"""Funções reutilizáveis para montar a base de pedidos da Olist.

Concentra a carga dos CSVs, os joins, a criação do alvo `atrasou`, o cálculo
de distância e as features disponíveis no momento da compra. O notebook
importa daqui para que a mesma lógica possa ser reaproveitada na etapa de
modelagem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ARQUIVOS = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}

COLUNAS_DATA = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

# CEP é lido como texto para preservar zeros à esquerda (ex.: "01037").
COLUNAS_TEXTO = {
    "customer_zip_code_prefix": str,
    "seller_zip_code_prefix": str,
    "geolocation_zip_code_prefix": str,
}

# Limites aproximados do território brasileiro, usados para descartar
# coordenadas claramente erradas da tabela de geolocalização.
LIMITES_BRASIL = {"lat": (-33.8, 5.3), "lng": (-73.99, -34.79)}

# Colunas que só existem DEPOIS da compra. Nunca podem virar feature.
COLUNAS_PROIBIDAS = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "review_score",
    "review_id",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
    "shipping_limit_date",
]


def carregar_tabelas(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Lê os 9 CSVs da Olist, já convertendo datas e CEPs."""
    raw_dir = Path(raw_dir)
    tabelas = {}
    for nome, arquivo in ARQUIVOS.items():
        df = pd.read_csv(
            raw_dir / arquivo,
            dtype=COLUNAS_TEXTO,
            parse_dates=COLUNAS_DATA.get(nome, []),
            encoding="utf-8-sig",  # o arquivo de tradução tem BOM
        )
        tabelas[nome] = df
    return tabelas


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em linha reta (km) entre dois pontos na superfície da Terra."""
    raio_terra_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * raio_terra_km * np.arcsin(np.sqrt(a))


def agregar_geolocalizacao(geo: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por prefixo de CEP com a latitude/longitude média.

    Antes de tirar a média, remove coordenadas fora do Brasil (erros de
    cadastro), que puxariam a média para pontos absurdos.
    """
    lat_ok = geo["geolocation_lat"].between(*LIMITES_BRASIL["lat"])
    lng_ok = geo["geolocation_lng"].between(*LIMITES_BRASIL["lng"])
    return (
        geo[lat_ok & lng_ok]
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
    )


def filtrar_entregues(orders: pd.DataFrame) -> pd.DataFrame:
    """Mantém só pedidos entregues e com data de entrega preenchida."""
    mascara = (orders["order_status"] == "delivered") & orders[
        "order_delivered_customer_date"
    ].notna()
    return orders[mascara].copy()


def criar_alvo(orders: pd.DataFrame) -> pd.Series:
    """atrasou = 1 se a entrega real foi depois da data estimada."""
    return (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    ).astype(int)


def agregar_itens(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    """Leva `order_items` (várias linhas por pedido) para uma linha por pedido.

    Cada linha de `order_items` é uma unidade vendida. Somamos preço, frete,
    peso e volume, contamos itens/produtos/vendedores e escolhemos um
    "item principal" (o de maior preço) para definir a categoria principal e
    o vendedor principal do pedido.
    """
    prod = products.merge(translation, on="product_category_name", how="left")
    prod["volume_cm3"] = (
        prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    )
    itens = order_items.merge(
        prod[
            [
                "product_id",
                "product_category_name",
                "product_category_name_english",
                "product_weight_g",
                "volume_cm3",
            ]
        ],
        on="product_id",
        how="left",
    )

    agregado = itens.groupby("order_id").agg(
        valor_total=("price", "sum"),
        frete_total=("freight_value", "sum"),
        n_itens=("order_item_id", "count"),
        n_produtos=("product_id", "nunique"),
        n_vendedores=("seller_id", "nunique"),
        peso_total_g=("product_weight_g", "sum"),
        volume_total_cm3=("volume_cm3", "sum"),
    )
    # Se algum item do pedido não tem peso/volume, a soma ficaria subestimada;
    # nesse caso marcamos o total como ausente para ser imputado depois.
    falta_peso = itens["product_weight_g"].isna().groupby(itens["order_id"]).any()
    falta_vol = itens["volume_cm3"].isna().groupby(itens["order_id"]).any()
    agregado.loc[falta_peso[falta_peso].index, "peso_total_g"] = np.nan
    agregado.loc[falta_vol[falta_vol].index, "volume_total_cm3"] = np.nan

    principal = (
        itens.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True])
        .drop_duplicates("order_id")
        .set_index("order_id")[["seller_id", "product_category_name"]]
        .rename(
            columns={
                "seller_id": "seller_id_principal",
                "product_category_name": "categoria_principal",
            }
        )
    )
    return agregado.join(principal).reset_index()


def montar_base_pedidos(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta todas as tabelas em uma base com UMA linha por pedido entregue."""
    orders = filtrar_entregues(t["orders"])
    orders["atrasou"] = criar_alvo(orders)

    itens = agregar_itens(t["order_items"], t["products"], t["translation"])
    geo = agregar_geolocalizacao(t["geolocation"])

    base = (
        orders.merge(t["customers"], on="customer_id", how="left")
        .merge(itens, on="order_id", how="inner")  # pedido sem item não tem o que prever
        .merge(
            t["sellers"].rename(columns={"seller_id": "seller_id_principal"}),
            on="seller_id_principal",
            how="left",
        )
        .merge(
            geo.rename(columns={"zip_prefix": "customer_zip_code_prefix", "lat": "lat_cliente", "lng": "lng_cliente"}),
            on="customer_zip_code_prefix",
            how="left",
        )
        .merge(
            geo.rename(columns={"zip_prefix": "seller_zip_code_prefix", "lat": "lat_vendedor", "lng": "lng_vendedor"}),
            on="seller_zip_code_prefix",
            how="left",
        )
    )
    base["distancia_km"] = haversine_km(
        base["lat_vendedor"], base["lng_vendedor"], base["lat_cliente"], base["lng_cliente"]
    )
    return base


def criar_features_compra(base: pd.DataFrame) -> pd.DataFrame:
    """Features que já são conhecidas no instante da compra.

    A data estimada de entrega é informada ao cliente no checkout, então o
    prazo prometido é legítimo. Datas de aprovação, envio e entrega real não.
    """
    compra = base["order_purchase_timestamp"]
    f = pd.DataFrame(index=base.index)
    f["order_id"] = base["order_id"]
    f["data_compra"] = compra
    f["prazo_prometido_dias"] = (
        base["order_estimated_delivery_date"] - compra
    ).dt.total_seconds() / 86400
    f["mes_compra"] = compra.dt.month
    f["dia_semana_compra"] = compra.dt.dayofweek  # 0 = segunda
    f["valor_total"] = base["valor_total"]
    f["frete_total"] = base["frete_total"]
    f["razao_frete_preco"] = base["frete_total"] / base["valor_total"]
    f["n_itens"] = base["n_itens"]
    f["n_vendedores"] = base["n_vendedores"]
    f["peso_total_g"] = base["peso_total_g"]
    f["volume_total_cm3"] = base["volume_total_cm3"]
    f["categoria_principal"] = base["categoria_principal"]
    f["estado_cliente"] = base["customer_state"]
    f["estado_vendedor"] = base["seller_state"]
    f["mesmo_estado"] = (base["customer_state"] == base["seller_state"]).astype(int)
    f["distancia_km"] = base["distancia_km"]
    f["atrasou"] = base["atrasou"]
    return f


class AgrupadorCategoriasRaras(BaseEstimator, TransformerMixin):
    """Preenche ausentes e agrupa categorias raras em "outros".

    As categorias "frequentes" são aprendidas SÓ no treino (frequência
    relativa >= `min_freq`). No teste, qualquer categoria fora dessa lista
    (rara ou nunca vista) vira "outros", então o one-hot seguinte nunca
    encontra coluna nova.
    """

    def __init__(self, min_freq: float = 0.01, valor_ausente: str = "desconhecido",
                 rotulo_raras: str = "outros"):
        self.min_freq = min_freq
        self.valor_ausente = valor_ausente
        self.rotulo_raras = rotulo_raras

    def _preparar(self, X):
        return pd.DataFrame(X).astype(object).fillna(self.valor_ausente)

    def fit(self, X, y=None):
        X = self._preparar(X)
        self.colunas_ = list(X.columns)
        self.frequentes_ = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.frequentes_[col] = set(freq[freq >= self.min_freq].index)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._preparar(X)
        for col in X.columns:
            X[col] = X[col].where(X[col].isin(self.frequentes_[col]), self.rotulo_raras)
        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.colunas_ if input_features is None else input_features, dtype=object)


class Winsorizador(BaseEstimator, TransformerMixin):
    """Limita cada coluna numérica aos percentis aprendidos no treino.

    Não remove linhas: valores extremos continuam sendo os maiores da base,
    mas deixam de dominar a escala (importante para a regressão logística).
    Por padrão só a cauda superior é limitada (`inferior=0` mantém o mínimo):
    as variáveis do problema são limitadas em zero, e valores baixos como
    frete grátis ou distância zero são legítimos.
    """

    def __init__(self, inferior: float = 0.0, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.limites_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.limites_sup_ = np.nanquantile(X, self.superior, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.limites_inf_, self.limites_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
