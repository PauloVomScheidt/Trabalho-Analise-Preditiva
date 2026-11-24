# Previsão de atraso na entrega — Olist (Análise Preditiva)

Trabalho da disciplina de **Análise Preditiva**.

**Integrantes:** Guilherme Theis e Paulo Henrique

## Problema

Prever, **no momento da compra**, se um pedido feito no marketplace Olist será entregue com atraso.

- **Tipo:** classificação binária.
- **Variável alvo `atrasou`:** `1` se `order_delivered_customer_date > order_estimated_delivery_date`; caso contrário, `0`.
- **Escopo:** apenas pedidos com `order_status == "delivered"` e com data de entrega preenchida.
- A nota de avaliação (`review_score`) entra **somente** na análise exploratória, para medir o impacto do atraso na satisfação. Ela **não** é usada como feature.

## Dataset

[Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 9 arquivos CSV, cerca de 100 mil pedidos, de 2016 a 2018.

## Estrutura

```
.
├── data/
│   ├── raw/            # CSVs originais (fora do Git)
│   └── processed/      # bases tratadas (fora do Git)
├── notebooks/
│   └── 01_eda_preprocessamento.ipynb
├── src/
│   └── features.py     # funções reutilizáveis (joins, alvo, distância)
├── reports/figures/    # gráficos exportados pelo notebook
├── requirements.txt
└── README.md
```

## Como criar o ambiente

Requer Python 3.12.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name analise-preditiva --display-name "Python (analise-preditiva)"
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name analise-preditiva --display-name "Python (analise-preditiva)"
```

## Como baixar os dados

**Opção 1 (automática):** com o ambiente ativado, execute:
```bash
python -c "import kagglehub, shutil, glob, os; p = kagglehub.dataset_download('olistbr/brazilian-ecommerce'); [shutil.copy(f, 'data/raw/') for f in glob.glob(os.path.join(p, '*.csv'))]"
```
Se o Kaggle pedir autenticação, gere um token em *Kaggle → Settings → API → Create New Token* e salve o `kaggle.json` em `~/.kaggle/` (Windows: `C:\Users\<usuario>\.kaggle\`). Esse arquivo nunca deve ser versionado.

**Opção 2 (manual):** baixe o zip pela página do dataset (botão *Download*) e extraia os 9 CSVs diretamente em `data/raw/`.

## Como rodar o notebook

```bash
jupyter notebook notebooks/01_eda_preprocessamento.ipynb
```
Selecione o kernel **Python (analise-preditiva)**. Para executar tudo pela linha de comando e salvar as saídas:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_preprocessamento.ipynb
```

O notebook gera os gráficos em `reports/figures/` e as bases tratadas em `data/processed/`.
