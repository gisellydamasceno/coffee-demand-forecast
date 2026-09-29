# Coffee Demand Forecast — Previsão de Demanda e Receita

Produto de dados de previsão de **receita diária** de uma máquina de café,
desenhado como **POC escalável** para uma rede de N pontos de venda.

> Dataset: [ihelon/coffee-sales](https://www.kaggle.com/datasets/ihelon/coffee-sales) (Kaggle).

---

## 1. Problema de negócio

Uma máquina de café registra vendas ao longo de ~1 ano. A dor priorizada é a
**previsibilidade de receita**: antecipar quanto será faturado nos próximos dias
apoia planejamento financeiro, reposição de insumos e dimensionamento operacional.

O projeto entrega:
- Previsão de **receita diária** (alvo principal) com horizonte configurável.
- Comparação honesta entre **baselines** e **modelos** (estatísticos e de ML).
- **Backtesting temporal** e métricas escolhidas pelo custo do erro no negócio.
- **Arquitetura de produção** (AWS) com equivalência para Azure/Databricks.

> Recorte de escopo: a previsão é feita no **agregado diário** (não por produto/dia),
> porque o volume é baixo (~9-10 vendas/dia) e a quebra por produto seria dominada
> por ruído. O mix de produtos é usado como **insight de negócio**, não como alvo.

---

## 2. Principais resultados (backtesting, 5 dobras temporais)

Alvo: receita diária. Métrica de decisão: **MASE** (< 1 supera o baseline sazonal).

| Modelo | MAE (R$) | MASE | Leitura |
|---|---|---|---|
| **LightGBM** | **123,0** | **0,82** | Melhor — erro ~18% menor que o baseline |
| SARIMA | 146,5 | 0,97 | Bate o baseline por pouco |
| Seasonal Naive | 148,9 | 0,99 | Baseline (régua) |
| Moving Average | 155,5 | 1,03 | Empata com o baseline |
| Prophet | 158,7 | 1,05 | Próximo do baseline |
| Naive | 230,2 | 1,53 | Piso |

**Aprendizado:** modelo mais complexo não garante melhor resultado. Por isso
comparamos tudo contra uma régua (Seasonal Naive) antes de concluir.

---

## 3. Estrutura do projeto

```
.
├── config/config.yaml         # parâmetros (fora do código)
├── data/{raw,processed}/      # dados (gitignored)
├── notebooks/                 # 01_eda, 02_modeling (narrados)
├── src/coffee_forecast/
│   ├── config.py              # config validada com Pydantic
│   ├── data/                  # load (ingestão+schema) e preprocess
│   ├── features/              # engenharia de features (anti-vazamento)
│   ├── models/                # baseline, train (SARIMA/Prophet/LightGBM),
│   │                          # evaluate (backtesting), metrics
│   ├── eda.py                 # funções de EDA reutilizáveis
│   └── pipeline.py            # orquestração ponta a ponta (CLI)
├── tests/                     # pytest (features, métricas, preprocess, baseline)
├── pyproject.toml             # deps + config de tooling
└── .pre-commit-config.yaml    # hooks de qualidade
```

---

## 4. Como executar

### Pré-requisitos
- Python 3.10+
- Conta Kaggle configurada para `kagglehub` (ou coloque os CSVs em `data/raw/`).

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -e ".[dev]"
```

### Rodar o pipeline completo (EDA + backtesting + previsão)

```bash
python -m coffee_forecast.pipeline --stage all
```

Estágios individuais:

```bash
python -m coffee_forecast.pipeline --stage eda      # só EDA + figuras
python -m coffee_forecast.pipeline --stage train    # só modelagem
python -m coffee_forecast.pipeline --stage train --no-mlflow   # sem tracking
```

Saídas em `data/processed/`: `backtest_summary.csv`, `backtest_folds.csv`,
`future_forecast.csv`, `results_summary.txt`, `eda_summary.txt`.

### Ver os experimentos no MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# abrir http://localhost:5000
```

### Rodar os testes

```bash
pytest -q
```

---

## 5. Decisões técnicas (o racional)

| Decisão | Escolha | Porquê |
|---|---|---|
| Granularidade | Agregado diário | Volume baixo; previsão por produto/dia seria ruído |
| Alvo | Receita | Dor de negócio priorizada (financeiro) |
| Baseline | Naive, Seasonal Naive, Média móvel | Régua honesta antes de sofisticar |
| Modelos | SARIMA, Prophet, LightGBM | 3 paradigmas p/ comparar trade-offs |
| Sem deep learning | Descartado | ~380 pontos não justificam; evita overfitting |
| Validação | Walk-forward temporal | Nunca vazar o futuro |
| Métrica de decisão | MASE | Diz se o modelo supera o baseline |
| Métrica de comunicação | MAE (R$) | Traduz o erro para o gestor |
| Pricing/elasticidade | Descartado | `money` reflete forma de pagamento, não experimentação de preço |
| Tracking | MLflow | Ponte direta para MLOps (AWS/Databricks) |

---

## 6. Boas práticas de engenharia

- Separação `src/` (produção) vs `notebooks/` (exploração).
- Configuração validada com **Pydantic**, parâmetros em `config.yaml`.
- **Testes** com pytest, incluindo teste anti-vazamento temporal das features.
- **Ruff** (lint + format) e **pre-commit** para qualidade automática.
- Ingestão com **validação de schema** (fail-fast).
- Rastreamento de experimentos com **MLflow**.

---

## 7. Arquitetura de produção

A POC roda localmente (Python + MLflow); o mesmo racional escala para produção.
Detalhes completos em [`architecture.md`](architecture.md).

**Fluxo (arquitetura Medalhão):**

```
Máquina/PDV → Ingestão → Bronze (cru) → Silver (limpo) → Gold (features)
   → Treino/Retreino → MLflow Registry → Inferência batch → BI / API
   → Observabilidade (erro real vs. previsto, drift) → trigger de retreino
```

- **Camadas Medalhão:** Bronze (dado bruto imutável), Silver (limpo/validado),
  Gold (features e agregações prontas para o modelo).
- **Governança:** promoção Staging → Production e rollback via MLflow Registry;
  um novo modelo só é promovido se superar o baseline e o modelo vigente.
- **Observabilidade:** performance (real vs. previsto), data drift e alertas,
  com retreino agendado ou disparado por degradação.
- **Escalabilidade 1 → N:** o mesmo desenho suporta uma rede de máquinas, via
  previsão hierárquica e um modelo global multi-máquina (em vez de um por máquina).
- **Stack:** referência em AWS (S3, Glue/EMR, SageMaker/MLflow, Step Functions,
  CloudWatch) com **equivalência direta em Azure/Databricks** (ADLS+Delta,
  Databricks, Unity Catalog, Workflows, Lakehouse Monitoring). O **MLflow** é a
  peça comum aos dois ecossistemas e já é usado nesta POC.

---

## 8. Limitações e evoluções futuras

- **Limitações:** uma única máquina; volume baixo; sem variáveis externas
  (clima, feriados locais, preço).
- **Riscos:** drift de comportamento; poucos dados para eventos raros.
- **Próximos passos:** previsão hierárquica (rede de máquinas), incorporar
  clima/feriados, modelo global multi-máquina, testar elasticidade de preço
  quando houver experimentação real de preço.
```
