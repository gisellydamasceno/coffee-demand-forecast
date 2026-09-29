"""Preparação dos dados: limpeza e agregação para série temporal diária.

Decisões-chave:
- Reindexação do calendário completo: uma vending machine tem dias sem venda.
  Se esses dias não existirem na série, a frequência diária fica quebrada e a
  sazonalidade semanal é distorcida. Preenchemos dias sem venda com 0.
- Agregamos ao nível DIÁRIO (receita e nº de transações). Foi a granularidade
  escolhida por ter sinal (vs. ruído da previsão por produto/dia).
- Mantemos uma agregação por produto/dia como insumo secundário (mix, EDA),
  não como alvo principal.
"""

from __future__ import annotations

import pandas as pd

from coffee_forecast.config import Config


def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Converte a coluna de data para datetime (normalizada à meia-noite)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    n_bad = out[date_col].isna().sum()
    if n_bad:
        raise ValueError(f"{n_bad} datas não puderam ser parseadas em '{date_col}'.")
    return out


def build_daily_series(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Agrega transações em uma série temporal diária.

    Produz, por dia:
        - revenue: soma de 'money' (receita)
        - transactions: contagem de transações (demanda)
        - avg_ticket: receita / transações (ticket médio)

    Dias sem venda são incluídos com revenue=0 e transactions=0.

    Returns:
        DataFrame indexado por data diária contínua.
    """
    date_col = cfg.target.date_col
    value_col = cfg.target.value_col

    data = _parse_dates(df, date_col)

    daily = (
        data.groupby(date_col)
        .agg(revenue=(value_col, "sum"), transactions=(value_col, "size"))
        .sort_index()
    )

    # Reindexa para um calendário diário contínuo (preenche dias sem venda).
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq=cfg.forecast.frequency)
    daily = daily.reindex(full_idx)
    daily[["revenue", "transactions"]] = daily[["revenue", "transactions"]].fillna(0.0)
    daily.index.name = date_col

    # Ticket médio: evita divisão por zero em dias sem venda.
    daily["avg_ticket"] = (daily["revenue"] / daily["transactions"]).where(
        daily["transactions"] > 0, 0.0
    )

    return daily


def build_product_daily(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Agrega receita e transações por produto e dia (insumo secundário/EDA)."""
    date_col = cfg.target.date_col
    value_col = cfg.target.value_col
    product_col = cfg.target.product_col

    data = _parse_dates(df, date_col)
    return (
        data.groupby([date_col, product_col])
        .agg(revenue=(value_col, "sum"), transactions=(value_col, "size"))
        .reset_index()
        .sort_values([date_col, product_col])
    )


def add_calendar_context(daily: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de calendário úteis para EDA e features simples."""
    out = daily.copy()
    idx = out.index
    out["dow"] = idx.dayofweek  # 0=segunda ... 6=domingo
    out["day_name"] = idx.day_name()
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return out


def prepare(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pipeline de preparação padrão: série diária + contexto de calendário."""
    daily = build_daily_series(df, cfg)
    return add_calendar_context(daily)
