"""Engenharia de features para o modelo de ML (LightGBM).

Princípio crítico: evitar VAZAMENTO TEMPORAL.
Toda feature em uma data t usa apenas informação disponível até t-1
(lags e janelas móveis deslocados com shift). Nunca usamos o próprio valor
de t nem valores futuros para prever t.

Grupos de features:
- Calendário: dia da semana, mês, fim de semana, dia do mês.
- Lags: valor de 1, 7, 14, 28 dias atrás (memória de curto e médio prazo).
- Médias móveis: tendência recente suavizada (deslocadas para não vazar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de calendário derivadas do índice de datas."""
    out = df.copy()
    idx = out.index
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_month"] = idx.day
    out["week_of_year"] = idx.isocalendar().week.astype(int)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # Codificação cíclica do dia da semana (captura continuidade dom->seg).
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return out


def add_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    """Adiciona features de lag do alvo (deslocadas => sem vazamento)."""
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target: str, windows: list[int]) -> pd.DataFrame:
    """Adiciona médias e desvios móveis do alvo.

    Importante: aplicamos shift(1) ANTES do rolling para que a janela em t
    use apenas valores até t-1 (sem incluir o próprio t).
    """
    out = df.copy()
    shifted = out[target].shift(1)
    for w in windows:
        out[f"{target}_rollmean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        out[f"{target}_rollstd_{w}"] = shifted.rolling(window=w, min_periods=1).std()
    return out


def build_feature_matrix(
    daily: pd.DataFrame,
    target: str,
    lags: list[int],
    rolling_windows: list[int],
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Constrói a matriz de features X e o vetor alvo y.

    Args:
        daily: série diária indexada por data, contendo a coluna `target`.
        target: nome da coluna alvo (ex.: 'revenue').
        lags: lista de defasagens.
        rolling_windows: janelas das médias/desvios móveis.
        dropna: se True, remove as primeiras linhas sem histórico suficiente.

    Returns:
        (X, y) alinhados por data.
    """
    df = daily[[target]].copy()
    df = add_calendar_features(df)
    df = add_lag_features(df, target, lags)
    df = add_rolling_features(df, target, rolling_windows)

    if dropna:
        df = df.dropna()

    y = df[target]
    x = df.drop(columns=[target])
    return x, y
