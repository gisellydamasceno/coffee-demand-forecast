"""Fixtures compartilhadas para os testes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_series() -> pd.Series:
    """Série diária sintética com sazonalidade semanal clara (180 dias)."""
    idx = pd.date_range("2024-01-01", periods=180, freq="D")
    dow_effect = np.array([100, 120, 130, 110, 90, 60, 50])  # seg..dom
    values = dow_effect[idx.dayofweek] + np.linspace(0, 20, len(idx))
    return pd.Series(values, index=idx, name="revenue")


@pytest.fixture
def daily_frame(daily_series: pd.Series) -> pd.DataFrame:
    df = daily_series.to_frame()
    df["transactions"] = (daily_series / 30).round()
    return df
