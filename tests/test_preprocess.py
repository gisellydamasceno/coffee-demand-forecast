"""Testes da preparação de dados (agregação e reindexação)."""

from __future__ import annotations

import pandas as pd

from coffee_forecast.config import load_config
from coffee_forecast.data.preprocess import build_daily_series


def _tiny_raw() -> pd.DataFrame:
    """Dataset mínimo com um dia sem venda no meio (para testar reindex)."""
    return pd.DataFrame(
        {
            "date": ["2024-03-01", "2024-03-01", "2024-03-03"],  # falta 02/03
            "datetime": [
                "2024-03-01 10:00:00",
                "2024-03-01 12:00:00",
                "2024-03-03 09:00:00",
            ],
            "cash_type": ["card", "card", "cash"],
            "card": ["A", "B", None],
            "money": [30.0, 20.0, 25.0],
            "coffee_name": ["Latte", "Americano", "Latte"],
        }
    )


def test_daily_aggregation_and_reindex():
    cfg = load_config()
    daily = build_daily_series(_tiny_raw(), cfg)
    # deve haver 3 dias (01, 02, 03) mesmo sem venda em 02
    assert len(daily) == 3
    # dia 01: receita 50, 2 transações
    assert daily.loc["2024-03-01", "revenue"] == 50.0
    assert daily.loc["2024-03-01", "transactions"] == 2
    # dia 02: sem venda -> preenchido com 0
    assert daily.loc["2024-03-02", "revenue"] == 0.0
    assert daily.loc["2024-03-02", "transactions"] == 0


def test_avg_ticket_no_division_by_zero():
    cfg = load_config()
    daily = build_daily_series(_tiny_raw(), cfg)
    # dia sem venda: ticket médio = 0 (sem erro de divisão)
    assert daily.loc["2024-03-02", "avg_ticket"] == 0.0
