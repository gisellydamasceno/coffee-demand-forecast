"""Testes da engenharia de features — foco em VAZAMENTO TEMPORAL.

O teste mais importante: garantir que nenhuma feature em t use informação
de t ou do futuro. Se este teste passa, a validação do modelo é confiável.
"""

from __future__ import annotations

import numpy as np

from coffee_forecast.features.build_features import (
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
)


def test_lag_features_do_not_leak(daily_frame):
    df = add_lag_features(daily_frame[["revenue"]], "revenue", [1, 7])
    # lag_1 em t deve ser igual a revenue em t-1
    assert df["revenue_lag_1"].iloc[5] == daily_frame["revenue"].iloc[4]
    assert df["revenue_lag_7"].iloc[10] == daily_frame["revenue"].iloc[3]


def test_rolling_features_do_not_include_current(daily_frame):
    df = add_rolling_features(daily_frame[["revenue"]], "revenue", [3])
    # a média móvel(3) em t usa t-1,t-2,t-3 (shift(1) antes do rolling)
    expected = daily_frame["revenue"].iloc[7:10].mean()
    assert np.isclose(df["revenue_rollmean_3"].iloc[10], expected)


def test_build_matrix_alignment(daily_frame):
    x, y = build_feature_matrix(daily_frame, "revenue", [1, 7], [7], dropna=True)
    assert len(x) == len(y)
    assert not x.isna().any().any()
    # o alvo não pode estar entre as features
    assert "revenue" not in x.columns


def test_target_not_in_features(daily_frame):
    x, _ = build_feature_matrix(daily_frame, "revenue", [1], [7], dropna=True)
    leaking = [c for c in x.columns if c == "revenue"]
    assert leaking == []
