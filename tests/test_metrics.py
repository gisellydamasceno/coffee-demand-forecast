"""Testes das métricas de avaliação."""

from __future__ import annotations

import numpy as np

from coffee_forecast.models.metrics import mae, mase, rmse, smape


def test_mae_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 3.0])  # um erro grande
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_smape_bounded_and_zero_on_perfect():
    y = np.array([10.0, 20.0])
    assert smape(y, y) == 0.0
    y_pred = np.array([0.0, 40.0])
    assert 0 <= smape(y, y_pred) <= 200


def test_mase_less_than_one_when_better_than_seasonal_naive():
    # treino com sazonalidade semanal + ruído (para o Seasonal Naive ter erro > 0,
    # senão o denominador do MASE é zero e o resultado é nan por definição).
    rng_gen = np.random.default_rng(0)
    base = np.tile([10, 12, 14, 11, 9, 6, 5], 10).astype(float)
    y_train = base + rng_gen.normal(0, 1.0, size=base.shape)
    # previsão quase perfeita vs valores reais
    y_true = np.array([10, 12, 14, 11, 9, 6, 5], dtype=float)
    y_pred = y_true + 0.1
    score = mase(y_true, y_pred, y_train, seasonal_period=7)
    assert score < 1.0


def test_mase_returns_nan_when_seasonal_naive_is_perfect():
    # Documenta o comportamento: série sazonal perfeita => denominador 0 => nan.
    y_train = np.tile([10, 12, 14, 11, 9, 6, 5], 10).astype(float)
    y_true = np.array([10, 12, 14, 11, 9, 6, 5], dtype=float)
    score = mase(y_true, y_true, y_train, seasonal_period=7)
    assert np.isnan(score)
