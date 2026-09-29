"""Testes dos baselines."""

from __future__ import annotations

import numpy as np

from coffee_forecast.models.baseline import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def test_naive_repeats_last_value(daily_series):
    model = NaiveForecaster().fit(daily_series)
    preds = model.predict(5)
    assert np.allclose(preds, daily_series.iloc[-1])


def test_seasonal_naive_repeats_last_season(daily_series):
    model = SeasonalNaiveForecaster(seasonal_period=7).fit(daily_series)
    preds = model.predict(7)
    assert np.allclose(preds, daily_series.iloc[-7:].to_numpy())


def test_seasonal_naive_tiles_beyond_period(daily_series):
    model = SeasonalNaiveForecaster(seasonal_period=7).fit(daily_series)
    preds = model.predict(10)
    assert len(preds) == 10
    # os 3 primeiros do 2º ciclo repetem os 3 primeiros do último ciclo
    assert np.allclose(preds[7:10], daily_series.iloc[-7:-4].to_numpy())


def test_moving_average_is_mean_of_window(daily_series):
    model = MovingAverageForecaster(window=7).fit(daily_series)
    preds = model.predict(3)
    expected = daily_series.iloc[-7:].mean()
    assert np.allclose(preds, expected)
