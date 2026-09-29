"""Modelos de baseline para previsão de série temporal.

Por que baselines importam:
Um baseline é um modelo de referência propositalmente simples. Ele responde
"bom comparado a quê?". Nenhum modelo sofisticado deve ser levado a produção
se não superar, de forma consistente, um baseline barato de manter.

Baselines implementados:
- Naive: previsão = último valor observado.
- SeasonalNaive: previsão = valor do mesmo dia da semana anterior (período m).
  É a régua principal aqui, pois a série tem forte sazonalidade semanal.
- MovingAverage: previsão = média móvel dos últimos N dias.

Todos seguem uma interface comum (fit/predict) para serem intercambiáveis
com os modelos mais complexos no backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaselineForecaster:
    """Interface comum para os baselines."""

    name: str = "baseline"

    def fit(self, y: pd.Series) -> BaselineForecaster:
        self._y = y.astype(float)
        return self

    def predict(self, horizon: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class NaiveForecaster(BaselineForecaster):
    """Previsão = último valor observado, repetido no horizonte."""

    name = "naive"

    def predict(self, horizon: int) -> np.ndarray:
        last = self._y.iloc[-1]
        return np.repeat(last, horizon)


class SeasonalNaiveForecaster(BaselineForecaster):
    """Previsão = valores do último ciclo sazonal (m dias atrás).

    Para horizonte > m, o padrão do último ciclo é repetido ciclicamente.
    """

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 7) -> None:
        self.m = seasonal_period

    def predict(self, horizon: int) -> np.ndarray:
        last_season = self._y.iloc[-self.m :].to_numpy()
        reps = int(np.ceil(horizon / self.m))
        return np.tile(last_season, reps)[:horizon]


class MovingAverageForecaster(BaselineForecaster):
    """Previsão = média dos últimos `window` valores, repetida no horizonte."""

    name = "moving_average"

    def __init__(self, window: int = 7) -> None:
        self.window = window

    def predict(self, horizon: int) -> np.ndarray:
        avg = self._y.iloc[-self.window :].mean()
        return np.repeat(avg, horizon)


def get_baselines(seasonal_period: int = 7) -> list[BaselineForecaster]:
    """Retorna a lista padrão de baselines usada no backtesting."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonal_period=seasonal_period),
        MovingAverageForecaster(window=seasonal_period),
    ]
