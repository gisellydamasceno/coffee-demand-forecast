"""Métricas de avaliação, escolhidas pelo custo do erro no negócio.

Decisão: não reportamos "todas as métricas".
Escolhemos poucas, justificadas pela dor de negócio priorizada — PREVISIBILIDADE
DE RECEITA:

- MAE (primária): erro médio na unidade original (R$). Comunicação direta:
  "erramos em média R$ X por dia". Fácil de traduzir para o gestor.
- MASE (primária de decisão): erro relativo ao Seasonal Naive. Em número único
  diz se o modelo VALE o custo vs. o baseline. MASE < 1 => supera o baseline.
- RMSE (secundária): penaliza erros grandes; relevante se um erro grande num dia
  causa dano desproporcional (ex.: dimensionamento).
- sMAPE (secundária): erro percentual simétrico, comparável entre séries; usar
  com cautela em dias de volume muito baixo.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE em % (0 a 200). Trata denominador zero como erro 0."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    ratio = np.where(denom == 0, 0.0, 2.0 * diff / denom)
    return float(np.mean(ratio) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 7,
) -> float:
    """Mean Absolute Scaled Error, escalado pelo erro do Seasonal Naive no treino.

    Denominador = erro absoluto médio do Seasonal Naive dentro da amostra de treino.
    MASE < 1: o modelo é melhor que o baseline sazonal ingênuo.
    MASE > 1: o modelo NÃO justifica sua complexidade.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    y_train = np.asarray(y_train, float)

    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 7,
) -> dict[str, float]:
    """Calcula o conjunto de métricas para um fold."""
    return {
        "mae": mae(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, seasonal_period),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
