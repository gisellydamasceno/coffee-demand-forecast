"""Modelos de previsão com interface comum (fit/predict).

Escada de complexidade crescente (do mais simples ao mais sofisticado):
- Baselines (baseline.py): referência de comparação.
- SARIMA (estatístico clássico): interpretável, projetado para sazonalidade,
  intervalos de previsão nativos. Bom com histórico curto.
- Prophet (aditivo): robusto, decomponível (fácil de comunicar), lida com
  sazonalidades múltiplas e feriados, incerteza nativa.
- LightGBM (ML de boosting): captura não linearidades via features de
  lag/calendário; permite feature importance (interpretabilidade + comunicação).

Deep learning (LSTM etc.) foi DESCARTADO de propósito: ~380 pontos diários não
justificam a complexidade e o risco de overfitting. Recusar over-engineering
também é uma decisão técnica.

Todos os modelos expõem:
    fit(y: pd.Series[, X])  -> self
    predict(horizon:int[, X]) -> np.ndarray  (previsão pontual)
Alguns expõem predict_interval(...) para bandas de incerteza.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------


class SarimaForecaster:
    """SARIMA via statsmodels.

    Ordens default escolhidas para uma série diária com sazonalidade semanal:
    (1,1,1)(1,1,1,7). Ajustáveis via config/experimentação.
    """

    name = "sarima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> SarimaForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y.astype(float),
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False)
        return self

    def predict(self, horizon: int, X: pd.DataFrame | None = None) -> np.ndarray:
        forecast = self._result.get_forecast(steps=horizon)
        return np.asarray(forecast.predicted_mean, dtype=float)

    def predict_interval(self, horizon: int, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (lower, upper) do intervalo de previsão (1-alpha de confiança)."""
        forecast = self._result.get_forecast(steps=horizon)
        ci = forecast.conf_int(alpha=alpha)
        return np.asarray(ci.iloc[:, 0]), np.asarray(ci.iloc[:, 1])


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------


class ProphetForecaster:
    """Prophet (Meta) para série diária com sazonalidade semanal/anual."""

    name = "prophet"

    def __init__(self, weekly: bool = True, yearly: bool = False) -> None:
        # yearly desligado por padrão: com ~1 ano de histórico (<730 dias) a
        # sazonalidade anual fica sub-identificada e piora o modelo. Decisão
        # técnica alinhada ao aviso do próprio Prophet.
        self.weekly = weekly
        self.yearly = yearly
        self._model = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> ProphetForecaster:
        from prophet import Prophet

        df = pd.DataFrame({"ds": y.index, "y": y.astype(float).to_numpy()})
        self._last_date = pd.Timestamp(y.index[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Prophet(
                weekly_seasonality=self.weekly,
                yearly_seasonality=self.yearly,
                daily_seasonality=False,
                interval_width=0.9,
            )
            self._model.fit(df)
        return self

    def _future(self, horizon: int) -> pd.DataFrame:
        future_dates = pd.date_range(
            self._last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        return pd.DataFrame({"ds": future_dates})

    def predict(self, horizon: int, X: pd.DataFrame | None = None) -> np.ndarray:
        fcst = self._model.predict(self._future(horizon))
        return np.asarray(fcst["yhat"], dtype=float)

    def predict_interval(self, horizon: int, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        fcst = self._model.predict(self._future(horizon))
        return np.asarray(fcst["yhat_lower"]), np.asarray(fcst["yhat_upper"])


# ---------------------------------------------------------------------------
# LightGBM (recursivo, com features de lag/calendário)
# ---------------------------------------------------------------------------


class LightGBMForecaster:
    """LightGBM para previsão multi-step por estratégia recursiva.

    Treina um regressor sobre features de calendário + lags + médias móveis.
    Na previsão, gera um dia por vez, realimentando a previsão anterior para
    recalcular os lags (previsão recursiva). Isso mantém coerência temporal.
    """

    name = "lightgbm"

    def __init__(self, lags: list[int], rolling_windows: list[int], **lgb_params) -> None:
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.lgb_params = lgb_params or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbose": -1,
        }
        self._model = None
        self._history: pd.Series | None = None
        self._target = "revenue"

    def _make_features(self, series: pd.Series) -> pd.DataFrame:
        from coffee_forecast.features.build_features import build_feature_matrix

        df = series.to_frame(name=self._target)
        x, _ = build_feature_matrix(
            df, self._target, self.lags, self.rolling_windows, dropna=False
        )
        return x

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> LightGBMForecaster:
        import lightgbm as lgb

        from coffee_forecast.features.build_features import build_feature_matrix

        self._history = y.astype(float).copy()
        self._target = y.name or "revenue"
        df = y.to_frame(name=self._target)
        x, target = build_feature_matrix(
            df, self._target, self.lags, self.rolling_windows, dropna=True
        )
        self._feature_names = list(x.columns)
        self._model = lgb.LGBMRegressor(**self.lgb_params)
        self._model.fit(x, target)
        return self

    def predict(self, horizon: int, X: pd.DataFrame | None = None) -> np.ndarray:
        history = self._history.copy()
        preds = []
        for _ in range(horizon):
            next_date = history.index[-1] + pd.Timedelta(days=1)
            extended = pd.concat([history, pd.Series([np.nan], index=[next_date])])
            feats = self._make_features(extended)
            x_next = feats.loc[[next_date], self._feature_names]
            yhat = float(self._model.predict(x_next)[0])
            yhat = max(yhat, 0.0)  # receita não pode ser negativa
            history.loc[next_date] = yhat
            preds.append(yhat)
        return np.asarray(preds, dtype=float)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self._model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Fábrica de modelos
# ---------------------------------------------------------------------------


def build_models(cfg) -> dict:
    """Instancia todos os modelos (não-baseline) a partir da config."""
    return {
        "sarima": SarimaForecaster(seasonal_order=(1, 1, 1, cfg.forecast.seasonal_period)),
        "prophet": ProphetForecaster(),
        "lightgbm": LightGBMForecaster(
            lags=cfg.features.lags, rolling_windows=cfg.features.rolling_windows
        ),
    }
