"""Backtesting temporal (walk-forward) e avaliação comparativa de modelos.

Por que walk-forward e NUNCA split aleatório:
Em série temporal, embaralhar quebra a ordem do tempo e vaza o futuro no
treino, inflando artificialmente a performance. O correto é backtesting:
treinar com o passado, prever o futuro, deslizar a janela e repetir.

Estratégia: expanding window.
- Janela de treino cresce a cada dobra (começa em min_train_size).
- A cada dobra, prevê-se o próximo bloco de `test_size` dias.
- Métricas são calculadas por dobra e agregadas (média).

A métrica MASE usa o histórico de treino de cada dobra como referência do
Seasonal Naive, o que a torna comparável entre dobras e modelos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from coffee_forecast.config import Config
from coffee_forecast.models.baseline import get_baselines
from coffee_forecast.models.metrics import evaluate_all
from coffee_forecast.models.train import build_models


@dataclass
class FoldResult:
    fold: int
    model: str
    metrics: dict[str, float]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class BacktestReport:
    per_fold: list[FoldResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.per_fold:
            row = {"fold": r.fold, "model": r.model, **r.metrics}
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Média das métricas por modelo, ordenada pela métrica de decisão (MASE)."""
        df = self.to_frame()
        agg = df.groupby("model").mean(numeric_only=True).drop(columns=["fold"])
        return agg.sort_values("mase")


def _make_splits(
    n: int, n_splits: int, test_size: int, min_train_size: int
) -> list[tuple[int, int]]:
    """Gera índices (train_end, test_end) para expanding window.

    A última dobra termina no fim da série; as anteriores recuam de test_size.
    """
    splits = []
    for k in range(n_splits):
        test_end = n - (n_splits - 1 - k) * test_size
        train_end = test_end - test_size
        if train_end < min_train_size:
            continue
        splits.append((train_end, test_end))
    return splits


def run_backtest(
    daily: pd.DataFrame,
    cfg: Config,
    target: str | None = None,
    use_mlflow: bool = True,
) -> BacktestReport:
    """Executa o backtesting comparando baselines e modelos.

    Args:
        daily: série diária preparada (indexada por data).
        cfg: configuração.
        target: coluna alvo (default: cfg.target.primary -> 'revenue').
        use_mlflow: registra métricas/parâmetros no MLflow se True.

    Returns:
        BacktestReport com métricas por dobra.
    """
    target = target or cfg.target.primary
    y_full = daily[target].astype(float)
    n = len(y_full)
    m = cfg.forecast.seasonal_period

    splits = _make_splits(
        n,
        cfg.backtest.n_splits,
        cfg.backtest.test_size,
        cfg.backtest.min_train_size,
    )
    report = BacktestReport()

    mlflow = None
    if use_mlflow:
        import os

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow as _mlflow

        _mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        _mlflow.set_experiment(cfg.mlflow.experiment_name)
        mlflow = _mlflow

    def _eval_model(model_name: str, make_model, is_baseline: bool) -> None:
        fold_metrics = []
        run_ctx = mlflow.start_run(run_name=f"{target}-{model_name}") if mlflow else None
        try:
            if mlflow:
                mlflow.log_param("model", model_name)
                mlflow.log_param("target", target)
                mlflow.log_param("is_baseline", is_baseline)
                mlflow.log_param("n_splits", len(splits))
                mlflow.log_param("test_size", cfg.backtest.test_size)
                mlflow.log_param("seasonal_period", m)
            for i, (train_end, test_end) in enumerate(splits):
                y_train = y_full.iloc[:train_end]
                y_test = y_full.iloc[train_end:test_end]
                horizon = len(y_test)

                model = make_model()
                model.fit(y_train)
                y_pred = model.predict(horizon)
                y_pred = np.asarray(y_pred, dtype=float)[:horizon]

                metrics = evaluate_all(
                    y_test.to_numpy(), y_pred, y_train.to_numpy(), seasonal_period=m
                )
                fold_metrics.append(metrics)
                report.per_fold.append(
                    FoldResult(
                        fold=i,
                        model=model_name,
                        metrics=metrics,
                        train_end=y_full.index[train_end - 1],
                        test_start=y_full.index[train_end],
                        test_end=y_full.index[test_end - 1],
                    )
                )
            if mlflow and fold_metrics:
                mean_metrics = {
                    f"{k}_mean": float(np.mean([fm[k] for fm in fold_metrics]))
                    for k in fold_metrics[0]
                }
                mlflow.log_metrics(mean_metrics)
        finally:
            if run_ctx is not None:
                mlflow.end_run()

    # Baselines
    for bl in get_baselines(seasonal_period=m):
        _eval_model(bl.name, lambda bl=bl: type(bl)(**_baseline_kwargs(bl, m)), is_baseline=True)

    # Modelos
    models = build_models(cfg)
    for name, proto in models.items():
        _eval_model(name, lambda proto=proto: _clone_model(proto, cfg), is_baseline=False)

    return report


def _baseline_kwargs(bl, m: int) -> dict:
    """Recupera os kwargs necessários para reinstanciar um baseline."""
    from coffee_forecast.models.baseline import MovingAverageForecaster, SeasonalNaiveForecaster

    if isinstance(bl, SeasonalNaiveForecaster):
        return {"seasonal_period": m}
    if isinstance(bl, MovingAverageForecaster):
        return {"window": bl.window}
    return {}


def _clone_model(proto, cfg: Config):
    """Cria uma instância nova do mesmo tipo do protótipo (fold independente)."""
    from coffee_forecast.models.train import (
        LightGBMForecaster,
        ProphetForecaster,
        SarimaForecaster,
    )

    if isinstance(proto, SarimaForecaster):
        return SarimaForecaster(order=proto.order, seasonal_order=proto.seasonal_order)
    if isinstance(proto, ProphetForecaster):
        return ProphetForecaster(weekly=proto.weekly, yearly=proto.yearly)
    if isinstance(proto, LightGBMForecaster):
        return LightGBMForecaster(
            lags=proto.lags, rolling_windows=proto.rolling_windows, **proto.lgb_params
        )
    raise TypeError(f"Modelo não suportado para clonagem: {type(proto)}")
