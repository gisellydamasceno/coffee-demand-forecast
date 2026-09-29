"""Pipeline de orquestração ponta a ponta.

Estágios:
- eda:   ingestão + preparação + geração de figuras e relatório de qualidade.
- train: backtesting comparativo (baselines vs modelos) + seleção do melhor +
         treino final na série completa + previsão futura com intervalos.
- all:   executa eda e train em sequência.

Uso:
    python -m coffee_forecast.pipeline --stage all
    python -m coffee_forecast.pipeline --stage train --no-mlflow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from coffee_forecast.config import PROJECT_ROOT, load_config
from coffee_forecast.data.load import load_raw
from coffee_forecast.data.preprocess import build_product_daily, prepare


def _processed_dir(cfg) -> Path:
    d = PROJECT_ROOT / cfg.data.processed_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_eda(cfg, download: bool = True) -> pd.DataFrame:
    """Ingestão + preparação + artefatos de EDA (figuras e relatório)."""
    from coffee_forecast import eda

    raw = load_raw(cfg, download=download)
    daily = prepare(raw, cfg)

    # Persiste dados processados (reprodutibilidade e reuso nos notebooks).
    out = _processed_dir(cfg)
    daily.to_parquet(out / "daily.parquet")
    build_product_daily(raw, cfg).to_parquet(out / "product_daily.parquet")

    # Artefatos visuais e relatórios
    eda.plot_series(daily["revenue"], "Receita diária", "revenue_daily.png", "R$")
    eda.plot_series(
        daily["revenue"].rolling(7).mean(),
        "Receita diária (média móvel 7 dias)",
        "revenue_ma7.png",
        "R$",
    )
    eda.dow_hour_heatmap(raw)

    adf = eda.adf_test(daily["revenue"])
    pareto = eda.product_pareto(raw, cfg.target.product_col)
    quality = eda.data_quality_report(raw)

    report_path = out / "eda_summary.txt"
    lines = [
        "=== EDA SUMMARY ===",
        f"Período: {daily.index.min().date()} a {daily.index.max().date()} ({len(daily)} dias)",
        f"Receita total: R$ {daily['revenue'].sum():,.2f}",
        f"Receita média/dia: R$ {daily['revenue'].mean():,.2f}",
        f"Transações média/dia: {daily['transactions'].mean():.2f}",
        "",
        "--- Teste ADF (estacionariedade) ---",
        f"ADF stat: {adf['adf_stat']:.4f} | p-value: {adf['p_value']:.4f} | "
        f"estacionária(5%): {adf['is_stationary_5pct']}",
        "",
        "--- Pareto de produtos ---",
        pareto.to_string(),
        "",
        "--- Qualidade dos dados ---",
        quality.to_string(),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return daily


def run_train(cfg, daily: pd.DataFrame | None = None, use_mlflow: bool = True) -> None:
    """Backtesting comparativo + treino final do melhor modelo + previsão."""
    from coffee_forecast.models.evaluate import run_backtest
    from coffee_forecast.models.train import build_models

    if daily is None:
        processed = _processed_dir(cfg) / "daily.parquet"
        daily = pd.read_parquet(processed) if processed.exists() else run_eda(cfg)

    target = cfg.target.primary
    report = run_backtest(daily, cfg, target=target, use_mlflow=use_mlflow)
    summary = report.summary()

    out = _processed_dir(cfg)
    report.to_frame().to_csv(out / "backtest_folds.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv")

    # Seleciona o melhor modelo pela métrica de decisão (menor MASE).
    best_name = summary.index[0]

    # Treino final na série completa + previsão futura com intervalo (se houver).
    y_full = daily[target].astype(float)
    horizon = cfg.forecast.horizon
    models = build_models(cfg)

    future_lines = [f"Melhor modelo (menor MASE médio): {best_name}", "", summary.to_string(), ""]
    if best_name in models:
        model = models[best_name].fit(y_full)
        yhat = model.predict(horizon)
        future_idx = pd.date_range(y_full.index[-1] + pd.Timedelta(days=1), periods=horizon)
        fc = pd.DataFrame({"forecast": yhat}, index=future_idx)
        if hasattr(model, "predict_interval"):
            lo, hi = model.predict_interval(horizon)
            fc["lower"] = lo
            fc["upper"] = hi
        fc.to_csv(out / "future_forecast.csv")
        future_lines.append("--- Previsão futura ---")
        future_lines.append(fc.round(2).to_string())
    else:
        future_lines.append("(Melhor modelo é um baseline; previsão futura não persistida.)")

    (out / "results_summary.txt").write_text("\n".join(future_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coffee demand forecast pipeline")
    parser.add_argument("--stage", choices=["eda", "train", "all"], default="all")
    parser.add_argument("--no-mlflow", action="store_true", help="Desativa o tracking MLflow")
    parser.add_argument("--no-download", action="store_true", help="Usa cópia local dos dados")
    args = parser.parse_args()

    cfg = load_config()
    daily = None
    if args.stage in ("eda", "all"):
        daily = run_eda(cfg, download=not args.no_download)
    if args.stage in ("train", "all"):
        run_train(cfg, daily=daily, use_mlflow=not args.no_mlflow)
    print("pipeline done")


if __name__ == "__main__":
    main()
