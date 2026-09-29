"""Funções de análise exploratória (EDA) reutilizáveis.

A EDA serve a dois públicos:
- Técnico: fundamentar decisões de modelagem (sazonalidade, estacionariedade).
- Negócio: gerar insights acionáveis (horário de pico, mix de produtos).

Cada função retorna dados/figuras reutilizáveis pelos notebooks. As figuras
são salvas em data/processed/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from coffee_forecast.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de qualidade: tipos, nulos e % de nulos por coluna."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_null": df.isna().sum(),
            "pct_null": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )


def adf_test(series: pd.Series) -> dict[str, float]:
    """Teste de estacionariedade Augmented Dickey-Fuller.

    p-value < 0.05 sugere série estacionária (útil para decidir diferenciação
    em modelos ARIMA). Retorna estatística, p-value e nº de lags usados.
    """
    stat, pvalue, usedlag, nobs, *_ = adfuller(series.dropna())
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(usedlag),
        "n_obs": int(nobs),
        "is_stationary_5pct": bool(pvalue < 0.05),
    }


def stl_decompose(series: pd.Series, period: int = 7):
    """Decomposição STL (tendência + sazonalidade + resíduo)."""
    result = STL(series, period=period, robust=True).fit()
    return result


def plot_series(series: pd.Series, title: str, filename: str, ylabel: str = "") -> Path:
    """Plota e salva uma série temporal."""
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(12, 4))
    series.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Data")
    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def dow_hour_heatmap(df_raw: pd.DataFrame, filename: str = "heatmap_dow_hour.png") -> Path:
    """Heatmap de volume por dia da semana x hora (insight de negócio)."""
    _ensure_figures_dir()
    data = df_raw.copy()
    dt = pd.to_datetime(data["datetime"], errors="coerce")
    data["dow"] = dt.dt.dayofweek
    data["hour"] = dt.dt.hour
    pivot = data.pivot_table(index="dow", columns="hour", values="money", aggfunc="size").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    # Escala teal alinhada ao template (claro -> teal escuro): quanto mais
    # escuro, maior o volume. Custom para combinar com a identidade visual.
    from matplotlib.colors import LinearSegmentedColormap

    teal_cmap = LinearSegmentedColormap.from_list(
        "brand_teal", ["#F2F7F6", "#8FC3C1", "#0E7C7B", "#0A5E5D"]
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=teal_cmap)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][: len(pivot.index)])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Hora do dia")
    ax.set_title("Volume de transações por dia da semana x hora")
    fig.colorbar(im, ax=ax, label="nº de transações")
    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def product_pareto(df_raw: pd.DataFrame, product_col: str = "coffee_name") -> pd.DataFrame:
    """Pareto de produtos: participação e participação acumulada no volume."""
    counts = df_raw[product_col].value_counts()
    pareto = pd.DataFrame({"transactions": counts})
    pareto["pct"] = (pareto["transactions"] / pareto["transactions"].sum() * 100).round(2)
    pareto["cum_pct"] = pareto["pct"].cumsum().round(2)
    return pareto
