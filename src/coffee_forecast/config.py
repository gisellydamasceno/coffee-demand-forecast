"""Carregamento e validação da configuração do projeto.

Usa Pydantic para validar o config.yaml. Vantagens:
- Parâmetros ficam fora do código (reprodutibilidade, sem hardcode).
- Validação de tipos falha cedo e com mensagem clara.
- Um único ponto de verdade para datas, horizonte, features, etc.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    kaggle_dataset: str
    primary_file: str
    experimental_file: str
    raw_dir: str
    processed_dir: str


class TargetConfig(BaseModel):
    primary: str
    date_col: str
    value_col: str
    product_col: str


class ForecastConfig(BaseModel):
    frequency: str
    seasonal_period: int
    horizon: int


class BacktestConfig(BaseModel):
    n_splits: int
    test_size: int
    min_train_size: int


class FeaturesConfig(BaseModel):
    lags: list[int]
    rolling_windows: list[int]


class MLflowConfig(BaseModel):
    experiment_name: str
    tracking_uri: str


class Config(BaseModel):
    data: DataConfig
    target: TargetConfig
    forecast: ForecastConfig
    backtest: BacktestConfig
    features: FeaturesConfig
    mlflow: MLflowConfig
    random_seed: int = Field(default=42)


def _project_root() -> Path:
    """Raiz do projeto (dois níveis acima deste arquivo: src/coffee_forecast/)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> Config:
    """Carrega a configuração a partir de config/config.yaml.

    Args:
        path: caminho opcional para um YAML alternativo.

    Returns:
        Objeto Config validado.
    """
    if path is None:
        path = _project_root() / "config" / "config.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# Caminhos absolutos derivados da raiz do projeto, para uso em todo o código.
PROJECT_ROOT = _project_root()
