"""Ingestão de dados: download do dataset e validação de schema.

Decisões de engenharia:
- A ingestão é isolada da preparação: uma etapa baixa/valida, outra transforma.
- Validamos o schema na entrada (fail-fast). Em produção, dados fora do
  contrato esperado devem quebrar cedo, não silenciosamente adiante.
- Guardamos uma cópia local em data/raw para reprodutibilidade offline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from coffee_forecast.config import PROJECT_ROOT, Config, load_config

# Contrato de schema esperado para o arquivo principal (index_1.csv).
EXPECTED_COLUMNS = {"date", "datetime", "cash_type", "card", "money", "coffee_name"}


def download_dataset(cfg: Config) -> Path:
    """Baixa o dataset via kagglehub e copia os CSVs para data/raw.

    Returns:
        Caminho da pasta data/raw com os arquivos locais.
    """
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(cfg.data.kaggle_dataset))
    raw_dir = PROJECT_ROOT / cfg.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    for csv in cache_path.rglob("*.csv"):
        shutil.copy2(csv, raw_dir / csv.name)

    return raw_dir


def validate_schema(df: pd.DataFrame, expected: set[str] = EXPECTED_COLUMNS) -> None:
    """Valida que o DataFrame contém as colunas esperadas.

    Raises:
        ValueError: se faltar alguma coluna do contrato.
    """
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema inválido. Colunas ausentes: {sorted(missing)}. "
            f"Colunas encontradas: {sorted(df.columns)}"
        )


def load_raw(cfg: Config, filename: str | None = None, download: bool = True) -> pd.DataFrame:
    """Carrega o arquivo bruto (baixando se necessário) e valida o schema.

    Args:
        cfg: configuração do projeto.
        filename: nome do CSV a carregar (default: arquivo primário do config).
        download: se True, baixa/atualiza a cópia local antes de ler.

    Returns:
        DataFrame bruto validado.
    """
    filename = filename or cfg.data.primary_file
    raw_dir = PROJECT_ROOT / cfg.data.raw_dir
    local_path = raw_dir / filename

    if download or not local_path.exists():
        download_dataset(cfg)

    df = pd.read_csv(local_path)
    validate_schema(df)
    return df


if __name__ == "__main__":
    config = load_config()
    path = download_dataset(config)
    frame = load_raw(config, download=False)
    print(f"Dados baixados em: {path}")
    print(f"Linhas: {len(frame)} | Colunas: {list(frame.columns)}")
