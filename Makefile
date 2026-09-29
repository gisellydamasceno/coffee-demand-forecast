# Atalhos de conveniência. No Windows, se não houver 'make',
# use os comandos Python equivalentes documentados no README.
.PHONY: setup ingest eda train test lint format mlflow-ui clean

setup:
	python -m pip install -e ".[dev]"
	pre-commit install

ingest:
	python -m coffee_forecast.data.load

eda:
	python -m coffee_forecast.pipeline --stage eda

train:
	python -m coffee_forecast.pipeline --stage train

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns

clean:
	rm -rf mlruns data/processed/* docs/figures/*.png
