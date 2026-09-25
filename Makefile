PY ?= python
export PYTHONPATH := src:.

.PHONY: setup data pipeline sql quality features train powerbi api app test lint mlflow-ui clean

setup:            ## install everything
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

data:             ## download 12 months of BTS data + reference files (~360 MB)
	$(PY) scripts/download_data.py

pipeline:         ## ingest -> SQL -> DQ gate -> features -> all models -> exports
	$(PY) -m flightops.pipeline

sql:
	$(PY) -m flightops.sqlrunner staging marts

quality:
	$(PY) -m flightops.quality

features:
	$(PY) -m flightops.features

train:
	$(PY) -m flightops.models.delay_classifier
	$(PY) -m flightops.models.delay_regression
	$(PY) -m flightops.models.cancellation
	$(PY) -m flightops.models.forecasting
	$(PY) -m flightops.models.clustering
	$(PY) -m flightops.models.anomaly
	$(PY) -m flightops.models.explain

powerbi:          ## star-schema exports for Power BI + app marts
	$(PY) -m flightops.powerbi_export

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

app:
	streamlit run app/streamlit_app.py --server.port 8501

test:
	$(PY) -m pytest -q

lint:
	ruff check .

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

clean:
	rm -rf data/interim data/warehouse mlruns .pytest_cache .ruff_cache
