# MLOps Platform

A self-hosted, mini Vertex AI / Azure ML: upload a CSV, get it auto-cleaned, train and compare several models with hyperparameter tuning, and serve the best one behind a prediction API — all tracked in MLflow.

## Features

- Upload any CSV dataset via a REST API or the Streamlit UI
- Automatic profiling (dtypes, missing values, cardinality) and cleaning (imputation, encoding, scaling) via scikit-learn pipelines
- Trains and compares Logistic/Linear Regression, Decision Tree, Random Forest, and XGBoost
- Hyperparameter tuning with `RandomizedSearchCV`
- Every run and model logged to MLflow (params, metrics, artifacts)
- Best model auto-selected and served via a `/predict` endpoint
- Dockerized (Postgres + MLflow + API + Streamlit UI) with a GitHub Actions CI pipeline

## Tech stack

Python · FastAPI · scikit-learn · XGBoost · MLflow · PostgreSQL · Streamlit · Docker · GitHub Actions

## Architecture

```
frontend (Streamlit)  ──HTTP──>  backend (FastAPI)  ──>  PostgreSQL   (datasets, run metadata)
                                        │
                                        └──────────────>  MLflow      (experiments, model artifacts)
```

## Project layout

```
backend/
  app/
    api/            REST endpoints (datasets, training, predict)
    core/           config + db session
    ml/pipelines/   preprocessing + training/tuning logic
    models/         SQLAlchemy models
    schemas/        Pydantic request/response schemas
  tests/
frontend/
  app.py            Streamlit UI: upload -> train -> results -> predict
docker/             Dockerfiles for backend + frontend
docker-compose.yml  Postgres + MLflow + api + frontend
.github/workflows/  CI: tests + image builds
```

## Running locally with Docker (recommended)

```bash
copy .env.example .env
docker compose up --build -d
```

`API_KEY` must be a unique local value in `.env`; the API and Streamlit services receive the same value. The included `.env` is ignored by Git and is suitable for a local-only setup.

| Service | Local URL / port |
| --- | --- |
| Streamlit | http://localhost:8503 |
| FastAPI | http://localhost:8002 |
| FastAPI docs | http://localhost:8002/docs |
| MLflow | http://localhost:5001 |
| PostgreSQL | localhost:5433 |

### Windows 11 and WSL2 prerequisites

- Docker Desktop with the WSL 2 engine enabled and integration enabled for your preferred Linux distribution.
- Docker Engine and Docker Compose V2 available from PowerShell (`docker info` and `docker compose version`).
- At least 8 GB of Docker-assigned memory is recommended for simultaneous MLflow, PostgreSQL, API and model-training workloads.
- Ensure ports 8501, 8000, 5000 and 5432 are available. The project can run from the Windows filesystem; moving it into the WSL filesystem is optional and can improve bind-mount performance.

### Operations

```bash
# Start or rebuild the complete stack
docker compose up --build -d

# Check readiness and view logs
docker compose ps
docker compose logs -f

# Stop containers without deleting persistent data
docker compose down

# Restart services while retaining datasets, metadata and MLflow artifacts
docker compose restart

# Run backend tests in the API container
docker compose exec api pytest tests -q
```

PostgreSQL data is held in the named `pgdata` volume. Uploaded datasets and MLflow artifacts are retained in `data/uploads/` and `mlruns/`; do not use `docker compose down -v` unless you explicitly intend to remove the database volume.

### Troubleshooting

- If `docker info` cannot reach the engine, start Docker Desktop and confirm WSL 2 integration in Docker Desktop settings.
- If a service remains unhealthy, inspect it with `docker compose logs <service>`. Compose waits for PostgreSQL, MLflow and the API health checks before starting dependent services.
- If a port is already bound, identify the owning process before changing the host port mapping; do not stop unrelated applications.
- To reset only application code after edits, rebuild the affected service: `docker compose up --build -d api` or `docker compose up --build -d frontend`.

## Running without Docker

Requires Python 3.11 (newer ML packages don't yet ship wheels for 3.12/3.13 on all platforms) and a running Postgres instance.

```bash
# backend
cd backend
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy ..\.env.example .env   # then edit DATABASE_URL / MLFLOW_TRACKING_URI
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API overview

| Endpoint | Method | Description |
|---|---|---|
| `/datasets/upload` | POST | Upload a CSV, get back column profiling |
| `/datasets/{id}` | GET | Fetch dataset metadata |
| `/training/run` | POST | Train + tune all models, log to MLflow, pick the best |
| `/training/{run_id}` | GET | Fetch a training run's status/metrics |
| `/training/predict` | POST | Predict using a completed run's best model |

## Testing

```bash
docker compose exec api pytest tests -q
```

## Roadmap

- API key auth on the prediction endpoint
- Model registry promotion (staging -> production) via MLflow Model Registry
- Auto-generated PDF/HTML comparison report
- React frontend as an alternative to Streamlit

## License

MIT — see [LICENSE](LICENSE).
