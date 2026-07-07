# Federated Sensor Workflow

This project runs an end-to-end federated next-event prediction workflow on the
Chinook smart-home sensor dataset in `data/`.

Each participant is represented as one federated client. The main server sends a
query to all clients, each client filters its own local traces with an
LTL-over-finite-traces pattern, and only matching clients train a local
next-event model. Clients return model parameters and accuracy for visibility.

## Dataset Mapping

- `data/adl_error/*.csv` and `data/adl_noerror/*.csv` contain participant trial
  logs named `pXX.tN.csv`.
- One participant (`p01`, `p02`, ...) is one federated client.
- One trial file is one trace.
- One event token is `sensor_message`, for example `M07_ON`.
- By default, trials 1-4 train the local model and trial 5 evaluates it.

## Client Filtering

The SOWCompact-style filtering step is implemented with the retained
`fpm/ltl.py` evaluator. A query is evaluated locally on every client.

Examples:

- Empty query: all clients participate.
- `F(M01_ON)`: participant has at least one trace where `M01_ON` occurs.
- `F(M07_ON & X(F M23_ON))`: `M07_ON` occurs and later `M23_ON` occurs.
- `G(!M14_ON)`: `M14_ON` never occurs.

A client is included when at least one local trace satisfies the query, unless a
higher minimum matching trace fraction is supplied.

## Models

The main model is a local **decision tree** trained on prefix features from the
previous three sensor events. Matching clients return the fitted tree structure
(`rules`, `nodes`, `classes`) together with accuracy for visibility.

Other available baselines:

- `tree`: decision tree over the previous three events (default)
- `frequency`: always predicts the most frequent local event
- `markov`: predicts from first-order transition counts
- `logreg`: logistic regression over the previous three events, with Markov
  fallback when the local filtered data is too small

Returned parameters are JSON-serializable for visibility in the dashboard.

## One-Command Docker Run

Start the coordinator plus all participant clients:

```bash
docker compose up --build
```

For day-to-day development, use the dev overlay so Python changes reload
automatically without rebuilding images:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

After the first build, you can usually omit `--build` and just run the same
command again. Uvicorn watches `fpm/`, `shared/`, and `CASAS2/` inside each
container and restarts only the affected service when a file changes.

Rebuild images only when dependencies change (`requirements.txt`, `Dockerfile`):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Regenerate compose files when participants in `data/` change:

```bash
python3 scripts/generate_compose.py
```

Open the dashboard:

```text
http://localhost:8080
```

The dashboard lets you choose a model, enter an LTL filter, send the query, and
inspect the matched client group, returned accuracies, parameter summaries, and
raw logs.

## Docker Grouped Workflow

The dashboard also supports the CASAS2-style grouped workflow. Enable
`Behavioral grouping`, keep `CASAS2 80/20` selected for parity with
`CASAS2/outputs/grouped/`, and use an LTL filter such as `F(M01_ON)`.

The grouped request still broadcasts `/train` to every client for local
federated accuracies. The coordinator then fetches each client's `/profile`,
clusters matched clients, runs the pooled grouped evaluation from `DATA_DIR`,
and writes artifacts to `fpm/outputs/grouped/`.

Equivalent API request:

```bash
curl -X POST http://localhost:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"model":"tree","ltl":"F(M01_ON)","group":true,"n_clusters":"auto","eval_protocol":"casas2"}'
```

Grouped artifacts include:

- `cluster_assignments.json`
- `behavioral_profiles.csv`
- `cluster_summary.txt`
- `cluster_dendrogram.png`
- `group_*_workflow.json`
- `group_*_workflow.png`
- `grouped_comparison.csv`
- `grouped_comparison.txt`
- `ltl_filter.json`

For the reference parity check:

```bash
python3 CASAS2/grouped_main.py --ltl "F(M01_ON)"
diff CASAS2/outputs/grouped/grouped_comparison.csv fpm/outputs/grouped/grouped_comparison.csv
```

## Local Development

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run one client:

```bash
PARTICIPANT=p01 DATA_DIR=data uvicorn fpm.client:app --port 8001
```

Run the coordinator against that client:

```bash
CLIENTS=http://127.0.0.1:8001 DATA_DIR=data uvicorn fpm.server:app --port 8080
```

Send a query:

```bash
curl -X POST http://127.0.0.1:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"model":"tree","ltl":"F(M01_ON)"}'
```

Run a grouped query locally by starting the same single-client coordinator as
above, or by setting `CLIENTS` to the full client list generated in
`docker-compose.yml`. The grouped evaluator itself can be checked without
running clients:

```bash
python3 CASAS2/grouped_main.py --ltl "F(M01_ON)" --output-dir fpm/outputs/grouped
```

## Code Layout

- `fpm/dataset.py`: loads participant traces from `data/`.
- `fpm/ltl.py`: parses and evaluates LTLf filters.
- `fpm/models.py`: local next-event models and evaluation.
- `fpm/client.py`: FastAPI app for one participant client.
- `fpm/server.py`: FastAPI coordinator and dashboard API.
- `fpm/grouped.py`: shared grouped evaluation used by Docker and CASAS2 CLI.
- `fpm/static/index.html`: simple dashboard.
- `scripts/generate_compose.py`: writes `docker-compose.yml` from the dataset.
