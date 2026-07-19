# Bachelor Thesis Codebase — Complete Technical Description

This document describes the full codebase of the thesis project: a **federated next-event prediction workflow on the CASAS "Chinook" smart-home ADL sensor dataset**, with **LTL-based client filtering** and **behavioral client grouping (clustering)**. It is written as source material for an AI research assistant and aims to be exhaustive about data handling, filtering decisions, feature engineering, model choices, hyperparameters, grouping logic, evaluation protocols, and underlying assumptions.

---

## 1. Research Idea in One Paragraph

Each study participant of a smart-home experiment is treated as one **federated client** that keeps its raw sensor data locally. A central coordinator broadcasts a query to all clients. The query contains (a) an **LTLf pattern** (Linear Temporal Logic over finite traces) that each client evaluates against its own local traces to decide whether it participates, and (b) a **model choice**. Matching clients train a local next-event prediction model and return only model parameters and accuracy metrics (never raw data). On top of this, the coordinator can perform **behavioral grouping**: it collects privacy-preserving behavioral profile vectors from the matched clients, clusters them, trains one model per cluster on pooled data, and compares grouped prediction against a global model, per-client local models, and several federated/discovery baselines. The research question is essentially whether grouping behaviorally similar clients yields better next-event prediction (and more compact workflow models) than one global model or purely local models.

---

## 2. Dataset

### 2.1 Source and layout

- Dataset: CASAS smart-home ADL (Activities of Daily Living) data recorded in the "Chinook" apartment testbed (floor plan images: `data/Chinook.png`, `data/Chinook_Cabinet.png`). Recordings are from 2008.
- Location in repo: `data/adl_noerror/` and `data/adl_error/`.
- File naming: `pXX.tN.csv`, e.g. `p01.t3.csv` = participant `p01`, trial/task `3`.
- Counts: `adl_noerror` contains 120 files (24 participants × 5 trials), `adl_error` contains 100 files (20 participants × 5 trials). **44 participants total**, each with exactly 5 trials, 220 trace files overall.
- `adl_noerror` = trials where the participant performed the scripted ADL tasks without errors; `adl_error` = trials containing errors in task execution. **Both folders are included by default** (`--include-errors` defaults to true everywhere); the `source` folder is kept as metadata but is not used as a model feature.

### 2.2 CSV schema

Each trial CSV has four columns: `date`, `time`, `sensor`, `message`. Example rows:

```
date,time,sensor,message
2008-02-27,12:43:27.416392,M08,ON
2008-02-27,12:43:33.108756,I08,ABSENT
```

- `M*` sensors are motion sensors (`ON`/`OFF`).
- `I*` sensors are item sensors (`PRESENT`/`ABSENT`), e.g. `I03`, `I08`.
- `D*` = door (OPEN/CLOSE), `AD1-*` = analog sensors (e.g. water flow), `ASTERISK` rows carry `START`/`END`/`STOP_INSTRUCT` trial control markers.

### 2.3 Mapping to the federated setting

- **One participant = one federated client.**
- **One trial file = one trace** (one case in process-mining terms).
- **One row = one event.** Event tokens take two equivalent surface forms depending on the pipeline: `M07=ON` (CASAS2-style label) or `M07_ON` (LTL-atom form). `shared/ltl_filter.py::event_to_ltl_token` converts between them (`=` and `-` become `_`, uppercased).
- Trial number is also called **task** (1–5); each trial corresponds to a scripted ADL task, so the trial index carries semantic meaning and is used both for train/test splitting (federated protocol) and as an optional model feature / profile breakdown.

---

## 3. Data Loading, Row Filtering, and Why Rows Are Ignored

Loading happens in three near-identical loaders (kept consistent deliberately): `CASAS2/main.py::load_events` (global pandas table), `fpm/dataset.py::load_trace` (per-participant `SensorTrace` objects for the federated clients), and `fpm/casas_client.py::load_local_traces` (CASAS2-style local client models). All of them:

1. Parse `date + " " + time` into a timestamp (`pd.to_datetime(..., format="mixed")`).
2. **Sort rows chronologically** with a stable sort (`sort_values("timestamp", kind="stable")`). Assumption: the timestamp order defines the true event order; stable sorting preserves file order for identical timestamps (which do occur, e.g. two rows at `12:43:31.491254`).
3. **Drop rows** via the shared predicate `shared/sensor_filter.py::should_skip_sensor`:

### 3.1 Ignored rows (exact rules and rationale)

- **Analog sensors (`AD1-*`)** are skipped when `skip_analog=True` (the default everywhere). Rationale: they emit continuous numeric readings rather than discrete state changes, which don't fit the discrete next-event token vocabulary.
- **`EXCLUDED_SENSORS = {"M06", "M10", "M21", "M22", "I09", "E01", "ASTERISK"}`**:
  - `M06`, `M10`, `M21`, `M22`, `I09`, `E01`: rare sensors with negligible coverage (each **< 0.5 % of sensor-level labels** across the corpus). They add label classes with almost no training support and hurt macro-F1 without contributing signal.
  - `ASTERISK`: rows are trial `START`/`END`/`STOP_INSTRUCT` **control markers injected by the experimenters**, not resident behavior, hence not predictable ADL events.
- Traces that end up with **fewer than 2 events** after filtering/abstraction are dropped entirely from sample building (you cannot form a (prefix → next event) pair from fewer than 2 events).
- Files not matching the `pXX.tN.csv` name pattern are ignored.

No other row filtering happens; there is no outlier removal, no deduplication beyond the consecutive-collapse abstraction described next, and no time-gap-based session splitting.

---

## 4. Event Abstraction (Two Views)

`shared/event_abstraction.py` defines two abstraction levels, and **every evaluation is run at both levels** ("abstraction views" in the dashboard/results):

- **`sensor` (default view)**: a raw token like `M07=ON` / `M07=OFF` (or `M07_ON`) is collapsed to the sensor ID `M07`. After collapsing, **consecutive duplicates are removed** (`collapse_consecutive_events`), so a burst of `M13 ON/OFF/ON/OFF` toggles becomes a single `M13` "sensor visit". This turns the trace into a sequence of visited sensors/areas — closer to an activity/location sequence. The prediction task becomes "which sensor fires next".
- **`raw` view**: the full low-level token including state (`M07=ON`, `I08=ABSENT`) is kept, no collapsing. The prediction task is "which exact sensor-state event comes next". This is a harder task with roughly twice the classes.

Known states recognized for collapsing: `ON, OFF, PRESENT, ABSENT, OPEN, CLOSE, START, END, STOP_INSTRUCT`.

---

## 5. Feature Engineering and Sample Construction

### 5.1 Prefix-window samples (main representation)

The core supervised representation (identical in `CASAS2/main.py`, `fpm/casas_client.py`, and `fpm/grouped.py`) is:

- Sliding **window of the previous `WINDOW = 3` events** predicts the next event.
- For each trace and each position `i ≥ 1`: prefix = last 3 events before `i` (left-padded with the `<PAD>` token when fewer than 3 exist), label = event at position `i`.
- Each sample also records `client_id`, `case_id` (e.g. `p01.t3`), `task` (trial number 1–5), `position`, and `split` (`train`/`test`).

**Encoding (CASAS2/grouped pipelines):** a global event vocabulary `event_map` maps each event string to an integer index. Feature dict per sample: `{"e0": id_of_prefix_token_0, "e1": ..., "e2": ...}` — i.e. the three prefix positions as **integer-encoded ordinal features** (not one-hot). Tokens absent from the vocabulary encode as `-1`. Optional extra features:
- `participant_id` (integer client index) — used **only by the global CASAS2 model**, so the global tree can personalize by participant.
- `task` (trial number) and `pos` (position in the trace) — added **only for grouped models in the `raw` abstraction view** (`include_task=True`), where the extra context measurably helps.

A `sklearn.DictVectorizer(sparse=False)` turns the dicts into matrices. Note the deliberate consequence: because features are ordinal integer IDs, decision-tree splits are on event-ID thresholds; this mirrors the original CASAS2 baseline design and is kept for parity.

**Encoding (legacy `fpm/models.py` client models):** feature dict is `{"prev_1": <token>, "prev_2": <token>, "prev_3": <token>, "prefix_len": <int>}` with **string-valued categorical features**, which the DictVectorizer one-hot encodes. Missing history positions get `<START>`.

### 5.2 Vocabulary construction

- CASAS2 protocol (`build_vocabs`): the event map is built from labels of **all** samples plus all non-PAD prefix tokens; the client map is built from train samples only. Test-only labels therefore have valid encodings (a shared label-space assumption).
- Federated protocol (`_build_federated_vocabs` in `fpm/grouped.py`): labels come from train+test samples but prefix-feature tokens come from **train samples only**.

### 5.3 Behavioral profile features (for grouping)

`shared/grouping.py::build_client_profile` builds one vector per client from its (LTL-matched, training-only) traces:

- **`tr:` transition features**: normalized first-order bigram frequencies `P(next | current)`-style counts over the flattened event stream (all bigrams normalized to sum to 1 per client).
- **`fr:` frequency features**: normalized unigram event frequency distribution.
- **`tk:` per-task features** (`include_task_breakdown=True` in the pipeline): the frequency distribution computed separately per trial/task 1–5 (`task{n}:{token}`), capturing "how the client performs each scripted task".

Profiles are assembled into a dense matrix (union of feature keys, missing = 0) and **each row is L2-normalized** before clustering. In the Docker workflow the profiles are computed **on-device** by each client (`GET /profile`) and only the numeric vectors are sent to the coordinator — the privacy story: raw traces never leave the client; the coordinator falls back to computing profiles centrally from `DATA_DIR` only for the CLI/parity path.

---

## 6. Client Filtering with LTLf (How the Group of Clients Is Selected)

### 6.1 The evaluator

`fpm/ltl.py` is a self-written, dependency-free **LTL-over-finite-traces (LTLf) tokenizer, recursive-descent parser, and recursive evaluator**. Supported syntax:

- Atoms: event tokens like `M07_ON`, `I03_PRESENT` (an atom holds at position i iff `trace[i] == atom`).
- Boolean: `!`, `&`, `|`, `->`, `<->`, `true`, `false`.
- Temporal: `X` (next), `F` (finally/eventually), `G` (globally/always), `U` (until), `R` (release), `W` (weak until, `a W b ≡ (a U b) | G a`), `M` (strong release, `a M b ≡ b U (a & b)`).
- Finite-trace semantics: `X φ` is false at the last position; `F`/`G`/`U` quantify over the remaining finite suffix.

Example queries (from `fpm/queries.py` / `CASAS2/queries.py`): empty string = all clients; `F(M01_ON)`; `F(M07_ON & X(F M23_ON))` ("M07 turns on and strictly later M23 turns on"); `G(!M14_ON)` ("M14 never turns on"); `F(I03_PRESENT)` (matches roughly half the clients).

### 6.2 Client-level matching rules

Two closely related mechanisms exist:

- **Per-client training endpoint (`fpm/client.py /train`)**: the query is evaluated on the client's *training pool* traces (trials ≠ 5). A client **participates if at least one training trace satisfies the query** and additionally `matched_fraction ≥ min_match_fraction` (a request parameter, default 0.0). Non-matching clients return `matched: false` and train nothing.
- **Grouped pipeline pre-filter (`shared/ltl_filter.py::filter_clients_by_ltl`)**: evaluated per training trace; a client is kept when it has **≥ `min_matching_traces` (default 1) satisfying training traces**. Only the *satisfying* traces feed the behavioral profiles and the group-Markov training pools, whereas the grouped decision trees train on all prefix samples of matched clients. Grouping requires **at least 2 matched clients**, otherwise the run aborts with an error.

An empty query matches everyone (fraction 1.0). LTL matching always happens on **sensor-state tokens in `M07_ON` form** on the raw (uncollapsed) event sequence of the training trials only — the held-out evaluation trial never influences participation.

---

## 7. Grouping / Clustering of Clients

Implemented in `shared/grouping.py::cluster_clients`, orchestrated by `fpm/grouped.py`:

1. Build the L2-normalized profile matrix from matched clients (Section 5.3).
2. **Choose K**: if `n_clusters="auto"`, run **KMeans** for each K in `DEFAULT_K_RANGE = (2, 3, 4, 5, 6)` (skipping K ≥ n_clients) and keep the K with the best **silhouette score**. With ≤ 2 clients, K=1.
3. **Cluster**: `KMeans(n_clusters=k, random_state=0, n_init=10)` by default; `AgglomerativeClustering(n_clusters=k)` is available via `method="agglomerative"` (the dendrogram artifact always uses Ward-linkage hierarchical clustering for visualization).
4. **Merge tiny clusters**: any cluster with fewer than `MIN_CLUSTER_SIZE = 2` members is merged into the nearest cluster by centroid Euclidean distance, then labels are re-indexed contiguously.
5. Report the final silhouette score and persist artifacts: `cluster_assignments.json`, `behavioral_profiles.csv`, `cluster_summary.txt`, `cluster_dendrogram.png`.

Reference run on the full dataset (no LTL filter, auto K): **K=2 clusters (21 vs 23 clients), silhouette ≈ 0.104** — i.e. weak but present behavioral structure.

---

## 8. Models and Hyperparameters

### 8.1 Global CASAS2 baseline (`CASAS2/main.py`)

- `sklearn.tree.DecisionTreeClassifier(max_depth=25, min_samples_leaf=10, criterion="entropy", random_state=0)`.
- Features: `e0,e1,e2` (integer event IDs of the 3-event prefix) **+ `participant_id`**.
- Trained on all participants pooled; this is the centralized "upper bound with data sharing" reference.
- Reference metrics (raw event level, 80/20 within-trace split, 44 participants, 53 classes, 6,616 train / 1,816 test samples): **accuracy 0.644, macro-F1 0.168, weighted-F1 0.617**. (The sample and class counts in `global_metrics.csv` match the raw abstraction view, not the collapsed sensor view.)

### 8.2 Per-cluster grouped trees (`fpm/grouped.py`)

- One `DecisionTreeClassifier` per cluster, same hyperparameters as the global model (`max_depth=25`, `min_samples_leaf=10`, entropy, `random_state=0`), trained only on prefix samples of that cluster's (LTL-matched) clients.
- Features: `e0,e1,e2`; in the `raw` view additionally `task` and `pos`.
- At prediction time each test sample is **routed to its client's cluster model**; clients without a cluster (LTL-excluded) fall back to the global model.
- **Hybrid prediction in the `raw` view** (`_predict_grouped_hybrid_raw`): the cluster tree's prediction is used only when its `predict_proba` confidence ≥ **`GROUPED_MARKOV_CONFIDENCE = 0.67`**; otherwise the sample falls back to the cluster's Markov model (or the global Markov model when the cluster Markov has never seen the context). Rationale: trees over ordinal event IDs are weak on rare raw-event contexts; the confident-tree/Markov-backoff hybrid recovers accuracy.

### 8.3 Variable-order Markov models (`shared/discovery_baseline.py::MarkovPredictor`)

- Counts n-gram context → next-event transitions up to `max_order` and **backs off** from the longest matching context down to bigram, then unigram (last event), then the global most-frequent event.
- Global Markov baseline: `use_trigram=True, max_order=4` trained on all training traces.
- Group Markov models: one per cluster, trained on the cluster's pooled LTL-matched traces **plus the global training pool** (smoothing so small clusters aren't starved); `max_order=4`. Routed prediction falls back to the global model when the group model lacks the context.
- Local client variant `casas_markov` (`fpm/casas_client.py`): trigram (`max_order=3`) fitted on that client's training traces only.

### 8.4 Local client models exposed by the FastAPI clients (`fpm/client.py`)

Selectable per query (`model` field): `casas_tree` (default), `casas_markov`, plus legacy models `tree`, `frequency`, `markov`, `logreg`.

- **`casas_tree`**: exact CASAS2 recipe locally — 3-event integer-ID prefix features, `DecisionTreeClassifier(max_depth=25, min_samples_leaf=10, entropy, random_state=0)`, no participant feature. If the local (filtered) data has < 2 label classes, it returns unfitted with a constant-invalid prediction (accuracy 0).
- **`casas_markov`**: local trigram Markov with backoff, as above.
- **`tree`** (legacy `fpm/models.py::DecisionTreeModel`): `DecisionTreeClassifier(max_depth=8, random_state=0)` over **one-hot categorical** features `prev_1..prev_3` + `prefix_len`; window 3. Falls back to a first-order Markov model when there are < 2 training pairs or < 2 label classes.
- **`frequency`**: always predicts the client's most frequent event (majority baseline).
- **`markov`**: first-order transition counts with frequency fallback.
- **`logreg`**: `LogisticRegression(max_iter=500, solver="lbfgs")` over the same one-hot prefix features, with the same Markov fallback for too-small data.

Clients return JSON-serializable parameters for dashboard inspection: tree structure (`rules` via `export_text`, node list with splits and class distributions, `n_nodes`, `n_leaves`, `classes`), Markov transition tables, or logreg coefficients. This "parameter visibility" is a demonstration feature, **not** a secure aggregation scheme.

### 8.5 Comparison baselines computed by the grouped evaluation

For every grouped run the pipeline reports (per abstraction view):

1. **Global** — pooled decision tree (Section 8.1 hyperparameters, with `participant_id`).
2. **Grouped (K=…)** — per-cluster trees (+ hybrid Markov in raw view), routed by cluster.
3. **Per-client local** — one tree per client (same tree hyperparameters, no participant feature); clients whose data has < 2 classes fall back to the global model.
4. **Local tree ensemble** — averages `predict_proba` across *all* per-client trees (a naive "model-sharing without data-sharing" ensemble).
5. **FedAvg SGD** — a genuine federated-averaging simulation: per client a `SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1, tol=None)` does one local epoch per round; coefficients/intercepts are averaged **weighted by client sample count** over **10 rounds**; features are the sparse one-hot prefix dicts scaled with `MaxAbsScaler`. This is the classic FL reference point.
6. **Markov global** and **Markov grouped** — the discovery-style baselines of Section 8.3.

Metrics per approach: **accuracy, macro-F1, weighted-F1**, plus workflow-graph size (nodes/edges) and training time. Per-cluster accuracy is reported for the grouped approach (computed only on test samples of LTL-matched clients).

Reference numbers (full data, no filter, sensor view, CASAS2 protocol): Global 0.481 accuracy vs Grouped (K=2) 0.494, with the grouped workflow graphs being smaller (20 nodes/41 edges vs 23/56). (The 0.644 figure in Section 8.1 is the global baseline *with* the `participant_id` feature on the raw-label task from `CASAS2/main.py` — the two tables answer different questions and shouldn't be conflated.)

---

## 9. Train/Test Split Protocols (Two, Deliberately)

Both protocols exist because the thesis compares against the original CASAS2 baseline while also wanting a realistic federated holdout:

- **`casas2` protocol (default in the dashboard/grouped runs)**: an **80/20 chronological split inside every trial trace** (`train_fraction=0.8`). The first 80 % of each trace's positions are training samples, the final 20 % test samples (with guards: split index at least 1 and at most len−1). All 5 trials contribute to both sides. This reproduces `CASAS2/outputs/grouped/` exactly (a `diff` parity check is part of the workflow).
- **`federated` protocol**: **trials 1–4 train, trial 5 (`EVAL_TRIAL = 5`) is the local held-out evaluation trace** per participant. This is a true "unseen episode" evaluation. If a client has no trial 5 after filtering, the code falls back to the 80/20 within-trace split.

The federated clients' `/train` endpoint always fits on the non-holdout trials; the LTL filter also only ever sees the training trials (no test leakage into participation decisions).

---

## 10. System Architecture

- **`fpm/client.py`** — FastAPI app, one instance per participant (env `PARTICIPANT`, `DATA_DIR`). Endpoints: `GET /health`, `GET /info` (trace/token stats), `POST /train` (LTL match check + local training + metrics + parameters at both abstraction levels), `GET /profile` (LTL-filtered behavioral profile vector + tokenized training traces).
- **`fpm/server.py`** — FastAPI coordinator + dashboard API. `POST /api/query` broadcasts `/train` to all clients concurrently (httpx, 180 s timeout), optionally fetches `/profile` from all clients, runs the grouped evaluation in a thread (300 s timeout), and returns a full JSON run record (kept in memory, last 25 runs). Serves artifacts and the static dashboard (`fpm/static/index.html`, at `http://localhost:8080`).
- **Docker**: `scripts/generate_compose.py` generates `docker-compose.yml` with one `client-pXX` service per participant discovered in `data/` (44 services) plus the `server`; a dev overlay adds live-reload volumes. Everything runs from one image (`Dockerfile`, deps in `requirements.txt`: fastapi, httpx, matplotlib, networkx, numpy, pandas, pydantic, scikit-learn, scipy, uvicorn).
- **CLI parity path**: `CASAS2/main.py` (global baseline) and `CASAS2/grouped_main.py` (grouped evaluation without any client processes) call the same shared code (`fpm/grouped.py::run_grouped_evaluation`), guaranteeing that the Docker workflow and the reproducible CLI produce identical artifacts.
- **`shared/workflow_graph.py`** — builds compact probabilistic **workflow graphs from model predictions**: for each test prefix, an edge (last-observed-event → predicted-next-event) is counted; edges with conditional probability < **0.05** are pruned. Graph size (nodes/edges/density) is used as a model-compactness metric, and PNG/JSON artifacts are produced globally and per cluster, at both abstraction levels.

### Code layout

| Path | Role |
|---|---|
| `CASAS2/main.py` | Centralized global decision-tree baseline + sample/vocab/vectorize primitives reused everywhere |
| `CASAS2/grouped_main.py` | CLI for the grouped evaluation (casas2 protocol) |
| `CASAS2/queries.py`, `fpm/queries.py` | Named example LTL queries |
| `fpm/dataset.py` | Trace loading, tokenization, LTL trace filtering, split protocols |
| `fpm/ltl.py` | LTLf tokenizer/parser/evaluator (pure Python) |
| `fpm/models.py` | Legacy local models (tree/frequency/markov/logreg) + evaluation |
| `fpm/casas_client.py` | CASAS2-style local client models (casas_tree/casas_markov) |
| `fpm/client.py` / `fpm/server.py` | Federated client / coordinator FastAPI apps |
| `fpm/grouped.py` | The full grouped evaluation pipeline (both protocols, both abstractions, all baselines) |
| `shared/sensor_filter.py` | Row-dropping rules (excluded sensors, analog skip) |
| `shared/event_abstraction.py` | sensor/raw abstraction, consecutive-collapse |
| `shared/grouping.py` | Behavioral profiles, KMeans/agglomerative clustering, auto-K, artifacts |
| `shared/ltl_filter.py` | Client-level LTL pre-filter, token mapping, artifacts |
| `shared/discovery_baseline.py` | Variable-order Markov predictor, group-Markov training/routing |
| `shared/evaluation.py` | ApproachResult, accuracy/macro-F1/weighted-F1, comparison tables |
| `shared/workflow_graph.py` | Predictive workflow graph construction, PNG/JSON export |
| `scripts/generate_compose.py` | Generates docker-compose files from the dataset |

---

## 11. Assumptions and Design Decisions (Explicit List)

1. **One participant = one client; one trial = one trace.** No session segmentation inside a trial; a trial is assumed to be one coherent task execution.
2. **Timestamps define event order**; stable sort resolves ties by original file order. No timing/duration features are used anywhere — the models are purely order-based (inter-event times are discarded).
3. **Error trials are included** in training and evaluation by default; erroneous behavior is considered part of the behavior to model, not noise.
4. **Rare sensors are noise**: sensors under 0.5 % coverage and experimenter control markers are excluded (Section 3.1). Analog sensor readings are excluded as non-discrete.
5. **A 3-event history window is sufficient context** for next-event prediction (WINDOW=3 everywhere; the Markov baselines go up to order 4 with backoff).
6. **Shared label space**: the event vocabulary is built centrally (including test labels); in a strict FL deployment this corresponds to a publicly known sensor vocabulary, which is realistic for a fixed smart-home installation.
7. **Federation is simulated**: clients are separate processes/containers with logically isolated data, but all read from the same mounted `data/` directory, and the pooled grouped evaluation runs centrally from `DATA_DIR`. There is no secure aggregation, no differential privacy, no encryption; model parameters are returned in plaintext deliberately ("visibility" for the dashboard). Privacy claims are limited to "raw traces never leave the client" in the profile/training exchange.
8. **LTL filtering is evaluated only on training trials**, so participation decisions cannot leak held-out data.
9. **Grouping needs ≥ 2 matched clients and ≥ 2 members per cluster** (tiny clusters are merged); K is capped at 6 and chosen by silhouette.
10. **Unseen tokens map to −1** in the integer encoding; unfittable local models (single-class data) fall back to Markov/frequency predictors or return unfitted with zero accuracy rather than crashing.
11. **Reproducibility**: all stochastic components are seeded (`random_state=0` for trees/KMeans, per-round seeds for FedAvg); CLI and Docker paths share one implementation and are parity-checked by diffing artifacts.

---

## 12. Outputs / Artifacts (What a Run Produces)

Written to `CASAS2/outputs/grouped/` (CLI) or `fpm/outputs/grouped/` (Docker volume):

- `ltl_filter.json`, `ltl_filter_summary.txt` — query, matched/excluded clients, matched case IDs.
- `behavioral_profiles.csv` — full profile vectors per client with cluster assignment.
- `cluster_assignments.json`, `cluster_summary.txt`, `cluster_dendrogram.png`.
- `grouped_comparison.csv/.txt` (sensor view) and `grouped_comparison_raw.csv/.txt` (raw view) — the approach-comparison tables with accuracy/macro-F1/weighted-F1/graph size/train time and per-cluster accuracies.
- `global_workflow.{json,png}`, `group_{id}_workflow.{json,png}` and `*_raw_*` variants — predictive workflow graphs.
- `CASAS2/outputs/global_metrics.csv` — the centralized baseline metrics.

The dashboard additionally shows per-client federated results: matched flag and fraction, per-abstraction accuracy/F1, parameter summaries (e.g. tree node/leaf/class counts or Markov state/transition counts), cluster membership, and raw JSON logs.
