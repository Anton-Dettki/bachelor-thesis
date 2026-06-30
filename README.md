# bachelor-thesis

Federated Process Mining (SOWCompact-style) on the **dailylog2016** ADL dataset (7 subjects, one trace per day).

Early-week DFG experiments on the Chinook sensor dataset live under [`first-week-tests/`](first-week-tests/) ([`first-week-tests/DATASET.md`](first-week-tests/DATASET.md)).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Graphviz must be installed on the system (e.g. `brew install graphviz`).

Place the dataset at `dailylog2016_dataset/` with layout `subjectN/data/activity.xes` (and `activity.csv`). By default the pipeline reads **`activity.xes`**, matching the original [SOWCompact server app](https://bitbucket.org/spilab/serverapp/src/master/).

### Timestamp source: `xes` vs `csv`

Event logs can be built from two timestamp sources, configured in [`fpm/settings.py`](fpm/settings.py) (`TIMESTAMP_SOURCE`) or overridden per run with `build_event_logs.py --timestamp-source {xes,csv}`:

| Mode | Timestamps | Case ids | Use for |
|------|-----------|----------|---------|
| `xes` (default) | Synthetic: every trace starts at `2015-01-01 00:00:00` with 1-second increments | `caseN` | **SOWCompact Section 7 reproduction** (`compare_sowcompact.py`) — keep this default |
| `csv` | Real `attr_endtime` from `activity.csv` (e.g. `11.03.15 08:07:44`) | `dayN` | Predictive / temporal next-activity work (real event and day ordering) |

Notes:

- Case ids differ between modes (`caseN` for XES vs `dayN` for CSV) because they come from different source files.
- `activity.csv` contains a few records not present in `activity.xes`, so **CSV mode is not metric-compatible** with the paper's Section 7 reference values. `compare_sowcompact.py` should stay on XES mode.
- After changing the timestamp source you must rebuild downstream artifacts:

```bash
python scripts/build_event_logs.py --timestamp-source csv   # or xes
python scripts/build_splits.py
python scripts/build_prefix_datasets.py
```

#### Methodology (thesis)

Two timestamp modes exist because the thesis combines **structural FPM reproduction** with **temporal next-activity prediction**. They answer different questions and must not be mixed without rebuilding.

**CSV is the standard for all predictive/temporal artifacts.** Real `attr_endtime` values from `activity.csv` give chronologically correct first-event ordering per day-trace, so the 80/20 temporal train/validation split holds out genuinely later days and avoids temporal leakage. This is what [`scripts/run_phase3.py`](scripts/run_phase3.py) produces by default (`--timestamp-source csv`, keep repeats).

**XES mode is for SOWCompact Section 7 structural reproduction only.** Synthetic per-second increments preserve within-trace event order, and pm4py XES import order (`@@case_index`) reflects chronological day order even though every trace shares the same synthetic epoch. That makes XES mode methodologically sound for DFG structure, arc weights, and compression metrics — but it is **not** a substitute for real wall-clock temporal evaluation.

**Provenance is recorded in generated artifacts.** Each `output/event_logs/subjectN/log_stats.json` and `output/splits/subjectN/split_manifest.json` (and the global split manifest) includes `"timestamp_source": "csv"` or `"xes"`, so downstream tables can be traced to the mode that produced them. If you change mode, rebuild event logs and all downstream steps (splits, prefix datasets, models).

## Run everything from scratch

All generated artifacts go under `output/` (gitignored). To rerun as if starting fresh:

```bash
source .venv/bin/activate
cd "/Users/anton/Semester 6 /bachelor-thesis"

rm -rf output

# Step 1 — per-subject event logs (use --no-collapse-repeats for paper-aligned metrics)
python scripts/build_event_logs.py --no-collapse-repeats

# Step 2–5 — baseline + all Section 7 scenarios + comparison table
python scripts/compare_sowcompact.py --run
```

Optional SOWCompact verification:

```bash
python scripts/discover_individual_models.py
python scripts/run_pattern_query.py --scenario scenario1_shopping_mealprep
python scripts/run_social_mining.py --scenario scenario1_shopping_mealprep
python scripts/verify_pipeline.py
python scripts/verify_federation.py
```

## Phase 3 — run everything

Run the full next-activity prediction pipeline (splits → prefix datasets → local + federated + group models) in one command:

```bash
source .venv/bin/activate
cd "/Users/anton/Semester 6 /bachelor-thesis"

# Optional: start from a clean output/ (also removes SOWCompact artifacts)
# rm -rf output

python scripts/run_phase3.py
```

**Standard settings:** CSV timestamps (`--timestamp-source csv`) and **keep repeats** (no `--collapse-repeats`). Real `attr_endtime` values give chronologically correct train/val splits and day ordering for predictive work. The SOWCompact block above uses XES timestamps instead — the two modes share `output/event_logs/` and are not interchangeable without rebuilding.

**SOWCompact caveat:** Phase 3 overwrites event logs with CSV-mode data. To return to Section 7 reproduction, rebuild with:

```bash
python scripts/build_event_logs.py --timestamp-source xes --no-collapse-repeats
```

**Resume mid-pipeline** with skip flags, e.g. after event logs already exist:

```bash
python scripts/run_phase3.py --skip-event-logs
```

Other useful flags: `--scenarios scenario2_no_sport,scenario3_movement_transportation`, `--min-train-traces 5`, `--window 3`, `--collapse-repeats` (override default keep-repeats).

**Steps executed** (see individual sections below for manual runs):

1. `build_event_logs.py` — CSV timestamps, keep repeats
2. `build_splits.py` — temporal 80/20 train/val per subject + global
3. `build_prefix_datasets.py` — prefix → next-activity rows
4. `train_local_models.py` — frequency, Markov, decision tree per scope
5. `run_federated_prediction.py` — global federated Markov/Frequency + FedAvg logistic regression
6. `build_group_prefix_datasets.py` — LTL-group prefix datasets (viable scenarios)
7. `run_group_prediction.py` — per-scenario group vs global vs local comparison

**Comparison artifacts** (primary thesis tables):

| Path | Description |
|------|-------------|
| `output/models/comparison.csv` | Local baselines: scope × model × accuracy / macro-F1 / top-3 |
| `output/models/federated/comparison.csv` | Global: local vs centralized vs federated vs ensemble |
| `output/models/federated/parity.json` | Federated params and metrics == centralized global |
| `output/models/group/<scenario>/comparison.csv` | Group: centralized, federated, global, local variants |
| `output/models/group/<scenario>/parity.json` | Group federated == group centralized |

## SOWCompact FPM Pipeline

Recreating the SOWCompact pipeline to implement Federated Process Mining.

### Tasks

- [x] **Data Collection**: Generate event logs from device data
- [x] **Local Discovery**: Alpha+ on device (`Phone` class)
- [x] **Filter Logic**: LTL pattern query resolver

**Communication layer**

- [x] FastAPI phone server per subject
- [x] Aggregator broadcasts LTL queries and merges matching traces

**Server logic**

- [x] In-process orchestrator and HTTP federation
- [x] Heuristic Miner on integrated logs (pm4py, `dependency_thresh` default)

### Pipeline commands (step by step)

```bash
source .venv/bin/activate

# Step 1: build per-subject event logs from activity.xes
python scripts/build_event_logs.py --no-collapse-repeats

# Step 2: discover individual Alpha+ models
python scripts/discover_individual_models.py

# Step 3: filter traces by LTL query
python scripts/run_pattern_query.py --scenario scenario1_shopping_mealprep
python scripts/run_pattern_query.py --query "G(!Sport)" --write-filtered

# Step 4: aggregate + Heuristic Miner (SOW model)
python scripts/run_social_mining.py --scenario scenario1_shopping_mealprep
python scripts/run_social_mining.py --query "G(!Sport)"

# Step 5 (Phase D): federated mining over HTTP
for s in 1 2 3 4 5 6 7; do
  python scripts/run_phone_server.py --subject $s &
done
python scripts/run_federated_mining.py --scenario scenario1_shopping_mealprep

# Compare to SOWCompact paper Section 7 (regenerates baseline + scenarios)
python scripts/compare_sowcompact.py --run
python scripts/compare_sowcompact.py --with-quality   # fitness/precision (slow)
```

### Outputs

| Path | Description |
|------|-------------|
| `output/event_logs/subjectN/event_log.xes` | Per-subject event logs (from `activity.xes`) |
| `output/individual/subjectN/model.pnml` | Individual Alpha+ models |
| `output/filtered/<query>/subjectN/filtered_log.xes` | Traces matching a query |
| `output/sow/<query>/integrated_log.xes` | Federated integrated log |
| `output/sow/<query>/model.pnml` | SOW model (Heuristic Miner) |
| `output/baseline_full/true/` | Full-log baseline (`F(true)`) |
| `output/comparison/sowcompact_comparison.json` | Structured comparison vs paper |

Integrated logs prefix each case id with the subject (`subject1:day1`) so days from different users do not collide.

### Paper comparison metrics

`scripts/compare_sowcompact.py` compares against SOWCompact Section 7 reference values:

- **`arcs`** — directly-follows graph edge count (`dfg_arcs`), not Petri-net arc count
- **`sum_arc_weights`** — sum of DFG transition frequencies from the heuristics net
- **`integrated_log_kb`** — serialized XES size (yours is ~2× the paper due to namespaced case ids, timestamps, and pm4py encoding; compare **% of baseline**, not absolute KB)

With `--no-collapse-repeats` and XES source logs, the full-log baseline should align with the paper on structure: **12 activities**, **116 DFG arcs**, **1309** arc-weight sum.

## Phase 3 — Predictive Splits (Next-Activity Prediction)

> **Quick start:** `python scripts/run_phase3.py` runs all Phase 3 steps end-to-end (see [Phase 3 — run everything](#phase-3--run-everything) above).

Temporal **80/20 train/validation splits** at the trace (day) level for next-activity prediction. The most recent 20% of days are held out for validation; no events from the same day appear in both splits.

Trace ordering adapts to the timestamp source automatically:

- **XES mode** (synthetic timestamps, all traces share one epoch): traces are ordered by `@@case_index`, the XES import order, which reflects chronological day order.
- **CSV mode** (real timestamps, distinct first-event times per trace): traces are ordered by their **first-event timestamp**, so validation days fall chronologically after training days.

```bash
# Requires event logs from step 1
python scripts/build_splits.py

# Optional: custom fraction or single subject
python scripts/build_splits.py --val-fraction 0.20 --subject 1
```

Per-subject splits support on-device training (each phone trains on its own `train.xes`). The global pooled split (`output/splits/global/`) unions all subjects' train days vs validation days (case ids namespaced as `subjectN:caseM`) for the global baseline model.

| Path | Description |
|------|-------------|
| `output/splits/subjectN/train.xes` | Per-phone training days (~80%) |
| `output/splits/subjectN/val.xes` | Per-phone validation days (~20%) |
| `output/splits/subjectN/split_manifest.json` | Split metadata (case ids, counts, bounds) |
| `output/splits/global/train.xes` | Pooled training log (all subjects) |
| `output/splits/global/val.xes` | Pooled validation log (all subjects) |

Example per-subject trace counts at default 20%: subject1 11/3, subject2 13/3, subject5 1/1 (only 2 days total).

### Prefix datasets

Turn each train/val split into prefix -> next-activity rows (window size 3, zero-padded). Each row has encoded columns `e0`, `e1`, `e2` (prefix activities) and `next_activity` (target). The builder automatically adds leakage-free numeric features from the current prefix only: timestamp/calendar signals, real duration features when CSV start/end times are available, compact history counts, recency, transition/window statistics, and subject/case context. These are used by the decision tree and logistic-regression ML baselines and are intentionally **not** consumed by the activity-only Markov/Frequency baselines, so federated parity stays comparable.

Activities are encoded with a **declared canonical taxonomy** (`ACTIVITY_TAXONOMY` in [`fpm/loader.py`](fpm/loader.py)), not a vocabulary derived from each scope's train+val logs. This matters for two reasons:

- **No validation leakage:** the label space is fixed independent of the split, so validation rows whose target activity is absent from training are removed before encoding. `Shopping` and `Sport` targets are also removed from train and validation because those classes are too sparse for reliable training.
- **Shared id space for federation:** every subject and the global scope use the *same* activity -> id mapping, so per-subject Markov transition counts (serialized by integer id) remain summable for a future federated "sum counts" aggregation.

`vocab.json` is therefore identical across all scopes.

```bash
# Requires splits from the step above
python scripts/build_prefix_datasets.py

# Optional: custom window or single subject
python scripts/build_prefix_datasets.py --window 3 --subject 1
```

| Path | Description |
|------|-------------|
| `output/prefix/subjectN/train.csv` | Encoded prefix samples from training days |
| `output/prefix/subjectN/val.csv` | Encoded prefix samples from validation days |
| `output/prefix/subjectN/vocab.json` | Canonical activity name -> integer id mapping (identical for every scope) |
| `output/prefix/subjectN/prefix_manifest.json` | Sample counts, window size, and feature-column metadata |
| `output/prefix/global/train.csv` | Pooled training prefix dataset |
| `output/prefix/global/val.csv` | Pooled validation prefix dataset |

### Local baseline predictors

Train and evaluate **local-only** next-activity predictors on prefix datasets. No HTTP, federation, or aggregation — each scope (subject or global) fits on its own `train.csv` and evaluates on `val.csv`.

**Predictors:**

| Predictor | Description |
|----------|-------------|
| **Frequency** | Predicts the single most common `next_activity` in training data (global majority class). Ignores prefix columns `e0`, `e1`, `e2`. |
| **Markov (order-1)** | Estimates `P(next \| e2)` from transition counts in training data. Uses Laplace smoothing; falls back to the marginal next-activity distribution when `e2` is `<PAD>` (id 0) or the context was unseen in training. Uses only the last prefix position despite window size 3 — see order-3 variant below for a symmetric baseline. |
| **Markov (order-3)** | Estimates `P(next \| e0, e1, e2)` from tuple-context transition counts. Same Laplace smoothing and marginal fallback when any prefix slot is `<PAD>` or the context was unseen. Count-based and additive for federation; symmetric with the decision tree input window. |
| **Logistic regression** | Numpy softmax regression on one-hot encoded prefix features plus automatic numeric temporal/duration/history/context columns. Federated with iterative FedAvg (`logreg`), not exact count summation. |
| **Decision tree** | sklearn `DecisionTreeClassifier` on one-hot encoded prefix features (`e0`, `e1`, `e2` over the canonical vocabulary) plus automatic numeric temporal/duration/history/context columns. Not additive for federation (unlike Markov counts). |

Requires prefix datasets from `build_prefix_datasets.py` first. Requires **scikit-learn** (see `requirements.txt`).

```bash
# Train and evaluate all predictors for all subjects + global
python scripts/train_local_models.py

# Optional: single subject or subset of predictors
python scripts/train_local_models.py --subject 1
python scripts/train_local_models.py --baselines frequency,markov,markov_order3,logreg,tree
python scripts/train_local_models.py --baselines markov,markov_order3
python scripts/train_local_models.py --baselines logreg
python scripts/train_local_models.py --baselines tree
```

| Path | Description |
|------|-------------|
| `output/models/subjectN/metrics.json` | Accuracy, macro-F1, and top-3 accuracy per predictor |
| `output/models/subjectN/frequency.json` | Majority class id and next-activity counts |
| `output/models/subjectN/markov.json` | Order-1 transition counts and marginal counts (JSON-serializable, additive for federation) |
| `output/models/subjectN/markov_order3.json` | Order-3 tuple-context transition counts and marginal counts (additive for federation) |
| `output/models/subjectN/logreg.json` | Local softmax logistic regression weights |
| `output/models/subjectN/tree.json` | Decision tree metadata (params, classes, feature importances) |
| `output/models/subjectN/predictions.csv` | Per-row predictions (`case_id`, `position`, `baseline`, `y_true`, `y_pred`) |
| `output/models/subjectN/confusion_matrices/<model>.csv/.png` | Validation confusion matrix per predictor |
| `output/models/subjectN/learning_curves/<model>.csv/.png` | Validation metrics over increasing training-set fractions |
| `output/models/comparison.csv` | Cross-scope comparison table (scope × model × metrics) |
| `output/models/comparison.json` | Same comparison data in JSON for thesis write-up |
| `output/models/global/` | Same artifacts for the pooled global scope |

Metrics (accuracy, macro-F1, top-3) are computed with numpy for consistency across all predictors. Macro-F1 averages per-class F1 over classes present in validation labels, with zero-division treated as 0.

### Global federated prediction

Each phone trains **additive** count-based models (Markov order-1, Markov order-3, Frequency) on its own prefix `train.csv` and exposes parameters over HTTP — raw event logs never leave the device. An aggregator collects `GET /predict/params/{model}` from every phone, **sums** the counts, and evaluates the merged global model. Because Markov/Frequency are pure sufficient statistics, the federated global model is **mathematically identical** to the centralized model trained on pooled `output/prefix/global/train.csv` (verified via `parity.json`).

The `logreg` model uses iterative FedAvg instead of additive sufficient statistics. The aggregator initializes global softmax weights, sends them to `POST /predict/fedavg/logreg/update`, averages returned local weights by each phone's `n_train`, and repeats for `--rounds` rounds. `parity.json` marks exact parity as not applicable for `logreg`; compare centralized vs federated by metrics.

The decision tree is **not** federated here: it is not additive and cannot be merged by summing counts. Pass `--models tree` to include a centralized global tree and an **ensemble** row alongside the federated models.

**Prediction-level ensemble (default):** with `--ensemble` (default on), each subject's independently trained local model predicts on the shared `output/prefix/global/val.csv` rows. The server averages each model's predicted probability vectors with **equal weight** (soft vote), takes the argmax as the final next-activity prediction, and writes a `variant=ensemble` row next to `variant=centralized` and `variant=federated` in `comparison.csv`. This is the thesis-friendly comparison between "one model trained on all data centrally" and "combine on-device model predictions without sharing raw logs". Disable with `--no-ensemble`.

Requires prefix datasets and (for local comparison rows) artifacts from `train_local_models.py`.

```bash
# In-process ASGI federation (default; no live servers needed)
python scripts/run_federated_prediction.py
python scripts/run_federated_prediction.py --models logreg --rounds 50 --local-epochs 1
python scripts/run_federated_prediction.py --models frequency,markov,markov_order3,logreg,tree

# Disable prediction-level ensemble if you only want param-merge federated rows
python scripts/run_federated_prediction.py --no-ensemble

# Against live phone servers (start run_phone_server.py per subject first)
for s in 1 2 3 4 5 6 7; do
  python scripts/run_phone_server.py --subject $s &
done
python scripts/run_federated_prediction.py --phones \
  http://127.0.0.1:8001 http://127.0.0.1:8002 ... http://127.0.0.1:8007
```

| Path | Description |
|------|-------------|
| `output/models/federated/markov.json` | Merged global order-1 Markov transition counts |
| `output/models/federated/markov_order3.json` | Merged global order-3 Markov transition counts |
| `output/models/federated/frequency.json` | Merged global frequency counts |
| `output/models/federated/logreg.json` | FedAvg logistic regression weights |
| `output/models/federated/metrics.json` | Federated and ensemble model metrics per scope |
| `output/models/federated/comparison.csv` | Local vs centralized vs federated vs ensemble comparison |
| `output/models/federated/confusion_matrices/<model>_<variant>.csv/.png` | Global validation confusion matrices for centralized, federated, and ensemble variants |
| `output/models/federated/parity.json` | Exact equality check: federated == centralized |

Phone APIs: `GET /predict/params/{model}` for additive `markov`, `markov_order3`, and `frequency`; `POST /predict/fedavg/logreg/update` for FedAvg logistic regression.

**Phase 4 note:** Compact predictive workflow graphs built from order-1 Markov counts reflect **last-event transitions only** (`P(next | e2)`). Graphs from order-3 counts reflect **full-prefix contexts** (`P(next | e0, e1, e2)`) and may require context-state nodes or projection to a simple activity-to-activity DFG. Choose the variant that matches the thesis comparison you report.

### Group-based prediction (LTL)

Users are grouped by **behavioral similarity** using LTL scenario queries from `fpm/queries.py` (`SCENARIO_QUERIES`). A group is defined at **day level**: each trace (one day) is individually filtered by the query; only matching days in the temporal train/val splits contribute to that group's prefix datasets and federated parameter merge. Canonical vocabulary and no validation leakage follow the same rules as global prefix build.

**Hybrid pipeline:** centralized group prefix datasets are the parity reference and train the decision tree; the federated HTTP path mirrors the thesis narrative — phones filter their own train prefix rows by LTL (via split `train.xes` case ids) and return additive Markov/Frequency counts or FedAvg `logreg` updates. Raw event logs never leave devices. Non-matching phones respond with **empty counts** for additive models or zero-row FedAvg updates for `logreg`.

**Scenario viability** (matching train / val traces, pooled over all subjects):

| Scenario | Train traces | Val traces | Notes |
|----------|-------------:|-----------:|-------|
| `scenario2_no_sport` | 40 | 14 | Primary scenario; all 7 subjects contribute |
| `scenario3_movement_transportation` | 23 | 8 | Viable |
| `scenario1_shopping_mealprep` | 9 | 4 | Small; subjects 5–7 contribute nothing |
| `scenario4_social_eat_transport` | 11 | 3 | Marginal val set; metrics noisy |
| `scenario5_no_eat_no_social` | 1 | 2 | Skipped by default (`--min-train-traces 5`) |

Requires prefix datasets, splits, and (for global federated comparison rows) artifacts from `run_federated_prediction.py`.

```bash
# Build group prefix datasets (all viable scenarios)
python scripts/build_group_prefix_datasets.py

# Run group prediction for one scenario
python scripts/run_group_prediction.py --scenario scenario2_no_sport
python scripts/run_group_prediction.py --scenario scenario3_movement_transportation

# Custom query or threshold
python scripts/build_group_prefix_datasets.py --scenario scenario1_shopping_mealprep --min-train-traces 1
python scripts/run_group_prediction.py --query "G(!Sport)"
```

| Path | Description |
|------|-------------|
| `output/prefix/group/<scenario>/train.csv` | Pooled group train prefix rows (matching days only) |
| `output/prefix/group/<scenario>/val.csv` | Pooled group validation prefix rows |
| `output/prefix/group/<scenario>/membership.json` | Matching case ids per subject/split (audit trail) |
| `output/models/group/<scenario>/markov.json` | Group federated order-1 Markov counts |
| `output/models/group/<scenario>/markov_order3.json` | Group federated order-3 Markov counts |
| `output/models/group/<scenario>/frequency.json` | Group federated frequency counts |
| `output/models/group/<scenario>/logreg.json` | Group FedAvg logistic regression weights |
| `output/models/group/<scenario>/tree.json` | Group centralized decision tree (not federated) |
| `output/models/group/<scenario>/comparison.csv` | Group: centralized, federated, global, local, and local_group variants |
| `output/models/group/<scenario>/parity.json` | Group federated == group centralized check |

**Comparison variants** (`comparison.csv` columns: `scope`, `model`, `variant`, metrics):

| Variant | Description |
|---------|-------------|
| `group_centralized` | Model trained on pooled group `train.csv` |
| `group_federated` | Merged additive counts from phones with LTL filter |
| `global_centralized` | Global model evaluated on group val set |
| `global_federated` | Global federated model evaluated on group val set |
| `local` | Per-subject model trained on the subject's **full** train split, evaluated on that subject's group val rows |
| `local_pooled` | Sample-weighted average of per-subject `local` metrics |
| `local_group` | Per-subject model trained on **LTL-filtered** train rows (same filter as phone-side federated path), evaluated on group val rows |
| `local_group_pooled` | Sample-weighted average of per-subject `local_group` metrics |

**Decision tree** is group-centralized only: unlike Markov/Frequency counts, a fitted tree is not additive and cannot be merged by summing parameters across phones.

Phone API for group federation: `GET /predict/params/{model}?query=<ltl>` — optional `query` filters train prefix rows by LTL-matching case ids from the phone's split.

**Known limitations:** sparse groups (scenario1, scenario4) yield small val sets; subject5 has only 2 days total; phones with zero matching train days contribute empty counts (recorded in `metrics.json` contributions).

### Phase 3 smoke verification

After running the prediction pipeline, verify artifacts and parity before Phase 4:

```bash
python scripts/verify_prediction_pipeline.py
```

Checks performed:

- Prefix datasets and local model metrics exist for all subjects and global scope
- Global federated `parity.json` reports `params_equal` and `metrics_equal` for additive models
- Group prefix row counts match `membership.json` (expanded via per-subject prefix `case_id` counts)
- Federated and group `comparison.csv` files contain the expected model/variant rows (including `local_group` / `local_group_pooled`)

Use `--skip-group` to validate only global/local prediction artifacts when group datasets have not been built yet.

## LTL Pattern Query Language

The pattern query resolver uses **Linear Temporal Logic over finite traces (LTLf)**.
Each trace is one day of activities (ordered by timestamp). A subject "meets" the pattern when enough day-traces satisfy it (default: at least 1).

### Operators

| Symbol | Name | Meaning (on a finite day-trace) |
|--------|------|----------------------------------|
| `!` | Not | Negation: the sub-formula is false at the current position |
| `X` | Next | The sub-formula holds at the **next** event |
| `F` | Finally (Eventually) | The sub-formula holds at **some** event from here onward |
| `G` | Globally (Always) | The sub-formula holds at **every** event from here onward |
| `U` | Until | Left formula holds until the right formula becomes true |
| `R` | Release | Right formula holds until (and including when) left becomes true |
| `W` | Weak Until | Left until right, or left forever if right never occurs |
| `M` | Strong Release | Right until left and right both hold |
| `&` | And | Both sub-formulas hold at the current position |
| `\|` | Or | At least one sub-formula holds at the current position |
| `->` | Implies | If left holds, then right must hold (at the same position) |
| `<->` | Iff | Left and right have the same truth value |
| `()` | Grouping | Override default precedence |

Constants: `true`, `false`

Activity names are identifiers matching the event log, e.g. `Shopping`, `Mealpreparation`, `EatingDrinking`.

### Operator precedence (high → low)

1. Atoms, `true`/`false`, parentheses
2. Unary: `!`, `X`, `F`, `G`
3. Binary temporal: `U`, `R`, `W`, `M`
4. `&`
5. `|`
6. `->`
7. `<->`

### Common patterns

| Intent | Example query |
|--------|----------------|
| Activity A occurs on some day | `F(A)` |
| Activity A never occurs on a day | `G(!A)` |
| A then later B (same day) | `F(A & X(F B))` |
| A immediately followed by B | `F(A & X B)` |
| Two global constraints | `G(!A) & G(!B)` |

### Validation scenario queries

Predefined queries in `fpm/queries.py` (`SCENARIO_QUERIES`), aligned with the SOWCompact paper (Section 7). The paper’s `→` between activities means “eventually followed by”, encoded as `X(F …)` not bare `X`:

| Scenario | Query |
|----------|-------|
| 1 – Shopping before meal prep | `F(Shopping & X(F Mealpreparation))` |
| 2 – Day without sport | `G(!Sport)` |
| 3 – Movement then transportation | `F(Movement & X(F Transportation))` |
| 4 – Socializing → eating → transport | `F(Socializing & X(F(EatingDrinking & X(F Transportation))))` |
| 5 – No eating/drinking and no socializing | `G(!EatingDrinking) & G(!Socializing)` |

Run a predefined scenario:

```bash
python scripts/run_pattern_query.py --scenario scenario1_shopping_mealprep
```

Compare against paper reference values:

```bash
python scripts/compare_sowcompact.py --run
```

### LTL operator reference

| Operator | Symbol |
|----------|--------|
| Finally (Eventually) | `F` |
| Globally (Always) | `G` |
| Next | `X` |
| Until, Release, Weak Until, Strong Release | `U`, `R`, `W`, `M` |
| Logical connectives | `!`, `&`, `\|`, `->`, `<->` |
