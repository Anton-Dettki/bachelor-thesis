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

Optional verification and full pipeline steps:

```bash
python scripts/discover_individual_models.py
python scripts/run_pattern_query.py --scenario scenario1_shopping_mealprep
python scripts/run_social_mining.py --scenario scenario1_shopping_mealprep
python scripts/verify_pipeline.py
python scripts/verify_federation.py
python scripts/build_splits.py
python scripts/build_prefix_datasets.py
```

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

Temporal **75/25 train/validation splits** at the trace (day) level for next-activity prediction. The most recent 25% of days are held out for validation; no events from the same day appear in both splits.

Trace ordering adapts to the timestamp source automatically:

- **XES mode** (synthetic timestamps, all traces share one epoch): traces are ordered by `@@case_index`, the XES import order, which reflects chronological day order.
- **CSV mode** (real timestamps, distinct first-event times per trace): traces are ordered by their **first-event timestamp**, so validation days fall chronologically after training days.

```bash
# Requires event logs from step 1
python scripts/build_splits.py

# Optional: custom fraction or single subject
python scripts/build_splits.py --val-fraction 0.25 --subject 1
```

Per-subject splits support on-device training (each phone trains on its own `train.xes`). The global pooled split (`output/splits/global/`) unions all subjects' train days vs validation days (case ids namespaced as `subjectN:caseM`) for the global baseline model.

| Path | Description |
|------|-------------|
| `output/splits/subjectN/train.xes` | Per-phone training days (~75%) |
| `output/splits/subjectN/val.xes` | Per-phone validation days (~25%) |
| `output/splits/subjectN/split_manifest.json` | Split metadata (case ids, counts, bounds) |
| `output/splits/global/train.xes` | Pooled training log (all subjects) |
| `output/splits/global/val.xes` | Pooled validation log (all subjects) |

Example per-subject trace counts at default 25%: subject1 10/4, subject2 12/4, subject5 1/1 (only 2 days total).

### Prefix datasets

Turn each train/val split into prefix -> next-activity rows (window size 3, zero-padded). Each row has encoded columns `e0`, `e1`, `e2` (prefix activities) and `next_activity` (target).

Activities are encoded with a **declared canonical taxonomy** (`ACTIVITY_TAXONOMY` in [`fpm/loader.py`](fpm/loader.py)), not a vocabulary derived from each scope's train+val logs. This matters for two reasons:

- **No validation leakage:** the label space is fixed independent of the split, so a validation-only activity (e.g. `Mealpreparation` for subject5) never enters the vocabulary or the Markov smoothing denominator as a side effect of being in val. The model is trained purely on train statistics; unseen-in-train targets are simply predicted incorrectly, as they should be.
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
| `output/prefix/subjectN/prefix_manifest.json` | Sample counts and window size |
| `output/prefix/global/train.csv` | Pooled training prefix dataset |
| `output/prefix/global/val.csv` | Pooled validation prefix dataset |

### Local baseline predictors

Train and evaluate **local-only** next-activity predictors on prefix datasets. No HTTP, federation, or aggregation — each scope (subject or global) fits on its own `train.csv` and evaluates on `val.csv`.

**Predictors:**

| Predictor | Description |
|----------|-------------|
| **Frequency** | Predicts the single most common `next_activity` in training data (global majority class). Ignores prefix columns `e0`, `e1`, `e2`. |
| **Markov (order-1)** | Estimates `P(next \| e2)` from transition counts in training data. Uses Laplace smoothing; falls back to the marginal next-activity distribution when `e2` is `<PAD>` (id 0) or the context was unseen in training. |
| **Decision tree** | sklearn `DecisionTreeClassifier` on one-hot encoded prefix features (`e0`, `e1`, `e2` over the canonical vocabulary). Not additive for federation (unlike Markov counts). |

Requires prefix datasets from `build_prefix_datasets.py` first. Requires **scikit-learn** (see `requirements.txt`).

```bash
# Train and evaluate all predictors for all subjects + global
python scripts/train_local_models.py

# Optional: single subject or subset of predictors
python scripts/train_local_models.py --subject 1
python scripts/train_local_models.py --baselines frequency,markov,tree
python scripts/train_local_models.py --baselines tree
```

| Path | Description |
|------|-------------|
| `output/models/subjectN/metrics.json` | Accuracy, macro-F1, and top-3 accuracy per predictor |
| `output/models/subjectN/frequency.json` | Majority class id and next-activity counts |
| `output/models/subjectN/markov.json` | Order-1 transition counts and marginal counts (JSON-serializable, additive for future federation) |
| `output/models/subjectN/tree.json` | Decision tree metadata (params, classes, feature importances) |
| `output/models/subjectN/predictions.csv` | Per-row predictions (`case_id`, `position`, `baseline`, `y_true`, `y_pred`) |
| `output/models/comparison.csv` | Cross-scope comparison table (scope × model × metrics) |
| `output/models/comparison.json` | Same comparison data in JSON for thesis write-up |
| `output/models/global/` | Same artifacts for the pooled global scope |

Metrics (accuracy, macro-F1, top-3) are computed with numpy for consistency across all predictors. Macro-F1 averages per-class F1 over classes present in validation labels, with zero-division treated as 0.

### Global federated prediction

Each phone trains **additive** count-based models (Markov, Frequency) on its own prefix `train.csv` and exposes parameters over HTTP — raw event logs never leave the device. An aggregator collects `GET /predict/params/{model}` from every phone, **sums** the counts, and evaluates the merged global model. Because Markov/Frequency are pure sufficient statistics, the federated global model is **mathematically identical** to the centralized model trained on pooled `output/prefix/global/train.csv` (verified via `parity.json`).

The decision tree is **not** federated here: it is not additive and cannot be merged by summing counts.

Requires prefix datasets and (for local comparison rows) artifacts from `train_local_models.py`.

```bash
# In-process ASGI federation (default; no live servers needed)
python scripts/run_federated_prediction.py

# Against live phone servers (start run_phone_server.py per subject first)
for s in 1 2 3 4 5 6 7; do
  python scripts/run_phone_server.py --subject $s &
done
python scripts/run_federated_prediction.py --phones \
  http://127.0.0.1:8001 http://127.0.0.1:8002 ... http://127.0.0.1:8007
```

| Path | Description |
|------|-------------|
| `output/models/federated/markov.json` | Merged global Markov transition counts |
| `output/models/federated/frequency.json` | Merged global frequency counts |
| `output/models/federated/metrics.json` | Federated model metrics per scope |
| `output/models/federated/comparison.csv` | Local vs centralized vs federated comparison |
| `output/models/federated/parity.json` | Exact equality check: federated == centralized |

Phone API: `GET /predict/params/{model}` where `model` is `markov` or `frequency`.

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
