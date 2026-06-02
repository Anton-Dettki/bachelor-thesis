# bachelor-thesis

Dataset: https://zenodo.org/records/15712834

See [DATASET.md](DATASET.md) for sensor and activity documentation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Graphviz must be installed on the system (e.g. `brew install graphviz`).

## SOWCompact FPM Pipeline

Recreating the SOWCompact pipeline to implement Federated Process Mining.

### Tasks

- [x] **Data Collection**: Generate event logs from device data
- [x] **Local Discovery**: Alpha algorithm builds behavioral models on device (in the `Phone` class)
- [x] **Filter Logic**: Pattern Query Resolver that interprets LTL operators

**Est. communication layer**

- [ ] Deploy a mobile API for communication
- [x] Ensure aggregator can send LTL strings to multiple devices and merge incoming XES traces into one integrated log (in-process; HTTP deferred)

**Server Logic**

- [x] **Environment**: Setup a server (in-process orchestrator for Phase C)
- [x] **Heuristic Miner**: Apply to integrated data

### Pipeline commands

```bash
source .venv/bin/activate

# Step 1: build per-subject event logs from activity.csv
python scripts/build_event_logs.py

# Step 2: discover individual Alpha+ models from event logs
python scripts/discover_individual_models.py

# Step 3: run an LTL pattern query (filter matching traces)
python scripts/run_pattern_query.py --scenario scenario1_shopping_mealprep
python scripts/run_pattern_query.py --query "G(!Sport)" --write-filtered

# Step 4: aggregate matching traces and discover SOW model (Heuristic Miner)
python scripts/run_social_mining.py --scenario scenario1_shopping_mealprep
python scripts/run_social_mining.py --query "G(!Sport)"
```

Outputs:

- `output/event_logs/subjectN/event_log.xes` — generated event logs
- `output/individual/subjectN/model.pnml` — individual Alpha+ models
- `output/filtered/<query>/subjectN/filtered_log.xes` — traces matching a query
- `output/sow/<query>/integrated_log.xes` — federated log from all matching phones
- `output/sow/<query>/model.pnml` — SOW model (Heuristic Miner on integrated log)

Integrated logs prefix each trace case id with the subject (`subject1:day1`) so days from different users do not collide.

## LTL Pattern Query Language

The pattern query resolver uses **Linear Temporal Logic over finite traces (LTLf)**.
Each trace is one day of activities (ordered by timestamp). A query is evaluated
per trace; a subject "meets" the pattern when enough day-traces satisfy it
(default: at least 1).

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

### SOWCompact scenario queries (ASCII)

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

> **Note:** The paper sometimes writes `F(Shopping -> Mealpreparation)`, but `->` is
> material implication at a **single** event (one activity cannot be two things at once).
> For "A then later B", use `F(A & X(F B))` instead.

### Paper ↔ implementation mapping

| SOWCompact (Section 5) | This project |
|------------------------|--------------|
| Finally (◇) | `F` |
| Globally (□) | `G` |
| Next | `X` |
| Until, Release, Weak Until, Strong Release | `U`, `R`, `W`, `M` |
| Logical connectives | `!`, `&`, `\|`, `->`, `<->` |
