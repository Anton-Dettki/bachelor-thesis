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

## Build a DFG

The pipeline reads sensor CSV traces directly from the zip archives in `dataset/`,
converts them to a pm4py event log, discovers a directly-follows graph (DFG), and
writes artifacts to `output/dfg/`.

```bash
source .venv/bin/activate

# All no-error traces (default)
python scripts/build_dfg.py

# Single ADL task, e.g. task 1 = phone call
python scripts/build_dfg.py --task 1

# Error variant with sparser graph
python scripts/build_dfg.py --variant adl_error --activity-coverage 0.5 --path-coverage 0.1
```

### Mapping (raw CSV → event log)

| Field | Column / rule |
|---|---|
| Case (trace) | CSV filename, e.g. `p16.t1` |
| Activity | `sensor` + `_` + `message`, e.g. `M07_ON` |
| Timestamp | `date` + `time` |

### Outputs

- `output/dfg/<variant>.png` — DFG visualization
- `output/dfg/<variant>.json` — edge frequencies and summary stats
- `output/dfg/<variant>_event_log.csv` — filtered event log used for discovery
