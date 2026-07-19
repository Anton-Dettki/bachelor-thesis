# CASAS Smart Home Dataset Context

This project uses the CASAS Smart Home "scripted activities, with and without activity errors" dataset, published on Zenodo as record 15712834:

https://zenodo.org/records/15712834
    
DOI: `10.5281/zenodo.15712834`

The dataset was created by Diane Cook and contributors from the CASAS smart-home research group at Washington State University. It contains ambient sensor event logs collected in the CASAS Chinook smart apartment while participants performed scripted activities of daily living.

The dataset is designed for smart-home activity recognition and activity-quality/error-detection research. It records timestamped sensor events from a real apartment testbed rather than simulated traces. Each row in the CSV files represents one sensor event with four fields: `date`, `time`, `sensor`, and `message`. The raw event stream therefore captures the temporal order of motion, item-use, door, water, burner, and phone-use events during each scripted activity.

The Zenodo record provides four files:

- `adl_noerror.zip`: activity trials performed without scripted errors.
- `adl_error.zip`: activity trials where a specific scripted error is introduced.
- `Chinook.png`: floor plan of the smart apartment with sensor locations.
- `Chinook_Cabinet.png`: close-up of the instrumented Chinook cabinet.

The license is Creative Commons Attribution 4.0 International, so the data can be reused with proper attribution. The Zenodo page asks users to cite Cook & Schmitter-Edgecombe, 2009, "Assessing the quality of activities in a smart environment," Methods of Information in Medicine.

## Activity Structure

Each participant performs five activities of daily living. In the no-error condition, the activities are:

1. Phone call: the participant goes to the dining-room phone, looks up a number, calls it, listens to cooking instructions, and writes a summary.
2. Wash hands: the participant washes and dries their hands at the kitchen sink.
3. Cook: the participant prepares oatmeal using water, pot, oats, bowl, raisins, and brown sugar.
4. Eat: the participant brings the oatmeal and medicine container to the dining room and eats.
5. Clean: the participant takes dishes to the sink and cleans them with soap and water.

In the error condition, each task includes one scripted error:

1. Phone call: the participant dials the wrong number first and must redial.
2. Wash hands: the participant leaves the water running.
3. Cook: the participant leaves the burner on.
4. Eat: the participant forgets the medicine container.
5. Clean: the participant cleans dishes without water.

Files are named as `pXX.tN.csv`, where `pXX` is the participant ID and `tN` is the task number. For example, `p01.t1.csv` is participant `p01`, task 1.

## Sensors

The dataset contains multiple categories of sensors:

- Motion sensors such as `M01`, `M07`, `M13`, etc., with messages like `ON` and `OFF`.
- Item sensors such as `I01` to `I08`, representing objects such as oatmeal, raisins, brown sugar, bowl, measuring spoon, medicine container, pot, and phone book. These typically use `PRESENT` and `ABSENT`.
- Door sensors such as `D01`, using `OPEN` and `CLOSE`.
- Analog sensors such as `AD1-A`, `AD1-B`, and `AD1-C`, representing sink water and burner readings.
- A phone-use or marker sensor represented as `ASTERISK` in the code.

In this repository, some sensors are filtered out before modeling. The shared filter in `shared/sensor_filter.py` excludes rare or non-predictive sensors: `M06`, `M10`, `M21`, `M22`, `I09`, `E01`, and `ASTERISK`. By default it also skips analog `AD1*` sensors. This means most modeling uses discrete motion, item, and door events rather than continuous analog values or trial markers.

## Local Dataset State In This Repository

The local extracted dataset is stored in `data/`. In this repository, the extracted files contain:

- 220 CSV traces total.
- 44 participant IDs total.
- 120 no-error traces in `data/adl_noerror/`.
- 100 error traces in `data/adl_error/`.
- 5 task files per participant ID.
- 11,586 raw CSV rows before filtering.
- 8,550 retained events after the repository's default sensor filtering.
- 30 unique raw sensor IDs before filtering.
- Loaded trace lengths after filtering range from 9 to 179 events, with an average of about 38.9 events per trace.

There is an important local-vs-Zenodo detail: the Zenodo record describes 20 participants in each dataset, but the local extracted repository data has 44 participant IDs. The no-error folder contains 24 participant IDs, while the error folder contains 20 different participant IDs. The IDs do not overlap between `adl_noerror` and `adl_error`. The code therefore treats these as 44 separate participants/clients, not as paired before/after or no-error/error versions of the same people.

## How The Code Loads The Dataset

Dataset loading is implemented in `fpm/dataset.py`. The code treats the CASAS data as participant-specific event traces.

The central data object is `SensorTrace`, which contains:

- `participant`: the participant/client ID, such as `p01`.
- `trial`: the task number, such as `1` to `5`.
- `source`: the source folder, either `adl_noerror` or `adl_error`.
- `path`: the CSV path.
- `events`: a tuple of processed event tokens.
- `event_count`: number of retained events.

The loader reads every CSV using pandas, checks for the required columns `date`, `time`, `sensor`, and `message`, combines `date` and `time` into a timestamp, sorts rows chronologically, filters unwanted sensors, and converts each remaining row into an event token.

In the federated loader, raw CSV rows are converted into tokens with this format:

```text
SENSOR_MESSAGE
```

Examples:

```text
M07_ON
M07_OFF
I08_PRESENT
D01_OPEN
```

This conversion is done by `sensor_token(sensor, message)` in `fpm/dataset.py`. It uppercases the sensor/message pair and replaces non-alphanumeric characters with underscores so the tokens are easy to use in LTL formulas and models.

`participant_ids()` scans `data/adl_*/*.csv` and extracts participant IDs from filenames. `load_participant()` loads all traces for one participant across both folders. `load_all()` loads a dictionary mapping each participant ID to its list of `SensorTrace` objects.

## How Participants Map To Federated Clients

The project's main interpretation is federated learning over smart-home behavior. Each participant ID is treated as one federated client. This is described in `README.md` and implemented in the client/server code.

The mapping is:

- One participant, such as `p01`, is one client.
- One CSV file, such as `p01.t3.csv`, is one trace.
- One row in the CSV becomes one event token after filtering.
- A participant's traces remain local to that participant/client.

The Docker setup creates one FastAPI client service per participant. The helper script `scripts/generate_compose.py` scans the dataset and writes Docker Compose services for the participants.

## Train/Test Splits

The repository supports two evaluation protocols.

The default federated protocol uses task 5 as the holdout evaluation trace. This is controlled by `EVAL_TRIAL = 5` in `fpm/dataset.py`. In this protocol:

- Trials/tasks 1-4 are training traces.
- Trial/task 5 is the evaluation trace.
- If a participant does not have a task-5 trace in a filtered subset, fallback splitting logic can use an internal chronological split.

The CASAS2-style protocol uses an 80/20 chronological split inside each trace. This is implemented in `CASAS2/main.py`, `fpm/casas_client.py`, and `fpm/grouped.py`. In this protocol:

- Every trace is sorted by timestamp.
- The first 80% of events in each trace are training data.
- The final 20% are test data.
- This is used for parity with the CASAS2 baseline scripts.

## Prediction Task

The machine learning task is next-event prediction. The models learn to predict the next sensor or activity event given a short history of previous events.

For each trace, the code creates training examples from prefixes:

```text
previous events -> next event
```

The default window is the previous three events. For example, if a trace contains:

```text
M07, M13, I08, D01
```

then a sample may use:

```text
prefix = (M07, M13, I08)
label = D01
```

In the CASAS2 baseline, prefixes are padded with `<PAD>` if fewer than three previous events are available. Samples are represented by the `Sample` dataclass in `CASAS2/main.py`, with fields for client ID, case ID, task, position, prefix, label, and split.

## Event Abstraction Levels

The repository evaluates the dataset at two abstraction levels, implemented in `shared/event_abstraction.py`.

The `raw` abstraction preserves low-level sensor state tokens. Examples:

```text
M07=ON
M07=OFF
I08=PRESENT
D01=CLOSE
```

The `sensor` abstraction collapses event states to sensor or activity IDs. Examples:

```text
M07=ON  -> M07
M07=OFF -> M07
I08=PRESENT -> I08
D01=CLOSE -> D01
```

The sensor abstraction also removes consecutive duplicate events. This means a rapid toggle sequence from the same sensor can become a single sensor-level activity. The sensor-level view is the main default view because it focuses on movement or object-use sequence rather than low-level state transitions.

## LTL Filtering

A major part of the project is filtering clients using Linear Temporal Logic over finite traces. This is implemented in `fpm/ltl.py`, `fpm/dataset.py`, and `shared/ltl_filter.py`.

An LTL query is evaluated locally against each participant's training traces. A participant/client is included only if enough of its local training traces satisfy the query.

Examples used in the README include:

```text
F(M01_ON)
F(M07_ON & X(F M23_ON))
G(!M14_ON)
```

Meaning:

- `F(M01_ON)`: eventually sensor event `M01_ON` occurs.
- `F(M07_ON & X(F M23_ON))`: `M07_ON` occurs and later `M23_ON` occurs.
- `G(!M14_ON)`: `M14_ON` never occurs.

For the federated API, each client filters its own traces in `fpm/client.py`. The server sends the query to every client, but the filtering is local. A client returns whether it matched, the fraction of matching traces, and the number of matched traces. This models a privacy-aware federated workflow where the coordinator does not need raw event logs from all participants to decide local participation.

For grouped evaluation, `shared/ltl_filter.py` applies the same idea over prepared training traces and records:

- matched clients,
- excluded clients,
- matched case IDs,
- matched traces per client,
- minimum number of matching traces required.

The grouped workflow writes this information to `ltl_filter.json` and `ltl_filter_summary.txt`.

## Models Used With The Dataset

The local client models are implemented in `fpm/models.py`. The main model is a decision tree over prefix features from the previous three events. Other local baselines include:

- `frequency`: always predicts the most frequent event in the local training data.
- `markov`: predicts based on first-order transition counts.
- `logreg`: logistic regression over prefix features, with Markov fallback if too little data is available.
- `tree`: decision tree over prefix features.

The client trains only after it passes the LTL filter. It then evaluates on the local evaluation split and returns accuracy, counts, and JSON-serializable model parameters. For tree models, the returned parameters include feature names, classes, number of nodes, number of leaves, exported rules, and tree node metadata.

The CASAS2 baseline in `CASAS2/main.py` trains a global decision tree classifier using scikit-learn. It converts prefix features into dictionaries, encodes events as integer classes, vectorizes features with `DictVectorizer`, and evaluates with accuracy, macro F1, and weighted F1.

## Federated Server Workflow

The coordinator is implemented in `fpm/server.py`. It exposes a dashboard and an API endpoint `/api/query`.

A typical query contains:

- model type,
- LTL filter,
- minimum match fraction,
- whether behavioral grouping is enabled,
- evaluation protocol,
- whether baselines should be included.

The server sends `/train` requests to all participant clients. Each client independently loads its participant data, filters traces by LTL, trains if matched, evaluates, and returns metrics. The server aggregates the responses into a run object containing matched clients, client metrics, model summaries, grouped results if requested, and output artifact links.

## Behavioral Grouping Workflow

The grouped workflow is implemented in `fpm/grouped.py`. It uses the CASAS dataset to compare global, local, grouped, Markov, and federated-style baselines.

The grouped workflow proceeds as follows:

1. Load and preprocess the CASAS traces.
2. Build prefix-to-next-event samples.
3. Apply the LTL filter to decide which clients are eligible for grouping.
4. Build behavioral profiles from each matched client's filtered training traces.
5. Cluster matched clients using their behavioral profiles.
6. Train grouped models using clients assigned to each cluster.
7. Compare grouped models against global and baseline models.
8. Write artifacts such as cluster assignments, behavioral profiles, workflow graphs, and comparison CSV/TXT files.

The grouped evaluation supports both `sensor` and `raw` abstraction views. It can also generate workflow graphs from predicted transitions. These graphs summarize predicted event transitions and are saved as JSON and PNG artifacts.

## Important Interpretation For Research

In this project, the CASAS dataset is not used primarily for activity classification. Instead, it is transformed into a sequential prediction benchmark for federated process/workflow modeling.

The key research interpretation is:

- Each participant is a separate data owner/client.
- Each participant has a small set of private smart-home traces.
- The coordinator can issue temporal behavior queries using LTL.
- Only clients whose local traces satisfy the query participate.
- Models predict future sensor/activity events from previous events.
- Behavioral grouping clusters similar participants before training grouped models.
- Results compare global learning, local learning, grouped learning, Markov baselines, and simulated federated averaging.

This makes the dataset useful for studying privacy-aware, query-driven federated workflow mining and next-event prediction in smart-home environments.

## Main Source References

- Dataset record: https://zenodo.org/records/15712834
- Dataset loader: `fpm/dataset.py`
- Local client workflow: `fpm/client.py`
- Coordinator/server workflow: `fpm/server.py`
- Models: `fpm/models.py`
- Grouped evaluation: `fpm/grouped.py`
- CASAS2 baseline: `CASAS2/main.py`
- Event abstraction: `shared/event_abstraction.py`
- Sensor filtering: `shared/sensor_filter.py`
- LTL filtering: `shared/ltl_filter.py`
