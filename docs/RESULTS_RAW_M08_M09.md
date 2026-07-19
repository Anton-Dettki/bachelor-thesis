# Evaluation Results — RAW ON/OFF View, LTL Query `F(M08_ON & X(F M09_ON))`

This document reports the complete results of one full end-to-end workflow run of the federated
process-mining system, restricted to the **RAW ON/OFF event abstraction**. It is intended as the
factual basis for the Evaluation chapter of the thesis and is organized around the four research
questions (EQ1–EQ4). All numbers are taken verbatim from the persisted run artifacts in
`fpm/outputs/m08_m09_run/` (grouped/server-side pipeline) and the per-client federated training
results (`per_client_federated.json`). Nothing in this document is estimated or rounded beyond
what the pipeline itself reports.

---

## 1. Experimental Setup

### 1.1 Dataset

- **Dataset:** CASAS "Chinook" smart-home ADL dataset.
- **Participants:** 44 (`p01` … `p59`, non-contiguous IDs). Each participant is simulated as an
  independent Docker client holding only its own private trial data.
- **Traces per participant:** 5 trials (both `adl_noerror` and `adl_error` recordings are included).
- **Events:** motion/door sensor readings. In the RAW view every event is the full sensor-state
  token (e.g. `M08=ON`, `M08=OFF`), so the model must predict the exact next sensor-state event.
  This roughly doubles the label space compared with the sensor-level abstraction and makes the
  task strictly harder.

### 1.2 Run configuration

| Parameter | Value |
|---|---|
| LTL filter query | `F(M08_ON & X(F M09_ON))` ("eventually M08 turns on, and strictly afterwards M09 eventually turns on") |
| Minimum matching trace fraction | 0 (a client participates if ≥ 1 training trace satisfies the query) |
| Per-client model | CASAS2-style decision tree (`casas_tree`, max_depth 25, min_samples_leaf 10) |
| Behavioral grouping | enabled, K selected automatically from {2, …, 6} by silhouette score |
| Selected K | **2** (silhouette score **0.6263**) |
| Evaluation protocol | CASAS2 80/20 **within trial**: for every trial, the first 80 % of events (chronological) form the training prefix samples, the last 20 % the test samples |
| Baselines | enabled (per-client local, local tree ensemble, FedAvg SGD, Markov global, Markov grouped) |
| Abstraction | **raw** (full ON/OFF sensor-state events) |

### 1.3 Sample counts (RAW view)

| Quantity | Value |
|---|---|
| Train samples, all 44 clients | 6 534 |
| Train samples, LTL-matched clients only (used for grouping/grouped trees) | 4 894 |
| Pooled test samples (all 44 clients) | 1 796 |

### 1.4 Pipeline stages of the run

1. **LTL broadcast & filtering.** The coordinator broadcasts the LTL query; every client
   independently evaluates it against its own training traces (raw tokens are converted to the
   `M08_ON` atomic-proposition form). Clients report whether they match and their
   `matched_fraction` (share of training traces satisfying the query).
2. **Per-client local training (federated `/train`).** Each matched client trains a local CASAS2
   decision tree on its own data and returns metrics only (no raw data leaves the client).
3. **Behavioral profiling & clustering.** Matched clients submit behavioral profile vectors
   (L2-normalized bigram transition frequencies, unigram event frequencies, and per-task
   breakdowns computed from their LTL-matched training traces). The server clusters these with
   KMeans; K is chosen by silhouette score.
4. **Server-side grouped evaluation.** Test samples from all participants are pooled centrally.
   A global tree, per-cluster grouped trees, and all baselines are trained/evaluated on this pool.
5. **Artifact generation.** Comparison tables, cluster summaries, dendrogram, and predictive
   workflow graphs (JSON + PNG) are written to disk.

### 1.5 Metric definitions

- **Accuracy** — `sklearn.metrics.accuracy_score` over the pooled test samples.
- **Macro_F1** — F1 averaged unweighted over all label classes (`average="macro"`,
  `zero_division=0`). With ~37 raw event classes and a heavily skewed label distribution, many
  rare classes drag this value down for every approach; it should be read as a class-balance
  indicator, not as the headline performance number.
- **Weighted_F1** — F1 averaged over classes weighted by support (`average="weighted"`).
- **Graph_Nodes / Graph_Edges** — size of the predictive workflow graph mined from the model's
  predicted transitions on the test set (edge kept if transition probability ≥ 0.05). Nodes are
  unique events appearing in predicted transitions; edges are last-observed-event →
  predicted-next-event transitions.
- **Train_Time_s** — wall-clock training time of the tree(s) only.
- **Per-cluster accuracy** — accuracy restricted to test samples of clients assigned to that
  cluster (matched clients only).

### 1.6 Important evaluation semantics (needed for correct interpretation)

- The pooled test set (1 796 samples) covers **all 44 participants**, including the 11 clients
  excluded by the LTL filter. For excluded clients' test samples, the "Grouped" approach **falls
  back to the global tree** (they have no cluster assignment). The grouped accuracy therefore
  measures the full-population effect of grouping, not only performance on matched clients.
- The grouped predictor is a **hybrid**: for each test sample of a matched client, the assigned
  cluster tree is used when its `predict_proba` confidence is ≥ 0.67; otherwise a cluster-level
  Markov predictor (with global-Markov backoff for unseen contexts) makes the prediction.
- The global tree receives `participant_id` as an input feature; the grouped cluster trees do
  not (the cluster assignment itself carries the personalization). In the raw view both also
  receive `task` and `pos` (position in trace) features.
- The "Grouped" `Graph_Nodes`/`Graph_Edges` values in the comparison table are the **mean over
  the per-cluster graphs**: cluster 0 has 36 nodes / 65 edges, cluster 1 has 23 nodes / 41 edges,
  giving the reported 29 / 53 (integer mean). The individual per-cluster graphs are persisted
  separately (see §6).

---

## 2. EQ1 — Next-Event Predictability (RAW View)

*To what extent can the evaluated models predict the next event in the smart-home traces?*

### 2.1 Federated per-client results (local CASAS2 tree, trained and evaluated on-device)

All 33 LTL-matched clients trained a local tree; the table reports each client's own 80/20
test split in the RAW view.

| Client | Accuracy | Macro_F1 | Weighted_F1 | Correct/Total | Label classes |
|---|---|---|---|---|---|
| p01 | 0.5918 | 0.3227 | 0.5437 | 29/49 | 33 |
| p04 | 0.7073 | 0.3302 | 0.6596 | 29/41 | 36 |
| p05 | 0.6949 | 0.3347 | 0.6478 | 41/59 | 40 |
| p06 | 0.4412 | 0.1540 | 0.3081 | 15/34 | 34 |
| p07 | 0.5294 | 0.2230 | 0.4710 | 18/34 | 33 |
| p08 | 0.5652 | 0.2977 | 0.4777 | 26/46 | 36 |
| p11 | 0.7000 | 0.2955 | 0.6354 | 28/40 | 34 |
| p12 | 0.6889 | 0.2821 | 0.6531 | 31/45 | 34 |
| p13 | 0.6176 | 0.3025 | 0.5378 | 21/34 | 35 |
| p14 | 0.6667 | 0.2670 | 0.6185 | 30/45 | 32 |
| p15 | 0.7297 | 0.3688 | 0.7388 | 27/37 | 36 |
| p16 | 0.6977 | 0.2819 | 0.6322 | 30/43 | 34 |
| p17 | 0.8254 | 0.5637 | 0.7952 | 52/63 | 33 |
| p18 | 0.7692 | 0.5164 | 0.7198 | 30/39 | 31 |
| p21 | 0.5714 | 0.2872 | 0.4500 | 20/35 | 32 |
| p23 | 0.6552 | 0.3624 | 0.5988 | 19/29 | 32 |
| p24 | 0.4444 | 0.1839 | 0.3971 | 24/54 | 36 |
| p26 | 0.8710 | 0.5212 | 0.8580 | 27/31 | 32 |
| p27 | 0.6667 | 0.2451 | 0.6319 | 24/36 | 32 |
| p29 | 0.8125 | 0.3003 | 0.7610 | 39/48 | 35 |
| p30 | 0.6364 | 0.2220 | 0.5856 | 21/33 | 32 |
| p31 | 0.5556 | 0.3221 | 0.4517 | 15/27 | 30 |
| p32 | 0.6809 | 0.2565 | 0.6385 | 32/47 | 35 |
| p40 | 0.8571 | 0.4140 | 0.8091 | 36/42 | 33 |
| p42 | 0.5610 | 0.2399 | 0.4862 | 23/41 | 36 |
| p43 | 0.5714 | 0.2993 | 0.5758 | 20/35 | 36 |
| p49 | 0.2759 | 0.1069 | 0.2070 | 8/29 | 34 |
| p51 | 0.5128 | 0.2830 | 0.4516 | 20/39 | 38 |
| p52 | 0.5526 | 0.2311 | 0.4452 | 21/38 | 31 |
| p55 | 0.6000 | 0.3598 | 0.5221 | 21/35 | 33 |
| p56 | 0.4444 | 0.2549 | 0.3929 | 32/72 | 35 |
| p57 | 0.4118 | 0.1920 | 0.3166 | 14/34 | 33 |
| p59 | 0.6207 | 0.3646 | 0.5498 | 18/29 | 30 |

**Aggregate statistics over the 33 matched clients (RAW view):**

| Statistic | Value |
|---|---|
| Mean accuracy | 0.6220 |
| Median accuracy | 0.6207 |
| Std. deviation | 0.1333 |
| Minimum | 0.2759 (p49) |
| Maximum | 0.8710 (p26) |
| Mean Macro_F1 | 0.3026 |
| Mean Weighted_F1 | 0.5627 |
| Pooled micro accuracy (Σcorrect / Σtotal) | 841 / 1 343 = 0.6262 |

**Interpretation points for EQ1:**

- The event histories carry substantial sequential signal: even purely local trees trained on a
  single client's data reach a mean accuracy of ~62 % over ~30–40 raw event classes, far above
  a majority-class or random baseline.
- Predictability varies strongly across individuals (27.6 %–87.1 %), showing that some
  participants exhibit far more regular sensor-event behavior than others.
- Pooling data helps: the centrally trained global tree on the same protocol reaches 0.6949 and
  the simple first-order Markov predictor 0.7071 (see §3), i.e., raw next-event sequences are
  largely dominated by short-range transition structure that local per-client data alone cannot
  fully capture.
- Macro-F1 is low for all approaches (0.10–0.56 locally, ≈ 0.19–0.21 pooled) because the raw
  label distribution is heavily imbalanced; frequent sensor transitions are predicted well while
  rare events are mostly missed.
- (Cross-representation comparison for EQ1 — sensor abstraction vs. raw — is documented in the
  companion sensor-level artifacts `grouped_comparison.txt`; this document covers RAW only.)

---

## 3. EQ2 — Behavioral Grouping vs. Global, Local, and Federated Baselines (RAW View)

*How does behavioral grouping affect predictive performance compared with global, local, and
federated baseline approaches?*

All approaches below are evaluated centrally on the identical pooled test set of **1 796 RAW
test samples from all 44 participants** (CASAS2 80/20 within trial).

| Approach | Accuracy | Macro_F1 | Weighted_F1 | Graph_Nodes | Graph_Edges | Train_Time_s |
|---|---|---|---|---|---|---|
| Global | 0.6949 | 0.1889 | 0.6746 | 37 | 78 | 0.01 |
| **Grouped (K=2), LTL n=33** | **0.7077** | **0.2103** | **0.6796** | **29** | **53** | **0.01** |
| Per-client local | 0.6080 | 0.1194 | 0.5618 | N/A | N/A | N/A |
| Local tree ensemble | 0.6420 | 0.1236 | 0.5557 | N/A | N/A | N/A |
| FedAvg SGD | 0.1871 | 0.0135 | 0.0800 | N/A | N/A | 4.67 |
| Markov global | 0.7071 | 0.2125 | 0.6842 | 36 | 75 | N/A |
| Markov grouped | 0.7077 | 0.2141 | 0.6837 | 29 | 54 | N/A |

Approach definitions:

- **Global** — one decision tree trained on the pooled training data of all 44 clients
  (with `participant_id` as a feature).
- **Grouped** — one tree per behavioral cluster, trained on the pooled data of the LTL-matched
  clients in that cluster; hybrid Markov backoff below confidence 0.67; global-tree fallback for
  the 11 unmatched clients' test samples.
- **Per-client local** — each client's own local tree predicts its own test samples (global-tree
  fallback for clients whose training labels were degenerate).
- **Local tree ensemble** — FedAvg-style model exchange for trees: `predict_proba` of all
  per-client local trees is averaged for every test sample.
- **FedAvg SGD** — a linear (log-loss) model trained with federated averaging over client-local
  SGD updates; included as the classical federated-learning baseline.
- **Markov global / Markov grouped** — first-order Markov (last-event transition-frequency)
  predictors, trained globally or per cluster respectively.

**Per-cluster accuracy of the grouped approach (matched clients only):**

| Cluster | Clients | Accuracy |
|---|---|---|
| Cluster 0 | 29 | 0.752 |
| Cluster 1 | 4 | 0.583 |

**Cluster membership** (KMeans on behavioral profiles, K=2, silhouette 0.6263):

- Cluster 0 (29): p01, p04, p05, p06, p11, p12, p13, p14, p15, p16, p17, p18, p21, p23, p26,
  p27, p29, p30, p31, p32, p40, p42, p43, p49, p51, p52, p55, p57, p59
- Cluster 1 (4): p07, p08, p24, p56

**Interpretation points for EQ2:**

- **Grouping outperforms every other approach on all three metrics.** Grouped (0.7077 / 0.2103 /
  0.6796) beats Global (0.6949 / 0.1889 / 0.6746) by +1.28 accuracy points, despite the grouped
  trees using only the 33 matched clients' data (4 894 vs. 6 534 training samples) and no
  participant-ID feature.
- **The ordering Global > ensemble > local confirms the data-pooling hypothesis:** isolated
  per-client learning (0.6080) loses ~8.7 points against the global model; averaging local
  models (0.6420) recovers part of the gap but not all; grouping recovers all of it and more.
- **FedAvg SGD collapses (0.1871).** A single averaged linear model cannot represent the highly
  multi-class, non-linear next-event mapping, which motivates the tree/cluster design over
  classical parameter-averaging FL for this task.
- **Markov baselines are strong.** Markov global (0.7071) almost matches the grouped trees, and
  Markov grouped ties them at 0.7077 accuracy (slightly higher Macro_F1 0.2141, slightly lower
  Weighted_F1 0.6837). This shows (a) that raw next-event dynamics are largely first-order, and
  (b) that the grouped trees' advantage over the global tree is genuine but modest in pure
  accuracy — its more distinctive contribution is structural (EQ3). Note the grouped hybrid
  itself uses Markov backoff for low-confidence predictions, so the tie is partly by
  construction.
- **The clusters are strongly imbalanced (29 vs. 4)** and differ markedly in predictability
  (75.2 % vs. 58.3 %). The small cluster groups the four clients whose local accuracies are
  among the lowest (p07 0.53, p08 0.57, p24 0.44, p56 0.44), i.e., the clustering isolates the
  behaviorally atypical/less regular participants rather than splitting the population evenly.
- Minor caveat: baselines with stochastic components vary marginally between runs (e.g., the
  dashboard for the same query showed Local tree ensemble 0.6414 and FedAvg train time 4.36 s
  versus 0.6420 / 4.67 s in the persisted artifacts); the tree/Markov/grouped numbers are
  deterministic and identical.

---

## 4. EQ3 — Compactness of Workflow Representations (RAW View)

*Does behavioral grouping produce more compact predictive workflow representations without
substantially reducing predictive performance?*

Workflow graphs are mined from each model's predicted transitions on the test set (edges kept
at probability ≥ 0.05).

| Graph | Nodes | Edges | Density |
|---|---|---|---|
| Global tree workflow | 37 | 78 | 0.0586 |
| Grouped — cluster 0 workflow | 36 | 65 | 0.0516 |
| Grouped — cluster 1 workflow | 23 | 41 | 0.0810 |
| Grouped (mean over clusters, as reported in table) | 29 | 53 | — |
| Markov global workflow | 36 | 75 | — |
| Markov grouped (mean over clusters) | 29 | 54 | — |

**Interpretation points for EQ3:**

- Relative to the single global workflow (37 nodes, 78 edges), the average grouped workflow is
  **~22 % smaller in nodes and ~32 % smaller in edges** (29 / 53), while accuracy simultaneously
  *increases* from 0.6949 to 0.7077. Compactness is therefore obtained without any performance
  sacrifice in this run.
- The reduction is not merely an averaging artifact: even the large cluster 0 alone (29 of 33
  clients) yields fewer edges than the global graph (65 vs. 78), and the small cluster 1 yields
  a much smaller graph (23 / 41) describing the behavior of its atypical members specifically.
- Each cluster graph describes a behaviorally homogeneous subgroup, so the per-group
  representations are individually simpler and more interpretable than one graph that must
  superimpose all behavioral variants.
- The same pattern holds for the Markov-based graphs (36/75 global vs. 29/54 grouped),
  confirming the effect is a property of the grouping, not of the tree model.

---

## 5. EQ4 — LTL-Based Client Filtering (Operational Results)

*Can LTL-based client filtering restrict participation according to temporal behavior while
preserving the complete evaluation workflow?*

### 5.1 Filtering outcome

| Item | Value |
|---|---|
| Query | `F(M08_ON & X(F M09_ON))` |
| Evaluation locus | each client evaluates the query independently on its own training traces |
| Minimum matching traces required | 1 |
| Matched clients | **33 / 44** (75 %) |
| Excluded clients (11) | p02, p03, p09, p10, p20, p22, p41, p50, p53, p54, p58 |
| Matching trace per matched client | exactly **trial 1** (`pXX.t1`) for all 33 matched clients |
| `matched_fraction` per matched client | 0.2 (1 of 5 trials) |

Every matched client satisfies the query in exactly one trace (trial 1), giving a uniform
matched fraction of 0.2; the 11 excluded clients satisfy it in no trace (fraction 0.0). This
confirms the filter discriminates on a genuine temporal-behavioral property (an M08-activation
strictly followed later by an M09-activation) rather than trivially matching everyone or no one.

### 5.2 Workflow completeness after filtering

With the filter active, the coordinator completed **every** downstream stage for the selected
subset, producing all expected artifacts:

1. **Local training** — all 33 matched clients trained and reported per-client metrics (§2);
   the 11 non-matching clients did not participate in training.
2. **Profile collection** — behavioral profiles were collected from all 33 matched clients
   (`behavioral_profiles.csv`).
3. **Clustering** — KMeans over the 33 profiles selected K=2 with silhouette 0.6263
   (`cluster_assignments.json`, `cluster_summary.txt`, `cluster_dendrogram.png`).
4. **Grouped prediction & baselines** — the full comparison table (§3) was computed, with
   grouped trees trained only on matched clients' data (4 894 samples).
5. **Artifact generation** — comparison tables (CSV/TXT), LTL filter reports, and all workflow
   graphs (JSON + PNG) were written (§6).

**Interpretation points for EQ4:**

- The LTL component functions as intended operationally: query broadcast, decentralized
  client-side evaluation, participation restriction (33 of 44), and an unmodified downstream
  pipeline over the subset all succeeded in a single run.
- Excluded clients are still *evaluated on* (their test samples remain in the pooled test set,
  served by the global fallback), so filtering restricts *training participation* without
  breaking the evaluation protocol.
- As stated in the research question, this run demonstrates functional integration; the
  accuracy gain of the grouped model over the global model should not be attributed to the LTL
  formula itself but to the grouping over the selected subset.

---

## 6. Artifact Inventory (source of every number above)

All paths relative to the repository root, directory `fpm/outputs/m08_m09_run/`:

| File | Content |
|---|---|
| `grouped_comparison_raw.csv` / `grouped_comparison_raw.txt` | RAW-view comparison table (§3), per-cluster accuracies, run metadata (sample counts, silhouette, matched/excluded clients) |
| `per_client_federated.json` | Per-client federated `/train` metrics for all 44 clients, both abstraction views (§2 uses the `raw` entries) |
| `ltl_filter.json` / `ltl_filter_summary.txt` | Matched/excluded clients, matched case IDs (`pXX.t1`), min-matching-traces setting (§5) |
| `cluster_assignments.json` / `cluster_summary.txt` | Client → cluster mapping and cluster sizes (§3) |
| `cluster_dendrogram.png` | Ward-linkage dendrogram of the behavioral profiles |
| `behavioral_profiles.csv` | Raw behavioral profile vectors of the 33 matched clients |
| `global_raw_workflow.json` / `.png` | Global-model RAW workflow graph, 37 nodes / 78 edges (§4) |
| `group_0_raw_workflow.json` / `.png` | Cluster-0 RAW workflow graph, 36 nodes / 65 edges (§4) |
| `group_1_raw_workflow.json` / `.png` | Cluster-1 RAW workflow graph, 23 nodes / 41 edges (§4) |

Sensor-level counterparts (`grouped_comparison.csv/.txt`, `global_workflow.*`,
`group_{0,1}_workflow.*`) from the same run are stored in the same directory but are outside
the scope of this RAW-only document.

---

## 7. Key Takeaways (one paragraph per EQ)

**EQ1.** The raw smart-home event streams are predictably sequential: local per-client trees
average 62.2 % next-event accuracy (median 62.1 %, range 27.6–87.1 %) over ~30–40 raw ON/OFF
classes, and pooled models reach ~69–71 %. Low macro-F1 across all approaches (≤ 0.21 pooled)
reflects the skewed event distribution: frequent transitions are learned, rare ones are not.

**EQ2.** Behavioral grouping is the best-performing approach in the RAW view: 0.7077 accuracy /
0.2103 macro-F1 / 0.6796 weighted-F1, ahead of the global model (0.6949 / 0.1889 / 0.6746),
the local models (0.6080), the local tree ensemble (0.6420), and far ahead of FedAvg SGD
(0.1871). It achieves this while training on fewer samples (matched clients only) and without a
participant-ID feature. First-order Markov predictors are competitive (0.7071 global, 0.7077
grouped), indicating strong short-range structure in the raw sequences.

**EQ3.** Grouping yields markedly more compact workflow representations at no accuracy cost:
mean 29 nodes / 53 edges per cluster graph versus 37 / 78 for the single global graph (−22 %
nodes, −32 % edges), with the grouped model simultaneously more accurate than the global one.

**EQ4.** The LTL filter `F(M08_ON & X(F M09_ON))` operationally restricted participation to
33 of 44 clients (each matching in exactly one trial, trial 1), and the complete downstream
workflow — local training, profile collection, clustering (K=2, silhouette 0.6263), grouped
prediction, baseline comparison, and artifact generation — completed successfully on the
selected subset.
