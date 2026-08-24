# Build a consumer-credit research workbench and test whether graphs earn new TinyMesh primitives

Decision state: Decided
Repositories: `drukpa1455/quantcredit` research repository;
`spatioterra-ai/tinymesh` reusable runtime
Inspected TinyMesh revision: `dad041a41cf7df7b379dc0e1a9769d6805a12042`
Sources: 2026-08-22 through 2026-08-23 public research research-preparation
discussion; SEC asset-level disclosure; Zed REPL contract; pinned PyG Temporal
reference

## Repository decision

`quantcredit` is the domain research repository and this file is its planning
source. TinyMesh product code remains unchanged.

```text
quantcredit
  SEC acquisition + schema semantics
  loan panel + targets + censoring
  credit features + GBM/survival baselines
  graph lowering + training + evaluation
  notebooks + figures + research conclusions
                    |
                    | imports released primitives
                    v
tinymesh
  Graph + sparse aggregation
  reusable message-passing equations
  reusable temporal equations
  CPU/Metal correctness + complexity contracts
```

The dependency points one way: `quantcredit -> tinymesh`. TinyMesh never imports
credit data, tasks, models, policy, or conclusions. Experimental credit code may
prototype a missing equation locally. It moves into TinyMesh only after the
credit experiment identifies a stable mathematical contract and TinyMesh's
independent parity, gradient, shape, sparse-work, documentation, and live-caller
gate passes.

`quantcredit` is preferred over `quantcc`, which hides the domain, and
`quantdebt`, which implies a broader fixed-income or corporate-debt scope. The
name also avoids prejudging graphs as the answer: tabular, survival, calibration,
and portfolio research are first-class outcomes even if no graph model wins.

### Can one artifact serve learning, research, and research rehearsal?

- **Decision informed:** Whether the project should own an `.ipynb`, a plain
  script, or both, and what belongs in reusable helper modules.
- **Stop condition:** One representation is readable by a person and an agent,
  executes cell-by-cell in the chosen editor, remains reviewable in Git, and can
  reproduce a clean top-to-bottom analysis without duplicated source truth.

#### Sources

- [Zed REPL documentation](https://zed.dev/docs/repl), which specifies Python
  `# %%` cell boundaries, inline Jupyter-kernel execution, and project-environment
  kernel selection
- [Jupytext percent-format documentation](https://jupytext.readthedocs.io/en/latest/formats-scripts.html),
  which specifies titled code and Markdown cells in ordinary text

#### Findings

- **Observed:** Zed executes blocks between `# %%` markers as independent
  Jupyter cells in a normal Python file and renders results inline.
- **Observed:** Percent scripts are ordinary diffable text and may give cells
  stable semantic titles. Notebook execution counts such as "cell 3" are
  positional and change when cells are inserted or reordered.
- **Inferred:** A generated `.ipynb` adds synchronization and output-churn costs
  without improving the primary Zed workflow. It may be exported for sharing,
  but it should not be another authored artifact.
- **Decision:** `notebooks/credit_research.py` is the canonical executable
  narrative. Its titled `# %%` sections own questions, invocations, compact
  observations, and conclusions. `src/quantcredit/` owns deterministic reusable
  operations. The notebook never owns a second implementation of stable logic.
- **Decision:** The toolkit is executable study material, not an assumed
  analysis dependency. Every helper has a small explicit contract and may be
  explained or rewritten directly if an research environment disallows
  personal packages.

## Research

### Can public auto-loan performance support a trustworthy experiment?

- **Decision informed:** Whether SEC Form ABS-EE automobile-loan filings can
  support an out-of-time loan panel without private data or invented labels.
- **Stop condition:** One bounded trust-year is acquired reproducibly, its XML
  schema and identifiers are validated, consecutive asset states reconcile, and
  every candidate target is classified as observed, derived, censored, or
  unusable.

#### Sources

- [SEC ABS technical specifications](https://www.sec.gov/submit-filings/technical-specifications),
  current ABS Version 3.1 dated 2023-09-18
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [SEC Release 33-9638](https://www.sec.gov/files/rules/final/2014/33-9638.pdf),
  which requires automobile-loan asset-level information under Schedule AL
- [Ford Credit Auto Owner Trust 2024-A July 2025 filing](https://www.sec.gov/Archives/edgar/data/2014176/000201417625000033/0002014176-25-000033-index.htm),
  an observed monthly Form ABS-EE filing with an EX-102 asset data file

#### Findings

- **Observed:** EX-102 is the XML Asset Data File governed by Schedule AL. The
  official schema is the semantic authority; issuer labels and convenient
  dataframe types are not.
- **Observed:** EDGAR submission history identifies monthly filings, while the
  asset XML files can exceed 100 MB. SEC recommends bulk access for large API
  workloads and requires compliant automated access.
- **Observed:** Ford Credit Auto Owner Trust 2024-A, CIK `0002014176`, has one
  Form ABS-EE report for every month from 2025-01-31 through 2025-12-31.
- **Inferred:** A local ignored source cache plus a tracked checksum manifest is
  the smallest reproducible boundary. Committing consumer-level rows is
  unnecessary and creates avoidable privacy and repository-size costs.
- **Decision:** Stage 1 uses that twelve-report trust-year only. It decides data
  semantics; it makes no model-quality or cross-originator claim.

### Which PyG and PyG Temporal ideas can change TinyMesh's design?

- **Decision informed:** Which missing primitive, if any, should follow the
  lending experiment.
- **Stop condition:** Each candidate has a live lending caller, a smaller
  TinyMesh representation, and an observation that would promote or reject it.

#### Sources

- TinyMesh
  [`Graph`](https://github.com/spatioterra-ai/tinymesh/blob/dad041a41cf7df7b379dc0e1a9769d6805a12042/src/tinymesh/graph.py),
  [`StaticGraphTemporalSignal`](https://github.com/spatioterra-ai/tinymesh/blob/dad041a41cf7df7b379dc0e1a9769d6805a12042/src/tinymesh/temporal.py),
  and
  [`GINEConv`](https://github.com/spatioterra-ai/tinymesh/blob/dad041a41cf7df7b379dc0e1a9769d6805a12042/src/tinymesh/nn/__init__.py)
  at the inspected revision
- Pinned PyG Temporal revision
  [`fe555bc30ee197755c4b58a89407033a5f383415`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415)
- [PyG heterogeneous graphs](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html),
  [PNAConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html),
  [TemporalData](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.TemporalData.html),
  and [TGNMemory](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.TGNMemory.html)

#### Findings

- **Observed:** TinyMesh already provides immutable sparse topology, shared
  leading lanes, static scalar edge weights, edge-vector GINE messages, and
  fixed-node temporal recurrence. It does not provide relation-specific
  operators, per-lane edge values, explicit temporal presence masks, changing
  topology, or event memory.
- **Observed:** The pinned PyG Temporal heterogeneous iterators retain separate
  dictionaries for node types and relation triples. `LRGCN` applies
  relation-specific graph convolutions inside recurrent gates. These are useful
  equation and boundary references, not evidence that TinyMesh needs equivalent
  containers.
- **Observed:** PNA combines aggregators and degree scalers. Consumer-credit
  loan-to-cohort graphs are likely to contain low-degree loan leaves and
  high-degree state, vintage, vehicle, and originator hubs.
- **Observed:** `TemporalData` models ordered interaction events and TGN stores
  per-node memory and last-update times. Monthly ABS-EE snapshots are not
  transaction events.
- **Inferred:** A heterogeneous lending graph can be lowered as a typed quiver:
  one sparse `Graph`, `node_type[N]`, and `edge_type[E]`. Separate graph
  dictionaries would duplicate topology ownership.
- **Decision:** Existing `GINEConv` is the first graph challenger. A
  relation-specific convolution is considered only after a relation-erasure
  control shows that one shared edge-conditioned transform is insufficient.
- **Decision:** Segment-level monthly monitoring uses existing temporal models
  first. Per-lane edge values or masks graduate only when the experiment has a
  concrete value or missingness contract that static inputs cannot represent.
- **Decision:** TGN/HGT, changing topology, general heterogeneous containers,
  and dense learned adjacency are non-goals for this epic.

## Outcome

### Current behavior

`quantcredit` now owns twelve pinned EX-102 documents, bounded acquisition,
typed streaming snapshots, identity and continuity validation, an aggregate-only
transition audit, one explicit three-report target, and executable notebook
sections `00` through `05`, including a label-maturity-aware chronological
split and Sapphire aggregate figures. It does not yet materialize the modeling
population or provide the classical credit-analysis toolkit. TinyMesh can
express a homogeneous edge-aware loan graph and a fixed-node monthly segment
graph, but there is no evidence that either adds useful information beyond a
tabular model. Adding relational or dynamic-temporal APIs now would violate
TinyMesh's live-caller and graduation rules.

### Desired behavior

A contributor can clone `quantcredit`, select its `uv` environment in Zed, open
one ordinary Python file, and learn or reproduce the complete public,
causally-aligned auto-loan analysis from source audit through model conclusions.
The same project answers four questions in order:

1. What do the public records actually observe, and which credit target and
   temporal evaluation are defensible?
2. How well does a small, explainable classical model rank and calibrate the
   accepted outcome, and where does it fail economically?
3. Does relationship structure improve an out-of-time credit prediction over
   a shallow gradient-boosted model receiving the same loan and historical
   cohort information?
4. If so, which smallest missing sparse equation—not a framework—does the
   measured failure of existing TinyMesh primitives require?

The result is complete whether it promotes one primitive or rejects the graph
claim. The retained research record states what the evidence proves, what it
does not prove, and what would reopen the decision.

### Scope

- Validate and audit one checksum-pinned SEC ABS-EE auto-loan panel.
- Define loan state, censoring, missingness, and candidate outcomes from official
  schema semantics.
- Maintain one canonical Zed-compatible percent-script notebook with stable,
  semantically named sections and clean-kernel execution.
- Build a small tested toolkit for source validation, targets, chronological
  splits, leakage and drift diagnostics, calibration, and expected-loss
  analysis; keep orchestration and conclusions in the notebook.
- Compare a shallow GBM, the same GBM with flattened past-only cohort features,
  and an existing-GINE loan/context graph under one out-of-time protocol.
- Preserve false-relation, relation-erasure, and node-local controls.
- Explore fixed-node monthly segment monitoring with existing recurrence only
  after the static experiment closes.
- Promote a runtime primitive only through the existing graduation gate.

### Non-goals

- Applicant approval, decline, adverse-action, or causal credit-policy claims;
  ABS-EE contains securitized originated loans, not rejected applications.
- Production underwriting, individual lending decisions, or investment advice.
- A general SEC, XML, dataframe, trainer, heterogeneous-graph, or credit-risk
  framework.
- A second hand-maintained `.ipynb`, notebook-output snapshots in Git, or a
  Jupytext synchronization requirement. Exported notebooks and HTML are
  disposable presentation artifacts.
- A hidden "solve the analysis" function or reliance on `quantcredit` being
  installable during an research. The project teaches transparent operations;
  it does not bypass analysis rules.
- Using an agent, external reference, or personal code when an analysis does
  not permit it. The notebook is always preparation material and becomes a live
  aid only under the researcher's stated rules.
- Private bureau, bank-account, email, address, identity, or customer data.
- Scraping all EDGAR filings or checking raw source files into Git.
- HGT, TGN, EvolveGCN, AGCRN, arbitrary changing topology, or a model zoo.
- A neural network replacing an explainable production GBM. The graph is a
  research challenger; later distillation is a separate decision.

## Behavioral contract

- **INV-1 — Source truth:** Every observation names the issuing-entity CIK,
  accession, reporting period, EX-102 URL, byte count, SHA-256 digest, and ABS
  schema version. Acquisition rejects drift and partial files.
- **INV-2 — Stable identity:** A loan snapshot is keyed by issuing-entity CIK,
  schema-defined asset identifier, and reporting period. Duplicate keys,
  non-monotone periods, and contradictory immutable origination fields fail
  loudly rather than being silently deduplicated.
- **INV-3 — Explicit state:** Missing, not applicable, zero, prepaid, charged
  off, recovered, and no longer reported remain distinguishable whenever the
  source permits. Disappearance alone is censoring, not default or prepayment.
- **INV-4 — Causal availability:** A prediction at cutoff `t` may use only
  fields filed for periods at or before `t`. Labels begin after `t`; train-only
  normalization, cohort statistics, graph context, model selection, and
  threshold selection never read validation or test outcomes.
- **INV-5 — Matched information:** The enriched GBM and graph challenger receive
  the same past-only cohort facts. Graph lift cannot be reported as structural
  lift when the tabular control lacks equivalent information.
- **INV-6 — Independent scoring:** An evaluation loan may receive messages from
  context frozen at its cutoff, but no evaluation loan may change another
  evaluation loan's prediction. Current-cohort test features do not aggregate
  through shared context.
- **INV-7 — Sparse execution:** Graph carriers remain `O(N + E)` and graph work
  remains `O((N + E)H)` for fixed hidden width. No `[N,N]`, `[N,E]`, or dense
  learned-adjacency carrier enters a network-scale path.
- **INV-8 — Ownership:** SEC semantics, extraction, features, targets, splits,
  models, training, and claims remain in `quantcredit`. `src/tinymesh/` changes
  only after a stable mathematical contract has a live caller, parity, gradient,
  shape, and sparse-work evidence.
- **INV-9 — Repository privacy:** Git retains source manifests, synthetic
  fixtures, aggregate observations, and research decisions only. Raw or sampled
  consumer-level loan rows remain in an ignored local cache and never enter
  commits, logs, or run envelopes.
- **INV-10 — Bounded acquisition:** Downloads use an identifying user agent,
  finite connect/read timeouts, atomic `.part` files, declared byte ceilings,
  and SEC access guidance. Retry applies only to absent or checksum-rejected
  local files and never accepts unknown partial success.
- **INV-11 — Canonical notebook:** `notebooks/credit_research.py` is the only
  authored notebook. Every cell starts with a stable `# %% <ID> — <title>`
  marker. References use the semantic ID and title, never execution count or
  ordinal position.
- **INV-12 — Restart determinism:** Starting from a fresh project environment
  and empty kernel, running canonical cells from top to bottom either produces
  the declared aggregate observations or fails at the first unmet source or
  invariant. No cell depends on untracked interactive state or execution order.
- **INV-13 — Thin narrative:** Stable transformations and metrics have one
  tested owner under `src/quantcredit/`; notebook cells select parameters,
  invoke those operations, inspect compact results, and state conclusions.
  Helpers accept explicit inputs and do not read notebook globals.
- **INV-14 — Explainable toolkit:** A helper's name, signature, return value,
  and test expose the operation it performs. No helper chooses a target, split,
  model, threshold, or conclusion implicitly, and no generic "find insights" or
  "solve dataset" interface exists. The common call uses minimal required
  arguments and safe defaults, while remaining simple enough to reproduce live.

- **AC-1 — Data boundary:** A clean-revision experiment validates all twelve
  Ford 2024-A reports for 2025, emits aggregate schema/identity/transition/
  missingness counts, and reproduces them from checksum-pinned sources.
- **AC-2 — Target decision:** The Stage 1 research record accepts or rejects
  each proposed default, prepayment, and loss target with field-level lineage
  and censoring evidence. No ambiguous target advances.
- **AC-3 — Matched comparison:** The static study reports identical temporal
  folds and loan populations for raw GBM, enriched GBM, GINE, relation-erased,
  false-relation, and node-local arms; exceptions are explicit invalid results.
- **AC-4 — Decision metrics:** Every model reports sample count, event rate,
  AUROC, average precision, log loss, Brier score, calibration by score band,
  and a declared ranking statistic. AUC alone cannot decide the result.
- **AC-5 — Structural claim:** Graph value requires improvement over the
  enriched GBM and all false-structure controls on validation across at least
  three declared seeds, followed by one frozen test evaluation. Statistical
  uncertainty and practical effect size are reported.
- **AC-6 — Runtime gate:** No new public primitive lands unless the experiment
  identifies a stable missing equation and the proposed owner passes host or
  dense parity, first-order gradient, leading-shape, CPU and Metal, and sparse
  storage/work checks.
- **AC-7 — Negative closure:** If the enriched GBM or a simpler existing model
  wins, the exact negative result is retained and no compensating architecture
  search or API is added within this epic.
- **AC-8 — Executable study surface:** With the project environment selected in
  Zed, the canonical percent script exposes runnable cells for setup, source
  audit, schema and missingness, state transitions, target and censoring,
  temporal split, classical baseline, error analysis, economic interpretation,
  and graph study. Every reached section runs from a restarted kernel without
  manual variable repair.
- **AC-9 — Portable classical analysis:** Tests and the notebook demonstrate
  chronological splitting, train-only preprocessing, AUROC, average precision,
  log loss, Brier score, score-band calibration, and exposure-weighted expected
  loss using small transparent functions that can be explained without the
  package.

## Technical design

### Current system and owners

- `Graph` owns one immutable homogeneous directed topology and sparse
  aggregation. `Graph.sum` accepts only one shared `edge_weight[E]`.
- `GINEConv` owns edge-vector messages but uses one shared edge transformation.
  It already accepts arbitrary leading node lanes while edge features remain
  shared.
- `StaticGraphTemporalSignal` owns aligned `[T,N,F]` and `[T,N,Y]` tensors for
  one fixed graph and fixed node identities. It has no presence or target mask.
- TinyMesh's `experiments.CATALOG`, `experiments.run`, and `docs/research/` own
  TinyMesh-only runnable policy and promoted-runtime evidence. Consumer-credit
  acquisition, semantics, model results, and conclusions do not belong there.
- `quantcredit` owns the filing declaration, twelve verified EX-102 pins, a
  reusable bounded HTTPS transfer, SEC acquisition policy, and the canonical
  notebook shell. Typed snapshot parsing and continuity validation are complete;
  aggregate transition evidence, targets, and models remain incomplete.

### Proposed flow

```text
SEC submissions + EX-102 XML
            |
            v
ignored checksum-pinned source cache
            |
            v
quantcredit.source/panel ------> aggregate semantic audit
            |                              |
            |                              `--> target decision
            v
past-only loan panel
      +-----+-------------------+
      |                         |
      v                         v
raw loan features       historical cohort facts
      |                         |
      +--------+----------------+
               |
       +-------+---------+
       |                 |
       v                 v
shallow GBM         typed-quiver lowering
enriched GBM        Graph + node/edge tensors
                         |
                         v
                 existing GINE + controls
       |                 |
       +--------+--------+
                v
       out-of-time matched evidence
                |
                +--> canonical # %% notebook
                |      questions + compact evidence + conclusions
                |
       +--------+------------------+
       |                           |
       v                           v
reject graph claim       name missing equation
                                 |
                                 v
                       runtime graduation proof
```

The repository shape is deliberately small:

```text
quantcredit/
  pyproject.toml                 environment and dependency ownership
  uv.lock                        exact resolved environment
  sources/
    ford-credit-auto-owner-trust-2024-a.json
                                 pinned public-source declaration
  notebooks/
    credit_research.py           canonical # %% executable narrative
  src/quantcredit/
    fetch.py                     bounded HTTPS transfer + verified cache
    source.py                    source declaration + manifest checks
    acquire.py                   bounded SEC acquisition + checksum pinning
    panel.py                     schema facts, identity, state transitions
    audit.py                     bounded aggregate audit command
    targets.py                   explicit outcomes and censoring
    splits.py                    chronological populations and folds
    diagnostics.py               missingness, drift, leakage, calibration tables
    metrics.py                   declared predictive and economic metrics
    graphs.py                    experiment-only typed-quiver lowering, later
  tests/
    fixtures/                    synthetic schema-shaped records only
  docs/research/                 revision-bound conclusions, not raw runs
  plans/
    consumer-credit-research.md  this specification after initialization
  data/                          ignored rebuildable cache and derived tables
```

Files appear only when their owner becomes necessary. The names above define
responsibilities, not a requirement to create empty modules in Stage 1.

The environment deliberately matches likely research tools: pandas for tables,
scikit-learn for preprocessing and metrics, XGBoost for the shallow GBM,
Matplotlib for deterministic figure ownership, and Seaborn for axes-level
statistical vocabulary. `ipykernel` supports Zed's REPL. TinyMesh is added only
when Stage 3 begins. The lockfile, not this specification, owns exact resolved
package versions.

### Canonical notebook flow

```text
# %% 00 — Contract and environment
# %% 01 — Acquire and verify sources
# %% 02 — Schema, identity, and missingness
# %% 03 — Loan-state transitions
# %% 04 — Target and censoring decision
# %% 05 — Chronological split
# %% 06 — Shallow GBM baseline
# %% 07 — Calibration and error analysis
# %% 08 — Expected-loss interpretation
# %% 09 — Matched cohort controls
# %% 10 — Graph challenger and falsification
# %% 11 — Conclusion and reopening conditions
```

Later sections may initially contain Markdown-only contracts until their stage
is implemented. IDs and meanings remain stable; new detail uses subordinate IDs
such as `07a`, rather than renumbering the narrative. The notebook imports the
installed editable project selected by Zed's toolchain/kernel picker. `uv` owns
the project environment; no system-wide package installation is required.

### Data boundary

The Stage 1 pilot is Ford Credit Auto Owner Trust 2024-A, CIK
`0002014176`, for these reporting periods and accessions:

| Report period | Accession |
| --- | --- |
| 2025-01-31 | `0002014176-25-000006` |
| 2025-02-28 | `0002014176-25-000010` |
| 2025-03-31 | `0002014176-25-000016` |
| 2025-04-30 | `0002014176-25-000020` |
| 2025-05-31 | `0002014176-25-000024` |
| 2025-06-30 | `0002014176-25-000028` |
| 2025-07-31 | `0002014176-25-000033` |
| 2025-08-31 | `0002014176-25-000037` |
| 2025-09-30 | `0002014176-25-000042` |
| 2025-10-31 | `0002014176-25-000047` |
| 2025-11-30 | `0002014176-25-000051` |
| 2025-12-31 | `0002014176-26-000002` |

The tracked `sources/ford-credit-auto-owner-trust-2024-a.json` declaration pins
each accession, official archive URL, EX-102 URL, byte count, and digest.
Acquisition streams each document into an ignored cache and rejects drift before
evidence can run. The executable experiment never trusts a mutable "latest"
query. An identifying SEC user agent is an explicit runtime input,
conventionally supplied through `QUANTCREDIT_SEC_USER_AGENT`; its personal value
is neither committed nor printed.

Parsing is streaming and standard-library-first because each asset file is
large, DOM materialization is unnecessary, and acquisition should have a small
failure surface. Parsed host facts use explicit types and retain source element
names and missingness. Tests use a synthetic minimal schema-shaped fixture
covering absent, nil, zero, duplicate, changed-immutable, delinquent, prepaid,
charged-off, recovered, and disappearing assets.

Stage 1 emits aggregate JSON only:

```text
schema + file counts
loan and snapshot counts
field presence/type/domain counts
identity duplicates and continuity
immutable-field contradictions
state transition counts
candidate-target lineage and censoring counts
```

### Static graph study

Stage 3 is refined only after Stage 2 fixes the classical incumbent. Its
default representation is a typed quiver lowered into one ordinary graph:

```text
historical context --relation--> scored loan

context types: state, origination vintage, vehicle cohort, trust/originator
relation types: located_in, originated_in, secured_by, issued_by
```

Each node type is projected into one shared hidden width by experiment-owned
linear maps. `edge_type[E]` becomes a one-hot or learned edge feature consumed
by existing `GINEConv`. Context values are computed only from information
available before the prediction cutoff. Edges point from frozen context to
scored loans so one scored loan cannot affect another.

The matched controls are:

1. shallow GBM on raw loan fields;
2. the same GBM plus flattened historical context values;
3. a node-local neural control with the graph parameter budget matched where
   practical;
4. GINE with true relations;
5. GINE with relation identity erased;
6. GINE with an isomorphic, degree-preserving false relation assignment.

Train, validation, and test are separated by prediction cutoff, never random
loan rows. Model selection uses validation only. The final test opens once
after the target, horizons, features, topology, seeds, budget, and primary
metric are frozen.

### Temporal monitoring study

After the static result closes, Stage 5 may aggregate persistent nodes such as
`originator × risk band × vehicle class × state` by reporting month. It begins
with `StaticGraphTemporalSignal` and existing node-local, TGCN, GConvGRU, and
DiffusionGRU controls.

The first possible temporal promotion questions are deliberately separate:

- **Presence masks:** required only if a fixed segment universe contains
  semantically distinct inactive and observed-zero rows.
- **Per-lane edge values:** required only if a fixed edge set has observed,
  causal monthly values whose static replacement measurably fails.
- **Dynamic topology:** required only if fixed-union topology plus explicit
  presence cannot express the study without changing semantics.
- **Event memory:** requires genuinely irregular applicant, inquiry,
  transaction, or payment events. Monthly ABS-EE snapshots cannot justify it.

### Decisions

- **D-1:** Keep the entire lending effort research-only until the repository's
  existing promotion gate passes. This accepts some experiment-local
  duplication to avoid an API without evidence.
- **D-2:** Use SEC ABS-EE rather than LendingClub or synthetic records. ABS-EE
  supplies recurring asset performance closer to buy-and-hold auto-loan
  investing, at the cost of observing only securitized originated loans.
- **D-3:** Start with one twelve-month trust-year. This bounds acquisition and
  semantic variance; it cannot support cross-originator generalization.
- **D-4:** Flatten heterogeneous semantics into one sparse graph plus type
  features. A typed host boundary remains the source of truth; execution does
  not need a heterogeneous container.
- **D-5:** Require enriched tabular and false-topology controls. A comparison
  against raw tabular features alone cannot distinguish graph structure from
  additional cohort information.
- **D-6:** Preserve independent loan scoring. This excludes transductive
  evaluation-loan aggregation even when labels are hidden.
- **D-7:** Rank possible runtime work by observed need: relation-specific
  aggregation, temporal masks, per-lane edge values, then degree-aware
  multi-aggregation. Dynamic topology and event memory remain outside this
  dataset's default path.
- **D-8:** Never retain consumer-level source rows in Git. Reproducibility comes
  from official URLs, exact checksums, deterministic extraction, synthetic
  contract fixtures, and aggregate observations.
- **D-9:** Use one canonical Python percent script, not paired authored
  `.py`/`.ipynb` files. This gives Zed inline Jupyter execution while preserving
  ordinary code review and agent-readable text; rich exports remain disposable.
- **D-10:** Keep a functional core under `src/quantcredit/` and the imperative,
  explanatory shell in `notebooks/credit_research.py`. This accepts a few
  explicit notebook calls in exchange for one owner per transformation.
- **D-11:** Optimize the toolkit for comprehension and causal correctness, not
  API breadth. Prefer named dataframe-in/table-out functions over classes,
  registries, automatic insight generation, or a generic pipeline abstraction.
  Optimize the common path for live recall and hand-coding, not package magic.
- **D-12:** Keep `quantcredit` independently useful if the graph hypothesis
  fails. SEC semantics, temporal validation, classical modeling, calibration,
  and economic interpretation are primary project outcomes.
- **D-13:** Practice with pandas, scikit-learn, XGBoost, Matplotlib, and Seaborn
  rather than inventing table, metric, booster, or plotting layers. `quantcredit`
  helpers compose these libraries around credit-specific invariants and remain
  small enough to rewrite or explain during an analysis.

### Failure and operational behavior

- A missing filing, unexpected form, absent EX-102, redirect outside SEC,
  unsupported schema, oversized response, checksum mismatch, malformed XML,
  duplicate snapshot key, or invalid field domain fails acquisition or audit
  with accession and element context.
- Downloads are resumable only by restarting an absent `.part` file; partial
  bytes are not treated as a cache hit. Existing verified files are reused.
- Source drift is repaired by an explicit manifest revision after inspecting
  the new filing, never by silently replacing the checksum.
- Experiment envelopes contain aggregate counts and metrics only. Errors must
  not print source rows.
- A notebook cell must not catch an invariant failure merely to let later cells
  run. Recovery is explicit: repair or reacquire the source, restart the kernel,
  and rerun from the owning section.
- Inline Zed outputs are ephemeral and excluded from repository truth. A
  decision-changing run is retained as aggregate machine-readable evidence plus
  a revision-bound Markdown conclusion, never by trusting visible cell state.
- The data cache is rebuildable from the manifest and has no durability claim.
  The SEC archive is the public source of truth; the tracked research record is
  the decision source of truth.
- No paid compute, API key, or authenticated source is required. A later paid or
  private-data stage requires a separate specification and immediate approval.

## Delivery graph

```text
Epic: executable consumer-credit research from public records to graph decision
  |
  +-- Stage 1: executable notebook + trustworthy SEC loan-state boundary
  |     +-- 1.1 Establish the workbench and acquire the trust-year
  |     +-- 1.2 Parse and validate loan snapshots (depends on 1.1)
  |     `-- 1.3 Audit transitions and decide targets in the notebook
  |                 (depends on 1.2)
  |
  +-- Stage 2: classical credit-analysis toolkit and baseline
  |             (depends on Stage 1 target decision)
  |
  +-- Stage 3: matched enriched-GBM/GINE experiment
  |             (depends on Stage 2 baseline)
  |
  +-- Stage 4: reject graph claim or graduate one missing equation
  |             (depends on Stage 3 evidence)
  |
  `-- Stage 5: fixed-segment temporal monitoring
                (depends on Stage 1 semantics; refined after Stage 3)
```

### Stage 1: Executable notebook and trustworthy SEC loan-state boundary

- **Outcome:** A fresh `quantcredit` checkout opens as an executable percent
  notebook in Zed, deterministically audits twelve monthly reports, and makes an
  evidence-backed target/censoring decision without adding a TinyMesh API.
- **Depends on:** Public SEC archive availability and the official ABS Version
  3.1 schema.
- **Invalidating assumption:** The schema-defined asset identifier may not remain
  stable enough across monthly reports to construct a loan panel, or terminal
  outcomes may not be distinguishable from reporting disappearance.
- **Proof:** Clean-kernel execution through notebook section `04`, synthetic
  contract tests, and one revision-bound aggregate audit from all twelve
  checksum-pinned EX-102 files.

#### Issue 1.1: Establish the workbench and acquire the Ford 2024-A trust-year

**What and why:** Initialize `quantcredit` with one locked `uv` environment, one
canonical Zed percent script, a bounded source declaration, and an ephemeral
acquisition boundary. This is the first independently usable study surface: a
contributor can verify exact public inputs without committing loan rows.

**Done when:**

- The declaration contains exactly the twelve accessions above, official SEC
  archive URLs, reporting periods, discovered EX-102 filenames, byte counts,
  SHA-256 digests, schema version, and access-policy metadata. **INV-1**
- The declaration's sole tracked owner is
  `sources/ford-credit-auto-owner-trust-2024-a.json`; generated caches never
  become alternate manifests.
- Acquisition uses an identifying user agent, finite timeout, byte ceiling,
  atomic temporary file, verified cache reuse, and cleanup after failure.
  **INV-10**
- Raw downloads land only in an ignored caller-selected directory. Neither the
  default path nor errors expose or retain loan rows. **INV-9**
- Tests prove rejection of a missing EX-102, excess bytes, checksum drift,
  partial response, and non-SEC document URL without network access.
- `notebooks/credit_research.py` contains stable sections `00` through `11`,
  with sections `00` and `01` executable and later sections stating their
  governing question without placeholder implementations. **INV-11**
- The Zed-selected `.venv` is project-local, reproducible through `uv sync
  --locked`, and includes an `ipykernel`; no global Python mutation is required.

**How to verify:**

- `uv sync --locked`
- `uv run --locked python -m unittest tests.test_source tests.test_acquire`
- `uv run --locked python -m quantcredit.source --help`
- `uv run --locked python -m quantcredit.acquire --help`
- Restart the Zed kernel and run notebook sections `00` and `01` in order.
- Run acquisition once against an empty temporary directory, rerun against the
  verified cache, and compare the manifest observation; do not retain the raw
  directory in Git.

**Agent notes:**

- Parent: this specification / Stage 1
- Depends on: none
- Fixed decisions: repository name `quantcredit`; Python `# %%` source is
  canonical; standard-library acquisition; no mutable latest query during
  evidence; no consumer rows or notebook outputs in Git

**Out of scope:**

- XML field parsing, target semantics, `.ipynb` export, graph construction,
  model training, and source expansion beyond the declared trust-year.

#### Issue 1.2: Parse and validate monthly loan snapshots

**What and why:** Stream the official auto-loan XML into explicit host facts and
prove identity, field-state, and transition preconditions before any feature or
label exists.

**Status:** Complete at implementation revision `a860203`.

**Done when:**

- The parser validates the official v3.1 namespace/root against the
  manifest-declared schema version and validates required identity/reporting
  elements before yielding typed snapshots.
- Canonical loan and snapshot keys enforce **INV-2**; contradictions include
  accession and element names but never raw row dumps.
- Source states preserve the distinctions in **INV-3**.
- Memory use is bounded by one XML element plus aggregate audit state rather
  than total source rows.
- A synthetic fixture covers valid continuity and every named error/state case
  without containing real consumer records. **INV-9**

**How to verify:**

- `uv run --locked python -m unittest tests.test_panel`
- Measure peak resident memory on one declared EX-102 file and record the file
  size, loan count, command, device, and inspected revision.

**Evidence:**

- The official v3.1 XSD qualifies `assetData` and its children with
  `http://www.sec.gov/edgar/document/absee/autoloan/assetdata`; the pinned Ford
  documents use that namespace but do not encode a version attribute. The
  manifest therefore owns version identity while the document proves the
  namespace/root contract.
- The XSD requires `assetTypeNumber` and `assetNumber`; this research boundary
  additionally requires both reporting dates and checks the ending date against
  the filing declaration. Schema-defined repeated fields remain tuples rather
  than being silently collapsed.
- Optional absence remains `missing`, numeric zero remains a reported value,
  and zero-balance code `99` remains field-specific `unavailable`; the parser
  does not invent a generic not-applicable state or interpret disappearance as
  an event.
- `loanMaturityDate` changes for continuing loans and vehicle descriptors receive
  rare corrections, so neither is classified as an immutable origination fact.
- All twelve pinned documents passed from merged revision `e3aa924`: 408,052 snapshots, 38,224
  loans, 12 periods, no duplicate snapshot keys, and no contradictions among the
  retained immutable origination fields.
- On an arm64 Apple M4 with 32 GiB RAM, the 136,232,097-byte January file yielded
  38,155 snapshots in 15.47 seconds with 28,540,928 bytes maximum resident set
  size. A relative-path reproduction command is:

  ```console
  /usr/bin/time -l uv run --locked python -c 'from pathlib import Path; from quantcredit.panel import read_snapshots; from quantcredit.source import load_manifest; manifest=load_manifest(); filing=manifest.filings[0]; path=Path("data/sec") / filing.accession / "autoloanmonthlydeal1153pool.xml"; print(sum(1 for _ in read_snapshots(path, manifest, filing)))'
  ```

**Agent notes:**

- Parent: this specification / Stage 1
- Depends on: Issue 1.1
- Fixed decisions: `quantcredit.panel` owns SEC semantics; streaming parser; no
  public TinyMesh dataset API

**Out of scope:**

- Choosing a prediction target, filling missing values, deriving cohort
  features, lowering a graph, or training a model.

#### Issue 1.3: Audit transitions and decide usable targets

**What and why:** Turn validated snapshots into aggregate evidence that accepts
or rejects default, prepayment, and loss targets before modeling can hide a
semantic error.

**Status:** Complete at implementation revision `d620625`.

**Done when:**

- The bounded `python -m quantcredit.audit` command emits only aggregate JSON
  and fails rather than emitting partial evidence. **AC-1**, **INV-9**
- The audit reports source coverage, identity continuity, missingness, each
  observed state transition, disappearance/reappearance, and immutable-field
  contradictions.
- Each candidate target has a field-level derivation, horizon, competing-event
  policy, censoring rule, and observed/derived/rejected decision. **AC-2**
- Notebook sections `02` through `04` call the same tested owners, show compact
  aggregate evidence, and reproduce from a restarted kernel. **INV-12**,
  **INV-13**, **AC-8**
- `docs/research/consumer-credit-data.md` binds the decision to exact source
  checksums, repository revision, command, result, limits, and conditions for
  reopening it.

**How to verify:**

- `uv run --locked python -m unittest tests.test_panel tests.test_targets`
- `uv run --locked python -m quantcredit.audit`
- Restart the Zed kernel and run notebook sections `00` through `04` in order.
- `uv run --locked --group lint ruff check .`
- `uv run --locked --group lint mypy`
- `uv run --locked python -m unittest discover -s tests -p 'test_*.py'`
- `uv build`

**Agent notes:**

- Parent: this specification / Stage 1
- Depends on: Issue 1.2
- Fixed decisions: no target inferred from disappearance; notebook remains the
  narrative shell; research record closes negative evidence; do not add model
  or TinyMesh runtime code

**Out of scope:**

- GBM/GINE comparison, feature selection, hyperparameter search, source
  expansion, and runtime promotion.

**Evidence:**

- All twelve exact source pins produced 408,052 snapshots, 38,224 loan
  identities, 369,668 contiguous transitions across 41 observed transition
  types, zero duplicate keys, and zero retained immutable contradictions.
- The accepted three-report serious-delinquency-or-charge-off target has 2,855
  positives and 287,209 fully observed negatives. The audit separately reports
  18,724 competing terminal events, 88,887 right-censored cutoffs, and 61,013
  ineligible cutoffs.
- Standalone prepayment is rejected because Schedule AL code 1 combines prepaid
  and matured loans. Ultimate net loss is rejected because the bounded panel
  does not observe a complete recovery horizon.
- Exact pins, command, results, limits, and reopening conditions are retained in
  `docs/research/consumer-credit-data.md`.

### Stage 2: Classical credit-analysis toolkit and baseline

- **Outcome:** The canonical notebook and transparent helper modules produce a
  causally valid shallow-GBM baseline, calibration and error diagnostics, and
  exposure-aware economic interpretation suitable for rehearsal and research.
- **Depends on:** Stage 1 accepts at least one target with sufficient events and
  uncensored horizon coverage; source expansion is pinned separately if twelve
  reports cannot support the chosen horizon and temporal folds.
- **Invariants:** **INV-4**, **INV-9**, and **INV-11** through **INV-14**.
- **Invalidating assumption:** Securitized loan performance may lack the fields,
  horizon, or event count required for meaningful expected-loss analysis; a
  rank-only target may be the honest stopping point.
- **Proof:** **AC-4**, **AC-8**, and **AC-9** across identical temporal
  populations, plus a revision-bound classical-baseline conclusion.
- **Refine after:** Stage 1 fixes target semantics, event frequency, useful
  fields, horizon, and issuer consistency. Do not finalize module-level issues
  before that evidence.

#### Issue 2.1: Freeze the causal chronological protocol

**What and why:** Choose prediction cutoffs whose earlier labels have fully
matured before the next fold begins. A random row split or adjacent monthly
cutoffs would let future performance influence an apparently earlier model.

**Status:** Complete at implementation revision `09731d4`.

**Decision:** For the twelve-report pilot and three-report target horizon, train
at 2025-01-31 with labels observed through 2025-04-30, validate at 2025-05-31
with labels observed through 2025-08-31, and test once at 2025-09-30 with labels
observed through 2025-12-31. A longer panel may include every earlier training
cutoff whose full horizon ends before validation.

**Done when:**

- `causal_split` rejects nonpositive horizons, shuffled, duplicate,
  gapped, and insufficient report sequences.
- Every training label horizon ends before validation, and the validation label
  horizon ends before test. **INV-4**
- Notebook section `05` invokes the tested owner and displays the cutoffs and
  label-maturity dates without implementing split arithmetic in the cell.

**How to verify:**

- `uv run --locked python -m unittest tests.test_splits`
- Restart the Zed kernel and run notebook sections `00` through `05` in order.
- `uv run --locked --group lint ruff check .`
- `uv run --locked --group lint mypy`

#### Issue 2.1a: Make aggregate evidence visually inspectable

**What and why:** Render the semantic audit and causal split as deterministic
figures so rare states, transitions, censoring, and label maturity are visible
before modeling compresses them into metrics.

**Status:** Complete at implementation revision `474ec40`.

**Decision:** Matplotlib owns canonical figures and static export; Seaborn is an
axes-level statistical companion. Plotly and Bokeh remain absent until a
specific browser interaction or linked-selection contract earns their runtime
and export complexity. The scoped Sapphire theme snapshots selected visual
tokens inspected from Reia revision
`0ad104c8bfbf7a08232ca45fefea8509e22d9fce`; `quantcredit` never imports Reia.

**Done when:**

- `plot_audit(audit)` shows reported population, state prevalence, row-normalized
  transitions, and target disposition from aggregate values only. **INV-9**
- `plot_split(split)` distinguishes feature cutoffs from label maturity for all
  three folds. **INV-4**
- `sapphire()` scopes Matplotlib state and restores the caller's prior theme.
- Notebook sections `04a` and `05a` call the same tested visual owners.
- Tests verify figure semantics, scales, lanes, and theme cleanup; full-resolution
  synthetic and real aggregate renders receive visual inspection.

#### Issue 2.1b: Make canonical research results fluent

**What and why:** Keep live notebook syntax close to the research language:
produce an aggregate audit or causal split, inspect it, then plot that same
value without remembering a second module-level verb.

**Status:** Complete at implementation revision `1806551`.

**Decision:** `quantcredit.audit(...)` returns an `Audit` value and
`quantcredit.split(...)` returns a `CausalSplit`; both expose `.plot()` as a
small notebook affordance. The methods delegate to `plot_audit` and
`plot_split`, which remain the single functional plotting owners. The short
constructors directly alias the existing descriptive owners; they add no
wrappers or parallel paths.

**Done when:**

- The canonical notebook uses `qc.audit(...)`, `audit.plot()`, `qc.split(...)`,
  and `split.plot()`.
- Aggregate audit fields are named attributes rather than an untyped outer
  dictionary.
- Audit CLI JSON, functional plot calls, and existing causal behavior remain
  intact.
- API, audit, split, visual, type, lint, notebook, and package checks pass.

**Next issue:** Materialize only eligible loan-cutoff examples at these dates,
declare past-only features and excluded leakage fields, and report fold-level
population and event counts before importing a model.

### Stage 3: Matched static GBM/GINE experiment

- **Outcome:** Determine whether true typed relations add out-of-time predictive
  value beyond the same flattened past-only cohort information.
- **Depends on:** Stage 2 establishes the raw GBM, temporal populations,
  train-only transformations, metrics, and economic interpretation.
- **Invariants:** **INV-4** through **INV-9**.
- **Invalidating assumption:** Existing GINE may be equivalent to or worse than
  the enriched GBM once information and evaluation are matched.
- **Proof:** **AC-3** through **AC-5**, with one frozen final test query.
- **Refine after:** Stage 2 fixes the incumbent, remaining error structure, and
  whether historical cohort facts are available without leakage.

### Stage 4: Reject the graph claim or graduate one missing equation

- **Outcome:** Retain a negative decision, or land exactly one reusable primitive
  whose missing contract caused a measured Stage 3 limitation.
- **Depends on:** Stage 3 controls isolate structural value and name the smallest
  missing equation.
- **Invariants:** **INV-7** and **INV-8**.
- **Invalidating assumption:** Relation-specific transforms, degree-aware
  aggregation, or another new equation may not improve the incumbent; repeated
  experiment syntax alone is not a primitive.
- **Proof:** **AC-6** and **AC-7**, plus the repository's complete promotion and
  delivery gates.
- **Refine after:** Stage 3 evidence. Do not preselect `RelationalConv`, PNA, or
  another API now.

### Stage 5: Fixed-segment temporal monitoring

- **Outcome:** Determine whether existing sparse recurrence improves monthly
  delinquency/prepayment monitoring over seasonal, persistence, GBM, and
  node-local controls, then identify whether masks or per-lane edge values are
  genuinely missing.
- **Depends on:** Stage 1 semantics and a stable segment universe; refine after
  Stage 3 to reuse accepted features and avoid parallel policy definitions.
- **Invariants:** **INV-3** through **INV-9**.
- **Invalidating assumption:** Monthly aggregation may erase the information
  needed for graph structure, or existing static topology may already suffice.
- **Proof:** Matched out-of-time topology controls and an explicit static-edge/
  dynamic-edge or zero/missing counterexample before any API proposal.
- **Refine after:** Stage 3 closes.

## Coverage

| Contract | Delivered by | Proven by |
| --- | --- | --- |
| INV-1, INV-10 | Stage 1 / Issue 1.1 | Acquisition unit failures and pinned manifest |
| INV-2, INV-3 | Stage 1 / Issue 1.2 | Synthetic parser/transition contract tests |
| INV-9 | Stage 1 / Issues 1.1–1.3 | Git inventory and aggregate-only envelopes |
| INV-11–INV-13, AC-8 | Stage 1 / Issues 1.1 and 1.3 | Stable markers and clean-kernel execution through section `04` |
| AC-1, AC-2 | Stage 1 / Issue 1.3 | Revision-bound data audit and research record |
| INV-4, INV-14, AC-9 | Stage 2 | Temporal split, train-only lineage, metric tests, and notebook replay |
| AC-4 | Stage 2 | Revision-bound classical-baseline evidence |
| INV-4–INV-6 | Stage 3 | Split, feature-lineage, and independent-score tests |
| AC-3–AC-5 | Stages 2–3 | Matched validation controls and frozen test evidence |
| INV-7, INV-8 | Stage 4 | Sparse-work inspection and promotion gate |
| AC-6, AC-7 | Stage 4 | Primitive evidence or retained negative decision |

## Open decisions

- **O-1 — Target (decided):** Use first serious delinquency or charge-off within
  three subsequent reports, with every other reported zero-balance code as a
  competing event. Censor missing follow-up and insufficient future reports.
  Reject standalone prepayment and ultimate net loss for this bounded panel.
- **O-2 — Source expansion:** Recommend adding at least one non-prime issuer and
  more vintages only after the one-trust parser and target audit land. Cross-
  originator claims require an issuer-held-out test; the Stage 1 pilot cannot
  support them. Blocks only generalization claims, not Stage 1.
- **O-3 — Primary decision statistic:** Recommend validation log loss with
  calibration and average precision as co-gates; report AUROC for research and
  industry familiarity. Freeze the economic/ranking statistic after Stage 1
  establishes target prevalence and available exposure/loss fields. Blocks the
  Stage 2 protocol.
- **O-4 — Runtime candidate:** No recommendation is binding before Stage 3. If
  true GINE beats enriched tabular controls in Stage 3 but relation erasure
  closes the lift, test a relation-specific sparse equation. If error instead
  tracks degree,
  test degree-aware aggregation. If neither occurs, add no primitive.
- **O-5 — Survival analysis:** Recommend a fixed-horizon classifier when Stage 1
  proves complete horizon observation for the scored population. If excluding
  right-censored loans materially shrinks or selects the population, Stage 2
  must compare an explicit time-to-event baseline before interpreting default
  probabilities. Do not add a survival dependency before that evidence.
