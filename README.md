# HeartScreen

[![ci](https://github.com/jasonjesuraja06/heartscreen/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonjesuraja06/heartscreen/actions/workflows/ci.yml)

- **0.8271 +/- 0.0084** mean cross-validated challenge F1 over 8,528 CinC 2017 records
- **1,961 recording-hours screened in 15.3 minutes** on an M4 Pro CPU, no GPU
- **0.921 sensitivity, 0.963 specificity** at window level on MIT-BIH Long-Term AF

Raw run artifacts behind these numbers (per-fold logs and summaries, screening
outputs, baseline feature table, trained parameters) are attached to
[release v1.0.0](https://github.com/jasonjesuraja06/heartscreen/releases/tag/v1.0.0).
Full tables and reproduction commands: [docs/results.md](docs/results.md).

Arrhythmia discovery engine for single-lead wearable ECG: a JAX/Flax residual
CNN classifier for the PhysioNet/CinC 2017 task, wrapped in a screening
pipeline that ranks candidate AF episodes in multi-hour recordings.

## Results

Stratified 5-fold cross-validation on the 8,528-record public training set,
revised (v3) labels, scored with the official challenge metric (mean F1 over
Normal, AF, Other). The official hidden test set was never released; see the
protocol notes below.

| Metric | Value |
|---|---|
| Mean CV challenge F1 | 0.8271 +/- 0.0084 |
| Per-class F1, Normal / AF / Other / Noisy | 0.903 / 0.810 / 0.768 / 0.614 |
| RR-feature logistic regression baseline | 0.547 +/- 0.022 |
| Model size | 821,348 parameters |
| Training throughput (M4 Pro CPU) | 140 windows/s |

For context, the four winning 2017 entries scored 0.83 on the withheld test
set, a protocol not directly comparable to cross-validation; the baseline row
shows what the screening pipeline's own hand-built rhythm features achieve
under the identical protocol. Details in [docs/results.md](docs/results.md).

Screening the MIT-BIH Long-Term AF Database with the model retrained on all
8,528 records, on the same CPU:

| Metric | Value |
|---|---|
| Recording-hours screened | 1,961 (84 Holter recordings) |
| End-to-end wall clock | 15.3 min (127.8 recording-hours/min) |
| Window-level sensitivity / specificity / PPV | 0.921 / 0.963 / 0.965 |
| Windows scored | 470,460 (52.7% annotated AF) |
| Healthy-cohort false alarms (NSRDB, 18 subjects) | 0.9 vetted candidates per patient-day |

The PPV holds only because this cohort is AF-enriched; it would fall in a
low-prevalence population, which is what the healthy-cohort row measures.
Full tables, figures, and reproduction commands:
[docs/results.md](docs/results.md).

Where the reduction happens, stage by stage, with the count kept at each step.
The sensitivity and specificity in the last band are measured on this
AF-enriched cohort and do not transfer to a screening population.

![screening funnel](docs/figures/screening_funnel.png)

Top-ranked candidates from the screening run with detected R peaks. The last
panel is the highest-scoring episode that vetting rejected: the model scored
it 1.00, but its QRS-band power ratio of 0.296 fell below the 0.30
signal-quality gate.

![top screening candidates](docs/figures/top_candidates.png)

## Pipeline

Atrial fibrillation is intermittent and often asymptomatic, so it is missed by
spot checks and found by long-duration monitoring, which produces far more
signal than anyone reads by hand. HeartScreen trains a compact
821k-parameter classifier on labeled 30 s single-lead records and makes it
the cheap first tier of a screening stack. The model scans a long recording
window by window, R-peak and signal-quality checks vet whatever it flags,
and the survivors reach a human as a ranked candidate list.

```mermaid
flowchart LR
  A["long recording<br/>resampled to 300 Hz"] --> B["bandpass<br/>0.5 to 40 Hz"]
  B --> C["30 s windows<br/>15 s stride"]
  C --> D["residual 1D CNN<br/>821k params, jit"]
  D -->|every window| E["merge runs of<br/>p(AF) at or above 0.5"]
  E -->|candidates only| F["R-peak and<br/>signal-quality vetting"]
  F --> G["ranked<br/>candidate list"]
```

Why each component is built this way, with the tradeoffs behind the filter
band, window length, class weighting, and model size, is in
[docs/design.md](docs/design.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/); it installs the pinned Python
interpreter and locked dependencies.

```
uv sync
uv run pytest -q                                     # 25 tests pass without data, 2 need CinC
./scripts/download_cinc2017.sh
uv run python -m heartscreen.evaluate --smoke        # ~10 s end-to-end check
uv run python -m heartscreen.evaluate                # full 5-fold CV, hours
```

Screening additionally needs the deployment model and the LTAF data:

```
./scripts/download_ltafdb.sh
uv run python -m heartscreen.train                   # train on all records
uv run python -m heartscreen.screening               # rank candidates
```

## Layout

```
heartscreen/
  data.py            CinC 2017 loading, labels, preprocessed cache
  preprocessing.py   bandpass, normalization, windowing, batching
  models.py          residual 1D CNN with mask-aware normalization
  train.py           fold training, jit steps, deployment training entry
  evaluate.py        stratified CV driver, challenge metric
  screening.py       sliding-window inference, vetting, candidate ranking
configs/             default and smoke configurations
docs/                design rationale, results, figures
scripts/             dataset downloads, dataset and funnel figures, jit benchmark, baseline
tests/               unit tests; data-dependent tests skip without data
```

## Evaluation protocol and limitations

All classifier numbers are cross-validation on the public training set with a
fixed seed. Each fold's final-epoch model is evaluated, with no early
stopping or model selection on the validation fold. Folds split records; patient identities are not
published for this dataset, so patient-level splitting is not possible.
Screening agreement on LTAF is measured across an acquisition mismatch the
model never saw in training (128 Hz Holter telemetry resampled to 300 Hz)
and the vetting thresholds are ranking heuristics, not diagnostic rules.

HeartScreen is a research prototype and is not a medical device.

## License

MIT
