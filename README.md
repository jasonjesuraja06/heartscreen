# HeartScreen

Arrhythmia discovery engine for single-lead wearable ECG: a JAX/Flax residual
CNN classifier for the PhysioNet/CinC 2017 task, wrapped in a screening
pipeline that ranks candidate AF episodes in multi-hour recordings.

Atrial fibrillation is intermittent and often asymptomatic, so it is missed by
spot checks and found by long-duration monitoring, which produces far more
signal than anyone reads by hand. HeartScreen trains a compact classifier
(821k parameters) on labeled 30 s single-lead records, then uses it as the
cheap first tier of a screening stack: sliding-window inference over long
recordings, followed by signal-quality and RR-irregularity vetting of
candidates via R-peak detection, yielding a ranked list a human can review.

## Results

Stratified 5-fold cross-validation on the 8,528-record public training set,
revised (v3) labels, scored with the official challenge metric (mean F1 over
Normal, AF, Other). The official hidden test set was never released; see the
protocol notes below.

| Metric | Value |
|---|---|
| Mean CV challenge F1 | 0.827 +/- 0.008 |
| Pooled per-class F1 (N / A / O / ~) | 0.903 / 0.810 / 0.768 / 0.614 |
| Model size | 821,348 parameters |
| Training throughput (M4 Pro CPU) | 140 windows/s |

Screening the MIT-BIH Long-Term AF Database (84 recordings, 1,961
recording-hours) runs end to end in 15.5 min on the same CPU (126.7
recording-hours/min) and agrees with the rhythm annotations at window level
with sensitivity 0.921, specificity 0.963, and PPV 0.965 across 470,460
windows. Full tables, figures, and reproduction commands:
[docs/results.md](docs/results.md).

## Quickstart

```
uv sync
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
scripts/             dataset downloads, dataset figures, jit benchmark
tests/               unit tests; data-dependent tests skip without data
```

## Evaluation protocol and limitations

The CinC 2017 hidden test set is not public, so all classifier numbers are
stratified 5-fold cross-validation on the public training set with a fixed
seed; published hidden-test scores from challenge entries are cited in
docs/results.md as context and are not directly comparable. Each fold's
final-epoch model is evaluated, with no early stopping or model selection on
the validation fold. Screening agreement on LTAF is measured under a real
domain shift (128 Hz Holter telemetry resampled to 300 Hz) and the vetting
thresholds are ranking heuristics, not diagnostic rules.

HeartScreen is a research prototype and is not a medical device.

## License

MIT
