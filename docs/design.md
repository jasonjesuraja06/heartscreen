# Design

## Problem

Classify single-lead ECG recordings from the PhysioNet/CinC 2017 challenge into
Normal (N), atrial fibrillation (A), Other rhythm (O), and Noisy (~), then reuse
the classifier as the detection stage of a screening pipeline for long
ambulatory recordings. The public training set has 8,528 records at 300 Hz,
9 to 61 s long (median 30 s), with class counts N 5,076 / A 758 / O 2,415 / ~ 279
under the revised v3 labels. The revised labels are used because the organizers
corrected several hundred annotations after the challenge.

## Preprocessing

Records are filtered with a zero-phase order-4 Butterworth bandpass at
0.5 to 40 Hz. The low cut removes baseline wander without touching clinically
relevant low-frequency content; the high cut removes powerline interference and
most muscle noise while keeping QRS energy, which concentrates between 5 and
25 Hz. Zero-phase filtering (forward-backward) avoids phase distortion of the
QRS complex that a causal filter would introduce.

Each record is z-scored individually. Absolute amplitude on a handheld
single-lead device depends on skin contact and hand pressure, so it carries
little rhythm information and much inter-record variance.

The network consumes fixed 30 s windows (9,000 samples). 30 s is the median
record length and holds roughly 30 to 40 beats, enough RR intervals to expose
the irregularity that separates AF from sinus rhythm. Shorter records are
right-padded and carry an explicit padding mask; longer records are randomly
cropped during training (a free augmentation) and evaluated as the average of
logits over up to 3 evenly spaced crops.

Filtering and normalization run once over the whole dataset and are cached to a
single flat array with offsets (5 s for all 8,528 records), so epochs never
touch wfdb or scipy.

## Model

A residual 1D CNN in Flax with 821,348 parameters: a stride-2 stem (kernel 15)
plus stride-2 max pool, then 4 stages of 2 residual blocks (kernel 7) at widths
32/64/96/128, each stage after the first opening with a stride-2 block. Total
downsampling is 32x, and each output step sees a receptive field of 1,291
samples (about 4.3 s); global pooling then aggregates rhythm evidence across
the full window. The head is a single dense layer to 4 logits.

The size is a deliberate CPU budget decision: this machine has no CUDA GPU, and
at about 820k parameters a full 5-fold cross-validation run fits in a few hours
while staying far from the regime where 8,528 training records overfit badly.

Normalization is a mask-aware GroupNorm. Group statistics are per-sample,
which makes a record's logits independent of how much padding its batch
neighbors carry, removes train/eval statistics handling, and keeps inference
deterministic. Plain GroupNorm would still fold a sample's own padded zeros
into its time-axis statistics, rescaling the 11% of records shorter than 30 s
by their pad fraction (measured logit shifts above 1.0 for a 9 s record), so
statistics are computed over valid positions only and padded positions are
re-zeroed after every normalization. Padding is always on the right and SAME
convolutions use ceil division of lengths, so a sample with v valid input
steps has exactly ceil(v / 32) valid output steps; valid lengths are tracked
through every stride-2 op and the global average pool masks positions beyond
them. Both properties are unit-tested: logits are invariant to batch
neighbors' padding and to the amount of padding in the sample's own window.

## Training

AdamW with a cosine schedule (2 warmup epochs) and weight decay 1e-4, batch
size 64. The loss is cross-entropy weighted by inverse class frequency, scaled
so the mean weight over training samples is 1 (weights 0.42 / 2.81 / 0.88 /
7.64 for N / A / O / ~ at the observed distribution); without it the 8.9% AF
class is underserved by an optimizer that can buy accuracy cheaply on Normal. Augmentation: random crop,
amplitude scaling in [0.8, 1.2], and random polarity flip, the last because
the recording device is held in either hand and the training set genuinely
contains inverted records.

Train and eval steps are jit-compiled. The eval path pads the final partial
batch to the training batch shape so XLA compiles exactly one program per
shape. The measured step-level speedup of jit over eager on this CPU is 1.9x
(380 ms vs 704 ms per 64-window step).

Each fold trains for a fixed 60 epochs and the final-epoch model is evaluated.
No early stopping or best-epoch selection is done on the validation fold, so
the reported score is not inflated by model selection on the same data.

## Evaluation protocol

The official hidden test set was never released, so the model is evaluated by
stratified 5-fold cross-validation on the public training set, scored with the
official challenge metric: the mean of per-class F1 for N, A, and O (noisy F1
is reported but excluded, as in the challenge). Stratification preserves the
skewed class ratios in every fold. Per-fold scores, their mean and standard
deviation, and the pooled confusion matrix are all reported. Published
hidden-test scores from challenge entries are quoted in results.md as context
only; they are not directly comparable to cross-validation on the training set.

## Screening pipeline

Long recordings (MIT-BIH Long-Term AF Database, 84 Holter records of 24 to
25 hours at 128 Hz) are resampled to 300 Hz, bandpassed once, and scanned by
the classifier over 30 s windows with a 15 s stride, z-scoring each window.
Consecutive windows with AF probability at or above 0.5 merge into candidate
episodes ranked by mean probability.

The design is cheap-model-everywhere, expensive-vetting-on-candidates: XQRS
R-peak detection runs only on one representative window per candidate episode.
Vetting evidence per candidate: flatline fraction and 5 to 25 Hz spectral power
ratio (signal quality), beat count plausibility, and RR irregularity
(coefficient of variation of RR, RMSSD). A candidate passes vetting when the
signal is usable and CV(RR) is at least 0.10; on CinC records this separates
AF (measured example: 0.19) from sinus rhythm (0.04) by a wide margin. The
thresholds are screening heuristics meant to rank and annotate candidates for
human review, not diagnostic rules.

Where rhythm annotations exist, window-level agreement is reported: a window
counts as AF truth when at least half of it is annotated AFIB or AFL, and as an
AF prediction when the argmax class is A. The classifier is trained on 30 s
AliveCor snippets and applied to Holter telemetry resampled from 128 Hz, a real
domain shift; agreement numbers are reported under that caveat rather than
tuned away.

The deployment model used for screening is trained on all 8,528 records with
the CV-validated recipe; its expected quality is the CV estimate.
