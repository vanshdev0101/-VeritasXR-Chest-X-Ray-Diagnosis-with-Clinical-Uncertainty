# VeritasXR — A Comparative Study of CNN Architectures for Chest Radiograph Classification

A custom dual-pathway CNN evaluated against three ImageNet-pretrained CNNs and a
classical computer-vision baseline on chest radiographs, under a single shared
training protocol.

The architecture runs two convolutional branches in parallel over a shared stem,
differing only in kernel size, and uses the disagreement between their
predictions as an uncertainty estimate. At 4.2M parameters it reaches an AUPRC
of 0.6426 on the held-out test set against 0.7009 for the strongest pretrained
baseline, with overlapping 95% confidence intervals.

**Not a medical device.** This is a research artifact. It is not validated for
clinical use and must not inform any diagnosis or treatment decision.

## Results

Test set: 4,006 radiographs, 22.5% positive. Three seeds per model, reported as
mean ± standard deviation.

| Model | Params | AUPRC | AUROC | F1 | Bal. acc |
|---|---|---|---|---|---|
| EfficientNet-B0 | 4.01M | **0.6993 ± 0.0033** | 0.8781 ± 0.0026 | 0.6182 | 0.7880 |
| DenseNet121 | 6.96M | 0.6872 ± 0.0034 | 0.8728 ± 0.0016 | 0.6166 | 0.7893 |
| ResNet50 | 23.51M | 0.6747 ± 0.0072 | 0.8675 ± 0.0032 | 0.6267 | 0.7772 |
| **VeritasXR** | **4.22M** | 0.6410 ± 0.0039 | 0.8531 ± 0.0028 | 0.5988 | 0.7670 |
| HOG + linear SVM | 6,085 feat | 0.3823 ± 0.0010 | 0.7283 ± 0.0009 | 0.5398 | 0.7207 |

Differences are tested by paired bootstrap over 2,000 resamples: both models
are scored on the *same* resampled test cases, so the shared difficulty of
those cases cancels.

| Comparison | Δ AUPRC | 95% CI | p |
|---|---|---|---|
| EfficientNet-B0 − VeritasXR | +0.0581 | [+0.0402, +0.0762] | < 0.0001 |
| ResNet50 − VeritasXR | +0.0422 | [+0.0230, +0.0621] | < 0.0001 |
| DenseNet121 − VeritasXR | +0.0397 | [+0.0215, +0.0588] | < 0.0001 |
| HOG + SVM − VeritasXR | −0.2593 | [−0.2857, −0.2330] | < 0.0001 |

Four readings:

- **Every pretrained baseline beats VeritasXR, significantly.** The gap is
  0.040–0.058 AUPRC and the paired intervals exclude zero. ImageNet
  initialisation is worth roughly 6–9% relative AUPRC on this task, and
  training from scratch on 18,677 images does not close it.
- **Capacity is not the limiting factor.** ResNet50 carries 5.6x the parameters
  of EfficientNet-B0, finishes third, and has the highest seed-to-seed variance
  of any model (±0.0072). On a dataset this size more model buys instability,
  not accuracy.
- **Deep learning earns its place, measurably.** The classical control sits
  0.259 AUPRC below the custom model, with the paired interval far from zero.
  That gap is a result rather than an assumption.
- **The cheap pathway does most of the work.** In the ablation, the 3x3 branch
  alone (1.01M parameters) outperforms the 7x7 branch alone (3.30M), and comes
  within 0.0126 of the full model using 24% of its parameters.

A note on method, because the two analyses disagree: the marginal 95% intervals
for VeritasXR and the pretrained models overlap, which invites the conclusion
that the difference is unresolved. That inference is wrong. Overlapping
marginal intervals do not imply the absence of a significant difference; the
paired test above is the correct comparison and it separates every pair at
p < 0.0001.

**Accuracy is deliberately absent from the headline.** At 22.5% prevalence a
model that always predicts negative scores 77.5% accuracy while detecting
nothing. Accuracy is recorded in the run artifacts for completeness and is never
used to select a checkpoint; AUPRC is.

## Ablation

Each variant removes or replaces one component of the custom model. Single
seed, so deltas are read against VeritasXR's own seed-to-seed standard
deviation of ±0.0039.

| Variant | Params | AUPRC | Δ vs full | Reading |
|---|---|---|---|---|
| `full` | 4,215,352 | 0.6426 | — | — |
| `scalar_merge` | 4,166,008 | 0.6399 | −0.0027 | within noise |
| `fixed_merge` | 4,166,006 | 0.6392 | −0.0034 | within noise |
| `concat_merge` | 4,264,310 | 0.6341 | −0.0085 | ~2 std |
| `no_se` | 4,177,672 | 0.6335 | −0.0091 | ~2 std |
| `local_only` | **1,008,348** | 0.6300 | −0.0126 | ~3 std |
| `global_only` | 3,302,108 | 0.6233 | −0.0193 | ~5 std |

Two findings and one negative result:

- **The 3x3 pathway outperforms the 7x7 pathway at a third of the cost.**
  `local_only` beats `global_only` by 0.0067 while using 1.01M parameters
  against 3.30M. The expensive branch is not carrying its weight.
- **76% of the model can go for ~2% relative AUPRC.** `local_only` is 0.0126
  below `full` at 24% of the parameters.
- **The per-input merge gate is not doing measurable work.** Gate, dataset-wide
  scalar and fixed 50/50 span 0.0034, which is smaller than the model's own
  seed-to-seed deviation. On this task the gate is indistinguishable from
  averaging the two pathways.

Deltas below roughly two standard deviations should not be treated as
conclusions until the variants are re-run across seeds.

## Architecture

```
input 1x224x224
  |
  stem: Conv 11x11 s2 -> Conv 3x3 -> MaxPool          64x56x56
  |
  +-- local pathway  (3x3 kernels)  3 bottleneck stages -> SE -> GAP   384
  +-- global pathway (7x7 kernels)  3 bottleneck stages -> SE -> GAP   384
  |
  gate: softmax(MLP([local; global]))                 per-input weights
  merged = w0*local + w1*global
  |
  +-- classifier -> verdict (2 logits)
  +-- per-pathway linear heads -> Jensen-Shannon divergence -> uncertainty
```

| Decision | Reason |
|---|---|
| 11x11 stem | Radiographic signal is diffuse texture rather than sharp edges, so the first layer looks wide |
| Two kernel scales | Each unit in the 3x3 branch sees 75 px of the 224 px field, against 187 px in the 7x7 branch — local detail against whole-lung structure |
| Squeeze-excitation | Channel attention using global context, 8,464 parameters per branch |
| Per-input merge gate | Blend weights are predicted from the image, not learned as a dataset-wide constant |
| Derived uncertainty | Jensen-Shannon divergence between two independently supervised pathway heads. No part of the uncertainty path is a free parameter, so it cannot silently go untrained |

The uncertainty signal is epistemic: pathways that agree while both being wrong
score as confident. Adding the predictive entropy of the merged verdict would
cover the aleatoric component.

## Dataset

**RSNA Pneumonia Detection Challenge** — 26,684 labelled radiographs of adults
from multiple sites, annotated by one to three board-certified radiologists with
bounding boxes around opacities.

| Class | Images | Share |
|---|---|---|
| No Lung Opacity / Not Normal | 11,821 | 44.3% |
| Normal | 8,851 | 33.2% |
| Lung Opacity (`Target=1`) | 6,012 | 22.5% |

Split by patient into 18,677 train / 4,001 validation / 4,006 test. Splits are
stratified on the **three-class** label rather than the binary target, so the
hard middle class — radiographically abnormal but not pneumonia, and 44% of the
data — stays proportional across splits. `tests/test_no_leakage.py` asserts no
patient appears in more than one split, and fails if a leak is injected.

The primary task is binary pneumonia detection; every manifest carries the
three-class label, so the three-class variant is a configuration change.

## Comparability

Every model reads one configuration file. Loss, optimiser, learning rate,
schedule, augmentation, mixed-precision policy, early-stopping rule and
checkpoint-selection metric are shared, so differences in the results table are
attributable to the model rather than to the setup.

Only two things vary per model, both recorded in `configs/default.yaml` with
their justification:

- **Channel count and normalisation.** An ImageNet-pretrained backbone needs
  three channels and its own statistics for its weights to mean anything; the
  custom model is natively single-channel. This is correct usage, not a
  confound.
- **The classical baseline has no epochs.** Learning rate and schedule are
  undefined for an SVM, so its departures are listed in a `deviations` field in
  its own results rather than glossed over. It shares the data, the splits, the
  class-weighting and the results contract.

Augmentation deliberately excludes horizontal flipping. Mirroring a chest
radiograph places the heart on the right, which is situs inversus — a rare
pathology, not a valid view of a normal chest.

## Performance

Measured on an RTX 3070 Ti Laptop (8 GB), batch 32, 8 dataloader workers.

| Model | it/s | min/epoch |
|---|---|---|
| VeritasXR | 7.94 | 1.22 |
| ResNet50 | 5.04 | 1.93 |
| EfficientNet-B0 | 3.66 | 2.66 |
| DenseNet121 | 2.11 | 4.61 |

DenseNet121 is the slowest of the four despite holding 6.96M parameters against
ResNet50's 23.51M: its concatenation-heavy design is memory-bandwidth bound.
Throughput here is set by per-model GPU behaviour, not by parameter count.

**Measured and rejected: GPU-side batched augmentation.** The data loader in
isolation reaches 27.9 it/s decoding only, against 13.3 with single-channel CPU
augmentation, which suggested augmentation was the bottleneck. Moving it to the
GPU as per-sample affine matrices measured **1.02x end to end**, and 0.93x for
VeritasXR — the loader-only figures were taken with all cores idle and no GPU
work competing, and augmentation was never on the critical path. Worth recording
so it is not rediscovered.

## Models compared

| Model | Class | Params |
|---|---|---|
| HOG + linear SVM | classical computer vision, no deep learning | 6,085 features |
| ResNet50 | residual CNN, ImageNet pretrained | 23.51M |
| DenseNet121 | dense CNN, ImageNet pretrained (CheXNet backbone) | 6.96M |
| EfficientNet-B0 | efficient CNN, ImageNet pretrained | 4.01M |
| VeritasXR | custom dual-pathway CNN, trained from scratch | 4.22M |

Parameter counts are measured with a two-class head, not the ImageNet
1000-class figures usually quoted. EfficientNet-B0 is the size-matched
comparison for VeritasXR; the HOG + SVM entry is the control for whether deep
learning is needed at all.

## Layout

```
configs/default.yaml          the shared training recipe
scripts/
├── prepare_rsna.py           DICOM -> PNG cache + patient-wise manifests
└── export_web_data.py        run artifacts -> docs/data/summary.json
src/
├── data/                     manifest-backed dataset
├── models/                   architecture and model registry
├── training/                 shared harness, classical baseline
└── evaluation/               metric plots
tests/
├── test_no_leakage.py        split integrity, with an injected-leak control
└── test_model.py             gradient coverage, per-input gate, parameter budget
docs/                         the static site
train.py                      entry point
```

## Setup

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Requires an NVIDIA GPU for training. The dataset is the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
on Kaggle; accept the competition rules and place the archive in `data/raw/`.

## Running it

```bash
# 1. build the dataset: DICOM -> PNG cache + patient-wise manifests
python scripts/prepare_rsna.py --raw data/raw --out data/processed

# 2. the gate: splits must be leakage-free before anything trains
python tests/test_no_leakage.py

# 3. architecture checks: gradient coverage, per-input gate, parameter budget
python tests/test_model.py

# 4. smoke test -- one epoch over a few batches for all five models
python train.py --smoke

# 5. a single run, every model across every seed, or the ablation sweep
python train.py --model veritasxr --seed 0
python train.py --model all
python train.py --model veritasxr --ablation all

# 6. regenerate the site's data
python scripts/export_web_data.py
```

Results land in `results/<model>_seed<seed>/` as `metrics.json`,
`progress.csv` and `test_scores.npz`. Per-sample test scores are saved so that
bootstrap intervals and significance tests can be run without retraining.

## Limitations

- Single dataset, single task, no external validation cohort.
- One shared learning rate for every model rather than a per-model sweep.
- Labels derive from radiologist report review, not confirmed microbiology.
- Uncertainty is epistemic only.
- Results in the table above are from one seed; multi-seed intervals are
  regenerated by re-running the exporter once additional seeds complete.
