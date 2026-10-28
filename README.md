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

Test set: 4,006 radiographs, 22.5% positive. Bootstrap 95% confidence intervals
over 2,000 resamples.

| Model | Params | AUPRC | 95% CI | AUROC | F1 | Bal. acc |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 4.01M | 0.7009 | [0.6699, 0.7288] | 0.8766 | 0.6065 | 0.7860 |
| ResNet50 | 23.51M | 0.6849 | [0.6536, 0.7140] | 0.8718 | 0.6312 | 0.7811 |
| DenseNet121 | 6.96M | 0.6824 | [0.6529, 0.7120] | 0.8706 | 0.6287 | 0.7962 |
| **VeritasXR** | **4.22M** | 0.6426 | [0.6111, 0.6742] | 0.8492 | 0.6020 | 0.7642 |
| HOG + linear SVM | 6,085 feat | 0.3834 | [0.3585, 0.4059] | 0.7297 | 0.5434 | 0.7240 |

Three readings worth stating plainly:

- **VeritasXR's interval overlaps all three pretrained baselines.** On a single
  seed it is not statistically distinguishable from models initialised on
  ImageNet, despite training from random initialisation on 18,677 images. Only
  the classical control separates cleanly.
- **Capacity is not the limiting factor.** ResNet50 carries 5.6x the parameters
  of EfficientNet-B0 and does not outperform it. On a dataset this size, more
  model buys nothing.
- **Deep learning earns its place, measurably.** The HOG + SVM control sits
  0.26 AUPRC below the deep models, which is what makes that gap a result
  rather than an assumption.

**Accuracy is deliberately absent from the headline.** At 22.5% prevalence a
model that always predicts negative scores 77.5% accuracy while detecting
nothing. Accuracy is recorded in the run artifacts for completeness and is never
used to select a checkpoint; AUPRC is.

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
| Two kernel scales | The 3x3 branch resolves ~119 px of the 224 px field, the 7x7 branch ~215 px — local detail against whole-lung structure |
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

## Ablation

The constructor flags are the ablation. `python train.py --model veritasxr
--ablation all` sweeps every variant:

| Variant | Tests |
|---|---|
| `full` | — |
| `no_se` | Does squeeze-excitation earn its parameters? |
| `local_only` | Is the 3x3 branch sufficient alone? (1.01M params) |
| `global_only` | Is the 7x7 branch sufficient alone? (3.30M params) |
| `fixed_merge` | Does the merge weight do anything at all? |
| `scalar_merge` | Is a dataset-wide weight as good as a per-input gate? |
| `concat_merge` | Does summing the pathways lose information against concatenating? |

## Limitations

- Single dataset, single task, no external validation cohort.
- One shared learning rate for every model rather than a per-model sweep.
- Labels derive from radiologist report review, not confirmed microbiology.
- Uncertainty is epistemic only.
- Results in the table above are from one seed; multi-seed intervals are
  regenerated by re-running the exporter once additional seeds complete.
