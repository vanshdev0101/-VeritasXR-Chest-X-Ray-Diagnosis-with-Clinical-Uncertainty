# VeritasXR — A Comparative Study of CNN Architectures for Chest Radiograph Classification

A custom dual-pathway CNN (VeritasXR) evaluated against classical machine learning and
three pretrained CNN baselines on chest radiographs, under a single shared training
protocol.

> **Status: refactor in progress on `pure-ml-refactor`.**
> The project is being converted from an inference-serving demo into a pure ML/DL study.
> No results from the previous version are carried forward — see *Why the rewrite*.

## Scope

**In scope:** architecture design, training, evaluation, statistical comparison,
explainability.

**Out of scope** (removed in S0): ONNX export, TensorRT compilation, Triton Inference
Server, FastAPI serving, latency benchmarking.

## Why the rewrite

A review of the previous version found three problems that invalidated its headline
claims:

1. **The uncertainty head was never trained.** The training loop discarded the model's
   second output (`verdicts, _ = model(imgs)`), so the uncertainty head's 528,577
   parameters received no gradient and remained at their initialisation — while still
   being served as a clinical triage recommendation.
2. **The benchmark table compared different models.** The "PyTorch baseline" latency was
   a hardcoded ResNet50 literal that was never measured, and the Triton figure came from
   `xray_classifier` — a model deleted in `b7ed44e`. The only valid comparison present
   was ~3.3x, not the 13x claimed.
3. **The model comparison held nothing constant.** VeritasXR and the ResNet baselines
   differed in loss function, epoch count, learning rate, schedule, initialisation, input
   channels, augmentation and checkpoint-selection metric. The accuracy gap was therefore
   not attributable to architecture.

## Plan

| Stage | Work | Status |
|---|---|---|
| S0 | Remove the serving stack, restructure into a package | **done** |
| S1 | Migrate to RSNA Pneumonia with patient-level splits | next |
| S2 | One training harness shared by every model | |
| S3 | Fix and slim the custom CNN; add a self-ablation | |
| S4 | Train every model across 3 seeds | |
| S5 | Metrics with bootstrap CIs and significance testing | |
| S6 | Grad-CAM localisation scored against radiologist boxes | |
| S7 | Report and figures generated from results files | |

## Dataset change

The previous dataset (Kaggle `chest_xray`) is **pediatric only, ages 1–5, from a single
hospital**, with an official validation split of 16 images. Results obtained on it cannot
be claimed to generalise to adults, and a 16-image validation set makes checkpoint
selection noise-dominated.

It is replaced by the **RSNA Pneumonia Detection Challenge** dataset. Figures below are
measured from the download and counted per image, not per annotation row:

| | Kaggle `chest_xray` | RSNA Pneumonia |
|---|---|---|
| Labelled images | 5,856 | 26,684 |
| Population | pediatric, 1–5 years | adults |
| Sites | 1 hospital | multi-site (from NIH CXR14) |
| Labels | binary, physician | 1–3 board-certified radiologists |
| Positive rate | 74% pneumonia | 22.5% lung opacity |
| Label granularity | 2 classes | 3 classes + boxes |
| Bounding boxes | none | 9,555 boxes over 6,012 images |
| Validation split | 16 images | patient-wise, ~15% (~4,000) |

Per-image class distribution:

| Class | Images | Share |
|---|---|---|
| No Lung Opacity / Not Normal | 11,821 | 44.3% |
| Normal | 8,851 | 33.2% |
| Lung Opacity (`Target=1`) | 6,012 | 22.5% |

Note that RSNA is *more* imbalanced than the dataset it replaces, in the opposite
direction. At 22.5% prevalence **accuracy is not a usable metric** — a model that always
predicts negative scores 77.5%. S5 therefore reports AUROC, AUPRC, F1 and balanced
accuracy, and the imbalance is handled identically for every model by the shared harness.

To be explicit about what the dataset change does and does not fix: it does **not** make
the models comparable. That was never a data problem — it came from training each model
under a different loss, schedule and selection metric, and S2's single harness is what
fixes it. What the dataset buys is 4.6x the images, an adult multi-site population,
radiologist labels rather than one physician's, a validation set large enough for
checkpoint selection to mean anything, and the bounding boxes that make S6 possible.

Splits are stratified on the 3-class label rather than the binary target, so the hard
middle class — abnormal but not pneumonia — stays proportional across train, val and
test. The primary task remains binary pneumonia detection for continuity with the
previous version; every manifest carries the `class` column, so the 3-class variant is a
one-line change.

## Performance notes

Measured on an RTX 3070 Ti Laptop (8 GB), batch 32, 8 dataloader workers.

Training-step throughput, and what the bottleneck is *not*:

| Model | it/s | min/epoch |
|---|---|---|
| VeritasXR | 7.94 | 1.22 |
| ResNet50 | 5.04 | 1.93 |
| EfficientNet-B0 | 3.66 | 2.66 |
| DenseNet121 | 2.11 | 4.61 |

DenseNet121 is the slowest of the four despite having 6.96M parameters against
ResNet50's 23.51M: its concatenation-heavy design is memory-bandwidth bound, so
throughput here is set by per-model GPU behaviour rather than by parameter count.

**Rejected: GPU-side batched augmentation.** The data loader in isolation reaches
27.9 it/s decoding only, against 13.3 with single-channel CPU augmentation and
9.3 with three-channel, which suggested augmentation was the limiter. Moving it
to the GPU as per-sample affine matrices (one resample instead of two) measured
**1.02x end to end** across the four models, and 0.93x for VeritasXR. The
loader-only figures were misleading because they were taken with all cores idle
and no GPU work competing; in real training the augmentation was never on the
critical path. Not adopted -- the implementation is in the history if it is ever
worth revisiting at a larger batch size.

One trap worth recording: `torchvision.transforms.v2` applied to a batched
tensor draws a single set of random parameters for the whole batch, so every
image in a batch receives the identical crop and rotation. That would look like
a clean speedup while quietly collapsing augmentation diversity.

## Models compared

| Model | Class | Params |
|---|---|---|
| HOG + linear SVM | classical ML, no deep learning | 6,085 features |
| ResNet50 | residual CNN, ImageNet pretrained | 23.51M |
| DenseNet121 | dense CNN, ImageNet pretrained (CheXNet backbone) | 6.96M |
| EfficientNet-B0 | efficient CNN, ImageNet pretrained | 4.01M |
| VeritasXR | custom dual-pathway CNN, trained from scratch | 17.02M, slimming to ~6M in S3 |

Parameter counts are measured with a two-class head, not the ImageNet
1000-class figures that are usually quoted.

The HOG + SVM entry is the control for whether deep learning is needed at all.
EfficientNet-B0 is the size-matched rival once VeritasXR is slimmed.

## Layout

```
src/
├── data/            dataset and manifest loading        (S1)
├── models/
│   └── veritasxr.py custom dual-pathway CNN
├── training/
│   ├── train_veritasxr.py
│   └── train_resnet_ablation.py   kept for reference; superseded in S2
└── evaluation/
    └── analysis.py  EDA and metric plots
```

## Setup

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Running it

```bash
# 1. build the dataset: DICOM -> PNG cache + patient-wise manifests
python scripts/prepare_rsna.py --raw data/raw --out data/processed

# 2. the gate: splits must be leakage-free before anything trains
python tests/test_no_leakage.py

# 3. smoke test -- one epoch over a few batches for all five models
python train.py --smoke

# 4. a single run, or every model across every seed
python train.py --model veritasxr --seed 0
python train.py --model all
```

Results land in `results/<model>_seed<seed>/` as `metrics.json` plus
`test_scores.npz`. The per-sample test scores are saved so that bootstrap
confidence intervals and significance tests can be run later without
retraining anything.

Developed against an NVIDIA RTX 3070 (8 GB). Training configs in S2 are sized for that;
adjust batch size for other cards.
