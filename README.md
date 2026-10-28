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

It is replaced by the **RSNA Pneumonia Detection Challenge** dataset:

| | Kaggle `chest_xray` | RSNA Pneumonia |
|---|---|---|
| Images | 5,856 | ~30,000 |
| Population | pediatric, 1–5 years | adults |
| Sites | 1 hospital | multi-site (from NIH CXR14) |
| Labels | binary, physician | 1–3 board-certified radiologists |
| Class balance | 26 / 74 | ~50 / 50 |
| Bounding boxes | none | yes, 6 radiologists |
| Validation split | 16 images | built patient-wise, ~15% |

The balanced classes remove the need for the asymmetric loss weighting that previously
made the custom model and the baselines incomparable. The bounding boxes enable S6:
scoring each model's Grad-CAM against radiologist-drawn ground truth, which replaces the
removed uncertainty-triage claim with a measurable one.

## Models compared

| Model | Class | Params |
|---|---|---|
| HOG + linear SVM | classical ML, no deep learning | — |
| ResNet50 | residual CNN, ImageNet pretrained | 25.6M |
| DenseNet121 | dense CNN, ImageNet pretrained (CheXNet backbone) | 8.0M |
| EfficientNet-B0 | efficient CNN, ImageNet pretrained | 5.3M |
| VeritasXR | custom dual-pathway CNN, trained from scratch | 17.0M, slimming to ~6M in S3 |

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
pip install -r requirements.txt
```

Developed against an NVIDIA RTX 3070 (8 GB). Training configs in S2 are sized for that;
adjust batch size for other cards.
