"""Model and transform construction, keyed by name.

Every model in the comparison is built here so the training harness stays
agnostic about which one it is holding. The ImageNet baselines get their own
normalisation statistics and three channels -- that is required for pretrained
weights to mean anything, not a confound -- while everything about the training
recipe itself comes from the shared config.
"""
from torchvision import models, transforms

from src.models.veritasxr import build_ablation, get_model as get_veritasxr

NORMALIZE = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "half": ([0.5], [0.5]),
}


def _replace_head(name, model, num_classes):
    """Swap each torchvision classifier for a `num_classes` output."""
    import torch.nn as nn
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise KeyError(name)
    return model


def build_model(name, spec, num_classes=2, ablation=None):
    """Return an nn.Module for `name`. `spec` is config's per_model[name].

    `ablation` selects a VeritasXR variant (see veritasxr.ABLATIONS); the
    variants are how the custom model's own design decisions get tested.
    """
    if name == "veritasxr":
        return (build_ablation(ablation, num_classes=num_classes) if ablation
                else get_veritasxr(num_classes=num_classes))
    if ablation:
        raise ValueError(f"--ablation applies to veritasxr, not {name!r}")
    builders = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
        "efficientnet_b0": (models.efficientnet_b0,
                            models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    }
    if name not in builders:
        raise KeyError(f"unknown model {name!r}; known: "
                       f"{['veritasxr'] + list(builders)}")
    fn, weights = builders[name]
    model = fn(weights=weights if spec.get("pretrained") else None)
    return _replace_head(name, model, num_classes)


def build_transforms(spec, cfg, train):
    """Shared augmentation pipeline; only channels and normalisation vary.

    Images come off disk as single-channel PNGs at the cache resolution. For the
    three-channel baselines the single channel is replicated, which is what
    feeding a radiograph to an ImageNet backbone means in practice.
    """
    size = cfg["data"]["image_size"]
    aug = cfg["augment"]
    ch = spec.get("in_channels", 1)

    ops = []
    if train:
        ops += [
            transforms.RandomResizedCrop(
                size, scale=tuple(aug["crop_scale"]), ratio=tuple(aug["crop_ratio"]),
            ),
            transforms.RandomRotation(aug["rotation_degrees"]),
            transforms.ColorJitter(brightness=aug["brightness"],
                                   contrast=aug["contrast"]),
        ]
    else:
        ops += [transforms.Resize((size, size))]

    if ch == 3:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.ToTensor())

    norm = spec.get("normalize", "half")
    if norm != "none":
        mean, std = NORMALIZE[norm]
        if ch == 3 and len(mean) == 1:
            mean, std = mean * 3, std * 3
        ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
