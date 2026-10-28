"""Manifest-backed RSNA dataset.

The manifests written by scripts/prepare_rsna.py hold one row per image, so
there is no grouping to do here -- the patient-level split already happened at
preparation time and tests/test_no_leakage.py guards it.

Images are cached as single-channel PNGs and loaded as mode "L". Fanning out to
three channels for the ImageNet-pretrained baselines is the transform's job, so
this class stays agnostic about which model is consuming it.
"""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RSNADataset(Dataset):
    def __init__(self, manifest, root, transform=None, with_boxes=False):
        self.df = pd.read_csv(manifest)
        self.root = Path(root)
        self.transform = transform
        self.with_boxes = with_boxes

    def __len__(self):
        return len(self.df)

    @property
    def targets(self):
        """Exposed for class-balance checks and weighted samplers."""
        return self.df.target.to_numpy()

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        target = int(row.target)
        if not self.with_boxes:
            return img, target
        # Normalised [x, y, w, h] rows; variable count per image, so a loader
        # using with_boxes=True needs its own collate_fn. S6 adds that when the
        # localisation scoring needs batching.
        boxes = torch.tensor(json.loads(row.boxes), dtype=torch.float32).reshape(-1, 4)
        return img, target, boxes
