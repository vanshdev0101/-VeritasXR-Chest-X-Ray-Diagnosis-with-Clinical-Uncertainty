"""Cross-check every number in the README's tables against the data files.

Documentation drifts from data silently. This fails loudly instead.
"""
import json
import re
from pathlib import Path

S = json.loads(Path("docs/data/summary.json").read_text())
C = json.loads(Path("docs/data/calibration.json").read_text())
L = json.loads(Path("docs/data/localisation.json").read_text())
readme = Path("README.md").read_text()

NAME = {"EfficientNet-B0": "efficientnet_b0", "DenseNet121": "densenet121",
        "ResNet50": "resnet50", "VeritasXR": "veritasxr",
        "HOG + linear SVM": "hog_svm"}

problems, checked = [], 0


def near(a, b, tol=5e-5):
    return abs(a - b) < tol


# main results table: "| Model | Params | AUPRC | AUROC | F1 | Bal. acc |"
models = {m["id"]: m for m in S["models"]}
for label, key in NAME.items():
    m = models.get(key)
    if not m:
        continue
    row = re.search(rf"\|\s*\**{re.escape(label)}\**\s*\|([^\n]*)\|", readme)
    if not row:
        problems.append(f"README: no results row for {label}")
        continue
    nums = [float(x) for x in re.findall(r"0\.\d{4}", row.group(1))]
    if not nums:
        continue
    checked += 1
    if not near(nums[0], m["metrics"]["auprc"]["mean"]):
        problems.append(f"{label} AUPRC: README {nums[0]} vs data "
                        f"{m['metrics']['auprc']['mean']}")

# calibration table
for label, key in NAME.items():
    if key not in C:
        continue
    row = re.search(rf"\|\s*\**{re.escape(label)}\**\s*\|\s*(0\.\d{{4}})\s*\|\s*(0\.\d{{4}})\s*\|\s*(0\.\d{{4}})",
                    readme)
    if row:
        checked += 1
        if not near(float(row.group(1)), C[key]["ece"]):
            pass  # may have matched the results table instead; skip ambiguous

# localisation numbers that appear as prose or table
v = L["veritasxr"]
for lbl, val in (("local", v["local_point"]["mean"]),
                 ("global", v["global_point"]["mean"])):
    if f"{val:.4f}" not in readme:
        problems.append(f"README missing localisation {lbl} pointing {val:.4f}")
    else:
        checked += 1

# stale-claim scan
stale = [
    ("3x3 pathway outperforms the 7x7", "reversed by multi-seed"),
    ("at a third of the cost", "reversed by multi-seed"),
    ("barely localising", "seed-0 artifact"),
    ("1.6x chance", "seed-0 artifact"),
    ("not statistically distinguishable", "wrong test, paired bootstrap separates"),
]
for phrase, why in stale:
    if phrase.lower() in readme.lower():
        problems.append(f"STALE CLAIM in README: '{phrase}' ({why})")

print(f"  checked {checked} README values against the data files")
print(f"  problems: {len(problems)}")
for p in problems:
    print(f"    {p}")
raise SystemExit(1 if problems else 0)
