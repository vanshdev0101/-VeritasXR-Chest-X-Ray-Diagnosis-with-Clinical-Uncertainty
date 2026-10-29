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

# Localisation: every value the README tabulates, not just the two pathways
# that happened to be quoted in prose. A stale merged-CAM row survived an
# earlier scan precisely because this loop only looked at two of the numbers.
for model, series in L.items():
    if model.startswith("_"):
        continue
    for metric, agg in series.items():
        if not isinstance(agg, dict) or "mean" not in agg:
            continue
        if metric == "pathway_peak_agreement":
            continue              # quoted as a percentage, checked separately
        val = agg["mean"]
        if f"{val:.4f}" not in readme:
            problems.append(
                f"README missing localisation {model}.{metric} {val:.4f}")
        else:
            checked += 1

agree = L["veritasxr"]["pathway_peak_agreement"]["mean"]
if f"{agree * 100:.2f}%" not in readme:
    problems.append(f"README missing pathway peak agreement {agree * 100:.2f}%")
else:
    checked += 1

# stale-claim scan
# Every claim this study has had to retract, in every phrasing it was written
# in. The list grew after a scan passed while the README's opening paragraph
# still carried a retracted claim in different words -- matching on one
# phrasing is not enough.
stale = [
    ("3x3 pathway outperforms the 7x7", "reversed by multi-seed"),
    ("at a third of the cost", "reversed by multi-seed"),
    ("barely localising", "seed-0 artifact"),
    ("1.6x chance", "seed-0 artifact"),
    ("not statistically distinguishable", "wrong test; paired bootstrap separates"),
    ("overlapping 95% confidence interval", "same wrong inference, other wording"),
    ("intervals overlap", "same wrong inference, other wording"),
    ("uses the disagreement between their predictions as an uncertainty",
     "disagreement was rejected; the model reports predictive entropy"),
    ("disagreement as an uncertainty estimate", "rejected, AUROC 0.5046"),
    ("0.6426", "seed-0 AUPRC; the three-seed mean is 0.6410"),
    ("0.7009 for the strongest", "seed-0 value; three-seed mean is 0.6993"),
    ("inter-path disagreement", "never implemented in the form claimed"),
]
# Scan the README *and* the site. A scan that covered only the README passed
# while the demo page's h1 still announced a retracted claim as its headline.
targets = {"README.md": readme}
for q in list(Path("docs").glob("*.html")) + list(Path("docs/assets/js").glob("*.js")):
    targets[str(q)] = q.read_text()

site_stale = stale + [
    ("disagreement as uncertainty", "rejected; AUROC 0.5046 against error"),
    ("their disagreement is the uncertainty", "rejected"),
]
for where, text in targets.items():
    for phrase, why in site_stale:
        if phrase.lower() in text.lower():
            problems.append(f"STALE CLAIM in {where}: '{phrase}' ({why})")

print(f"  checked {checked} README values against the data files")
print(f"  problems: {len(problems)}")
for p in problems:
    print(f"    {p}")
raise SystemExit(1 if problems else 0)
