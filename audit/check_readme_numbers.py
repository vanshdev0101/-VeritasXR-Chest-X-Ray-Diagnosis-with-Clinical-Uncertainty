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

# Uncertainty signals. The demo's triage bands rest on this table, and it
# shipped from an 800-image single-seed run.
upath = Path("docs/data/uncertainty.json")
if upath.is_file():
    U = json.loads(upath.read_text())
    checked += 1
    if len(U.get("seeds", [])) < 3:
        problems.append(f"uncertainty.json ran on {U.get('seeds')} seeds")
    checked += 1
    if U.get("n", 0) < 4000:
        problems.append(f"uncertainty.json scored only {U.get('n')} images")
    for key, label in (("entropy", "Predictive entropy of the verdict"),
                       ("disagreement", "Jensen-Shannon pathway disagreement"),
                       ("entropy_plus_5x_disagreement", "Entropy + 5x disagreement")):
        want = U["auroc"][key]
        row = re.search(rf"\|\s*{re.escape(label)}\s*\|([^\n]*)", readme)
        if not row:
            problems.append(f"README uncertainty: no row for {label}")
            continue
        nums = [float(x) for x in re.findall(r"0\.\d{4}", row.group(1))]
        checked += 1
        if not nums or not near(nums[0], want["mean"]):
            problems.append(f"{label}: README {nums[:1]} vs data {want['mean']}")

# Paired-significance table, and the cross-check that caught it being
# single-seed: the paired delta must reconcile with the difference of the two
# mean AUPRCs in the results table. It did not, for a year of commits, because
# the delta came from seed 0 and the means from three.
ref_auprc = models["veritasxr"]["metrics"]["auprc"]["mean"] if "veritasxr" in models else None
for label, key in NAME.items():
    m = models.get(key)
    if not m or key == "veritasxr" or "vs_veritasxr" not in m:
        continue
    v = m["vs_veritasxr"]
    expect = m["metrics"]["auprc"]["mean"] - ref_auprc
    checked += 1
    if abs(v["diff"] - expect) > 2e-3:
        problems.append(
            f"{label}: paired delta {v['diff']:+.4f} does not reconcile with the "
            f"difference of mean AUPRCs ({expect:+.4f}) -- is it single-seed?")
    checked += 1
    if v.get("seeds", 1) < 3:
        problems.append(f"{label}: paired test ran on {v.get('seeds', 1)} seed(s)")
    row = re.search(rf"\|\s*{re.escape(label)} − VeritasXR\s*\|([^\n]*)", readme)
    if not row:
        problems.append(f"README significance: no row for {label}")
        continue
    nums = re.findall(r"[+−-]0\.\d{4}", row.group(1))
    if nums:
        checked += 1
        got = float(nums[0].replace("−", "-").replace("+", ""))
        if not near(got, v["diff"]):
            problems.append(f"{label} paired delta: README {got} vs data {v['diff']}")

# Calibration table. This block used to search the whole README with a regex
# that could match the results table instead, and then `pass` on a mismatch --
# so it verified nothing while counting as a check. It now reads the section it
# is about, and fails.
section = readme.split("## Calibration")[1].split("## Localisation")[0] \
    if "## Calibration" in readme else ""
for label, key in NAME.items():
    if key not in C or not section:
        continue
    row = re.search(rf"\|\s*\**{re.escape(label)}\**\s*\|([^\n]*)", section)
    if not row:
        problems.append(f"README calibration: no row for {label}")
        continue
    nums = [float(x) for x in re.findall(r"0\.\d{4}", row.group(1))]
    if len(nums) < 4:
        problems.append(f"README calibration row for {label}: unreadable")
        continue
    for got, want, what in ((nums[0], C[key]["ece"], "ECE"),
                            (nums[1], C[key]["ece_std"], "ECE std"),
                            (nums[2], C[key]["mce"], "MCE"),
                            (nums[3], C[key]["brier"], "Brier")):
        checked += 1
        if not near(got, want):
            problems.append(f"{label} {what}: README {got} vs data {want}")

# Temperature-scaling table, if the probe has been run.
tpath = Path("docs/data/temperature.json")
if tpath.is_file():
    T = json.loads(tpath.read_text())
    tsec = readme.split("calibration is cheap to buy".title())[-1] \
        if "Calibration is cheap to buy" in readme else readme
    for label, key in NAME.items():
        if key not in T:
            continue
        row = re.search(rf"\|\s*\**{re.escape(label)}\**\s*\|\s*(\d\.\d{{3}})\s*\|([^\n]*)",
                        tsec)
        if not row:
            problems.append(f"README temperature: no row for {label}")
            continue
        a = T[key]["agg"]
        checked += 1
        if not near(float(row.group(1)), a["temperature"], 5e-4):
            problems.append(f"{label} temperature: README {row.group(1)} "
                            f"vs data {a['temperature']}")
        nums = [float(x) for x in re.findall(r"0\.\d{4}", row.group(2))]
        for got, want, what in zip(nums, (a["before"]["ece"], a["after"]["ece"],
                                          a["before"]["brier"], a["after"]["brier"]),
                                   ("ECE before", "ECE after",
                                    "Brier before", "Brier after")):
            checked += 1
            if not near(got, want):
                problems.append(f"{label} {what}: README {got} vs data {want}")

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
    ("disagreement as an uncertainty estimate", "rejected; AUROC 0.5334, chance"),
    ("0.6426", "seed-0 AUPRC; the three-seed mean is 0.6410"),
    ("0.7009 for the strongest", "seed-0 value; three-seed mean is 0.6993"),
    ("inter-path disagreement", "never implemented in the form claimed"),
    ("the one measurement VeritasXR wins",
     "calibration was seed-0; three-seed mean ranks it second"),
    ("best model in the study by a factor of 3.6", "seed-0 calibration artifact"),
    ("12x lower than resnet50", "seed-0 calibration artifact"),
    ("3.6x lower than the next best", "seed-0 calibration artifact"),
    ("0.0059", "seed-0 ECE; the three-seed mean is 0.0440"),
    ("claims 78.55%", "seed-0 calibration artifact"),
    ("0.7563", "800-image single-seed entropy AUROC; full test set gives 0.7747"),
    ("on 800 test images", "the uncertainty table is the full test set now"),
    ("auroc 0.76.", "unlabelled single-seed figure"),
]
# Scan the README *and* the site. A scan that covered only the README passed
# while the demo page's h1 still announced a retracted claim as its headline.
targets = {"README.md": readme}
for q in list(Path("docs").glob("*.html")) + list(Path("docs/assets/js").glob("*.js")):
    targets[str(q)] = q.read_text()

site_stale = stale + [
    ("disagreement as uncertainty", "rejected; AUROC 0.5334 against error"),
    ("their disagreement is the uncertainty", "rejected"),
]
# A retraction has to be allowed to name the number it is retracting. Lines
# that do so are marked, and only those lines are exempt -- the phrase is still
# a failure anywhere else, including elsewhere in the same file.
RETRACTION = ("previously reported", "was one seed", "seed 0 was its best",
              "across the three seeds", "across three seeds its ece",
              "read as a 12", "retracted", "seed-0 artifact", "no longer")


def exempt(line):
    return any(m in line.lower() for m in RETRACTION)


for where, text in targets.items():
    lines = text.splitlines()
    for phrase, why in site_stale:
        # The exemption covers the sentence, not the line: prose wraps, so a
        # retracted number often lands a line below the words retracting it.
        hits = [ln for i, ln in enumerate(lines)
                if phrase.lower() in ln.lower()
                and not any(exempt(x) for x in lines[max(0, i - 2):i + 2])]
        if hits:
            problems.append(f"STALE CLAIM in {where}: '{phrase}' ({why})"
                            f"\n      {hits[0].strip()[:100]}")

print(f"  checked {checked} README values against the data files")
print(f"  problems: {len(problems)}")
for p in problems:
    print(f"    {p}")
raise SystemExit(1 if problems else 0)
