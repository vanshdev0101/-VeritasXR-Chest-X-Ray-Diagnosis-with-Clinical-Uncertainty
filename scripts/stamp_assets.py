"""Stamp the site's asset URLs with the current commit.

GitHub Pages serves assets with cache-control: max-age=600, so a returning
visitor can hold a stale script or model for ten minutes. For the stylesheet
that is cosmetic; for the model it is not. New JavaScript paired with a cached
older model disagree about output names and the page fails silently -- showing
a dash where a number should be, with nothing in the console.

Run before committing a site change:

    python scripts/stamp_assets.py
"""
import re
import subprocess
from pathlib import Path

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True, check=True).stdout.strip()
docs = Path("docs")
changed = []

for name in ("index.html", "results.html"):
    p = docs / name
    s = p.read_text()
    out = re.sub(r'(assets/(?:css|js)/[\w.]+)(\?v=[0-9a-f]+)?', rf'\1?v={sha}', s)
    if out != s:
        p.write_text(out)
        changed.append(name)

js = docs / "assets/js/demo.js"
s = js.read_text()
out = re.sub(r"const MODEL_VERSION = '[0-9a-f]+';", f"const MODEL_VERSION = '{sha}';", s)
if out != s:
    js.write_text(out)
    changed.append("demo.js")

print(f"stamped {sha}: {', '.join(changed) if changed else 'nothing to change'}")
