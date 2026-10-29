# VeritasXR site

Static, no build step, no backend. Served from this directory by GitHub Pages.

    index.html      live demo - the model runs in the browser via onnxruntime-web
    results.html    comparative study, charts generated from the run artifacts
    data/           summary.json, written by scripts/export_web_data.py
    model/          veritasxr.onnx - 16.9 MB, exported from models/veritasxr_seed0.pt

Regenerate the data after new runs:

    python scripts/export_web_data.py

Re-export the demo model after retraining, and verify it before shipping:

    python scripts/export_onnx.py --checkpoint models/veritasxr_seed0.pt
    python scripts/check_onnx_parity.py

Preview locally. Use an absolute path: `--directory docs` resolves against the
current shell, and http.server does not complain about a missing directory --
it just returns 404 for everything, which looks like missing files.

    python -m http.server 8077 --directory ~/git/veritasxr/docs

## Design

Generated with the ui-ux-pro-max skill; see `design-system/veritasxr/MASTER.md`.
Minimalism & Swiss Style, variance 3, motion 3 (subtle), density 7. The
"Enterprise Gateway" page pattern the search returned was discarded — it is a
sales template (Contact Sales, Solutions by Industry, logo carousels) and this
is a research artifact. The style, palette and Fira Code / Fira Sans pairing
are kept.

Three accessibility decisions that shaped the build, each from a high-severity
guideline rather than taste:

- **Dragging is never the only route.** WCAG 2.2 AA requires a single-pointer
  alternative to drag operations, so "Choose a file" is the primary control and
  drag-and-drop is a shortcut.
- **Nothing is encoded by colour alone.** Triage bands carry an icon and a text
  label as well as a hue; chart series are separated by dash pattern and point
  shape, not just colour.
- **Every chart has a table.** Each figure ships a "View as table" disclosure so
  the underlying numbers are reachable without perceiving the chart at all.
