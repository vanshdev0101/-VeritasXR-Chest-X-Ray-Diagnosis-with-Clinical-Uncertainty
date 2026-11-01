# Presentation

The project defence deck, and everything needed to rebuild it.

| File | What it is |
|---|---|
| [`VeritasXR.pdf`](VeritasXR.pdf) | The deck, 15 slides — read this one |
| `VeritasXR.pptx` | The editable deck |
| [`SLIDE-CONTENT.md`](SLIDE-CONTENT.md) | Slide-by-slide text, plus the appendix material cut to fit |
| `diagrams/*.svg` | Architecture figures, in three themes (`light`, `dark`, `canva`) |
| `charts/*.png` | Result figures, generated from this repo's own run artefacts |
| `build_deck.py` | Composes the deck on the template |
| `make_diagrams.py` | Draws the architecture figures |
| `make_charts.py` | Draws the result figures |

## Where the numbers come from

No figure in this deck is typed in by hand.

- `make_diagrams.py` reads the real layer shapes and parameter counts by
  instantiating `src/models/veritasxr.py`, so the architecture diagrams cannot
  drift from the code.
- `make_charts.py` reads `docs/data/summary.json` and
  `docs/data/uncertainty.json`, and recomputes the per-seed ablation deltas
  from each `results/<run>_seed<N>/metrics.json`.

Retrain anything and the slides follow:

```bash
python presentation/make_diagrams.py      # architecture figures
python presentation/make_charts.py        # result figures
python presentation/build_deck.py         # the deck itself
```

## Rebuilding the deck

`build_deck.py` composes onto a Canva template, which is **not redistributed
here** — the finished presentation is mine to publish, the template is not.
Point the script at your own copy:

```bash
export VERITASXR_TEMPLATE="/path/to/your-template.pptx"
python presentation/build_deck.py
```

The script clones the template's own image-free slides as design bases and
composes the rest, so a different template needs its slide indices adjusted.

## Notes

- PNG renders of the diagrams are generated, not tracked. Run
  `make_diagrams.py` to produce them (it needs `rsvg-convert`); the SVGs are
  the source.
- Tables are drawn from shapes rather than PowerPoint tables, because Canva's
  PPTX import does not carry native tables across.
- The literature survey cites eleven studies from 2022–2026. Every DOI was
  checked against Crossref, and every reference on the slide is a live link.
