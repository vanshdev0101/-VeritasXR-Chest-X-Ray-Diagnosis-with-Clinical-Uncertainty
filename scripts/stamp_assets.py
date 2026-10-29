"""Stamp the site's asset URLs with a hash of their own contents.

GitHub Pages serves assets with cache-control: max-age=600, so a returning
visitor can hold a stale script or model for ten minutes. For the stylesheet
that is cosmetic; for the model it is not. New JavaScript paired with a cached
older model disagree about output names and the page fails silently.

The version token is a content hash rather than the commit SHA. A commit SHA
cannot be correct: the stamp has to be written before committing, but the SHA
is only known after, so the stamped value always lagged by one commit. A
content hash is known at write time, is stable when nothing changed, and
changes exactly when the asset does -- which is what the cache needs to know.

    python scripts/stamp_assets.py
"""
import hashlib
import re
from pathlib import Path

DOCS = Path("docs")


def digest(path: Path, n: int = 8) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def main():
    model = DOCS / "model" / "veritasxr.onnx"
    versions = {p.as_posix().removeprefix("docs/"): digest(p)
                for p in DOCS.glob("assets/**/*") if p.suffix in (".css", ".js")}

    changed = []
    for name in ("index.html", "results.html"):
        p = DOCS / name
        if not p.is_file():
            continue
        s = orig = p.read_text()
        for rel, h in versions.items():
            s = re.sub(rf'({re.escape(rel)})(\?v=[0-9a-f]+)?', rf'\1?v={h}', s)
        if s != orig:
            p.write_text(s)
            changed.append(name)

    if model.is_file():
        js = DOCS / "assets/js/demo.js"
        s = orig = js.read_text()
        s = re.sub(r"const MODEL_VERSION = '[0-9a-f]+';",
                   f"const MODEL_VERSION = '{digest(model)}';", s)
        if s != orig:
            js.write_text(s)
            changed.append("demo.js (model hash)")
            # demo.js just changed, so its own stamp in the HTML is now stale.
            h = digest(js)
            for name in ("index.html",):
                p = DOCS / name
                t = re.sub(r"(assets/js/demo\.js)(\?v=[0-9a-f]+)?", rf"\1?v={h}",
                           p.read_text())
                p.write_text(t)

    print(f"stamped by content hash: {', '.join(changed) if changed else 'nothing to change'}")
    for rel, h in sorted(versions.items()):
        print(f"  {rel:<28} {h}")
    if model.is_file():
        print(f"  {'model/veritasxr.onnx':<28} {digest(model)}")


if __name__ == "__main__":
    main()
