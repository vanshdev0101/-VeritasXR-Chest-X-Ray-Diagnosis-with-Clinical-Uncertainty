"""Architecture diagrams for the deck, in light and dark, from the real shapes.

Numbers are the measured ones (scripts print them from the built model), not
redrawn from memory: stem 44,864 params -> 64x56x56, each pathway 384x7x7,
local 863,128, global 3,156,888, total 4,215,352.

    python make_diagrams.py && ls diagrams/
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).parent / "diagrams"
FONT = "Inter, DejaVu Sans, Helvetica, sans-serif"
FS = 1.4   # type scale: these figures are read at 1/2 slide width or less
THEMES = {
    "light": dict(ink="#13161a", mute="#5b6570", line="#8a949e", fill="#ffffff",
                  tint="#eef2f6", accent="#b3261e", accent_tint="#fbe9e7"),
    "dark":  dict(ink="#e9edf2", mute="#9aa5b1", line="#5c6773", fill="#1b1f24",
                  tint="#242a31", accent="#ff6b5e", accent_tint="#3a2320"),
    # Matches the Canva deck template: cream ground, warm brown ink, the
    # template's own olive rule and amber card fill.
    "canva": dict(ink="#524545", mute="#8a7a72", line="#c2b49f", fill="#fbf6f1",
                  tint="#f3ebe1", accent="#a2560f", accent_tint="#ffca8c"),
}


def fit(text, size, box_w, bold=False):
    """Shrink a line until it fits its box.

    Box widths here are hand-placed, so scaling the type up globally pushed
    the longest sub-labels straight through their borders. Estimated from
    Inter's average advance width; conservative by design.
    """
    need = len(str(text)) * size * (0.58 if bold else 0.54)
    avail = box_w - 16
    return size * avail / need if need > avail else size


class SVG:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.p = w, h, t, []

    def box(self, x, y, w, h, label, sub=None, accent=False, r=10):
        """Lays the label and sub-lines out as one centred block.

        Positioning them from the box centre instead clips the last line as
        soon as a box carries three, which is most of them here.
        """
        c = self.t
        lines = [] if sub is None else sub.split("\n")
        h = max(h, 15 * FS * len(lines) + 34 * FS)
        fill, stroke = (c["accent_tint"], c["accent"]) if accent else (c["tint"], c["line"])
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        top = y + (h - (20 * FS + 15 * FS * len(lines))) / 2
        self.text(x + w / 2, top + 16 * FS, label, fit(label, 15 * FS, w, True),
                  weight=600, fill=c["accent"] if accent else c["ink"])
        for i, line in enumerate(lines):
            self.text(x + w / 2, top + 34 * FS + i * 15 * FS, line,
                      fit(line, 12 * FS, w), fill=c["mute"])

    def text(self, x, y, s, size=13, weight=400, fill=None, anchor="middle", italic=False):
        s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self.p.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
                      f'font-weight="{weight}" fill="{fill or self.t["ink"]}" '
                      f'text-anchor="{anchor}"'
                      f'{" font-style=\"italic\"" if italic else ""}>{s}</text>')

    def arrow(self, x1, y1, x2, y2, label=None, accent=False, dash=False):
        c = self.t["accent"] if accent else self.t["line"]
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.p.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c}" stroke-width="1.8" '
                      f'fill="none" marker-end="url(#a{"x" if accent else ""})"{d}/>')
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 7, label, 11.5 * FS, fill=self.t["mute"])

    def elbow(self, x1, y1, x2, y2, label=None, accent=False):
        """Right-angled connector: out, across, in."""
        c = self.t["accent"] if accent else self.t["line"]
        mx = (x1 + x2) / 2
        self.p.append(f'<path d="M{x1},{y1} H{mx} V{y2} H{x2}" stroke="{c}" '
                      f'stroke-width="1.8" fill="none" marker-end="url(#a{"x" if accent else ""})"/>')
        if label:
            self.text(mx, (y1 + y2) / 2, label, 11.5 * FS, fill=self.t["mute"])

    def render(self, name):
        c = self.t
        defs = "".join(
            f'<marker id="a{sfx}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{col}"/></marker>'
            for sfx, col in (("", c["line"]), ("x", c["accent"])))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}"><defs>{defs}</defs>'
               f'<rect width="{self.w}" height="{self.h}" fill="{c["fill"]}"/>'
               + "".join(self.p) + "</svg>")
        p = OUT / f"{name}.svg"
        p.write_text(svg)
        subprocess.run(["rsvg-convert", "-z", "2", "-o", str(p.with_suffix(".png")), str(p)],
                       check=True)


def overview(t):
    s = SVG(1600, 625, t)
    s.text(40, 46, "VeritasXR — forward pass", 22 * FS, 700, anchor="start")
    s.text(40, 70, "4,215,352 parameters · one 224x224 greyscale radiograph in, "
                   "verdict + uncertainty + two attention maps out", 13 * FS, fill=t["mute"], anchor="start")

    s.box(40, 290, 150, 90, "Input", "1 x 224 x 224\ngreyscale")
    s.arrow(195, 335, 245, 335)
    s.box(245, 275, 180, 120, "Stem", "11x11 s2 -> 3x3\n-> maxpool s2\n64 x 56 x 56\n44,864 params")
    # split
    s.elbow(430, 335, 520, 190)
    s.elbow(430, 335, 520, 480)

    s.box(520, 130, 300, 120, "Local pathway  (3x3)", "3 bottleneck stages, stride 2 each\n"
          "+ squeeze-excitation\n384 x 7 x 7  ·  863,128 params (20.5%)")
    s.box(520, 420, 300, 120, "Global pathway  (7x7)", "same depth, wider receptive field\n"
          "+ squeeze-excitation\n384 x 7 x 7  ·  3,156,888 params (74.9%)", accent=True)
    s.text(670, 565, "the only ablation-proven component", 12 * FS, 600, fill=t["accent"])

    s.box(860, 155, 130, 70, "GAP", "-> 384")
    s.box(860, 445, 130, 70, "GAP", "-> 384")
    s.arrow(825, 190, 855, 190)
    s.arrow(825, 480, 855, 480)

    s.box(1030, 290, 190, 110, "Merge gate", "concat 768 -> 64 -> 2\nsoftmax, per input\n49,346 params")
    s.elbow(995, 190, 1025, 320)
    s.elbow(995, 480, 1025, 370)

    s.box(1270, 275, 170, 80, "Classifier", "384 -> 256 -> 2\n99,586 params")
    s.arrow(1225, 320, 1265, 320)
    s.box(1270, 395, 170, 70, "Verdict", "P(normal), P(pneumonia)")
    s.arrow(1355, 360, 1355, 390)

    # per-path heads and the two derived signals
    s.box(1030, 110, 190, 75, "Per-path heads", "2 x Linear(384, 2)\n1,540 params")
    s.arrow(995, 150, 1025, 150)
    s.elbow(995, 460, 1025, 170)
    s.box(1270, 95, 170, 95, "Disagreement", "Jensen-Shannon\nAUROC 0.5334\nreported, not used")
    s.arrow(1225, 148, 1265, 143)
    s.box(1270, 500, 170, 95, "Uncertainty", "entropy of verdict\nAUROC 0.7747\nthis is what ships", accent=True)
    s.arrow(1355, 467, 1355, 497, accent=True)

    s.render("1-architecture-overview")


def bottleneck(t):
    s = SVG(1200, 410, t)
    s.text(40, 46, "Bottleneck block — the unit both pathways are built from", 22 * FS, 700, anchor="start")
    s.text(40, 70, "Three stages per pathway, stride 2 on each, so 56x56 becomes 7x7. "
                   "k is 3 in the local pathway and 7 in the global one.", 13 * FS, fill=t["mute"], anchor="start")

    y = 180
    s.box(40, y, 130, 80, "in", "C x H x W")
    s.arrow(175, y + 40, 215, y + 40)
    s.box(215, y, 170, 80, "1x1 reduce", "C -> C_out/2\nBN + ReLU")
    s.arrow(390, y + 40, 430, y + 40)
    s.box(430, y, 190, 80, "k x k, stride 2", "the only place the\nkernel size differs", accent=True)
    s.arrow(625, y + 40, 665, y + 40)
    s.box(665, y, 170, 80, "1x1 expand", "-> C_out\nBN")
    s.arrow(840, y + 40, 915, y + 40)
    s.box(915, y, 110, 80, "+  ReLU", None)
    s.arrow(1030, y + 40, 1070, y + 40)
    s.box(1070, y, 90, 80, "out", "C_out")

    # shortcut
    s.p.append(f'<path d="M105,{y+85} V{y+170} H970 V{y+85}" stroke="{t["line"]}" '
               f'stroke-width="1.8" fill="none" stroke-dasharray="5 4" marker-end="url(#a)"/>')
    s.text(537, y + 190, "projection shortcut: 1x1 stride-2 conv + BN, because stride and width both change",
           12, fill=t["mute"])
    s.text(600, 420, "Why a bottleneck: the first version stacked three full-width kxk convolutions. "
                     "At k=7 and 256 channels that one choice held 59% of a 17M-parameter model.",
           12.5, italic=True, fill=t["mute"])
    s.render("2-bottleneck-block")


def se_block(t):
    s = SVG(1200, 415, t)
    s.text(40, 46, "Squeeze-and-Excitation — channel attention at the end of each pathway", 22 * FS, 700, anchor="start")
    s.text(40, 70, "Lets a pathway turn its own channels up or down per image. "
                   "Measured across three seeds; the effect is real on two of them.", 13 * FS, fill=t["mute"], anchor="start")

    y = 160
    s.box(40, y, 150, 90, "384 x 7 x 7", "feature map")
    s.arrow(195, y + 45, 240, y + 45)
    s.box(240, y, 150, 90, "global avg pool", "-> 384 numbers\none per channel")
    s.arrow(395, y + 45, 440, y + 45)
    s.box(440, y, 170, 90, "Linear 384 -> 24", "ReLU\nreduction 16")
    s.arrow(615, y + 45, 655, y + 45)
    s.box(655, y, 170, 90, "Linear 24 -> 384", "sigmoid\ngains in (0, 1)")
    s.arrow(830, y + 45, 880, y + 45)
    s.box(880, y, 180, 90, "per-channel multiply", "384 x 7 x 7\nre-weighted", accent=True)
    s.p.append(f'<path d="M115,{y+95} V{y+150} H970 V{y+95}" stroke="{t["line"]}" '
               f'stroke-width="1.8" fill="none" stroke-dasharray="5 4" marker-end="url(#a)"/>')
    s.text(540, y + 172, "the original map is what gets scaled", 12 * FS, fill=t["mute"])
    s.text(600, 380, "Ablation: removing SE costs accuracy on two seeds of three, so its "
                     "contribution is real but not separable at this sample size.",
           12.5, italic=True, fill=t["mute"])
    s.render("3-squeeze-excitation")


def merge_gate(t):
    s = SVG(1200, 470, t)
    s.text(40, 46, "The merge gate — how the two pathways are combined", 22 * FS, 700, anchor="start")
    s.text(40, 70, "A weight predicted from the image itself, not a constant — and the only "
                   "merge of four tried that is never the worst.", 13 * FS, fill=t["mute"], anchor="start")

    s.box(40, 130, 170, 75, "local  384", "pooled features")
    s.box(40, 265, 170, 75, "global  384", "pooled features")
    s.elbow(215, 167, 290, 215)
    s.elbow(215, 302, 290, 230)
    s.box(290, 190, 150, 90, "concat", "768")
    s.arrow(445, 235, 485, 235)
    s.box(485, 190, 170, 90, "Linear 768 -> 64", "ReLU")
    s.arrow(660, 235, 700, 235)
    s.box(700, 190, 150, 90, "Linear 64 -> 2", "softmax")
    s.arrow(855, 235, 895, 235)
    s.box(895, 175, 190, 120, "w_local, w_global", "sum to 1, differ\nper image", accent=True)
    s.arrow(990, 300, 990, 350, accent=True)
    s.box(850, 350, 280, 80, "merged = w_l * local + w_g * global", "384 -> classifier")

    s.text(600, 450, "Three alternatives were trained: a fixed 50/50 average, one global learned scalar "
                     "(which converged to 0.510/0.490), and concatenation. None separates by seed.",
           12.5, italic=True, fill=t["mute"])
    s.render("4-merge-gate")


def input_gate(t):
    s = SVG(1240, 450, t)
    s.text(40, 46, "The input gate — refusing what the model cannot judge", 22 * FS, 700, anchor="start")
    s.text(40, 70, "Runs on the pixels before inference, because the classifier has no way to abstain.",
           13, fill=t["mute"], anchor="start")

    s.box(40, 170, 150, 90, "uploaded image", "any PNG or JPEG")
    s.arrow(195, 215, 235, 215)
    s.box(235, 170, 170, 90, "greyscale 224", "16x16 pool,\nz-scored per image")
    s.arrow(410, 215, 450, 215)
    s.box(450, 120, 200, 60, "1 · colour", "every training film\nis greyscale", r=8)
    s.box(450, 225, 200, 60, "2 · contrast", "blank or near-blank", r=8)
    s.box(450, 315, 200, 60, "3 · layout", "distance + correlation\nvs the mean chest", r=8)
    s.arrow(655, 155, 700, 205)
    s.arrow(655, 250, 700, 225)
    s.arrow(655, 345, 700, 248)
    s.box(700, 185, 150, 80, "any one fails", None, accent=True)
    s.arrow(855, 200, 920, 160, accent=True)
    s.arrow(855, 250, 920, 300)
    s.box(920, 115, 230, 90, "refused", "no verdict shown\n21/22 wrong body parts\n11/11 photographs", accent=True)
    s.box(920, 265, 230, 90, "passed to the model", "costs 3.8% of genuine\nchest radiographs")

    s.text(40, 420, "Each cut sits at the 2nd percentile of the validation split.",
           12 * FS, italic=True, fill=t["mute"], anchor="start")
    s.render("5-input-gate")


for name, theme in THEMES.items():
    for fn in (overview, bottleneck, se_block, merge_gate, input_gate):
        fn(theme)
        last = sorted(OUT.glob("*.svg"))[-1]
    for f in list(OUT.glob("*.svg")) + list(OUT.glob("*.png")):
        if not f.stem.endswith(name) and "-light" not in f.stem and "-dark" not in f.stem:
            f.rename(f.with_name(f"{f.stem}-{name}{f.suffix}"))
print("\n".join(sorted(p.name for p in OUT.iterdir())))
