"""Build the 14-slide VeritasXR deck on the Canva template.

Clones the template's own image-free slides (1, 3, 6, 9, 11) as design bases so
the title, numbered rows and closing keep their decoration, and composes the
rest with the template's palette and fonts.

    python build_deck.py && soffice --headless --convert-to pdf VeritasXR.pptx
"""
import copy
import os
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
# The Canva template is not redistributed with this repo -- see README.md. Point
# VERITASXR_TEMPLATE at your own copy to rebuild the deck from scratch.
TEMPLATE = Path(os.environ.get(
    "VERITASXR_TEMPLATE",
    Path.home() / "Downloads/Green and Yellow Simple Creative Networking Presentation.pptx"))
OUT = HERE / "VeritasXR.pptx"
DIA = HERE / "diagrams"
CHARTS = HERE / "charts"

INK = RGBColor(0x52, 0x45, 0x45)
MUTE = RGBColor(0x6B, 0x5B, 0x53)
AMBER = RGBColor(0xFF, 0xCA, 0x8C)
AMBER_DK = RGBColor(0xA2, 0x56, 0x0F)
OLIVE = RGBColor(0xCD, 0xC2, 0x9C)
CREAM = RGBColor(0xFB, 0xF6, 0xF1)
SAND = RGBColor(0xF3, 0xEB, 0xE1)
HEAD, BODY = "Prachason Neue", "Inter"

W, H = Inches(20), Inches(11.25)
M = Inches(1.3)                      # side margin


# ---------- template plumbing ----------
def walk(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:       # group
            yield from walk(sh.shapes)


def theme_link_colour(prs, rgb="A2560F"):
    """Recolour hyperlinks in the theme.

    PowerPoint paints link text with the theme's hlink colour and ignores the
    run colour, so the default blue cannot be overridden from the run — this is
    the only place it can be set. folHlink is set to the same value so a link
    does not turn purple once the presenter has clicked it.
    """
    for part in prs.part.package.iter_parts():
        if not str(part.partname).endswith("theme1.xml"):
            continue
        xml = part.blob.decode("utf-8")
        for tag in ("hlink", "folHlink"):
            xml = re.sub(rf'(<a:{tag}><a:srgbClr val=")[0-9A-Fa-f]{{6}}',
                         rf'\g<1>{rgb}', xml)
        part._blob = xml.encode("utf-8")
    return prs


def paint(slide, rgb=CREAM):
    """The template paints its cream ground per slide, so new slides need it too."""
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb
    return slide


def clone(prs, src):
    """Copy a template slide's shapes, and its ground, onto a new blank slide."""
    new = paint(prs.slides.add_slide(prs.slide_layouts[6]))
    for sh in src.shapes:
        new.shapes._spTree.append(copy.deepcopy(sh._element))
    return new


def retext(shape, text, size=None, color=None, bold=None, font=None, align=None):
    """Replace a text frame's words, keeping its paragraph structure.

    Several template boxes centre their text with a leading empty paragraph.
    Deleting it lifts the text clear of the pill drawn behind it, so this
    writes into the paragraph that already holds runs and only blanks the rest.
    """
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    # "has runs" is not enough: some boxes centre their text with a spacer
    # paragraph holding an empty run, and writing into that lifts the words
    # clear of the pill drawn behind them.
    target = next((p for p in paras if p.text.strip()),
                  next((p for p in paras if p.runs), paras[0]))
    for p in paras:
        if p is not target:
            for r in list(p.runs):
                r.text = ""
    for r in list(target.runs)[1:]:
        r._r.getparent().remove(r._r)
    if align is not None:
        target.alignment = align
    run = target.runs[0] if target.runs else target.add_run()
    src = run.font
    run.text = str(text)
    if font:
        src.name = font
    if size:
        src.size = Pt(size)
    if bold is not None:
        src.bold = bold
    if color is not None:
        src.color.rgb = color
    return shape


def pick(slide, needle, left_lt=None, left_gt=None):
    """All shapes whose text starts with `needle`, optionally filtered by column,
    returned top to bottom. Document order is not layout order in this template."""
    out = [sh for sh in walk(slide.shapes)
           if sh.has_text_frame and sh.text_frame.text.strip().startswith(needle)
           and (left_lt is None or sh.left < left_lt)
           and (left_gt is None or sh.left > left_gt)]
    return sorted(out, key=lambda s_: s_.top)


def find(slide, needle):
    for sh in walk(slide.shapes):
        if sh.has_text_frame and needle.lower() in sh.text_frame.text.lower():
            return sh
    raise KeyError(needle)


# ---------- composition ----------
def blank(prs):
    return paint(prs.slides.add_slide(prs.slide_layouts[6]))


def tbox(slide, x, y, w, h, text, size=18, bold=False, color=INK, font=BODY,
         align=PP_ALIGN.LEFT, line=1.25, anchor=MSO_ANCHOR.TOP, space=6):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, lineitem in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(space)
        bullet, txt = (lineitem if isinstance(lineitem, tuple) else (False, lineitem))
        run = p.add_run()
        run.text = ("•   " if bullet else "") + str(txt)
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return box


def card(slide, x, y, w, h, fill=SAND, radius=0.04):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def title(slide, text, sub=None):
    tbox(slide, M, Inches(0.72), W - 2 * M, Inches(1.0), text, 40, True, INK, HEAD)
    if sub:
        tbox(slide, M, Inches(1.85), W - 2 * M, Inches(0.6), sub, 17, False, MUTE)


def picture(slide, name, y, max_h, x=None, max_w=None, folder=None):
    """Place a figure, scaled to fit, centred unless x is given."""
    from PIL import Image
    path = (folder or DIA) / name
    iw, ih = Image.open(path).size
    avail_w = max_w or (W - 2 * M)
    w = min(avail_w, Emu(int(max_h * iw / ih)))
    h = Emu(int(w * ih / iw))
    left = x if x is not None else Emu(int((W - w) / 2))
    return slide.shapes.add_picture(str(path), left, y, w, h)


def _wrap(text, max_chars):
    """Greedy word wrap, to size a cell before anything is drawn."""
    words, lines, cur = str(text).split(), 1, 0
    for word in words:
        need = len(word) + (1 if cur else 0)
        if cur + need > max_chars and cur:
            lines, cur = lines + 1, len(word)
        else:
            cur += need
    return lines


def table(slide, x, y, w, rows, widths, row_h=Inches(0.52), size=14,
          head_fill=OLIVE, emphasis=(), head_size=None):
    """A table drawn from rectangles and text boxes rather than a PowerPoint table.

    Canva's PPTX import does not carry native tables across reliably, and a
    comparison table that silently vanishes is worse than one that is plainer.
    Shapes always survive, and this keeps the same palette and alignment.
    """
    n_col = len(rows[0])
    total = sum(widths)
    col_w = [int(w * ww / total) for ww in widths]
    col_x, run = [], int(x)
    for cw in col_w:
        col_x.append(run)
        run += cw

    def plain(v):
        return v[0] if isinstance(v, tuple) else v

    def numeric(v):
        return str(plain(v)).lstrip("+-−±").replace(",", "").replace(".", "").replace("%", "").isdigit()

    right = [all(numeric(r[j]) for r in rows[1:]) for j in range(n_col)]
    pad = Inches(0.14)

    top = int(y)
    for i, row in enumerate(rows):
        pt_size = head_size or size if i == 0 else size
        lines = 1
        for j, val in enumerate(row):
            chars = max(int(((col_w[j] - 2 * pad) / 12700) / (0.52 * pt_size)), 8)
            lines = max(lines, sum(_wrap(ln, chars) for ln in str(plain(val)).split("\n")))
        h = max(int(row_h), int(Pt(lines * pt_size * 1.25)) + 2 * int(pad))
        for j, val in enumerate(row):
            fill = (head_fill if i == 0 else AMBER if i in emphasis
                    else SAND if i % 2 else CREAM)
            cell = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, col_x[j], top, col_w[j], h)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            cell.line.fill.background()
            cell.shadow.inherit = False
            tf = cell.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = pad
            tf.margin_top = tf.margin_bottom = Inches(0.04)
            text, url = val if isinstance(val, tuple) else (val, None)
            parts = str(text).split("\n")
            for k, part in enumerate(parts):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.RIGHT if right[j] else PP_ALIGN.LEFT
                p.line_spacing = 1.22
                run_ = p.add_run()
                run_.text = part
                run_.font.name = BODY
                run_.font.size = Pt(pt_size)
                run_.font.bold = i == 0 or i in emphasis
                # The last line of a linked cell carries the hyperlink, and shows
                # the URL as text so the citation survives being printed.
                if url and k == len(parts) - 1:
                    run_.hyperlink.address = url
                    run_.font.underline = True
                    run_.font.color.rgb = AMBER_DK
                else:
                    run_.font.color.rgb = INK
        top += h
    return top


def label_pills(slide, texts, size=20):
    """Write over the template's pills.

    Their text lives inside a scaled group whose child transform puts the
    baseline above the pill once the string changes. Rather than fight the
    transform, blank the originals and lay centred boxes over the pill
    rectangles, which are plain top-level groups on the same row.
    """
    # Bounded on both sides: the decorative edge tabs share this row.
    pills = sorted((sh for sh in slide.shapes
                    if sh.shape_type == 6 and Inches(7.2) < sh.top < Inches(8.4)
                    and Inches(2) < sh.left and sh.left + sh.width < Inches(18)
                    and Inches(2) < sh.width < Inches(6)), key=lambda s_: s_.left)
    for sh in walk(slide.shapes):
        if sh.has_text_frame and sh.text_frame.text.strip() in (
                "JUNE 5TH", "2:00 PM", "www.reallygreatsite.com"):
            for para in sh.text_frame.paragraphs:
                for r in para.runs:
                    r.text = ""
    for pill, txt in zip(pills, texts):
        tbox(slide, pill.left, pill.top + Inches(0.17), pill.width, Inches(0.5),
             txt, size, True, INK, BODY, PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return pills


# ---------- the deck ----------
def build():
    prs = Presentation(str(TEMPLATE))
    prs.slide_width, prs.slide_height = W, H
    t = list(prs.slides)
    keep = len(t)

    # 1 — title
    s = clone(prs, t[0])
    retext(find(s, "CREATIVE NETWORKING"), "VERITASXR")
    # 30pt, not the template's 42: the longer subtitle wraps onto the pill row.
    retext(find(s, "CONNECT. SHARE"), "A DUAL-PATHWAY CNN FOR CHEST RADIOGRAPHS", size=30)
    label_pills(s, ["RSNA · 26,684", "3 SEEDS", "4.22M PARAMETERS"], 18)

    # 2 — the problem and the three questions
    s = clone(prs, t[2])
    retext(find(s, "OUR VISION"), "A CONFIDENT WRONG ANSWER IS WORSE THAN NONE", size=36)
    body = pick(s, "Lorem", left_lt=Inches(9))[0]
    retext(body,
           "Pneumonia is a leading cause of death in children and the elderly, and the "
           "chest radiograph is the first-line investigation. Reading one reliably needs "
           "a radiologist, and demand far exceeds supply. Most deployed models return a "
           "label and a probability — and nothing that says this is a case I do not "
           "understand.")
    qs = ["Does a purpose-built architecture beat ImageNet transfer learning on 18,677 images?",
          "Does every component of it earn the parameters it costs?",
          "Does disagreement between the two pathways work as an uncertainty signal?"]
    for sh, q in zip(pick(s, "Lorem", left_gt=Inches(9)), qs):
        retext(sh, q, size=15)

    # 3 — literature survey
    s = blank(prs)
    title(s, "Literature survey",
          "Eleven studies, 2022–2026. Each metric is that paper's own headline figure on "
          "its own dataset — not comparable across rows.")
    lit = [
        ["Paper", "Year", "Authors", "Algorithm", "Reported performance", "Reference"],
        ["Expert-level detection of pathologies from unannotated chest X-rays via "
         "self-supervised learning (CheXzero)", "2022",
         "Tiu, Talius, Patel, Langlotz, Ng, Rajpurkar",
         "Contrastive image–text pretraining; ViT-B/32, zero-shot",
         "Mean AUC 0.889 on CheXpert; matches 3 radiologists",
         ("Nat. Biomed. Eng. 6:1399\ndoi.org/10.1038/s41551-022-00936-9",
          "https://doi.org/10.1038/s41551-022-00936-9")],
        ["A deep CNN for pneumonia detection in X-ray images with attention ensemble",
         "2024", "An, Chen, Shao",
         "EfficientNetB0 + DenseNet121 ensemble with self-attention fusion",
         "Acc 95.19% · AUC 0.9564 · F1 96.06%",
         ("Diagnostics 14(4):390\ndoi.org/10.3390/diagnostics14040390",
          "https://doi.org/10.3390/diagnostics14040390")],
        ["Pneumonia detection based on RSNA dataset and anchor-free deep learning detector",
         "2024", "Wu, Zhang, Wang, Ding, Cao, Liu et al.",
         "Anchor-free detector; ResNet-50 + 5-level FPN, focal loss",
         "AP 51.5 · AR10 96.6 — the same RSNA data this project uses",
         ("Sci. Rep. 14:1929\ndoi.org/10.1038/s41598-024-52156-7",
          "https://doi.org/10.1038/s41598-024-52156-7")],
        ["Enhancing pediatric pneumonia diagnosis through masked autoencoders", "2024",
         "Yoon, Kang",
         "Masked-autoencoder pretraining on adult CXR, fine-tuned on paediatric",
         "Acc 95.89% · AUC 0.996 binary; 3-class acc 93.86%",
         ("Sci. Rep. (2024)\ndoi.org/10.1038/s41598-024-56819-3",
          "https://doi.org/10.1038/s41598-024-56819-3")],
        ["Binary classification of pneumonia using a modified CLAHE algorithm", "2025",
         "Buriboev, Abduvaitov, Jeon",
         "Adaptive CLAHE preprocessing + 3-block CNN",
         "Acc 98.7% · AUC 0.992 · F1 97.9% (single dataset)",
         ("Sensors 25(13):3976\ndoi.org/10.3390/s25133976",
          "https://doi.org/10.3390/s25133976")],
        ["PELM: a deep learning model for early detection of pneumonia", "2025",
         "Yanar, Hardalaç, Ayturan",
         "Ensemble of CNN backbones across NIH, CheXpert, PadChest, Kaggle",
         "Acc 96% · AUC 0.91 · F1 95% (44,836 images)",
         ("Appl. Sci. 15(12):6487\ndoi.org/10.3390/app15126487",
          "https://doi.org/10.3390/app15126487")],
        ["RepViT-CXR: channel replication for vision transformers in CXR classification",
         "2025", "Ahmed",
         "Grey-to-3-channel replication to reuse an ImageNet-pretrained ViT",
         "Acc 99.0% · AUC 0.990 (paediatric); 99.9% on TB-CXR",
         ("arXiv:2509.08234\narxiv.org/abs/2509.08234", "https://arxiv.org/abs/2509.08234")],
        ["Benchmarking uncertainty and its disentanglement in multi-label CXR classification",
         "2025", "Baur, Samek, Ma",
         "13 uncertainty-quantification methods on ResNet and ViT (MIMIC-CXR-JPG)",
         "No single winner: strengths are method- and architecture-specific",
         ("UNSURE 2025 (MICCAI)\narxiv.org/abs/2508.04457", "https://arxiv.org/abs/2508.04457")],
        ["Diagnostic accuracy of AI in chest radiography for pneumonia and lung cancer",
         "2025", "Essa", "Meta-analysis pooling 15 studies (2017–2025)",
         "Pooled sensitivity 88% · specificity 90% · SROC AUC ≈ 0.95",
         ("Eur. J. Radiol. Open 15:100701\ndoi.org/10.1016/j.ejro.2025.100701",
          "https://doi.org/10.1016/j.ejro.2025.100701")],
        ["Pediatric pneumonia detection: transfer learning vs custom CNNs", "2025",
         "Roy Choudhury",
         "ResNet50 / DenseNet121 / EfficientNet-B0, frozen vs fine-tuned, vs CNNs from scratch",
         "Fine-tuned ResNet50: acc 99.43% · AUC 0.9993; fine-tuning beats frozen by 5.5 pp",
         ("arXiv:2601.00837\narxiv.org/abs/2601.00837", "https://arxiv.org/abs/2601.00837")],
        ["Pneumonia detection from enhanced chest X-rays based on a Double SGAN model",
         "2026", "Xu, Zhang",
         "Double SGAN augmentation + ResNet18 with spatial attention",
         "Acc 95.83% · AUC 0.993 vs 0.892 on real data alone",
         ("Sci. Rep. (2026)\ndoi.org/10.1038/s41598-026-39785-w",
          "https://doi.org/10.1038/s41598-026-39785-w")],
    ]
    table(s, M, Inches(2.5), W - 2 * M, lit, [22, 4, 13, 22, 22, 17], Inches(0.5), 10.5,
          head_size=11.5)

    # 4 — the gap and the hypothesis
    s = clone(prs, t[1])
    retext(find(s, "WHAT IS CREATIVE"), "THE GAP, AND THE HYPOTHESIS")
    gap_body = find(s, "Lorem ipsum")
    gap_body.height = Inches(3.4)
    retext(gap_body,
           "Three patterns run through those eleven papers. Reported accuracy is near "
           "saturation — 95–99% — but almost always on one small, single-source, often "
           "paediatric dataset. Pretraining, not architecture, moves the numbers: "
           "fine-tuning beats a frozen backbone by 5.5 points, and self-supervised "
           "pretraining reaches radiologist level without labels. Uncertainty is "
           "unsettled — a 2025 benchmark of 13 methods finds no single winner, and most "
           "studies report none.\n\n"
           "So: can a compact dual-pathway CNN earn useful accuracy and a free uncertainty "
           "signal in one forward pass — and survive the replication these papers rarely "
           "report?\n\n"
           "Measured over three seeds, the answer splits. The uncertainty works — AUROC "
           "0.7747, no ensemble. The accuracy does not follow.",
           size=16, align=PP_ALIGN.LEFT)

    # 5 — dataset and protocol
    s = clone(prs, t[4])
    retext(find(s, "CHALLENGES AND"), "DATASET AND PROTOCOL")
    heads = ["26,684", "70 / 15 / 15", "3 SEEDS"]
    bodies = [
        "RSNA Pneumonia Detection Challenge. Adult radiographs from multiple sites, "
        "boxed by one to three board-certified radiologists. 22.5% positive, and the "
        "hard middle class — abnormal but not pneumonia — is 44% of the data.",
        "Split by patient into 18,677 / 4,001 / 4,006, stratified on the three-class "
        "label so the middle class stays proportional. A leakage test asserts no patient "
        "crosses splits, and fails when a leak is injected.",
        "Every model reads one config: same loss, optimiser, learning rate, schedule, "
        "augmentation and early stopping. Three seeds each, so no claim rests on one run.",
    ]
    num_boxes = [sh for sh in walk(s.shapes)
                 if sh.has_text_frame and sh.text_frame.text.strip() in ("01", "02", "03")]
    txt_boxes = [sh for sh in walk(s.shapes)
                 if sh.has_text_frame and sh.text_frame.text.startswith("Lorem")]
    for sh, v in zip(sorted(num_boxes, key=lambda s_: s_.left), heads):
        retext(sh, v, size=30)
    for sh, v in zip(sorted(txt_boxes, key=lambda s_: s_.left), bodies):
        retext(sh, v, size=14)

    # 6 — architecture overview
    s = blank(prs)
    title(s, "Architecture", "One stem, two kernel scales, three outputs — 4,215,352 parameters")
    picture(s, "1-architecture-overview-canva.png", Inches(2.5), Inches(6.4))
    tbox(s, M, Inches(9.4), W - 2 * M, Inches(1.1),
         "Two branches read a shared stem and differ only in kernel size: 3x3 sees local "
         "opacity texture, 7x7 sees whole-lung structure. A gate predicted from the image "
         "blends them. The next three slides take one component at a time.", 16, False, MUTE)

    # 7 — stem and bottleneck
    s = blank(prs)
    title(s, "Inside a pathway", "A wide stem, then three bottleneck stages that halve the map each time")
    picture(s, "2-bottleneck-block-canva.png", Inches(2.45), Inches(4.3),
            x=M, max_w=W - 2 * M)
    y = Inches(7.1)
    for i, (h, b) in enumerate([
            ("Stem — 44,864 params",
             "Conv 11x11 stride 2, then 3x3, then maxpool, to 64 x 56 x 56. Wide on purpose: "
             "radiographic signal is diffuse texture, not the sharp edges a 3x3 first layer is tuned for."),
            ("Bottleneck — the unit both pathways repeat",
             "1x1 reduce, k x k at half width, 1x1 expand, plus a projection shortcut. "
             "Three stages per pathway take 56x56 down to 7x7."),
            ("Why it matters — 17.0M to 4.2M params",
             "The first version stacked three full-width k x k convolutions. At k=7 and 256 "
             "channels that one choice held 59% of the model and ran 4-5x slower than bigger baselines.")]):
        x = M + i * Inches(6.05)
        card(s, x, y, Inches(5.6), Inches(2.6), SAND if i < 2 else AMBER)
        tbox(s, x + Inches(0.4), y + Inches(0.32), Inches(4.8), Inches(0.5), h, 16, True, INK)
        tbox(s, x + Inches(0.4), y + Inches(1.0), Inches(4.8), Inches(1.4), b, 13.5, False, INK)

    # 8 — SE and the merge gate
    s = blank(prs)
    title(s, "Attention, and how the pathways combine",
          "Both were trained with and without, across three seeds. Here is what each contributes.")
    picture(s, "3-squeeze-excitation-canva.png", Inches(2.45), Inches(3.5), x=M,
            max_w=Inches(8.3))
    picture(s, "4-merge-gate-canva.png", Inches(2.45), Inches(3.5),
            x=M + Inches(8.8), max_w=Inches(8.3))
    for i, (h, b) in enumerate([
            ("Squeeze-and-Excitation",
             "Pool each of 384 channels to one number, learn a gain for each through "
             "384 -> 24 -> 384 with a sigmoid, multiply the map by it. Removing it costs "
             "accuracy on two seeds of three — real, but not separable at this sample size."),
            ("The merge gate — 49,346 params",
             "Concatenate both pooled vectors to 768, predict two weights that sum to 1, blend. "
             "Of the four merges trained it is the only one never ranked worst, and it replaced "
             "a global scalar that had converged to 0.510 / 0.490 — an average with extra steps.")]):
        x = M + i * Inches(8.8)
        card(s, x, Inches(6.5), Inches(8.3), Inches(2.9), SAND)
        tbox(s, x + Inches(0.45), Inches(6.85), Inches(7.4), Inches(0.5), h, 18, True, INK)
        tbox(s, x + Inches(0.45), Inches(7.6), Inches(7.4), Inches(1.6), b, 14.5, False, INK)

    # 9 — the uncertainty output
    s = clone(prs, t[5])
    retext(find(s, "BENEFITS OF"), "THE UNCERTAINTY OUTPUT")
    retext(find(s, "Lorem ipsum"),
           "The architecture is named for pathway disagreement. Two independent heads, one "
           "per pathway, make that disagreement measurable at all — the previous version "
           "merged before the head, which makes it unobservable by construction.\n\n"
           "Measured against the model's own errors, it scores chance. Predictive entropy "
           "is what ships instead, and the negative result is reported rather than dropped.",
           size=16)
    sig = [("0.5334", "Pathway disagreement — Jensen-Shannon divergence between the two heads. "
                      "This is chance. Both heads train on the same labels with the same loss, "
                      "so they converge toward each other."),
           ("0.7747", "Predictive entropy of the merged verdict. The signal that ships, and what "
                      "the demo's triage bands are read from."),
           ("0.7675", "Entropy and disagreement combined — worse than entropy alone. The rejected "
                      "signal adds nothing.")]
    nums = [sh for sh in walk(s.shapes)
            if sh.has_text_frame and sh.text_frame.text.strip() in ("01", "02", "03")]
    txts = [sh for sh in walk(s.shapes)
            if sh.has_text_frame and sh.text_frame.text.startswith("Lorem")]
    for sh, (n, _) in zip(sorted(nums, key=lambda s_: s_.top), sig):
        retext(sh, n, size=34)
    for sh, (_, b) in zip(sorted(txts, key=lambda s_: s_.top), sig):
        retext(sh, b, size=13.5)

    # 10 — results
    s = blank(prs)
    title(s, "How it compares, and why",
          "4,006 test radiographs · 22.5% positive · AUPRC and AUROC are mean ± SD over three seeds")
    res = [["Model", "Params", "AUPRC", "AUROC", "it/s", "Why it lands here"],
           ["EfficientNet-B0", "4.01M", "0.6993 ± 0.0033", "0.8781", "3.66",
            "ImageNet features plus compound scaling. Best accuracy at the smallest "
            "pretrained size, and the best calibrated (ECE 0.0343)."],
           ["DenseNet121", "6.96M", "0.6872 ± 0.0034", "0.8728", "2.11",
            "The CheXNet backbone; dense connections reuse features. Slowest of the "
            "four — concatenation makes it memory-bandwidth bound, not parameter bound."],
           ["ResNet50", "23.51M", "0.6747 ± 0.0072", "0.8675", "5.04",
            "5.6x EfficientNet's parameters and still third, with the highest seed "
            "variance of any model. On 18,677 images more capacity buys instability."],
           ["VeritasXR", "4.22M", "0.6410 ± 0.0039", "0.8531", "7.94",
            "The only model here with no pretraining — that single difference is worth "
            "5–9% relative AUPRC. Fastest of the five, and it ships a measured "
            "uncertainty (entropy, AUROC 0.7747 against its own errors) in the same "
            "forward pass: no ensemble, no repeated passes."],
           ["HOG + linear SVM", "6,085 feat", "0.3823 ± 0.0010", "0.7283", "—",
            "Hand-crafted gradient histograms cannot represent diffuse opacity texture. "
            "The control that shows the deep models are earning their place."]]
    table(s, M, Inches(2.75), W - 2 * M, res, [13, 8, 13, 8, 6, 52], Inches(1.08), 14.5,
          emphasis={4})
    tbox(s, M, Inches(9.6), W - 2 * M, Inches(1.3), [
        (True, "The ordering is significant, not noise: a paired bootstrap run per seed, "
               "Holm-corrected, separates every pair on every seed. The gap to the "
               "pretrained models is 0.034–0.058 AUPRC."),
        (True, "The cause is pretraining, not architecture or capacity — and not the shared "
               "learning rate either. Retrained at 1/3x and 3x, the model most handicapped "
               "by the shared rate was DenseNet121, a baseline; tuning each model widens the gap."),
    ], 15, False, INK)

    # 11 — the results as pictures
    s = blank(prs)
    title(s, "The same result, in three pictures",
          "Every curve and point is read from the run artefacts, not retyped")
    for i, (fname, caption) in enumerate([
            ("1-pr-curves.png",
             "VeritasXR tracks the pretrained models closely until recall ≈ 0.3, then "
             "falls behind — and sits far above the classical control throughout."),
            ("2-auprc-vs-params.png",
             "ResNet50 carries 5.6x EfficientNet-B0's parameters and lands below it. "
             "Size is not what separates these models."),
            ("3-uncertainty-auroc.png",
             "The signal the architecture is named for sits on the chance line. "
             "Predictive entropy, free in the same pass, clears it by 0.27.")]):
        x = M + i * Inches(5.95)
        picture(s, fname, Inches(2.9), Inches(4.6), x=x, max_w=Inches(5.6),
                folder=CHARTS)
        tbox(s, x, Inches(7.3), Inches(5.6), Inches(1.4), caption, 14, False, INK)
    card(s, M, Inches(9.2), W - 2 * M, Inches(1.2), AMBER)
    tbox(s, M + Inches(0.5), Inches(9.52), W - 2 * M - Inches(1.0), Inches(0.8),
         "One story, three ways: the gap to transfer learning is real and holds on every "
         "seed, it is not explained by model size, and the uncertainty that works is the "
         "one the architecture did not set out to use.", 16, True, INK)

    # 11 — ablation
    s = blank(prs)
    title(s, "Does each component earn its parameters?",
          "Seven variants, three seeds each. One result survives replication.")
    abl = [["Variant", "Params", "AUPRC", "Δ vs full"],
           ["concat_merge", "4,264,310", "0.6422", "+0.0011"],
           ["full", "4,215,352", "0.6410", "baseline"],
           ["fixed_merge", "4,166,006", "0.6394", "−0.0016"],
           ["scalar_merge", "4,166,008", "0.6360", "−0.0051"],
           ["no_se", "4,177,672", "0.6348", "−0.0062"],
           ["global_only", "3,302,108", "0.6337", "−0.0073"],
           ["local_only (3x3 branch alone)", "1,008,348", "0.6205", "−0.0205"]]
    table(s, M, Inches(2.7), Inches(8.6), abl, [40, 22, 19, 19], Inches(0.62), 15,
          emphasis={7})
    # The chart carries what the "across the seeds" column used to say, per seed.
    picture(s, "4-ablation-per-seed.png", Inches(2.9), Inches(4.2),
            x=M + Inches(9.1), max_w=Inches(8.3), folder=CHARTS)
    tbox(s, M, Inches(8.0), W - 2 * M, Inches(1.8), [
        (True, "Only the 7x7 global pathway survives. Removing it costs accuracy on every seed — "
               "the one component in this model that replicates."),
        (True, "Everything else changes sign with the seed. Squeeze-excitation, the merge gate and "
               "the merge strategy each help on one seed and hurt on another."),
        (True, "global_only separates on seed 0 and then reverses on seed 1. Any of these would have "
               "looked meaningful from a single run."),
    ], 15, False, INK)

    # 12 — four reversals
    s = clone(prs, t[8])
    retext(find(s, "TOOLS TO BOOST"), "WHAT THE THIRD SEED CORRECTED")
    rev = [("A 12x calibration win for VeritasXR",
            "Second, not first: ECE 0.0440 against EfficientNet-B0's 0.0343. One temperature "
            "fitted on validation puts it last."),
           ("The cheap 3x3 branch beats the expensive 7x7",
            "Reverses across seeds. Seed 0 caught global_only at its worst and local_only near its best."),
           ("Pathway disagreement measures uncertainty",
            "AUROC 0.5334 against the model's own errors, over the full test set on three seeds. Chance.")]
    nums = [sh for sh in walk(s.shapes)
            if sh.has_text_frame and sh.text_frame.text.strip() in ("01", "02", "03")]
    txts = [sh for sh in walk(s.shapes)
            if sh.has_text_frame and sh.text_frame.text.startswith("Lorem")]
    for sh, (h, b) in zip(sorted(txts, key=lambda s_: s_.top), rev):
        retext(sh, f"{h}  —  {b}", size=14)
    for sh, n in zip(sorted(nums, key=lambda s_: s_.top), ["01", "02", "03"]):
        retext(sh, n)
    tbox(s, Inches(1.8), Inches(10.05), Inches(16.4), Inches(0.9),
         "A fourth: p < 0.0001 across all paired tests became p = 0.014 at its weakest once the "
         "test ran per seed. Each of these was a clean, attractive result from a single run. "
         "Catching all four is what three seeds and a per-seed paired test are for.", 15, False, INK)

    # 13 — the input gate
    s = blank(prs)
    title(s, "Refusing what the model cannot judge",
          "The classifier has no way to abstain, so the refusal happens on the pixels before it")
    picture(s, "5-input-gate-canva.png", Inches(2.6), Inches(3.6), x=M, max_w=Inches(11.0))
    x2 = M + Inches(11.6)
    card(s, x2, Inches(2.6), Inches(5.7), Inches(2.5), AMBER)
    tbox(s, x2 + Inches(0.4), Inches(2.9), Inches(4.9), Inches(0.4),
         "Measured on the deployed model", 16, True, INK)
    tbox(s, x2 + Inches(0.4), Inches(3.45), Inches(4.9), Inches(1.5), [
        (True, "4 of 7 photographs: highest-confidence band"),
        (True, "A flat white field: 68% pneumonia"),
        (True, "A radiograph of a foot: P = 0.911"),
    ], 13.5, False, INK)
    gate = [["What it is given", "Turned away"],
            ["Radiographs of the wrong body part (22)", "95%  (21/22)"],
            ["Ordinary photographs (11)", "100%"],
            ["Rotated, inverted, noise, flat field", "99.8–100%"],
            ["Genuine chest radiographs — the cost", "3.8%"]]
    table(s, x2, Inches(5.5), Inches(5.7), gate, [67, 33], Inches(0.6), 13.5, emphasis={4})
    tbox(s, M, Inches(9.5), Inches(17.4), Inches(1.0),
         "A 16x16 layout template fitted on the training split: three cuts — colour, contrast, "
         "and distance and correlation against the mean chest — each at the 2nd validation "
         "percentile. The one that still gets through is a horse hoof. Passing is not evidence "
         "an image is a chest radiograph, only that nothing obvious says otherwise.", 14.5, False, MUTE)

    # 14 — conclusion
    s = clone(prs, t[10])
    # The template's display box is 4.9in tall and centres its text, so anything
    # placed above 6.4in collides with the second line of the headline.
    retext(find(s, "THANK YOU"), "SMALL, FAST, MEASURED", size=112)
    tbox(s, Inches(3.8), Inches(6.5), Inches(12.5), Inches(0.8),
         "A 4.22M CNN from scratch — and a study that checked its own claims.",
         26, False, INK, HEAD, PP_ALIGN.CENTER)
    label_pills(s, ["7.94 it/s — fastest", "RUNS IN-BROWSER", "NEXT LEVER: PRETRAINING"], 17)

    # drop the template's own slides
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides)[:keep]:
        rid = sld.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        prs.part.drop_rel(rid)
        xml_slides.remove(sld)

    theme_link_colour(prs)
    prs.save(str(OUT))
    print(f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides -> {OUT}")


if __name__ == "__main__":
    build()
