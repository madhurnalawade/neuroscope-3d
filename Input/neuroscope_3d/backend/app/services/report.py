"""Per-study report (HTML page and PDF): volumes, agreement, evaluation, provenance, limitations, disclaimer."""

import base64
import datetime
import html
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import app.core.paths  # noqa: E402,F401
from app.core.paths import REPO_ROOT  # noqa: E402
from app.services import viewer_data as vd  # noqa: E402
from app.services.study_store import ValidatedStudy  # noqa: E402

from brainseg.constants import DISCLAIMER, LIMITATIONS, MODEL_INFO, PREPROCESSING  # noqa: E402

EVALUATION_FILE = REPO_ROOT / "evaluation" / "metrics.json"
SPLIT_FILE = REPO_ROOT / "evaluation" / "split.json"
REPORT_METHOD = ("Fixed template (Input/neuroscope_3d/backend/app/services/report.py) filled only with values "
                 "computed or recorded by this platform; figure drawn with matplotlib from the uploaded FLAIR and "
                 "the label maps. No generative AI or external service writes, edits or illustrates this report.")
PATIENT_CAVEAT = ("The Decathlon release has no patient identifiers; each case ID is assumed to be one patient, "
                  "which cannot be confirmed from the data.")
LABEL_CMAP = ListedColormap(["none", "#d6452c", "#78a870", "none", "#e8b030"])
REGION_ORDER = ["NCR", "ED", "ET", "TC", "WT"]
REGION_DESC = {
    "NCR": "label 1", "ED": "label 2", "ET": "label 4", "TC": "labels 1+4", "WT": "labels 1+2+4",
}


def load_evaluation() -> Optional[Dict[str, Any]]:
    if EVALUATION_FILE.exists():
        return json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    return None


def load_split() -> Optional[Dict[str, Any]]:
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return None


def evaluation_parts(ev, split):
    """(heading, paragraphs, table rows) for the dataset-level evaluation, shared by HTML and PDF."""
    if not ev:
        return ("Part B — dataset-level evaluation", ["Not computed: evaluation/metrics.json is missing. Run "
                "scripts/evaluate.py on at least 10 labelled cases to fill this section."], None)
    s = ev["summary"]
    n = ev["n_cases"]
    trained = (split or {}).get("training_or_tuning_cases_used_by_us")
    trained_txt = ("none (the pretrained model was used as-is; no training or tuning by us)" if trained == []
                   else ", ".join(trained) if trained else "not provided")
    paras = [
        "These numbers describe the model on other cases, not the case above.",
        f"Data: {ev['dataset']}. Split by case ID (patient-level): {n} evaluation cases: {', '.join(ev['case_ids'])}. "
        f"Cases we trained or tuned on: {trained_txt}. {PATIENT_CAVEAT}",
        f"Evaluation run {ev.get('generated_utc', 'not provided')}, code commit "
        f"{ev.get('code_commit', 'not provided')} (as recorded in evaluation/metrics.json).",
    ]
    rows = [["Region", "Dice mean ± SD (n)", "Sensitivity mean ± SD (n)", "HD95 median (mm)"]]
    for r in ("WT", "TC", "ET"):
        n_sens = n - int(s[r].get("sensitivity_undefined_cases", 0) or 0)
        rows.append([r, f"{s[r]['dice_mean']:.3f} ± {s[r]['dice_sd']:.3f} ({n})",
                     f"{_fmt(s[r]['sensitivity_mean'])} ± {_fmt(s[r]['sensitivity_sd'])} ({n_sens})",
                     _fmt(s[r]["hd95_median_mm"], 1)])
    undefined = [r for r in ("WT", "TC", "ET") if s[r].get("sensitivity_undefined_cases")]
    if undefined:
        paras.append("Sensitivity is undefined when the expert marked no voxels of a region; such cases are left "
                     "out of that region's mean (" + ", ".join(f"{r}: {s[r]['sensitivity_undefined_cases']}"
                                                         for r in undefined) + ").")
    return (f"Part B — dataset-level evaluation on {n} labelled cases we did not train or tune on", paras, rows)


# ---------------------------------------------------------------- figure

def _views(arr, x, y, z):
    """Radiological display: axial/coronal show patient Right on screen-left; sagittal shows Anterior on left."""
    return {
        "Axial": (arr[::-1, ::-1, z].T, ("R", "L", "A", "P")),
        "Coronal": (arr[::-1, y, ::-1].T, ("R", "L", "S", "I")),
        "Sagittal": (arr[x, ::-1, ::-1].T, ("A", "P", "S", "I")),
    }


def snapshot_png(study: ValidatedStudy, with_expert: bool = True) -> bytes:
    ref, ornt, canon_affine = vd._canonical(study)
    sp = [float(np.linalg.norm(canon_affine[:3, i])) for i in range(3)]
    flair = vd._reorient(np.asarray(nib.load(study.modalities["FLAIR"]["path"], mmap=False).dataobj, dtype=np.float32), ornt)
    rows = [("Model prediction", vd._labels(study, "pred"))]
    if with_expert and vd._labels(study, "expert") is not None:
        rows.append(("Expert mask (reference)", vd._labels(study, "expert")))
    rows = [(t, vd._reorient(l, ornt)) for t, l in rows if l is not None]
    lab0 = rows[0][1] if rows else None
    if lab0 is not None and lab0.any():
        x, y, z = (int(round(c)) for c in np.argwhere(lab0 > 0).mean(axis=0))
    else:
        x, y, z = (s // 2 for s in flair.shape)
    nz = flair[flair != 0]
    vmax = float(np.percentile(nz, 99.5)) if nz.size else 1.0
    aspects = {"Axial": sp[1] / sp[0], "Coronal": sp[2] / sp[0], "Sagittal": sp[2] / sp[1]}

    fig, axes = plt.subplots(max(len(rows), 1), 3, figsize=(10, 3.5 * max(len(rows), 1)), squeeze=False)
    for r, (title, lab) in enumerate(rows or [("FLAIR", None)]):
        img_views = _views(flair, x, y, z)
        lab_views = _views(lab, x, y, z) if lab is not None else {}
        for c, name in enumerate(("Axial", "Coronal", "Sagittal")):
            ax = axes[r][c]
            im, (left, right, top, bottom) = img_views[name]
            ax.imshow(im, cmap="gray", vmin=0, vmax=vmax, aspect=aspects[name], interpolation="nearest")
            if name in lab_views:
                ax.imshow(lab_views[name][0], cmap=LABEL_CMAP, vmin=0, vmax=4, alpha=0.45,
                          aspect=aspects[name], interpolation="nearest")
            for txt, pos in ((left, (0.02, 0.5)), (right, (0.98, 0.5)), (top, (0.5, 0.97)), (bottom, (0.5, 0.03))):
                ax.text(*pos, txt, transform=ax.transAxes, color="#f2ede3", fontsize=11, fontweight="bold",
                        ha="center", va="center")
            ax.set_title(f"{title} — {name}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"FLAIR with overlay: red NCR (1), green ED (2), yellow ET (4). Canonical voxel ({x}, {y}, {z}). "
                 "Suspected regions predicted by the model.", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- data

def report_data(study: ValidatedStudy) -> Dict[str, Any]:
    m = vd.meta(study)
    rec = study.metadata.get("prediction") or {}
    return {
        "study_id": study.study_id,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "meta": m,
        "inputs": rec.get("provenance", {}).get("inputs", {}),
        "evaluation": load_evaluation(),
        "split": load_split(),
        "limitations": LIMITATIONS,
        "model": MODEL_INFO,
        "preprocessing": PREPROCESSING,
        "disclaimer": DISCLAIMER,
    }


def _backup_text(m) -> str:
    b = m.get("backup")
    return (f"PRECOMPUTED BACKUP CASE — {b['case']}, generated {b['generated_utc']} by this pipeline. "
            "This is NOT a live run on an uploaded case.") if b else ""


def _backup_html(m) -> str:
    t = _backup_text(m)
    return (f"<p style='background:#c8402a;color:#fff6ec;padding:9px 14px;font-weight:600'>"
            f"{html.escape(t)}</p>") if t else ""


def _source_text(m) -> str:
    s = m.get("source_4d")
    if not s:
        return ""
    order = ", ".join(f"ch{i} = {mod}" for i, mod in enumerate(s["channel_order"]))
    return (f"Uploaded as one 4-channel file {s['file']} (shape {s['shape']}, SHA-256 {s['sha256']}); "
            f"split server-side as {order}. The per-sequence hashes above are of the split channels.")


def _trust_rows(m):
    """(paragraph, table rows) describing TTA uncertainty and, if computed, modality ablation."""
    out = []
    u = m.get("uncertainty")
    if u:
        out.append(("Test-time augmentation: the model was re-run on the scan mirrored along each axis "
                    f"({u['passes']} passes; the label map is from the un-augmented pass). "
                    f"{u['wt_uncertain_fraction'] * 100:.1f}% of WT voxels are disputed between passes; WT is "
                    f"{u['wt_unanimous_mL']:.2f} mL where all passes agree and {u['wt_any_mL']:.2f} mL where any pass "
                    "predicts it.",
                    [["Region", "Volume range across passes (mL)", "Lowest Dice vs main pass"]]
                    + [[r, f"{x['volume_mL_min']:.2f} – {x['volume_mL_max']:.2f}", f"{min(x['dice_vs_main_per_pass']):.3f}"]
                       for r, x in u["regions"].items()]))
    a = m.get("ablation")
    if a:
        out.append((f"{a['label']}. {a['method']}",
                    [["Input change", "WT Dice vs full", "TC Dice vs full", "ET Dice vs full"]]
                    + [[f"without {s.upper()}"] + [f"{v[r]['dice_vs_full']:.2f}" for r in ("WT", "TC", "ET")]
                       for s, v in a["removed_sequence"].items()]))
    return out


def _trust_html(m) -> str:
    blocks = _trust_rows(m)
    if not blocks:
        return ""
    html_out = "<h2>How much to trust it (this case)</h2>"
    for para, rows in blocks:
        html_out += (f"<p>{html.escape(para)}</p><table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in rows[0])
                     + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
                                         for r in rows[1:]) + "</table>")
    return html_out


def _fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


# ---------------------------------------------------------------- expert comparison (this case only)

EXPERT_SCOPE = ("Compares the model with the reference supplied for this case only. It is not the benchmark: "
                "dataset-level scores are reported separately, on other cases.")
MASK_METRIC_NOTE = ("Dice = 2 x overlap / (model + expert voxels); sensitivity = overlap / expert voxels; "
                    "precision = overlap / model voxels; IoU = overlap / union; HD95 = 95th-percentile symmetric "
                    "surface distance in mm. Regions use BraTS codes WT = 1+2+4, TC = 1+4, ET = 4. "
                    "A dash means undefined (an empty region).")
READING_NOTE = ("Entered volumes are compared by volume only: difference = model - entered; % = difference / entered. "
                "Dice, sensitivity and the other overlap scores need a voxel-wise expert mask and are not computed "
                "from volumes. Equal volumes do not mean the same voxels.")


def _pct(v):
    return "—" if v is None else f"{v:+.1f}%"


def expert_comparison_parts(m):
    """(mask rows, mask notes, reading rows, reading meta) for the report; rows include a header. None = absent."""
    mask_rows = None
    if m.get("agreement"):
        mask_rows = [["Region", "Dice", "Sensitivity", "Precision", "IoU", "HD95 (mm)", "Model voxels", "Expert voxels"]]
        for r in ("WT", "TC", "ET"):
            v = m["agreement"][r]
            mask_rows.append([r, _fmt(v["dice"]), _fmt(v["sensitivity"]), _fmt(v.get("precision")), _fmt(v.get("iou")),
                              _fmt(v["hd95_mm"], 1), f"{v['pred_voxels']:,}", f"{v['ref_voxels']:,}"])
    read_rows = None
    rc = m.get("readings_comparison")
    if rc:
        read_rows = [["Region", "Model (mL)", "Entered (mL)", "Model - entered (mL)", "Absolute (mL)", "Difference (%)"]]
        for r, v in rc.items():
            read_rows.append([r, f"{v['model_mL']:.2f}", f"{v['doctor_mL']:.2f}", f"{v['difference_mL']:+.2f}",
                              f"{v['abs_difference_mL']:.2f}", _pct(v["percent_difference"])])
    return mask_rows, m.get("expert_notes") or [], read_rows, m.get("doctor_readings")


def _expert_html(m) -> str:
    mask_rows, notes, read_rows, readings = expert_comparison_parts(m)
    if not mask_rows and not read_rows:
        return ""
    e = html.escape

    def tab(rows):
        return ("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in rows[0]) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
    out = f"<h2>Expert comparison (this case only)</h2><p class=muted>{e(EXPERT_SCOPE)}</p>"
    if mask_rows:
        fn = (m.get("expert_mask") or {}).get("filename")
        out += f"<h3>Agreement with the expert mask{(' · ' + e(fn)) if fn else ''}</h3>" + tab(mask_rows)
        out += "".join(f"<p class=warn>{e(n)}</p>" for n in notes) + f"<p class=muted>{e(MASK_METRIC_NOTE)}</p>"
    if read_rows:
        out += (f"<h3>Doctor's volume readings · entered {e(readings['entered_utc'])} UTC</h3>" + tab(read_rows)
                + f"<p class=muted>{e(READING_NOTE)}</p>")
    return out


# ---------------------------------------------------------------- HTML

def render_html(study: ValidatedStudy) -> str:
    d = report_data(study)
    m = d["meta"]
    e = html.escape
    img = base64.b64encode(snapshot_png(study)).decode()
    sp = m["spacing_mm"]
    vox = m["voxel_volume_mm3"]

    def vol_rows():
        out = []
        mv, ev = m["volumes"]["model"] or {}, m["volumes"]["expert"] or {}
        for r in REGION_ORDER:
            a = mv.get(r)
            b = ev.get(r)
            out.append(
                f"<tr><td>{r} <span class=muted>({REGION_DESC[r]})</span></td>"
                f"<td>{a['voxels'] if a else '—'}</td><td>{a['voxels'] * vox if a else 0:,.1f}</td>"
                f"<td><b>{a['mL']:.2f}</b></td>"
                + (f"<td>{b['voxels']}</td><td><b>{b['mL']:.2f}</b></td>" if m["has_expert"] else "")
                + "</tr>"
            )
        return "".join(out)

    agreement = _expert_html(m)

    ev = d["evaluation"]
    ev_head, ev_paras, ev_rows = evaluation_parts(ev, d["split"])
    evaluation = f"<h2>{e(ev_head)}</h2>" + "".join(f"<p>{e(p)}</p>" for p in ev_paras)
    if ev_rows:
        evaluation += ("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in ev_rows[0]) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in ev_rows[1:])
                       + "</table>")
    if ev:
        evaluation += f"<p class=warn><b>Overlap disclosure:</b> {e(ev['overlap_disclosure'])}</p>"

    warnings = m["suitability"].get("warnings", [])
    warn_html = ("<ul>" + "".join(f"<li>{e(w)}</li>" for w in warnings) + "</ul>") if warnings else "<p>None.</p>"
    prov = m["provenance"] or {}
    inputs = "".join(
        f"<tr><td>{e(k)}</td><td>{e(v['file'])}</td><td class=mono>{e(v['sha256'])}</td></tr>" for k, v in d["inputs"].items()
    )
    expert_cols = "<th>Expert voxels</th><th>Expert mL</th>" if m["has_expert"] else ""

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Voxels report</title>
<link rel="stylesheet" href="/static/base.css">
<style>
:root {{ --fg:#171614; --muted:#7a7468; --bg:#f2ede3; --line:#c9bfab; --warn:#5a3d06; --warnbg:#f3dfb4; --accent:#c8402a; }}
body {{ font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; color:var(--fg); background:var(--bg); max-width: 980px;
       margin: 0 auto; padding: 0 16px 40px; line-height: 1.5; }}
* {{ border-radius: 0 !important; }}
h1 {{ font: 400 clamp(32px, 5vw, 52px)/1.02 "Instrument Serif", Georgia, serif; margin: 22px 0 6px; letter-spacing: -.01em; }}
h2 {{ font: 400 28px/1.1 "Instrument Serif", Georgia, serif; margin: 34px 0 10px; padding-top: 10px; border-top: 1px solid var(--fg); }}
h3 {{ font: 500 11px "IBM Plex Mono", Consolas, monospace; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
td, th {{ border-bottom:1px solid var(--line); padding:6px 10px 6px 0; text-align:left; vertical-align: top; }}
th {{ font: 500 10.5px "IBM Plex Mono", Consolas, monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); border-bottom-color: var(--fg); }}
.disclaimer {{ background: var(--fg); color: var(--bg); padding: 9px 14px; font: 500 12px/1.5 "IBM Plex Mono", Consolas, monospace; letter-spacing: .03em; }}
.warn {{ background:var(--warnbg); color:var(--warn); padding:8px 12px; }}
.muted {{ color: var(--muted); }} .mono {{ font-family: "IBM Plex Mono", Consolas, monospace; font-size: .75rem; word-break: break-all; }}
img {{ max-width: 100%; border: 1px solid var(--fg); }}
.toolbar {{ display: flex; gap: 18px; font: 400 13px "IBM Plex Mono", Consolas, monospace; margin: 14px 0 0; }}
@media print {{ .toolbar {{ display:none; }} body {{ background: #fff; }} }}
</style></head><body>
<div class=disclaimer>{e(d['disclaimer'])}</div>
{_backup_html(m)}
<p class=toolbar><a href="/api/report/{e(d['study_id'])}.pdf">Download PDF</a><a href="/viewer/{e(d['study_id'])}">Open viewer</a>
<a href="/api/segmentation/prediction/{e(d['study_id'])}">Prediction NIfTI</a></p>
<h1>Voxels — estimated tumour-region volumes</h1>
<p class=muted>Study {e(d['study_id'])} · report generated {e(d['generated_utc'])}</p>
<p>Part A describes this case: the suspected tumour regions predicted by the model and their estimated volumes.
Part B describes how the model performed on other, labelled cases. The two are separate.</p>

<h2>Part A — this case: input</h2>
<table><tr><th>Sequence</th><th>File</th><th>SHA-256</th></tr>{inputs}</table>
<p>Grid {m['original_shape']} voxels, spacing {', '.join(f'{s:.3f}' for s in sp)} mm, orientation {e(m['original_orientation'])}.
Expert mask: {'supplied' if m['has_expert'] else 'not supplied'}.</p>
{f"<p class=mono>{e(_source_text(m))}</p>" if _source_text(m) else ""}
<h3>Suitability warnings</h3>{warn_html}

<h2>This case — estimated volumes of suspected tumour regions predicted by the model</h2>
<p><b>Formula:</b> V = N × (Δx × Δy × Δz) = N × ({sp[0]:.3f} × {sp[1]:.3f} × {sp[2]:.3f}) mm³ = N × {vox:.3f} mm³;
&nbsp; mL = mm³ / 1000 (1 mL = 1,000 mm³). N is the number of voxels with the region's BraTS codes.</p>
<table><tr><th>Region</th><th>Model voxels N</th><th>Model mm³</th><th>Model mL</th>{expert_cols}</tr>{vol_rows()}</table>
<p><img alt="Axial, coronal and sagittal FLAIR slices with predicted label overlay" src="data:image/png;base64,{img}"></p>
{agreement}
{_trust_html(m)}
{evaluation}

<h2>Provenance</h2>
<table>
<tr><td>Model</td><td>{e(MODEL_INFO['name'])}, version {e(MODEL_INFO['version'])}</td></tr>
<tr><td>Licence</td><td>{e(MODEL_INFO['licence'])}</td></tr>
<tr><td>Model training data</td><td>{e(MODEL_INFO['training_data'])}</td></tr>
<tr><td>Preprocessing</td><td>{e(PREPROCESSING)}</td></tr>
<tr><td>Code commit</td><td class=mono>{e(str(prov.get('code_commit', '—')))}</td></tr>
<tr><td>Inference timestamp (UTC)</td><td>{e(str(prov.get('timestamp_utc', '—')))}</td></tr>
<tr><td>Device / inference time</td><td>{e(str(prov.get('device', '—')))} / {e(str(prov.get('inference_seconds', '—')))} s</td></tr>
<tr><td>Report text and figure</td><td>{e(REPORT_METHOD)}</td></tr>
</table>

<h2>Limitations</h2><ul>{''.join(f'<li>{e(x)}</li>' for x in LIMITATIONS)}</ul>
<div class=disclaimer>{e(d['disclaimer'])}</div>
</body></html>"""


# ---------------------------------------------------------------- PDF
# Two pages: page 1 is this case only; page 2 is dataset-level evaluation, provenance and limits.
# Every number comes from the study record, evaluation/metrics.json or evaluation/split.json.

PDF_PREPROCESSING = ("Reorient to RAS, resample to 1 mm, z-score each sequence; SegResNet sliding-window inference; "
                     "probabilities resampled back to the input grid, threshold 0.5, written as BraTS codes 1 / 2 / 4.")
PDF_LIMITS = [
    "Research and teaching prototype. Not a diagnostic tool, not clinically validated; no diagnosis, grading, "
    "prognosis or treatment advice.",
    "Model trained on BraTS 2018 (adult, pre-operative glioma; skull-stripped, co-registered, 1 mm). Other tumour "
    "types, post-operative or non-skull-stripped scans may fail.",
    "Volumes are voxel counts of the predicted regions times voxel volume, so they carry every segmentation error. "
    "ET and NCR are the least reliable regions.",
    "Dataset-level scores are probably optimistic: the evaluation cases likely overlap the model's training data.",
]
REGION_NAME = {"WT": "Whole tumour", "TC": "Tumour core", "ET": "Enhancing tumour", "NCR": "Necrotic core", "ED": "Oedema"}


def _case_label(m, inputs) -> str:
    """Case name from the uploaded file name (4-channel upload, or four files sharing a stem); else 'not provided'."""
    s = m.get("source_4d")
    if s:
        return re.sub(r"\.nii(\.gz)?$", "", s["file"], flags=re.I)
    stems = {re.sub(r"^(t1gd|t1|t2|flair)_|_(t1ce|t1gd|t1|t2|flair)\.nii(\.gz)?$", "", v["file"], flags=re.I)
             for v in inputs.values()}
    return stems.pop() if len(stems) == 1 else "not provided"


def render_pdf(study: ValidatedStudy) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    d = report_data(study)
    m = d["meta"]
    sp, vox = m["spacing_mm"], m["voxel_volume_mm3"]
    ink, muted, rule, accent = (colors.HexColor(c) for c in ("#171614", "#6b665c", "#c9bfab", "#c8402a"))
    e = html.escape
    W = A4[0] - 30 * mm

    def style(name, **kw):
        base = dict(fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=ink)
        base.update(kw)
        return ParagraphStyle(name, **base)

    body, small = style("body"), style("small", fontSize=7.3, leading=9.6, textColor=muted)
    mono = style("mono", fontName="Courier", fontSize=6.8, leading=8.6)
    h1 = style("h1", fontName="Helvetica-Bold", fontSize=17, leading=20)
    h2 = style("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=10, spaceAfter=5)
    num = style("num", fontName="Helvetica-Bold", fontSize=22, leading=24)
    tag = style("tag", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=muted)

    def table(rows, widths, align_right_from=1):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7),
            ("TEXTCOLOR", (0, 0), (-1, 0), muted), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ink),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, rule), ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    case = _case_label(m, d["inputs"])
    prov = m["provenance"] or {}
    model_v = m["volumes"]["model"] or {}
    expert_v = m["volumes"]["expert"] or {}
    agree = m["agreement"] or {}

    # ---- page 1: this case
    head_t = Table([[Paragraph(f"Voxels &nbsp;·&nbsp; case {e(case)}", h1),
                     Paragraph(f"Report {e(d['generated_utc'][:16].replace('T', ' '))} UTC<br/>Study ref "
                               f"{e(d['study_id'].rsplit('_', 1)[-1][:6])}",
                               style("r", fontSize=7.5, leading=10, textColor=muted, alignment=2))]],
                   colWidths=[W * 0.72, W * 0.28])
    head_t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story = [
        head_t, Spacer(1, 5),
        Paragraph(e(DISCLAIMER), style("disc", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                                       textColor=colors.HexColor("#f2ede3"), backColor=ink,
                                       borderPadding=(4, 5, 4, 5))),
        Spacer(1, 8),
    ]
    if _backup_text(m):
        story += [Paragraph(e(_backup_text(m)), style("bk", fontName="Helvetica-Bold", textColor=colors.white,
                                                       backColor=accent, borderPadding=4)), Spacer(1, 8)]

    src = m.get("source_4d")
    if src:
        what = f"one 4-channel file <b>{e(src['file'])}</b> (channels {e(', '.join(src['channel_order']))})"
    else:
        what = "four sequence files (" + e(", ".join(v["file"] for v in d["inputs"].values())) + ")"
    warnings = m["suitability"].get("warnings", [])
    story += [
        Paragraph("1 &nbsp; This case", h2),
        Paragraph(f"Input: {what}; grid {' × '.join(str(x) for x in m['original_shape'])} voxels at "
                  f"{' × '.join(f'{x:.2f}' for x in sp)} mm, {e(m['original_orientation'])}. Expert mask: "
                  f"{'supplied' if m['has_expert'] else 'not supplied'}. Suitability warnings: "
                  f"{e('; '.join(warnings)) if warnings else 'none'}.", body),
        Spacer(1, 8),
        Paragraph("ESTIMATED VOLUME OF THE SUSPECTED TUMOUR REGION PREDICTED BY THE MODEL", tag),
        Spacer(1, 3),
    ]
    tiles = []
    for r, codes in (("WT", "1 + 2 + 4"), ("TC", "1 + 4"), ("ET", "4")):
        v = model_v[r]
        tiles.append([Paragraph(f"{v['mL']:.2f} <font size=10>mL</font>", num),
                      Paragraph(f"<b>{r}</b> · {REGION_NAME[r]} (codes {codes})", body),
                      Paragraph(f"{v['voxels']:,} voxels × {vox:.3f} mm³ = {v['mm3']:,.0f} mm³", small)])
    tile_t = Table([tiles], colWidths=[W / 3] * 3)
    tile_t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.2, ink), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                ("TOPPADDING", (0, 0), (-1, -1), 5)]))
    story += [tile_t, Spacer(1, 5), Paragraph(
        f"V = N × Δx·Δy·Δz = N × {sp[0]:.3f} × {sp[1]:.3f} × {sp[2]:.3f} mm³ = N × {vox:.3f} mm³; "
        "1 mL = 1,000 mm³. N = voxels the model labelled with the region's BraTS codes.", small), Spacer(1, 8)]

    head = ["Region", "Codes", "Model voxels", "Model mL"]
    if m["has_expert"]:
        head += ["Expert mL", "Dice", "Sensitivity"]
    rows = [head]
    for r in ("WT", "TC", "ET", "NCR", "ED"):
        a = model_v[r]
        row = [f"{r}  {REGION_NAME[r]}", REGION_DESC[r].replace("labels ", "").replace("label ", ""),
               f"{a['voxels']:,}", f"{a['mL']:.2f}"]
        if m["has_expert"]:
            g = agree.get(r) or {}
            row += [f"{expert_v[r]['mL']:.2f}", _fmt(g.get("dice")) if g else "—",
                    _fmt(g.get("sensitivity")) if g else "—"]
        rows.append(row)
    widths = [44 * mm, 18 * mm, 24 * mm, 20 * mm] + ([22 * mm, 18 * mm, 22 * mm] if m["has_expert"] else [])
    story.append(table(rows, widths, align_right_from=2))
    if m["has_expert"]:
        story.append(Paragraph("Dice and sensitivity compare the model with the expert mask supplied for this case "
                               "only (computed for WT, TC and ET).", small))

    png = snapshot_png(study, with_expert=False)
    img = Image(io.BytesIO(png))
    scale = W / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    story += [Spacer(1, 8), KeepTogether([img, Paragraph(
        "FLAIR with the model's predicted regions: red NCR (1), green ED (2), yellow ET (4). Radiological view "
        "(patient right on image left); slices through the centre of the predicted region.", small)])]

    u = m.get("uncertainty")
    if u:
        w = u["regions"]["WT"]
        story += [Spacer(1, 6), Paragraph(
            f"<b>Stability.</b> Re-running the model on the scan mirrored along each axis ({u['passes']} passes) "
            f"gives WT between {w['volume_mL_min']:.2f} and {w['volume_mL_max']:.2f} mL; "
            f"{u['wt_uncertain_fraction'] * 100:.1f}% of WT voxels change between passes.", body)]

    mask_rows, notes, read_rows, readings = expert_comparison_parts(m)
    if mask_rows or read_rows:
        block = [Paragraph("Expert comparison &nbsp;·&nbsp; this case only", h2),
                 Paragraph(e(EXPERT_SCOPE), small), Spacer(1, 4)]
        if mask_rows:
            fn = (m.get("expert_mask") or {}).get("filename")
            block += [Paragraph(f"EXPERT MASK{(' · ' + e(fn)) if fn else ''}", tag), Spacer(1, 2),
                      table(mask_rows, [16 * mm, 18 * mm, 22 * mm, 20 * mm, 16 * mm, 22 * mm, 33 * mm, 33 * mm])]
            block += [Paragraph(e(n), style("en", fontSize=7.5, leading=10, textColor=colors.HexColor("#5a3d06")))
                      for n in notes]
            block += [Paragraph(e(MASK_METRIC_NOTE), small), Spacer(1, 6)]
        if read_rows:
            block += [Paragraph(f"DOCTOR'S VOLUME READINGS · ENTERED {e(readings['entered_utc'][:16].replace('T', ' '))} UTC",
                                tag), Spacer(1, 2),
                      table(read_rows, [16 * mm, 26 * mm, 26 * mm, 38 * mm, 28 * mm, 30 * mm]),
                      Paragraph(e(READING_NOTE), small)]
        story += [Spacer(1, 4), KeepTogether(block)]

    # ---- page 2: other cases, provenance, limits
    story.append(PageBreak())
    ev, split = d["evaluation"], d["split"]
    story.append(Paragraph("2 &nbsp; Benchmark: how the model performs on other cases", h2))
    if ev:
        s = ev["summary"]
        n = ev["n_cases"]
        trained = (split or {}).get("training_or_tuning_cases_used_by_us")
        trained_txt = ("none (pretrained model used as-is)" if trained == []
                       else ", ".join(trained) if trained else "not provided")
        story.append(Paragraph(
            f"Dataset-level benchmark, not about this case and separate from any expert comparison above. {n} labelled cases from {e(ev['dataset'])}, split by case ID "
            f"(one case ID is taken as one patient; the release has no patient identifiers). Cases we trained or "
            f"tuned on: {e(trained_txt)}.", body))
        rows = [["Region", "Dice mean ± SD", "Sensitivity mean ± SD", "Cases (n)"]]
        for r in ("WT", "TC", "ET"):
            n_sens = n - int(s[r].get("sensitivity_undefined_cases", 0) or 0)
            rows.append([f"{r}  {REGION_NAME[r]}", f"{s[r]['dice_mean']:.3f} ± {s[r]['dice_sd']:.3f}",
                         f"{_fmt(s[r]['sensitivity_mean'])} ± {_fmt(s[r]['sensitivity_sd'])}",
                         f"{n}" + (f" ({n_sens} for sensitivity)" if n_sens != n else "")])
        story += [Spacer(1, 5), table(rows, [50 * mm, 38 * mm, 42 * mm, 50 * mm]), Spacer(1, 4),
                  Paragraph("Evaluation cases: " + e(", ".join(ev["case_ids"])) + ". Sensitivity is undefined when "
                            "the expert marked no voxels of a region; those cases are left out of that mean.", small),
                  Spacer(1, 6),
                  Paragraph("<b>Overlap with training data.</b> " + e(ev["overlap_disclosure"]),
                            style("ov", fontSize=7.8, leading=10.5, backColor=colors.HexColor("#f3dfb4"),
                                  textColor=colors.HexColor("#5a3d06"), borderPadding=5)), Spacer(1, 6)]
    else:
        story.append(Paragraph("Not computed: evaluation/metrics.json is missing. Needed: scripts/evaluate.py run on "
                               "at least 10 labelled cases that were not used for training or tuning.", body))

    story.append(Paragraph("3 &nbsp; Provenance", h2))
    if src:
        hashes = [Paragraph(f"{e(src['file'])}: {e(src['sha256'])}", mono)]
    else:
        hashes = [Paragraph(f"{e(k)}: {e(v['sha256'])}", mono) for k, v in d["inputs"].items()]
    prov_rows = [
        ["Model", Paragraph(f"{e(MODEL_INFO['name'])} v{e(MODEL_INFO['version'])}, {e(MODEL_INFO['licence'])}; "
                            f"trained by its authors on {e(MODEL_INFO['training_data'])}", body)],
        ["Preprocessing", Paragraph(e(PDF_PREPROCESSING), body)],
        ["Input SHA-256", hashes],
        ["Code commit", Paragraph(e(str(prov.get("code_commit", "not provided"))), mono)],
        ["Inference", Paragraph(f"{e(str(prov.get('timestamp_utc', 'not provided')))} UTC on "
                                f"{e(str(prov.get('device', 'not provided')))}, "
                                f"{e(str(prov.get('inference_seconds', 'not provided')))} s", body)],
        ["Evaluation run", Paragraph(f"{e(str((ev or {}).get('generated_utc', 'not provided')))}, commit "
                                     f"{e(str((ev or {}).get('code_commit', 'not provided'))[:12])}", body)],
        ["Report", Paragraph("Fixed template in app/services/report.py filled with values recorded by this platform; "
                             "figure drawn with matplotlib. No generative AI writes or illustrates this report. "
                             "The segmentation itself is AI model output.", body)],
    ]
    pt = Table(prov_rows, colWidths=[30 * mm, W - 30 * mm])
    pt.setStyle(TableStyle([("FONT", (0, 0), (0, -1), "Helvetica-Bold", 7.5), ("TEXTCOLOR", (0, 0), (0, -1), muted),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.3, rule), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(pt)

    story.append(Paragraph("4 &nbsp; Limits", h2))
    story += [Paragraph(f"{i}. &nbsp;{e(x)}", body) for i, x in enumerate(PDF_LIMITS, 1)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(rule)
        canvas.line(15 * mm, 11 * mm, A4[0] - 15 * mm, 11 * mm)
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(muted)
        canvas.drawString(15 * mm, 7 * mm, "Voxels · research and teaching prototype · not a diagnosis · "
                                           "not for clinical use")
        canvas.drawRightString(A4[0] - 15 * mm, 7 * mm, f"case {case} · page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                      bottomMargin=16 * mm, title=f"Voxels report · case {case}",
                      author="Voxels (fixed template)").build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
