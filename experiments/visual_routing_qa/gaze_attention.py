"""Where does Qwen3-VL look when it is told where the wearer looked?

Rebuilds the oracle inputs of oracle.py (same questions, frames, text context, prompts, image resizing) for
oracle_full / oracle_text / oracle_mark / oracle_saliency, plus a placebo of each gaze condition (same format,
gaze point moved to a random spot >= 0.25 away, per frame). One forward pass per condition; for every text
decoder layer the attention of the LAST position (where the answer letter is predicted) over the visual tokens
is saved, head-averaged, as a map on each frame's merged-token grid.

The model's own attention is never materialised: a custom attention function runs sdpa for the output and,
for the text layers only, separately computes softmax(q_last . K) for the last query row.

    python gaze_attention.py --n 30            # run (GPU)
    python gaze_attention.py --analyze         # summarise saved maps (CPU)
    python gaze_attention.py --qids-file run_data/crop84_qids.json --out-dir run_data/gaze_attention_crop84

Outputs: run_data/gaze_attention/{qid}.npz (maps) and {qid}.json (answers, gaze points, frame ids).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle  # noqa: E402
from oracle import PROMPTS, base, build_choices, draw_mark, load_questions, oracle_slots, saliency_map  # noqa: E402
from gaze_common import gaze_at, load_gaze  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONDS = ("oracle_full", "oracle_text", "oracle_mark", "oracle_saliency",
         "placebo_text", "placebo_mark", "placebo_saliency")
PLACEBO_MIN_DIST = 0.25
CAPTURE: dict[int, "torch.Tensor"] = {}   # layer_idx -> (L,) head-mean attention of the last position


def placebo_xy(xy, seed_key: str):
    rng = random.Random(int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16))
    while True:
        nx, ny = rng.random(), rng.random()
        if (nx - xy[0]) ** 2 + (ny - xy[1]) ** 2 >= PLACEBO_MIN_DIST ** 2:
            return nx, ny


def build(cond: str, frames: list, qid: str):
    """Images, image tags and note for one condition; mirrors oracle.build_input (placebo: shown xy is random)."""
    kind = cond.split("_", 1)[1]                                   # full / text / mark / saliency
    shown = {(k, i): (placebo_xy(xy, f"{qid}|{k}|{i}") if cond.startswith("placebo") else xy)
             for k, i, _, xy in frames}
    images, tags = [], []
    note = PROMPTS.get(f"oracle_{kind}")
    if kind == "full":
        for k, i, img, _ in frames:
            images.append(img); tags.append(f"{k}:{i}")
    elif kind == "text":
        lines = [f"Frame {n + 1}: Gaze({round(shown[(k, i)][0], 3)}, {round(shown[(k, i)][1], 3)})"
                 for n, (k, i, _, _) in enumerate(frames)]
        note = "Gaze information for the relevant frames:\n" + "\n".join(lines) + "\n" + note
        for k, i, img, _ in frames:
            images.append(img); tags.append(f"{k}:{i}")
    elif kind == "mark":
        for k, i, img, _ in frames:
            images.append(draw_mark(img.copy(), *shown[(k, i)])); tags.append(f"{k}:{i}")
    elif kind == "saliency":
        for key in dict.fromkeys(k for k, *_ in frames):
            group = [f for f in frames if f[0] == key]
            images.append(saliency_map(group[0][2].size, [shown[(k, i)] for k, i, _, _ in group]))
            tags.append(f"map:{key}")
            for k, i, img, _ in group:
                images.append(img); tags.append(f"{k}:{i}")
    return images, tags, note, {f"{k}:{i}": list(v) for (k, i), v in shown.items()}


def register_attention():
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, repeat_kv

    def lastrow_sdpa(module, query, key, value, attention_mask, **kwargs):
        if isinstance(module, Qwen3VLTextAttention):
            with torch.no_grad():
                k = repeat_kv(key, module.num_key_value_groups)            # (B, H, L, D)
                s = (query[:, :, -1:, :].float() @ k.float().transpose(-1, -2)) * kwargs.get("scaling", module.scaling)
                if attention_mask is not None and attention_mask.dim() == 4:
                    m = attention_mask[:, :, -1:, : k.shape[-2]]
                    s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m.float()
                CAPTURE[module.layer_idx] = torch.softmax(s, -1)[0, :, 0].mean(0).cpu()   # (L,)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    AttentionInterface.register("lastrow_sdpa", lastrow_sdpa)


def run(args):
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from types import SimpleNamespace
    from worldmm.llm import PromptTemplateManager
    from worldmm.llm.qwen3vl import Qwen3VLModel

    rows, order, target_keys, clips = load_questions(args)
    gaze = load_gaze(str(args.gaze))
    if args.qids_file:                                           # an explicit question set (e.g. the crop-84 cases)
        wanted = {str(q) for q in json.load(open(args.qids_file))}
        qids = [q for q in order if q in wanted and all(gaze.get(k, {}).get("samples") for k in target_keys[q])]
        print(f"{args.qids_file}: {len(wanted)} listed, {len(qids)} with target clips and gaze", flush=True)
    else:
        pool = [q for q in order if not rows[q]["need_audio"] and len(target_keys[q]) == 1
                and gaze.get(target_keys[q][0], {}).get("samples")]
        qids = sorted(random.Random(args.seed).sample(pool, min(args.n, len(pool))), key=order.index)
        print(f"candidates (visual-grounded, one target clip, gaze present): {len(pool)}; running {len(qids)}", flush=True)

    register_attention()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="lastrow_sdpa", device_map={"": 0}).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in letters]
    image_token = model.config.image_token_id
    wm = SimpleNamespace(prompt_template_manager=PromptTemplateManager(), qa_template_name="qa_egolife")
    pre = Qwen3VLModel.__new__(Qwen3VLModel)                     # only for its image preprocessing (thumbnail 512)
    pre.max_size, pre.max_size_video = (512, 512), (256, 256)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    for qid in qids:
        if (out / f"{qid}.json").exists():
            continue
        row = rows[qid]
        context = base.text_record(args, qid, row)
        slots = oracle_slots(target_keys[qid], clips, args.video_root)
        frames = []
        from PIL import Image
        for key, vr, idx, fps in slots:
            frames.append((key, idx, Image.fromarray(vr[idx].asnumpy()).convert("RGB"), gaze_at(gaze[key], idx / fps, tol=0.5)))
        record = {"ID": qid, "question": row["question"], "answer": row["answer"], "type": row.get("type"),
                  "target_keys": target_keys[qid], "real_xy": {f"{k}:{i}": list(xy) for k, i, _, xy in frames},
                  "conds": {}}
        arrays = {}
        for cond in args.conds:
            images, tags, note, shown = build(cond, frames, qid)
            prompt = base.qa_prompt(wm, row["question"], build_choices(row), context["retrieved_items"], images, note=note)
            messages = pre._preprocess_prompt(prompt)
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = process_vision_info(messages)[0]
            inputs = proc(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
            CAPTURE.clear()
            with torch.no_grad():
                logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(logits[letter_ids], 0).tolist()
            ids = inputs["input_ids"][0]
            pos = (ids == image_token).nonzero().flatten().cpu().numpy()
            grids = inputs["image_grid_thw"].cpu().numpy()                     # (n_images, 3) in 16-px patches
            att = torch.stack([CAPTURE[i] for i in sorted(CAPTURE)]).numpy()  # (n_layers, L)
            maps, start = {}, 0
            for tag, (t, h, w) in zip(tags, grids):
                n = int(t * h * w // 4)
                gh, gw = int(h // 2), int(w // 2)
                maps[tag] = att[:, pos[start:start + n]].reshape(att.shape[0], gh, gw)
                start += n
            assert start == len(pos), (start, len(pos))
            frame_tags = [t for t in tags if not t.startswith("map:")]
            arrays[cond] = np.stack([maps[t] for t in frame_tags]).astype(np.float16)   # (frames, layers, gh, gw)
            if any(t.startswith("map:") for t in tags):
                arrays[cond + "__map"] = np.stack([maps[t] for t in tags if t.startswith("map:")]).astype(np.float16)
            record["conds"][cond] = {
                "probs": dict(zip(letters, p)), "pred": letters[int(np.argmax(p))], "frame_tags": frame_tags,
                "shown_xy": shown if cond != "oracle_full" else None, "n_tokens": int(ids.shape[0]),
                "visual_mass_per_layer": att[:, pos].sum(1).round(4).tolist(),
            }
            orig = HERE / "run_data" / "results" / cond / f"{qid}.json"
            if orig.exists():
                record["conds"][cond]["original_response"] = base.read_json(orig).get("response")
        np.savez_compressed(out / f"{qid}.npz", **arrays)
        base.write_json(out / f"{qid}.json", record)
        print(f"{qid}: " + " ".join(f"{c.split('_',1)[0][0]}{c.split('_',1)[1][:3]}={record['conds'][c]['pred']}"
                                    for c in args.conds) + f"  gold={row['answer']}  tokens={record['conds']['oracle_full']['n_tokens']}",
              flush=True)


def gaze_concentration(m: np.ndarray, xy, radius: float) -> float:
    """Share of a frame's attention inside `radius` of xy, divided by that disc's share of the area (1 = uniform)."""
    gh, gw = m.shape
    ys, xs = np.mgrid[0:gh, 0:gw]
    inside = ((xs + 0.5) / gw - xy[0]) ** 2 + ((ys + 0.5) / gh - xy[1]) ** 2 <= radius ** 2
    tot = float(m.sum())
    if tot <= 0 or inside.sum() == 0:
        return float("nan")
    return (float(m[inside].sum()) / tot) / (inside.sum() / inside.size)


def analyze(args):
    recs = sorted(args.out_dir.glob("*.json"))
    if not recs:
        sys.exit("no results")
    rows = []
    for jp in recs:
        r = json.load(open(jp))
        z = np.load(jp.with_suffix(".npz"))
        for cond, c in r["conds"].items():
            A = z[cond].astype(np.float32)                         # (frames, layers, gh, gw)
            for li in range(A.shape[1]):
                real = [gaze_concentration(A[f, li], r["real_xy"][t], args.radius) for f, t in enumerate(c["frame_tags"])]
                shown = ([gaze_concentration(A[f, li], c["shown_xy"][t], args.radius) for f, t in enumerate(c["frame_tags"])]
                         if c["shown_xy"] else real)
                w = A[:, li].sum((1, 2))                           # weight frames by the attention they receive
                rows.append((r["ID"], cond, li, np.nansum(np.array(real) * w) / w.sum(), np.nansum(np.array(shown) * w) / w.sum(),
                             float(c["visual_mass_per_layer"][li]), c["pred"] == r["answer"]))
    import collections
    agg = collections.defaultdict(list)
    for qid, cond, li, real, shown, vis, ok in rows:
        agg[(cond, li)].append((real, shown, vis))
    nl = 1 + max(li for _, _, li, *_ in rows)
    bands = {"early (0-11)": range(0, 12), "middle (12-23)": range(12, 24), "late (24-35)": range(24, nl)}
    conds = [c for c in CONDS if any(k[0] == c for k in agg)]
    nq = len(recs)
    print(f"{nq} questions · radius {args.radius} · concentration = attention share in the disc / area share (1 = uniform)\n")
    print(f"{'condition':18s} " + " ".join(f"{b:>22s}" for b in bands) + "   accuracy")
    for cond in conds:
        cells = []
        for b, ls in bands.items():
            v = np.array([x for li in ls for x in agg[(cond, li)]])
            cells.append(f"real {np.nanmean(v[:, 0]):5.2f} shown {np.nanmean(v[:, 1]):5.2f}")
        acc = np.mean([ok for q, c, li, *_, ok in rows if c == cond and li == 0])
        print(f"{cond:18s} " + " ".join(f"{x:>22s}" for x in cells) + f"   {100 * acc:5.1f}%")
    print("\nvisual-token share of the last position's attention (mean over layers):")
    for cond in conds:
        v = np.array([x[2] for li in range(nl) for x in agg[(cond, li)]])
        print(f"  {cond:18s} {v.mean():.3f}")
    if "oracle_full" in conds:
        baseline_and_cases(args, recs, bands)


def region(shape, pts, radius):
    gh, gw = shape
    ys, xs = np.mgrid[0:gh, 0:gw]
    cx, cy = (xs + 0.5) / gw, (ys + 0.5) / gh
    m = np.zeros(shape, bool)
    for x, y in pts:
        m |= (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2
    return m


def conc_mask(a, m):
    t = float(a.sum())
    return float("nan") if t <= 0 or m.sum() == 0 else (float(a[m].sum()) / t) / (m.sum() / m.size)


def baseline_and_cases(args, recs, bands):
    """Concentration relative to oracle_full at the same point (removes position bias such as corner sinks),
    the saliency map itself, and whether attention moved more on questions the condition fixed or broke."""
    R = args.radius
    mid = bands["middle (12-23)"]
    shift = {}                                   # (method, kind) -> list of per-frame (cond, full) at the point
    per_q = []                                   # (qid, method, outcome, ratio at real point, middle layers)
    sal = {}                                     # (cond, band) -> (map share, equal share, trajectory conc.)
    for jp in recs:
        r = json.load(open(jp))
        z = np.load(jp.with_suffix(".npz"))
        tags = r["conds"]["oracle_full"]["frame_tags"]
        F = z["oracle_full"].astype(np.float32)
        # outcome as graded in the original oracle run (the set was selected on it); this run's pred if absent
        def is_ok(cond, qid=r["ID"], pred=None):
            orig = HERE / "run_data" / "results" / cond / f"{qid}.json"
            return bool(base.read_json(orig).get("correct")) if orig.exists() else r["conds"][cond]["pred"] == r["answer"]
        full_ok = is_ok("oracle_full")
        for kind in ("text", "mark", "saliency"):
            for prefix, key in (("oracle", "real"), ("placebo", "placebo")):
                cond = f"{prefix}_{kind}"
                if cond not in r["conds"]:
                    continue
                C = z[cond].astype(np.float32)
                pts = r["real_xy"] if prefix == "oracle" else r["conds"][cond]["shown_xy"]
                for b, ls in bands.items():
                    for li in ls:
                        for f, t in enumerate(tags):
                            shift.setdefault((kind, key, b), []).append(
                                (gaze_concentration(C[f, li], pts[t], R), gaze_concentration(F[f, li], pts[t], R)))
                if prefix == "oracle":
                    ok = is_ok(cond)
                    outcome = "fixed" if ok and not full_ok else "broke" if full_ok and not ok else "same"
                    num = np.nanmean([gaze_concentration(C[f, li], pts[t], R) for li in mid for f, t in enumerate(tags)])
                    den = np.nanmean([gaze_concentration(F[f, li], pts[t], R) for li in mid for f, t in enumerate(tags)])
                    per_q.append((r["ID"], kind, outcome, num / den))
            if kind == "saliency":
                for prefix in ("oracle", "placebo"):
                    cond = f"{prefix}_saliency"
                    if cond + "__map" not in z.files:
                        continue
                    M = z[cond + "__map"].astype(np.float32)                 # (clips, layers, gh, gw)
                    C = z[cond].astype(np.float32)
                    keys = list(dict.fromkeys(t.split(":")[0] for t in tags))
                    shown = r["conds"][cond]["shown_xy"]
                    for b, ls in bands.items():
                        for li in ls:
                            mm = float(M[:, li].sum()); fm = float(C[:, li].sum())
                            tc = np.nanmean([conc_mask(M[ci, li], region(M.shape[2:], [shown[t] for t in tags if t.startswith(k + ":")], R))
                                             for ci, k in enumerate(keys)])
                            sal.setdefault((cond, b), []).append((mm / (mm + fm), len(keys) / (len(keys) + len(tags)), tc))

    print(f"\nrelative to oracle_full at the same point (x = condition / no-gaze), r={R}:")
    print(f"{'method':9s} {'layers':15s} | at REAL gaze (oracle_*)     | at PLACEBO point (placebo_*)")
    for kind in ("text", "mark", "saliency"):
        for b in bands:
            cells = []
            for key in ("real", "placebo"):
                v = np.array(shift.get((kind, key, b), [(np.nan, np.nan)]), dtype=float)
                a, c = np.nanmean(v[:, 0]), np.nanmean(v[:, 1])
                cells.append(f"{a:5.2f} vs {c:5.2f} (x{a / c:4.2f})")
            print(f"{kind:9s} {b:15s} | {cells[0]:27s} | {cells[1]}")
    if sal:
        print("\nsaliency map: share of the visual attention on the map(s) (equal-share in brackets), "
              "and concentration on the drawn trajectory inside the map")
        for (cond, b), v in sal.items():
            v = np.array(v, dtype=float)
            print(f"  {cond:17s} {b:15s} map {np.nanmean(v[:, 0]):.3f} ({np.nanmean(v[:, 1]):.3f})  trajectory {np.nanmean(v[:, 2]):.2f}")
    print("\nper question, middle layers: attention at the real gaze relative to oracle_full, by what the condition did to the answer")
    print(f"{'method':9s} | {'fixed (full wrong -> right)':>28s} | {'broke (full right -> wrong)':>28s} | {'same':>18s}")
    for kind in ("text", "mark", "saliency"):
        cells = []
        for oc in ("fixed", "broke", "same"):
            v = [x[3] for x in per_q if x[1] == kind and x[2] == oc]
            cells.append(f"n={len(v):2d}  x{np.nanmedian(v):4.2f}" if v else "n= 0")
        print(f"{kind:9s} | {cells[0]:>28s} | {cells[1]:>28s} | {cells[2]:>18s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--gaze", type=Path, default=base.ROOT / "experiments/gaze_crop/gaze_points.json")
    ap.add_argument("--video-root", type=Path, default=Path(oracle.VIDEO_ROOT))
    ap.add_argument("--out", type=Path, default=HERE / "run_data", help="holds text_context/")
    ap.add_argument("--out-dir", type=Path, default=HERE / "run_data" / "gaze_attention")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qids-file", type=Path, help="JSON list of question ids to run instead of a random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--conds", nargs="+", choices=CONDS, default=list(CONDS))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--radius", type=float, default=0.1, help="analysis: disc radius around the gaze point")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
