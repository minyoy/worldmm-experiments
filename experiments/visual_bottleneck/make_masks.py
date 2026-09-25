#!/usr/bin/env python3
"""
Exp.3 step 1: detect the task-relevant region in each frames16/{qid}/ frame and write binary masks.

Pipeline (per frame): GroundingDINO (text prompt -> boxes) -> SAM (boxes -> segmentation) -> union mask.
Both models are loaded through `transformers` (GroundingDINO-tiny, SAM2.1 if available else SAM-ViT-base),
so no extra repositories are needed. Detection is per frame; there is no cross-frame tracking, which is
why the review step exists.

Outputs
    mask_prompts.json                      (created by --init-prompts, then edited by hand)
    masks/{qid}/{frame}.png                binary mask, 255 = relevant region
    masks/{qid}/detections.json            boxes / scores / labels per frame
    masks/{qid}/overlay_{frame}.jpg        mask overlay for review
    review_masks.html                      (--review) one page to approve / reject / annotate every qid

Usage (from repo root):
    python experiments/visual_bottleneck/make_masks.py --init-prompts      # writes mask_prompts.json template
    # edit mask_prompts.json: one GroundingDINO prompt per qid, e.g. "screwdriver . hand ."
    python experiments/visual_bottleneck/make_masks.py                     # detect + segment
    python experiments/visual_bottleneck/make_masks.py --review            # build review_masks.html
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP_DIR, FRAMES16_DIR, MASKS_DIR, SUBSET_PATH, load_json, load_subset, save_json  # noqa: E402

PROMPTS_PATH = os.path.join(EXP_DIR, "mask_prompts.json")
REVIEW_HTML = os.path.join(EXP_DIR, "review_masks.html")

# Relevant-region rule per question flavour (PLAN.md section 4). The rule string is only guidance
# for whoever writes the prompt; GroundingDINO gets the prompt text.
RULES = {
    "object_identity":            "target object",
    "object_location":            "target object + supporting surface / surroundings",
    "person_object_interaction":  "object + hand / person interacting with it",
    "spatial_relation":           "both related objects",
    "fine_grained_action":        "hand + manipulated object",
}
TYPE_DEFAULT_RULE = {
    "EntityLog": "object_identity",
    "RelationMap": "person_object_interaction",
    "EventRecall": "fine_grained_action",
    "HabitInsight": "person_object_interaction",
    "TaskMaster": "object_identity",
}


def init_prompts(subset_path: str, out_path: str) -> None:
    subset = load_subset(subset_path)
    prompts = load_json(out_path) if os.path.exists(out_path) else {}
    for e in subset["entries"]:
        if not e.get("masking") or e["ID"] in prompts:
            continue
        kw = (e.get("keywords") or "").strip()
        prompts[e["ID"]] = {
            "question": e["question"],
            "keywords": kw,
            "reason": e.get("reason", ""),
            "rule": TYPE_DEFAULT_RULE.get(e["type"], "object_identity"),
            "prompt": " . ".join(w for w in kw.replace(",", " ").split() if len(w) > 2) + " ." if kw else "",
        }
    save_json(prompts, out_path)
    print(f"wrote {out_path} with {len(prompts)} entries; edit each 'prompt' (GroundingDINO phrases separated by ' . ')")
    print("rules:", json.dumps(RULES, indent=2))


class Detector:
    def __init__(self, device: str, dino_id: str, box_thr: float, text_thr: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = device
        self.box_thr, self.text_thr = box_thr, text_thr
        self.proc = AutoProcessor.from_pretrained(dino_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()

    def __call__(self, img: Image.Image, prompt: str) -> Dict[str, Any]:
        inputs = self.proc(images=img, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[img.size[::-1]],
        )[0]
        labels = res.get("text_labels", res.get("labels", []))
        return {"boxes": [[float(v) for v in b] for b in res["boxes"].tolist()],
                "scores": [float(s) for s in res["scores"].tolist()],
                "labels": [str(l) for l in labels]}


class Segmenter:
    """SAM2 via transformers when available, else SAM. Boxes in -> union binary mask out."""

    def __init__(self, device: str, sam2_id: str, sam_id: str):
        import torch
        self.torch = torch
        self.device = device
        try:
            from transformers import Sam2Model, Sam2Processor
            self.proc = Sam2Processor.from_pretrained(sam2_id)
            self.model = Sam2Model.from_pretrained(sam2_id).to(device).eval()
            self.name = sam2_id
        except Exception as e:  # older transformers or model not available
            from transformers import SamModel, SamProcessor
            print(f"SAM2 unavailable ({type(e).__name__}); falling back to {sam_id}")
            self.proc = SamProcessor.from_pretrained(sam_id)
            self.model = SamModel.from_pretrained(sam_id).to(device).eval()
            self.name = sam_id

    def __call__(self, img: Image.Image, boxes: List[List[float]]) -> np.ndarray:
        h, w = img.size[1], img.size[0]
        if not boxes:
            return np.zeros((h, w), dtype=np.uint8)
        inputs = self.proc(images=img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.proc.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]  # (num_boxes, 1, H, W) bool
        union = masks.any(dim=0).any(dim=0).numpy()
        return (union.astype(np.uint8) * 255)


def overlay(img: Image.Image, mask: np.ndarray, boxes: List[List[float]]) -> Image.Image:
    from PIL import ImageDraw
    base = img.convert("RGBA")
    color = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = Image.fromarray((mask > 0).astype(np.uint8) * 110, mode="L")
    color.putalpha(alpha)
    out = Image.alpha_composite(base, color)
    d = ImageDraw.Draw(out)
    for b in boxes:
        d.rectangle(b, outline=(255, 255, 0, 255), width=2)
    return out.convert("RGB")


def run_detection(args) -> None:
    prompts = load_json(args.prompts)
    subset = load_subset(args.subset)
    qids = [e["ID"] for e in subset["entries"] if e.get("masking")]
    det = Detector(args.device, args.dino, args.box_threshold, args.text_threshold)
    seg = Segmenter(args.device, args.sam2, args.sam)
    print(f"segmenter: {seg.name}")

    for qid in qids:
        p = prompts.get(qid, {})
        prompt = (p.get("prompt") or "").strip()
        src = os.path.join(args.frames16_dir, qid)
        if not prompt or not os.path.isdir(src):
            print(f"[{qid}] skipped (no prompt or no frames16)")
            continue
        out_dir = os.path.join(args.masks_dir, qid)
        if os.path.exists(os.path.join(out_dir, "detections.json")) and not args.overwrite:
            continue
        os.makedirs(out_dir, exist_ok=True)
        detections: Dict[str, Any] = {"prompt": prompt, "rule": p.get("rule"), "frames": {}}
        for fname in sorted(f for f in os.listdir(src) if f.endswith(".jpg")):
            img = Image.open(os.path.join(src, fname)).convert("RGB")
            d = det(img, prompt)
            if args.max_boxes and len(d["boxes"]) > args.max_boxes:
                order = np.argsort(d["scores"])[::-1][:args.max_boxes]
                d = {k: [v[i] for i in order] for k, v in d.items()}
            mask = seg(img, d["boxes"])
            Image.fromarray(mask, mode="L").save(os.path.join(out_dir, fname.replace(".jpg", ".png")))
            overlay(img, mask, d["boxes"]).save(os.path.join(out_dir, f"overlay_{fname}"), quality=85)
            d["mask_area_frac"] = float((mask > 0).mean())
            detections["frames"][fname] = d
        save_json(detections, os.path.join(out_dir, "detections.json"))
        covered = sum(1 for v in detections["frames"].values() if v["boxes"])
        print(f"[{qid}] '{prompt}' -> boxes on {covered}/{len(detections['frames'])} frames")


def build_review(args) -> None:
    subset = load_subset(args.subset)
    prompts = load_json(args.prompts) if os.path.exists(args.prompts) else {}
    entries = [e for e in subset["entries"] if e.get("masking")]
    rel = lambda p: os.path.relpath(p, EXP_DIR)
    blocks = []
    for e in entries:
        qid = e["ID"]
        mdir = os.path.join(args.masks_dir, qid)
        det_path = os.path.join(mdir, "detections.json")
        if not os.path.exists(det_path):
            continue
        det = load_json(det_path)
        thumbs = "".join(
            f'<figure><img src="{rel(os.path.join(mdir, "overlay_" + f))}" loading="lazy">'
            f'<figcaption>{f} · {len(v["boxes"])} box · {v["mask_area_frac"]*100:.1f}%</figcaption></figure>'
            for f, v in sorted(det["frames"].items())
        )
        choices = " ".join(f"({k}) {v}" for k, v in sorted(e["choices"].items()))
        blocks.append(f"""
<section data-qid="{qid}">
  <h2>Q{qid} · {e['type']} · answer {e['answer']}</h2>
  <p><b>{e['question']}</b><br>{choices}</p>
  <p>prompt: <code>{det['prompt']}</code> · rule: {det.get('rule')} · reason: {e.get('reason','')}</p>
  <div class="grid">{thumbs}</div>
  <label><input type="radio" name="d{qid}" value="approve"> approve</label>
  <label><input type="radio" name="d{qid}" value="reject"> reject</label>
  <label><input type="radio" name="d{qid}" value="fix"> fix (note boxes below)</label>
  <textarea placeholder='notes, or manual boxes as JSON: {{"00_03.jpg": [[x1,y1,x2,y2]]}}'></textarea>
</section>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>mask review</title>
<style>
body{{font-family:system-ui;margin:16px;max-width:1400px}} section{{border-top:2px solid #ccc;padding:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(8,1fr);gap:4px}} figure{{margin:0}} img{{width:100%}}
figcaption{{font-size:10px;color:#666}} textarea{{width:100%;height:48px}} label{{margin-right:12px}}
#bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc}}
</style>
<div id="bar"><b>{len(blocks)} questions</b> · decisions are saved in this browser ·
<button onclick="exportJSON()">export decisions JSON</button></div>
{''.join(blocks)}
<script>
const KEY='mask_review_decisions';
const state=JSON.parse(localStorage.getItem(KEY)||'{{}}');
document.querySelectorAll('section').forEach(s=>{{
  const q=s.dataset.qid, st=state[q]||{{}};
  s.querySelectorAll('input[type=radio]').forEach(r=>{{ if(r.value===st.decision) r.checked=true;
    r.onchange=()=>{{state[q]={{...(state[q]||{{}}),decision:r.value}}; localStorage.setItem(KEY,JSON.stringify(state));}};}});
  const t=s.querySelector('textarea'); t.value=st.note||'';
  t.oninput=()=>{{state[q]={{...(state[q]||{{}}),note:t.value}}; localStorage.setItem(KEY,JSON.stringify(state));}};
}});
function exportJSON(){{const a=document.createElement('a');
  a.href='data:application/json,'+encodeURIComponent(JSON.stringify(state,null,2));
  a.download='review_decisions.json'; a.click();}}
</script>"""
    with open(REVIEW_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {REVIEW_HTML} ({len(blocks)} questions). Open it in a browser, then export review_decisions.json "
          f"next to it for apply_masks.py")
    if len(blocks) < len(entries):
        print(f"note: {len(entries) - len(blocks)} masking questions have no detections yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET_PATH)
    ap.add_argument("--frames16-dir", default=FRAMES16_DIR)
    ap.add_argument("--masks-dir", default=MASKS_DIR)
    ap.add_argument("--prompts", default=PROMPTS_PATH)
    ap.add_argument("--init-prompts", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--sam", default="facebook/sam-vit-base")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--max-boxes", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.init_prompts:
        init_prompts(args.subset, args.prompts)
    elif args.review:
        build_review(args)
    else:
        if not os.path.exists(args.prompts):
            sys.exit(f"{args.prompts} not found; run with --init-prompts first and edit the prompts")
        run_detection(args)


if __name__ == "__main__":
    main()
