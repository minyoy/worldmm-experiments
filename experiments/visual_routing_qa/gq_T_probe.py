"""Does GazeQwen's gaze effect depend on the number of temporal steps T? (small probe, single-clip oracle questions)

The original training code builds videos with `max_frames: 3` -> 2 frames -> T=1 (evaluation default 8 frames -> T=4); our
oracle runs use ~30 frames -> T=15. Here the same questions are run with n = 2 / 8 / 30 frames picked evenly from the
clip's 1 fps frames. Each temporal step t is stamped with the real clip time of its first frame; fixations stay in real clip
seconds (the clip is a single one, so no timeline remapping). Two prompts: A = our setting (system prompt + retrieved text
memory), B = GazeQwen's own prompt (video + "Question/Options/The best option is:", default system prompt).

Conditions per (prompt, n, question): lora_only (hooks inactive), inject_only (never-active fixation), gaze, placebo
(same times/durations, random centres >= 0.25 away). `active_steps` = temporal steps at which at least one fixation is
active, i.e. where gaze can enter the model at all.

    cd ~/gazeqwen && PYTHONPATH=".:third_party/vjepa2" python .../gq_T_probe.py --stage run --attn eager
    python .../gq_T_probe.py --stage analyze        # stdlib only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "run_data/results/tprobe"          # --tag appends a suffix
CONDS = ("lora_only", "inject_only", "gaze", "placebo")


DETECTOR_OVERRIDE = None


def run(args) -> None:
    import numpy as np
    import torch

    import oracle_gazeqwen as og
    from gq_smoke import GQ, fixations, random_placebo

    base, oracle = og.base, og.oracle
    ids = json.loads((HERE / "run_data/t_probe_ids.json").read_text())["ids"]
    if args.limit:
        ids = ids[:args.limit]
    ns = [int(v) for v in args.ns.split(",")]
    kinds = args.prompts.split(",")

    class A:                                                   # what oracle.load_questions reads
        pool, qa = base.ROOT / "experiments/gaze_crop/pool_all.json", base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json"
        out, limit = HERE / "run_data", None
        video_root = Path(oracle.VIDEO_ROOT)

    rows, order, target_keys, clips = oracle.load_questions(A)
    gaze = oracle.load_gaze(str(base.ROOT / "experiments/gaze_crop/gaze_points.json"))
    system = og.qa_system_prompt()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks
    from gazeqwen.lora import apply_lora, load_lora_state_dict
    from gazeqwen.model import GazeLens
    from gazeqwen.train import build_mcqa_prompt, get_abcd_token_ids
    from gazeqwen.vjepa_features import VJEPAFeatureExtractor

    dev = torch.device("cuda")
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn).eval()
    ck = torch.load(GQ / "checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
    f_theta = GazeLens(**ck["config"]).to(dev).eval()
    f_theta.load_state_dict(ck["state_dict"])
    apply_lora(model, rank=ck["lora_config"]["rank"], alpha=ck["lora_config"]["alpha"])
    print(f"LoRA tensors loaded: {load_lora_state_dict(model, ck['lora_state_dict'])}", flush=True)
    ctx = GazeLensContext(f_theta)
    register_gazelens_hooks(model, ctx)
    vjepa = VJEPAFeatureExtractor(device=dev, checkpoint_path=str(GQ / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"))
    abcd = get_abcd_token_ids(proc.tokenizer)
    captured = {}
    model.lm_head.register_forward_hook(lambda m, i, o: captured.__setitem__("h", i[0][0, -1].detach()))

    def make_inputs(kind, row, context, frames):
        if kind == "A":
            return og.build_inputs(proc, system, row, context, frames, dev)
        choices = base.build_choices(row)
        letters = sorted(choices)
        prompt = build_mcqa_prompt(row["question"], [f"{k}. {choices[k]}" for k in letters])
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}],
                                        tokenize=False, add_generation_prompt=True)
        return proc(text=[text], videos=[frames], fps=1.0, return_tensors="pt").to(dev), letters

    def answer_probs(letters):
        w = model.lm_head.weight[torch.tensor([abcd["ABCD".index(c)] for c in letters], device=dev)].float()
        return torch.softmax(w @ captured["h"].float(), 0).cpu().tolist()

    for qid in ids:
        row = rows[qid]
        key = target_keys[qid][0]
        todo = [(k, n, c) for k in kinds for n in ns for c in CONDS if not (RESULTS / f"{k}_n{n}" / c / f"{qid}.json").exists()]
        if not todo:
            continue
        try:
            slots = oracle.oracle_slots(target_keys[qid], clips, A.video_root)
            frames_all, meta_all = og.load_video(slots, args.max_side)
            context = base.text_record(A, qid, row)
        except Exception as exc:
            print(f"{qid}: prepare failed {type(exc).__name__}: {exc}", flush=True)
            continue
        N = len(meta_all)
        scan = fixations(gaze[key]["samples"], args.min_dur, DETECTOR_OVERRIDE)
        placebo = random_placebo(scan, args.seed + int(qid))
        for n in ns:
            n_eff = min(n, N - N % 2)
            idx = np.linspace(0, N - 1, n_eff).round().astype(int)
            frames = frames_all[idx]
            step_times = [meta_all[idx[2 * t]][1] for t in range(n_eff // 2)]
            active = sum(any(m - d / 2 <= t <= m + d / 2 for _, _, m, d in scan.tolist()) for t in step_times)
            for kind in kinds:
                if not any(k == kind and nn == n for k, nn, _ in todo):
                    continue
                t0 = time.time()
                inputs, letters = make_inputs(kind, row, context, frames)
                grid = inputs["video_grid_thw"][0]
                T, H, W = int(grid[0]), int(grid[1]) // 2, int(grid[2]) // 2
                feats = vjepa.extract_from_raw_frames(frames, T, H, W)
                plan = {"lora_only": None, "inject_only": torch.tensor(og.INACTIVE), "gaze": scan, "placebo": placebo}
                for cond, sp in plan.items():
                    path = RESULTS / f"{kind}_n{n}" / cond / f"{qid}.json"
                    if path.exists():
                        continue
                    res = {"ID": qid, "prompt": kind, "n_frames": int(n_eff), "T": T, "cond": cond, "answer": row["answer"],
                           "need_audio": bool(row["need_audio"]), "n_tokens": int(inputs["input_ids"].shape[1]),
                           "n_fix": int(len(scan)), "active_steps": int(active), "step_times": [round(t, 2) for t in step_times]}
                    try:
                        with torch.no_grad():
                            if sp is None:
                                model(**inputs, logits_to_keep=1)
                            else:
                                with ctx.active(sp.to(dev), step_times, backbone_features=feats):
                                    model(**inputs, logits_to_keep=1)
                        probs = answer_probs(letters)
                        pred = letters[int(np.argmax(probs))]
                        res.update({"probs": dict(zip(letters, probs)), "response": pred, "correct": pred == row["answer"], "status": "ok"})
                    except Exception as exc:
                        res.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                        torch.cuda.empty_cache()
                    base.write_json(path, res)
                print(f"{qid} prompt {kind} n={n_eff} T={T} active_steps={active}/{T} tokens={inputs['input_ids'].shape[1]} "
                      f"{time.time() - t0:.1f}s", flush=True)


def exact_p(up: int, down: int) -> float:
    n = up + down
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, x) for x in range(min(up, down) + 1)) / 2 ** n)


def analyze(args) -> None:
    R = {}
    for f in RESULTS.glob("*/*/*.json"):
        d = json.loads(f.read_text())
        if d["status"] == "ok":
            R.setdefault((d["prompt"], d["n_frames"], d["T"]), {}).setdefault(d["cond"], {})[d["ID"]] = d
    lines = ["# GazeQwen: gaze 효과 vs 시간 스텝 수 T (소규모 프로브)", "",
             "정답 클립 1개 + fixation 1개 이상(2.5 s 필터)인 문항. 프롬프트 A = 우리 설정(텍스트 메모리), B = GazeQwen 원래 프롬프트. "
             "`활성 스텝` = fixation이 활성인 시간 스텝이 있는 문항의 비율과 평균 개수. 사후·탐색용이며 표본이 작다.", ""]
    for kind in ("A", "B"):
        keys = sorted(k for k in R if k[0] == kind)
        if not keys:
            continue
        lines += [f"## 프롬프트 {kind}", "",
                  "| 프레임 | T | N | 활성 스텝 있는 문항 | 평균 활성 스텝 | lora_only | inject_only | gaze | placebo |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        detail = []
        for k in keys:
            C = R[k]
            ids = sorted(set.intersection(*[set(C[c]) for c in CONDS if c in C]), key=int) if all(c in C for c in CONDS) else []
            if not ids:
                continue
            n = len(ids)
            act = [C["gaze"][i]["active_steps"] for i in ids]
            acc = {c: sum(C[c][i]["correct"] for i in ids) for c in CONDS}
            lines.append(f"| {k[1]} | {k[2]} | {n} | {sum(a > 0 for a in act)}/{n} | {sum(act) / n:.2f} | "
                         + " | ".join(f"{acc[c]} ({100 * acc[c] / n:.0f}%)" for c in CONDS) + " |")
            pc = lambda c, i: C[c][i]["probs"][C[c][i]["answer"]]            # noqa: E731
            l1 = lambda a, b, i: 0.5 * sum(abs(C[a][i]["probs"][x] - C[b][i]["probs"][x]) for x in C[a][i]["probs"])  # noqa: E731
            row = [f"| {k[1]} (T={k[2]}) |"]
            for a, b in (("gaze", "inject_only"), ("gaze", "placebo")):
                up = sum(C[a][i]["correct"] and not C[b][i]["correct"] for i in ids)
                dn = sum(C[b][i]["correct"] and not C[a][i]["correct"] for i in ids)
                chg = sum(C[a][i]["response"] != C[b][i]["response"] for i in ids)
                row.append(f" {chg} | {up}/{dn} (p={exact_p(up, dn):.2f}) | {sum(pc(a, i) - pc(b, i) for i in ids) / n:+.4f} | "
                           f"{sum(l1(a, b, i) for i in ids) / n:.4f} |")
            detail.append("".join(row))
        lines += ["", "| 프레임 | gaze↔inject_only 답 바뀜 | 개선/악화 | ΔP(정답) | 평균 확률 이동 | gaze↔placebo 답 바뀜 | 개선/악화 | ΔP(정답) | 평균 확률 이동 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"] + detail + [""]
    out = HERE / f"run_data/analysis/gazeqwen_T_probe{args.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("run", "analyze"), required=True)
    ap.add_argument("--ns", default="2,8,30", help="frames per video; T = frames / 2")
    ap.add_argument("--prompts", default="A,B")
    ap.add_argument("--attn", default="eager", choices=("eager", "sdpa"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--min-dur", type=float, default=2.5)
    ap.add_argument("--radius", type=float, help="fixation radius (normalised). With it, the detector is called like StreamGaze: "
                    "duration_thresh=--min-dur, gap_thresh=--gap, no dropout tolerance beyond --dropout")
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--tag", default="", help="suffix for the results folder / analysis file, e.g. _sg096")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="only the first N questions of the list")
    args = ap.parse_args()
    global RESULTS, DETECTOR_OVERRIDE
    RESULTS = HERE / f"run_data/results/tprobe{args.tag}"
    if args.radius is not None:
        DETECTOR_OVERRIDE = dict(radius_thresh=args.radius, duration_thresh=args.min_dur, gap_thresh=args.gap,
                                 dropout_thresh=args.dropout)
    run(args) if args.stage == "run" else analyze(args)


if __name__ == "__main__":
    main()
