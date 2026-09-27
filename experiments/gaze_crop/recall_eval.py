#!/usr/bin/env python3
"""
Visual retrieval recall per arm. No LLM, no answer generation -- just whether the target 30-sec
clip lands in the top-k for the question.

This is the metric the experiment turns on. In the visual-bottleneck run, oracle visual frames were
worth +12.2 points on the 82 questions where text retrieval failed, while real visual retrieval put
the target in the top-3 for 5% of questions. Recall is what a crop can move; accuracy is what recall
buys, later and more expensively.

Per question the candidate set is the pool restricted to clips the agent could have seen
(ts_end <= query_time), so nothing from the future is ranked. Arms are compared only on questions
that are evaluable in every compared arm, and the comparison is paired: same questions, same query
embeddings, same candidate sets. Only the pixels behind the clip embeddings differ.

    python experiments/gaze_crop/recall_eval.py                          # all arms in emb/
    python experiments/gaze_crop/recall_eval.py --arms full center@0.5 gaze@0.5 --ks 1 3 5 10
    python experiments/gaze_crop/recall_eval.py --query-source keywords
    python experiments/gaze_crop/recall_eval.py --self-test              # no GPU: random embeddings
"""

import argparse
import glob
import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    ANALYSIS_DIR, EMB_DIR, EXP_DIR, POOL_PATH, RESULTS_DIR, VB_DIR, clip_start_sec, load_json,
    save_json, unit,
)

DEFAULT_KS = (1, 3, 5, 10, 20, 50)
MAIN_K = 3  # WorldMM's visual top-k, and what condition E' actually consumed


def arm_name_from_file(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_arms(emb_dir: str, wanted: Optional[List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
    """arm name -> {clip_key: embedding}. Files are emb/<arm>.npz as written by embed_arms.py."""
    files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no arm embeddings in {emb_dir}; run embed_arms.py first")
    out = {}
    for f in files:
        name = arm_name_from_file(f)
        z = np.load(f, allow_pickle=False)
        emb = z["emb"].astype(np.float32)
        bad = int((np.linalg.norm(emb, axis=1) == 0).sum()) + int((~np.isfinite(emb)).any(axis=1).sum())
        if bad:
            print(f"WARNING: arm '{name}' has {bad} zero/non-finite embedding rows")
        out[name] = {"keys": z["keys"].tolist(), "emb": unit(emb)}
    if wanted:
        canon = {n.replace("@", "_").replace(".", ""): n for n in out}
        picked = {}
        for w in wanted:
            k = w.replace("@", "_").replace(".", "")
            if k in canon:
                picked[canon[k]] = out[canon[k]]
            elif w in out:
                picked[w] = out[w]
            else:
                raise SystemExit(f"arm '{w}' has no {os.path.join(emb_dir, k + '.npz')}; have {sorted(out)}")
        out = picked
    return out


def agent_queries(results_path: str) -> Dict[str, List[str]]:
    """The text queries the agent itself sent to visual memory, from a condition-E results file."""
    out: Dict[str, List[str]] = {}
    if not os.path.exists(results_path):
        return out
    for r in load_json(results_path):
        qs = [c["query"] for c in (r.get("visual_calls") or []) if c.get("kind") == "text"]
        seen, uniq = set(), []
        for q in qs:
            q = (q or "").strip()
            if q and q not in seen:
                seen.add(q)
                uniq.append(q)
        if uniq:
            out[str(r["ID"])] = uniq
    return out


def question_queries(pool: Dict[str, Any], source: str, results_path: str) -> Tuple[Dict[str, List[str]], int]:
    agent = agent_queries(results_path) if source == "agent" else {}
    qmap, fell_back = {}, 0
    for q in pool["questions"]:
        if source == "question":
            qmap[q["ID"]] = [q["question"]]
        elif source == "keywords":
            kw = (q.get("keywords") or "").strip()
            qmap[q["ID"]] = [kw] if kw else [q["question"]]
            fell_back += not kw
        elif source == "agent":
            if q["ID"] in agent:
                qmap[q["ID"]] = agent[q["ID"]]
            else:
                qmap[q["ID"]] = [q["question"]]
                fell_back += 1
        else:
            raise ValueError(source)
    return qmap, fell_back


def encode_queries(texts: List[str], cache_path: str, self_test: bool, dim: int) -> Dict[str, np.ndarray]:
    """VLM2Vec's text side -- the same call VisualMemory._retrieve_by_similarity makes."""
    cache: Dict[str, np.ndarray] = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        cache = {k: v for k, v in zip(z["texts"].tolist(), z["emb"])}
    todo = [t for t in texts if t not in cache]
    if todo:
        if self_test:
            for t in todo:
                seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
                cache[t] = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
        else:
            from worldmm.embedding import EmbeddingModel  # late import: --self-test needs no GPU
            model = EmbeddingModel()
            print(f"encoding {len(todo)} distinct queries with encode_vis_query ...")
            emb = np.asarray(model.encode_vis_query(todo), dtype=np.float32)
            for t, v in zip(todo, emb):
                cache[t] = v
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        keys = sorted(cache)
        np.savez(cache_path, texts=np.array(keys), emb=np.stack([cache[k] for k in keys]))
    return {t: unit(cache[t].astype(np.float32)) for t in texts}


def bootstrap_ci(a: np.ndarray, b: np.ndarray, iters: int = 5000, seed: int = 0) -> Tuple[float, float]:
    """95% CI of mean(a) - mean(b), resampling questions (paired)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(iters, n))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_PATH)
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--arms", nargs="+", default=None, help="default: every npz in emb/")
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--main-k", type=int, default=MAIN_K, help="k used for the arm-vs-arm tests")
    ap.add_argument("--target-tolerance-sec", type=float, default=0.0, metavar="SEC",
                    help="count a clip as a hit when it starts within SEC of an annotated target "
                         "clip on the same day. 0 (default) is the strict benchmark definition: "
                         "only the one 30-sec clip EgoLife tagged. Raising it asks a different "
                         "question -- 'did the arm land near the evidence' rather than 'on it' -- "
                         "and mechanically inflates recall, so the table prints the chance level "
                         "and the targets-per-question that go with each setting. Compare arms "
                         "only at the same tolerance.")
    ap.add_argument("--query-source", choices=("question", "keywords", "agent"), default="question",
                    help="'question': the raw question text, which is what condition E' retrieved "
                         "with (default). 'keywords': the QA's own keywords, a friendlier query. "
                         "'agent': the queries the 8B issued in condition E (it searched visual "
                         "memory for almost no questions, so most fall back to the question).")
    ap.add_argument("--agent-results", default=os.path.join(VB_DIR, "results", "E.json"))
    ap.add_argument("--query-cache", default=os.path.join(EXP_DIR, ".cache_queries.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="replace query encoding with deterministic random vectors: exercises the "
                         "whole ranking / stats path without a GPU. Numbers are meaningless.")
    args = ap.parse_args()

    pool = load_json(args.pool)
    arms = load_arms(args.emb_dir, args.arms)
    dim = next(iter(arms.values()))["emb"].shape[1]
    arm_names = list(arms)
    print(f"arms: {arm_names}  (dim {dim})")
    for n in arm_names:
        print(f"  {n}: {len(arms[n]['keys'])} clips")

    qmap, fell_back = question_queries(pool, args.query_source, args.agent_results)
    if fell_back:
        print(f"query source '{args.query_source}': {fell_back} questions fell back to the question text")
    all_texts = sorted({t for v in qmap.values() for t in v})
    qemb = encode_queries(all_texts, args.query_cache, args.self_test, dim)

    ts_end = {r["key"]: r["ts_end"] for r in pool["clips"]}
    ks = sorted(args.ks)

    # Neighbour tolerance. The benchmark tags exactly one 30-sec clip per question (the one holding
    # target_time), so an arm that ranks the clip right before the evidence first scores a miss.
    # Widening asks whether the arm landed in the right place at all. It is a DIFFERENT question,
    # not a fairer version of the same one: every extra clip admitted is another way to score a
    # hit, so recall rises even for a random ranker. The chance column below is what keeps it
    # honest -- read the arm against chance at its own tolerance, never against another one.
    tol = float(args.target_tolerance_sec)
    day_of = {r["key"]: r["date"] for r in pool["clips"]}
    start_of = {r["key"]: clip_start_sec(r["video_path"]) for r in pool["clips"]}

    def expand(target_keys: Set[str]) -> Set[str]:
        if tol <= 0:
            return set(target_keys)
        out = set(target_keys)
        anchors = [(day_of[t], start_of[t]) for t in target_keys
                   if t in day_of and start_of.get(t) is not None]
        if not anchors:
            return out
        for key, st in start_of.items():
            if st is None:
                continue
            d = day_of[key]
            if any(d == ad and abs(st - a) <= tol for ad, a in anchors):
                out.add(key)
        return out

    expanded = {q["ID"]: expand(set(q["target_keys"])) for q in pool["questions"]}
    if tol > 0:
        sizes = sorted(len(v) for v in expanded.values())
        print(f"\ntarget tolerance {tol:.0f}s: targets per question went "
              f"{sorted(len(set(q['target_keys'])) for q in pool['questions'])[len(sizes)//2]} -> "
              f"{sizes[len(sizes)//2]} (median), max {sizes[-1]}. Recall below is NOT comparable "
              f"to the strict table; use the chance column.")

    # hits[arm][k] -> {qid: 0/1};  rank[arm] -> {qid: best target rank or None}
    hits: Dict[str, Dict[int, Dict[str, int]]] = {n: {k: {} for k in ks} for n in arm_names}
    ranks: Dict[str, Dict[str, Optional[int]]] = {n: {} for n in arm_names}
    pool_sizes: Dict[str, Dict[str, int]] = {n: {} for n in arm_names}
    skipped: Dict[str, List[str]] = {n: [] for n in arm_names}

    for q in pool["questions"]:
        qid, qt = q["ID"], q["query_time"]
        targets = expanded[qid]
        qvecs = np.ascontiguousarray(np.stack([qemb[t] for t in qmap[qid]]).T)   # [d, nq]
        for n in arm_names:
            keys, emb = arms[n]["keys"], arms[n]["emb"]
            vis = [i for i, k in enumerate(keys) if ts_end[k] <= qt]
            vis_keys = [keys[i] for i in vis]
            tgt_present = targets & set(vis_keys)
            if not tgt_present or len(vis) < 2:
                skipped[n].append(qid)          # arm cannot be scored on this question
                continue
            pool_sizes[n][qid] = len(vis)
            sub = emb[vis]                       # [m, d]
            # some numpy/BLAS builds (2.0 + Accelerate) raise spurious divide/overflow warnings
            # here even for finite inputs; load_arms already checked the rows are finite.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = sub @ qvecs               # [m, nq]
            order_per_query = [np.argsort(-sims[:, j]) for j in range(sims.shape[1])]
            best = None
            for pos, i in enumerate(order_per_query[0], 1):
                if vis_keys[i] in tgt_present:
                    best = pos
                    break
            ranks[n][qid] = best
            for k in ks:
                topk = {vis_keys[i] for order in order_per_query for i in order[:k]}
                hits[n][k][qid] = int(bool(tgt_present & topk))

    # questions every arm could score -- the paired set
    common = sorted(set.intersection(*[set(hits[n][ks[0]]) for n in arm_names]), key=int)
    print(f"\nquestions: {len(pool['questions'])} | scorable in every arm: {len(common)}")
    for n in arm_names:
        if skipped[n]:
            print(f"  {n}: {len(skipped[n])} unscorable (target clip has no embedding in this arm, "
                  f"or nothing was visible yet)")

    def arr(n: str, k: int) -> np.ndarray:
        return np.array([hits[n][k][qid] for qid in common], dtype=np.float64)

    print(f"\nrecall@k over {len(common)} questions (union over the question's queries)")
    header = f"{'arm':>14} | " + " | ".join(f"@{k:<7}" for k in ks) + " | med.rank"
    print(header)
    print("-" * len(header))
    for n in arm_names:
        row = " | ".join(f"{100 * arr(n, k).mean():6.1f}%" for k in ks)
        rs = [r for r in (ranks[n].get(q) for q in common) if r]
        med = f"{int(np.median(rs))}" if rs else "-"
        print(f"{n:>14} | {row} | {med:>8}")

    # What a ranker that knows nothing would score on THIS question set, at THIS tolerance:
    # per question, the chance of landing a target in k draws from its own visible pool. Recall
    # only means something against this line -- a tolerance that admits more targets lifts both.
    ref = arm_names[0]
    n_tgt, n_vis = [], []
    for qid in common:
        vis = pool_sizes[ref].get(qid)
        if not vis:
            continue
        keys = set(arms[ref]["keys"])
        n_tgt.append(len(expanded[qid] & keys))
        n_vis.append(vis)
    def chance_at(t: int, v: int, k: int) -> float:
        """P(a random k-subset of v contains at least one of t targets).

        NOT k*t/v: that union bound double-counts the ways two targets both land in the k, and
        runs past 1 once k*t approaches v -- at tolerance 300s (t~21, k=50) it reported 52% where
        the truth is 45%, which is the difference between an arm looking below chance and above.
        """
        if t <= 0 or v <= 0:
            return 0.0
        if t >= v:
            return 1.0
        miss = 1.0
        for i in range(min(k, v)):
            num = v - t - i
            if num <= 0:
                return 1.0
            miss *= num / (v - i)
        return 1.0 - miss

    if n_vis:
        chance = [100 * float(np.mean([chance_at(t, v, k) for t, v in zip(n_tgt, n_vis)]))
                  for k in ks]
        # expected rank of the BEST of t targets under a random ranking is (v+1)/(t+1), not v/2 --
        # v/2 is the t=1 case and would flatter every tolerance above 0.
        mr_chance = float(np.median([(v + 1) / (t + 1) for t, v in zip(n_tgt, n_vis) if t > 0]))
        print(f"{'chance':>14} | " + " | ".join(f"{c:6.2f}%" for c in chance)
              + f" | {int(mr_chance):>8}")
        print(f"{'':>14}   (targets/question median {int(np.median(n_tgt))}, "
              f"visible pool median {int(np.median(n_vis))}, tolerance {tol:.0f}s)")

    mk = args.main_k if args.main_k in ks else ks[0]
    pairs = []
    print(f"\npaired comparisons at k={mk}  (win = only this arm found the target)")
    for i, a in enumerate(arm_names):
        for b in arm_names[i + 1:]:
            ha, hb = arr(a, mk), arr(b, mk)
            a_only = int(((ha == 1) & (hb == 0)).sum())
            b_only = int(((ha == 0) & (hb == 1)).sum())
            diff = 100 * (ha.mean() - hb.mean())
            lo, hi = bootstrap_ci(ha, hb)
            pairs.append({"a": a, "b": b, "k": mk, "diff_pp": diff,
                          "ci95_pp": [100 * lo, 100 * hi], "a_only": a_only, "b_only": b_only})
            print(f"  {a} - {b}: {diff:+5.1f}pp  95% CI [{100*lo:+5.1f}, {100*hi:+5.1f}]  "
                  f"({a} only {a_only}, {b} only {b_only})")

    # breakdowns at the main k
    def group(field: str) -> Dict[str, Dict[str, float]]:
        meta = {q["ID"]: q for q in pool["questions"]}
        out: Dict[str, Dict[str, float]] = {}
        vals = sorted({str(meta[q][field]) for q in common})
        for v in vals:
            qids = [q for q in common if str(meta[q][field]) == v]
            out[v] = {"n": len(qids),
                      **{n: 100 * float(np.mean([hits[n][mk][q] for q in qids])) for n in arm_names}}
        return out

    breakdowns = {f: group(f) for f in ("need_audio", "type", "gap_bin")}
    print(f"\nrecall@{mk} by need_audio / type / gap  (need_audio=True questions are the ones oracle "
          f"frames could not fix either)")
    for field, table in breakdowns.items():
        print(f"  {field}")
        for v, row in table.items():
            cells = "  ".join(f"{n} {row[n]:5.1f}%" for n in arm_names)
            print(f"    {v:>12} (n={row['n']:>3})  {cells}")

    src = args.query_source
    out_path = args.out or os.path.join(RESULTS_DIR, f"recall_{src}.json")
    md_path = args.markdown or os.path.join(ANALYSIS_DIR, f"recall_{src}.md")
    summary = {
        "pool": args.pool, "emb_dir": args.emb_dir, "query_source": src, "self_test": args.self_test,
        "n_questions": len(pool["questions"]), "n_scored": len(common), "ks": ks, "main_k": mk,
        "arms": {n: {"n_clips": len(arms[n]["keys"]),
                     "recall": {str(k): float(arr(n, k).mean()) for k in ks},
                     "median_rank": (lambda rs: int(np.median(rs)) if rs else None)(
                         [r for r in (ranks[n].get(q) for q in common) if r]),
                     "unscorable": skipped[n]} for n in arm_names},
        "pairs": pairs,
        "breakdowns": breakdowns,
        "per_question": [{"ID": q, "queries": qmap[q],
                          "pool_size": {n: pool_sizes[n].get(q) for n in arm_names},
                          "rank": {n: ranks[n].get(q) for n in arm_names},
                          f"hit@{mk}": {n: hits[n][mk].get(q) for n in arm_names}} for q in common],
    }
    save_json(summary, out_path)

    lines = [f"# Visual retrieval recall by crop arm (`--query-source {src}`)", ""]
    if args.self_test:
        lines += ["> **--self-test run: query embeddings are random. Numbers mean nothing.**", ""]
    lines += [f"Pool: {pool['n_clips']} clips ({pool['n_targets']} targets + {pool['n_distractors']} "
              f"distractors, scope `{pool['distractor_scope']}`), scored on {len(common)} of "
              f"{len(pool['questions'])} questions.", "",
              "| arm | " + " | ".join(f"R@{k}" for k in ks) + " | median rank |",
              "|---|" + "---|" * (len(ks) + 1)]
    for n in arm_names:
        rs = [r for r in (ranks[n].get(q) for q in common) if r]
        lines.append(f"| `{n}` | " + " | ".join(f"{100 * arr(n, k).mean():.1f}%" for k in ks) +
                     f" | {int(np.median(rs)) if rs else '-'} |")
    lines += ["", f"Paired differences at k={mk}:", "",
              "| arms | diff (pp) | 95% CI | a only | b only |", "|---|---|---|---|---|"]
    for p in pairs:
        lines.append(f"| `{p['a']}` - `{p['b']}` | {p['diff_pp']:+.1f} | "
                     f"[{p['ci95_pp'][0]:+.1f}, {p['ci95_pp'][1]:+.1f}] | {p['a_only']} | {p['b_only']} |")
    lines += ["", "How to read it: `center@R` is the control. `gaze@R` beating `full` but not "
                  "`center@R` means cropping helped and gaze did not.", ""]
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
