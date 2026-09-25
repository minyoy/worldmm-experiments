#!/usr/bin/env python3
"""
Is the 30-sec retrieval miss HippoRAG's fault or the embedding's?

Condition E delivers the target 30-sec caption into the multiscale filter's candidate list for only
~40% of questions, using the authors' own OpenIE cache. This script replays the agent's own episodic
queries against the same 30-sec captions with the same text encoder, but with NOTHING else: no OpenIE,
no entity/fact graph, no PPR, no LLM filter. Plain cosine similarity, top-k.

    DPR-only recall  >  HippoRAG recall   ->  the graph/PPR stage is losing the target
    DPR-only recall ~=  HippoRAG recall   ->  the encoder (or the query) is the ceiling
    both low, but recall@50 high          ->  the target is reachable, top-k is just too small

Everything comes from a finished run's results file, so no re-evaluation is needed. Caption embeddings
are cached to .npy, so only the first run pays for encoding 6,223 captions.

Usage (from repo root, needs the GPU for the encoder):
    python experiments/visual_bottleneck/dpr_recall.py
    python experiments/visual_bottleneck/dpr_recall.py --results experiments/visual_bottleneck/results/B.json
    python experiments/visual_bottleneck/dpr_recall.py --dry-run     # no GPU: data sanity + HippoRAG side only
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ANALYSIS_DIR, CAPTION_30SEC, EXP_DIR, clip_display_key, load_json, save_json, ts_int,
)

# HippoRAG's per-granularity candidate count for 30sec (EpisodicMemory.retrieve default).
HIPPORAG_30SEC_TOP_K = 10
DEFAULT_KS = (1, 3, 5, 10, 20, 50, 100)


def load_questions(results_path: str) -> List[Dict[str, Any]]:
    """Per question: the agent's episodic queries, its query_time, and its target caption keys.

    filter_calls is written by FilterRecorder in eval_egolife.py, one entry per multiscale-filter call,
    i.e. one per episodic search the agent performed. Questions that never searched episodic memory
    (semantic only, or answered straight away) carry no filter_calls and are skipped: there is no query
    to replay for them.
    """
    out = []
    for r in load_json(results_path):
        calls = r.get("filter_calls") or []
        if not calls or not r.get("target_clip_keys"):
            continue
        seen, queries = set(), []
        for c in calls:
            q = (c.get("query") or "").strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
        if not queries:
            continue
        out.append({
            "ID": r["ID"],
            "type": r.get("type"),
            "queries": queries,
            "question": r.get("question", ""),
            "query_time": r["query_time"],
            "targets": set(r["target_clip_keys"]),
            "evaluate": r.get("evaluate"),
            # what HippoRAG actually delivered at the 30-sec scale, for the side-by-side
            "hippo_30sec": {x["key"] for c in calls for x in c["candidates"] if x["granularity"] == "30sec"},
        })
    return out


def load_captions(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """30-sec captions as (indexed text, display key, end timestamp), in file order."""
    caps = load_json(path)
    texts = [c["text"] for c in caps]          # the exact string EpisodicMemory hands to HippoRAG
    keys = [clip_display_key(c) for c in caps]
    ends = np.array([ts_int(c["date"], c["end_time"]) for c in caps], dtype=np.int64)
    return texts, keys, ends


def cached_captions(texts: List[str], cache_path: str) -> np.ndarray:
    """Reuse a previous run's caption embeddings when they match the caption count."""
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            print(f"caption embeddings: loaded {emb.shape} from {cache_path}")
            return emb
        print(f"caption embeddings: cache has {emb.shape[0]} rows for {len(texts)} captions, re-encoding")
    return None


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(EXP_DIR, "results_robust", "E.json"),
                    help="a finished run's results file; must contain filter_calls")
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--emb-cache", default=os.path.join(EXP_DIR, ".cache_dpr", "captions_30sec.npy"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--out", default=None, help="default: analysis/dpr_recall_<query-source>.json")
    ap.add_argument("--query-source", choices=("agent", "question", "oracle"), default="agent",
                    help="What to retrieve with. 'agent': the queries the 8B actually issued (default). "
                         "'question': the raw question text, to test whether the agent's rewriting hurts. "
                         "'oracle': the target caption's own text -- a ceiling check, should be near 100%%; "
                         "anything less means the encoder cannot even match a caption to itself in context.")
    ap.add_argument("--dry-run", action="store_true", help="skip encoding; report data sanity and the HippoRAG side")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(ANALYSIS_DIR, f"dpr_recall_{args.query_source}.json")

    qs = load_questions(args.results)
    texts, keys, ends = load_captions(args.captions)
    key_set = set(keys)
    key_to_text = dict(zip(keys, texts))

    if args.query_source == "question":
        for item in qs:
            item["queries"] = [item["question"]] if item["question"] else item["queries"]
    elif args.query_source == "oracle":
        for item in qs:
            oracle = [key_to_text[k] for k in sorted(item["targets"]) if k in key_to_text]
            if oracle:
                item["queries"] = oracle
    print(f"query source: {args.query_source}")
    print(f"questions with episodic queries: {len(qs)}   30-sec captions: {len(texts)}")

    # Targets are 30-sec caption entries (common.target_clips), so they must all be known keys.
    unknown = {k for q in qs for k in q["targets"] if k not in key_set}
    if unknown:
        print(f"WARNING: {len(unknown)} target keys are not 30-sec caption keys, e.g. {sorted(unknown)[:3]}")

    hippo_hit = sum(1 for q in qs if q["targets"] & q["hippo_30sec"])
    n = len(qs)
    print(f"\nHippoRAG, 30-sec scale, top-{HIPPORAG_30SEC_TOP_K} per query, union over the question's "
          f"queries: {hippo_hit}/{n} = {100 * hippo_hit / n:.1f}%")
    if args.dry_run:
        qc = [len(q["queries"]) for q in qs]
        print(f"[dry run] distinct queries per question: mean {sum(qc) / len(qc):.2f}, max {max(qc)}")
        print("[dry run] stopping before the encoder")
        return

    caption_emb = cached_captions(texts, args.emb_cache)
    from worldmm.embedding import EmbeddingModel  # imported late so --dry-run needs no GPU
    model = EmbeddingModel()
    if caption_emb is None:
        print(f"encoding {len(texts)} captions ...")
        caption_emb = np.asarray(model.encode_text(texts, batch_size=args.batch_size), dtype=np.float32)
        os.makedirs(os.path.dirname(args.emb_cache), exist_ok=True)
        np.save(args.emb_cache, caption_emb)
        print(f"caption embeddings: {caption_emb.shape} saved to {args.emb_cache}")
    caption_emb = unit(caption_emb.astype(np.float32))

    all_queries = sorted({q for item in qs for q in item["queries"]})
    print(f"encoding {len(all_queries)} distinct queries ...")
    q_emb = unit(np.asarray(model.encode_text(all_queries, batch_size=args.batch_size), dtype=np.float32))
    q_index = {q: i for i, q in enumerate(all_queries)}

    recall = {k: 0 for k in args.ks}
    recall_first = {k: 0 for k in args.ks}      # round-1 query only, no union
    best_rank: List[int] = []
    per_question: List[Dict[str, Any]] = []

    for item in qs:
        # Only captions the agent could have seen at this question's query time.
        visible = np.flatnonzero(ends <= item["query_time"])
        sims_by_query = []
        for q in item["queries"]:
            s = caption_emb[visible] @ q_emb[q_index[q]]
            order = visible[np.argsort(-s)]
            sims_by_query.append(order)

        # rank of the best-ranked target under the first query, for the "how far down is it" picture
        first = sims_by_query[0]
        rank = next((i + 1 for i, idx in enumerate(first) if keys[idx] in item["targets"]), None)
        if rank:
            best_rank.append(rank)

        hit_at = {}
        for k in args.ks:
            union = {keys[i] for order in sims_by_query for i in order[:k]}
            hit_at[k] = bool(item["targets"] & union)
            recall[k] += hit_at[k]
            recall_first[k] += bool(item["targets"] & {keys[i] for i in first[:k]})

        per_question.append({
            "ID": item["ID"], "type": item["type"], "evaluate": item["evaluate"],
            "n_queries": len(item["queries"]), "visible_captions": int(visible.size),
            "first_query_target_rank": rank,
            "hippo_hit": bool(item["targets"] & item["hippo_30sec"]),
            "dpr_hit": {str(k): hit_at[k] for k in args.ks},
        })

    pct = lambda x: f"{x}/{n} = {100 * x / n:.1f}%"
    print(f"\nDPR only (cosine, same encoder, no graph / no LLM filter), 30-sec scale")
    print(f"{'k':>5} | {'union over queries':>20} | {'first query only':>18}")
    print("-" * 50)
    for k in args.ks:
        print(f"{k:>5} | {pct(recall[k]):>20} | {pct(recall_first[k]):>18}")

    same_k = HIPPORAG_30SEC_TOP_K
    if same_k in recall:
        d = 100 * (recall[same_k] - hippo_hit) / n
        print(f"\nat the same k={same_k}: DPR {100 * recall[same_k] / n:.1f}% vs HippoRAG "
              f"{100 * hippo_hit / n:.1f}%  ({d:+.1f}%p)")
        both = sum(1 for p in per_question if p["hippo_hit"] and p["dpr_hit"][str(same_k)])
        only_d = sum(1 for p in per_question if not p["hippo_hit"] and p["dpr_hit"][str(same_k)])
        only_h = sum(1 for p in per_question if p["hippo_hit"] and not p["dpr_hit"][str(same_k)])
        print(f"  both {both} | DPR only {only_d} | HippoRAG only {only_h} | neither {n - both - only_d - only_h}")

    if best_rank:
        br = np.array(best_rank)
        print(f"\nrank of the target under the first query (found in {len(br)}/{n}): "
              f"median {int(np.median(br))}, p25 {int(np.percentile(br, 25))}, p75 {int(np.percentile(br, 75))}")

    summary = {
        "results_file": args.results, "query_source": args.query_source, "n_questions": n,
        "hipporag_30sec_top10": hippo_hit,
        "dpr_recall_union": {str(k): recall[k] for k in args.ks},
        "dpr_recall_first_query": {str(k): recall_first[k] for k in args.ks},
        "per_question": per_question,
    }
    save_json(summary, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
