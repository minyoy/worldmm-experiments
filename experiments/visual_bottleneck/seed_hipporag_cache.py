#!/usr/bin/env python3
"""
Seed the HippoRAG OpenIE cache for the 30-sec granularity from the prebuilt
output/metadata OpenIE results, so evaluation does not re-run OpenIE on the
6,223 thirty-second captions.

Prebuilt format (output/metadata/episodic_memory/A1_JAKE/openie_results_gpt-5-mini.json):
    {"ner_results": {chunk-<md5>: [entity, ...]}, "triple_results": {chunk-<md5>: [[s, p, o], ...]}}
HippoRAG cache format (.cache/episodic_memory/30sec/openie_results_ner_gpt-5-mini.json):
    {"docs": [{"idx": chunk-<md5>, "passage": text, "extracted_entities": [...], "extracted_triples": [...]}],
     "avg_ent_chars": float, "avg_ent_words": float}

The chunk id is md5(passage) with a "chunk-" prefix in both, so the join key is the caption text.
The file name is fixed by HippoRAG's default config (llm_name=gpt-5-mini) regardless of the
retriever model passed at eval time.

Usage (from repo root):
    python experiments/visual_bottleneck/seed_hipporag_cache.py [--force]
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE_ROOT, CAPTION_30SEC, METADATA_DIR, SUBJECT, load_json, save_json  # noqa: E402

CACHE_LLM_LABEL = "gpt-5-mini"  # HippoRAG BaseConfig.llm_name default


def chunk_id(text: str) -> str:
    return "chunk-" + hashlib.md5(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--openie", default=os.path.join(METADATA_DIR, "episodic_memory", SUBJECT, "openie_results_gpt-5-mini.json"))
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--granularity", default="30sec")
    ap.add_argument("--force", action="store_true", help="overwrite an existing cache file instead of merging")
    args = ap.parse_args()

    caps = load_json(args.captions)
    oie = load_json(args.openie)
    ner, triples = oie["ner_results"], oie["triple_results"]

    docs, missing = [], []
    for c in caps:
        cid = chunk_id(c["text"])
        if cid not in triples:
            missing.append(cid)
            continue
        docs.append({
            "idx": cid,
            "passage": c["text"],
            "extracted_entities": list(ner.get(cid, [])),
            "extracted_triples": [list(t) for t in triples[cid]],
        })
    print(f"captions: {len(caps)} | matched: {len(docs)} | missing: {len(missing)}")
    if missing:
        print("WARNING: some captions have no prebuilt OpenIE; HippoRAG will run OpenIE for them at eval time")

    out_path = os.path.join(args.cache_root, args.granularity, f"openie_results_ner_{CACHE_LLM_LABEL}.json")
    if os.path.exists(out_path) and not args.force:
        existing = load_json(out_path).get("docs", [])
        have = {d["idx"] for d in existing}
        added = [d for d in docs if d["idx"] not in have]
        docs = existing + added
        print(f"existing cache has {len(existing)} docs; adding {len(added)}")

    ents = [e for d in docs for e in d["extracted_entities"]]
    n = len(ents)
    payload = {
        "docs": docs,
        "avg_ent_chars": round(sum(len(e) for e in ents) / n, 4) if n else 0,
        "avg_ent_words": round(sum(len(e.split()) for e in ents) / n, 4) if n else 0,
    }
    save_json(payload, out_path, indent=None)
    print(f"wrote {len(docs)} docs -> {out_path}")


if __name__ == "__main__":
    main()
