#!/usr/bin/env python3
"""
EgoLifeQA evaluation runner for the visual-bottleneck experiment.

Copy of eval/eval_egolife.py with a --condition branch. The original file and src/worldmm are left
as they are (except the visual-similarity frame bug fix in src/worldmm/memory/visual/memory.py).

Conditions
    A        question only
    B        question + text memory (WorldMM loop with visual disabled); caches text_context/{qid}.json
    B_replay question + B's cached text context, answered once (D without the frames). Same prompt as
             B's own final answer, so B's run counts as one B_replay sample
    C        question + oracle visual (frames/{qid})
    D        question + text memory (from B cache) + oracle visual
    E        original WorldMM loop (episodic + semantic + visual)
    E_prime  text memory (from B cache) + visual similarity top-k for the question text
    C1 / D1  same as C / D but with the fixed-16-frame oracle (frames16/{qid}); masking set only
    C2 / D2  relevant-region masked oracle   (masked_frames/relevant/{qid})
    C3 / D3  irrelevant-region masked oracle (masked_frames/irrelevant/{qid})

Usage (from repo root, all conditions):
    python experiments/visual_bottleneck/eval_egolife.py --condition B \
        --retriever-model qwen3vl-8b --respond-model qwen3vl-8b
Run B before D and E_prime (they read B's text_context cache).
"""

import argparse
import copy
import json
import logging
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from string import Template

from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult
from worldmm.memory.utils import MemorySearchOutput, ReasoningOutput, RetrievedItem
from worldmm.memory.visual.memory import _is_time_range_query

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CACHE_ROOT, FRAMES16_DIR, FRAMES_DIR, MASKED_FRAMES_DIR, METADATA_DIR, RESULTS_DIR, SUBSET_PATH,
    TEXT_CONTEXT_DIR, VIDEO_ROOT, build_choices, load_subset, resolve_video_path, subset_by_id,
)

CONDITIONS = ["A", "B", "B_replay", "C", "D", "E", "E_prime", "C1", "D1", "C2", "D2", "C3", "D3"]
TEXT_CONDITIONS = {"B", "B_replay", "D", "E", "E_prime", "D1", "D2", "D3"}
NEEDS_TEXT_CACHE = {"B_replay", "D", "E_prime", "D1", "D2", "D3"}
# Only these two actually run the retrieval loop over episodic memory, so only they depend on the
# index holding exactly the captions up to query_time. The rest read B's cached text context.
RETRIEVES_EPISODIC = {"B", "E"}
# Which conditions need index() at all. A, C, D and the masking variants answer from pre-extracted
# frames and B's cached text, so building the HippoRAG graph for them is pure waste -- and a real
# hazard: a graph.pickle left truncated by an earlier OOM kill crashes them on load even though they
# never read it. E_prime only queries visual memory, so it indexes that one alone.
NEEDS_FULL_INDEX = {"B", "E"}
NEEDS_VISUAL_INDEX = {"E_prime"}


# ---------------------------------------------------------------------------
# Unchanged helpers from eval/eval_egolife.py
# ---------------------------------------------------------------------------

def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])
    if pred_norm == gold_candidate:
        return True
    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True
    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}"),
    ]
    return pred_norm in full_patterns


# ---------------------------------------------------------------------------
# Experiment additions
# ---------------------------------------------------------------------------

# memory_reasoning template with the Visual memory type removed (used for condition B).
# Text is otherwise identical to src/worldmm/llm/templates/memory_reasoning.py.
MEMORY_REASONING_TEXT_ONLY = """
You are a reasoning agent for a video memory retrieval system.
Your job is to decide whether to stop and answer, or to search memory for more evidence.
When searching, you must select exactly one memory type and form a query.

# Decision Modes:
1. **search**: Retrieve memory to begin, continue, or extend progress toward the answer
   - Choose one memory type and form a keyword(phrase)-style search query.
2. **answer**: Stop searching because the accumulated results are sufficient.
   - No memory type selection is needed.

# Memory Types:
1. Episodic: Specific events/actions. Stores memories of past events and actions. Query by EVENT/ACTION.
2. Semantic: Entities/relationships. Stores factual knowledge about entities and their relationships, roles, and habits. Query by ENTITY/CONCEPT.

# Context Inputs:
- Current Query
- Round History: Log of past retrieval rounds. Each round is written in this format:

  ### Round N
  Decision: <search|answer>
  Memory: <episodic|semantic>
  Search Query: <query text>
  Retrieved:
  <retrieved items>

# STRICT OUTPUT RULES:
- Always decide **first**: "search" or "answer".
- If decision = "search": Must include "selected_memory" with exactly one memory type and one query.
- If decision = "answer": Do NOT include "selected_memory".
- Always output in valid JSON only, no extra commentary.

# Output Format:
{
 "decision": "search" | "answer",
 "selected_memory": {
   "memory_type": "episodic" | "semantic",
   "search_query": <str>
 } # Omit if decision = "answer"
}

# Few-shot Examples:
## Example 1
Query: Who gives the graduation gift to Maria?
Round History: []

### Response:
{
 "decision": "search",
 "selected_memory": {
   "memory_type": "episodic",
   "search_query": "Maria graduation gift giver"
 }
}

## Example 2
Query: Who gives the graduation gift to Maria?
Round History:
### Round 1
Decision: search
Memory: episodic
Search Query: Maria graduation gift giver
Retrieved:
[('Luis', 'hands', 'wrapped gift to Maria')]

### Response:
{
 "decision": "search",
 "selected_memory": {
   "memory_type": "semantic",
   "search_query": "Luis relation to Maria"
 }
}

## Example 3
Query: Who gives the graduation gift to Maria?
Round History:
### Round 1
Decision: search
Memory: episodic
Search Query: Maria graduation gift giver
Retrieved:
[('Luis', 'hands', 'wrapped gift to Maria')]

### Round 2
Decision: search
Memory: semantic
Search Query: Luis relation to Maria
Retrieved:
[('Luis', 'is brother of', 'Maria')]

### Response:
{
 "decision": "answer"
}
"""


def disable_visual_memory(world_memory: WorldMemory) -> None:
    """Condition B: the agent never sees the visual option, and a stray 'visual' decision returns nothing."""
    world_memory.prompt_template_manager.templates["memory_reasoning"] = [
        {"role": "system", "content": Template(MEMORY_REASONING_TEXT_ONLY)}
    ]
    world_memory.retrieve_from_visual = lambda query, top_k=None, retrieved_set=None: ({}, retrieved_set or set())


class VisualRetrievalRecorder:
    """Wrap WorldMemory.retrieve_from_visual to log which clips each visual search returned (condition E)."""

    def __init__(self, world_memory: WorldMemory):
        self.wm = world_memory
        self.orig = world_memory.retrieve_from_visual
        self.calls: List[Dict[str, Any]] = []
        world_memory.retrieve_from_visual = self._wrapped

    def _wrapped(self, query, top_k=None, retrieved_set=None):
        result, retrieved_set = self.orig(query, top_k=top_k, retrieved_set=retrieved_set)
        self.calls.append({
            "query": query,
            "kind": "time_range" if _is_time_range_query(query) else "text",
            "clips": list(result.keys()) if isinstance(result, dict) else [],
            "num_images": sum(len(v) for v in result.values()) if isinstance(result, dict) else 0,
        })
        return result, retrieved_set

    def reset(self) -> None:
        self.calls = []


class FilterRecorder:
    """Wrap EpisodicMemory._filter_with_llm to log what the multiscale filter saw and what it kept.

    Episodic retrieval hands the retriever LLM 10+5+5+3 = 23 HippoRAG candidates (one list per
    granularity) and asks for final_top_k of them; WorldMemory then drops already-seen captions and
    keeps episodic_top_k. Nothing about that step is logged, so a miss cannot be attributed: was the
    target never among the 23 (HippoRAG), or was it there and discarded (Qwen)? This records both
    sides per call. Only ids, times and scores are stored, not caption text, to keep results small.
    """

    def __init__(self, world_memory: WorldMemory):
        self.em = world_memory.episodic_memory
        self.orig = self.em._filter_with_llm
        self.calls: List[Dict[str, Any]] = []
        self.em._filter_with_llm = self._wrapped

    @staticmethod
    def _key(entry) -> str:
        return entry.to_display_str().split("]")[0].lstrip("[")   # "DAY1 11:34:30 - DAY1 11:34:59"

    def _wrapped(self, query, candidates, final_top_k):
        result = self.orig(query=query, candidates=candidates, final_top_k=final_top_k)
        self.calls.append({
            "query": query,
            "final_top_k": final_top_k,
            "candidates": [{"id": e.id, "granularity": e.granularity, "key": self._key(e), "score": float(sc)}
                           for e, sc in candidates],
            "selected": [{"id": e.id, "granularity": e.granularity, "key": self._key(e)} for e in result],
        })
        return result

    def reset(self) -> None:
        self.calls = []


def _span(key: str) -> Tuple[int, int]:
    """'DAY1 11:34:30 - DAY1 11:34:59' -> (start_sec, end_sec) on one absolute axis."""
    def sec(s: str) -> int:
        d, hms = s.split()
        h, m, sec_ = (int(x) for x in hms.split(":"))
        return int(d[3:]) * 86400 + h * 3600 + m * 60 + sec_
    a, b = key.split(" - ")
    return sec(a), sec(b)


def filter_recall(calls: List[Dict[str, Any]], target_keys: List[str]) -> Dict[str, Any]:
    """Did any target clip overlap a candidate the filter saw / a caption the filter kept?"""
    if not target_keys:
        return {"in_candidates": None, "in_selected": None}
    targets = [_span(k) for k in target_keys]
    def hit(key: str) -> bool:
        a, b = _span(key)
        return any(a <= t1 and t0 <= b for t0, t1 in targets)
    in_cand = any(hit(c["key"]) for call in calls for c in call["candidates"])
    in_sel = any(hit(c["key"]) for call in calls for c in call["selected"])
    return {"in_candidates": in_cand, "in_selected": in_sel}


def answer_with_context(
    world_memory: WorldMemory,
    query: str,
    choices: Optional[Dict[str, str]],
    text_items: List[RetrievedItem],
    images: List[Image.Image],
) -> str:
    """The answer-generation tail of WorldMemory.answer() (memory.py, 'Generate final answer' onward),
    with the retrieved context supplied directly instead of collected by the retrieval loop."""
    full_query = f"Query: {query}"
    if choices:
        choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
        full_query += f"\nChoices: {choices_str}"

    items = list(text_items)
    if images:
        items.append(RetrievedItem(memory_type="visual", content=list(images), query="", round_num=0))

    qa_prompt = world_memory.prompt_template_manager.render(world_memory.qa_template_name)
    qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
    qa_content.extend(world_memory._render_retrieved_items_for_qa(items))
    if choices:
        qa_content.append({
            "type": "text",
            "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
        })
    qa_messages = copy.deepcopy(qa_prompt)
    qa_messages.append({"role": "user", "content": qa_content})
    return world_memory.respond_llm_model.generate(qa_messages)


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frames(frames_root: str, qid: str) -> List[Image.Image]:
    d = os.path.join(frames_root, qid)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no cached frames for {qid} under {frames_root}")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [Image.open(os.path.join(d, f)).convert("RGB") for f in files]


def save_text_context(path: str, qa_result: QAResult) -> None:
    items = [
        {"memory_type": it.memory_type, "content": it.content, "query": it.query, "round_num": it.round_num}
        for it in qa_result.retrieved_items if it.memory_type in ("episodic", "semantic")
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"round_history": qa_result.round_history, "retrieved_items": items,
                   "num_rounds": qa_result.num_rounds}, f, indent=2, ensure_ascii=False)


def load_text_context(path: str) -> Tuple[List[RetrievedItem], Dict[str, Any]]:
    ctx = load_json(path)
    items = [RetrievedItem(memory_type=it["memory_type"], content=it["content"], query=it["query"],
                           round_num=it["round_num"]) for it in ctx["retrieved_items"]]
    return items, ctx


def reset_episodic_index(cache_root: str, granularities=("30sec", "3min", "10min", "1h")) -> None:
    """Delete HippoRAG's indexed state, keeping the OpenIE result cache.

    Why this is needed. HippoRAG.index() does:

        self.chunk_embedding_store.insert_strings(docs)          # captions up to query_time
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()   # EVERYTHING in the store

    and builds the graph from chunk_to_rows. The store is persisted to parquet, so whatever was
    indexed before is still there on the next run. In the original flow that is harmless: the cache
    starts empty and the store grows question by question, exactly in step with query_time. But once
    the store already holds the full week (because a previous condition finished, or because
    build_episodic_cache.py pre-built it), the very first question sees all seven days and future
    captions leak into retrieval.

    So the indexed state has to start empty for every run, which is what the original gets for free.
    The OpenIE results are a different kind of cache: they are keyed by caption text hash and carry
    no notion of time, so they are kept and reused (that is what saves the API calls).

    Removed:  {cache_root}/{granularity}/{llm}_{embedding}/   (vdb_*.parquet, graph.pickle)
    Kept:     {cache_root}/{granularity}/openie_results_*.json
    """
    removed = []
    for g in granularities:
        gdir = os.path.join(cache_root, g)
        if not os.path.isdir(gdir):
            continue
        for name in os.listdir(gdir):
            sub = os.path.join(gdir, name)
            if os.path.isdir(sub):          # the "{llm}_{embedding}" working dir
                shutil.rmtree(sub)
                removed.append(f"{g}/{name}")
    if removed:
        logger.info(f"Reset episodic index state ({len(removed)} dirs): {', '.join(removed)}")
    else:
        logger.info("Episodic index state already empty")


def frames_root_for(condition: str, args) -> Optional[str]:
    return {
        "C": args.frames_dir, "D": args.frames_dir,
        "C1": args.frames16_dir, "D1": args.frames16_dir,
        "C2": os.path.join(args.masked_dir, "relevant"), "D2": os.path.join(args.masked_dir, "relevant"),
        "C3": os.path.join(args.masked_dir, "irrelevant"), "D3": os.path.join(args.masked_dir, "irrelevant"),
    }.get(condition)


_JSON_DECODER = json.JSONDecoder()
_FENCE_RE = re.compile(r"```(?:json)?\s*|```", re.IGNORECASE)


def robust_parse_reasoning_response(response: str) -> ReasoningOutput:
    """Drop-in for WorldMemory._parse_reasoning_response (--robust-reasoning).

    The original does re.search(r'\{.*\}', DOTALL) and json.loads the match. That regex is greedy, so
    any second brace after the JSON (a trailing example, a repeated object, prose with braces) makes
    the match span both and json.loads fails; the original then silently returns "answer". This
    version tries progressively more naive readings and only gives up after all of them:

        1. strip ``` fences, then raw_decode from each "{" until one object parses
        2. regex the three fields out of whatever text there is
        3. fall back to "answer" as the original does, but log the response head so it is visible
    """
    text = _FENCE_RE.sub("", response or "").strip()

    # 1. first parseable JSON object anywhere in the text
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            data, _ = _JSON_DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "decision" in data:
            decision = str(data.get("decision", "answer")).lower().strip()
            selected = None
            mem = data.get("selected_memory")
            if decision == "search" and isinstance(mem, dict):
                selected = MemorySearchOutput(
                    memory_type=str(mem.get("memory_type", "")).lower().strip(),
                    search_query=str(mem.get("search_query", "")).strip(),
                )
            return ReasoningOutput(decision=decision, selected_memory=selected, reason=data.get("reason"))

    # 2. field-level regex, for truncated or otherwise broken JSON
    m_dec = re.search(r'"decision"\s*:\s*"(search|answer)"', text, re.IGNORECASE)
    if m_dec:
        decision = m_dec.group(1).lower()
        selected = None
        if decision == "search":
            m_type = re.search(r'"memory_type"\s*:\s*"(episodic|semantic|visual)"', text, re.IGNORECASE)
            m_q = re.search(r'"search_query"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if m_type and m_q and m_q.group(1).strip():
                selected = MemorySearchOutput(memory_type=m_type.group(1).lower(), search_query=m_q.group(1).strip())
        logger.warning(f"Reasoning JSON rescued by regex (decision={decision}): {text[:120]!r}")
        return ReasoningOutput(decision=decision, selected_memory=selected)

    # 3. original fallback, made visible
    logger.warning(f"Reasoning response unparseable, defaulting to answer: {text[:200]!r}")
    return ReasoningOutput(decision="answer")


# Patterns that name a choice in a long-form answer. Checked in order; within one pattern the LAST
# match wins, because a rambling answer ("I will pick C ... no ... I must pick B") ends on its decision.
_LENIENT_LETTER_PATTERNS = [
    re.compile(r"(?:final answer|answer is|answer:|i (?:will |must |would )?(?:pick|choose|select|go with))\s*(?:option\s*)?\**\(?([A-D])\)?\b", re.IGNORECASE),
    re.compile(r"\*\*\(?([A-D])\)?(?:[.:)]\s*[^*]{0,80})?\*\*"),          # **C** / **C. Dumplings**
    re.compile(r"(?:^|\n)\s*\(?([A-D])[.)]\s"),                               # a line starting "C. ..." / "(C) ..."
]


def extract_choice_letter_lenient(text: str, choices: Dict[str, str]) -> Optional[str]:
    """Letter extraction for --robust-reasoning.

    1. the original rule (a leading letter) so clean answers score identically
    2. explicit decision phrases, bold letters, or a line that starts with a letter: last match wins
    3. the text of exactly one choice appearing verbatim in the answer
    Returns None when nothing applies, which the caller scores as wrong (same as the original).
    """
    letter = extract_choice_letter(text)
    if letter in choices:
        return letter
    for pat in _LENIENT_LETTER_PATTERNS:
        found = [m.group(1).upper() for m in pat.finditer(text)]
        if found:
            return found[-1]
    low = text.lower()
    mentioned = [k for k, v in choices.items() if v and normalize(v) in low]
    return mentioned[0] if len(mentioned) == 1 else None


def evaluate_prediction_lenient(prediction: str, gold_letter: str, choices: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """Original scoring first; if that fails, the lenient extraction. Returns (correct, letter used)."""
    if evaluate_prediction(prediction, gold_letter, choices):
        return True, gold_letter
    letter = extract_choice_letter_lenient(prediction, choices)
    return letter == gold_letter, letter


def build_llms(args) -> Tuple[LLMModel, LLMModel]:
    """Retriever and responder. When both names match, load one model and share it (one 8B copy on GPU).

    Generation parameters are left exactly as the original eval script leaves them: nothing is passed,
    so the model's own generation_config applies. Forcing greedy decoding (do_sample=False) was tried
    and reverted; it sends Qwen into degenerate repetition on the long 1h captions, which then overruns
    max_new_tokens and fails JSON parsing. See PLAN.md section 6.
    """
    if args.retriever_model == args.respond_model:
        shared = LLMModel(model_name=args.respond_model, fps=1)
        logger.info(f"Sharing one LLM instance for retriever and responder: {args.respond_model}")
        return shared, shared
    retriever = LLMModel(model_name=args.retriever_model, fps=1)
    responder = LLMModel(model_name=args.respond_model, fps=1)
    return retriever, responder


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Evaluation with WorldMM (visual-bottleneck conditions)")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--retriever-model", type=str, default="qwen3vl-8b", help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="qwen3vl-8b", help="LLM model for iterative reasoning and generating answers")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top-k for visual retrieval")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    # experiment additions
    parser.add_argument("--condition", type=str, required=True, choices=CONDITIONS)
    parser.add_argument("--subset", type=str, default=SUBSET_PATH)
    parser.add_argument("--frames-dir", type=str, default=FRAMES_DIR)
    parser.add_argument("--frames16-dir", type=str, default=FRAMES16_DIR)
    parser.add_argument("--masked-dir", type=str, default=MASKED_FRAMES_DIR)
    parser.add_argument("--text-context-dir", type=str, default=TEXT_CONTEXT_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--episodic-cache-root", type=str, default=CACHE_ROOT,
                        help="HippoRAG cache built by build_episodic_cache.py. Absolute by default so "
                             "the cache is found no matter which directory the run starts from; "
                             "WorldMemory's own default is the relative '.cache/episodic_memory'.")
    parser.add_argument("--semantic-path", type=str, default=None,
                        help="default: output/metadata/semantic_memory/{subject}/semantic_consolidation_results_gpt-5-mini.json")
    parser.add_argument("--visual-path", type=str, default=None,
                        help="default: output/metadata/visual_memory/{subject}/visual_embeddings.pkl")
    parser.add_argument("--video-root", type=str, default=VIDEO_ROOT,
                        help="Directory holding A1_JAKE/DAY*/ mp4 files. The captions store repo-relative "
                             "paths ('data/EgoLife/A1_JAKE/...'); that prefix is replaced with this. "
                             "Only conditions E / E_prime read video files.")
    parser.add_argument("--keep-index-cache", action="store_true",
                        help="Do NOT clear HippoRAG's indexed state at startup (conditions B and E only). "
                             "Off by default because a store left over from a previous run leaks future "
                             "captions into the first questions; see reset_episodic_index(). The OpenIE "
                             "cache is kept either way.")
    parser.add_argument("--robust-reasoning", action="store_true",
                        help="Two lenient parsers, generation untouched. (a) Round-decision JSON: first "
                             "parseable object, then field regex, instead of greedy-regex + json.loads. "
                             "(b) Final answer: when the response is not a bare letter, take the last "
                             "explicitly chosen / bolded letter, else the single choice text quoted "
                             "verbatim. The raw response is still stored. Off = original behaviour.")
    parser.add_argument("--resume", action="store_true", help="skip IDs already present in the results file")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing results file (without --resume it is refused)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed the sampler per question (seed*1_000_003 + ID) so a resumed run draws the "
                             "same samples. Must be used with a --results-dir other than the default")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N subset questions (smoke test)")
    args = parser.parse_args()

    condition = args.condition
    subject = args.subject
    data_dir = args.data_dir
    semantic_path = args.semantic_path or os.path.join(METADATA_DIR, "semantic_memory", subject, "semantic_consolidation_results_gpt-5-mini.json")
    visual_path = args.visual_path or os.path.join(METADATA_DIR, "visual_memory", subject, "visual_embeddings.pkl")
    output_path = os.path.join(args.results_dir, f"{condition}.json")

    # ---- subset -------------------------------------------------------------
    subset = load_subset(args.subset)
    meta_by_id = subset_by_id(subset)
    wanted_ids = [e["ID"] for e in subset["entries"]]
    if condition in ("C1", "D1", "C2", "D2", "C3", "D3"):
        wanted_ids = [e["ID"] for e in subset["entries"] if e.get("masking")]
    if args.limit:
        wanted_ids = wanted_ids[:args.limit]

    if args.seed is not None and os.path.abspath(args.results_dir) == os.path.abspath(RESULTS_DIR):
        raise SystemExit(f"--seed writes a repeat run; give it its own --results-dir so {RESULTS_DIR} "
                         f"(the first run) is not touched")
    if os.path.exists(output_path) and not args.resume and not args.overwrite:
        raise SystemExit(f"{output_path} already exists. Use --resume to continue it or --overwrite to replace it")

    results: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and os.path.exists(output_path):
        results = load_json(output_path)
        done_ids = {str(r["ID"]) for r in results}
        logger.info(f"Resuming: {len(done_ids)} questions already in {output_path}")

    if condition in NEEDS_TEXT_CACHE:
        missing = [q for q in wanted_ids if not os.path.exists(os.path.join(args.text_context_dir, f"{q}.json"))]
        if missing:
            raise FileNotFoundError(f"condition {condition} needs B's text_context for {len(missing)} questions "
                                    f"(e.g. {missing[:3]}); run --condition B first")

    # ---- episodic index state -----------------------------------------------
    # Start from an empty index so the graph grows with query_time, as it does on a fresh run of the
    # original script. Only B and E search episodic memory; every other condition either ignores text
    # memory or replays B's cached context, so clearing for them would just cost a re-embedding pass.
    if condition in RETRIEVES_EPISODIC and not args.keep_index_cache:
        reset_episodic_index(args.episodic_cache_root)

    # ---- models -------------------------------------------------------------
    # The episodic cache must already hold all four granularities (build_episodic_cache.py).
    # Without it the retriever model would run OpenIE mid-evaluation, which is both off-spec and slow.
    missing_cache = [g for g in ("30sec", "3min", "10min", "1h")
                     if not os.path.exists(os.path.join(args.episodic_cache_root, g,
                                                        "openie_results_ner_gpt-5-mini.json"))]
    if missing_cache and condition in TEXT_CONDITIONS:
        logger.warning(f"Episodic cache missing for {missing_cache} under {args.episodic_cache_root}. "
                       f"OpenIE will run during evaluation with --retriever-model {args.retriever_model}. "
                       f"Run build_episodic_cache.py first.")

    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm_model, respond_llm_model = build_llms(args)
    prompt_template_manager = PromptTemplateManager()

    logger.info("Initializing WorldMemory...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        episodic_cache_root=args.episodic_cache_root,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )
    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    if args.robust_reasoning:
        world_memory._parse_reasoning_response = robust_parse_reasoning_response
        logger.info("robust reasoning: lenient round-decision JSON parsing + lenient answer-letter extraction "
                    "(generation params unchanged)")

    if condition == "B":
        disable_visual_memory(world_memory)
    recorder = VisualRetrievalRecorder(world_memory) if condition == "E" else None
    filter_rec = FilterRecorder(world_memory) if condition in RETRIEVES_EPISODIC else None

    # ---- data ---------------------------------------------------------------
    logger.info("Loading data...")
    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    eval_data = [row for row in load_json(eval_data_path) if str(row["ID"]) in set(wanted_ids)]
    eval_data.sort(key=lambda r: wanted_ids.index(str(r["ID"])))

    episodic_caption_dir = os.path.join(data_dir, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json") for g in granularities}
    episodic_captions_30sec = load_json(episodic_caption_files["30sec"])
    semantic_results = load_json(semantic_path)

    logger.info("Loading data into WorldMemory...")
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    world_memory.load_semantic_triples(data=semantic_results)
    # Load first, remap second. visual_embeddings.pkl is keyed by the ORIGINAL repo-relative video
    # path, so load_visual_clips() must see the captions unchanged or every embedding lookup misses
    # and VisualMemory ends up with nothing to index. Only after the lookup is done do we point the
    # stored clip paths at the real video directory, which is what frame extraction opens.
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    n_remapped = 0
    for clip in world_memory.visual_memory.clips:
        resolved = resolve_video_path(clip.video_path, args.video_root)
        if resolved != clip.video_path:
            clip.video_path = resolved
            n_remapped += 1
    with_emb = sum(1 for c in world_memory.visual_memory.clips if c.embedding is not None)
    logger.info(f"Video root: {args.video_root} ({n_remapped} clip paths remapped, "
                f"{with_emb}/{len(world_memory.visual_memory.clips)} clips have embeddings)")
    if with_emb == 0:
        logger.error("No clip has an embedding. Visual retrieval will return nothing. "
                     "Check that visual_embeddings.pkl matches the caption video_path values.")
    if condition in ("E", "E_prime"):
        sample = next((c.video_path for c in world_memory.visual_memory.clips if c.video_path), None)
        if sample and not os.path.exists(sample):
            logger.warning(f"Video file not found: {sample}. Frame extraction will return nothing. "
                           f"Check --video-root.")

    frames_root = frames_root_for(condition, args)

    # ---- evaluation loop ----------------------------------------------------
    logger.info(f"Condition {condition}: evaluating {len(eval_data)} questions ...")
    evaluate_true = sum(int(r["evaluate"]) for r in results)

    for row in tqdm(eval_data):
        ID = str(row['ID'])
        if ID in done_ids:
            continue
        meta = meta_by_id[ID]
        question = row['question']
        answer = row['answer']
        choices = build_choices(row)
        query_time = int(row['query_time']["date"][-1] + row['query_time']["time"].zfill(8))

        logger.info(f"Processing ID {ID}: {question[:50]}...")
        if args.seed is not None:
            seed_everything(args.seed * 1_000_003 + int(ID))

        # Index only what this condition actually reads (see NEEDS_FULL_INDEX above).
        if condition in NEEDS_FULL_INDEX:
            world_memory.index(query_time)
        elif condition in NEEDS_VISUAL_INDEX:
            world_memory.visual_memory.index(query_time)

        qa_result: Optional[QAResult] = None
        text_ctx: Dict[str, Any] = {}
        images: List[Image.Image] = []
        retrieved_clips: List[str] = []
        n_images = 0
        visual_used = False
        visual_query_kind: Optional[str] = None
        if recorder:
            recorder.reset()
        if filter_rec:
            filter_rec.reset()

        try:
            if condition == "A":
                response = answer_with_context(world_memory, question, choices, [], [])

            elif condition in ("B", "E"):
                qa_result = world_memory.answer(query=question, choices=choices, until_time=query_time)
                response = qa_result.answer
                if condition == "B":
                    save_text_context(os.path.join(args.text_context_dir, f"{ID}.json"), qa_result)
                else:
                    visual_used = any(r.get("memory_type") == "visual" for r in qa_result.round_history)
                    if recorder and recorder.calls:
                        visual_query_kind = recorder.calls[0]["kind"]
                        retrieved_clips = [c for call in recorder.calls for c in call["clips"]]
                        # answer() consumes the images internally, so count them from the recorder
                        n_images = sum(call["num_images"] for call in recorder.calls)

            elif condition == "B_replay":
                text_items, text_ctx = load_text_context(os.path.join(args.text_context_dir, f"{ID}.json"))
                response = answer_with_context(world_memory, question, choices, text_items, [])

            elif condition in ("C", "C1", "C2", "C3"):
                images = load_frames(frames_root, ID)
                response = answer_with_context(world_memory, question, choices, [], images)

            elif condition in ("D", "D1", "D2", "D3"):
                text_items, text_ctx = load_text_context(os.path.join(args.text_context_dir, f"{ID}.json"))
                images = load_frames(frames_root, ID)
                response = answer_with_context(world_memory, question, choices, text_items, images)

            elif condition == "E_prime":
                text_items, text_ctx = load_text_context(os.path.join(args.text_context_dir, f"{ID}.json"))
                vis = world_memory.visual_memory.retrieve(query=question, top_k=args.visual_top_k, as_context=True)
                if isinstance(vis, dict):
                    retrieved_clips = list(vis.keys())
                    images = [im for ims in vis.values() for im in ims]
                visual_used = bool(images)
                visual_query_kind = "text"
                response = answer_with_context(world_memory, question, choices, text_items, images)

            else:
                raise ValueError(condition)

        except Exception as e:
            logger.error(f"Error processing ID {ID}: {e}")
            response = "Error"

        if args.robust_reasoning:
            evaluate, pred_letter = evaluate_prediction_lenient(response, answer, choices)
            if len(response.strip()) > 3:
                logger.info(f"ID {ID} long-form answer ({len(response)} chars) -> lenient letter {pred_letter}")
        else:
            evaluate = evaluate_prediction(response, answer, choices)
            pred_letter = None
        evaluate_true += int(evaluate)

        frec = filter_recall(filter_rec.calls, meta.get("target_clip_keys", [])) if filter_rec else {}
        if filter_rec and filter_rec.calls:
            n_cand = sum(len(c["candidates"]) for c in filter_rec.calls)
            n_sel = sum(len(c["selected"]) for c in filter_rec.calls)
            logger.info(f"ID {ID} filter: {len(filter_rec.calls)} calls, {n_cand} candidates -> {n_sel} kept; "
                        f"target in candidates={frec.get('in_candidates')} in selected={frec.get('in_selected')}")

        result_entry = {
            "ID": ID,
            "condition": condition,
            "type": row['type'],
            "need_audio": meta["need_audio"],
            "gap": meta["gap"],
            "gap_bin": meta["gap_bin"],
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "evaluate": evaluate,
            "pred_letter_lenient": pred_letter,
            "query_time": query_time,
            "target_clips": meta["target_clips"],
            "target_clip_keys": meta["target_clip_keys"],
            "num_frames": len(images) or n_images,
            "num_rounds": qa_result.num_rounds if qa_result else text_ctx.get("num_rounds", 0),
            "round_history": qa_result.round_history if qa_result else text_ctx.get("round_history", []),
            "visual_used": visual_used,
            "visual_query_kind": visual_query_kind,
            "retrieved_clips": retrieved_clips,
            "visual_calls": recorder.calls if recorder else [],
            "filter_calls": filter_rec.calls if filter_rec else [],
            "target_in_filter_candidates": frec.get("in_candidates"),
            "target_in_filter_selected": frec.get("in_selected"),
        }
        results.append(result_entry)

        logger.info(
            f"ID {ID} Answer: {response}, Gold: {answer}, Correct: {evaluate} "
            f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true/len(results):.4f}"
        )

        # write incrementally so a crash keeps progress (and --resume can continue)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Complete  [condition {condition}]")
    logger.info(f"Subject: {subject}")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*50}")

    world_memory.cleanup()


if __name__ == "__main__":
    main()
