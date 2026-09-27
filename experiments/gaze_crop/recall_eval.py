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
    ANALYSIS_DIR, EMB_DIR, EXP_DIR, POOL_PATH, RESULTS_DIR, VB_DIR, clip_span,
    exact_paired_p, load_json, save_json, target_spans, unit,
)

DEFAULT_KS = (1, 3, 5, 10, 20, 50)
MAIN_K = 3  # WorldMM's visual top-k, and what condition E' actually consumed


def arm_name_from_file(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_arms(emb_dir: str, wanted: Optional[List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
    """emb/*.npz 를 읽어 arm 별 클립 벡터를 올린다.

    입력:  emb_dir  "experiments/gaze_crop/emb"
           wanted   ["full", "gazef@0.5"] 또는 None(=emb/ 안의 모든 npz)
                    "gazef@0.5" 와 파일명 "gazef_05.npz" 를 서로 맞춰준다
    출력:  {"full": {"keys": [클립key, ...], "emb": (N, d) float32 배열}, ...}
           emb 는 unit() 으로 L2 정규화돼 있다. 그래서 나중에 내적 하나가 코사인 유사도가 된다.

    디버깅 팁: 0벡터나 NaN 행이 있으면 WARNING 을 찍는다. 그게 뜨면 임베딩 단계가
    일부 클립에서 실패한 것이므로 recall 을 믿지 말 것.
    6,223 규모로 임베딩된 arm 은 full 과 gazef@0.5 뿐이다(2단계에서 그 둘만 돌렸다).
    """
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
    """조건 E 결과 파일에서 '에이전트가 실제로 던진 검색어' 를 꺼낸다.

    입력:  results_path  visual_bottleneck/results/E.json
    출력:  {질문ID: [검색어, ...]}  중복 제거. 파일이 없으면 {}
    --query-source agent 일 때만 쓴다.
    """
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
    """검색어 텍스트를 클립 벡터와 같은 공간으로 임베딩한다.

    입력:  texts       ["Who plans to grow flowers", ...]  (중복 제거된 전체 목록)
           cache_path  .cache_queries.npz  -- 있으면 재사용, 없는 것만 새로 인코딩
           self_test   True 면 GPU 없이 해시 기반 난수 벡터 (숫자는 무의미, 배선 점검용)
           dim         클립 벡터 차원. self_test 때만 쓴다
    출력:  {텍스트: 정규화된 벡터}

    ★ model.encode_vis_query 를 쓰는 게 핵심. 실제 시스템의
      VisualMemory._retrieve_by_similarity 와 같은 호출이어야 측정이 의미가 있다.
      클립 쪽 encode_video 와 짝이 맞는 텍스트 인코더다.
    """
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

    # ------------------------------------------------------------------
    # 정답 클립 판정: 주석된 시점(target_time) 기준, 파생된 클립 기준이 아니다
    # ------------------------------------------------------------------
    # EgoLifeQA 원본이 주는 건 target_time 뿐이다. "어느 클립이 정답인가" 는
    # common.py 의 target_clips() 가 "시점이 클립 구간 안에 들어가는가" 로 만든 파생물이고,
    # 실측해보면 두 군데가 깨진다.
    #   - 주석 시점이 클립 경계 3초 이내인 문항이 19.2% -> 격자가 조금만 달랐어도 옆 클립이 정답
    #   - 클립 사이 틈(중앙값 3초)에 떨어져 정답이 0개가 되는 문항 13개
    # 그래서 여기서는 "클립 구간이 target_time ±tol 에 걸치는가" 로 판정한다.
    #   tol=0  -> 기존 엄격 정의와 동일 (500문항 중 497개 일치, 나머지 3개는 위 연쇄 버그)
    #   tol>0  -> '정확히 맞혔나' 가 아니라 '근처에 갔나' 라는 다른 질문. 정답 칸이 늘어나
    #             recall 이 저절로 오르므로 아래 chance 행과 같이 읽어야 한다.
    tol = float(args.target_tolerance_sec)

    # spans:      {질문ID: [(시작초, 끝초), ...]}   지금 데이터는 항상 시작==끝 (한 시점)
    # clip_spans: {클립key: (시작초, 끝초)}          풀 전체 6,223개
    spans = {q["ID"]: target_spans(q.get("target_time"), (q.get("query_time") or {}).get("date"))
             for q in pool["questions"]}
    clip_spans = {r["key"]: clip_span(r) for r in pool["clips"]}

    def expand(qid: str, target_keys: Set[str]) -> Set[str]:
        """질문 하나의 정답 클립 key 집합을 만든다.

        입력:  qid          "6"
               target_keys  build_pool 이 넣어둔 파생 정답 {"DAY1_A1_JAKE_11343000"}
        출력:  정답으로 칠 클립 key 집합
               tol=0  -> {"DAY1_A1_JAKE_11343000"}           1칸
               tol=30 -> {"...11340000", "...11343000", "...11350000"}   3칸

        디버깅 팁: 반환 크기가 이상하면 spans[qid] 를 먼저 찍어볼 것.
        시점이 2~3개로 쪼개져 있어야 정상이고, 하나인데 수백 칸이 나오면 tol 이 너무 크다.
        """
        sp = spans.get(qid) or []
        if not sp:
            # target_time 이 없는 옛날 pool 파일 -> 기존 파생 정답으로 폴백
            return set(target_keys)
        out = set()
        for key, (cs, ce) in clip_spans.items():
            # 두 구간 [cs,ce] 와 [a-tol, b+tol] 이 겹치는가 (양끝 포함)
            if any(cs <= b + tol and ce >= a - tol for a, b in sp):
                out.add(key)
        # ★ target_keys 를 합집합으로 얹지 않는다.
        #   그건 common.py 의 target_clips() 가 만든 값이고, 연쇄 타임스탬프를 구간으로
        #   잘못 읽어 ID 279 를 2,430칸으로 부풀린다(16문항, 전부 held-out 380 쪽).
        #   spans 가 잡히면 그게 주석에 더 충실하므로 그것만 쓴다.
        return out

    expanded = {q["ID"]: expand(q["ID"], set(q["target_keys"])) for q in pool["questions"]}
    if tol > 0:
        sizes = sorted(len(v) for v in expanded.values())
        print(f"\ntarget tolerance {tol:.0f}s: targets per question went "
              f"{sorted(len(set(q['target_keys'])) for q in pool['questions'])[len(sizes)//2]} -> "
              f"{sizes[len(sizes)//2]} (median), max {sizes[-1]}. Recall below is NOT comparable "
              f"to the strict table; use the chance column.")

    # ------------------------------------------------------------------
    # 채점 루프: 질문 x arm 마다 클립을 유사도로 줄 세우고 정답 순위를 기록
    # ------------------------------------------------------------------
    # 여기서 채워지는 4개가 이후 모든 표의 원재료다. 디버깅할 때 먼저 볼 것.
    #   hits[arm][k][qid]  0/1   정답이 상위 k 안에 들어왔나
    #   ranks[arm][qid]    int   정답이 처음 나온 순위(1부터). 없으면 None
    #   pool_sizes[arm][qid] int 그 질문이 실제로 뒤진 후보 수(6,223 이 아니다!)
    #   skipped[arm]       list  채점 불가 질문 ID
    hits: Dict[str, Dict[int, Dict[str, int]]] = {n: {k: {} for k in ks} for n in arm_names}
    ranks: Dict[str, Dict[str, Optional[int]]] = {n: {} for n in arm_names}
    pool_sizes: Dict[str, Dict[str, int]] = {n: {} for n in arm_names}
    skipped: Dict[str, List[str]] = {n: [] for n in arm_names}

    for q in pool["questions"]:
        qid, qt = q["ID"], q["query_time"]
        targets = expanded[qid]                  # 위에서 만든 정답 클립 key 집합
        qvecs = np.ascontiguousarray(np.stack([qemb[t] for t in qmap[qid]]).T)   # [d, nq]
        for n in arm_names:
            keys, emb = arms[n]["keys"], arms[n]["emb"]
            # ★ 미래 차단: 질문 시각 이전에 끝난 클립만 후보. 그래서 문항마다 후보 수가
            #   다르다(96 ~ 5,966개, 중앙값 2,489). recall 을 6,223 기준으로 해석하면 틀린다.
            vis = [i for i, k in enumerate(keys) if ts_end[k] <= qt]
            vis_keys = [keys[i] for i in vis]
            tgt_present = targets & set(vis_keys)
            if not tgt_present or len(vis) < 2:
                # 정답이 아직 안 일어났거나(질문 시각 이전에 없음) 이 arm 에 임베딩이 없음.
                # 채점 불가로 빼고, 아래 common 계산에서 모든 arm 공통 문항만 남긴다.
                skipped[n].append(qid)
                continue
            pool_sizes[n][qid] = len(vis)
            sub = emb[vis]                       # [m, d]
            # some numpy/BLAS builds (2.0 + Accelerate) raise spurious divide/overflow warnings
            # here even for finite inputs; load_arms already checked the rows are finite.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                # 양쪽 다 unit() 으로 정규화돼 있으므로 내적 = 코사인 유사도. [후보수, 검색어수]
                sims = sub @ qvecs
            # 검색어마다 유사도 내림차순 정렬. order_per_query[j][0] 이 j번째 검색어의 1위.
            order_per_query = [np.argsort(-sims[:, j]) for j in range(sims.shape[1])]
            best = None
            for pos, i in enumerate(order_per_query[0], 1):
                if vis_keys[i] in tgt_present:
                    best = pos                   # 정답이 처음 나온 순위 (1부터)
                    break
            ranks[n][qid] = best
            for k in ks:
                # 검색어가 여러 개면 각자의 상위 k 를 합집합으로 본다.
                # 주의: ranks 는 위에서 order_per_query[0] 만 쓰므로 검색어가 2개 이상이면
                # rank 와 hit@k 의 기준이 달라진다. --query-source question 은 1개라 무관.
                topk = {vis_keys[i] for order in order_per_query for i in order[:k]}
                hits[n][k][qid] = int(bool(tgt_present & topk))

    # 모든 arm 이 채점할 수 있었던 질문만 남긴다 = 짝지은 비교의 대상.
    # 이게 있어야 arm 간 차이가 "문항 구성이 달라서" 생기는 일이 없다.
    common = sorted(set.intersection(*[set(hits[n][ks[0]]) for n in arm_names]), key=int)
    print(f"\nquestions: {len(pool['questions'])} | scorable in every arm: {len(common)}")
    for n in arm_names:
        if skipped[n]:
            print(f"  {n}: {len(skipped[n])} unscorable (target clip has no embedding in this arm, "
                  f"or nothing was visible yet)")

    def arr(n: str, k: int) -> np.ndarray:
        """arm n 의 recall@k 를 문항별 0/1 벡터로. 길이는 항상 len(common).

        출력 예) array([0., 0., 1., 0., ...])  -> .mean() 이 곧 recall@k
        모든 arm 이 같은 순서의 같은 문항이라 그대로 짝지어 빼면 된다.
        """
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
        """아무것도 모르는 랭커가 상위 k 안에 정답을 넣을 확률.

        입력:  t 정답 클립 수, v 후보(가시 풀) 수, k 상위 몇 개
        출력:  확률 0~1.  chance_at(1, 2489, 3) = 0.0012,  chance_at(21, 2489, 50) = 0.348

        recall 은 이 값과 비교해야만 의미가 있다. tolerance 를 올리면 t 가 늘어나
        recall 이 저절로 오르는데, 이 값도 같이 오르기 때문.
        ★ tolerance 가 다른 두 recall 을 직접 비교하지 말 것. 각자 자기 chance 와 비교.

        k*t/v (합집합 상한) 를 쓰면 안 된다. 정답 두 개가 동시에 들어오는 경우를 중복해서
        세고, k*t 가 v 에 가까워지면 1 을 넘어간다. tolerance 300s(t~21, k=50)에서
        그 식은 52%, 실제는 45% 였고 -- arm 이 우연 이하로 보이느냐 아니냐가 갈렸다.
        여기서는 정확식 1 - C(v-t,k)/C(v,k) 를 곱셈으로 푼다.
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
            miss *= num / (v - i)        # 정답을 i번째까지 계속 피할 확률
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
            pv = exact_paired_p(a_only, b_only)
            pairs.append({"a": a, "b": b, "k": mk, "diff_pp": diff,
                          "a_only": a_only, "b_only": b_only,
                          "mcnemar_p": pv, "n_discordant": a_only + b_only})
            flag = "" if pv < 0.05 else "  <- not significant"
            print(f"  {a} - {b}: {diff:+5.1f}pp  ({a} only {a_only}, {b} only {b_only}, "
                  f"discordant {a_only + b_only})  McNemar p={pv:.3f}{flag}")

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
              "| arms | diff (pp) | a only | b only | discordant | McNemar p |",
              "|---|---|---|---|---|---|"]
    for p in pairs:
        lines.append(f"| `{p['a']}` - `{p['b']}` | {p['diff_pp']:+.1f} | {p['a_only']} | "
                     f"{p['b_only']} | {p['n_discordant']} | {p['mcnemar_p']:.3f} |")
    lines += ["", "Only the discordant questions carry information about which arm is better, and "
                  "McNemar asks whether their split is further from even than a coin would give. "
                  "Six discordant all one way is the first split reaching p < 0.05; below that, a "
                  "clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', "
                  "not 'no difference'."]
    lines += ["", "How to read it: `center@R` is the control. `gaze@R` beating `full` but not "
                  "`center@R` means cropping helped and gaze did not.", ""]
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
