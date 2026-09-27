#!/usr/bin/env python3
"""gazef vs center 를 k=3 이진 적중이 아니라 순위 수준에서 본다.

k=3 McNemar 는 500문항 중 top-3 경계를 넘나든 47문항만 쓴다. 순위 sign test 는 500문항을
다 쓰므로 검정력이 훨씬 높고, README 3.3 이 full-gazef 에 이미 쓴 것과 같은 통계다.
마지막 블록이 핵심: 이득이 순위 어디에서 나는지(깊은 곳이면 검색에 쓸모없다).

  python rank_probe.py            # experiments/gaze_crop 에서
"""
import json, math
from statistics import median

def signtest(b, w):
    n = b + w
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n)

PAIRS = [("center_05", "gazef_05"), ("full", "gazef_05"),
         ("full", "center_05"), ("gazef_05_randf", "gazef_05")]
BANDS = [(1, 10), (11, 50), (51, 200), (201, 10 ** 9)]

for tol in (60, 0):
    d = json.load(open(f"results/recall_arms_tol{tol}.json"))
    pq = d["per_question"]
    print(f"=== tol {tol}s, n_scored={d['n_scored']} ===")
    for a, b in PAIRS:
        ok = [q["rank"] for q in pq if q["rank"].get(a) and q["rank"].get(b)]
        imp = sum(1 for r in ok if r[b] < r[a])
        wor = sum(1 for r in ok if r[b] > r[a])
        print(f"  {b} vs {a}: better {imp} / worse {wor} / tie {len(ok)-imp-wor}, "
              f"sign p={signtest(imp, wor):.2g}, "
              f"median {median(r[a] for r in ok):.0f} -> {median(r[b] for r in ok):.0f}")

    a, b = "center_05", "gazef_05"
    print(f"  -- {b} vs {a}, 시작 순위 구간별 --")
    for lo, hi in BANDS:
        sel = [q["rank"] for q in pq if q["rank"].get(a) and lo <= q["rank"][a] <= hi]
        if not sel:
            continue
        imp = sum(1 for r in sel if r[b] < r[a])
        wor = sum(1 for r in sel if r[b] > r[a])
        print(f"     {a} rank {lo}-{hi if hi < 10**9 else '+'}: n={len(sel):3d} "
              f"better {imp:3d} worse {wor:3d} median dRank {median(r[a]-r[b] for r in sel):+.0f}")
    for k in (3, 10):
        gain = sum(1 for q in pq if q["rank"].get(a) and q["rank"][a] > k and q["rank"][b] <= k)
        loss = sum(1 for q in pq if q["rank"].get(a) and q["rank"][a] <= k and q["rank"][b] > k)
        print(f"     top-{k} 진입 +{gain} / 이탈 -{loss} (net {gain-loss:+d})")
