#!/usr/bin/env bash
# EgoServe(A1_JAKE) 의 88 개 검색 질의를 full / center / randf / gazef 4 조건으로 채점한다.
# 임베딩은 ../gaze_crop/emb 것을 그대로 쓰므로 새로 만드는 건 질의 벡터(GPU 약간)뿐이다.
#
#   CUDA_VISIBLE_DEVICES=1 bash run_recall.sh            # gpu3 에서
#
# 질의 = 정답 장면의 텍스트 묘사(episodic: past_observation, long_term: memory_key). build_egoserve_pool.py 참고.
# tol 0 이 주 결과, 30/60 은 강건성 확인.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GC="$HERE/../gaze_crop"
PY="${PY:-python}"
ARMS="${ARMS:-full center_05 gazef_05_randf gazef_05}"
TOLS="${TOLS:-0 30 60}"
QLABEL="${QLABEL:-pastobs}"      # 결과 파일 이름에 붙는 질의 이름
mkdir -p "$HERE/results" "$HERE/analysis/tables"
[ -f "$HERE/pool_egoserve.json" ] || "$PY" "$HERE/build_egoserve_pool.py"
echo "gpu: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
for QS in question; do
  for T in $TOLS; do
    echo "=== query=$QS tol=${T}s ==="
    "$PY" "$GC/recall_eval.py" --pool "$HERE/pool_egoserve.json" --arms $ARMS \
        --query-source "$QS" --target-tolerance-sec "$T" \
        --out "$HERE/results/recall_egoserve_${QLABEL}_tol${T}.json" \
        --markdown "$HERE/analysis/tables/recall_egoserve_${QLABEL}_tol${T}.md"
  done
done
