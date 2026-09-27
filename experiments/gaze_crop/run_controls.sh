#!/usr/bin/env bash
# 대조군 두 개를 실제 인덱스 크기(6,223클립)로 올리고 500문항으로 채점한다.
#
# 이 실험에 남은 단 하나의 미결 질문 -- gazef@R 이 full 을 이긴 이유가 (a) 자르기냐 (b) 박스가
# 움직인 것이냐 (c) 그 움직임이 시선이어서냐 -- 를 가른다. (a) 는 center@R, (b) 는 _randf 가 끊는다.
#
#   nohup bash experiments/gaze_crop/run_controls.sh > /dev/null 2>&1 &            # GPU 1장, 직렬 ~4h
#   GPUS=1,2 nohup bash experiments/gaze_crop/run_controls.sh > /dev/null 2>&1 &    # GPU 2장, 병렬 ~2h
#   tail -f experiments/gaze_crop/logs/controls_*.log
#
# 환경변수: GPUS(예 "1,2" — arm 하나당 GPU 하나씩 병렬. 비우면 현재 CUDA_VISIBLE_DEVICES 그대로 직렬),
#           VIDEO_ROOT(기본 /datasets/EgoLife), RATIO(0.5), PY(python), TOLS("0 60"),
#           MAX_CPUS=N (공유 머신에서 전체를 N코어로 묶는다), EMBED_ONLY=1 (채점 없이 임베딩만),
#           SKIP_CENTER=1 / SKIP_RANDF=1 (한쪽만 돌릴 때. 둘 다 1이면 채점만)
#
# 왜 arm 단위 병렬인가: 두 arm 은 서로 다른 npz 를 쓰는 완전히 독립적인 작업이라 GPU 하나씩 주면
# 벽시계가 그대로 절반이 된다. 한 arm 을 여러 GPU 로 쪼개는 것(클립 분할)은 npz 병합이 필요해서
# arm 이 2개뿐인 지금은 얻는 것이 없다.
#
# 이어받기: embed_arms.py 가 클립 키로 이어받으므로 중간에 죽어도 같은 명령을 다시 부르면 된다.
# 623클립은 1단계에서 이미 들어 있어 각 arm 마다 ~5,600개만 새로 만든다(arm 당 ~2시간).

set -u

# MAX_CPUS: 디코딩(decord)과 BLAS/OMP 풀이 각각 "모든 코어"를 기본값으로 쓰므로, 공유 머신에서는
# 스레드 수와 affinity 를 같이 묶어야 한다(run_stage1.sh 와 같은 방식). 병렬 모드에서는 arm 두 개가
# 동시에 디코딩하므로 DECORD 스레드는 절반씩 나눠 준다.
MAX_CPUS="${MAX_CPUS:-}"
if [ -n "$MAX_CPUS" ] && [ -z "${WORLDMM_CPU_PINNED:-}" ]; then
  if ! [ "$MAX_CPUS" -ge 1 ] 2>/dev/null; then
    echo "FATAL: MAX_CPUS must be a positive integer, got '$MAX_CPUS'"; exit 1
  fi
  export OMP_NUM_THREADS="$MAX_CPUS" MKL_NUM_THREADS="$MAX_CPUS" \
         OPENBLAS_NUM_THREADS="$MAX_CPUS" NUMEXPR_NUM_THREADS="$MAX_CPUS" \
         DECORD_NUM_THREADS="$MAX_CPUS" WORLDMM_CPU_PINNED=1
  if command -v taskset >/dev/null 2>&1; then
    exec taskset -c "0-$((MAX_CPUS - 1))" bash "$0" "$@"
  fi
  echo "WARNING: MAX_CPUS=$MAX_CPUS but taskset is missing; only thread counts are capped."
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
RATIO="${RATIO:-0.5}"
VIDEO_ROOT="${VIDEO_ROOT:-/datasets/EgoLife}"
TOLS="${TOLS:-0 60}"
POOL_ALL="$HERE/pool_all.json"
RANDF_GAZE="$HERE/gaze_random_frames.json"
CENTER_ARM="center@$RATIO"
GAZEF_ARM="gazef@$RATIO"
RANDF_SUFFIX="_randf"
GAZEF_FILE="gazef_$(echo "$RATIO" | tr -d '.')"                  # gazef_05
RANDF_NAME="${GAZEF_FILE}${RANDF_SUFFIX}"                        # gazef_05_randf

# GPUS="1,2" -> ("1" "2").  비어 있으면 현재 CUDA_VISIBLE_DEVICES 를 그대로 쓰는 직렬 실행.
IFS=', ' read -r -a GPU_ARR <<< "${GPUS:-}"
PARALLEL=0
[ "${#GPU_ARR[@]}" -ge 2 ] && PARALLEL=1

mkdir -p "$HERE/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$HERE/logs/controls_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "=== $(date '+%F %T')  $* ==================================================="; }

# ---------------------------------------------------------------------------
# preflight. 4시간을 태운 뒤 "풀이 없다"를 알게 되는 일이 없도록 먼저 다 본다.
# ---------------------------------------------------------------------------
say "0. preflight"
FATAL=0
[ -f "$POOL_ALL" ] || { echo "FATAL: $POOL_ALL 없음 -- build_pool.py --all-clips --questions questions_500.json --out pool_all.json"; FATAL=1; }
[ -f "$HERE/emb/full.npz" ] || { echo "FATAL: emb/full.npz 없음 (2단계가 돌지 않았다)"; FATAL=1; }
[ -f "$HERE/emb/$GAZEF_FILE.npz" ] || { echo "FATAL: emb/gazef_*.npz 없음 -- 비교 대상이 없다"; FATAL=1; }
[ -f "$HERE/gaze_points.json" ] || { echo "FATAL: gaze_points.json 없음 -- prepare_gaze.py 를 먼저"; FATAL=1; }
[ -d "$VIDEO_ROOT" ] || { echo "FATAL: VIDEO_ROOT=$VIDEO_ROOT 없음"; FATAL=1; }
[ "$RANDF_GAZE" != "$HERE/gaze_points.json" ] || { echo "FATAL: randf 좌표 파일이 실제 gaze 파일과 같다"; FATAL=1; }
[ "$FATAL" = "0" ] || exit 1
echo "pool      : $POOL_ALL  ($("$PY" -c "import json,sys;print(json.load(open('$POOL_ALL'))['n_clips'],'clips /',len(json.load(open('$POOL_ALL'))['questions']),'questions')"))"
echo "arms      : $CENTER_ARM, $RANDF_NAME  (비교 대상 full, $GAZEF_ARM)"
echo "tolerance : $TOLS"
if [ "$PARALLEL" = "1" ]; then
  echo "gpus      : ${GPU_ARR[0]} <- $CENTER_ARM,  ${GPU_ARR[1]} <- $RANDF_NAME  (병렬, arm 로그는 별도 파일)"
elif [ "${#GPU_ARR[@]}" = "1" ]; then
  echo "gpus      : ${GPU_ARR[0]} (직렬). 두 장으로 반 나눠 돌리려면 GPUS=1,2"
else
  echo "gpus      : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} (직렬). 두 장이면 GPUS=1,2"
fi
echo "log       : $LOG"

# ---------------------------------------------------------------------------
# randf 의 좌표 파일은 GPU 를 쓰지 않으므로 병렬 분기 전에 여기서 만든다(경쟁 방지).
# 반드시 pool_all 기준으로: 기본값(pool.json, 623클립)으로 만들면 5,600클립이 "gaze 없음"으로
# 빠지고, recall_eval 이 그 문항을 전부 채점 불가로 버린다.
if [ "${SKIP_RANDF:-0}" != "1" ]; then
  say "1. randf 좌표 파일"
  if [ -f "$RANDF_GAZE" ] && "$PY" - "$RANDF_GAZE" "$POOL_ALL" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1])); g = g.get("clips", g)
pool = {c["key"] for c in json.load(open(sys.argv[2]))["clips"]}
missing = len(pool - set(g))
print(f"  {sys.argv[1]}: {len(g)} clips, pool 중 누락 {missing}")
sys.exit(1 if missing else 0)
PYEOF
  then
    echo "  기존 파일 재사용"
  else
    echo "  (다시 만든다)"
    "$PY" "$HERE/prepare_gaze.py" --pool "$POOL_ALL" --pseudo random-frames \
        --out "$RANDF_GAZE" || exit 1
  fi
fi

# ---------------------------------------------------------------------------
# arm 임베딩. 병렬 모드면 arm 하나당 GPU 하나씩 동시에, 아니면 차례로.
# ---------------------------------------------------------------------------
embed_center() {
  "$PY" "$HERE/embed_arms.py" --pool "$POOL_ALL" --arms "$CENTER_ARM" --video-root "$VIDEO_ROOT"
}
embed_randf() {
  "$PY" "$HERE/embed_arms.py" --pool "$POOL_ALL" --arms "$GAZEF_ARM" \
      --gaze "$RANDF_GAZE" --arm-suffix "$RANDF_SUFFIX" --video-root "$VIDEO_ROOT"
}

say "2. arm 임베딩 (center = 자르기 효과, randf = 움직임 효과)"
RC_C=0; RC_R=0
if [ "$PARALLEL" = "1" ]; then
  # 두 프로세스가 동시에 디코딩하므로 decord 스레드는 반으로 나눈다(MAX_CPUS 를 준 경우).
  if [ -n "$MAX_CPUS" ]; then
    export DECORD_NUM_THREADS="$(( MAX_CPUS / 2 > 0 ? MAX_CPUS / 2 : 1 ))"
  fi
  LOG_C="$HERE/logs/controls_${STAMP}_center.log"
  LOG_R="$HERE/logs/controls_${STAMP}_randf.log"
  if [ "${SKIP_CENTER:-0}" != "1" ]; then
    echo "  gpu ${GPU_ARR[0]}: $CENTER_ARM  -> $LOG_C"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" embed_center > "$LOG_C" 2>&1 &
    PID_C=$!
  fi
  if [ "${SKIP_RANDF:-0}" != "1" ]; then
    echo "  gpu ${GPU_ARR[1]}: $RANDF_NAME -> $LOG_R"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[1]}" embed_randf > "$LOG_R" 2>&1 &
    PID_R=$!
  fi
  # 두 arm 의 진행률은 각자 로그에 있다. 여기서는 둘 다 끝날 때까지 기다리고 종료 코드만 본다.
  [ -n "${PID_C:-}" ] && { wait "$PID_C"; RC_C=$?; tail -n 3 "$LOG_C"; }
  [ -n "${PID_R:-}" ] && { wait "$PID_R"; RC_R=$?; tail -n 3 "$LOG_R"; }
else
  if [ "${SKIP_CENTER:-0}" != "1" ]; then
    [ "${#GPU_ARR[@]}" = "1" ] && export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
    embed_center; RC_C=$?
  fi
  if [ "${SKIP_RANDF:-0}" != "1" ]; then
    [ "${#GPU_ARR[@]}" = "1" ] && export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
    embed_randf; RC_R=$?
  fi
fi
[ "$RC_C" = "0" ] || { echo "FATAL: $CENTER_ARM 임베딩 실패 (rc=$RC_C)"; exit 1; }
[ "$RC_R" = "0" ] || { echo "FATAL: $RANDF_NAME 임베딩 실패 (rc=$RC_R)"; exit 1; }

if [ "${SKIP_RANDF:-0}" != "1" ]; then
  # 덮어쓰기 사고 검사. --arm-suffix 를 빼면 무작위 좌표가 실험군 arm(gazef_05.npz, 2시간짜리)을
  # 조용히 덮어쓴다. 각 npz 옆의 meta json 이 어떤 좌표 파일로 만들어졌는지 적어 두므로 그것을 본다.
  [ -f "$HERE/emb/$RANDF_NAME.npz" ] || { echo "FATAL: emb/$RANDF_NAME.npz 가 생기지 않았다"; exit 1; }
  "$PY" - "$HERE/emb/$GAZEF_FILE.meta.json" "$RANDF_GAZE" <<'PYEOF' || exit 1
import json, os, sys
meta_path, randf = sys.argv[1], sys.argv[2]
if not os.path.exists(meta_path):
    sys.exit(0)
used = json.load(open(meta_path)).get("gaze_file", "")
if os.path.basename(str(used)) == os.path.basename(randf):
    print(f"FATAL: {meta_path} 가 무작위 좌표({used})로 만들어졌다 -- 실험군 arm 이 덮어써졌다. "
          f"emb/{os.path.basename(meta_path).replace('.meta.json','')}.npz 를 지우고 실제 gaze 로 다시 만들 것")
    sys.exit(1)
print(f"  ok: 실험군 arm 은 여전히 {used} 로 만들어진 것")
PYEOF
fi

if [ "${EMBED_ONLY:-0}" = "1" ]; then
  say "EMBED_ONLY=1 -> 채점 전에 멈춘다. 채점만 다시 돌리려면:"
  echo "  SKIP_CENTER=1 SKIP_RANDF=1 bash experiments/gaze_crop/run_controls.sh"
  exit 0
fi

# ---------------------------------------------------------------------------
# 채점은 GPU 가 거의 필요 없다(질의 임베딩은 .cache_queries.npz 에 이미 있다). tolerance 두 값 모두.
say "3. recall, 4 arms x tolerance {$TOLS}"
ARMS="full $CENTER_ARM $GAZEF_ARM $RANDF_NAME"
for TOL in $TOLS; do
  # shellcheck disable=SC2086
  "$PY" "$HERE/recall_eval.py" --pool "$POOL_ALL" --arms $ARMS --target-tolerance-sec "$TOL" \
      --out "$HERE/results/recall_arms_tol$TOL.json" \
      --markdown "$HERE/analysis/recall_arms_tol$TOL.md" || exit 1
done

say "4. digest"
"$PY" "$HERE/digest.py"
cat <<'MSG'

--------------------------------------------------------------------------
읽는 법 (k=3, 같은 문항·같은 질의·같은 candidate set):

  gazef > center  그리고  gazef > randf   -> 시선의 순간 위치가 정보다
  gazef ~ randf  > center                 -> 이득은 박스가 움직인 것 자체 (시선은 부수적)
  gazef ~ center                          -> 이득은 자르기 자체. gaze 는 접는다

판정은 이 두 쌍(gazef-center, gazef-randf)만 본다. arm 4개면 pair 가 6개이고,
6개를 다 보고 이긴 쌍을 고르면 그게 다중비교다. 불일치 건수가 6건 미만인 쌍은
방향이 한쪽으로 몰려도 p<0.05 가 될 수 없으니 "차이 없음"이 아니라 "검정력 부족"이다.
--------------------------------------------------------------------------
MSG
say "$(date '+%F %T')  controls done. log: $LOG"
