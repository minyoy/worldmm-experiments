#!/usr/bin/env bash
# Stage 1: visual retrieval recall on the 623-clip pool. No LLM, no answer generation.
# Run from the repo root, in the environment with torch + decord + the VLM2Vec weights.
#
#   bash experiments/gaze_crop/run_stage1.sh                  # 1st pass: renders the transform
#                                                             #   candidates and stops for your call
#   GAZE_TRANSFORM=rot90cw bash experiments/gaze_crop/run_stage1.sh   # 2nd pass: the real run, ~30 min
#   nohup GAZE_TRANSFORM=rot90cw bash experiments/gaze_crop/run_stage1.sh > /dev/null 2>&1 &
#
# Knobs: VIDEO_ROOT, GAZE_ROOT (default data/EgoLife/EyeGaze/A1_JAKE), RATIO (0.5),
#        GAZE_TRANSFORM (required for the real run), N_DISTRACTORS (500),
#        SKIP_CONTROLS=1 (drop the pkl and random-crop arms),
#        NO_GAZE=1 (run full vs centre crop only, on purpose),
#        MAX_CPUS=N (cap the whole run at N cores, i.e. N*100% in top -- see below)
#
# Ends by printing whether the gaze arms beat `full`, which is what decides stage 2.
# It never starts stage 2 itself -- that is run_stage2.sh, and it costs 3-4 hours.
set -uo pipefail          # NOT -e: a late failure must still reach the digest

# MAX_CPUS: hold the whole run to N cores on a shared machine. Decode (decord) and the BLAS/OMP
# pools each default to "every core", so capping needs both a hard ceiling and a thread count --
# affinity alone leaves N*ncore threads fighting over N cores. taskset pins this shell, and every
# python it starts inherits that mask, so one re-exec covers all the steps below.
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
  echo "WARNING: MAX_CPUS=$MAX_CPUS but taskset is missing. Thread counts are capped, but nothing"
  echo "         enforces the ceiling -- libraries that ignore those vars can still use every core."
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VIDEO_ROOT="${VIDEO_ROOT:-${WORLDMM_VIDEO_ROOT:-/datasets/EgoLife}}"
GAZE_ROOT="${GAZE_ROOT:-$REPO/data/EgoLife/EyeGaze/A1_JAKE}"
RATIO="${RATIO:-0.5}"
GAZE_TRANSFORM="${GAZE_TRANSFORM:-}"
N_DISTRACTORS="${N_DISTRACTORS:-500}"
SKIP_CONTROLS="${SKIP_CONTROLS:-0}"
PY="${PY:-python}"

GAZEF_ARM="gazef@$RATIO"      # the box follows the gaze frame by frame
mkdir -p "$HERE/logs"
LOG="$HERE/logs/stage1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO" || exit 1

say() { echo; echo "=== $* ==================================================" ; }

say "$(date '+%F %T')  stage 1: retrieval recall"
echo "video root: $VIDEO_ROOT"
echo "gaze root : $GAZE_ROOT"
echo "ratio     : $RATIO"
echo "cpus      : ${MAX_CPUS:-unbounded}${MAX_CPUS:+ cores (taskset 0-$((MAX_CPUS - 1)))}"
echo "log       : $LOG"

say "0. preflight"
[ -d "$VIDEO_ROOT" ] || { echo "FATAL: video root missing: $VIDEO_ROOT"; exit 1; }
HAS_GAZE=0
"$PY" "$HERE/check_gaze.py" --gaze-root "$GAZE_ROOT" && HAS_GAZE=1 || true
if [ "$HAS_GAZE" != "1" ] && [ "${NO_GAZE:-0}" != "1" ]; then
  echo
  echo "FATAL: no gaze data found, so the gaze arms cannot be built."
  echo "       Looked in: $GAZE_ROOT"
  echo "       Download it there, or point GAZE_ROOT elsewhere:"
  echo "           huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze \\"
  echo "               --repo-type dataset --include \"EyeGaze/A1_JAKE/*\" --local-dir data/EgoLife"
  echo "       Deliberately measuring full vs centre crop only? Rerun with NO_GAZE=1."
  exit 1
fi

say "1. questions: all 500"
"$PY" "$HERE/build_questions.py" || exit 1

say "2. pool (623 clips)"
"$PY" "$HERE/build_pool.py" --n-distractors "$N_DISTRACTORS" || exit 1

say "2. gaze"
ARMS="full center@$RATIO"
if [ "$HAS_GAZE" = "1" ] && [ -z "$GAZE_TRANSFORM" ]; then
  # The gaze CSVs hold yaw/pitch, not pixels, so the sign/rotation mapping to EgoLife's re-exported
  # video has to be settled before any embedding. It is settled by looking, not by a score: this
  # renders the candidates and stops.
  "$PY" "$HERE/compare_transforms.py" --gaze-root "$GAZE_ROOT" --video-root "$VIDEO_ROOT" \
      --ratio "$RATIO" || exit 1
  cat <<'MSG'

--------------------------------------------------------------------------
Stopping here on purpose: the gaze->image transform is yours to choose.

Open the variants/*_variants.jpg listed above. Pick the tile whose crop box is on what the wearer
was plainly looking at, then rerun with that label:

    GAZE_TRANSFORM=<label> bash experiments/gaze_crop/run_stage1.sh

If no tile looks right, the projection itself may be off -- try
    python experiments/gaze_crop/compare_transforms.py --gaze-root ... --projection pinhole
or a different --focal-px, and look again.
--------------------------------------------------------------------------
MSG
  exit 0
fi

if [ "$HAS_GAZE" = "1" ]; then
  echo "transform: $GAZE_TRANSFORM  (chosen by you)"
  "$PY" "$HERE/prepare_gaze.py" --gaze-root "$GAZE_ROOT" --video-root "$VIDEO_ROOT" \
      --transform "$GAZE_TRANSFORM" --dump-overlay 6 || exit 1
  echo ">>> overlay/*.jpg is the receipt for this transform -- keep it with the results."
  ARMS="$ARMS $GAZEF_ARM"
  if [ "$SKIP_CONTROLS" != "1" ]; then
    "$PY" "$HERE/prepare_gaze.py" --pseudo random-frames \
        --out "$HERE/gaze_random_frames.json"
  fi
else
  echo "no gaze -> full vs centre crop only"
fi

say "3. embeddings: $ARMS"
# shellcheck disable=SC2086
"$PY" "$HERE/embed_arms.py" --arms $ARMS --video-root "$VIDEO_ROOT" || exit 1
if [ "$SKIP_CONTROLS" != "1" ]; then
  "$PY" "$HERE/embed_arms.py" --arms pkl
  # 움직이지만 시선은 아닌 박스. gazef 가 이겼을 때 "움직임 자체의 효과"를 끊어 내는 유일한 arm.
  [ "$HAS_GAZE" = "1" ] && "$PY" "$HERE/embed_arms.py" --arms "$GAZEF_ARM" \
      --gaze "$HERE/gaze_random_frames.json" --arm-suffix _randf --video-root "$VIDEO_ROOT"
fi

# 출력 경로는 항상 명시한다 (recall_eval 의 기본값은 recall_scratch_* 로 떨어진다).
# 이름 규칙: recall_{문항집합}_{tolerance}.{json,md}  -- 여기는 623클립 풀 + 엄격 채점.
say "4. recall"
for SRC in question keywords; do
  "$PY" "$HERE/recall_eval.py" --query-source "$SRC" --target-tolerance-sec 0 \
      --out "$HERE/results/recall_stage1_${SRC}_tol0.json" \
      --markdown "$HERE/analysis/recall_stage1_${SRC}_tol0.md" \
      || { [ "$SRC" = "question" ] && exit 1; }
done

say "5. verdict"
"$PY" "$HERE/gate.py" --results "$HERE/results/recall_stage1_question_tol0.json" --arms "$GAZEF_ARM"
VERDICT=$?
"$PY" "$HERE/digest.py"
say "$(date '+%F %T')  stage 1 done. log: $LOG"
if [ "$VERDICT" = "0" ]; then
  echo
  echo "A gaze arm beat full. Stage 2 (answer accuracy, 3-4 h) is worth running:"
  echo "    bash experiments/gaze_crop/run_stage2.sh"
else
  echo
  echo "No gaze arm beat full. Read analysis/DIGEST.md before spending 4 hours on stage 2."
  echo "Force it anyway with:  FORCE=1 bash experiments/gaze_crop/run_stage2.sh"
fi
