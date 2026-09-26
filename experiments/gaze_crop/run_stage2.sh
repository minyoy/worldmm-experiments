#!/usr/bin/env bash
# Stage 2: answer accuracy. Extends the winning arm to all 6,223 clips, then runs condition E'
# with that arm as the visual index. ~3-4 hours. Run stage 1 first.
#
#   bash experiments/gaze_crop/run_stage2.sh                  # carries stage 1's winner
#   ARM=gazef@0.5 bash experiments/gaze_crop/run_stage2.sh    # pick the arm yourself
#   FORCE=1 bash experiments/gaze_crop/run_stage2.sh          # run even though stage 1 said no
#   VISUAL_TOP_K="3 20" bash experiments/gaze_crop/run_stage2.sh   # QA at both depths, one run
#   nohup bash experiments/gaze_crop/run_stage2.sh > /dev/null 2>&1 &
#
# Knobs: VIDEO_ROOT, GAZE_ROOT, ARM, FORCE, MIN_GAIN_PP (0.0), SKIP_QA=1 (embeddings + recall only),
#        VISUAL_TOP_K (default 3; space-separated list runs step 6 once per value),
#        MAX_CPUS=N (cap the whole run at N cores)
#
# k lives in the results path (results_qa/<arm>_k<K>/), because --resume keys off that file: with a
# shared path the second k would skip all 120 answered questions and silently report the first k's
# answers as its own.
#
# What it does NOT change: the LLM sees the retrieved clip's own uncropped frames, exactly as
# condition E' always did. Only which clips get retrieved changes. Showing the LLM cropped frames
# is a different experiment; mixing the two would make the result unreadable.
set -uo pipefail

# Hold the run to N cores on a shared box; see run_stage1.sh for why both the mask and the thread
# counts are needed. Re-exec happens before any child is forked, so every step inherits it.
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
REPO="$(cd "$HERE/../.." && pwd)"
VB="$REPO/experiments/visual_bottleneck"
VIDEO_ROOT="${VIDEO_ROOT:-${WORLDMM_VIDEO_ROOT:-/datasets/EgoLife}}"
GAZE_ROOT="${GAZE_ROOT:-$REPO/data/EgoLife/EyeGaze/A1_JAKE}"
ARM="${ARM:-}"
FORCE="${FORCE:-0}"
MIN_GAIN_PP="${MIN_GAIN_PP:-0.0}"
SKIP_QA="${SKIP_QA:-0}"
VISUAL_TOP_K="${VISUAL_TOP_K:-3}"
PY="${PY:-python}"

for K in $VISUAL_TOP_K; do
  [ "$K" -ge 1 ] 2>/dev/null || { echo "FATAL: VISUAL_TOP_K must be positive integers, got '$VISUAL_TOP_K'"; exit 1; }
done

POOL_ALL="$HERE/pool_all.json"
mkdir -p "$HERE/logs"
LOG="$HERE/logs/stage2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO" || exit 1

say() { echo; echo "=== $* ==================================================" ; }

say "$(date '+%F %T')  stage 2: answer accuracy"
echo "log : $LOG"
echo "k   : $VISUAL_TOP_K  (visual-top-k for condition E')"
echo "cpus: ${MAX_CPUS:-unbounded}${MAX_CPUS:+ cores (taskset 0-$((MAX_CPUS - 1)))}"

say "0. preflight"
FATAL=0
[ -d "$VIDEO_ROOT" ] || { echo "FATAL: video root missing: $VIDEO_ROOT"; FATAL=1; }
[ -f "$HERE/results/recall_question.json" ] || { echo "FATAL: stage 1 has not run (no results/recall_question.json)"; FATAL=1; }

# E' replays condition B's cached text context; without it eval_egolife.py refuses
N_CTX=$(ls "$VB/text_context"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "B text_context cache: $N_CTX files"
if [ "$N_CTX" -lt 120 ] && [ "$SKIP_QA" != "1" ]; then
  echo "FATAL: stage 2's QA needs 120 cached text contexts from condition B ($VB/text_context)."
  echo "       Run condition B first, or rerun with SKIP_QA=1 for embeddings + recall only."
  FATAL=1
fi

# which arm goes forward, and was it actually worth it
GATE_OUT=$("$PY" "$HERE/gate.py" --results "$HERE/results/recall_question.json" \
           --min-gain-pp "$MIN_GAIN_PP" 2>&1)
GATE_RC=$?
echo "$GATE_OUT"
[ -z "$ARM" ] && ARM=$(echo "$GATE_OUT" | grep -oE '^BEST=.*$' | tail -1 | cut -d= -f2)
[ -z "$ARM" ] && { echo "FATAL: could not decide which arm to carry; pass ARM=gazef@0.5"; FATAL=1; }
if [ "$GATE_RC" != "0" ] && [ "$FORCE" != "1" ]; then
  echo "stage 1 did not clear the bar (gate exit $GATE_RC). Stopping."
  echo "Rerun with FORCE=1 if you want the accuracy number anyway."
  FATAL=1
fi
[ "$FATAL" = "1" ] && { echo "aborting before spending GPU time"; exit 1; }

# the transform stage 1 settled on; the full-pool gaze file must use the same one
TRANSFORM=$("$PY" -c "
import json,sys
try: print(json.load(open('$HERE/gaze_points.json')).get('transform','none'))
except Exception: print('none')")
echo "arm: $ARM   transform: $TRANSFORM"

say "1. pool: all 6,223 clips"
"$PY" "$HERE/build_pool.py" --all-clips --out "$POOL_ALL" || exit 1

case "$ARM" in
  gaze*|gazef*)
    say "2. gaze over the full pool"
    "$PY" "$HERE/prepare_gaze.py" --pool "$POOL_ALL" --gaze-root "$GAZE_ROOT" \
        --video-root "$VIDEO_ROOT" --transform "$TRANSFORM" || exit 1 ;;
esac

say "3. embeddings: full + $ARM  (stage 1's 623 clips are reused; ~5,600 left per arm)"
"$PY" "$HERE/embed_arms.py" --pool "$POOL_ALL" --arms full "$ARM" \
    --video-root "$VIDEO_ROOT" || exit 1

say "4. recall at the real index size"
# --arms is explicit: recall_eval scores a question only when every loaded arm can score it, so a
# leftover 623-clip arm in emb/ would gut the comparison
"$PY" "$HERE/recall_eval.py" --pool "$POOL_ALL" --arms full "$ARM" \
    --out "$HERE/results/recall_allclips.json" \
    --markdown "$HERE/analysis/recall_allclips.md"

# How far apart the two arms actually are at each k. Recall alone cannot say: two arms can score the
# same and still disagree on which questions they got. Only the discordant counts (a_only + b_only)
# bound how much the accuracy numbers in step 6 could possibly differ -- if they are ~0 at some k,
# the LLM sees the same context in both arms there and any gap it reports is noise. Scoring only,
# no embedding, so this is seconds per k. Printed for reading in the morning; nothing branches on it.
say "4b. how much do the arms disagree at each k?"
for K in 1 3 5 10 20 50; do
  "$PY" "$HERE/recall_eval.py" --pool "$POOL_ALL" --arms full "$ARM" --main-k "$K" \
      --out "$HERE/results/recall_allclips_k$K.json" 2>/dev/null \
      | sed -n "/paired comparisons at k=$K/,/^$/p"
done

if [ "$SKIP_QA" = "1" ]; then
  say "SKIP_QA=1 -> stopping before the LLM"
  "$PY" "$HERE/digest.py"
  exit 0
fi

say "5. export the arms as visual_embeddings.pkl"
for A in full "$ARM"; do
  "$PY" "$HERE/export_pkl.py" --arm "$A" --pool "$POOL_ALL" || exit 1
done

say "6. condition E' per index x per k (~34 min each)"
ARM_FILE="$(echo "$ARM" | tr -d '.' | tr '@' '_')"
for K in $VISUAL_TOP_K; do
  for A in full "$ARM_FILE"; do
    OUT="$HERE/results_qa/${A}_k${K}"
    echo
    echo ">>> E' with emb/$A.pkl at visual-top-k=$K -> results_qa/${A}_k${K}"
    [ -f "$OUT/E_prime.json" ] && echo "(resuming; answered questions are skipped)"
    # a failure here must not kill the remaining k: each pair is a standalone result
    "$PY" "$VB/eval_egolife.py" --condition E_prime \
        --visual-path "$HERE/emb/$A.pkl" \
        --visual-top-k "$K" \
        --results-dir "$OUT" \
        --video-root "$VIDEO_ROOT" --resume \
      || echo "WARNING: E' failed for $A at k=$K -- continuing with the rest"
  done
done

say "7. parser sanity check"
"$PY" "$HERE/rescore.py" "$HERE"/results_qa/*/E_prime.json

say "8. digest"
"$PY" "$HERE/digest.py"
say "$(date '+%F %T')  stage 2 done. log: $LOG"
