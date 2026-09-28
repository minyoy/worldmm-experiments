#!/usr/bin/env bash
# Frame selection vs. gaze crop vs. both, scored on retrieval recall.
#
# Runs on the FULL 6,223-clip pool (stage 2's pool_all.json), 500 questions. Three arms to embed
# at ~3.3 s/clip measured on gpu2 -> ~5.5-6 h. Start it detached:
#
#   nohup bash run_recall.sh > /dev/null 2>&1 &     # ~6 h, checkpoints every 50 clips
#   tail -f logs/recall_*.log
#
#   POOL=../gaze_crop/pool.json bash run_recall.sh  # stage-1's 623 clips instead (~35 min, but
#                                                   # 120 questions: McNemar needs 6 discordant
#                                                   # one way to reach p<0.05, so a real effect
#                                                   # can easily fail to show)
#   SELECT=refill bash run_recall.sh         # 16 frames, all inside fixations
#   RADIUS=0.03 bash run_recall.sh
#   SKIP_EMBED=1 bash run_recall.sh          # rescore what is already in emb/
#
# Interrupting it is safe: embed_fix_arms.py resumes from the npz, so rerunning picks up where it
# stopped. What is NOT safe is changing --radius/--select mid-way; that discards the arm and
# starts over (by design -- see embed_fix_arms.py).
#
# The comparison is four arms, and every one of them answers a question the others cannot:
#
#   full             baseline: 16 uniform frames, whole frame
#   gazef@R          crop follows the gaze  (the gaze_crop experiment's arm)
#   fixsel           frame SELECTION alone: the fixation-covered subset of full's own frames
#   fixsel_ctl       count-matched control: as many frames as fixsel, uniformly spaced,
#                    fixations ignored.  fixsel - fixsel_ctl is the selection effect;
#                    fixsel - full mixes it with "fewer frames"
#   fixsel_gazef@R   selection + crop: the arm this experiment is for
#
# recall_eval.py scores a question only when EVERY loaded arm can, so a clip dropped for having
# no fixation frame is dropped from all five columns and the table stays paired. Its
# "unscorable" line is how many questions that cost -- read it before the recall numbers.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GC="$HERE/../gaze_crop"
POOL="${POOL:-$GC/pool_all.json}"
GAZE="${GAZE:-$GC/gaze_points.json}"
VIDEO_ROOT="${VIDEO_ROOT:-${WORLDMM_VIDEO_ROOT:-/datasets/EgoLife}}"
RATIO="${RATIO:-0.5}"
SELECT="${SELECT:-subset}"
RADIUS="${RADIUS:-0.05}"
SKIP_EMBED="${SKIP_EMBED:-0}"
PY="${PY:-python}"

FIX_ARMS=(fixsel "fixsel_gazef@$RATIO" fixsel_ctl)
ALL_ARMS=(full "gazef@$RATIO" "${FIX_ARMS[@]}")
TAG="$(basename "$POOL" .json)_${SELECT}_r${RADIUS/./}"

mkdir -p "$HERE/analysis" "$HERE/results" "$HERE/logs"
LOG="$HERE/logs/recall_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log  : $LOG"
echo "pool : $POOL"
echo "arms : ${ALL_ARMS[*]}"

FATAL=0
[ -f "$POOL" ] || { echo "FATAL: $POOL missing. stage 2 builds it:"; \
                    echo "  python $GC/build_questions.py"; \
                    echo "  python $GC/build_pool.py --all-clips --out $POOL"; FATAL=1; }
[ -f "$GAZE" ] || { echo "FATAL: $GAZE missing. Run $GC/prepare_gaze.py --pool $POOL"; FATAL=1; }
[ -d "$VIDEO_ROOT" ] || { echo "FATAL: video root missing: $VIDEO_ROOT"; FATAL=1; }

# full and gazef@R are the gaze_crop experiment's; this script never rebuilds them, so that a
# rerun here cannot quietly change the baseline the fixation arms are being judged against.
#
# Their row COUNT is checked too, not just the file. The npz path does not carry the pool size, so
# a leftover 623-clip full.npz from stage 1 sits exactly where the 6,223-clip one belongs -- and
# recall_eval.py scores a question only when every arm can, so the whole table would silently
# collapse onto stage 1's clips while still printing 500 questions at the top.
if [ "$FATAL" = "0" ]; then
  N_POOL=$("$PY" -c "import json,sys; print(json.load(open('$POOL'))['n_clips'])") || exit 1
  echo "pool clips: $N_POOL"
  for A in full "gazef@$RATIO"; do
    F="$GC/emb/$(echo "$A" | tr -d '.' | tr '@' '_').npz"
    if [ ! -f "$F" ]; then
      echo "FATAL: $F missing. Build the baseline arms in gaze_crop first (~2.7 h for $N_POOL clips):"
      echo "  python $GC/embed_arms.py --pool $POOL --arms full gazef@$RATIO --video-root $VIDEO_ROOT"
      FATAL=1
      continue
    fi
    N_ARM=$("$PY" -c "
import numpy as np
print(len(np.load('$F', allow_pickle=False)['keys']))") || exit 1
    echo "  $A: $N_ARM embeddings"
    # 90%: a handful of clips legitimately lack a readable video or any gaze, and stage 2's own
    # logs show that shortfall. An arm built for a 10x smaller pool is nowhere near this line.
    if [ "$N_ARM" -lt $((N_POOL * 9 / 10)) ]; then
      echo "FATAL: $A holds $N_ARM embeddings for a $N_POOL-clip pool. That arm was built for a"
      echo "       different pool; finish it before comparing against it:"
      echo "  python $GC/embed_arms.py --pool $POOL --arms $A --video-root $VIDEO_ROOT"
      FATAL=1
    fi
  done
fi
[ "$FATAL" = "1" ] && { echo; echo "aborting before spending GPU time"; exit 1; }

if [ "$SKIP_EMBED" != "1" ]; then
  echo; echo "=== embedding ${FIX_ARMS[*]} ==="
  "$PY" "$HERE/embed_fix_arms.py" --pool "$POOL" --gaze "$GAZE" --video-root "$VIDEO_ROOT" \
      --arms "${FIX_ARMS[@]}" --select "$SELECT" --radius "$RADIUS" || exit 1
fi

echo; echo "=== recall ==="
"$PY" "$GC/recall_eval.py" --pool "$POOL" --arms "${ALL_ARMS[@]}" --target-tolerance-sec 0 \
    --out "$HERE/results/recall_${TAG}_tol0.json" \
    --markdown "$HERE/analysis/recall_${TAG}_tol0.md" || exit 1

# The pairwise table above is at k=3 only. The arms can tie on recall and still disagree about
# which questions they got; the discordant counts at each k bound how big any real difference
# could be, and a near-zero count means "underpowered", not "no effect".
# Same run as the table above with a different --main-k: identical recall, identical ranks, only
# the printed pairwise block differs. Recomputable from the json above, so it goes to
# results/discordance/ rather than next to the result that matters.
echo; echo "=== disagreement per k ==="
DISC="$HERE/results/discordance"
mkdir -p "$DISC"
for K in 1 3 5 10 20 50; do
  "$PY" "$GC/recall_eval.py" --pool "$POOL" --arms "${ALL_ARMS[@]}" --main-k "$K" \
      --out "$DISC/recall_k$K.json" --markdown "$DISC/recall_k$K.md" 2>/dev/null \
      | sed -n "/paired comparisons at k=$K/,/^$/p"
done
echo "done. table: $HERE/analysis/recall_${TAG}_tol0.md"
