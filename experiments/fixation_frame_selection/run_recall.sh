#!/usr/bin/env bash
# Frame selection vs. gaze crop vs. both, scored on retrieval recall.
#
# Runs on the FULL 6,223-clip pool (stage 2's pool_all.json), 500 questions. Four fixation arms,
# ~1.05 s decode + ~0.76 s per arm per clip on gpu2: ~7 h from scratch, ~3 h when only
# fixsel_gazef is new (the others resume from emb/). Start it detached:
#
#   CUDA_VISIBLE_DEVICES=3 nohup bash run_recall.sh > /dev/null 2>&1 &   # checkpointed
#   sleep 5; tail -f logs/recall_*.log               # the log does not exist for the first second
#
# The GPU comes from CUDA_VISIBLE_DEVICES, inherited as-is (same convention as gaze_crop's
# run_controls.sh). Set it: on a shared box an unset value grabs every card. To stop the run,
# kill the process GROUP -- `kill <pid>` leaves the python child holding GPU memory:
#
#   kill -- -<pid>   # the pid nohup printed
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
# Every arm answers a question the others cannot:
#
#   full             baseline: 16 uniform frames, whole frame
#   center@R         centre crop: the gaze_crop experiment's crop-without-gaze control
#   gazef@R          crop follows the gaze  (the gaze_crop experiment's arm)
#   fixsel           frame SELECTION alone: the fixation-covered subset of full's own frames
#   fixsel_ctl       count-matched control: as many frames as fixsel, uniformly spaced,
#                    fixations ignored.  fixsel - fixsel_ctl is the selection effect;
#                    fixsel - full mixes it with "fewer frames"
#   fixsel_gaze@R    selection + crop on each fixation's CENTROID (one box per fixation, like
#                    gaze_crop's gaze@R holds one per clip). Called fixsel_gazef before 2026-09-29.
#   fixsel_gazef@R   selection + gazef's own per-frame crop (nearest gaze sample).
#                    fixsel_gazef - gazef is the selection effect on top of the gaze crop
#
# recall_eval.py scores a question only when EVERY loaded arm can, so a clip dropped for having
# no fixation frame is dropped from every column and the table stays paired. Its
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
TOLS="${TOLS:-0 30 60}"     # tol 0 is the pre-registered result; 30/60 are robustness checks
PY="${PY:-python}"

FIX_ARMS=(fixsel "fixsel_gaze@$RATIO" "fixsel_gazef@$RATIO" fixsel_ctl)
BASE_ARMS=(full "center@$RATIO" "gazef@$RATIO")
ALL_ARMS=("${BASE_ARMS[@]}" "${FIX_ARMS[@]}")
TAG="$(basename "$POOL" .json)_${SELECT}_r${RADIUS/./}"

mkdir -p "$HERE/analysis/tables/question_query" "$HERE/results" "$HERE/logs"
LOG="$HERE/logs/recall_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log  : $LOG"
echo "pool : $POOL"
echo "arms : ${ALL_ARMS[*]}"
# Recorded because the s/clip numbers in this log are only comparable against another run on the
# same card, and an unset value means "every GPU on the box" -- worth seeing in the log, not
# discovering from nvidia-smi three hours in.
echo "gpu  : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: all GPUs>}"

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
  for A in "${BASE_ARMS[@]}"; do
    F="$GC/emb/$(echo "$A" | tr -d '.' | tr '@' '_').npz"
    if [ ! -f "$F" ]; then
      echo "FATAL: $F missing. Build the baseline arms in gaze_crop first (~2.7 h for $N_POOL clips):"
      echo "  python $GC/embed_arms.py --pool $POOL --arms $A --video-root $VIDEO_ROOT"
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
# The centroid arm was renamed fixsel_gazef -> fixsel_gaze. Its old npz, if still under the old name,
# is not silently reused as the new per-frame fixsel_gazef (embed_fix_arms.py would detect it from
# the meta and rebuild), but that throws ~1.3 h of centroid embeddings away. Move it first.
R_TAG="$(echo "$RATIO" | tr -d '.')"
OLD="$GC/emb/fixsel_gazef_$R_TAG.npz"
NEW="$GC/emb/fixsel_gaze_$R_TAG.npz"
if [ -f "$OLD" ] && [ ! -f "$NEW" ] && ! grep -q crop_centre "${OLD%.npz}.meta.json" 2>/dev/null; then
  echo "FATAL: $OLD holds centroid crops from before the rename (no crop_centre in its meta)."
  echo "       It is fixsel_gaze@$RATIO now. Move it, then rerun:"
  echo "  mv $OLD $NEW"
  echo "  mv ${OLD%.npz}.meta.json ${NEW%.npz}.meta.json"
  FATAL=1
fi
[ "$FATAL" = "1" ] && { echo; echo "aborting before spending GPU time"; exit 1; }

if [ "$SKIP_EMBED" != "1" ]; then
  echo; echo "=== embedding ${FIX_ARMS[*]} ==="
  "$PY" "$HERE/embed_fix_arms.py" --pool "$POOL" --gaze "$GAZE" --video-root "$VIDEO_ROOT" \
      --arms "${FIX_ARMS[@]}" --select "$SELECT" --radius "$RADIUS" || exit 1
fi

# One table per tolerance. Recall across tolerances is not comparable (more target clips lift
# every arm); each table carries its own chance row.
for T in $TOLS; do
  echo; echo "=== recall, tol ${T}s ==="
  "$PY" "$GC/recall_eval.py" --pool "$POOL" --arms "${ALL_ARMS[@]}" --target-tolerance-sec "$T" \
      --out "$HERE/results/recall_${TAG}_tol${T}.json" \
      --markdown "$HERE/analysis/tables/question_query/recall_${TAG}_tol${T}.md" || exit 1
done
echo; echo "figures: python $HERE/plot_recall_curves.py"

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
echo "done. tables: $HERE/analysis/tables/question_query/recall_${TAG}_tol{$(echo $TOLS | tr ' ' ',')}.md"
