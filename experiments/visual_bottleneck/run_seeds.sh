#!/usr/bin/env bash
# Two extra sampled runs of the single-shot conditions, so every condition has 3 samples:
#   run 0 = the existing results/ (B.json doubles as B_replay: identical final-answer prompt)
#   run 1 = results_seeds/seed1/   run 2 = results_seeds/seed2/
# results/ is never written. Resumable: rerun the same command and finished questions are skipped.
#
#   cd ~/WorldMM && source .venv/bin/activate
#   CUDA_VISIBLE_DEVICES=2 nohup bash experiments/visual_bottleneck/run_seeds.sh > /dev/null 2>&1 &
#   # when done (or any time, partial runs are reported as such):
#   python experiments/visual_bottleneck/analyze_seeds.py
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
SEEDS_DIR="$HERE/results_seeds"
LOG_DIR="$HERE/logs/seeds"
mkdir -p "$SEEDS_DIR" "$LOG_DIR"
cd "$REPO" || exit 1

# D vs B_replay first: that pair is the headline comparison, so it is complete even if time runs out.
RUNS="D:1 B_replay:1 D:2 B_replay:2 A:1 C:1 E_prime:1 A:2 C:2 E_prime:2"

for run in $RUNS; do
  cond="${run%%:*}"; seed="${run##*:}"
  out="$SEEDS_DIR/seed$seed"
  log="$LOG_DIR/${cond}_seed${seed}.log"
  echo "$(date '+%F %T')  $cond seed $seed -> $out/$cond.json  (log: $log)"
  "$PY" "$HERE/eval_egolife.py" --condition "$cond" --seed "$seed" \
      --results-dir "$out" --resume >> "$log" 2>&1 \
    || echo "  FAILED ($cond seed $seed), see $log; continuing"
done

echo "$(date '+%F %T')  all runs finished"
"$PY" "$HERE/analyze_seeds.py"
