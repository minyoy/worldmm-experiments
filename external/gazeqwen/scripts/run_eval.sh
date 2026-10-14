#!/bin/bash
# usage: run_eval.sh MODE TAG [extra eval.py args...]
cd /home/minyoy/gazeqwen
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-3}
export PYTHONPATH=".:third_party/vjepa2" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=$1; TAG=$2; shift 2; mkdir -p results
/home/minyoy/miniconda3/envs/gazeqwen/bin/python -u -m gazeqwen.eval --mode $MODE \
  --qa_files $(ls dataset/qa/*.json) --video_root dataset/videos --fixation_root dataset/fixations \
  --split_file splits/test_egtea_egoexo.json --split test \
  --model_path /home/minyoy/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --checkpoint checkpoints/gazeqwen/best_model.pt --vjepa_checkpoint checkpoints/vjepa2_1_vitb_dist_vitG_384.pt \
  --output_file results/${TAG}_test38.json --resume "$@"
