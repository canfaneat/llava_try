#!/bin/bash

SPLIT="mmbench_dev_20230712"

# Extract model path argument for naming output files
MODEL_PATH_ARG=$(echo "$@" | grep -oP '(?<=--model-path )[^ ]+')
CKPT=$(basename "$MODEL_PATH_ARG") # Extract the last part of the path (e.g., llava-v1.5-7b)

# Run the evaluation using all arguments passed to the script
python -m llava.eval.model_vqa_mmbench \
    --question-file /kaggle/input/mmbench/mmbench_dev_20230712.tsv \
    --answers-file /kaggle/working/output/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    "$@" # Pass all script arguments
# Create directory for upload based on the checkpoint name
UPLOAD_DIR="./playground/data/eval/mmbench/answers_upload/$SPLIT"
mkdir -p "$UPLOAD_DIR"

# Run the conversion script using the extracted checkpoint name
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir "$UPLOAD_DIR" \
    --experiment "$CKPT"
