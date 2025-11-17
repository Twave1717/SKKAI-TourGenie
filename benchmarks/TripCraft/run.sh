#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRIPCRAFT_DB_ROOT="${TRIPCRAFT_DB_ROOT:-$REPO_DIR/TripCraft_database}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output}"
export MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"     # qwen / phi4 also supported
export OPENAI_API_KEY="${OPENAI_API_KEY}"
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"
export DAY="${DAY:-3day}"                                           # 3day/5day/7day
export SET_TYPE="${SET_TYPE:-3day_gpt4o_orig}"
export STRATEGY="${STRATEGY:-direct_og}"                            # direct_og / direct_param
export CSV_FILE="${CSV_FILE:-$REPO_DIR/tripcraft/tripcraft_3day.csv}"

mkdir -p "$OUTPUT_DIR"

cd "$REPO_DIR/tools/planner"

python sole_planning_mltp.py \
    --day $DAY \
    --set_type $SET_TYPE \
    --output_dir $OUTPUT_DIR \
    --csv_file $CSV_FILE \
    --model_name $MODEL_NAME \
    --strategy $STRATEGY
