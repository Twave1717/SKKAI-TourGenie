#!/bin/bash

# =============================================================================
# End-to-End Persona Augmentation Pipeline
# =============================================================================
#
# This script runs the complete 3-stage persona augmentation pipeline:
# - Stage 1: MMR-based persona retrieval (k×10 candidates)
# - Stage 1.5: LLM alpha survey (importance scores)
# - Stage 1.7: Solvable conflict selection (final N personas)
#
# API Key:
#   Automatically loaded from KY/aug_stage/.env file (no manual setup needed)
#
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   --mode <batch|iterative>  Pipeline mode (default: batch)
#   --max_records <N>         Limit to N records (default: 0 = all)
#   --model <model>           LLM model (default: gpt-4.1)
#   --workers <N>             Stage 1 parallel workers (default: 8)
#   --skip_stage1             Skip Stage 1 if already done
#   --help                    Show this help message
#
# Examples:
#   # Run full pipeline (batch mode, all 1000 records)
#   ./run.sh
#
#   # Test with 10 records (iterative mode)
#   ./run.sh --mode iterative --max_records 10
#
#   # Skip Stage 1, only run Stage 1.5+1.7
#   ./run.sh --skip_stage1 --max_records 100
#
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

# Default parameters
MODE="batch"
MAX_RECORDS=0
MODEL="gpt-4.1"
WORKERS=8
SKIP_STAGE1=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --max_records)
            MAX_RECORDS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --skip_stage1)
            SKIP_STAGE1=true
            shift
            ;;
        --help)
            sed -n '3,36p' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run './run.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "batch" && "$MODE" != "iterative" ]]; then
    echo "Error: --mode must be 'batch' or 'iterative'"
    exit 1
fi

# =============================================================================
# Environment Check
# =============================================================================

echo "========================================================================"
echo "  Persona Augmentation Pipeline"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Mode: $MODE"
echo "  - Max records: $([ $MAX_RECORDS -eq 0 ] && echo 'all (1000)' || echo $MAX_RECORDS)"
echo "  - Model: $MODEL"
echo "  - Workers: $WORKERS"
echo "  - Skip Stage 1: $SKIP_STAGE1"
echo ""

# Load .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    set +u  # Temporarily disable undefined variable check
    # Export variables from .env (skip comments and empty lines)
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)
    set -u  # Re-enable undefined variable check
    echo "✓ Environment variables loaded"
    echo ""
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo ""
fi

# Check API key
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo ""
    echo "Please either:"
    echo "  1. Add OPENAI_API_KEY to $ENV_FILE"
    echo "  2. Set it manually: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "✓ OPENAI_API_KEY found"
echo ""

# =============================================================================
# Stage 1: MMR-based Persona Retrieval
# =============================================================================

if [[ "$SKIP_STAGE1" == false ]]; then
    echo "========================================================================"
    echo "  Stage 1: MMR-based Persona Retrieval"
    echo "========================================================================"
    echo ""
    echo "Retrieving k×10 candidate personas from Stravl dataset..."
    echo ""

    STAGE1_ARGS="--split test --write_meta"

    if [[ $MAX_RECORDS -gt 0 ]]; then
        STAGE1_ARGS="$STAGE1_ARGS --max_records $MAX_RECORDS"
    fi

    if [[ $WORKERS -gt 1 ]]; then
        STAGE1_ARGS="$STAGE1_ARGS --workers $WORKERS"
    fi

    python3 run_stage1.py $STAGE1_ARGS

    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Stage 1 completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Stage 1 failed"
        exit 1
    fi
else
    echo "========================================================================"
    echo "  Stage 1: SKIPPED"
    echo "========================================================================"
    echo ""
    echo "Using existing Stage 1 outputs from outputs/stage1/test/data/"
    echo ""
fi

# =============================================================================
# Stage 1.5 + 1.7: Mode Selection
# =============================================================================

if [[ "$MODE" == "batch" ]]; then
    # =========================================================================
    # Batch Mode: Stage 1.5 → Stage 1.7 (Sequential)
    # =========================================================================

    echo "========================================================================"
    echo "  Stage 1.5: LLM Alpha Survey (Batch API)"
    echo "========================================================================"
    echo ""
    echo "Submitting batch job to OpenAI..."
    echo "  - Model: $MODEL"
    echo "  - Cost: ~\$53.29 for 9,200 personas (63% savings)"
    echo "  - Processing time: 0-24 hours"
    echo ""

    STAGE1_5_ARGS="--model $MODEL"

    if [[ $MAX_RECORDS -gt 0 ]]; then
        STAGE1_5_ARGS="$STAGE1_5_ARGS --max_records $MAX_RECORDS"
    fi

    python3 run_stage1_5.py $STAGE1_5_ARGS

    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Stage 1.5 completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Stage 1.5 failed"
        exit 1
    fi

    echo "========================================================================"
    echo "  Stage 1.7: Solvable Conflict Selection"
    echo "========================================================================"
    echo ""
    echo "Selecting final N personas with solvable conflicts..."
    echo ""

    STAGE1_7_ARGS="--write_meta"

    if [[ $MAX_RECORDS -gt 0 ]]; then
        STAGE1_7_ARGS="$STAGE1_7_ARGS --max_records $MAX_RECORDS"
    fi

    python3 run_stage1_7.py $STAGE1_7_ARGS

    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Stage 1.7 completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Stage 1.7 failed"
        exit 1
    fi

    # Output summary
    STAGE1_5_DIR="outputs/stage1_5/test/data"
    STAGE1_7_DIR="outputs/stage1_7/test/data"

else
    # =========================================================================
    # Iterative Mode: Stage 1.5 + 1.7 (Integrated)
    # =========================================================================

    echo "========================================================================"
    echo "  Stage 1.5 + 1.7: Iterative Mode (Integrated)"
    echo "========================================================================"
    echo ""
    echo "Running iterative pipeline (process 1 persona per round)..."
    echo "  - Model: $MODEL"
    echo "  - Cost: \$11.58 - \$34.75 (best case 92% savings)"
    echo "  - Processing time: 1-3 hours"
    echo ""

    ITERATIVE_ARGS="--model $MODEL --write_meta"

    if [[ $MAX_RECORDS -gt 0 ]]; then
        ITERATIVE_ARGS="$ITERATIVE_ARGS --max_records $MAX_RECORDS"
    fi

    python3 run_stage1_5_1_7_iterative.py $ITERATIVE_ARGS

    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Iterative pipeline completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Iterative pipeline failed"
        exit 1
    fi

    # Output summary
    STAGE1_5_DIR="outputs/stage1_5_iterative/test/data"
    STAGE1_7_DIR="outputs/stage1_7_iterative/test/data"
fi

# =============================================================================
# Final Summary
# =============================================================================

echo "========================================================================"
echo "  Pipeline Completed Successfully!"
echo "========================================================================"
echo ""
echo "Outputs:"
echo "  - Stage 1:   outputs/stage1/test/data/"
echo "  - Stage 1.5: $STAGE1_5_DIR"
echo "  - Stage 1.7: $STAGE1_7_DIR"
echo ""

# Count outputs
STAGE1_COUNT=$(find outputs/stage1/test/data -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
STAGE1_5_COUNT=$(find "$STAGE1_5_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
STAGE1_7_COUNT=$(find "$STAGE1_7_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')

echo "Statistics:"
echo "  - Stage 1 outputs:   $STAGE1_COUNT files"
echo "  - Stage 1.5 outputs: $STAGE1_5_COUNT files"
echo "  - Stage 1.7 outputs: $STAGE1_7_COUNT files (final)"
echo ""

# Sample output
if [[ $STAGE1_7_COUNT -gt 0 ]]; then
    SAMPLE_FILE=$(find "$STAGE1_7_DIR" -name "*.json" | head -1)
    echo "Sample output:"
    echo "  $SAMPLE_FILE"
    echo ""
    echo "Content preview:"
    head -20 "$SAMPLE_FILE" | sed 's/^/  /'
    echo "  ..."
    echo ""
fi

echo "Next steps:"
echo "  1. Validate outputs: python validate_outputs.py"
echo "  2. Analyze conflicts: python analyze_conflicts.py"
echo "  3. Run experiments: python run_experiments.py"
echo ""
echo "Done! 🎉"
