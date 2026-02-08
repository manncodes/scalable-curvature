#!/bin/bash
# Setup script for scalable-curvature data preparation
#
# Usage:
#   ./setup_data.sh                    # Prepare FineWeb-Edu (sample-10BT, ~10B tokens)
#   ./setup_data.sh --subset sample-100BT  # Use larger subset (~100B tokens)
#   ./setup_data.sh --skip-env         # Skip conda environment setup
#
# This script:
#   1. Creates/activates conda environment (optional)
#   2. Installs dependencies
#   3. Downloads and tokenizes FineWeb-Edu dataset
#   4. Creates train.bin and val.bin in data/fineweb/

set -e

# Parse arguments
SKIP_ENV=false
SUBSET="sample-10BT"
NUM_PROC=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --num-proc)
            NUM_PROC="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Scalable Curvature - Data Setup"
echo "============================================"
echo ""

# Step 1: Environment setup
if [ "$SKIP_ENV" = false ]; then
    echo "[1/3] Setting up conda environment..."

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please install conda or use --skip-env"
        exit 1
    fi

    # Create environment if it doesn't exist
    if ! conda env list | grep -q "^curvature "; then
        echo "Creating conda environment 'curvature'..."
        conda create -n curvature python=3.10 -y
    else
        echo "Conda environment 'curvature' already exists"
    fi

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate curvature

    echo "Installing dependencies..."
    pip install torch numpy pandas scipy wandb tqdm transformers tiktoken datasets
else
    echo "[1/3] Skipping environment setup (--skip-env)"
fi

echo ""
echo "[2/3] Checking dependencies..."

# Verify required packages
python -c "import tiktoken; import datasets; import numpy; print('Dependencies OK')" || {
    echo "Error: Missing dependencies. Run without --skip-env or install manually:"
    echo "  pip install torch numpy pandas scipy wandb tqdm transformers tiktoken datasets"
    exit 1
}

echo ""
echo "[3/3] Preparing FineWeb-Edu dataset (${SUBSET})..."
echo "This may take a while depending on your internet connection..."
echo ""

# Build command
CMD="python data/fineweb/prepare.py --dataset_subset $SUBSET"
if [ -n "$NUM_PROC" ]; then
    CMD="$CMD --num_proc $NUM_PROC"
fi

# Run data preparation
$CMD

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Data files created:"
ls -lh data/fineweb/*.bin 2>/dev/null || echo "  (no .bin files found)"
echo ""
echo "To start training:"
echo "  python demo_nanogpt.py"
echo ""
echo "To use split LLaMA architecture:"
echo "  python demo_nanogpt.py --abc llama_split \\"
echo "      --path_8b /path/to/llama-8b \\"
echo "      --path_70b /path/to/llama-70b"
echo "============================================"
