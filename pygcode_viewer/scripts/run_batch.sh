#!/bin/bash
# Run batch visualization on the full dataset
#
# Usage:
#   ./run_batch.sh              # Process all images
#   ./run_batch.sh --limit 10   # Limit to 10 images per folder (for testing)
#   ./run_batch.sh --dry-run    # Preview what would be processed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYGCODE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PYGCODE_DIR"

# Activate virtual environment
source venv/bin/activate

echo "Running batch visualization..."
echo "This will process 500 images across 4 folders."
echo "Estimated time: ~4 hours (30s per image)"
echo ""

# Run the batch script with all arguments passed through
python scripts/batch_visualize.py "$@"

echo ""
echo "Done! Output saved to: $PYGCODE_DIR/outputs/batch/"
