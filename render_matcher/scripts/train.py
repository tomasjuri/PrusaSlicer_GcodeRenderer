#!/usr/bin/env python3
"""
Train Render Matcher CNN.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --device cuda
    python scripts/train.py --resume runs/run_20240101_120000/best.pt
    
TensorBoard:
    tensorboard --logdir runs/
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from render_matcher.trainer import main

if __name__ == "__main__":
    main()
