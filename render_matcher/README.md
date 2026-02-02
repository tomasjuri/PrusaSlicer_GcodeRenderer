# Render Matcher

CNN-based render vs reality image matching for 3D print verification.

## Overview

This project implements a Siamese neural network to compare rendered G-code visualizations against real camera captures of 3D prints. The network outputs a similarity score (0-1) that can be thresholded to determine if the print matches the expected render.

## Architecture

```
Source Image ──→ [ResNet Backbone] ──→ [Embedding Head] ──→ embedding_a ─┐
                                                                          ├──→ [Comparison MLP] ──→ score
Render Image ──→ [ResNet Backbone] ──→ [Embedding Head] ──→ embedding_b ─┘
                 (shared weights)
```

- **Backbone**: Pretrained ResNet-18/34/50 or EfficientNet-B0
- **Embedding**: 256-dimensional L2-normalized vectors
- **Comparison**: Concatenation + difference + product → MLP → Sigmoid

## Installation

```bash
cd render_matcher
pip install -e .
```

## Data Format

The training data should be organized as follows:

```
pygcode_viewer/outputs/batch/
├── session1/
│   ├── img_ZN100_Z20.2_X90_Y130_Z70_source.jpg
│   ├── img_ZN100_Z20.2_X90_Y130_Z70_render.png
│   └── ...
└── session2/
    └── ...
```

- `*_source.jpg`: Real camera capture
- `*_render.png`: Rendered G-code visualization

## Training

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train on specific device
python scripts/train.py --config configs/default.yaml --device cuda

# Resume from checkpoint
python scripts/train.py --resume runs/run_20240101_120000/best.pt
```

## Configuration

Edit `configs/default.yaml` to customize:

- **model**: Backbone, embedding dimension, pretrained weights
- **data**: Data path, patch size, negative sample settings
- **training**: Batch size, epochs, learning rate, threshold
- **augmentations**: Flip, rotation, brightness/contrast
- **logging**: TensorBoard settings, sample grid size

## Monitoring

```bash
# Start TensorBoard
tensorboard --logdir runs/
```

TensorBoard logs include:
- Training/validation loss and accuracy curves
- Precision, recall, F1 metrics
- 10x10 sample pairs grid
- Prediction visualizations with scores

## Outputs

Training outputs are saved to `runs/run_TIMESTAMP/`:
- `config.yaml`: Training configuration
- `samples_grid.png`: Visualization of training samples
- `best.pt`: Best model checkpoint
- `final.pt`: Final model checkpoint
- TensorBoard event files

## Usage (Inference)

```python
import torch
from render_matcher import SiameseNetwork, load_config

# Load model
config = load_config("configs/default.yaml")
model = SiameseNetwork(config.model)
model.load_state_dict(torch.load("runs/best.pt")["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    score = model(source_tensor, render_tensor)
    match = score.item() >= config.training.threshold
```

## License

AGPL-3.0-or-later
