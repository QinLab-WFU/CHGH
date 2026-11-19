# CHGH

This repository provides a preview of the CHGH project for cross-modal hashing.
It contains the data loader, the CLIP-based hashing model, and a training
scaffold. The full source code will be made publicly available if the paper is
accepted.

## Contents

| Path | Description |
| --- | --- |
| `dataset/base.py` | Paired image and text loading, CLIP preprocessing, and multi-label targets. |
| `dataset/dataloader.py` | Metadata loading and seeded train, query, and retrieval splits. |
| `model/hash_model_a.py` | CLIP checkpoint loading and separate image and text hash heads. |
| `train/hash_train_IDGH_CGF.py` | Standard training loop with an explicit base-only mode. |

## Using the preview

Install PyTorch and TorchVision for your platform, then run:

```bash
python -m pip install -r requirements.txt
```

The loader accepts a MATLAB caption file with a `caption` field or a text file
with one caption per line, a MATLAB image index with an `index` field, and a
MATLAB label file with a `category` field. Keep all three files in the same
sample order. For relative image paths in the index, pass `image_root` to the
loader. The training split is a subset of the retrieval database, and the query
split is disjoint from both. The dataset files themselves are not distributed
here.

```python
from dataset import dataloader
from model import DHaPH

train, query, retrieval = dataloader(
    "data/caption.mat", "data/index.mat", "data/label.mat",
    image_root="data/images", maxWords=77, imageResolution=224,
    query_num=5000, train_num=10000, seed=1814,
)
model = DHaPH(outputDim=32, clipPath="ViT-B-32.pt")
```

Use a local [OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/32 checkpoint
from a trusted source. The example initializes the hash heads; the checkpoint
contains the backbone weights.

The preview does not include CHGH's generation and feedback implementation, a
training entry point, or the original retrieval evaluation utilities. The
training scaffold requires a caller-supplied objective and optimizer.
Its explicit base-only mode is provided for inspecting the training flow;
it does not reproduce the CHGH results reported in the paper.
