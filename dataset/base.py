import random
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor


def _text_values(entry):
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, bytes):
        return [entry.decode("utf-8")]
    if isinstance(entry, (tuple, list)):
        return [text for value in entry for text in _text_values(value)]
    if isinstance(entry, np.ndarray):
        return [text for value in entry.flat for text in _text_values(value)]
    return []


class BaseDataset(Dataset):
    """Return an image, token IDs, its multi-label target, and its split-local index."""

    def __init__(
        self, captions, indexs, labels, *, is_train=True, maxWords=77,
        imageResolution=224, npy=False, image_root=None,
    ):
        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.npy = npy
        self.image_root = Path(image_root) if image_root is not None else None
        self.maxWords = maxWords
        resize = Resize(
            imageResolution if is_train else (imageResolution, imageResolution),
            interpolation=InterpolationMode.BICUBIC,
        )
        transforms = [resize]
        if is_train:
            transforms.append(CenterCrop(imageResolution))
        transforms += [
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.indexs)

    def _load_image(self, index):
        if self.npy:
            return self.transform(Image.fromarray(self.indexs[index]).convert("RGB"))
        entries = _text_values(self.indexs[index])
        if len(entries) != 1 or not entries[0].strip():
            raise ValueError(f"Expected one image path for sample {index}")
        path = Path(entries[0].strip())
        if self.image_root is not None and not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def _load_text(self, index):
        captions = _text_values(self.captions[index])
        if not captions:
            raise ValueError(f"Missing caption for sample {index}")
        caption = captions[random.randrange(len(captions))]
        return clip.tokenize(caption, context_length=self.maxWords, truncate=True)[0].long()

    def __getitem__(self, index):
        return self._load_image(index), self._load_text(index), torch.from_numpy(self.labels[index]), index

    def get_all_label(self):
        return torch.tensor(self.labels, dtype=torch.float32)
