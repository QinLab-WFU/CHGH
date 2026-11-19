from pathlib import Path

import clip
import torch
from torch import nn


class LinearHash(nn.Module):
    def __init__(self, inputDim=2048, outputDim=64):
        super().__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, features):
        return torch.tanh(self.drop_out(self.fc(features)))


class DHaPH(nn.Module):
    """CLIP backbone with separate linear hash heads for images and text."""

    def __init__(self, outputDim=64, clipPath="ViT-B-32.pt"):
        super().__init__()
        checkpoint = Path(clipPath)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint}")
        self.clip, _ = clip.load(str(checkpoint), device="cpu", jit=False)
        embedding_dim = self.clip.text_projection.shape[1]
        self.image_hash = LinearHash(embedding_dim, outputDim)
        self.text_hash = LinearHash(embedding_dim, outputDim)
        self.train()

    def encode_image(self, image):
        return self.image_hash(self.clip.encode_image(image))

    def encode_text(self, text):
        return self.text_hash(self.clip.encode_text(text))

    def forward(self, image, text):
        return self.encode_image(image), self.encode_text(text)
