from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset


def split_data(captions, indexs, labels, query_num=5000, train_num=10000, seed=None):
    if not len(captions) == len(indexs) == len(labels):
        raise ValueError("Caption, image index, and label counts must agree")
    if query_num < 1 or train_num < 1 or query_num + train_num > len(indexs):
        raise ValueError("Query and training sizes must be positive and fit in the dataset")
    order = np.random.RandomState(seed).permutation(len(indexs))
    split = (order[:query_num], order[query_num:query_num + train_num], order[query_num:])
    return tuple(tuple(array[indices] for indices in split) for array in (indexs, captions, labels))


def dataloader(
    captionFile, indexFile, labelFile, maxWords=77, imageResolution=224,
    query_num=5000, train_num=10000, seed=None, npy=False, *, image_root=None,
):
    """Load metadata and return datasets in the order (train, query, retrieval)."""
    suffix = Path(captionFile).suffix.lower()
    if suffix == ".mat":
        captions = loadmat(captionFile)["caption"]
    elif suffix == ".txt":
        captions = np.asarray(Path(captionFile).read_text(encoding="utf-8").splitlines())
    else:
        raise ValueError("Caption file must be .mat or .txt")
    labels = loadmat(labelFile)["category"]
    indexs = np.load(indexFile, allow_pickle=False) if npy else loadmat(indexFile)["index"]
    if len(labels) > 1 and captions.ndim == 2 and captions.shape == (1, len(labels)):
        captions = captions[0]
    if not npy and len(labels) > 1 and indexs.ndim == 2 and indexs.shape == (1, len(labels)):
        indexs = indexs[0]
    split_indexs, split_captions, split_labels = split_data(
        captions, indexs, labels, query_num=query_num, train_num=train_num, seed=seed,
    )
    return tuple(
        BaseDataset(
            split_captions[part], split_indexs[part], split_labels[part],
            is_train=part == 1, maxWords=maxWords, imageResolution=imageResolution,
            npy=npy, image_root=image_root,
        )
        for part in (1, 0, 2)
    )
