import os

import torch
from datasets import load_dataset, load_from_disk

from config import config
from preprocess import transform_ds


def load_butterflies():
    if os.path.exists(config.dataset_path):
        return load_from_disk(config.dataset_path)
    dataset = load_dataset(config.dataset_name, split="train")
    dataset.save_to_disk(config.dataset_path)
    return dataset


def build_dataloader():
    dataset = transform_ds(load_butterflies())
    return torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True
    )
