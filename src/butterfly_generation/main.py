from datasets import load_dataset, load_from_disk
import os
import torch

from config import config
from preprocess import transform_ds

if os.path.exists(config.dataset_path):
    dataset = load_from_disk(config.dataset_path)
else:
    dataset = load_dataset(config.dataset_name, split="train")
    dataset.save_to_disk(config.dataset_path)

transformed_ds = transform_ds(dataset)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)