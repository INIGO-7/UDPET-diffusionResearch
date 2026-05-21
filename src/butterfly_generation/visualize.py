from datasets import load_dataset, load_from_disk
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from model import noise_scheduler
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

timesteps = torch.LongTensor([50])

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for i, image in enumerate(dataset[:4]["image"]):
    axs[0, i].imshow(image)
    axs[0, i].set_axis_off()

for i, sample in enumerate(transformed_ds[:4]["images"]):
    sample_image = sample.unsqueeze(0)
    noise = torch.randn(sample_image.shape)
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
    noisy_array = ((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]
    axs[1, i].imshow(noisy_array)
    axs[1, i].set_axis_off()

axs[0, 0].set_title("Original")
axs[1, 0].set_title("Noisy (t=50)")

plt.tight_layout()
plt.show()
