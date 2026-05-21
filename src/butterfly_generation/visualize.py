import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler

from data import load_butterflies
from preprocess import transform_ds

dataset = load_butterflies()
original_images = list(dataset[:4]["image"])  # capture PIL images before set_transform rewrites the accessor

transformed_ds = transform_ds(dataset)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

timesteps = torch.LongTensor([50])

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for i, image in enumerate(original_images):
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
