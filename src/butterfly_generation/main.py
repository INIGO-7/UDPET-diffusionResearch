import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import config
from data import build_dataloader
from model import build_model, build_noise_scheduler
from train import train_loop


def main():
    train_dataloader = build_dataloader()
    model = build_model()
    noise_scheduler = build_noise_scheduler()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)


if __name__ == "__main__":
    main()
