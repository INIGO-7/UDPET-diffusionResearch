"""Shared training engine for both pipelines.

The only thing that differs between Pipeline A (supervised) and Pipeline B
(unconditional) is how the model's input is built from each batch. That
divergence is isolated in the `prepare_model_input` callback, leaving the
optimization loop, EMA, mixed precision, checkpointing and logging
identical and centralized in this module.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm


def _save_checkpoint(
    accelerator: Accelerator,
    model,
    ema: EMAModel | None,
    noise_scheduler,
    save_dir: Path,
) -> None:
    """Save EMA-applied weights + scheduler config. Non-EMA weights are restored after."""
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if ema is not None:
        ema.store(unwrapped.parameters())
        ema.copy_to(unwrapped.parameters())
    unwrapped.save_pretrained(save_dir / "unet")
    noise_scheduler.save_pretrained(save_dir / "scheduler")
    if ema is not None:
        ema.restore(unwrapped.parameters())


def run_training(
    cfg,
    model,
    noise_scheduler,
    train_loader,
    output_dir: Path,
    prepare_model_input: Callable[[dict, torch.Tensor], torch.Tensor],
    tracker_name: str,
) -> None:
    """Run the v-prediction training loop with EMA, accumulation and tensorboard.

    Args:
        cfg: a SupervisedConfig or UnconditionalConfig.
        model: a diffusers UNet2DModel.
        noise_scheduler: a DDPMScheduler (configured for v-prediction).
        train_loader: yields dicts with at least "full" (the clean target). Pipeline A's
            loader also yields "low".
        output_dir: where checkpoints and logs are written.
        prepare_model_input: builds the actual U-Net input from (batch, noisy_full).
            For Pipeline A: torch.cat([noisy_full, batch["low"]], dim=1).
            For Pipeline B: just noisy_full.
        tracker_name: tensorboard run name.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,
        num_training_steps=len(train_loader) * cfg.train.num_epochs,
    )

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(output_dir / "logs"),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(tracker_name)

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    ema = (
        EMAModel(model.parameters(), decay=cfg.train.ema_decay)
        if cfg.train.use_ema
        else None
    )

    global_step = 0
    for epoch in range(cfg.train.num_epochs):
        progress = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress.set_description(f"Epoch {epoch}")

        for batch in train_loader:
            clean = batch["full"]
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=clean.device,
            ).long()

            noisy = noise_scheduler.add_noise(clean, noise, timesteps)
            model_input = prepare_model_input(batch, noisy)
            v_target = noise_scheduler.get_velocity(clean, noise, timesteps)

            with accelerator.accumulate(model):
                v_pred = model(model_input, timesteps, return_dict=False)[0]
                loss = F.mse_loss(v_pred, v_target)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if ema is not None and accelerator.sync_gradients:
                    ema.step(model.parameters())

            progress.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # End-of-epoch checkpoint
        if accelerator.is_main_process:
            is_last = epoch == cfg.train.num_epochs - 1
            if (epoch + 1) % cfg.train.save_model_epochs == 0 or is_last:
                save_dir = output_dir / f"checkpoint-epoch-{epoch:03d}"
                _save_checkpoint(accelerator, model, ema, noise_scheduler, save_dir)
                accelerator.print(f"Saved checkpoint to {save_dir}")

    accelerator.end_training()
