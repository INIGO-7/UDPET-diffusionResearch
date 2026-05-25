"""Shared training engine for both pipelines.

The only thing that differs between Pipeline A (supervised) and Pipeline B
(unconditional) is how the model's input is built from each batch. That
divergence is isolated in the `prepare_model_input` callback, leaving the
optimization loop, EMA, mixed precision, checkpointing and logging
identical and centralized in this module.
"""

import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm


def resolve_resume_path(output_dir: Path, resume_setting: str) -> Path | None:
    """Resolve `resume_setting` against `output_dir`.

    Accepts either "latest" (newest checkpoint-epoch-* under output_dir) or a
    specific subdir name. Returns None if nothing matches.
    """
    if not output_dir.is_dir():
        return None
    if resume_setting == "latest":
        candidates = sorted(output_dir.glob("checkpoint-epoch-*"))
        return candidates[-1] if candidates else None
    candidate = output_dir / resume_setting
    return candidate if candidate.is_dir() else None


def _save_checkpoint(
    accelerator: Accelerator,
    model,
    ema: EMAModel | None,
    noise_scheduler,
    save_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    """Save inference-ready EMA weights AND full training state for resume.

    Layout under `save_dir`:
        unet/            EMA-applied U-Net (consumed by reconstruct/evaluate)
        scheduler/       noise scheduler config
        training_state/  accelerator.save_state output: live model params,
                         optimizer, lr scheduler, RNG, and EMA shadow params
                         (EMA is registered with the accelerator below)
        metadata.json    epoch + global_step needed to resume the loop
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if ema is not None:
        ema.store(unwrapped.parameters())
        ema.copy_to(unwrapped.parameters())
    unwrapped.save_pretrained(save_dir / "unet")
    noise_scheduler.save_pretrained(save_dir / "scheduler")
    if ema is not None:
        ema.restore(unwrapped.parameters())

    accelerator.save_state(str(save_dir / "training_state"))
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step}, f)


def run_training(
    cfg,
    model,
    noise_scheduler,
    train_loader,
    output_dir: Path,
    prepare_model_input: Callable[[dict, torch.Tensor], torch.Tensor],
    tracker_name: str,
    resume_from: Path | None = None,
    preview_sampler: Callable[[torch.nn.Module], dict[str, torch.Tensor]] | None = None,
    preview_references: dict[str, torch.Tensor] | None = None,
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
        resume_from: directory of a prior checkpoint (containing training_state/ and
            metadata.json) to resume from. None starts fresh.
        preview_sampler: optional callable invoked at each checkpoint with the
            EMA-applied unwrapped U-Net in eval mode. Must return a dict mapping
            TensorBoard tag -> CHW image tensor (values roughly in [0, 1]).
        preview_references: optional dict of fixed reference images logged once
            at the first preview step (e.g. ground-truth full + low-dose inputs).
    """
    # Free speedup for any residual fp32 matmuls on Ampere+/Blackwell GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    # Register EMA so its shadow params are part of accelerator.save_state/load_state.
    if ema is not None:
        accelerator.register_for_checkpointing(ema)

    first_epoch = 0
    global_step = 0
    if resume_from is not None:
        accelerator.load_state(str(resume_from / "training_state"))
        with open(resume_from / "metadata.json") as f:
            meta = json.load(f)
        first_epoch = meta["epoch"] + 1
        global_step = meta["global_step"]
        if accelerator.is_main_process:
            accelerator.print(
                f"Resumed from {resume_from} (epoch {meta['epoch']}, step {global_step})"
            )

    preview_refs_logged = False

    for epoch in range(first_epoch, cfg.train.num_epochs):
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
                _save_checkpoint(
                    accelerator, model, ema, noise_scheduler, save_dir, epoch, global_step
                )
                accelerator.print(f"Saved checkpoint to {save_dir}")

                if preview_sampler is not None:
                    writer = accelerator.get_tracker("tensorboard", unwrap=True)
                    if not preview_refs_logged and preview_references is not None:
                        for tag, img in preview_references.items():
                            writer.add_image(
                                tag, img.clamp(0, 1), global_step, dataformats="CHW"
                            )
                        preview_refs_logged = True

                    # Mirror the EMA swap from _save_checkpoint so previews use
                    # the same weights that will be loaded at inference time.
                    unwrapped = accelerator.unwrap_model(model)
                    if ema is not None:
                        ema.store(unwrapped.parameters())
                        ema.copy_to(unwrapped.parameters())
                    was_training = unwrapped.training
                    unwrapped.eval()

                    # Isolate preview RNG so the deterministic init noise reused
                    # across checkpoints doesn't perturb the training trajectory.
                    cpu_state = torch.random.get_rng_state()
                    cuda_state = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    )
                    torch.manual_seed(cfg.train.seed)
                    try:
                        images = preview_sampler(unwrapped)
                    finally:
                        torch.random.set_rng_state(cpu_state)
                        if cuda_state is not None:
                            torch.cuda.set_rng_state_all(cuda_state)
                        if was_training:
                            unwrapped.train()
                        if ema is not None:
                            ema.restore(unwrapped.parameters())

                    for tag, img in images.items():
                        writer.add_image(
                            tag, img.clamp(0, 1), global_step, dataformats="CHW"
                        )
                    writer.flush()

    accelerator.end_training()
