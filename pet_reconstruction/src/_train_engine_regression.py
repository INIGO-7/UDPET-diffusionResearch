"""Shared training engine for the regression baselines (no diffusion).

This mirrors `_train_engine.run_training` — same Accelerator setup, EMA,
gradient accumulation, checkpointing, resume and TensorBoard previews — but the
inner loop is a plain supervised regression: predict the clean full-dose slice
from the low-dose input and minimize MSE in the asinh [-1, 1] model space. There
is no noise scheduler, no timestep sampling and no iterative sampling.

The model-specific pieces are isolated in three callbacks so the same engine
serves both Baseline A (the diffusion U-Net as a regressor) and a future
Baseline B (a plain CNN such as RED-CNN):
    prepare_model_input(batch) -> Tensor    build the network input from a batch
    model_forward(model, x)    -> Tensor    run the network (e.g. add a constant
                                            timestep for the diffusers U-Net)
    save_model_fn(model, dir)               persist inference weights
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


def _default_save_model(model, save_dir: Path) -> None:
    """Default: a diffusers model saved under `unet/` (matches the diffusion layout)."""
    model.save_pretrained(save_dir / "unet")


def _save_checkpoint(
    accelerator: Accelerator,
    model,
    ema: EMAModel | None,
    save_dir: Path,
    epoch: int,
    global_step: int,
    save_model_fn: Callable,
) -> None:
    """Save inference-ready EMA weights AND full training state for resume.

    Layout mirrors the diffusion engine minus the `scheduler/` subdir (a regressor
    has no noise scheduler):
        unet/            EMA-applied weights (consumed by evaluate/reconstruct)
        training_state/  accelerator.save_state: live params, optimizer, lr
                         scheduler, RNG, and the EMA shadow params
        metadata.json    epoch + global_step needed to resume the loop
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if ema is not None:
        ema.store(unwrapped.parameters())
        ema.copy_to(unwrapped.parameters())
    save_model_fn(unwrapped, save_dir)
    if ema is not None:
        ema.restore(unwrapped.parameters())

    accelerator.save_state(str(save_dir / "training_state"))
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step}, f)


def run_regression_training(
    cfg,
    model,
    train_loader,
    output_dir: Path,
    prepare_model_input: Callable[[dict], torch.Tensor],
    model_forward: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    tracker_name: str,
    resume_from: Path | None = None,
    save_model_fn: Callable = _default_save_model,
    preview_sampler: Callable[[torch.nn.Module], dict[str, torch.Tensor]] | None = None,
    preview_references: dict[str, torch.Tensor] | None = None,
    validation_fn: Callable[[torch.nn.Module], dict[str, float]] | None = None,
) -> None:
    """Run the MSE regression training loop with EMA, accumulation and tensorboard.

    Args:
        cfg: a RegressionUNetConfig (or any config exposing the same `.train`/`.data`).
        model: the regression network.
        train_loader: yields dicts with "full" (clean target) and "low" (input).
        output_dir: where checkpoints and logs are written.
        prepare_model_input: builds the network input from a batch (e.g. batch["low"]).
        model_forward: runs the network on that input and returns the prediction.
        tracker_name: tensorboard run name.
        resume_from: directory of a prior checkpoint to resume from, or None.
        save_model_fn: persists the (EMA-applied) inference weights for a checkpoint.
        preview_sampler: optional callable invoked at each checkpoint with the
            EMA-applied unwrapped model in eval mode; returns {tag: CHW image}.
        preview_references: optional fixed reference images logged once.
        validation_fn: optional callable invoked at each checkpoint with the
            EMA-applied unwrapped model in eval mode; returns {metric_name ->
            value}, each logged to TensorBoard under "val/<metric_name>". For the
            regressors the training loss already tracks convergence, but these
            task metrics make the baselines directly comparable to the diffusion
            pipelines on the same axes.
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

    # Forward-only compiled handle. It SHARES parameters with `model`, so EMA,
    # the optimizer, gradient clipping, save_state/load_state and save_pretrained
    # all keep operating on `model` (clean state-dict keys, resume unaffected);
    # only the training forward goes through the compiled graph.
    forward_model = torch.compile(model) if getattr(cfg.train, "use_compile", False) else model

    ema = (
        EMAModel(model.parameters(), decay=cfg.train.ema_decay)
        if cfg.train.use_ema
        else None
    )
    if ema is not None:
        accelerator.register_for_checkpointing(ema)

    first_epoch = 0
    global_step = 0
    if resume_from is not None:
        accelerator.load_state(str(resume_from / "training_state"))
        if ema is not None:
            ema.to(accelerator.device)
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
            target = batch["full"]
            model_input = prepare_model_input(batch)

            with accelerator.accumulate(model):
                pred = model_forward(forward_model, model_input)
                loss = F.mse_loss(pred, target)
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
                    accelerator, model, ema, save_dir, epoch, global_step, save_model_fn
                )
                accelerator.print(f"Saved checkpoint to {save_dir}")

                if preview_sampler is not None or validation_fn is not None:
                    writer = accelerator.get_tracker("tensorboard", unwrap=True)
                    if (
                        preview_sampler is not None
                        and not preview_refs_logged
                        and preview_references is not None
                    ):
                        for tag, img in preview_references.items():
                            writer.add_image(
                                tag, img.clamp(0, 1), global_step, dataformats="CHW"
                            )
                        preview_refs_logged = True

                    # Mirror the EMA swap from _save_checkpoint so previews and
                    # validation use the same weights loaded at inference time.
                    unwrapped = accelerator.unwrap_model(model)
                    if ema is not None:
                        ema.store(unwrapped.parameters())
                        ema.copy_to(unwrapped.parameters())
                    was_training = unwrapped.training
                    unwrapped.eval()
                    images: dict[str, torch.Tensor] = {}
                    metrics: dict[str, float] = {}
                    try:
                        if validation_fn is not None:
                            metrics = validation_fn(unwrapped)
                        if preview_sampler is not None:
                            images = preview_sampler(unwrapped)
                    finally:
                        if was_training:
                            unwrapped.train()
                        if ema is not None:
                            ema.restore(unwrapped.parameters())

                    for name, value in metrics.items():
                        writer.add_scalar(f"val/{name}", value, global_step)
                    if metrics:
                        summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
                        accelerator.print(f"[val] epoch {epoch}: {summary}")
                    for tag, img in images.items():
                        writer.add_image(
                            tag, img.clamp(0, 1), global_step, dataformats="CHW"
                        )
                    writer.flush()

    accelerator.end_training()
