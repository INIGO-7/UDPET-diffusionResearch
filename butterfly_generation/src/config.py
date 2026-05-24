from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 4
    eval_batch_size = 4
    num_epochs = 200    # Don't change when resuming a run! it'd misaling the LR schedule.
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 50  # The weights are heavy, so avoid granular disk writes
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-256"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    overwrite_output_dir = True
    seed = 0

    # "latest" auto-detects the most recent checkpoint; set to None to always start fresh
    resume_from_checkpoint = "latest"

    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset_path = "data/butterflies"


config = TrainingConfig()
