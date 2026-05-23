from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 512  # the generated image resolution
    train_batch_size = 1
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-512"  # the model name locally

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    overwrite_output_dir = True
    seed = 0

    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset_path = "data/butterflies"


config = TrainingConfig()
