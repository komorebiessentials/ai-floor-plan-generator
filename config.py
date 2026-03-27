"""
Training configuration for the floor plan generator.

KEY CONCEPT: Hyperparameters
These are the knobs you tune to control how your model learns.
They're not learned by the model — YOU set them before training.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- Image settings ---
    # Floor plans are represented as images. The model works with fixed-size
    # square images. 64x64 is small enough to train fast, large enough to
    # capture room layouts.
    image_size: int = 64
    channels: int = 3  # RGB

    # --- Model architecture ---
    # latent_dim: The size of the compressed representation.
    # Think of it as: the model compresses a 64x64x3 image (12,288 values)
    # down to just `latent_dim` numbers. Those numbers capture the "essence"
    # of the floor plan. To generate new plans, you sample random numbers
    # in this space and decode them back to images.
    latent_dim: int = 128
    hidden_dims: tuple = (32, 64, 128, 256)  # Feature maps per conv layer

    # --- Training ---
    # batch_size: How many floor plans the model sees at once before updating.
    # Larger = more stable gradients but needs more memory.
    batch_size: int = 32

    # learning_rate: How big of a step the optimizer takes.
    # Too high = training explodes. Too low = training is painfully slow.
    learning_rate: float = 1e-4

    # epochs: How many times the model sees the ENTIRE dataset.
    # 1 epoch = 1 full pass through all training images.
    epochs: int = 100

    # kl_weight: Balances reconstruction quality vs. generation diversity.
    # Higher = more diverse but blurrier. Lower = sharper but less variety.
    kl_weight: float = 0.005

    # --- Data ---
    num_synthetic_samples: int = 5000  # How many floor plans to generate
    data_dir: str = "data"
    output_dir: str = "output"
    checkpoint_dir: str = "checkpoints"

    # --- Device ---
    device: str = "auto"  # "auto", "cuda", or "cpu"
