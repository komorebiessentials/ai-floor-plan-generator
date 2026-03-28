"""
Generate Floor Plans from a Trained Model

After training, this script loads the saved model and generates
new floor plans by sampling from the latent space.

KEY CONCEPT: Inference vs Training
- Training: Model learns from data (slow, uses gradients)
- Inference: Model generates predictions (fast, no gradients needed)

You can also do cool things like:
- Interpolation: Blend between two floor plan styles
- Exploration: Vary one latent dimension to see what it controls
"""

import os
import argparse

import torch
from torchvision.utils import save_image, make_grid

from config import Config
from model import FloorPlanVAE


def load_model(checkpoint_path, device):
    """Load a trained model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = FloorPlanVAE(
        channels=cfg.channels,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(loss: {checkpoint['loss']:.4f})")

    return model, cfg


def generate_random(model, cfg, device, num_samples=16, output_path="generated.png"):
    """Generate random floor plans."""
    samples = model.generate(num_samples, device)
    samples = (samples + 1) / 2  # Denormalize to [0, 1]

    grid = make_grid(samples, nrow=4, padding=2)
    save_image(grid, output_path)
    print(f"Generated {num_samples} floor plans -> {output_path}")


def interpolate(model, cfg, device, steps=10, output_path="interpolation.png"):
    """
    Interpolate between two random floor plans.

    KEY CONCEPT: Latent Space Interpolation
    Because the VAE's latent space is smooth (thanks to KL regularization),
    you can smoothly transition between two floor plans by interpolating
    their latent vectors. The intermediate points produce valid floor plans
    that are a "blend" of the two endpoints.
    """
    z1 = torch.randn(1, cfg.latent_dim, device=device)
    z2 = torch.randn(1, cfg.latent_dim, device=device)

    images = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2  # Linear interpolation
        with torch.no_grad():
            img = model.decoder(z)
        images.append(img)

    images = torch.cat(images, dim=0)
    images = (images + 1) / 2

    grid = make_grid(images, nrow=steps, padding=2)
    save_image(grid, output_path)
    print(f"Interpolation ({steps} steps) -> {output_path}")


def explore_dimension(model, cfg, device, dim=0, steps=10,
                      output_path="explore_dim.png"):
    """
    Vary a single latent dimension to see what it controls.

    Each dimension in the latent space may correspond to a different
    aspect of the floor plan: number of rooms, overall shape,
    room size distribution, etc.
    """
    z_base = torch.zeros(1, cfg.latent_dim, device=device)

    images = []
    for value in torch.linspace(-3, 3, steps):
        z = z_base.clone()
        z[0, dim] = value
        with torch.no_grad():
            img = model.decoder(z)
        images.append(img)

    images = torch.cat(images, dim=0)
    images = (images + 1) / 2

    grid = make_grid(images, nrow=steps, padding=2)
    save_image(grid, output_path)
    print(f"Dimension {dim} exploration -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate floor plans")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of floor plans to generate")
    parser.add_argument("--interpolate", action="store_true",
                        help="Generate interpolation between two plans")
    parser.add_argument("--explore-dim", type=int, default=None,
                        help="Explore a specific latent dimension")
    parser.add_argument("--output-dir", default="generated",
                        help="Output directory")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg = load_model(args.checkpoint, device)

    # Generate random samples
    generate_random(model, cfg, device, args.num_samples,
                    os.path.join(args.output_dir, "random_samples.png"))

    # Optionally interpolate
    if args.interpolate:
        interpolate(model, cfg, device,
                    output_path=os.path.join(args.output_dir, "interpolation.png"))

    # Optionally explore a dimension
    if args.explore_dim is not None:
        explore_dimension(model, cfg, device, dim=args.explore_dim,
                          output_path=os.path.join(args.output_dir,
                                                   f"dim_{args.explore_dim}.png"))
