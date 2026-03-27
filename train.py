"""
Training Script for Floor Plan VAE

KEY CONCEPT: Training Loop
================================
Training a neural network is an iterative optimization process:

1. FORWARD PASS:  Feed images through the model, get predictions
2. COMPUTE LOSS:  Measure how bad the predictions are
3. BACKWARD PASS: Compute gradients (which direction to adjust each weight)
4. UPDATE:        Adjust weights using the gradients (optimizer.step())
5. REPEAT:        Go back to step 1 with the next batch

Each full pass through the dataset is called an "epoch."
You typically train for many epochs (50-200+).

KEY CONCEPT: VAE Loss Function
================================
The VAE loss has two parts:

1. Reconstruction Loss (MSE):
   "How well can the decoder recreate the original image?"
   = Mean Squared Error between input and reconstruction.
   This teaches the model to compress/decompress accurately.

2. KL Divergence Loss:
   "How close is the learned latent distribution to a standard normal N(0,1)?"
   This regularizes the latent space, ensuring it's smooth and continuous.
   Without it, the encoder might map each image to a random isolated point,
   and the decoder couldn't generate anything coherent from random samples.

Total Loss = Reconstruction Loss + kl_weight * KL Divergence
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import Config
from dataset import create_dataloader
from model import FloorPlanVAE


def vae_loss(reconstruction, original, mu, log_var, kl_weight):
    """
    Compute the VAE loss.

    Args:
        reconstruction: Decoder output (batch_size, 3, 64, 64)
        original: Input image (batch_size, 3, 64, 64)
        mu: Latent mean (batch_size, latent_dim)
        log_var: Latent log variance (batch_size, latent_dim)
        kl_weight: How much to weight the KL term

    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # Reconstruction loss: how different is the output from the input?
    recon_loss = F.mse_loss(reconstruction, original, reduction="sum") / original.shape[0]

    # KL divergence: how far is the latent distribution from N(0,1)?
    # This formula comes from the analytical KL divergence between
    # two Gaussian distributions. See Kingma & Welling (2013).
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / original.shape[0]

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def get_device(config_device):
    """Determine which device to use for training."""
    if config_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU (training will be slower)")
    else:
        device = torch.device(config_device)
    return device


def save_samples(model, device, epoch, output_dir, num_samples=16):
    """Generate and save sample floor plans from the model."""
    model.eval()
    samples = model.generate(num_samples, device)

    # Denormalize: [-1, 1] -> [0, 1] for saving
    samples = (samples + 1) / 2

    grid = make_grid(samples, nrow=4, padding=2)
    save_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
    save_image(grid, save_path)
    model.train()
    return grid


def train():
    """
    Main training function.

    This ties everything together:
    1. Set up data, model, optimizer
    2. Run the training loop
    3. Save checkpoints and sample images
    """
    cfg = Config()

    # --- Setup ---
    device = get_device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Create data loader
    dataloader = create_dataloader(cfg.data_dir, cfg.image_size, cfg.batch_size)

    # Create model
    model = FloorPlanVAE(
        channels=cfg.channels,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
    ).to(device)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer: Adam is a good default choice
    # KEY CONCEPT: The optimizer decides HOW to update weights given the gradients.
    # Adam adapts the learning rate for each parameter individually,
    # which is more effective than a single global learning rate (plain SGD).
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True,
    )

    # TensorBoard writer for logging
    writer = SummaryWriter(log_dir="runs/floor_plan_vae")

    # --- Training Loop ---
    print(f"\nStarting training for {cfg.epochs} epochs...")
    print(f"Logging to TensorBoard: runs/floor_plan_vae")
    print("-" * 60)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        # Progress bar for this epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for batch in pbar:
            # Move batch to device (GPU if available)
            batch = batch.to(device)

            # --- Forward Pass ---
            reconstruction, mu, log_var = model(batch)

            # --- Compute Loss ---
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, batch, mu, log_var, cfg.kl_weight,
            )

            # --- Backward Pass ---
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients

            # Gradient clipping: prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # --- Update Weights ---
            optimizer.step()

            # --- Logging ---
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.2f}",
                "recon": f"{recon_loss.item():.2f}",
                "kl": f"{kl_loss.item():.2f}",
            })

            # Log to TensorBoard
            writer.add_scalar("batch/total_loss", loss.item(), global_step)
            writer.add_scalar("batch/recon_loss", recon_loss.item(), global_step)
            writer.add_scalar("batch/kl_loss", kl_loss.item(), global_step)

        # --- Epoch Summary ---
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches

        print(f"Epoch {epoch}: Loss={avg_loss:.2f} "
              f"(Recon={avg_recon:.2f}, KL={avg_kl:.2f})")

        # Log epoch metrics
        writer.add_scalar("epoch/total_loss", avg_loss, epoch)
        writer.add_scalar("epoch/recon_loss", avg_recon, epoch)
        writer.add_scalar("epoch/kl_loss", avg_kl, epoch)
        writer.add_scalar("epoch/learning_rate",
                          optimizer.param_groups[0]["lr"], epoch)

        # Update learning rate
        scheduler.step(avg_loss)

        # --- Save Samples Every 5 Epochs ---
        if epoch % 5 == 0 or epoch == 1:
            grid = save_samples(model, device, epoch, cfg.output_dir)
            writer.add_image("generated_samples", grid, epoch)
            print(f"  -> Samples saved to {cfg.output_dir}/")

        # --- Save Best Model ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": cfg,
            }, os.path.join(cfg.checkpoint_dir, "best_model.pt"))

        # --- Save Periodic Checkpoint ---
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": cfg,
            }, os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    # --- Training Complete ---
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}/")
    print(f"Sample images saved to: {cfg.output_dir}/")
    print(f"View training curves: tensorboard --logdir runs/")

    writer.close()


if __name__ == "__main__":
    train()
