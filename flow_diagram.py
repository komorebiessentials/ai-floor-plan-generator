"""
FLOOR PLAN VAE — COMPLETE FLOW DIAGRAM
=======================================

This file documents the entire data + training + generation pipeline.
Run: python -c "from flow_diagram import print_flow; print_flow()"


================================================================================
                        OVERALL PIPELINE FLOW
================================================================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        PHASE 1: DATA CREATION                         │
  │                        (data_generator.py)                            │
  │                                                                       │
  │   config.py ──► num_samples = 5000                                    │
  │                 image_size  = 64x64                                    │
  │                                                                       │
  │   ┌───────────────┐    ┌──────────────────┐    ┌────────────────┐     │
  │   │ Empty Canvas  │───►│ BSP Partitioning │───►│ Assign Colors  │     │
  │   │  64x64 white  │    │ Split into rooms │    │ per room type  │     │
  │   └───────────────┘    └──────────────────┘    └───────┬────────┘     │
  │                                                        │              │
  │                              ┌─────────────────────────┘              │
  │                              ▼                                        │
  │                        ┌───────────┐    ┌───────────────────────┐     │
  │                        │ Add Walls │───►│ Save as PNG to data/  │     │
  │                        │ Add Doors │    │ x5000 images          │     │
  │                        └───────────┘    └───────────────────────┘     │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     PHASE 2: DATA LOADING                             │
  │                     (dataset.py)                                      │
  │                                                                       │
  │   data/*.png                                                          │
  │       │                                                               │
  │       ▼                                                               │
  │   ┌──────────────────┐   ┌──────────────┐   ┌────────────────────┐   │
  │   │ FloorPlanDataset │──►│  Transform   │──►│    DataLoader      │   │
  │   │ Load single PNG  │   │ Resize 64x64 │   │ Batch 32 images    │   │
  │   │ __getitem__(idx) │   │ ToTensor     │   │ Shuffle each epoch │   │
  │   │                  │   │ Norm [-1, 1] │   │ Pin memory (GPU)   │   │
  │   └──────────────────┘   └──────────────┘   └────────┬───────────┘   │
  └──────────────────────────────────────────────────────┬───────────────┘
                                                         │
                              Batches of 32 images       │
                              Shape: (32, 3, 64, 64)     │
                                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     PHASE 3: TRAINING                                 │
  │                     (train.py + model.py)                             │
  │                                                                       │
  │   For each epoch (1 to 100):                                          │
  │     For each batch (32 images):                                       │
  │                                                                       │
  │     ┌─────────────────────── FORWARD PASS ──────────────────────┐     │
  │     │                                                           │     │
  │     │  Input Image ──► ENCODER ──► (mu, log_var) ──► SAMPLE    │     │
  │     │  (32,3,64,64)              (32, 128) each    ──► z       │     │
  │     │                                              (32, 128)   │     │
  │     │                                                  │       │     │
  │     │                                                  ▼       │     │
  │     │                              Reconstructed ◄── DECODER   │     │
  │     │                              (32,3,64,64)                │     │
  │     └──────────────────────────────────┬────────────────────────┘     │
  │                                        │                              │
  │     ┌─────────────────── COMPUTE LOSS ─┴────────────────────────┐     │
  │     │                                                           │     │
  │     │  Total Loss = Reconstruction Loss + kl_weight * KL Loss   │     │
  │     │               ▲                              ▲            │     │
  │     │               │                              │            │     │
  │     │    MSE(input, output)          -0.5 * Σ(1 + log_var       │     │
  │     │    "How close is the                  - mu²               │     │
  │     │     reconstruction?"                  - exp(log_var))     │     │
  │     │                                "How close to N(0,1)?"     │     │
  │     └──────────────────────────────────┬────────────────────────┘     │
  │                                        │                              │
  │     ┌─────────────────── BACKWARD PASS ┴────────────────────────┐     │
  │     │                                                           │     │
  │     │  loss.backward()    ──►  Compute gradients for all weights│     │
  │     │  optimizer.step()   ──►  Update weights using Adam        │     │
  │     │                                                           │     │
  │     └───────────────────────────────────────────────────────────┘     │
  │                                                                       │
  │   Every 5 epochs: Save sample images to output/                       │
  │   Best loss:      Save model to checkpoints/best_model.pt             │
  │   TensorBoard:    Log losses to runs/                                 │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     PHASE 4: GENERATION                               │
  │                     (generate.py)                                     │
  │                                                                       │
  │   Load checkpoints/best_model.pt                                      │
  │                                                                       │
  │   Option A: Random Generation                                         │
  │   ┌──────────────────┐    ┌─────────┐    ┌────────────────────┐      │
  │   │ Sample z ~ N(0,1)│───►│ DECODER │───►│ New Floor Plan!    │      │
  │   │ (16, 128)        │    │         │    │ (16, 3, 64, 64)   │      │
  │   └──────────────────┘    └─────────┘    └────────────────────┘      │
  │                                                                       │
  │   Option B: Interpolation                                             │
  │   ┌────┐         ┌────┐                                               │
  │   │ z1 │────┐    │ z2 │────┐   Blend z1 and z2 at different ratios   │
  │   └────┘    │    └────┘    │   to smoothly morph between two plans    │
  │             ▼              ▼                                           │
  │        z = (1-α)·z1 + α·z2  ──► DECODER ──► Blended floor plan      │
  │        α goes from 0.0 to 1.0                                        │
  │                                                                       │
  │   Option C: Dimension Exploration                                     │
  │        Start with z = zeros                                           │
  │        Vary z[dim] from -3 to +3 ──► DECODER ──► See what changes    │
  └─────────────────────────────────────────────────────────────────────────┘


================================================================================
                     VAE MODEL ARCHITECTURE (model.py)
================================================================================

  ENCODER (Compresses image to latent vector)
  ──────────────────────────────────────────────

  Input: (batch, 3, 64, 64)
    │
    ▼
  Conv2d(3→32)  + BatchNorm + LeakyReLU    ──► (batch, 32, 32, 32)
    │                                           spatial: 64→32 (stride=2)
    ▼
  Conv2d(32→64) + BatchNorm + LeakyReLU    ──► (batch, 64, 16, 16)
    │                                           spatial: 32→16
    ▼
  Conv2d(64→128) + BatchNorm + LeakyReLU   ──► (batch, 128, 8, 8)
    │                                           spatial: 16→8
    ▼
  Conv2d(128→256) + BatchNorm + LeakyReLU  ──► (batch, 256, 4, 4)
    │                                           spatial: 8→4
    ▼
  Flatten                                  ──► (batch, 4096)
    │                                           256 * 4 * 4 = 4096
    ├──► Linear(4096→128) ──► mu               (batch, 128)
    │
    └──► Linear(4096→128) ──► log_var          (batch, 128)


  REPARAMETERIZATION TRICK
  ────────────────────────
  z = mu + exp(0.5 * log_var) * epsilon
  where epsilon ~ N(0, 1)


  DECODER (Expands latent vector back to image)
  ───────────────────────────────────────────────

  Input: z (batch, 128)
    │
    ▼
  Linear(128→4096)                          ──► (batch, 4096)
    │
    ▼
  Reshape                                   ──► (batch, 256, 4, 4)
    │
    ▼
  ConvTranspose2d(256→128) + BN + LeakyReLU ──► (batch, 128, 8, 8)
    │                                            spatial: 4→8
    ▼
  ConvTranspose2d(128→64) + BN + LeakyReLU  ──► (batch, 64, 16, 16)
    │                                            spatial: 8→16
    ▼
  ConvTranspose2d(64→32) + BN + LeakyReLU   ──► (batch, 32, 32, 32)
    │                                            spatial: 16→32
    ▼
  ConvTranspose2d(32→3) + Tanh              ──► (batch, 3, 64, 64)
    │                                            spatial: 32→64
    ▼
  Output: Reconstructed image in [-1, 1]


================================================================================
              DATA FLOW THROUGH ONE TRAINING STEP
================================================================================

  ┌──────────┐   load    ┌──────────┐  batch   ┌──────────────────────┐
  │  5000    │──────────►│DataLoader│────────►│ 32 images            │
  │  PNGs    │           │shuffle   │         │ shape: (32,3,64,64)  │
  │  on disk │           │batch=32  │         │ values: [-1, 1]      │
  └──────────┘           └──────────┘         └──────────┬───────────┘
                                                         │
                         ┌───────────────────────────────┘
                         ▼
                    ┌──────────┐
                    │ ENCODER  │
                    │ 4 conv   │
                    │ layers   │
                    └────┬─────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
                ┌──────┐  ┌─────────┐
                │  mu  │  │ log_var │    Each: (32, 128)
                └──┬───┘  └────┬────┘
                   │           │
                   └─────┬─────┘
                         ▼
                  ┌──────────────┐
                  │ REPARAM TRICK│    z = mu + std * noise
                  │  Sample z    │    (32, 128)
                  └──────┬───────┘
                         │
                         ▼
                    ┌──────────┐
                    │ DECODER  │
                    │ 4 deconv │
                    │ layers   │
                    └────┬─────┘
                         │
                         ▼
                  ┌──────────────┐
                  │Reconstructed │    (32, 3, 64, 64)
                  │   Images     │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       ┌─────────────┐     ┌──────────────┐
       │ Recon Loss  │     │  KL Loss     │
       │MSE(in, out) │     │ regularize   │
       │             │     │ latent space │
       └──────┬──────┘     └──────┬───────┘
              │                   │
              └────────┬──────────┘
                       ▼
              ┌────────────────┐
              │  Total Loss    │
              │ = recon + β*KL │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ loss.backward()│   Compute gradients
              │ optimizer.step │   Update all weights
              └────────────────┘


================================================================================
                    FILE DEPENDENCY MAP
================================================================================

  config.py ◄─────────────────────────────────────────────┐
     │                                                     │
     │  (imported by all other files)                      │
     │                                                     │
     ├──► data_generator.py   [standalone — run first]     │
     │         │                                           │
     │         ▼                                           │
     │       data/*.png  (5000 floor plan images)          │
     │         │                                           │
     │         ▼                                           │
     ├──► dataset.py  ◄──── loads images from data/        │
     │         │                                           │
     │         ▼                                           │
     ├──► model.py    ◄──── defines the VAE architecture   │
     │         │                                           │
     │         ▼                                           │
     ├──► train.py    ◄──── uses dataset.py + model.py     │
     │         │              saves to checkpoints/        │
     │         ▼                                           │
     └──► generate.py ◄──── loads from checkpoints/        │
               │              uses model.py                │
               ▼                                           │
             generated/*.png  (new floor plans!)           │
                                                           │
  requirements.txt  (pip dependencies — not imported) ─────┘
"""


def print_flow():
    """Print the flow diagram to the terminal."""
    import inspect
    # Get the module docstring which contains our diagrams
    print(__doc__)


if __name__ == "__main__":
    print_flow()
