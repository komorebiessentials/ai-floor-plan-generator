"""
Variational Autoencoder (VAE) for Floor Plan Generation

KEY CONCEPT: How a VAE Works
================================

A VAE has two halves:

1. ENCODER (compresses):
   Floor Plan Image (64x64x3) -> Latent Vector (128 numbers)

   It takes a full image and squeezes it down to a tiny representation.
   Those 128 numbers capture the "essence": number of rooms, layout style,
   room sizes, etc.

2. DECODER (expands):
   Latent Vector (128 numbers) -> Floor Plan Image (64x64x3)

   It takes the tiny representation and reconstructs the full image.

Training teaches both halves simultaneously:
- The encoder learns WHAT information matters
- The decoder learns HOW to reconstruct from that information

KEY CONCEPT: Why "Variational"?
================================
A plain autoencoder just compresses and decompresses. A VAE adds randomness:
instead of encoding to a single point, it encodes to a DISTRIBUTION
(mean + variance). During training, we sample from this distribution.

Why? This forces the latent space to be smooth and continuous. Nearby points
in latent space produce similar floor plans. This means you can:
- Sample random points to generate NEW floor plans
- Interpolate between two floor plans smoothly
- Control specific features by moving along latent dimensions

KEY CONCEPT: Convolutional Layers
================================
Images have spatial structure — nearby pixels are related. Convolutional
layers exploit this by sliding small filters across the image, detecting
local patterns (edges, corners, room boundaries).

Encoder: Conv layers progressively SHRINK the spatial size while
         INCREASING the number of feature channels.
         64x64x3 -> 32x32x32 -> 16x16x64 -> 8x8x128 -> 4x4x256

Decoder: ConvTranspose layers do the REVERSE — expanding spatial size
         while decreasing channels.
         4x4x256 -> 8x8x128 -> 16x16x64 -> 32x32x32 -> 64x64x3
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Compresses a floor plan image into a latent distribution (mu, log_var).
    """

    def __init__(self, channels, hidden_dims, latent_dim):
        super().__init__()

        # Build convolutional layers
        layers = []
        in_channels = channels
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    # Conv2d: Applies learned filters to detect features.
                    # kernel_size=3: Each filter looks at a 3x3 patch.
                    # stride=2: Moves 2 pixels at a time, halving spatial size.
                    # padding=1: Pads edges so we don't lose border info.
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),

                    # BatchNorm: Normalizes activations to stabilize training.
                    # Without it, values can explode or vanish through layers.
                    nn.BatchNorm2d(h_dim),

                    # LeakyReLU: Activation function.
                    # Introduces non-linearity — without it, stacking layers
                    # would be equivalent to a single linear transformation.
                    # "Leaky" means it allows small negative values through
                    # (unlike ReLU which kills them), preventing "dead neurons."
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = h_dim

        self.conv_layers = nn.Sequential(*layers)

        # After convolutions, flatten and project to latent space.
        # With 64x64 input and 4 stride-2 convolutions: 64 / 2^4 = 4
        # So we have hidden_dims[-1] * 4 * 4 values.
        flat_size = hidden_dims[-1] * 4 * 4

        # Two separate projections: one for mean, one for log-variance.
        # Together they define the latent distribution.
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_log_var = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        """
        Input:  x of shape (batch_size, 3, 64, 64)
        Output: mu, log_var each of shape (batch_size, latent_dim)
        """
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # (batch, channels*4*4)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    """
    Reconstructs a floor plan image from a latent vector.
    """

    def __init__(self, channels, hidden_dims, latent_dim):
        super().__init__()

        # Reverse hidden dims for the decoder
        hidden_dims = list(reversed(hidden_dims))

        # Project from latent space back to spatial features
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        self.initial_channels = hidden_dims[0]

        # Build transposed convolutional layers (upsampling)
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    # ConvTranspose2d: The "reverse" of Conv2d.
                    # stride=2 DOUBLES the spatial size (upsamples).
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1],
                        kernel_size=3, stride=2, padding=1, output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                )
            )

        # Final layer: project back to RGB channels
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                # Tanh squashes output to [-1, 1], matching our input normalization.
                nn.Tanh(),
            )
        )

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z):
        """
        Input:  z of shape (batch_size, latent_dim)
        Output: reconstructed image of shape (batch_size, 3, 64, 64)
        """
        x = self.fc(z)
        x = x.view(-1, self.initial_channels, 4, 4)
        x = self.deconv_layers(x)
        return x


class FloorPlanVAE(nn.Module):
    """
    The complete VAE model.

    Combines encoder + decoder and implements the reparameterization trick.
    """

    def __init__(self, channels=3, hidden_dims=(32, 64, 128, 256), latent_dim=128):
        super().__init__()
        self.encoder = Encoder(channels, hidden_dims, latent_dim)
        self.decoder = Decoder(channels, hidden_dims, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        KEY CONCEPT: Reparameterization Trick

        Problem: We need to SAMPLE from the latent distribution during
        training, but sampling is random — you can't backpropagate gradients
        through randomness.

        Solution: Instead of sampling z ~ N(mu, sigma), we compute:
            z = mu + sigma * epsilon,  where epsilon ~ N(0, 1)

        The randomness (epsilon) is external to the computation graph,
        so gradients flow through mu and sigma normally. Clever!
        """
        std = torch.exp(0.5 * log_var)  # sigma = exp(log_var / 2)
        eps = torch.randn_like(std)      # Random noise from N(0, 1)
        return mu + eps * std

    def forward(self, x):
        """Full forward pass: encode -> sample -> decode."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def generate(self, num_samples, device):
        """
        Generate new floor plans by sampling from the latent space.

        After training, the latent space is organized so that random
        samples from N(0, 1) decode into realistic floor plans.
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decoder(z)
