"""
PyTorch Dataset for Floor Plans

KEY CONCEPT: Dataset & DataLoader
In PyTorch, training data flows through two abstractions:

1. Dataset — Knows how to load ONE sample (one floor plan image).
   You implement __len__ (how many samples?) and __getitem__ (give me sample #i).

2. DataLoader — Wraps a Dataset and handles:
   - Batching: Groups samples into batches (e.g., 32 images at a time)
   - Shuffling: Randomizes order each epoch (prevents the model from
     memorizing the order rather than learning patterns)
   - Parallel loading: Uses multiple CPU workers to load data faster

The pipeline: Disk -> Dataset -> DataLoader -> Model
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FloorPlanDataset(Dataset):
    """
    Loads floor plan images from a directory.

    Each image is transformed to a tensor with values in [-1, 1].
    Why [-1, 1]? Neural networks work best with normalized inputs.
    Raw pixel values (0-255) would create huge gradients early in training.
    """

    def __init__(self, data_dir, image_size=64):
        self.data_dir = data_dir
        self.image_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {data_dir}. "
                f"Run `python data_generator.py` first to create training data."
            )

        # Transforms: Resize -> Convert to tensor -> Normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),           # [0, 255] -> [0, 1]
            transforms.Normalize(            # [0, 1]   -> [-1, 1]
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self):
        """How many floor plans are in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return one floor plan.

        Returns a tensor of shape (3, image_size, image_size).
        The 3 channels are R, G, B.
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def create_dataloader(data_dir, image_size, batch_size, num_workers=2):
    """
    Create a DataLoader ready for training.

    KEY CONCEPT: Batching
    Instead of updating the model after every single image, we update after
    seeing a "batch" of images. This is called mini-batch gradient descent.

    Why? Single-image updates are noisy (one weird floor plan could send
    the model in a bad direction). Averaging gradients over a batch of 32
    gives a much more reliable signal of which direction to update.
    """
    dataset = FloorPlanDataset(data_dir, image_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,         # Randomize order each epoch
        num_workers=num_workers,
        pin_memory=True,      # Speeds up CPU->GPU transfer
        drop_last=True,       # Drop incomplete last batch for consistent sizes
    )

    print(f"Dataset: {len(dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")

    return dataloader
