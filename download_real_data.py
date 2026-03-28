"""
Real Floor Plan Data Pipeline
==============================

This script downloads and preprocesses REAL floor plan images for training.

KEY CONCEPT: Why Real Data Matters
-----------------------------------
Our synthetic data (data_generator.py) creates simple colored rectangles.
Real floor plans have:
  - Irregular room shapes (not just rectangles)
  - Furniture, fixtures, labels
  - Various drawing styles (architectural, schematic, hand-drawn)
  - Different scales, orientations, resolutions

Training on real data teaches the model these complexities.

SUPPORTED DATASETS:
  1. CubiCasa5K     — 5,000 Finnish real estate floor plans (HuggingFace)
  2. pseudo-fp-12k  — 12,000 pseudo floor plans for generative models (HuggingFace)
  3. local          — Your own folder of floor plan images

USAGE:
  python download_real_data.py --source cubicasa      # Download CubiCasa5K
  python download_real_data.py --source pseudo12k     # Download 12K pseudo plans
  python download_real_data.py --source local --path /my/images/  # Use your own
"""

import os
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ============================================================================
#  STEP 1: DOWNLOAD — Get raw images from the internet
# ============================================================================

def download_cubicasa(raw_dir):
    """
    Download CubiCasa5K from HuggingFace.

    KEY CONCEPT: HuggingFace Datasets
    ----------------------------------
    HuggingFace hosts thousands of ML datasets. The `datasets` library
    lets you download them with one line of code. It handles:
      - Downloading files
      - Caching (won't re-download if already cached)
      - Streaming (can process without downloading everything first)

    CubiCasa5K contains ~5,000 real floor plan images scraped from
    Finnish real estate websites. Each has an image + SVG annotations.
    We only need the images for our VAE training.
    """
    from datasets import load_dataset

    print("Downloading CubiCasa5K from HuggingFace...")
    print("(This may take a few minutes on first run — data is cached after)")
    print()

    # load_dataset downloads and caches the data automatically
    dataset = load_dataset("Claudio9701/cubicasa5k", split="train")

    os.makedirs(raw_dir, exist_ok=True)
    saved = 0

    print(f"Processing {len(dataset)} floor plans...")
    for i, sample in enumerate(tqdm(dataset, desc="Saving images")):
        # The dataset stores images as PIL Image objects
        # The exact column name may vary — let's find the image column
        img = None
        for key in sample:
            if hasattr(sample[key], "save"):  # It's a PIL Image
                img = sample[key]
                break

        if img is not None:
            img.save(os.path.join(raw_dir, f"cubicasa_{i:05d}.png"))
            saved += 1

    print(f"\nSaved {saved} raw floor plan images to {raw_dir}/")
    return saved


def download_pseudo12k(raw_dir):
    """
    Download pseudo-floor-plan-12k from HuggingFace.

    This dataset has 12,000 floor plan-style images specifically
    designed for training generative models. Cleaner than real
    architectural drawings — a good middle ground between our
    synthetic data and messy real-world data.
    """
    from datasets import load_dataset

    print("Downloading pseudo-floor-plan-12k from HuggingFace...")
    print()

    dataset = load_dataset("zimhe/pseudo-floor-plan-12k", split="train")

    os.makedirs(raw_dir, exist_ok=True)
    saved = 0

    print(f"Processing {len(dataset)} floor plans...")
    for i, sample in enumerate(tqdm(dataset, desc="Saving images")):
        img = None
        for key in sample:
            if hasattr(sample[key], "save"):
                img = sample[key]
                break

        if img is not None:
            img.save(os.path.join(raw_dir, f"pseudo_{i:05d}.png"))
            saved += 1

    print(f"\nSaved {saved} raw floor plan images to {raw_dir}/")
    return saved


def load_local_images(source_dir, raw_dir):
    """
    Copy/link images from a local directory.

    Use this if you have your own collection of floor plan images.
    Just point --path to a folder containing PNG/JPG files.
    """
    import shutil

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = [
        f for f in source_path.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in {source_dir}. "
            f"Supported formats: {extensions}"
        )

    os.makedirs(raw_dir, exist_ok=True)

    print(f"Copying {len(image_files)} images from {source_dir}...")
    for i, src_file in enumerate(tqdm(image_files, desc="Copying")):
        dst_file = os.path.join(raw_dir, f"local_{i:05d}{src_file.suffix}")
        shutil.copy2(src_file, dst_file)

    print(f"\nCopied {len(image_files)} images to {raw_dir}/")
    return len(image_files)


# ============================================================================
#  STEP 2: PREPROCESS — Convert raw images to training-ready format
# ============================================================================

def preprocess_images(raw_dir, processed_dir, target_size=64):
    """
    Preprocess raw floor plan images for training.

    KEY CONCEPT: Preprocessing Pipeline
    -------------------------------------
    Raw images from the internet come in all shapes and sizes:
      - Different resolutions (500x400, 2000x3000, etc.)
      - Different aspect ratios (square, portrait, landscape)
      - Some have borders, watermarks, text labels
      - Different color schemes

    We need to normalize them all to a consistent format:
      - Fixed size: 64x64 (or whatever image_size is in config)
      - RGB color (3 channels)
      - Clean, consistent format

    The preprocessing steps:
      1. Convert to RGB (some may be grayscale or RGBA)
      2. Pad to square (so resizing doesn't distort)
      3. Resize to target size
      4. Save as PNG

    WHY PAD TO SQUARE?
    If an image is 1000x500 and we resize to 64x64, rooms would be
    stretched horizontally. Padding to 1000x1000 first preserves
    the original proportions.

    WHY NOT JUST CROP?
    Cropping might cut off rooms! Padding adds empty space, which
    is fine — the model learns that floor plans don't always fill
    the entire image.
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Find all images in raw directory
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    raw_files = sorted([
        f for f in Path(raw_dir).iterdir()
        if f.suffix.lower() in extensions
    ])

    if not raw_files:
        print(f"No images found in {raw_dir}/")
        return 0

    print(f"\nPreprocessing {len(raw_files)} images to {target_size}x{target_size}...")
    print(f"Output: {processed_dir}/")
    print()

    processed = 0
    skipped = 0

    for i, filepath in enumerate(tqdm(raw_files, desc="Preprocessing")):
        try:
            img = Image.open(filepath)

            # --- Step 2a: Convert to RGB ---
            # Some images are RGBA (with transparency), grayscale, or palette mode.
            # Our model expects 3-channel RGB.
            if img.mode != "RGB":
                # For RGBA: paste onto white background to handle transparency
                if img.mode == "RGBA":
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # Use alpha as mask
                    img = background
                else:
                    img = img.convert("RGB")

            # --- Step 2b: Pad to square ---
            # This preserves aspect ratio when we resize
            w, h = img.size
            max_dim = max(w, h)
            if w != h:
                # Create a white square canvas
                padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
                # Center the image on the canvas
                offset_x = (max_dim - w) // 2
                offset_y = (max_dim - h) // 2
                padded.paste(img, (offset_x, offset_y))
                img = padded

            # --- Step 2c: Resize to target size ---
            # LANCZOS is the highest quality downsampling filter.
            # It preserves sharp edges (important for walls and room boundaries).
            img = img.resize((target_size, target_size), Image.LANCZOS)

            # --- Step 2d: Save ---
            output_path = os.path.join(processed_dir, f"fp_{i:05d}.png")
            img.save(output_path)
            processed += 1

        except Exception as e:
            # Some images may be corrupted or in unsupported formats
            print(f"\n  Skipped {filepath.name}: {e}")
            skipped += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Training-ready images saved to: {processed_dir}/")
    return processed


# ============================================================================
#  STEP 3: VISUALIZE — Preview what the training data looks like
# ============================================================================

def visualize_samples(processed_dir, num_samples=16, output_path="data_preview.png"):
    """
    Create a grid preview of the processed training data.

    Always look at your data before training! You want to verify:
      - Images look like floor plans (not corrupted)
      - Consistent quality
      - Reasonable variety
    """
    import matplotlib.pyplot as plt

    images_dir = Path(processed_dir)
    image_files = sorted(images_dir.glob("*.png"))[:num_samples]

    if not image_files:
        print(f"No images found in {processed_dir}/")
        return

    cols = 4
    rows = (len(image_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    if rows == 1:
        axes = [axes]

    for idx, ax_row in enumerate(axes):
        if not hasattr(ax_row, "__iter__"):
            ax_row = [ax_row]
        for jdx, ax in enumerate(ax_row):
            img_idx = idx * cols + jdx
            if img_idx < len(image_files):
                img = Image.open(image_files[img_idx])
                ax.imshow(img)
                ax.set_title(image_files[img_idx].name, fontsize=8)
            ax.axis("off")

    plt.suptitle(f"Training Data Preview ({len(image_files)} samples)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Preview saved to {output_path}")


# ============================================================================
#  MAIN — Tie it all together
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess real floor plan data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_real_data.py --source cubicasa
  python download_real_data.py --source pseudo12k
  python download_real_data.py --source local --path ./my_floor_plans/
  python download_real_data.py --source cubicasa --size 128
        """,
    )
    parser.add_argument(
        "--source",
        choices=["cubicasa", "pseudo12k", "local"],
        default="cubicasa",
        help="Which dataset to use (default: cubicasa)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to local images (required if --source=local)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Target image size (default: 64)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data_raw",
        help="Directory for raw downloaded images (default: data_raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory for processed training images (default: data)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Generate a preview grid of processed images",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FLOOR PLAN DATA PIPELINE")
    print("=" * 60)
    print()

    # ── Step 1: Download ──
    print("STEP 1: Download / Load raw images")
    print("-" * 40)

    if args.source == "cubicasa":
        download_cubicasa(args.raw_dir)
    elif args.source == "pseudo12k":
        download_pseudo12k(args.raw_dir)
    elif args.source == "local":
        if not args.path:
            parser.error("--path is required when --source=local")
        load_local_images(args.path, args.raw_dir)

    # ── Step 2: Preprocess ──
    print()
    print("STEP 2: Preprocess images")
    print("-" * 40)
    num_processed = preprocess_images(args.raw_dir, args.output_dir, args.size)

    # ── Step 3: Visualize ──
    if args.preview and num_processed > 0:
        print()
        print("STEP 3: Generate preview")
        print("-" * 40)
        visualize_samples(args.output_dir)

    # ── Summary ──
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Source:     {args.source}")
    print(f"  Raw data:   {args.raw_dir}/")
    print(f"  Processed:  {args.output_dir}/")
    print(f"  Image size: {args.size}x{args.size}")
    print(f"  Total:      {num_processed} training images")
    print()
    print("  Next step: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
