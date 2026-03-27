"""
Synthetic Floor Plan Data Generator

KEY CONCEPT: Training Data
A model learns by example. If you want it to generate floor plans, you need
to show it thousands of floor plans. Each one is a "training sample."

The model learns patterns like:
- Rooms are rectangular
- Rooms connect to each other
- There's usually a hallway/corridor
- Bathrooms are smaller than living rooms

We generate synthetic (fake but realistic) floor plans as colored images:
- White  = walls/background
- Colors = different room types (bedroom, kitchen, bathroom, etc.)
"""

import os
import random

import numpy as np
from PIL import Image, ImageDraw

# Room types with their associated colors (RGB)
ROOM_TYPES = {
    "living_room": (173, 216, 230),   # Light blue
    "bedroom": (144, 238, 144),       # Light green
    "kitchen": (255, 218, 185),       # Peach
    "bathroom": (221, 160, 221),      # Plum
    "hallway": (245, 245, 220),       # Beige
    "dining": (255, 255, 200),        # Light yellow
    "closet": (200, 200, 200),        # Light gray
}

WALL_COLOR = (40, 40, 40)
BACKGROUND_COLOR = (255, 255, 255)
DOOR_COLOR = (139, 90, 43)


def generate_floor_plan(image_size=64):
    """
    Generate a single synthetic floor plan image.

    The algorithm:
    1. Start with a blank canvas (the building outline)
    2. Use a simple space-partitioning approach to divide the space into rooms
    3. Color each room based on its type
    4. Draw walls between rooms
    5. Add doors connecting adjacent rooms

    This is a SIMPLIFIED version of real floor plan generation. Real
    architectural floor plans follow building codes, structural constraints,
    etc. But this gives the model enough patterns to learn from.
    """
    img = Image.new("RGB", (image_size, image_size), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Building boundary (leave margin for outer walls)
    margin = 3
    bx1, by1 = margin, margin
    bx2, by2 = image_size - margin, image_size - margin

    # Draw outer walls
    draw.rectangle([bx1, by1, bx2, by2], outline=WALL_COLOR, width=2)

    # Partition the space into rooms using recursive splitting
    rooms = _partition_space(bx1 + 2, by1 + 2, bx2 - 2, by2 - 2, depth=0)

    # Assign room types
    room_type_names = list(ROOM_TYPES.keys())
    assigned_rooms = []
    for room in rooms:
        rtype = random.choice(room_type_names)
        assigned_rooms.append((room, rtype))

    # Fill rooms with colors
    for (x1, y1, x2, y2), rtype in assigned_rooms:
        color = ROOM_TYPES[rtype]
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # Draw walls between rooms
    for (x1, y1, x2, y2), _ in assigned_rooms:
        draw.rectangle([x1, y1, x2, y2], outline=WALL_COLOR, width=1)

    # Add doors between adjacent rooms
    _add_doors(draw, assigned_rooms)

    return img


def _partition_space(x1, y1, x2, y2, depth, max_depth=3):
    """
    Recursively split a rectangle into smaller rooms.

    KEY CONCEPT: Binary Space Partitioning (BSP)
    This is a classic algorithm in game development and architecture.
    You take a rectangle and split it in half (either horizontally or
    vertically), then repeat on each half. This naturally creates
    room-like subdivisions.
    """
    width = x2 - x1
    height = y2 - y1
    min_room_size = 10

    # Stop splitting if room is too small or we've gone deep enough
    if depth >= max_depth or width < min_room_size * 2 or height < min_room_size * 2:
        return [(x1, y1, x2, y2)]

    # Randomly decide to stop splitting (adds variety)
    if depth > 1 and random.random() < 0.3:
        return [(x1, y1, x2, y2)]

    rooms = []

    # Choose split direction based on aspect ratio (split the longer side)
    if width > height:
        split_horizontal = False
    elif height > width:
        split_horizontal = True
    else:
        split_horizontal = random.choice([True, False])

    if split_horizontal:
        # Split top/bottom
        split_y = random.randint(y1 + min_room_size, y2 - min_room_size)
        rooms.extend(_partition_space(x1, y1, x2, split_y, depth + 1))
        rooms.extend(_partition_space(x1, split_y, x2, y2, depth + 1))
    else:
        # Split left/right
        split_x = random.randint(x1 + min_room_size, x2 - min_room_size)
        rooms.extend(_partition_space(x1, y1, split_x, y2, depth + 1))
        rooms.extend(_partition_space(split_x, y1, x2, y2, depth + 1))

    return rooms


def _add_doors(draw, assigned_rooms):
    """Add doors between adjacent rooms."""
    for i, ((x1a, y1a, x2a, y2a), _) in enumerate(assigned_rooms):
        for j, ((x1b, y1b, x2b, y2b), _) in enumerate(assigned_rooms):
            if i >= j:
                continue

            # Check if rooms share a vertical wall
            if abs(x2a - x1b) <= 2 or abs(x2b - x1a) <= 2:
                overlap_y1 = max(y1a, y1b)
                overlap_y2 = min(y2a, y2b)
                if overlap_y2 - overlap_y1 > 6:
                    door_y = (overlap_y1 + overlap_y2) // 2
                    door_x = x2a if abs(x2a - x1b) <= 2 else x2b
                    draw.rectangle(
                        [door_x - 1, door_y - 2, door_x + 1, door_y + 2],
                        fill=DOOR_COLOR,
                    )

            # Check if rooms share a horizontal wall
            if abs(y2a - y1b) <= 2 or abs(y2b - y1a) <= 2:
                overlap_x1 = max(x1a, x1b)
                overlap_x2 = min(x2a, x2b)
                if overlap_x2 - overlap_x1 > 6:
                    door_x = (overlap_x1 + overlap_x2) // 2
                    door_y = y2a if abs(y2a - y1b) <= 2 else y2b
                    draw.rectangle(
                        [door_x - 2, door_y - 1, door_x + 2, door_y + 1],
                        fill=DOOR_COLOR,
                    )


def generate_dataset(num_samples, image_size, output_dir):
    """
    Generate a full dataset of floor plan images.

    KEY CONCEPT: Dataset
    A dataset is just a folder of training examples. Each image is one example.
    More examples = the model sees more variety = better generalization.
    5,000 samples is a reasonable starting point for a small model.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} floor plan images...")
    for i in range(num_samples):
        img = generate_floor_plan(image_size)
        img.save(os.path.join(output_dir, f"floor_plan_{i:05d}.png"))

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples}")

    print(f"Dataset saved to {output_dir}/")


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    generate_dataset(cfg.num_synthetic_samples, cfg.image_size, cfg.data_dir)
