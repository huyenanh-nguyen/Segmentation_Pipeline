#!/usr/bin/env python3
"""Stack only *_instances TIFF frames and convert TYX stacks to GIFs."""

from pathlib import Path
import re

import numpy as np
import tifffile as tiff
from PIL import Image


BASE_DIR = Path.cwd()

# Input folder created by the watershed inference script
SOURCE_DIR = BASE_DIR / "unet_predictions_watershed_pt2"

# Output folders
STACK_DIR = BASE_DIR / "unet_instance_stack_pt2"
GIF_DIR = BASE_DIR / "unet_instance_gif_pt2"

GIF_FRAME_DURATION_MS = 100
GIF_LOOP = 0


def frame_number(path: Path) -> int:
    """Extract frame index from names such as frame_0012_instances.tif."""
    match = re.search(r"frame[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot extract frame number from {path.name}")
    return int(match.group(1))


def find_instance_files(folder: Path) -> list[Path]:
    """Find only TIFF files containing '_instances' and sort by frame index."""
    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and "_instances" in p.stem.lower()
    ]
    return sorted(files, key=frame_number)


def stack_unet_instances_strict() -> None:
    STACK_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        print(f"Error: Source directory does not exist: {SOURCE_DIR}")
        return

    subfolders = sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir())
    if not subfolders:
        print(f"No subfolders found inside: {SOURCE_DIR}")
        return

    for subfolder in subfolders:
        try:
            tiff_files = find_instance_files(subfolder)
        except ValueError as exc:
            print(f"Skipping '{subfolder.name}': {exc}")
            continue

        if not tiff_files:
            print(f"Skipping '{subfolder.name}': no *_instances TIFFs found.")
            continue

        print(f"\nProcessing '{subfolder.name}' ({len(tiff_files)} frames)")
        for path in tiff_files[:3]:
            print(f"  - {path.name}")
        if len(tiff_files) > 3:
            print(f"  - ... and {len(tiff_files) - 3} more")

        frames = []
        expected_shape = None

        for path in tiff_files:
            frame = np.asarray(tiff.imread(path))

            if frame.ndim != 2:
                raise ValueError(
                    f"{path.name}: expected 2D labels, got {frame.shape}"
                )

            if not np.issubdtype(frame.dtype, np.integer):
                if not np.all(np.isfinite(frame)) or not np.allclose(
                    frame, np.round(frame)
                ):
                    raise ValueError(f"{path.name}: labels are not integers")
                frame = np.round(frame).astype(np.int32)

            if np.any(frame < 0):
                raise ValueError(f"{path.name}: negative instance IDs found")

            if expected_shape is None:
                expected_shape = frame.shape
            elif frame.shape != expected_shape:
                raise ValueError(
                    f"{path.name}: shape {frame.shape} differs from "
                    f"expected {expected_shape}"
                )

            frames.append(frame)

        stack = np.stack(frames, axis=0)
        output_path = STACK_DIR / f"{subfolder.name}_instances_stack.tif"

        tiff.imwrite(
            output_path,
            stack,
            bigtiff=True,
            compression="zlib",
            photometric="minisblack",
            metadata={"axes": "TYX"},
        )

        print(f"Saved {output_path} | shape={stack.shape} | dtype={stack.dtype}")

    print(f"\nStacking finished: {STACK_DIR}")


def label_frame_to_rgb(frame: np.ndarray) -> Image.Image:
    """
    Render categorical instance IDs as RGB.
    Background is black; each ID gets a deterministic color.
    """
    labels = np.asarray(frame)

    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got {labels.shape}")

    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.allclose(
            labels, np.round(labels)
        ):
            raise ValueError("Instance labels must be integers")
        labels = np.round(labels).astype(np.int64)

    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError("Instance labels must be non-negative")

    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    instance_ids = np.unique(labels)
    instance_ids = instance_ids[instance_ids > 0]

    for instance_id in instance_ids:
        value = int(instance_id)
        red = (value * 67 + 53) % 256
        green = (value * 131 + 97) % 256
        blue = (value * 149 + 193) % 256

        if red + green + blue < 180:
            red = (red + 100) % 256
            green = (green + 80) % 256
            blue = (blue + 60) % 256

        rgb[labels == instance_id] = (red, green, blue)

    return Image.fromarray(rgb)


def convert_tiff_stacks_to_gifs() -> None:
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    if not STACK_DIR.exists():
        print(f"Error: Stack directory does not exist: {STACK_DIR}")
        return

    tiff_files = sorted(
        list(STACK_DIR.glob("*.tif")) + list(STACK_DIR.glob("*.tiff"))
    )
    if not tiff_files:
        print(f"No TIFF stacks found inside: {STACK_DIR}")
        return

    for tiff_file in tiff_files:
        print(f"\nConverting '{tiff_file.name}'...")
        stack = np.asarray(tiff.imread(tiff_file))

        if stack.ndim != 3:
            print(
                f"Skipping: expected TYX (3D), got shape {stack.shape}"
            )
            continue
        if stack.shape[0] == 0:
            print("Skipping empty stack.")
            continue

        gif_frames = [label_frame_to_rgb(frame) for frame in stack]
        output_path = GIF_DIR / f"{tiff_file.stem}.gif"

        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=GIF_FRAME_DURATION_MS,
            loop=GIF_LOOP,
            optimize=False,
        )

        print(f"Saved {output_path} ({len(gif_frames)} frames)")

    print(f"\nGIF conversion finished: {GIF_DIR}")


if __name__ == "__main__":
    stack_unet_instances_strict()
    convert_tiff_stacks_to_gifs()
