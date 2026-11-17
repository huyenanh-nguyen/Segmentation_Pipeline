
#!/usr/bin/env python3
"""Stack tracked *_instances TIFF masks and convert TYX stacks to colored GIFs."""

from pathlib import Path
import re

import numpy as np
import tifffile as tiff
from PIL import Image


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path.cwd()

# Output folder created by run_tracking_corrected.py
SOURCE_DIR = BASE_DIR / "unet_predictions_tracked"

# Output folders
STACK_DIR = BASE_DIR / "unet_instance_stack"
GIF_DIR = BASE_DIR / "unet_instance_gif"

GIF_FRAME_DURATION_MS = 100
GIF_LOOP = 0


# ============================================================
# FIND AND SORT INSTANCE MASKS
# ============================================================

def frame_number(path: Path) -> int:
    """Extract frame index, e.g. frame_0012_instances.tif -> 12."""

    match = re.search(
        r"frame[_-]?(\d+)",
        path.stem,
        flags=re.IGNORECASE
    )

    if match is None:
        raise ValueError(
            f"Cannot extract frame number from {path.name}"
        )

    return int(match.group(1))


def find_instance_files(folder: Path) -> list[Path]:
    """Find only *_instances TIFFs and sort by frame number."""

    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and "_instances" in p.stem.lower()
    ]

    return sorted(files, key=frame_number)


# ============================================================
# VALIDATE INSTANCE MASKS
# ============================================================

def validate_instance_frame(path: Path) -> np.ndarray:
    """Read one 2D integer instance-label image."""

    frame = np.asarray(tiff.imread(path))

    if frame.ndim != 2:
        raise ValueError(
            f"{path.name}: expected 2D labels, got shape {frame.shape}"
        )

    if not np.issubdtype(frame.dtype, np.integer):

        if not np.all(np.isfinite(frame)) or not np.allclose(
            frame, np.round(frame)
        ):
            raise ValueError(
                f"{path.name}: labels are not integers"
            )

        frame = np.round(frame).astype(np.int32)

    if np.any(frame < 0):
        raise ValueError(
            f"{path.name}: negative instance IDs found"
        )

    return frame


# ============================================================
# STACK TRACKED INSTANCE MASKS
# ============================================================

def stack_tracked_instances() -> None:

    STACK_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Tracking output folder does not exist: {SOURCE_DIR}"
        )

    recording_folders = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_dir()
    )

    if not recording_folders:
        print(f"No recording folders found inside: {SOURCE_DIR}")
        return

    for recording_folder in recording_folders:

        # Instance TIFFs are inside instance_masks/
        instance_dir = recording_folder / "instance_masks"

        if not instance_dir.is_dir():
            print(
                f"Skipping '{recording_folder.name}': "
                "no instance_masks folder."
            )
            continue

        tiff_files = find_instance_files(instance_dir)

        if not tiff_files:
            print(
                f"Skipping '{recording_folder.name}': "
                "no *_instances TIFFs found."
            )
            continue

        print(
            f"\nProcessing '{recording_folder.name}' "
            f"({len(tiff_files)} frames)"
        )

        frames = []
        expected_shape = None
        seen_frame_numbers = set()

        for path in tiff_files:

            frame_index = frame_number(path)

            if frame_index in seen_frame_numbers:
                raise ValueError(
                    f"Duplicate frame index {frame_index} "
                    f"in {instance_dir}"
                )

            seen_frame_numbers.add(frame_index)

            frame = validate_instance_frame(path)

            if expected_shape is None:
                expected_shape = frame.shape

            elif frame.shape != expected_shape:
                raise ValueError(
                    f"{path.name}: shape {frame.shape} differs "
                    f"from expected {expected_shape}"
                )

            frames.append(frame)

        stack = np.stack(frames, axis=0)

        output_path = STACK_DIR / (
            f"{recording_folder.name}_instances_stack.tif"
        )

        tiff.imwrite(
            output_path,
            stack,
            bigtiff=True,
            compression="zlib",
            photometric="minisblack",
            metadata={"axes": "TYX"},
        )

        print(
            f"Saved {output_path} | "
            f"shape={stack.shape} | dtype={stack.dtype}"
        )

    print(f"\nStacking finished: {STACK_DIR}")


# ============================================================
# COLOR INSTANCE LABELS FOR GIF
# ============================================================

def label_frame_to_rgb(frame: np.ndarray) -> Image.Image:
    """
    Render categorical instance IDs as RGB.

    Background = black.
    Each ID receives a deterministic color.
    The same ID therefore keeps the same color across frames.
    """

    labels = np.asarray(frame)

    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2D frame, got {labels.shape}"
        )

    if not np.issubdtype(labels.dtype, np.integer):

        if not np.all(np.isfinite(labels)) or not np.allclose(
            labels, np.round(labels)
        ):
            raise ValueError("Instance labels must be integers")

        labels = np.round(labels).astype(np.int64)

    labels = labels.astype(np.int64, copy=False)

    if np.any(labels < 0):
        raise ValueError("Instance labels must be non-negative")

    rgb = np.zeros(
        (*labels.shape, 3),
        dtype=np.uint8
    )

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

        rgb[labels == instance_id] = (
            red, green, blue
        )

    return Image.fromarray(rgb)


# ============================================================
# CONVERT TIFF STACKS TO GIF
# ============================================================

def convert_tiff_stacks_to_gifs() -> None:

    GIF_DIR.mkdir(parents=True, exist_ok=True)

    if not STACK_DIR.is_dir():
        raise FileNotFoundError(
            f"Stack directory does not exist: {STACK_DIR}"
        )

    tiff_files = sorted(
        list(STACK_DIR.glob("*.tif"))
        + list(STACK_DIR.glob("*.tiff"))
    )

    if not tiff_files:
        print(f"No TIFF stacks found inside: {STACK_DIR}")
        return

    for tiff_file in tiff_files:

        print(f"\nConverting '{tiff_file.name}'...")

        stack = np.asarray(
            tiff.imread(tiff_file)
        )

        if stack.ndim != 3:
            print(
                f"Skipping: expected TYX (3D), "
                f"got shape {stack.shape}"
            )
            continue

        if stack.shape[0] == 0:
            print("Skipping empty stack.")
            continue

        # GIF supports a maximum of 256 colors per frame.
        gif_frames = [
            label_frame_to_rgb(frame).quantize(
                colors=256,
                method=Image.Quantize.MEDIANCUT
            )
            for frame in stack
        ]

        output_path = GIF_DIR / f"{tiff_file.stem}.gif"

        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=GIF_FRAME_DURATION_MS,
            loop=GIF_LOOP,
            optimize=False,
            disposal=2,
        )

        print(
            f"Saved {output_path} "
            f"({len(gif_frames)} frames)"
        )

    print(f"\nGIF conversion finished: {GIF_DIR}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    stack_tracked_instances()
    convert_tiff_stacks_to_gifs()
