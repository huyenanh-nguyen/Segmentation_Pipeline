#!/usr/bin/env python3
"""Strictly stack only _instances TIFF frames from unet_predictions into multi-page TIFFs
and save them into unet_stack.
"""

from pathlib import Path
import numpy as np
import tifffile as tiff


def stack_unet_instances_strict() -> None:
    source_dir = Path("/u/hanguy/Segmentation_UI/unet_predictions")
    output_dir = Path("/u/hanguy/Segmentation_UI/unet_stack")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    subfolders = [item for item in source_dir.iterdir() if item.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found inside '{source_dir}'.")
        return

    for subfolder in subfolders:
        # Strict filter: Must end with .tif / .tiff AND contain '_instances' in the name
        tiff_files = []
        for f in subfolder.iterdir():
            if f.is_file() and f.suffix.lower() in [".tif", ".tiff"] and "_instances" in f.name.lower():
                tiff_files.append(f)
        
        # Sort numerically by frame index extracted from the filename
        tiff_files = sorted(
            tiff_files,
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
        )
        
        if not tiff_files:
            print(f"Skipping folder '{subfolder.name}' (no instance files matched).")
            continue
        
        print(f"\nProcessing folder '{subfolder.name}' - Found {len(tiff_files)} instance files:")
        # Print a couple of matched files to verify they are strictly instances
        for tf in tiff_files[:3]:
            print(f"  - {tf.name}")
        if len(tiff_files) > 3:
            print(f"  - ... and {len(tiff_files) - 3} more.")

        # Read frames into a list and stack along the time axis (axis 0)
        frames = [tiff.imread(f) for f in tiff_files]
        stack = np.stack(frames, axis=0)  # Shape becomes (T, Y, X)
        
        output_path = output_dir / f"{subfolder.name}_instances_stack.tif"
        
        tiff.imwrite(
            output_path,
            stack,
            bigtiff=True,
            compression="zlib",
            photometric="minisblack",
            metadata={"axes": "TYX"}
        )
        print(f"-> Saved strict instance stack to: {output_path}")

    print(f"\nAll done! Stacks saved to: {output_dir}")


if __name__ == "__main__":
    stack_unet_instances_strict()


# [turning tif into gif]

from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image


def convert_tiff_stacks_to_gifs() -> None:
    # Define exact absolute paths
    source_dir = Path("/u/hanguy/Segmentation_UI/unet_stack")
    output_dir = Path("/u/hanguy/Segmentation_UI/unet_gif")
    
    # Create the output directory if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Find all multi-page TIFF stacks in the source directory
    tiff_files = sorted(list(source_dir.glob("*.tif")) + list(source_dir.glob("*.tiff")))
    
    if not tiff_files:
        print(f"No TIFF stack files found inside '{source_dir}'.")
        return

    for tiff_file in tiff_files:
        print(f"Processing '{tiff_file.name}'...")
        
        # Read the multi-page TIFF stack (shape: T, Y, X)
        stack = tiff.imread(tiff_file)
        
        if stack.ndim != 3:
            print(f"Skipping '{tiff_file.name}': Expected a 3D stack (T, Y, X), but got shape {stack.shape}.")
            continue
        
        frames = []
        for frame in stack:
            # Normalize instance values or categorical labels to 8-bit (0-255) for valid GIF visualization
            min_val = frame.min()
            max_val = frame.max()
            
            if max_val > min_val:
                normalized = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros(frame.shape, dtype=np.uint8)
            
            # Convert NumPy array to a Pillow image
            pil_img = Image.fromarray(normalized)
            frames.append(pil_img)
        
        if not frames:
            print(f"Skipping '{tiff_file.name}': No valid frames extracted.")
            continue
        
        # Define output GIF file path inside 'unet_gif'
        output_path = output_dir / f"{tiff_file.stem}.gif"
        
        # Save as an animated GIF (duration=100ms per frame, loop=0 for infinite loop)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        print(f"-> Saved animated GIF to: {output_path}")

    print(f"\nAll GIFs successfully created and saved to: {output_dir}")


if __name__ == "__main__":
    convert_tiff_stacks_to_gifs()