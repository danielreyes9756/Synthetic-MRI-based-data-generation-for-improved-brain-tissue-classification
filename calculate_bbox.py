#!/usr/bin/env python3
# calculate_bbox.py - TFM by Daniel Reyes García
import sys
import nibabel as nib
import numpy as np


def calculate_bbox(mask_path):
    """Calculate bounding box from volumetric mask."""
    try:
        img = nib.load(mask_path)
        data = img.get_fdata()
    except Exception as e:
        print(f"[ERROR] Mask was not loaded {mask_path}: {e}", file=sys.stderr)
        return None

    # Find coordinates of all non-zero voxels (foreground mask)
    coords = np.argwhere(data)
    
    if coords.size == 0:
        print("[WARNING] Empty mask, returning original shape.", file=sys.stderr)
        return img.shape, (0, 0, 0)

    # Minimum and maximum voxel coordinates of the mask (bounding box corners)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    bbox_size = max_coords - min_coords + 1
    
    bbox_start = min_coords
    
    print(f"{bbox_size[0]},{bbox_size[1]},{bbox_size[2]};{bbox_start[0]},{bbox_start[1]},{bbox_start[2]}")
    return bbox_size, bbox_start


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: calculate_bbox.py <mask.nii.gz>", file=sys.stderr)
        sys.exit(1)
    
    mask_path = sys.argv[1]
    calculate_bbox(mask_path)