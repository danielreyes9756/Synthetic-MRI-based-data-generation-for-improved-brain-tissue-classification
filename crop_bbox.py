#!/usr/bin/env python3
# crop_bbox.py - TFM by Daniel Reyes García
import sys
import nibabel as nib
import numpy as np


def crop_bbox(input_path, output_path, start_coords, bbox_size):
    """Crop volumen based on a calculated bounding box."""
    img = nib.load(input_path)
    data = img.get_fdata()
    
    sx, sy, sz = start_coords
    dx, dy, dz = bbox_size

    # Compute the ending coordinates of the bounding box crop
    end_coords = [sx + dx, sy + dy, sz + dz]
    
    # Perform the actual 3D cropping operation
    cropped_data = data[sx:end_coords[0], sy:end_coords[1], sz:end_coords[2]]

    # Update the affine matrix so that spatial coordinates remain correct
    # Shift the origin by applying the affine transform to the start offset
    new_affine = img.affine.copy()
    new_affine[:3, 3] = new_affine[:3, 3] + np.dot(new_affine[:3, :3], start_coords)

    nib.save(nib.Nifti1Image(cropped_data, new_affine, img.header), output_path)
    print(f"[INFO] Cropped image saved to {output_path}")



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: crop_bbox.py <input.nii.gz> <output.nii.gz> <start_x,y,z> <size_x,y,z>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    start_coords = tuple(map(int, sys.argv[3].split(',')))
    bbox_size = tuple(map(int, sys.argv[4].split(',')))
    
    crop_bbox(input_path, output_path, start_coords, bbox_size)