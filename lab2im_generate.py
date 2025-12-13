# lab2im_generate.py - TFM by Daniel Reyes García
import os
import gc
import sys
import time
import random
import numpy as np
import nibabel as nib
import tensorflow as tf
from lab2im import utils
from lab2im.image_generator import ImageGenerator
from lab2im.layers import RandomSpatialDeformation, RandomFlip, GaussianBlur, IntensityAugmentation, MimicAcquisition, BiasFieldCorruption
import imageio

GM_LABELS = set([3,42,9,10,11,12,13,48,49,50,51,52,26,58,
                 17,18,53,54,8,47,16,19,55,27,59,28,60])
WM_LABELS = set([2,41,7,46,251,252,253,254,255,5001,5002])
CSF_LABELS = set([4,43,5,44,14,15,72,24])


def load_nifti(path):
    """Load a NIfTI file and return its data in canonical orientation."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata()


def pad_slice_to_256(slice_img, is_label=False):
    """Pad slices to 256x256 so all images have same size. Labels are padded with background=0."""
    target = 256
    h, w = slice_img.shape

    pad_h = (target - h) // 2
    pad_w = (target - w) // 2

    if is_label:
        bg_value = 0
    else:
        bg_value = np.min(slice_img)

    padded = np.full((256, 256), fill_value=bg_value, dtype=slice_img.dtype)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = slice_img
    return padded

def build_hard_seg_from_pves(base_dir, case_basename):
    """Create a hard segmentation from FSL-FAST PVE maps."""
    pve0 = os.path.join(base_dir, f"{case_basename}_pve_0.nii.gz")
    pve1 = os.path.join(base_dir, f"{case_basename}_pve_1.nii.gz")
    pve2 = os.path.join(base_dir, f"{case_basename}_pve_2.nii.gz")

    if not (os.path.exists(pve0) and os.path.exists(pve1) and os.path.exists(pve2)):
        raise FileNotFoundError("PVE files not found " + base_dir)

    pve0_d = load_nifti(pve0)
    pve1_d = load_nifti(pve1)
    pve2_d = load_nifti(pve2)

    # Stack maps into shape (3, H, W, D) and take argmax
    stacked = np.stack([pve0_d, pve1_d, pve2_d], axis=0)
    hard = np.argmax(stacked, axis=0).astype(np.int32)
    return hard


def remap_fslfast_seg_to_common(seg_hard):
    """Map FSLFAST output labels to the unified {1=GM, 2=WM, 3=CSF} scheme."""
    out = np.zeros_like(seg_hard, dtype=np.int32)
    out[seg_hard == 3] = 1
    out[seg_hard == 2] = 2
    out[seg_hard == 1] = 3
    return out


def load_seg_for_tool(seg_path, tool):
    """Load segmentation for each tool. FSLFast has its own mapping logic;
       other tools use label-sets based on GM/WM/CSF groupings."""
    if tool == "FSLFast":
        if os.path.exists(seg_path):
            seg_hard = load_nifti(seg_path).astype(int)
        else:
            # Fallback to PVE files if hard segmentation not found
            base_dir = os.path.dirname(seg_path)
            case_basename = os.path.basename(seg_path).replace("_seg.nii.gz","").replace("_pveseg.nii.gz","")
            seg_hard = build_hard_seg_from_pves(base_dir, case_basename)
        seg = remap_fslfast_seg_to_common(seg_hard)
    else:
         # General remapping for SynthSeg, Samseg, FastSurfer.
        seg = remap_classes(load_nifti(seg_path))
    return seg


def remap_classes(seg):
    """Convert many anatomical labels into the simplified GM=1, WM=2, CSF=3."""
    out = np.zeros_like(seg, dtype=np.int32)
    out[np.isin(seg, list(GM_LABELS))]  = 1
    out[np.isin(seg, list(WM_LABELS))]  = 2 
    out[np.isin(seg, list(CSF_LABELS))] = 3
    return out


def apply_random_transforms(image, label):
    """Apply random spatial and intensity augmentations. These operations mimic real MRI variability."""
    im = image[np.newaxis, ..., np.newaxis].astype(np.float32)
    lab = label[np.newaxis, ..., np.newaxis].astype(np.int32)

    im, lab = RandomSpatialDeformation(nonlin_std=0.05, scaling_bounds=0.02)([im, lab])
    im = IntensityAugmentation(noise_std=0.05, gamma_std=0.03, contrast_inversion=False)(im)
    
    if np.random.rand() < 0.5:
        im, lab = RandomFlip()([im, lab])

    if np.random.rand() < 0.2:
        im = GaussianBlur(sigma=np.random.uniform(0, 0.1))(im)

    if np.random.rand() < 0.3:
        im = BiasFieldCorruption()(im)

    return im[0, ..., 0], lab[0, ..., 0]


def save_slice_as_png(volume, save_path, is_label=False):
    """Take the center slice of a 3D volume, normalize it, pad to 256, and save."""
    slice_idx = volume.shape[2] // 2
    slice_img = volume[:, :, slice_idx]

    if hasattr(slice_img, "numpy"):
        slice_img = slice_img.numpy()

    if is_label:
        slice_img = slice_img.astype(np.uint8)
    else:
        slice_img = ((slice_img - np.min(slice_img)) / np.ptp(slice_img) * 255).astype(np.uint8)
    
    slice_img = pad_slice_to_256(slice_img)
    imageio.imwrite(save_path, slice_img)



def generate_synthetic_images(seg_path, out_dir, subject, start_idx, end_idx, tool):
     """Main function to generate synthetic MRI slices using Lab2im. Uses ImageGenerator to synthesize volumes, then applies random augmentations."""
    utils.mkdir(out_dir)
    seg = load_seg_for_tool(seg_path, tool)

    # Labels generation corresponding to the classes present in the segmentation
    generation_labels = np.unique(seg)
    generation_classes = np.arange(len(generation_labels))

    original_seg_path = os.path.join(out_dir, "original_seg.nii.gz")
    utils.save_volume(seg, np.eye(4), None, original_seg_path)

    # Create the synthetic image generatorvia lab2im.
    brain_generator = ImageGenerator(
        labels_dir=original_seg_path,
        generation_labels=generation_labels,
        generation_classes=generation_classes,
        prior_distributions="uniform",
        prior_means=[120, 80],
        prior_stds=[15, 10],
        blur_range=0.1,
        target_res=None,
        output_shape=None
    )

    for n in range(start_idx, end_idx+1):
        start = time.time()
        im, lab = brain_generator.generate_image()
        im, lab = apply_random_transforms(im, lab)
        save_slice_as_png(im, os.path.join(out_dir, f"img_{n}.png"))
        save_slice_as_png(lab, os.path.join(out_dir, f"lab_{n}.png"), is_label=True)

        # Free memory (important in case of low resources)
        del im, lab
        gc.collect()

        end = time.time()
        print(f"[{subject}] Generated {n} in {end-start:.2f}s")



def generate_for_split(split: str, out_dir_base: str, cases_dict: dict, n_images_dict: dict, batch_size: int = 50):
    for tool in TOOLS:
        for stripper in STRIPPERS:
            for case in cases_dict[tool]:
                # workaround problematic case
                if tool == "FSLFast" and stripper == "hdbet" and (case in ["sub-01", "sub-04"]):
                    continue

                n_images = n_images_dict.get(tool, 200)
                if tool == "FSLFast" and stripper == "hdbet":
                    n_images = N_IMAGES_FSLFAST_HDBET

                
                if tool == "SynthSeg":
                    seg_path = os.path.join(BASE_DIR, tool, "ds002330", stripper, case, f"{case}_pveseg.nii.gz")
                elif tool == "Samseg":
                    seg_path = os.path.join(BASE_DIR, tool, "ds002330", stripper, case, "seg.nii.gz")
                elif tool == "FastSurfer":
                    seg_path = os.path.join(BASE_DIR, tool, "ds002330", stripper, case, "mri", "aseg.auto_noCCseg.nii.gz")
                elif tool == "FSLFast":
                    seg_path = os.path.join(BASE_DIR, tool, "ds002330", stripper, f"{case}_seg.nii.gz")

                out_dir_case = os.path.join(out_dir_base, tool, "ds002330", stripper, case)
                utils.mkdir(out_dir_case)

                for batch_start in range(1, n_images+1, batch_size):
                    batch_end = min(batch_start + batch_size - 1, n_images)
                    print(f"[{split}] Processing {tool} - {stripper} - {case} images {batch_start}-{batch_end}")
                    generate_synthetic_images(seg_path, out_dir_case, f"{case}_{tool}", batch_start, batch_end, tool)


def set_seed(seed: int = 4224):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == "__main__":
    set_seed()

    BASE_DIR = r"./outputs/SEG"
    TOOLS = ["SynthSeg", "Samseg", "FastSurfer", "FSLFast"]
    STRIPPERS = ["hdbet", "synthstrip"]
    BATCH_SIZE = 50
    N_IMAGES_FSLFAST_HDBET = 300
    
    # TRAIN
    out_dir_train = r"./outputs/SYNTH/train"
    cases_train = {
        "FastSurfer": ["sub-01", "sub-02", "sub-03"],
        "FSLFast": ["sub-01", "sub-02", "sub-03"],
        "SynthSeg": ["sub-01", "sub-02", "sub-03"],
        "Samseg": ["sub-01", "sub-02", "sub-03"],
    }
    n_images_train = {
        "SynthSeg": 200,
        "Samseg": 200,
        "FastSurfer": 200,
        "FSLFast": 300
    }
    generate_for_split("TRAIN", out_dir_train, cases_train, n_images_train, BATCH_SIZE)

    # TEST
    BATCH_SIZE = 10
    N_IMAGES_FSLFAST_HDBET = 60
    out_dir_test = r"./outputs/SYNTH/test"
    cases_test = {
        "FastSurfer": ["sub-04", "sub-05"],
        "FSLFast": ["sub-04", "sub-05"],
        "SynthSeg": ["sub-04", "sub-05"],
        "Samseg": ["sub-04", "sub-05"],
    }
    n_images_test = {
        "SynthSeg": 30,
        "Samseg": 30,
        "FastSurfer": 30,
        "FSLFast": 60
    }
    generate_for_split("TEST", out_dir_test, cases_test, n_images_test, BATCH_SIZE)