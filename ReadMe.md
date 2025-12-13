# TFM
Generation of synthetic MRI images using  lab2im for automatic classification of brain tissues. 
---
**Abstract**

The present work focuses on the generation of realistic synthetic magnetic resonance imaging (MRI) data using lab2im [1], with the goal of enhancing the automatic classification of brain tissues. Specifically, segmentation maps of gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF) are employed as structural inputs to generate anatomically consistent MRI volumes.

The proposed workflow establishes an end-to-end preprocessing and synthetic image generation pipeline.Original MRI datasets obtained from OpenNeuro are standardized in resolution
and contrast using SynthSR [2], followed by skull-stripping with SynthStrip [3] to remove non-brain tissues. Tissue segmentation is then performed with SynthSeg [1], producing high-quality GM, WM, and CSF masks under variable contrast conditions. These masks serve as input to lab2im, a probabilistic generative model based on Gaussian Mixture Models (GMMs) that synthesizes new MRI volumes with realistic intensity distributions and spatial variations. To enrich the dataset, spatial deformations and intensity-based augmentations are applied to the synthetic images, improving diversity and robustness. The final dataset, composed of real and synthetic MRI images with corresponding labels, is used to train deep learning models, for tissue classification.

# Integrated Pipeline for Brain Tissue Segmentation Using Synthetic Data

## 1. Image Harmonization and Preprocessing

All MRI volumes are first standardized to ensure spatial and intensity consistency across subjects.

### Contrast Harmonization with SynthSR

**SynthSR [2]** is applied to all T1-weighted images to map them into a common high-resolution, contrast-harmonized space. This step serves as the unified entry point of the pipeline and reduces variability due to acquisition differences.

*SynthSR maps heterogeneous T1-weighted MRI volumes into a common high-resolution and contrast-normalized space, reducing inter-subject variability.*
![Original MRI](figures/raw.png)
![SynthSR normalized output](figures/SR-normalization.png)

### Skull-Stripping

Two state-of-the-art brain extraction methods are applied independently:

- **SynthStrip [3]**
![Skull-stripping by SynthStrip (normalization and crop included)](figures/normalized_crop_included.png)

- **HD-Bet [4]**
Isensee, F., Schell, M., Tursunova, I., Brugnara, G., Bonekamp, D., Neuberger, U.,  
Wick, A., Schlemmer, H. P., Heiland, S., Wick, W., Bendszus, M.,  
Maier-Hein, K. H., & Kickingereder, P. (2019).  
*Automated brain extraction of multi-sequence MRI using artificial neural networks.* 

![Skull-stripping by hdber (normalization and crop included)](figures/normalized_crop_included_hdbet.png)

This creates parallel preprocessing branches that allow direct comparison of skull-stripping strategies.

### Intensity Normalization and Cropping

After skull-stripping, images undergo:

- intensity normalization
- automatic bounding-box computation
- cropping to remove non-brain background

This ensures consistent spatial support across subjects.

---

## 2. Tissue Segmentation and Label Harmonization

Each preprocessed image is segmented using four different tools:

- **FastSurfer [6]**
    ![seg by FastSurfer and HD-Bet](figures/seg_fastsurfer_hdbet_01.png)
    ![seg by FastSurfer and SynthStrip](figures/seg_fastsurfer_synthstrip_01.png)
- **FSL FAST [7]**
    ![seg by FSLFast and HD-Bet](figures/seg_fslfat_hdbet_02.png)
    ![seg by FSLFast and SynthStrip](figures/seg_fslfast_synthstrip_01.png)
- **SAMSEG [1]**
    ![seg by SAMSEG and HD-Bet](figures/seg_fastsurfer_hdbet_01.png)
    ![seg by SAMSEG and SynthStrip](figures/seg_samseg_synthstrip_01.png)
- **SynthSeg [1]**
    ![seg by SynthSeg and HD-Bet](figures/seg_SynthSeg_hdbet_01.png)
    ![seg by SynthSeg and SynthStrip](figures/seg_SynthSeg_synthstrip_01.png)

Since these tools produce heterogeneous anatomical label sets, all outputs are remapped into a unified tissue representation:

- **Gray Matter (GM)**
- **White Matter (WM)**
- **Cerebrospinal Fluid (CSF)**

This harmonization enables fair comparison across segmentation methods.

---

## 3. Synthetic Data Generation

Synthetic MRI images are generated from the simplified tissue maps using **lab2im** [1].


### Slice or 2DImage Synthesis

Lab2im generates anatomically plausible MRI images by sampling tissue-specific intensity distributions and applying spatial deformations.

### Data Augmentation

To increase realism and variability, the following transformations are applied:

- random spatial deformations
- intensity augmentation
- Gaussian blur
- bias field corruption
- random flipping

### 2D Slice Extraction

From each synthesized volume, the central axial slice is extracted, padded to **256 × 256**, and saved as PNG. These slices constitute the final training and test datasets.

Each combination of **(segmentation method × skull-stripping method)** results in a distinct synthetic dataset.

At this point labels are also remapped to have only 4 possible values {0: background, 1: GM, 2: WM, 3: CSF}

![lab2im case](figures/lab2im-case.png)

---

## 4. Model Training and Evaluation

### Model

**nnUNet [5]** is used to train 2D tissue segmentation models on each synthetic dataset.

### Self-Configuring Framework

nnUNet automatically adapts its architecture and training strategy to the data, enabling fair comparison without manual hyperparameter tuning.

### Evaluation

Performance is assessed using Dice similarity coefficients for **GM**, **WM**, and **CSF**, allowing analysis of:

- overall segmentation quality
- tissue-specific failure modes
- robustness across preprocessing strategies


## Scripts & Env
There are 6 scripts (.sh and .py) in the main folder:
- ``workflow.sh`` (main workflow - ``WSL``)
- ``functions.sh`` (helper for ``workflow.sh``)
- ``calculate_bbox.py`` (helper for ``workflow.sh``)
- ``crop_bbox.py`` (helper for ``workflow.sh``)
- ``lab2im_generate.py`` (image synthesis by ``lab2im``, important run after ``workflow.sh``)
- ``train_eval_test_nnunet.py`` (train and evaluation by ``nnUnet`` , important run after ``lab2im_generate.py``)

There are 4 envs placed on a folder called envs:
- ``environment_tfm-fastsurfer`` (for ``workflow.sh`` and related ``WSL``)
- ``environment_tfm-lab2im`` (for ``lab2im_generate.py``)
- ``environment_tfm_nnunet`` (for ``train_eval_test_nnunet.py``)
- ``environment_tfm-analytics`` (for ``analitycs.ipynb``)

Final analysis can be found on ``analitycs.ipynb``


## Outputs

By ``worflow.sh``
```
- SR: Super-resolution made by SynthSR
  - SynthSR
- STRIP: Skull-stripping (hdbet or synthseg depending on the user selection) and manual normalization (crop)
    - hdbet
    - synthseg
- SEG: Segmentation (FastSufer, FSLFast, SyntSeg or SamSeg depending on the user selection)
    - FastSufer
    - FSLFast
    - SyntSeg
    - SamSeg
```

By ``lab2im_generate.py``
```
- SYNTH: Contains the synthetic images generated by lab2im separated by segmenter, stripper and sub.
    - train
        - ...
    - test
        - ...
```

By ``train_eval_test_nnunet.py``
```     
- nnunet
    - preprocessed
    - raw_data
    - results
```

### Notes:
``lab2im_generate.py`` the number of synthetic images generated per case is controlled by the variables ``n_images_train`` and ``n_images_test``, as well as the constant ``N_IMAGES_FSLFAST_HDBET``. These values are defined in the ``if __name__ == "__main__":`` block and can be easily modified by the user to change the size of the training or testing datasets.

Furthermore, the script contains a specific workaround for the FSLFast tool when paired with the hdbet stripper:

**workaround problematic case**
```
if tool == "FSLFast" and stripper == "hdbet" and (case in ["sub-01", "sub-04"]):
    continue
```

This conditional statement explicitly skips the image generation process for the ``FSLFast/hdbet`` combination on subjects ``sub-01`` (in the ``TRAIN`` split) and ``sub-04`` (in ``TEST`` splits). This is implemented to bypass issues where the segmentation step failed or produced unusable results for these specific cases, thus preventing subsequent errors during the synthetic image generation process.

### Image Configuration

- **format:** .png
- **size:** 256x256
- **normalized:** true
- **amount_synthetics:** all (660 per case generated by lab2im {600: train and validation, 60: test})
- **labels:** {0: background, 1: GM, 2: WM, 3: CSF}

### Notes:
The total number of synthetic images generated (`amount_synthetics`) is currently **limited** and lower than initially planned for a comprehensive train/validation split.

This limitation arose primarily due to several **technical and environmental challenges**:

1.  **Version and Compatibility Errors:** Significant time was spent resolving complex versioning conflicts and compatibility issues involving Python libraries (e.g., TensorFlow, PyTorch, CUDA) and the specific GPU hardware used for accelerated processing.
2.  **Extended Development Time:** These technical hurdles, combined with minor developmental setbacks, resulted in considerable delays to the project timeline.

As this task is highly time-consuming, the final dataset size had to be constrained to meet the project deadline. However, the existing dataset still follows the structured methodology derived from the multi-tool workflow.

## Dataset

### Neuroimaging predictors of creativity in healthy adults (ds002330)

- **Authors:** Sunavsky, A., Poppenk, J.
- **link:** https://openneuro.org/datasets/ds002330/versions/1.1.0
- **doi:** doi:10.18112/openneuro.ds002330.v1.1.0
- **Participants:** 66
- **Uploaded by:** Jordan Poppenk on 2019-11-21 - over 5 years ago
- **Last Updated:** 2020-01-14 - over 5 years ago
- **License:** CC0
- **cite**: Sunavsky, A. and Poppenk, J. (2020). Neuroimaging predictors of creativity in healthy adults. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds002330.v1.1.0

## Tools and Resources

All software tools used in this work are publicly available and widely adopted
in the neuroimaging community, ensuring reproducibility and methodological transparency.


| Tool | Description | Reference Link |
| :--- | :--- | :--- |
| **lab2im** | Framework for synthesizing realistic anatomical MRI images from label maps. | [https://github.com/vcasellesb/lab2im/tree/master/lab2im](https://github.com/vcasellesb/lab2im/tree/master/lab2im) |
| **nnUNet** | Self-configuring, state-of-the-art segmentation and classification framework. | [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| **SynthSR** | Super-Resolution tool for converting MR volumes into a high-resolution, T1w harmonized space. | [https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR) |
| **SynthSeg** | Deep learning segmentation tool robust to image quality and contrast variability. | [https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) |
| **FSLFast** | FMRIB's Automated Segmentation Tool (FAST) for tissue segmentation (used for Partial Volume Estimation). | [https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FAST.html](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FAST.html) |
| **Samseg** | Probabilistic atlas-based segmentation tool from the FreeSurfer suite. | [https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg](https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg) |
| **FreeSurfer** | Comprehensive suite for processing and analyzing brain MRI data. (Contextual link for related tools). | [https://surfer.nmr.mgh.harvard.edu/fswiki](https://surfer.nmr.mgh.harvard.edu/fswiki) |
| **HD-Bet** | High Definition Brain Extraction Tool, a deep learning model for skull-stripping. | [https://github.com/MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET) |
| **SynthStrip** | Contrast-agnostic skull-stripping model used for accurate brain extraction. | [https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) |





# Cites:
```
[1] Billot, B., Greve, D. N., Puonti, O., Thiran, J. P., Van Leemput, K., & Iglesias, J. E. (2020).
A learning strategy for contrast-agnostic MRI segmentation.
Medical Image Analysis, 60, 101618.

[2] Iglesias, J. E., et al. (2023).
SynthSR: Super-resolution and contrast harmonization of MRI using deep learning.
NeuroImage, 266, 119387.

[3] Hoopes, A., Mora, J. S., Dalca, A. V., Fischl, B., & Hoffmann, M. (2022).
SynthStrip: Skull-stripping for any brain image.
NeuroImage, 260, 119474.
https://doi.org/10.1016/j.neuroimage.2022.119474

[4] Isensee, F., et al. (2019).
Automated brain extraction of multi-sequence MRI using artificial neural networks.
Human Brain Mapping, 40(17), 4952–4964.
https://doi.org/10.1002/hbm.24750

[5] Isensee, F., et al. (2020).
nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18, 203–211.
https://doi.org/10.1038/s41592-020-01008-z

[6] Fischl, B. (2012).
FreeSurfer.
NeuroImage, 62(2), 774–781.
https://doi.org/10.1016/j.neuroimage.2012.01.021

[7] Zhang, Y., Brady, M., & Smith, S. (2001).
Segmentation of brain MR images through a hidden Markov random field model.
IEEE Transactions on Medical Imaging, 20(1), 45–57.
https://doi.org/10.1109/42.906424

```