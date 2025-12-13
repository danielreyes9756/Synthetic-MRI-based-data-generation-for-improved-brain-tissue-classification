#!/bin/bash
# functions.sh - TFM by Daniel Reyes García

set -e

# Avoid multi-thread nondeterminism
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Colors
RED="\033[1;31m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RESET="\033[0m"

# Show help message
show_help() {
cat << EOF
Usage: $0 --i <DATASET_PATH> [--l <LIMIT>] [--sm <SEGMENTATION_METHODS>] [--skm <SKULLSTIPPING_METHODS>] [--t <THREADS>] [--dry-run] [--help]

Arguments:
  --i DATASET_PATH                      Path to the dataset (required)
  --l LIMIT                             Number of subjects to process (default: all)
  --sm SEGMENTATION_METHODS             Segmentation methods: synthseg | fslfast | reconall | fastsurfer | all (default: all)
  --skm SKULLSTIPPING_METHODS           Skull-Stripping methods: synthstrip | watershed | all (default: all)
  --t THREAD                            In case of using CPU instead of GPU you can adapt the number of threads (but at least in wsl this produces performance issues)


Options:
  --dry-run                Show what would be done, but do NOT execute
  --help                   Show this help message

This pipeline performs:
  1. Super-Resolution: Improve image resolution and voxel. 
  2. Skull-Stripping: Remove ... parts.
  3. Normalize: normalize intensities and crop.
  4. Segmentation: Segment parts/tissius.
  3. Image Generation:  Generate synthetic images  (deactivated due to performance issues)

outputs stored at:
  ~/TFM/outputs/
    ├── SR
    |   └── SynthSR/
    ├── STRIP
    |   ├── synthstrip/ (optional depends on --skm)
    |   |   └── stripped, normalized and cropped images (care with FastSurfer crop doesn't work)
    |   └── hdbet/  (optional depends on --skm)
    |   |   └── stripped, normalized and cropped images (care with FastSurfer crop doesn't work)
    ├── SEG  
    |   ├── SynthSeg/   (optional depends on --sm)
    |   ├── FSLFast/    (optional depends on --sm)
    |   ├── FastSurfer/ (optional depends on --sm)
    |   └── Samseg/     (optional depends on --sm)
    └── SYNTH (deactivated due to performance issues)
        └── lab2im/
EOF
exit 0
}

# Logging functions
log_info()    { echo -e "${CYAN}[INFO]:${RESET} $1\n" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]:${RESET} $1\n" >&2; }
log_warning() { echo -e "${YELLOW}[WARNING]:${RESET} $1\n" >&2; }
log_error()   { echo -e "${RED}[ERROR]:${RESET} $1\n" >&2; }

# Command runner, handles dry-run
run_cmd() {
    local cmd="$1"
    local msg="$2"

    if [ "$DRYRUN" = true ]; then
        log_info "[DRY-RUN] $msg"
    else
        log_info "$msg"
        eval "$cmd" >&2
    fi
}

# Performs super-resolution and voxel isotropy resample with mri_synthsr (SynthSR)
perform_synthsr() {
    local sub="$1"
    local t1_input="$2"
    local outdir="$3/$sub"
    local threads="$4"
    
    mkdir -p "$outdir"
    local sr_output="$outdir/${sub}_T1w.nii.gz"


    if [ ! -f "$sr_output" ]; then
        if ! run_cmd "conda run -n fastsurfer mri_synthsr --i '$t1_input' --o '$sr_output'" \
        "Running SynthSR for $sub"; then
            log_error "SynthSR failed for subject: $sub"
            exit 1
        fi
    else
        log_error "SynthSR outputput already exist"
        exit 1
    fi
    echo "$sr_output"
}

# Performs skull-stripping with mri_synthstrip (SynthStrip)
perform_synthstrip() {
    local sub="$1"
    local sr_output="$2"
    local outdir="$3/$sub"

    mkdir -p "$outdir"
    local strip_output="$outdir/${sub}_T1w_brain.nii.gz"

    if [ ! -f "$strip_output" ]; then
        if ! run_cmd "conda run -n fastsurfer mri_synthstrip --i '$sr_output' --o '$strip_output'" \
            "Running SynthStrip for $sub"; then
            log_error "SynthStrip failed for subject: $sub"
            exit 1
        fi
    else
        log_error "SynthStrip outputput already exist"
        exit 1
    fi
    echo "$strip_output"
}

# Performs skull-stripping using hd-bet (HD-BET)
perform_hd_bet() {
    local sub="$1"
    local sr_output="$2"
    local outdir="$3/$sub"

    mkdir -p "$outdir"
    local strip_output="$outdir/${sub}_T1w_brain.nii.gz"

    if [ ! -f "$strip_output" ]; then
        if ! run_cmd "conda run -n hdbet hd-bet -i '$sr_output' -o '$strip_output'" \
            "Running HD-BET skull-stripping for $sub"; then
            log_error "HD-BET failed for subject: $sub"
            exit 1
        fi
    else
        log_error "HD-BET outputput already exist"
        exit 1
    fi
    echo "$strip_output"
}

# Performs segmentation with mri_synthseg (SynthSeg)
perform_synthseg() {
    local sub="$1"
    local strip_output="$2"
     local outdir="$3/$sub"

    mkdir -p "$outdir"
    local seg_output="$outdir/${sub}_pveseg.nii.gz"

    if [ ! -f "$seg_output" ]; then
        if ! run_cmd "conda run -n fastsurfer mri_synthseg --i '$strip_output' --o '$seg_output'" \
            "Running SynthSeg for $sub"; then
            log_error "SynthSeg failed for subject: $sub"
            exit 1
        fi
    else
        log_error "SynthSeg outputput already exist"
        exit 1
    fi
    echo "$seg_output"
}

# Performs segmentation with fast (FSLFast)
perform_fslfast() {
    local sub="$1"
    local strip_output="$2"
    local outdir="$3"

    mkdir -p "$outdir"
    local seg_output="$outdir/${sub}_pveseg.nii.gz"

    if [ ! -f "$seg_output" ]; then
        if ! run_cmd "conda run -n fastsurfer fast -v -n 3 -o '$outdir/$sub' '$strip_output'" \
            "Running FSLFast for $sub"; then
            log_error "FSLFast failed for subject: $sub"
            exit 1
        fi
    else
        log_error "FSLFast outputput already exist"
        exit 1
    fi
    echo "$seg_output"
}

# Performs segmentation with run_fastsurfer (FastSurfer)
perform_fastsurfer() {
    local sub="$1"
    local strip_output="$2"
    local outdir="$3"

    mkdir -p "$outdir"

    local seg_output="$outdir/$sub/mri/aseg.auto_noCCseg.mgz"
    local seg_output_corrected="$outdir/$sub/mri/aseg.auto_noCCseg.nii.gz"

    if [ ! -f "$seg_output" ]; then
        export FASTSURFER_HOME=~/FastSurfer
        chmod +x $FASTSURFER_HOME/run_fastsurfer.sh

        if ! run_cmd "
            export TORCH_USE_CUDA_DSA=1 &&
            conda run -n fastsurfer $FASTSURFER_HOME/run_fastsurfer.sh \
                --t1 '$strip_output' \
                --sid '$sub' \
                --sd '$outdir' \
                --fs_license ~/license.txt \
                --seg_only" \
                "Running FastSurfer for $sub"; then
            log_error "FastSurfer failed for $sub"
            exit 1
        fi
    else
        log_error "FastSurfer outputput exists"
        exit 1
    fi

    if [ ! -f "$seg_output_corrected" ]; then
        if ! run_cmd "conda run -n fastsurfer mri_convert '$seg_output' '$seg_output_corrected'" \
            "Converting FastSurfer aseg.mgz to aseg.nii.gz"; then
            log_error "Convert failed for $sub"
            exit 1
        fi
    else
        log_error "FastSurfer corrected already exist"
        exit 1
    fi
    echo "$seg_output_corrected"
}

# Performs segmentation with SAMSEG (FreeSurfer)
perform_samseg() {
    local sub="$1"
    local strip_output="$2"
    local outdir="$3/$sub"

     mkdir -p "$outdir"

    local seg_output="$outdir/seg.mgz"
    local seg_output_corrected="$outdir/seg.nii.gz"

    if [ ! -f "$seg_output" ]; then
        if ! run_cmd "conda run -n fastsurfer run_samseg -i '$strip_output' -o '$outdir'" \
            "Running SAMSEG for $sub"; then
            log_error "SAMSEG failed for $sub"
            exit 1
        fi
    else
        log_error "SAMSEG corrected already exist"
        exit 1
    fi

    if [ ! -f "$seg_output_corrected" ]; then
        if ! run_cmd "conda run -n fastsurfer mri_convert '$seg_output' '$seg_output_corrected'" \
            "Converting FastSurfer aseg.mgz to aseg.nii.gz"; then
            log_error "Convert failed for $sub"
            exit 1
        fi
    else
        log_error "SAMSEG corrected already exist"
        exit 1
    fi
    echo "$seg_output_corrected"
}

# Performs normalization with mri_normalize
perform_normalize() {
    local sub="$1"
    local strip_output="$2"
    local outdir="$3/$sub"

    mkdir -p "$outdir"
    local norm_output="$outdir/${sub}_T1w_brain_norm.nii.gz"

    if [ ! -f "$norm_output" ]; then
        if ! run_cmd "conda run -n fastsurfer mri_normalize '$strip_output' '$norm_output'" \
            "Normalizing intensities using FreeSurfer mri_normalize for $sub"; then
            log_error "Normalize failed for $sub"
            exit 1
        fi
    else
        log_error "Normalize outputput already exist"
        exit 1
    fi
    echo "$norm_output"
}

perform_crop() {
    local sub="$1"
    local strip_output="$2"
    local outdir="$3/$sub"
    local start_coords="$4"
    local crop_size="$5"
    local output_prefix=$(basename "${strip_output%.nii.gz}")
    local crop_output="$outdir/${output_prefix}_crop.nii.gz"

    mkdir -p "$outdir"

    if [ ! -f "$crop_output" ]; then
        if ! run_cmd "conda run -n fastsurfer python ~/crop_bbox.py '$strip_output' '$crop_output' '$start_coords' '$crop_size'" \
            "Cropping image $sub from $start_coords to size $crop_size"; then
            log_error "Cropping failed for $sub"
            exit 1
        fi
    else
        log_error "Crop outputput already exist"
        exit 1
    fi

    echo "$crop_output"
}

# Performs image synthetization lab2im_generate
# perform_lab2im() {
#     local sub="$1"
#     local seg_input="$2"
#     local outdir="$3/$sub"
#     local numberimages="$4" 

#     mkdir -p "$outdir"

#     local synth_output="$outdir/${sub}_T1w_synth.nii.gz"

#     if [ ! -f "$synth_output" ]; then
#         # this fix ram issue 
#         for n in $(seq 1 $numberimages); do
#             if ! run_cmd "conda run -n lab2im python ~/lab2im_generate.py '$seg_input' '$outdir' '$sub' '$n'" \
#                 "Generating synthetic image $n for $sub"; then
#                 log_error "Lab2im failed for image $n of $sub"
#                 exit 1
#             fi
#         done
#     else
#         log_error "Lab2im outputput already exist"
#         exit 1
#     fi
# }