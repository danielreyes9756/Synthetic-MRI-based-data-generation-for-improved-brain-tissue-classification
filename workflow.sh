#!/bin/bash
# workflow.sh - TFM by Daniel Reyes García

source functions.sh
set -e

# Defaults
DATASET_PATH=""
LIMIT="all"
THREADS=1
DRYRUN=false
SEGMENTATION_METHODS=("all")
SKULLSTIPPING_METHODS=("all")
# NUMBERIMAGES=1500
step=1

# ------------------ ARG PARSING ------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --i|--input) DATASET_PATH="$2"; shift 2 ;;
        --l|--limit) LIMIT="$2"; shift 2 ;;
        --sm|--segmentation) IFS=', ' read -r -a SEGMENTATION_METHODS <<< "$2"; shift 2 ;;
        --skm|--skullstrip) IFS=', ' read -r -a SKULLSTIPPING_METHODS <<< "$2"; shift 2 ;;
        --t|--threads) THREADS="$2"; shift 2 ;;
        --dry-run) DRYRUN=true; shift ;;
        # --n|--numberimages) NUMBERIMAGES="$2"; shift 2 ;;
        --help) show_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ------------------ PARAM VALIDATION ------------------

[ -z "$DATASET_PATH" ] && log_error "Dataset path cannot be empty" && exit 1
[ ! -d "$DATASET_PATH" ] && log_error "Dataset path '$DATASET_PATH' not found" && exit 1
# if ! [[ "$NUMBERIMAGES" =~ ^[0-9]+$ ]]; then
#     log_error "Invalid amount of images require a number"
#     exit 1
# fi

VALID_SM_METHODS=("synthseg" "fslfast" "fastsurfer" "samseg" "all")
for sm in "${SEGMENTATION_METHODS[@]}"; do
    if [[ ! " ${VALID_SM_METHODS[*]} " =~ " $sm " ]]; then
        log_error "Invalid segmentation method '$sm'. Valid: synthseg, fslfast, samseg, fastsurfer, all"
        exit 1
    fi
done

if [[ " ${SEGMENTATION_METHODS[*]} " =~ " all " ]]; then
    SEGMENTATION_METHODS=("synthseg" "fslfast" "samseg" "fastsurfer")
fi

VALID_SKM_METHODS=("synthstrip" "hdbet" "all")

for skm in "${SKULLSTIPPING_METHODS[@]}"; do
    if [[ ! " ${VALID_SKM_METHODS[*]} " =~ " $skm " ]]; then
        log_error "Invalid skull-stripping method '$skm'. Valid: synthstrip, hdbet, all"
        exit 1
    fi
done

if [[ " ${SKULLSTIPPING_METHODS[*]} " =~ " all " ]]; then
    SKULLSTIPPING_METHODS=("synthstrip" "hdbet")
fi

# ------------------ PATHS ------------------

DATASET_NAME=$(basename "$DATASET_PATH")
OUTDIR_BASE=~/TFM/outputs

# Super-resolution
SYNTHSR_OUTDIR="$OUTDIR_BASE/SR/SynthSR/$DATASET_NAME"

# Skull-stripping
SYNTHSTRIP_OUTDIR="$OUTDIR_BASE/STRIP/SynthStrip/$DATASET_NAME"
HDBET_OUTDIR="$OUTDIR_BASE/STRIP/hd-bet/$DATASET_NAME"

# Segmentations
SYNTHSEG_OUTDIR="$OUTDIR_BASE/SEG/SynthSeg/$DATASET_NAME"
FSLFAST_OUTDIR="$OUTDIR_BASE/SEG/FSLFast/$DATASET_NAME"
SAMSEG_OUTDIR="$OUTDIR_BASE/SEG/Samseg/$DATASET_NAME"
FASTSURFER_OUTDIR="$OUTDIR_BASE/SEG/FastSurfer/$DATASET_NAME"

# Synthetic images
LAB2IM_OUTDIR="$OUTDIR_BASE/SYNTH/lab2im/$DATASET_NAME"

# ------------------ SUBJECTS ------------------

mapfile -t SUBJECTS < <(ls "$DATASET_PATH" | grep -E "^sub-[0-9]+" | sort)
TOTAL=${#SUBJECTS[@]}
N_TO_PROC=$([[ "$LIMIT" == "all" ]] && echo "$TOTAL" || echo $(( LIMIT > TOTAL ? TOTAL : LIMIT )))

# ------------------ WORKFLOW ------------------

for (( i=0; i< N_TO_PROC; i++ )); do
    SUB=${SUBJECTS[$i]}
    BASENAME="${SUB}_T1w"
    T1="$DATASET_PATH/$SUB/anat/${SUB}_T1w.nii.gz"
    
    if [ ! -f "$T1" ]; then
        log_warning "T1 not found for $SUB, skipping..."
        continue
    fi

    log_info "Processing $SUB..."
    start=$(date +%s)

    {   
        # Super Resolution
        SR_OUT=$(perform_synthsr "$SUB" "$T1" "$SYNTHSR_OUTDIR" "$THREADS")
        log_info "Normalization (Skull-Stripping + Intensity Normalization + Crop Bounding Box)"

        for skm in "${SKULLSTIPPING_METHODS[@]}"; do
            log_info ">> Skull-Stripping ($skm)"
            case $skm in
                synthstrip)
                    SS_FILE=$(perform_synthstrip "$SUB" "$SR_OUT" "$SYNTHSTRIP_OUTDIR")
                    NORM_DIR="$SYNTHSTRIP_OUTDIR"
                    ;;
                hdbet)
                    SS_FILE=$(perform_hd_bet "$SUB" "$SR_OUT" "$HDBET_OUTDIR")
                    NORM_DIR="$HDBET_OUTDIR"
                    ;;
                *)
                    log_warning "Unknown skull stripping method: $skm"
                    exit 1
                    ;;
            esac
            
            log_info ">> Applying intensity normalization for $skm output..."
            NORM_FILE=$(perform_normalize "$SUB" "$SS_FILE" "$NORM_DIR")

            log_info ">> Calculating BBOX for $skm output..."
            BBOX_OUTPUT=$(conda run -n fastsurfer python ~/calculate_bbox.py "$NORM_FILE")

            if [ $? -ne 0 ] || [ -z "$BBOX_OUTPUT" ] || ! echo "$BBOX_OUTPUT" | grep -q ';'; then
                log_error "Failed to calculate BBOX for $SUB ($skm). Skipping segmentation loop."
                exit 1
            fi

            BBOX_SIZE=$(echo "$BBOX_OUTPUT" | cut -d ';' -f 1)
            BBOX_START=$(echo "$BBOX_OUTPUT" | cut -d ';' -f 2)
            
            log_info ">> Performing cropping for $skm output..."
            CROP_FILE=$(perform_crop "$SUB" "$NORM_FILE" "$NORM_DIR" "$BBOX_START" "$BBOX_SIZE")

            # Segmentations
            log_info "Segmenting"

            for sm in "${SEGMENTATION_METHODS[@]}"; do
                log_info ">> Applying $sm"

                case $sm in
                    synthseg)
                        SEG_FILE=$(perform_synthseg "$SUB" "$CROP_FILE" "$SYNTHSEG_OUTDIR/${skm}")
                        ;;
                    fslfast)
                        SEG_FILE=$(perform_fslfast "$SUB" "$CROP_FILE" "$FSLFAST_OUTDIR/${skm}")
                        ;;
                    fastsurfer)
                        SEG_FILE=$(perform_fastsurfer "$SUB" "$CROP_FILE" "$FASTSURFER_OUTDIR/${skm}")
                        ;;
                    samseg)
                        SEG_FILE=$(perform_samseg "$SUB" "$CROP_FILE" "$SAMSEG_OUTDIR/${skm}")
                        ;;
                    *)
                        log_warning "Unknown segmentation method: $sm"
                        continue
                        ;;
                esac
                
                log_info "Image Synthesis and Slice Selection"
                log_info ">> Applying lab2im and manual slice selection (middle slice)"
                # perform_lab2im "$SUB" "$SEG_FILE" "$LAB2IM_OUTDIR/${sm}/${skm}" "$NUMBERIMAGES"
            done

        done
        
        end=$(date +%s)
        log_success "Subject $SUB done in $((end-start)) sec"
    } || {
        log_error "Error while processing $SUB, skipping..."
    }
    
done

log_success ""
log_success "=============================="
log_success "   WORKFLOW Done   "
log_success "=============================="