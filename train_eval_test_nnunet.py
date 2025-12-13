import os
import json
from pathlib import Path
import shutil
import subprocess

BASE_OUTPUT = Path("./outputs/nnunet")
RAW_DATA_ROOT = BASE_OUTPUT / "raw_data"
PREPROCESSED_ROOT = BASE_OUTPUT / "preprocessed"
RESULTS_ROOT = BASE_OUTPUT / "results"

# export for nnUNet (required)
os.environ["nnUNet_raw"] = str(RAW_DATA_ROOT.resolve())
os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_ROOT.resolve())
os.environ["nnUNet_results"] = str(RESULTS_ROOT.resolve())


SYNTH_ROOT = Path("./outputs/SYNTH")
LABELS = {
    "background": 0,
    "GM": 1,
    "WM": 2,
    "CSF": 3
}
conda_env = "tfm-nnUnet"


def collect_images_by_config():
    data = {}
    
    for split in ["train", "test"]:
        split_root = SYNTH_ROOT / split
        if not split_root.exists():
            continue
        
        for segmenter_dir in split_root.iterdir():
            if not segmenter_dir.is_dir():
                continue
            
            segmenter = segmenter_dir.name
            ds_dir = segmenter_dir / "ds002330"
            if not ds_dir.exists():
                continue
            
            for stripper_dir in ds_dir.iterdir():
                if not stripper_dir.is_dir():
                    continue
                
                stripper = stripper_dir.name
                key = (segmenter, stripper)
                
                if key not in data:
                    data[key] = {"train": [], "test": []}
                
                for subject_dir in stripper_dir.iterdir():
                    if not subject_dir.is_dir():
                        continue
                    
                    img_paths = sorted(subject_dir.glob("img_*.png"))
                    lab_paths = sorted(subject_dir.glob("lab_*.png"))
                    
                    if len(img_paths) != len(lab_paths):
                        print(f"MISMATCH: {subject_dir}")
                        continue
                    
                    for img, lab in zip(img_paths, lab_paths):
                        data[key][split].append({
                            "img": img,
                            "lab": lab,
                        })
    
    return data


def create_nnunet_datasets(data, starting_id=1):
    dataset_meta = {}
    dataset_id = starting_id
    
    for (segmenter, stripper), splits in data.items():
        name = f"Dataset{dataset_id:03d}_{segmenter}_{stripper}"
        dataset_meta[name] = f"{dataset_id:03d}"

        output_dir = RAW_DATA_ROOT / name
        imagesTr_dir = output_dir / "imagesTr"
        labelsTr_dir = output_dir / "labelsTr"
        imagesTs_dir = output_dir / "imagesTs"

        imagesTr_dir.mkdir(parents=True, exist_ok=True)
        labelsTr_dir.mkdir(parents=True, exist_ok=True)
        imagesTs_dir.mkdir(parents=True, exist_ok=True)

        # train dataset
        for idx, item in enumerate(splits["train"]):
            case = f"{idx:05d}"
            shutil.copy2(item["img"], imagesTr_dir / f"{name}_{case}_0000.png")
            shutil.copy2(item["lab"], labelsTr_dir / f"{name}_{case}.png")

        # test dataset
        for idx, item in enumerate(splits["test"]):
            case = f"{idx:05d}"
            shutil.copy2(item["img"], imagesTs_dir / f"{name}_{case}_0000.png")

        # dataset.json
        dataset_json = {
            "channel_names": {"0": "MRI"},
            "labels": LABELS,
            "numTraining": len(splits["train"]),
            "file_ending": ".png"
        }

        with open(output_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=4)

        dataset_id += 1

    return dataset_meta


def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def run_nnunet_pipeline(dataset_meta):
    conda_env = "tfm-nnUnet"
    model_config = "2d"
    device = "cuda"
    folds = [0]

    for dataset_name, dataset_id in dataset_meta.items():
        # precporcess
        cmd = f"conda run -n {conda_env} nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
        run_cmd(cmd)

        for fold in folds:
            # train
            cmd = (
                f"conda run -n {conda_env} nnUNetv2_train {dataset_id} "
                f"{model_config} {fold} --npz -device {device}"
            )
            run_cmd(cmd)

            # prediction
            input_folder = RAW_DATA_ROOT / dataset_name / "imagesTs"
            output_folder = RESULTS_ROOT / dataset_name / "test_results" / f"fold_{fold}"
            output_folder.mkdir(parents=True, exist_ok=True)

            cmd = (
                f"conda run -n {conda_env} nnUNetv2_predict "
                f"-i {input_folder} -o {output_folder} "
                f"-d {dataset_id} -c {model_config} -f {fold} "
                f"-chk checkpoint_best.pth --save_probabilities -device {device}"
            )
            run_cmd(cmd)

            # postprocessing
            cmd = f"conda run -n {conda_env} nnUNetv2_find_best_configuration {dataset_id} -c 2d -f 0"
            run_cmd(cmd)


            input_folder = RESULTS_ROOT / dataset_name / "test_results" / f"fold_{fold}"
            postprocess_folder = RESULTS_ROOT / dataset_name / "post_process" / f"fold_{fold}"
            postprocess_folder.mkdir(parents=True, exist_ok=True)

            crossval_results_folder = RESULTS_ROOT / dataset_name / "nnUNetTrainer__nnUNetPlans__2d" / f"crossval_results_folds_{fold}"

            pp_file = crossval_results_folder / "postprocessing.pkl"
            plans_json = crossval_results_folder / "plans.json"
            dataset_json = crossval_results_folder / "dataset.json"

            cmd = (
                f"conda run -n {conda_env} nnUNetv2_apply_postprocessing "
                f"-i {input_folder} "
                f"-o {postprocess_folder} "
                f"-pp_pkl_file {pp_file} "
                f"-plans_json {plans_json} "
                f"-dataset_json {dataset_json}"
            )
            run_cmd(cmd)

if __name__ == "__main__":
    data = collect_images_by_config()
    dataset_meta = create_nnunet_datasets(data, starting_id=1)
    run_nnunet_pipeline(dataset_meta)
