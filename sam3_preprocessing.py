import argparse
from dotenv import load_dotenv
import os
import glob
import json
from huggingface_hub import login
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm.auto import tqdm
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import shutil
import subprocess
import zipfile

from utils import *

def run_sam3_preprocessing(split_name, processor_model, is_instance, config):
    """
    is_instance=True : Crop individual tooth instances (Alphadent Instance version)
    is_instance=False: Crop the entire oral region ROI (Alphadent ROI version)
    """
    # 1. Configuration and Path Initialization
    # Distinguish between Instance crop and ROI oral region crop
    mode_str = "INSTANCE" if is_instance else "ROI"
    
    # Path configuration
    base_output_dir = "./data/alphadent_instance" if is_instance else "./data/alphadent_roi" 
    
    # SAM-3 prompt differentiation based on crop mode
    prompt = "teeth" if is_instance else "The complete intraoral area including all teeth and gingiva"
    margin_ratio = 0.15 # Apply 15% margin on all sides for both modes
    
    print(f"🚀 [{split_name.upper()}] Starting {mode_str} Preprocessing (Prompt: {prompt})")

    # Source image and label paths (extracted via loader.py)
    source_img_dir = f"data/alphadent_extracted/images/{split_name}"
    source_lbl_dir = f"data/alphadent_extracted/labels/{split_name}"

    image_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")) +
                         glob.glob(os.path.join(source_img_dir, "*.png")))

    # Create directory structure
    for sub in ["images", "labels", "metadata"]:
        os.makedirs(os.path.join(base_output_dir, sub, split_name), exist_ok=True)

    total_count = 0

    # 2. Start Image Processing Loop
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        file_base = os.path.splitext(os.path.basename(img_path))[0]

        # [Common] Load and resize image
        raw_image = Image.open(img_path).convert("RGB")
        raw_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img_np = np.array(raw_image)
        img_h, img_w = img_np.shape[:2]

        # [Common] SAM-3 Inference
        # Use inference_mode and autocast for optimized performance
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            inference_state = processor_model.set_image(raw_image)
            output = processor_model.set_text_prompt(state=inference_state, prompt=prompt)
        masks = output["masks"] 
        scores = output["scores"]

        if len(masks) == 0:
            print(f"⚠️ {file_base}: Detection failed. Skipping.")
            continue

        # [Common] Load source labels (Train/Valid only)
        raw_lines = []
        lbl_path = os.path.join(source_lbl_dir, file_base + ".txt")
        if split_name != "test" and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                raw_lines = f.readlines()

        metadata_list = []

        # ---------------------------------------------------------
        # 3. Branching Logic: Mask Processing and Cropping
        # ---------------------------------------------------------
        
        # Prepare the list of masks to process
        if is_instance:
            # Process each individual mask separately
            process_masks = [(masks[i].cpu().numpy().squeeze(), float(scores[i]), i) 
                             for i in range(len(masks)) if scores[i] >= config['sam_threshold']]
        else:
            # Combine all detected masks into a single ROI
            combined_mask = torch.any(masks, dim=0).cpu().numpy().squeeze()
            avg_score = float(torch.mean(scores))
            process_masks = [(combined_mask, avg_score, None)]

        for mask_2d, score, idx in process_masks:
            y_indices, x_indices = np.where(mask_2d > 0)
            if len(x_indices) == 0: continue

            # Calculate Bbox and margin padding
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            w_box, h_box = x_max - x_min, y_max - y_min
            pad_x, pad_y = int(w_box * margin_ratio), int(h_box * margin_ratio)

            x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
            x2, y2 = min(img_w, x_max + pad_x), min(img_h, y_max + pad_y)
            crop_w, crop_h = x2 - x1, y2 - y1

            # Crop and save image
            cropped_img = img_np[y1:y2, x1:x2]

            # Filename differentiation: instance_{idx} vs cropped
            suffix = f"instance_{idx:02d}" if is_instance else "cropped"
            instance_name = f"{file_base}_{suffix}"

            img_save_path = os.path.join(base_output_dir, "images", split_name, f"{instance_name}.png")
            cv2.imwrite(img_save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            # Label Remapping
            yolo_lines = []
            if split_name != "test":
                for line in raw_lines:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = parts[0]
                    norm_coords = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                    
                    # Restore original pixel coordinates
                    gt_pts = (norm_coords * [img_w, img_h]).astype(np.float32)

                    if is_instance:
                        # [Instance Mode] Extract only relevant labels using IoU check
                        gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(gt_mask, [gt_pts.astype(np.int32)], 1)
                        intersection = np.logical_and(mask_2d > 0, gt_mask > 0).sum()
                        gt_area = (gt_mask > 0).sum()
                        if (intersection / gt_area if gt_area > 0 else 0) < config['iou_filter_th']:
                            continue
                    
                    # [Common] Transform coordinates to crop coordinate system and normalize
                    local_pts = (gt_pts - [x1, y1]) / [crop_w, crop_h]
                    local_pts = np.clip(local_pts, 0, 1)
                    str_pts = " ".join([f"{p:.6f}" for p in local_pts.flatten()])
                    yolo_lines.append(f"{cid} {str_pts}")

            # Save remapped labels
            lbl_save_path = os.path.join(base_output_dir, "labels", split_name, f"{instance_name}.txt")
            with open(lbl_save_path, "w") as f:
                if yolo_lines: f.write("\n".join(yolo_lines))

            # Append metadata
            metadata_list.append({
                "instance_name": instance_name,
                "crop_coords": [int(x1), int(y1), int(x2), int(y2)],
                "original_size": [img_w, img_h],
                "score": score
            })
            if is_instance: total_count += 1

        # Save JSON metadata
        if metadata_list:
            json_path = os.path.join(base_output_dir, "metadata", split_name, f"{file_base}.json")
            with open(json_path, 'w') as f:
                json.dump(metadata_list, f, indent=4)
        
        if not is_instance: total_count += 1

    print(f"✅ {split_name.upper()} processing complete. ({mode_str} Total: {total_count})")

def extract_with_progress(zip_path, extract_dir):
    """Unzip files with a tqdm progress bar."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc=f"📦 Extracting {os.path.basename(zip_path)}", unit="file", leave=False):
            zip_ref.extract(file, extract_dir)

def prepare_directories():
    """Remove existing incomplete folders and create a fresh directory structure."""
    base_data_dir = "./data"
    sub_dirs = ["alphadent_instance", "alphadent_roi"]

    # 1. Create data folder if missing
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
        print(f"📁 Created base directory: {base_data_dir}")

    # 2. Reset Instance/ROI directories for a clean start
    for sub in sub_dirs:
        target_path = os.path.join(base_data_dir, sub)
        if os.path.exists(target_path):
            shutil.rmtree(target_path) # Delete existing folder
            print(f"🗑️ Deleted existing directory: {target_path}")
        os.makedirs(target_path)
        print(f"✅ Created fresh directory: {target_path}")

def download_all_from_drive():
    """Download preprocessed datasets from GDrive and extract with tqdm progress."""
    prepare_directories()

    files = {
        "alphadent_roi.zip": ("1ZQCzfbQF0uPXtSz_-NHBYnArZoRei-IF", "./data/alphadent_roi"),
        "alphadent_instance.zip": ("1L45ztC85-ZCq76mcg4AXD87gHTBOnO3t", "./data/alphadent_instance")
    }

    for filename, (file_id, target_extract_dir) in files.items():
        zip_path = os.path.join("./data", filename)
        
        print(f"\n🚚 Downloading {filename}...")
        subprocess.run(["gdown", "--id", file_id, "-O", zip_path], check=True)
        
        # Execute progress-aware extraction
        extract_with_progress(zip_path, target_extract_dir)
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"🗑️ Cleaned up temporary file: {zip_path}")

    print("\n✨ All preprocessed datasets are ready!")

def main():
    # 1. Argument Parser Setup
    parser = argparse.ArgumentParser(description="SAM-3 Teeth Segmentation Preprocessing Pipeline")
    
    # --mode: Choose between 'roi' and 'instance' (Runs both if omitted)
    parser.add_argument("--mode", type=str, choices=["roi", "instance"], 
                        help="Execution mode: 'roi' or 'instance'. If omitted, both will run.")
    
    # --split: Choose data split (Runs all if omitted)
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"],
                        help="Data split: 'train', 'valid', or 'test'. If omitted, all splits will run.")
    
    # Download-only option
    parser.add_argument("--from_drive", action="store_true", help="Download preprocessed data from Google Drive instead of processing")
    
    args = parser.parse_args()

    # Case: From Drive download (Skip processing)
    if args.from_drive:
        download_all_from_drive()
        return 

    # 2. Environment Setup (Prepare directories and model)
    prepare_directories()

    load_dotenv()
    hf_token = os.getenv('HUGGINGFACE_API_KEY')
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model().to(device)
    model.eval()
    processor = Sam3Processor(model)
    config = load_config()
    set_seed(42)

    # 3. Determine Execution Targets
    modes_to_run = [args.mode] if args.mode else ["roi", "instance"]
    splits_to_run = [args.split] if args.split else ["train", "valid", "test"]

    # 4. Execute Pipeline
    for m in modes_to_run:
        is_instance = (m == "instance")
        for s in splits_to_run:
            run_sam3_preprocessing(s, processor, is_instance=is_instance, config=config)

if __name__ == "__main__":
    main()