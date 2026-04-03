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
    is_instance=True : 개별 치아 인스턴스 크롭 (알파덴트 인스턴스 버전)
    is_instance=False: 구강 영역 전체 ROI 크롭 (알파덴트 ROI 버전)
    """
    # 1. 설정 및 경로 초기화
    # 인스턴스 크롭 / ROI 구강영역 크롭으로 구분
    mode_str = "INSTANCE" if is_instance else "ROI"
    # 경로 구분
    base_output_dir = "./data/alphadent_instance" if is_instance else "./data/alphadent_roi" 
    # 크롭 방식에 따른 SAM-3 프롬프트 구분
    prompt = "teeth" if is_instance else  "The complete intraoral area including all teeth and gingiva"
    margin_ratio = 0.15 # 두 방식 모두 상하좌우 15% 마진 두어 크롭
    
    print(f"🚀 [{split_name.upper()}] {mode_str} 전처리 시작 (Prompt: {prompt})")

    # loader.py 를 통해 받은 원본 이미지와 라벨 경로
    source_img_dir = f"data/alphadent_extracted/images/{split_name}"
    source_lbl_dir = f"data/alphadent_extracted/labels/{split_name}"

    # 출력 경로
    base_output_dir = "data/alphadent_instance" if is_instance else "data/alphadent_roi"

    image_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")) +
                         glob.glob(os.path.join(source_img_dir, "*.png")))

    for sub in ["images", "labels", "metadata"]:
        os.makedirs(os.path.join(base_output_dir, sub, split_name), exist_ok=True)

    total_count = 0

    # 2. 이미지 루프 시작
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        file_base = os.path.splitext(os.path.basename(img_path))[0]

        # [공통] 이미지 로드 및 리사이즈
        raw_image = Image.open(img_path).convert("RGB")
        raw_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img_np = np.array(raw_image)
        img_h, img_w = img_np.shape[:2]

        # [공통] SAM-3 추론
        # inference_mode와 autocast를 중첩해서 사용.
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            inference_state = processor_model.set_image(raw_image)
            output = processor_model.set_text_prompt(state=inference_state, prompt=prompt)
        masks = output["masks"] 
        scores = output["scores"]

        if len(masks) == 0:
            print(f"⚠️ {file_base}: 탐지 실패. 스킵합니다.")
            continue

        # [공통] 원본 라벨 로드 (Train/Valid 전용)
        raw_lines = []
        lbl_path = os.path.join(source_lbl_dir, file_base + ".txt")
        if split_name != "test" and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                raw_lines = f.readlines()

        metadata_list = []

        # ---------------------------------------------------------
        # 3. 분기 로직: 마스크 처리 및 크롭
        # ---------------------------------------------------------
        
        # 처리할 마스크 리스트 준비
        if is_instance:
            # 개별 마스크를 각각 처리
            process_masks = [(masks[i].cpu().numpy().squeeze(), float(scores[i]), i) 
                             for i in range(len(masks)) if scores[i] >= config['sam_threshold']]
        else:
            # 모든 마스크를 하나로 합쳐서 ROI 생성
            combined_mask = torch.any(masks, dim=0).cpu().numpy().squeeze()
            avg_score = float(torch.mean(scores))
            process_masks = [(combined_mask, avg_score, None)]

        for mask_2d, score, idx in process_masks:
            y_indices, x_indices = np.where(mask_2d > 0)
            if len(x_indices) == 0: continue

            # Bbox 및 마진 계산
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            w_box, h_box = x_max - x_min, y_max - y_min
            pad_x, pad_y = int(w_box * margin_ratio), int(h_box * margin_ratio)

            x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
            x2, y2 = min(img_w, x_max + pad_x), min(img_h, y_max + pad_y)
            crop_w, crop_h = x2 - x1, y2 - y1

            # 이미지 크롭 및 저장
            cropped_img = img_np[y1:y2, x1:x2]

            # 인스턴스 크롭의 경우 instance_{인스턴스 인덱스} 로 파일명 구분
            # ROI 크롭은 단순히 파일 명에 _cropped 로 파일명 구분
            suffix = f"instance_{idx:02d}" if is_instance else "cropped"
            instance_name = f"{file_base}_{suffix}"

            img_save_path = os.path.join(base_output_dir, "images", split_name, f"{instance_name}.png")
            cv2.imwrite(img_save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            # 라벨 리매핑
            yolo_lines = []
            if split_name != "test":
                for line in raw_lines:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = parts[0]
                    norm_coords = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                    
                    # 원본 픽셀 좌표 복원
                    gt_pts = (norm_coords * [img_w, img_h]).astype(np.float32)

                    if is_instance:
                        # [Instance 모드] IoU 체크로 현재 치아에 해당하는 라벨만 추출
                        gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(gt_mask, [gt_pts.astype(np.int32)], 1)
                        intersection = np.logical_and(mask_2d > 0, gt_mask > 0).sum()
                        gt_area = (gt_mask > 0).sum()
                        if (intersection / gt_area if gt_area > 0 else 0) < config['iou_filter_th']:
                            continue
                    
                    # [공통] 크롭 좌표계로 변환 및 정규화
                    local_pts = (gt_pts - [x1, y1]) / [crop_w, crop_h]
                    local_pts = np.clip(local_pts, 0, 1)
                    str_pts = " ".join([f"{p:.6f}" for p in local_pts.flatten()])
                    yolo_lines.append(f"{cid} {str_pts}")

            # 라벨 저장
            lbl_save_path = os.path.join(base_output_dir, "labels", split_name, f"{instance_name}.txt")
            with open(lbl_save_path, "w") as f:
                if yolo_lines: f.write("\n".join(yolo_lines))

            # 메타데이터 추가
            metadata_list.append({
                "instance_name": instance_name,
                "crop_coords": [int(x1), int(y1), int(x2), int(y2)],
                "original_size": [img_w, img_h],
                "score": score
            })
            if is_instance: total_count += 1

        # JSON 메타데이터 저장
        if metadata_list:
            json_path = os.path.join(base_output_dir, "metadata", split_name, f"{file_base}.json")
            with open(json_path, 'w') as f:
                json.dump(metadata_list, f, indent=4)
        
        if not is_instance: total_count += 1

    print(f"✅ {split_name.upper()} 완료! ({mode_str} 총합: {total_count})")

def extract_with_progress(zip_path, extract_dir):
    """zip 파일을 tqdm 프로그레스 바와 함께 압축 해제 (NameError 해결)"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc=f"📦 Extracting {os.path.basename(zip_path)}", unit="file", leave=False):
            zip_ref.extract(file, extract_dir)

def prepare_directories():
    """기존의 불완전한 폴더를 삭제하고 새 구조를 만듭니다."""
    base_data_dir = "./data"
    sub_dirs = ["alphadent_instance", "alphadent_roi"]

    # 1. data 폴더가 없으면 생성
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
        print(f"📁 {base_data_dir} 폴더를 생성했습니다.")

    # 2. 기존 인스턴스/ROI 폴더 삭제 후 재생성 (깔끔한 시작)
    for sub in sub_dirs:
        target_path = os.path.join(base_data_dir, sub)
        if os.path.exists(target_path):
            shutil.rmtree(target_path) # 폴더 전체 삭제
            print(f"🗑️ 기존 {target_path} 폴더를 삭제했습니다.")
        os.makedirs(target_path)
        print(f"✅ 새 {target_path} 폴더를 생성했습니다.")


def download_all_from_drive():
    """구글 드라이브에서 다운로드 후 tqdm 바와 함께 압축 해제"""
    prepare_directories()

    files = {
        "alphadent_roi.zip": ("1ZQCzfbQF0uPXtSz_-NHBYnArZoRei-IF", "./data/alphadent_roi"),
        "alphadent_instance.zip": ("1L45ztC85-ZCq76mcg4AXD87gHTBOnO3t", "./data/alphadent_instance")
    }

    for filename, (file_id, target_extract_dir) in files.items():
        zip_path = os.path.join("./data", filename)
        
        print(f"\n🚚 {filename} 다운로드 중...")
        subprocess.run(["gdown", "--id", file_id, "-O", zip_path], check=True)
        
        # 🔥 수정된 압축 해제 로직 호출
        extract_with_progress(zip_path, target_extract_dir)
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"🗑️ 임시 파일 {zip_path} 삭제 완료.")

    print("\n✨ 모든 데이터셋 세팅이 완료되었습니다!")

def main():
    # 1. 인자 파서 설정
    parser = argparse.ArgumentParser(description="SAM-3 Teeth Segmentation Preprocessing")
    
    # --mode: roi, instance 중 선택 (기본값은 둘 다 실행하기 위해 None)
    parser.add_argument("--mode", type=str, choices=["roi", "instance"], 
                        help="Execution mode: 'roi' or 'instance'. If omitted, both will run.")
    
    # --split: train, valid, test 중 선택 (기본값은 모두 실행하기 위해 None)
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"],
                        help="Data split: 'train', 'valid', or 'test'. If omitted, all splits will run.")
    # 다운로드 전용 옵션
    parser.add_argument("--from_drive", action="store_true", help="Download preprocessed data from Google Drive")
    
    args = parser.parse_args()

    # 만약 --from_drive 옵션을 주면 전처리를 안 하고 다운로드만 수행
    if args.from_drive:
        download_all_from_drive()
        return # 전처리 로직 실행 안 하고 종료

    # 2. 일반 전처리 모드 (기존 데이터를 지우지 않고 필요한 폴더만 생성)
    prepare_directories()

    # 2. 실행 환경 준비
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

    # 3. 실행할 대상 리스트 결정
    modes_to_run = [args.mode] if args.mode else ["roi", "instance"]
    splits_to_run = [args.split] if args.split else ["train", "valid", "test"]

    # 4. 루프 실행
    for m in modes_to_run:
        is_instance = (m == "instance")
        for s in splits_to_run:
            run_sam3_preprocessing(s, processor, is_instance=is_instance, config=config)

if __name__ == "__main__":
    main()