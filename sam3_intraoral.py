import os
import sys
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def run_intraoral_crop_pipeline(split_name, processor_model, BASE_OUTPUT_DIR = '/data/alphadent_roi',
                                margin_ratio = config['margin_ratio'], 
                                prompt = config['intraoral_prompt']):
    print(f"🚀 [{split_name.upper()}] 구강 영역 통합 크롭 시작 (Margin: {margin_ratio})")

    source_img_dir = f"/data/alphadent_extracted/images/{split_name}"
    source_lbl_dir = f"/data/alphadent_extracted/labels/{split_name}"

    image_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")) +
                         glob.glob(os.path.join(source_img_dir, "*.png")))

    # 폴더 구조 생성
    for sub in ["images", "labels", "metadata"]:
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, sub, split_name), exist_ok=True)

    total_images = 0

    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        file_base = os.path.splitext(os.path.basename(img_path))[0]

        # 1. 이미지 로드 및 전처리
        raw_image = Image.open(img_path).convert("RGB")
        raw_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img_np = np.array(raw_image)
        img_h, img_w = img_np.shape[:2]

        # 2. SAM-3 추론 (영역 중심 프롬프트)
        inference_state = processor_model.set_image(raw_image)
        # 단일 영역 포착을 위해 프롬프트 수정
        output = processor_model.set_text_prompt(
            state=inference_state,
            prompt=prompt
        )

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            print(f"⚠️ {file_base}: 영역 탐지 실패. 스킵합니다.")
            continue

        # 3. 단일 통합 바운딩 박스 계산 (Global ROI)
        # 모든 마스크를 하나로 합침
        combined_mask = torch.any(masks, dim=0).cpu().numpy().squeeze()
        y_indices, x_indices = np.where(combined_mask > 0)

        if len(x_indices) == 0: continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # 통합 영역 기준 마진 적용
        w_box, h_box = x_max - x_min, y_max - y_min
        pad_x, pad_y = int(w_box * margin_ratio), int(h_box * margin_ratio)

        x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
        x2, y2 = min(img_w, x_max + pad_x), min(img_h, y_max + pad_y)

        # 4. 이미지 크롭 및 저장
        cropped_img = img_np[y1:y2, x1:x2]
        crop_w, crop_h = x2 - x1, y2 - y1
        instance_name = f"{file_base}_cropped" # 요구사항: suffix _cropped

        image_save_path = os.path.join(BASE_OUTPUT_DIR, "images", split_name, f"{instance_name}.png")
        cv2.imwrite(image_save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

        # 5. 라벨 리매핑 (모든 원본 라벨을 크롭 좌표계로 변환)
        yolo_lines = []
        lbl_path = os.path.join(source_lbl_dir, file_base + ".txt")

        if split_name != "test" and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = parts[0]
                    norm_coords = list(map(float, parts[1:]))

                    # 전역 좌표 -> 크롭 내부 좌표로 변환
                    # [x, y, x, y...] 순서이므로 2개씩 처리
                    new_coords = []
                    pts = np.array(norm_coords).reshape(-1, 2)
                    for pt in pts:
                        # 원본 이미지 픽셀 좌표로 복원
                        gx, gy = pt[0] * img_w, pt[1] * img_h
                        # 크롭 좌표계로 이동 및 정규화
                        lx = (gx - x1) / crop_w
                        ly = (gy - y1) / crop_h
                        # 0~1 사이로 클리핑 (크롭 영역 밖으로 나간 점 처리)
                        new_coords.extend([np.clip(lx, 0, 1), np.clip(ly, 0, 1)])

                    str_pts = " ".join([f"{p:.6f}" for p in new_coords])
                    yolo_lines.append(f"{cid} {str_pts}")

            # 라벨 파일 저장
            label_save_path = os.path.join(BASE_OUTPUT_DIR, "labels", split_name, f"{instance_name}.txt")
            with open(label_save_path, "w") as f:
                if yolo_lines: f.write("\n".join(yolo_lines))

        # 6. 메타데이터 저장 (기존 형식 유지)
        avg_score = float(torch.mean(scores)) # 전체 탐지 신뢰도 평균
        metadata = [{
            "instance_name": instance_name,
            "crop_coords": [int(x1), int(y1), int(x2), int(y2)],
            "original_size": [img_w, img_h],
            "score": avg_score
        }]

        json_path = os.path.join(BASE_OUTPUT_DIR, "metadata", split_name, f"{file_base}.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        total_images += 1

    print(f"✅ {split_name.upper()} 완료! 처리된 이미지: {total_images}")



# 현재 위치 확인
print(f"현재 디렉토리: {os.getcwd()}")

try:
    import sam3
    import iopath
    print("✅ SAM3 and dependency libraries have been sucessfully loaded.")
except ImportError as e:
    print(f"❌ Library still not found: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model().to(device)
model.eval() # 추론 모드로 설정

# 프로세서 설정
processor = Sam3Processor(model)
print(f"Sucessfully loaded SAM 3 on {device}")

run_intraoral_crop_pipeline("valid", processor) # expected 1 min
run_intraoral_crop_pipeline("train", processor) # expected 17 mins
run_intraoral_crop_pipeline("test", processor) # expected 2 mins