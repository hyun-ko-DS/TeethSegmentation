import os
import sys
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
 
 
def run_instance_crop_pipeline(split_name, processor_model, BASE_OUTPUT_DIR = '/data/alphadent_instance',
                                margin_ratio = config['margin_ratio'], 
                                prompt = config['instance_prompt']):
    
    source_img_dir = f"/data/alphadent_extracted/images/{split_name}"
    source_lbl_dir = f"/data/alphadent_extracted/labels/{split_name}"

    image_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")) +
                         glob.glob(os.path.join(source_img_dir, "*.png")))

    # 폴더 구조 생성
    for sub in ["images", "labels", "metadata"]:
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, sub, split_name), exist_ok=True)

    total_instances = 0

    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        file_base = os.path.splitext(os.path.basename(img_path))[0]

        # 1. 이미지 로드
        raw_image = Image.open(img_path).convert("RGB")
        raw_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        img_np = np.array(raw_image)
        img_h, img_w = img_np.shape[:2]

        # 2. SAM-3 추론
        inference_state = processor_model.set_image(raw_image)
        output = processor_model.set_text_prompt(state=inference_state, prompt=prompt)
        masks = output["masks"].cpu().numpy() if torch.is_tensor(output["masks"]) else output["masks"]
        scores = output["scores"].cpu().numpy() if torch.is_tensor(output["scores"]) else output["scores"]

        # 3. 원본 라벨 로드 (Test 제외)
        raw_lines = []
        lbl_path = os.path.join(source_lbl_dir, file_base + ".txt")
        if split_name != "test" and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                raw_lines = f.readlines()

        metadata_list = []

        # 4. 개별 치아 인스턴스 루프
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score < SAM_THRESHOLD: continue
            mask_2d = mask.squeeze()
            if np.sum(mask_2d) == 0: continue

            # [V5 로직] Bbox + Margin
            y_indices, x_indices = np.where(mask_2d > 0)
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            w_box, h_box = x_max - x_min, y_max - y_min

            pad_x, pad_y = int(w_box * MARGIN_RATIO), int(h_box * MARGIN_RATIO)
            x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
            x2, y2 = min(img_w, x_max + pad_x), min(img_h, y_max + pad_y)

            # 🔥 사각형 크롭 (배경 유지)
            cropped_img = img_np[y1:y2, x1:x2]
            instance_name = f"{file_base}_instance_{i:02d}"

            # 이미지 저장
            instance_save_path = os.path.join(BASE_OUTPUT_DIR, "images", split_name, f"{instance_name}.png")
            cv2.imwrite(instance_save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            # [GT 필터링] IoU Check (Train/Valid 전용)
            yolo_lines = []
            if split_name != "test":
                for line in raw_lines:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = int(parts[0])
                    norm_coords = list(map(float, parts[1:]))

                    gt_pts = (np.array(norm_coords).reshape(-1, 2) * [img_w, img_h]).astype(np.int32)
                    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.fillPoly(gt_mask, [gt_pts], 1)

                    intersection = np.logical_and(mask_2d > 0, gt_mask > 0).sum()
                    gt_area = (gt_mask > 0).sum()
                    overlap = intersection / gt_area if gt_area > 0 else 0

                    if overlap >= IOU_FILTER_TH:
                        local_pts = (gt_pts - [x1, y1]) / [x2-x1, y2-y1]
                        local_pts = np.clip(local_pts, 0, 1)
                        str_pts = " ".join([f"{p:.6f}" for p in local_pts.flatten()])
                        yolo_lines.append(f"{cid} {str_pts}")

                # 라벨 저장
                label_save_path = os.path.join(BASE_OUTPUT_DIR, "labels", split_name, f"{instance_name}.txt")
                with open(label_save_path, "w") as f:
                    if yolo_lines: f.write("\n".join(yolo_lines))

            # 메타데이터 수집
            metadata_list.append({
                "instance_name": instance_name,
                "crop_coords": [int(x1), int(y1), int(x2), int(y2)],
                "original_size": [img_w, img_h],
                "score": float(score)
            })
            total_instances += 1

        # JSON 메타데이터 저장
        if metadata_list:
            json_path = os.path.join(BASE_OUTPUT_DIR, "metadata", split_name, f"{file_base}.json")
            with open(json_path, 'w') as f:
                json.dump(metadata_list, f, indent=4)

    print(f"✅ {split_name.upper()} 완료! 생성된 인스턴스: {total_instances}")