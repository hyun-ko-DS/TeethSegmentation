import argparse
import cv2
import glob
import json
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

from caf import *
from utils import *

# ============================================================
# 1. Configuration Class
# ============================================================
class WMFConfig:
    def __init__(self, config):
        self.canvas_w = config['canvas_w']
        self.canvas_h = config['canvas_w']
        self.iou_thr = config['wmf_iou_thres']
        self.mask_thr = config['wmf_mask_thres']
        self.weights = config['wmf_weights']
        self.single_model_thr = config['wmf_single_model_thres']
        self.agreement_boost_thr = config['wmf_agreement_boost_thres']

# ============================================================
# 2. Utility Functions (Mask Fusion and Calculation)
# ============================================================
def poly_to_mask(poly_str, config):
    """Converts a polygon string to a binary mask."""
    mask = np.zeros((config.canvas_h, config.canvas_w), dtype=np.float32)
    try:
        coords = np.array(list(map(float, poly_str.split()))).reshape(-1, 2)
        coords[:, 0] *= config.canvas_w
        coords[:, 1] *= config.canvas_h
        pts = coords.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1.0)
    except Exception:
        pass
    return mask

def mask_to_poly(mask, config):
    """Converts a binary mask back into a polygon string."""
    mask_ui8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_ui8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    cnt = max(contours, key=cv2.contourArea)
    poly = cnt.reshape(-1, 2).astype(np.float32)
    poly[:, 0] /= config.canvas_w
    poly[:, 1] /= config.canvas_h
    return " ".join(map(str, poly.flatten()))

def get_iou(m1, m2):
    """Calculates Intersection over Union for two masks."""
    it = np.logical_and(m1, m2).sum()
    un = np.logical_or(m1, m2).sum()
    return it / un if un > 0 else 0

def draw_predictions_on_image(image, polygons, class_names, colors, is_gt=False):
    """
    Draws only the outlines of masks on the image with class-specific colors.
    """
    h, w = image.shape[:2]
    vis_img = image.copy()

    for poly_data in polygons:
        class_id = poly_data['class_id']
        pts = poly_data['points'].astype(np.int32)
        conf = poly_data.get('conf', 1.0)

        # 1. Color conversion (0~1 float -> 0~255 BGR)
        raw_color = colors[class_id % len(colors)]
        color_bgr = (
            int(np.round(raw_color[2] * 255)), # Blue
            int(np.round(raw_color[1] * 255)), # Green
            int(np.round(raw_color[0] * 255))  # Red
        )

        # 2. Draw outlines (Thickness adjusted for high resolution)
        thickness = max(3, int(w / 800))
        cv2.polylines(vis_img, [pts], isClosed=True, color=color_bgr, thickness=thickness)

        # Skip text labels for Ground Truth
        if is_gt:
            continue

        # 3. Setup text label style
        label_text = f"{class_names[class_id]}: {conf:.2f}"
        font_scale = max(0.5, w / 1600)
        text_thickness = max(1, int(w / 1400))
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        padding = int(8 * font_scale)

        # 4. Position and boundary correction for text
        txt_x, txt_y = pts[0][0], pts[0][1] - 10
        if txt_x + tw + padding > w: txt_x = w - tw - padding * 2
        txt_x = max(padding, txt_x)
        if txt_y < th + padding * 2: txt_y = pts[0][1] + th + padding * 2

        # 5. Draw text background box and label
        cv2.rectangle(vis_img, (int(txt_x - padding), int(txt_y - th - padding)),
                      (int(txt_x + tw + padding), int(txt_y + baseline + padding)), color_bgr, -1)

        # Determine text color based on background brightness
        brightness = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]
        txt_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

        cv2.putText(vis_img, label_text, (int(txt_x), int(txt_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color,
                    thickness=text_thickness, lineType=cv2.LINE_AA)

    return vis_img

# ============================================================
# 3. Ensemble Core Logic
# ============================================================
def perform_wmf_direct(model_outputs, config):
    """
    Performs Weighted Mask Fusion on raw model detections.
    Clusters masks based on IoU and fuses them into a single ensemble result.
    """
    weights = config.weights
    num_models = len(model_outputs)
    
    # Extract patient_id from the first available detection
    current_pid = "unknown"
    for m_out in model_outputs:
        if m_out:
            current_pid = m_out[0]['patient_id']
            break

    # [Step 1] Clustering
    clusters = []
    for model_idx, detections in enumerate(model_outputs):
        weight = weights[model_idx]
        for det in detections:
            det_mask = poly_to_mask(det['poly'], config)
            det['mask'] = det_mask
            det['model_weight'] = weight
            
            matched = False
            for cluster in clusters:
                if det['class_id'] == cluster[0]['class_id']:
                    iou = get_iou(det_mask, cluster[0]['mask'])
                    if iou >= config.iou_thr:
                        cluster.append(det)
                        matched = True
                        break
            if not matched:
                clusters.append([det])

    # [Step 2] Fusion and Correction
    ensemble_rows = []
    for cluster in clusters:
        # Require at least two models to agree
        if len(cluster) < 2:
            continue

        fused_mask = np.zeros((config.canvas_h, config.canvas_w), dtype=np.float32)
        total_weight, weighted_conf = 0, 0

        for det in cluster:
            w = det['model_weight']
            fused_mask += (det['mask'] * w)
            weighted_conf += (det['confidence'] * w)
            total_weight += w

        avg_conf = weighted_conf / total_weight
        agreement_ratio = len(cluster) / num_models
        
        # Boost confidence based on model agreement
        final_conf = avg_conf * (agreement_ratio ** config.agreement_boost_thr)

        fused_mask /= total_weight
        fused_mask = cv2.GaussianBlur(fused_mask, (3, 3), 0)
        final_mask = (fused_mask >= config.mask_thr).astype(np.uint8)

        final_poly = mask_to_poly(final_mask, config)
        if final_poly:
            ensemble_rows.append({
                'patient_id': current_pid,  
                'class_id': cluster[0]['class_id'],
                'confidence': final_conf,
                'poly': final_poly
            })

    if not ensemble_rows:
        return pd.DataFrame(columns=['patient_id', 'class_id', 'confidence', 'poly'])

    return pd.DataFrame(ensemble_rows).sort_values(by='confidence', ascending=False)

def run_wmf_ensemble(models, model_names, is_roi_list, config_dict, paths_list, is_valid=True, weight_type="pt"):
    """
    Main ensemble pipeline.
    Saves outputs to ./results/{valid or test}/wmf_ensemble
    """
    wmf_config = WMFConfig(config_dict)
    
    # Class mapping for labeling
    CLASS_INFO = {
        0: {'name': 'Abrasion'}, 1: {'name': 'Filling'}, 2: {'name': 'Crown'},
        3: {'name': 'Caries Class 1'}, 4: {'name': 'Caries Class 2'},
        5: {'name': 'Caries Class 3'}, 6: {'name': 'Caries Class 4'},
        7: {'name': 'Caries Class 5'}, 8: {'name': 'Caries Class 6'}
    }
    CLASS_NAMES = [CLASS_INFO[i]['name'] for i in range(len(CLASS_INFO))]
    
    # 1. Dynamic Directory Setup
    mode = "valid" if is_valid else "test"
    save_dir = os.path.join("results", mode, "wmf_ensemble")
    os.makedirs(save_dir, exist_ok=True)

    # Path keys mapping
    img_path_key = f"{mode}_images_path"
    meta_path_key = f"{mode}_metadata_path"
    orig_img_key = f"original_{mode}_images_dir"
    orig_lbl_key = f"original_{mode}_labels_dir"

    print(f"🔥 Starting WMF Ensemble [Mode: {mode.upper()}]")
    print(f"📂 Results will be saved to: {os.path.abspath(save_dir)}")
    
    target_files = glob.glob(os.path.join(paths_list[0][orig_img_key], "*.jpg")) + \
                   glob.glob(os.path.join(paths_list[0][orig_img_key], "*.png"))
    
    all_ensemble_results = []

    # 2. Per-image ensemble loop
    for img_path in tqdm(target_files, desc=f"Processing {mode.upper()}"):
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        
        meta_path = os.path.join(paths_list[0][meta_path_key], f"{file_id}.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, 'r') as f:
            meta_list = json.load(f)
            target_w, target_h = meta_list[0]['original_size']
        
        wmf_config.canvas_w, wmf_config.canvas_h = target_w, target_h
        all_model_detections = []

        # [Step A] Model prediction and coordinate restoration
        for idx, model in enumerate(models):
            is_roi = is_roi_list[idx]
            curr_paths = paths_list[idx]
            model_detections = []
            
            search_pattern = f"{file_id}_cropped*" if is_roi else f"{file_id}_instance_*"
            crop_paths = glob.glob(os.path.join(curr_paths[img_path_key], search_pattern))
            
            if not crop_paths:
                all_model_detections.append([])
                continue

            imgsz = config_dict['roi_image_size'] if is_roi else config_dict['instance_image_size']
            
            # task="segment" if ONNX or tensorRT
            predict_kwargs = {
                "source": crop_paths,
                "imgsz": imgsz,
                "conf": config_dict['conf_thres'],
                "iou": config_dict['iou_thres'],
                "retina_masks": False, # TensorRT 환경에서도 False가 훨씬 빠릅니다.
                "verbose": False
            }
            
            # TensorRT나 ONNX일 경우 task 재명시 (안전장치)
            if weight_type in ["onnx", "engine"]:
                predict_kwargs["task"] = "segment"

            # # --- 수정된 추론 루프 ---
            # all_results = []
            # for cp in crop_paths:
            #     predict_kwargs["source"] = cp  # 단일 파일 경로 주입 (배치 사이즈 1 유지)
            #     res = model.predict(**predict_kwargs)
            #     all_results.extend(res)

            # # 후처리 로직 (기존과 동일하되 리스트 순회)
            # for r in all_results:
            #     crop_name = os.path.splitext(os.path.basename(r.path))[0]
            #     try:
            #         with open(os.path.join(curr_paths[meta_path_key], f"{file_id}.json"), 'r') as f:
            #             m_list = json.load(f)
                    
            #         # ROI 모델은 첫 번째 메타데이터, Instance 모델은 이름 매칭
            #         meta = m_list[0] if is_roi else next(m for m in m_list if m['instance_name'] == crop_name)
            #         x_off, y_off = meta['crop_coords'][:2]
            #     except Exception:
            #         continue
                
            results = model.predict(**predict_kwargs)
            for r in results:
                crop_name = os.path.splitext(os.path.basename(r.path))[0]
                try:
                    with open(os.path.join(curr_paths[meta_path_key], f"{file_id}.json"), 'r') as f:
                        m_list = json.load(f)
                    meta = m_list[0] if is_roi else next(m for m in m_list if m['instance_name'] == crop_name)
                    x_off, y_off = meta['crop_coords'][:2]
                except Exception:
                    continue

                if r.masks is None:
                    continue
                for i, mask_coords in enumerate(r.masks.xy):
                    global_pts = mask_coords.copy()
                    global_pts[:, 0] += x_off
                    global_pts[:, 1] += y_off
                    
                    norm_poly = global_pts.copy()
                    norm_poly[:, 0] /= target_w
                    norm_poly[:, 1] /= target_h
                    model_detections.append({
                        'patient_id': file_id,
                        'class_id': int(r.boxes.cls[i]),
                        'confidence': float(r.boxes.conf[i]),
                        'poly': " ".join(map(str, norm_poly.flatten()))
                    })
            all_model_detections.append(model_detections)

        # [Step B] Execute WMF Ensemble
        fused_df = perform_wmf_direct(all_model_detections, wmf_config)
        
        if not is_valid and not fused_df.empty:
            all_ensemble_results.append(fused_df)

        # [Step C] Visualization
        orig_raw = cv2.imread(img_path)
        orig_img = cv2.resize(orig_raw, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        preds = []
        for _, row in fused_df.iterrows():
            coords = (np.array(list(map(float, row['poly'].split()))).reshape(-1, 2) * [target_w, target_h]).astype(np.int32)
            preds.append({'class_id': row['class_id'], 'points': coords, 'conf': row['confidence']})

        gt_list = []
        gt_label_text = "Ground Truth"
        if is_valid:
            lbl_path = os.path.join(paths_list[0][orig_lbl_key], f"{file_id}.txt")
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        p = line.strip().split()
                        if not p: continue
                        gt_list.append({'class_id': int(p[0]), 'points': (np.array(list(map(float, p[1:]))).reshape(-1, 2) * [target_w, target_h]).astype(np.int32)})
        else:
            gt_label_text = "Test Original Image"

        img_gt = draw_predictions_on_image(orig_img.copy(), gt_list, CLASS_NAMES, config_dict['colors'], is_gt=True)
        img_pred = draw_predictions_on_image(orig_img.copy(), preds, CLASS_NAMES, config_dict['colors'], is_gt=False)
        
        cv2.putText(img_gt, gt_label_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(img_pred, f"WMF Ensemble ({mode.upper()})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        combined = np.hstack((img_gt, img_pred))
        cv2.imwrite(os.path.join(save_dir, f"{file_id}_wmf_{mode}.jpg"), combined)

    # 3. [Step D] Create submission.csv for Test mode
    if not is_valid and all_ensemble_results:
        final_df = pd.concat(all_ensemble_results, ignore_index=True)
        final_df = final_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        final_df['id'] = range(1, len(final_df) + 1)
        
        csv_path = os.path.join(save_dir, 'submission.csv')
        final_df[['id', 'patient_id', 'class_id', 'confidence', 'poly']].to_csv(csv_path, index=False)
        print(f"✅ Submission CSV generated successfully! Location: {csv_path}")

    print(f"✅ {mode.upper()} ensemble process finished!")

# ============================================================
# 4. Entry Point (Argparse)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="WMF Ensemble for Teeth Segmentation")
    parser.add_argument("--data", type=str, required=True, choices=["valid", "test"],
                        help="Choose data split for ensemble: 'valid' or 'test'")
    # 1. weight_type에 engine 추가
    parser.add_argument("--weight_type", type=str, default="pt", choices=["pt", "onnx", "engine"],
                        help="Select weight file format: 'pt', 'onnx', or 'engine'")
    args = parser.parse_args()

    print(f"📦 Loading models for ensemble as {args.weight_type.upper()} format...")
    
    model_names = ['model_365', 'model_360', 'model_357', 'model_355']
    models = []
    
    for name in model_names:
        suffix = name.rsplit("_", 1)[-1]
        ext = args.weight_type
        weight_path = os.path.join("models", name, f"best_{suffix}.{ext}")
        
        if os.path.exists(weight_path):
            print(f"   • Loading {name} from {weight_path}")
            
            # 2. TensorRT(.engine) 및 ONNX 로드 시 Task 명시
            # TensorRT는 task="segment"를 명시해야 Predictor가 올바르게 설정됩니다.
            if ext in ["onnx", "engine"]:
                models.append(YOLO(weight_path, task="segment"))
            else:
                models.append(YOLO(weight_path))
        else:
            print(f"   ⚠️ Warning: Weights not found for {name} at {weight_path}")
    
    # 2. Flag whether the model uses ROI-based (True) or Instance-based (False) crops
    is_roi_list = [True, False, True, True]
    
    # 3. Define path dictionaries for ROI-based preprocessed data
    paths_roi = {
        'output_dir': './data/alphadent_roi',
        'valid_images_path': './data/alphadent_roi/images/valid',
        'test_images_path': './data/alphadent_roi/images/test',
        'valid_metadata_path': './data/alphadent_roi/metadata/valid',
        'test_metadata_path': './data/alphadent_roi/metadata/test',
        'original_valid_images_dir': './data/alphadent_extracted/images/valid',
        'original_test_images_dir': './data/alphadent_extracted/images/test',
        'original_valid_labels_dir': './data/alphadent_extracted/labels/valid'
    }

    # 4. Define path dictionaries for Instance-based preprocessed data
    paths_instance = {
        'output_dir': './data/alphadent_instance',
        'valid_images_path': './data/alphadent_instance/images/valid',
        'test_images_path': './data/alphadent_instance/images/test',
        'valid_metadata_path': './data/alphadent_instance/metadata/valid',
        'test_metadata_path': './data/alphadent_instance/metadata/test',
        'original_valid_images_dir': './data/alphadent_extracted/images/valid',
        'original_test_images_dir': './data/alphadent_extracted/images/test',
        'original_valid_labels_dir': './data/alphadent_extracted/labels/valid'
    }

    # Map each model to its respective path dictionary
    paths_list = [paths_roi, paths_instance, paths_roi, paths_roi]

    # 5. Load global configuration and trigger the ensemble pipeline
    config = load_config()
    is_valid_flag = (args.data == "valid")
    
    # 3. run_wmf_ensemble에 weight_type 전달
    run_wmf_ensemble(
        models=models,
        model_names=model_names,
        is_roi_list=is_roi_list,
        config_dict=config,
        paths_list=paths_list,
        is_valid=(args.data == "valid"),
        weight_type=args.weight_type
    )

if __name__ == "__main__":
    main()