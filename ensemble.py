import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


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
    mask = np.zeros((config.canvas_h, config.canvas_w), dtype=np.float32)
    try:
        coords = np.array(list(map(float, poly_str.split()))).reshape(-1, 2)
        coords[:, 0] *= config.canvas_w
        coords[:, 1] *= config.canvas_h
        pts = coords.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1.0)
    except:
        pass
    return mask

def mask_to_poly(mask, config):
    mask_ui8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_ui8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return ""
    cnt = max(contours, key=cv2.contourArea)
    poly = cnt.reshape(-1, 2).astype(np.float32)
    poly[:, 0] /= config.canvas_w
    poly[:, 1] /= config.canvas_h
    return " ".join(map(str, poly.flatten()))

def get_iou(m1, m2):
    it = np.logical_and(m1, m2).sum()
    un = np.logical_or(m1, m2).sum()
    return it / un if un > 0 else 0

def run_wmf_ensemble(models, model_names, is_roi_list, config_dict, paths_list, is_valid=True):
    """
    is_valid=True: Validation mode (with GT visualization)
    is_valid=False: Test mode (Visualizatin w/o GT + Create submission.csv)
    """
    wmf_config = WMFConfig(config_dict)
    
    # 1. 폴더 구조 설정 (valid/test 분기)
    mode = "valid" if is_valid else "test"
    base_output_dir = os.path.join(paths_list[0]['output_dir'], mode)
    # 시각화 파일과 CSV가 저장될 최종 경로
    save_dir = os.path.join(base_output_dir, 'wmf_ensemble')
    os.makedirs(save_dir, exist_ok=True)

    # 경로 키 설정
    img_path_key = f"{mode}_images_path"
    meta_path_key = f"{mode}_metadata_path"
    orig_img_key = f"original_{mode}_images_dir"
    orig_lbl_key = f"original_{mode}_labels_dir"

    print(f"🔥 WMF 앙상블 시작 [Mode: {mode.upper()}]")
    print(f"📂 저장 및 CSV 경로: {save_dir}")
    
    target_files = glob.glob(os.path.join(paths_list[0][orig_img_key], "*.jpg")) + \
                   glob.glob(os.path.join(paths_list[0][orig_img_key], "*.png"))
    
    all_ensemble_results = []

    # 2. 이미지별 앙상블 루프
    for img_path in tqdm(target_files, desc=f"Processing {mode.upper()}"):
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # 메타데이터 로드
        meta_path = os.path.join(paths_list[0][meta_path_key], f"{file_id}.json")
        if not os.path.exists(meta_path): continue
        with open(meta_path, 'r') as f:
            meta_list = json.load(f)
            target_w, target_h = meta_list[0]['original_size']
        
        wmf_config.canvas_w, wmf_config.canvas_h = target_w, target_h
        all_model_detections = []

        # [Step A] 각 모델별 예측 및 좌표 복원
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
            results = model.predict(source=crop_paths, imgsz=imgsz, conf=config_dict['conf_thres'], 
                                    iou=config_dict['iou_thres'], retina_masks=True, verbose=False)

            for r in results:
                crop_name = os.path.splitext(os.path.basename(r.path))[0]
                try:
                    with open(os.path.join(curr_paths[meta_path_key], f"{file_id}.json"), 'r') as f:
                        m_list = json.load(f)
                    meta = m_list[0] if is_roi else next(m for m in m_list if m['instance_name'] == crop_name)
                    x_off, y_off = meta['crop_coords'][:2]
                except: continue

                if r.masks is None: continue
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

        # [Step B] WMF 앙상블 실행
        fused_df = perform_wmf_direct(all_model_detections, wmf_config)
        
        # Test 모드일 경우 CSV용 데이터 저장 (patient_id가 포함된 fused_df 사용)
        if not is_valid and not fused_df.empty:
            all_ensemble_results.append(fused_df)

        # [Step C] 시각화 (기존 로직 유지)
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

    # 3. [Step D] Test 모드일 경우 최종 submission.csv 저장
    if not is_valid and all_ensemble_results:
        final_df = pd.concat(all_ensemble_results, ignore_index=True)
        # 전체 데이터 통합 정렬
        final_df = final_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        final_df['id'] = range(1, len(final_df) + 1)
        
        # [수정] submission.csv를 wmf_ensemble 폴더 내부에 저장
        csv_path = os.path.join(save_dir, 'submission.csv')
        final_df[['id', 'patient_id', 'class_id', 'confidence', 'poly']].to_csv(csv_path, index=False)
        print(f"✅ Submission CSV 생성 완료! 위치: {csv_path}")

    print(f"✅ {mode.upper()} 앙상블 프로세스 완료!")

def perform_wmf_direct(model_outputs, config):
    """
    model_outputs 리스트에서 patient_id를 추출하여 결과 DataFrame에 포함합니다.
    """
    weights = config.weights
    num_models = len(model_outputs)
    
    # [추가] patient_id 추출 (detections 중 하나라도 있으면 가져옴)
    current_pid = "unknown"
    for m_out in model_outputs:
        if m_out:
            current_pid = m_out[0]['patient_id']
            break

    # [Step 1] 클러스터링
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
            if not matched: clusters.append([det])

    # [Step 2] 융합 및 보정
    ensemble_rows = []
    for cluster in clusters:
        if len(cluster) < 2: continue

        fused_mask = np.zeros((config.canvas_h, config.canvas_w), dtype=np.float32)
        total_weight, weighted_conf = 0, 0

        for det in cluster:
            w = det['model_weight']
            fused_mask += (det['mask'] * w)
            weighted_conf += (det['confidence'] * w)
            total_weight += w

        avg_conf = weighted_conf / total_weight
        agreement_ratio = len(cluster) / num_models
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

    # 결과가 없을 경우 빈 DataFrame 반환 (컬럼명 명시)
    if not ensemble_rows:
        return pd.DataFrame(columns=['patient_id', 'class_id', 'confidence', 'poly'])

    return pd.DataFrame(ensemble_rows).sort_values(by='confidence', ascending=False)

models = [model_365, model_360, model_357, model_355]
model_names = ['model_365', 'model_360', 'model_357', 'model_355']
is_roi_list = [True, False, True, True]
paths_list = [paths_roi, paths_instance, paths_roi, paths_roi]

# VALID DATA 개별 모델 추론 + 앙상블
run_wmf_ensemble(
    models=models,
    model_names=model_names,
    is_roi_list=is_roi_list,
    config_dict=config, # 공유되는 단일 config 딕셔너리
    paths_list=paths_list,
    is_valid = True
)

# TEST DATA 개별 모델 추론 + 앙상블
run_wmf_ensemble(
    models=models,
    model_names=model_names,
    is_roi_list=is_roi_list,
    config_dict=config,
    paths_list=paths_list,
    is_valid = False
)