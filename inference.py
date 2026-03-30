def run_test_inference(model_path, source_dir, 
    imgsz=IMAGE_SIZE, conf=CONF_THRES, iou=IOU_THRES,
    config = config):
    print(f"CONF THRESH: {config['conf_thres']}")
    print(f"IOU THRESH: {config['iou_thres']}")
    """
    테스트 이미지에 대해 모델 추론을 수행하고 결과를 반환합니다.
    """
    print(f"🚀 모델 로드 중: {model_path}")
    model = YOLO(model_path)

    print(f"📂 추론 시작: {source_dir}")
    results = model.predict(
        source=source_dir,
        imgsz=imgsz,
        conf=config['conf_thres'],
        iou = config['iou_thres'],
        agnostic_nms = False,
        device=0,
        retina_masks=config['retina_masks'],
        verbose=False,
        save=False,
        stream=False
    )
    return results

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()
run_test_inference(model_path="best_365.pt", source_dir="data/test", config=config)
run_test_inference(model_path="best_360.pt", source_dir="data/test", config=config)
run_test_inference(model_path="best_357.pt", source_dir="data/test", config=config)
run_test_inference(model_path="best_355.pt", source_dir="data/test", config=config)


