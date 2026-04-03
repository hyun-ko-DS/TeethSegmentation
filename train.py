import argparse
import os
from pathlib import Path
import torch

from ultralytics import YOLO
from ultralytics.utils import loss

from loss import nwd_iou_loss_patch, load_config
from utils import *

# 1. Argparse 설정: 모델명 파라미터 받기
def get_args():
    parser = argparse.ArgumentParser(description="YOLO Teeth Segmentation Training")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=["model_365", "model_360", "model_357", "model_355"],
                        help="Select model version to train (365, 360, 357, 355)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()

def prepare_model_environment(model_name):
    """모델별 서브폴더 생성 및 기존 가중치 경로 반환"""
    base_models_dir = Path("models")
    model_dir = base_models_dir / model_name
    
    # 폴더가 없으면 생성
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 폴더 생성 완료: {model_dir}")

    # 가중치 파일 경로: models/model_365/best_model_365.pt
    weight_path = model_dir / f"best_{model_name}.pt"
    
    # 만약 기존 가중치가 없다면 기본 YOLOv11n-seg.pt 등을 사용하도록 예외 처리
    if not weight_path.exists():
        print(f"⚠️ {weight_path}를 찾을 수 없습니다. 기본 모델(yolo11n-seg.pt)로 시작합니다.")
        return "yolo11n-seg.pt", model_dir
    
    print(f"✅ 기존 모델 로드: {weight_path}")
    return str(weight_path), model_dir

def main():
    args_cli = get_args()
    config = load_config() # nwd_alpha 등이 들어있는 json
    
    # 2. NWD Loss 몽키 패치 (이전과 동일)
    if not hasattr(loss, 'original_bbox_iou'):
        loss.original_bbox_iou = loss.bbox_iou
    loss.bbox_iou = nwd_iou_loss_patch
    print("✅ NWD Loss 패치가 성공적으로 주입되었습니다.")

    # 3. 모델 경로 및 저장 환경 세팅
    weight_path, save_dir = prepare_model_environment(args_cli.model_name)

    # 4. YOLO 모델 초기화
    model = YOLO(weight_path)

    # 5. 학습 인자 구성
    train_args = {
        "data": "dataset.yaml",       # 데이터셋 설정 파일
        "epochs": args_cli.epochs,
        "imgsz": args_cli.imgsz,
        "batch": args_cli.batch,
        "device": 0,                  # A6000 (첫 번째 GPU)
        "optimizer": "AdamW",
        "lr0": 1e-3,
        "project": "models",          # 'models' 폴더를 프로젝트 루트로
        "name": args_cli.model_name,  # 서브폴더명을 모델명으로 지정 (models/model_365/...)
        "exist_ok": True,             # 기존 폴더에 덮어쓰기/이어하기 가능하게 설정
        "save": True,
        "plots": True
    }

    print("\n" + "="*60)
    print(f"🔥 Expert Mode Training Start: {args_cli.model_name}")
    print(f"📌 Start Weight: {weight_path}")
    print(f"📌 Save Directory: {save_dir}")
    print("="*60)

    # 6. 학습 시작
    model.info()
    results = model.train(**train_args)

if __name__ == "__main__":
    main()