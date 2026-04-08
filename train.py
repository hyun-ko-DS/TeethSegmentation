import argparse
from pathlib import Path
from typing import Tuple, Optional
import gdown

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.utils import loss

from caf import *
from loss import *
from utils import *

# Register custom module
tasks.CAFBlock = CAFBlock

# Google Drive file IDs
DRIVE_IDS = {
    "model_365": {
        "pt": "10b1nx9PUgQWOVVPSRx7m98sJxURfqxhp",
        "onnx": "1xwpTu5knpAI9igwqOIBy373eUi7MlySI",
        "json": "1We0eBkbrG4_SBn5UVnR_smyLRUpryRBr",
        "engine": "13fyssWX7NhWRQXjRv2Y9n4IC01-X2Hdb"
    },
    "model_360": {
        "pt": "1Kj0-T9xiKdRugcHqaef2NpQ3hyNMmzY4",
        "onnx": "1C-RiOOO8Gf7G0M-ff8aJZoJn8LFpFsWU",
        "json": "1oOUOEEpgQulWgIEFS9aGtqoFfX2EeXXZ",
        "engine": "1UlKvPplyNb6CYoDRvvyhyvV5C7nk93_A"
    },
    "model_357": {
        "pt": "1rIHJakSahRRVOO1qfZjlAVxEUhO-yFP2",
        "onnx": "1YURCE37EI0PP1xc8_jcewHseyjI4JAf1",
        "json": "1_EjLeM0JaGBrrFAmnYyRt5pBrOnLMhT5",
        "engine": "1dgbgJH5_8uf3h0ARhs1PGOT-NYTGDMTy"
    },
    "model_355": {
        "pt": "1f3AI8eawYGetpj_KOV9ywrC4QZQjFj50",
        "onnx": "1eObLMQ9tbLwSl2g5NA1lmIwlS5wHNOhO",
        "json": "1PsJrDJ8wqTsMPlz0EJt0WA8pCt5LvIf8",
        "engine": "1J9YzgeeUL2UVuARTkp9dTbM_McnsnxWu"
    },
}

_TRAIN_SKIP_KEYS = frozenset({
    "model_name", "project_name", "run_name", "data", "project",
    "conf_thres", "iou_thres", "caf_alpha", "nwd_alpha", "caf_dilation_rates"
})

def get_args():
    parser = argparse.ArgumentParser(
        description="YOLO Teeth Segmentation: Training and Checkpoint Management"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "download", "from_drive"],
        help="train: Scratch training (no GDrive download). "
             "from_drive: Download best_*.pt and start training from it. "
             "download: Only fetch best_*.pt from GDrive.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["model_365", "model_360", "model_357", "model_355"],
        help="Model variant name.",
    )
    return parser.parse_args()

def get_all_paths(model_name: str):
    """모델 이름에 따른 모든 로컬 경로 반환"""
    suffix = model_name.rsplit("_", 1)[-1]
    model_dir = Path("models") / model_name
    return {
        "pt": model_dir / f"best_{suffix}.pt",
        "onnx": model_dir / f"best_{suffix}.onnx",
        "json": model_dir / f"config_{suffix}.json",
        "yaml": model_dir / f"yaml_{suffix}.yaml" # YAML은 보통 로컬 생성
    }

def model_checkpoint_paths(model_name: str) -> Tuple[Path, Path]:
    suffix = model_name.rsplit("_", 1)[-1]
    model_dir = Path("models") / model_name
    weight_path = model_dir / f"best_{suffix}.pt"
    return model_dir, weight_path

def resolve_model_paths(model_name: str) -> Tuple[Path, Path, Path]:
    suffix = model_name.rsplit("_", 1)[-1]
    model_dir = Path("models") / model_name
    config_path = model_dir / f"config_{suffix}.json"
    yaml_path = model_dir / f"yaml_{suffix}.yaml"
    return model_dir, config_path, yaml_path

def download_resource(model_name: str, file_type: str, target_path: Path):
    """공통 다운로드 로직"""
    ids = DRIVE_IDS.get(model_name)
    if not ids or file_type not in ids or ids[file_type] == "1-XXXXX":
        print(f"⚠️ Skip: No valid GDrive ID for {model_name} ({file_type})")
        return

    file_id = ids[file_type]
    print(f"📡 Downloading {file_type} for {model_name}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdown.download(id=file_id, output=str(target_path), quiet=False)
    
    if not target_path.is_file():
        print(f"❌ Failed to download {target_path}")

def run_train(model_name: str, use_drive_weights: bool = False) -> None:
    paths = get_all_paths(model_name)

    # 1. 필수 파일(Config, YAML) 확인 및 자동 다운로드 시도
    if not paths["json"].exists():
        download_resource(model_name, "json", paths["json"])
    
    if not paths["json"].is_file() or not paths["yaml"].is_file():
        raise FileNotFoundError(f"Missing config or yaml for {model_name}. Please check local 'models' folder.")

    # 2. Load Config
    config = load_config(str(paths["json"]))
    
    # 3. Inject CAF & NWD Loss Patch (기존 로직 유지)
    if model_name in ["model_365", "model_355"]:
        CAFBlock.runtime_alpha = config["caf_alpha"]
        CAFBlock.runtime_dilation_rates = tuple(config["caf_dilation_rates"])
        print(f"✨ CAF Config Injected: Alpha={config['caf_alpha']}")

    nwd_alpha = config["nwd_alpha"]
    if not hasattr(loss, "original_bbox_iou"):
        loss.original_bbox_iou = loss.bbox_iou
    loss.bbox_iou = make_nwd_iou_loss_patch(float(nwd_alpha))
    print(f"✅ NWD Loss Patch Applied (Alpha={nwd_alpha})")

    # 4. Handle Model Initialization
    if use_drive_weights:
        if not paths["pt"].exists():
            download_resource(model_name, "pt", paths["pt"])
        
        model = YOLO(str(paths["pt"]))
        start_weights = str(paths["pt"])
    else:
        print(f"🚀 Starting training FROM SCRATCH using {paths['yaml'].name}")
        model = YOLO(str(paths["yaml"]))
        start_weights = "None (Scratch)"

    # 5. Build Training Arguments (기존 로직 유지)
    train_args = {k: v for k, v in config.items() if k not in _TRAIN_SKIP_KEYS}
    train_args.update({
        "data": str(paths["yaml"]),
        "project": "models",
        "name": model_name,
        "exist_ok": True,
        "save": True
    })

    # Execute
    model.train(**train_args)

def main():
    args_cli = get_args()
    paths = get_all_paths(args_cli.model_name)
    
    if args_cli.mode == "download":
        for f_type in ["pt", "onnx", "json"]:
            download_resource(args_cli.model_name, f_type, paths[f_type])
        print(f"✅ All resources for {args_cli.model_name} checked.")
    
    elif args_cli.mode in ["from_drive", "train"]:
        run_train(args_cli.model_name, use_drive_weights=(args_cli.mode == "from_drive"))

if __name__ == "__main__":
    main()