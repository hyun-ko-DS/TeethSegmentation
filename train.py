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
DRIVE_BEST_PT_IDS = {
    "model_365": "10b1nx9PUgQWOVVPSRx7m98sJxURfqxhp",
    "model_360": "1Kj0-T9xiKdRugcHqaef2NpQ3hyNMmzY4",
    "model_357": "1rIHJakSahRRVOO1qfZjlAVxEUhO-yFP2",
    "model_355": "1f3AI8eawYGetpj_KOV9ywrC4QZQjFj50",
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

def download_best_pt_from_drive(model_name: str, weight_path: Path) -> None:
    print(f"📡 Initializing GDrive download for {model_name}...")
    file_id = DRIVE_BEST_PT_IDS.get(model_name)
    if not file_id:
        raise RuntimeError(f"No GDrive ID found for {model_name}.")
    
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(weight_path), quiet=False)
    
    if not weight_path.is_file():
        raise RuntimeError(f"Download failed for {weight_path}")

def run_train(model_name: str, use_drive_weights: bool = False) -> None:
    """
    Main training pipeline.
    If use_drive_weights is True, it downloads/uses best_*.pt.
    If False, it initializes from yaml (scratch).
    """
    model_dir, config_path, yaml_path = resolve_model_paths(model_name)
    _, weight_path = model_checkpoint_paths(model_name)

    if not config_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError(f"Missing config or yaml in {model_dir}")

    # 1. Load Config
    config = load_config(str(config_path))
    
    # 2. Inject CAF (Only for specific models)
    if model_name in ["model_365", "model_355"]:
        CAFBlock.runtime_alpha = config["caf_alpha"]
        CAFBlock.runtime_dilation_rates = tuple(config["caf_dilation_rates"])
        print(f"✨ CAF Config Injected: Alpha={config['caf_alpha']}")

    # 3. Apply NWD Loss Patch
    nwd_alpha = config["nwd_alpha"]
    if not hasattr(loss, "original_bbox_iou"):
        loss.original_bbox_iou = loss.bbox_iou
    loss.bbox_iou = make_nwd_iou_loss_patch(float(nwd_alpha))
    print(f"✅ NWD Loss Patch Applied (Alpha={nwd_alpha})")

    # 4. Handle Model Initialization
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if use_drive_weights:
        # Mode: from_drive
        if not weight_path.exists():
            download_best_pt_from_drive(model_name, weight_path)
        else:
            print(f"Using existing local checkpoint: {weight_path}")
        model = YOLO(str(weight_path))
        start_weights = str(weight_path)
    else:
        # Mode: train (Scratch)
        print(f"🚀 Starting training FROM SCRATCH using {yaml_path.name}")
        model = YOLO(str(yaml_path))
        start_weights = "None (Scratch)"

    # 5. Build Training Arguments
    train_args = {}
    for key, value in config.items():
        if key not in _TRAIN_SKIP_KEYS:
            train_args[key] = value

    train_args.update({
        "data": str(yaml_path),
        "project": "models",
        "name": model_name,
        "exist_ok": True,
        "save": True
    })

    # 6. Pre-flight Summary
    print("\n" + "=" * 60)
    print(f"🔥 TRAINING MODE: {'FROM DRIVE' if use_drive_weights else 'SCRATCH'}")
    print(f"📍 Model Name   : {model_name}")
    print(f"📍 Start Weights: {start_weights}")
    print(f"📍 Save Dir     : {model_dir}")
    print("=" * 60 + "\n")

    # Execute
    model.train(**train_args)

def main():
    args_cli = get_args()
    
    if args_cli.mode == "download":
        model_dir, weight_path = model_checkpoint_paths(args_cli.model_name)
        download_best_pt_from_drive(args_cli.model_name, weight_path)
    
    elif args_cli.mode == "from_drive":
        run_train(args_cli.model_name, use_drive_weights=True)
    
    elif args_cli.mode == "train":
        run_train(args_cli.model_name, use_drive_weights=False)

if __name__ == "__main__":
    main()