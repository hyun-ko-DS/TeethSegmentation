import argparse
from pathlib import Path
from typing import Tuple
import gdown

from ultralytics import YOLO
from ultralytics.utils import loss

from caf import CAFBlock
from loss import make_nwd_iou_loss_patch
from utils import *

# Google Drive file IDs for best_{suffix}.pt (aligned with models/model_*/ structure)
DRIVE_BEST_PT_IDS = {
    "model_365": "10b1nx9PUgQWOVVPSRx7m98sJxURfqxhp",
    "model_360": "1Kj0-T9xiKdRugcHqaef2NpQ3hyNMmzY4",
    "model_357": "1rIHJakSahRRVOO1qfZjlAVxEUhO-yFP2",
    "model_355": "1f3AI8eawYGetpj_KOV9ywrC4QZQjFj50",
}

# Config keys that should not be passed directly to the YOLO train() method
_TRAIN_SKIP_KEYS = frozenset(
    {
        "model_name",
        "project_name",
        "run_name",
        "data",
        "project",
        "conf_thres",
        "iou_thres",
        "caf_alpha",
        "nwd_alpha",
        "caf_dilation_rates"
    }
)

def get_args():
    parser = argparse.ArgumentParser(
        description="YOLO Teeth Segmentation: Download checkpoints or execute training pipeline."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "download"],
        help="download: only fetch best_*.pt from GDrive. train: run the full training pipeline.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["model_365", "model_360", "model_357", "model_355"],
        help="Model variant (e.g., model_365 for weights located in models/model_365/).",
    )
    return parser.parse_args()

def model_checkpoint_paths(model_name: str) -> Tuple[Path, Path]:
    """Returns the directory and weight path for a given model variant."""
    suffix = model_name.rsplit("_", 1)[-1]
    model_dir = Path("models") / model_name
    weight_path = model_dir / f"best_{suffix}.pt"
    return model_dir, weight_path

def resolve_model_paths(model_name: str) -> Tuple[Path, Path, Path]:
    """Resolves paths for the specific model directory, JSON config, and dataset YAML."""
    suffix = model_name.rsplit("_", 1)[-1]
    model_dir = Path("models") / model_name
    config_path = model_dir / f"config_{suffix}.json"
    yaml_path = model_dir / f"yaml_{suffix}.yaml"
    return model_dir, config_path, yaml_path

def build_train_args(cfg, yaml_path, model_name):
    """Constructs the YOLO train() arguments by merging JSON config with forced path overrides."""
    train_args = {}
    for key, value in cfg.items():
        if key in _TRAIN_SKIP_KEYS:
            continue
        train_args[key] = value

    train_args["data"] = str(yaml_path)
    train_args["project"] = "models"
    train_args["name"] = model_name
    train_args["exist_ok"] = True
    train_args["save"] = True

    return train_args

def download_best_pt_from_drive(model_name: str, weight_path: Path) -> None:
    """Downloads the specific model checkpoint from Google Drive using gdown."""
    print("[1/5] Resolving Google Drive file ID and destination...")
    print(f"      Model Name  : {model_name}")
    print(f"      Destination : {weight_path.resolve()}")

    file_id = DRIVE_BEST_PT_IDS.get(model_name)
    if not file_id:
        raise RuntimeError(f"No GDrive ID found for {model_name}. Please update DRIVE_BEST_PT_IDS.")

    print("[2/5] Ensuring parent directory exists...")
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    print("[3/5] Starting download (byte progress via tqdm)...")
    gdown.download(id=file_id, output=str(weight_path), quiet=False)

    print("[4/5] Verifying integrity on disk...")
    if not weight_path.is_file():
        raise RuntimeError(f"Download failed: {weight_path} is missing.")
    
    size_mb = weight_path.stat().st_size / (1024 * 1024)
    print(f"[5/5] Success. Saved {weight_path.name} ({size_mb:.2f} MiB) to {weight_path.parent.resolve()}")

def run_download_only(model_name: str) -> None:
    """Execution mode to only fetch weights without starting training."""
    print("=" * 60)
    print("MODE: DOWNLOAD ONLY — Fetching checkpoints from GDrive")
    print("=" * 60)
    model_dir, weight_path = model_checkpoint_paths(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    download_best_pt_from_drive(model_name, weight_path)

def prepare_model_environment(model_name: str) -> Tuple[str, Path]:
    """Ensures environment is ready and downloads weights if not present locally."""
    model_dir, weight_path = model_checkpoint_paths(model_name)

    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created model directory: {model_dir}")

    if weight_path.exists():
        print(f"Using existing local checkpoint: {weight_path.resolve()}")
    else:
        print("Local checkpoint missing. Initializing GDrive download...")
        download_best_pt_from_drive(model_name, weight_path)

    return str(weight_path), model_dir

def run_train(model_name: str) -> None:
    """Main training pipeline: config loading, model patching, and execution."""
    model_dir, config_path, yaml_path = resolve_model_paths(model_name)
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Config missing: {config_path}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML missing: {yaml_path}")

    # 1. Load JSON Configuration
    config = load_config(str(config_path))
    
    # 2. Inject CAF Parameters for specific models (360, 355)
    if model_name in ["model_360", "model_355"]:
        caf_alpha = config["caf_alpha"]
        caf_dilations = config["caf_dilation_rates"]
        
        # Inject values into the CAFBlock class before model instantiation
        CAFBlock.runtime_alpha = caf_alpha
        CAFBlock.runtime_dilation_rates = tuple(caf_dilations)
        print(f"✨ CAF Config Injected: Alpha={caf_alpha}, Dilations={caf_dilations}")

    # 3. Apply NWD Loss Patch
    nwd_alpha = config["nwd_alpha"]
    if not hasattr(loss, "original_bbox_iou"):
        loss.original_bbox_iou = loss.bbox_iou
    loss.bbox_iou = make_nwd_iou_loss_patch(float(nwd_alpha))
    print(f"✅ NWD Loss Patch Applied (Alpha={nwd_alpha})")

    # 4. Prepare Environment and Instantiate YOLO
    weight_path, save_dir = prepare_model_environment(model_name)
    model = YOLO(weight_path)
    
    # 5. Build Training Arguments
    train_args = build_train_args(config, yaml_path, model_name)

    # 6. Detailed Parameter Summary (Pre-flight Check)
    print("\n" + "=" * 60)
    print(f"🚀 EXPERT MODE TRAINING: {model_name}")
    print("-" * 60)
    print(f"📍 Config File  : {config_path}")
    print(f"📍 Data YAML    : {yaml_path}")
    print(f"📍 Start Weights: {weight_path}")
    print(f"📍 Save Dir     : {save_dir}")
    
    print("-" * 60)
    print("🔍 Full Configuration Parameters:")
    for key in sorted(config.keys()):
        value = config[key]
        print(f"   • {key:<18}: {value}")
    print("=" * 60 + "\n")

    # Execute training
    model.info()
    model.train(**train_args)

def main():
    args_cli = get_args()
    if args_cli.mode == "download":
        run_download_only(args_cli.model_name)
    else:
        run_train(args_cli.model_name)

if __name__ == "__main__":
    main()