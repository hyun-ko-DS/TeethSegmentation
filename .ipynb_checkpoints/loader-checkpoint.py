from dotenv import load_dotenv
import huggingface_hub
from huggingface_hub import login
import os
from pathlib import Path
import shutil
import wandb
import zipfile

def run_loading_pipeline(login_wandb=False):
    if login_wandb:
        load_dotenv(".env")
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            login(token=hf_key)
            print("Hugging Face logged in successfully")
        else:
            print("HUGGINGFACE_API_KEY not found. Skipping hugging face login.")

    cache_dir = huggingface_hub.snapshot_download(
    repo_id="ZFTurbo/AlphaDent",
    repo_type="dataset"
    )

    print(f"\nDataset cached at: {cache_dir}")

    # ZIP 파일 찾기
    zip_path = None
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.zip') and 'AlphaDent' in file:
                zip_path = os.path.join(root, file)
                break
        if zip_path:
            break

    if zip_path:
        print(f"Found ZIP file: {os.path.basename(zip_path)}")
    else:
        raise FileNotFoundError("ZIP file not found")

    extract_dir = Path(__file__).resolve().parent / "data" / "alphadent_extracted"
    need_extract = not extract_dir.exists() or not any(extract_dir.iterdir())

    if need_extract:
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExtracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed")
    else:
        print(f"\nDataset already extracted at {extract_dir}")
    return str(extract_dir)

run_loading_pipeline()