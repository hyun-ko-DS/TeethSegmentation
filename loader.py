from dotenv import load_dotenv
import huggingface_hub
from huggingface_hub import login
import os
from pathlib import Path
import shutil
import zipfile

def run_loading_pipeline(
):
    if login_wandb:
        load_dotenv(".env")
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if wandb_key:
            login(token=hf_token)
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

    # 압축 해제
    extract_dir = '/data/AlphaDent_extracted'

    if not os.path.exists(extract_dir):
        print(f"\nExtracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed")
    else:
        print(f"\nDataset already extracted at {extract_dir}")
    return extract_dir