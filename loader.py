import huggingface_hub
import os
import shutil
import wandb
import zipfile
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path


def run_loading_pipeline(login_wandb=False):
    """
    Downloads the AlphaDent dataset from Hugging Face and extracts the contents.
    """
    # 1. Authentication Stage
    if login_wandb:
        load_dotenv(".env")
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            login(token=hf_key)
            print("✅ Hugging Face logged in successfully")
        else:
            print("⚠️ HUGGINGFACE_API_KEY not found. Skipping Hugging Face login.")

    # 2. Download Stage
    print("🚚 Initializing dataset download from Hugging Face Hub...")
    cache_dir = huggingface_hub.snapshot_download(
        repo_id="ZFTurbo/AlphaDent",
        repo_type="dataset"
    )
    print(f"📍 Dataset cached at: {cache_dir}")

    # 3. ZIP File Discovery
    zip_path = None
    # Traverse the cache directory to find the AlphaDent ZIP file
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.zip') and 'AlphaDent' in file:
                zip_path = os.path.join(root, file)
                break
        if zip_path:
            break

    if zip_path:
        print(f"📦 Found ZIP file: {os.path.basename(zip_path)}")
    else:
        raise FileNotFoundError("❌ ZIP file not found in the cached directory.")

    # 4. Extraction Stage
    extract_dir = Path(__file__).resolve().parent / "data" / "alphadent_extracted"
    
    # Check if extraction is necessary
    need_extract = not extract_dir.exists() or not any(extract_dir.iterdir())

    if need_extract:
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Extracting contents to: {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Extraction completed successfully.")
    else:
        print(f"ℹ️ Dataset already exists at: {extract_dir}")
        
    return str(extract_dir)


if __name__ == "__main__":
    # Execute the loading pipeline
    run_loading_pipeline()