"""Upload model AND preprocessor to HuggingFace Hub"""

import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 60)
    print("🚀 Uploading Model + Preprocessor to HuggingFace Hub")
    print("=" * 60)
    
    # Configuration
    repo_id = os.getenv('HF_MODEL_REPO', 'Terorra/fd_model_jedha')
    token = os.getenv('HF_TOKEN')
    
    if not token:
        print("❌ HF_TOKEN not found in .env")
        return
    
    api = HfApi()
    
    # Upload preprocessor
    print("\n📦 Uploading preprocessor.pkl...")
    try:
        api.upload_file(
            path_or_fileobj="../04_models/preprocessor.pkl",
            path_in_repo="preprocessor.pkl",
            repo_id=repo_id,
            token=token
        )
        print("✅ Preprocessor uploaded")
    except Exception as e:
        print(f"❌ Preprocessor upload failed: {e}")
        return
    
    # Upload model
    print("\n📦 Uploading fraud_model.pkl...")
    try:
        api.upload_file(
            path_or_fileobj="../04_models/fraud_model.pkl",
            path_in_repo="fraud_model.pkl",
            repo_id=repo_id,
            token=token
        )
        print("✅ Model uploaded")
    except Exception as e:
        print(f"❌ Model upload failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ Upload completed!")
    print(f"📍 Repository: https://huggingface.co/{repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    main()