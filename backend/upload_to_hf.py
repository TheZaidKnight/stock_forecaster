#!/usr/bin/env python3
"""
Upload trained model artifacts to Hugging Face Hub.

This script supports pushing ARIMA/LSTM/GRU checkpoints and ancillary files
to a model repo under your HF account.

Usage:
  python backend/upload_to_hf.py \
    --repo my-username/stock-forecaster-aapl \
    --path artifacts/2025-10-06 \
    --private \
    --commit "Add AAPL models for 2025-10-06"

Auth:
  Set the environment variable HUGGINGFACE_HUB_TOKEN with a write token:
    set HUGGINGFACE_HUB_TOKEN=hf_xxx     (Windows)
    export HUGGINGFACE_HUB_TOKEN=hf_xxx  (Linux/macOS)

Artifacts directory layout (example):
  artifacts/
    2025-10-06/
      arima/
        params.json
        forecast.npy
      lstm/
        model.keras
        scaler.pkl
        forecast.npy
      gru/
        model.keras
        scaler.pkl
        forecast.npy
      metadata.json

The script will create the repo if missing and upload all files preserving
relative structure.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def ensure_repo(repo_id: str, private: bool, token: str | None) -> None:
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id)
        print(f"✅ Repo exists: {repo_id}")
    except Exception:
        print(f"ℹ️  Creating repo: {repo_id} (private={private})")
        create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)


def push_folder(repo_id: str, local_path: Path, commit_message: str, token: str | None) -> None:
    print(f"📤 Uploading from {local_path} to {repo_id} ...")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(local_path),
        commit_message=commit_message or "Upload model artifacts",
        path_in_repo="",
        ignore_patterns=["*.tmp", "*.log", "__pycache__/*"],
        token=token,
    )
    print("✅ Upload complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Target repo id, e.g. username/repo-name")
    parser.add_argument("--path", required=True, help="Local artifacts directory to upload")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist")
    parser.add_argument("--commit", default="", help="Commit message")
    parser.add_argument("--token", default=None, help="HF token (fallback to HUGGINGFACE_HUB_TOKEN env)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("❌ HUGGINGFACE_HUB_TOKEN environment variable not set.")
        print("   Create a write token at https://huggingface.co/settings/tokens and set it.")
        return 1

    artifacts = Path(args.path)
    if not artifacts.exists() or not artifacts.is_dir():
        print(f"❌ Artifacts path not found or not a directory: {artifacts}")
        return 1

    ensure_repo(args.repo, private=args.private, token=token)
    push_folder(args.repo, artifacts, args.commit, token=token)
    return 0


if __name__ == "__main__":
    sys.exit(main())


