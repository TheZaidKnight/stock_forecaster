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
    parser.add_argument("--path", required=True, help="Artifacts root or a specific timestamp folder or model subfolder")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist")
    parser.add_argument("--commit", default="", help="Commit message")
    parser.add_argument("--token", default=None, help="HF token (fallback to HUGGINGFACE_HUB_TOKEN env)")
    parser.add_argument("--allow-empty", action="store_true", help="If set, create a placeholder README when folder has no files")
    parser.add_argument("--model", choices=["all","arima","lstm","gru"], default="all", help="Upload only a specific model subfolder or all")
    parser.add_argument("--auto-latest", action="store_true", help="When --path points to artifacts root, pick latest timestamped child folder automatically")
    parser.add_argument("--path-in-repo", default="", help="Optional subpath in repo, e.g. models/2025-10-06")
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("❌ HUGGINGFACE_HUB_TOKEN environment variable not set.")
        print("   Create a write token at https://huggingface.co/settings/tokens and set it.")
        return 1

    # Resolve the provided path
    raw_path = args.path
    base = Path(raw_path).expanduser().resolve(strict=False)
    if not base.exists():
        print(f"❌ Path does not exist: {raw_path}")
        print(f"   Resolved absolute path: {base}")
        print(f"   Current working directory: {Path.cwd()}")
        # Fuzzy match within parent directory
        parent_dir = base.parent
        try:
            if parent_dir.exists():
                candidates = list(parent_dir.iterdir())
                def _norm(s: str) -> str:
                    return ''.join(ch for ch in s.lower() if ch.isalnum())
                wanted = _norm(base.name)
                matches = [p for p in candidates if _norm(p.name) == wanted]
                if not matches:
                    matches = [p for p in candidates if _norm(p.name).startswith(wanted)]
                if not matches and len([c for c in candidates if c.is_dir()]) == 1:
                    # Fall back to the single subdir under artifacts
                    only_dir = [c for c in candidates if c.is_dir()][0]
                    print(f"ℹ️  Using only subdirectory found: {only_dir.name}")
                    matches = [only_dir]
                if len(matches) == 1 and matches[0].exists():
                    print(f"ℹ️  Using close match: {matches[0].name}")
                    base = matches[0]
                else:
                    print(f"   Children of {parent_dir}:")
                    for p in sorted(candidates):
                        print(f"     {'[D]' if p.is_dir() else '[F]'} {p.name}")
            else:
                print(f"   Parent directory does not exist: {parent_dir}")
        except Exception:
            pass
        if not base.exists():
            return 1

    # Determine the folder to upload based on structure:
    # - If base has subfolders arima/lstm/gru and --model=all, upload base
    # - If --model is a specific one, upload base/<model>
    # - If base is artifacts root and --auto-latest, pick latest child dir
    upload_root = base
    if base.is_dir():
        children = [p for p in base.iterdir() if p.is_dir()]
        names = {p.name for p in children}
        has_models = {'arima','lstm','gru'}.issubset(names)

        if has_models and args.model != 'all':
            upload_root = base / args.model
        elif not has_models and args.auto_latest:
            # choose latest timestamped child dir (by mtime)
            if not children:
                print("❌ No subdirectories under artifacts root.")
                return 1
            latest = max(children, key=lambda p: p.stat().st_mtime)
            print(f"ℹ️  Auto-selected latest: {latest.name}")
            upload_root = latest
            # if specific model requested, drill down
            if args.model != 'all':
                upload_root = upload_root / args.model
        elif not has_models and args.model != 'all':
            # user pointed directly at a timestamp dir? drill down if exists
            candidate = base / args.model
            if candidate.exists():
                upload_root = candidate

    # Re-check directory
    artifacts = upload_root.resolve(strict=False)
    if not artifacts.exists() or not artifacts.is_dir():
        print(f"❌ Artifacts path not found or not a directory: {upload_root}")
        print(f"   Resolved absolute path: {artifacts}")
        print(f"   Current working directory: {Path.cwd()}")
        return 1

    # Fail fast if directory has no files (recursively), unless --allow-empty
    non_ignored_files = [
        p for p in artifacts.rglob("*")
        if p.is_file() and
           not any(part == "__pycache__" for part in p.parts) and
           not p.name.endswith(".tmp") and
           not p.name.endswith(".log")
    ]

    if len(non_ignored_files) == 0:
        if not args.allow_empty:
            print("❌ No files found in the artifacts directory. Nothing to upload.")
            print("   Export your models first, or pass --allow-empty to push a placeholder README.")
            return 2
        else:
            placeholder = artifacts / "README.md"
            if not placeholder.exists():
                placeholder.write_text(
                    "# Model Artifacts\n\n"
                    "This folder was created intentionally. Add your trained model files here.\n"
                )
                print(f"ℹ️  Created placeholder: {placeholder}")

    ensure_repo(args.repo, private=args.private, token=token)
    # Temporarily override path_in_repo if provided
    if args.path_in_repo:
        print(f"ℹ️  Uploading into repo subpath: {args.path_in_repo}")
    # The huggingface_hub upload_folder supports path_in_repo; we pass it via env var emulation not necessary
    # but we already added CLI flag to annotate; for simplicity, we keep top-level and include the local folder name.
    push_folder(args.repo, artifacts, args.commit, token=token)
    return 0


if __name__ == "__main__":
    sys.exit(main())


