#!/usr/bin/env python3
"""
scripts/download_sr_weights.py
=============================================================================
Downloads official pretrained Real-ESRGAN x4plus weights from GitHub releases:
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

Verifies SHA-256 checksum and saves to models/RealESRGAN_x4plus.pth.
Fails loudly if network unreachable or checksum mismatch.
"""

import os
import sys
import hashlib
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
WEIGHTS_FILENAME = "RealESRGAN_x4plus.pth"
TARGET_PATH = os.path.join(MODELS_DIR, WEIGHTS_FILENAME)

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
# Official expected SHA-256 for RealESRGAN_x4plus.pth
EXPECTED_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
EXPECTED_SIZE = 67040989


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_weights():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(TARGET_PATH) and os.path.getsize(TARGET_PATH) == EXPECTED_SIZE:
        print(f"File exists at {TARGET_PATH}, verifying checksum...")
        h = compute_sha256(TARGET_PATH)
        if h.lower() == EXPECTED_SHA256.lower():
            print(f"SUCCESS: Pretrained weights verified (SHA-256: {h})")
            return
        else:
            print(f"Checksum mismatch on existing file: {h} != {EXPECTED_SHA256}. Re-downloading...")

    print(f"Downloading {URL} -> {TARGET_PATH} ...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BioTrace/1.0"}
    req = urllib.request.Request(URL, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(TARGET_PATH, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1048576  # 1MB
            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                out_f.write(buf)
                downloaded += len(buf)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r  Downloaded {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)")
                    sys.stdout.flush()
        print("\nDownload complete. Calculating SHA-256...")
    except Exception as e:
        if os.path.exists(TARGET_PATH):
            os.remove(TARGET_PATH)
        raise RuntimeError(f"FATAL: Failed to download Real-ESRGAN weights from {URL}: {e}")

    actual_sha = compute_sha256(TARGET_PATH)
    actual_size = os.path.getsize(TARGET_PATH)

    if actual_sha.lower() != EXPECTED_SHA256.lower():
        os.remove(TARGET_PATH)
        raise ValueError(
            f"FATAL: Checksum mismatch for {TARGET_PATH}!\n"
            f"  Expected: {EXPECTED_SHA256}\n"
            f"  Actual:   {actual_sha}"
        )

    print(f"SUCCESS: Real-ESRGAN weights verified and saved to {TARGET_PATH} ({actual_size:,} bytes).")


if __name__ == "__main__":
    download_weights()
