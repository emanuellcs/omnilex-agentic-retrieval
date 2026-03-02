#!/usr/bin/env python3
"""Install llama-cpp-python with GPU support.

Auto-detects CUDA version and installs the appropriate prebuilt wheel.
Supports CUDA 12.1-12.5. Falls back to CPU if no compatible CUDA found.

Usage:
    python scripts/install_llama_gpu.py
    python scripts/install_llama_gpu.py --cuda 12.1  # Force specific version
    python scripts/install_llama_gpu.py --cpu        # Force CPU version
"""

import argparse
import re
import subprocess
import sys

SUPPORTED_CUDA = ["12.5", "12.4", "12.3", "12.2", "12.1"]
WHEEL_BASE_URL = "https://abetlen.github.io/llama-cpp-python/whl"


def get_cuda_version() -> str | None:
    """Detect installed CUDA version from nvcc or nvidia-smi."""
    # Try nvcc first (more reliable)
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # nvidia-smi shows "CUDA Version: X.Y"
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def find_compatible_cuda(detected: str) -> str | None:
    """Find the best compatible CUDA wheel version."""
    major_minor = detected.split(".")
    if len(major_minor) < 2:
        return None

    major = int(major_minor[0])
    minor = int(major_minor[1])

    # Only CUDA 12.x wheels are available
    if major != 12:
        return None

    # Find the highest compatible version <= detected
    for cuda_ver in SUPPORTED_CUDA:
        ver_parts = cuda_ver.split(".")
        ver_minor = int(ver_parts[1])
        if ver_minor <= minor:
            return cuda_ver

    return None


def install_llama_cpp(cuda_version: str | None = None, force_cpu: bool = False) -> bool:
    """Install llama-cpp-python with appropriate GPU/CPU support."""
    if force_cpu:
        print("Installing CPU version (forced)...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "llama-cpp-python",
        ]
    elif cuda_version:
        cuda_tag = f"cu{cuda_version.replace('.', '')}"
        wheel_url = f"{WHEEL_BASE_URL}/{cuda_tag}"
        print(f"Installing GPU version for CUDA {cuda_version}...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "llama-cpp-python",
            "--extra-index-url",
            wheel_url,
        ]
    else:
        print("Installing CPU version (no compatible CUDA found)...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "llama-cpp-python",
        ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Install llama-cpp-python with GPU support"
    )
    parser.add_argument(
        "--cuda",
        type=str,
        help="Force specific CUDA version (e.g., 12.1, 12.4)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only installation",
    )
    args = parser.parse_args()

    if args.cpu:
        success = install_llama_cpp(force_cpu=True)
    elif args.cuda:
        if args.cuda not in SUPPORTED_CUDA:
            print(f"Warning: CUDA {args.cuda} may not have prebuilt wheels.")
            print(f"Supported versions: {', '.join(SUPPORTED_CUDA)}")
        success = install_llama_cpp(cuda_version=args.cuda)
    else:
        # Auto-detect
        detected = get_cuda_version()
        if detected:
            print(f"Detected CUDA version: {detected}")
            compatible = find_compatible_cuda(detected)
            if compatible:
                print(f"Using compatible wheel for CUDA {compatible}")
                success = install_llama_cpp(cuda_version=compatible)
            else:
                print(f"No prebuilt wheel for CUDA {detected}.")
                print(f"Supported: {', '.join(SUPPORTED_CUDA)}")
                print("Falling back to CPU version.")
                success = install_llama_cpp(force_cpu=True)
        else:
            print("No CUDA installation detected.")
            success = install_llama_cpp(force_cpu=True)

    if success:
        print("\n✓ Installation complete!")
        # Verify
        try:
            from omnilex.llm import has_cuda_support

            if has_cuda_support():
                print("✓ GPU support enabled")
            else:
                print("→ Running in CPU mode")
        except ImportError:
            pass
    else:
        print("\n✗ Installation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
