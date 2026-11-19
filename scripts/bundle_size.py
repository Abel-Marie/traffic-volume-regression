"""Measure the service bundle as the hosting platform would install it.

Installs service/requirements.txt for Linux (the platform Vercel runs on), whatever machine this
script runs on, into a temporary folder and adds the service's own files. Vercel's limit for a Python
function is 250 MB unzipped.

    python scripts/bundle_size.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 250


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and "__pycache__" not in f.parts) / 1e6


def install_for_linux(target: str) -> None:
    requirements = str(ROOT / "service" / "requirements.txt")
    if shutil.which("uv"):
        command = ["uv", "pip", "install", "-q", "--python-platform", "x86_64-manylinux_2_28", "--python-version", "3.12",
                   "--target", target, "-r", requirements]
    else:
        command = [sys.executable, "-m", "pip", "install", "-q", "--target", target, "--platform", "manylinux_2_28_x86_64",
                   "--python-version", "3.12", "--only-binary=:all:", "-r", requirements]
    subprocess.check_call(command)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        install_for_linux(tmp)
        deps = folder_mb(Path(tmp))
    own = folder_mb(ROOT / "service")
    total = deps + own
    print(f"Linux dependencies {deps:.1f} MB + service files {own:.2f} MB = {total:.1f} MB (limit {LIMIT_MB} MB)")
    return 0 if total < LIMIT_MB else 1


if __name__ == "__main__":
    sys.exit(main())
