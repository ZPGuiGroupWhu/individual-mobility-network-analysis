"""Run the public reproducibility workflow in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script_name: str, args: list[str]) -> None:
    """Run one workflow script with the current Python interpreter."""
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run data checks, feature analysis, optional downstream steps, and summary."""
    args = sys.argv[1:]
    for script in [
        "00_prepare_data.py",
        "01_feature_analysis.py",
        "02_multiscale_core_periphery.py",
        "03_network_compressibility.py",
        "04_summarize_results.py",
    ]:
        run(script, args)


if __name__ == "__main__":
    main()

