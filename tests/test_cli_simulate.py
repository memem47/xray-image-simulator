"""
CLI integration test.

Invokes the simulator as a module:

    python -m simulator.simulate --kvp 70 --mas 5 ...

This avoids relying on a legacy wrapper at repo root
"""
import subprocess, tempfile, os
import sys
import numpy as np
from PIL import Image

PY = sys.executable # full path to current Python interpreter

def test_simulate_cli():
    with tempfile.TemporaryDirectory() as tmp:
        png_path = os.path.join(tmp, "out.png")
        cmd = [
            PY, "-m", "simulator.simulate",
            "--kvp", "70",
            "--mas", "5",
            "--cone-scale", "0.3",
            "--cone-offset", "20", "40",
            "--out", png_path
        ]
        # Capture stdout/stderr for debugging; fail if returncode != 0
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Sanity: confirm script message
        assert "Saved image to" in result.stdout

        # File exists & looks like a grayscale PNG
        assert os.path.exists(png_path)
        img = np.array(Image.open(png_path))
        assert img.ndim == 2 and img.max() > img.min()