"""Isolated access to the small ZXing finder/binarizer helper executables."""
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def run_helper(name, image):
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(__file__).resolve().parents[1] / "cpp/zxing-cpp_finder" / (name + suffix)
    if not executable.is_file():
        raise FileNotFoundError(f"Build the ZXing helper first: {executable}")
    # The existing helpers write zxing-cpp_finder/output.png relative to cwd.
    # A private directory makes calls independent of cwd and of other scans.
    with tempfile.TemporaryDirectory(prefix="csqrd-") as folder:
        workspace = Path(folder)
        (workspace / "zxing-cpp_finder").mkdir()
        source = workspace / "input.png"
        if not cv2.imwrite(str(source), np.asarray(image)):
            raise OSError("Could not write helper input")
        result = subprocess.run([str(executable), str(source)], cwd=workspace,
                                capture_output=True, text=True, check=True, timeout=30)
        binary = cv2.imread(str(workspace / "zxing-cpp_finder/output.png"), cv2.IMREAD_COLOR)
        if binary is None:
            raise OSError("ZXing helper did not produce its binary image")
        return result.stdout, binary
