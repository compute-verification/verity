"""Island `e100kb` on the placement direction's branch: runs islands/placement/train.py as class e100kb."""
import runpy
import sys
from pathlib import Path

sys.argv[1:1] = ["--class", "e100kb"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "placement" / "train.py"), run_name="__main__")
