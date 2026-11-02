"""Island `e1mb` on the sparse branch: runs `islands/sparse/train.py` (the sparse island's script) with `--klass e1mb`.

    python guard.py islands/e1mb/train.py --start s50 --out SUB --mech lines [TRAIN_ARGS...]
"""
import runpy
import sys
from pathlib import Path

sys.argv[1:1] = ["--klass", "e1mb"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "sparse" / "train.py"), run_name="__main__")
