"""Island `e1mb` on the lowrank-restart branch: runs `islands/lowrank-restart/lowrank_restart.py` as class e1mb (its
docstring has the mechanisms and their flags).

    python guard.py islands/e1mb/train.py --start s100 --out SUB --mech relora|galore|full|subspace [FLAGS]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lowrank-restart"))

from lowrank_restart import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("e1mb"))
