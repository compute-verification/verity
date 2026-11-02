"""Island `lowrank-restart`: runs `lowrank_restart.py` as the class `--class` names (its docstring has the mechanisms and
their flags).

    python guard.py islands/lowrank-restart/train.py --class e1mb --start s100 --out SUB --mech relora|galore|full|subspace
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowrank_restart import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
