"""Island `lowrank-restart`: runs `lowrank_restart.py` (its docstring has the mechanisms and their flags);
`--program delta_grad.py` runs the compressed-training-state program instead.

    python benchmarks/covert_witness/launch.py s-t20 --island lowrank-restart --note "galore ..." -- \\
        --max-e-bytes 1000000 --mech relora|stack|galore|full|subspace ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowrank_restart import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
