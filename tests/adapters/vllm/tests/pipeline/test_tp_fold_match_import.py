"""`verity_vllm.pipeline.tp.fold_match` imports without numpy, torch or the core `verity`.  The command table imports it when
`tp-fold-match` is named, and only its `main`, when it runs, reads the fold worker's flags from `verity_vllm.pipeline.match`,
which loads the core."""
from __future__ import annotations

import subprocess
import sys


def test_importing_the_module_loads_neither_numpy_nor_torch_nor_the_core():
    code = ("import sys; before = set(sys.modules); import verity_vllm.pipeline.tp.fold_match; "
            "print(sorted({m.split('.')[0] for m in set(sys.modules) - before} & {'numpy', 'torch', 'verity'}))")
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[]"
