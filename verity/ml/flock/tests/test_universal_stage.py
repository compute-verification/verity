"""`verity_flock.universal_stage`, the universal unit's class statement from arrays (``pod/rec_inner.py universal --fast``):
its gate circuit is `universal.traced`'s, and the files it stages are `class_statement.stage`'s, byte for byte."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
POD = HERE.parent / "pod"
ROOT = HERE.parents[3]

#: (G, N_IN, N_OUT): odd input counts, outputs that leave the output group's last word part empty (8,5,3; 24,9,5), and
#: 87-rec-private.sh's default class; each netlist fills 3/4 of the gates, the rest padding
CLASSES = [(8, 5, 3), (16, 16, 8), (24, 9, 5)]


@pytest.mark.parametrize("cls", [(4, 3, 2), *CLASSES, (40, 7, 6)])
def test_trace_is_universal_traced(cls):
    from verity_flock import universal_stage as US

    from verity.ml.private_circuits import universal as U

    want = U.traced(*cls)
    got = US.trace(*cls).traced()
    assert (got.n_inputs, got.outputs, got.counts) == (want.n_inputs, want.outputs, want.counts)
    assert got.gates == [(k, tuple(ops)) for k, ops in want.gates]


_STAGE_BOTH = r"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import rec_inner as RI
RI._benchmark()
from verity_flock import class_statement as CS
CS.cache_slot_circuits()
res = {}
for cls, n in json.loads(sys.argv[3]):
    for fast in (False, True):
        d = Path(sys.argv[2]) / f"{cls}-{fast}"
        rec = RI.universal(cls, n, d, 0, fast=fast, salt_key=bytes(range(32)))
        files = {f"{s}/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest() for s in ("stage", "other") for f in sorted((d / s).iterdir())}
        res[f"{cls}/{fast}"] = {"record": rec, "files": files}
print(json.dumps(res))
"""


@pytest.mark.slow
def test_fast_staging_writes_class_statement_stage_files(tmp_path):
    """Both statements `rec_inner.universal` stages (the registered netlist's and the other's) through today's path and through
    `universal_stage`, under one salt key: every file (``circuit.txt``, ``inst-n.bin``, ``pub-n.bin``, ``partition.json``) has
    the same SHA-256, and the records agree. One process for all classes, the slot circuits composed once (about a minute):
    the layout, the composed circuit and the salted rows are only checked whole by staging them."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (ROOT, ROOT / "catalog", ROOT / "kernels",
                                                                        ROOT / "verity" / "ml" / "flock" / "python")))
    todo = [[",".join(map(str, c)), 1 + c[2] % 4] for c in CLASSES]
    p = subprocess.run([sys.executable, "-c", _STAGE_BOTH, str(POD), str(tmp_path), json.dumps(todo)], env=env, capture_output=True,
                       text=True, check=False)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout)
    for cls, n in todo:
        today, fast = res[f"{cls}/False"], res[f"{cls}/True"]
        names = ("circuit.txt", f"inst-{n}.bin", "partition.json", f"pub-{n}.bin")
        assert sorted(today["files"]) == [f"{s}/{f}" for s in ("other", "stage") for f in names]
        assert fast["files"] == today["files"], cls
        assert fast["record"] == today["record"], cls
        assert today["record"]["netlist"]["padded_gates"] > 0
