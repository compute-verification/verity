"""Core's pin of the torch loop's checkpoint role (`checkpoint.ROLE`, its files and manifest entry circuits' `checkpoint`
writes): credit holds a period's open to the digest of those files, and the auditor's model registry registers their rows
under a window Program of the role's own linears, refusing one of any other."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from verity.ml.pouw import credit as CR
from verity.ml.pouw import weights as Wt
from verity.ml.circuits import codec
from verity_torch_loop import checkpoint as C
from verity_torch_loop import loop as L

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "benchmarks" / "live_challenger"))
import models as M  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "h1window_case", ROOT / "verity/ml/pouw/tests/test_circuit_h1window.py")
H1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H1)

KEY = bytes(range(32))


def descriptor(linears) -> bytes:
    calls = tuple((name, 0) for name, _k, _n in linears)
    desc = codec.encode_program(H1.HW.program(linears, calls, 4, H1.TILE, prefix="h1/", unit=H1.unit))
    return codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"})


def test_core_pins_the_loops_committed_checkpoint():
    """``models.json``'s entry of the role is each committed file's SHA-512 and their canonical JSON's digest, the one
    credit holds an open's model to, with the manifest's repo, revision and recipe."""
    entry = M.manifest_entry(ROOT, C.ROLE)
    pin = json.loads((ROOT / M.PINS).read_text())[C.ROLE]
    files = {n: (C.CHECKPOINT / n).read_bytes() for n in entry["files"]}
    assert pin["files"] == {n: hashlib.sha512(b).hexdigest() for n, b in files.items()}
    assert CR.PINNED_MODELS[C.ROLE] == pin["checkpoint_sha512"] == C.checkpoint_sha512(files)
    assert M.pinned_checkpoint(ROOT, C.ROLE) == pin["checkpoint_sha512"]
    assert (pin["repo"], pin["revision"], pin["recipe"]) == (entry["repo"], entry["revision"], entry["recipe"]["id"])


def test_the_registry_commits_the_loops_rows_for_a_program_of_its_linears(tmp_path):
    """`models.register` (the gate's ``model-register``) takes the committed checkpoint for a descriptor of the role's two
    linears, its b/<128> root the loop's own int8 codes and s_w row by row; a descriptor of them reversed is refused."""
    ck = tmp_path / "ckpt"
    ck.mkdir(mode=0o755)
    for n in M.manifest_entry(ROOT, C.ROLE)["files"]:
        shutil.copyfile(C.CHECKPOINT / n, ck / n)
        (ck / n).chmod(0o644)
    m = L.Model()
    lin = [(m.op_path(i), m.h, m.h) for i in range(len(m.layers))]
    d = descriptor(lin)
    circuit = hashlib.sha512(d).hexdigest()
    line = M.register(ROOT, C.ROLE, circuit, d, ck, KEY, pinned="ab" * 20, now="2026-10-25T20:00:00Z")
    rows = [[layer.wq.numpy()[j].tobytes() + layer.sw.numpy().astype("<f4")[j:j + 1].tobytes() for j in range(m.h)]
            for layer in m.layers]
    want = Wt.weights_roots(circuit, "h1/", [(m.h, m.h)] * len(m.layers), lambda w: rows[w], KEY)
    assert line["roots"] == {str(m.h): want[m.h].hex()} and line["weights_root"] == Wt.weights_root(want).hex()
    assert line["checkpoint_sha512"] == CR.PINNED_MODELS[C.ROLE] and line["model"] == C.ROLE
    other = descriptor(lin[::-1])
    with pytest.raises(M.Refused, match="the descriptor's linears are not"):
        M.register(ROOT, C.ROLE, hashlib.sha512(other).hexdigest(), other, ck, KEY, pinned="ab" * 20, now="t")
