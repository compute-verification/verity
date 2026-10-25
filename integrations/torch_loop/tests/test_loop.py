"""The torch loop on the serving interface: its model computes its template's function, its window is the one core's
verifier derives (and a port under another domain is refused), and a tampering loop's outputs are not its Definition's."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.serving import interface as I, rows as CS
from verity_torch_loop import loop as L

RUN = "cd" * 64 + "/rank0"


def served(tamper: bool = False, h: int = 64) -> tuple[L.TorchLoop, I.ServingWindow]:
    loop = L.TorchLoop(L.Model(h=h), [3, 7], steps=2, tamper=tamper)
    win = I.ServingWindow(loop.workload(), RUN, D.parse_law("stratified:2"), cap=16)
    loop.serve(win)
    return loop, win


def test_the_loops_outputs_are_its_templates():
    loop, win = served()
    assert loop.units == 2 * 2 * 2 * 64 and win.window.population == loop.units
    assert win.mismatches() == {}


def test_a_tampering_loops_outputs_are_not():
    _, win = served(tamper=True)
    (tid, bad), = win.mismatches().items()
    assert len(bad) == win.window.population


def test_core_accepts_the_written_window(tmp_path):
    _, win = served()
    c = win.close(salt_key=bytes(32))
    win.write(c, tmp_path, {"of": {"stack": "torch-loop"}}, registrar=R)
    rec = json.loads((tmp_path / "registration.json").read_text())
    w = win.window
    layer = CS.leaf_layer({m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(w.members)})
    v = R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer)
    assert v.ok, v.to_json()
    assert rec["window"]["kind"] == "served" and rec["window"]["workload"] == "torch-loop/mlp-w8a8-h64"
    bad = {**rec, "roots": [{**rec["roots"][0], "domain": "00" * 64}, *rec["roots"][1:]]}
    assert not R.check(bad, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer).ok
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["members"][0]["rows"][0][2:6] == ["r0", 0, 0, "model.layers.0.mlp.up_proj"]


def test_the_loop_loads_no_vllm():
    code = "import sys, verity_torch_loop.loop; print(sorted(m for m in sys.modules if m.split('.')[0] in ('verity_vllm', 'vllm')))"
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=Path(L.__file__).parents[1])
    assert got.stdout.strip() == "[]"
