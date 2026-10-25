"""The torch loop as an integration of Verity's (`verity.ml.windows.interface`): its model computes its template's function,
its written window is the one core's registration check derives (and a port under another domain is refused), the
verifier's window from its records alone is the loop's, the circuit it registers before running is the one it writes, a
tampering loop's outputs are not its Definition's, and it loads no vLLM."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as IF, rows as CS
from verity_torch_loop import loop as L, serve as S

LAW = D.parse_law("stratified:2")


def served(tamper: bool = False, h: int = 64) -> tuple[L.TorchLoop, IF.Window]:
    loop = L.TorchLoop(L.Model(h=h), [3, 7], steps=2, tamper=tamper)
    return loop, loop.window(LAW, cap=16)


def test_the_loops_outputs_are_its_templates():
    loop, win = served()
    assert loop.units == 2 * 2 * 2 * 64 and win.window.population == loop.units
    assert win.mismatches() == {}


def test_a_tampering_loops_outputs_are_not():
    _, win = served(tamper=True)
    (tid, bad), = win.mismatches().items()
    assert tid == descriptor_id(L.template(64)) and len(bad) == win.window.population


def test_core_accepts_the_written_window_and_the_verifier_rebuilds_it_from_the_records(tmp_path):
    loop, win = served()
    win.write(win.close(salt_key=bytes(32)), tmp_path, {"loop": "torch-loop"}, registrar=R)
    rec = json.loads((tmp_path / "registration.json").read_text())
    w = win.window
    layer = CS.leaf_layer({m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(w.members)})
    v = R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer)
    assert v.ok, v.to_json()
    assert rec["window"]["kind"] == "served" and rec["window"]["run"] == loop.run_root and "compiler" not in rec["window"]
    bad = {**rec, "roots": [{**rec["roots"][0], "domain": "00" * 64}, *rec["roots"][1:]]}
    assert not R.check(bad, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer).ok
    mine, doc = IF.of_records(tmp_path, LAW, IF.resolve)
    assert mine.partition_digest == w.partition_digest and [m.key for m in mine.members] == [m.key for m in w.members]
    labels = json.loads((tmp_path / "index.json").read_text())
    assert labels["members"][0]["labels"][0][2] == {"request": "r0", "step": 0, "layer": 0, "op": "model.layers.0.mlp.up_proj"}


def test_the_circuit_it_registers_before_running_is_the_one_it_writes(tmp_path, capsys):
    shape = ["--law", "subset:2", "--requests", "3", "--steps", "1", "--h", "64", "--cap", "32"]
    assert S.main(["--circuit-only", *shape]) == 0
    before = json.loads(capsys.readouterr().out)["circuit"]
    assert S.main(["--out", str(tmp_path), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["circuit"] == before == hashlib.sha512((tmp_path / "circuit.json").read_bytes()).hexdigest()
    assert got["mismatched"] == {} and got["population"] == 128
    assert S.main(["--out", str(tmp_path / "t"), *shape, "--tamper"]) == 0
    tampered = json.loads(capsys.readouterr().out)
    assert tampered["circuit"] == before and list(tampered["mismatched"].values()) == [128]


def test_the_loop_loads_no_vllm():
    code = ("import sys, verity_torch_loop.loop, verity_torch_loop.serve; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('verity_vllm', 'vllm')))")
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=Path(L.__file__).parents[1])
    assert got.stdout.strip() == "[]"
