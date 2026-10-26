"""The torch loop as an integration of Verity's (`verity.ml.windows.interface`): its model computes its template's function,
its written window is the one core's registration check derives (and a port under another domain is refused), the
verifier's window from its records alone is the loop's, the circuit it registers before running is the one it writes (two
runs of one shape: two roots and two circuits), its
compiler registers its code's digest at its commit, compiles its committed schedule to the circuit it registers and refuses
advice that is not a schedule, its unit
composes into C-Flock's unit statement, a tampering loop's outputs are not its Definition's, and it loads no vLLM."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as IF, rows as CS
from verity_torch_loop import compile as C, loop as L, serve as S

LAW = D.parse_law("stratified:2")


def served(tamper: bool = False) -> tuple[L.TorchLoop, IF.Window]:
    loop = L.TorchLoop(L.Model(), [3, 7], steps=2, tamper=tamper)
    return loop, loop.window(LAW, cap=16)


def test_the_loops_outputs_are_its_templates():
    loop, win = served()
    assert loop.units == 2 * 2 * 2 * L.H and win.window.population == loop.units
    assert win.mismatches() == {}


def test_a_tampering_loops_outputs_are_not():
    _, win = served(tamper=True)
    (tid, bad), = win.mismatches().items()
    assert tid == descriptor_id(L.template(L.H)) and len(bad) == win.window.population


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
    shape = ["--law", "subset:2", "--requests", "3", "--steps", "1", "--cap", "32"]
    assert S.main(["--circuit-only", *shape]) == 0
    before = json.loads(capsys.readouterr().out)["circuit"]
    assert S.main(["--out", str(tmp_path), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["circuit"] == before == hashlib.sha512((tmp_path / "circuit.json").read_bytes()).hexdigest()
    assert got["mismatched"] == {} and got["population"] == 2 * L.H
    assert S.main(["--out", str(tmp_path / "t"), *shape, "--tamper"]) == 0
    tampered = json.loads(capsys.readouterr().out)
    assert tampered["circuit"] == before and list(tampered["mismatched"].values()) == [2 * L.H]


def test_two_runs_of_one_shape_are_two_roots_each_registering_its_own_circuit(tmp_path, capsys):
    shape = ["--law", "subset:2", "--requests", "3", "--steps", "1", "--cap", "32"]
    got = {}
    for run in ("a", "b"):
        assert S.main(["--circuit-only", *shape, "--run-id", run]) == 0
        before = json.loads(capsys.readouterr().out)["circuit"]
        assert S.main(["--out", str(tmp_path / run), *shape, "--run-id", run]) == 0
        got[run] = json.loads(capsys.readouterr().out)
        assert got[run]["circuit"] == before
    assert got["a"]["run"] != got["b"]["run"] and got["a"]["circuit"] != got["b"]["circuit"]


def test_its_compiler_compiles_the_committed_schedule_to_the_circuit_it_registers(tmp_path, capsys):
    shape = ["--law", "subset:2", "--requests", "3", "--steps", "1", "--cap", "32"]
    assert S.main(["--out", str(tmp_path / "w"), "--opening", str(tmp_path / "opening.json"), *shape]) == 0
    capsys.readouterr()
    rec = json.loads((tmp_path / "w" / "registration.json").read_text())["window"]
    data, root = IF.opened_advice(C.compiler(), json.loads((tmp_path / "opening.json").read_text()))
    assert rec["compiler"] == C.compiler() and rec["advice"] == {"root": root, "bits": 8 * len(data)}
    compile_ = ["--law", "subset:2", "--cap", "32", "--out", str(tmp_path / "c.json"), "--advice"]
    (tmp_path / "advice.bin").write_bytes(data)
    assert C.main([*compile_, str(tmp_path / "advice.bin")]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"bits": 8 * len(data)}
    theirs, mine = (json.loads(p.read_text()) for p in (tmp_path / "c.json", tmp_path / "w" / "circuit.json"))
    assert theirs["run"] == theirs["context"] == "" and IF.circuit_of(theirs) == IF.circuit_of(mine)
    (tmp_path / "other.bin").write_bytes(C.schedule_bytes([3, 7], 1))
    assert C.main([*compile_, str(tmp_path / "other.bin")]) == 0
    assert IF.circuit_of(json.loads((tmp_path / "c.json").read_text())) != IF.circuit_of(mine)


def test_its_compiler_registers_its_code_s_digest_at_its_commit_and_the_verifier_s_entry_names_that_code(capsys):
    c = C.compiler()
    assert c == {"name": C.NAME, "digest": IF.compiler_digest(C.NAME, C.REPO, C.CODE), "commit": IF.commit_of(C.REPO)}
    assert IF.advice_binding(c, "ab" * 64, 8)["compiler"] == c
    assert C.main(["--table", "py", "-m", "verity_torch_loop.compile", "--law", "subset:2"]) == 0
    assert json.loads(capsys.readouterr().out) == {C.NAME: {
        "digest": c["digest"], "argv": ["py", "-m", "verity_torch_loop.compile", "--law", "subset:2"],
        "code": list(C.CODE), "pythonpath": list(C.PYTHONPATH)}}
    assert all((C.REPO / p).is_dir() for p in (*C.CODE, *C.PYTHONPATH))


@pytest.mark.parametrize("advice", [b"", b"not json", b'{"requests": [3], "steps": 1}', b'{"requests":[16],"steps":1}',
                                    b'{"requests":[3],"steps":0}', b'{"requests":[],"steps":1}', b'{"requests":[3],"steps":1,"x":0}'])
def test_its_compiler_refuses_advice_that_is_not_a_schedule(tmp_path, capsys, advice):
    (tmp_path / "a.bin").write_bytes(advice)
    assert C.main(["--advice", str(tmp_path / "a.bin"), "--out", str(tmp_path / "c.json")]) == C.REFUSED
    assert "not" in capsys.readouterr().err and not (tmp_path / "c.json").exists()


@pytest.mark.slow
def test_its_unit_composes_into_c_flocks_unit_statement(tmp_path):
    """The statement a drawn member is proved under, from the circuit record's rows (a5's composer does the same): C-Flock
    refuses a unit none of whose rows is longer than one SHA-512 block, which only the end to end would otherwise find. ~12 s,
    ~1.1 GB."""
    from verity_flock.class_statement import UnitStager

    _, win = served()
    (tid, st), = win.circuit_record()["statements"].items()
    s = UnitStager(L.template(L.H), "unit/window", [(p, sc, int(b), list(ps)) for p, sc, b, ps in st["in_rows"]],
                   [(sc, tuple(ix)) for sc, ix in st["out_rows"]], circuit=tmp_path / "unit.circuit",
                   bind={"partition": "ab" * 64, "program": "cd" * 64}, reads=st["reads"])
    assert tid == descriptor_id(L.template(L.H)) and len(s.class_sha512) == 128 and (tmp_path / "unit.circuit").stat().st_size


def test_the_loop_loads_no_vllm():
    code = ("import sys, verity_torch_loop.checkpoint, verity_torch_loop.compile, verity_torch_loop.loop, verity_torch_loop.serve; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('verity_vllm', 'vllm')))")
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=Path(L.__file__).parents[1])
    assert got.stdout.strip() == "[]"
