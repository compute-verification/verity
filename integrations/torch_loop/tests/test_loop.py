"""The torch loop as an integration of Verity's (`verity.ml.windows.interface`): its model computes its template's function,
its written window is the one core's registration check derives (and a port under another domain is refused), the
verifier's window from its records alone is the loop's, the circuit it registers before running is the one it writes (two
runs of one shape: two roots and two circuits), its
compiler registers its code's digest at its commit, compiles its committed schedule to the circuit it registers and refuses
advice that is not a schedule, its unit
composes into C-Flock's unit statement, a tampering loop's outputs are not its Definition's, and it loads no vLLM. With
registered weights its record leaves half its input bits free, every unit's weight rows open the pinned root at the rows
the verifier derives from the record (Lean's registered reads, mirrored on the host), and other weights served under the
registered salts open none. Its activation rows are committed per window, each the rows its units recorded, under domains
binding the run root, named in the registration (so fixed before the draw) and taking nothing from the free inputs. Its
wired circuit is its Definitions, every activation wired (a tampered one its quantize unit's refusal), and with registered
weights leaves only the requests' own rows free."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as IF, rows as CS, wiring as WI
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


def test_under_a_work_law_it_registers_its_work_table_and_its_output_units_are_the_last_linears(tmp_path, capsys):
    shape = ["--law", "work:4", "--requests", "3,7", "--steps", "2", "--cap", "8"]
    assert S.main(["--circuit-only", *shape]) == 0
    before = json.loads(capsys.readouterr().out)
    assert S.main(["--out", str(tmp_path), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["circuit"] == before["circuit"]
    assert json.loads((tmp_path / "circuit.json").read_text())["work_table"] == {descriptor_id(L.template(L.H)): 1}
    last = sorted(u for m in json.loads((tmp_path / "index.json").read_text())["members"] for lo, hi, lab in m["labels"]
                  if lab["layer"] == 1 for u in range(lo, hi))
    assert [u for lo, hi in before["outputs"] for u in range(lo, hi)] == last and len(last) == got["population"] // 2


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


WEIGHTS = L.commit_weights(L.Model())


def registered(tmp_path: Path, tamper: bool = False, tamper_weights: bool = False) -> tuple[IF.Window, CS.Committed, dict]:
    loop = L.TorchLoop(L.Model(), [3, 7], steps=2, tamper=tamper, registered=True, tamper_weights=tamper_weights)
    win = loop.window(D.parse_law("subset:2"), cap=8, weights=WEIGHTS)
    c = win.close(salt_key=bytes(32))
    win.write(c, tmp_path, {"loop": "torch-loop"}, registrar=R)
    return win, c, json.loads((tmp_path / "window.json").read_text())


def opened_reads(c: CS.Committed, k: int, header: dict, file: dict, weights: dict | None = None) -> str | None:
    """Lean's registered reads of member ``k`` (`Flock.Registered.checkPort`), on the host: each table row's ``b || c`` opens
    the pinned root at its stated position, and each instance's ref names the row the verifier's file reads for its unit.
    None when they accept, else why not."""
    m = c.window.members[k]
    for j, (p, port) in enumerate((p, e) for p, e in file["ports"].items()):
        entry, bc, value = header[p], c.bc[m.port(p)], (weights or WEIGHTS)[port["value"]]
        assert (port["domain"], port["root"]) == (value.domain.domain_id.hex(), value.root.hex())
        for r, (pos, path) in enumerate(zip(entry["positions"], entry["paths"])):
            if not REG.verify_read(value.domain, value.root, pos, bc[r * CS.BC_BYTES:(r + 1) * CS.BC_BYTES], bytes.fromhex(path),
                                   port["row_schema"]):
                return f"{p}: table row {r} does not open the registered root at position {pos}"
        col = [q for q, _ in m.rows].index(p)
        if [entry["positions"][int(i)] for i in c.refs[k][:, col]] != port["reads"]:
            return f"{p}: an instance reads another position than the verifier's program"
    return None


def test_registered_weights_leave_half_the_inputs_free_and_open_the_pinned_root(tmp_path):
    win, c, written = registered(tmp_path)
    assert win.mismatches() == {}
    doc = json.loads((tmp_path / "circuit.json").read_text())
    assert IF.input_count(doc) == {"units": 1024, "inputs": 2_162_688, "wired": 0, "registered": 1_081_344, "ingress": 0,
                                   "free": 1_081_344}
    rec = json.loads((tmp_path / "registration.json").read_text())
    w = win.window
    layer = CS.leaf_layer({m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(w.members)})
    assert R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer).ok
    mine, doc = IF.of_records(tmp_path, D.parse_law("subset:2"), IF.resolve)
    assert mine.members == w.members
    pin = L.held_pin(L.pin(L.Model(), WEIGHTS))
    files = IF.registered_reads(doc, mine, pin, registered=REG)
    assert len(files) == len(mine.members) == 128
    for k, m in enumerate(mine.members):
        assert opened_reads(c, k, written["members"][k]["registered"], files[m.key]) is None
    m = mine.members[-1]
    layer_of, n0 = divmod(m.base % 256, 128)
    assert files[m.key]["ports"]["wrow"]["reads"] == [128 * layer_of + n0 + i for i in range(8)]
    other = {**files[m.key], "ports": {**files[m.key]["ports"], "wrow": {**files[m.key]["ports"]["wrow"],
                                                                         "reads": [0] * 8}}}
    assert "another position" in opened_reads(c, len(mine.members) - 1, written["members"][-1]["registered"], other)


def test_other_weights_under_the_registered_salts_open_no_pinned_root(tmp_path):
    """The prover serves other weights (its outputs theirs, so every unit's Definition holds) under the registered value's
    salts: Lean's registered reads refuse every member, its rows opening no pinned root."""
    win, c, written = registered(tmp_path, tamper_weights=True)
    assert win.mismatches() == {}
    mine, doc = IF.of_records(tmp_path, D.parse_law("subset:2"), IF.resolve)
    files = IF.registered_reads(doc, mine, L.held_pin(L.pin(L.Model(), WEIGHTS)), registered=REG)
    why = [opened_reads(c, k, written["members"][k]["registered"], files[m.key]) for k, m in enumerate(mine.members)]
    assert all(w is not None and "does not open" in w for w in why)


def test_a_tampering_loop_with_registered_weights_is_still_its_definitions_refusal(tmp_path):
    win, c, written = registered(tmp_path, tamper=True)
    assert len(win.mismatches()[descriptor_id(L.template(L.H))]) == 1024
    mine, doc = IF.of_records(tmp_path, D.parse_law("subset:2"), IF.resolve)
    files = IF.registered_reads(doc, mine, L.held_pin(L.pin(L.Model(), WEIGHTS)), registered=REG)
    assert opened_reads(c, 0, written["members"][0]["registered"], files[mine.members[0].key]) is None


def test_its_activations_are_the_units_rows_committed_under_the_run_and_take_nothing_from_the_free_inputs():
    loop, win = served()
    with pytest.raises(ValueError, match="has not served"):
        L.TorchLoop(L.Model(), [3]).commit_activations()
    acts = loop.commit_activations()
    tid, h, calls = descriptor_id(L.template(L.H)), L.H, 2 * 2 * 2
    recorded = win._recorded[tid]
    assert [len(recorded)] * 2 == [acts[v].leaves for v in L.ACTIVATIONS] == [calls] * 2
    for g, v in enumerate(L.ACTIVATIONS):
        assert all(b.n == h and (b.words[g] == b.words[g][0]).all() and tuple(b.words[g][0]) == acts[v].rows[i]
                   for i, b in enumerate(recorded))
        r = acts[v]
        assert r.words == dict(IF.layout(L.template(h)).rows)[v] and r.window == loop.run_root
        assert r.domain.domain_id == REG.domain(L.Model().name, v, r.words, calls, window=loop.run_root).domain_id
        assert all(REG.verify_read(r.domain, r.root, i, r.bcs[i], r.path(i), REG.SCHEMA_V2) for i in range(calls))
    other = L.TorchLoop(L.Model(), [3, 7], steps=2, run_id="other")
    other.window(LAW, cap=16)
    theirs = other.commit_activations()
    assert all(theirs[v].rows == acts[v].rows and theirs[v].domain.domain_id != acts[v].domain.domain_id for v in L.ACTIVATIONS)
    n = IF.input_count(win.circuit_record())
    assert n["free"] == n["inputs"] == 2_162_688


def test_serving_names_its_activations_in_the_registration_and_keeps_their_salts_private(tmp_path, capsys):
    shape = ["--law", "subset:2", "--requests", "3", "--steps", "1", "--cap", "32"]
    assert S.main(["--out", str(tmp_path), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    rec = json.loads((tmp_path / "registration.json").read_text())
    acts = rec["window"]["activations"]
    assert acts["program"] == L.Model().name and {v: e["root"] for v, e in acts["values"].items()} == got["activations"]
    assert [acts["values"][v]["leaves"] for v in L.ACTIVATIONS] == [2, 2] and R.digest(rec) == got["registration"]
    private = tmp_path / S.ACTIVATIONS_PRIVATE
    opening = json.loads(private.read_text())
    assert private.stat().st_mode & 0o077 == 0 and opening["window"] == got["run"] == rec["window"]["run"]
    assert all(len(opening["salts"][v]) == 2 for v in L.ACTIVATIONS)
    w = IF.of_records(tmp_path, D.parse_law("subset:2"), IF.resolve)[0]
    layer = CS.leaf_layer({m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(w.members)})
    assert R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer).ok


def test_the_weights_pin_names_the_role_and_its_domains_are_their_programs(tmp_path, capsys):
    shape = ["--law", "subset:2", "--requests", "3", "--steps", "1", "--cap", "32"]
    assert S.main(["--commit-weights", str(tmp_path / "opening.json"), "--pin", str(tmp_path / "pin.json")]) == 0
    roots = json.loads(capsys.readouterr().out)
    pin = json.loads((tmp_path / "pin.json").read_text())
    assert pin["checkpoint_role"] == L.ROLE and {v: e["root"] for v, e in pin["values"].items()} == roots
    assert L.held_pin(pin) == pin["values"] and pin["values"]["w"]["leaves"] == 256 and pin["values"]["sw"]["words"] == 2
    with pytest.raises(ValueError, match="another value or domain"):
        L.held_pin({**pin, "program": "torch-loop/other"})
    assert {v: r.root.hex() for v, r in L.opened(L.Model(), json.loads((tmp_path / "opening.json").read_text())).items()} == roots
    assert S.main(["--circuit-only", "--registered", *shape]) == 0
    before = json.loads(capsys.readouterr().out)["circuit"]
    assert S.main(["--out", str(tmp_path / "w"), "--registered", "--weights", str(tmp_path / "opening.json"), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["circuit"] == before and got["mismatched"] == {}
    assert S.main(["--circuit-only", *shape]) == 0
    assert json.loads(capsys.readouterr().out)["circuit"] != before
    (tmp_path / "advice.bin").write_bytes(C.schedule_bytes([3], 1))
    assert C.main(["--law", "subset:2", "--cap", "32", "--registered", "--out", str(tmp_path / "c.json"), "--advice",
                   str(tmp_path / "advice.bin")]) == 0
    assert IF.circuit_of(json.loads((tmp_path / "c.json").read_text())) == IF.circuit_of(
        json.loads((tmp_path / "w" / "circuit.json").read_text()))


def wired(tmp_path: Path, registered: bool = False, **kw) -> tuple[L.TorchLoop, IF.Window, CS.Committed]:
    loop = L.TorchLoop(L.Model(), [3, 7], steps=2, wired=True, registered=registered, **kw)
    win = loop.window(D.parse_law("stratified:3"), cap=1, weights=WIRED_WEIGHTS if registered else None)
    c = win.close(salt_key=bytes(32))
    win.write(c, tmp_path, {"loop": "torch-loop"}, registrar=R)
    return loop, win, c


WIRED_WEIGHTS = L.commit_weights(L.Model(), wired=True)


def test_the_wired_loop_is_its_definitions_each_quantize_unit_reading_its_linears_row_and_each_linear_its_codes(tmp_path):
    """Each linear one unit and each ReLU and quantize one, every activation wired: the loop's ``quantize`` is
    ``QuantRowS8_v1``'s, core accepts the window, the wiring check holds on its public bodies, the verifier's members from
    the records are the loop's, and only the requests' own rows (and the weights, unregistered) are free."""
    loop, win, c = wired(tmp_path)
    h, rs = L.H, 2 * 2
    assert [(descriptor_id(fn), n) for fn, n in win.circuit.templates] == [
        (descriptor_id(L.linear_template(h)), 2 * rs), (descriptor_id(L.quant_template(h, True)), rs),
        (descriptor_id(L.quant_template(h, False)), 2)]
    assert loop.units == win.window.population == 14 and win.mismatches() == {}
    w = win.window
    publics = {m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(w.members)}
    rec = json.loads((tmp_path / "registration.json").read_text())
    assert R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=CS.leaf_layer(publics)).ok
    assert WI.check(w.members, publics) == {} and sum(bool(m.wired) for m in w.members) == 14 - 2
    mine, _doc = IF.of_records(tmp_path, D.parse_law("stratified:3"), IF.resolve)
    assert mine.members == w.members
    n = IF.input_count(json.loads((tmp_path / "circuit.json").read_text()))
    request_rows = 2 * 16 * (h // 2 + 2)
    assert n["wired"] == 16 * (6 * h + 6 * (h // 2 + 2)) and n["free"] == request_rows + 8 * 16 * (h * h // 2 + 2 * h)
    coordinates, _ = served()
    assert loop.outputs.keys() == coordinates.outputs.keys()
    assert all(torch.equal(loop.outputs[k], coordinates.outputs[k]) for k in loop.outputs)


def test_a_tampered_activation_is_its_quantize_units_refusal_and_a_linear_reading_another_row_the_wiring_checks():
    """Codes flipped after each quantize: every linear still holds (it computes on the codes it reads), and every quantize unit
    is not its Definition, which its proof refuses when drawn. A linear that reads other codes than its writer's doesn't
    commit, and a committed string that isn't its writer's is the wiring check's refusal."""
    loop = L.TorchLoop(L.Model(), [3, 7], steps=2, wired=True, tamper_activations=True)
    win = loop.window(D.parse_law("stratified:3"), cap=1)
    assert win.mismatches() == {descriptor_id(L.quant_template(L.H, True)): [0, 1, 2, 3],
                                descriptor_id(L.quant_template(L.H, False)): [0, 1]}
    with pytest.raises(ValueError, match="give wired"):
        L.TorchLoop(L.Model(), [3], tamper_activations=True)
    honest = L.TorchLoop(L.Model(), [3, 7], steps=2, wired=True)
    bad = honest.window(D.parse_law("stratified:3"), cap=1)
    bad._recorded[descriptor_id(L.linear_template(L.H))][-1].words[0][0, 0] ^= 1
    with pytest.raises(ValueError, match="is not the o0 row"):
        bad.close(salt_key=bytes(32))


def test_registered_weights_in_the_wired_loop_leave_only_the_requests_rows_free_and_open_the_pinned_root(tmp_path):
    """The weights one row a linear: each linear unit reads its layer's row, so the record leaves the two requests' input rows
    free and nothing else, each member's rows open the pinned root (Lean's registered reads, on the host), the wiring check
    reads the shared-row members' tables at their refs, and other weights under the registered salts open no pinned root."""
    loop, win, c = wired(tmp_path, registered=True)
    assert win.mismatches() == {} and {v: (r.leaves, r.words) for v, r in WIRED_WEIGHTS.items()} == {
        "w": (2, L.H * L.H // 2), "sw": (2, 2 * L.H)}
    n = IF.input_count(json.loads((tmp_path / "circuit.json").read_text()))
    assert n["free"] == 2 * 16 * (L.H // 2 + 2) == 2112 and n["registered"] == 8 * 16 * (L.H * L.H // 2 + 2 * L.H)
    mine, doc = IF.of_records(tmp_path, D.parse_law("stratified:3"), IF.resolve)
    written = json.loads((tmp_path / "window.json").read_text())
    files = IF.registered_reads(doc, mine, L.held_pin(L.pin(L.Model(), WIRED_WEIGHTS)), registered=REG)
    assert len(files) == 8
    for k, m in enumerate(mine.members):
        if m.key in files:
            assert opened_reads(c, k, written["members"][k]["registered"], files[m.key], WIRED_WEIGHTS) is None
            assert files[m.key]["ports"]["w"]["reads"] == [(m.base // 4)]
    publics = {m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    assert WI.check(mine.members, publics) == {}
    op = L.opening(L.Model(), WIRED_WEIGHTS)
    assert op["rows"] == "linear" and all(r.root == WIRED_WEIGHTS[v].root for v, r in L.opened(L.Model(), op).items())
    _, other, oc = wired(tmp_path / "other", registered=True, tamper_weights=True)
    assert other.mismatches() == {}
    written = json.loads((tmp_path / "other" / "window.json").read_text())
    k = next(k for k, m in enumerate(other.window.members) if m.key in files)
    assert "does not open" in opened_reads(oc, k, written["members"][k]["registered"], files[other.window.members[k].key],
                                           WIRED_WEIGHTS)


def test_serving_the_wired_loop_registers_the_circuit_it_writes_and_its_compiler_compiles_it(tmp_path, capsys):
    shape = ["--law", "stratified:3", "--requests", "3", "--steps", "2", "--cap", "1", "--wired"]
    assert S.main(["--circuit-only", *shape]) == 0
    before = json.loads(capsys.readouterr().out)["circuit"]
    assert S.main(["--out", str(tmp_path / "w"), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["circuit"] == before and got["mismatched"] == {} and got["population"] == 4 + 2 + 1
    rec = json.loads((tmp_path / "w" / "registration.json").read_text())
    assert [e["leaves"] for e in rec["window"]["activations"]["values"].values()] == [1, 1]
    assert S.main(["--out", str(tmp_path / "t"), *shape, "--tamper-activations"]) == 0
    assert sorted(json.loads(capsys.readouterr().out)["mismatched"].values()) == [1, 2]
    (tmp_path / "advice.bin").write_bytes(C.schedule_bytes([3], 2))
    assert C.main(["--law", "stratified:3", "--cap", "1", "--wired", "--out", str(tmp_path / "c.json"), "--advice",
                   str(tmp_path / "advice.bin")]) == 0
    capsys.readouterr()
    assert IF.circuit_of(json.loads((tmp_path / "c.json").read_text())) == IF.circuit_of(
        json.loads((tmp_path / "w" / "circuit.json").read_text()))
    assert S.main(["--circuit-only", *shape[:-1]]) == 0
    assert json.loads(capsys.readouterr().out)["circuit"] != before


@pytest.mark.slow
@pytest.mark.parametrize("relu", [True, False])
def test_the_wired_loops_quantize_unit_composes_into_c_flocks_unit_statement(tmp_path, relu):
    """As `test_its_unit_composes_into_c_flocks_unit_statement`, for the wired circuit's quantize units, whose one long row is
    their 256-byte input (C-Flock refuses a unit none of whose rows is longer than one SHA-512 block). ~15 s, ~2.6 GB each.
    The linear (a 16 KiB weight row) composes too, in ~85 s and ~6 GB, which only the end to end needs."""
    from verity_flock.class_statement import UnitStager

    win = L.TorchLoop(L.Model(), [3, 7], steps=2, wired=True).window(D.parse_law("stratified:3"), cap=1)
    tid = descriptor_id(L.quant_template(L.H, relu))
    st = win.circuit_record()["statements"][tid]
    s = UnitStager(L.quant_template(L.H, relu), "unit/window", [(p, sc, int(b), list(ps)) for p, sc, b, ps in st["in_rows"]],
                   [(sc, tuple(ix)) for sc, ix in st["out_rows"]], circuit=tmp_path / "unit.circuit",
                   bind={"partition": "ab" * 64, "program": "cd" * 64}, reads=st["reads"])
    assert len(s.class_sha512) == 128 and (tmp_path / "unit.circuit").stat().st_size
