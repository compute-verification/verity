"""The audit from a window's records alone (`records_audit`), without proving, on a window written through the interface
(`verity.ml.windows.interface`) under core's registration:

* the verifier's window is the circuit record's, each template resolved in Verity's catalog, and the record must be the one
  the registration binds; a window with row tables is refused until its records carry their segment roots;
* the compile check runs the registered compiler as the verifier's table names it, on advice that opens the registered root,
  and refuses by name a compiler it doesn't hold, an opening of another root, a compiler that refuses the advice, one whose
  circuit is another or that writes none, and a length other than the registered one or none;
* a compiler registered under a commit runs from a checkout of that commit, not the auditing tree's, and a commit whose code
  doesn't hash to the registered digest is refused;
* the leak bound's count is the circuit's unwired input bits, reported beside the advice's charge and vacuous past it;
* `source` and the negatives refuse what the records don't bind, and `fixed_before_draw` refuses advice re-committed after
  the coins;
* the gate's coins (``--challenger gate:USER@HOST``) reach the challenger's session with the peer and the key record;
* none of it imports `verity_vllm`, and the served compiler's command (``prove_path.py circuit``) refuses advice it can't
  compile and charges the length the registration does;
* that command compiles over its own trace of each Program, never the prover's row dir, so a row dir whose names make
  another circuit under unchanged digests is refused by name and one whose names make the same circuit is ignored."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.windows import interface as IF, rows as CS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "one_stage"))
import records_audit as RA  # noqa: E402

K = 64
COORD = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
TID = descriptor_id(COORD)
LAW = "stratified:2"
RUN = "ab" * 64
COMPILER = {"name": "test/compiler", "digest": "cd" * 64}
ADVICE = bytes(range(40))

FAKE = '''import argparse, json, sys
ap = argparse.ArgumentParser()
for f in ("--record", "--expect", "--advice", "--out", "--say", "--write"):
    ap.add_argument(f)
ap.add_argument("--bits", type=int)
a = ap.parse_args()
data = open(a.advice, "rb").read()
if data.hex() != a.expect:
    sys.stderr.write("advice this compiler can't compile")
    sys.exit(2)
if a.write != "no":
    open(a.out, "w").write(open(a.record).read())
print(a.say if a.say is not None else json.dumps({"bits": 8 * len(data) if a.bits is None else a.bits}))
'''


def _write(d: Path, n: int = 5, cap: int = 4, advice: bool = True, compiler: dict = COMPILER) -> SimpleNamespace:
    """A window of ``n`` coordinates in members of ``cap``, its advice committed under ``compiler`` (or none), written and
    registered through the interface under core's registration."""
    reg = IF.commit_advice(compiler, ADVICE)
    bound = IF.advice_binding(compiler, reg.root.hex(), 8 * len(ADVICE)) if advice else None
    win = IF.Window(IF.Circuit("test", ((COORD, n),)), RUN, D.parse_law(LAW), cap, advice=bound)
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (n, K)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)], {"unit": 0})
    win.write(win.close(salt_key=bytes(32)), d, {"label": "test"}, registrar=R)
    doc = json.loads((d / "circuit.json").read_text())
    return SimpleNamespace(dir=d, doc=doc, rec=json.loads((d / "registration.json").read_text()),
                           opening=IF.advice_opening(ADVICE, reg), bound=bound)


def _compilers(tmp: Path, doc: dict, **kw) -> dict:
    """The verifier's table: `COMPILER` as the fake command, which writes ``doc`` less its run fields for `ADVICE` alone."""
    body = json.dumps({**doc, "run": "", "context": ""}).encode()
    record = tmp / f"record-{hashlib.sha512(body).hexdigest()[:16]}.json"
    (tmp / "compiler.py").write_text(FAKE)
    record.write_bytes(body)
    argv = [sys.executable, str(tmp / "compiler.py"), "--record", str(record), "--expect", ADVICE.hex(),
            *(x for k, v in kw.items() for x in (f"--{k}", str(v)))]
    return {COMPILER["name"]: {"digest": COMPILER["digest"], "argv": argv}}


@pytest.fixture
def written(tmp_path):
    return _write(tmp_path / "w")


def test_the_verifiers_window_is_the_circuit_record_under_verity_s_catalog(written):
    mine, doc, sets = RA.verifier_window(written.dir, D.parse_law(LAW), None)
    assert doc == written.doc and mine.run == RUN and mine.partition_digest == written.rec["partition"]
    assert [(m.template, m.n) for m in mine.members] == [(TID, 4), (TID, 1)]
    assert set(sets) == {TID} and callable(sets[TID][0]) and descriptor_id(sets[TID][1]) == TID
    w = written.rec["window"]
    assert w["circuit"] == hashlib.sha512((written.dir / "circuit.json").read_bytes()).hexdigest()
    assert (w["compiler"], w["advice"]) == (written.bound["compiler"], written.bound["advice"])
    (written.dir / "circuit.json").write_text(json.dumps({**doc, "name": "other"}))
    with pytest.raises(ValueError, match="not the circuit record the registration binds"):
        RA.verifier_window(written.dir, D.parse_law(LAW), None)


def test_the_compile_check_runs_the_registered_compiler_on_the_opened_advice(written, tmp_path):
    w, doc, op = written.rec["window"], written.doc, written.opening
    v, bound = RA.compile_check(doc, w, op, _compilers(tmp_path, doc), tmp_path / "c")
    assert v["ok"] and bound == written.bound and v["detail"]["bits"] == 8 * len(ADVICE)
    assert IF.circuit_of(json.loads((tmp_path / "c" / "compiled.json").read_text())) == IF.circuit_of(doc)
    other = dict(op, salts=[("00" if op["salts"][0][:2] != "00" else "01") + op["salts"][0][2:], *op["salts"][1:]])
    short = {**doc, "templates": [[TID, 4]]}
    for name, (compilers, window, opening), codes in (
            ("a compiler the verifier doesn't hold", ({}, w, op), ["compiler", "code", "compiles", "circuit", "length"]),
            ("another digest", (_compilers(tmp_path, doc), {**w, "compiler": {**COMPILER, "digest": "00" * 64}}, op),
             ["compiler", "advice", "code", "compiles", "circuit", "length"]),
            ("an opening of another root", (_compilers(tmp_path, doc), w, other),
             ["advice", "code", "compiles", "circuit", "length"]),
            ("a compiler refusing the advice", (_compilers(tmp_path, doc, expect="00"), w, op), ["compiles", "circuit", "length"]),
            ("another circuit", (_compilers(tmp_path, short), w, op), ["circuit"]),
            ("another length", (_compilers(tmp_path, doc, bits=8 * len(ADVICE) + 1), w, op), ["length"]),
            ("a length that isn't a whole number", (_compilers(tmp_path, doc, say='{"bits": "320"}'), w, op), ["length"]),
            ("a length that's a boolean", (_compilers(tmp_path, doc, say='{"bits": true}'), w, op), ["length"]),
            ("no length line", (_compilers(tmp_path, doc, say="compiled"), w, op), ["length"])):
        v, bound = RA.compile_check(doc, window, opening, compilers, tmp_path / "c")
        assert (v["codes"], bound) == (codes, None), name
    v, _ = RA.compile_check(doc, w, op, _compilers(tmp_path, doc, expect="00"), tmp_path / "c")
    assert v["detail"]["refused"] and "can't compile" in v["detail"]["stderr"]


def test_a_command_that_exits_0_without_writing_its_circuit_fails_the_circuit_check(written, tmp_path):
    """The command writes inside a fresh directory, so a stale ``compiled.json`` from an earlier compile is never compared."""
    w, doc, op = written.rec["window"], written.doc, written.opening
    assert RA.compile_check(doc, w, op, _compilers(tmp_path, doc), tmp_path / "c")[0]["ok"]
    assert (tmp_path / "c" / "compiled.json").exists()
    v, bound = RA.compile_check(doc, w, op, _compilers(tmp_path, doc, write="no"), tmp_path / "c")
    assert (v["codes"], bound) == (["circuit"], None) and "output" in v["detail"]
    assert not (tmp_path / "c" / "compiled.json").exists()


def test_source_and_the_negatives_refuse_what_the_records_dont_bind(written, tmp_path):
    w, doc = written.rec["window"], written.doc
    compilers = _compilers(tmp_path, doc)
    v, bound = RA.compile_check(doc, w, written.opening, compilers, tmp_path / "c")
    held = {"run": RUN, "circuit": w["circuit"], "compiled": v["ok"], **bound, "compilers": compilers,
            "opening": written.opening, "output": json.loads((tmp_path / "c" / "compiled.json").read_text())}
    assert RA.source(w, held)["ok"]
    assert RA.source(w, {**held, "run": "00" * 64})["codes"] == ["run"]
    assert RA.source(w, {**held, "compiled": False})["codes"] == ["compiled"]
    got = RA.negatives(w, held, doc, tmp_path / "n")
    assert set(got) == {"registration-naming-another-circuit", "registration-naming-another-run",
                        "registration-naming-another-compiler", "registration-naming-another-advice-root",
                        "registration-charging-another-length", "registration-binding-no-compiler-or-advice",
                        "opening-under-another-salt", "circuit-record-not-the-compilers-output"}
    assert all(x["refused"] for x in got.values()), got
    fixed = _write(tmp_path / "fixed", advice=False)
    fw = fixed.rec["window"]
    held = {"run": RUN, "circuit": fw["circuit"], "compiled": True, "compiler": None, "advice": None}
    assert "compiler" not in fw and RA.source(fw, held)["ok"]
    got = RA.negatives(fw, held, fixed.doc, tmp_path / "n")
    assert set(got) == {"registration-naming-another-circuit", "registration-naming-another-run",
                        "registration-binding-a-compiler-for-a-fixed-circuit"} and all(x["refused"] for x in got.values())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           *args], check=True, capture_output=True, text=True).stdout.strip()


def test_a_window_is_compiled_at_its_registered_commit_from_a_tree_at_another(tmp_path, monkeypatch):
    """A registered compiler is (name, digest, commit): the audit compiles from its own checkout of the commit, refused unless
    the digest it computes there is the registered one. The compiler's code at commit A charges the advice's length; at B
    it charges a byte more. From a tree at B, a window registered under A is audited with A's code, and one naming B under
    A's digest, a commit the repository doesn't hold, or a compiler whose table entry names no code is refused."""
    repo = tmp_path / "repo"
    (repo / "fake").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "fake" / "compiler.py").write_text(FAKE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "A")
    a, da = IF.commit_of(repo), IF.compiler_digest(COMPILER["name"], repo, ["fake"])
    (repo / "fake" / "compiler.py").write_text(FAKE.replace("8 * len(data) if", "8 * len(data) + 8 if"))
    _git(repo, "commit", "-qam", "B")
    b, db = IF.commit_of(repo), IF.compiler_digest(COMPILER["name"], repo, ["fake"])
    assert a != b and da != db

    compiler = {"name": COMPILER["name"], "digest": da, "commit": a}
    s = _write(tmp_path / "w", compiler=compiler)
    w, doc, op = s.rec["window"], s.doc, s.opening
    assert w["compiler"] == compiler
    (tmp_path / "record.json").write_text(json.dumps({**doc, "run": "", "context": ""}))
    entry = {"digest": da, "argv": [sys.executable, "fake/compiler.py", "--record", str(tmp_path / "record.json"),
                                    "--expect", ADVICE.hex()], "code": ["fake"], "pythonpath": ["."]}
    table = {COMPILER["name"]: entry}

    v, bound = RA.compile_check(doc, w, op, table, tmp_path / "c", repo)
    assert v["ok"], v
    assert (v["detail"]["at"], v["detail"]["bits"], bound["compiler"]) == (a, 8 * len(ADVICE), compiler)
    held = {"run": RUN, "circuit": w["circuit"], "compiled": True, **bound, "compilers": table, "opening": op, "repo": repo,
            "output": json.loads((tmp_path / "c" / "compiled.json").read_text())}
    assert RA.source(w, held)["ok"]
    got = RA.negatives(w, held, doc, tmp_path / "n")
    assert got["registration-naming-another-commit"] == {"refused": True, "codes": ["compiler"]}
    assert all(x["refused"] for x in got.values()), got

    monkeypatch.setattr(RA, "tree_root", lambda: repo)
    for name, (window, compilers, why) in {
            "the auditing tree's code, at B": ({**w, "compiler": {"name": COMPILER["name"], "digest": da}}, table, "hashes to"),
            "commit B under A's digest": ({**w, "compiler": {**compiler, "commit": b}}, table, "hashes to"),
            "a commit the repository doesn't hold": ({**w, "compiler": {**compiler, "commit": "e" * 40}}, table, "check out"),
            "a table entry naming no code": (w, {COMPILER["name"]: {k: x for k, x in entry.items() if k != "code"}}, "no code"),
    }.items():
        v, bound = RA.compile_check(doc, window, op, compilers, tmp_path / "c", repo)
        assert (v["codes"], bound) == (["code", "compiles", "circuit", "length"], None), name
        assert why in v["detail"]["code"], (name, v["detail"])
    v, _ = RA.compile_check(doc, {**w, "compiler": {**compiler, "digest": db, "commit": b}}, op, table, tmp_path / "c", repo)
    assert v["codes"][:2] == ["compiler", "advice"]
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_the_leak_count_is_the_circuits_unwired_input_bits_and_vacuous_past_the_advices_charge(written):
    """2a's audited window, 8 units of SiluMul_v2{I=8192}, counts its two bf16 rows of 8192 a unit, 2,097,152 bits, against
    1,687 declared advice bits: vacuous. A wired port isn't an input, a shared-row port counts its table's rows, and an
    external port is reported but not subtracted until a layout lists the ingress at its steps."""
    per_unit = sum(16 * w for _p, w in IF.layout(COORD).rows)
    mine, doc = IF.of_records(written.dir, D.parse_law(LAW), IF.resolve)
    got = RA.leak_count(mine.members, doc["external"], 8 * len(ADVICE))
    assert got == {"inputs": 5 * per_unit, "wired": 0, "ingress": 0, "registered": 0, "external": 0,
                   "other_inputs": 5 * per_unit, "advice_bits": 8 * len(ADVICE), "bound": "vacuous"}
    assert RA.leak_count(mine.members, (), 5 * per_unit)["bound"] == "within the charge"
    silu = [CS.Member("SiluMul_v2{I=8192}", (("g", 8192), ("u", 8192)), (("o0", 8192),), 8)]
    assert RA.leak_count(silu, (), 1687)["other_inputs"] == 2_097_152 and RA.leak_count(silu, (), 1687)["bound"] == "vacuous"
    m0, m1 = mine.members
    x = 16 * dict(m0.rows)["x"]
    got = RA.leak_count([replace(m0, wired=(("x", "o0", (0, 1, 2, 3)),)), replace(m1, tables=(("x", 3),))], [f"{TID}/wrow"], 0)
    assert (got["wired"], got["inputs"]) == (4 * x, 5 * per_unit + 2 * x)
    assert got["other_inputs"] == got["inputs"] - 4 * x and got["external"] == 5 * x


def test_a_window_with_row_tables_is_refused_until_its_records_carry_their_segment_roots(written, monkeypatch):
    mine, doc = IF.of_records(written.dir, D.parse_law(LAW), IF.resolve)
    assert not mine.row_tables
    monkeypatch.setattr(IF, "of_records", lambda d, law, resolve: (SimpleNamespace(row_tables=("t0", "t1")), doc))
    with pytest.raises(ValueError, match="2 row tables.*segment roots.*refused until they do"):
        RA.verifier_window(written.dir, D.parse_law(LAW), None)


def test_the_audit_needs_the_run_root_the_run_published(written, capsys):
    base = ["audit", "--window", str(written.dir), "--law", LAW, "--circuit", "00" * 64, "--m0-pythonpath", ".", "--bin", "b",
            "--lean", "l", "--firewall", "f", "--out", "o"]
    with pytest.raises(SystemExit) as e:
        RA.main(base)
    assert e.value.code == 2 and "--run" in capsys.readouterr().err


def test_the_advice_is_fixed_before_the_draw_and_refused_re_committed_after_it(written):
    import a5 as A5

    rec, held = written.rec, {**written.bound, "opening": written.opening, "run": RUN, "circuit": written.rec["window"]["circuit"],
                              "compiled": True}
    receipt = {"registration": R.digest(rec), "verifier": A5.VERIFIER, "received": 1}
    assert RA.fixed_before_draw(receipt, rec, written.bound)["ok"]
    assert RA.fixed_before_draw({**receipt, "registration": "00" * 64}, rec, written.bound)["codes"] == ["after-the-draw"]
    assert RA.fixed_before_draw(receipt, rec, written.bound, "challenger")["codes"] == ["after-the-draw"]
    assert RA.fixed_before_draw(receipt, rec, None)["codes"] == ["compiler", "advice"]
    late = RA.after_draw(receipt, rec, held)
    assert late == {"refused": True, "codes": ["after-the-draw"], "before_draw_checks_pass": True}


def test_the_gates_coins_take_the_peer_and_key_record_to_the_challengers_session(written, tmp_path, monkeypatch):
    """``--challenger gate:USER@HOST`` hands ``--gate-key``, ``--peer`` and ``--challenger-key`` to `a5_coins.start`, as
    `prove_path.py audit` does; the command refuses a gate without the peer or the key record, and any other challenger."""
    import a5_coins as AC

    args = ["audit", "--window", str(written.dir), "--law", LAW, "--circuit", written.rec["window"]["circuit"], "--run", RUN,
            "--port", "10000",
            "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "out"), "--main", "main.txt"]
    for bad in (["--peer", "vy-nebius-1"], ["--challenger-key", "key.json"]):
        with pytest.raises(SystemExit, match="2"):
            RA.main([*args, "--challenger", "gate:research@gate", *bad])
    with pytest.raises(SystemExit, match="2"):
        RA.main([*args, "--challenger", "remote"])

    class Opened(Exception):
        pass

    seen = {}

    def start(challenger, root, ranks, dest, main, **kw):
        seen.update(challenger=challenger, root=root, ranks=ranks, main=main, **kw)
        raise Opened

    monkeypatch.setattr(AC, "start", start)
    with pytest.raises(Opened):
        RA.main([*args, "--challenger", "gate:research@gate", "--peer", "vy-nebius-1", "--challenger-key", "key.json",
                 "--gate-key", "gate.key"])
    assert seen == {"challenger": "gate:research@gate", "root": RUN, "ranks": [0], "main": "main.txt", "ssh_key": "gate.key",
                    "peer": "vy-nebius-1", "key_record": "key.json"}


def test_the_audit_from_the_records_imports_no_vllm(written, tmp_path):
    """The verifier's window, the compile check, `source`, the negatives and the draw-order check, in a fresh process: no
    `verity_vllm` module is loaded."""
    compilers = tmp_path / "compilers.json"
    compilers.write_text(json.dumps(_compilers(tmp_path, written.doc)))
    (tmp_path / "opening.json").write_text(json.dumps(written.opening))
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(HERE.parent)!r})
import records_audit as RA
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
import a5 as A5
d, t = Path({str(written.dir)!r}), Path({str(tmp_path)!r})
mine, doc, sets = RA.verifier_window(d, D.parse_law({LAW!r}), None)
rec = json.loads((d / "registration.json").read_text())
compilers, opening = json.loads((t / "compilers.json").read_text()), json.loads((t / "opening.json").read_text())
v, bound = RA.compile_check(doc, rec["window"], opening, compilers, t / "c")
held = {{"run": mine.run, "circuit": rec["window"]["circuit"], "compiled": v["ok"], **bound, "compilers": compilers,
        "opening": opening, "output": json.loads((t / "c" / "compiled.json").read_text())}}
assert v["ok"] and RA.source(rec["window"], held)["ok"]
assert all(x["refused"] for x in RA.negatives(rec["window"], held, doc, t / "n").values())
receipt = {{"registration": R.digest(rec), "verifier": A5.VERIFIER}}
assert RA.fixed_before_draw(receipt, rec, bound)["ok"] and RA.after_draw(receipt, rec, held)["refused"]
print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] == "verity_vllm")))
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    assert json.loads(p.stdout.strip().splitlines()[-1]) == []


def test_the_served_compilers_command_charges_the_registered_length_and_refuses_what_it_cant_compile(tmp_path, monkeypatch, capsys):
    """``prove_path.py circuit``, the served window's registered compiler as `records_audit.compile_check` runs it, on the
    synthetic serve-openai run of `test_prove_path` (its I2 compile stubbed, and the window over its Programs a coordinate
    window): it writes a circuit record the audit reads (`interface.window_of_record`) under no run and prints the length the
    registration charges, and exits `records_audit.REFUSED` on advice it can't compile."""
    import prove_path as PP
    from test_prove_path import _served
    from verity_vllm.pipeline import serving_rows as SRW

    s = _served(tmp_path, monkeypatch)
    loaded = SimpleNamespace(progs={"r0": {}}, dirs={"r0": "d"}, arrivals={"r0": 0})
    monkeypatch.setattr(PP.I2Compiler, "load", lambda self, rank, world, requests: loaded)
    window = IF.Window(IF.Circuit("test", ((COORD, 5),)), "", D.parse_law(LAW), 4).window
    monkeypatch.setattr(SRW, "population", lambda progs, partition, run, dirs, **kw: (window, []) if run == "" else None)
    (tmp_path / "part.json").write_text(json.dumps({"templates": [TID], "law": LAW}))
    (tmp_path / "advice.bin").write_bytes(s.data)
    base = ["circuit", "--serve-run", str(s.run), "--window", "0", "--partition", str(tmp_path / "part.json"), "--out",
            str(tmp_path / "out.json")]
    capsys.readouterr()
    assert PP.main([*base, "--advice", str(tmp_path / "advice.bin")]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {"bits": s.bound["advice"]["bits"]}
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["format"] == IF.CIRCUIT and (out["run"], out["context"]) == ("", "") and set(out) == set(IF.circuit_record(
        window, "x", IF.templates_of(window)))
    assert IF.window_of_record(out, D.parse_law(LAW), IF.resolve).partition_digest == window.partition_digest
    (tmp_path / "bad.bin").write_bytes(b"\xff" * 7)
    assert PP.main([*base, "--advice", str(tmp_path / "bad.bin")]) == RA.REFUSED
    assert "prove_path.py circuit:" in capsys.readouterr().err


def test_a_row_dir_whose_names_are_permuted_under_its_digests_is_refused_or_ignored_by_the_compile_check(tmp_path, monkeypatch):
    """No Program digest covers a Build's row names or its weights fields' names (nor its ``supplied``, correspondence or
    ``prefix_cache.json``: these TINY Builds carry none, so their names are their correspondence), and the served compiler's
    command reads none of them from the prover: it takes no row dir and compiles over its own trace of each Program I2
    compiled (`prove_path.own_builds`; the trace `test_prove_path._tiny_served`'s stub). Against a prover's row dir permuted
    under unchanged digests (each Build still I2's Program, `build_of`), the compile check refuses by name (``circuit``) the
    circuit.json whose names make another circuit (a layer-1 SiluMul row and layer 0's down_proj swap names: five layer-0
    rows), and ignores names that make the same one (layers 0 and 1's SiluMul rows swapped, which moves only the prover's own
    reads; two weights fields swapped): the verifier's record is its own trace's in each case."""
    import prove_path as PP
    from test_prove_path import LAYER0, _circuit, _row_dir, _tiny_served

    s = _tiny_served(tmp_path, monkeypatch)
    (tmp_path / "part.json").write_text(json.dumps(LAYER0))
    (tmp_path / "advice.bin").write_bytes(s.data)
    out = tmp_path / "mine" / "circuit.json"
    out.parent.mkdir()
    assert PP.main(["circuit", "--serve-run", str(s.run), "--window", "0", "--partition", str(tmp_path / "part.json"),
                    "--advice", str(tmp_path / "advice.bin"), "--out", str(out), "--vllm-python", "pinned-python"]) == 0
    assert [py for py, _ in s.traced] == ["pinned-python"] * len(s.units) and list(out.parent.iterdir()) == [out]
    mine = json.loads(out.read_text())
    own, own_rows = _circuit(PP.Capped(PP.i2_of_serve(str(s.run), 0, str(tmp_path / "own"), None, None, trace="py")), s.data, LAYER0)
    assert IF.circuit_of(own) == IF.circuit_of(mine)
    copy = ("import json, shutil, sys; a = sys.argv; shutil.copy(a[1], a[a.index('--out') + 1]); "
            f"print(json.dumps({{'bits': {s.bound['advice']['bits']}}}))")
    compilers = {s.bound["compiler"]["name"]: {"digest": s.bound["compiler"]["digest"], "argv": [sys.executable, "-c", copy, str(out)]}}
    opening = json.loads(Path(s.a.opening).read_text())
    reads = lambda rows: [(rid, int(r["i"])) for rid, _pi, r, _d in rows[0]]  # noqa: E731
    for k, (name, names, params, codes, same_reads) in enumerate((
            ("honest", None, False, [], True),
            ("a layer-1 SiluMul row and layer 0's down_proj swapped", (("SiluMul_v2", 1), ("Gemm_v2", 0)), False, ["circuit"], False),
            ("layers 0 and 1's SiluMul rows swapped", (("SiluMul_v2", 0), ("SiluMul_v2", 1)), False, [], False),
            ("layers 0 and 1's down_proj weights swapped", None, True, [], True))):
        rows = _row_dir(tmp_path / f"rows{k}", s, names, params)
        theirs, their_rows = _circuit(PP.Capped(PP.i2_of_serve(str(s.run), 0, str(rows), None, None)), s.data, LAYER0)
        v, bound = RA.compile_check(theirs, s.bound, opening, compilers, tmp_path / "c")
        assert (v["codes"], bound is None, reads(their_rows) == reads(own_rows)) == (codes, bool(codes), same_reads), name
