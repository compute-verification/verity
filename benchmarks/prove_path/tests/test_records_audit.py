"""The audit from a window's records alone (`records_audit`), without proving, on a window written through the interface
(`verity.ml.windows.interface`) under core's registration:

* the verifier's window is the circuit record's, each template resolved in Verity's catalog, and the record must be the one
  the registration binds; a window with row tables is refused until its records carry their segment roots;
* the compile check runs the registered compiler as the verifier's table names it, on advice that opens the registered root,
  and refuses by name a compiler it doesn't hold, an opening of another root, a compiler that refuses the advice, one whose
  circuit is another or that writes none, and a length other than the registered one or none;
* a compiler registered under a commit runs from a checkout of that commit, not the auditing tree's, and a commit whose code
  doesn't hash to the registered digest is refused;
* the leak bound's count is the circuit's unwired input bits, reported beside the advice's charge and vacuous past it, and
  a record that declares registered reads counts those of a parameter the verifier pinned registered and the rest free
  (`declared_count`);
* a circuit reading registered values is audited against the entries the verifier pinned, each under its own domain
  (`pinned`), and refused before any draw without them;
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

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R
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
    win = IF.ExecutionBlockBuilder(IF.Circuit("test", ((COORD, n),)), RUN, D.parse_law(LAW), cap, advice=bound)
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


def test_a_registry_template_resolves_in_a_fresh_audit_process():
    """A window's io shell (`ScanSlots_v2`) is a registry Definition, which `interface.resolve`'s own walk doesn't import: in a
    fresh process, which has loaded nothing else, `catalog_resolve` resolves its id to itself."""
    from verity.ml.definitions.registry import boolean_io_shell as BX

    occ, lp, mx, ids = (tuple(f + 36333 * j for j in range(2)) for f in (376, 168, 200, 474))
    tid = descriptor_id(bind(BX.ScanSlotsV2, NS=4548, NP=4, IDB=17, K0=12, K=45, OCC=occ, LP=lp, MAX=mx, IDS=ids))
    code = f"""
import sys
sys.path.insert(0, {str(HERE.parent)!r})
import records_audit as RA
from verity.ml.circuits.codec import descriptor_id
assert descriptor_id(RA.catalog_resolve(sys.argv[1])) == sys.argv[1]
"""
    p = subprocess.run([sys.executable, "-c", code, tid], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]


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


def _registered_circuit(n: int = 2, free: str | None = None) -> IF.Circuit:
    """``n`` coordinates whose every input row is a registered read of one parameter ``a``, but port ``free``'s."""
    rows, at, reads = IF.layout(COORD).rows, 0, []
    for p, words in rows:
        if p != free:
            reads.append(IF.Read(TID, p, "a", at=0, radix=(n,), base=at, step=(words,)))
        at += n * words
    return IF.Circuit("test", ((COORD, n),), registered=IF.Registered({"a": at}, reads=tuple(reads)))


#: the verifier's pin of ``a``, and a window registration's root for it
PIN_A = {"a": "ab" * 64}


def test_a_declared_records_count_is_its_free_inputs_and_the_advice_against_the_charge(tmp_path, capsys):
    """A stated record whose inputs are declared is counted from its description alone: with every input row registered at
    a root the verifier pinned before the run, the root the window's registration records, nothing is free, so its other
    inputs are the advice's bits, the charge; a port left free adds its rows and the bound is vacuous, and so does every
    registered read when the verifier pins no root, or when the window's root is another than the pin or none.
    ``records_audit.py count`` prints the same from the record's and the registration's files (core's, its window's
    ``registered_roots`` or its committed values' entries, or an integration's own, whose ``window`` is its index), and
    refuses a pin with no root."""
    doc, inputs = _registered_circuit().description(), 2 * sum(16 * w for _p, w in IF.layout(COORD).rows)
    assert RA.declared_count(doc, 1687, pins=PIN_A, roots=PIN_A) == {
        "units": 2, "inputs": inputs, "wired": 0, "registered": inputs, "ingress": 0, "free": 0, "other_inputs": 1687,
        "advice_bits": 1687, "bound": "within the charge"}
    unpinned = RA.declared_count(doc, 1687)
    assert (unpinned["registered"], unpinned["free"], unpinned["bound"]) == (0, inputs, "vacuous")
    assert RA.declared_count(doc, 1687, pins=PIN_A, roots={"a": "cd" * 64}) == unpinned == RA.declared_count(doc, 1687, pins=PIN_A)
    x = 16 * dict(IF.layout(COORD).rows)["x"]
    loose = RA.declared_count(_registered_circuit(free="x").description(), 1687, pins=PIN_A, roots=PIN_A)
    assert (loose["free"], loose["other_inputs"], loose["bound"]) == (2 * x, 2 * x + 1687, "vacuous")
    with pytest.raises(ValueError, match=r"\{param: root\}"):
        RA.declared_count(doc, 1687, pins={"a"})
    (tmp_path / "c.json").write_bytes(IF.circuit_bytes(doc))
    def group(root: str) -> dict:
        return {"program": "p", "values": {"a": {"value": "a", "root": root}}}

    pouw = {"context": RUN, "kind": "served", "public": {}}
    regs = {"core": {"window": {"run": RUN, "registered_roots": PIN_A}}, "own": {"window": 63, "registered_roots": PIN_A},
            "other": {"window": {"run": RUN, "registered_roots": {"a": "cd" * 64}}}, "none": {"window": 63},
            "activations": {"window": {**pouw, "activations": group("ab" * 64)}},
            "registered": {"window": {**pouw, "activations": group("AB" * 64), "registered": group("ab" * 64)}}}
    for name, reg in regs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(reg))
    for pins, reg, roots in [((), None, {}), ((), "core", PIN_A), (("a=" + "ab" * 64,), "core", PIN_A),
                             (("a=" + "AB" * 64,), "own", PIN_A), (("a=" + "ab" * 64,), "other", {"a": "cd" * 64}),
                             (("a=" + "ab" * 64,), "none", {}), (("a=" + "ab" * 64,), None, {}),
                             (("a=" + "ab" * 64,), "activations", PIN_A), (("a=" + "ab" * 64,), "registered", PIN_A)]:
        capsys.readouterr()
        argv = ["count", "--circuit", str(tmp_path / "c.json"), "--bits", "1687", *(f"--pin={p}" for p in pins),
                *(["--registration", str(tmp_path / f"{reg}.json")] if reg else [])]
        assert RA.main(argv) == 0
        got = json.loads(capsys.readouterr().out)
        assert got == RA.declared_count(doc, 1687, pins=dict(RA.pin_arg(p) for p in pins), roots=roots)
        assert got["free"] == (0 if pins and roots == PIN_A else inputs), (pins, reg)
    entry = {"value": "a", "root": PIN_A["a"], "leaves": 1, "words": 4, "domain": REG.domain("p", "a", 4, 1).domain_id.hex()}
    (tmp_path / "pins.json").write_text(json.dumps(PIN_A))
    (tmp_path / "registration.json").write_text(json.dumps({"program": "p", "values": {"a": entry}}))
    (tmp_path / "other-domain.json").write_text(json.dumps({"program": "q", "values": {"a": entry}}))
    shape = {k: v for k, v in entry.items() if k != "root"}
    (tmp_path / "record.json").write_text(json.dumps({"of_record": True, "registered_entries": {"a": shape}}))
    for f in ("pins.json", "registration.json"):
        capsys.readouterr()
        assert RA.main(["count", "--circuit", str(tmp_path / "c.json"), "--bits", "1687", "--pins", str(tmp_path / f),
                        "--registration", str(tmp_path / "core.json")]) == 0
        assert json.loads(capsys.readouterr().out)["free"] == 0
    with pytest.raises(ValueError, match="a weights record states each weight's shape, not its root"):
        RA.pins_of((), [str(tmp_path / "record.json")])
    for bad in (["--pin", "a"], ["--pin", "a=zz"], ["--pin", "a=" + "ab" * 64, "--pin", "a=" + "cd" * 64],
                ["--pin", "a=" + "ab" * 64, "--pins", str(tmp_path / "pins.json")], ["--pins", str(tmp_path / "record.json")],
                ["--pins", str(tmp_path / "other-domain.json")]):
        with pytest.raises(SystemExit):
            RA.main(["count", "--circuit", str(tmp_path / "c.json"), "--bits", "1687", *bad])
    with pytest.raises(ValueError, match="counted from its members"):
        RA.declared_count({"templates": doc["templates"]}, 0)


def test_a_windows_roots_are_its_registered_roots_and_its_committed_values_entries_roots():
    """`window_roots`: the roots a window's registration records for its parameters are its ``registered_roots`` and each
    entry's root of the committed values it names, ``{program, values}`` (`ENTRY_GROUPS`: per-window ``activations``,
    PoUW's operands among them, and pinned ``registered``, PoUW's weights), one root a parameter; a group that isn't one,
    an entry with no root of hex digits, or two roots for one parameter are refused."""
    def group(**roots: str) -> dict:
        return {"program": "p", "values": {v: {"value": v, "root": r} for v, r in roots.items()}}

    from verity.core.protocols.pouw import operands as OP

    w = {"context": RUN, "registered_roots": {"e": "01" * 64}, "activations": group(x="ab" * 64, e="01" * 64),
         "registered": group(w="cd" * 64)}
    assert RA.window_roots(w) == {"e": "01" * 64, "x": "ab" * 64, "w": "cd" * 64}
    a, b = (REG.Registered.commit("test/model", v, [[i] * 4], schema=REG.SCHEMA_V2) for i, v in enumerate("ab"))
    pouw = {"context": RUN, "kind": "served", "public": {}, OP.KEY: OP.block("test/h1", {"a": a}),
            "registered": OP.block("test/model", {"b": b})}
    assert RA.window_roots(pouw) == {"a": a.root.hex(), "b": b.root.hex()}
    assert RA.window_roots({"run": RUN, "registered": None}) == {} == RA.registration_roots({"window": 63})
    assert RA.registration_roots({"window": w}) == RA.window_roots(w)
    for bad, why in [({"registered": {"w": "cd" * 64}}, r"is \{program, values"), ({"activations": group(x=7)}, "hex digits"),
                     ({"activations": group(x="zz")}, "hex digits"), ({"registered": group(w=None)}, "hex digits"),
                     ({**w, "registered": group(x="cd" * 64)}, r"two roots for \['x'\]"),
                     ({"registered_roots": {"w": "ab" * 64}, "registered": group(w="cd" * 64)}, "two roots")]:
        with pytest.raises(ValueError, match=why):
            RA.window_roots(bad)


def test_the_window_circuit_command_writes_i2s_circuit_whose_count_is_the_charge(tmp_path, monkeypatch, capsys):
    """``prove_path.py window-circuit`` on the synthetic serve-openai run of `test_prove_path` (I2's window circuit stubbed,
    one whose every input is registered): it writes the circuit's record from its description (members stated at its cap,
    no run) or, ``--description``, the description, prints the length the registration charges with the count, and exits
    `records_audit.REFUSED` where I2's circuit refuses the advice. ``records_audit.py count`` over the record finds the
    charge. Each count holds the pin to the window's root its advice registration records: where it records another root,
    or none, every registered read is free."""
    import prove_path as PP
    from test_prove_path import _served
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict.build import Unexpressible

    s, circuit, refuse = _served(tmp_path, monkeypatch), _registered_circuit(), []

    def lower(registered, compiled, cls):
        assert cls == WC.CLASS_OF_RECORD and compiled.requests
        if refuse:
            raise Unexpressible("a family the window's lowering doesn't take")
        return circuit

    monkeypatch.setattr(WC, "window_circuit", lower)
    (tmp_path / "advice.bin").write_bytes(s.data)
    bits = s.bound["advice"]["bits"]
    reg_file = s.run / "advice" / "w0" / "registration.json"
    served = json.loads(reg_file.read_text())
    reg_file.write_text(json.dumps({**served, "registered_roots": PIN_A}))
    base = ["window-circuit", "--serve-run", str(s.run), "--window", "0", "--row-dir", s.a.row_dir, "--advice",
            str(tmp_path / "advice.bin"), "--cap", "4", "--pin", "a=" + PIN_A["a"]]
    capsys.readouterr()
    assert PP.main([*base, "--out", str(tmp_path / "c.json")]) == 0
    got = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (got["bits"], got["count"]) == (bits, IF.input_count(circuit.description(), pins=PIN_A, roots=PIN_A))
    assert got["count"]["free"] == 0
    rec = json.loads((tmp_path / "c.json").read_text())
    assert rec == IF.descriptor_record(circuit, 4) and (rec["cap"], rec["registered"]) == (4, circuit.registered.doc())
    assert PP.main([*base, "--out", str(tmp_path / "d.json"), "--description"]) == 0
    assert json.loads((tmp_path / "d.json").read_text()) == circuit.description()
    capsys.readouterr()
    assert RA.main(["count", "--circuit", str(tmp_path / "c.json"), "--bits", str(bits), "--pin", "a=" + PIN_A["a"],
                    "--registration", str(reg_file)]) == 0
    count = json.loads(capsys.readouterr().out)
    assert (count["free"], count["other_inputs"], count["bound"]) == (0, bits, "within the charge")
    for theirs in ({**served, "registered_roots": {"a": "cd" * 64}}, served):
        reg_file.write_text(json.dumps(theirs))
        assert PP.main([*base, "--out", str(tmp_path / "c.json")]) == 0
        got = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (got["count"]["registered"], got["count"]["free"]) == (0, got["count"]["inputs"])
    refuse.append(True)
    assert PP.main([*base, "--out", str(tmp_path / "e.json")]) == RA.REFUSED
    assert "prove_path.py window-circuit: I2's window circuit refuses the advice" in capsys.readouterr().err


def _registered(d: Path, n: int = 8, cap: int = 4) -> SimpleNamespace:
    """A window of ``n`` coordinates each reading its ``wrow`` from a registered value of 4 rows (unit u row u % 4), committed
    by core's registered rows, and the verifier's pin of it."""
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 1 << 16, (4, K // 2), dtype=np.uint16)
    value = REG.Registered.commit("test/model", "w", rows.tolist(), schema=REG.SCHEMA_V2)
    read = IF.Read(TID, "wrow", "w", 0, (n // 4, 4), 0, (0, K // 2))
    win = IF.ExecutionBlockBuilder(IF.Circuit("test", ((COORD, n),), registered=IF.Registered({"w": 4 * K // 2}, reads=(read,))), RUN,
                    D.parse_law(LAW), cap, registered={"w": value})
    x = rng.integers(-128, 128, (n, K)).astype(np.int8)
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), rows[np.arange(n) % 4].view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)])
    win.write(win.close(), d, {"label": "test"}, registrar=R)
    return SimpleNamespace(dir=d, value=value, pin={"program": "test/model", "values": {"w": value.entry()}})


def test_registered_reads_are_audited_against_the_pinned_entries_and_leave_the_free_inputs(tmp_path, capsys):
    got = _registered(tmp_path / "w")
    mine, doc, _sets = RA.verifier_window(got.dir, D.parse_law(LAW), None)
    pins = RA.pinned(got.pin)
    files = IF.registered_reads(doc, mine, pins, registered=REG)
    assert [f["ports"]["wrow"]["reads"] for f in files.values()] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert all(f["ports"]["wrow"]["root"] == got.value.root.hex() for f in files.values())
    per_unit = sum(16 * w for _p, w in IF.layout(COORD).rows)
    rec = json.loads((got.dir / "registration.json").read_text())
    roots, pin = RA.registration_roots(rec), {v: e["root"] for v, e in pins.items()}
    assert roots == rec["window"]["registered_roots"] == pin == {"w": got.value.root.hex()}
    count = RA.declared_count(doc, 0, pins=pin, roots=roots)
    assert (count["inputs"], count["registered"], count["free"]) == (8 * per_unit, 8 * 8 * K, 8 * per_unit - 8 * 8 * K)
    assert count["other_inputs"] == count["free"] and count["bound"] == "vacuous"
    assert RA.declared_count(doc, 0, pins={"w": "cd" * 64}, roots=roots)["free"] == 8 * per_unit
    with pytest.raises(ValueError, match="another value or domain"):
        RA.pinned({**got.pin, "program": "test/other"})
    with pytest.raises(ValueError, match="a pin is"):
        RA.pinned({"values": {}})
    base = ["audit", "--window", str(got.dir), "--law", LAW, "--circuit", "00" * 64, "--run", RUN, "--m0-pythonpath", ".",
            "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "o")]
    assert RA.main(base) == 1
    assert "didn't pin" in json.loads((tmp_path / "o" / "summary.json").read_text())["refused"]["records"]
    (tmp_path / "circuit.json").write_bytes((got.dir / "circuit.json").read_bytes())
    assert RA.main(["count", "--circuit", str(tmp_path / "circuit.json"), "--bits", "0", "--pin", f"w={pin['w']}",
                    "--registration", str(got.dir / "registration.json")]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["free"] == count["free"]


def test_a_windows_committed_activations_are_held_to_domains_binding_its_run():
    rows = [[1, 2], [3, 4]]
    mine = {v: REG.Registered.commit("test/model", v, rows, schema=REG.SCHEMA_V2, window=RUN).entry() for v in ("x", "sx")}
    w = {"run": RUN, "activations": {"program": "test/model", "values": mine}}
    got = RA.committed_activations(w)
    assert got["ok"] and got["detail"]["values"]["x"] == {"root": mine["x"]["root"], "leaves": 2, "words": 2}
    assert RA.committed_activations({"run": RUN}) is None
    elsewhere = REG.Registered.commit("test/model", "x", rows, schema=REG.SCHEMA_V2, window="cd" * 64).entry()
    for bad, code in (({**w, "run": "cd" * 64}, "domains"), ({**w, "activations": {**w["activations"], "program": "other"}}, "domains"),
                      ({**w, "activations": {"program": "test/model", "values": {**mine, "x": elsewhere}}}, "domains"),
                      ({**w, "activations": {"program": "test/model", "values": {}}}, "activations"),
                      ({**w, "activations": {"program": "test/model"}}, "activations")):
        v = RA.committed_activations(bad)
        assert not v["ok"] and code in v["codes"], (bad, v)


def _crafted(reads: tuple[IF.Read, ...], params: dict[str, int], n: int = 33) -> dict:
    """A stated record of ``n`` coordinates declaring ``reads`` of ``params`` (no wires)."""
    return {"templates": [[TID, n]], "wires": [], "registered": IF.Registered(params, reads=reads).doc()}


#: coordinates 0..31 write their one-word ``o0`` as leaves 0..31 of ``y``, which coordinate 32's ``x`` (32 words) reads: a
#: tie; coordinates 0..31 read their ``x`` from ``e``, which no output port writes
TIE = (IF.Read(TID, "o0", "y", 0, (32,), 0, (1,)), IF.Read(TID, "x", "y", 32, (1,), 0, (32,)))
INPUT_ONLY = (IF.Read(TID, "x", "e", 0, (32,), 0, (32,)),)
PARAMS = {"y": 32, "e": 32 * 32}


def test_a_value_is_a_tie_only_when_output_ports_write_every_leaf_its_input_ports_take():
    """`tied_values`: ``y``, written by the coordinates' outputs and read by another's input, is a tie; ``e``, read only
    by input ports, isn't; nor is a value whose readers take leaves no output writes; a value nothing reads isn't either."""
    doc = _crafted(TIE + INPUT_ONLY, PARAMS)
    assert RA.tied_values(doc, {"y", "e", "z"}) == {"y"}
    assert RA.tied_values(doc, {"e"}) == set() and RA.tied_values(doc, ()) == set()
    half = (IF.Read(TID, "o0", "p", 0, (16,), 0, (1,)), IF.Read(TID, "wrow", "p", 32, (1,), 0, (32,)))
    assert RA.tied_values(_crafted(half, {"p": 32}), {"p"}) == set()


def test_an_input_only_window_value_is_free_in_the_count_and_named():
    """`declared_count` with ``untied``: the input rows read from a value no output port writes leave ``registered`` for
    ``free``, by exactly those rows, and the count names it; a tie's reads stay registered. Both are pins, as the audit
    pins every value it opens."""
    doc = _crafted(TIE + INPUT_ONLY, PARAMS)
    per_unit = 16 * sum(w for _p, w in IF.layout(COORD).rows)
    roots = {v: f"{i:02x}" * 64 for i, v in enumerate(PARAMS, 1)}
    before, after = (RA.declared_count(doc, 0, pins=roots, roots=roots),
                     RA.declared_count(doc, 0, pins=roots, roots=roots, untied={"e"}))
    assert before["registered"] == 16 * 32 + 32 * 16 * 32 and before["free"] == 33 * per_unit - before["registered"]
    assert after["registered"] == 16 * 32 and after["free"] - before["free"] == 32 * 16 * 32 and after["untied"] == ["e"]
    assert "untied" not in before


def test_the_windows_own_values_the_circuit_reads_are_pinned_only_when_they_are_ties():
    """`window_pins`: the per-window values the circuit reads are the registration's entries, each held to a domain binding
    the window's run root, split into ties (pinned) and the untied (read only by input ports, counted free); one it
    doesn't read is neither, and one committed under another run, or naming another value, is refused."""
    rows = [[1, 2], [3, 4]]
    mine = {v: REG.Registered.commit("test/model", v, rows, schema=REG.SCHEMA_V2, window=RUN).entry() for v in ("x", "y", "e")}
    w = {"run": RUN, "activations": {"program": "test/model", "values": mine}}
    circuit = _crafted(TIE + INPUT_ONLY, PARAMS)
    assert RA.window_pins(w, circuit) == ({"y": mine["y"]}, {"e": mine["e"]})
    assert RA.window_pins(w, {}) == ({}, {}) and RA.window_pins({"run": RUN}, circuit) == ({}, {})
    elsewhere = REG.Registered.commit("test/model", "y", rows, schema=REG.SCHEMA_V2, window="cd" * 64).entry()
    for bad in ({**w, "run": "cd" * 64}, {**w, "activations": {"program": "test/model", "values": {"y": elsewhere}}},
                {**w, "activations": {"program": "test/model", "values": {"y": mine["x"]}}}):
        with pytest.raises(ValueError, match="another value or a domain"):
            RA.window_pins(bad, circuit)


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


def test_the_audit_reserves_a_port_for_every_session_its_draw_can_open(tmp_path, monkeypatch):
    """A session listens on one port a member with a proved unit (closure included), so 20 proved members need 20 ports:
    the audit reserves one a member, not a rank's 16 (`a5.RANK_PORTS`), and a draw needing more than ``--ports`` is
    refused before any session listens, since the next one could reach another audit's verifier."""
    import a5 as A5

    got = _write(tmp_path / "w", n=20, cap=1)
    mine, _doc, _sets = RA.verifier_window(got.dir, D.parse_law(LAW), None)
    assert len(mine.members) == 20 > A5.RANK_PORTS
    every = {"law": "work", "units": [18, 19], "closure": list(range(18))}
    assert A5.port_refusal(mine, every, 10000, A5.RANK_PORTS) == {
        "sessions-past-the-reserved-ports": {"sessions": 20, "ports": 16, "from": 10000}}
    assert A5.port_refusal(mine, every, 10000, 20) is None and A5.port_refusal(mine, every, 10000, None) is None
    assert A5.port_refusal(mine, {"law": "subset", "units": [3, 7]}, 10000, 2) is None

    class Opened(Exception):
        pass

    seen = {}

    def free_ports(span):
        seen["span"] = span
        return 20000

    def audit(a, sets, out, mine, served, source, port, summary, more, **kw):
        seen.update(port=port, ports=kw["ports"])
        raise Opened

    monkeypatch.setattr(RA, "free_ports", free_ports)
    monkeypatch.setattr(A5, "audit", audit)
    args = ["audit", "--window", str(got.dir), "--law", LAW, "--circuit", got.rec["window"]["circuit"], "--run", RUN,
            "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "out")]
    for extra, want in (([], {"span": 20, "port": 20000, "ports": 20}), (["--port", "10000", "--ports", "16"],
                                                                         {"port": 10000, "ports": 16})):
        seen.clear()
        with pytest.raises(Opened):
            RA.main([*args, *extra])
        assert seen == want


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
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R
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
    window = IF.ExecutionBlockBuilder(IF.Circuit("test", ((COORD, 5),)), "", D.parse_law(LAW), 4).window
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


def test_the_output_class_is_the_given_ranges_and_every_uncut_registered_read_of_an_output_port():
    """`output_class`: the verifier's own ranges, with each uncut registered read of an ``o<j>`` port adding its reader's
    units at the template's base in the population; a read of an input port, or a cut one, adds none."""
    circuit = {"templates": [["a", 8], ["b", 16]], "registered": {"reads": [
        {"reader": "b", "port": "o0", "at": 4, "radix": [2, 2], "cut": [0, 1]},
        {"reader": "a", "port": "o1", "at": 0, "radix": [2]},
        {"reader": "b", "port": "w", "at": 0, "radix": [16], "cut": [0, 1]},
        {"reader": "b", "port": "o0", "at": 12, "radix": [4], "cut": [1, 2]}]}}
    assert RA.output_class(circuit) == [[0, 2], [12, 16]]
    assert RA.output_class(circuit, [[2, 4], [20, 24]]) == [[0, 4], [12, 16], [20, 24]]
    assert RA.output_class({"templates": [["a", 8]]}, [[4, 8]]) == [[4, 8]]
    assert RA.output_class({"templates": [["a", 8]]}) == []
    tie = {"templates": [["a", 8], ["b", 8]], "registered": {"reads": [
        {"reader": "a", "port": "o0", "param": "y", "at": 0, "radix": [8]},
        {"reader": "b", "port": "x", "param": "y", "at": 0, "radix": [8]},
        {"reader": "b", "port": "o0", "param": "out", "at": 4, "radix": [4]}]}}
    assert RA.output_class(tie, own={"y"}) == [[12, 16]]
    assert RA.output_class(tie) == [[0, 8], [12, 16]]


def test_a_served_output_that_the_next_step_reads_keeps_its_units_in_the_output_class():
    """A decode loop's served ``out``, pinned from outside the window, is read by an output port and by the next step's
    input port: it isn't one of the window's own values, so its writers stay in the class (keying the exclusion on any
    value an input port reads would leave the class empty)."""
    decode = {"templates": [["a", 8], ["b", 8]], "registered": {"reads": [
        {"reader": "b", "port": "o0", "param": "out", "at": 0, "radix": [8]},
        {"reader": "a", "port": "x", "param": "out", "at": 0, "radix": [8]}]}}
    assert RA.output_class(decode) == [[8, 16]]
    assert RA.output_class(decode, own={"y"}) == [[8, 16]]


def test_a_row_dir_whose_names_are_permuted_under_its_digests_is_refused_or_ignored_by_the_compile_check(tmp_path, monkeypatch):
    """No Program digest covers a Build's row names or its weights fields' names (nor its ``supplied``, correspondence or
    ``prefix_cache.json``), and the served compiler's command reads none of them from the prover: it takes no row dir and
    compiles over its own Build of each Program I2 compiled, beside the names its compile gives it (`prove_path.own_builds`).
    Against a prover's row dir permuted under unchanged digests (each Build still I2's Program, `build_of`, its correspondence
    record permuted with its names), the compile check refuses by name (``circuit``) the circuit.json whose names make another
    circuit (a layer-1 SiluMul row and layer 0's first down_proj row swap names: one more SiluMul row in layer 0's scope), and
    ignores names that make the same one (layers 0 and 1's first SiluMul rows swapped, which moves only the prover's own reads;
    two weights fields swapped): the verifier's record is its own compile's in each case."""
    import prove_path as PP
    from test_prove_path import LAYER0, _circuit, _row_dir, _tiny_served

    s = _tiny_served(tmp_path, monkeypatch)
    (tmp_path / "part.json").write_text(json.dumps(LAYER0))
    (tmp_path / "advice.bin").write_bytes(s.data)
    out = tmp_path / "mine" / "circuit.json"
    out.parent.mkdir()
    assert PP.main(["circuit", "--serve-run", str(s.run), "--window", "0", "--partition", str(tmp_path / "part.json"),
                    "--advice", str(tmp_path / "advice.bin"), "--out", str(out)]) == 0
    assert s.compiles == [True] and list(out.parent.iterdir()) == [out]
    mine = json.loads(out.read_text())
    own, own_rows = _circuit(PP.Capped(PP.i2_of_serve(str(s.run), 0, str(tmp_path / "own"), None, None, own=True)), s.data, LAYER0)
    assert IF.circuit_of(own) == IF.circuit_of(mine)
    copy = ("import json, shutil, sys; a = sys.argv; shutil.copy(a[1], a[a.index('--out') + 1]); "
            f"print(json.dumps({{'bits': {s.bound['advice']['bits']}}}))")
    compilers = {s.bound["compiler"]["name"]: {"digest": s.bound["compiler"]["digest"], "argv": [sys.executable, "-c", copy, str(out)]}}
    opening = json.loads(Path(s.a.opening).read_text())
    reads = lambda rows: [(rid, int(r["i"])) for rid, _pi, r, _d in rows[0]]  # noqa: E731
    for k, (name, names, params, codes, same_reads) in enumerate((
            ("honest", None, False, [], True),
            ("a layer-1 SiluMul row and layer 0's down_proj swapped", ("model.model.layers.1.mlp.act_fn/", "model.model.layers.0.mlp.down_proj/"),
             False, ["circuit"], False),
            ("layers 0 and 1's SiluMul rows swapped", ("model.model.layers.0.mlp.act_fn/", "model.model.layers.1.mlp.act_fn/"), False, [], False),
            ("layers 0 and 1's down_proj weights swapped", None, True, [], True))):
        rows = _row_dir(tmp_path / f"rows{k}", s, names, params)
        theirs, their_rows = _circuit(PP.Capped(PP.i2_of_serve(str(s.run), 0, str(rows), None, None)), s.data, LAYER0)
        v, bound = RA.compile_check(theirs, s.bound, opening, compilers, tmp_path / "c")
        assert (v["codes"], bound is None, reads(their_rows) == reads(own_rows)) == (codes, bool(codes), same_reads), name
