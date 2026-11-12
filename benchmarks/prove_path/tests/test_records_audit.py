"""The audit from a window's records alone (`records_audit`), without proving, on a window written through the interface
(`verity.commitments.windows.interface`) under core's registration:

* the verifier's window is the circuit record's, each template resolved in Verity's catalog, and the record must be the one
  the registration binds; a window with row tables is refused until its records carry their segment roots;
* the compile check runs the registered compiler as the verifier's table names it, on advice that opens the registered root,
  and refuses by name a compiler it doesn't hold, an opening of another root, a compiler that refuses the advice, one whose
  circuit is another or that writes none, and a length other than the registered one or none;
* a compiler registered under a commit runs from a checkout of that commit, not the auditing tree's, and a commit whose code
  doesn't hash to the registered digest is refused;
* the leak bound's count is the circuit's unwired input bits, reported beside the advice's charge and vacuous past it, and
  a record that declares registered reads counts those of a parameter the verifier pinned registered and the rest free
  (`declared_count`); the audit counts a record that lists its members and declares its window's ingress so too, and
  prints the window figure beside the run figure with its premise;
* a circuit reading registered values is audited against the entries the verifier pinned, each under its own domain
  (`pinned`), and refused before any sample without them, a record that lists its members (a served window's) as one that
  states them; one whose program takes values only tied to the earlier registration that registered them first
  (`tie_of`), its readers' reads registered at that record's roots;
* a window's audit is one compute audit of the network certifier's period (`period`), its scan state's reads registered;
* a declared day (``--day``, `declared_day`) hands R2 the registration log's declarations, so a program other than the
  day's is refused;
* `source` and the negatives refuse what the records don't bind, and `fixed_before_draw` refuses advice re-committed after
  the coins;
* the gate's coins (``--challenger gate:USER@HOST``) reach the challenger's session with the peer and the key record;
* ``warm`` fills a5's compose cache from the verifier's window, its composers keyed by what they compose from, a table
  read's by the verifier's executable too, and the audit takes that cache;
* none of it imports `verity_vllm`, and the served compiler's command (``prove_path.py circuit``) refuses advice it can't
  compile and charges the length the registration does;
* that command compiles over its own trace of each Program, never the prover's row dir, so a row dir whose names make
  another circuit under unchanged digests is refused by name and one whose names make the same circuit is ignored."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.protocols.spot.verifier.one_stage import draw as D, registered as REG, registration as R
from verity.circuits.codec import canonical_json, descriptor_id, encode_program, program_digest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch, evaluate_words
from verity.definitions.ops import gemm as G
from verity.commitments.windows import interface as IF, rows as CS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "one_stage"))
import records_audit as RA  # noqa: E402
import served_integrity as SI  # noqa: E402

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
    win = IF.Window(IF.Circuit("test", ((COORD, n),)), RUN, D.parse_design(LAW), cap, advice=bound)
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
    mine, doc, sets = RA.verifier_window(written.dir, D.parse_design(LAW), None)
    assert doc == written.doc and mine.run == RUN and mine.partition_digest == written.rec["partition"]
    assert [(m.template, m.n) for m in mine.members] == [(TID, 4), (TID, 1)]
    assert set(sets) == {TID} and callable(sets[TID][0]) and descriptor_id(sets[TID][1]) == TID
    w = written.rec["window"]
    assert w["circuit"] == hashlib.sha512((written.dir / "circuit.json").read_bytes()).hexdigest()
    assert (w["compiler"], w["advice"]) == (written.bound["compiler"], written.bound["advice"])
    (written.dir / "circuit.json").write_text(json.dumps({**doc, "name": "other"}))
    with pytest.raises(ValueError, match="not the circuit record the registration binds"):
        RA.verifier_window(written.dir, D.parse_design(LAW), None)


def test_a_registry_template_resolves_in_a_fresh_audit_process():
    """A window's io shell (`ScanSlots_v2`) is a registry Definition, which `interface.resolve`'s own walk doesn't import: in a
    fresh process, which has loaded nothing else, `catalog_resolve` resolves its id to itself."""
    from verity.definitions.registry import boolean_io_shell as BX

    occ, lp, mx, ids = (tuple(f + 36333 * j for j in range(2)) for f in (376, 168, 200, 474))
    tid = descriptor_id(bind(BX.ScanSlotsV2, NS=4548, NP=4, IDB=17, K0=12, K=45, OCC=occ, LP=lp, MAX=mx, IDS=ids))
    code = f"""
import sys
sys.path.insert(0, {str(HERE.parent)!r})
import records_audit as RA
from verity.circuits.codec import descriptor_id
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
    1,687 advice bits: vacuous. A wired port isn't an input, a shared-row port counts its table's rows, and an
    external port is reported but not subtracted until a layout lists the ingress at its steps."""
    per_unit = sum(16 * w for _p, w in IF.layout(COORD).rows)
    mine, doc = IF.of_records(written.dir, D.parse_design(LAW), IF.resolve)
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
    with pytest.raises(ValueError, match="or its listed members"):
        RA.declared_count({"templates": doc["templates"]}, 0)


def test_a_windows_roots_are_its_registered_roots_and_its_committed_values_entries_roots():
    """`window_roots`: the roots a window's registration records for its parameters are its ``registered_roots`` and each
    entry's root of the committed values it names, ``{program, values}`` (`ENTRY_GROUPS`: per-window ``activations``,
    PoUW's operands among them, and pinned ``registered``, PoUW's weights), one root a parameter; a group that isn't one,
    an entry with no root of hex digits, or two roots for one parameter are refused."""
    def group(**roots: str) -> dict:
        return {"program": "p", "values": {v: {"value": v, "root": r} for v, r in roots.items()}}

    from verity.protocols.pouw.verifier import operands as OP

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

    def lower(registered, compiled, cls, tables=None):
        assert cls == WC.CLASS_OF_RECORD and compiled.requests and tables is None
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


def test_class_circuit_from_phi_alone_is_window_circuits_record_and_serve_phi_names_the_runs_compiler_before_it_serves(
        tmp_path, monkeypatch, capsys):
    """``prove_path.py class-circuit --serve F`` builds the class's circuit from φ alone, no advice, row dir or window, and
    writes ``window-circuit``'s record of the run's window byte for byte, at the class of record and cut at Pearl+'s
    linears under ``--pearl-tables`` alike (I2's class circuit stubbed, `window_circuit.window_circuit` left to call it).
    ``serve-phi`` states what serve.json will record of φ from this device (`prove_path.device_target` stubbed with the
    run's), and ``compiler --serve`` over it names the compiler ``compiler --serve-run`` names once the run has served."""
    import prove_path as PP
    from test_prove_path import _served
    from verity_vllm.program.predict import window_circuit as WC

    s, circuit, seen = _served(tmp_path, monkeypatch), _registered_circuit(), []

    def lower(registered, cls, programs=None, grain=None, name=None, provable=True, tables=None):
        seen.append((registered, cls, tables))
        return circuit

    monkeypatch.setattr(WC, "class_circuit", lower)
    monkeypatch.setattr(WC, "takes", lambda *a: None)
    (tmp_path / "advice.bin").write_bytes(s.data)
    serve = json.loads((s.run / "serve.json").read_text())
    monkeypatch.setattr(PP, "device_target", lambda async_scheduling=False: {
        "target": serve["target"], "execution": serve.get("execution") or serve["engine"]["execution"]})
    phi = tmp_path / "phi.json"
    assert PP.main(["serve-phi", "--model", serve["model"], "--revision", serve["revision"], "--out", str(phi)]) == 0
    for tables in ((), ("--pearl-tables", "pearl-c-h100-v1-h3")):
        seen.clear()
        assert PP.main(["window-circuit", "--serve-run", str(s.run), "--window", "0", "--row-dir", s.a.row_dir, "--advice",
                        str(tmp_path / "advice.bin"), "--cap", "4", "--out", str(tmp_path / "w.json"), *tables]) == 0
        for given in (s.run / "serve.json", phi):
            assert PP.main(["class-circuit", "--serve", str(given), "--cap", "4", "--out", str(tmp_path / "c.json"),
                            *tables]) == 0
            assert (tmp_path / "c.json").read_bytes() == (tmp_path / "w.json").read_bytes()
        assert len(seen) == 3 and seen[0] == seen[1] == seen[2] and seen[0][1] == WC.CLASS_OF_RECORD
        assert (seen[0][2] is None) == (not tables) and (not tables or seen[0][2].scheme == "pearl-c-h100-v1-h3")
    capsys.readouterr()
    assert PP.main(["compiler", "--serve-run", str(s.run)]) == 0
    served = json.loads(capsys.readouterr().out)
    assert PP.main(["compiler", "--serve", str(phi)]) == 0
    assert json.loads(capsys.readouterr().out) == served


def test_the_window_circuit_command_under_h1s_w_reads_its_tables_with_no_key(tmp_path, monkeypatch, capsys):
    """``prove_path.py window-circuit --window-registration W``, the verifier's record of a window H1 tied: I2's window
    circuit reads H1's tables of W's Program (`h1_tie`'s, as window-commit's circuit with W reads them), and the command
    takes no key, which is H1's prover's alone; without W it reads none. Refused for a W under another Program."""
    import prove_path as PP
    from test_prove_path import _served
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.protocols.pouw import h1 as PH

    s, circuit, seen = _served(tmp_path, monkeypatch), _registered_circuit(), []

    def lower(registered, compiled, cls, tables=None):
        seen.append((cls.outputs, tables))
        return circuit

    monkeypatch.setattr(WC, "window_circuit", lower)
    (tmp_path / "advice.bin").write_bytes(s.data)
    layout = {"calls": [["a", 0], ["b", 0]] * 2, "linears": [["a", 64, 8], ["b", 64, 8]], "m": 8}
    (s.run / "h1").mkdir()
    (s.run / "h1" / PH.PROGRAM_META).write_text(json.dumps({"program_digest": "p" * 128, "steps": 2, "layout": layout}))
    (tmp_path / "w.json").write_text(json.dumps({"program": "p" * 128, "registered": []}))
    (tmp_path / "other.json").write_text(json.dumps({"program": "q" * 128}))
    base = ["window-circuit", "--serve-run", str(s.run), "--window", "0", "--row-dir", s.a.row_dir, "--advice",
            str(tmp_path / "advice.bin"), "--class", "4,32,2"]
    assert PP.main([*base, "--out", str(tmp_path / "c.json")]) == 0
    assert PP.main([*base, "--out", str(tmp_path / "t.json"), "--window-registration", str(tmp_path / "w.json")]) == 0
    assert seen == [(2, None), (2, WC.Tables(64, 8, 2))]
    with pytest.raises(SystemExit):
        PP.main([*base, "--out", str(tmp_path / "k.json"), "--window-registration", str(tmp_path / "w.json"), "--h1-key", "k"])
    capsys.readouterr()
    assert PP.main([*base, "--out", str(tmp_path / "o.json"), "--window-registration", str(tmp_path / "other.json")]) == RA.REFUSED
    assert "is no registration under the serve's H1 window Program" in capsys.readouterr().err


def _registered(d: Path, n: int = 8, cap: int = 4, name: str = "w", own: bool = False,
                listed: bool = False) -> SimpleNamespace:
    """A window of ``n`` coordinates each reading its ``wrow`` from a registered value ``name`` of 4 rows (coordinate u row u % 4),
    committed by core's registered rows, and the verifier's pin of it; ``own``: the window's own value instead, committed
    under its run root and registered in its ``activations``, as `prove_path.py window-commit` commits the scan's state;
    ``listed``: its record lists its members and declares the read beside them, a served window's form."""
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 1 << 16, (4, K // 2), dtype=np.uint16)
    value = REG.Registered.commit("test/model", name, rows.tolist(), schema=REG.SCHEMA_V2, window=RUN if own else None)
    read = IF.Read(TID, "wrow", name, 0, (n // 4, 4), 0, (0, K // 2))
    win = IF.Window(IF.Circuit("test", ((COORD, n),), registered=IF.Registered({name: 4 * K // 2}, reads=(read,))), RUN,
                    D.parse_design(LAW), cap, registered={name: value})
    if listed:
        stated = win.circuit_record()
        win.circuit_record = lambda: IF.circuit_record(win.window, stated["name"], stated["templates"], stated["statements"],
                                                       declared={"registered": stated["registered"]})
    x = rng.integers(-128, 128, (n, K)).astype(np.int8)
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), rows[np.arange(n) % 4].view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)])
    entry = {"program": "test/model", "values": {name: value.entry()}}
    win.write(win.close(), d, {"label": "test", **({"activations": entry} if own else {})}, registrar=R)
    return SimpleNamespace(dir=d, value=value, pin=entry)


def test_registered_reads_are_audited_against_the_pinned_entries_and_leave_the_free_inputs(tmp_path, capsys):
    got = _registered(tmp_path / "w")
    mine, doc, _sets = RA.verifier_window(got.dir, D.parse_design(LAW), None)
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
            "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "o"), "--challenger", "local", "--main", "m.txt"]
    assert RA.main(base) == 1
    assert "didn't pin" in json.loads((tmp_path / "o" / "summary.json").read_text())["refused"]["records"]
    (tmp_path / "circuit.json").write_bytes((got.dir / "circuit.json").read_bytes())
    assert RA.main(["count", "--circuit", str(tmp_path / "circuit.json"), "--bits", "0", "--pin", f"w={pin['w']}",
                    "--registration", str(got.dir / "registration.json")]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["free"] == count["free"]


def test_an_expert_tables_registration_is_a_pin_as_written():
    """A TP rank's registration of the per-expert cut's tables (``verity-vllm tp-register-tables``) is ``--registered`` as written,
    each table under its rank's program: moved to another rank's program it refuses, and two ranks' registrations name each table
    twice, so a rank's window pins its own rank's."""
    from verity_vllm.pipeline import expert_tables as ET

    E, N, K, label = 8, 4, 64, "model.model.layers.3.mlp.experts.w13_weight"
    fields = [(label, [E, N, K], 8), (label + "_scale_inv", [E, 1, 1], 32)]
    tables = ET.tables_of(fields, {"w13": 2})
    rng = np.random.default_rng(4)
    words = {label: rng.integers(0, 256, (E, N, K), dtype=np.uint8), label + "_scale_inv": rng.random((E, 1, 1), dtype=np.float32)}

    def registration(rank: int) -> dict:
        prog, nonce = ET.program("test/model", "abc", 2, rank, "model."), ET.nonces(tables, [f for f, _s, _b in fields], rank)
        values = {p: e for f, shape, _b in fields for p, e in ET.entries(prog, words[f], shape, tables[f], bytes(32), nonce).items()}
        return json.loads(json.dumps({"schema": ET.SCHEMA, "program": prog, "repo": "test/model", "revision": "abc", "world": 2,
                                      "rank": rank, "prefix": "model.", "blocks": {"w13": 2}, "values": values, "nonces": nonce}))

    one, zero = registration(1), registration(0)
    assert set(RA.pinned(one)) == set(one["values"]) and len(one["values"]) == N // 2 + 1
    with pytest.raises(ValueError, match="another value or domain"):
        RA.pinned({**one, "program": zero["program"]})
    with pytest.raises(ValueError, match="pinned twice"):
        RA.pinned([one, zero])


def test_a_listed_records_registered_weight_reads_are_audited_and_counted_as_a_stated_records(tmp_path, capsys):
    """A record that lists its members (a served window's form) declares its registered reads beside them, as a served
    window reads its model's weights: the verifier's window takes it, each member's ``--registered`` file is the one the
    stated record of the same window makes, and its count is the stated record's, the reads registered where the verifier
    pinned the parameter at the root the window's registration records. Without a pin the audit refuses the window before
    any sample; the count names the parameter ``unpinned`` and counts its reads free, as it does for a pin of another root
    or a registration that records none."""
    law = D.parse_design(LAW)
    stated, listed = _registered(tmp_path / "s"), _registered(tmp_path / "l", listed=True)
    (ms, sdoc, _), (ml, ldoc, _) = (RA.verifier_window(g.dir, law, None) for g in (stated, listed))
    assert "cap" in sdoc and "members" in ldoc and "cap" not in ldoc and ldoc["registered"] == sdoc["registered"]
    assert ml.members == ms.members and all(dict(m.tables)["wrow"] == 4 for m in ml.members)
    pins = RA.pinned(listed.pin)
    files = IF.registered_reads(ldoc, ml, pins, registered=REG)
    assert files == IF.registered_reads(sdoc, ms, pins, registered=REG) and len(files) == 2
    pin = RA.pin_roots(pins)
    roots = RA.registration_roots(json.loads((listed.dir / "registration.json").read_text()))
    count = RA.declared_count(ldoc, 0, pins=pin, roots=roots)
    assert roots == pin and count == {**RA.declared_count(sdoc, 0, pins=pin, roots=roots), IF.WINDOW_INGRESS: 0}
    assert count["registered"] == 8 * 8 * K and "unpinned" not in count
    for p, r in ((None, roots), ({"w": "cd" * 64}, roots), (pin, {})):
        free = RA.declared_count(ldoc, 0, pins=p, roots=r)
        assert (free["registered"], free["free"], free["unpinned"]) == (0, count["free"] + 8 * 8 * K, ["w"])
    base = ["audit", "--window", str(listed.dir), "--law", LAW, "--circuit", "00" * 64, "--run", RUN, "--m0-pythonpath",
            ".", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "o"), "--challenger", "local",
            "--main", "m.txt"]
    assert RA.main(base) == 1
    assert "didn't pin" in json.loads((tmp_path / "o" / "summary.json").read_text())["refused"]["records"]
    for pins_, want in (([f"--pin=w={pin['w']}"], count), ([], RA.declared_count(ldoc, 0))):
        capsys.readouterr()
        assert RA.main(["count", "--circuit", str(listed.dir / "circuit.json"), "--bits", "0", *pins_,
                        "--registration", str(listed.dir / "registration.json")]) == 0
        assert json.loads(capsys.readouterr().out.splitlines()[-1]) == want


def test_a_tie_pins_the_circuits_values_at_the_earlier_registrations_tables():
    """`tie_of`: a circuit whose program takes values (a served window's H1 tables, by their Commit names) is tied to the
    earlier registration W that registered them first: one-stage's ``log`` is W, each value's ``registered`` words are W's
    entry's, its ``outputs`` W's table of it under the prefix (`credit.window_table`) and its schema v2, and the audit pins
    each at W's     entry under the value's own name. With neither values nor W there's no tie; values without W, W without values, a
    value W registers other than once (with no prefix, by its own name: told to give H1's) and a value that is no Commit
    table are refused."""
    rows = np.arange(4 * K // 2, dtype=np.uint16).reshape(4, K // 2)
    a, other, ids = (REG.Registered.commit("h1/program", v, r.tolist(), schema=REG.SCHEMA_V2).entry()
                     for v, r in ((f"h1/a/{K}@0", rows), (f"h1/a/{K}@1", rows), ("h1/ids@0", rows[:, :4])))
    w = {"format": R.FORMAT_V2, "registered": [a, other, ids], "previous": None}
    values = [f"row-table/0/k{K}", "input-ids/0"]
    reg = {"params": {values[0]: 4 * K // 2, values[1]: 16}, "tables": [], "reads": [], "roots": [], "openings": []}
    doc = {"registered": {**reg, "values": values}}
    tied, pins = RA.tie_of(doc, w, "h1/")
    d = R.digest(w)
    assert tied == {"log": [w], "registered": {values[0]: K // 2, values[1]: 4},
                    "outputs": {values[0]: {"record": d, "value": f"h1/a/{K}@0"}, values[1]: {"record": d, "value": "h1/ids@0"}},
                    "schemas": {v: REG.SCHEMA_V2 for v in values}}
    assert pins == {values[0]: {**a, "value": values[0]}, values[1]: {**ids, "value": values[1]}}
    assert RA.tie_of({}, None, None) == RA.tie_of({"registered": reg}, None, "h1/") == ({}, {})
    named = {"registered": {**reg, "params": {"w": 16}, "values": ["w"]}}
    for circuit, earlier, prefix, why in (
            (doc, None, "h1/", "tie the audit to it"), ({"registered": reg}, w, "h1/", "takes no values"),
            (doc, w, None, "give --tie-prefix"), (doc, {**w, "registered": [a, other]}, "h1/", "as 'h1/ids@0' 0 times"),
            (doc, {**w, "registered": [a, a, ids]}, "h1/", f"as 'h1/a/{K}@0' 2 times"), (doc, w, "g/", "0 times"),
            (named, w, "h1/", "is no Commit name")):
        with pytest.raises(ValueError, match=why):
            RA.tie_of(circuit, earlier, prefix)


def test_a_tie_with_no_prefix_takes_pearls_values_by_their_own_names_in_32_bit_seg_rows():
    """`tie_of` with no prefix ties a circuit cut at its linears, whose values are round s's call c's A rows and y
    (``pearl-a/<s>/<c>``, ``pearl-y/<s>/<c>``: FP32 words in ``row-seg/v1`` rows, `interface.Registered.formats`), to
    Pearl+'s window record W, which registers each under the value's own name: its ``registered`` words are W's entry's
    (K for A, N for y), its ``outputs`` W's entry of that name, its schema ``row-seg/v1`` and its pin W's 32-bit entry.
    A value W doesn't register or registers twice is refused, and so is a prefix (Pearl+'s names are no Commit table)."""
    rows = {"pearl-a/0/0": np.arange(16 * K, dtype=np.uint32).reshape(16, K),
            "pearl-y/0/0": np.arange(5 * 2 * K, dtype=np.uint32).reshape(5, 2 * K)}
    a, y = (REG.Registered.commit("pearl/program", v, r.tolist(), word_bits=32, schema=REG.SCHEMA_SEG).entry()
            for v, r in rows.items())
    w = {"format": R.FORMAT_V2, "registered": [a, y], "previous": None}
    seg = {"row_schema": REG.SCHEMA_SEG, "word_bits": 32}
    doc = {"registered": {"params": {v: 2 * r.size for v, r in rows.items()}, "tables": [], "reads": [], "roots": [],
                          "openings": [], "values": list(rows), "formats": {v: seg for v in rows}}}
    tied, pins = RA.tie_of(doc, w, None)
    d = R.digest(w)
    assert tied == {"log": [w], "registered": {"pearl-a/0/0": K, "pearl-y/0/0": 2 * K},
                    "outputs": {v: {"record": d, "value": v} for v in rows}, "schemas": {v: REG.SCHEMA_SEG for v in rows}}
    assert pins == {"pearl-a/0/0": a, "pearl-y/0/0": y} and a[REG.WIDTH_KEY] == y[REG.WIDTH_KEY] == 32
    for earlier, prefix, why in (({**w, "registered": [a]}, None, "'pearl-y/0/0' as 'pearl-y/0/0' 0 times"),
                                 ({**w, "registered": [a, y, a]}, None, "'pearl-a/0/0' 2 times"),
                                 (w, "h1/", "is no Commit name")):
        with pytest.raises(ValueError, match=why):
            RA.tie_of(doc, earlier, prefix)


def test_a_ties_summary_names_w_and_for_pearls_w_its_development_verifier():
    """The summary's ``tie`` names W's digest and the values tied to it. Tied to Pearl+'s W (its window names a Pearl-C
    scheme) it also says what holds W's roots to the credited ones, Pearl+'s verdict, which reads the prover's salt key and
    opens drawn rows in the clear; tied to H1's W, whose window names none, it doesn't."""
    entry = {"value": "pearl-a/0/0", "root": "00" * 64, "leaves": 16, "words": K, "domain": "00" * 32, REG.WIDTH_KEY: 32}
    outputs = {"pearl-a/0/0": {"record": "", "value": "pearl-a/0/0"}}
    pearl = {"format": R.FORMAT_V2, "window": {"kind": "served", "entry": "pouw:pearl-c-h100-v1-h3",
                                               "scheme": "pearl-c-h100-v1-h3", "salt": "00" * 64},
             "registered": [entry], "previous": None}
    h1 = {**pearl, "window": {"kind": "served", "entry": "toy-window", "open_root": "00" * 64, "refused": []}}
    got = RA.tie_row({"log": [pearl], "outputs": outputs})
    assert got == {"record": R.digest(pearl), "values": ["pearl-a/0/0"], "credited_by": RA.PEARL_CREDIT}
    assert "development verifier" in got["credited_by"] and "salt key" in got["credited_by"]
    assert RA.tie_row({"log": [h1], "outputs": outputs}) == {"record": R.digest(h1), "values": ["pearl-a/0/0"]}


@dataclass(frozen=True)
class _Reused(REG.Registered):
    """W's table as the circuit's value (`pouw_h1_commit.Reused`): W's rows, salts and tree, by the circuit's name of it, so
    its entry is W's but in name; there's no binding to recompute W's domain from."""

    domain_id: bytes = b""

    @property
    def domain(self):
        raise ValueError(f"{self.value}: a reused table has W's domain id alone")

    def entry(self) -> dict:
        return {"value": self.value, "root": self.root.hex(), "leaves": self.leaves, "words": self.words,
                "domain": self.domain_id.hex()}


def _tied(d: Path, n: int = 8, cap: int = 4) -> SimpleNamespace:
    """A window of ``n`` coordinates each reading its ``wrow`` from round 0's A table of K (4 rows, coordinate u row u % 4),
    a value its circuit's program takes (`IF.Registered.values`) and an earlier registration W registered first as
    ``h1/a/<K>@0``: the window's registration is chained right after W and registers the table by W's entry, as
    ``prove_path.py window-commit --window-registration W`` does. W is written beside the window as ``w.json``."""
    rng = np.random.default_rng(2)
    rows = rng.integers(0, 1 << 16, (4, K // 2), dtype=np.uint16)
    theirs = REG.Registered.commit("h1/program", f"h1/a/{K}@0", rows.tolist(), schema=REG.SCHEMA_V2)
    w = {"format": R.FORMAT_V2, "registered": [theirs.entry()], "previous": None}
    name = f"row-table/0/k{K}"
    value = _Reused(**{**{f.name: getattr(theirs, f.name) for f in fields(theirs)}, "value": name},
                    domain_id=theirs.domain.domain_id)
    read = IF.Read(TID, "wrow", name, 0, (n // 4, 4), 0, (0, K // 2))
    circuit = IF.Circuit("test", ((COORD, n),), registered=IF.Registered({name: 4 * K // 2}, reads=(read,), values=(name,)))
    win = IF.Window(circuit, RUN, D.parse_design(LAW), cap, registered={name: value})
    x = rng.integers(-128, 128, (n, K)).astype(np.int8)
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), rows[np.arange(n) % 4].view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)])
    win.write(win.close(), d, {"label": "test"}, registrar=R, previous=R.digest(w))
    (d.parent / "w.json").write_text(json.dumps(w))
    return SimpleNamespace(dir=d, w=w, w_file=d.parent / "w.json", name=name, entry=theirs.entry(),
                           rec=json.loads((d / "registration.json").read_text()))


@pytest.mark.parametrize("text, population, rule", [
    ("stratified:1", 561, "stratified-rate:1/561"), ("subset:3", 40, "rate:3/40"), ("subset:9", 4, "rate:4/4"),
    ("work:2", 10, "work-rate:2/10"), ("bernoulli:1/8", 10, "bernoulli:1/8"),
])
def test_the_audits_rule_is_the_rate_of_the_windows_design_which_a_stream_takes(text, population, rule):
    """A stream's rule is a rate (`service.Law` refuses a fixed count), so the audit states the window's fixed design K of
    N as the rate K/N of its kind (`rule_of`), which over N proof blocks samples K: one a stratum at ``stratified:1``."""
    from verity.protocols.spot.verifier import service as S

    assert RA.rule_of(D.parse_design(text), population) == rule
    assert S.Law(rule).text == rule
    with pytest.raises(ValueError, match="a fixed count"):
        S.Law(text if not text.startswith("bernoulli") else "subset:1")
    if text.startswith("subset"):
        assert S.selection_law(S.Law(rule), population) == {"law": "subset", "k": min(int(text.split(":")[1]), population)}
    if text == "stratified:1":
        assert S.selection_law(S.Law(rule), population, [1, 200, 360]) == D.parse_design(text)


def test_the_audit_ties_a_window_whose_circuit_takes_values_to_the_earlier_registration(tmp_path, monkeypatch, capsys):
    """A window whose circuit takes values (`_tied`) is audited tied to the earlier registration W that registered them
    first (``--tie W --tie-prefix P``), and refused before any session without the tie: the tie hands a5 one-stage's
    ``log``, ``registered``, ``outputs`` and ``schemas`` (`tie_of`), the value's readers read W's table at W's root and
    count registered (the root the window's registration records for the value), the value is the window's own to the
    output class (its readers sampled like any other unit), and the summary names W's digest and the values. The sessions
    (`a5.audit`) are stubbed and sample nothing, so the audit doesn't pass."""
    import a5 as A5
    import a5_coins as AC

    got = _tied(tmp_path / "w")
    assert got.rec["previous"] == R.digest(got.w) and got.rec["registered"] == [{**got.entry, "value": got.name}]
    seen, owned = {}, []

    def audit(a, sets, out, mine, served, source, port, summary, more, **kw):
        seen.update(tie=a.tie, outputs=a.outputs, registered=kw["registered"], declared=a.declared)
        ruled.append((a.rule, A5.rated(mine, a.rule) == mine, mine.population))
        return {"audit": {"accepted": True}}, True

    def output_class(circuit, given=(), own=frozenset()):
        owned.append(set(own))
        return []

    ruled: list = []

    session = SimpleNamespace(skip=lambda r: None, join=lambda: None, evidence=lambda: {}, shutdown=lambda: None)
    monkeypatch.setattr(RA, "free_ports", lambda span: 20000)
    monkeypatch.setattr(A5, "audit", audit)
    monkeypatch.setattr(AC, "start", lambda *a, **kw: (session, session, session))
    monkeypatch.setattr(RA, "output_class", output_class)
    base = ["audit", "--window", str(got.dir), "--law", LAW, "--circuit", got.rec["window"]["circuit"], "--run", RUN,
            "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--skip-negatives", "--challenger",
            "local", "--main", "m.txt"]
    for extra, why in (([], "tie the audit to it"), (["--tie", str(got.w_file)], "give --tie-prefix"),
                       (["--tie", str(got.w_file), "--tie-prefix", "g/"], "0 times")):
        assert RA.main([*base, "--out", str(tmp_path / "refused"), *extra]) == 1
        assert why in json.loads((tmp_path / "refused" / "summary.json").read_text())["refused"]["records"]
        assert not seen and not owned
    assert RA.main([*base, "--out", str(tmp_path / "out"), "--tie", str(got.w_file), "--tie-prefix", "h1/"]) == 1
    tied, pins = RA.tie_of(json.loads((got.dir / "circuit.json").read_text()), got.w, "h1/")
    assert seen["tie"] == tied and tied["log"] == [got.w] and pins == {got.name: {**got.entry, "value": got.name}}
    (rule, same, n), = ruled
    assert rule == f"stratified-rate:2/{n}" and same, "the stream's rule is a rate, and its design the window's own"
    assert tied["outputs"] == {got.name: {"record": R.digest(got.w), "value": f"h1/a/{K}@0"}} and owned == [{got.name}]
    assert len(seen["registered"]) == 2 and all(f["ports"]["wrow"]["root"] == got.entry["root"]
                                                 and f["ports"]["wrow"]["domain"] == got.entry["domain"]
                                                 for f in seen["registered"].values())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["tie"] == {"record": R.digest(got.w), "values": [got.name]}
    named = {**summary["tie"], "previous": R.digest(got.w), "chained": True}
    assert json.loads((tmp_path / "out" / "tie.json").read_text()) == named, "named before any proof, as the TIE line"
    assert f"TIE {json.dumps(named, sort_keys=True)}" in capsys.readouterr().out
    assert not (tmp_path / "refused" / "tie.json").exists()
    assert summary["registered"]["roots"] == summary["registered"]["pins"] == {got.name: got.entry["root"]}
    assert summary["leak"]["registered"] == 8 * 8 * K and "unpinned" not in summary["leak"]
    assert seen["declared"] == {} and "day" not in summary
    from verity.hardware import registration as G
    day = G.chain("remote", [G.Declaration(0, got.rec["program"])])
    (tmp_path / "log.json").write_bytes(G.committed("remote", day))
    named = tmp_path / "w-day.json"
    named.write_text(json.dumps({**got.w, "window": {"day": 0, "declaration": G.Declaration(0, got.rec["program"]).digest()}}))
    seen.clear()
    assert RA.main([*base, "--out", str(tmp_path / "unnamed"), "--tie", str(got.w_file), "--tie-prefix", "h1/", "--day", "0",
                    "--day-log", str(tmp_path / "log.json")]) == 1
    assert "not one of day 0's in --day-log" in json.loads(
        (tmp_path / "unnamed" / "summary.json").read_text())["refused"]["records"] and not seen
    assert RA.main([*base, "--out", str(tmp_path / "day"), "--tie", str(named), "--tie-prefix", "h1/", "--day", "0",
                    "--day-log", str(tmp_path / "log.json")]) == 1
    assert seen["declared"] == {"day": 0, "declarations": [day[0].body()]}
    assert json.loads((tmp_path / "day" / "summary.json").read_text())["day"] == {
        "day": 0, "declared": [got.rec["program"]], "source": RA.DAY_SOURCES["development"]}
    root = RA.verifier_window(got.dir, D.parse_design(LAW), None)[0].run
    lines, _key, record = _device_ledger(tmp_path / "device", [_declared(got.rec["program"], blocks=[]),
                                                               *_window(S1, 1, root)])
    (ledger := tmp_path / "windows.jsonl").write_text("".join(x + "\n" for x in lines))
    device = tmp_path / "w-device.json"
    device.write_text(json.dumps({**got.w, "window": {"day": DAY0,
                                                      "declaration": G.Declaration(DAY0, got.rec["program"]).digest()}}))
    seen.clear()
    assert RA.main([*base, "--out", str(tmp_path / "device-day"), "--tie", str(device), "--tie-prefix", "h1/",
                    "--day-ledger", str(ledger), "--challenger-key", str(record)]) == 1
    assert seen["declared"] == {"day": DAY0, "declarations": [{"blocks": [], "day": DAY0, "program": got.rec["program"]}]}
    assert json.loads((tmp_path / "device-day" / "summary.json").read_text())["day"] == {
        "day": DAY0, "declared": [got.rec["program"]], "source": RA.DAY_SOURCES["device"]}


def test_a_declared_day_holds_the_registrations_program_to_the_days_one_declaration(written, tmp_path):
    """``--day N --day-log LOG`` (`declared_day`): the declarations of a registration log core's check accepts, which R2
    takes with the day. The window's program declared for its day passes R2; a day declared to another program is refused
    (``R2-declared``), and a day the log doesn't declare too (``R2-undeclared``). A log declaring a day twice is core's
    ``duplicate-declaration``, and day options without ``--day`` are refused, each before any session."""
    from verity.hardware import registration as G

    mine, _doc, _sets = RA.verifier_window(written.dir, D.parse_design(LAW), None)
    program, log = written.rec["program"], tmp_path / "log.json"
    entries = G.chain("remote", [G.Declaration(3, program, (TID,)), G.Declaration(4, "0" * 128)])
    log.write_bytes(G.committed("remote", entries))
    assert RA.declared_day(None, None, None) == {}
    day = RA.declared_day(3, str(log), None)
    assert day == {"day": 3, "declarations": [e.body() for e in entries]}

    def r2(d: dict) -> list[str]:
        v = R.check(written.rec, mine.program, mine.query, D.parse_design(LAW), CS.SCHEME, {}, **d)
        return sorted(c for c in v.codes if c.startswith("R2"))
    assert r2(day) == []
    assert r2({**day, "day": 4}) == ["R2-declared"] and r2({**day, "day": 5}) == ["R2-undeclared"]
    twice = tmp_path / "twice.json"
    twice.write_bytes(G.committed("remote", G.chain("remote", [G.Declaration(3, program), G.Declaration(3, "0" * 128)])))
    with pytest.raises(ValueError, match="core's check refuses it, entry 1: duplicate-declaration"):
        RA.declared_day(3, str(twice), None)
    with pytest.raises(ValueError, match="go with --day"):
        RA.declared_day(None, str(log), None)
    with pytest.raises(ValueError, match="needs --day-log"):
        RA.declared_day(3, None, None)


def test_a_declared_day_holds_the_tied_ws_naming_to_a_declaration_of_that_day_in_the_log(tmp_path):
    """Under ``--tie`` W's window names the day's declaration (``day``, ``declaration``: `serve_day.Day.naming`), and
    `declared_day` holds it to the log: W naming day 3's declaration by its digest rebuilt with no ``previous`` passes;
    naming another day, a digest the log doesn't hold for day 3 (day 4's), the log's chained digest of day 3's entry, or a
    day without ``--day`` is refused before any session. A W naming no day under no ``--day`` (Pearl+'s, every window
    without the switch) is as before."""
    from verity.hardware import registration as G

    log = tmp_path / "log.json"
    entries = G.chain("remote", [G.Declaration(3, "5d" * 64), G.Declaration(4, "6e" * 64)])
    log.write_bytes(G.committed("remote", entries))
    d3, d4 = G.Declaration(3, "5d" * 64).digest(), G.Declaration(4, "6e" * 64).digest()
    assert entries[0].previous and entries[0].digest() != d3

    def w(**named: object) -> dict[str, object]:
        return {"format": "toy", "window": {"kind": "served", **named}}
    assert RA.declared_day(3, str(log), None, w(day=3, declaration=d3))["day"] == 3
    assert RA.declared_day(None, None, None, w()) == {}
    for bad in (w(day=4, declaration=d3), w(day=3, declaration=d4), w(day=3, declaration=entries[0].digest()), w()):
        with pytest.raises(ValueError, match="not one of day 3's in --day-log"):
            RA.declared_day(3, str(log), None, bad)
    with pytest.raises(ValueError, match="names day 3's declaration, and the audit has no --day"):
        RA.declared_day(None, None, None, w(day=3, declaration=d3))


DAY0, PIN, S1, S2 = 20746, "c" * 40, "s20261020T093000Z-0001", "s20261020T093000Z-0002"


def _auditor():
    import a5_coins as AC
    return AC._module("a5_auditor", HERE.parents[2] / "verity" / "composite" / "device" / "live_challenger" / "auditor.py")


def _device_ledger(tmp_path: Path, events: list[tuple[str, str | None, dict]]) -> tuple[list[str], object, Path]:
    """The challenger device's own ledger of ``events`` (kind, session, body), chained and signed by the live challenger's
    code (`verity/composite/device/live_challenger/auditor.py`): its lines, its key and the key record ``--challenger-key`` reads."""
    from verity.protocols.spot.device import challenger as Ch
    from verity.commitments.crypto import lms
    A = _auditor()
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("vy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line(PIN, "d" * 40, "2026-10-20T09:00:00Z") + "\n")
    host = A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")
    params = lms.Params(64, 4, 5)
    first, sign, key_sha = A.key_signer(host, params, workers=1)
    for kind, session, body in events:
        A.append(host, kind, session, lambda lines, b=body: {**b, **({"key_sha512": key_sha} if kind == "open" else {})},
                 first, sign)
    public = (host.key / "public").read_bytes()
    record = tmp_path / "challenger-key.json"
    record.write_text(json.dumps({"schema": Ch.KEY_RECORD_SCHEMA, "W": params.W, "H": params.H, "pubkey": public.hex(),
                                  "pubkey_sha512": hashlib.sha512(public).hexdigest()}))
    return [ln.decode() for ln in A.read_ledger(host.ledger)], Ch.ChallengerKey(params.W, params.H, public), record


def _declared(program: str, **kw: object) -> tuple[str, None, dict]:
    return "declaration", None, {"by": "hash", "blocks": ["t"], "day": DAY0, "declared_utc": "2026-10-20T09:30:00Z",
                                 "pinned": PIN, "program": program, **kw}


def _window(session: str, reserve_seq: int, root: str, by: str = "hash", reserved_at: str = "2026-10-20T09:30:00Z",
            expires: str = "2026-10-20T09:40:00Z", opened_at: str = "2026-10-20T09:31:00Z") -> list[tuple[str, str, dict]]:
    """A lane's reserve and v2 open of the window at ``root``, on day DAY0 by its signed ``started_utc`` (``opened_at``)."""
    reserved = {"by": by, "peer": "vy-nebius-1", "scheme": "verity/test/coins/v1", "epoch": 0,
                "families": [["pouw/salt", 1], ["pouw/round", 3]], "minutes": 10, "started_utc": reserved_at,
                "expires_utc": expires}
    opened = {"window": {"root": root}, "reserve": reserve_seq, **{k: v for k, v in reserved.items() if k in _auditor().RESERVED},
              "opened_by": by, "started_utc": opened_at, "host": "vy-challenger-1.x", "pinned": PIN}
    return [("reserve", session, reserved), ("open", session, opened)]


def _verify(key: object):
    import a5_coins as AC
    return AC.lms_verify


def test_the_device_day_is_the_challengers_declared_day_over_its_own_ledger_and_holds_ws_naming_to_its_entry(
        written, tmp_path):
    """``--day-ledger`` (`device_day`) reads the challenger's ``declared_day`` over the device's own signed ledger: the
    UTC day of the window's signed open and the declarations its lane signed before it, which R2 takes in place of
    ``--day``. W's naming is held to the Declaration rebuilt from the device's entry (no ``previous``): the prover log's
    chained digest, or another day's, is refused."""
    from verity.hardware import registration as G
    mine, _doc, _sets = RA.verifier_window(written.dir, D.parse_design(LAW), None)
    program, root = written.rec["program"], "01" * 64
    lines, key, _record = _device_ledger(tmp_path, [_declared(program), *_window(S1, 1, root)])
    day = RA.device_day(lines, root, key, _verify(key))
    assert day == {"day": DAY0, "declarations": [{"blocks": ["t"], "day": DAY0, "program": program}]}
    assert not [c for c in R.check(written.rec, mine.program, mine.query, D.parse_design(LAW), CS.SCHEME, {}, **day).codes
                if c.startswith("R2")]
    rebuilt = G.Declaration(DAY0, program, ("t",)).digest()
    assert RA.device_day(lines, root, key, _verify(key), {"window": {"day": DAY0, "declaration": rebuilt}}) == day
    chained = G.chain("remote", [G.Declaration(DAY0, program, ("t",))])[-1].digest()
    for bad in ({"day": DAY0, "declaration": chained}, {"day": DAY0 + 1, "declaration": rebuilt}, {}):
        with pytest.raises(ValueError, match=f"not one of day {DAY0}'s in the challenger's ledger"):
            RA.device_day(lines, root, key, _verify(key), {"window": bad})


def test_a_window_opened_before_its_days_declaration_or_of_a_lane_that_never_declares_is_r2_undeclared(written, tmp_path):
    """memory-accounting's first two points. A declaration signed after some of the day's windows opened leaves those
    windows with no declaration (``declared_day`` reads only what the lane signed before the open), so R2 refuses them
    (``R2-undeclared``) and W's naming of the later declaration is refused, while a window opened after it passes. A lane
    that never declares isn't gated at the challenger, so its windows open, but under ``--day-ledger`` they read no
    declaration (another lane's binds them not) and R2 refuses them too; with the switch off R2 reads no day."""
    from verity.hardware import registration as G
    mine, _doc, _sets = RA.verifier_window(written.dir, D.parse_design(LAW), None)
    program = written.rec["program"]

    def r2(d: dict) -> list[str]:
        v = R.check(written.rec, mine.program, mine.query, D.parse_design(LAW), CS.SCHEME, {}, **d)
        return sorted(c for c in v.codes if c.startswith("R2"))
    late, key, _ = _device_ledger(tmp_path / "late",
                                  [*_window(S1, 0, "01" * 64), _declared(program), *_window(S2, 3, "02" * 64)])
    before = RA.device_day(late, "01" * 64, key, _verify(key))
    assert before == {"day": DAY0, "declarations": []} and r2(before) == ["R2-undeclared"]
    with pytest.raises(ValueError, match=f"not one of day {DAY0}'s in the challenger's ledger"):
        RA.device_day(late, "01" * 64, key, _verify(key),
                      {"window": {"day": DAY0, "declaration": G.Declaration(DAY0, program, ("t",)).digest()}})
    assert r2(RA.device_day(late, "02" * 64, key, _verify(key))) == []

    never, key, _ = _device_ledger(tmp_path / "never", [_declared(program, by="vllm"), *_window(S1, 1, "01" * 64)])
    got = RA.device_day(never, "01" * 64, key, _verify(key))
    assert got == {"day": DAY0, "declarations": []} and r2(got) == ["R2-undeclared"] and r2({}) == []


def test_the_device_day_reads_the_devices_own_ledger_and_refuses_a_copy_pous_cut(written, tmp_path):
    """memory-accounting's third point: PoUS cuts its copies of the ledger to the entries a window cites
    (`pous.period.cited`), which drops the day's declaration; ``declared_day`` refuses such a copy as a ledger (its chain
    no longer runs from seq 0), so R2 reads the device's own ledger or nothing, never a copy that would read undeclared."""
    from verity.protocols.pous.verifier import period as PP
    program, root = written.rec["program"], "01" * 64
    lines, key, _ = _device_ledger(tmp_path, [_declared(program), *_window(S1, 1, root)])
    cut = [lines[i] for i in PP.cited([json.loads(x) for x in lines], {S1})]
    assert lines[0] not in cut and len(cut) == len(lines) - 1
    with pytest.raises(ValueError, match="the auditor's ledger is refused"):
        RA.device_day(cut, root, key, _verify(key))
    assert RA.device_day(lines, root, key, _verify(key))["declarations"][0]["program"] == program


def test_a_window_that_crosses_utc_midnight_belongs_to_its_opens_day(written, tmp_path):
    """Soundness's midnight fixture: a window whose open the device signs at 23:58 on day D and whose end (its steps
    and close) falls on D+1 is a window of day D, its open's ``started_utc``. Declared for D, R2 accepts it under D,
    though the lane declares D+1 after midnight while it runs, and W's naming of D+1's declaration is refused. Declared
    only for D+1, after midnight, it reads no declaration and R2 refuses it (``R2-undeclared``). Declared only for D+1
    ahead of the open, the device would have refused the open, and a ledger holding it is refused."""
    from verity.hardware import registration as G
    mine, _doc, _sets = RA.verifier_window(written.dir, D.parse_design(LAW), None)
    program, root = written.rec["program"], "01" * 64

    def r2(d: dict) -> list[str]:
        v = R.check(written.rec, mine.program, mine.query, D.parse_design(LAW), CS.SCHEME, {}, **d)
        return sorted(c for c in v.codes if c.startswith("R2"))
    def late(reserve_seq: int) -> list[tuple[str, str, dict]]:
        return _window(S1, reserve_seq, root, reserved_at="2026-10-20T23:55:00Z", expires="2026-10-21T00:05:00Z",
                       opened_at="2026-10-20T23:58:00Z")
    next_day = _declared(program, day=DAY0 + 1, declared_utc="2026-10-21T00:01:00Z")
    end = ("end", S1, {"rc": 0, "closed": True, "ended": "closed by the client", "files": {}, "released": None,
                       "complete": False, "record_leaf": None, "commitment": None, "refused_hellos": 0,
                       "ended_utc": "2026-10-21T00:04:00Z"})

    lines, key, _ = _device_ledger(tmp_path / "d", [_declared(program), *late(1), next_day, end])
    got = RA.device_day(lines, root, key, _verify(key))
    assert got == {"day": DAY0, "declarations": [{"blocks": ["t"], "day": DAY0, "program": program}]} and r2(got) == []
    on_d = G.Declaration(DAY0, program, ("t",)).digest()
    assert RA.device_day(lines, root, key, _verify(key), {"window": {"day": DAY0, "declaration": on_d}}) == got
    with pytest.raises(ValueError, match=f"not one of day {DAY0}'s in the challenger's ledger"):
        RA.device_day(lines, root, key, _verify(key),
                      {"window": {"day": DAY0 + 1, "declaration": G.Declaration(DAY0 + 1, program, ("t",)).digest()}})

    lines, key, _ = _device_ledger(tmp_path / "d1", [*late(0), next_day, end])
    got = RA.device_day(lines, root, key, _verify(key))
    assert got == {"day": DAY0, "declarations": []} and r2(got) == ["R2-undeclared"]

    ahead = _declared(program, day=DAY0 + 1, declared_utc="2026-10-20T23:50:00Z")
    lines, key, _ = _device_ledger(tmp_path / "ahead", [ahead, *late(1), end])
    with pytest.raises(ValueError, match=f"opens on day {DAY0}, which its lane hash hasn't declared"):
        RA.device_day(lines, root, key, _verify(key))


def _history(d: Path, n: int = 4) -> SimpleNamespace:
    """``n`` units of ``Gemm_v2{K=16, N=8}``, two a member, their outputs history table ``latent``'s rows (capacity 16),
    the window written to ``d`` with the members' public bodies."""
    from verity.commitments import history as HT
    g8 = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
    gid, rng = descriptor_id(g8), np.random.default_rng(3)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    x, w = bf16(rng.standard_normal((n, 16))), bf16(rng.standard_normal((n, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(g8, x, w, kernel="reference"), dtype=np.uint16).reshape(n, 8)
    table = HT.HistoryTable("program", "latent", 8, 16)
    table.append(y)
    runs = ({"start": 0, "count": n, "writer": gid, "out": "o0", "radix": [n], "base": 0, "step": [1]},)
    circuit = IF.Circuit("history", ((g8, n),),
                         registered=IF.Registered({}, histories=(IF.History("latent", "program", gid, "o0", 8, runs, 16),)))
    win = IF.Window(circuit, RUN, D.parse_design(LAW), 2, registered={"latent": table})
    win.record(g8, [x, w], [y])
    win.write(win.close(), d, {"label": "test"}, registrar=R)
    return SimpleNamespace(dir=d, table=table, rec=json.loads((d / "registration.json").read_text()),
                           publics={m.key: (d / f"m{k}-public.bin").read_bytes() for k, m in enumerate(win.window.members)})


def test_a5_holds_a_history_tables_registered_root_to_the_one_over_its_writers_bodies(tmp_path, monkeypatch):
    """A window with a history table (`interface.History`) is audited with a5 handed the table as the verifier of record
    takes it (``history``, `ml.windows.history.params`): from the window's public bodies it computes the table's root, to
    which it holds the root the window's registration records, the root the table is pinned at (`history.pins`). A
    registration that records no root for the table is refused before any session. The sessions (`a5.audit`) are stubbed
    and sample nothing, so the audit doesn't pass."""
    import a5 as A5
    import a5_coins as AC
    from verity.commitments import history as HT

    got, seen = _history(tmp_path / "w"), {}

    def audit(a, sets, out, mine, served, source, port, summary, more, **kw):
        seen.update(history=kw["history"])
        return {"audit": {"accepted": True}}, True

    session = SimpleNamespace(skip=lambda r: None, join=lambda: None, evidence=lambda: {}, shutdown=lambda: None)
    monkeypatch.setattr(RA, "free_ports", lambda span: 20000)
    monkeypatch.setattr(A5, "audit", audit)
    monkeypatch.setattr(AC, "start", lambda *a, **kw: (session, session, session))
    monkeypatch.setattr(RA, "output_class", lambda circuit, given=(), own=frozenset(): [])
    base = ["audit", "--window", str(got.dir), "--law", LAW, "--circuit", got.rec["window"]["circuit"], "--run", RUN,
            "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--skip-negatives", "--challenger",
            "local", "--main", "m.txt"]
    assert RA.main([*base, "--out", str(tmp_path / "out")]) == 1
    hist = seen["history"](got.publics)
    assert hist["registered"] == {"latent": got.table.root.hex()} == IF.registered_roots(got.rec["window"])
    assert HT.roots_of(CS.leaf_layer(got.publics), hist) == {"latent": (got.table.root, got.table.domain.domain_id)}
    seen.clear()
    rec = {**got.rec, "window": {**got.rec["window"], "registered_roots": {}}}
    (got.dir / "registration.json").write_text(json.dumps(rec))
    assert RA.main([*base, "--out", str(tmp_path / "refused")]) == 1
    assert "records no root" in json.loads((tmp_path / "refused" / "summary.json").read_text())["refused"]["records"]
    assert not seen


def _audited(window: Path, out: Path, registration: dict | None = None, accepted: bool = True) -> Path:
    """What the records audit leaves under its ``--out`` ``out`` for ``window``: the population Program it sampled, its
    summary, and an audit record of ``registration`` (default the window's) accepting a subset sample of 2 proof blocks;
    or, not ``accepted``, a check before the sample that failed."""
    mine, _doc, _sets = RA.verifier_window(window, D.parse_design(LAW), None)
    rec = registration or json.loads((window / "registration.json").read_text())
    (out / "window").mkdir(parents=True)
    (out / "window" / "program.json").write_bytes(canonical_json(encode_program(mine.program)))
    if not accepted:
        (out / "summary.json").write_text(json.dumps({"checks_before_draw": {"registration": False}}))
        return out
    (out / "summary.json").write_text(json.dumps({"checks_before_draw": {"registration": True},
                                                  "fixed_before_draw": {"ok": True}}))
    (out / "window" / "audit.json").write_text(json.dumps({
        "accepted": True, "registration": rec, "proved_units": [1, 5], "checks": {"draw": {"ok": True}},
        "draw": {"law": "subset", "k": 2, "population": rec["population"], "units": [1, 5]},
        "wrong_units": {"delta": 0.5, "bound": 3}}))
    return out


def test_a_windows_audit_is_a_period_audit_whose_scan_state_reads_count_registered(tmp_path, capsys):
    """`period`: a window whose circuit reads its own scan state (`RA.TIED`) gives the network certifier's period its
    registration, the Program the audit sampled (the one the registration names), the audit's answer, ``steps`` 0 and
    ``free``: the audit's count, but with the state's reads registered (the period ties them to its scan's state tree),
    where the audit's own count leaves them free. ``records_audit.py period`` writes the same and says it in one line.
    Refused by name: a window that commits no scan state, a circuit the registration doesn't bind, an audit that wrote no
    Program or another than the registration names, and an answer committing to another registration."""
    got = _registered(tmp_path / "w", name=RA.TIED, own=True)
    audit = _audited(got.dir, tmp_path / "a")
    rec, doc = (json.loads((got.dir / f).read_text()) for f in ("registration.json", "circuit.json"))
    p = RA.period(got.dir, audit, {})
    per_unit = sum(16 * w for _p, w in IF.layout(COORD).rows)
    assert p["free"] == 8 * per_unit - 8 * 8 * K and p["steps"] == 0
    assert p["registration"] == rec and p["answer"] == SI.answer(audit, rec) and p["answer"]["verdict"] == "accept"
    assert program_digest(p["program"]) == rec["program"] and p["answer"]["commitment"] == R.digest(rec)
    ties, untied = RA.window_pins(rec["window"], doc)
    assert ties == {} and list(untied) == [RA.TIED]
    audited = RA.declared_count(doc, 0, pins=RA.pin_roots(untied), roots=RA.window_roots(rec["window"]), untied=set(untied))
    assert audited["free"] == 8 * per_unit and audited["untied"] == [RA.TIED]
    capsys.readouterr()
    assert RA.main(["period", "--window", str(got.dir), "--audit", str(audit), "--out", str(tmp_path / "p.json")]) == 0
    assert json.loads((tmp_path / "p.json").read_text()) == p
    assert json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix("PERIOD ")) == {
        "verdict": "accept", "reason": None, "free": p["free"], "commitment": R.digest(rec), "program": rec["program"]}
    refused = RA.period(got.dir, _audited(got.dir, tmp_path / "r", accepted=False), {})["answer"]
    assert refused["verdict"] == "reject" and refused["commitment"] == R.digest(rec)

    other = _registered(tmp_path / "x")
    unbound = tmp_path / "u"
    shutil.copytree(got.dir, unbound)
    (unbound / "circuit.json").write_text(json.dumps({**doc, "name": "other"}))
    bare, renamed = tmp_path / "b", tmp_path / "n"
    for d in (bare, renamed):
        shutil.copytree(audit, d)
    (bare / "window" / "program.json").unlink()
    (renamed / "window" / "program.json").write_text(json.dumps({**p["program"], "name": "other"}))
    for window, audited_by, why in [
            (other.dir, _audited(other.dir, tmp_path / "xa"), "commits no verity/certifier/scan-state/v1 its circuit reads"),
            (unbound, audit, "is not the circuit record the registration binds"),
            (got.dir, bare, "wrote no Program"),
            (got.dir, renamed, "isn't the one the registration names"),
            (got.dir, _audited(got.dir, tmp_path / "o", {**rec, "window": {**rec["window"], "label": "other"}}),
             "commits to another registration than the window's")]:
        with pytest.raises(ValueError, match=why):
            RA.period(window, audited_by, {})
    assert RA.main(["period", "--window", str(unbound), "--audit", str(audit), "--out", str(tmp_path / "q.json")]) == 1
    assert "records_audit.py period: " in capsys.readouterr().err and not (tmp_path / "q.json").exists()


#: a `ScanSlots_v2` of two lanes on a 40-word state row (`test_io_shell`'s)
SLOTS = {"NS": 40, "NP": 4, "IDB": 17, "K0": 2, "K": 3,
         "OCC": (3, 300), "LP": (40, 340), "MAX": (80, 380), "IDS": (120, 420)}


def _io_shell_window(d: Path, n: int = 8, cap: int = 4, shell: int = 3) -> SimpleNamespace:
    """`_registered`'s window of ``n`` coordinates reading the window's own scan state (`RA.TIED`), with ``shell`` units
    of an io shell's `ScanSlots_v2` at `SLOTS` beside them, on random state rows, each with the outputs its Definition
    gives it."""
    from verity.definitions.registry import boolean_io_shell as BX

    fn = bind(BX.ScanSlotsV2, **SLOTS)
    state = np.random.default_rng(2).integers(0, 1 << 16, (shell, SLOTS["NS"]), dtype=np.uint16)
    got = np.asarray(evaluate_words(fn, state.astype(np.uint32)), dtype=np.uint32).reshape(shell, -1)
    outs = np.split(got, np.cumsum([count // 2 for _o, count in IF.layout(fn).outs])[:-1], axis=1)
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 1 << 16, (4, K // 2), dtype=np.uint16)
    value = REG.Registered.commit("test/model", RA.TIED, rows.tolist(), schema=REG.SCHEMA_V2, window=RUN)
    read = IF.Read(TID, "wrow", RA.TIED, 0, (n // 4, 4), 0, (0, K // 2))
    registered = IF.Registered({RA.TIED: 4 * K // 2}, reads=(read,))
    win = IF.Window(IF.Circuit("test", ((COORD, n), (fn, shell)), registered=registered), RUN, D.parse_design(LAW), cap,
                    registered={RA.TIED: value})
    x = rng.integers(-128, 128, (n, K)).astype(np.int8)
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), rows[np.arange(n) % 4].view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)])
    win.record(fn, [state], [np.ascontiguousarray(o) for o in outs])
    assert win.mismatches() == {}
    entry = {"program": "test/model", "values": {RA.TIED: value.entry()}}
    win.write(win.close(), d, {"label": "test", "activations": entry}, registrar=R)
    return SimpleNamespace(dir=d, shell=fn)


def test_a_period_counts_an_io_shell_window_in_a_process_that_loaded_no_registry_definition(tmp_path):
    """``records_audit.py period`` as V3c runs it, in a process of its own, on a window whose circuit holds an io
    shell's `ScanSlots_v2` beside the coordinates that read its scan state. `ScanSlots_v2` is a registry Definition,
    which `interface.resolve`'s own walk doesn't import, so the period resolves the circuit's templates in Verity's
    whole catalog (`catalog_resolve`): it counts the shell's state rows free beside the coordinates' own, where
    `interface.resolve` alone refuses the window ("names no Definition in Verity's catalog")."""
    got = _io_shell_window(tmp_path / "w")
    audit = _audited(got.dir, tmp_path / "a")
    p = subprocess.run([sys.executable, str(HERE.parent / "records_audit.py"), "period", "--window", str(got.dir),
                        "--audit", str(audit), "--out", str(tmp_path / "p.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    coord, state = (sum(16 * w for _p, w in IF.layout(fn).rows) for fn in (COORD, got.shell))
    period = json.loads((tmp_path / "p.json").read_text())
    assert period["free"] == 8 * coord - 8 * 8 * K + 3 * state and period["answer"]["verdict"] == "accept"


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
    mine, doc = IF.of_records(written.dir, D.parse_design(LAW), IF.resolve)
    assert not mine.row_tables
    monkeypatch.setattr(IF, "of_records", lambda d, law, resolve: (SimpleNamespace(row_tables=("t0", "t1")), doc))
    with pytest.raises(ValueError, match="2 row tables.*segment roots.*refused until they do"):
        RA.verifier_window(written.dir, D.parse_design(LAW), None)


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
    receipt = {"registration": R.digest(rec), "verifier": A5.CHALLENGER, "received": 1}
    assert RA.fixed_before_draw(receipt, rec, written.bound)["ok"]
    assert RA.fixed_before_draw({**receipt, "registration": "00" * 64}, rec, written.bound)["codes"] == ["after-the-draw"]
    assert RA.fixed_before_draw({**receipt, "verifier": "another stream"}, rec, written.bound)["codes"] == ["after-the-draw"]
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
    with pytest.raises(SystemExit, match="2"):
        RA.main(args)

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
    the audit reserves one a member, not a rank's 16 (`a5.RANK_PORTS`), and a sample needing more than ``--ports`` is
    refused before any session listens, since the next one could reach another audit's verifier."""
    import a5 as A5
    import a5_coins as AC

    got = _write(tmp_path / "w", n=20, cap=1)
    mine, _doc, _sets = RA.verifier_window(got.dir, D.parse_design(LAW), None)
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

    session = SimpleNamespace(skip=lambda r: None, join=lambda: None, evidence=lambda: {}, shutdown=lambda: None)
    monkeypatch.setattr(RA, "free_ports", free_ports)
    monkeypatch.setattr(A5, "audit", audit)
    monkeypatch.setattr(AC, "start", lambda *a, **kw: (session, session, session))
    args = ["audit", "--window", str(got.dir), "--law", LAW, "--circuit", got.rec["window"]["circuit"], "--run", RUN,
            "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "out"),
            "--challenger", "local", "--main", "m.txt"]
    for extra, want in (([], {"span": 20, "port": 20000, "ports": 20}), (["--port", "10000", "--ports", "16"],
                                                                         {"port": 10000, "ports": 16})):
        seen.clear()
        with pytest.raises(Opened):
            RA.main([*args, *extra])
        assert seen == want


def test_the_audit_counts_a_listed_record_with_window_ingress_and_prints_its_window_figure_with_its_premise(
        written, tmp_path, monkeypatch, capsys):
    """A record that lists its members and declares its window's ingress (here each coordinate's ``x`` and ``sx``, rows
    before the window) is counted by `declared_count`, not `leak_count`: the audit's ``RECORDS`` line and ``summary.json``
    carry the run figure, the window ingress still free in it, and beside it the window figure with its premise
    (`WINDOW_PREMISE`). The sessions (`a5.audit`) are stubbed and sample nothing, so the audit doesn't pass; its leak count
    is the records' alone."""
    import a5 as A5
    import a5_coins as AC

    mine, stated, _sets = RA.verifier_window(written.dir, D.parse_design(LAW), None)
    ingress = [{"reader": TID, "port": p, "at": 0, "units": 5, "step": 0} for p in ("x", "sx")]
    doc = IF.circuit_record(mine, stated["name"], stated["templates"], stated["statements"], stated["external"],
                            declared={IF.WINDOW_INGRESS: ingress})
    body = IF.circuit_bytes(doc)
    (written.dir / "circuit.json").write_bytes(body)
    rec = {**written.rec, "window": {**written.rec["window"], "circuit": hashlib.sha512(body).hexdigest()}}
    (written.dir / "registration.json").write_text(json.dumps(rec))
    assert "cap" not in doc and IF.of_records(written.dir, D.parse_design(LAW), IF.resolve)[0].members == mine.members
    session = SimpleNamespace(skip=lambda r: None, join=lambda: None, evidence=lambda: {}, shutdown=lambda: None)
    monkeypatch.setattr(RA, "free_ports", lambda span: 20000)
    monkeypatch.setattr(A5, "audit", lambda *a, **kw: ({"audit": {"accepted": True}}, True))
    monkeypatch.setattr(AC, "start", lambda *a, **kw: (session, session, session))
    capsys.readouterr()
    assert RA.main(["audit", "--window", str(written.dir), "--law", LAW, "--circuit", rec["window"]["circuit"], "--run", RUN,
                    "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "out"),
                    "--skip-negatives", "--challenger", "local", "--main", "m.txt"]) == 1
    rows = dict(IF.layout(COORD).rows)
    inputs, window = 5 * 16 * sum(rows.values()), 5 * 16 * (rows["x"] + rows["sx"])
    figure = {"ingress": window, "free": inputs - window, "other_inputs": inputs - window, "bound": "vacuous",
              "premise": RA.WINDOW_PREMISE}
    printed = [json.loads(x.removeprefix("RECORDS ")) for x in capsys.readouterr().out.splitlines() if x.startswith("RECORDS ")]
    assert [p["leak"] for p in printed] == [{"other_inputs": inputs, "free": inputs, "registered": 0, "advice_bits": 0,
                                             "bound": "vacuous", "window": figure}]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["leak"] == RA.declared_count(doc, 0) and summary["leak"]["window"] == figure


def test_warm_fills_the_compose_cache_the_audit_then_takes(written, tmp_path, monkeypatch, capsys):
    """``warm`` hands `a5.warm` the verifier's window and its composers, each saying what it composes from (the record's
    statement under the stager's label, which a5's cache keys), with ``--compose-cache`` and ``--jobs``; the audit hands a5
    that cache, and none without ``--compose-cache``."""
    import a5 as A5
    import a5_coins as AC

    seen = {}

    def warm(a, sets, mine, jobs=1):
        seen.update(cache=a.compose_cache, jobs=jobs, members=[m.template for m in mine.members],
                    inputs={t: c.inputs for t, (c, _fn) in sets.items()})
        return [{"template": TID, "seconds": 1.0, "kept": False, "pin": "ab", "k_log": 3}]

    monkeypatch.setattr(A5, "warm", warm)
    cache = str(tmp_path / "cache")
    assert RA.main(["warm", "--window", str(written.dir), "--law", LAW, "--compose-cache", cache, "--jobs", "2"]) == 0
    assert seen == {"cache": cache, "jobs": 2, "members": [TID, TID],
                    "inputs": {TID: ["unit/window", written.doc["statements"][TID]]}}
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])["template"] == TID and json.loads(lines[-1].removeprefix("WARM "))["templates"] == 1

    class Opened(Exception):
        pass

    def audit(a, *args, **kw):
        seen["audit"] = a.compose_cache
        raise Opened

    session = SimpleNamespace(skip=lambda r: None, join=lambda: None, evidence=lambda: {}, shutdown=lambda: None)
    monkeypatch.setattr(A5, "audit", audit)
    monkeypatch.setattr(AC, "start", lambda *a, **kw: (session, session, session))
    monkeypatch.setattr(RA, "free_ports", lambda span: 20000)
    monkeypatch.setattr(AC, "start", lambda *a, **kw: (session, session, session))
    args = ["audit", "--window", str(written.dir), "--law", LAW, "--circuit", written.rec["window"]["circuit"], "--run", RUN,
            "--m0-pythonpath", "", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "out"),
            "--skip-negatives", "--challenger", "local", "--main", "m.txt"]
    for extra, want in ((["--compose-cache", cache], cache), ([], None)):
        with pytest.raises(Opened):
            RA.main([*args, *extra])
        assert seen["audit"] == want


def test_a_table_reads_composer_is_keyed_by_the_verifiers_executable_and_no_other_composer_is(tmp_path, monkeypatch):
    """A table read's unit is the template the verifier's executable prints (`read_tmpl`), and that executable is no file of
    the tree: its composer's ``inputs`` hold the executable's digest, by `compose_circuit.template_inputs`'s rule
    (`read_tmpl.is_read`, `read_tmpl.digest`), so a5's key for it (`a5.compose_key`) moves with the executable and comes back
    with it, and a compose cache kept under another build is not taken. Every other template's composer never asks for the
    executable, and its inputs and its key are the statement's alone; the Definition decides, not the statement's form. The
    read here is the toy template under the read's name (`read_tmpl.READ`), which is all `is_read` asks of a Definition."""
    import a5 as A5
    import compose_circuit as CM
    from verity_flock import read_tmpl as RT

    st, flat = {"layout": "iv/hybrid/4", "structural": True}, {"reads": "inline"}
    with monkeypatch.context() as m:
        m.setattr(RT, "READ", COORD.base_id)
        assert RT.is_read(COORD)
        builds, keys = [tmp_path / "flock-read-tmpl-1", tmp_path / "flock-read-tmpl-2"], []
        for exe, text in zip((*builds, builds[0]), (b"one build", b"another build", b"one build")):
            exe.write_bytes(text)
            m.setattr(RT, "EXE", str(exe))
            compose = RA.composer(COORD, st)
            assert compose.inputs == ["unit/window", st, hashlib.sha512(text).hexdigest()]
            keys.append(A5.compose_key(compose, COORD, TID))
        assert keys[0] == keys[2] != keys[1]
        assert RA.composer(COORD, flat).inputs == ["unit/window", flat, hashlib.sha512(b"one build").hexdigest()]
    assert not RT.is_read(COORD)
    monkeypatch.setattr(RT, "digest", lambda: pytest.fail("a template that reads no table asked for the verifier's executable"))
    compose = RA.composer(COORD, flat)
    assert compose.inputs == ["unit/window", flat] and RA.composer(COORD, st).inputs == ["unit/window", st]
    assert A5.compose_key(compose, COORD, TID) == CM.cache_key(
        ["a5/own", TID, program_digest(A5.Program(COORD)), ["unit/window", flat]], files=(Path(CM.__file__), Path(RA.__file__)))
    assert A5.compose_key(compose, COORD, TID) not in keys


def test_the_audit_from_the_records_imports_no_vllm(written, tmp_path):
    """The verifier's window, the compile check, `source`, the negatives and the sample-order check, in a fresh process: no
    `verity_vllm` module is loaded."""
    compilers = tmp_path / "compilers.json"
    compilers.write_text(json.dumps(_compilers(tmp_path, written.doc)))
    (tmp_path / "opening.json").write_text(json.dumps(written.opening))
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(HERE.parent)!r})
import records_audit as RA
from verity.protocols.spot.verifier.one_stage import draw as D, registered as REG, registration as R
import a5 as A5
d, t = Path({str(written.dir)!r}), Path({str(tmp_path)!r})
mine, doc, sets = RA.verifier_window(d, D.parse_design({LAW!r}), None)
rec = json.loads((d / "registration.json").read_text())
compilers, opening = json.loads((t / "compilers.json").read_text()), json.loads((t / "opening.json").read_text())
v, bound = RA.compile_check(doc, rec["window"], opening, compilers, t / "c")
held = {{"run": mine.run, "circuit": rec["window"]["circuit"], "compiled": v["ok"], **bound, "compilers": compilers,
        "opening": opening, "output": json.loads((t / "c" / "compiled.json").read_text())}}
assert v["ok"] and RA.source(rec["window"], held)["ok"]
assert all(x["refused"] for x in RA.negatives(rec["window"], held, doc, t / "n").values())
receipt = {{"registration": R.digest(rec), "verifier": A5.CHALLENGER}}
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
    window = IF.Window(IF.Circuit("test", ((COORD, 5),)), "", D.parse_design(LAW), 4).window
    monkeypatch.setattr(SRW, "population", lambda progs, partition, run, dirs, **kw: (window, [[]]) if run == "" else None)
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
    assert IF.window_of_record(out, D.parse_design(LAW), IF.resolve).partition_digest == window.partition_digest
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
