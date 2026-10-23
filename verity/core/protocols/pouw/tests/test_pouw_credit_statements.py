"""`credit.window` and `credit.period_credit` on a window's statements, each call's proof files (PRF-132): credit runs the
verifier of record itself (`one_stage.proofs.verify`) on the circuit file the period's open committed for the call's K,
with the partition object and the ``--registered`` file it builds from the committed descriptor and the window's entries,
and credits nothing from a verdict the evidence carries.

The verifier here is a stand-in `flock-verify` (`FAKE`), so the test runs without a C-Flock proof of an H1 window: it
answers as the Lean verifier does, one ``VERDICT`` line a session, and accepts a session only when its proof is the
SHA-512 of every file it was handed (the circuit file, the public file, the program, the partition object and the
``--registered`` file) and the units its session drew. `prove` makes those proofs with the files a prover would hold,
the ``--registered`` file `h1window.table_reads_file`'s form, so a window is credited only when credit handed the
verifier the files the prover proved against. `sampled_proofs/tests/test_proofs.py` holds the real verifier to
rejecting tampered proof bytes and to the circuit file's digest; `test_pouw_credit.py` holds `credit.registered_file`
to `h1window.table_reads_file` byte for byte."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

PKG = Path(__file__).resolve().parents[1] / "lean/check/tests"
for _name in ("pouw_rounds_cases", "pouw_credit_cases"):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, PKG / f"{_name}.py")
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
cases, rc = sys.modules["pouw_credit_cases"], sys.modules["pouw_rounds_cases"]

SEEDS = range(3)
FAKE = f"""#!{sys.executable}
import hashlib, json, sys
from pathlib import Path
a = sys.argv[1:]
opt = {{a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and a[i] != "--session"}}
sessions = a[a.index("--session") + 1:]
files = [Path(opt[k]).read_bytes() for k in ("--circuit", "--public", "--program", "--partition", "--registered")]
bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
reg = json.loads(files[4])
for s in sessions:
    rec = json.loads((Path(s) / "session.json").read_text())
    ok = rec["proof"] == hashlib.sha512(bound + json.dumps(rec["unit_draw"]["units"]).encode()).hexdigest()
    print("VERDICT " + json.dumps({{"session": s, "accepted": ok, "why": "" if ok else "the proof does not verify",
                                   "program_sha512": hashlib.sha512(files[2]).hexdigest(),
                                   "registered": {{p: {{"value": e["value"], "root": e["root"]}}
                                                  for p, e in reg["ports"].items()}} if ok else {{}}}}))
"""


@pytest.fixture
def verifier(tmp_path, monkeypatch) -> Path:
    exe = tmp_path / "flock-verify"
    exe.write_text(FAKE)
    exe.chmod(0o755)
    monkeypatch.setattr(PF, "executable", lambda: exe)
    return exe


def circuit(k: int) -> bytes:
    return f"circuit file of K = {k}'s template\n".encode()


def files_of(layout: dict) -> dict:
    """The period-open's `credit.CIRCUIT_FILES`: each K's circuit file's SHA-512."""
    return {f"k{k}": hashlib.sha512(circuit(k)).hexdigest() for _n, k, _c in layout["linears"]}


def _layout(d: dict) -> dict:
    return Cr.layout_of(bytes.fromhex(d["descriptor"]))[0]


def _units(d: dict, k: int) -> list[int]:
    got, why = Cr.drawn(d["draw"]["coin"]["coin"], d["window"]["root"], d["draw"]["root"],
                        {"receipts": d["draw"]["receipts"]}, _layout(d), k)
    assert why is None, why
    return got


def prove(d: dict, k: int, to: Path) -> list[dict]:
    """The window's statements as an honest prover delivers them: for each call with a drawn unit, its K's circuit file,
    a public file whose instances are the call's drawn units, and one session drawing all of them, its proof bound to
    the files the verifier is handed (the call's ``--registered`` file as `h1window.table_reads_file` writes it, at the
    roots the window registers)."""
    raw, lay = bytes.fromhex(d["descriptor"]), _layout(d)
    program, shape = hashlib.sha512(raw).hexdigest(), Cr.shape_of(raw)
    roots = {}
    for e in d["entries"]:
        roots.setdefault(e["value"], []).append(e["root"])
    sizes = [size for size, _w in Cr.call_work(lay)]
    by_call: dict[int, list[int]] = {}
    for u in _units(d, k):
        c = next(c for c in range(len(sizes)) if sum(sizes[:c]) <= u < sum(sizes[:c + 1]))
        by_call.setdefault(c, []).append(u)
    ks = {name: kk for name, kk, _n in lay["linears"]}
    out = []
    for c, units in sorted(by_call.items()):
        base = to / f"call-{c}"
        (base / "session").mkdir(parents=True)
        (base / "circuit.txt").write_bytes(circuit(ks[lay["calls"][c][0]]))
        (base / "pub.bin").write_bytes(json.dumps({"units": {"indices": units}}).encode() + b"\n")
        reg, why = Cr.registered_file(program, lay, shape, c, roots, d["window"]["root"], len(d["rounds"]))
        assert why is None, why
        files = [base.joinpath("circuit.txt").read_bytes(), base.joinpath("pub.bin").read_bytes(), raw,
                 json.dumps(Cr.partition_of(program, shape)).encode(), json.dumps(reg).encode()]
        bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
        drew = list(range(len(units)))
        (base / "session" / "session.json").write_text(json.dumps(
            {"unit_draw": {"units": drew}, "proof": hashlib.sha512(bound + json.dumps(drew).encode()).hexdigest()}))
        out.append({"call": c, "circuit": str(base / "circuit.txt"), "public": str(base / "pub.bin"),
                    "sessions": [str(base / "session")]})
    return out


def window(seed: int, tmp: Path) -> tuple[dict, list[dict]]:
    d = cases.window_instance(random.Random(f"statements/{seed}"), "honest")
    return d, prove(d, d["k"], tmp)


def credit(d: dict, statements, files=None):
    return Cr.window(*(d[k] for k in cases.WINDOW_ARGS[:-2]), statements,
                     files_of(_layout(d)) if files is None else files, d["k"], rc.lms_verify)


def _credited(d: dict) -> list[int]:
    return [i for i, row in enumerate(d["calls"]) if row[Cr.CREDIT]]


@pytest.mark.parametrize("seed", SEEDS)
def test_an_honest_windows_statements_credit_its_bits(seed, tmp_path, verifier):
    d, statements = window(seed, tmp_path)
    assert credit(d, statements) == (_credited(d), None)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_window_carrying_verdicts_and_no_proof_files_is_refused(seed, tmp_path, verifier):
    """The evidence carries proof files, never verdicts: the drawn units' accepted outcomes in place of the statements,
    or a statement carrying a verdict beside its files, credit nothing."""
    d, statements = window(seed, tmp_path)
    assert credit(d, d["proofs"]) == (None, ("proof", Cr.STATEMENTS_FORM))
    assert credit(d, [{**statements[0], "verdict": "accept"}, *statements[1:]]) == (None, ("proof", Cr.STATEMENTS_FORM))
    assert credit(d, [{k: v for k, v in s.items() if k != "sessions"} for s in statements]) == (
        None, ("proof", Cr.STATEMENTS_FORM))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("part", ["proof", "public", "drawn-units"])
def test_tampered_proof_files_fail_the_window(seed, part, tmp_path, verifier):
    """One byte of a session's proof or of its public file changed: the verifier of record rejects the session, so the
    window is refused whole (no partial credit, no retry)."""
    d, statements = window(seed, tmp_path)
    s = random.Random(seed).choice(statements)
    if part == "proof":
        p = Path(s["sessions"][0]) / "session.json"
        rec = json.loads(p.read_text())
        rec["proof"] = ("0" if rec["proof"][0] != "0" else "1") + rec["proof"][1:]
        p.write_text(json.dumps(rec))
    elif part == "public":
        p = Path(s["public"])
        p.write_bytes(p.read_bytes().replace(b"\n", b" \n"))
    else:
        p = Path(s["sessions"][0]) / "session.json"
        rec = json.loads(p.read_text())
        rec["unit_draw"]["units"] = rec["unit_draw"]["units"][:-1] or [1]
        p.write_text(json.dumps(rec))
    got, refused = credit(d, statements)
    assert got is None and refused[0] == "failed", refused
    assert refused[1].startswith(f"call {s['call']}'s session ") and "no partial credit" in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_circuit_file_other_than_the_committed_one_is_refused_before_the_verifier_runs(seed, tmp_path, verifier,
                                                                                          monkeypatch):
    d, statements = window(seed, tmp_path)
    files = files_of(_layout(d))
    k = next(iter(files))
    monkeypatch.setattr(PF, "executable", lambda: (_ for _ in ()).throw(AssertionError("the verifier ran")))
    for other in ({**files, k: "ab" * 64}, {x: v for x, v in files.items() if x != k}):
        got, refused = credit(d, statements, other)
        assert got is None and refused[0] == "circuit", refused
        assert ("is not the one the auditor committed" in refused[1] if k in other
                else refused[1] == Cr.no_statement_files(k[1:]))
    assert credit(d, statements, {k: "zz"}) == (None, ("circuit", Cr.FILES_FORM))


@pytest.mark.parametrize("seed", SEEDS)
def test_statements_the_draw_did_not_ask_for_are_refused(seed, tmp_path, verifier):
    """A call proved twice (a retry), a statement of a call no drawn unit is in, a statement whose session proved a unit
    of another call, and a drawn call with no statement: each refuses the window."""
    d, statements = window(seed, tmp_path)
    assert credit(d, statements + [statements[0]])[1] == (
        "outcome", f"call {statements[0]['call']} has two statements: a call's drawn units are proved in one "
                   "statement and verified once, and a retry never reproves them")
    lay = _layout(d)
    idle = [c for c in range(len(lay["calls"])) if c not in {s["call"] for s in statements}]
    if idle:
        assert credit(d, statements + [{**statements[0], "call": idle[0]}])[1] == (
            "outcome", f"call {idle[0]} has a statement and no drawn unit: only the draw's units are proved")
    got, refused = credit(d, statements[1:])
    assert got is None and refused[0] == "missing", refused
    assert credit(d, [{**statements[0], "call": len(lay["calls"])}])[1] == ("proof", Cr.STATEMENTS_FORM)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_session_proving_another_calls_unit_is_refused(seed, tmp_path, verifier):
    d, statements = window(seed, tmp_path)
    s = statements[0]
    lay = _layout(d)
    sizes = [size for size, _w in Cr.call_work(lay)]
    pub = Path(s["public"])
    indices = json.loads(pub.read_text())["units"]["indices"]
    other = sum(sizes[:s["call"] + 1]) % sum(sizes)
    pub.write_bytes(json.dumps({"units": {"indices": [other, *indices[1:]]}}).encode() + b"\n")
    raw = bytes.fromhex(d["descriptor"])
    program, shape = hashlib.sha512(raw).hexdigest(), Cr.shape_of(raw)
    roots = {}
    for e in d["entries"]:
        roots.setdefault(e["value"], []).append(e["root"])
    reg, _ = Cr.registered_file(program, lay, shape, s["call"], roots, d["window"]["root"],
                                len(d["rounds"]))
    files = [Path(s["circuit"]).read_bytes(), pub.read_bytes(), raw, json.dumps(Cr.partition_of(program, shape)).encode(),
             json.dumps(reg).encode()]
    bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
    p = Path(s["sessions"][0]) / "session.json"
    rec = json.loads(p.read_text())
    rec["proof"] = hashlib.sha512(bound + json.dumps(rec["unit_draw"]["units"]).encode()).hexdigest()
    p.write_text(json.dumps(rec))
    got, refused = credit(d, statements)
    assert got is None and refused == ("proof", f"call {s['call']}'s session {s['sessions'][0]} proved unit {other}, "
                                                f"which is no unit of call {s['call']}: a call's statement proves its "
                                                "own call's units")


def test_a_verifier_that_cannot_run_leaves_the_window_refused(tmp_path, monkeypatch):
    d, statements = window(0, tmp_path)

    def broken():
        raise PF.LeanError("lake build flock-verify failed")

    monkeypatch.setattr(PF, "executable", broken)
    got, refused = credit(d, statements)
    assert got is None and refused[0] == "proof" and "could not be verified, so the window stays refused" in refused[1]


def _period(seed: int, tmp: Path, files: dict | None = "own"):
    rng = random.Random(f"statements-period/{seed}")
    probe = cases.period_instance(random.Random(f"statements-period/{seed}"))
    lay = _layout(probe["windows"][0])
    committed = {} if files is None else {Cr.CIRCUIT_FILES: files_of(lay) if files == "own" else files}
    d = cases.period_instance(rng, committed_too=committed)
    k, _ = Cr.committed_k(d["ledger"], d["name"])
    windows = []
    for i, w in enumerate(d["windows"]):
        statements = prove({**w, "k": k}, k, tmp / f"w{i}")
        windows.append({**{x: v for x, v in w.items() if x != "proofs"}, "statements": statements})
    return d, windows


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_of_statements_on_its_committed_circuit_files_earns(seed, tmp_path, verifier):
    d, windows = _period(seed, tmp_path)
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["name"], windows, rc.lms_verify)
    assert refused is None
    assert got["windows"] == [_credited(w) for w in windows]
    assert Cr.period_credit(d["key"], d["ledger"], d["name"], d["windows"], rc.lms_verify) == (
        None, ("period", Cr.PERIOD_FORM))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_committing_no_circuit_files_earns_nothing(seed, tmp_path, verifier):
    d, windows = _period(seed, tmp_path, files=None)
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["name"], windows, rc.lms_verify)
    assert got is None and refused[0] == "circuit" and f"commits no circuit files ({Cr.CIRCUIT_FILES}" in refused[1]
    d, windows = _period(seed, tmp_path / "other", files={"k1": "ab" * 64})
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["name"], windows, rc.lms_verify)
    assert got is None and refused[0] == "window" and "circuit: the period's open commits no circuit file for K = " in \
        refused[1], refused
