"""`credit.window` and `credit.period_credit` on a window's statements, each call's proof files: credit runs the
verifier of record itself (`one_stage.proofs.verify`) on the circuit file the period's open committed for the call's K,
with the partition object and the ``--registered`` file it builds from the committed descriptor and the window's entries,
and credits nothing from a verdict the evidence carries.

The verifier here is a stand-in `flock-verify` (`FAKE`), so the test runs without a C-Flock proof of an H1 window: it
answers as the Lean verifier does, one ``VERDICT`` line a session, and accepts a session only when its proof is the
SHA-512 of every file it was handed (the circuit file, the public file, the program, the partition object and the
``--registered`` file, those two as the JSON they hold, whose key order the verifier doesn't read) and the units its
session drew. `prove` makes those proofs with the files a prover would hold,
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
from verity.core.protocols.pouw import live_reads as LR
from verity.core.protocols.sampled_proofs import challenger as Ch
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
files[3:] = [json.dumps(json.loads(f), sort_keys=True).encode() for f in files[3:]]
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


LIVE_CIRCUIT = b"circuit file of the live Program's template\n"


def files_of(layout: dict) -> dict:
    """The period-open's `credit.CIRCUIT_FILES`: each K's circuit file's SHA-512, and the live template's."""
    return {**{f"k{k}": hashlib.sha512(circuit(k)).hexdigest() for _n, k, _c in layout["linears"]},
            Cr.LIVE: hashlib.sha512(LIVE_CIRCUIT).hexdigest()}


def _session(d: dict) -> str:
    return Ch.audited_session(d["ledger"], d["window"], Ch.ChallengerKey.from_json(d["key"]), rc.lms_verify).session


def _bind(base: Path, files: list[bytes], drew: list[int]) -> None:
    bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
    (base / "session" / "session.json").write_text(json.dumps(
        {"unit_draw": {"units": drew}, "proof": hashlib.sha512(bound + json.dumps(drew).encode()).hexdigest()}))


def _max_tokens(d: dict) -> list[int] | None:
    """The max_tokens the window's period's open committed (`credit.committed_length`), None when it commits none."""
    opens = [e["body"] for e in map(json.loads, d["ledger"]) if e["kind"] == "period-open"]
    length = opens[0].get(Cr.LENGTH_FIELD) if opens else None
    return length.get("max_tokens") if isinstance(length, dict) else None


def live_reads(d: dict, s: int, m: int | None = None) -> dict:
    """Round s's live@<s> ``--registered`` file (`live_reads.reads_file`) at the window's tables, its session, the
    manifest's m_s (or `m`) and the max_tokens its period's open committed."""
    raw, live = bytes.fromhex(d["descriptor"]), Cr.live_of(bytes.fromhex(d["live"]["descriptor"]))[0]
    program, prefix, steps = hashlib.sha512(raw).hexdigest(), Cr._prefix(raw), len(d["rounds"])
    roots = {e["value"]: e["root"] for e in d["entries"]}
    ms = m if m is not None else d["calls"][s * (len(d["calls"]) // steps)][Cr.SERVED_ROWS]
    return LR.reads_file(program, live["T"], steps, s, roots, d["window"]["root"], _session(d), ms, prefix, _max_tokens(d))


def prove_live(d: dict, to: Path, m: dict[int, int] | None = None) -> dict:
    """The window's live evidence as an honest prover delivers it (`credit.LIVE_STATEMENTS`): its live descriptor and
    each round's statement, the live template's circuit file, a public file whose one instance is unit s, and one session
    drawing it, its proof bound to the files the verifier is handed (round s's ``--registered`` file at the manifest's
    m_s, or ``m[s]``)."""
    desc = d["live"]["descriptor"]
    raw = bytes.fromhex(desc)
    unit = Cr.live_of(raw)[0]["unit"]
    partition = {"format": Cr.PARTITION_FORMAT, "program": hashlib.sha512(raw).hexdigest(),
                 "query": {"name": Cr.QUERY[0], "version": Cr.QUERY[1], "params": {"templates": [unit]}}}
    out = []
    for s in range(len(d["rounds"])):
        base = to / f"live-{s}"
        (base / "session").mkdir(parents=True)
        (base / "circuit.txt").write_bytes(LIVE_CIRCUIT)
        (base / "pub.bin").write_bytes(json.dumps({"units": {"indices": [s]}}).encode() + b"\n")
        reg = live_reads(d, s, (m or {}).get(s))
        _bind(base, [LIVE_CIRCUIT, base.joinpath("pub.bin").read_bytes(), raw,
                     json.dumps(partition, sort_keys=True).encode(), json.dumps(reg, sort_keys=True).encode()], [0])
        out.append({"round": s, "circuit": str(base / "circuit.txt"), "public": str(base / "pub.bin"),
                    "sessions": [str(base / "session")]})
    return {"descriptor": desc, "statements": out}


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
        reg, why = Cr.registered_file(program, lay, shape, c, roots, d["window"]["root"], len(d["rounds"]), d["salt"])
        assert why is None, why
        files = [base.joinpath("circuit.txt").read_bytes(), base.joinpath("pub.bin").read_bytes(), raw,
                 json.dumps(Cr.partition_of(program, shape), sort_keys=True).encode(),
                 json.dumps(reg, sort_keys=True).encode()]
        bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
        drew = list(range(len(units)))
        (base / "session" / "session.json").write_text(json.dumps(
            {"unit_draw": {"units": drew}, "proof": hashlib.sha512(bound + json.dumps(drew).encode()).hexdigest()}))
        out.append({"call": c, "circuit": str(base / "circuit.txt"), "public": str(base / "pub.bin"),
                    "sessions": [str(base / "session")]})
    return out


def window(seed: int, tmp: Path) -> tuple[dict, list[dict]]:
    """An honest window, its ``live`` evidence its live statements (`prove_live`), and its calls' statements."""
    d = cases.window_instance(random.Random(f"statements/{seed}"), "honest")
    return {**d, "live": prove_live(d, tmp)}, prove(d, d["k"], tmp)


def credit(d: dict, statements, files=None, run=PF.verify):
    return Cr.window(*(d[k] for k in cases.WINDOW_ARGS[:-4]), statements, d["live"],
                     files_of(_layout(d)) if files is None else files, d["k"], d["rows"], rc.lms_verify, run=run)


def _credited(d: dict) -> list[int]:
    return [i for i, row in enumerate(d["calls"]) if row[Cr.CREDIT] and row[Cr.SERVED_ROWS] > 0]


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
                                len(d["rounds"]), d["salt"])
    files = [Path(s["circuit"]).read_bytes(), pub.read_bytes(), raw,
             json.dumps(Cr.partition_of(program, shape), sort_keys=True).encode(),
             json.dumps(reg, sort_keys=True).encode()]
    bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
    p = Path(s["sessions"][0]) / "session.json"
    rec = json.loads(p.read_text())
    rec["proof"] = hashlib.sha512(bound + json.dumps(rec["unit_draw"]["units"]).encode()).hexdigest()
    p.write_text(json.dumps(rec))
    got, refused = credit(d, statements)
    assert got is None and refused == ("proof", f"call {s['call']}'s session {s['sessions'][0]} proved unit {other}, "
                                                f"which is no unit of call {s['call']}: a call's statement proves its "
                                                "own call's units")


def _files(d: dict, s: dict) -> tuple[str, dict, dict, dict]:
    """Statement ``s``'s program digest, layout, shape and the ``--registered`` file credit builds for its call."""
    raw, lay = bytes.fromhex(d["descriptor"]), _layout(d)
    program, shape = hashlib.sha512(raw).hexdigest(), Cr.shape_of(raw)
    roots = {}
    for e in d["entries"]:
        roots.setdefault(e["value"], []).append(e["root"])
    reg, why = Cr.registered_file(program, lay, shape, s["call"], roots, d["window"]["root"], len(d["rounds"]), d["salt"])
    assert why is None, why
    return program, lay, shape, reg


def _seeded(d: dict, s: dict, roots: tuple[str, str]) -> dict:
    """Statement ``s``'s ``--registered`` file with its public rows the prover's, seeded under ``roots`` in place of its
    call's own: every other port credit's."""
    program, lay, shape, reg = _files(d, s)
    pub = reg["ports"][Cr.PUB_PORT]
    u0, n = pub[Cr.G.READS_FROM], pub["leaves"]
    rows, salts, got = Cr.public_table(program, pub["value"], d["salt"], d["window"]["root"], u0, n, roots)
    return {**reg, "ports": {**reg["ports"], Cr.PUB_PORT: {**pub, "root": got.root.hex(), "known": {
        "rows": rows, "salts": [x.hex() for x in salts]}}}}


def _rebind(d: dict, s: dict, reg: dict) -> None:
    """Statement ``s``'s session reproved against ``reg`` in place of the file credit builds."""
    raw = bytes.fromhex(d["descriptor"])
    program, shape = hashlib.sha512(raw).hexdigest(), Cr.shape_of(raw)
    files = [Path(s["circuit"]).read_bytes(), Path(s["public"]).read_bytes(), raw,
             json.dumps(Cr.partition_of(program, shape), sort_keys=True).encode(),
             json.dumps(reg, sort_keys=True).encode()]
    bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
    p = Path(s["sessions"][0]) / "session.json"
    rec = json.loads(p.read_text())
    rec["proof"] = hashlib.sha512(bound + json.dumps(rec["unit_draw"]["units"]).encode()).hexdigest()
    p.write_text(json.dumps(rec))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_units_public_row_is_its_calls_own_salt_position_and_roots(seed, tmp_path, verifier):
    """RCPT: the public row a unit's seed and fold keys are derived from is credit's own. Call c's statement reads it at
    row t of ``<prefix>pub@<c>``, the window's Program salt, the unit's position u0 + t and the call's roots
    (root(a/<K>@<s>), root(b/<K>)) as the window's entries register them, at that known content; and a statement whose
    units were seeded under another call's roots or a made-up root was proved against another file, so its session is
    rejected and the window refused whole."""
    d, statements = window(seed, tmp_path)
    for s in statements:
        _program, lay, _shape, reg = _files(d, s)
        c, pub = s["call"], reg["ports"][Cr.PUB_PORT]
        k = {name: kk for name, kk, _n in lay["linears"]}[lay["calls"][c][0]]
        a, b = (reg["ports"][f"a_k{k}_0"], reg["ports"][f"b_k{k}_0"])
        u0, size = sum(n for n, _w in Cr.call_work(lay)[:c]), Cr.call_work(lay)[c][0]
        assert pub["value"].endswith(f"pub@{c}") and pub["leaves"] == size and pub["words"] == 100
        assert (pub[Cr.G.READS_FROM], pub["reads"]) == (u0, list(range(size)))
        assert pub["known"]["rows"] == [Cr.CC.public_words(bytes.fromhex(d["salt"]), u0 + t, (
            bytes.fromhex(a["root"]), bytes.fromhex(b["root"]))) for t in range(size)]
        assert pub["known"]["salts"] == [Cr.CC.public_row_salt(d["window"]["root"], u0 + t).hex() for t in range(size)]
    assert credit(d, statements) == (_credited(d), None)
    s = random.Random(seed).choice(statements)
    _program, lay, _shape, reg = _files(d, s)
    own = (next(p for x, p in reg["ports"].items() if x.startswith("a_k"))["root"],
           next(p for x, p in reg["ports"].items() if x.startswith("b_k"))["root"])
    others = {e["root"] for e in d["entries"] if Cr.is_a(e["value"])} - {own[0]}
    for roots in [(r, own[1]) for r in sorted(others)[:1]] + [(own[0], "cd" * 64), ("ab" * 64, own[1])]:
        _rebind(d, s, _seeded(d, s, roots))
        got, refused = credit(d, statements)
        assert got is None and refused[0] == "failed" and refused[1].startswith(f"call {s['call']}'s session "), refused


def window_j10r(seed: int, tmp: Path) -> tuple[dict, list[dict]]:
    """An honest window over J10-R units, each call of at least two row tiles, so a call's units' blocks differ, and its
    ``live`` evidence (`window`'s)."""
    d = cases.window_instance(random.Random(f"statements-j10r/{seed}"), "j10r")
    return {**d, "live": prove_live(d, tmp)}, prove(d, d["k"], tmp)


def _j10r_rows(d: dict, s: dict, call: int, blocks: list[tuple[int, int]]) -> dict:
    """Statement ``s``'s ``--registered`` file with its public rows the prover's, each unit's ``call`` and ``blocks``
    `call` and `blocks` (one a unit of its call) in place of the ones credit derives: every other port credit's."""
    program, _lay, _shape, reg = _files(d, s)
    pub, w = reg["ports"][Cr.PUB_PORT], d["window"]["root"]
    u0, n = pub["reads"].index(0), pub["leaves"]
    roots = tuple(bytes.fromhex(next(p for x, p in reg["ports"].items() if x.startswith(f"{t}_k"))["root"]) for t in "ab")
    rows = [Cr.CC.public_words(bytes.fromhex(d["salt"]), u0 + t, roots, (call, blocks[t])) for t in range(n)]
    salts = [Cr.CC.public_row_salt(w, u0 + t) for t in range(n)]
    got = Cr.G.Registered.commit(program, pub["value"], rows, salts, Cr.G.WORD_BITS, Cr.G.SCHEMA_V2, window=w)
    return {**reg, "ports": {**reg["ports"], Cr.PUB_PORT: {**pub, "root": got.root.hex(), "known": {
        "rows": rows, "salts": [x.hex() for x in salts]}}}}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_j10r_units_public_row_holds_the_call_and_blocks_credit_derives(seed, tmp_path, verifier):
    """RT-ALGO-16: a J10-R window's unit's public row is today's, then its ``call``, its call's first position u0 (the
    one its seed_c is drawn at), and its ``blocks``, `core_check.j10r_blocks` of its tile (I, J) = (t // (n/C),
    t % (n/C)) at the call's slot M, its K and the tile: credit derives both from the committed layout, as it builds
    the rest of the row. A statement whose units drew their noise at another unit's blocks, at another call's u0, or at
    blocks one off was proved against another file, so its session is rejected and the window refused whole."""
    d, statements = window_j10r(seed, tmp_path)
    for s in statements:
        _program, lay, shape, reg = _files(d, s)
        c, pub = s["call"], reg["ports"][Cr.PUB_PORT]
        k, n = next((k, n) for name, k, n in lay["linears"] if name == lay["calls"][c][0])
        (r, cc), m = lay["tile"], lay["m"]
        a, b = (reg["ports"][f"a_k{k}_0"], reg["ports"][f"b_k{k}_0"])
        u0, size = sum(x for x, _w in Cr.call_work(lay)[:c]), Cr.call_work(lay)[c][0]
        blocks = [Cr.CC.j10r_blocks(m, k, (r, cc), t // (n // cc), t % (n // cc)) for t in range(size)]
        assert shape["tiles"][k]["j10r"] and pub["words"] == 112 and size >= 2 and len(set(blocks)) == size
        assert pub["known"]["rows"] == [Cr.CC.public_words(bytes.fromhex(d["salt"]), u0 + t, (
            bytes.fromhex(a["root"]), bytes.fromhex(b["root"])), (u0, blocks[t])) for t in range(size)]
    assert credit(d, statements) == (_credited(d), None)
    s = random.Random(seed).choice(statements)
    _program, lay, _shape, reg = _files(d, s)
    pub = reg["ports"][Cr.PUB_PORT]
    u0, size = pub["reads"].index(0), pub["leaves"]
    k, n = next((k, n) for name, k, n in lay["linears"] if name == lay["calls"][s["call"]][0])
    own = [Cr.CC.j10r_blocks(lay["m"], k, tuple(lay["tile"]), t // (n // lay["tile"][1]), t % (n // lay["tile"][1]))
           for t in range(size)]
    starts = [sum(x for x, _w in Cr.call_work(lay)[:c]) for c in range(len(lay["calls"]))]
    other = next((x for x in starts if x != u0), u0 + size)
    for call, blocks in ((u0, own[1:] + own[:1]), (other, own), (u0, [(b0 + 1, b1) for b0, b1 in own]),
                         (u0, [(b0, b1 - 1) for b0, b1 in own])):
        _rebind(d, s, _j10r_rows(d, s, call, blocks))
        got, refused = credit(d, statements)
        assert got is None and refused[0] == "failed" and refused[1].startswith(f"call {s['call']}'s session "), refused


@pytest.mark.parametrize("seed", SEEDS)
def test_a_session_that_opened_another_public_table_root_is_refused(seed, tmp_path, verifier):
    """A verifier finding that a session opened its call's public table at another root than the rows credit built
    (another call's roots, or made up) refuses the window (``registration``): no entry registers that table, so
    `_registration` holds it only through `_verified`."""
    d, statements = window(seed, tmp_path)
    s = random.Random(seed).choice(statements)
    _program, _lay, _shape, reg = _files(d, s)
    pub = reg["ports"][Cr.PUB_PORT]

    def run(proof, **kw):
        got, why = PF.verify(proof, **kw)
        return [PF.Finding(f.session, f.accepted, f.why, f.program, {**f.registered, Cr.PUB_PORT: {
            "value": pub["value"], "root": "ab" * 64}} if kw["registered"] == reg else f.registered, f.units)
            for f in got or []], why

    got, refused = credit(d, statements, run=run)
    assert got is None and refused == ("registration", (
        f"call {s['call']}'s session {s['sessions'][0]} opened a public table at another port, value or root than the "
        f"verifier's own {pub['value']} at port {Cr.PUB_PORT}: a unit's public row is the window's salt, its position "
        "and its call's roots, as credit builds them"))


def _ok_rows(d: dict, s: dict, rows: list[list[int]]) -> dict:
    """Statement ``s``'s ``--registered`` file with the window's ok rows the prover's `rows` (row u unit u's one word),
    under their public salts, in place of credit's: every other port credit's."""
    program, _lay, _shape, reg = _files(d, s)
    ok, w = reg["ports"][Cr.OK_PORT], d["window"]["root"]
    salts = [Cr.CC.ok_salt(w, u) for u in range(ok["leaves"])]
    got = Cr.G.Registered.commit(program, ok["value"], rows, salts, Cr.G.WORD_BITS, Cr.G.SCHEMA_V2, window=w)
    return {**reg, "ports": {**reg["ports"], Cr.OK_PORT: {**ok, "root": got.root.hex(), "known": {
        "rows": rows, "salts": [x.hex() for x in salts]}}}}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_units_ok_row_is_the_verifiers_one_word_1(seed, tmp_path, verifier):
    """OKROW in credit: a statement's output row ``ok`` (o1) reads row u of the window's ok rows ``<prefix>ok``, the one
    word 1 under the public `core_check.ok_salt` of the window's open root and u, at that known content, one row a unit
    of the window (h1_prove's `register_ok`). A statement whose unit computed 0, or whose units all did, was proved
    against another file, so its session is rejected and the window refused whole."""
    d, statements = window(seed, tmp_path)
    w, population = d["window"]["root"], sum(n for n, _w in Cr.call_work(_layout(d)))
    for s in statements:
        program, lay, shape, reg = _files(d, s)
        c, ok = s["call"], reg["ports"][Cr.OK_PORT]
        u0, size = sum(n for n, _w in Cr.call_work(lay)[:c]), Cr.call_work(lay)[c][0]
        assert ok["value"] == f"{shape['prefix']}ok" and ok["leaves"] == population and ok["words"] == 1
        assert (ok[Cr.G.READS_FROM], ok["reads"]) == (u0, list(range(u0, u0 + size)))
        assert ok["known"] == {"rows": [[1]] * population, "salts": [Cr.CC.ok_salt(w, u).hex() for u in range(population)]}
        assert ok["domain"] == Cr.G.domain(program, ok["value"], 1, population, window=w).domain_id.hex()
    assert credit(d, statements) == (_credited(d), None)
    rng = random.Random(seed)
    s = rng.choice(statements)
    sizes = [n for n, _w in Cr.call_work(_layout(d))]
    u = sum(sizes[:s["call"]]) + rng.randrange(sizes[s["call"]])
    zero = [[1]] * population
    zero[u] = [0]
    for rows in (zero, [[0]] * population):
        _rebind(d, s, _ok_rows(d, s, rows))
        got, refused = credit(d, statements)
        assert got is None and refused[0] == "failed" and refused[1].startswith(f"call {s['call']}'s session "), refused


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("opened", ["another root", "another value", "none"])
def test_a_session_that_opened_another_ok_root_is_refused(seed, opened, tmp_path, verifier):
    """A verifier finding that a session opened the window's ok rows at another root than the rows credit registered,
    at another value, or not at all refuses the window (``registration``): no entry registers them, so `_registration`
    holds them only through `_verified`."""
    d, statements = window(seed, tmp_path)
    s = random.Random(seed).choice(statements)
    _program, _lay, _shape, reg = _files(d, s)
    ok = reg["ports"][Cr.OK_PORT]

    def found(f):
        got = {p: e for p, e in f.registered.items() if p != Cr.OK_PORT}
        if opened == "another root":
            got[Cr.OK_PORT] = {"value": ok["value"], "root": "ab" * 64}
        elif opened == "another value":
            got[Cr.OK_PORT] = {"value": ok["value"] + "@0", "root": ok["root"]}
        return got

    def run(proof, **kw):
        got, why = PF.verify(proof, **kw)
        return [PF.Finding(f.session, f.accepted, f.why, f.program, found(f) if kw["registered"] == reg else f.registered,
                           f.units) for f in got or []], why

    got, refused = credit(d, statements, run=run)
    assert got is None and refused == ("registration", (
        f"call {s['call']}'s session {s['sessions'][0]} opened the ok rows at another port, value or root than the "
        f"verifier's own {ok['value']} at port {Cr.OK_PORT}, or not at all: a unit's ok row is the one word 1 under its "
        "public ok salt, as credit registers it, so a unit that computed 0 opens none of it"))


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
        live = prove_live({**w, "key": d["key"], "ledger": d["ledger"]}, tmp / f"w{i}")
        windows.append({**{x: v for x, v in w.items() if x != "proofs"}, "statements": statements, "live": live})
    return d, windows


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_of_statements_on_its_committed_circuit_files_earns(seed, tmp_path, verifier):
    d, windows = _period(seed, tmp_path)
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows, rc.lms_verify)
    assert refused is None
    assert got["windows"] == [_credited(w) for w in windows]
    assert Cr.period_credit(d["key"], d["ledger"], d["pins"], d["main"], d["name"], d["windows"], rc.lms_verify) == (
        None, ("period", Cr.PERIOD_FORM))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_committing_no_circuit_files_earns_nothing(seed, tmp_path, verifier):
    d, windows = _period(seed, tmp_path, files=None)
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows, rc.lms_verify)
    assert got is None and refused[0] == "circuit" and f"commits no circuit files ({Cr.CIRCUIT_FILES}" in refused[1]
    d, windows = _period(seed, tmp_path / "other", files={"k1": "ab" * 64})
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows, rc.lms_verify)
    assert got is None and refused[0] == "window" and "circuit: the period's open commits no circuit file for K = " in \
        refused[1], refused


@pytest.mark.parametrize("seed", SEEDS)
def test_a_live_statement_proved_at_another_m_is_rejected(seed, tmp_path, verifier):
    """Credit builds round s's live ``--registered`` file itself, its ``m@<s>`` at the manifest's m_s: a live statement
    the prover proved against an ``m@<s>`` of another m (the serving process stated m_s, the live unit counted another)
    was proved against another file, so the verifier rejects its session and the window is refused (``live``)."""
    d, statements = window(seed, tmp_path)
    assert credit(d, statements) == (_credited(d), None)
    s = random.Random(seed).randrange(len(d["rounds"]))
    m = d["calls"][s * (len(d["calls"]) // len(d["rounds"]))][Cr.SERVED_ROWS]
    d = {**d, "live": prove_live(d, tmp_path / "other-m", {s: m + 1 if m < d["rows"] else m - 1})}
    got, refused = credit(d, statements)
    assert got is None and refused[0] == "live" and refused[1].startswith(f"round {s}'s live@{s} session "), refused
    assert "was rejected by the verifier of record" in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_live_evidence_carrying_outcomes_or_no_committed_file_is_refused(seed, tmp_path, verifier, monkeypatch):
    """The live evidence carries each round's proof files, never verdicts (`credit.LIVE_STATEMENTS`), on the circuit
    file the period's open committed for the live template (``circuit_files.live``); a round proved twice or past the
    window's rounds, or a session that proved another unit than its round's, refuses the window."""
    d, statements = window(seed, tmp_path)
    outcomes = cases.window_instance(random.Random(f"statements/{seed}"), "honest")["live"]
    assert credit({**d, "live": outcomes}, statements) == (None, ("live", Cr.LIVE_STATEMENTS_FORM))
    files = {x: v for x, v in files_of(_layout(d)).items() if x != Cr.LIVE}
    assert credit(d, statements, files) == (None, ("circuit", Cr.no_live_files()))
    live = d["live"]["statements"]
    steps = len(d["rounds"])
    for extra in ([*live, live[0]], [*live, {**live[0], "round": steps}]):
        got, refused = credit({**d, "live": {**d["live"], "statements": extra}}, statements)
        assert got is None and refused[0] == "live" and "each round's live@<s> is one statement" in refused[1]
    got, refused = credit({**d, "live": {**d["live"], "statements": live[1:]}}, statements)
    assert got is None and refused == ("live", refused[1]) and "round 0's live@0 has no proof outcome" in refused[1]
    pub = Path(live[0]["public"])
    pub.write_bytes(json.dumps({"units": {"indices": [steps]}}).encode() + b"\n")
    reg = live_reads(d, 0)
    raw = bytes.fromhex(d["live"]["descriptor"])
    partition = {"format": Cr.PARTITION_FORMAT, "program": hashlib.sha512(raw).hexdigest(),
                 "query": {"name": Cr.QUERY[0], "version": Cr.QUERY[1],
                           "params": {"templates": [Cr.live_of(raw)[0]["unit"]]}}}
    _bind(pub.parent, [LIVE_CIRCUIT, pub.read_bytes(), raw, json.dumps(partition, sort_keys=True).encode(),
                       json.dumps(reg, sort_keys=True).encode()], [0])
    got, refused = credit(d, statements)
    assert got is None and refused[0] == "live" and f"proved units [{steps}], not unit 0" in refused[1], refused


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_whose_live_statement_fails_earns_nothing(seed, tmp_path, verifier):
    d, windows = _period(seed, tmp_path)
    i = random.Random(seed).randrange(len(windows))
    p = Path(windows[i]["live"]["statements"][0]["sessions"][0]) / "session.json"
    rec = json.loads(p.read_text())
    rec["proof"] = ("0" if rec["proof"][0] != "0" else "1") + rec["proof"][1:]
    p.write_text(json.dumps(rec))
    got, refused = Cr.period_credit(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows, rc.lms_verify)
    assert got is None and refused[0] == "window" and "live: round 0's live@0 session " in refused[1], refused
