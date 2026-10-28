"""`certifier_case`: certifier/v0's case of a run and its compute audits, held to the period check's Python mirror
(`period_case`, which ``network-period`` is held to):

* a compute audit the integrity check accepted is re-decided at δ (one-stage's bound on wrong units), and its refusal
  kept as it is; an audit record missing, refused or for another registration is an error;
* `case_of`'s case of signed link-windows and the period's audit is the clock form's: the mirror runs one-stage v2 on
  each link-window's proof (undecided with no session's proof), refuses a link-window v2 refuses, a case naming another
  first window and an audit registered over another Program, and once v2 accepts the proofs accepts with K and eps at
  each δ, the audit charged its Program's units' committed bits (16 a unit, or 40);
* `run_windows` reads each link-window's registration, one-stage v2 evidence, delivered salts and the files its proof
  names off N2's verifier directory, with the link's v2 parameters and circuit file; a link-window's clock is the
  client's certificate without the session or the sync;
* `rows_case_of`'s case is the rows form's: a signed rows period of the mirror's (`period_case.build`), taken apart into
  a warden run's windows and scans and rebuilt, is the same commitment and evidence, and the mirror refuses it at the
  wiring alone (`period_case.UNWIRED`, the rows form's last refusal) and without that refusal accepts it;
* each term of a case is decided at its share of the δ (`term_delta`): a signed clock case of two audits, its
  link-windows' proofs accepted, is accepted with eps that δ, an unsplit one with eps their sum, which `holds_at`
  refuses.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.certifier.active import TOKEN_BYTES
from verity.core.protocols.certifier.certificate import SALT_BYTES, Certificate, Direction, clock_commitment
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.commitments.window import window_domain
from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import block_certifier as BC  # noqa: E402
import certifier_case as L  # noqa: E402
import certifier_serve as cs  # noqa: E402


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


PERIOD = _module("period_case", ROOT / "verity/core/protocols/network/tests/period_case.py")
ZK = PERIOD.ZK

PROMPTS, STEPS, H = [[3], [7]], 2, 128
FRAMES = STEPS * -(-4 * H // TOKEN_BYTES)
LOOP = {"format": R.FORMAT, "protocol": R.PROTOCOL, "program": "aa" * 64, "partition": "bb" * 64, "population": 1024,
        "law": {"law": "subset", "k": 2}}
DRAW = {"law": "subset", "k": 2, "population": 1024, "units": [700, 5]}
ACCEPTED = {"accepted": True, "registration": LOOP, "draw": DRAW, "proved_units": [5, 700]}
#: the audit's Program, the period mirror's stand-in population of LOOP's 1024 units, each committing 16 bits, and
#: LOOP registered over it
PROGRAM = json.loads(PERIOD.audit_descriptor(((1024, 2),)))
AUDITED = LOOP | {"program": PERIOD.program_sha512(PROGRAM)}
#: a loop window's free inputs: 1,024 units of four input rows
FREE = 1024 * (1024 + 32 + 1024 + 32)


def audited(delta: Fraction, program: dict = PROGRAM) -> tuple[dict, dict]:
    """LOOP registered over ``program`` and its accepted audit re-decided at ``delta``."""
    reg = LOOP | {"program": PERIOD.program_sha512(program)}
    return reg, L.audit_answer(reg, {"verdict": "accept"}, ACCEPTED | {"registration": reg}, delta)


def plan(periods: int) -> tuple[dict, dict]:
    probed = {"url": "stub", "body": cs.request_body("stub", STEPS, PROMPTS[0]),
              "responses": [{"events": FRAMES, "first_s": 0.0, "last_s": 0.2}]}
    return cs.served_plan(probed, len(PROMPTS) * periods, in_flight=len(PROMPTS), frames=FRAMES, prompts=PROMPTS,
                          window_per_group=True)


@pytest.mark.parametrize("delta, bound", [(Fraction(1, 100), 921), (Fraction(1, 10 ** 6), 1022)])
def test_an_accepted_audit_is_re_decided_at_delta(delta, bound):
    a = L.audit_answer(LOOP, {"verdict": "accept"}, ACCEPTED, delta)
    CC.answer_of(a)
    assert a["public"]["bound"] == bound
    assert CC.integrity(a["profile"])["delta"] == delta
    assert a["public"]["drawn"] == a["public"]["proved"] == [5, 700]


def test_a_refused_audit_stays_refused_and_a_record_for_another_registration_is_an_error():
    refused = {"verdict": "reject", "reason": "records", "commitment": R.digest(LOOP), "profile": None,
               "protocol": R.PROTOCOL, "public": None}
    assert L.audit_answer(LOOP, refused, None, Fraction(1, 100)) == refused
    for record in (None, ACCEPTED | {"accepted": False}, ACCEPTED | {"registration": LOOP | {"population": 1023}}):
        with pytest.raises(RuntimeError, match="audit record"):
            L.audit_answer(LOOP, {"verdict": "accept"}, record, Fraction(1, 100))


def session_run(windows: list[int], tmp: Path, refused: frozenset[int] = frozenset()) -> dict:
    """A run as `run_windows` reads it: N2's link on the planned profile, under one of the one-stage difftest's
    programs registered with a law that draws no unit (so one-stage v2 leaves each link-window undecided: no case here
    carries a proof it accepts), its link-windows registered under their served domains and chained, each with v2's
    evidence (no session, the mirror's public file, `one_stage_zk_case.undecided`) and two delivered salts, and signed
    by a small deployed-size LMS key; a ``refused`` link-window's evidence names a unit no session proved, which v2
    refuses. The circuit and public files are under ``tmp``."""
    report, _ = plan(len(windows))
    setup = cs.ar.Setup(report)
    params = lms.Params(64, 4, 4)
    key = lms.keygen(params, hashlib.sha512(b"certifier leg").digest())
    session, link, run = hashlib.sha512(b"session").digest()[:lms.SESSION_BYTES], "n2", "r-leg"
    sync = ClockSync(setup.sync_set.deltas[0], setup.sync_set.rhos[0])
    q = iter(range(2 ** params.H))
    salts = iter(hashlib.sha512(b"salt %d" % i).digest()[:SALT_BYTES] for i in range(1 << 10))
    prog, law = PERIOD.OS.SETUPS[0], PERIOD.EMPTY_LAWS[0]
    part = {"program": prog.sha512, "query": prog.query}
    (tmp / "circuit.txt").write_bytes(ZK.CIRCUIT)

    def sign(d: Direction, w: int, digest: bytes) -> dict:
        c = Certificate(key.I, link, d, w, digest, b"")
        c = dataclasses.replace(c, signature=lms.sign_at(key, next(q), c.message(session)).encode())
        return L._cert({"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(),
                        "link": c.link, "signature": c.signature.hex(), "window": c.window})

    xs, previous = {}, None
    for w in windows:
        root = hashlib.sha512(b"root %d" % w).hexdigest()
        domain = window_domain(prog.sha512, PERIOD.OS.P.digest(part), PERIOD.VALUE, PERIOD.SCHEMA, PERIOD.LEAVES,
                               f"{run}/{link}/{w}")
        reg = PERIOD.OS._json(PERIOD.OS.R.record(
            prog.program, prog.query, law, PERIOD.OS.SCHEME,
            [{"port": PERIOD.VALUE, "root": root, "leaves": PERIOD.LEAVES, "domain": domain}], PERIOD.served(w),
            previous=previous, partition=part))
        previous = R.digest(reg)
        one = ZK.undecided(prog, law, reg, {PERIOD.VALUE: root})
        if w in refused:
            one["proved_units"] = [0]
        (tmp / f"w{w:03d}-pub-1.bin").write_bytes(ZK.PUBLIC)
        xs[w] = {"registration": reg, "one_stage": one,
                 "salts": {k: hashlib.sha512(b"%s %d" % (k.encode(), w)).hexdigest() * 3 for k in ("o0", "pub")},
                 "blobs": [str(tmp / f"w{w:03d}-pub-1.bin")],
                 "certificates": [sign(Direction.EGRESS, w, hashlib.sha512(b"e%d" % w).digest()),
                                  sign(Direction.INGRESS, w, hashlib.sha512(b"i%d" % w).digest())],
                 "clock": L.clock_evidence({"session": session.hex()}
                                           | sign(Direction.EGRESS, w, clock_commitment(next(salts), sync)))}
    dep = {"certifier_key": {"params": [params.N, params.W, params.H], "public": lms.public_key(key).hex()},
           "links": {"n2": link}, "run_id": run, "session": session.hex()}
    one_stage = {"anchors": [], "delta": "1/1099511627776", "law": law, "leaf_layer": None, "log": [],
                 "partition": part, "program": prog.desc, "proofs": {"circuit": ZK.sha(ZK.CIRCUIT), "registered": None},
                 "query": prog.query, "registered": None, "scheme": PERIOD.OS.SCHEME, "work_table": None}
    return {"deployment": dep, "calibration": report, "plan": plan(len(windows))[1], "one_stage": one_stage,
            "circuit": str(tmp / "circuit.txt"), "windows": xs}


def decided(case: dict) -> dict:
    """The mirror's answer on a case of `case_of`, its blobs read off their paths as ``network-period check`` reads
    them."""
    return PERIOD.reference({**case, "blobs": [Path(b).read_bytes() for b in case["blobs"]]})


def proofs_accepted(monkeypatch) -> None:
    """One-stage v2 accepting each link-window's proof, as on a real session's proof: the mirror runs no prover and
    raises on an acceptance, so here the rest of the check (its K, eps and profile) runs as if v2 had accepted."""
    monkeypatch.setattr(PERIOD, "_proved", lambda *a: None)


@pytest.mark.parametrize("delta", BC.BLOCK_DELTAS)
def test_the_case_is_the_period_checks(delta, tmp_path, monkeypatch):
    """`case_of`'s case is the clock form's: the link's ends and run, N2's one-stage v2 parameters with the stream's
    registrations before the period as their log, each audit's Program, each link-window's v2 evidence and salts, and
    the files its proof names as blobs; the mirror passes every check of each link-window up to one-stage v2 and runs
    v2 on it (undecided: no session's proof), refuses a link-window v2 refuses, a case naming another first window and
    an audit registered over another Program, and once v2 accepts the proofs accepts with K and eps at each δ, the
    audit charged its Program's units' committed bits (16 a unit, or 40)."""
    rw = session_run([0, 1, 2], tmp_path)
    audits = [audited(delta)]
    case = L.case_of(rw, [1, 2], audits, 1, [FREE], [PROGRAM])
    p = case["params"]
    assert sorted(p) == PERIOD.PARAMS_KEYS and p["run"] == "r-leg" and p["link"] == "n2"
    assert p["one_stage"] == {**rw["one_stage"], "log": [rw["windows"][0]["registration"]]}
    assert p["audits"] == [{"free": FREE, "program": PROGRAM, "steps": 2 * cs.WINDOW_BUCKETS}]
    assert case["blobs"] == [rw["circuit"], *rw["windows"][1]["blobs"], *rw["windows"][2]["blobs"]]
    e = json.loads(bytes.fromhex(case["evidence"]))
    assert all(sorted(x) == PERIOD.CLOCK_WINDOW_KEYS for x in e["windows"])
    a = decided(case)
    assert (a["verdict"], a["reason"]) == ("undecided", "window 1: one-stage v2: verifier of record"), a
    assert decided(L.case_of(rw, [1, 2], audits, 0, [FREE], [PROGRAM]))["verdict"] == "reject"
    refused = session_run([0, 1, 2], tmp_path, frozenset({2}))
    a = decided(L.case_of(refused, [1, 2], audits, 1, [FREE], [PROGRAM]))
    assert a["verdict"] == "reject" and a["reason"].startswith("window 2: one-stage v2: "), a
    wide = json.loads(PERIOD.audit_descriptor(((1024, 5),)))
    other = decided(L.case_of(rw, [1, 2], audits, 1, [FREE], [wide]))
    assert other["verdict"] == "reject" and "not its Program" in other["reason"], other
    with pytest.raises(ValueError, match="free-input counts"):
        L.case_of(rw, [1, 2], audits, 1, [], [PROGRAM])
    with pytest.raises(ValueError, match="Programs"):
        L.case_of(rw, [1, 2], audits, 1, [FREE], [])
    proofs_accepted(monkeypatch)
    a = decided(case)
    assert a["verdict"] == "accept", a["reason"]
    pub = a["public"]
    bound = audits[0][1]["public"]["bound"]
    assert pub["eps"] == str(delta) and pub["windows"] == 2 and pub["cells"] == 0 and bound > 0
    assert pub["audits"][0]["wrong_units"] == bound and pub["audits"][0]["jitter"] == 0
    assert pub["audits"][0]["free"] == FREE and pub["audits"][0]["faulty_bits"] == 16 * bound
    assert pub["K"] == pub["schedules"] + pub["audits"][0]["faulty_sets"] + 16 * bound + FREE
    a40 = decided(L.case_of(rw, [1, 2], [audited(delta, wide)], 1, [FREE], [wide]))
    assert a40["public"]["audits"][0]["faulty_bits"] == 40 * bound and a40["public"]["K"] == pub["K"] + 24 * bound


def test_run_windows_reads_what_n2_writes(tmp_path):
    """`run_windows` reads a run's link-windows off N2's verifier directory as `n2_zk.member` writes it (each
    link-window's registration, one-stage v2 evidence, delivered salts, public file and sessions' files, and the link's
    v2 parameters and circuit file) and the client's certificates."""
    (tmp_path / "made").mkdir()
    rw = session_run([0, 1], tmp_path / "made")
    s, run = tmp_path / "s", tmp_path / "s" / "rehearsal" / "honest"
    v = s / "n2zk" / "honest" / "verifier"
    for d in (run / "client", v):
        d.mkdir(parents=True)
    for f, x in (("deployment.json", rw["deployment"]), ("plan.json", rw["plan"]),
                 ("calibration.json", rw["calibration"])):
        (run / f).write_text(json.dumps(x))
    (run / "client" / "certificates.jsonl").write_text("".join(
        json.dumps(c | {"session": "x"}) + "\n" for x in rw["windows"].values() for c in x["certificates"]))
    (run / "client" / "clocks.jsonl").write_text("".join(
        json.dumps(x["clock"] | {"session": "x"}) + "\n" for x in rw["windows"].values()))
    (s / "n2zk" / "summary.json").write_text(json.dumps({"members": {"honest": {"windows": [
        {"window": w} for w in rw["windows"]]}}}))
    (v / "one-stage.json").write_text(json.dumps(rw["one_stage"]))
    (v / "circuit.txt").write_bytes(ZK.CIRCUIT)
    for w, x in rw["windows"].items():
        pre = f"w{w:03d}-"
        (v / f"{pre}registration.json").write_text(json.dumps(x["registration"]))
        (v / f"{pre}one-stage.json").write_text(json.dumps(x["one_stage"]))
        (v / f"{pre}delivered.json").write_text(json.dumps({"salts": x["salts"], "output": 3}))
        (v / f"{pre}pub-1.bin").write_bytes(ZK.PUBLIC)
        (v / f"{pre}session-s0-session.json").write_bytes(b"{}")
    got = L.run_windows({"dir": str(s)}, "honest")
    assert (got["one_stage"], got["circuit"]) == (rw["one_stage"], str(v / "circuit.txt"))
    for w, x in rw["windows"].items():
        pre = f"w{w:03d}-"
        assert got["windows"][w] == x | {"blobs": [str(v / f"{pre}pub-1.bin"), str(v / f"{pre}session-s0-session.json")]}


def test_a_clock_certificate_is_the_evidence_without_the_session_or_the_sync(tmp_path):
    rw = session_run([1], tmp_path)
    c = rw["windows"][1]["clock"]
    assert sorted(c) == ["digest", "direction", "key_id", "link", "signature", "window"]
    assert L.clock_evidence({"session": rw["deployment"]["session"], **c}) == c


def test_a_rows_case_is_the_rows_forms():
    import warden_serve as WS

    schedule = {"bucket_ms": 100, "cell_bytes": 512, "delay": 27, "egress_cells": 65, "every": 16,
                "ingress_buckets": 12, "ingress_cells": 47, "lanes": 4, "queue": 4, "slots": 4, "window": 32}
    dep = {"key": {"params": [64, 1, 6], "public": "ab" * 24, "key_id": "ab" * 16}, "params": schedule,
           "session": "cd" * 32}

    def rows(d: str, w: int) -> dict:
        return {"direction": d, "window": w, "root": f"{d}{w}"}

    def cert(link: str, d: str, w: int) -> dict:
        return {"digest": f"{link}{d}{w}", "direction": d, "key_id": "ab" * 16, "link": link, "signature": "ee",
                "window": w}

    rw = {"deployment": dep, "windows": {w: {
        "registrations": {d: rows(d, w) for d in ("egress", "ingress")},
        "certificates": {n: cert(link, d.value, w) for n, (link, d) in WS.CERTIFIED.items()}} for w in (3, 4)}}
    scan, desc = {**LOOP, "program": "dd" * 64}, {"schema": "verity-ir/descriptor/v1", "root": "scan"}
    scans = {w: (scan | {"partition": f"scan/{w}"}, {"verdict": "accept", "w": w}, desc) for w in (3, 4)}
    audits = [(LOOP, {"verdict": "accept"})]
    case = L.rows_case_of(rw, [3, 4], scans, audits, 3, [FREE], [PROGRAM])
    p = case["params"]
    assert sorted(p) == ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule",
                         "session", "windows"]
    assert (p["first"], p["windows"], p["link"], p["program"], p["key"]) == (3, 2, WS.LINK, "dd" * 64, "ab" * 24)
    assert p["audits"] == [{"free": FREE, "program": PROGRAM, "steps": 64}] and p["scan"] == desc
    assert p["schedule"] == schedule and p["lms"] == {"H": 6, "N": 64, "W": 1, "toy": False}
    c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    assert c == {"audits": [LOOP], "windows": [scans[3][0], scans[4][0]]}
    assert e["audits"] == [{"verdict": "accept"}] and [x["answer"]["w"] for x in e["windows"]] == [3, 4]
    for w, x in zip((3, 4), e["windows"]):
        assert sorted(x) == ["answer", "certificates", "registrations"]
        assert [(r["direction"], r["window"]) for r in x["registrations"]] == [("ingress", w), ("egress", w)]
        assert [(c["link"], c["direction"]) for c in x["certificates"]] == [
            (WS.LINK, "ingress"), (WS.LINK, "egress"), (f"{WS.LINK}-fails", "egress")]
    with pytest.raises(ValueError, match="free-input counts"):
        L.rows_case_of(rw, [3, 4], scans, audits, 3, [], [PROGRAM])
    with pytest.raises(RuntimeError, match="2 Programs"):
        L.rows_case_of(rw, [3, 4], scans | {4: (LOOP, {}, desc)}, audits, 3, [FREE], [PROGRAM])
    with pytest.raises(RuntimeError, match="2 descriptors"):
        L.rows_case_of(rw, [3, 4], scans | {4: (scans[4][0], {}, desc | {"root": "other"})}, audits, 3, [FREE],
                       [PROGRAM])


def mirror_rows_run(monkeypatch, seed: int) -> tuple[dict, dict, dict, dict, list[int]]:
    """A signed rows period of the mirror's (`period_case.build`) and the warden run it is the case of: (case,
    commitment, evidence, run as `rows_case_of` reads it, link-windows)."""
    import warden_serve as WS

    case = PERIOD.build("rows-honest", seed)
    p = case["params"]
    c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    lp = p["lms"]
    monkeypatch.setattr(WS, "LINK", p["link"])
    dep = {"key": {"params": [lp["N"], lp["W"], lp["H"]], "public": p["key"]}, "params": p["schedule"],
           "session": p["session"]}
    ws = list(range(p["first"], p["first"] + p["windows"]))
    rw = {"deployment": dep, "windows": {w: {
        "registrations": dict(zip(("ingress", "egress"), x["registrations"])),
        "certificates": dict(zip((WS.INGRESS, WS.EGRESS, WS.FAILS), x["certificates"]))}
        for w, x in zip(ws, e["windows"])}}
    return case, c, e, rw, ws


def test_a_signed_rows_period_rebuilt_by_rows_case_of_is_the_mirrors_and_refused_at_the_wiring(monkeypatch):
    case, c, e, rw, ws = mirror_rows_run(monkeypatch, 19)
    p = case["params"]
    assert not p["lms"]["toy"] and len(c["audits"]) == 2 and p["first"] == 3
    scans = {w: (reg, x["answer"], p["scan"]) for w, reg, x in zip(ws, c["windows"], e["windows"])}
    free, programs = [a["free"] for a in p["audits"]], [a["program"] for a in p["audits"]]
    got = L.rows_case_of(rw, list(ws), scans, list(zip(c["audits"], e["audits"])), p["first"], free, programs)
    assert (got["commitment"], got["evidence"]) == (case["commitment"], case["evidence"])
    assert sorted(got["params"]) == PERIOD.ROWS_PARAMS_KEYS
    T = p["schedule"]["window"]
    assert got["params"]["scan"] == p["scan"]
    assert got["params"]["audits"] == [{"free": f, "program": q, "steps": len(ws) * T} for f, q in zip(free, programs)]
    a = PERIOD.reference(got)
    assert (a["verdict"], a["reason"]) == ("reject", PERIOD.UNWIRED), a
    monkeypatch.setattr(PERIOD, "_unwired", lambda p: None)
    a = PERIOD.reference(got)
    assert (a["verdict"], a["public"]["windows"], len(a["public"]["scans"])) == ("accept", len(ws), len(ws)), a


@pytest.mark.parametrize("delta", BC.BLOCK_DELTAS)
def test_each_term_is_decided_at_its_share_of_the_periods_delta(monkeypatch, delta, tmp_path):
    assert L.term_delta(delta, 5) == delta / 5
    with pytest.raises(ValueError, match="at least one term"):
        L.term_delta(delta, 0)
    accept = {"verdict": "accept"}
    rw, second = session_run([1, 2], tmp_path), AUDITED | {"partition": "cc" * 64}
    records = [(AUDITED, ACCEPTED | {"registration": AUDITED}), (second, ACCEPTED | {"registration": second})]
    with monkeypatch.context() as m:
        proofs_accepted(m)
        for d, eps in ((L.term_delta(delta, len(records)), delta), (delta, 2 * delta)):
            audits = [(r, L.audit_answer(r, accept, rec, d)) for r, rec in records]
            a = decided(L.case_of(rw, [1, 2], audits, 1, [FREE, FREE], [PROGRAM, PROGRAM]))
            assert a["verdict"] == "accept" and Fraction(a["public"]["eps"]) == eps
    L.holds_at(0, a, 2 * delta)
    with pytest.raises(RuntimeError, match="term_delta"):
        L.holds_at(0, a, delta)
    L.holds_at(0, {"verdict": "reject", "public": None}, delta)
