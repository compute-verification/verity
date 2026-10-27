"""The communication leg (`legs/certifier.py`) on what it reads and what it writes, without a certifier session:

* a served plan of the period's prompts, a link-window per group, puts period p's requests in link-window p, each with
  its own prompt, and has no period past the session's; the toy's stub serves those bodies (``ignore_eos``, no
  ``min_tokens``) and refuses any other, a body that still sends ``min_tokens`` among them;
* `run_windows` reads each link-window's registration, one-stage v2 evidence, delivered salts and the files its proof
  names off N2's verifier directory, with the link's v2 parameters and circuit file, and `_keep` keeps that circuit file;
* the integrity leg's accepted audit is re-decided at the leg's δ (one-stage's bound on wrong units), and its refusal
  kept as it is;
* the certified text matches the registered outputs by stream, else by content, and a changed byte is no match; a
  period's relayed outputs are its link-window's responses in the plan's order, and a short or non-hex one is refused;
* a tied period reports its integrity audit's output class and Lean's refusals at the output port's registered read,
  and the leg asserts an honest relay's accepted audit proved the class and a tampered relay's was refused there;
* `case_of`'s case of signed link-windows and the period's audit is the period check's (its Python mirror, which
  `network-period` is held to): the mirror runs one-stage v2 on each link-window's proof (undecided with no session's
  proof), refuses a link-window v2 refuses, a case naming another first window and an audit registered over another
  Program, and once v2 accepts the proofs accepts with K and eps at each δ, the audit charged its Program's units'
  committed bits (16 a unit, or 40);
* a loop period's value files give its requests' prompt tokens, steps and width, and its registered outputs are the
  stub's for those tokens (`torch_loop_stub.step_outputs`), so an honest certifier session's egress carries them; its
  audit's Program, rebuilt from its records (`audit_program`), is its registration's, of its population's units, each
  committing a 16-bit output (slow);
* the free inputs are every unit's input rows less the wired, the external and an uncut registered read's; a member's
  units read the same rows off a shared-row member's tables through its refs as off a plain member's bodies (each
  written by `rows.Committed`), and a body of another size or a ref past its table is refused; where #4426's
  registered weights are in the tree, a ``--registered`` period's shared-row members read unit for unit as the plain
  period's, and its free inputs are `interface.input_count`'s (slow); a wired window gives the leg the per-coordinate
  window's requests and outputs at a cap that splits its records across members, and a record that spans other than
  one unit a request is refused; a ``--registered --wired`` period gives it its requests, the stub's outputs (each
  wired down_proj unit's whole ``o0`` row) and its free inputs, the requests' step-0 inputs alone (slow);
* ``CERTIFIER_PATH`` picks the path, the certifier-only one by default, and the warden path's stubbed lines are the
  other's less its records' gap; ``WARDEN_TAMPER`` names the stub's tampers;
* of the windows the runner audits at once, one makes the run's session and the others read it (`once`), and the
  session proves N2's members one firewalled session at a time (``n2_zk.py prove --jobs 1``);
* the warden session's run parses as the gate's scan rehearsal at the scan's own schedule, a period's link-windows are
  its group's in ``groups.json``, and the certified ids match the registered outputs' by connection, else by content,
  and a stub's changed byte, extra id or lost id is no match;
* only the warden session's public files are published, its certificates without the heads and salts they sign;
* a session step that fails (the leg's relay or the runner's, N2's prove, a warden rehearse) raises an error carrying
  its log's last `LOG_TAIL` lines and naming the log's copy where custody keeps it (the leg's ``out``, the runner's
  ``--out/certifier-session``), since the scratch's is gone by the time anyone reads the error;
* the scan's answer of link-window 0 is its audit record re-decided at the leg's δ, a refusal a reject, beside its
  Program's descriptor, and `rows_case_of`'s case is the rows form's: a signed rows period of the mirror's
  (`period_case.build`), taken apart into a warden run's windows and scans and rebuilt, is the same commitment and
  evidence, and the mirror refuses it by name at the scan's answer (one-stage v2 can't run the scan's proof);
* each term of a case is decided at its share of the δ (`term_delta`): a signed clock case of two audits, its
  link-windows' proofs accepted, is accepted with eps that δ, an unsplit one with eps their sum, which `holds_at`
  refuses;
* `block_leg` turns a period's accepted answer, clock form (r20261025-213746-7692's period 0) or rows form
  (r20261025-235229-305e's), into the block record's certifier leg, K held at each δ its exact eps is at most, and
  refuses a rejected answer; where #4454's ``verity/block-record/v1`` is in the tree, each leg fits its schema, is
  `block_record.leg`'s of the runner's row, reproduces its fixture's certifier leg from that leg's answer, and a fixture
  record carrying it checks clean.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import threading
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.certifier.certificate import SALT_BYTES, Certificate, Direction, clock_commitment
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.commitments.window import window_domain
from verity.ml.crypto import lms
from verity_all_protocols import runner as RN, table as T
from verity_all_protocols.table import check_answer

L = T.load("certifier")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "network_traces"))
import certifier_serve as cs  # noqa: E402
import torch_loop_stub as S  # noqa: E402


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

PROMPTS, STEPS, H = [[3], [7]], 2, 128
FRAMES = STEPS * -(-4 * H // L.TOKEN_BYTES)
LOOP = {"format": R.FORMAT, "protocol": R.PROTOCOL, "program": "aa" * 64, "partition": "bb" * 64, "population": 1024,
        "law": {"law": "subset", "k": 2}}
DRAW = {"law": "subset", "k": 2, "population": 1024, "units": [700, 5]}
ACCEPTED = {"accepted": True, "registration": LOOP, "draw": DRAW, "proved_units": [5, 700]}
#: the audit's Program, the period mirror's stand-in population of LOOP's 1024 units, each committing 16 bits, and
#: LOOP registered over it
PROGRAM = json.loads(PERIOD.audit_descriptor(((1024, 2),)))
AUDITED = LOOP | {"program": PERIOD.program_sha512(PROGRAM)}


def audited(delta: Fraction, program: dict = PROGRAM) -> tuple[dict, dict]:
    """LOOP registered over ``program`` and its accepted audit re-decided at ``delta``."""
    reg = LOOP | {"program": PERIOD.program_sha512(program)}
    return reg, L.audit_answer(reg, {"verdict": "accept"}, ACCEPTED | {"registration": reg}, delta)


def plan(periods: int) -> tuple[dict, dict]:
    probed = {"url": "stub", "body": cs.request_body("stub", STEPS, PROMPTS[0]),
              "responses": [{"events": FRAMES, "first_s": 0.0, "last_s": 0.2}]}
    return cs.served_plan(probed, len(PROMPTS) * periods, in_flight=len(PROMPTS), frames=FRAMES, prompts=PROMPTS,
                          window_per_group=True)


def test_a_period_is_its_own_link_window_with_its_own_prompts_and_the_last_takes_the_tail():
    report, p = plan(3)
    assert [L.period_windows(p, i, len(PROMPTS), 3, [0, 1, 2]) for i in range(3)] == [[0], [1], [2]]
    assert [L.period_windows(p, i, len(PROMPTS), 3, [0, 1, 2, 3, 4]) for i in range(3)] == [[0], [1], [2, 3, 4]]
    assert [q["body"]["prompt"] for q in p["requests"]] == [[3], [7]] * 3
    for period, periods in ((3, 3), (2, 2)):
        with pytest.raises(RuntimeError, match=f"carries {min(3, periods)} periods, not period {period}"):
            L.period_windows(p, period, len(PROMPTS), periods, [0, 1, 2])
    assert cs.ar.Setup(report).sync_set.deltas


def test_the_stub_serves_each_fixed_plan_body_of_the_loops_steps():
    _, p = plan(1)
    model = f"torch-loop/mlp-w8a8-h{H}"
    assert all(q["body"]["ignore_eos"] is True and "min_tokens" not in q["body"] for q in p["requests"])
    assert [S.refusal(q["body"] | {"model": model}, STEPS) for q in p["requests"]] == [None, None]
    assert "steps" in S.refusal(cs.request_body(model, STEPS + 1, [3]), STEPS)
    assert "greedy" in S.refusal(cs.request_body(model, STEPS, [3], temperature=1.0), STEPS)
    assert "ignore_eos" in S.refusal(cs.request_body(model, STEPS, [3]) | {"ignore_eos": False,
                                                                           "return_token_ids": True}, STEPS)


def test_the_stub_refuses_a_body_that_still_sends_min_tokens():
    model = f"torch-loop/mlp-w8a8-h{H}"
    fixed = cs.request_body(model, STEPS, [3])
    varied = cs.varied_body(model, {"prompt": [3], "max_tokens": STEPS, "seed": 1}) | {"return_token_ids": True}
    assert S.refusal(fixed, STEPS) is None and S.refusal(varied, STEPS) is None
    for body in (fixed | {"min_tokens": STEPS}, fixed | {"min_tokens": 1}, varied | {"min_tokens": 1}):
        assert "min_tokens" in S.refusal(body, STEPS)


@pytest.mark.parametrize("delta, bound", [(Fraction(1, 100), 921), (Fraction(1, 10 ** 6), 1022)])
def test_an_accepted_audit_is_re_decided_at_the_legs_delta(delta, bound):
    a = L.audit_answer(LOOP, {"verdict": "accept"}, ACCEPTED, delta)
    assert check_answer(a) == a and a["public"]["bound"] == bound
    assert CC.integrity(a["profile"])["delta"] == delta
    assert a["public"]["drawn"] == a["public"]["proved"] == [5, 700]


def test_a_refused_audit_stays_refused_and_a_record_for_another_registration_is_an_error():
    refused = {"verdict": "reject", "reason": "records", "commitment": R.digest(LOOP), "profile": None,
               "protocol": R.PROTOCOL, "public": None}
    assert L.audit_answer(LOOP, refused, None, Fraction(1, 100)) == refused
    for record in (None, ACCEPTED | {"accepted": False}, ACCEPTED | {"registration": LOOP | {"population": 1023}}):
        with pytest.raises(RuntimeError, match="audit record"):
            L.audit_answer(LOOP, {"verdict": "accept"}, record, Fraction(1, 100))


def test_the_certified_text_is_matched_to_the_registered_outputs():
    outputs = {(f"r{i}", k): bytes([16 * i + k]) * 2 * H for i in range(2) for k in range(STEPS)}
    text = [b"".join(outputs[f"r{i}", k] for k in range(STEPS)).hex().encode() for i in range(2)]
    events = [t[j:j + L.TOKEN_BYTES] for t in text for j in range(0, len(t), L.TOKEN_BYTES)]
    reqs = [{"stream": 1, "session": "s.0"}, {"stream": 2, "session": "s.1"}]
    m = L.match_outputs(reqs, events, FRAMES, outputs, STEPS)
    assert m["equal"] and m["events"] == 2 * FRAMES and [r["by"] for r in m["requests"]] == ["position", "position"]
    swapped = events[FRAMES:] + events[:FRAMES]
    m = L.match_outputs(reqs, swapped, FRAMES, outputs, STEPS)
    assert m["equal"] and [r["by"] for r in m["requests"]] == ["content", "content"]
    bad = list(events)
    bad[FRAMES + 3] = (b"0" if bad[FRAMES + 3][:1] != b"0" else b"1") + bad[FRAMES + 3][1:]
    m = L.match_outputs(reqs, bad, FRAMES, outputs, STEPS)
    assert not m["equal"] and [r["by"] for r in m["requests"]] == ["position", None]
    extra = events[:FRAMES] + [b"00"] + events[FRAMES:]
    assert [r["by"] for r in L.match_outputs(reqs, extra, FRAMES, outputs, STEPS)["requests"]] == ["position", None]
    assert not L.match_outputs(reqs, events + [b"00"], FRAMES, outputs, STEPS)["equal"]
    assert not L.match_outputs(reqs, events[:-1], FRAMES, outputs, STEPS)["equal"]
    with pytest.raises(ValueError, match="plan requests"):
        L.match_outputs(reqs[:1], events, FRAMES, outputs, STEPS)


def test_a_periods_relayed_outputs_are_its_link_windows_responses_in_the_plans_order(tmp_path, monkeypatch):
    report, p = plan(2)
    run = tmp_path / "rehearsal" / "honest"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps(p))
    (run / "calibration.json").write_text(json.dumps(report))
    said = [bytes([16 * i + k for k in range(STEPS)]) * 2 * H for i in range(2)]
    events = [t[j:j + L.TOKEN_BYTES] for t in (s.hex().encode() for s in said) for j in range(0, len(t), L.TOKEN_BYTES)]
    got: dict[str, list] = {"events": events}
    monkeypatch.setattr(L, "certified_events", lambda r, fb, ws: got["events"] if ws == [1] else [])
    rec = {"dir": str(tmp_path), "prompts": [3, 7], "frames": FRAMES, "h": H, "steps": STEPS}
    assert L.relayed(rec, "honest", 1) == (1, said)
    with pytest.raises(RuntimeError, match="canonical events"):
        L.relayed(rec, "honest", 0)
    got["events"] = events[:-1] + [b"zz" * (L.TOKEN_BYTES // 2)]
    with pytest.raises(RuntimeError, match="not the hex of"):
        L.relayed(rec, "honest", 1)


def test_a_tied_period_reports_its_audits_output_class_and_lean_refusals_of_the_output_port(tmp_path):
    (tmp_path / "p").mkdir()
    audit = tmp_path / "out" / "integrity" / "audit"
    audit.mkdir(parents=True)
    period = {"period": 0}
    (tmp_path / "p" / "period.json").write_text(json.dumps(period))
    assert L.tie_of(tmp_path / "p", tmp_path / "out" / "communication") is None
    tie = {"value": "out", "port": "o0", "stratum": "output class", "strategy": "byte", "window": 0}
    (tmp_path / "p" / "period.json").write_text(json.dumps(period | {"tie": tie}))
    why = {"lean-verify/m17": "registered reads: port o0: table row 0 does not open out's registered root at position 0",
           "lean-verify/m3": "zk inner: τ does not match its commitment"}
    (audit / "summary.json").write_text(json.dumps({"audit": {"why": why},
                                                    "outputs": {"units": 128, "proved": False}}))
    got = L.tie_of(tmp_path / "p", tmp_path / "out" / "communication")
    assert got == {**tie, "outputs": {"units": 128, "proved": False}, "refused": {"lean-verify/m17": why["lean-verify/m17"]}}
    L.tie_holds(0, got, "reject")
    for t, integrity in ((got | {"refused": {}}, "reject"), (got, "accept"),
                         (got | {"strategy": "honest", "refused": {}}, "accept")):
        with pytest.raises(RuntimeError, match="registered|output class"):
            L.tie_holds(0, t, integrity)
    L.tie_holds(0, got | {"strategy": "honest", "outputs": {"units": 128, "proved": True}}, "accept")
    L.tie_holds(0, got | {"strategy": "honest"}, "reject")
    assert L.STUBBED["tied"] == (L._PROXY_GAP, L._OFF_HOST_GAP) and "out-records.jsonl" in L.PUBLIC_SESSION
    assert not any("witness" in f for f in L.PUBLIC_SESSION)


ZK = PERIOD.ZK


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


@pytest.mark.parametrize("delta", L.DELTAS)
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
    v2 parameters and circuit file) and the client's certificates, and `_keep` keeps the member's circuit file but no
    statement's copy of it or prover's file."""
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
    kept = tmp_path / "kept"
    for f in ("member/verifier/circuit.txt", "member/w000/statement/circuit.txt", "member/w000/statement/inst-1.bin",
              "member/prover-instances.bin", "member/verifier/w000-pub-1.bin"):
        (s / "src" / f).parent.mkdir(parents=True, exist_ok=True)
        (s / "src" / f).write_bytes(b"x")
    L._keep(s / "src", kept)
    assert sorted(str(f.relative_to(kept)) for f in kept.rglob("*") if f.is_file()) == [
        "member/verifier/circuit.txt", "member/verifier/w000-pub-1.bin"]


def test_a_clock_certificate_is_the_evidence_without_the_session_or_the_sync(tmp_path):
    rw = session_run([1], tmp_path)
    c = rw["windows"][1]["clock"]
    assert sorted(c) == ["digest", "direction", "key_id", "link", "signature", "window"]
    assert L.clock_evidence({"session": rw["deployment"]["session"], **c}) == c


def circuit(wires: list = (), external: list = ()) -> dict:
    """A loop window's ``circuit.json`` as `torch_loop` writes it: one template of 1,024 units, four input rows."""
    t = 'ScaledMmS8Coordinate_v1{K=128,DOT={"fn":"BlackwellS8ImmaDot32_v1"}}'
    rows = [["x", "s8", 1024, ["x"]], ["sx", "u32", 32, ["sx"]], ["wrow", "s8", 1024, ["w"]],
            ["sw", "u32", 32, ["sw"]]]
    return {"templates": [[t, 1024]], "statements": {t: {"in_rows": rows}}, "wires": list(wires),
            "external": list(external)}


FREE = 1024 * (1024 + 32 + 1024 + 32)


def test_free_inputs_are_every_units_input_rows_less_the_wired_and_the_external():
    c = circuit()
    t = c["templates"][0][0]
    assert L.free_inputs(c) == {"bits": FREE, "per_template": {t: FREE}}
    assert L.free_inputs(circuit(external=[f"{t}/x"]))["bits"] == FREE - 1024 * 1024
    wire = {"reader": t, "port": "w", "writer": t, "out": "o0", "at": 512, "radix": [256], "base": 0, "step": [1]}
    assert L.free_inputs(circuit(wires=[wire]))["bits"] == FREE - 256 * 1024
    with pytest.raises(ValueError, match="no template"):
        L.free_inputs(circuit(external=["other/x"]))
    reads = [{"reader": t, "port": p, "param": p, "at": 0, "radix": [4, 2, 128], "base": 0, "step": [0, 128 * w, w]}
             for p, w in (("w", 64), ("sw", 2))]
    assert L.free_inputs(circuit() | {"registered": {"params": {}, "reads": reads}})["bits"] == 1024 * (1024 + 32)
    half = {**reads[0], "at": 256, "radix": [4, 128]}
    assert L.free_inputs(circuit(wires=[wire]) | {"registered": {"reads": [half]}})["bits"] == FREE - 512 * 1024
    assert L.free_inputs(circuit() | {"registered": {"reads": [reads[0] | {"cut": [0, 2]}]}})["bits"] == FREE
    with pytest.raises(ValueError, match="registered read's reader"):
        L.free_inputs(circuit() | {"registered": {"reads": [reads[0] | {"reader": "other"}]}})


def test_a_shared_row_members_units_read_their_rows_through_its_refs(tmp_path):
    from types import SimpleNamespace

    from verity.ml.windows import rows as CS

    rng = np.random.default_rng(0)
    n, ports, outs, reads = 6, (("x", 3), ("w", 2)), (("o0", 1),), np.array([0, 1, 1, 0, 1, 0])
    x, w = rng.integers(0, 1 << 16, (n, 3), dtype=np.uint16), rng.integers(0, 1 << 16, (2, 2), dtype=np.uint16)
    o0 = np.zeros((n, 64), np.uint16)
    o0[:, 0] = rng.integers(0, 1 << 16, n, dtype=np.uint16)
    want = np.concatenate([x, w[reads], o0], axis=1)

    def written(d: Path, m: CS.Member, rows: dict, refs: np.ndarray | None) -> dict:
        size = {p: len(r) for p, r in rows.items()}
        c = CS.Committed(SimpleNamespace(members=[m]), {}, {}, {m.port(p): bytes([7]) * CS.BC_BYTES * k for p, k in size.items()},
                         {m.port(p): bytes([9]) * CS.SALT_BYTES * k for p, k in size.items()},
                         {m.port(p): r for p, r in rows.items()}, {}, {} if refs is None else {0: refs})
        d.mkdir()
        (d / "m0-public.bin").write_bytes(c.public_body(0))
        (d / "m0-private.bin").write_bytes(c.private_body(0))
        return {"k": 0, "name": m.key, "n": n, "ports": [list(p) for p in m.all_rows],
                **({"shared_rows": {**dict(m.tables), **{p: n for p, _ in m.out_rows}}} if m.tables else {})}

    plain = written(tmp_path / "plain", CS.Member("t", ports, outs, n, name="t/c0"), {"x": x, "w": w[reads], "o0": o0},
                    None)
    refs = np.stack([np.arange(n), reads, np.arange(n)], axis=1).astype(np.uint32)
    shared = written(tmp_path / "shared", CS.Member("t", ports, outs, n, 0, (("x", n), ("w", 2)), name="t/c0"),
                     {"x": x, "w": w, "o0": o0}, refs)
    assert shared["shared_rows"] == {"x": n, "w": 2, "o0": n}
    assert np.array_equal(L.member_rows(tmp_path / "plain", plain), want)
    assert np.array_equal(L.member_rows(tmp_path / "shared", shared), want)
    with pytest.raises(ValueError, match="shared-row member's"):
        L.member_rows(tmp_path / "shared", shared | {"shared_rows": {"x": n, "w": 3, "o0": n}})
    with pytest.raises(ValueError, match="instances' rows"):
        L.member_rows(tmp_path / "plain", plain | {"n": n - 1})
    pub = tmp_path / "shared" / "m0-public.bin"
    pub.write_bytes(pub.read_bytes()[:-4 * n * 3] + np.where(refs == 1, 2, refs).astype("<u4").tobytes())
    with pytest.raises(ValueError, match="past its port's table"):
        L.member_rows(tmp_path / "shared", shared)


def n2(exit: int = 0, **member) -> dict:
    m = {"ok": True, "leaked": [], "verifier_found": [], "structured": [], "control_holds": True}
    return {"exit": exit, "ok": exit == 0, "members": {"honest": m, "extra": m | member}}


def test_n2s_prove_is_refused_on_a_leak_a_blind_scan_or_its_exit():
    assert L.n2_refusal(n2()) is None
    assert "leaked" in L.n2_refusal(n2(leaked=["salt"]))
    assert "leaked" in L.n2_refusal(n2(verifier_found=["head"]))
    assert "control" in L.n2_refusal(n2(control_holds=False))
    assert "control" in L.n2_refusal(n2(control_holds=None))
    assert "exited 1" in L.n2_refusal(n2(1, ok=False))


def test_only_the_sessions_public_files_are_published(tmp_path):
    s = tmp_path / "scratch" / L.SESSION
    for strategy in ("honest", "extra"):
        run = s / "rehearsal" / strategy
        for f in (*L.PUBLIC_SESSION, "plan.json", "requests.records", "responses.records", "shim.jsonl",
                  "certifier/clocks.jsonl", "client/syncs.jsonl"):
            (run / f).parent.mkdir(parents=True, exist_ok=True)
            (run / f).write_text(f)
    out = tmp_path / "out" / L.SESSION
    L.publish({"dir": str(s), "strategies": ["honest", "extra"]}, out)
    got = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())
    assert got == sorted(f"{st}/{f}" for st in ("honest", "extra") for f in L.PUBLIC_SESSION)


def test_the_path_is_certifier_paths_and_the_certifier_only_one_by_default(monkeypatch):
    monkeypatch.setattr(L, "warden_leg", lambda *a: "warden")
    monkeypatch.setattr(L, "certifier_leg", lambda *a: "certifier")
    monkeypatch.delenv("CERTIFIER_PATH", raising=False)
    assert L.leg(Path("p"), "node", Path("o")) == "certifier"
    for path in L.PATHS:
        monkeypatch.setenv("CERTIFIER_PATH", path)
        assert L.leg(Path("p"), "node", Path("o")) == path
    monkeypatch.setenv("CERTIFIER_PATH", "proxy")
    with pytest.raises(RuntimeError, match="CERTIFIER_PATH"):
        L.leg(Path("p"), "node", Path("o"))
    assert L.STUBBED["warden"][:2] == L.STUBBED["certifier"][1:]


def test_windows_audited_at_once_make_the_runs_one_session_once(tmp_path):
    s, entered, go, made, got = tmp_path / L.SESSION, threading.Event(), threading.Event(), [], {}

    def first(d: Path) -> dict:
        entered.set()
        go.wait()
        made.append("first")
        (d / "session.json").write_text(json.dumps({"by": "first"}))
        return {"by": "first"}

    def second(d: Path) -> dict:
        made.append("second")
        return {"by": "second"}

    a = threading.Thread(target=lambda: got.update(a=L.once(s, first)))
    a.start()
    entered.wait()
    b = threading.Thread(target=lambda: got.update(b=L.once(s, second)))
    b.start()
    go.set()
    a.join()
    b.join()
    assert made == ["first"] and got == {"a": {"by": "first"}, "b": {"by": "first"}}


def test_the_session_proves_n2s_members_one_firewalled_session_at_a_time(tmp_path, monkeypatch):
    period, s, calls = tmp_path / "period", tmp_path / L.SESSION, []
    period.mkdir()
    s.mkdir()
    (period / "period.json").write_text(json.dumps({"periods": 2}))

    class Stub:
        def terminate(self) -> None:
            pass

        def wait(self) -> None:
            pass

    def run(argv, log, env=None) -> int:
        calls.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text("{}")
        return 0

    monkeypatch.setattr(L, "_tools", lambda d: {"flock-circuit": "fc", "flock-verify": "fv", "flock-firewall": "ff",
                                                 "m0-pythonpath": ""})
    monkeypatch.setattr(L, "lean_exes", lambda: {"network-records": "nr"})
    monkeypatch.setattr(L, "requests_of", lambda d: ([1, 2], 3, 4))
    monkeypatch.setattr(L, "_stub", lambda *a, **k: (Stub(), "http://stub"))
    monkeypatch.setattr(L, "_run", run)
    monkeypatch.setenv("CERTIFIER_TAMPER", "extra")
    L._session(period, s)
    n2 = next(c for c in calls if Path(c[1]).name == "n2_zk.py")
    assert n2[n2.index("--rounds") + 1] == "honest,extra" and n2[n2.index("--jobs") + 1] == "1"


def test_the_warden_sessions_tampers_are_the_stubs(monkeypatch):
    monkeypatch.delenv("WARDEN_TAMPER", raising=False)
    assert L.warden_tampers() == list(L.WARDEN_TAMPERS) and set(L.WARDEN_TAMPERS) <= set(S.TAMPERS)
    monkeypatch.setenv("WARDEN_TAMPER", "")
    assert L.warden_tampers() == []
    monkeypatch.setenv("WARDEN_TAMPER", "extra,byte")
    assert L.warden_tampers() == ["extra", "byte"]
    monkeypatch.setenv("WARDEN_TAMPER", "drop")
    with pytest.raises(RuntimeError, match="WARDEN_TAMPER"):
        L.warden_tampers()


TOOLS = {"flock-circuit": "fc", "flock-verify": "fv", "flock-firewall": "ff", "m0-pythonpath": ""}


def failing(monkeypatch, fails: set[str]) -> None:
    """The session's stub, Lean executables and requests stubbed; each step it runs writes 100 lines to its log, and
    the steps in ``fails`` (``rehearse``, ``prove``) exit 3 with nothing else, the others as if they succeeded."""
    class Stub:
        def terminate(self) -> None:
            pass

        def wait(self) -> None:
            pass

    def run(argv, log, env=None) -> int:
        log.write_text("".join(f"{log.stem} line {i}\n" for i in range(100)))
        step = argv[2]
        if step not in fails:
            done = Path(argv[argv.index("--out") + 1]) / "summary.json"
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text("{}")
        return 3 if step in fails else 0

    monkeypatch.setattr(L, "lean_exes", lambda: {"network-records": "nr"})
    monkeypatch.setattr(L, "requests_of", lambda d: ([3, 7], STEPS, H))
    monkeypatch.setattr(L, "_stub", lambda *a, **k: (Stub(), "http://127.0.0.1:1"))
    monkeypatch.setattr(L, "scan_schedule", lambda env=None: [])
    monkeypatch.setattr(L, "_run", run)


def carries_tail(error: Exception, why: str, kept: Path) -> bool:
    head, *tail = str(error).splitlines()
    return (head == f"{why} ({kept}); its last {L.LOG_TAIL} lines:"
            and tail == [f"{kept.stem} line {i}" for i in range(100 - L.LOG_TAIL, 100)]
            and kept.read_text() == "".join(f"{kept.stem} line {i}\n" for i in range(100)))


def test_a_failed_relay_or_prove_carries_its_logs_tail_and_keeps_the_log_where_custody_does(tmp_path, monkeypatch):
    from verity_all_protocols import tie as TIE

    period, out = tmp_path / "scratch" / "w0", tmp_path / "out"
    period.mkdir(parents=True)
    (period / "period.json").write_text(json.dumps({"periods": 1, "tools": TOOLS}))
    failing(monkeypatch, {"rehearse"})
    with pytest.raises(RuntimeError) as got:
        L.session(period, out / "w0" / "certifier")
    assert carries_tail(got.value, "certifier_serve.py rehearse exited 3 with no summary",
                        out / "w0" / "certifier" / "rehearse.log")
    monkeypatch.setattr(TIE, "load", lambda name: L)
    monkeypatch.setattr(TIE, "_loop", lambda: Namespace(H=H))
    with pytest.raises(RuntimeError) as got:
        TIE.relay(tmp_path / "runner-scratch", [3, 7], STEPS, 1, TOOLS, False, TIE.OC.OutShape("aa" * 64, "out", 1, H),
                  out)
    assert carries_tail(got.value, "certifier_serve.py rehearse exited 3 with no summary",
                        out / L.SESSION / "rehearse.log")
    failing(monkeypatch, {"prove"})
    with pytest.raises(RuntimeError) as got:
        L.session(period, out / "w1" / "certifier")
    assert carries_tail(got.value, "n2_zk.py prove exited 3 with no summary", out / "w1" / "certifier" / "n2zk.log")


def test_a_failed_warden_rehearse_carries_its_logs_tail_and_keeps_the_log_in_the_legs_out(tmp_path, monkeypatch):
    failing(monkeypatch, {"rehearse"})
    (tmp_path / "s").mkdir()
    with pytest.raises(RuntimeError) as got:
        L.warden_rehearse(tmp_path / "s", [3, 7], STEPS, H, 1, 7, ["honest"], {}, tmp_path / "out")
    assert carries_tail(got.value, "warden_serve.py rehearse (honest) exited 3 with no check.json",
                        tmp_path / "out" / "rehearse-honest.log")


def test_a_warden_run_is_the_gates_scan_rehearsal_at_the_scans_schedule(tmp_path):
    import scan_zk
    import warden_serve as WS

    argv = L.warden_argv("http://127.0.0.1:8123", tmp_path / "run", tmp_path / "prompts.json", 6, STEPS, 7,
                         L.scan_schedule())
    a = WS.parser().parse_args(argv[2:])
    assert (a.command, a.upstream, a.out, a.prompts) == ("rehearse", ("127.0.0.1", 8123), tmp_path / "run",
                                                         tmp_path / "prompts.json")
    assert a.ids and a.window_per_group and not a.fake
    assert (a.requests, a.max_tokens, a.height, a.temperature, a.top_p) == (6, STEPS, 7, 0.0, 1.0)
    assert WS._params(a) == scan_zk.PARAMS


def test_a_warden_period_is_its_groups_link_windows():
    for G in (1, 2):
        groups = {"groups": [{"group": g, "requests": [2 * g, 2 * g + 1], "windows": [g * G, (g + 1) * G]}
                             for g in range(3)]}
        assert [L.warden_period_windows(groups, p, 2) for p in range(3)] == [
            list(range(p * G, (p + 1) * G)) for p in range(3)]
    with pytest.raises(RuntimeError, match="3 groups, not period 3's"):
        L.warden_period_windows(groups, 3, 2)
    with pytest.raises(RuntimeError, match="not period 1's 3"):
        L.warden_period_windows(groups, 1, 3)


def test_the_certified_ids_are_matched_to_the_registered_outputs():
    model = S.L.Model(h=H)
    outputs = {(f"r{i}", k): o for i, t in enumerate([3, 7]) for k, o in enumerate(S.step_outputs(model, t, STEPS))}
    ids = {o: [S.scan_id(outputs[f"r{i}", k]) for k in range(STEPS)] for i, o in enumerate([5, 2])}
    m = L.match_ids([5, 2], ids, outputs, STEPS)
    assert m["equal"] and m["others"] == [] and [r["by"] for r in m["requests"]] == ["position", "position"]
    assert [r["by"] for r in L.match_ids([2, 5], ids, outputs, STEPS)["requests"]] == ["content", "content"]
    for tampered in (S.tampered([outputs["r1", k] for k in range(STEPS)], "byte"),
                     S.tampered([outputs["r1", k] for k in range(STEPS)], "extra"), [outputs["r1", 0]]):
        m = L.match_ids([5, 2], ids | {2: [S.scan_id(o) for o in tampered]}, outputs, STEPS)
        assert not m["equal"] and [r["by"] for r in m["requests"]] == ["position", None]
    m = L.match_ids([5, 2], ids | {9: [1]}, outputs, STEPS)
    assert not m["equal"] and m["others"] == [9]
    assert [r["by"] for r in L.match_ids([5, None], ids, outputs, STEPS)["requests"]] == ["position", "content"]
    assert not L.match_ids([5, None], {5: ids[5]}, outputs, STEPS)["equal"]
    with pytest.raises(ValueError, match="registered requests"):
        L.match_ids([5], ids, outputs, STEPS)


def test_only_the_warden_sessions_public_files_are_published(tmp_path):
    s = tmp_path / "scratch" / L.WARDEN_SESSION
    witness = ("cells.salts", "requests.records", "responses.records", "fails.records", "traffic.json",
               "client.json", "check.json", "warden.json", "stalls.json", "serve.log", "client.log")
    cert = {"digest": "aa", "direction": "egress", "key_id": "bb", "link": "warden", "signature": "cc", "window": 0}
    for strategy in ("honest", "late"):
        run = s / "rehearsal" / strategy
        run.mkdir(parents=True)
        for f in (*L.WARDEN_PUBLIC, *witness):
            (run / f).write_text(f)
        (run / "certified.jsonl").write_text(json.dumps({"certificate": cert, "head": "dd", "salt_c": "ee"}) + "\n"
                                             + json.dumps({"refused": "signer", "head": "ff"}) + "\n")
    (s / "prompts.json").write_text("[[3], [7]]")
    out = tmp_path / "out" / L.WARDEN_SESSION
    L.warden_publish({"dir": str(s), "strategies": ["honest", "late"]}, out)
    got = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())
    assert got == sorted(f"{st}/{f}" for st in ("honest", "late") for f in (*L.WARDEN_PUBLIC, "certificates.jsonl"))
    assert [json.loads(x) for x in (out / "late" / "certificates.jsonl").read_text().splitlines()] == [
        {"certificate": cert}, {"refused": "signer"}]


SCAN_SUMMARY = {"T": 32, "bits": {"ingress/row": 16384, "ingress/sin": 3, "ingress/prog": 5, "ingress/view": 11,
                                  "egress/state": 7, "egress/row": 65536}}


def test_a_scan_window_is_its_audit_records_answer_and_a_refused_one_a_reject(tmp_path):
    d = tmp_path / "scan" / "honest"
    (d / "honest").mkdir(parents=True)
    (d / "joint").mkdir()
    (d / "summary.json").write_text(json.dumps(SCAN_SUMMARY))
    (d / "joint" / "program.json").write_text(json.dumps(PROGRAM))
    (d / "honest" / "registration.json").write_text(json.dumps(LOOP))
    sess = {"scan_prove": {"honest": {"dir": str(d), "exit": 0}}}
    for record, verdict in ((ACCEPTED, "accept"), (ACCEPTED | {"accepted": False}, "reject"), (None, "reject")):
        (d / "honest" / "audit.json").unlink(missing_ok=True)
        if record is not None:
            (d / "honest" / "audit.json").write_text(json.dumps(record))
        reg, a, desc = L.scan_window(sess, "honest", Fraction(1, 100))
        assert reg == LOOP and desc == PROGRAM
        assert check_answer(a) == a and a["verdict"] == verdict and a["commitment"] == R.digest(LOOP)
        assert verdict == "reject" or a["public"]["bound"] == 921
    assert L.scan_window(sess, "late", Fraction(1, 100)) is None
    assert L.scan_window({}, "honest", Fraction(1, 100)) is None


def test_a_rows_case_is_the_rows_forms():
    import warden_serve as WS

    schedule = {"bucket_ms": 100, "cell_bytes": 512, "delay": 21, "egress_cells": 65, "ingress_buckets": 12,
                "ingress_cells": 47, "lanes": 4, "queue": 4, "slots": 4, "window": 32}
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
    commitment, evidence, run as `warden_windows` reads it, link-windows)."""
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


def test_a_signed_rows_period_rebuilt_by_rows_case_of_is_the_mirrors_and_refused_at_the_scans_answer(monkeypatch):
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
    assert a["verdict"] == "reject" and a["reason"].startswith(
        f"window {ws[0]}: the scan's answer: the check takes no answer for a link-window"), a


@pytest.mark.parametrize("delta", L.DELTAS)
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


ANSWERS = json.loads((Path(__file__).parent / "fixtures" / "certifier-period-answers.json").read_text())
CLOCK_RUN, ROWS_RUN = "r20261025-213746-7692", "r20261025-235229-305e"
E2E = ROOT / "benchmarks" / "e2e_demo"
BLOCK_SCHEMA = E2E / "verity_e2e_demo" / "schemas" / "block-record.v1.schema.json"


def _block_leg(run: str, delta: str) -> dict:
    return L.block_leg(ANSWERS[run]["answers"][delta], server=ANSWERS[run]["node"], runs=[run])


def _held(value: int) -> dict:
    return {"holds": True, "value": value, "epsilon": None, "by": "run"}


def _not_held(eps: str, delta: str) -> dict:
    return {"holds": False, "why": f"the run established it except with {eps}, more than {delta}"}


def test_a_period_answer_of_either_form_is_the_block_records_certifier_leg():
    clock = _block_leg(CLOCK_RUN, "1/100")
    assert (clock["protocol"], clock["owner"], clock["server"], clock["source"], clock["runs"]) == (
        L.PROTOCOL, "network-accounting", ANSWERS[CLOCK_RUN]["node"], "real", [CLOCK_RUN])
    assert clock["answer"] == {k: ANSWERS[CLOCK_RUN]["answers"]["1/100"][k]
                               for k in ("verdict", "reason", "commitment", "public")}
    assert clock["quantity"] == {"symbol": "K", "value": 2178452, "unit": "bits", "windows": 1}
    assert (clock["error"], clock["error_from"]) == ("1/100", "quantity.eps")
    assert clock["at"] == {"1/100": _held(2178452), "1/1000000": _not_held("1/100", "1/1000000")}
    assert _block_leg(CLOCK_RUN, "1/1000000")["at"] == {"1/100": _held(2180068), "1/1000000": _held(2180068)}
    rows = _block_leg(ROWS_RUN, "1/100")
    assert "scans" in rows["answer"]["public"] and rows["quantity"]["value"] == 12200606
    assert rows["error"] == "1/50" and rows["at"] == {"1/100": _not_held("1/50", "1/100"),
                                                       "1/1000000": _not_held("1/50", "1/1000000")}
    rows6 = _block_leg(ROWS_RUN, "1/1000000")
    assert rows6["error"] == "1/500000" and rows6["at"] == {"1/100": _held(12202528),
                                                            "1/1000000": _not_held("1/500000", "1/1000000")}
    answer = ANSWERS[CLOCK_RUN]["answers"]["1/100"]
    with pytest.raises(ValueError, match="certifies no K"):
        L.block_leg(ANSWERS[CLOCK_RUN]["tampered"], server="n", runs=[CLOCK_RUN])
    with pytest.raises(ValueError, match="not verity/certifier/v0"):
        L.block_leg(answer | {"protocol": "verity/sampled-proofs/one-stage/v1"}, server="n", runs=[CLOCK_RUN])
    with pytest.raises(ValueError, match="real or fixture"):
        L.block_leg(answer, server="n", runs=[CLOCK_RUN], source="synthetic")
    with pytest.raises(ValueError, match="at least one run"):
        L.block_leg(answer, server="n", runs=[])


@pytest.mark.skipif(not BLOCK_SCHEMA.exists(), reason="verity/block-record/v1 (#4454) is not in this tree yet")
def test_the_block_leg_fits_the_block_records_schema_its_fixture_and_its_runner_path():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(BLOCK_SCHEMA.read_text())
    B = _module("verity_e2e_demo_block_record", E2E / "verity_e2e_demo" / "block_record.py")
    assert [str(d) for d in L.BLOCK_DELTAS] == [str(d) for d in B.DELTAS]
    leg_schema = jsonschema.Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                                                  "$ref": "#/$defs/leg"})
    for run, x in ANSWERS.items():
        for delta, a in x["answers"].items():
            got = L.block_leg(a, server=x["node"], runs=[run])
            leg_schema.validate(got)
            row = {"answer": a, "quantity": a["public"], "owner": "network-accounting", "runs": [run]}
            assert B.leg("certifier", row, server=x["node"], runs=[run]) == got
    rec = json.loads((E2E / "tests" / "fixtures" / "block-record-r2-period0.json").read_text())
    fx = rec["legs"]["certifier"]
    assert L.block_leg({**fx["answer"], "protocol": fx["protocol"]}, server=fx["server"], runs=fx["runs"]) == fx
    for run, delta in ((CLOCK_RUN, "1/100"), (ROWS_RUN, "1/1000000")):
        legs = rec["legs"] | {"certifier": L.block_leg(ANSWERS[run]["answers"][delta], server=rec["server"],
                                                       runs=[run])}
        composed = B.compose(rec["server"], legs) | {"onsite": rec["composed"]["onsite"]}
        mine = rec | {"legs": legs, "composed": composed, "monitoring_period": B.monitoring_period(
            rec["monitoring_period"]["deployment"], Fraction(composed["joint_horizon"]["error_sum"]))}
        jsonschema.Draft202012Validator(schema).validate(mine)
        assert B.check(mine) == {}
        assert composed["joint_horizon"]["at"]["1/1000000"]["not_established"] == ["certifier"]


@pytest.mark.slow
def test_a_loop_window_registers_what_the_stub_serves(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3,7", steps=STEPS, tamper=set(), node="test-node", tools={},
                  pouw=None, delta="1/100", start=0, windows=1, weights=None, session_jobs=1)
    d, _ = RN.loop_window(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    reg = json.loads((d / "registration.json").read_text())
    audit = PERIOD._audit({"free": 0, "program": L.audit_program(d), "steps": 0}, "the audit")
    assert (audit.program, audit.units, audit.bits) == (reg["program"], reg["population"], 16)
    units = L.loop_units(d)
    assert L.requests_of(d, units) == ([3, 7], STEPS, H)
    outputs = L.registered_outputs(units)
    model = S.L.Model(h=H)
    for i, token in enumerate([3, 7]):
        assert [outputs[f"r{i}", k] for k in range(STEPS)] == S.step_outputs(model, token, STEPS)


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(S.L, "commit_weights"), reason="registered weights (#4426) are not in this tree yet")
def test_a_registered_weights_window_reads_unit_for_unit_as_the_plain_one(tmp_path):
    from verity.ml.windows import interface as IF

    a = Namespace(law="subset:2", cap=8, requests="3,7", steps=STEPS, tamper=set(), node="test-node", tools={},
                  pouw=None, delta="1/100", start=0, windows=1, weights=None, session_jobs=1)
    plain, _ = RN.loop_window(0, "test", a, tmp_path / "plain", tmp_path / "plain-out")
    weights = {"opening": tmp_path / "opening.json", "pin": tmp_path / "pin.json"}
    RN.serve(["--commit-weights", str(weights["opening"]), "--pin", str(weights["pin"])])
    d, _ = RN.loop_window(0, "test", Namespace(**{**vars(a), "weights": weights}), tmp_path / "reg", tmp_path / "reg-out")
    assert json.loads((d / "period.json").read_text())["registered"] == str(weights["pin"])
    assert all("shared_rows" in m for m in json.loads((d / "window.json").read_text())["members"])
    got, want = L.loop_units(d), L.loop_units(plain)
    assert sorted(got) == sorted(want)
    for u, (lab, rows) in want.items():
        assert got[u][0] == lab and all(np.array_equal(got[u][1][p], r) for p, r in rows.items())
    circuit = json.loads((d / "circuit.json").read_text())
    assert L.free_inputs(circuit)["bits"] == IF.input_count(circuit, pins={"w", "sw"})["free"] == 1024 * (1024 + 32)
    assert L.free_inputs(json.loads((plain / "circuit.json").read_text()))["bits"] == FREE


def test_a_wired_windows_requests_and_registered_outputs_are_the_per_coordinate_windows(tmp_path):
    """The wired circuit records each linear once for all requests (a record's ``requests``, one a unit) and its quantize
    units carry no layer: the leg reads the same prompts, steps and outputs off it as off the per-coordinate circuit, at
    a cap that splits each record across members as at one that doesn't; a record whose units aren't one a request is
    refused."""
    from verity_torch_loop import serve

    shape = ["--requests", "3,7", "--steps", "2", "--law", "stratified:3", "--run-id", "allp-cert-wired"]
    assert serve.main([*shape, "--out", str(tmp_path / "coord")]) == 0
    for cap in (1, 2):
        assert serve.main([*shape, "--out", str(tmp_path / f"wired{cap}"), "--cap", str(cap), "--wired"]) == 0
    coord, *wired = (L.loop_units(tmp_path / d) for d in ("coord", "wired1", "wired2"))
    for cap, units in zip((1, 2), wired):
        assert L.requests_of(tmp_path / f"wired{cap}", units) == L.requests_of(tmp_path / "coord", coord) == ([3, 7], 2, 128)
        outputs = L.registered_outputs(units)
        assert outputs == L.registered_outputs(coord) and sorted(outputs) == [(r, s) for r in ("r0", "r1") for s in (0, 1)]
        assert all(len(v) == 2 * 128 for v in outputs.values())
    d = tmp_path / "wired1"
    index = json.loads((d / "index.json").read_text())
    index["members"][1]["labels"][0][2]["requests"] = ["r1", "r0"]
    (d / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match=r"spans units \[0\], not one a request of 2"):
        L.loop_units(d)


@pytest.mark.slow
def test_a_wired_registered_window_gives_the_leg_its_requests_outputs_and_free_inputs(tmp_path):
    from verity.ml.windows import interface as IF

    weights = {"opening": tmp_path / "opening.json", "pin": tmp_path / "pin.json"}
    RN.serve(["--commit-weights", str(weights["opening"]), "--pin", str(weights["pin"]), "--wired"])
    a = Namespace(law="work:14", cap=2, requests="3,7", steps=STEPS, tamper=set(), node="test-node", tools={},
                  pouw=None, delta="1/100", start=0, windows=1, weights=weights, session_jobs=1, wired=True)
    d, _ = RN.loop_window(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    circuit = json.loads((d / "circuit.json").read_text())
    assert circuit["name"] == f"torch-loop/mlp-w8a8-h{H}/wired" and len(circuit["templates"]) == 3
    units = L.loop_units(d)
    assert L.requests_of(d, units) == ([3, 7], STEPS, H)
    outputs = L.registered_outputs(units)
    model = S.L.Model(h=H)
    for i, token in enumerate([3, 7]):
        assert [outputs[f"r{i}", k] for k in range(STEPS)] == S.step_outputs(model, token, STEPS)
    text = [b"".join(outputs[f"r{i}", k] for k in range(STEPS)).hex().encode() for i in range(2)]
    events = [t[j:j + L.TOKEN_BYTES] for t in text for j in range(0, len(t), L.TOKEN_BYTES)]
    assert L.match_outputs([{"stream": 1, "session": "s.0"}, {"stream": 2, "session": "s.1"}], events, FRAMES, outputs,
                           STEPS)["equal"]
    # every activation is a tie and every weight registered: only the requests' step-0 up_proj inputs are free
    assert L.free_inputs(circuit)["bits"] == IF.input_count(circuit)["free"] == 2 * (1024 + 32)
