"""The communication leg (`legs/certifier.py`) on what it reads and what it writes, without a certifier session:

* a served plan of the period's prompts, a link-window per group, puts period p's requests in link-window p, each with
  its own prompt, and has no period past the session's; the stub serves those bodies and refuses any other;
* N2's outcome of record is one-stage's answer on the link-window's one unit, and its refusal a reject;
* the integrity leg's accepted audit is re-decided at the leg's δ (one-stage's bound on wrong units), and its refusal
  kept as it is;
* the certified text matches the registered outputs by stream, else by content, and a changed byte is no match;
* `case_of`'s case of signed link-windows and the period's audit is the period check's (its Python mirror, which
  `network-period` is held to): accepted with K and eps at each δ, refused when N2 refused a link-window or the case
  names another first window;
* a loop period's value files give its requests' prompt tokens, steps and width, and its registered outputs are the
  stub's for those tokens (`torch_loop_stub.step_outputs`), so an honest certifier session's egress carries them (slow).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.certifier.certificate import Certificate, Direction, clock_digest
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registration as R
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


def plan(periods: int) -> tuple[dict, dict]:
    probed = {"url": "stub", "body": cs.request_body("stub", STEPS, PROMPTS[0]),
              "responses": [{"events": FRAMES, "first_s": 0.0, "last_s": 0.2}]}
    return cs.served_plan(probed, len(PROMPTS) * periods, in_flight=len(PROMPTS), frames=FRAMES, prompts=PROMPTS,
                          window_per_group=True)


def link_window(w: int, program: str = "cc" * 64) -> dict:
    return {"format": R.FORMAT, "protocol": R.PROTOCOL, "program": program, "partition": f"n2/{w}", "population": 1,
            "law": {"law": "subset", "k": 1}, "window": {"kind": "served", "window": w}}


def test_a_period_is_its_own_link_window_with_its_own_prompts():
    report, p = plan(3)
    assert [L.period_windows(p, i, len(PROMPTS)) for i in range(3)] == [[0], [1], [2]]
    assert [q["body"]["prompt"] for q in p["requests"]] == [[3], [7]] * 3
    with pytest.raises(RuntimeError, match="carries 3 periods, not period 3"):
        L.period_windows(p, 3, len(PROMPTS))
    assert cs.ar.Setup(report).sync_set.deltas


def test_the_stub_serves_each_fixed_plan_body_of_the_loops_steps():
    _, p = plan(1)
    model = f"torch-loop/mlp-w8a8-h{H}"
    assert [S.refusal(q["body"] | {"model": model}, STEPS) for q in p["requests"]] == [None, None]
    assert S.refusal(cs.request_body(model, STEPS + 1, [3]), STEPS)
    assert S.refusal(cs.request_body(model, STEPS, [3], temperature=1.0), STEPS)
    assert S.refusal(cs.request_body(model, STEPS, [3]) | {"min_tokens": 1}, STEPS)


def test_n2s_outcome_is_one_stages_answer_on_the_link_windows_one_unit():
    reg = link_window(4)
    a = L.answer_of_outcome(reg, {"units": [0]}, True, None)
    assert check_answer(a) == a and a["verdict"] == "accept" and a["commitment"] == R.digest(reg)
    p = CC.integrity(a["profile"])
    assert p["delta"] == L.N2_DELTA and p["drawn"] == [("unit", 1)]
    assert [(lv["name"], lv["size"], lv["draw"]) for lv in p["levels"]] == [("unit", 1, {"subset": 1})]
    assert a["public"]["proved"] == a["public"]["drawn"] == [0]
    r = L.answer_of_outcome(reg, {}, False, "N2's verifier of record: {}")
    assert check_answer(r) == r and r["verdict"] == "reject" and r["profile"] is None and r["public"] is None


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


def session_run(windows: list[int], refused: frozenset[int] = frozenset()) -> dict:
    """A run as `run_windows` reads it, its link-windows signed by a small deployed-size LMS key."""
    report, _ = plan(len(windows))
    setup = cs.ar.Setup(report)
    params = lms.Params(64, 4, 3)
    key = lms.keygen(params, hashlib.sha512(b"certifier leg").digest())
    session, link = hashlib.sha512(b"session").digest()[:lms.SESSION_BYTES], "n2"
    sync = ClockSync(setup.sync_set.deltas[0], setup.sync_set.rhos[0])
    q = iter(range(2 ** params.H))

    def sign(d: Direction, w: int, digest: bytes) -> dict:
        c = Certificate(key.I, link, d, w, digest, b"")
        c = dataclasses.replace(c, signature=lms.sign_at(key, next(q), c.message(session)).encode())
        return L._cert({"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(),
                        "link": c.link, "signature": c.signature.hex(), "window": c.window})

    xs = {}
    for w in windows:
        reg = link_window(w)
        xs[w] = {"registration": reg,
                 "answer": L.answer_of_outcome(reg, {"units": [0]}, w not in refused, "N2's verifier of record: {}"),
                 "certificates": [sign(Direction.EGRESS, w, hashlib.sha512(b"e%d" % w).digest()),
                                  sign(Direction.INGRESS, w, hashlib.sha512(b"i%d" % w).digest())],
                 "clock": {"certificate": sign(Direction.EGRESS, w, clock_digest(sync)),
                           "sync": [sync.delta, sync.rho.numerator, sync.rho.denominator]}}
    dep = {"certifier_key": {"params": [params.N, params.W, params.H], "public": lms.public_key(key).hex()},
           "links": {"n2": link}, "session": session.hex()}
    return {"deployment": dep, "calibration": report, "windows": xs}


@pytest.mark.parametrize("delta", L.DELTAS)
def test_the_case_is_the_period_checks(delta):
    rw = session_run([1, 2])
    audits = [(LOOP, L.audit_answer(LOOP, {"verdict": "accept"}, ACCEPTED, delta))]
    case = L.case_of(rw, [1, 2], audits, STEPS, 1)
    assert sorted(case["params"]) == PERIOD.PARAMS_KEYS
    a = PERIOD.reference(case)
    assert a["verdict"] == "accept", a["reason"]
    pub = a["public"]
    bound = audits[0][1]["public"]["bound"]
    assert pub["eps"] == str(delta) and pub["windows"] == 2 and pub["cells"] == 0
    assert pub["audits"][0]["wrong_units"] == bound and pub["audits"][0]["jitter"] == 0
    assert pub["K"] == pub["schedules"] + pub["audits"][0]["faulty_sets"] + 16 * bound
    assert PERIOD.reference(L.case_of(rw, [1, 2], audits, STEPS, 0))["verdict"] == "reject"
    refused = session_run([1, 2], frozenset({2}))
    assert PERIOD.reference(L.case_of(refused, [1, 2], audits, STEPS, 1))["verdict"] == "reject"


@pytest.mark.slow
def test_a_loop_period_registers_what_the_stub_serves(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3,7", steps=STEPS, tamper=set(), node="test-node", tools={})
    d, _ = RN.loop_period(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    units = L.loop_units(d)
    assert L.requests_of(d, units) == ([3, 7], STEPS, H)
    outputs = L.registered_outputs(units)
    model = S.L.Model(h=H)
    for i, token in enumerate([3, 7]):
        assert [outputs[f"r{i}", k] for k in range(STEPS)] == S.step_outputs(model, token, STEPS)
