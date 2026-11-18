"""Refresh (`audit.Params.refresh`), its slot schedule (`rota.Schedule`), held to the Lean of record's form
(`Poub/Session.lean` with refresh, DRAW256's) against the Python alone: this branch's `Poub.Session` has no refresh, so
`ra_cases.REFRESH_SCENARIOS`, the cases `test_refresh_lean.py` sends the Lean, replay through `ra_cases.session` only,
and join `SCENARIOS` (held to the Lean by `test_session.py`) where the two meet. With refresh off, every recorded
session is as it was before refresh, byte for byte."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import ra_cases as M

from verity.protocols.resource_accounting.memory.poub.verifier import audit, coins, ledger

OFF_DIGEST = ("543a23a0082c8c8855250c645f9faf1a3335afa6462ce9459703f993742a63dc"
              "fe04996af6026e75ccae86e8ac270fd3fbeb664b44f704adf0fda9ff3a6a6bd1")
"""SHA-512 over every recorded scenario's case, outcome and, accepted, its record's public form, profile and ledger, as
`off_digest` lists them, before refresh, under the challenger's coin commitment as it now forms it."""


def _ledger(rec: audit.Record, **kw) -> audit.GammaTerms:
    return audit.ledger(rec, capacity_bytes_per_s=1e18, chip_bytes=0, rebuild=audit.Rebuild(rec.codec, 0.0), **kw)


def off_digest() -> str:
    h = hashlib.sha512()
    for name, (world, kw, build, _) in M.SCENARIOS.items():
        r = M.Run(world, **kw)
        build(r)
        out = {"name": name, "case": r.case(), "outcome": r.s.outcome}
        if r.s.outcome == "accepted":
            rec = r.s.record()
            out |= {"public": rec.public(), "profile": rec.profile(), "ledger": _ledger(rec).to_json()}
        h.update(json.dumps(out, sort_keys=True).encode() + b"\n")
    return h.hexdigest()


def run(name: str) -> M.Run:
    world, kw, build, _ = M.REFRESH_SCENARIOS[name]
    r = M.Run(world, **kw)
    build(r)
    return r


@pytest.mark.slow
def test_refresh_off_leaves_every_recorded_session_byte_for_byte() -> None:
    """Every recorded session's events, coins, results, record and ledger, pinned before refresh: what no test of one
    field catches is a byte of the off session moved anywhere. A change that moves the off session on purpose re-pins
    `OFF_DIGEST`."""
    assert off_digest() == OFF_DIGEST


def test_the_recorded_sessions_held_to_the_lean_name_no_refresh() -> None:
    assert all("refresh" not in M.recorded(name)[0]["params"] for name in M.SCENARIOS)
    assert not set(M.SCENARIOS) & set(M.REFRESH_SCENARIOS)


@pytest.mark.parametrize("name", list(M.REFRESH_SCENARIOS))
def test_python_only_a_refresh_scenario_reaches_its_outcome_recorded_and_replayed(name: str) -> None:
    case, got = M.recorded(name)
    out = M.session(copy.deepcopy(case))
    assert got == out["outcome"] == M.REFRESH_SCENARIOS[name][3]


def test_the_refresh_scenarios_reach_every_new_failure_and_an_accepted_session() -> None:
    reached = {s[3] for s in M.REFRESH_SCENARIOS.values()}
    assert set(M.REFRESH_FAILURES) == set(audit.FAILURES[-3:])
    assert reached >= {*M.REFRESH_FAILURES, "lifetime", "append", "malformed", "accepted"}


def test_the_honest_refresh_session_opens_rekeys_of_registered_and_fresh_payloads() -> None:
    """Windows 1 and 2 open the registered re-keys' blocks under the registration's root, window 2 K1's under slot 0's
    fresh append's (store indices 6 and 7)."""
    case, _ = M.recorded("honest-refresh")
    origins = {o.get("origin_position") for e in case["events"] if e["kind"] == "verify" for o in e["openings"]}
    assert {0, 1, 2, 3, 4, 5, 6, 7} <= origins


def test_the_honest_refresh_record_names_refresh_and_each_passs_probation() -> None:
    """DRAW256's: slot 0's fresh append (2 blocks) is born in window 0 and held in probation from window 0's second
    pass through window 1; K1 and K2 inherit its birth, out of probation at windows 2 and 4."""
    case, _ = M.recorded("honest-refresh")
    rec = M.session(copy.deepcopy(case))["record"]
    assert rec["refresh"] is True and rec["appended"] == 18
    assert [w["probation"] for w in rec["windows"]] == [[0, 2], [2, 2], [0, 0], [0, 0], [0, 0]]
    assert case["params"]["fresh_slots"] == 1


def test_each_rejection_is_the_last_event_and_names_what_drew_it() -> None:
    """Every refresh scenario's rejection is its last event, so none is reached by a step before the one it tests."""
    for name in M.REFRESH_SCENARIOS:
        case, outcome = M.recorded(name)
        failed = [k for k, x in enumerate(M.session(copy.deepcopy(case))["results"]) if "failure" in x]
        assert failed == ([] if outcome == "accepted" else [len(case["events"]) - 1]), name


def test_an_empty_slot_takes_no_coin_and_shifts_nothing() -> None:
    """With window 3's registered re-key (slot 6) skipped, window 4's appends keep slots 9 and 10, and the append coins
    are the appends received, in order."""
    r = run("refresh-skipped-rekey")
    slots = [e["slot"] for e in r.events if e["kind"] == "append"]
    assert slots == [0, 1, 3, 4, 9, 10] and r.s.outcome == "accepted"
    assert r.s.coins.appends == len(slots) == len(r.append_coins)


def test_refresh_off_names_neither_refresh_nor_probation_anywhere() -> None:
    r = M.Run(M.World(), windows=2)
    M.honest_appends(r)
    rec = r.s.record()
    assert "refresh" not in rec.public() and all("probation" not in w.to_json() for w in rec.windows)
    assert "probation" not in _ledger(rec).to_json() and _ledger(rec).probation is None


def test_a_refresh_records_ledger_credits_appended_blocks_on_its_schedule_alone_with_no_carrier() -> None:
    rec = run("honest-refresh").s.record()
    schedule = audit.Scheduled(rec.registration, rec.coins, 0)
    plain, credited = _ledger(rec), _ledger(rec, schedule=schedule)
    assert sum(w.post_seed[0] for w in rec.windows) > 0
    assert credited.timing < plain.timing and credited.carrier == 0 == plain.carrier
    assert credited.steering == 0 and plain.steering is None
    assert credited.probation == max(a / (a + n) for w in rec.windows for a, n in zip(w.probation, w.lengths))
    assert credited.to_json()["probation"] == credited.probation > 0

    class Holds:
        holds = True

        def carrier_bytes(self, read_bytes: int) -> float:
            return 0.0

    with pytest.raises(ValueError, match="no stacking verdict"):
        _ledger(rec, schedule=schedule, credit_post_seed=Holds())
    with pytest.raises(ValueError, match="schedule check"):
        _ledger(rec, schedule=audit.Scheduled(rec.registration, bytes(64), 0))


def test_declared_outputs_take_a_refresh_records_schedule_and_refuse_it_for_any_other() -> None:
    rec = run("honest-refresh").s.record()
    node = ledger.Node(1e18, 1e18, 0, {rec.codec.field.name: 0.0}, 1e18, 1e18)
    g = ledger.gamma(rec, node, 1e18, schedule=audit.Scheduled(rec.registration, rec.coins, 0))
    assert g.to_json()["probation"] == g.ledger.probation
    r = M.Run(M.World(), windows=2)
    M.honest_appends(r)
    off = r.s.record()
    with pytest.raises(ValueError, match="only a refresh record"):
        ledger.gamma(off, node, 1e18, schedule=audit.Scheduled(off.registration, off.coins, 0))


def test_a_coin_leaves_only_after_its_own_append_is_committed() -> None:
    """The session asks the challenger for an append's coin only inside `append`, after the append's root (a re-key's,
    its declaration) is received and checked: every append coin opened is opened after its own root, in order, and an
    append refused releases none."""
    r = run("honest-refresh")
    roots = [bytes.fromhex(e["root"]) for e in r.events if e["kind"] == "append"]
    assert r.s.coins.afters(coins.APPEND) == tuple(roots)
    assert [coins.coin(o) for o in r.s.coins.opened(coins.APPEND)] == r.append_coins
    refused = run("refresh-coin-before-commit")
    sent = [e for e in refused.events if e["kind"] == "append"]
    assert len(sent) == 1 and refused.s.coins.appends == 0 and refused.s.outcome == "append"


def _session(blocks: int) -> audit.AuditRun:
    reg = audit.register(M.SOURCE, M.NODE, blocks, bytes(64), M.P512, nonce=bytes(32))
    params = audit.Params(blocks, 100, 2, window_ns=10, codec=M.P512, refresh=True, lifetime=2, fresh_slots=1)
    s = audit.AuditRun(params, reg, None, coins.local(reg.digest, 0, 2, M.Entropy(b"coins")), not_before_ns=0,
                       clock=M.Clock())
    s.start()
    return s


@pytest.mark.parametrize(("count", "failure"), [(512, None), (513, "origin")])
def test_a_fresh_slot_holds_at_most_rekey_max_blocks(count: int, failure: str | None) -> None:
    assert audit.REKEY_MAX == 512
    s = _session(6)
    s.append(hashlib.sha512(b"fresh").digest(), count, slot=0)
    assert (s.failed and s.failed.failure) == failure


def test_a_registered_rekey_is_its_whole_class_run_past_rekey_max() -> None:
    """N = 1200 at L = 2: window 1's registered slot (1) re-keys class 1, blocks 600 … 1199, in one append."""
    s = _session(1200)
    s.clock.now = 10
    s.boundary()
    origin = ("registered", 600)
    assert isinstance(s.append(audit.rekey_root(s.registration.digest, s.extent, 600, origin), 600, origin, 1), tuple)


def test_refresh_is_a_bool_and_fresh_slots_are_refreshs() -> None:
    with pytest.raises(ValueError, match="refresh is a bool"):
        audit.Params(6, 100, 2, refresh=1)
    with pytest.raises(ValueError, match="fresh_slots"):
        audit.Params(6, 100, 2, fresh_slots=1)
    assert audit.Params(6, 100, 2).schedule is None
    assert audit.Params(6, 100, 5, refresh=True, lifetime=2, fresh_slots=1).schedule.total == 13
