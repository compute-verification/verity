"""`audit.ledger`'s γ terms (`ledger.gamma`'s `Gamma.ledger`) held to their Lean (`Poub.Gamma`, served by
`poub-difftest`'s `gamma`) as `Verifier.ledgers` holds them (`verifier.gamma_held`), refusals compared word for word:
the records of the recorded sessions (`ra_cases.SCENARIOS`) the verifier accepts, and generated records, paced and
not, g = 1 and g > 1 windows a coin, both codecs, passes partly after the codec seed, at capacities each check passes
and ones a window's or the session's refuses. `named_bits` holds `Gamma.namedBits` to `audit.named_bits` exactly, over
every pass of up to 4096 receipts and spread counts up to 2^40."""

from __future__ import annotations

import json
import random
import shutil
from fractions import Fraction

import pytest
import ra_cases as M
import verity_difftest as D
from hypothesis import strategies as st

from verity.protocols.resource_accounting.memory.poub.verifier import audit, line, receipts
from verity.protocols.resource_accounting.memory.poub.verifier.session import lean_verifier as lv
from verity.protocols.resource_accounting.memory.poub.verifier.session import verifier as V

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

server = D.server_fixture(D.Port(M.EXE, "gamma", lambda v: v, st.nothing(), package=M.PACKAGE, of_record="lean",
                                 rejects=M.REFUSALS))

DELTAS = (2.0 ** -40, 1e-9, 1e-3, 0.5)
SEEDS = range(160)


def reads(record: dict) -> tuple[int, list[tuple[int, int]]]:
    """The receipts a record certifies and each window's (receipts, span in ns)."""
    ws = [(sum(w["lengths"]), w["deadline_ns"] - w["released_ns"]) for w in record["windows"]]
    return sum(u for u, _ in ws), ws


def least_capacity(record: dict) -> Fraction:
    """The least R (bytes a second) every capacity check of `audit.ledger` passes at."""
    total, ws = reads(record)
    whole = record["windows"][-1]["deadline_ns"] - record["windows"][0]["released_ns"]
    return max(Fraction(u * audit.BLOCK_BYTES * 10 ** 9, s) for u, s in [(total, whole), *ws])


def held(server: D.Server, registration: dict, record: dict, samples: int, delta: float, capacity: float, chip: int,
         squarings: float) -> str:
    """The Lean's γ terms of `record` held to `audit.ledger`'s: "refused" where both refuse alike, else "held"."""
    reg = M.registration_of(registration)
    rec = lv.record_of(record, reg, bytes(64), delta, samples)
    try:
        python = audit.ledger(rec, capacity_bytes_per_s=capacity, chip_bytes=chip,
                              rebuild=audit.Rebuild(rec.codec, squarings)).to_json()
    except ValueError as e:
        python = {"refused": str(e)}
    kind, text = server.call("gamma", {
        "registration": registration, "record": record, "delta": line.text(Fraction(delta)),
        "block_bytes": audit.BLOCK_BYTES, "chip_bytes": chip, "capacity_bytes_per_s": line.text(Fraction(capacity)),
        "squarings_per_s": line.text(Fraction(squarings))})
    assert kind == "output", text
    lean = json.loads(text)
    if "refused" in python:
        assert lean == python
        return "refused"
    V.gamma_held(lean, python)
    return "held"


def generated(seed: int) -> tuple[dict, dict, int, float, float, int, float]:
    """Record `seed`'s inputs to `held`: a registration of a random codec and size, windows of random passes (each
    reading at most every chunk, some after the codec seed), sampled, paced or not, g windows a coin; a capacity at or
    above the least every check passes, or for one seed in four below a window's or the session's."""
    r = random.Random(seed)
    blocks = r.choice([1, 7, 64, 1000])
    reg = audit.Registration(M.NODE, blocks, bytes([seed % 256]) * 64, bytes(32), r.choice([M.P512, M.P512_K3]),
                             receipts.DEFAULT, False, receipts.LAYOUT, receipts.SKETCH_ROWS)
    n, samples = r.randint(1, 6), r.choice([1, 3, 30, 10 ** 6])
    g = r.choice([1, 1, 2, 3]) if n > 1 else 1
    ws, t, step = [], r.randint(0, 10 ** 12), 0
    for _ in range(n):
        k = r.randint(1, 4)
        lengths = [r.randint(1, blocks) for _ in range(k)]
        span = r.choice([1, 1000, 10 ** 6, 10 ** 9, 37 * 10 ** 9 + 11])
        ws.append({"lengths": lengths, "steps": list(range(step, step + k)),
                   "post_seed": [r.choice([0, 0, r.randint(0, x)]) for x in lengths],
                   "heads": [None, "ab" * 64][:k] + [None] * (k - 2), "stale": list(lengths),
                   "arrived_ns": [t + span] * k, "released_ns": t, "deadline_ns": t + span, "opened_ns": t + span + 1,
                   "drawn": min(samples, sum(lengths))})
        t, step = t + span + r.randint(2, 10 ** 6), step + k + r.randint(0, 2)
    record = {"windows": ws, "blocks": blocks, "appended": 0, "released": 0, "lifetime": 2,
              "pace": r.choice([None, {"k": 2, "window_ns": 1}]), "appends": None, "registration": reg.digest.hex()}
    if g > 1:
        record |= {"groups_per_coin": g, "pace": {"k": 2, "window_ns": 1, "in_flight": 2}}
        for a in range(0, n, g):
            left = min(samples, sum(sum(w["lengths"]) for w in ws[a:a + g]))
            for w in ws[a:a + g]:
                w["drawn"] = min(left, sum(w["lengths"]))
                left -= w["drawn"]
    least = least_capacity(record)
    capacity = float(least * Fraction(r.choice([1, 2, 1000]))) * (1 + 2 ** -50)
    if seed % 4 == 3:
        capacity = float(least) * r.choice([0.5, 0.999])
    return (M.registration_json(reg), record, samples, r.choice(DELTAS), capacity,
            r.choice([0, 1 << 20, 96 << 20]), r.choice([0.0, 1e12, 3.7e15]))


def test_ledgers_gamma_terms_are_held_to_the_leans_on_generated_records(server) -> None:
    got = [held(server, *generated(seed)) for seed in SEEDS]
    print(f"generated records: {got.count('held')} held, {got.count('refused')} refused alike")
    assert got.count("held") >= len(SEEDS) // 2 and got.count("refused") >= len(SEEDS) // 8, got


ACCEPTED = [n for n, s in M.SCENARIOS.items() if s[3] == "accepted"]


@pytest.mark.parametrize("name", ACCEPTED)
def test_ledgers_gamma_terms_are_held_to_the_leans_on_a_recorded_sessions_record(server, name) -> None:
    case, _ = M.recorded(name)
    out = M.session(case)
    assert out["outcome"] == "accepted"
    record = out["record"]
    capacity = float(least_capacity(record)) * 4
    for delta in DELTAS:
        got = held(server, case["registration"], record, case["params"]["samples"], delta, capacity, 96 << 20, 1e12)
        assert got == "held"


def test_a_capacity_too_small_for_a_recorded_session_is_refused_alike(server) -> None:
    case, _ = M.recorded(ACCEPTED[0])
    record = M.session(case)["record"]
    assert held(server, case["registration"], record, case["params"]["samples"], 1e-9,
                float(least_capacity(record)) / 2, 0, 0.0) == "refused"


def test_a_ledger_the_lean_disagrees_with_is_refused_naming_both(server) -> None:
    reg, record, samples, delta, capacity, chip, squarings = generated(0)
    rec = lv.record_of(record, M.registration_of(reg), bytes(64), delta, samples)
    python = audit.ledger(rec, capacity_bytes_per_s=capacity, chip_bytes=chip,
                          rebuild=audit.Rebuild(rec.codec, squarings)).to_json()
    off = json.loads(json.dumps(python))
    off["windows"][0]["terms"]["chip"] += 2 ** -30
    kind, text = server.call("gamma", {
        "registration": reg, "record": record, "delta": line.text(Fraction(delta)), "block_bytes": audit.BLOCK_BYTES,
        "chip_bytes": chip, "capacity_bytes_per_s": line.text(Fraction(capacity)),
        "squarings_per_s": line.text(Fraction(squarings))})
    lean = json.loads(text)
    assert V.gamma_held(lean, python) == lean
    with pytest.raises(ValueError, match=r"disagree at \.windows\[0\]\.terms\.chip: the Lean's \""):
        V.gamma_held(lean, off)
    with pytest.raises(ValueError, match=r"not of `GammaTerms.to_json`'s shape, so they hold nothing: ValueError"):
        V.gamma_held({**lean, "windows": []}, python)


NAMED = [*range(4097), *(1 << e for e in range(13, 41)), *random.Random(0).sample(range(4097, 1 << 40), 64)]


def test_named_bits_is_audits_exactly(server) -> None:
    for k in NAMED:
        kind, text = server.call("named_bits", {"k": k})
        assert (kind, json.loads(text)) == ("output", audit.named_bits(k)), k
