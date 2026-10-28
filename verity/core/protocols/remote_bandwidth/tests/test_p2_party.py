"""The P2 session's party (`p2_party.py`): its registration, plan and openings round-trip their JSON, and a registration
whose digest isn't its fields' or a plan that isn't one of its units is refused; a session's parameters are the
auditor's, and a request carrying other values is refused by name; its answer in the core interface's form is one
`core_check` reads, and the prover reads a challenge in either form. Its sessions, decided by the Lean verifier with
these books beside it, are `lean/tests/test_p2_party.py`'s, which run where Lake is."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from test_registrations import hosts
from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.pouw import core_check
from verity.core.protocols.remote_bandwidth import coins, p2, p2_party, receipts

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = b"the verifier's secret"
NODE = b"p2-node-under-test"


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


@pytest.fixture(scope="module")
def store() -> p2.Store:
    doc = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    blocks = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = doc["params"]
    params = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    commitment, salt, n = bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]), params.blocks_per_segment
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2.Store(p2.register(SOURCE, NODE, commitment, vk, site=hosts(NODE.decode())), blocks)


def test_the_registration_and_openings_round_trip_their_json(store) -> None:
    reg = store.registration
    assert p2_party.registration_of(json.loads(json.dumps(p2_party.registration_json(reg)))) == reg
    bad = p2_party.registration_json(reg) | {"rows": 64}
    with pytest.raises(ValueError, match="digest"):
        p2_party.registration_of(bad)
    prover = p2.Prover(store, entropy=Entropy(b"s"))
    ps = prover.pass_under(0, 0, b"\x01" * 64)
    o = prover.open(ps, 1)
    assert p2_party.opening_of(json.loads(json.dumps(p2_party.opening_json(o)))) == o
    assert p2_party.opening_of({"pass": 0}) is None


def test_the_registrations_plan_round_trips_its_json_and_a_bad_one_is_refused(store) -> None:
    reg = store.registration
    plan = receipts.Plan(((1, 1),), reg.blocks)
    doc = json.loads(json.dumps(p2_party.registration_json(reg, plan)))
    assert p2_party.registration_of(doc) == reg and p2_party.plan_of(doc, reg) == plan
    assert p2_party.plan_of(p2_party.registration_json(reg), reg) == receipts.Plan.whole(reg.blocks)
    for bad in ([[0, reg.blocks + 1]], [[0, 1], [1, 1]], [], "0,1", [[0]]):
        with pytest.raises(ValueError):
            p2_party.plan_of(doc | {"plan": bad}, reg)


def _held(params) -> tuple[int, ...]:
    return params.deadline_ns, params.min_passes, params.samples, params.window_ns, params.windows


def test_the_honest_request_passes_and_the_params_are_the_auditors(store) -> None:
    """A request carrying the auditor's values as they are passes, and `audit.Params` gets the auditor's: `SESSION`,
    the served bandwidth piece's, by default, or the values a toy party is held to."""
    reg = store.registration
    assert p2_party.SESSION == {"deadline_ns": 10**9, "passes": 1, "samples": 1024, "window_ns": 10**9, "windows": 2}
    params = p2_party.params_of(reg, p2_party.request_of(reg.node, **p2_party.SESSION))
    assert _held(params) == (10**9, 1, 1024, 10**9, 2)
    assert (params.blocks, params.codec, params.rows) == (reg.blocks, reg.codec, reg.rows)
    toy = {"deadline_ns": 400, "passes": 2, "samples": 4, "window_ns": 800, "windows": 3}
    assert _held(p2_party.params_of(reg, p2_party.request_of(reg.node, **toy), toy)) == (400, 2, 4, 800, 3)


@pytest.mark.parametrize("key, value", [("deadline_ns", 2 * 10**9), ("samples", 1023), ("window_ns", 10**9 // 2),
                                        ("windows", 1), ("passes", 2)])
def test_a_request_off_the_auditors_session_is_refused_by_name(store, key, value) -> None:
    """A longer allowance, fewer openings, a shorter window, fewer windows or another pass count than the auditor's is
    refused, naming the parameter and both values."""
    reg = store.registration
    request = p2_party.request_of(reg.node, **p2_party.SESSION | {key: value})
    with pytest.raises(ValueError) as e:
        p2_party.params_of(reg, request)
    assert str(e.value) == f"the request's {key} is not the auditor's ({value}, not {p2_party.SESSION[key]})"
    toy = p2_party.SESSION | {key: value}
    assert _held(p2_party.params_of(reg, request, toy)) == tuple(toy[k] for k in sorted(toy))


def test_a_request_off_the_auditors_session_in_two_values_names_both(store) -> None:
    reg = store.registration
    request = p2_party.request_of(reg.node, **p2_party.SESSION | {"deadline_ns": 5 * 10**9, "samples": 16})
    with pytest.raises(ValueError, match=r"^the request's deadline_ns is not the auditor's \(5000000000, not "
                                         r"1000000000\); the request's samples is not the auditor's \(16, not 1024\)$"):
        p2_party.params_of(reg, request)


def test_the_party_refuses_session_parameters_that_are_not_positive(tmp_path, capsys) -> None:
    path = tmp_path / "registration.json"
    path.write_text("{}")
    with pytest.raises(SystemExit):
        p2_party.main(["--registration", str(path), "--stdio", "--deadline-ns", "0"])
    assert "the session's parameters are positive integers" in capsys.readouterr().err


ACCEPTED = {"protocol": p2_party.PROTOCOL, "registration": "ab" * 64, "unit_bytes": p2.UNIT_BYTES, "units": 40,
            "plan_units": 32, "windows": [{"passes": 1, "draws": 8, "verdict": None}] * 2, "reason": None,
            "verdict": "accept", "receipts": 64, "wrong_at_most": 4, "delta": str(Fraction(1, 2**40)), "bound": {},
            "draws": [8, 8], "q_bytes": 60 * p2.UNIT_BYTES, "span_ns": 2 * 10**9, "credited_bytes_per_s": 246720,
            "host": "auditor"}


def test_an_accepted_answer_is_a_core_accept_with_its_eps_and_gamma() -> None:
    a = p2_party.core_answer(ACCEPTED, r_bytes_per_s=10**6)
    assert core_check.answer_of(a)["verdict"] == "accept"
    assert a["commitment"] == "ab" * 64 and a["reason"] is None
    assert Fraction(a["public"]["eps"]) == Fraction(1, 2**40) + Fraction(16, 2**64)
    assert a["public"]["q_bytes"] == 60 * p2.UNIT_BYTES and a["public"]["span_ns"] == 2 * 10**9
    assert Fraction(a["public"]["gamma"]) == 1 - Fraction(60 * p2.UNIT_BYTES, 2 * 10**6)
    assert a["profile"]["wrong"] == 4 and a["profile"]["plan_units"] == 32 and a["profile"]["delta"] == "1/1099511627776"
    without = p2_party.core_answer({k: v for k, v in ACCEPTED.items() if k != "plan_units"})
    assert "gamma" not in without["public"] and without["profile"]["plan_units"] == 40


@pytest.mark.parametrize("answer, verdict, says", [
    ({"verdict": "reject", "failure": "receipt", "window": 0, "at": [1, 3]}, "reject", "receipt at window 0"),
    ({"verdict": "undecided"}, "undecided", "not yet verified"),
    ({"verdict": "refused", "reason": "the request's node is not the registration's"}, "reject", "refused"),
    ({"verdict": "prover-error", "reason": "OSError: gone"}, "reject", "prover-error"),
])
def test_any_other_answer_is_a_core_refusal_with_its_reason(answer: dict, verdict: str, says: str) -> None:
    a = p2_party.core_answer({"protocol": p2_party.PROTOCOL, "registration": "cd" * 64} | answer)
    assert core_check.answer_of(a)["verdict"] == verdict
    assert says in a["reason"] and a["public"] is None and a["profile"] is None
    with pytest.raises(ValueError, match="not a"):
        p2_party.core_answer(answer)


def test_the_prover_reads_a_challenge_released_or_as_its_whole_opening() -> None:
    r = coins.Released(0, coins.CHALLENGE, 0, b"\x01" * 64, b"\x02" * 32)
    assert p2_party.released_of(r.to_json()) == r
    assert p2_party.released_of(r.to_json() | {"path": ["03" * 64], "salt": "04" * 32}) == r
    with pytest.raises(ValueError):
        p2_party.released_of(r.to_json() | {"path": []})
