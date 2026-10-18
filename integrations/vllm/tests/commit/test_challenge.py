"""The one challenge module: legacy derivations stay byte-identical to the outputs recorded before they moved onto it, and
the non-legacy forms draw through `verity.ml.randomness.derive` from an auditor key (decision 42)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.ml.randomness import derive
from verity_vllm.check.replay import sample as SR
from verity_vllm.commit import binding as B
from verity_vllm.commit import challenge as CH

from tests.commit import test_binding_negatives as T

V = json.loads(Path(__file__).with_name("challenge_legacy_vectors.json").read_text())
ROOT = bytes.fromhex(V["run_root"])
AUDITOR_KEY = b"auditor key 1041"


def _vus() -> list[dict]:
    return [{"stratum": f"s{i % 5}|a|b|{'prefill' if i % 2 else 'decode'}", "op_path": f"m.layers.{i % 3}.mlp", "step": i % 4, "i": i}
            for i in range(83)]


def _ident(x):
    return tuple(tuple(y) if isinstance(y, list) else y for y in x)


def test_legacy_is_the_default_until_the_re_baseline_epoch():
    assert CH.LEGACY is True


@pytest.mark.parametrize("case", V["positions"], ids=lambda c: f"k{c['k']}")
def test_legacy_positions_are_byte_identical(case):
    got = CH.challenge_positions(ROOT, case["leaves_per_step"], case["k"], salt=bytes.fromhex(case["salt"]))
    assert [list(p) for p in got] == case["positions"]


@pytest.mark.parametrize("case", V["seeds"], ids=lambda c: c["challenge"])
def test_legacy_replay_seed_is_byte_identical(case):
    assert SR.challenge_seed(case["challenge"], ROOT) == (case["seed"], case["source"])


@pytest.mark.parametrize("case", V["samples"], ids=lambda c: f"per{c['per_stratum']}")
def test_legacy_stratified_sample_is_byte_identical(case):
    picked, strata, tgt = SR.sample(_vus(), case["per_stratum"], case["seed"], targets=case["targets"])
    assert [v["i"] for v in picked] == case["picked"] and strata == case["strata"] and tgt == case["targets_out"]


def test_legacy_identities_are_byte_identical():
    c, run, bmap = T.commit(T.produce())
    pop = list(B.expected_population(T.structure(), T.workload()))
    for case in V["identities"]:
        assert (run.run_root.hex(), bmap["digest"]) == (case["run_root"], case["map_digest"]), "the fixture commit changed, not the challenge"
        picks = B.challenge_identities(pop, bmap, run.run_root, case["k"], salt=bytes.fromhex(case["salt"]))
        assert [[ident, leaf] for ident, _e, leaf in picks] == [[_ident(i), leaf] for i, leaf in case["picks"]]


def test_non_legacy_needs_a_source_and_legacy_refuses_one():
    with pytest.raises(ValueError, match="needs a source"):
        CH.challenge_positions(ROOT, [10], 3, legacy=False)
    with pytest.raises(ValueError, match="legacy"):
        CH.challenge_positions(ROOT, [10], 3, source=AUDITOR_KEY)
    with pytest.raises(ValueError, match="needs a source"):
        CH.identity_picker(ROOT, "00" * 32, legacy=False)
    with pytest.raises(ValueError, match="needs a source"):
        CH.replay_key(b"", ROOT)


def test_non_legacy_positions_follow_the_source_root_and_layout():
    lps = [10, 0, 20, 5]
    a = CH.challenge_positions(ROOT, lps, 64, legacy=False, source=AUDITOR_KEY)
    assert len(a) == 64 and all(0 <= off < lps[s] for s, off in a)
    assert a == CH.challenge_positions(ROOT, lps, 64, legacy=False, source=AUDITOR_KEY)
    assert a != CH.challenge_positions(ROOT, lps, 64)
    assert a != CH.challenge_positions(ROOT, lps, 64, legacy=False, source=b"another key")
    assert a != CH.challenge_positions(hashlib.sha256(b"x").digest(), lps, 64, legacy=False, source=AUDITOR_KEY)
    assert a != CH.challenge_positions(ROOT, lps, 64, b"salt", legacy=False, source=AUDITOR_KEY)
    assert a != CH.challenge_positions(ROOT, [10, 0, 20, 6], 64, legacy=False, source=AUDITOR_KEY)


def test_non_legacy_identities_draw_through_the_picker():
    c, run, bmap = T.commit(T.produce())
    pop = list(B.expected_population(T.structure(), T.workload()))
    fresh = B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", legacy=False, source=AUDITOR_KEY)
    assert fresh and all(e is not None for _, e, _ in fresh)
    assert fresh == B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", legacy=False, source=AUDITOR_KEY)
    assert fresh != B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t")
    pick = CH.identity_picker(ROOT, "ab" * 32, legacy=False, source=AUDITOR_KEY)
    assert all(0 <= pick(i, 7) < 7 for i in range(100))


def test_non_legacy_replay_sample_is_the_former_proof_unit_draw():
    key = CH.replay_key(AUDITOR_KEY, ROOT)
    assert key == derive(AUDITOR_KEY, "verity/sampled-proofs/vu-sample/v1", {"run_root": ROOT})
    picked, strata, _ = SR.sample(_vus(), 3, 0, key=key)
    assert all(s["sampled"] == 3 for s in strata.values()) and len(picked) == 3 * len(strata)
    assert [v["stratum"] for v in picked[:len(strata)]] == sorted(strata)  # round-robin: every stratum's first pick first
    assert picked == SR.sample(_vus(), 3, 0, key=key)[0]
    assert picked != SR.sample(_vus(), 3, 0, key=CH.replay_key(b"another key", ROOT))[0]


def test_stratum_picks_takes_a_seed_or_a_key():
    with pytest.raises(ValueError):
        CH.stratum_picks([[1, 2]], 1, None)
    with pytest.raises(ValueError):
        CH.stratum_picks([[1, 2]], 1, 1, key=CH.replay_key(AUDITOR_KEY, ROOT))
    assert CH.stratum_picks([[1, 2], [3]], 5, None, key=CH.replay_key(AUDITOR_KEY, ROOT)) == [[1, 2], [3]]
    with pytest.raises(ValueError):
        CH.stratum_picks([[1, 2]], -1, None, key=CH.replay_key(AUDITOR_KEY, ROOT))


# --- a run of record (`commit.record`): live draws only --------------------------------------------------------------------

class _Run:
    def __init__(self, root: bytes, leaves: list[int]):
        self.run_root, self.extra = root, {}
        self.steps = [type("S", (), {"n_leaves": n})() for n in leaves]


RECORD_ROOT = hashlib.sha512(b"record").digest()


def test_a_run_of_record_draws_only_from_its_live_source():
    run = _Run(RECORD_ROOT, [5, 0, 9, 3])
    with pytest.raises(ValueError, match="no live key"):
        CH.positions_of(run, 8)
    key = CH.draw_live(run, AUDITOR_KEY.hex())
    assert key == AUDITOR_KEY and run.extra["challenge"] == {"key": AUDITOR_KEY.hex(), "origin": "unattested", "after_run_root": RECORD_ROOT.hex()}
    want = derive(AUDITOR_KEY, CH.POSITIONS_DOMAIN, {"run_root": RECORD_ROOT, "salt": b"p", "layout": "5,0,9,3"})
    cum = [5, 5, 14, 17]
    got = CH.positions_of(run, 40, b"p")
    assert [(cum[s - 1] if s else 0) + i for s, i in got] == [want.uniform(17, j) for j in range(40)]
    assert CH.identities_kw(run) == {"legacy": False, "source": AUDITOR_KEY}
    assert CH.kernel_seed_of(run) == derive(AUDITOR_KEY, CH.KERNEL_CHECK_DOMAIN, {"run_root": RECORD_ROOT}).uniform(1 << 32)


def test_own_coins_are_fresh_and_bound_to_the_root_they_follow():
    a, b = _Run(RECORD_ROOT, [4]), _Run(RECORD_ROOT, [4])
    assert CH.draw_live(a) != CH.draw_live(b) and a.extra["challenge"]["origin"] == "own"
    a.run_root = hashlib.sha512(b"another root").digest()
    with pytest.raises(ValueError, match="no live key"):
        CH.positions_of(a, 2)
    with pytest.raises(ValueError, match="at least 16 bytes"):
        CH.key_challenger("00" * 8)


def test_the_openings_take_their_challenge_from_a_challenger_seam_and_only_the_service_challengers_is_of_record(monkeypatch):
    """`draw_live` asks a `Challenger` (root in, `LiveChallenge` out); positions are a function of the root and that challenge alone.
    A key handed in or the run's own bytes are never attested; the service challenger's is, once `ATTEST` accepts its provenance."""
    asked = []

    def service(root: bytes) -> CH.LiveChallenge:
        asked.append(root)
        return CH.LiveChallenge(AUDITOR_KEY, root, CH.ATTESTED, {"opening": "stand-in"})

    run = _Run(RECORD_ROOT, [6, 6])
    assert CH.draw_live(run, service) == AUDITOR_KEY and asked == [RECORD_ROOT]
    assert run.extra["challenge"]["provenance"] == {"opening": "stand-in"}
    assert CH.positions_of(run, 5) == CH.record_positions(RECORD_ROOT, [6, 6], 5, source=AUDITOR_KEY)
    assert "is plugged in yet" in CH.attested_reason(run.extra["challenge"])
    monkeypatch.setattr(CH, "ATTEST", lambda ch: None if ch.get("provenance") else "no opening")
    assert CH.attested_reason(run.extra["challenge"]) is None
    for given in (AUDITOR_KEY.hex(), None):
        other = _Run(RECORD_ROOT, [6, 6])
        CH.draw_live(other, given)
        assert "not the service challenger's" in CH.attested_reason(other.extra["challenge"])
    with pytest.raises(ValueError, match="not a live challenge after its root"):
        CH.draw_live(_Run(RECORD_ROOT, [1]), lambda root: CH.LiveChallenge(AUDITOR_KEY, bytes(64), CH.ATTESTED))


def test_reference_rows_from_a_live_key_are_a_subset_of_the_pending_rows():
    got = CH.reference_rows(AUDITOR_KEY, 120, 30, 8)
    assert got == sorted(derive(AUDITOR_KEY, CH.REFERENCE_ROWS_DOMAIN, {"rows_total": 120, "pending": 30}).subset(8, 30, "rows"))
    assert len(set(got)) == 8 and all(0 <= r < 30 for r in got)
    assert CH.reference_rows(AUDITOR_KEY, 120, 3, 8) == [0, 1, 2]
    assert CH.reference_rows(b"another auditor key", 120, 30, 8) != got


def test_a_recorded_vllm_v1_run_reads_the_legacy_draws():
    run = _Run(ROOT, [7, 2])
    assert CH.draw_live(run) is None and "challenge" not in run.extra
    assert CH.positions_of(run, 12, b"s") == CH.challenge_positions(ROOT, [7, 2], 12, b"s")
    assert CH.identities_kw(run) == {} and CH.kernel_seed_of(run) == CH.kernel_check_seed(ROOT)
    with pytest.raises(ValueError, match="64 bytes"):
        CH.record_positions(ROOT, [7, 2], 3, source=AUDITOR_KEY)
