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
