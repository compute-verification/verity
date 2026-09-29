"""The one challenge module: legacy derivations stay byte-identical to the outputs recorded before they moved onto it, and
the non-legacy forms are the verifier's (deployment decision A24): keyed by its secret, drawn after a receipt, and a run's
call indices registered once across its windows (X-SPC-18)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.randomness import derive
from verity_vllm.check.replay import sample as SR
from verity_vllm.commit import binding as B
from verity_vllm.commit import challenge as CH

from tests.commit import test_binding_negatives as T

V = json.loads(Path(__file__).with_name("challenge_legacy_vectors.json").read_text())
ROOT = bytes.fromhex(V["run_root"])
BEACON = b"the verifier's secret 1041"
RECEIPT = hashlib.sha256(b"receipt of window 0").digest()


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


def test_non_legacy_needs_the_verifiers_secret_and_a_receipt_and_legacy_refuses_both():
    with pytest.raises(ValueError, match="needs a source"):
        CH.challenge_positions(ROOT, [10], 3, legacy=False, receipt=RECEIPT)
    with pytest.raises(ValueError, match="after a receipt"):
        CH.challenge_positions(ROOT, [10], 3, legacy=False, source=BEACON)
    with pytest.raises(ValueError, match="legacy"):
        CH.challenge_positions(ROOT, [10], 3, source=BEACON)
    with pytest.raises(ValueError, match="legacy"):
        CH.challenge_positions(ROOT, [10], 3, receipt=RECEIPT)
    with pytest.raises(ValueError, match="needs a source"):
        CH.identity_picker(ROOT, "00" * 32, legacy=False, receipt=RECEIPT)
    with pytest.raises(ValueError, match="after a receipt"):
        CH.identity_picker(ROOT, "00" * 32, legacy=False, source=BEACON)
    with pytest.raises(ValueError, match="needs a source"):
        CH.replay_key(b"", ROOT, RECEIPT)
    with pytest.raises(ValueError, match="after a receipt"):
        CH.replay_key(BEACON, ROOT, b"")


def test_non_legacy_positions_follow_the_source_root_and_layout():
    lps = [10, 0, 20, 5]
    kw = {"legacy": False, "source": BEACON, "receipt": RECEIPT}
    a = CH.challenge_positions(ROOT, lps, 64, **kw)
    assert len(a) == 64 and all(0 <= off < lps[s] for s, off in a)
    assert a == CH.challenge_positions(ROOT, lps, 64, **kw)
    assert a != CH.challenge_positions(ROOT, lps, 64)
    assert a != CH.challenge_positions(ROOT, lps, 64, **{**kw, "source": b"another secret"})
    assert a != CH.challenge_positions(ROOT, lps, 64, **{**kw, "receipt": hashlib.sha256(b"another receipt").digest()})
    assert a != CH.challenge_positions(hashlib.sha256(b"x").digest(), lps, 64, **kw)
    assert a != CH.challenge_positions(ROOT, lps, 64, b"salt", **kw)
    assert a != CH.challenge_positions(ROOT, [10, 0, 20, 6], 64, **kw)


def test_non_legacy_identities_draw_through_the_picker():
    c, run, bmap = T.commit(T.produce())
    pop = list(B.expected_population(T.structure(), T.workload()))
    fresh = B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", legacy=False, source=BEACON, receipt=RECEIPT)
    assert fresh and all(e is not None for _, e, _ in fresh)
    assert fresh == B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", legacy=False, source=BEACON, receipt=RECEIPT)
    assert fresh != B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t")
    pick = CH.identity_picker(ROOT, "ab" * 32, legacy=False, source=BEACON, receipt=RECEIPT)
    assert all(0 <= pick(i, 7) < 7 for i in range(100))


def test_non_legacy_replay_sample_is_the_two_stage_draw():
    key = CH.replay_key(BEACON, ROOT, RECEIPT)
    assert key == derive(BEACON, CH.REPLAY_DOMAIN, {"run_root": ROOT, "receipt": RECEIPT})
    picked, strata, _ = SR.sample(_vus(), 3, 0, key=key)
    assert all(s["sampled"] == 3 for s in strata.values()) and len(picked) == 3 * len(strata)
    assert [v["stratum"] for v in picked[:len(strata)]] == sorted(strata)  # round-robin: every stratum's first pick first
    assert picked == SR.sample(_vus(), 3, 0, key=key)[0]
    assert picked != SR.sample(_vus(), 3, 0, key=CH.replay_key(b"another secret", ROOT, RECEIPT))[0]


def test_stratum_picks_takes_a_seed_or_a_key():
    with pytest.raises(ValueError):
        CH.stratum_picks([[1, 2]], 1, None)
    with pytest.raises(ValueError):
        CH.stratum_picks([[1, 2]], 1, 1, key=CH.replay_key(BEACON, ROOT, RECEIPT))
    assert CH.stratum_picks([[1, 2], [3]], 5, None, key=CH.replay_key(BEACON, ROOT, RECEIPT)) == [[1, 2], [3]]


def test_the_run_salt_is_the_verifiers_over_the_weights_root_and_the_run():
    w = hashlib.sha256(b"weights").digest()
    salt = CH.run_salt(BEACON, w, "r1")
    assert len(salt) == 32 and salt == CH.run_salt(BEACON, w, "r1")
    assert salt == derive(BEACON, CH.SALT_DOMAIN, {"weights_root": w, "run_id": "r1"}).stream("salt", 32)
    assert len({salt, CH.run_salt(b"another secret", w, "r1"), CH.run_salt(BEACON, ROOT, "r1"), CH.run_salt(BEACON, w, "r2")}) == 4
    with pytest.raises(ValueError, match="needs a source"):
        CH.run_salt(b"", w, "r1")


def test_a_run_registers_each_call_index_once_across_its_windows():
    run = CH.RunWindows("r1")
    r0 = run.register(0, [(0, 0, "qkv"), (0, 0, "o")], ROOT)
    r1 = run.register(1, [(1, 0, "qkv"), (1, 0, "o")], ROOT)
    assert len({r0, r1}) == 2 and run.windows == {0: r0, 1: r1}
    assert r0 == CH.RunWindows("r1").register(0, [(0, 0, "o"), (0, 0, "qkv")], ROOT)       # the receipt binds the set, not its order
    assert r0 != CH.RunWindows("r2").register(0, [(0, 0, "qkv"), (0, 0, "o")], ROOT)
    with pytest.raises(ValueError, match="re-registers call indices of earlier windows"):
        run.register(2, [(2, 0, "qkv"), (0, 0, "o")], ROOT)
    with pytest.raises(ValueError, match="never re-registered"):
        run.register(1, [(9, 9, "x")], ROOT)
    with pytest.raises(ValueError, match="twice"):
        run.register(3, [(3, 0, "o"), (3, 0, "o")], ROOT)
    assert 2 not in run.windows and 3 not in run.windows                                       # a refused window leaves no trace
    run.abort(4)
    with pytest.raises(ValueError, match="rejected: window 4 aborted"):
        run.register(5, [(5, 0, "o")], ROOT)
    key = CH.replay_key(BEACON, ROOT, r1)                                                      # the draw follows the receipt
    assert key != CH.replay_key(BEACON, ROOT, r0)
