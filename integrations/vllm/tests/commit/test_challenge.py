"""The one challenge module: every sample derives through `verity.ml.randomness.derive` from the run's live coin and is
refused without one, and the legacy replay seeds stay byte-identical to the outputs recorded before they moved onto it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.ml.randomness import derive
from verity_vllm.acquire.committer_api import RunCommitment, StepCommitment
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


@pytest.mark.parametrize("case", V["seeds"], ids=lambda c: c["challenge"])
def test_legacy_replay_seed_is_byte_identical(case):
    assert SR.challenge_seed(case["challenge"], ROOT) == (case["seed"], case["source"])


@pytest.mark.parametrize("case", V["samples"], ids=lambda c: f"per{c['per_stratum']}")
def test_legacy_stratified_sample_is_byte_identical(case):
    picked, strata, tgt = SR.sample(_vus(), case["per_stratum"], case["seed"], targets=case["targets"])
    assert [v["i"] for v in picked] == case["picked"] and strata == case["strata"] and tgt == case["targets_out"]


def _run(roots: list[bytes], errors: tuple[str, ...] = ()) -> RunCommitment:
    steps = [StepCommitment(i, r, r, 4, 64, 1, b"", 0.0, 0.0, 0) for i, r in enumerate(roots)]
    width = {len(r) for r in roots}
    root = (ROOT if width == {32} else hashlib.sha512(ROOT).digest()) if width in ({32}, {64}) else b""
    return RunCommitment(root, steps, {}, 0.0, 0.0, 0, 0, extra={"errors": list(errors)})


def test_a_run_with_an_uncommitted_step_has_no_root_to_draw_from():
    """The Commit's opening draw (`pipeline.commit`, the config run's ``--bounded-staging``) takes the root through
    `run_root_of`: rows-sha512's 64-byte root or vllm-v1's 32; a step the committer refused is a placeholder root and the
    run has none, refused by its steps and first error, not by the root's length."""
    step = hashlib.sha256(b"step").digest()
    assert CH.run_root_of(_run([step, step])) == ROOT
    rows = hashlib.sha512(b"step").digest()
    assert CH.run_root_of(_run([rows, rows])) == hashlib.sha512(ROOT).digest()
    with pytest.raises(CH.NoRunRoot, match=r"1 of 2 steps not committed \(steps \[1\]\)"):
        CH.run_root_of(_run([rows, b""]))
    err = "bounded staging: hit step 0 ran 6434560 bytes against a plan of 23539456"
    with pytest.raises(CH.NoRunRoot, match=r"2 of 3 steps not committed \(steps \[0, 2\]\); first error: bounded staging: hit step 0"):
        CH.run_root_of(_run([b"", step, b""], (err,)))
    with pytest.raises(CH.NoRunRoot, match="first error: none recorded"):
        CH.run_root_of(_run([b""]))
    with pytest.raises(ValueError, match=r"a run root is 32 bytes \(vllm-v1\) or 64 \(SHA-512\), not 0"):
        CH.challenge_positions(b"", [4], 1, source=AUDITOR_KEY)


def test_a_sample_without_its_coin_is_refused():
    """No sample has a default: a caller names its coin, and a missing one (None or empty) is refused by name."""
    with pytest.raises(TypeError, match="source"):
        CH.challenge_positions(ROOT, [10], 3)
    with pytest.raises(TypeError, match="source"):
        CH.identity_picker(ROOT, "00" * 32)
    for missing in (None, b""):
        with pytest.raises(ValueError, match="needs a source: the run's live coin"):
            CH.challenge_positions(ROOT, [10], 3, source=missing)
        with pytest.raises(ValueError, match="needs a source: the run's live coin"):
            CH.identity_picker(ROOT, "00" * 32, source=missing)
    with pytest.raises(ValueError, match="needs a source: the run's live coin"):
        CH.replay_key(b"", ROOT)


def test_positions_follow_the_source_root_and_layout():
    lps = [10, 0, 20, 5]
    a = CH.challenge_positions(ROOT, lps, 64, source=AUDITOR_KEY)
    assert len(a) == 64 and all(0 <= off < lps[s] for s, off in a) and {s for s, _ in a} == {0, 2, 3}
    assert a == CH.challenge_positions(ROOT, lps, 64, source=AUDITOR_KEY)
    assert a != CH.challenge_positions(ROOT, lps, 64, source=b"another key")
    assert a != CH.challenge_positions(hashlib.sha256(b"x").digest(), lps, 64, source=AUDITOR_KEY)
    assert a != CH.challenge_positions(ROOT, lps, 64, b"salt", source=AUDITOR_KEY)
    assert a != CH.challenge_positions(ROOT, [10, 0, 20, 6], 64, source=AUDITOR_KEY)


def test_positions_draw_over_a_sha512_root_and_refuse_other_sizes():
    root = hashlib.sha512(b"tp run root").digest()
    lps = [10, 0, 20, 5]
    a = CH.challenge_positions(root, lps, 64, source=AUDITOR_KEY)
    assert len(a) == 64 and all(0 <= off < lps[s] for s, off in a)
    assert a != CH.challenge_positions(root[:32], lps, 64, source=AUDITOR_KEY)
    for n in (0, 31, 48, 65):
        with pytest.raises(ValueError, match=f"32 bytes .* or 64 .*not {n}"):
            CH.challenge_positions(bytes(n), lps, 4, source=AUDITOR_KEY)


def test_identities_sample_through_the_picker():
    c, run, bmap = T.commit(T.produce())
    pop = list(B.expected_population(T.structure(), T.workload()))
    fresh = B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", source=AUDITOR_KEY)
    assert fresh and all(e is not None for _, e, _ in fresh)
    assert fresh == B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", source=AUDITOR_KEY)
    assert fresh != B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", source=b"another key")
    with pytest.raises(ValueError, match="needs a source: the run's live coin"):
        B.challenge_identities(pop, bmap, run.run_root, 16, salt=b"t", source=None)
    pick = CH.identity_picker(ROOT, "ab" * 32, source=AUDITOR_KEY)
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
