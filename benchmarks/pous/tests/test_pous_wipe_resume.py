"""The resume attack against the streamed-label wipe (`benchmarks/pous/wipe_resume/resume.py`) and the band fill
(`pi2_state_attack.py`), at a tiny size: the streamed wipe accepts the honest prover and rejects the naive cheat and the
resume prover however long it searches, and accepts a prover only when a premise is broken (`K` handed to it, or the
labels kept in an undeclared tier); the band and dense labels are rebuilt bit for bit from five SHAKE256 states.
The clock is injected (each reading advances it), the coins fixed, and the verdict is the Python reference's."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("verity.core.protocols.pous")

WIPE_RESUME = Path(__file__).resolve().parents[1] / "wipe_resume"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"wipe_resume_{name}", WIPE_RESUME / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tick:
    """A clock that advances 1 µs at each reading."""

    def __init__(self) -> None:
        self.now = 10**9

    def __call__(self) -> int:
        self.now += 1000
        return self.now


@pytest.fixture(scope="module")
def streamed(tmp_path_factory):
    resume = _load("resume")
    from verity.core.protocols.pous import streamed_wipe as sw

    def python(signed, commitment, *, trusted, schemes, decide):
        v = sw.verdict(signed, commitment, trusted=trusted, schemes=schemes)
        return v, {"python": {"accepted": v.accepted, "egress_mib": v.egress_mib, "reasons": list(v.reasons)}}

    out = resume.run(n=256, k=400, limit_ns=1_000_000, decide="python", scratch=tmp_path_factory.mktemp("wipe-resume"),
                     clock=Tick(), untimed_tries=1 << 10, sample_sessions=0, judge=python, coin_secret=b"fixed coins")
    return resume, {s["prover"]: s for s in out["sessions"]}, out


def test_every_prover_ends_as_expected(streamed):
    _, by, out = streamed
    assert out["as_expected"], {p: (s["accepted"], s["ended"], s["reasons"]) for p, s in by.items()}
    assert [p for p, s in by.items() if s["accepted"]] == ["honest", "key-leak", "untiered"]


def test_the_naive_cheat_keeps_its_prior_content_and_answers_wrong(streamed):
    _, by, _ = streamed
    assert by["naive"]["kept"] and by["naive"]["ended"] == "wrong"


def test_the_resume_prover_keeps_63_of_64_bytes_and_never_finds_the_key(streamed):
    resume, by, _ = streamed
    for name, tries in (("resume", None), ("resume-untimed", 1 << 10)):
        s = by[name]
        assert s["kept"] and s["prior_share"] > 0.95 and s["ended"] == "wrong" and not s["key_found"]
        assert s["kept_label_bytes"] == resume.KEPT == 1
        if tries is not None:
            assert s["keys_searched"] == tries
    assert by["resume"]["read_ns"]["max"] <= by["resume"]["limit_ns"]


def test_only_a_broken_premise_keeps_the_whole_store(streamed):
    _, by, _ = streamed
    for name in ("key-leak", "untiered"):
        assert by[name]["accepted"] and by[name]["kept"] and by[name]["prior_share"] == 1.0
    assert not by["honest"]["kept"]


def test_the_band_and_dense_labels_rebuild_from_five_states():
    attack = _load("pi2_state_attack")
    m = 8192
    for rec in attack.run(blocks=3, m=m, d=2):
        assert rec["rebuilt_bit_exact"] == 3
        assert rec["answer_sequential_keccak_f"] == -(-(m // 8) // attack.RATE) - 1
        assert rec["answer_sequential_keccak_f"] < rec["pi2_call_keccak_f"]
