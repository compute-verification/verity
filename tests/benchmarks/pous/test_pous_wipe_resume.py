"""The resume attack against the streamed-label wipe (`benchmarks/pous/wipe_resume/resume.py`) and the band fill
(`pi2_state_attack.py`), at a tiny size: the streamed wipe accepts the honest prover and rejects the naive cheat and the
resume prover however long it searches, and accepts a prover only when a premise is broken (`K` handed to it, or the
labels kept in an undeclared tier); the band and dense labels are rebuilt bit for bit from five SHAKE256 states.
The clock is injected (each reading advances it), the coins fixed, and no Lean judges (`decide="python"`): what the
check passed is the Python reference's side, and no session is recorded accepted on its word."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("verity.protocols.resource_accounting.memory.pous.verifier")

WIPE_RESUME = Path(__file__).resolve().parents[3] / "benchmarks" / "pous" / "wipe_resume"
POUS_TESTS = Path(__file__).resolve().parents[2] / "verity" / "protocols" / "resource_accounting" / "memory" / "pous" / "verifier"


def _load(name: str, where: Path = WIPE_RESUME):
    spec = importlib.util.spec_from_file_location(f"wipe_resume_{name}", where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def streamed(tmp_path_factory):
    resume = _load("resume")
    out = resume.run(n=256, k=400, limit_ns=1_000_000, decide="python", scratch=tmp_path_factory.mktemp("wipe-resume"),
                     clock=_load("pous_fakes", POUS_TESTS).Clock(10 ** 9, 1_000), untimed_tries=1 << 10,
                     sample_sessions=0, coin_secret=b"fixed coins")
    return resume, {s["prover"]: s for s in out["sessions"]}, out


def _passed(s) -> bool:
    return s["verdicts"]["python"]["accepted"]


def test_every_prover_ends_as_expected(streamed):
    _, by, out = streamed
    assert out["as_expected"], {p: (s["outcome"], s["ended"], s["reasons"]) for p, s in by.items()}
    assert [p for p, s in by.items() if _passed(s)] == ["honest", "key-leak", "untiered"]
    assert {s["expected_by"] for s in by.values()} == {"python", "verdict"}


def test_with_no_lean_no_session_is_recorded_accepted_on_the_pythons_word(streamed):
    """Under `decide="python"` no Lean judges: a session the Python accepts is `undecided` (`wipe_verdict.NO_JUDGE`),
    not accepted, its accept only in the Python's own side; one it rejects stays rejected."""
    resume, by, _ = streamed
    wv = resume._wipe_verdict()
    for p, s in by.items():
        assert s["accepted"] is False, p
        if _passed(s):
            assert (s["outcome"], s["reasons"], s["expected_by"]) == ("undecided", [wv.NO_JUDGE], "python"), p
        else:
            assert s["outcome"] == "reject" and s["expected_by"] == "verdict", p


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
        assert _passed(by[name]) and by[name]["kept"] and by[name]["prior_share"] == 1.0
    assert not by["honest"]["kept"]


def test_the_band_and_dense_labels_rebuild_from_five_states():
    attack = _load("pi2_state_attack")
    m = 8192
    for rec in attack.run(blocks=3, m=m, d=2):
        assert rec["rebuilt_bit_exact"] == 3
        assert rec["answer_sequential_keccak_f"] == -(-(m // 8) // attack.RATE) - 1
        assert rec["answer_sequential_keccak_f"] < rec["pi2_call_keccak_f"]
