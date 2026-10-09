"""V*'s verifier-registered values against the verifier's own coins (`Flock/RecCoins.lean`, `verify --rec-inner-*`): the
smallest level of a recorded recursive session (`r20261009-111638-6720`, class (16,16,8) at 64 instances: level 3,
`rec-open/l16-h4`, 106 openings), restaged here by `rec_vstage.stage` from the fetched fixture `rec-coins/16-16-8-n64` (the
inner statement, the inner session's public record and the prover's, V*'s fold, points and running-sum salts, the level's
recorded `pub.bin` and `registered.json`, and its recorded outer session). The honest level is accepted at the honest inner
session; at another inner session of the same statement (the run's other-coins control) its coefficients are not the
verifier's, and a level whose `rec-coef` V* registered one bit off is refused at that opening, though its registration is
consistent. Staging the level takes about a minute and each verify sets up its 158 MB circuit, so the module is slow; it
catches what no test before it can: the Lean verifier's coefficients, positions and salts at real coins, against the ones
V*'s Python registers."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_vstage as S

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
SET = HERE / "fixtures" / "rec-coins" / "16-16-8-n64"
LEVEL = 3
UNIT = "rec-open/l16-h4"
#: the recorded level's circuit (`v/L3/circuit.txt`), which its outer session proves
CIRCUIT_SHA512 = ("4c47dec94a141028c1bb722b703f0c5d5624b736cfecbedb142cd0512c411e1fc91b0317d2de6a74c5897476217bd89904dabdcfc71e84"
                  "28ff1c6ecaa6c399f9")

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def forged(tmp_path_factory) -> Path:
    """Level 3 staged with rep 0's opening 0 `coef` one bit off (bit 0 of its first word) where V* computes it, so its
    registration, the running sums and the prover's row all agree with it. The circuit, program and partition do not read
    the coefficients: they are the recorded level's."""
    assert (SET / "u" / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/rec-coins`")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    honest = S.coefficients

    def coefficients(vf, qs):
        cs = honest(vf, qs)
        c = cs[(0, LEVEL, 0)]
        cs[(0, LEVEL, 0)] = [c[0] ^ 1, *c[1:]]
        return cs

    d = tmp_path_factory.mktemp("rec-coins")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "coefficients", coefficients)
        res = S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d, levels=[LEVEL],
                      algebra=False, pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin")
    out = d / f"L{LEVEL}"
    assert res["levels"][f"L{LEVEL}"]["opened"] == res["levels"][f"L{LEVEL}"]["instances"] == 106
    assert hashlib.sha512((out / "circuit.txt").read_bytes()).hexdigest() == CIRCUIT_SHA512
    return out


def _verify(level: Path, public: Path, registered: Path, inner: str) -> dict:
    p = subprocess.run([str(BIN), "verify", "--zk", "--statement", "verity/flock-circuit", "--circuit", str(level / "circuit.txt"),
                        "--public", str(public), "--partition", str(level / "partition.json"), "--program",
                        str(level / "program.json"), "--registered", str(registered),
                        "--rec-inner-circuit", str(SET / "u" / "stage" / "circuit.txt"),
                        "--rec-inner-public", str(SET / "u" / "stage" / "pub-64.bin"),
                        "--rec-inner-partition", str(SET / "u" / "stage" / "partition.json"),
                        "--rec-inner-session", str(SET / inner / "session"), "--session", str(SET / "outer-L3" / "session")],
                       capture_output=True, text=True, check=False)
    (v,) = [json.loads(line[8:]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]
    return v


def test_the_honest_level_is_accepted_at_the_honest_inner_session(forged):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", "inner-honest")
    assert v["accepted"], v.get("why")


def test_the_level_is_refused_at_another_inner_sessions_coins(forged):
    v = _verify(forged, SET / "L3" / "pub.bin", SET / "L3" / "registered.json", "inner-other-coins")
    assert not v["accepted"]
    assert v["why"].startswith(f"setup: rec-coins: {UNIT}: level {LEVEL} rep "), v["why"]
    assert "the statement's coef row is not the verifier's rec-coef row" in v["why"]


def test_a_coef_registered_one_bit_off_is_refused_at_its_opening(forged):
    v = _verify(forged, forged / "pub.bin", forged / "registered.json", "inner-honest")
    assert v["why"] == (f"setup: rec-coins: {UNIT}: level {LEVEL} rep 0 opening 0: the statement's coef row is not the "
                        "verifier's rec-coef row 0")
