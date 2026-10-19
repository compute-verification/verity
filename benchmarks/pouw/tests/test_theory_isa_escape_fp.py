"""isa_escape_fp.py, run as written: per device (sm_120, H200, B200), no tensor form is open to an elementwise route for
an FP8, FP4 or int8-on-FP word, the IMMA and H200 FP4 ties are found, k3/k3h/k4/k9 hold on every operand pair, premise
(a) holds on the models' replays and fails just outside it, and every mutation is caught. Its model transcription is held
to the registry's Python models here, which the stdlib script can't import."""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from verity.ml.circuits.silicon import models as M

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "isa_escape_fp.py"


def _script():
    spec = importlib.util.spec_from_file_location("isa_escape_fp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_isa_escape_fp(tmp_path):
    """The verdicts, the constructions and the premise replays are claims only running the script checks."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 6
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 70


def _acc(rng: random.Random, lo: int, hi: int) -> int:
    r = rng.random()
    if r < 0.1:
        return rng.choice([0, 0x80000000])
    if r < 0.2:
        return (rng.randrange(2) << 31) | rng.randrange(1, 1 << 23)
    return (rng.randrange(2) << 31) | (rng.randrange(lo, hi) << 23) | rng.randrange(1 << 23)


def test_transcription_matches_the_registry_models():
    """The script replays premise (a) on its own transcription of the bit-exact models; this checks that transcription
    word for word against the registry's (the Lean definitions transcribe the same Python) on random finite steps:
    sm_120's and Hopper's E4M3 group sums, and sm_120's NVF4 and MXF4 align-add."""
    s = _script()
    rng = random.Random(20261019)
    finite = [c for c in range(256) if c % 128 != 127]
    for model, (width, floor, _) in ((M.BLACKWELL_SM120_E4M3_M16N8K32, s.PIPELINES["sm_120"]),
                                     (M.HOPPER_E4M3_K32, s.PIPELINES["sm_90a"])):
        assert (model.groups, model.width, model.zero_exponent) == ((32,), width, floor)
        for _ in range(3000):
            acc = _acc(rng, 1, 230)
            a = [rng.choice(finite) for _ in range(32)]
            b = [rng.choice(finite) for _ in range(32)]
            assert s.word_value(M.tc_dot(model, acc, a, b)) == s.fp8_step(width, floor, acc, a, b), (model.name, acc, a, b)
    scales = {"ue4m3": list(range(0x08, 0x50)), "ue8m0": list(range(110, 146))}
    for model, fmt in ((M.BLACKWELL_SM120_NVF4, "ue4m3"), (M.BLACKWELL_SM120_MXF4, "ue8m0")):
        ns = 64 // s.SCALES[fmt][1]
        for _ in range(3000):
            acc = _acc(rng, 60, 190)
            a = [rng.randrange(16) for _ in range(64)]
            b = [rng.randrange(16) for _ in range(64)]
            sa = [rng.choice(scales[fmt]) for _ in range(ns)]
            sb = [rng.choice(scales[fmt]) for _ in range(ns)]
            assert s.fp4_step(fmt, acc, a, b, sa, sb) == model.step_scaled(acc, a, b, sa, sb), (model.name, acc)


def test_premise_a_on_the_b200_fp4_models():
    """Premise (a) for B200's `kind::mxf4nvf4` and `kind::mxf4` at p = 24, replayed on the registry's pinned sm_100
    models (no Lean model exists for them): every small-grid step writes the exact sum."""
    s = _script()
    rng = random.Random(20261020)
    for model, fmt in ((M.BLACKWELL_SM100_NVF4, "ue4m3"), (M.BLACKWELL_SM100_MXF4, "ue8m0")):
        n = 0
        for acc, a, b, sa, sb, c, prods in s.fp4_cases(rng, fmt, 400):
            if not s.small_grid(24, c, prods):
                continue
            n += 1
            assert s.word_value(model.step_scaled(acc, a, b, sa, sb)) == c + sum(prods, Fraction(0)), model.name
        assert n >= 300, n
