"""U's CPU pin (`verity.core.protocols.pouw.schemes.pearl_c_u`): the integer grouped sum equals `verity.ml.circuits.silicon`'s two step models, the pin
equals `pearl_c`'s reference, and the pinned vectors (cut from the H100 kernel's gate problem) reproduce."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import pearl_c_u_vectors as UV
import pearl_c_vectors as CV
from verity.ml.circuits.silicon.models import HOPPER_BF16_WGMMA_K16, HOPPER_E4M3_K32
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c_u as U

VECTORS = json.loads((Path(__file__).with_name("vectors") / "pearl_c_u.json").read_text())


def e4m3_codes(rng: random.Random, n: int) -> list[int]:
    out = []
    for _ in range(n):
        r = rng.random()
        c = 0 if r < 0.1 else rng.randrange(1, 8) if r < 0.2 else rng.randrange(0, 0x7F)
        out.append(c | (0x80 if rng.random() < 0.5 else 0))
    return out


def acc_word(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.1:
        return rng.choice([0, 0x80000000])
    if r < 0.25:
        return rng.randrange(2) << 31 | rng.randrange(1, 1 << 23)
    return rng.randrange(2) << 31 | rng.randint(60, 190) << 23 | rng.randrange(1 << 23)


def test_status_is_cpu_pinned():
    assert U.STATUS == VECTORS["status"] == "cpu-pinned"
    assert VECTORS["gate"]["agree"] and VECTORS["problem"]["shape"] == [256, 384, 1024]


def test_grouped_sum_is_the_tc_models():
    """The integer statement against HOPPER_BF16_WGMMA_K16 and HOPPER_E4M3_K32 on random words: subnormals, zeros, both
    signs, accumulators from subnormal to large."""
    rng = random.Random("pearl-c-u/grouped")
    for _ in range(1500):
        acc = acc_word(rng)
        x = [UV.bf16(rng, 60, 190, 0.1) for _ in range(16)]
        y = [UV.bf16(rng, 60, 190, 0.1) for _ in range(16)]
        assert U.k16(acc, x, y) == HOPPER_BF16_WGMMA_K16.step(acc, x, y)
    for _ in range(1500):
        acc = acc_word(rng) & 0x87FFFFFF if rng.random() < 0.5 else acc_word(rng)
        a, b = e4m3_codes(rng, 32), e4m3_codes(rng, 32)
        assert U.atom(acc, a, b) == HOPPER_E4M3_K32.step(acc, a, b)


def test_pin_is_the_reference():
    """chain and u against pearl_c's chain and peel on random codes and factors, partial windows included, and the peel
    factors against pearl_c's on test instances of both formings."""
    rng = random.Random("pearl-c-u/reference")
    for k in (32, 96, 128, 160, 512):
        for _ in range(20):
            a, b = e4m3_codes(rng, k), e4m3_codes(rng, k)
            assert U.chain(a, b) == C.chain(a, b)
    for _ in range(200):
        c = C.P.f32_bits(rng.uniform(-1e4, 1e4))
        pa = [C.P.f32_to_bf16(rng.uniform(-8, 8)) for _ in range(U.PEEL_K)]
        pb = [C.P.f32_to_bf16(rng.uniform(-8, 8)) for _ in range(U.PEEL_K)]
        assert U.u(c, pa, pb) == C.peel(c, pa, pb)
    for forming, make in (("v0", CV.rows), ("v1", CV.rows_f32)):
        s = C.PearlC(k_min=32, tile=(16, 16), forming=forming)
        ctx = CV.ctx_of("pin", (16, 128, 16), 3, 9)
        noise = s.noise(ctx)
        a, e_a = s._side(make("pin", "A", 4, 128), range(4), 0, noise.seed_a, noise.seed_b, 128)
        b, e_b = s._side(make("pin", "B", 4, 128), range(4), 1, noise.seed_b, noise.seed_b, 128)
        be_a, be_b = C.scaled_noise(a, e_a), C.scaled_noise(b, e_b)
        fa, fb = C._basis(noise.seed_b, 0, 128, s.line_norm), C._basis(noise.seed_b, 1, 128, s.line_norm)
        g = [[U.chain(UV.column(fb, p), UV.column(fa, q)) for q in range(U.R)] for p in range(U.R)]
        for codes, be, got in zip(a.codes, be_a, C.peel_factors_a(a.codes, be_a, noise.seed_b, 128, s.line_norm)):
            assert U.pa_row(be, [U.chain(codes, UV.column(fb, p)) for p in range(U.R)]) == got
        for codes, be, got in zip(b.codes, be_b, C.peel_factors_b(b.codes, be_b, noise.seed_b, 128, s.line_norm)):
            assert U.pb_row(be, [U.chain(codes, UV.column(fa, q)) for q in range(U.R)], g) == got


def test_vectors_reproduce():
    for acc, x, y, w, _ in VECTORS["k16"]["gate"] + VECTORS["k16"]["edge"]:
        assert U.k16(acc, x, y) == HOPPER_BF16_WGMMA_K16.step(acc, x, y) == w
    for forming, f in VECTORS["formings"].items():
        pa = {r["i"]: r["pa"] for r in f["rows_a"]}
        pb = {r["j"]: r["pb"] for r in f["rows_b"]}
        for r in f["rows_a"]:
            assert U.pa_row(r["be"], r["af"]) == r["pa"]
        for r in f["rows_b"]:
            trace: dict = {}
            assert U.pb_row(r["be"], r["bf"], f["g"], trace) == r["pb"]
            assert trace == {"acc": r["acc"], "diff": r["diff"]}
        for cell in f["cells"]:
            steps: list = []
            assert U.u(cell["c"], pa[cell["i"]], pb[cell["j"]], steps) == cell["u"] and steps == cell["steps"]
        for ch in f["chains"]:
            a, b = bytes.fromhex(ch["a"]), bytes.fromhex(ch["b"])
            trace = {}
            assert U.chain(a, b, trace) == C.chain(a, b) == ch["c"]
            assert trace == {"atoms": ch["atoms"], "totals": ch["totals"]} and len(ch["totals"]) == len(a) // U.WINDOW
            assert next(c for c in f["cells"] if (c["i"], c["j"]) == (ch["i"], ch["j"]))["c"] == ch["c"]


@pytest.mark.slow
def test_vectors_are_pinned():
    """Regenerated from the scheme (without the gate directory, whose hashes the file records), the vectors are the
    file's."""
    fresh = json.loads(UV.dumps(UV.vectors()))
    assert {k: v for k, v in fresh.items() if k != "gate"} == {k: v for k, v in VECTORS.items() if k != "gate"}
