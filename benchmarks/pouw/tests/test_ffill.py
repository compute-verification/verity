"""ffill, the H200's FFMA filler (dc/ffill_ref.py, dc/ffill.cuh): the reference rounds once where float64-then-cast
rounds twice, agrees with the integer FP32 pipe on finite words and pins sm_90's NaN and subnormals; the operand map,
the coins and the chains; the pinned receipts; ffill_ref's checks of ffill.cu's dumps; the header's host build against
the reference; and the design that prices it. The GPU side needs an H200 (dc/ffill.sh)."""
from __future__ import annotations

import importlib.util
import random
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.silicon import fp32 as silicon
from verity_machine_model.eps import design
from verity_machine_model.model import device

DC = Path(__file__).resolve().parents[1] / "dc"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F = _load("ffill_ref")


def _f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def _bits(x) -> int:
    return int(np.float32(x).view(np.uint32))


def test_one_rounding_not_two():
    a, b, c, want = F.DOUBLE_ROUNDING
    assert F.fma(a, b, c) == want == silicon.fma(a, b, c)
    twice = _bits(np.float32(np.float64(_f32(a)) * np.float64(_f32(b)) + np.float64(_f32(c))))
    assert twice == 0x40400002 != want


def test_the_reference_is_the_fp32_pipe_on_finite_words():
    r = random.Random(1)
    n = 0
    while n < 3000:
        a, b, c = (r.getrandbits(32) for _ in range(3))
        if any((w >> 23) & 0xFF == 0xFF for w in (a, b, c)):
            continue
        assert F.fma(a, b, c) == silicon.fma(a, b, c), (hex(a), hex(b), hex(c))
        n += 1
    for _ in range(3000):     # the filler's own operands
        w = r.randrange(-F.WORD_BOUND + 1, F.WORD_BOUND)
        s = 0x3F800001 | (r.getrandbits(23) & 0x7FFFFE) | (r.getrandbits(1) << 31)
        x = 0x3F800000 + (r.getrandbits(28) & 0x0FFFFFFF)
        assert F.fma(F.operand(w), s, x) == silicon.fma(F.operand(w), s, x)


def test_sm90s_nan_and_subnormals():
    nan = 0x7FFFFFFF
    assert F.fma(0x7FC00000, 0x3F800000, 0) == F.fma(0xFF800001, 0, 0) == nan
    assert F.fma(0x7F800000, 0, 0x3F800000) == F.fma(0x7F800000, 0x3F800000, 0xFF800000) == nan
    assert F.fma(0x00800000, 0x3F000000, 0) == 0x00400000                 # 2^-127 kept, not flushed
    assert F.fma(0x00000001, 0x3F800000, 0x80000000) == 0x00000001
    assert F.fma(0x3F800000, 0x3F800000, 0xBF800000) == 0                 # 1 - 1 = +0
    assert F.fma(0x80000000, 0x3F800000, 0x80000000) == 0x80000000        # -0 + -0 = -0
    assert F.fma(0x7F7FFFFF, 0x40000000, 0) == 0x7F800000
    assert F.fma(0x3F800001, 0x3F800000, 0x33800000) == 0x3F800002        # a tie, to even
    assert F.fma(0x3F800000, 0x3F800000, 0x33800000) == 0x3F800000


def test_the_operand_map():
    for w in (-(1 << 22) + 1, -12345, 0, 1, 777, (1 << 22) - 1):
        assert _f32(F.operand(w)) == 1.5 * 2 ** 23 + w
    r = random.Random(2)
    for w in [r.randrange(-F.WORD_BOUND + 1, F.WORD_BOUND) for _ in range(2000)] + [F.WORD_BOUND - 1, 1 - F.WORD_BOUND]:
        u = F.operand(w)
        assert u >> 31 == 0 and 0 < (u >> 23) < 0xFF, w            # a positive normal number


def test_the_coins():
    r = random.Random(3)
    seen = set()
    for _ in range(500):
        s, t, x0, y0 = F.coins(r.getrandbits(64), r.getrandbits(64), r.getrandbits(20))
        assert s & 1 and t & 1 and s != t and s >> 23 == t >> 23 == 127 and s > 0x3F800000 and t > 0x3F800000
        assert x0 >> 23 == y0 >> 23 == 127
        seen.add(s)
    assert len(seen) == 500
    assert F.coins(7, 11, 5) == (0x3FFD97A7, 0x3FFC9475, 0x3FF5446F, 0x3FEB361F)


def test_every_step_rounds_and_moves_the_chain():
    """For |w| < 2^22 f·s needs about 48 bits, so nearly every step rounds, and no step is absorbed: a word moved by
    2^12, above any ulp x reaches, moves both receipt words (a move under x's ulp can round away)."""
    r = random.Random(4)
    s, t, x0, y0 = F.coins(1, 2, 3)
    words = [r.randrange(-(1 << 22) + (1 << 12), (1 << 22) - (1 << 12)) for _ in range(3 * 40)]
    base = F.chain(words, s, t, x0, y0)
    for i in range(0, len(words), 7):
        w = list(words)
        w[i] += 1 << 12
        got = F.chain(w, s, t, x0, y0)
        assert got[0] != base[0] and got[1] != base[1], i
    rounded = 0
    x = x0
    for n, w in enumerate(words):
        f, sg = F.operand(w), (n & 1) << 31
        exact = _f32(f) * _f32(s ^ sg) + _f32(x)            # float64 is exact here: 48 + a few bits
        x = F.fma(f, s ^ sg, x)
        rounded += _f32(x) != exact
    assert rounded > 0.9 * len(words)


def test_no_chain_is_affine_in_its_words():
    r = random.Random(5)
    c = F.coins(9, 9, 9)
    zero = F.chain([0] * 30, *c)
    for _ in range(20):
        a = [r.randrange(-(1 << 21), 1 << 21) for _ in range(30)]
        b = [r.randrange(-(1 << 21), 1 << 21) for _ in range(30)]
        sx = [p + q for p, q in zip(a, b)]
        lin = tuple((_f32(F.chain(a, *c)[k]) + _f32(F.chain(b, *c)[k]) - _f32(zero[k])) for k in (0, 1))
        if tuple(_f32(v) for v in F.chain(sx, *c)) != lin:
            return
    pytest.fail("the chain composed affinely on every sample")


def test_pinned_receipts():
    """A thread's receipt and 16 threads' in one digest: ffill.cu's chain kernel writes these words."""
    import hashlib
    r = random.Random(20261018)
    words = [r.randrange(-(1 << 27), 1 << 27) for _ in range(3 * 64)]
    assert F.receipt(7, 11, 5, words) == (0xD3A8908E, 0xD3A7E447)
    h = hashlib.sha512()
    for u in range(16):
        w = [r.randrange(-(1 << 27), 1 << 27) for _ in range(3 * 32)]
        h.update(struct.pack("<2I", *F.receipt(0xC0FFEE, 0xBADC0DE, u, w)))
    assert h.hexdigest()[:32] == "b933f42fb40ef8b4a6b121ad2abe5f2d"


def test_thread_words_take_the_filler_registers():
    segs = [list(range(100 * m, 100 * m + F.WORDS_PER_MMA)) for m in range(3)]
    assert F.thread_words(segs) == [100 * m + r for m in range(3) for r in F.REGS]


def test_the_checks_read_ffill_cus_dumps(tmp_path):
    cases = [F.DOUBLE_ROUNDING[:3], (0x7FC00000, 0, 0), (0x00800000, 0x3F000000, 0), (F.operand(5), 0x3FC00001, 3)]
    quads = [(a, b, c, F.fma(a, b, c)) for a, b, c in cases]
    (tmp_path / "fma.bin").write_bytes(b"".join(struct.pack("<4I", *q) for q in quads))
    got = F.check_fma(tmp_path / "fma.bin")
    assert (got["cases"], got["bad"], got["nan_results"], got["subnormal_results"]) == (4, 0, 1, 1)
    quads[0] = (*quads[0][:3], 0x40400002)                              # float64's answer
    (tmp_path / "fma.bin").write_bytes(b"".join(struct.pack("<4I", *q) for q in quads))
    got = F.check_fma(tmp_path / "fma.bin")
    assert got["bad"] == 1 and got["first"]["gpu"] == "0x40400002" and got["first"]["ref"] == "0x40400001"

    c0, c1, tile, threads, steps = 0x1234, 0x5678, 9, 3, 6
    r = random.Random(6)
    words = [r.randrange(-(1 << 27), 1 << 27) for _ in range(threads * steps)]
    co = F.coins(c0, c1, tile)
    rec = [v for u in range(threads) for v in F.chain(words[u * steps:(u + 1) * steps], *co)]

    def dump(receipts, coins=co):
        return (struct.pack("<3Q2I", c0, c1, tile, threads, steps) + struct.pack("<4I", *coins)
                + struct.pack(f"<{len(words)}i", *words) + struct.pack(f"<{len(receipts)}I", *receipts))

    (tmp_path / "chain.bin").write_bytes(dump(rec))
    got = F.check_chain(tmp_path / "chain.bin")
    assert (got["threads"], got["steps"], got["coins_ok"], got["bad"]) == (3, 6, True, 0)
    assert F.main(["check", str(tmp_path)]) == 1                        # fma.bin still holds the float64 word
    (tmp_path / "chain.bin").write_bytes(dump(rec[:3] + [rec[3] ^ 1] + rec[4:], coins=(co[1], co[0], *co[2:])))
    got = F.check_chain(tmp_path / "chain.bin")
    assert got["bad"] == 1 and got["first"]["thread"] == 1 and not got["coins_ok"]


def test_the_header_is_the_reference():
    """ffill.cuh's constants, and its coins and chain built for the host (the same __host__ __device__ code the kernels
    run), against ffill_ref."""
    src = (DC / "ffill.cuh").read_text()
    assert int(re.search(r"FF_BIAS = (0x[0-9A-Fa-f]+)u", src).group(1), 16) == F.BIAS
    assert int(re.search(r"FF_KEYS = (0x[0-9A-Fa-f]+)ull", src).group(1), 16) == F.FF_KEYS
    regs = re.search(r"FF_REGS\[FF_STEPS\] = \{([^}]*)\}", src).group(1)
    assert tuple(int(v) for v in regs.split(",")) == F.REGS
    assert int(re.search(r"FF_STEPS = (\d+)", src).group(1)) == len(F.REGS)


@pytest.mark.skipif(not shutil.which("g++"), reason="the host build of ffill.cuh needs g++")
def test_the_headers_coins_on_the_host(tmp_path):
    main = tmp_path / "coins.cpp"
    main.write_text('#include <cstdio>\n#include "ffill.cuh"\nint main() {\n'
                    '  unsigned long long a, b, c;\n'
                    '  while (scanf("%llu %llu %llu", &a, &b, &c) == 3) {\n'
                    '    uint32_t w[4]; ff_coins(a, b, c, w);\n'
                    '    printf("%u %u %u %u\\n", w[0], w[1], w[2], w[3]);\n  }\n}\n')
    exe = tmp_path / "coins"
    subprocess.run(["g++", "-O2", "-std=c++17", "-D__host__=", "-D__device__=", f"-I{DC}", "-o", str(exe), str(main)],
                   check=True)
    r = random.Random(7)
    keys = [(r.getrandbits(64), r.getrandbits(64), r.getrandbits(20)) for _ in range(200)] + [(7, 11, 5)]
    out = subprocess.run([str(exe)], input="".join(f"{a} {b} {c}\n" for a, b, c in keys), capture_output=True,
                         text=True, check=True).stdout.split("\n")
    for (a, b, c), line in zip(keys, out):
        assert tuple(int(v) for v in line.split()) == F.coins(a, b, c), (a, b, c)


def test_the_design_prices_ffill():
    f, = design("dither-2s-licpw-redux-l32", device("sm_90")).filler
    assert f.per_word == F.PER_WORD == 9 / 128 and f.mix == F.MIX and f.route == "lanes" and f.p == 0.0
    assert f.luck.startswith("Accounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit,")
    assert "ffill" in f.luck and "F32FmaV2" in f.luck
