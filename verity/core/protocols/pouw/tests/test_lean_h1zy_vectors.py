"""The Lean mirrors of a lane's chain words and of the dequant (`Security/Definitions/Pouw/H1Zy/Chain.lean`, `Dequant.lean`)
are held to `h1zy.lane_partial` and `h1zy.dequant` by vectors: `Security/Proofs/Pouw/H1Zy/Vectors.lean` states each one
as a theorem the kernel checks against the Lean, and this test checks the same vectors against the Python, at one
contraction per digit count the kernel can afford (`m = 1` at K = 32, `m = 2` at K = 64) and on the dequant's special
words. A change to either side's semantics then fails a build.

Run as a script, this file writes the vectors from the Python."""
from __future__ import annotations

import random
import re
import struct
from pathlib import Path

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy
from verity.core.protocols.pouw.circuit import seeds as SD

LEAN = Path(__file__).resolve().parents[5] / "Security"
VECTORS = LEAN / "Proofs" / "Pouw" / "H1Zy" / "Vectors.lean"
M32 = (1 << 32) - 1
LANE = re.compile(r"theorem (\w+) : lanePartial (\d+) (0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+) (\d+)\s+(\[\[.*?\]\])\s+(\[\[.*?\]\]) =\s+(\[.*?\]) :=",
                  re.S)
DEQUANT = re.compile(r"theorem (\w+) : dequant (0x[0-9a-f]+) (0x[0-9a-f]+) (0x[0-9a-f]+) = (0x[0-9a-f]+) :=")
#: the lanes the vectors take: (K, lane, generator seed)
LANES = ((32, 5, 1), (64, 13, 2))


def _port(bits: list[int]) -> int:
    """A port field's number: its leaves' bits, least significant first."""
    return sum((b & 1) << i for i, b in enumerate(bits))


def _bytes(n: int, halves: int) -> bytes:
    """The bytes whose leaves `_port` reads as `n`: half h the bits 32 h .. 32 h + 31, each half big-endian."""
    return b"".join(((n >> 32 * h) & M32).to_bytes(4, "big") for h in range(halves))


def _lane_inputs(S: int, P: int, R: int) -> tuple[bytes, int, tuple[bytes, bytes]]:
    salt, roots = _bytes(S, 16), _bytes(R, 32)
    pos = int.from_bytes(_bytes(P, 2), "big")
    if (_port(SD.leaves(salt, True)), _port(SD.position_leaves(pos, True)),
            _port(SD.leaves(roots[:64], True) + SD.leaves(roots[64:], True))) != (S, P, R):
        raise AssertionError("the port numbers are not the seeds' leaves")
    return salt, pos, (roots[:64], roots[64:])


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _dequant_cases() -> list[tuple[int, int, int]]:
    """z: zero, ±1, the s32 extremes, rounding cases above 2^24; scales: signed zeros, the infinities, NaNs, subnormals,
    the extremes, ordinary values; then seeded draws."""
    zs = [0, 1, M32, 1 << 31, (1 << 31) - 1, (1 << 24) + 1, (1 << 24) + 3, 0x7FFFFFC0, (-(1 << 24) - 1) & M32, 1000]
    fs = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 1, 0x007FFFFF, 0x7F7FFFFF, _f32(1.0),
          _f32(-0.5), _f32(1e-30), _f32(1 / 3)]
    out = [(z, fs[i % len(fs)], fs[(3 * i + 1) % len(fs)]) for i, z in enumerate(zs)]
    out += [(z, s, _f32(1.0)) for z in (1, 3, 255) for s in fs[:7]]
    rnd = random.Random(7)
    out += [(rnd.randrange(1 << 32), _f32(rnd.uniform(-1, 1) * 2.0 ** rnd.randrange(-130, 100)),
             _f32(rnd.uniform(-1, 1) * 2.0 ** rnd.randrange(-130, 100))) for _ in range(12)]
    return out


def _lit(x: list, indent: str = "    ") -> str:
    """A (nested) list of naturals as a Lean literal, wrapped at 120 columns."""
    out, line = [], indent
    for tok in re.findall(r"[^,]+(?:, )?", repr(x)):
        if len(line) + len(tok.rstrip()) > 120:
            out.append(line.rstrip())
            line = indent + " "
        line += tok
    return "\n".join(out + [line])


def render() -> str:
    """The vectors file, from the Python."""
    parts = ['import Definitions.Pouw.H1Zy.Chain\nimport Definitions.Pouw.H1Zy.Dequant\n\n/-!\n'
             "# A lane's chain words and the dequant against the Python (generated)\n\n"
             "Written by `verity/core/protocols/pouw/tests/test_lean_h1zy_vectors.py` from `h1zy.lane_partial` (a live leaf's)\n"
             "and `h1zy.dequant`, which that test holds these to; each is evaluated by the kernel.\n\n"
             "* `lane_k<K>_l<lane>`: `Pouw.H1Zy.lanePartial` at `K = 32` (one digit) and `K = 64` (two), on seeded codes, salt,\n"
             "  position and roots, the public inputs as their port fields' numbers.\n"
             "* `dequant_<i>`: `Pouw.H1Zy.dequant` on z's s32 word and the scales' f32 encodings: zero, the s32 extremes, rounding\n"
             "  cases, and scales of every kind (signed zeros, the infinities, NaNs, subnormals, the extremes), then seeded words.\n"
             "-/\n\nnamespace Pouw.H1Zy.Vectors\n\nopen Pouw.H1Zy\n"]
    for K, lane, seed in LANES:
        rnd = random.Random(seed)
        salt, ra, rb = (bytes(rnd.randrange(256) for _ in range(64)) for _ in range(3))
        pos = rnd.randrange(1 << 64)
        a = [[rnd.randrange(256) for _ in range(K)] for _ in range(4)]
        b = [[rnd.randrange(256) for _ in range(K)] for _ in range(8)]
        S, P = _port(SD.leaves(salt, True)), _port(SD.position_leaves(pos, True))
        R = _port(SD.leaves(ra, True) + SD.leaves(rb, True))
        want = h1zy.lane_partial(a, b, [1] * 32, salt, pos, (ra, rb), lane, (32, 32))
        parts.append(f"\ntheorem lane_k{K}_l{lane} : lanePartial {K} {S:#x}\n    {P:#x}\n    {R:#x} {lane}\n{_lit(a)}\n{_lit(b)} =\n"
                     f"{_lit(want)} := by\n  decide +kernel\n")
    for i, (z, sa, sw) in enumerate(_dequant_cases()):
        parts.append(f"\ntheorem dequant_{i} : dequant {z:#x} {sa:#x} {sw:#x} = {h1zy.dequant(z, sa, sw):#x} := by decide +kernel\n")
    return "".join(parts) + "\nend Pouw.H1Zy.Vectors\n"


def _ints(s: str) -> list:
    return eval(s, {"__builtins__": {}})  # a Lean list literal of naturals is a Python one


def test_the_lane_vectors_are_lane_partials() -> None:
    text = VECTORS.read_text()
    seen = set()
    for name, K, S, P, R, lane, a, b, want in LANE.findall(text):
        K, lane = int(K), int(lane)
        salt, pos, roots = _lane_inputs(int(S, 16), int(P, 16), int(R, 16))
        a, b = _ints(a), _ints(b)
        assert len(a) == 4 and len(b) == 8 and all(len(r) == K for r in a + b), name
        got = h1zy.lane_partial(a, b, [1] * 32, salt, pos, roots, lane, (32, 32))
        assert got == _ints(want), name
        seen.add(H.Layout(32, 32, K).m)
    assert seen == {1, 2}


def test_the_dequant_vectors_are_the_dequants() -> None:
    rows = DEQUANT.findall(VECTORS.read_text())
    assert len(rows) >= 30
    kinds = set()
    for name, z, sa, sw, y in rows:
        z, sa, sw, y = (int(v, 16) for v in (z, sa, sw, y))
        assert h1zy.dequant(z, sa, sw) == y, name
        kinds |= {"nan" if y == 0x7FFF else "inf" if y & 0x7FFF == 0x7F80 else "zero" if y & 0x7FFF == 0 else "finite",
                  "negative z" if z >> 31 else "z"}
    assert kinds == {"nan", "inf", "zero", "finite", "negative z", "z"}


def test_the_vectors_file_is_rendered() -> None:
    assert VECTORS.read_text() == render()


if __name__ == "__main__":
    VECTORS.write_text(render())
