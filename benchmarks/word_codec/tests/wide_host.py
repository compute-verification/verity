"""`wide.cu`'s field arithmetic and codec built for the host (WW_HOST, g++) against `wide.py`: encode and the raw decode
of random and edge words at each field and k. Not a suite test (it needs a C++ compiler); run it before a GPU run:

    uv run python benchmarks/word_codec/tests/wide_host.py [--words 300] [--fields 127,255,521] [--ks 1,2,3]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import wide as W  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/wide/host").digest()


def hexlimbs(v: int, f: W.Field) -> str:
    return " ".join(f"{x:08x}" for x in W.limbs(v, f))


def build(bits: int, kmax: int, out: Path) -> Path:
    exe = out / f"wide-host-{bits}"
    cmd = ["g++", "-O2", "-std=c++17", "-x", "c++", "-DWW_HOST", f"-DWW_BITS={bits}", f"-DWW_KMAX={kmax}",
           str(HERE.parent / "wide.cu"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    return exe


def edges(f: W.Field) -> list[int]:
    p = f.p
    return [0, 1, 2, 3, p - 1, p - 2, (p - 1) // 2, (p + 1) // 2, (1 << f.payload) - 1, 1 << (f.payload - 1),
            (1 << 32) - 1, 1 << 32, (1 << (f.bits - 1)) - 1, 1 << (f.bits - 1)]


def check(exe: Path, f: W.Field, k: int, n: int, rng: random.Random) -> list[str]:
    key = W.key(f, SEED, k)
    i_, c1 = (W.SQRT_M1, W.C1) if not f.three_mod_four else (0, 0)
    c1i = c1 * i_ % f.p
    lines = [str(k), *(hexlimbs(c, f) for c in key.c), hexlimbs(key.tau, f), hexlimbs(i_, f), hexlimbs(c1, f),
             hexlimbs(c1i, f)]
    cases = []
    for j in range(n):
        m = edges(f)[j] if j < len(edges(f)) and edges(f)[j] < 1 << f.payload else rng.getrandbits(f.payload)
        y = edges(f)[j] if j < len(edges(f)) else rng.randrange(f.p)
        i = rng.choice([0, 1, (1 << 40) + j, rng.getrandbits(48), (1 << 64) - 1 - j])
        cases.append((i, m, y))
        lines.append(f"{i} {hexlimbs(m, f)} {hexlimbs(y, f)}")
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    bad = []
    for (i, m, y), line in zip(cases, out.splitlines(), strict=True):
        v = [int(x, 16) for x in line.split()]
        enc = sum(x << (32 * e) for e, x in enumerate(v[:f.limbs]))
        dec = sum(x << (32 * e) for e, x in enumerate(v[f.limbs:]))
        if enc != W.encode_word(m, i, key):
            bad.append(f"{f.name} k={k} encode m={m:#x} i={i}")
        if dec != W.decode_raw(y, i, key):
            bad.append(f"{f.name} k={k} decode y={y:#x} i={i}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", type=int, default=300)
    ap.add_argument("--fields", default="127,255,521")
    ap.add_argument("--ks", default="1,2,3,4")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    rng = random.Random(5)
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for bits in (int(b) for b in a.fields.split(",")):
            exe = build(bits, 4, Path(tmp))
            for k in ks:
                b = check(exe, W.field(bits), k, a.words, rng)
                print(f"{W.field(bits).name} k={k}: {'ok' if not b else f'{len(b)} FAIL'}", flush=True)
                bad += b
    for line in bad[:20]:
        print(line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
