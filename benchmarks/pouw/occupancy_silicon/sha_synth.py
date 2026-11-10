"""Synthesize P-C's SHA-512 engine (sha.v plus a generated K ROM) with SI-ALG's flow, check each step's netlists bit
for bit against references, and check the references themselves by hashing with them against hashlib.

    python3 sha_synth.py --alg DIR_OF_SILICON_ALG --out OUT [--vectors N]

DIR_OF_SILICON_ALG is SI-ALG's benchmarks/pouw/silicon_alg (branch cursor/compute-research-silicon-alg-8e03), whose
synth.py runs unchanged. The K ROM is generated into OUT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = (1 << 64) - 1


def icbrt(n: int) -> int:
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while x ** 3 > n:
        x -= 1
    while (x + 1) ** 3 <= n:
        x += 1
    return x


def isqrt_frac(p: int) -> int:
    from math import isqrt
    return isqrt(p << 128) & M


def primes(n: int) -> list[int]:
    out, c = [], 2
    while len(out) < n:
        if all(c % q for q in out if q * q <= c):
            out.append(c)
        c += 1
    return out


K = [icbrt(p << 192) & M for p in primes(80)]
H0 = [isqrt_frac(p) for p in primes(8)]
rotr = lambda x, r: ((x >> r) | (x << (64 - r))) & M


def rnd(s: list[int], w: int, k: int) -> list[int]:
    a, b, c, d, e, f, g, h = s
    t1 = (h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & M & g)) + k + w) & M
    t2 = ((rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & M
    return [(t1 + t2) & M, a, b, c, (d + t1) & M, e, f, g]


def sched(w: list[int]) -> list[int]:
    g0 = rotr(w[1], 1) ^ rotr(w[1], 8) ^ (w[1] >> 7)
    g1 = rotr(w[14], 19) ^ rotr(w[14], 61) ^ (w[14] >> 6)
    return w[1:] + [(g1 + w[9] + g0 + w[0]) & M]


def sha512(msg: bytes) -> bytes:
    ml = len(msg) * 8
    msg += b"\x80" + b"\x00" * ((112 - len(msg) - 1) % 128) + ml.to_bytes(16, "big")
    hh = list(H0)
    for o in range(0, len(msg), 128):
        w = [int.from_bytes(msg[o + 8 * i:o + 8 * i + 8], "big") for i in range(16)]
        s = list(hh)
        for t in range(80):
            s = rnd(s, w[0], K[t])
            w = sched(w)
        hh = [(x + y) & M for x, y in zip(hh, s)]
    return b"".join(x.to_bytes(8, "big") for x in hh)


def kv(out: Path) -> Path:
    p = out / "sha_k.v"
    lines = ["module sha512_k(input [6:0] t, output reg [63:0] k);", "  always @* case (t)"]
    lines += [f"    7'd{i}: k = 64'h{x:016x};" for i, x in enumerate(K)]
    lines += ["    default: k = 64'h0;", "  endcase", "endmodule", ""]
    p.write_text("\n".join(lines))
    return p


def pack(ws: list[int]) -> int:
    return sum(x << (64 * i) for i, x in enumerate(ws))


def designs() -> dict:
    names = "abcdefgh"
    d = {}
    d["sha512_round"] = dict(
        gen=lambda rng, n: {k: [rng.getrandbits(64) for _ in range(n)] for k in list(names) + ["w", "k"]},
        ref=lambda v, i: dict(zip([f"{c}_n" for c in names],
                                  rnd([v[c][i] for c in names], v["w"][i], v["k"][i]))),
        regs=512, role="round: a..h")
    d["sha512_sched"] = dict(
        gen=lambda rng, n: {"w": [rng.getrandbits(1024) for _ in range(n)]},
        ref=lambda v, i: {"w_n": pack(sched([(v["w"][i] >> (64 * j)) & M for j in range(16)]))},
        regs=1024, role="message schedule, 16 words")
    d["sha512_k"] = dict(
        gen=lambda rng, n: {"t": [rng.randrange(128) for _ in range(n)]},
        ref=lambda v, i: {"k": K[v["t"][i]] if v["t"][i] < 80 else 0}, regs=7, role="K ROM + round counter")
    d["sha512_fin"] = dict(
        gen=lambda rng, n: {k: [rng.getrandbits(512) for _ in range(n)] for k in ("hh", "s")},
        ref=lambda v, i: {"hh_n": pack([(((v["hh"][i] >> (64 * j)) + (v["s"][i] >> (64 * j))) & M)
                                        for j in range(8)])},
        regs=512, role="chaining value + final add")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", type=int, default=2048)
    a = ap.parse_args()
    rng = random.Random(0)
    for n in (0, 1, 111, 112, 127, 128, 129, 300, 1000):
        m = bytes(rng.getrandbits(8) for _ in range(n))
        assert sha512(m) == hashlib.sha512(m).digest(), n
    print("reference SHA-512 == hashlib on 9 lengths")
    sys.path.insert(0, a.alg)
    import synth
    out = Path(a.out).resolve()        # yosys runs in a temporary directory, so the generated ROM's path must be absolute
    out.mkdir(parents=True, exist_ok=True)
    synth.RTL = [HERE / "sha.v", kv(out)]
    rows = {}
    for top, spec in designs().items():
        with tempfile.TemporaryDirectory() as td:
            r = synth.run_yosys(top, Path(td))
        nets = r.pop("nets")
        vec = spec["gen"](random.Random(f"{top}:0"), a.vectors)
        want = [spec["ref"](vec, i) for i in range(a.vectors)]
        chk = {}
        for kind, net in nets.items():
            got = synth.simulate(net, top, vec, a.vectors)
            bad = [i for i in range(a.vectors) for k in want[i] if got[k][i] != want[i][k]]
            chk[kind] = {"vectors": a.vectors, "mismatches": len(bad)}
        r.update(check=chk, ge_reg=spec["regs"] * synth.DFF_GE, role=spec["role"])
        r["ge"] = r["ge_comb"] + r["ge_reg"]
        rows[top] = r
        ok = all(c["mismatches"] == 0 for c in chk.values())
        print(f"{top:13s} comb {r['ge_comb']:8.1f} reg {r['ge_reg']:7.1f} total {r['ge']:8.1f} GE  depth {r['depth']:3d}"
              f"  check {'ok' if ok else 'FAIL'}", flush=True)
    tot = sum(r["ge"] for r in rows.values())
    eng = 0.5885                       # engines an SM on the H200 (occ_bounds.py)
    res = {"engine_ge": tot, "per_SM_H200_ge": eng * tot, "pct_H200_SM_array": 100 * eng * tot / (4096 * 817.9)}
    # sm_120: 1024 s8 MACs an SM-clock; A bytes a MAC scale with the tensor rate, so engines an SM scale by 1/4
    res["per_SM_sm120_ge"] = eng / 4 * tot
    res["pct_sm120_SM_array"] = 100 * eng / 4 * tot / (1024 * 817.9)
    print(json.dumps(res, indent=1))
    (out / "sha_synth.json").write_text(json.dumps({"rows": rows, "summary": res}, indent=1, default=str))
    return 0 if all(c["mismatches"] == 0 for r in rows.values() for c in r["check"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
