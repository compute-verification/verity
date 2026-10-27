"""The red team's P3 / band / dense `Π₂` state attack, on this tree's band and
dense codecs: any whole label `C_v` of the overwrite-chain schemes, rebuilt from five 200-byte post-absorb SHAKE256
states of the label's final (pad) `Π₂` call, about 1,000 bytes a 64 KiB block, in 481 sequential Keccak-f, with no
`Π₂` call and no chain walk.

The deployed reference `Π₂` is Feistel10 on `2m + 512` bits with `F_i(x) = SHAKE256(0x48 ‖ u8(i) ‖ x)[0:half]`,
`half = m/8 + 32` bytes (`primitives.FeistelPi`). The key is the first `m` bits of the pad call's output. For a Feistel
network from `(L_0, R_0)`,

    R_9 = L_0 ^ F_1(R_0) ^ F_3(R_2) ^ F_5(R_4) ^ F_7(R_6) ^ F_9(R_8),   key_v = first m/8 bytes of R_9.

The first `m/8` bytes of `L_0` are the pad block. So `key_v = pad ^` five squeeze streams, each the SHAKE256 output of
the post-absorb state reached after absorbing `0x48 ‖ u8(2i−1) ‖ R_{2i−2}`. A prover keeps those five states per block
and, at a read, squeezes them in parallel to `m/8` bytes: `⌈(m/8)/136⌉ − 1` sequential Keccak-f.

Ground truth is this tree's reference encoder (`band.reference`, `dense.encode_segment`); every rebuilt block is
compared bit for bit with it. The code is the red team's but for the imports, the parameters and the record.

    python pi2_state_attack.py [--blocks B] [--m BITS] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from verity.core.protocols.pous import band as band_codec  # noqa: E402
from verity.core.protocols.pous import dense as dense_codec  # noqa: E402
from verity.core.protocols.pous.labels import salt_value, tag  # noqa: E402
from verity.core.protocols.pous.primitives import (CAPACITY_BITS, KECCAK_F_NS, PI_DOMAIN, SHAKE256_RATE,  # noqa: E402
                                                   pad_block, pi2_keccak_f)

RATE = SHAKE256_RATE
MASKW = (1 << 64) - 1
RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
      0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
      0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
      0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
STATE_BYTES = 200


def rol(x, n):
    n %= 64
    return ((x << n) | (x >> (64 - n))) & MASKW if n else x


def keccak_f(a):
    for rc in RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ rol(c[(x + 1) % 5], 1) for x in range(5)]
        a = [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = rol(a[x][y], ROT[x][y])
        a = [[b[x][y] ^ (~b[(x + 1) % 5][y] & b[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        a[0][0] ^= rc
    return a


def absorb(msg):
    """SHAKE256 absorb to the post-absorb state (squeeze block 0)."""
    m = bytearray(msg) + b"\x1f"
    while len(m) % RATE:
        m += b"\x00"
    m[-1] |= 0x80
    a = [[0] * 5 for _ in range(5)]
    for off in range(0, len(m), RATE):
        blk = m[off:off + RATE]
        for k in range(RATE // 8):
            a[k % 5][k // 5] ^= int.from_bytes(blk[8 * k:8 * k + 8], "little")
        a = keccak_f(a)
    return a


def rate_bytes(a):
    return b"".join(a[k % 5][k // 5].to_bytes(8, "little") for k in range(RATE // 8))


def squeeze(a, nbytes):
    """Squeeze nbytes from post-absorb state a; returns (bytes, sequential Keccak-f used)."""
    out = bytearray()
    seq = 0
    while len(out) < nbytes:
        out += rate_bytes(a)
        if len(out) < nbytes:
            a = keccak_f(a)
            seq += 1
    return bytes(out[:nbytes]), seq


def xorb(*bs):
    out = bytearray(len(bs[0]))
    for b in bs:
        for i in range(len(b)):
            out[i] ^= b[i]
    return bytes(out)


def pad_call_chaining_value(params, pi2, t, v, blocks):
    """h going into block v's pad call: the overwrite chain over v's parents (newest first)."""
    m = params.m
    h = dense_codec.dom(t, v) << m
    for u in sorted(params.parents(v), reverse=True):
        x = blocks[u] | (h << m)
        out = pi2.forward(x)
        h = out >> m
    return h


def attack(name, params, prims, salt, seg, w_seg, encode_segment):
    """Encode one segment with the reference, then rebuild every block from five stored states; the record."""
    m = params.m
    mb = m // 8
    half = (2 * m + CAPACITY_BITS) // 16
    nbytes = (2 * m + CAPACITY_BITS) // 8
    pi2 = prims.h.pi2
    t = tag(salt_value(salt), seg)
    honest = encode_segment(params, prims, salt, seg, w_seg)
    w = [int.from_bytes(w_seg[i * mb:(i + 1) * mb], "little") for i in range(params.blocks_per_segment)]
    pad = pad_block(m).to_bytes(mb, "little")

    worst_seq = 0
    rebuilt = 0
    for v in range(params.blocks_per_segment):
        c_v = int.from_bytes(honest[v], "little")
        key_v_honest = c_v ^ w[v]
        h = pad_call_chaining_value(params, pi2, t, v, [int.from_bytes(b, "little") for b in honest])
        x_pad = pad_block(m) | (h << m)
        xb = x_pad.to_bytes(nbytes, "little")
        L, R = xb[:half], xb[half:]
        sigma_odd = {}
        for i in range(1, 11):
            pre = absorb(bytes([PI_DOMAIN, i]) + R)
            fi, _ = squeeze(pre, half)
            if i % 2 == 1:
                sigma_odd[i] = pre
            L, R = R, xorb(L, fi)
        assert int.from_bytes(L + R, "little") == pi2.forward(x_pad), f"{name}: my Feistel != the reference Pi2 at {v}"
        assert int.from_bytes((L + R)[:mb], "little") == key_v_honest, f"{name}: Feistel key != honest key at {v}"

        streams, seqs = [], []
        for i in (1, 3, 5, 7, 9):
            s, seq = squeeze(sigma_odd[i], mb)
            streams.append(s)
            seqs.append(seq)
        c_rebuilt = int.from_bytes(xorb(pad, *streams), "little") ^ w[v]
        assert c_rebuilt == c_v, f"{name}: rebuilt block != honest block at block {v}"
        rebuilt += 1
        worst_seq = max(worst_seq, max(seqs))

    full = pi2_keccak_f(m)
    kept = 5 * STATE_BYTES
    rec = {"scheme": name, "blocks": params.blocks_per_segment, "block_bytes": mb, "d": getattr(params, "d", None),
           "rebuilt_bit_exact": rebuilt, "kept_bytes_per_block": kept, "kept_share": kept / mb,
           "answer_sequential_keccak_f": worst_seq, "pi2_call_keccak_f": full,
           "answer_ns_at_keccak_f_ns": worst_seq * KECCAK_F_NS, "pi2_call_ns": full * KECCAK_F_NS,
           "d5_deadline_ns": 6 * full * KECCAK_F_NS}
    print(f"[{name}] B={params.blocks_per_segment} blocks of {mb} B: {rebuilt} rebuilt bit-exactly from {kept} B each "
          f"({kept / mb:.4%}); {worst_seq} sequential Keccak-f an answer vs {full} a Pi2 call; at {KECCAK_F_NS} ns "
          f"{worst_seq * KECCAK_F_NS / 1000:.2f} us vs the D=5 deadline {6 * full * KECCAK_F_NS / 1000:.1f} us")
    return rec


def cpu_answer_ns(m: int, reps: int = 200) -> dict:
    """An answer's work on this CPU in OpenSSL's Keccak: five SHAKE256 outputs of `m/8` bytes, one after another, and
    their XOR. `shake_256(x).digest(m/8)` for a short `x` is one absorb and the attack's `⌈(m/8)/136⌉ − 1` squeeze
    permutations, so it costs what resuming one stored state does; the median and the slowest of `reps`."""
    import hashlib
    import time

    mb = m // 8
    seeds = [bytes([PI_DOMAIN, i]) + bytes(32) for i in (1, 3, 5, 7, 9)]
    ts = []
    for _ in range(reps):
        a = time.perf_counter_ns()
        acc = 0
        for s in seeds:
            acc ^= int.from_bytes(hashlib.shake_256(s).digest(mb), "little")
        acc.to_bytes(mb, "little")
        ts.append(time.perf_counter_ns() - a)
    ts.sort()
    return {"block_bytes": mb, "reps": reps, "p50_ns": ts[len(ts) // 2], "max_ns": ts[-1],
            "d5_deadline_ns": 6 * pi2_keccak_f(m) * KECCAK_F_NS - 1}


def run(blocks: int, m: int, d: int = 12) -> list[dict]:
    salt = bytes(range(24))
    bp = band_codec.BandParams(blocks, m, d)
    w = bytes((i * 2654435761) & 0xFF for i in range(bp.segment_bytes))
    band = attack(f"band d={d}", bp, band_codec.reference(bp), salt, 0, w, band_codec.dense.encode_segment)
    dp = dense_codec.DenseParams(blocks, m)
    w2 = bytes((i * 40503) & 0xFF for i in range(dp.segment_bytes))
    dense = attack("dense", dp, dense_codec.reference(dp), salt, 0, w2, dense_codec.encode_segment)
    return [band, dense]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--m", type=int, default=524_288, help="label bits (default 524,288: the deployed 64 KiB)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    recs = run(args.blocks, args.m)
    cpu = cpu_answer_ns(args.m)
    print(f"[cpu] an answer's five {cpu['block_bytes']}-byte SHAKE256 streams and their XOR: p50 {cpu['p50_ns'] / 1000:.1f} us, "
          f"max {cpu['max_ns'] / 1000:.1f} us of {cpu['reps']}, vs the D=5 deadline {cpu['d5_deadline_ns'] / 1000:.1f} us")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pi2_state_attack.json").write_text(json.dumps({"blocks": recs, "cpu_answer": cpu}, indent=1) + "\n")


if __name__ == "__main__":
    main()
