"""fold_cp.cu's CPU reference, checked before any device run: its per-set checkpoint is CP-W's `warp_checkpoint`
(`segment.warp_checkpoint`: per part of a state, y then z, the XOR over the warp's lanes and chains), and the device's
route (each thread XORs its K lanes' 8 chains in registers, the 8 threads of a logical warp combine on the memory
system by red.global.xor into a zeroed pair) gives the same two words. The fold is fold_cp.cu's (and fold_bulk.cu's
`ref`): mode 10's fx1 absorb, chain 4 (t % 2) + e. Exit 0 when every set's checkpoint agrees three ways.

    python3 benchmarks/pouw/rt4_sheet/fold_cp_ref.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verity.core.protocols.pouw.circuit.segment import NCH, warp_checkpoint  # noqa: E402

M32 = 0xFFFFFFFF
K = 4                       # logical lanes a thread (fold_cp.cu's K)


def mix32(x: int) -> int:
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def KA(t: int) -> int:
    return mix32(0xA0000000 + 2 * t) | 1


def KK(t: int) -> int:
    return mix32(0xB0000000 + 2 * t + 1)


def ROT(t: int) -> int:
    return ((t * 7 + 5) % 31) + 1


def FK(q: int) -> int:
    return mix32(0xA0000000 + 2 * (16 + 32 * q)) | 1


def rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & M32


def absorb_set(S: list[int], w: list[int], s: int) -> None:
    """One set of fold_cp.cu's `ref` on a lane's 16-word state (y then z), in place."""
    y, z = S[:NCH], S[NCH:]
    for t in range(8):
        for e in range(4):
            c = 4 * (t % 2) + e
            p = (w[4 * t + e] + 32 * (s + 1)) & M32
            u = y[c] ^ ((p * FK(4 * t + e)) & M32) ^ KK(t)
            y[c] = (u * KA(t) + z[c]) & M32
            z[c] = (rotl(u, ROT(t)) + y[c]) & M32
    S[:NCH], S[NCH:] = y, z


def main() -> int:
    rng = random.Random(0xC0B1)
    lanes, sets = 128, 16                       # 4 logical warps, 16 sets
    st = [[rng.getrandbits(32) for _ in range(2 * NCH)] for _ in range(lanes)]
    W = [[rng.getrandbits(32) for _ in range(32)] for _ in range(lanes)]
    bad = 0
    for s in range(sets):
        for ln in range(lanes):
            absorb_set(st[ln], W[ln], s)
        for w in range(lanes // 32):
            warp = st[32 * w:32 * (w + 1)]
            honest = warp_checkpoint(warp)                                         # CP-W's definition
            kernel = [0, 0]                                                        # fold_cp.cu's ref formula
            for lane in warp:
                for j in range(NCH):
                    kernel[0] ^= lane[j]
                    kernel[1] ^= lane[NCH + j]
            device = [0, 0]                                                        # 8 threads' partials, REDG-merged
            for th in range(32 // K):
                py = pz = 0
                for lane in warp[K * th:K * (th + 1)]:
                    for j in range(NCH):
                        py ^= lane[j]
                        pz ^= lane[NCH + j]
                device[0] ^= py
                device[1] ^= pz
            bad += not (honest == kernel == device)
    print(f"fold_cp_ref: {lanes} lanes, {sets} sets, {lanes // 32} logical warps a set: "
          f"warp_checkpoint == ref formula == per-thread REDG merge on every set: {bad == 0} (bad {bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
