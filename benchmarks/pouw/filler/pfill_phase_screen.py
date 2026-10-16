"""HC-PROVED's screens of pfill-k's phase filler (psched_probe.sh's PROBES, ';'-separated `pscore pscore` rows):
`python3 benchmarks/pouw/filler/pfill_phase_screen.py NAME` prints screen NAME's probes. A row is (B, L, split SMs, phase
warps, phase S, phase tiles a block a layer); the split pumps pfill-k's arm filler (W 8, NC 4, S 256, 2^19), the phase
filler is blocks of NC 4 ram tiles of that many warps, which the build fixes: NVCC_APPEND_FLAGS="-DPFILL_BITS -DPF_PW=W"
(one width a screen). A tile does 16 W^2 S int8 MMAs of 4096 MACs.
"""

import sys

ROUNDS, K = 2, 3


def _at(w, rows):
    return [(b, l, f, w, s, t) for b, l, f, s, t in rows]


SCREENS = {
    # S 32 tiles sized from the leader's TABLE_V3 phase work at pfill-k's rate, a size below and above, at the leader's
    # split and pfill-power-split's where they differ (r20261016-083237-b004: every row 1.076-1.302)
    "s1": _at(8, [
        (32, 1024, 92, 8, 1), (32, 1024, 92, 16, 1),
        (32, 2048, 92, 16, 1), (32, 2048, 92, 32, 1), (32, 2048, 108, 16, 1),
        (32, 8192, 108, 32, 2), (32, 8192, 108, 32, 3), (32, 8192, 124, 32, 2),
        (32, 24576, 108, 32, 8), (32, 24576, 108, 32, 11), (32, 24576, 132, 32, 8),
        (48, 16384, 92, 32, 8), (48, 16384, 92, 32, 11), (48, 16384, 132, 32, 8),
        (128, 2048, 92, 32, 2), (128, 2048, 92, 32, 3), (128, 2048, 92, 32, 4),
        (128, 6144, 92, 32, 8), (128, 6144, 92, 32, 11), (128, 6144, 108, 32, 8),
        (132, 5120, 60, 32, 7), (132, 5120, 60, 32, 10),
        (144, 5120, 60, 32, 7), (144, 5120, 60, 32, 10),
        (256, 1024, 60, 32, 2), (256, 1024, 60, 32, 3),
        (256, 2048, 60, 32, 5), (256, 2048, 60, 32, 7), (256, 2048, 60, 32, 9),
        (256, 3072, 60, 32, 8), (256, 3072, 60, 32, 11),
    ]),
    # the narrow tile (-DPF_PW=4, 16K registers and 24 KB a block): s1's lowest phase work a layer (and twice it), and
    # a quarter of it at pfill-k's own split at 32/8K and 256/2K
    "s2": _at(4, [
        (32, 1024, 92, 32, 1), (32, 2048, 92, 64, 1),
        (32, 8192, 124, 64, 1), (32, 8192, 108, 64, 2), (32, 8192, 108, 64, 4),
        (32, 24576, 108, 64, 8), (48, 16384, 92, 64, 8),
        (128, 2048, 92, 64, 2), (128, 2048, 92, 64, 4), (128, 6144, 92, 64, 8),
        (132, 5120, 60, 64, 7), (144, 5120, 60, 64, 7),
        (256, 1024, 60, 64, 2), (256, 2048, 60, 32, 2), (256, 2048, 60, 64, 5), (256, 2048, 60, 64, 10),
        (256, 3072, 60, 64, 8),
    ]),
    # the narrower tile (-DPF_PW=2, 8K registers and 12 KB a block, so pv_dec keeps its 3 blocks an SM): s2's work a layer
    "s3": _at(2, [
        (32, 1024, 92, 128, 1), (32, 2048, 92, 128, 2),
        (32, 8192, 108, 128, 4), (32, 8192, 108, 128, 8),
        (32, 24576, 108, 128, 16), (48, 16384, 92, 128, 16),
        (128, 2048, 92, 128, 4), (128, 2048, 92, 128, 8), (128, 6144, 92, 128, 16),
        (132, 5120, 60, 128, 14), (144, 5120, 60, 128, 14),
        (256, 1024, 60, 128, 4), (256, 2048, 60, 128, 10), (256, 2048, 60, 128, 20),
        (256, 3072, 60, 128, 16),
    ]),
}


def probes(rows):
    return ";".join(f"DFILL_CHUNK_LOG2=19 pscore pscore {b} {l} {f} 8 4 256 {w} 4 {s} {t} {ROUNDS} {K} {i % 64}"
                    for i, (b, l, f, w, s, t) in enumerate(rows))


if __name__ == "__main__":
    print(probes(SCREENS[sys.argv[1]]))
