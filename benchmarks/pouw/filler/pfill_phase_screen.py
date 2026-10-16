"""HC-PROVED's screens of pfill-k's phase filler (psched_probe.sh's PROBES, ';'-separated `pscore pscore` rows):
`python3 benchmarks/pouw/filler/pfill_phase_screen.py NAME` prints screen NAME's probes. A row is (B, L, split SMs, phase S,
phase tiles a block a layer); the split pumps pfill-k's arm filler (W 8, NC 4, S 256, 2^19), the phase filler is 8-warp
blocks of NC 4 ram tiles. Build with NVCC_APPEND_FLAGS=-DPFILL_BITS.
"""

import sys

ROUNDS, K = 2, 3

SCREENS = {
    # S 32 tiles sized from the leader's TABLE_V3 phase work at pfill-k's rate, a size below and above, at the leader's
    # split and pfill-power-split's where they differ
    "s1": [
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
    ],
}


def probes(rows):
    return ";".join(f"DFILL_CHUNK_LOG2=19 pscore pscore {b} {l} {f} 8 4 256 8 4 {s} {t} {ROUNDS} {K} {i % 64}"
                    for i, (b, l, f, s, t) in enumerate(rows))


if __name__ == "__main__":
    print(probes(SCREENS[sys.argv[1]]))
