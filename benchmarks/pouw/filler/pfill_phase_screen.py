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
    # (r20261016-085255-cdd2: every row 1.12-2.0, its quarter-peak blocks outlast attention; s2 at PF_PW 4 1.019-1.088
    # below its cliff)
    "s3": _at(2, [
        (32, 1024, 92, 128, 1), (32, 2048, 92, 128, 2),
        (32, 8192, 108, 128, 4), (32, 8192, 108, 128, 8),
        (32, 24576, 108, 128, 16), (48, 16384, 92, 128, 16),
        (128, 2048, 92, 128, 4), (128, 2048, 92, 128, 8), (128, 6144, 92, 128, 16),
        (132, 5120, 60, 128, 14), (144, 5120, 60, 128, 14),
        (256, 1024, 60, 128, 4), (256, 2048, 60, 128, 10), (256, 2048, 60, 128, 20),
        (256, 3072, 60, 128, 16),
    ]),
    # PF_PW 4 toward ~70% of attention's layer time (B L 2 KB of KV at ~1.6 TB/s; a S 64 tile ~61 µs alone) where s2 had
    # room, lighter splits with a phase where pfill-k's split alone has little margin; W 0 rows are the split alone (dscore)
    "s4": [
        (1, 32768, 100, 0, 0, 0), (1, 32768, 92, 0, 0, 0),
        (32, 8192, 116, 4, 64, 2), (32, 8192, 108, 4, 64, 3), (32, 8192, 116, 0, 0, 0),
        (32, 24576, 124, 4, 64, 6), (32, 24576, 132, 4, 64, 4),
        (48, 16384, 108, 4, 64, 8), (48, 16384, 116, 4, 64, 8), (48, 16384, 124, 4, 64, 6),
        (128, 2048, 76, 4, 64, 2), (128, 2048, 76, 4, 64, 3),
        (128, 6144, 92, 4, 64, 11), (128, 6144, 92, 4, 64, 13), (128, 6144, 108, 4, 64, 8),
        (132, 5120, 60, 4, 64, 9), (132, 5120, 60, 4, 64, 10),
        (144, 5120, 60, 4, 64, 9), (144, 5120, 60, 4, 64, 11),
        (256, 1024, 44, 4, 64, 2), (256, 1024, 52, 4, 64, 2),
        (256, 2048, 60, 4, 64, 6), (256, 2048, 60, 4, 64, 7),
        (256, 3072, 60, 4, 64, 10), (256, 3072, 60, 4, 64, 12),
    ],
    # PF_PW 4 at 80 registers (-DPF_PMIN=6, 10K registers a block, so pv_dec keeps its 3 blocks an SM; ptxas: the MMA loop
    # spill-free, one 4-byte reload a tile): pfill-k-phase's rows and a tile past each, against s4's cliffs
    "s5": _at(4, [
        (32, 8192, 108, 64, 2), (32, 8192, 108, 64, 3), (32, 24576, 108, 64, 8), (32, 24576, 124, 64, 6),
        (48, 16384, 108, 64, 8), (48, 16384, 116, 64, 8),
        (128, 2048, 76, 64, 3), (128, 2048, 92, 64, 2), (128, 6144, 92, 64, 10), (128, 6144, 92, 64, 12),
        (132, 5120, 60, 64, 7), (132, 5120, 60, 64, 9), (144, 5120, 60, 64, 9), (144, 5120, 60, 64, 10),
        (256, 1024, 60, 64, 2), (256, 2048, 60, 64, 6), (256, 2048, 60, 64, 7),
        (256, 3072, 60, 64, 10), (256, 3072, 60, 64, 11),
    ]),
}


def probes(rows):
    def row(i, b, l, f, w, s, t):
        if w == 0:
            return f"DFILL_CHUNK_LOG2=19 dscore dscore step {b} {l} split {f} 8 4 256 {ROUNDS} {K} {i % 64}"
        return f"DFILL_CHUNK_LOG2=19 pscore pscore {b} {l} {f} 8 4 256 {w} 4 {s} {t} {ROUNDS} {K} {i % 64}"
    return ";".join(row(i, *r) for i, r in enumerate(rows))


if __name__ == "__main__":
    print(probes(SCREENS[sys.argv[1]]))
