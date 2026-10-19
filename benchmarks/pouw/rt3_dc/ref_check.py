"""Recompute the exact mode's reference through segment.py, so each atomic form's zero mismatches are against li itself.

sm90_cpatom.cu's and sm100_cpatom.cu's host reference and their kernels share one C++ transcription of li (`li_set`,
cpatom.cuh), so a kernel matching it says only that the device computes that transcription. This regenerates the run's
inputs (the same LCG and seed, the same layout: unit u, lane l, set s's 32 words at ((u·32 + l)·128 + 32 s)), absorbs
each lane's sets with `segment._li_absorb`, takes `segment.warp_checkpoint` after every 128-word segment (cp_seg) and
each lane's final 8 states (final), and compares them with the `ref` line the run printed (RT2-HOPPER's ref_check.py,
for the one placement every atomic form computes).

Run: `uv run python benchmarks/pouw/rt3_dc/ref_check.py OUT/exact.jsonl`. Exit 0 every word agrees and every kernel
line reads 0 mismatches, 1 otherwise.
"""
from __future__ import annotations

import json
import sys

from verity.core.protocols.pouw.circuit.segment import _li_absorb, warp_checkpoint

M64 = (1 << 64) - 1


def lcg(s: int) -> tuple[int, int]:
    s = (s * 6364136223846793005 + 1442695040888963407) & M64
    return s, s >> 32


def main() -> int:
    lines = [json.loads(x) for x in open(sys.argv[1]) if x.startswith("{")]
    ref = next(x for x in lines if x["probe"] == "ref")
    U, s = ref["units"], ref["seed"]
    w = []
    for _ in range(U * 32 * 128):
        s, v = lcg(s)
        w.append(v)
    st = []
    for _ in range(256):
        s, v = lcg(s)
        st.append(v)
    states = [st[8 * lane:8 * lane + 8] for lane in range(32)]
    cp_seg = []
    for u in range(U):
        for k in range(4):
            for lane in range(32):
                base = (u * 32 + lane) * 128 + 32 * k
                states[lane] = _li_absorb(states[lane], w[base:base + 32])
        cp_seg.append(warp_checkpoint(states)[0])
    final = [v for st8 in states for v in st8]
    bad = 0
    for name, mine in (("cp_seg", cp_seg), ("final", final)):
        diff = sum(a != b for a, b in zip(mine, ref[name])) + abs(len(mine) - len(ref[name]))
        print(f"{name}: {len(mine)} words, {diff} differ from the run's reference")
        bad += diff
    kernels = [x for x in lines if x["probe"] == "exact"]
    for x in kernels:
        print(f"{x['kernel']}: {x['mismatches']} mismatches over {x['words_checked']} words ({x['cuda']})")
        bad += x["mismatches"] != 0 or x["cuda"] != "no error"
    bad += not kernels
    print("REF", "PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
