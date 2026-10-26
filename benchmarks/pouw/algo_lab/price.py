"""DS-90, the H200's data-sheet machine, as prices a warp-word (32 committed words, one a lane), in SMSP-cycles.

A class's warp instruction costs what `machine_model/data/devices/sm_90/census/DS-90.csv` charges its census kind: 1 on
issue for every kind; 2 on the ALU pipe (ALU), 2 on the IMAD pipe (IMAD), 4 on SHFL, 1.961 on the uniform datapath (UNI);
an MMA its DS_tensor cycles over its MACs. The CSV is read, not copied, so the lab moves with the census.

Two prices DS-90 doesn't have are stated here:

- **Shared memory**, which DS-90 has no row for: 128 bytes an SM-clock (the CUDA Programming Guide's 32 banks of 32 bits
  a clock), 32 bytes an SMSP-cycle when the SM's four SMSPs share it. A wgmma reads its operands from it by descriptor.
- **IMAD.WIDE** is UNLISTED in DS-90 (issue only); it is priced here as an IMAD, its pipe on the hardware. **POPC** and
  **LEA.HI** go on ALU's row, with LOP3, IADD3, SHF, LEA and PRMT (`theory/fold_sass_count.py`'s rule). b1's BGMMA has no
  DS_tensor row, so a b1 MAC costs only its issue: a gap in DS-90 (O-REDTEAM's finding 1), flagged where it matters.
"""
from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

CENSUS = Path(__file__).resolve().parents[1] / "machine_model/data/devices/sm_90/census/DS-90.csv"
LANES = 32
SMEM_BYTES_A_CYCLE = 32
RESOURCES = ("tensor", "issue", "ALU", "IMAD", "SHFL", "uniform", "smem")

#: a lab class -> its DS-90 kind; MMAs carry their MACs a warp instruction (a quarter of the warpgroup's wgmma)
KIND = {"LOP3": "ALU", "IADD3": "ALU", "SHF": "ALU", "LEA": "ALU", "LEA.HI": "ALU", "PRMT": "ALU", "POPC": "ALU",
        "IMAD": "IMAD", "IMAD.HI": "IMAD", "IMAD.WIDE": "IMAD", "IDP4A": "IMAD", "FFMA": "FFMA", "F2I": "UNLISTED",
        "SHFL": "SHFL", "REDUX": "REDUX", "REDG": "REDG32", "UNI": "UNI", "LDS": "LDS128", "STS": "STS32", "STG": "STG32",
        "MMA.S8": "IGMMA_S8_N256", "MMA.S8.K16": "IMMA16816_S8", "MMA.BF16": "HGMMA_BF16_N256",
        "MMA.B1": "BGMMA_B1_N256"}
MMA_MACS = {"MMA.S8": 64 * 256 * 32 // 4, "MMA.S8.K16": 16 * 8 * 16, "MMA.BF16": 64 * 256 * 16 // 4,
            "MMA.B1": 64 * 256 * 256 // 4}
ROW = {"DS_issue": "issue", "DS_alu": "ALU", "DS_imad": "IMAD", "DS_shfl": "SHFL", "DS_uniform": "uniform",
       "DS_tensor": "tensor"}
#: classes counted a warp, not a lane
WARP_COUNTED = {"UNI"}
UNCOSTED = {"GMEM.B"}


def census() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with CENSUS.open() as f:
        for row in csv.DictReader(f):
            out.setdefault(row["kind"], {})[ROW[row["row"]]] = float(row["cycles"])
    return out


DS90 = census()


def loads(counts: dict[str, int], words: int) -> dict[str, float]:
    """SMSP-cycles a warp-word on each resource, of a program whose class counts are ``counts`` (lanes; MACs for an MMA;
    bytes for SMEM.B) over ``words`` committed words."""
    ww = words / LANES
    out = dict.fromkeys(RESOURCES, 0.0)
    for cls, n in counts.items():
        if cls in UNCOSTED:
            continue
        if cls == "SMEM.B":
            out["smem"] += n / SMEM_BYTES_A_CYCLE / ww
            continue
        instr = n / MMA_MACS[cls] if cls in MMA_MACS else n if cls in WARP_COUNTED else n / LANES
        for res, cyc in DS90[KIND[cls]].items():
            out[res] += instr * cyc / ww
    return out


def time(load: dict[str, float]) -> tuple[float, str]:
    r = max(load, key=load.get)
    return load[r], r


def relative(load: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    """Each resource over the reference's, and ``time`` over the reference's time."""
    out = {r: (load[r] / ref[r] if ref[r] else (float("inf") if load[r] else 1.0)) for r in RESOURCES}
    out["time"] = time(load)[0] / time(ref)[0]
    return out


def mix2(a: dict[str, float], b: dict[str, float]) -> tuple[Fraction, float]:
    """The share x of route b, mixed per word with route a, that minimises max_r ((1 - x) a_r + x b_r), and that max: the
    exact optimum of the convex piecewise-linear function, at x in {0, 1} or where two resources cross."""
    fa = {r: Fraction(a[r]).limit_denominator(1 << 20) for r in RESOURCES}
    fb = {r: Fraction(b[r]).limit_denominator(1 << 20) for r in RESOURCES}
    cands = {Fraction(0), Fraction(1)}
    for r in RESOURCES:
        for s in RESOURCES:
            den = (fb[r] - fa[r]) - (fb[s] - fa[s])
            if den:
                x = (fa[s] - fa[r]) / den
                if 0 <= x <= 1:
                    cands.add(x)
    best = min(cands, key=lambda x: max((1 - x) * fa[r] + x * fb[r] for r in RESOURCES))
    return best, float(max((1 - best) * fa[r] + best * fb[r] for r in RESOURCES))
