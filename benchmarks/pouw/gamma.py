"""γ accounting for Pearl's FP8 scheme under W1 on the RTX 4090 (the problem statement's cost model; Derived).

    uv run python benchmarks/pouw/gamma.py [--json OUT] [--workload benchmarks/pouw/workloads.json]

For each GEMM shape it prices the honest reference term by term in W1 units (1 unit = one int8 tensor-core MAC
at peak), marks each term checked or unchecked (an unchecked term is work an adversary can skip without detection), and
derives:

- Ω: W_ref over the work the scheme credits (its FP8 GEMM);
- γ's floor from unchecked work: skippable/W_ref (the scheme names no γ_0, so no γ);
- hashing (free in γ, reported as overhead): hashed bytes per useful MAC;
- the overhead against the plain GEMM of the same datatype, in W1 units.

Prices (problem statement §3.1, measured on one RTX 4090): FP8 `mma` with FP32 accumulate 2 per MAC;
BF16 `mma` with FP32 accumulate 4 per MAC; INT32 (IADD3, LOP3, SHF, IMAD, PRMT) 16 per lane-word; FP32 (FADD, FFMA, FMNMX) 8.
Terms marked Unverified are instruction counts not yet read off a compiled kernel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

INT32, FP32 = 16, 8
MAC_FP8_F32, MAC_BF16_F32 = 2, 4
R = 32                      # Pearl's noise rank
SCHEME = "pearl-fp8-v4"
WORKLOAD = Path(__file__).with_name("workloads.json")


@dataclass(frozen=True)
class Term:
    name: str
    units: int
    checked: bool
    note: str = ""


def pearl_terms(m: int, k: int, n: int) -> list[Term]:
    """Pearl's honest miner per unit (``schemes/pearl_kw``), A side and outputs; the per-weight B side is amortized."""
    mk, mn = m * k, m * n
    return [
        Term("GEMM m x k x n, FP8 mma with FP32 accumulate", m * k * n * MAC_FP8_F32, True, "Pearl credits m·k·n MACs"),
        Term("noise N = E·Fᵀ, one rank-32 FP8 atom per A entry", mk * R * MAC_FP8_F32, True),
        Term("noisy quantize: β·N, fma, clamp, cvt to e4m3 (about 5 instructions per entry)", mk * 40, True, "Unverified count"),
        Term("FP10 prequant of A: block max, scale, int8 codes", mk * 24, True, "Unverified count"),
        Term("row norms l2, linf from the planes", mk * 2 * FP32, True, "Unverified count"),
        Term("extract: IMAD + SHF per output feeding the ticket", mn * 2 * INT32, True),
        Term("peel: A·F_B, BF16 mma (m·k·r)", mk * R * MAC_BF16_F32, False, "the useful output is unchecked"),
        Term("peel: 2r correction columns per output, BF16 mma", mn * 2 * R * MAC_BF16_F32, False, "unchecked"),
        Term("unscale: one FFMA per output", mn * FP32, False, "unchecked"),
    ]


def pearl_hash_bytes(m: int, k: int, n: int) -> int:
    """A's FP10 planes (1.25 bytes per entry) and one 64-byte extract per 256-output tile; B's planes per weight."""
    return m * k * 5 // 4 + 64 * (m * n // 256)


@dataclass(frozen=True)
class Row:
    scheme: str
    shape: tuple[int, int, int]
    wref: int
    credited: int
    omega: str
    gamma_floor: str
    overhead_vs_plain_gemm: str
    hashed_bytes_per_useful_mac: float
    terms: list[dict]


def pct(x: Fraction) -> str:
    return f"{float(x) * 100:.3f}%"


def row(m: int, k: int, n: int) -> Row:
    terms, credited, plain = pearl_terms(m, k, n), m * k * n * MAC_FP8_F32, m * k * n * MAC_FP8_F32
    wref = sum(t.units for t in terms)
    skippable = sum(t.units for t in terms if not t.checked)
    omega = Fraction(wref, credited)
    return Row(SCHEME, (m, k, n), wref, credited, f"{float(omega):.5f}", pct(Fraction(skippable, wref)), f"{wref / plain:.4f}",
               pearl_hash_bytes(m, k, n) / (m * k * n), [asdict(t) for t in terms])


def uncovered_fraction(model: dict) -> float:
    """The share of a decode token's GEMM MACs that the LinearBase hook does not cover (lm_head)."""
    per_layer = sum(k * n for k, n in model["linears"].values())
    head = model["hidden"] * model["vocab"]
    return head / (per_layer * model["layers"] + head)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", type=Path, default=WORKLOAD)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    work = json.loads(args.workload.read_text())
    shapes = [tuple(s) for s in work["gamma_shapes"]]
    rows = [row(*shape) for shape in shapes]
    model = work["model"]
    out = {"schema": "verity-pouw/gamma/v2", "cost_model": "W1 (RTX 4090), sub-word extraction priced", "rows": [asdict(r) for r in rows],
           "model": model["id"], "uncovered_lm_head_fraction": uncovered_fraction(model)}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    print(f"{'scheme':14} {'(m, k, n)':22} {'Ω':>8} {'γ floor':>9} {'W1 vs plain':>11} {'hash B/MAC':>10}")
    for r in rows:
        print(f"{r.scheme:14} {str(r.shape):22} {r.omega:>8} {r.gamma_floor:>9} {r.overhead_vs_plain_gemm:>11} "
              f"{r.hashed_bytes_per_useful_mac:10.4f}")
    print(f"{model['id']}: lm_head is {out['uncovered_lm_head_fraction']:.1%} of a decode token's GEMM MACs (uncovered)")


if __name__ == "__main__":
    main()
