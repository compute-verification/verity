"""γ accounting for the PoUW schemes under W1 on the RTX 4090 (the problem statement's cost model; Derived).

    uv run python benchmarks/pouw/gamma.py [--json OUT] [--workload benchmarks/pouw/workloads.json]

For each scheme and GEMM shape it prices the honest reference term by term in W1 units (1 unit = one int8 tensor-core MAC
at peak), marks each term checked or unchecked (an unchecked term is work an adversary can skip without detection), and
derives:

- Ω: W_ref over the work the scheme's conjecture credits (NCP: the 3k-deep GEMM, 3·m·k·n; Pearl: its FP8 GEMM);
- γ where the scheme names γ_0: γ = 1 − (1 − γ_0)/Ω (Lean `gammaFromTTNCP`), the conservative reduction;
- γ's floor from unchecked work: skippable/W_ref, whatever the conjecture;
- the hashed bytes per useful MAC: the binding, every value the scheme checks;
- the overhead against the plain GEMM of the same datatype, in W1 units, once per binding: none (arithmetic only) and each
  hash in `HASHES`.  The headline is the total at the scheme's pinned hash (`PINNED`).

Hash calls are free in γ (problem statement §3.1: an adversary with free hashing is only stronger), but the honest prover
pays for them, so every total includes them.  γ holds only where every checked value is bound: a row without a binding
carries no γ ("—").

Prices (problem statement §3.1, measured on one RTX 4090): int8 `mma` 1 per MAC; FP8 `mma` with FP32 accumulate 2 per MAC;
BF16 `mma` with FP32 accumulate 4 per MAC; INT32 (IADD3, LOP3, SHF, IMAD, PRMT) 16 per lane-word; FP32 (FADD, FFMA, FMNMX) 8;
sub-word extraction 16 per word produced; hashes per byte as in `HASHES`.
Terms marked Unverified are instruction counts not yet read off a compiled kernel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

INT32, FP32, SUBWORD = 16, 8, 16
MAC_INT8, MAC_FP8_F32, MAC_BF16_F32 = 1, 2, 4
D = 16                      # NCP's checked depth
R = 32                      # Pearl's noise rank
GAMMA0_NCP = Fraction(1, 200)
WORKLOAD = Path(__file__).with_name("workloads.json")


@dataclass(frozen=True)
class Hash:
    units_per_byte: float   # W1 units on the RTX 4090
    gbps: float             # device throughput the timed scripts add as serial time
    status: str             # "measured" or "estimated"
    source: str


SHAKE256_UNITS, SHAKE256_GBPS = 525.0, 650.0


def _scaled(units: float) -> float:
    """An estimated hash's throughput: SHAKE256's measured rate scaled by the price ratio (all run on the INT32 pipe)."""
    return SHAKE256_GBPS * SHAKE256_UNITS / units


HASHES = {
    "shake256": Hash(SHAKE256_UNITS, SHAKE256_GBPS, "measured",
                     "521–548 units/B and 626–696 GB/s on one RTX 4090 (research store gpu-constants.md, runs "
                     "r20260927-202355-71cd and -201908-1942); hashcat's Keccak-256 agrees"),
    "sha256": Hash(340.0, _scaled(340.0), "estimated",
                   "1,404 INT32 instructions per 64-B block (the H100 build's count, Round 11) at 16 units; hashcat's "
                   "single-block 1.4 TB/s is an optimistic bound"),
    "turboshake128": Hash(205.0, _scaled(205.0), "estimated",
                          "Keccak-p[1600, 12] at a 168-B rate: the H100's measured ratio to Keccak-f (0.377, Round 11) "
                          "applied to SHAKE256's 4090 price"),
    "blake3": Hash(175.0, _scaled(175.0), "estimated", "about 700 INT32 instructions per 64-B block, by count"),
}
PINNED = {"ncp-v1": "shake256", "pearl-fp8-v4": "blake3"}   # the hash each scheme's tile digest uses


@dataclass(frozen=True)
class Term:
    name: str
    units: int
    checked: bool
    note: str = ""


def ncp_terms(m: int, k: int, n: int) -> list[Term]:
    """Route U at β = 128, f = 24, with sub-word extraction priced (``internal/pouw/new-crypto/ncp-accounting.md``, 03:50Z):
    the schedule whose biased accumulator is ``ncp-v1``'s checked values (Lean `wrefU 24`).  The retired unbiased sums would
    cost 36 units per position (γ = 1.08% at n = 2,048; ``route-u-checkpoints.md``)."""
    return [
        Term("GEMM m x 3k x n, int8 mma (u8 x s8)", 3 * m * k * n * MAC_INT8, True, "the credited work"),
        Term("E_1 fields from hash output, one LOP3 per 4", 4 * m * k, True),
        Term("E_1's complement fields, one LOP3 per 4", 4 * m * k, True),
        Term("A's complement, one LOP3 per 4", 4 * m * k, False, "skippable at zero input (bias in the accumulator)"),
        Term("L1, L2: one IADD per 4 each", 8 * m * k, False, "skippable at zero input"),
        Term("L3 = A' + E_1b' + E_2b', one IADD3 per 4", 4 * m * k, True),
        Term("E_2 = E_1·P (word-granular P: register moves)", 0, True, "free under W1"),
    ]


def ncp_hash_words(m: int, k: int, n: int) -> int:
    return m * n * (3 * k // D)


def pearl_terms(m: int, k: int, n: int) -> list[Term]:
    """Pearl's honest miner per unit (``schemes/pearl_kw``), A side and outputs; the per-weight B side is amortized.

    B̃·F_A is charged per unit: the whitepaper derives F_A from noise seed_A (§4.5, eq. 10), fresh for every activation
    commitment. Pearl #311's consensus (``zk-pow/src/api/fp8/noise.rs``) and ``pearl_kw`` key it by seed_B instead, which
    would amortize it with the B side."""
    mk, mn, nk = m * k, m * n, n * k
    return [
        Term("GEMM m x k x n, FP8 mma with FP32 accumulate", m * k * n * MAC_FP8_F32, True, "Pearl credits m·k·n MACs"),
        Term("noise N = E·Fᵀ, one rank-32 FP8 atom per A entry", mk * R * MAC_FP8_F32, True),
        Term("noisy quantize: β·N, fma, clamp, cvt to e4m3 (about 5 instructions per entry)", mk * 40, True, "Unverified count"),
        Term("FP10 prequant of A: block max, scale, int8 codes", mk * 24, True, "Unverified count"),
        Term("row norms l2, linf from the planes", mk * 2 * FP32, True, "Unverified count"),
        Term("extract: IMAD + SHF per output feeding the ticket", mn * 2 * INT32, True),
        Term("peel: A·F_B, BF16 mma (m·k·r)", mk * R * MAC_BF16_F32, False, "the useful output is unchecked"),
        Term("peel: B̃·F_A, BF16 mma (n·k·r)", nk * R * MAC_BF16_F32, False, "unchecked; F_A fresh per unit"),
        Term("peel: 2r correction columns per output, BF16 mma", mn * 2 * R * MAC_BF16_F32, False, "unchecked"),
        Term("unscale: one FFMA per output", mn * FP32, False, "unchecked"),
    ]


def pearl_hash_bytes(m: int, k: int, n: int) -> int:
    """A's FP10 planes (1.25 bytes per entry) and one 64-byte extract per 256-output tile; B's planes per weight."""
    return m * k * 5 // 4 + 64 * (m * n // 256)


@dataclass(frozen=True)
class Binding:
    hash: str | None            # None: no binding, the arithmetic alone
    status: str                 # the hash price's: "measured" or "estimated"; "none" without a binding
    total_vs_plain_gemm: str
    gamma: str | None           # None ("—") without a binding: the bound needs every checked value hashed


@dataclass(frozen=True)
class Row:
    scheme: str
    shape: tuple[int, int, int]
    wref: int
    credited: int
    omega: str
    gamma: str | None
    gamma_floor: str
    binding: str                        # the pinned hash
    total_vs_plain_gemm: str            # the headline: the arithmetic plus the binding at the pinned hash
    arithmetic_vs_plain_gemm: str       # without the binding, which carries no γ
    hashed_bytes_per_useful_mac: float
    hash_units_per_useful_mac: float    # at the pinned hash
    bindings: list[dict]                # none, then every hash in HASHES
    terms: list[dict]


def pct(x: Fraction) -> str:
    return f"{float(x) * 100:.3f}%"


def row(scheme: str, m: int, k: int, n: int) -> Row:
    if scheme == "ncp-v1":
        terms, credited, plain = ncp_terms(m, k, n), 3 * m * k * n, m * k * n * MAC_INT8
        hashed = 4 * ncp_hash_words(m, k, n)
    else:
        terms, credited, plain = pearl_terms(m, k, n), m * k * n * MAC_FP8_F32, m * k * n * MAC_FP8_F32
        hashed = pearl_hash_bytes(m, k, n)
    wref = sum(t.units for t in terms)
    skippable = sum(t.units for t in terms if not t.checked)
    omega = Fraction(wref, credited)
    gamma = None if scheme != "ncp-v1" else pct(1 - (1 - GAMMA0_NCP) / omega)

    def total(h: str | None) -> str:
        return f"{(wref + (hashed * HASHES[h].units_per_byte if h else 0)) / plain:.4f}"

    bindings = [Binding(None, "none", total(None), None)] + [Binding(h, v.status, total(h), gamma) for h, v in HASHES.items()]
    pinned = PINNED[scheme]
    return Row(scheme, (m, k, n), wref, credited, f"{float(omega):.5f}", gamma, pct(Fraction(skippable, wref)), pinned,
               total(pinned), total(None), hashed / (m * k * n), hashed * HASHES[pinned].units_per_byte / (m * k * n),
               [asdict(b) for b in bindings], [asdict(t) for t in terms])


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
    rows = [row(s, *shape) for s in ("ncp-v1", "pearl-fp8-v4") for shape in shapes]
    model = work["model"]
    out = {"schema": "verity-pouw/gamma/v2", "cost_model": "W1 (RTX 4090), sub-word extraction priced",
           "hashes": {h: asdict(v) for h, v in HASHES.items()}, "rows": [asdict(r) for r in rows],
           "model": model["id"], "uncovered_lm_head_fraction": uncovered_fraction(model)}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    print("W1 vs plain is the total: the arithmetic plus the binding at the scheme's pinned hash. Arithmetic alone carries no γ.")
    print(f"{'scheme':14} {'(m, k, n)':22} {'Ω':>8} {'γ':>9} {'γ floor':>9} {'W1 vs plain':>11} {'arith.':>8} "
          f"{'hash B/MAC':>10}  pinned")
    for r in rows:
        print(f"{r.scheme:14} {str(r.shape):22} {r.omega:>8} {r.gamma or '—':>9} {r.gamma_floor:>9} {r.total_vs_plain_gemm:>11} "
              f"{r.arithmetic_vs_plain_gemm:>8} {r.hashed_bytes_per_useful_mac:10.4f}  {r.binding}")
    print(f"\nPer binding ({', '.join(f'{h} {v.units_per_byte:g} units/B {v.status}' for h, v in HASHES.items())}):")
    print(f"{'scheme':14} {'(m, k, n)':22} {'binding':14} {'price':>9} {'W1 vs plain':>11} {'γ':>9}")
    for r in rows:
        for b in r.bindings:
            print(f"{r.scheme:14} {str(r.shape):22} {b['hash'] or 'none':14} {b['status']:>9} {b['total_vs_plain_gemm']:>11} "
                  f"{b['gamma'] or '—':>9}")
    print(f"{model['id']}: lm_head is {out['uncovered_lm_head_fraction']:.1%} of a decode token's GEMM MACs (uncovered)")


if __name__ == "__main__":
    main()
