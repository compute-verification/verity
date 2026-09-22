"""VOLE-based designated-verifier ZK (QuickSilver family) costed on the WP2 checker census.  A cost MODEL: every
figure is derived from the census (``security.census``) and from published benchmark tables cited in
:data:`ANCHORS`; nothing is measured.  Proof class of anything built from it: designated-verifier interactive ZK,
which is *not* one of the contract's public-coin classes -- see :data:`PROOF_CLASS_NOTE`.

Embedding (section 1 of the question).  IT-MAC commit-and-prove over ``F_p``, ``p = 2^61 - 1`` (the field of every
cited benchmark; ``p > 2^60`` so the checker's ``lt_pow`` products embed without wrap, checker F1), MACs in
``F_{p^r}`` (QuickSilver's subfield VOLE: the prover sends one ``F_p`` element per committed value whatever ``r``):

    committed value (word, hint, looked-up row column, product output, lookup helper)
                        1 VOLE correlation + 1 F_p element P->V (8 B)
    mul (incl. boolean) 1 multiplication check, batched: no per-check communication (QuickSilver Fig. 5;
                        booleans ``b(1-b)=0`` and products ``xy=z`` are degree-2 relations on committed values)
    lin                 free (local on MACs); one batched CheckZero
    range / table       three realisations, ``range_mode``:
        "logup"   Haboeck's identity with prover-side helpers ``h_j = 1/(z - w_j)``: 1 committed helper + 1 mul
                  check per query, ``|T|`` committed multiplicities per table per batch, table side linear.
                  [derivation: logUp in the IT-MAC commit-and-prove model; no VOLE paper states it]
        "rom"     Yang-Heath read-only memory (ePrint 2023/1115, USENIX Sec 2024, section 4.1): ``l+1`` inputs
                  per lookup (the ``l`` row columns are already committed here, so +1 version input) and two
                  fan-in-``(n+T)`` products for the permutation proof (2 mul checks per element), teardown
                  ``n(l+1)`` inputs.  [source]
        "bits"    bit decomposition in ``F_p``: one committed bit + one boolean check per bit of every range
                  check (3 384 bits per unit from the census widths); fixed tables by "logup".
                  [derivation]  (An ``F_2`` sVOLE realisation needs Mystique-style conversions, not modelled.)

Rounds: QuickSilver's online phase is 3 messages (P: commitments, V: chi, P: (U,V)); a lookup challenge ``z``
must follow the commitments and precede the helpers, so the sequential challenge depth is 2 (``z``, ``chi``) for
"logup"/"rom" and 1 for "bits".  The VOLE preprocessing (base OTs + LPN extension) is statement-independent and
constant-round; it is reported apart (``vole.preprocessing_rounds``), never inside the depth.

Soundness (``account_vole``): QuickSilver Theorem 2, ``(t+3)/p^r`` for ``t`` multiplication checks; Theorem 1,
``(d-1)/p^r`` for the extended sVOLE with ``d = 2``; logUp identity term per table (``accounting.logup_identity_error``
over ``F_{p^r}``, conditioned challenge space); the LPN/OT assumption behind the VOLE is recorded as a computational
term (``2^-kappa``), like the hash term of the candidates.  The MAC extension degree is the operating-point knob:
``first_extension`` finds the smallest ``r`` meeting a target (the analogue of the candidates' quadratic/cubic
Goldilocks extension).

Throughput (``ANCHORS``): QuickSilver Table 2 (m5.2xlarge, F_{2^61-1}, rho >= 40, i.e. ``r = 1``), AntMan Tables 1-2
(m5.8xlarge, p = 2^59 - 2^28 + 1), Yang-Heath Figure 10 (m5.2xlarge, F_{2^61-1}).  Wall time = gate-equivalents /
rate x ``ext_factor(r)`` (MAC arithmetic in ``F_{p^r}`` costs ``EXT_MULT_COST[r]`` base multiplications; half of the
per-gate work is taken to scale with it) -- an ASSUMPTION with a stated uncertainty factor (``UNCERTAINTY``).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from verity_numerical.security import accounting, census
from verity_numerical.security.accounting import (
    EXT_MULT_COST,
    MERSENNE61,
    Budget,
    Field,
    e2e_sweep,
    log2,
    logup_identity_error,
)

__all__ = [
    "ANCHORS",
    "PROOF_CLASS_NOTE",
    "RANGE_MODES",
    "UNCERTAINTY",
    "Anchor",
    "Embedding",
    "VoleParams",
    "account_vole",
    "census_estimate_vole",
    "embed",
    "first_extension",
    "main",
    "report",
    "table",
]

PROOF_CLASS_NOTE = (
    "designated-verifier interactive ZK (IT-MAC / VOLE): the verifier holds a secret key Delta; the transcript "
    "convinces nobody else.  Not a contract proof_class; the nearest label is COMPLETE_ZK_BACKEND with "
    "zk_mode='vole-itmac-designated-verifier' and this caveat, never compared against A/B without it."
)
RANGE_MODES = ("logup", "rom", "bits")
F_P: Field = Field.prime("F_{2^61-1}", MERSENNE61)
ELEMENT_BYTES = 8                     # one F_p element on the wire (61 bits, byte-aligned)
UNCERTAINTY = {"prover_seconds": 3.0, "verifier_seconds": 3.0, "transcript.bytes": 1.2}
ANTMAN_ELEMENTS_PER_GATE_TIMES_B = 0.0064 * 2048   # AntMan Table 1 (4 threads, 1 Gbps, |C| = 2^20): 13.1 / B
ANTMAN_SETUP_BYTES = 5_100_000                      # AntMan Table 1 caption: 5.1 MB BGV setup for every B


# ---------------------------------------------------------------------------------------------------------------------
# Published anchors (cite paper, table, hardware).  Rates are gates per second for the paper's own gate notion.
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    name: str
    paper: str
    where: str
    hardware: str
    threads: int
    bandwidth: str
    rate_per_s: float             # multiplication gates (1 VOLE + 1 check each) per second, or lookups per second
    field: str
    statistical_security: str
    bytes_per_gate: float = ELEMENT_BYTES
    url: str = ""
    note: str = ""


ANCHORS: dict[str, Anchor] = {
    "quicksilver_arith_1t": Anchor(
        "QuickSilver arithmetic, 1 thread, local-host / 2 Gbps",
        "Yang-Sarkar-Weng-Wang, QuickSilver, CCS 2021, ePrint 2021/076", "Table 2 (arithmetic, 1 thread, local-host)",
        "2x AWS m5.2xlarge (8 vCPU), ~0.1 ms latency", 1, "local-host", 4.8e6, "F_{2^61-1}, r=1", "rho >= 40",
        url="https://eprint.iacr.org/2021/076", note="includes sVOLE generation [YWL+20, WYKW21]; 1 F_p element per gate"),
    "quicksilver_arith_4t": Anchor(
        "QuickSilver arithmetic, 4 threads, local-host", "QuickSilver, ePrint 2021/076", "Table 2 (arithmetic, 4 threads)",
        "2x AWS m5.2xlarge", 4, "local-host", 8.9e6, "F_{2^61-1}, r=1", "rho >= 40", url="https://eprint.iacr.org/2021/076"),
    "quicksilver_arith_1t_50mbps": Anchor(
        "QuickSilver arithmetic, 1 thread, 100 Mbps", "QuickSilver, ePrint 2021/076", "Table 2 (arithmetic, 1 thread, 100 Mbps)",
        "2x AWS m5.2xlarge", 1, "100 Mbps", 1.2e6, "F_{2^61-1}, r=1", "rho >= 40", url="https://eprint.iacr.org/2021/076",
        note="bandwidth-bound: 1 element/gate"),
    "quicksilver_bool_1t": Anchor(
        "QuickSilver Boolean, 1 thread, local-host", "QuickSilver, ePrint 2021/076", "Table 2 (Boolean, 1 thread, local-host)",
        "2x AWS m5.2xlarge", 1, "local-host", 7.6e6, "F_2 values, F_{2^128} MACs", "rho ~ 100", bytes_per_gate=1 / 8,
        url="https://eprint.iacr.org/2021/076"),
    "quicksilver_bool_4t": Anchor(
        "QuickSilver Boolean, 4 threads, local-host", "QuickSilver, ePrint 2021/076", "Table 2 (Boolean, 4 threads, local-host)",
        "2x AWS m5.2xlarge", 4, "local-host", 15.8e6, "F_2 values, F_{2^128} MACs", "rho ~ 100", bytes_per_gate=1 / 8,
        url="https://eprint.iacr.org/2021/076"),
    "macncheese_arith_1t": Anchor(
        "Mac'n'Cheese arithmetic, 1 thread, 95 ms latency / 31.5 Mbps", "Baum-Malozemoff-Rosen-Scholl, CRYPTO 2021, ePrint 2020/1410",
        "abstract (1.5 us per multiplication gate)", "not stated in the abstract", 1, "31.5 Mbps", 1 / 1.5e-6, "F_{2^61-1}",
        "not read", url="https://eprint.iacr.org/2020/1410"),
    "wolverine_arith_5t": Anchor(
        "Wolverine arithmetic, 5 threads, 200 Mbps", "Weng-Yang-Katz-Wang, S&P 2021, ePrint 2020/925", "abstract (1.6 us/gate)",
        "not stated in the abstract; 400 MB memory", 5, "200 Mbps", 1 / 1.6e-6, "F_{2^61-1}", "rho = 40",
        bytes_per_gate=3 * ELEMENT_BYTES, url="https://eprint.iacr.org/2020/925", note="2-4 field elements per gate"),
    "antman_simd_16t": Anchor(
        "AntMan (B, |C|)-SIMD, 16 threads, 50 Mbps", "Weng-Yang-Yang-Xie-Wang, AntMan, CCS 2022, ePrint 2022/566",
        "Table 2 (AntMan-16, 50 Mbps; B=1024, |C|=2^21) and Table 1 (B=2048: 0.0064 elements/gate, 4 threads, 1 Gbps)",
        "2x AWS m5.8xlarge", 16, "50 Mbps", 14.07e6, "F_p, p = 2^59 - 2^28 + 1 (NOT > 2^60: fails checker F1 as benchmarked)",
        "soundness (2B+3)/|F| (Thm 2) + BGV/RLWE + PRG", bytes_per_gate=0.0064 * ELEMENT_BYTES,
        url="https://eprint.iacr.org/2022/566", note="needs a single-level BGV AHE: an extra (lattice) assumption; 6 rounds"),
    "yang_heath_rom": Anchor(
        "Yang-Heath ZK ROM, 1 thread, 1 Gbps, eps=8", "Yang-Heath, Two Shuffles Make a RAM, USENIX Sec 2024, ePrint 2023/1115",
        "Figure 10 (ROM, n=2^20, T=2^23, eps=8: 0.83 us/access, 24.9 B/access)", "2x AWS m5.2xlarge, single thread", 1,
        "1 Gbps", 1 / 0.83e-6, "F_{2^61-1}", "not restated here", bytes_per_gate=24.9, url="https://eprint.iacr.org/2023/1115"),
}


# ---------------------------------------------------------------------------------------------------------------------
# Embedding of the census
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Embedding:
    """Counts for one census object (unit or VU) under a range mode.  ``vole`` = VOLE correlations consumed =
    committed values; ``mul_checks`` = degree-2 relations checked; ``lin_free`` = local linear identities."""

    variant: str
    range_mode: str
    inputs: int                   # words + hints + looked-up row columns (census columns_committed_a)
    product_outputs: int          # mul minus booleans: committed product wires
    helpers: int                  # lookup helpers / ROM version inputs / decomposition bits
    mul_checks: int
    lin_free: int
    range_bits: int
    lookups: int

    @property
    def vole(self) -> int:
        return self.inputs + self.product_outputs + self.helpers

    @property
    def p2v_bytes(self) -> int:
        return self.vole * ELEMENT_BYTES

    def summary(self) -> dict:
        return {"variant": self.variant, "range_mode": self.range_mode, "vole_correlations": self.vole,
                "committed.inputs": self.inputs, "committed.product_outputs": self.product_outputs,
                "committed.helpers": self.helpers, "mul_checks": self.mul_checks, "lin_free": self.lin_free,
                "range_bits": self.range_bits, "lookups": self.lookups, "p2v_bytes": self.p2v_bytes}


def range_bits(c: census.UnitCensus) -> int:
    """Total bits of every range check of the census (the widths the checker asserts, not the chunked lookups)."""
    return sum(w * n for w, n in c.range_by_width.items())


def embed(c: census.UnitCensus, range_mode: str = "logup") -> Embedding:
    if range_mode not in RANGE_MODES:
        raise ValueError(f"range_mode must be one of {RANGE_MODES}")
    bits = range_bits(c)
    if range_mode == "logup":
        helpers, extra_mul = c.lookup, c.lookup
    elif range_mode == "rom":
        helpers, extra_mul = c.lookup, 2 * c.lookup
    else:
        helpers, extra_mul = bits + c.fixed_lookups, bits + c.fixed_lookups
    return Embedding(c.variant, range_mode, c.columns_committed_a, c.mul - c.booleans, helpers, c.mul + extra_mul,
                     c.lin_total, bits, c.lookup)


# ---------------------------------------------------------------------------------------------------------------------
# Parameters, soundness, census columns
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VoleParams:
    K: int
    B: int
    mac_extension: int = 3                    # r: MACs in F_{p^r}
    range_mode: str = "logup"
    anchor: str = "quicksilver_arith_1t"
    simd: bool = False                         # AntMan packing of the B identical VU circuits (communication only)
    # LPN / OT computational term, recorded as 2^-kappa_term like the candidates' hash term (their q^2/2^256 with
    # q=2^60 is 2^-136): matched computational budget, not a statement about a particular LPN parameter set.
    kappa_term: int = 136
    preprocessing_rounds: int = 3              # base OTs + sVOLE extension, statement-independent (assumption)
    tables: tuple = field(default_factory=lambda: accounting.CENSUS_TABLES)

    @property
    def units(self) -> int:
        if self.K % 16:
            raise ValueError("K must be a multiple of 16")
        return self.B * (self.K // 16)

    @property
    def mac_field(self) -> Field:
        return F_P.extension(self.mac_extension)


def _vu(p: VoleParams) -> census.UnitCensus:
    return census.census_vu(p.K)


def account_vole(p: VoleParams) -> Budget:
    """Soundness budget: QuickSilver batched multiplication check, extended sVOLE, logUp identities, LPN term."""
    Fm = p.mac_field
    vu = embed(_vu(p), p.range_mode)
    t = vu.mul_checks * p.B
    if p.range_mode != "bits":
        t += sum(2 * tb.rows for tb in p.tables) if p.range_mode == "rom" else 0
    b = Budget("VOLE-ZK (QuickSilver-style IT-MAC commit-and-prove, designated verifier)")
    b.add("batching", f"QuickSilver Thm 2: batched multiplication check, (t+3)/p^r, t={t}", Fraction(t + 3, Fm.size))
    b.add("batching", "QuickSilver Thm 1: extended sVOLE (d=2), (d-1)/p^r", Fraction(1, Fm.size))
    b.add("batching", "batched CheckZero of all linear identities, 1/p^r", Fraction(1, Fm.size))
    if p.range_mode == "bits":
        tables = [tb for tb in p.tables if tb.name != "T_range16"]
    else:
        tables = list(p.tables)
    for tb in tables:
        N = tb.queries_per_unit * p.units
        if p.range_mode == "rom":
            b.add("lookup", f"{tb.name}: permutation proof over n+T={tb.rows + N} (Schwartz-Zippel, degree n+T)",
                  Fraction(tb.rows + N, Fm.size))
        else:
            b.add("lookup", f"{tb.name}: logUp identity, N={N}, |T|={tb.rows}, cols={tb.columns}, conditioned on z not in image",
                  logup_identity_error(N, tb.rows, Fm, columns=tb.columns))
    if p.simd:
        b.add("batching", f"AntMan Thm 2: (2B+3)/|F| SIMD packing, B={p.B}", Fraction(2 * p.B + 3, Fm.size))
    b.add("commitment_hash", f"LPN / base-OT computational term, 2^-{p.kappa_term} (modelling choice, not a probability)",
          Fraction(1, 2**p.kappa_term), kind="computational",
          note="the VOLE functionality is realised from LPN + OT; recorded like the candidates' hash term")
    b.add("zk", "IT-MAC hiding is perfect; ZK against a malicious verifier holds in the F_VOLE-hybrid model "
          "(QuickSilver Thm 2 proof, malicious-verifier case); the VOLE protocol itself must be maliciously secure", 0, kind="zk")
    b.parameters.update({"K": p.K, "B": p.B, "units": p.units, "field": F_P.name, "mac_field": Fm.name,
                         "mac_extension": p.mac_extension, "range_mode": p.range_mode, "mul_checks": t,
                         "vole_correlations": vu.vole * p.B + _batch_committed(p),
                         "proof_class": PROOF_CLASS_NOTE})
    return b


def _batch_committed(p: VoleParams) -> int:
    """Per-batch committed values beyond the B VUs: multiplicities (logup), ROM teardown inputs, none for bits."""
    if p.range_mode == "logup":
        return sum(tb.rows for tb in p.tables)
    if p.range_mode == "rom":
        return sum(tb.rows * tb.columns for tb in p.tables)     # (l+1) inputs per row: the l-1 outputs + key + version
    return sum(tb.rows for tb in p.tables if tb.name != "T_range16")


def first_extension(target_log2: int, make, degrees: Sequence[int] = (1, 2, 3, 4)) -> tuple[int, Budget]:
    """Smallest MAC extension degree whose budget meets ``2**target_log2``; ``make(r)`` builds the params."""
    for r in degrees:
        bud = account_vole(make(r))
        if bud.meets(target_log2):
            return r, bud
    raise accounting.AccountingError(f"no extension degree in {tuple(degrees)} meets 2^{target_log2}")


def ext_factor(r: int) -> float:
    """Wall-time multiplier for MACs in F_{p^r} against the r=1 benchmarks: half of the per-gate work is MAC
    arithmetic scaling with EXT_MULT_COST[r] (ASSUMPTION; 1, 2, 3.5, 5 for r = 1..4)."""
    return (1 + EXT_MULT_COST[r]) / 2


def census_estimate_vole(p: VoleParams) -> dict[str, float]:
    """The contract's ``census.*`` / ``rounds.*`` / ``transcript.bytes`` / ``t.e2e.*`` columns plus ``vole.*`` and
    ``estimate.*`` (all ESTIMATES from the module docstring's model)."""
    vu_c = _vu(p)
    vu = embed(vu_c, p.range_mode)
    r = p.mac_extension
    a = ANCHORS[p.anchor]
    committed = vu.vole * p.B + _batch_committed(p)
    mul_checks = vu.mul_checks * p.B + (sum(2 * tb.rows for tb in p.tables) if p.range_mode == "rom" else 0)
    wires = vu_c.columns_all * p.B + vu.helpers * p.B
    # communication
    if p.simd:
        # AntMan Table 1: elements per gate halve as B doubles (0.82 at B=16 ... 0.0064 at B=2048): ~13.1/B, capped at 1
        per_gate = min(1.0, ANTMAN_ELEMENTS_PER_GATE_TIMES_B / p.B) * ELEMENT_BYTES
        p2v = int(per_gate * (vu.mul_checks + vu.vole) * p.B) + _batch_committed(p) * ELEMENT_BYTES + ANTMAN_SETUP_BYTES
    else:
        p2v = committed * ELEMENT_BYTES + 3 * r * ELEMENT_BYTES          # values + (U, V) + CheckZero MAC
    n_tables = len(p.tables) if p.range_mode != "bits" else len(p.tables) - 1
    v2p = (2 * n_tables + 1) * r * ELEMENT_BYTES + 32                    # z_T, beta_T, chi + a seed
    depth = 1 if p.range_mode == "bits" else 2
    protocol_rounds = depth + (6 if p.simd else 0)
    # time
    gate_eq = (committed + mul_checks) / 2
    rate = a.rate_per_s
    prover = gate_eq / rate * ext_factor(r)
    verifier = prover                                                     # ASSUMPTION: symmetric linear work
    # field ops: ~8 F_p ops per correlation (LPN extension) + 3 F_{p^r} mults per check (prover) as F_p equivalents
    field_ops = 8 * committed + 3 * EXT_MULT_COST[r] * mul_checks
    bytes_encoded = committed * (1 + r) * ELEMENT_BYTES                  # value + MAC materialised by the prover
    per_vu_bytes = vu.vole * (1 + r) * ELEMENT_BYTES
    out = {
        "census.witness_elements": wires,
        "census.committed_elements": committed,
        "census.bytes_encoded": bytes_encoded,
        "census.hash_bytes": 0,
        "census.field_ops": field_ops,
        "census.memory_passes": 2,
        "census.bytes_moved": 2 * bytes_encoded,
        "rounds.protocol": protocol_rounds,
        "rounds.sequential_depth": depth,
        "transcript.bytes": p2v + v2p,
        "verifier.seconds": verifier,
        "estimate.prover_seconds": prover,
        "estimate.transcript_p2v_bytes": p2v,
        "estimate.transcript_v2p_bytes": v2p,
        "estimate.peak_bytes": 4 * per_vu_bytes,                          # streaming: a few VUs in flight
        "estimate.transfer_seconds_1gbps": (p2v + v2p) / 125e6,
        "vole.correlations": committed,
        "vole.mul_checks": mul_checks,
        "vole.lin_free": vu.lin_free * p.B,
        "vole.range_bits": vu.range_bits * p.B,
        "vole.preprocessing_rounds": p.preprocessing_rounds,
        "vole.mac_extension": r,
        "vole.ext_factor": ext_factor(r),
        "vole.rate_anchor_per_s": rate,
        **e2e_sweep(depth, prover, verifier),
    }
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Tables and report
# ---------------------------------------------------------------------------------------------------------------------

def _make(K: int, B: int, mode: str, anchor: str, simd: bool = False):
    return lambda r: VoleParams(K=K, B=B, mac_extension=r, range_mode=mode, anchor=anchor, simd=simd)


def table(K: int = 1536, batches: Sequence[int] = (1, 64, 4096), targets: Sequence[int] = (-128, -80),
          range_mode: str = "logup", anchor: str = "quicksilver_arith_1t", simd: bool = False) -> list[dict]:
    rows = []
    for tgt in targets:
        for B in batches:
            r, bud = first_extension(tgt, _make(K, B, range_mode, anchor, simd))
            p = VoleParams(K=K, B=B, mac_extension=r, range_mode=range_mode, anchor=anchor, simd=simd)
            est = census_estimate_vole(p)
            rows.append({"target_log2": tgt, "B": B, "K": K, "range_mode": range_mode, "anchor": anchor, "simd": simd,
                         "mac_extension": r, "soundness_log2": round(log2(bud.total()), 1), **est})
    return rows


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1e9:
            return f"{v / 1e9:.3g}G"
        if abs(v) >= 1e6:
            return f"{v / 1e6:.3g}M"
        if abs(v) >= 1e3:
            return f"{v / 1e3:.3g}k"
        return f"{v:.3g}"
    return f"{v:,}"


def print_table(rows: list[dict], cols: Sequence[str]) -> None:
    head = ["tgt", "B", "r"] + list(cols)
    print(" | ".join(head))
    for row in rows:
        print(" | ".join([str(row["target_log2"]), str(row["B"]), str(row["mac_extension"])] + [_fmt(row[c]) for c in cols]))


def report(K: int = 1536) -> dict:
    """Everything ``backends/numerical/reports/vole.json`` holds: embeddings, tables per mode, comparison, citations."""
    unit, vu = census.census_unit(), census.census_vu(K)
    modes = {m: {"unit": embed(unit, m).summary(), "vu": embed(vu, m).summary()} for m in RANGE_MODES}
    tables = {m: table(K, range_mode=m) for m in RANGE_MODES}
    tables["logup_4threads"] = table(K, range_mode="logup", anchor="quicksilver_arith_4t")
    tables["logup_antman_simd"] = table(K, range_mode="logup", anchor="antman_simd_16t", simd=True)
    comparison_b4096 = {
        "A_transcript_bytes": 74.7e6, "B_transcript_bytes": 318e6,
        "A_depth": 381, "B_depth": 3,
        "vole_logup_transcript_bytes": tables["logup"][2]["transcript.bytes"],
        "vole_bits_transcript_bytes": tables["bits"][2]["transcript.bytes"],
        "vole_antman_transcript_bytes": tables["logup_antman_simd"][2]["transcript.bytes"],
        "vole_depth": 2,
        "sp1_core_seconds_K1536_B1": 3.58,
    }
    return {
        "proof_class": PROOF_CLASS_NOTE,
        "field": {"values": F_P.name, "macs": "F_{p^r}, r = first_extension(target)"},
        "embedding": modes,
        "tables": tables,
        "comparison_B4096": comparison_b4096,
        "uncertainty_factors": UNCERTAINTY,
        "anchors": {k: vars(a) for k, a in ANCHORS.items()},
        "assumptions": [
            "wall time = (VOLE correlations + multiplication checks)/2 / anchor rate x ext_factor(r); verifier = prover",
            "ext_factor(r) = (1 + EXT_MULT_COST[r]) / 2: half of the per-gate work is MAC arithmetic in F_{p^r}",
            ("one F_p element (8 B) per committed value; batched checks add O(r) elements; VOLE preprocessing "
             "communication (sublinear, LPN) not counted"),
            "logUp-in-IT-MAC is a derivation (helper + 1 mul per query); the ROM row is Yang-Heath's published construction",
            ("AntMan row: communication 13.1/B elements per gate from Table 1; its field p=2^59-2^28+1 fails checker F1 "
             "(p > 2^60) and its soundness (2B+3)/|F| needs an extension not benchmarked; BGV AHE assumption"),
            "streaming memory: peak = 4 VUs of (value + MAC) bytes",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--json":
        with open(args[1], "w") as f:
            json.dump(report(), f, indent=1, default=str)
        print(f"wrote {args[1]}")
        return
    K = int(args[0]) if args else 1536
    unit, vu = census.census_unit(), census.census_vu(K)
    for m in RANGE_MODES:
        print(f"== embedding {m}: unit {embed(unit, m).summary()}\n   vu   {embed(vu, m).summary()}")
    cols = ("vole.correlations", "vole.mul_checks", "transcript.bytes", "rounds.sequential_depth",
            "estimate.prover_seconds", "verifier.seconds", "census.field_ops", "estimate.peak_bytes", "t.e2e.rtt_10ms")
    for m in RANGE_MODES:
        print(f"\n== range_mode={m}, anchor=quicksilver_arith_1t (K={K})")
        print_table(table(K, range_mode=m), cols)
    print("\n== range_mode=logup, AntMan SIMD packing (communication only; see assumptions)")
    print_table(table(K, range_mode="logup", anchor="antman_simd_16t", simd=True), cols)


if __name__ == "__main__":       # pragma: no cover
    main()
