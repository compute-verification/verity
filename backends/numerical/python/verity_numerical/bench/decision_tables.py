"""Regenerate every table of ``backends/numerical/reports/DECISION_DRAFT.md`` (and its appendix) from the overhead
ledger (``backends/numerical/reports/ledger/*.jsonl``) and the per-lane reports.

    python -m verity_numerical.bench.decision_tables            # print every table
    python -m verity_numerical.bench.decision_tables --write    # refresh the tables in place between the markers
    python -m verity_numerical.bench.decision_tables --check    # exit 1 if the draft's tables are stale

Every row of every table names the ledger lane and run id it was read from.  The prover seconds per VU, the
overhead, the proof class, the hardware, the batch size and the soundness bound *as the accountant states it* are
taken from the ledger entry (looked up by lane + run id + label prefix; a missing entry is an error, so a row cannot
silently outlive its evidence).  Fields the ledger schema does not carry as columns (verifier seconds, proof bytes,
``rounds.sequential_depth``) are transcribed from that entry's ``note`` and carried here as constants tagged with the
run id -- change them only together with the entry they cite.  End-to-end times use ``bench.latency.e2e`` (prove +
verify + depth x RTT + bytes / link) at the 10 Gb/s link the a-verifier lane used; ``latency.md`` used 25 Gb/s.

Rows whose lane has not merged into ``main`` are marked PENDING and carry no ledger lookup; when the lane merges, add
the ledger key and drop the PENDING flag (``PENDING_LANES`` lists them with the files they will write).  A red-team
verdict that downgrades a ledger label is carried as ``label_override`` on the row: downgrades stand, nothing is upgraded.

The tables live in the draft between ``<!-- BEGIN TABLE:<name> -->`` / ``<!-- END TABLE:<name> -->`` markers; the
prose around them is the coordinator's.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_numerical.bench import latency, ledger

__all__ = ["REPORTS", "TABLES", "main", "render", "render_all", "update_markdown"]

REPORTS = ledger.DIR.parent
DRAFT = REPORTS / "DECISION_DRAFT.md"
APPENDIX = REPORTS / "DECISION_DRAFT_appendix.md"
LINK_GBPS = 10.0
RTTS_MS = (0.0, 1.0, 10.0, 50.0)
NATIVE_PS = 9.85  # ps per K=1536 VU, contract.NATIVE_REFERENCE


# --- ledger lookup ----------------------------------------------------------------------------------------------------


def _entries(directory: Path) -> list[dict[str, Any]]:
    return [e for rows in ledger.load_all(directory).values() for e in rows]


def _find(entries: Sequence[dict[str, Any]], lane: str, run_id: str | None, label: str) -> dict[str, Any]:
    hits = [e for e in entries if e["lane"] == lane and e.get("run_id") == run_id and e["label"].startswith(label)]
    if len(hits) != 1:
        raise LookupError(f"ledger key ({lane!r}, {run_id!r}, {label!r}) matched {len(hits)} entries, need exactly 1")
    return hits[0]


# --- row specifications ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One like-for-like row.  ``key`` = (lane, run_id, label prefix) in the ledger; the rest is transcribed from that
    entry's note (or from the named report) and tagged with its source."""

    design: str
    key: tuple[str, str | None, str] | None
    verify_s: float | None
    proof_bytes: float | None
    depth: int | None
    depth_note: str = ""
    prove_s: float | None = None  # per batch; when None, ledger seconds_per_vu x B
    verify_note: str = ""
    flags: str = ""
    pending: str = ""  # non-empty = PENDING: the file the lane will write
    verified: str = ""  # who accepted the proofs: the candidate's own verifier / an independent one / REJECTED
    label_override: str = ""  # a red-team downgrade of the ledger label (downgrades stand; never upgrade)
    # PENDING rows carry their own values (no ledger entry to read):
    manual: dict[str, Any] = field(default_factory=dict)


LIKE_FOR_LIKE: tuple[Row, ...] = (
    # --- the same-card rows: one H100 (vy-g4), gate4-h100, main = 38757ed, v1 checker, both statement-bound ---
    Row("A: GKR+LogUp+Ligero v1, BabyBear^6, a-gpu v2e kernel path -- gate4 same-card row",
        ("gate4-h100", "r20260922-091718-45c3", "A e2e non-ZK B=4096 (a-gpu v2e, main 38757ed)"),
        verify_s=6.06, proof_bytes=33.9e6, depth=438, verify_note="Python", verified="own Python verifier, 5/5 reps",
        flags="median of 5 warm reps [7.54, 5.02, 5.03, 4.40, 4.20]; spread is host-side Merkle openings 1.4-3.9 s"),
    Row("B-Ligero v1 (b-ligero2 prover), non-ZK -- gate4 same-card row",
        ("gate4-h100", "r20260922-092841-94ac", "B e2e non-ZK 4096 VUs as 25x170 (b-ligero2)"),
        verify_s=0.816, proof_bytes=110.8e6, depth=2, verified="own Python verifier, 3/3 reps",
        flags="25 x 170 VUs, t=196; hash + SHAKE expander host placeholders"),
    Row("B-Ligero v1 (b-ligero2 --zk), 8b masking -- gate4 same-card row",
        ("gate4-h100", "r20260922-092841-94ac", "B e2e HVZK 4096 VUs as 25x170 (b-ligero2 --zk)"),
        verify_s=0.880, proof_bytes=122.3e6, depth=2, verified="own Python verifier, 3/3 reps",
        label_override="NON_ZK_PROOF_DIAGNOSTIC (red-team-zk F1: 8b leaves q unmasked on H; ledger says COMPLETE_HVZK_BACKEND)",
        flags="zk masks 0.40 s host SHAKE = 14%"),
    # --- A's other GPU numbers, each with its verification status (no winner picked) ---
    Row("A: a-gpu2 v6 (packed GKR + graphed LogUp + INT8 opening), v1, BabyBear^6 X^6-22, H100 vy-g5",
        ("a-gpu2", "r20260922-094040-9e76", "full VU B=4096 H100 v6"),
        verify_s=1.34, proof_bytes=33.9e6, depth=438, verify_note="vectorised Python",
        verified="own Python verifier 4096/4096 x3, 52/52 neg rejected; independent Rust verifier REJECTS (W=31 at LogUp POW level 0, "
                 "r20260922-101240-8f71; W=22 verifier 2943ae0 at 'LogUp POW level 2: final check', r20260922-102428-2285 -- coordinator, not on main)",
        flags="reproduced from main at 1.80-1.93 s (r20260922-101240-8f71); a-protocol-diff localising the Rust/Python divergence (PENDING)"),
    Row("A: a-gpu v2e kernel path, H100 vy-g5 (a-gpu's own record)",
        ("a-gpu", "r20260922-074455-d0b6", "full VU B=4096 H100 v2e"),
        verify_s=3.017, proof_bytes=33.9e6, depth=438, verify_note="Rust 12T; Python 7.1 s",
        verified="own Python verifier; a-verifier F1: the kernel path at c4d472a REJECTED by both verifiers on vy-g5 and 4.43 s not reproduced (a-verifier-2 F1_TRIAGE.md PENDING)",
        flags="superseded by gate4's 5.02 s re-run of the same prover (accepted by the Python verifier on vy-g4)"),
    Row("A: a-gpu torch path (VERITY_GPU_TORCH_ONLY=1), H100 vy-g5",
        ("a-verifier", "r20260922-093721-1330", "A-GPU e2e w/ Rust verifier"),
        verify_s=3.017, proof_bytes=33.9e6, depth=438, prove_s=25.5, verify_note="Rust 12T",
        verified="INDEPENDENT Rust verifier + Python: accepted (the only A GPU proofs the Rust verifier accepts)",
        flags="prover 25.5 s median of 3 (r20260922-091240-4bf9)"),
    Row("A: checker v2 + limb-form epilogue, BabyBear^6, a-gpu-v2, A100 vy-a100b",
        ("a-gpu-v2", "r20260922-095013-db80", "v2 GPU BabyBear^6 limb epilogue full VU B=4096 A100"),
        verify_s=8.1, proof_bytes=21.1e6, depth=None, verify_note="Python", depth_note="B=64 run reports 279; not reported at 4096",
        verified="own Python verifier 4096/4096; 52/52 + 30 unit + 4 chain + 15/15 epilogue negatives rejected; no Rust verifier for v2",
        flags="same-box v1 control 7.93 s (r20260922-091544-4e02): v2 = 1.38x; lookup bucket launch-bound, a-gpu2's graphed rounds not applied"),
    Row("A: GKR checker v2, Goldilocks^3, CPU Rust (a-v2)",
        ("a-v2", "r20260922-081106-6067", "A on checker v2 (Goldilocks), 4096 VUs"),
        verify_s=12.2, proof_bytes=41.5e6, depth=328, verified="own Rust verifier",
        flags="run-vu chain.txt trust (red-team-v2 #9) fixed ff651a8, fixture refused r20260922-095128-05ac; not re-timed"),
    Row("A: GKR v1 BabyBear^6, CPU Rust (a-babybear)",
        ("a-babybear", "r20260922-065112-d687", "BabyBear^6 full VU B=4096"),
        verify_s=45.0, proof_bytes=33.39e6, depth=381, verified="own Rust verifier",
        depth_note="derived; latency lane measured 355 trips on Goldilocks"),
    # --- B-Ligero's faster (L40S) rows, labels as red-teamed ---
    Row("B-Ligero: fused tests, --zk + HM96 coins, L40S (b-zk)",
        ("b-zk", "r20260922-092404-2a1b", "B-Ligero full ZK (HM96), fused tests"),
        verify_s=0.96, proof_bytes=122.3e6, depth=3, verified="own Python verifier",
        label_override="NON_ZK_PROOF_DIAGNOSTIC (red-team-zk VERDICT r20260922-100305-2b13; ledger says COMPLETE_ZK_BACKEND)",
        flags="F1 q unmasked: 16150/16150 accumulators recovered; F2 simulator PROM-only; F3 measured mode is Fiat-Shamir (needs t~279), 2^-128.05 is the interactive bound; repair on b-zk-fix (PENDING)"),
    Row("B-Ligero: fused tests, --zk, no coin commitment, L40S (b-zk)",
        ("b-zk", "r20260922-092404-2a1b", "B-Ligero HVZK (no coin commitment), fused"),
        verify_s=0.94, proof_bytes=122.3e6, depth=2, verified="own Python verifier",
        label_override="NON_ZK_PROOF_DIAGNOSTIC (red-team-zk F1; ledger says COMPLETE_HVZK_BACKEND)"),
    Row("B-Ligero: fused tests, non-ZK, L40S (b-zk)",
        ("b-zk", "r20260922-092404-2a1b", "B-Ligero non-ZK, fused tests"),
        verify_s=0.96, proof_bytes=110.8e6, depth=3, depth_note="coin commitment left on; 2 without", verified="own Python verifier",
        flags="the fastest measured full-batch prover on any card; F3 applies: as run it is Fiat-Shamir, 2^-128.25 is the interactive bound"),
    Row("B-Ligero at the checker-v2 shape, eager torch, RTX 4090 (b-e2e-v2)",
        ("b-e2e-v2", "r20260922-094956-5453", "B-Ligero v2 e2e GPU, 4096 VUs = 7 proofs x 682"),
        verify_s=0.40 * 7, proof_bytes=9.2e6 * 7, depth=2, verify_note="0.40 s x 7 proofs", verified="own verifier 7/7; 52/52 + 39/39 rejected",
        flags="INCOMPLETE_CHECKER caveat (Python limb rows not z3-verified); kernels not wired: v2 real is SLOWER than v1 real; 180x above b-v2's synthetic 42.4 ms"),
    # --- B-AIR, C, SP1 ---
    Row("B-AIR: Plonky3 LogUp AIR, checker v2, BabyBear^6, CPU Rust (b-air-v2)",
        ("b-air-v2", "r20260922-090904-cd75", "checker v2 BabyBear^6 AIR + LogUp"),
        verify_s=0.731, proof_bytes=2.03e6, depth=26, verified="Plonky3's own verifier; none independent",
        flags="statement-bound (F1-F3 header); the only honest 2^-128 B-AIR row"),
    Row("B-AIR: checker v2, BabyBear^4, CPU Rust (b-air-v2)",
        ("b-air-v2", "r20260922-085312-82a9", "checker v2 BabyBear AIR + LogUp tuple lookups"),
        verify_s=0.455, proof_bytes=1.95e6, depth=26, verified="Plonky3's own verifier"),
    Row("B-AIR: v1 + LogUp range lookups, BabyBear^4, CPU Rust (b-lookup)",
        ("b-lookup", "r20260922-070654-177a", "BabyBear AIR + LogUp range lookups"),
        verify_s=0.441, proof_bytes=2.27e6, depth=26, verified="Plonky3's own verifier",
        flags="STATEMENT-UNBOUND on main (red-team-3 U1/U2 ACCEPTED); fix on lane/b-bind (PENDING): timing-only until then"),
    Row("B-AIR: v1 limbs, BabyBear^4, CPU Rust (b-lookup, after lead.window fix)",
        ("b-lookup", "r20260922-065546-a242", "SOUNDNESS FIX: LT_DIRECT"),
        verify_s=0.426, proof_bytes=3.47e6, depth=22, verified="Plonky3's own verifier",
        flags="STATEMENT-UNBOUND on main (red-team-vu F1/F2); fix on lane/soundness-fixes via b-bind (PENDING, +2^-105 term at ^4)"),
    Row("C: QuickSilver VOLE (dietmc), compressed v2, F384p, 1 thread (c-compress)",
        ("c-compress", "r20260922-080520-71b6", "4096 VUs, compressed relation v2, F384p (2^-128 headline)"),
        verify_s=1255.6, proof_bytes=20.88e9, depth=3, verified="designated verifier (dietmc), not red-teamed",
        verify_note="designated verifier in lock-step: latency.md sets verify = prove",
        depth_note="latency.md catalog; the v2 logUp adds challenges not counted as measured trips"),
    Row("C: same, F61p (sensitivity point, NOT a security level)",
        ("c-compress", "r20260922-075153-022f", "4096 VUs, compressed relation v2, F61p (full batch)"),
        verify_s=123.8, proof_bytes=3.545e9, depth=3, verify_note="as above", verified="designated verifier"),
    Row("SP1 (baseline): sound authenticated core, 1 VU per proof, RTX 4090",
        ("sp1", "r20260921-220722-019b", "SP1 sound, authenticated core"),
        verify_s=0.245, proof_bytes=5.9e6, depth=1, depth_note="Fiat-Shamir", verified="SP1's verifier",
        flags="B=1; 4096 x 3.58 s = 14,664 s if serial is an extrapolation; authentication INCLUDED"),
)


REPRICED: tuple[tuple[str, tuple[str, str | None, str], str], ...] = (
    # (design, ledger key, note)
    ("A v1 buckets (fused sumcheck + graphed logUp), H100", ("a-fusion", None, "A v1 re-priced: fused sumcheck + graphed logUp"), "70% measured"),
    ("A v2 buckets (fused sumcheck + 10-stream logUp), H100", ("a-fusion", None, "A v2 re-priced: fused sumcheck + 10-stream logUp"), "68% measured"),
    ("A v1 buckets, A100 anchor", ("anchor-a100", "r20260922-080250-cff5", "A re-priced, A100-measured kernels"), "99.7% measured; logUp launch-bound 59%"),
    ("B v1 buckets (structured linear test M=3), RTX 4090", ("b-lintest", "r20260922-073017-0924", "B re-priced, structured linear test M=3 lazy"), "89% measured"),
    ("B v2 buckets (blocked inversions + radix-4), RTX 4090", ("b-v2", "r20260922-081137-109d", "B at checker-v2 shape, blocked inv + radix-4"), "94% measured"),
    ("B v1 buckets, A100 anchor", ("anchor-a100", "r20260922-080250-cff5", "B re-priced, A100-measured kernels"), "80% measured"),
    ("A v1 buckets re-run by gate4-h100, H100 vy-g4 (floor 0.285 s)", ("gate4-h100", "r20260922-092450-5b79", "A buckets v1"), "70% measured; a-fusion confirmed within 2%"),
    ("A v2 buckets re-run by gate4-h100, H100 vy-g4 (floor 0.184 s)", ("gate4-h100", "r20260922-092450-5b79", "A buckets v2"), "68% measured"),
    ("B v1 buckets re-run by gate4-h100, H100 vy-g4 (floor 0.112 s)", ("gate4-h100", "r20260922-093147-3d6c", "B buckets v1"), "87% measured; L2-slice pipeline slower on H100"),
    ("B v2 buckets re-run by gate4-h100, H100 vy-g4 (floor 0.046 s)", ("gate4-h100", "r20260922-093147-3d6c", "B buckets v2"), "93% measured"),
    ("A v2 model (CPU ratios scaled), H100 @0.172", ("a-v2", "r20260922-081106-6067", "A GPU re-priced on v2"), "model, not run"),
    ("checker-min models: A v2 / B v2, H100 @0.172", ("checker-min", None, "A re-priced on checker v2, model"), "everything modelled"),
)


SOUNDNESS_KEYS: tuple[tuple[str, str | None, str], ...] = (
    ("red-team-vu", "r20260922-070024-e7d1", "B FORGED: any trace = any statement"),
    ("red-team-vu", "r20260922-070024-e7d1", "leaf/v2 hiding false"),
    ("b-lookup", "r20260922-065546-a242", "SOUNDNESS FIX: LT_DIRECT"),
    ("a-babybear", "r20260922-063837-b26c", "BabyBear^6 gate"),
    ("b-ligero2", "r20260922-073932-bc75", "SOUNDNESS: statement unbound"),
    ("red-team-3", "r20260922-085839-3976", "B LogUp FORGED: statement-unbound"),
    ("red-team-3", "r20260922-085839-3976", "B LogUp FORGED: degree_bits"),
    ("red-team-v2", "r20260922-094212-465e", "A-GKR v2 run-vu trusts bundle chain.txt"),
    ("a-verifier", "r20260922-093721-1330", "Rust BB^6 verifier accepts a-gpu B=4096"),
    ("red-team-zk", "r20260922-100305-2b13", "rt-zk F1 ACCEPTED-LEAK"),
    ("red-team-zk", "r20260922-100305-2b13", "rt-zk F2 BROKEN"),
    ("red-team-zk", "r20260922-100305-2b13", "rt-zk F3 MODEL MISMATCH"),
    ("red-team-zk", "r20260922-100305-2b13", "rt-zk coin commit HOLDS"),
    ("red-team-zk", "r20260922-100305-2b13", "rt-zk VERDICT"),
)


FIXED_PAIRS: tuple[tuple[str, tuple[str, str | None, str], tuple[str, str | None, str] | None, str], ...] = (
    ("A R16 sub-16-bit range wrap (a-babybear): Goldilocks 4096 VUs",
     ("a-chain", "r20260922-055358-9f6b", "full VU B=4096, headline"),
     ("a-babybear", "r20260922-065844-2b84", "Goldilocks^3 full VU B=4096, fixed range tables"),
     "fix removed sub-16-bit chunks: FASTER (297.6 -> 261.3 s) and shallower (depth 438 -> 381); same box, same threads"),
    ("B-AIR lead.window lower bound (b-lookup): BabyBear limbs 4096 VUs",
     ("b-chain", "r20260922-060503-6847", "BabyBear AIR at 2^-128 proven"),
     ("b-lookup", "r20260922-065546-a242", "SOUNDNESS FIX: LT_DIRECT"),
     "+30 columns: 168.12 -> 173.24 s (+3%, within shared-box noise)"),
    ("B-Ligero statement binding (b-ligero2): pre-fix numbers were <=85 VUs/proof only",
     ("b-ligero", None, "graph-replayed hints: 2.30 ms/VU"),
     ("b-ligero2", "r20260922-074449-a913", "4096 VUs as 49 x 85"),
     "pre-fix 2.30 ms/VU at 85 VUs (no run id) vs post-fix 1.98 ms/VU at 4096 = 49 x 85: not slower; every 4096-VU B-Ligero row is post-fix"),
    ("A run-vu chain spec from bundle (red-team-v2 #9, fixed ff651a8): verifier-side only",
     ("a-v2", "r20260922-081106-6067", "A on checker v2 (Goldilocks), 4096 VUs"),
     None, "verifier-owned v2_chain_spec() at ff651a8, 15/15 tests, forgery fixture refused (r20260922-095128-05ac); prover work unchanged by construction, not re-timed"),
    ("B-AIR F1/F2 statement binding (lane/soundness-fixes, UNMERGED)",
     ("b-chain", "r20260922-060503-6847", "BabyBear AIR at 2^-128 proven"),
     None, ("lane ledger r20260922-080527-f30c on vy-cpu3 (a ~3x slower box): 544.8 s bound vs 426-485 s unbound on the same box, "
            "six alternating 512-VU runs flip sign (box swings 40%): cost not resolvable above noise; adds n/|F| = 2^-105 at BabyBear^4")),
)


# --- census (per transition unit unless stated; every cell names its source) --------------------------------------------

CENSUS: tuple[tuple[str, ...], ...] = (
    # design, variant, mul, lin, lookups, hints, committed cols or rows / unit, committed elements / VU, committed bytes / VU, hash bytes / batch, source
    ("checker (reference arithmetisation)", "v1", "358", "137", "306", "225", "100 materialised cols", "--", "--", "--", "checker_min.md 1, 3.1"),
    ("checker", "v2", "33", "9", "89", "15", "45 materialised cols", "--", "--", "--", "checker_min.md 3.1 (z3 10/10 UNSAT)"),
    ("A GKR, Goldilocks^3", "v1", "358", "137", "306", "--", "468 cols", "44,937 (184.1M/4096)", "360 kB (8 B/el)", "6.19 GB (est.)", "a-chain r20260922-055358-9f6b; gkr/PROTOCOL.md 9.4"),
    ("A GKR, BabyBear^6", "v1", "322", "322", "382", "--", "444 cols (a-gpu circuit 452, 609 lookups)", "42,651 (174.7M/4096); a-gpu 177.8M", "171 kB (4 B/el)", "--", "a-babybear r20260922-063837-b26c; a-gpu r20260922-065037-41f8"),
    ("A GKR, Goldilocks^3", "v2", "26", "9", "86", "12", "264 cols; tables implicit, 5.14M multiplicities/proof", "26,611 (109.0M/4096 incl. tables)", "213 kB (8 B/el); 106 kB at 4 B", "--", "a-v2 r20260922-081106-6067; gkr/PROTOCOL.md 14.1"),
    ("B-AIR limbs, Goldilocks", "v1", "402 constraints", "--", "0 (2-bit limbs)", "--", "2,844 cols", "273k", "2.18 MB", "--", "b-chain r20260922-055326-699d"),
    ("B-AIR limbs, BabyBear (after fix)", "v1", "402+32 constraints", "--", "0", "--", "2,930 cols", "281k", "1.13 MB", "--", "b-lookup r20260922-065546-a242"),
    ("B-AIR + LogUp, BabyBear^4", "v1", "402 constraints", "--", "391 queries/row", "--", "863 cols (401 main + 456 aux + 6 pre)", "82.8k", "331 kB", "--", "b-lookup r20260922-070654-177a; direct/PROTOCOL.md 14"),
    ("B-AIR + LogUp, BabyBear^4 / ^6", "v2", "88 constraints", "--", "105 queries/row (27 helpers)", "15", "405 / 463 cols", "38.9k / 44.4k", "155.5 / 177.8 kB", "--", "b-air-v2 r20260922-085312-82a9 / -090904-cd75; V2_AIR.md 3"),
    ("B-Ligero torch prover (chain rows), BabyBear", "v1", "347 lin / 3,387 quad constraints", "--", "as bit + selector rows", "92 hint rows", "3,516 rows/unit (2,401 bit, 610 selector)", "337k (3516 x 96)", "1.35 MB (4 B/el, before RS rate 1/4)", "20.4 GB codeword at 310,602-row model shape", "ligero/PROTOCOL.md 1, 7b; hash_gpu.md"),
    ("B-Ligero model shape (b-v2 layout)", "v2", "--", "43.4M lin / 13.3k quad rows per batch", "helper triples in F_c", "--", "104,217 Ligero rows (v1 310,602)", "--", "388.8 kB (v1 1,164.7 kB)", "6.8 GB (104,217 x 65,536 B)", "b-v2 r20260922-075021-2b1b; V2_SHAPE.md"),
    ("C QuickSilver (dietmc)", "v1", "501,393 mults/VU", "--", "one-hot selectors", "--", "--", "993,095", "8.50 MB P->V (F61p) / 54.4 MB (F384p)", "n/a (IT-MACs)", "c-vole r20260922-053613-d110 / -061644-5d62"),
    ("C QuickSilver compressed", "v2", "73,066 mult checks/VU", "--", "36.6k 10-bit logUp chunks + 2.1k table queries", "--", "--", "101,009", "0.865 MB (F61p) / 5.10 MB (F384p)", "n/a", "c-compress r20260922-080520-71b6; vole/DECISION.md 2"),
)


ZK_STATUS: tuple[tuple[str, ...], ...] = (
    # candidate, HVZK, malicious-verifier ZK, audit, label as it stands, notes
    ("A (GKR)", "none implemented", "none", "designed (PROTOCOL.md 5.5, +0.002%, +1 depth); not built", "NON_ZK_PROOF_DIAGNOSTIC",
     "zk-construction model +1.43% HVZK; no --zk in any A prover"),
    ("B-Ligero (torch, 8b/8c)",
     ("IMPLEMENTED BUT BROKEN as a construction: columns, w, v, h masked correctly; the chain message q is unmasked on H -- "
     "16150/16150 accumulators recovered from the transcript (red-team-zk F1 ACCEPTED-LEAK, r20260922-100305-2b13)"),
     ("HM96 coin commitment HOLDS with a repair (2^-128 bind, 2^-191 hide); the 8c simulator exists only in the programmable-RO model (F2); "
     "the measured mode is Fiat-Shamir with a prover-chosen nonce, priced with the interactive bound (F3: FS needs t~279)"),
     ("red-team-zk VERDICT: COMPLETE_ZK_BACKEND -> NON_ZK_PROOF_DIAGNOSTIC for b-zk r20260922-092404-2a1b (REDTEAM_ZK_URGENT.md, redteam_zk.md); "
     "repair lane b-zk-fix PENDING (b_zk_fix.md / ledger b-zk-fix.jsonl)"),
     "NON_ZK_PROOF_DIAGNOSTIC (downgraded; ledger rows still say COMPLETE_ZK/HVZK_BACKEND)",
     ("for the CURRENT statement (operands a, b and words y16 public via statement_digest) privacy is vacuous: no privacy label is meaningful "
     "until operands are committed (auth-integration LEAF_V2)")),
    ("B-AIR (Plonky3)", "none implemented (modelled +1-1.6%)", "none; as run Fiat-Shamir depth 1", "--", "NON_ZK_PROOF_DIAGNOSTIC",
     "F19 weak-FS residual fixed only on lane/b-bind"),
    ("C (QuickSilver)", "native (IT-MAC)", "native in the F_sVOLE-hybrid, designated verifier", "not red-teamed",
     "COMPLETE_ZK_BACKEND (designated-verifier) -- the only candidate holding a ZK label", "assumption stack LPN + base OT + CRH; 20.88 GB/batch at 2^-128"),
)


PENDING_LANES: tuple[tuple[str, ...], ...] = (
    # lane, what it resolves, file(s) it will write, state at merge time
    ("b-zk-fix", "B-Ligero ZK repair: one-line q mask (AHIV22 u_add), interactive challenge derivation from coins alone, explicit interactive vs Fiat-Shamir modes each with the accountant's bound", "b_zk_fix.md / ledger b-zk-fix.jsonl", "running; no branch commits yet"),
    ("b-bind", "B-AIR statement binding: soundness-fixes landing (F1/F2 limb path) + LogUp path U1/U2 + F19", "lane/b-bind (24 commits ahead), ledger b-bind", "running on vy-cpu3"),
    ("a-protocol-diff", "localise the Rust-verifier / Python-verifier divergence on a-gpu2 proofs (W=22 verifier rejects at LogUp POW level 2)", "backends/gkr/verifier/PROTOCOL_DIFF.md (proposed; 1 commit ahead, no .md yet)", "running"),
    ("a-verifier-2", "F1 triage: why the a-gpu kernel path at c4d472a was rejected by both verifiers on vy-g5 while gate4 accepted it on vy-g4", "backends/gkr/gpu/F1_TRIAGE.md (11 commits ahead)", "running"),
    ("b-verifier", "independent Rust verifier for B-Ligero", "backends/ligero-verify/ + DISCREPANCIES.md, reports/b_verifier.md (9 commits ahead)", "running"),
    ("a-logup-union", "A LogUp union of tables (epilogue R8/R15/R16 tables not in the unit list)", "backends/gkr/gpu/HANDOFF_UNION.md, REDTEAM_URGENT_epilogue_tables.md (12 commits ahead)", "running"),
)


# --- rendering ---------------------------------------------------------------------------------------------------------


def _f(x: float | None, nd: int = 2) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def _sci(x: float | None) -> str:
    return "--" if x is None else f"{x:.2e}"


def _mb(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x / 1e9:.2f} GB" if x >= 1e9 else f"{x / 1e6:.1f} MB"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def _row_values(r: Row, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if r.key is None:
        m = dict(r.manual)
        m["B"] = m.get("B", 4096)
        m["seconds_per_vu"] = m["prove_s"] / m["B"]
        return m
    e = _find(entries, *r.key)
    B = e["B"] or 1
    spv = r.prove_s / B if r.prove_s is not None else e["seconds_per_vu"]
    prove = r.prove_s if r.prove_s is not None else (spv * B if spv else None)
    return {"prove_s": prove, "seconds_per_vu": spv, "overhead": e["overhead"] if r.prove_s is None else (spv / (NATIVE_PS * 1e-12) if spv else None),
            "proof_class": e["proof_class"], "hardware": e["hardware"], "security": e["security_target"], "run_id": e["run_id"],
            "lane": e["lane"], "B": e["B"]}


def _design(name: str, v: dict[str, Any], r: Row) -> latency.Design | None:
    if v.get("prove_s") is None or r.verify_s is None or r.depth is None or r.proof_bytes is None:
        return None
    return latency.Design(name=name, prove_s=v["prove_s"], verify_s=r.verify_s, depth=r.depth, bytes_pv=r.proof_bytes,
                          mem_per_batch_bytes=0.0, mem_capacity_bytes=1.0, B=v["B"], hardware=v["hardware"], proof_class=v["proof_class"], source="")


def render_like_for_like(entries: Sequence[dict[str, Any]]) -> str:
    header = ["design", "hardware", "B", "prover s", "ms/VU", "overhead x native", "verifier s", "proof", "depth",
              *[f"e2e @{int(r)} ms" for r in RTTS_MS], "soundness (as stated)", "label", "verified by", "lane / run id", "flags"]
    rows = []
    for r in LIKE_FOR_LIKE:
        v = _row_values(r, entries)
        d = _design(r.design, v, r)
        e2e = [_f(latency.e2e(d, rtt, LINK_GBPS)["total"]) if d else "--" for rtt in RTTS_MS]
        spv = v.get("seconds_per_vu")
        name = ("**PENDING** (" + r.pending + ") " if r.pending else "") + r.design
        depth = ("--" if r.depth is None else f"{r.depth}") + (f" ({r.depth_note})" if r.depth_note else "")
        verify = _f(r.verify_s) + (f" ({r.verify_note})" if r.verify_note else "")
        label = r.label_override or v["proof_class"]
        rows.append([name, v["hardware"], f"{v['B']}", _f(v.get("prove_s")), _f(spv * 1e3, 3) if spv else "--", _sci(v.get("overhead")),
                     verify, _mb(r.proof_bytes), depth, *e2e, v["security"], label, r.verified or "--", f"{v['lane']} / {v['run_id']}", r.flags or ""])
    note = (f"\nEnd-to-end = prove + verify + depth x RTT + proof bytes / {LINK_GBPS:.0f} Gb/s (`bench.latency.e2e`; the a-verifier "
            "lane's convention, `latency.md` used 25 Gb/s). Overhead = prover s/VU over 9.85 ps. Sub-batched rows: the soundness "
            "is the union bound over the sub-batches and the prover time is the sum.")
    return _table(header, rows) + note


def render_repriced(entries: Sequence[dict[str, Any]]) -> str:
    header = ["bucket re-price (ARITHMETIC_DIAGNOSTIC, not a proof)", "hardware", "overhead x native", "us/VU (derived)", "measured share", "lane / run id", "omitted"]
    rows = []
    for name, key, note in REPRICED:
        e = _find(entries, *key)
        us = e["overhead"] * NATIVE_PS * 1e-6 if e["overhead"] else None
        rows.append([name, e["hardware"], _sci(e["overhead"]), _f(us, 1), note, f"{e['lane']} / {e['run_id']}", "; ".join(e["omitted"])[:160]])
    return _table(header, rows)


def render_soundness(entries: Sequence[dict[str, Any]]) -> str:
    header = ["ledger row (lane / run id)", "label", "note (verbatim, truncated)"]
    rows = []
    for key in SOUNDNESS_KEYS:
        e = _find(entries, *key)
        rows.append([f"{e['lane']} / {e['run_id']}", e["label"], e["note"][:260] + ("..." if len(e["note"]) > 260 else "")])
    return _table(header, rows)


def render_fixed(entries: Sequence[dict[str, Any]]) -> str:
    header = ["fix", "before (lane / run id): ms/VU at B, soundness", "after (lane / run id): ms/VU at B, soundness", "delta / reading"]
    rows = []
    for name, before, after, note in FIXED_PAIRS:
        b = _find(entries, *before)
        bs = f"{b['lane']} / {b['run_id']}: {_f((b['seconds_per_vu'] or 0) * 1e3, 3)} ms/VU at B={b['B']}, {b['security_target'][:40]}"
        if after is None:
            as_ = "no ledger row on main"
        else:
            a = _find(entries, *after)
            as_ = f"{a['lane']} / {a['run_id']}: {_f((a['seconds_per_vu'] or 0) * 1e3, 3)} ms/VU at B={a['B']}, {a['security_target'][:40]}"
        rows.append([name, bs, as_, note])
    return _table(header, rows)


def render_census(entries: Sequence[dict[str, Any]]) -> str:
    del entries
    header = ["design", "checker", "field mul / unit", "linear / unit", "lookups / unit", "hints / unit", "committed cols or rows / unit",
              "committed elements / VU", "committed bytes / VU", "hash bytes / batch", "source"]
    return _table(header, CENSUS)


def render_zk(entries: Sequence[dict[str, Any]]) -> str:
    del entries
    header = ["candidate", "HVZK", "malicious-verifier ZK (HM96 coin commitment)", "audit", "label", "notes"]
    return _table(header, ZK_STATUS)


def render_pending(entries: Sequence[dict[str, Any]]) -> str:
    del entries
    header = ["lane (not merged)", "resolves", "file(s) it will write", "state"]
    return _table(header, PENDING_LANES)


def _merge_order(lanes: Iterable[str]) -> list[str]:
    """Lanes in the order they first appear on ``git log --first-parent --reverse main`` (merge commits say
    ``merge lane/<name>``; the sp1 and a-kernel lanes were committed directly and are matched by name in the subject).
    Lanes never mentioned are appended in ledger first-timestamp order by the caller."""

    try:
        log = subprocess.run(["git", "log", "--first-parent", "--reverse", "--format=%s", "main"], check=True, capture_output=True,
                             text=True, cwd=REPORTS).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    order: list[str] = []
    names = sorted(set(lanes), key=len, reverse=True)
    for line in log.splitlines():
        m = re.search(r"lane/([\w-]+)", line)
        if m and m.group(1) in names and m.group(1) not in order:
            order.append(m.group(1))
            continue
        for ln in names:
            if ln not in order and re.search(rf"(?:^|lane/|backends/){re.escape(ln)}(?![\w-])", line):
                order.append(ln)
                break
    return order


def render_lanes(entries: Sequence[dict[str, Any]]) -> str:
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_lane.setdefault(e["lane"], []).append(e)
    order = [ln for ln in _merge_order(by_lane) if ln in by_lane]
    order += sorted((ln for ln in by_lane if ln not in order), key=lambda ln: min(e["ts"] for e in by_lane[ln]))
    header = ["#", "lane (merge order)", "track", "entries", "headline ledger row (last breakthrough, else last entry)", "class", "overhead", "run id"]
    rows = []
    for i, ln in enumerate(order, 1):
        es = sorted(by_lane[ln], key=lambda e: e["ts"])
        head = [e for e in es if e["breakthrough"]] or es
        h = head[-1]
        rows.append([str(i), ln, h["track"], str(len(es)), h["label"], h["proof_class"], _sci(h["overhead"]), h["run_id"] or "--"])
    return _table(header, rows)


TABLES = {
    "like_for_like": render_like_for_like,
    "repriced": render_repriced,
    "soundness_ledger": render_soundness,
    "fixed_vs_forgeable": render_fixed,
    "census": render_census,
    "zk_status": render_zk,
    "pending": render_pending,
    "lanes": render_lanes,
}


def render(name: str, directory: Path = ledger.DIR) -> str:
    return TABLES[name](_entries(directory))


def render_all(directory: Path = ledger.DIR) -> dict[str, str]:
    entries = _entries(directory)
    return {name: fn(entries) for name, fn in TABLES.items()}


def update_markdown(text: str, tables: dict[str, str]) -> tuple[str, list[str]]:
    """Replace the content between ``<!-- BEGIN TABLE:x -->`` and ``<!-- END TABLE:x -->`` for every table found."""

    found = []

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in tables:
            return m.group(0)
        found.append(name)
        return f"<!-- BEGIN TABLE:{name} -->\n{tables[name]}\n<!-- END TABLE:{name} -->"

    return re.sub(r"<!-- BEGIN TABLE:([\w-]+) -->\n.*?\n<!-- END TABLE:\1 -->", sub, text, flags=re.DOTALL), found


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger-dir", type=Path, default=ledger.DIR)
    ap.add_argument("--write", action="store_true", help="refresh the tables inside the draft and the appendix")
    ap.add_argument("--check", action="store_true", help="exit 1 if the drafts' tables differ from the ledger")
    ap.add_argument("--table", choices=sorted(TABLES), help="print one table")
    ns = ap.parse_args(argv)
    tables = render_all(ns.ledger_dir)
    if ns.table:
        print(tables[ns.table])
        return 0
    if ns.write or ns.check:
        stale = 0
        for path in (DRAFT, APPENDIX):
            if not path.exists():
                continue
            old = path.read_text()
            new, found = update_markdown(old, tables)
            if new != old:
                stale += 1
                if ns.write:
                    path.write_text(new)
            print(f"{path.name}: tables {', '.join(found) or 'none'}{' (refreshed)' if ns.write and new != old else ''}")
        return 1 if (ns.check and stale) else 0
    for name, t in tables.items():
        print(f"\n### {name}\n\n{t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
