"""Regenerate every table of ``backends/numerical/reports/DECISION_DRAFT.md`` (and its appendix) from the overhead
ledger (the frozen one in the evidence store, ``ledger.frozen_dir()``, unless ``--ledger-dir`` names another) and the
per-lane reports.

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

REPORTS = Path(__file__).resolve().parents[3] / "reports"   # untracked (.gitignore): the drafts are working copies
DRAFT = REPORTS / "DECISION_DRAFT.md"
APPENDIX = REPORTS / "DECISION_DRAFT_appendix.md"
LINK_GBPS = 10.0
RTTS_MS = (0.0, 1.0, 10.0, 50.0)
NATIVE_PS = 9.85  # ps per K=1536 VU, contract.NATIVE_REFERENCE


# --- ledger lookup ----------------------------------------------------------------------------------------------------


def _entries(directory: Path | None = None) -> list[dict[str, Any]]:
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
        verify_s=6.06, proof_bytes=33.9e6, depth=438, verify_note="Python", verified="own Python verifier, 5/5 reps; Rust not run on this bundle",
        flags="median of 5 warm reps [7.54, 5.02, 5.03, 4.40, 4.20]; spread is host-side Merkle openings 1.4-3.9 s"),
    Row("B-Ligero v1 (b-ligero2 prover), non-ZK -- gate4 same-card row",
        ("gate4-h100", "r20260922-092841-94ac", "B e2e non-ZK 4096 VUs as 25x170 (b-ligero2)"),
        verify_s=0.816, proof_bytes=110.8e6, depth=2, verified="own Python verifier, 3/3 reps",
        flags="25 x 170 VUs, t=196; hash + SHAKE expander host placeholders; pre-b-zk-fix protocol (FS run priced with the interactive bound, red-team-zk F3)"),
    Row("B-Ligero v1 (b-ligero2 --zk), 8b masking -- gate4 same-card row",
        ("gate4-h100", "r20260922-092841-94ac", "B e2e HVZK 4096 VUs as 25x170 (b-ligero2 --zk)"),
        verify_s=0.880, proof_bytes=122.3e6, depth=2, verified="own Python verifier, 3/3 reps",
        label_override="NON_ZK_PROOF_DIAGNOSTIC (red-team-zk F1: 8b leaves q unmasked on H; ledger says COMPLETE_HVZK_BACKEND)",
        flags="zk masks 0.40 s host SHAKE = 14%; superseded as a ZK claim by the b-zk-fix rows"),
    # --- A: the independently verified GPU number and the rest of A's GPU history ---
    Row("A: a-gpu2 v6 (packed GKR + graphed LogUp + INT8 opening), v1, BabyBear^6 X^6-22, H100 vy-g5 -- INDEPENDENTLY VERIFIED",
        ("a-protocol-diff", "r20260922-104038-eabb", "A e2e a-gpu2 + Rust verifier (fixed) RTT0 5.19 s"),
        verify_s=3.282, proof_bytes=33.9e6, depth=438, prove_s=1.875, verify_note="Rust 16T on the pod's Xeon 8480+; 1.00 s at 12T on vy-cpu2 (a-verifier-2 r20260922-105647-c947)",
        verified="INDEPENDENT Rust verifier (887fc5c, no shared code) accepts after the sumcheck variable-order fix (PROTOCOL 4.2); 1320/1320 mutations + 44/44 negatives rejected; own Python verifier accepts",
        flags="different pod/run than gate4's A row; a-gpu2's own 1.790 s (r20260922-094040-9e76) was Python-only until a-protocol-diff; README claim 'transcript identical to v2e' was false"),
    Row("A: a-gpu2 prover + 1.00 s Rust verifier (a-verifier-2 e2e composition)",
        ("a-verifier-2", "r20260922-105647-c947", "A-GPU e2e w/ Rust verifier RTT0 2.82 s"),
        verify_s=1.000, proof_bytes=33.9e6, depth=438, prove_s=1.790, verify_note="Rust 12T vy-cpu2 (was 3.02 s)",
        verified="prover time from r20260922-094040-9e76 (Python-verified at the time; the same tree's B=4096 proofs are Rust-accepted since a-protocol-diff); verifier time measured on the torch-path proof",
        flags="composition of two measured numbers, not one run"),
    Row("A: a-gpu v2e kernel path, H100 vy-g5 (a-gpu's own record)",
        ("a-gpu", "r20260922-074455-d0b6", "full VU B=4096 H100 v2e"),
        verify_s=3.017, proof_bytes=33.9e6, depth=438, verify_note="Rust 12T; Python 7.1 s",
        verified="own Python verifier; a-verifier F1 rejection on vy-g5 = ENVIRONMENT fault (a-verifier-2: kernel path byte-identical to torch on a clean 4090, Rust-accepted; a-kernel-a100: stale Triton cache served W=31 binaries)",
        flags="code sound; the run itself is not reproducible from a poisoned pod; superseded by gate4 5.02 s and a-gpu2 1.875 s"),
    Row("A: a-gpu torch path (VERITY_GPU_TORCH_ONLY=1), H100 vy-g5",
        ("a-verifier", "r20260922-093721-1330", "A-GPU e2e w/ Rust verifier"),
        verify_s=3.017, proof_bytes=33.9e6, depth=438, prove_s=25.5, verify_note="Rust 12T (1.00 s after a-verifier-2)",
        verified="INDEPENDENT Rust + Python: accepted", flags="prover 25.5 s median of 3 (r20260922-091240-4bf9); bypasses every gpu/kernels.py kernel"),
    Row("A: v1 kernel path on the A100, merged main (FIELD_KEY fix), a-kernel-a100-2 -- SAME-CARD A100 PAIR, Rust-verified",
        ("a-kernel-a100", "r20260922-115019-03ea", "v1 GPU kernel path B=4096 A100 on merged main, Rust-verified"),
        verify_s=1.94, proof_bytes=33.87e6, depth=438, verify_note="Rust 12T on the pod; Python 0.83-0.92 s",
        verified="INDEPENDENT Rust verifier (main, order-fixed) ACCEPTED; own Python 3/3",
        flags="2.969 / 2.962 / 2.970 s; supersedes the 73b9f75 row (2.979 s Python-only, r20260922-112320-3ceb); 1.66x the H100 a-gpu2 1.79 s"),
    Row("A: checker-v2 (limb epilogue) kernel path on the A100, merged main, a-kernel-a100-2 -- SAME-CARD A100 PAIR, Rust-verified",
        ("a-kernel-a100", "r20260922-115019-03ea", "v2 GPU checker-v2 B=4096 A100 on merged main, Rust-verified"),
        verify_s=2.02, proof_bytes=21.13e6, depth=None, depth_note="not reported at 4096 (a-gpu-v2: 279 at B=64)", verify_note="Rust 12T; Python 0.99-1.03 s",
        verified="INDEPENDENT Rust verifier ACCEPTED (a-verifier-2 v2 support); own Python 3/3",
        flags="2.495 / 2.360 / 2.450 s; re-records a-gpu-v2's 5.62 / 5.73 s (W=31 tree); faster mostly in lookup (0.80 vs 2.78 s) -- a-kernel-a100 asks a-gpu-v2 to confirm which LogUp path its rows used; v2/v1 same card = 0.83x"),
    Row("A: checker v2 + limb-form epilogue, BabyBear^6, a-gpu-v2, A100 vy-a100b -- field X^6-31 tree (SUPERSEDED)",
        ("a-gpu-v2", "r20260922-095013-db80", "v2 GPU BabyBear^6 limb epilogue full VU B=4096 A100"),
        verify_s=8.1, proof_bytes=21.1e6, depth=None, verify_note="Python", depth_note="279 at B=64; not reported at 4096",
        verified="own Python verifier 4096/4096; 52/52 + 30 + 4 + 15/15 negatives rejected; no Rust verifier for this field",
        flags="SUPERSEDED by r20260922-115019-03ea: measured on a tree with W=31 (a_kernel_a100.md: stands as a timing over F_p[X]/(X^6-31), not the protocol field); recorded with the epilogue-table hole (a-logup-union); same-box v1 control 7.93 s (r20260922-091544-4e02)"),
    Row("A: GKR checker v2, Goldilocks^3, CPU Rust (a-v2) -- recorded with the epilogue-table hole",
        ("a-v2", "r20260922-081106-6067", "A on checker v2 (Goldilocks), 4096 VUs"),
        verify_s=12.2, proof_bytes=41.5e6, depth=328, verified="own Rust verifier",
        flags="a-logup-union: lookups proved for segment 0's tables only, epilogue R16/R15 unchecked -> any public word forgeable; fixed (fix cost <1%); superseded by a-logup-union's 67.36 s per-table row on vy-cpu (different box)"),
    Row("A: checker v2, per-table logUp, epilogue tables proved (a-logup-union control), CPU vy-cpu",
        ("a-logup-union", "r20260922-102556-cdcc", "A v2, per-table logUp (control), 4096 VUs"),
        verify_s=14.87, proof_bytes=41.57e6, depth=304, verified="own Rust verifier 4096/4096",
        flags="first v2 CPU row with the epilogue hole closed; union variant 70.08 s, depth 355, coins 3393 -> 761 (same run)"),
    Row("A: GKR v1 BabyBear^6, CPU Rust (a-babybear)",
        ("a-babybear", "r20260922-065112-d687", "BabyBear^6 full VU B=4096"),
        verify_s=45.0, proof_bytes=33.39e6, depth=381, verified="own Rust verifier", depth_note="derived; latency lane measured 355 trips on Goldilocks"),
    # --- B-Ligero: the audited ZK rows (b-zk-fix) and the downgraded predecessors (b-zk) ---
    Row("B-Ligero ZK, INTERACTIVE (HM96 coins, q masked; F1/F2 fixed), L40S (b-zk-fix) -- audited label",
        ("b-zk-fix", "r20260922-105905-2021", "B-Ligero ZK interactive (F1/F2 fixed)"),
        verify_s=0.967, proof_bytes=122.3e6, depth=3, verify_note="Python; Rust 1.24 s at 8T laptop (b-verifier-2)",
        verified="own Python verifier; INDEPENDENT Rust (ligero-verify) 108/108 honest, 264/264 neg+mut rejected; red-team-zk2 EARNED (mask rank 197/197, 0/16150 accumulators, coins-alone soundness holds)",
        flags="assumptions [hash, random-oracle (Merkle hiding)] (G3); D7: interactive proof FILES are not transferable (49/49 forged files accepted by both file verifiers) -- this label holds only with a LIVE verifier, like C"),
    Row("B-Ligero ZK, FIAT-SHAMIR (q x eps: t=288, D=7), L40S (b-zk-fix) -- audited label, transferable proof",
        ("b-zk-fix", "r20260922-105905-2021", "B-Ligero ZK Fiat-Shamir"),
        verify_s=1.071, proof_bytes=165e6, depth=1, verify_note="Python; Rust 1.65 s at 8T laptop",
        verified="own Python verifier; INDEPENDENT Rust 108/108 / 264/264; red-team-zk2 EARNED (rank 288/288, FS accounting reproduces 2^-128.04, t=288 minimal)",
        flags="the transferable B-Ligero ZK row; assumptions [hash, random-oracle]; D1 open (Python never checks params meet the target)"),
    Row("B-Ligero non-ZK, interactive control, L40S (b-zk-fix)",
        ("b-zk-fix", "r20260922-105905-2021", "B-Ligero non-ZK interactive control"),
        verify_s=0.967, proof_bytes=110.8e6, depth=3, verified="own Python; Rust 108/108 set", flags="ZK cost over this control +2%; the fastest 4096-VU prover on any card (1.366 s)"),
    Row("B-Ligero non-ZK, Fiat-Shamir control (t=285, D=7), L40S (b-zk-fix)",
        ("b-zk-fix", "r20260922-105905-2021", "B-Ligero non-ZK Fiat-Shamir ctrl"),
        verify_s=1.049, proof_bytes=149.8e6, depth=1, verified="own Python; Rust 108/108 set", flags="FS re-parameterisation: +5% prover, +8% verifier, +35% transcript vs interactive"),
    Row("B-Ligero: b-zk --zk + HM96 coins, fused tests, L40S (b-zk, PRE-FIX) -- downgraded",
        ("b-zk", "r20260922-092404-2a1b", "B-Ligero full ZK (HM96), fused tests"),
        verify_s=0.96, proof_bytes=122.3e6, depth=3, verified="own Python; b-verifier Rust 81/81 honest accepted (soundness, not privacy)",
        label_override="NON_ZK_PROOF_DIAGNOSTIC (red-team-zk VERDICT r20260922-100305-2b13; ledger says COMPLETE_ZK_BACKEND)",
        flags="F1 q leak real for this code (16150/16150 accumulators); superseded by the b-zk-fix rows"),
    Row("B-Ligero at the checker-v2 shape, eager torch, RTX 4090 (b-e2e-v2)",
        ("b-e2e-v2", "r20260922-094956-5453", "B-Ligero v2 e2e GPU, 4096 VUs = 7 proofs x 682"),
        verify_s=0.40 * 7, proof_bytes=9.2e6 * 7, depth=2, verify_note="0.40 s x 7 proofs", verified="own verifier 7/7; 52/52 + 39/39 rejected",
        flags="INCOMPLETE_CHECKER caveat (Python limb rows not z3-verified); kernels not wired: v2 real SLOWER than v1 real"),
    # --- B-AIR: bound rows supersede the unbound ones ---
    Row("B-AIR: Plonky3 LogUp AIR, checker v2, BabyBear^6, CPU Rust vy-cpu2 (b-air-v2)",
        ("b-air-v2", "r20260922-090904-cd75", "checker v2 BabyBear^6 AIR + LogUp"),
        verify_s=0.731, proof_bytes=2.03e6, depth=26, verified="Plonky3's own verifier; none independent; b-bind gate r20260922-111343-3ed9 re-asserts v2 BB^4/BB^6 binding at 64 VUs",
        flags="statement-bound at record time (v2 header); b-bind's vendored p3-fri changes the transcript (depth +1, upstream proofs no longer verify) -- not re-timed at 4096; v2 header does not absorb the LogUp table digest (tables bound via the recomputed preprocessed commitment)"),
    Row("B-AIR: checker v2, BabyBear^4, CPU Rust vy-cpu2 (b-air-v2)",
        ("b-air-v2", "r20260922-085312-82a9", "checker v2 BabyBear AIR + LogUp tuple lookups"),
        verify_s=0.455, proof_bytes=1.95e6, depth=26, verified="Plonky3's own verifier"),
    Row("B-AIR: v1 + LogUp, BabyBear^4, statement-BOUND, CPU Rust vy-cpu3 (b-bind) -- supersedes b-lookup's unbound row",
        ("b-bind", "r20260922-094310-af1a", "B LogUp bound: 4096 VUs BB^4"),
        verify_s=1.1, proof_bytes=2.37e6, depth=27, depth_note="26 + 1 for the vendored p3-fri order", verified="Plonky3's own verifier + b-bind negatives (52/52 x 5 modes, battery 0 FORGED, r20260922-111343-3ed9)",
        flags="prove 199.4 / 182.8 s bound vs 197.6 / 201.8 s unbound back to back (binding free within ~10% swing); vy-cpu3 is ~3.2x slower than vy-cpu2 (unbound 60.1 s there); BabyBear^5 213-218 s at 2^-136"),
    Row("B-AIR: v1 + LogUp range lookups, BabyBear^4, CPU Rust vy-cpu2 (b-lookup) -- RECORDED UNBOUND",
        ("b-lookup", "r20260922-070654-177a", "BabyBear AIR + LogUp range lookups"),
        verify_s=0.441, proof_bytes=2.27e6, depth=26, verified="Plonky3's own verifier",
        flags="statement-unbound at record time (red-team-3 U1/U2); bound since b-bind 8437aa5; timing stands (binding free within noise), soundness label at ^4 is 2^-105 once bound"),
    Row("B-AIR: v1 limbs, BabyBear^4, statement-BOUND (F1-F3), CPU Rust vy-cpu3 16T (soundness-fixes) -- supersedes b-chain/b-lookup limb rows",
        ("soundness-fixes", "r20260922-080527-f30c", "B full VU, statement-bound (SOUNDNESS FIX F1-F3)"),
        verify_s=None, proof_bytes=None, depth=None, verified="Plonky3's own verifier; gate r20260922-073904-7a6c negatives",
        flags="544.8 s on vy-cpu3 vs 426-485 s unbound on the same box across alternating runs: cost not resolvable above the box's 40% swing; +2^-105 term at ^4"),
    Row("B-AIR: v1 limbs, BabyBear^4, CPU Rust vy-cpu2 (b-lookup, after lead.window fix) -- RECORDED UNBOUND",
        ("b-lookup", "r20260922-065546-a242", "SOUNDNESS FIX: LT_DIRECT"),
        verify_s=0.426, proof_bytes=3.47e6, depth=22, verified="Plonky3's own verifier",
        flags="statement-unbound at record time (red-team-vu F1/F2); bound since b-bind; timing stands"),
    # --- C, SP1 ---
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
    ("b-bind", "r20260922-091130-525e", "SOUNDNESS FIX: B LogUp path statement-bound"),
    ("b-bind", "r20260922-094310-af1a", "SOUNDNESS FIX: F3 residual"),
    ("soundness-fixes", "r20260922-080527-f30c", "B full VU, statement-bound"),
    ("a-protocol-diff", "r20260922-104038-eabb", "A e2e a-gpu2 + Rust verifier (fixed)"),
    ("a-verifier-2", "r20260922-100739-bc57", "F1: kernel path bit-exact"),
    ("a-kernel-a100", "r20260922-112320-3ceb", "full VU B=4096 A100: packed path, Triton cache-key fix"),
    ("red-team-zk2", "r20260922-112029-2db3", "EARNED: COMPLETE_ZK_BACKEND interactive"),
    ("red-team-zk2", "r20260922-112029-2db3", "EARNED: COMPLETE_HVZK_BACKEND fiat-shamir"),
    ("red-team-zk2", "r20260922-112029-2db3", "G3 interactive ZK row"),
    ("b-verifier-2", "r20260922-111531-e2d3", "b-verifier-2.reject_rate"),
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
     "pre-fix 2.30 ms/VU at 85 VUs (no run id) vs post-fix 1.98 ms/VU at 4096 = 49 x 85: not slower"),
    ("B-AIR F1/F2 statement binding, limb path (soundness-fixes, merged via b-bind 8437aa5)",
     ("b-chain", "r20260922-060503-6847", "BabyBear AIR at 2^-128 proven"),
     ("soundness-fixes", "r20260922-080527-f30c", "B full VU, statement-bound (SOUNDNESS FIX F1-F3)"),
     "different boxes (vy-cpu2 168 s vs vy-cpu3 544.8 s, ~3.2x slower box); same-box alternating 512-VU runs flip sign: cost not resolvable above the 40% swing"),
    ("B-AIR U1/U2 statement binding, LogUp path (b-bind)",
     ("b-lookup", "r20260922-070654-177a", "BabyBear AIR + LogUp range lookups"),
     ("b-bind", "r20260922-094310-af1a", "B LogUp bound: 4096 VUs BB^4"),
     "bound vs unbound back to back on vy-cpu3: 199.4 / 182.8 s vs 197.6 / 201.8 s -- binding free within the ~10% repeat swing (vy-cpu2 unbound 60.1 s is a 3.2x faster box)"),
    ("A run-vu chain spec from bundle (red-team-v2 #9, fixed ff651a8): verifier-side only",
     ("a-v2", "r20260922-081106-6067", "A on checker v2 (Goldilocks), 4096 VUs"),
     None, "verifier-owned v2_chain_spec() at ff651a8, 15/15 tests, forgery fixture refused (r20260922-095128-05ac); prover work unchanged by construction, not re-timed"),
    ("A v2 epilogue tables never proved (a-logup-union): lookups for segment 0's tables only",
     ("a-v2", "r20260922-081106-6067", "A on checker v2 (Goldilocks), 4096 VUs"),
     ("a-logup-union", "r20260922-102556-cdcc", "A v2, per-table logUp (control), 4096 VUs"),
     "different boxes (vy-cpu2 54.2 s shared vs vy-cpu 67.36 s); fix adds 2 logUp instances + 98,304 multiplicities: < 1% by count (REDTEAM_URGENT_epilogue_tables.md)"),
    ("B-Ligero q leak F1 + simulator F2 + FS pricing F3 (red-team-zk -> b-zk-fix, re-audited by red-team-zk2)",
     ("b-zk", "r20260922-092404-2a1b", "B-Ligero full ZK (HM96), fused tests"),
     ("b-zk-fix", "r20260922-105905-2021", "B-Ligero ZK interactive (F1/F2 fixed)"),
     "1.41 -> 1.394 s interactive (2 extra mask rows, free); FS re-pricing t=197 -> 288, D 6 -> 7: 1.500 s (+8%), transcript 122 -> 165 MB"),
    ("A Triton cache poisoning (a-kernel-a100; environment, not soundness): A100 v1",
     ("a-gpu-v2", "r20260922-091544-4e02", "v1 GPU (a-gpu v2e) same-box control, B=4096 A100"),
     ("a-kernel-a100", "r20260922-112320-3ceb", "full VU B=4096 A100: packed path, Triton cache-key fix"),
     "7.93 s (a-gpu v2e path on the W=31 tree) -> 2.979 s (a-gpu2 packed path, fixed main): different provers, not a fix cost; FIELD_KEY constexpr adds a cache key only"),
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
    ("B-Ligero, b-zk-fix protocol (q masked, coins-alone challenges), FIAT-SHAMIR mode",
     "EARNED: COMPLETE_HVZK_BACKEND (red-team-zk2 r20260922-112029-2db3: mask rank 288/288 at the real layout, 0/16150 accumulators, FS accounting reproduces 2^-128.04 with t=288 minimal, D=7)",
     "n/a (non-interactive)", "red-team-zk2 EARNED; b-verifier-2 Rust 108/108 honest / 264/264 rejected / 372 Python agreements",
     "COMPLETE_HVZK_BACKEND (ledger b-zk-fix r20260922-105905-2021), assumptions [hash, random-oracle]",
     "THE transferable ZK proof: 1.500 s / 4096 VUs L40S, 165 MB, depth 1, verifier 1.07 s Python / 1.65 s Rust 8T. D1 open: the Python verifier never checks the file's params meet the target"),
    ("B-Ligero, b-zk-fix protocol, INTERACTIVE mode (HM96 step-0 coins)",
     "EARNED (rank 197/197, 0/16150)", "EARNED: COMPLETE_ZK_BACKEND (red-team-zk2), after G1 (simulator dummy run on the public constraints; PublicCheckVerifier advantage 1 -> 0, b-zk-fix-2 r20260922-113615-2dfa), G2 (rewind count wording), G3 (assumption 'random-oracle (Merkle hiding)' added)",
     "red-team-zk2 EARNED; b-verifier-2 Rust 108/108; **D7**: interactive proof FILES are not transferable -- a file verifier replays the file's coins and the simulator as cheating prover produced 49/49 accepted files for false claims in BOTH file verifiers",
     "COMPLETE_ZK_BACKEND (ledger b-zk-fix r20260922-105905-2021), assumptions [hash, random-oracle (Merkle hiding)] -- holds only with a LIVE / designated verifier, like C",
     "1.394 s / 4096 VUs L40S, 122 MB, depth 3, +2% over the non-ZK control"),
    ("B-Ligero, b-zk 8b/8c protocol (PRE-FIX: b-zk r20260922-092404-2a1b, b-ligero2 --zk, gate4 HVZK row)",
     "BROKEN: q unmasked on H, 16150/16150 accumulators recovered (red-team-zk F1)", "simulator PROM-only (F2); FS run priced with the interactive bound (F3)",
     "red-team-zk VERDICT r20260922-100305-2b13", "NON_ZK_PROOF_DIAGNOSTIC (downgraded; ledger rows still say COMPLETE_ZK/HVZK_BACKEND)", "historical; superseded by the b-zk-fix rows"),
    ("B-AIR (Plonky3)", "none implemented (modelled +1-1.6%)", "none; as run Fiat-Shamir depth 1", "--", "NON_ZK_PROOF_DIAGNOSTIC",
     "F19 weak-FS residual closed by b-bind's vendored p3-fri"),
    ("C (QuickSilver)", "native (IT-MAC)", "native in the F_sVOLE-hybrid, designated verifier", "not red-teamed",
     "COMPLETE_ZK_BACKEND (designated-verifier)", "assumption stack LPN + base OT + CRH; 20.88 GB/batch at 2^-128; like B-Ligero interactive, not a transferable proof"),
)


PENDING_LANES: tuple[tuple[str, ...], ...] = ()


SOUNDNESS_HISTORY: tuple[tuple[str, ...], ...] = (
    # hole, found by, fixed by, rows affected, cost of fix
    ("B-AIR statement unbound + degree_bits read from the proof (F1/F2)", "red-team-vu (r20260922-070024-e7d1 'B FORGED: any trace = any statement')",
     "soundness-fixes 3fe85aa -> main via b-bind 8437aa5 (header absorbed before the first challenge; verifier fixes degree_bits from B; y16 column checked at zeta, zeta*g)",
     "every v1 B-AIR row before 8437aa5: b-chain, b-lookup (both), latency B rows", "not resolvable above vy-cpu3's 40% swing (544.8 s bound vs 426-485 s unbound; soundness-fixes r20260922-080527-f30c); +2^-105 term at BabyBear^4"),
    ("B-AIR LogUp path (lookup_stark.rs) unbound + no degree_bits check (U1/U2)", "red-team-3 (r20260922-085839-3976)",
     "b-bind (r20260922-091130-525e; StatementColumnMismatch / DegreeBitsMismatch; table digest in the v1 header; v2 binds tables via the recomputed preprocessed commitment, digest NOT in the v2 header -- b-bind's stated residual)",
     "b-lookup r20260922-070654-177a (60.14 s) recorded unbound", "free within ~10% (199.4/182.8 s bound vs 197.6/201.8 s unbound, b-bind r20260922-094310-af1a)"),
    ("B-AIR F3/F19 residual: p3-fri sampled alpha before absorbing the zeta-openings (weak FS)", "red-team-3 / red-team-vu transcript audit",
     "b-bind: vendored p3-fri (PROTOCOL 15.8) -- transcript change, sequential depth +1, upstream-0.4.3 v2 proofs no longer verify",
     "every B-AIR row's transcript; b-air-v2 depth 26 -> 27", "not measured separately (inside the b-bind rows)"),
    ("B-AIR BabyBear leading-bit window direct bound forgeable", "red-team-babybear (r20260922-054949-dcc0) then b-lookup lead.window",
     "b-lookup LT_DIRECT (+30 columns; r20260922-065546-a242)", "b-chain BabyBear 168.12 s", "+3% (168.12 -> 173.24 s), within noise"),
    ("B-Ligero challenges unbound to the statement words (LLL-style forgery at <=85 VUs)", "b-ligero2 (r20260922-073932-bc75)",
     "b-ligero2 (statement hashed into the transcript before the coins)", "b-ligero rows (<=85 VUs, no run ids)", "none: 2.30 -> 1.98 ms/VU"),
    ("B-Ligero ZK: q unmasked on H (F1), simulator PROM-only (F2), FS run priced with the interactive bound (F3)", "red-team-zk (r20260922-100305-2b13)",
     "b-zk-fix (uniform zero-sum q mask; coins-alone challenges; explicit interactive / fiat-shamir modes each with the accountant's bound) + b-zk-fix-2 (G1-G3); re-audited EARNED by red-team-zk2 (r20260922-112029-2db3)",
     "b-zk r20260922-092404-2a1b, b-ligero2 --zk, gate4 HVZK row (labels downgraded)", "interactive 1.41 -> 1.394 s (free); FS re-pricing +8% prover, +35% transcript"),
    ("B-Ligero interactive proof files not transferable (D7): a file verifier replays the file's coins", "b-verifier-2 (DISCREPANCIES.md D7; 49/49 forged files accepted by both file verifiers)",
     "not a code fix: the transferable protocol is --mode fiat-shamir; interactive = live verifier only", "every interactive B-Ligero row, as a FILE", "n/a"),
    ("A R16 sub-16-bit range chunks wrapped mod p (BabyBear)", "a-babybear (r20260922-063837-b26c)", "a-babybear (range tables re-derived)",
     "a-chain Goldilocks 297.6 s", "negative: 297.6 -> 261.3 s, depth 438 -> 381 (r20260922-065844-2b84)"),
    ("A run-vu took the v2 chain spec from the bundle's chain.txt (#9)", "red-team-v2 (r20260922-094212-465e)", "ff651a8: verifier-owned v2_chain_spec(); fixture refused (r20260922-095128-05ac)",
     "a-v2 verifier (prover unaffected)", "zero by construction"),
    ("A v2 epilogue lookups (R16/R15 tables of the epilogue segment) never proved -> any public word forgeable in v2", "a-logup-union (REDTEAM_URGENT_epilogue_tables.md)",
     "a-logup-union (table union across segments; gate r20260922-110557-b462 [COORDINATOR: run id from the coordinator, not found in the report files])",
     "a-v2 54.2 s CPU row; a-gpu-v2's v2 rows", "< 1%: +2 logUp instances, +98,304 multiplicities"),
    ("A Triton disk cache keyed on source + arg specialisation, not on constexpr W: W=31 binaries served to the W=22 tree (environment, not soundness)", "a-kernel-a100 (r20260922-112320-3ceb; hypothesis for a-verifier F1 consistent with a-verifier-2's clean-4090 result)",
     "a-kernel-a100: FIELD_KEY = sha256(P, W, R) constexpr on every kernel (regression test fails 5/7 pre-fix)",
     "a-gpu 4.43 s (unreproducible on vy-g5), a-gpu-v2 A100 rows (self-consistent over X^6-31, not the protocol field; re-recorded 2.450 s v2 / 2.969 s v1 by a-kernel-a100-2 r20260922-115019-03ea), gate4's a-gpu2 NO_PROOF rows", "none (a cache key; first proof after the fix recompiles every kernel: warm-up 304 s v1 / 161 s v2, excluded)"),
    ("A sumcheck variable order: a-gpu2 binds the low bit first, a-gpu / the Rust verifier the top bit (Python verifier shared gpu/logup.py so could not see it)", "a-protocol-diff (a_protocol_diff.md F1)",
     "PROTOCOL 4.2 pinned to the new order; Rust verifier follows (887fc5c); a-gpu2 B=4096 proof accepted, 1320/1320 mutations + 44/44 negatives rejected", "a-gpu2 rows were Python-only until 40868e5", "zero (verifier-side)"),
)


VERIFICATION: tuple[tuple[str, ...], ...] = (
    # headline row, own verifier, independent Rust verifier, red-team
    ("A gate4 same-card 5.02 s (H100 vy-g4, r20260922-091718-45c3)", "Python: accepted 5/5", "not run on this bundle", "a-verifier-2: kernel path == torch bytes on a clean card; cache-poisoning explains the vy-g5 rejection"),
    ("A a-gpu2 1.875 s (H100 vy-g5, r20260922-104038-eabb)", "Python: accepted 4096/4096 x3; 52/52 negatives", "ACCEPTED after the order fix (887fc5c); 1320/1320 mutations + 44/44 negatives rejected", "a-protocol-diff (order), a-verifier-2 (F1), red-team-3 (a-gpu forgeries all rejected)"),
    ("A a-kernel-a100-2 A100 pair: v1 2.969 s / v2 2.450 s (r20260922-115019-03ea, merged main)", "Python: accepted 3/3 each; 52/52 v2 negatives (r20260922-113632-baa3)", "ACCEPTED both (1.94 s / 2.02 s at 12T, built on the pod from main); the earlier level-2 rejection (r20260922-112813-b2fe) was the pre-merge verifier", "a-kernel-a100 cache forensics (r20260922-111848-dde1); F1 on vy-g5 explained as the mirror image, not confirmed (no access to that pod's cache)"),
    ("A a-v2 CPU 54.2 s / a-logup-union 67.36 s (checker v2)", "Rust prover's own verifier", "a-verifier-2 Rust verifies checker-v2 proofs (B=64, 48/48 negatives, r20260922-103210-9cba); not at 4096", "red-team-v2 (#9 fixed), a-logup-union (epilogue tables fixed)"),
    ("B-Ligero gate4 same-card 2.39 s / 2.91 s (H100, r20260922-092841-94ac)", "Python: accepted 3/3", "b-verifier (pre-fix protocol) 81/81 honest, 270/274 rejected (4 = unbound HVZK coin bytes)", "red-team-zk: the --zk row's privacy is broken (F1); soundness F3: FS run, interactive bound"),
    ("B-Ligero b-zk-fix 1.394 s interactive / 1.500 s FS (L40S, r20260922-105905-2021)", "Python: 52/52 negatives, 252/252 mutations", "b-verifier-2 Rust: 108/108 honest, 264/264 rejected, 372/372 Python agreement; 1.24 / 1.65 s at 8T", "red-team-zk2: both labels EARNED; D7 (interactive files not transferable); D1 open"),
    ("B-AIR b-air-v2 32.80 s BB^6 (vy-cpu2, r20260922-090904-cd75)", "Plonky3 verifier", "none exists", "red-team-vu/-3 forgeries fixed by soundness-fixes/b-bind; 47/47 tests, 52/52 x 5 modes, battery 0 FORGED (r20260922-111343-3ed9)"),
    ("B-AIR b-bind 199.4 s BB^4 bound (vy-cpu3, r20260922-094310-af1a)", "Plonky3 verifier", "none exists", "b-bind gate: U1/U2 fixtures rejected (StatementColumnMismatch, DegreeBitsMismatch)"),
    ("C c-compress 1255.6 s F384p (r20260922-080520-71b6)", "dietmc designated verifier", "n/a (designated verifier)", "not red-teamed"),
    ("SP1 3.58 s/VU (r20260921-220722-019b)", "SP1 verifier", "n/a", "not red-teamed"),
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
        if not m.get("prove_s"):
            m["prove_s"] = None
        m["seconds_per_vu"] = m["prove_s"] / m["B"] if m["prove_s"] else None
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


def render_soundness_history(entries: Sequence[dict[str, Any]]) -> str:
    del entries
    header = ["hole", "found by", "fixed by", "rows affected", "cost of the fix"]
    return _table(header, SOUNDNESS_HISTORY)


def render_verification(entries: Sequence[dict[str, Any]]) -> str:
    del entries
    header = ["headline row", "own verifier", "independent Rust verifier", "red-team"]
    return _table(header, VERIFICATION)


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
                             text=True, cwd=Path(__file__).resolve().parent).stdout
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
    "soundness_history": render_soundness_history,
    "verification": render_verification,
    "lanes": render_lanes,
}


def render(name: str, directory: Path | None = None) -> str:
    return TABLES[name](_entries(directory))


def render_all(directory: Path | None = None) -> dict[str, str]:
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
    ap.add_argument("--ledger-dir", type=Path, default=None, help="default: the frozen ledger, from the evidence store")
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
