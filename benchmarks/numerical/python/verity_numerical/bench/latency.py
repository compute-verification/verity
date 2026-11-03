"""End-to-end latency as a first-class axis: what each design costs in wall-clock time per batch when every challenge
is a real network round trip, and what pipelining independent batches needs to recover throughput.

The protocol is INTERACTIVE (``backends/ZK_CONSTRUCTION.md`` 3.3: the verifier-coin commitment gives
malicious-verifier ZK at +1 depth, where Fiat-Shamir would force a re-parameterisation), so ``rounds.sequential_depth``
verifier messages are ``depth`` network round trips on the critical path.  Model (``archive/AGENTS.md`` "Latency is a
first-class axis"; the contract's ``t.e2e.rtt_<x>ms``)::

    t.e2e(rtt, bw) = t.prove + t.verify + depth x rtt + bytes / bw          (the honest serial number for one batch)

``t.prove`` is prover compute at zero-latency transport (``t.total``), ``t.verify`` the verifier's compute (its per-round
checks are serial with the prover's, so they are on the critical path; a verifier that only derives coins and defers
its checks would take them off it -- not assumed), ``bytes`` the total communication (``transcript.bytes`` plus the
coins), ``bw`` the link bandwidth.  Everything below is per *batch* (one proof of ``B`` VUs).

Pipelining.  Independent batches can be interleaved: while batch 1 waits for a coin the prover works on batch 2.  With
``N`` batches in flight the prover is busy ``N x t.prove`` out of every ``t.e2e``, so::

    throughput(N)  = min(N / t.e2e, 1 / t.prove, 1 / t.verify)   batches / s
    N_saturate     = ceil(t.e2e / t.prove)                        batches in flight that saturate the prover
    memory(N)      = N x mem_per_batch                            must fit the prover's memory

``mem_per_batch`` is what an in-flight batch holds between coins: for A the committed trace plus the sumcheck state
(peak RSS on the CPU; the codeword store + the phase-1 tables on a GPU), for B the trace plus the codeword store.  When
``N_saturate x mem_per_batch`` exceeds the machine, memory is the binding constraint and the achievable throughput is
``floor(capacity / mem_per_batch) / t.e2e``; :func:`memory_binding_rtt` is the RTT at which that happens.

Crossovers.  :func:`crossover_rtt` solves ``t.e2e_a(rtt) = t.e2e_b(rtt)`` for two designs: the RTT above which the
shallower protocol has the lower end-to-end latency despite a slower prover.

:data:`CATALOG` holds the designs the report runs (numbers from the recorded runs and reports named in each entry;
every one labelled with its proof class, hardware, and what is modelled rather than measured); :func:`from_envelope`
builds a design from a ``result.json`` that carries ``rounds.sequential_depth`` (the measured runs of this lane).
``python -m verity_numerical.bench.latency [--envelope NAME=result.json ...] [--out reports/latency]`` writes the
report (``latency.md`` + ``latency.json``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from verity_numerical.bench import census

__all__ = [
    "BANDWIDTHS_GBPS",
    "CATALOG",
    "RTTS_MS",
    "Design",
    "bandwidth_for_compute_bound",
    "crossover_rtt",
    "e2e",
    "from_envelope",
    "memory_binding_rtt",
    "n_saturate",
    "pipeline",
    "report",
    "sweep",
]

#: The contract's RTT sweep (ms) and the link speeds of the report (Gbit/s).
RTTS_MS: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 50.0)
BANDWIDTHS_GBPS: tuple[float, ...] = (10.0, 25.0, 100.0)
GIB = 1 << 30
GB = 1e9
MB = 1e6
#: The default native seconds per VU (``pipeline``'s ``native_s_per_vu``): FIRST's census line at K = 1536 (9.85 ps).
NATIVE_S_PER_VU = census.native_seconds_per_instance(census.HARDWARE["a100-sxm4-80gb/bf16"], 1536)


@dataclass(frozen=True)
class Design:
    """One design at one operating point.  Times in seconds per batch, bytes per batch."""

    name: str
    prove_s: float
    verify_s: float
    depth: int
    bytes_pv: float
    mem_per_batch_bytes: float
    mem_capacity_bytes: float
    B: int
    hardware: str
    proof_class: str
    source: str
    K: int = 1536
    protocol_rounds: int | None = None
    bytes_vp: float = 0.0
    interactive: bool = True
    verify_note: str = ""
    notes: str = ""

    @property
    def bytes_total(self) -> float:
        return self.bytes_pv + self.bytes_vp

    @property
    def compute_s(self) -> float:
        return self.prove_s + self.verify_s


# --- the model -----------------------------------------------------------------------------------------------------


def e2e(d: Design, rtt_ms: float, bw_gbps: float | None = None) -> dict[str, float]:
    """The serial end-to-end time of one batch and its parts: ``prove + verify + depth x rtt + bytes / bw``."""

    if rtt_ms < 0:
        raise ValueError("rtt_ms must be non-negative")
    network = d.depth * rtt_ms / 1000.0
    transfer = 0.0 if bw_gbps is None else d.bytes_total * 8 / (bw_gbps * 1e9)
    if bw_gbps is not None and bw_gbps <= 0:
        raise ValueError("bw_gbps must be positive")
    total = d.prove_s + d.verify_s + network + transfer
    return {"prove": d.prove_s, "verify": d.verify_s, "network": network, "transfer": transfer, "total": total,
            "network_share": network / total if total else 0.0}


def n_saturate(d: Design, rtt_ms: float, bw_gbps: float | None = None) -> int:
    """Batches in flight that keep the prover busy: ``ceil(t.e2e / t.prove)``."""

    return max(1, math.ceil(e2e(d, rtt_ms, bw_gbps)["total"] / d.prove_s - 1e-12))


def pipeline(d: Design, rtt_ms: float, bw_gbps: float | None = None, n_inflight: int | None = None,
             native_s_per_vu: float = NATIVE_S_PER_VU) -> dict[str, Any]:
    """Throughput with ``n_inflight`` batches interleaved (default: the saturating N, capped by memory)."""

    t = e2e(d, rtt_ms, bw_gbps)["total"]
    n_sat = n_saturate(d, rtt_ms, bw_gbps)
    n_mem = max(1, int(d.mem_capacity_bytes // d.mem_per_batch_bytes)) if d.mem_per_batch_bytes > 0 else n_sat
    n = n_inflight if n_inflight is not None else min(n_sat, n_mem)
    if n < 1:
        raise ValueError("n_inflight must be at least 1")
    prover_bound = 1.0 / d.prove_s
    verifier_bound = 1.0 / d.verify_s if d.verify_s > 0 else math.inf
    latency_bound = n / t
    batches_per_s = min(latency_bound, prover_bound, verifier_bound)
    binding = min((latency_bound, "in-flight batches (latency)"), (prover_bound, "prover compute"), (verifier_bound, "verifier compute"))[1]
    if n_inflight is None and n_sat > n_mem:
        binding = "memory (cannot hold N_saturate batches)"
    vu_per_s = batches_per_s * d.B
    return {
        "n_inflight": n, "n_saturate": n_sat, "n_memory_max": n_mem, "memory_bytes": n * d.mem_per_batch_bytes,
        "memory_fits": n * d.mem_per_batch_bytes <= d.mem_capacity_bytes if d.mem_per_batch_bytes > 0 else True,
        "batches_per_s": batches_per_s, "vu_per_s": vu_per_s,
        "seconds_per_vu": 1.0 / vu_per_s if vu_per_s else math.inf,
        "overhead_vs_native_peak": (1.0 / vu_per_s) / native_s_per_vu if vu_per_s else math.inf,
        "prover_utilisation": min(1.0, n * d.prove_s / t),
        "binding": binding,
    }


def memory_binding_rtt(d: Design, bw_gbps: float | None = None) -> float | None:
    """The RTT (ms) above which the saturating number of in-flight batches no longer fits in memory (``None`` when
    memory never binds, e.g. depth 0)."""

    if d.mem_per_batch_bytes <= 0 or d.depth == 0:
        return None
    n_mem = int(d.mem_capacity_bytes // d.mem_per_batch_bytes)
    base = e2e(d, 0.0, bw_gbps)["total"]
    # N_saturate(rtt) = ceil(t.e2e / prove) > n_mem  <=>  t.e2e > n_mem x prove
    rtt = (n_mem * d.prove_s - base) / d.depth * 1000.0
    return max(0.0, rtt)


def crossover_rtt(a: Design, b: Design, bw_gbps: float | None = None) -> float | None:
    """The RTT (ms) at which ``a`` and ``b`` have the same end-to-end latency (``None`` if parallel or the shallower
    design is already faster at RTT 0, i.e. no positive crossover)."""

    ca, cb = e2e(a, 0.0, bw_gbps)["total"], e2e(b, 0.0, bw_gbps)["total"]
    if a.depth == b.depth:
        return None
    rtt = (cb - ca) / (a.depth - b.depth) * 1000.0
    return rtt if rtt > 0 else None


def bandwidth_for_compute_bound(d: Design) -> float:
    """The link speed (Gbit/s) at which transferring the batch takes as long as proving it: below it the design is
    bandwidth-bound."""

    return d.bytes_total * 8 / d.prove_s / 1e9


def sweep(designs: Iterable[Design], rtts_ms: Sequence[float] = RTTS_MS, bws_gbps: Sequence[float] = BANDWIDTHS_GBPS) -> list[dict[str, Any]]:
    """One row per (design, RTT, bandwidth): the serial e2e and the pipelined throughput at the saturating N."""

    rows = []
    for d in designs:
        for rtt in rtts_ms:
            for bw in bws_gbps:
                row = {"design": d.name, "B": d.B, "rtt_ms": rtt, "bw_gbps": bw, **{f"e2e.{k}": v for k, v in e2e(d, rtt, bw).items()},
                       **{f"pipe.{k}": v for k, v in pipeline(d, rtt, bw).items()}}
                rows.append(row)
    return rows


# --- designs -------------------------------------------------------------------------------------------------------


def from_envelope(name: str, result: Mapping[str, Any], *, mem_capacity_bytes: float, mem_per_batch_bytes: float | None = None,
                  source: str | None = None, notes: str = "") -> Design:
    """A design from a ``result.json`` with ``t.total``, ``verifier.seconds``, ``rounds.sequential_depth``,
    ``transcript.bytes`` and (unless given) ``mem.peak_host_bytes`` / ``mem.peak_device_bytes``."""

    m = {x["name"]: x["value"] for x in result.get("measurements", []) if isinstance(x, Mapping) and "name" in x}
    fp = result.get("workload_fingerprint") or {}
    for key in ("t.total", "verifier.seconds", "rounds.sequential_depth", "transcript.bytes"):
        if key not in m:
            raise ValueError(f"{name}: envelope lacks {key}")
    mem = mem_per_batch_bytes if mem_per_batch_bytes is not None else m.get("mem.peak_device_bytes") or m.get("mem.peak_host_bytes")
    if mem is None:
        raise ValueError(f"{name}: no mem.peak_* measurement; pass mem_per_batch_bytes")
    hw = fp.get("hardware") or {}
    dev = (hw.get("gpu") or {}).get("name") or (hw.get("cpu") or {}).get("model") or "?"
    return Design(
        name=name, prove_s=float(m["t.total"]), verify_s=float(m["verifier.seconds"]), depth=int(m["rounds.sequential_depth"]),
        protocol_rounds=int(m["rounds.protocol"]) if "rounds.protocol" in m else None, bytes_pv=float(m["transcript.bytes"]),
        bytes_vp=32.0 * int(m.get("rounds.protocol", m["rounds.sequential_depth"])),  # one 32-byte coin per slot
        mem_per_batch_bytes=float(mem), mem_capacity_bytes=mem_capacity_bytes, B=int(fp.get("B", 0)), K=int(fp.get("K", 1536)),
        hardware=dev, proof_class=str(fp.get("proof_class", "?")), source=source or str(result.get("run_id", "?")), notes=notes,
    )


CPU2 = "AMD EPYC 9654 (vy-cpu2, 12 of 32 threads, shared with b-lookup and checker-min)"
CPU2_MEM = 256 * GB
GPU_4090 = "RTX 4090 24 GB"
GPU_4090_MEM = 24 * GB
GPU_H100 = "H100 80 GB"
GPU_H100_MEM = 80 * GB

#: The designs of the report at K = 1536.  ``prove_s`` is the zero-latency prover compute per batch (``t.total``).
#: CPU rows are the recorded runs of archive/gkr/README.md and archive/direct/README.md (the ``--rtt-ms`` runs of
#: this lane replace them when their envelopes are given); GPU rows are the re-priced kernel measurements of the
#: named reports (prover time measured or re-priced per bucket; memory per batch from the reports' device-memory
#: numbers; the verifier's time is the CPU verifier's, an assumption for a GPU pair).
CATALOG: dict[str, Design] = {
    "A-CPU Goldilocks^3": Design(
        name="A-CPU Goldilocks^3 (GKR+logUp+Ligero)", prove_s=261.3, verify_s=26.1, depth=355, protocol_rounds=3179, bytes_pv=69_693_032,
        bytes_vp=32 * 3179, mem_per_batch_bytes=33.8 * GB, mem_capacity_bytes=CPU2_MEM, B=4096, hardware=CPU2,
        proof_class="NON_ZK_PROOF_DIAGNOSTIC", source="archive/gkr/README.md r20260922-065844-2b84 (depth: measured trips of this lane, was 381 by the derived counter)",
        notes="memory = peak RSS (the committed trace + codeword store + sumcheck state)"),
    "A-CPU BabyBear^6": Design(
        name="A-CPU BabyBear^6 (GKR+logUp+Ligero)", prove_s=319.3, verify_s=45.0, depth=355, protocol_rounds=3176, bytes_pv=33_394_904,
        bytes_vp=32 * 3176, mem_per_batch_bytes=25.7 * GB, mem_capacity_bytes=CPU2_MEM, B=4096, hardware=CPU2,
        proof_class="NON_ZK_PROOF_DIAGNOSTIC", source="archive/gkr/README.md r20260922-065112-d687; depth as Goldilocks (same lookup chain)"),
    "A-GPU re-priced (4090, fused SIMT)": Design(
        name="A-GPU re-priced (4090 fused SIMT sumcheck, BabyBear^6)", prove_s=0.866, verify_s=45.0, depth=355, protocol_rounds=3176, bytes_pv=36 * MB,
        bytes_vp=32 * 3176, mem_per_batch_bytes=15.6 * GB + 3.0 * GB, mem_capacity_bytes=GPU_4090_MEM, B=4096, hardware=GPU_4090,
        proof_class="ARITHMETIC_DIAGNOSTIC (re-pricing)", source="reports/fused_sumcheck_4090.md: 0.866 s/batch (graphed x96, conservative; 0.641 s at census widths); ledger a-fused 2.15e7",
        verify_note="CPU BabyBear verifier 45 s used (no GPU verifier measured): dominates the serial e2e; a GPU verifier or deferred checks would change this row",
        notes="memory = phase-1 tables 15.6 GB (PROTOCOL.md 12.4) + codeword store 45,809 rows x 16,384 x 4 B = 3.0 GB; one batch per 24 GB card"),
    "A-GPU re-priced (4090) + 1 s verifier": Design(
        name="A-GPU re-priced (4090 fused SIMT) with a hypothetical 1 s verifier", prove_s=0.866, verify_s=1.0, depth=355, protocol_rounds=3176, bytes_pv=36 * MB,
        bytes_vp=32 * 3176, mem_per_batch_bytes=15.6 * GB + 3.0 * GB, mem_capacity_bytes=GPU_4090_MEM, B=4096, hardware=GPU_4090,
        proof_class="ARITHMETIC_DIAGNOSTIC (re-pricing + verifier assumption)", source="as the row above; verifier 1 s ASSUMED (no such verifier exists) to expose the depth x RTT term",
        verify_note="hypothetical: a GPU or 45x-parallel verifier; the row isolates the network term from the CPU verifier's 45 s",
        notes="same memory per batch as the row above"),
    "A-GPU re-priced (H100, packed model)": Design(
        name="A-GPU re-priced (H100 packed sumcheck, model)", prove_s=0.43, verify_s=45.0, depth=355, protocol_rounds=3176, bytes_pv=36 * MB,
        bytes_vp=32 * 3176, mem_per_batch_bytes=15.6 * GB + 3.0 * GB, mem_capacity_bytes=GPU_H100_MEM, B=4096, hardware=GPU_H100,
        proof_class="ARITHMETIC_DIAGNOSTIC (model)", source="ledger a-packed: A re-priced w/ packed sumcheck, model, 1.07e7 -> 0.43 s/batch",
        verify_note="CPU BabyBear verifier 45 s used (assumption)"),
    "B-CPU AIR (Plonky3 FRI, Goldilocks)": Design(
        name="B-CPU AIR (Plonky3 uni-stark FRI, Goldilocks^2)", prove_s=182.65, verify_s=0.148, depth=22, protocol_rounds=24, bytes_pv=3_231_120,
        bytes_vp=32 * 24, mem_per_batch_bytes=59.8 * GB, mem_capacity_bytes=CPU2_MEM, B=4096, hardware=CPU2,
        proof_class="NON_ZK_PROOF_DIAGNOSTIC", source="archive/direct/README.md r20260922-055326-699d (depth: measured observe->sample boundaries of this lane, was 24 derived); as run it is Fiat-Shamir (depth 1); the interactive reading is the campaign's",
        notes="t.total includes 24.2 s witness generation; memory = peak RSS (trace 2^19 x 2,844 + LDE + Merkle)"),
    "B-Ligero GPU re-priced (4090)": Design(
        name="B-Ligero GPU re-priced (4090, BabyBear, helper columns)", prove_s=0.259, verify_s=0.4, depth=3, protocol_rounds=7, bytes_pv=240 * MB,
        bytes_vp=32 * 7 + 16384 * 8, mem_per_batch_bytes=20.6 * GIB, mem_capacity_bytes=GPU_4090_MEM, B=4096, hardware=GPU_4090,
        proof_class="ARITHMETIC_DIAGNOSTIC (re-pricing, HVZK rows)", source="reports/b_measured_4090.md: 259 ms/batch, 95% measured; peak device memory 20.6 GiB (codeword store kept)",
        verify_note="0.4 s assumed (the CPU FRI verifier's range 0.15-0.4 s; the Ligero verifier re-encodes t = 192 columns of 310,602 rows)",
        notes="bytes: t = 192 opened columns x 310,602 rows x 4 B = 238 MB + tests (ZK_CONSTRUCTION.md 1); depth 3 interactive (4 with step 0)"),
    "SP1 (FS, B=1)": Design(
        name="SP1 core (Fiat-Shamir, one VU per proof)", prove_s=3.58, verify_s=0.245, depth=1, protocol_rounds=1, bytes_pv=5_898_276, bytes_vp=0,
        mem_per_batch_bytes=9.94 * GB, mem_capacity_bytes=GPU_4090_MEM, B=1, hardware="RTX 4090 (vy-sp1), SP1 6.4.0 cuda", interactive=False,
        proof_class="NON_ZK_PROOF (authentication included)", source="archive/sp1/benchmarks/dot_product/baselines.json r20260921-220722-019b",
        notes="depth 1: the proof is one message; the transferable baseline; peak GPU memory 9.9 GB per proof"),
    "C QuickSilver CPU (F61p, 2^-36)": Design(
        name="C QuickSilver CPU (dietmc F61p, 2^-36 sensitivity point)", prove_s=0.40835 * 4096, verify_s=0.40835 * 4096, depth=3, protocol_rounds=3,
        bytes_pv=8.58 * MB * 4096, bytes_vp=0.25 * MB * 4096, mem_per_batch_bytes=1.4 * GB, mem_capacity_bytes=128 * GB, B=4096,
        hardware="AMD EPYC 9655P (vy-cpu, 1 thread)", proof_class="COMPLETE_ZK_BACKEND (designated-verifier)",
        source="ledger c-vole: 4096 VUs dietmc F61p 0.408 s/VU; 8.58 MB/VU P->V",
        verify_note="verifier ~ prover (the designated verifier checks every MAC: 38.4 s vs 37.9 s in the F384p run)",
        notes="2^-36 statistical bound (F61p, r = 1): a sensitivity point, not the operating point; the 2^-128-class F384p run is 4.74 s/VU and 54.4 MB/VU"),
    "C QuickSilver GPU model": Design(
        name="C QuickSilver GPU model (tensor.md 3.05e6 overhead)", prove_s=3.05e6 * NATIVE_S_PER_VU * 4096, verify_s=3.05e6 * NATIVE_S_PER_VU * 4096, depth=3,
        protocol_rounds=3, bytes_pv=8.58 * MB * 4096, bytes_vp=0.25 * MB * 4096, mem_per_batch_bytes=4096 * 8.58 * MB, mem_capacity_bytes=GPU_H100_MEM, B=4096,
        hardware=GPU_H100, proof_class="cost model (designated-verifier)", source="reports/tensor.md C row: 3.05e6 at H100@0.172",
        verify_note="verifier = prover (model)", notes="bandwidth-bound: 35 GB per batch against 0.12 s of prover compute"),
}


# --- A-GKR: depth-reduction options (costed, not implemented) --------------------------------------------------
#
# A's critical path (measured by the transcript's trip counter, archive/gkr/src/main.rs) is
#     depth = 1 (message 2a) + max(lookup chain, checker chain) + 2 (Ligero r, Q)
#     lookup chain(n)  = 1 + sum_{k=0}^{n-1} (ceil(k / c) + 1)     the R16 fractional-GKR tree: level k is a k-round
#                                                                  sumcheck followed by one (mu, lambda) message
#     checker chain    = sum over the 6 layers of (ceil(b / c) + 2 ceil(s_in / c) + 1)   (no alpha on the last layer)
# Every instance (9 tables, 2 segments) already runs in lock-step; within an instance the rounds are inherently serial
# (each challenge must follow the round polynomial it binds).  Three levers shorten a chain: fewer levels (table
# sharding, n -> n - s), more variables per message (c), and CUTTING the chain by committing an intermediate layer so
# the pieces above and below it are independent given the commitment (the verifier opens the committed layer at the
# pieces' endpoints as Ligero linear functionals; the piece below starts from a random point sampled after com_1).

def lookup_chain(n: int, c: int = 1, cuts: Sequence[int] = ()) -> int:
    """Critical-path messages of the R16 fractional GKR of dimension ``n`` with ``c`` variables per message and the
    tree cut (committed) at the given levels: the longest piece plus the root's lambda message."""

    bounds = [0, *sorted(set(cuts)), n]
    pieces = [sum(math.ceil(k / c) + 1 for k in range(lo, hi)) for lo, hi in itertools.pairwise(bounds) if hi > lo]
    return 1 + max(pieces)


def checker_chain(layers: Sequence[tuple[int, int]], c: int = 1, cut_after: Sequence[int] = ()) -> int:
    """Critical-path messages of the checker GKR: ``layers`` = (b, s_in) per layer from the output layer down; a
    cut after layer i commits layer i's input wires so the layers below run in parallel with those above."""

    rounds = [math.ceil(b / c) + 2 * math.ceil(s / c) + (1 if i < len(layers) - 1 else 0) for i, (b, s) in enumerate(layers)]
    bounds = [0, *sorted(set(cut_after)), len(layers)]
    return max(sum(rounds[lo:hi]) for lo, hi in itertools.pairwise(bounds) if hi > lo)


def a_depth_options(n_r16: int, layers: Sequence[tuple[int, int]], committed: float, cpu: Design, gpu: Design,
                    sumcheck_share_gpu: float = 0.62, commit_share_gpu: float = 0.15) -> list[dict[str, Any]]:
    """The costed depth-reduction options for A at one (n_R16, checker layout): depth, prover-work factor, committed
    elements delta, e2e at 10 / 50 ms on the CPU and GPU rows, and the RTT above which each beats the baseline on
    the GPU (``prove x (w - 1) / (depth_base - depth_opt)``).  Work factors: c variables per message multiply the
    sumcheck buckets by (d+1)^c / ((d+1) c) (PROTOCOL.md 5.4; the GPU's sumcheck share is ``sumcheck_share_gpu``,
    fused_sumcheck_4090.md: 55-69%); committing more elements scales the commit/open buckets (``commit_share_gpu``)."""

    t = 65_536  # |R16|
    base_l, base_c = lookup_chain(n_r16), checker_chain(layers)
    base = 1 + max(base_l, base_c) + 2

    def row(option: str, l: int, ch: int, work_sum: float, extra_committed: float, note: str, work_other: float = 1.0) -> dict[str, Any]:
        depth = 1 + max(l, ch) + 2
        commit_frac = extra_committed / committed
        w_gpu = 1 + sumcheck_share_gpu * (work_sum - 1) + commit_share_gpu * commit_frac + (work_other - 1)
        w_cpu = 1 + 0.86 * (work_sum - 1) + 0.13 * commit_frac + (work_other - 1)  # CPU: lookup+arith 86%, commit+open 13% of 261 s
        out = {"option": option, "depth": depth, "lookup_chain": l, "checker_chain": ch, "prover_work_x": round(w_gpu, 3), "prover_work_x_cpu": round(w_cpu, 3),
               "committed_pct": 100 * commit_frac, "note": note}
        for tag, d, w in (("cpu", cpu, w_cpu), ("gpu", gpu, w_gpu)):
            dd = replace(d, prove_s=d.prove_s * w, depth=depth)
            out[f"e2e10_{tag}"] = e2e(dd, 10.0)["total"]
            out[f"e2e50_{tag}"] = e2e(dd, 50.0)["total"]
        if depth >= base:
            out["wins_above_rtt_ms_gpu"] = "never (deeper)" if w_gpu >= 1 else f"below {gpu.prove_s * (1 - w_gpu) / (depth - base) * 1000:.3g} ms only"
        elif w_gpu <= 1:
            out["wins_above_rtt_ms_gpu"] = "0 ms (also cheaper)"
        else:
            out["wins_above_rtt_ms_gpu"] = f"{gpu.prove_s * (w_gpu - 1) / (base - depth) * 1000:.3g} ms"
        return out

    rows = [row("baseline (b): data-parallel units, query wires committed, c = 1", base_l, base_c, 1.0, 0.0, "measured critical path")]
    for s in (1, 2, 5):
        rows.append(row(f"R16 sharding s = {s} (2^{s} tables)", lookup_chain(n_r16 - s), base_c, 1.0, (2**s - 1) * t,
                        f"+{(2**s - 1) * t:,} multiplicities; {2**s}x the R16 transcript (+{(2**s - 1) * 0.1:.1f} MB)"))
    for c in (2, 3):
        rows.append(row(f"c = {c} variables per message", lookup_chain(n_r16, c), checker_chain(layers, c), 4**c / (4 * c), 0.0,
                        f"pads x{(4**c - 1) / (3 * c):.1f}; sumcheck rounds send (d+1)^c - 1 values"))
    rows.append(row("c = 2 + sharding s = 5", lookup_chain(n_r16 - 5, 2), checker_chain(layers, 2), 2.0, 31 * t, "PROTOCOL.md 5.4 'both'"))
    for cuts in ((20,), (16, 21), (13, 17, 20, 22, 24)):
        extra = sum(2 * 6 * 2**ell for ell in cuts)  # p and q at level ell, 6 base coordinates each (BabyBear^6)
        rows.append(row(f"cut the R16 tree at levels {list(cuts)} [proposal]", lookup_chain(n_r16, 1, cuts), base_c, 1.0, extra,
                        f"commit p_l, q_l at the cut levels ({extra / 1e6:.1f}M elements); {2 * len(cuts)} more Ligero functionals; pieces run in parallel"))
    rows.append(row("cut the tree at [16, 21] + commit depth>=3 checker wires (32/unit) [proposal]", lookup_chain(n_r16, 1, (16, 21)), checker_chain(layers, 1, (2,)),
                    1.0, sum(2 * 6 * 2**ell for ell in (16, 21)) + 32 * 393_216, "checker chain halves too: the two levers that reach ~140 without extra sumcheck work"))
    rows.append(row("hint-free branch (query wires not committed; 5 layers after the lookups)", base_l + sum(math.ceil(b) + 2 * s + 1 for b, s in layers[1:]) + 1, base_c, 1.0,
                    -0.30 * committed, "ZK_CONSTRUCTION.md 4: prover 0.94x, committed -30%, depth ~630 by the derived counter", work_other=0.94))
    for m in (2, 4):
        rows.append(row(f"checkpoints every m = {m} units (sub-chain as a deep circuit)", base_l, checker_chain([(b - int(math.log2(m)), s) for b, s in layers] * m), 1.0, -190 / 44_937 * committed / m,
                        f"PROTOCOL.md 12.4: checker depth x~{m}, wire evaluations x{m}: deeper AND slower at every RTT", work_other=0.5 + 0.5 * m))
    return rows


# --- report --------------------------------------------------------------------------------------------------------

NOTES_MD = [
    "## 7. Reading the tables",
    "",
    "**What the depth is.** `rounds.sequential_depth` is now *measured*: A counts `Transcript::round_trip()` at every",
    "critical-path verifier message (the logUp `(mu, lambda)` pair is one message, so the derived counter's 381 is 355; the",
    "lock-step instances -- 9 tables, 2 segments -- share their messages, only the longest chain counts), B counts every",
    "`observe -> sample` boundary of the prover's Plonky3 challenger (22 at 2^19 rows, 16 at 2^13; the derived `log_n + 5`",
    "over-counted by 2). Both provers sleep one RTT at each such boundary under `--rtt-ms`, so `t.e2e.rtt_<x>ms` in the",
    "envelopes is wall-clock with the emulated link and `t.total` is the compute with the sleep subtracted.",
    "",
    "**Serial e2e (section 1).** At 50 ms every design's network term is `depth x 0.05 s`: A 17.8 s, B-AIR 1.1 s, B-Ligero",
    "0.15 s, SP1 0.05 s, C 0.15 s. On the CPU provers that is noise for A (261 s of compute) and for B-AIR (183 s); on the GPU",
    "re-pricings it is the whole story for A: 0.87 s of kernels behind 17.8 s of waiting (plus the 45 s CPU verifier, which is",
    "the honest serial number until a GPU verifier or deferred checks exist -- the table says so per row). B-Ligero at 0.26 s",
    "compute + 0.15 s network + 0.4 s verify stays under a second at 50 ms; its 240 MB of opened columns cost 0.08 s at 25 Gb/s.",
    "C is bandwidth-bound: 35 GB per batch of 4096 VUs needs ~2.3 Tb/s to hide behind 0.12 s of GPU compute (section 2).",
    "",
    "The row *A-GPU with a hypothetical 1 s verifier* isolates the network term: 1.9 s at 0 ms, 5.4 s at 10 ms, 19.6 s at",
    "50 ms -- 90% of the batch's wall-clock is waiting for coins.",
    "",
    "**Pipelining (section 3).** `N_sat` is how many batches the prover must hold to keep busy. With the real (45 s CPU)",
    "verifier A-GPU is verifier-bound at ~90 VU/s whatever the RTT. With the 1 s verifier it needs 3 batches in flight at",
    "0-1 ms, 7 at 10 ms, 23 at 50 ms, at 18.6 GB each (phase-1 tables 15.6 GB + codeword store 3.0 GB): one 24 GB card holds",
    "one batch, so memory binds at every RTT and A's throughput on a 4090 is one batch per e2e -- 2,200 VU/s at 0 ms, 750 at",
    "10 ms, 210 at 50 ms (overhead 4.7e7 -> 4.9e8). B-Ligero needs 3-4 batches (22 GB each with the codeword store kept):",
    "also more than one card holds, but its e2e is 0.74-0.89 s, so one batch per e2e is still 4,600-5,600 VU/s; the",
    "re-encode-on-demand store (b_measured_4090.md, 5.6 GB) would fit 4 batches and remove the bind. The CPU rows saturate",
    "at N = 2 (A: 68 GB, B-AIR: 120 GB of 256 GB) at every RTT up to 50 ms: compute-bound; memory binds only above",
    "4.3 s (A) / 25 s (B) of RTT.",
    "",
    "**Crossovers (section 4).** B-AIR (CPU) and B-Ligero (GPU) are already faster than A at 0 ms on the same class of",
    "hardware, so there is no A -> B crossover in the e2e sense: the shallower design wins everywhere and the gap grows by",
    "`(355 - depth_B) x RTT` -- 17 s per batch at 50 ms. The crossovers that exist: A-GPU (1 s verifier) loses to SP1's",
    "single-VU 3.8 s proof above 5.5 ms RTT and to the C GPU model above 28 ms; A-GPU beats B-CPU AIR up to 0.4-0.5 s of",
    "RTT (the CPU prover's 183 s is the gap). Within A, the RTT above which a depth-reduction option pays its extra prover",
    "work back (section 6, last column): sharding and the committed-cut proposals below 0.15 ms, `c = 2` above 3.4 ms,",
    "`c = 3` above 11 ms.",
    "",
    "**Can the verifier pre-commit all coins so the prover never waits? No.** The HM96 step-0 commitment hides the coins;",
    "opening coin `i` must follow the prover's message `i` or the prover learns it early and soundness is gone. The round trip",
    "is inherent to every coin that binds a prover message. **Can rounds be batched?** Yes, exactly when the prover's messages",
    "for those rounds do not depend on each other's coins: (i) the 9 tables and 2 segments already share messages (lock-step);",
    "(ii) the levels of the fractional tree and the layers of the checker are serial *only because* each level's claim is",
    "produced by the previous level's sumcheck -- committing an intermediate level (`p_l`, `q_l` of the tree, or the wires of",
    "a checker layer) makes the pieces above and below it independent given the commitment, so their rounds share round",
    "trips. That is the `cut` family in section 6: 26M committed elements (+14%) cut the lookup chain 352 -> 137, and 32",
    "committed wires per unit (+7%) cut the checker chain 197 -> 129; together depth 355 -> 140 at ~+3% GPU prover time.",
    "(iii) `c` variables per message halves or thirds the rounds of every sumcheck at 2x / 5.3x the sumcheck work. The",
    "checkpointed variant (PROTOCOL.md 12.4) is deeper *and* slower at every RTT -- it does not win under a latency objective",
    "either. The hint-free branch is 6% cheaper but ~170 rounds deeper: it wins only below ~0.5 ms RTT on the GPU.",
]


def _fmt_s(x: float) -> str:
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _fmt_bytes(x: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if x >= div:
            return f"{x / div:.1f} {unit}"
    return f"{x:.0f} B"


def report(designs: Sequence[Design], rtts_ms: Sequence[float] = RTTS_MS, bws_gbps: Sequence[float] = BANDWIDTHS_GBPS,
           measured: Sequence[Mapping[str, Any]] = (), options: Sequence[Mapping[str, Any]] = ()) -> tuple[str, dict[str, Any]]:
    """``(markdown, json)`` of the latency tables: serial e2e per design x RTT (at the middle bandwidth), the bandwidth
    sensitivity at RTT 0, the pipelining requirements per design x RTT, the crossovers, the memory-binding RTTs."""

    bw_mid = bws_gbps[len(bws_gbps) // 2]
    rows = sweep(designs, rtts_ms, bws_gbps)
    lines = ["# End-to-end latency: design x RTT (lane latency, 2026-09-22)", "",
             "Model: `t.e2e = t.prove + t.verify + depth x RTT + bytes / bandwidth` per batch (`verity_numerical.bench.latency`; the",
             "protocol is interactive, ZK_CONSTRUCTION.md 3.3, so every critical-path verifier message is one round trip). Every row",
             "carries its proof class and hardware; CPU rows are measured (vy-cpu2 shared with b-lookup and checker-min), GPU rows are",
             "re-priced kernels with the CPU verifier's time (an assumption). Pipelining: `N_sat = ceil(t.e2e / t.prove)` batches in",
             "flight saturate the prover if `N_sat x mem_per_batch` fits; otherwise memory binds.", ""]
    lines += ["## 1. Serial end-to-end latency per batch (s), link " + f"{bw_mid:g} Gb/s", "",
              "| design | B | proof class | hardware | prove s | verify s | depth | bytes | " + " | ".join(f"rtt {r:g} ms" for r in rtts_ms) + " | network share @50 ms |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(rtts_ms) + "---|"]
    for d in designs:
        e = [e2e(d, r, bw_mid) for r in rtts_ms]
        lines.append(f"| {d.name} | {d.B} | {d.proof_class} | {d.hardware} | {_fmt_s(d.prove_s)} | {_fmt_s(d.verify_s)} | {d.depth} | {_fmt_bytes(d.bytes_total)} | "
                     + " | ".join(_fmt_s(x["total"]) for x in e) + f" | {e2e(d, 50.0, bw_mid)['network_share']:.0%} |")
    lines += ["", "## 2. Bandwidth sensitivity: transfer time per batch (s) and the link speed at which transfer equals prover compute", "",
              "| design | bytes / batch | " + " | ".join(f"{b:g} Gb/s" for b in bws_gbps) + " | Gb/s for transfer = prove | bandwidth-bound below |", "|---|---|" + "---|" * len(bws_gbps) + "---|---|"]
    for d in designs:
        need = bandwidth_for_compute_bound(d)
        lines.append(f"| {d.name} | {_fmt_bytes(d.bytes_total)} | " + " | ".join(_fmt_s(e2e(d, 0.0, b)["transfer"]) for b in bws_gbps) + f" | {need:.2f} | {'yes at ' + str([b for b in bws_gbps if b < need]) if any(b < need for b in bws_gbps) else 'no'} |")
    lines += ["", f"## 3. Pipelining: in-flight batches to saturate the prover, memory, resulting VU/s and overhead (link {bw_mid:g} Gb/s)", "",
              "| design | mem / batch | capacity | " + " | ".join(f"N_sat @ {r:g} ms (mem, fits?)" for r in rtts_ms) + " | VU/s @ 0 / 10 / 50 ms | overhead @ 0 / 10 / 50 ms | binding @ 50 ms | memory binds above RTT |",
              "|---|---|---|" + "---|" * len(rtts_ms) + "---|---|---|---|"]
    for d in designs:
        ps = [pipeline(d, r, bw_mid) for r in rtts_ms]
        p0, p10, p50 = pipeline(d, 0.0, bw_mid), pipeline(d, 10.0, bw_mid), pipeline(d, 50.0, bw_mid)
        mb = memory_binding_rtt(d, bw_mid)
        lines.append(f"| {d.name} | {_fmt_bytes(d.mem_per_batch_bytes)} | {_fmt_bytes(d.mem_capacity_bytes)} | "
                     + " | ".join(f"{p['n_saturate']} ({_fmt_bytes(p['n_saturate'] * d.mem_per_batch_bytes)}, {'yes' if p['n_saturate'] * d.mem_per_batch_bytes <= d.mem_capacity_bytes else 'NO'})" for p in ps)
                     + f" | {p0['vu_per_s']:.3g} / {p10['vu_per_s']:.3g} / {p50['vu_per_s']:.3g} | {p0['overhead_vs_native_peak']:.2e} / {p10['overhead_vs_native_peak']:.2e} / {p50['overhead_vs_native_peak']:.2e} | {p50['binding']} | "
                     + ("never" if mb is None else f"{mb:.2f} ms") + " |")
    cross = []
    lines += ["", f"## 4. Crossovers: the RTT above which the shallower design has the lower serial e2e (link {bw_mid:g} Gb/s)", "", "| deeper | shallower | e2e @ 0 ms deeper / shallower | crossover RTT |", "|---|---|---|---|"]
    for a in designs:
        for b in designs:
            if a.depth > b.depth:
                x = crossover_rtt(a, b, bw_mid)
                ea, eb = e2e(a, 0.0, bw_mid)["total"], e2e(b, 0.0, bw_mid)["total"]
                cross.append({"deeper": a.name, "shallower": b.name, "e2e0_deeper": ea, "e2e0_shallower": eb, "crossover_rtt_ms": x})
                lines.append(f"| {a.name} | {b.name} | {_fmt_s(ea)} / {_fmt_s(eb)} | " + ("shallower already faster at 0 ms" if x is None else f"{x:.3g} ms") + " |")
    if measured:
        lines += ["", "## 5. Measured vs model (the `--rtt-ms` runs on vy-cpu2)", "",
                  "| run | design | B | RTT ms | depth (measured trips) | legacy depth | t.total s (compute) | verify s | sleep s | e2e measured s | model (same run's compute) | predicted from the RTT-0 run | predicted / measured |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        base0 = {(m["design"], m["B"]): m for m in measured if m["rtt_ms"] == 0}
        for m in measured:
            b0 = base0.get((m["design"], m["B"]))
            pred = (b0["t_total"] + b0["verify"] + m["depth"] * m["rtt_ms"] / 1000) if b0 else None
            lines.append(f"| {m['run_id']} | {m['design']} | {m['B']} | {m['rtt_ms']:g} | {m['depth']} | {m.get('depth_legacy', '-')} | {_fmt_s(m['t_total'])} | {_fmt_s(m['verify'])} | {_fmt_s(m['sleep'])} | {_fmt_s(m['e2e_measured'])} | {_fmt_s(m['e2e_model'])} | "
                         + (f"{_fmt_s(pred)} | {pred / m['e2e_measured']:.3f} |" if pred else "- | - |"))
            m["e2e_predicted_from_rtt0"] = pred
        lines += ["", "The sleep column is the measured wall-clock minus compute minus verify: `depth x RTT` to within the sleep overshoot (~0.1 ms per",
                  "trip). The last column is the honest test -- the RTT-0 run's compute plus `depth x RTT` against the wall-clock measured at",
                  "that RTT in a separate run; the residual is prover-compute drift on the shared box, not the network term."]
    if options:
        lines += ["", "## 6. Depth-reduction options for A (costed, not implemented)", "",
                  "| option | depth | prover work x | committed +% | e2e @ 10 ms (CPU / GPU) | e2e @ 50 ms (CPU / GPU) | RTT where it beats the baseline (GPU) | note |", "|---|---|---|---|---|---|---|---|"]
        for o in options:
            lines.append(f"| {o['option']} | {o['depth']} | {o['prover_work_x']:g} | {o['committed_pct']:+.1f} | {_fmt_s(o['e2e10_cpu'])} / {_fmt_s(o['e2e10_gpu'])} | {_fmt_s(o['e2e50_cpu'])} / {_fmt_s(o['e2e50_gpu'])} | {o['wins_above_rtt_ms_gpu']} | {o['note']} |")
    lines += ["", *NOTES_MD]
    doc = {"model": "t.e2e = prove + verify + depth x rtt + bytes / bw; throughput(N) = min(N / t.e2e, 1 / prove, 1 / verify); N_sat = ceil(t.e2e / prove)",
           "rtts_ms": list(rtts_ms), "bandwidths_gbps": list(bws_gbps), "designs": [asdict(d) for d in designs], "sweep": rows,
           "crossovers": cross, "memory_binding_rtt_ms": {d.name: memory_binding_rtt(d, bw_mid) for d in designs},
           "bandwidth_for_compute_bound_gbps": {d.name: bandwidth_for_compute_bound(d) for d in designs},
           "measured": list(measured), "options": list(options)}
    return "\n".join(lines) + "\n", doc


def measured_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """The measured-vs-model row of one ``--rtt-ms`` run: the envelope's ``t.e2e.rtt_<rtt>ms`` at the RTT it ran
    against ``t.total + verifier.seconds + depth x rtt`` (no transfer term: the run is on one host)."""

    m = {x["name"]: x["value"] for x in result.get("measurements", []) if isinstance(x, Mapping) and "name" in x}
    fp = result.get("workload_fingerprint") or {}
    rtt = float((fp.get("latency") or {}).get("rtt_ms", 0.0))
    depth = int(m["rounds.sequential_depth"])
    e2e_meas = float(m[f"t.e2e.rtt_{rtt:g}ms"])
    model = float(m["t.total"]) + float(m["verifier.seconds"]) + depth * rtt / 1000
    return {"run_id": result.get("run_id"), "design": (fp.get("software") or {}).get("backend", {}).get("name", "?"), "B": fp.get("B"), "rtt_ms": rtt, "depth": depth,
            "depth_legacy": m.get("depth_formula_legacy", m.get("direct.depth_formula_legacy")), "t_total": float(m["t.total"]), "verify": float(m["verifier.seconds"]),
            "sleep": e2e_meas - float(m["t.total"]) - float(m["verifier.seconds"]), "e2e_measured": e2e_meas, "e2e_model": model,
            "lookup_chain": m.get("depth_lookup_chain"), "checker_chain": m.get("depth_checker_chain")}


#: The checker GKR's per-layer (b, s_in) at B = 4096 (96 x 4096 = 393,216 unit copies, b = 19; 826 wires): fitted to
#: the measured checker chain 197 of r20260922-074606-be7b (6 x 19 + 2 x 39 + 5); the B = 64 run's 167 is
#: 6 x 13 + 2 x 42 + 5. The R16 fractional tree has n = 26 levels (352 = measured lookup chain).
A_CHECKER_LAYERS_B4096: tuple[tuple[int, int], ...] = ((19, 7), (19, 7), (19, 7), (19, 6), (19, 6), (19, 6))
A_R16_LEVELS_B4096 = 26
A_COMMITTED_B4096 = 44_937 * 4096


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envelope", action="append", default=[], metavar="KEY=PATH", help="replace CATALOG[KEY]'s prove/verify/depth/bytes/memory with a result.json")
    ap.add_argument("--measured", action="append", default=[], metavar="PATH", help="result.json of an --rtt-ms run for the measured-vs-model table")
    ap.add_argument("--out", default=None, help="write <out>.md and <out>.json")
    a = ap.parse_args(argv)
    designs = dict(CATALOG)
    for kv in a.envelope:
        key, path = kv.split("=", 1)
        base = designs[key]
        d = from_envelope(base.name, json.loads(Path(path).read_text()), mem_capacity_bytes=base.mem_capacity_bytes, source=path)
        designs[key] = replace(d, notes=base.notes, verify_note=base.verify_note)
    measured = [measured_row(json.loads(Path(p).read_text())) for p in a.measured]
    options = a_depth_options(A_R16_LEVELS_B4096, A_CHECKER_LAYERS_B4096, A_COMMITTED_B4096, designs["A-CPU Goldilocks^3"], designs["A-GPU re-priced (4090, fused SIMT)"])
    md, doc = report(list(designs.values()), measured=measured, options=options)
    if a.out:
        Path(a.out + ".md").write_text(md)
        Path(a.out + ".json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
