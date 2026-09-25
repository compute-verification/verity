"""The census: one registry of the fixed quantities the campaign's tables are computed against, each with a stable id.

* :data:`HARDWARE` -- the GPU lines we measure: a device and an operand dtype, with the datasheet **dense** tensor-core
  peak at FP32 accumulate (an FMA counts as two FLOP, sparsity figures are never used) and the source it was read from.
  A ``verity.proofs.target.Target``'s ``native_peak`` is one of these.
* :data:`NETWORKS` -- the reference network a Table 2 cell is evaluated at (:data:`REFERENCE_NETWORK`: same datacenter,
  1 ms RTT, 100 Gb/s) and the sensitivity profiles of the interaction drill-down (:data:`SENSITIVITY_NETWORKS`).
* :data:`WORKLOADS` -- the unit a table row proves (``GemmCoordinate<K>`` at the frozen K and reference batch).
* :data:`INSTANCE_SETS` -- the frozen instance set of each target: dataset, tier, size and the digest a result's
  ``instances`` ref must carry.

Ids are stable: a render cites them, and a site builds history over renders by them.  A changed quantity is a new entry
with a new id, never an edit of an old one.  Stdlib only; imports no other core module.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HARDWARE",
    "INSTANCE_SETS",
    "NETWORKS",
    "REFERENCE_NETWORK",
    "SENSITIVITY_NETWORKS",
    "VU_K1536",
    "WORKLOADS",
    "Hardware",
    "InstanceSet",
    "Network",
    "Workload",
]


@dataclass(frozen=True, slots=True)
class Hardware:
    """One measured GPU line: ``device`` (the datasheet part) running operand ``dtype`` (the instruction registry's
    ``a_dtype``: ``bf16``, ``e4m3``, ``e2m1``), with its dense tensor-core peak at the ``accumulate`` width in FLOP per
    second and the ``source`` it was read from.  ``memory_bytes`` is the reference part's device memory (a board reporting a
    very different total is a variant of the SKU, and this peak is not its denominator); ``None`` = unchecked."""

    id: str
    device: str
    dtype: str
    rate: str
    flop_per_second: float
    accumulate: str
    source: str
    memory_bytes: int | None = None

    def seconds_per_vu(self, K: int) -> float:
        """Native wall seconds of one ``GemmCoordinate<K>`` (``2K`` FLOP) at this peak."""

        if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
            raise ValueError(f"K must be a positive int, got {K!r}")
        return 2 * K / self.flop_per_second


_HARDWARE = (
    Hardware(
        id="a100-sxm4-80gb/bf16",
        device="NVIDIA A100 SXM4 80GB",
        dtype="bf16",
        rate="bf16 dense tensor-core peak, fp32 accumulate",
        flop_per_second=312e12,
        accumulate="fp32",
        source="NVIDIA A100 datasheet / Ampere whitepaper",
        memory_bytes=80 * 2**30,
    ),
    Hardware(
        id="h100-sxm5-80gb/bf16",
        device="NVIDIA H100 SXM5 80GB",
        dtype="bf16",
        rate="bf16 dense tensor-core peak, fp32 accumulate",
        flop_per_second=989.4e12,
        accumulate="fp32",
        source="NVIDIA Hopper whitepaper (BF16 1978.9 TFLOPS with sparsity)",
        memory_bytes=80 * 2**30,
    ),
    Hardware(
        id="h100-sxm5-80gb/e4m3",
        device="NVIDIA H100 SXM5 80GB",
        dtype="e4m3",
        rate="fp8 (e4m3) dense tensor-core peak, fp32 accumulate",
        flop_per_second=1978.9e12,
        accumulate="fp32",
        source="NVIDIA Hopper whitepaper (FP8 3957.8 TFLOPS with sparsity; FP32 accumulate at full rate on Hopper)",
        memory_bytes=80 * 2**30,
    ),
    Hardware(
        id="rtx-4090/e4m3",
        device="NVIDIA GeForce RTX 4090",
        dtype="e4m3",
        rate="fp8 (e4m3) dense tensor-core peak, fp32 accumulate",
        flop_per_second=330.3e12,
        accumulate="fp32",
        source=(
            "NVIDIA Ada whitepaper (FP8 with FP32 accumulate 330.3 / 660.6 sparse TFLOPS; FP16 accumulate is 660.6 -- "
            "Ada halves FP8 throughput with FP32 accumulate)"
        ),
        memory_bytes=24 * 2**30,
    ),
    Hardware(
        id="rtx-5090/e2m1",
        device="NVIDIA GeForce RTX 5090",
        dtype="e2m1",
        rate="fp4 (e2m1, block-scaled) dense tensor-core peak, fp32 accumulate",
        flop_per_second=1676e12,
        accumulate="fp32",
        source=(
            "NVIDIA RTX Blackwell GPU Architecture whitepaper, Appendix A, table 'GeForce RTX 5090 vs RTX 4090 vs RTX 3090', "
            "row 'Peak FP4 Tensor TFLOPS with FP32 Accumulate (FP4 AI TOPS)': 1676 dense / 3352 with the 2:4 sparsity "
            "feature (footnote 2); footnote 1: peak rates at the GPU boost clock (2407 MHz).  The card's headline "
            "'3352 AI TOPS' is the sparse figure; dense is used here.  4x the FP8-with-FP32-accumulate rate (419 dense)"
        ),
    ),
)
HARDWARE: dict[str, Hardware] = {h.id: h for h in _HARDWARE}


@dataclass(frozen=True, slots=True)
class Network:
    """A network profile the interaction formula ``t = compute + rounds x RTT + bytes / bandwidth`` is evaluated at."""

    id: str
    label: str
    rtt_s: float
    bandwidth_bps: float


#: The profile a Table 2 cell is evaluated at: same datacenter.
REFERENCE_NETWORK = Network("dc-1ms-100gbps", "same datacenter (reference)", 0.001, 100e9)
#: The interaction drill-down's other profiles: longer RTTs at the reference bandwidth, a slower link at the reference RTT.
SENSITIVITY_NETWORKS = (
    Network("rtt-20ms-100gbps", "20 ms RTT, 100 Gb/s", 0.020, 100e9),
    Network("rtt-100ms-100gbps", "100 ms RTT, 100 Gb/s", 0.100, 100e9),
    Network("dc-1ms-10gbps", "1 ms RTT, 10 Gb/s", 0.001, 10e9),
)
NETWORKS: dict[str, Network] = {n.id: n for n in (REFERENCE_NETWORK, *SENSITIVITY_NETWORKS)}


@dataclass(frozen=True, slots=True)
class Workload:
    """The unit a table row proves: ``GemmCoordinate<K>`` (``K`` multiply-accumulates from ``c_0 = 0``), measured at the
    reference ``batch`` of the frozen tables."""

    id: str
    K: int
    batch: int
    description: str


VU_K1536 = Workload("vu-k1536", 1536, 4096, "GemmCoordinate<1536>: one K = 1536 verification unit, batch 4096 (frozen 2026-09-22)")
WORKLOADS: dict[str, Workload] = {w.id: w for w in (VU_K1536,)}


@dataclass(frozen=True, slots=True)
class InstanceSet:
    """A frozen instance set: ``dataset`` / ``tier`` with ``size`` instances, identified by ``manifest_sha256`` (the sha256
    of a fixture's ``manifest.json``, or of a synthetic set's recipe string at ``size``).  ``seed`` is a synthetic set's
    generator seed (``None`` for a fixture); ``target`` names the ``verity.proofs.target`` target it is the set of."""

    id: str
    target: str
    dataset: str
    tier: str
    size: int
    manifest_sha256: str
    seed: int | None = None


_SEED = 20260922
_INSTANCE_SETS = (
    InstanceSet("vu-k1536", "first-campaign-target/2026-09-21", "bench-instances/v1", "vu-k1536", 4096,
                "059103cf9bd55ee83cbd4bb14ae6db2f60db2cb4ddf85cdc22b1cecee6e4eeea"),
    InstanceSet("vu-k1536-fp8-ada", "fp8-ada-mma-draft/2026-09-22", "bench-instances-fp8-ada/v1", "vu-k1536-fp8-ada", 4096,
                "e66ff0f21c8e67d13b4b2d4cee2afa35b8345656b314da80e6b342ba051f078d", _SEED),
    InstanceSet("vu-k1536-bf16-hopper", "bf16-hopper-mma-draft/2026-09-22", "bench-instances-bf16-hopper/v1", "vu-k1536-bf16-hopper",
                4096, "2a5babca16e82727611ae483a125b8d853e27e4c9deedda7c35927cc1e456869", _SEED),
    InstanceSet("vu-k1536-fp8-hopper", "fp8-hopper-wgmma-draft/2026-09-22", "bench-instances-fp8-hopper/v1", "vu-k1536-fp8-hopper",
                4096, "0ff750026b8d8df0418d3ec360ed6ade683a33b8264d66973021cbfbff7945bc", _SEED),
    InstanceSet("vu-k1536-nvfp4-sm120", "nvfp4-sm120-mma-draft/2026-09-22", "bench-instances-nvfp4-sm120/v1", "vu-k1536-nvfp4-sm120",
                4096, "d2d65f6523b49e673847746d725fbce3282f2734b0b424637c01e88f188a240a", _SEED),
)
INSTANCE_SETS: dict[str, InstanceSet] = {s.id: s for s in _INSTANCE_SETS}
