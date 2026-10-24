"""The VU families the benchmark proves: GLM-5.3's dot products at their served K, each one ``GemmCoordinate<K>`` (K MACs).

A workload is one VU family: the coordinate Definition a served GEMM's ``Q_word`` partition cuts it into, at that GEMM's K.
Its population is ``n_max`` independent instances of that Definition (``population``: core's ``population_program``, each
instance one unit of ``Q_template_instance`` v0, so the population is one stratum), staged once as one statement; each
session proves the B units the verifier draws from it by the stratified law, the draw the Lean verifier's soundness covers
(``instances``). Its unit is the coordinate class the ``Q_word`` partition of a ``Gemm_v2`` Call gives, gate for gate.

``bf16-k6144`` is the VU family behind GLM-5.3's bf16 GEMMs at hidden 6144 over ``HopperBF16WgmmaDot16_v1``
(``verity/ml/definitions/registry/bf16_gemm_check.py``): lm_head and the DSA indexer's ``wk_weights_proj``
(N = 160) are ``Gemm_v2``, whose coordinate is the VU; the MoE router (N = 256) past 16 tokens is cuBLAS's ``GemmF32Out_v1``
over the same dot chain, its output the f32 word rather than its bf16 rounding. The sites differ in N, the number of
coordinates a token row makes, not in the coordinate, so one VU family stands for them. ``fp8-coord-k*`` are served_units' block-scaled FP8
coordinates (``benchmarks/served_units/stage.py``'s ``UNITS``), declared with their profile and staged by that script; the
point job proves only the class-statement workloads today (``stager``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: ``MAX_STATEMENT_BITS`` the hill-climb proves at (2^35 witness bits): the GPU's ceiling for one statement
MAX_STATEMENT_BITS = 1 << 35


@dataclass(frozen=True)
class Workload:
    name: str
    #: the VU Definition a statement's instances are (one output coordinate)
    vu: str
    K: int
    #: the target the overhead divides by (``verity.ml.circuits.targets.target.TARGETS``): its anchor's dense peak
    profile: str
    #: the served GEMMs whose coordinates these are, each with its N per TP8 rank (None: vocabulary-sized)
    sites: dict[str, int | None]
    #: how the point job stages it: ``class-statement`` (verity_flock.class_statement --program-module) or ``served-unit``
    stager: str
    #: the population's instance count: the units a session's draw is taken from, and the largest B
    n_max: int
    #: block bits one coordinate takes in a statement (2^k_log), from staging; the cap is MAX_STATEMENT_BITS / 2^k_log
    k_log: int | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def cap(self) -> int:
        """The largest power-of-two B one statement holds: every coordinate is one 2^k_log block, the statement at most 2^35."""
        if self.k_log is None:
            return self.n_max
        return min(self.n_max, MAX_STATEMENT_BITS >> self.k_log)

    def population(self) -> Any:
        """The Program the statement is staged over: ``n_max`` independent instances of the VU Definition."""
        if self.stager != "class-statement":
            raise ValueError(f"{self.name}: staged by {self.stager}, not by the class sweep")
        from .population import population

        return population(self.vu, self.n_max)

    def to_json(self) -> dict:
        return {"name": self.name, "vu": self.vu, "K": self.K, "profile": self.profile, "sites": self.sites,
                "stager": self.stager, "n_max": self.n_max, "k_log": self.k_log, "cap": self.cap}


BF16 = "bf16-hopper-mma-draft/2026-09-22"
FP8 = "fp8-hopper-wgmma-draft/2026-09-22"

WORKLOADS: dict[str, Workload] = {w.name: w for w in (
    Workload("bf16-k6144", "GemmCoordinate_v2{K=6144,DOT=HopperBF16WgmmaDot16_v1}", 6144, BF16,
             {"lm_head": None, "indexer.wk_weights_proj": 160, "moe.router": 256}, "class-statement", 1024, k_log=25,
             notes="k_log 25 as the hill-climb's K = 8192 (K = 4096 packs into 24)",
             extra={"dot": "HopperBF16WgmmaDot16_v1"}),
    Workload("fp8-coord-k6144", "ScaledMmFp8BlockCoordinate_v2{K=6144,G=128}", 6144, FP8,
             {"fused_qkv_a": None, "dense.gate_up": None, "shared_expert.gate_up": None}, "served-unit", 1024,
             extra={"unit": "fp8-coord:K=6144"}),
    Workload("fp8-coord-k2048", "ScaledMmFp8BlockCoordinate_v2{K=2048,G=128}", 2048, FP8,
             {"q_b": None, "indexer.wq_b": None, "o_proj": None}, "served-unit", 2048, extra={"unit": "fp8-coord:K=2048"}),
    Workload("fp8-coord-k256", "ScaledMmFp8BlockCoordinate_v2{K=256,G=128}", 256, FP8,
             {"shared_expert.down_proj": None}, "served-unit", 4096, extra={"unit": "fp8-coord:K=256"}),
)}


def workload(name: str) -> Workload:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise ValueError(f"unknown workload {name!r}; one of {sorted(WORKLOADS)}") from None


def ladder(cap: int, lo: int = 1) -> list[int]:
    """B from ``lo`` to ``cap`` in powers of two (both powers of two)."""
    if cap < 1 or cap & (cap - 1) or lo < 1 or lo & (lo - 1) or lo > cap:
        raise ValueError(f"the ladder runs over powers of two, lo <= cap: got lo={lo}, cap={cap}")
    return [1 << i for i in range(int(math.log2(lo)), int(math.log2(cap)) + 1)]

