"""circuit-check's bindings of `verity.definitions.registry.boolean_dsa_norm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _glm53_served_roots() -> list[Any]:
    """The units written as Calls at GLM-5.3's served statics: the scale `DsaRMSNormSumSq_v3` of its two rows, q_c [2048] and
    kv_c [512] (the smallest row whose threads take a register past their second, a row of `DsaRMSNormSquareFma`), the apply
    unit `DsaRMSNormApply_v3` of 64 columns, and the whole row `DsaRMSNorm_v3` of each.  Each is over `targets.MAX_GATES`, so it
    holds a ``digests`` pin, which keeps it in ``--all``; its description is rows of Calls, and its Flock block is read off them
    (`verity_flock.circuit.unit_block`)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_norm

    rows = (boolean_dsa_norm.DsaRMSNormSumSqV3, boolean_dsa_norm.DsaRMSNormV3)
    return [bind(d, N=n, EPS=1e-5) for d in rows for n in (2048, 512)] + [bind(boolean_dsa_norm.DsaRMSNormApplyV3, G=64)]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaRMSNorm_v2` and its scale at its smallest row (two elements a thread, one FMA each), and its apply unit of 8 elements;
    `TritonBlockRMSNorm_v2` at a two-warp block whose second warp is half masked (48 of 64 threads live, two elements each);
    `DsaLatentRow_v2` over that smallest norm and GLM-5.3's k_pe head (64); and the units written as Calls at the same
    bindings, `DsaRMSNorm_v3`, `DsaRMSNormSumSq_v3` and `DsaRMSNormApply_v3`."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_norm

    return [
        ("DsaRMSNorm_v2", lambda: bind(boolean_dsa_norm.DsaRMSNormV2, N=256, EPS=1e-5)),
        ("DsaRMSNormSumSq_v2", lambda: bind(boolean_dsa_norm.DsaRMSNormSumSqV2, N=256, EPS=1e-5)),
        ("DsaRMSNormApply_v2", lambda: bind(boolean_dsa_norm.DsaRMSNormApplyV2, G=8)),
        ("TritonBlockRMSNorm_v2", lambda: bind(boolean_dsa_norm.TritonBlockRMSNormV2, N=96, BLOCK=128, WARPS=2, EPS=1e-6)),
        ("DsaLatentRow_v2", lambda: bind(boolean_dsa_norm.DsaLatentRowV2, KVL=256, R=64, EPS=1e-5)),
        ("DsaRMSNorm_v3", lambda: bind(boolean_dsa_norm.DsaRMSNormV3, N=256, EPS=1e-5)),
        ("DsaRMSNormSumSq_v3", lambda: bind(boolean_dsa_norm.DsaRMSNormSumSqV3, N=256, EPS=1e-5)),
        ("DsaRMSNormApply_v3", lambda: bind(boolean_dsa_norm.DsaRMSNormApplyV3, G=8)),
    ]
