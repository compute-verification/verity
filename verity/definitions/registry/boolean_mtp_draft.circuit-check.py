"""circuit-check's bindings of `verity.definitions.registry.boolean_mtp_draft`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _glm53_served_roots() -> list[Any]:
    """The units written as Calls at GLM-5.3's draft layer (hidden 6144, its 8192 block on 4 warps, rms_norm_eps 1e-5): the select,
    the scale and the whole row as its cut.  A unit over `targets.MAX_GATES` holds a ``digests`` pin, which keeps it in ``--all``."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_mtp_draft as BM

    eh = {"BLOCK": 8192, "WARPS": 4, "EPS": 1e-5}
    return [bind(BM.MtpEhZeroV2, H=6144), bind(BM.TritonBlockRMSNormSumSqV2, N=6144, **eh), bind(BM.MtpEhNormV2, H=6144, **eh)]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Each at a row of 256 in a block of 512 on one warp: two runs of 256, the second masked, so the scale takes a masked
    register; the select at 16."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_mtp_draft as BM

    small = {"BLOCK": 512, "WARPS": 1, "EPS": 1e-5}
    return [
        ("MtpEhZero_v2", lambda: bind(BM.MtpEhZeroV2, H=16)),
        ("TritonBlockRMSNormSumSq_v2", lambda: bind(BM.TritonBlockRMSNormSumSqV2, N=256, **small)),
        ("MtpEhNorm_v2", lambda: bind(BM.MtpEhNormV2, H=256, **small)),
    ]
