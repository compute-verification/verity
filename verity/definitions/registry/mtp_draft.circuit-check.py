"""circuit-check's bindings of `verity.definitions.registry.mtp_draft`'s Definitions, and of the draft step's other kernels at the
statics GLM-5.3's MTP draft runs them at, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`MtpEhNorm_v1` at GLM-5.3's draft layer (hidden 6144, its 8192 block on 4 warps, rms_norm_eps 1e-5), held to the block by
    its cut (`mtp_draft.eh_cut`), and the cut's own units at those statics: the position select and each half's scale
    (`dsa_norm.DsaRMSNormApply_v1{G=64}`, the apply, is `dsa_norm`'s binding)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import mtp_draft as MD

    eh = {"BLOCK": 8192, "WARPS": 4, "EPS": 1e-5}
    return [
        ("MtpEhNorm_v1/glm53", lambda: bind(MD.MtpEhNorm, H=6144, **eh)),
        ("MtpEhZero_v1/glm53", lambda: bind(MD.MtpEhZero, H=6144)),
        ("TritonBlockRMSNormSumSq_v1/glm53-mtp-eh", lambda: bind(MD.TritonBlockRMSNormSumSq, N=6144, **eh)),
    ]
