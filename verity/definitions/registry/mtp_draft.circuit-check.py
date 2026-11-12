"""circuit-check's bindings of `verity.definitions.registry.mtp_draft`'s Definitions, and of the draft step's other kernels at the
statics GLM-5.3's MTP draft runs them at, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`MtpEhNorm_v1` at GLM-5.3's draft layer (hidden 6144, its 8192 block on 4 warps, rms_norm_eps 1e-5), and each of its norms,
    `TritonBlockRMSNorm_v1` at those statics."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_norm as DN
    from verity.definitions.registry import mtp_draft as MD

    eh = {"BLOCK": 8192, "WARPS": 4, "EPS": 1e-5}
    return [
        ("MtpEhNorm_v1/glm53", lambda: bind(MD.MtpEhNorm, H=6144, **eh)),
        ("TritonBlockRMSNorm_v1/glm53-mtp-eh", lambda: bind(DN.TritonBlockRMSNorm, N=6144, **eh)),
    ]
