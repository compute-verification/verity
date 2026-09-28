"""circuit-check's bindings of `verity_vllm.program.registry.boolean_fp8_moe`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_fp8_moe

    return [
        ("MoeExpertFp8BlockCoordinate_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=32, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockCoordinate_v2/swap", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=32,
                                                             SWAP=True)),
        ("MoeExpertFp8BlockCoordinate_v2/kb64", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=64,
                                                             SWAP=False)),
        ("MoeExpertFp8BlockCoordinateW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateWV2, K=64, G=32, KB=32, SWAP=True)),
        ("MoeExpertFp8BlockGemm_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemmW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmWV2, K=32, N=40, G=32, KB=32, SWAP=True)),
        ("SiluMulFp8BlockQuantResult_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantResultV2)),
        ("SiluMulFp8BlockQuantGroup_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantGroupV2, G=4)),
        ("SiluMulFp8BlockQuant_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantV2, I=8, G=4)),
    ]
