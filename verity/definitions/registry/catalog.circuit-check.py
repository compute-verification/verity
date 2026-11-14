"""What circuit-check reads of the catalog (`circuit_check.targets.HOOKS`), handed by the module that loads it, so circuit-check
names no module of the catalog's.  circuit-check reads these before it imports any module with a sidecar, so this file
imports nothing at its top level: each function imports what it hands."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def enumerators() -> set[tuple[str, str]]:
    """The functions (module, qualified name) that list a package only to import each module listed (`cache.enumerators`)."""
    return {("verity.definitions.registry.catalog", "load"), ("verity.definitions.boolean", "_load"),
            (__name__, "load_catalog")}


def load_catalog() -> None:
    """Import every module of the registry and the kernels of the replay's rows and twins, which register with core."""
    import verity.definitions.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity.definitions.numerics import rows, twins  # noqa: F401


def preload() -> dict[str, Any]:
    from verity.definitions.numerics import twins

    return twins.preload()


def kernels() -> list[Any]:
    from verity.definitions.numerics import kernel_registry

    return kernel_registry.registered()


def table_probes() -> dict[str, tuple[int, Any, int, tuple[tuple[int, int], ...], Any]]:
    """`tanh_rn` (2^27 entries of two words) as parameter 1 of the Definitions that read it, probed at its ends and its
    regions' edges (`TANH_RN_REGIONS`); `TanhF32Rn_v2` reads entry i through the word of its region whose low 27 bits are i
    (past HI where no region word has them)."""
    from verity.definitions.registry import tables

    lo = tables.TANH_RN_LO

    def word(i: int) -> int:
        return 0x38000000 | i if (0x38000000 | i) >= lo else 0x40000000 | i

    return {
        "TableRead27x2_v1": (1, tables.tanh_rn_table, 2, tables.TANH_RN_REGIONS, lambda i: [[i], [i | 0xF8000000]]),
        "TanhF32Rn_v2": (1, tables.tanh_rn_table, 2, tables.TANH_RN_REGIONS,
                         lambda i: [[word(i)], [word(i) | 0x80000000]]),
    }


def lazy_operands() -> dict[str, tuple[str, ...]]:
    """The parameters over the input budget that are operands, not a registered table, by the id of their Definition without
    statics."""
    return {
        # weights: a linear's or a GEMM's [N][K], an MoE launch's every expert's (with Marlin's per-32 group scales), and the
        # vocabulary's rows
        "CublasSplitKGemmF32Out_v1": ("w",),
        "DeepGemmFp8Linear_v1": ("w",),
        "Embedding_v1": ("table",),
        "EmbeddingShard_v1": ("table",),
        "FlashInferFp8Linear_v1": ("w",),
        "GemmF32Out_v1": ("w",),
        "Gemm_v2": ("w",),
        "MarlinMoeExpertGemm_v1": ("wq", "ws"),
        "MarlinMoeExpertGemmW_v1": ("wq", "ws"),
        "MoeExpertFp8Gemm_v1": ("w",),
        "MoeExpertFp8GemmW_v1": ("w",),
        "ScaledMmFp8BlockGrouped_v1": ("w",),
        # activations: every rank's partial sums, and the latent rows DSA's top-k routes past INDEX_TOPK keys (Waksman's network,
        # the mask form's concentrator in words and in bits, and a row below its size class: its rows and their pad)
        "AllReduceMultimem_v1": ("parts",),
        "DsaButterflyRows_v1": ("rows",),
        "DsaButterflyRows_v2": ("rows",),
        "DsaButterflyRowsPad_v1": ("rows", "pad"),
        "DsaButterflyRowsPad_v2": ("rows", "pad"),
        "DsaWaksmanRows_v1": ("rows",),
    }


def operand_probes() -> dict[str, Any]:
    """The first and last rows a lazy operand's Definition reads, as its index parameters at a function of its statics: a
    shard's first and last token, an expert count's last expert."""
    return {
        "Embedding_v1": lambda S: [{"tok": 0}, {"tok": S["V"] - 1}],
        "EmbeddingShard_v1": lambda S: [{"tok": S["START"]}, {"tok": S["START"] + S["VS"] - 1}],
        "MarlinMoeExpertGemm_v1": lambda S: [{"e": S["E"] - 1}],
        "MarlinMoeExpertGemmW_v1": lambda S: [{"e": S["E"] - 1}],
        "MoeExpertFp8Gemm_v1": lambda S: [{"e": S["E"] - 1}],
        "MoeExpertFp8GemmW_v1": lambda S: [{"e": S["E"] - 1}],
    }


def callee_calls() -> dict[str, tuple[str, ...]]:
    """The families a served Call calls where that Call's own partition goes unchecked, each with the Calls that pass it."""
    return {
        "AllReduceMultimemVector_v1": ("AllReduceMultimem_v1",),
        "And2_v1": ("DsaButterflyTagBits_v1", "DsaRadixTies_v1", "MhcFusedSplit_v2"),
        "AttnDotStepA_v1": ("AttnBlockFirst_v3",),
        "AttnDotStep_v1": ("AttnBlockFirst_v3",),
        "Bf16AddF2fp_v2": ("AllReduceRmsNormMnnvl_v2", "RMSNormFusedCudaVector8_v4"),
        "Bf16MulBf16_v2": ("AllReduceRmsNormMnnvl_v2",),
        "Bf16ToF32_v2": ("AllReduceRmsNormMnnvl_v2", "RMSNormFusedCudaVector8_v4"),
        "Const16[0x0000]_v1": ("DsaButterflyRows_v1",),
        "Const1[0x0]_v1": (
            "DsaButterflyRows_v2",
            "DsaButterflySettings_v2",
            "DsaButterflyTagBits_v1",
            "DsaTopKMaskCounted_v2",
            "DsaTopKMask_v2",
            "MhcFusedSplit_v2",
        ),
        "Const1[0x1]_v1": ("DsaTopKMaskCounted_v2", "DsaTopKMask_v2"),
        "Const32[0x3727c5ac]_v2": ("AllReduceRmsNormMnnvl_v2", "RMSNormFusedCudaVector8_v4"),
        "Const32[0x43800000]_v2": ("AllReduceRmsNormMnnvl_v2",),
        "Const32[0x44060000]_v2": ("RMSNormFusedCudaVector8_v4",),
        "DsaButterflyCtl_v1": ("DsaButterflySettings_v2",),
        "DsaButterflyTagBits_v1": ("DsaButterflySettings_v2",),
        "DsaMqaLogit_v3": ("DsaMqaLogits_v3",),
        "DsaMqaLogitsKey_v1": ("DsaMqaLogits_v4",),
        "DsaMqaLogitsQ_v1": ("DsaMqaLogits_v4",),
        "DsaRMSNormMeanRsqrt_v1": ("DsaRMSNorm_v3",),
        "DsaRMSNormOut_v1": ("DsaRMSNorm_v3",),
        "DsaRMSNormScale_v1": ("DsaRMSNorm_v3",),
        "DsaRMSNormSquareFma_v1": ("DsaRMSNorm_v3",),
        "DsaRMSNormSquares_v1": ("DsaRMSNorm_v3",),
        "DsaRadixPass_v1": ("DsaTopKMaskCounted_v2", "DsaTopKMask_v2"),
        "DsaRadixTies_v1": ("DsaTopKMaskCounted_v2", "DsaTopKMask_v2"),
        "DsaRowValid_v1": ("DsaTopKMaskCounted_v2",),
        "DsaTopKKey_v2": ("DsaTopKMaskCounted_v2", "DsaTopKMask_v2"),
        "DsaWaksmanPick_v1": ("DsaButterflyRows_v1",),
        "DsaWaksmanPick_v2": ("DsaButterflyRows_v2",),
        "DsaWaksmanSwap_v1": ("DsaButterflyRows_v1",),
        "DsaWaksmanSwap_v2": ("DsaButterflyRows_v2",),
        "F2fpBf16_v2": ("AllReduceRmsNormMnnvl_v2", "RMSNormFusedCudaVector8_v4"),
        "F32AddFtz_v3": ("AttnBlockFirst_v3",),
        "F32AddFtz_v4": ("AllReduceRmsNormMnnvl_v2",),
        "F32Add_v3": ("DsaRMSNorm_v3", "RMSNormFusedCudaVector8_v4"),
        "F32Div_v3": ("RMSNormFusedCudaVector8_v4",),
        "F32Max_v2": ("AttnBlockFirst_v3",),
        "F32MulFtzConst_v1": ("AttnBlockFirst_v3",),
        "F32MulFtz_v4": ("AllReduceRmsNormMnnvl_v2",),
        "F32Mul_v3": ("RMSNormFusedCudaVector8_v4", "ScaledMmFp8BlockSharedScale_v2"),
        "FmaSubRowF32FtzConstElement_v1": ("AttnBlockFirst_v3",),
        "FmaSubRowF32FtzConstScalar_v1": ("AttnBlockFirst_v3",),
        "GuardNegInfZero_v2": ("AttnBlockFirst_v3",),
        "MhcFusedSplit_v2": ("MhcFusedPostPre_v2",),
        "MhcPreBigFuse_v2": ("MhcFusedPostPre_v2",),
        "MoeExpertFp8CoordinateW_v1": ("MoeExpertFp8GemmW_v1",),
        "MoeExpertFp8Coordinate_v1": ("MoeExpertFp8Gemm_v1",),
        "MoeExpertFp8ScaleRow_v1": ("MoeExpertFp8GemmW_v1", "MoeExpertFp8Gemm_v1"),
        "MufuEx2Ftz_v2": ("AttnBlockFirst_v3",),
        "MufuRcpFtz_v2": ("AllReduceRmsNormMnnvl_v2",),
        "Not_v1": ("DsaRadixTies_v1", "MhcFusedSplit_v2"),
        "RMSNormPairSumCuda_v2": ("RMSNormFusedCudaVector8_v4",),
        "RMSNormSumInOrderCuda_v2": ("RMSNormFusedCudaVector8_v4",),
        "RsqrtApprox_v2": ("AllReduceRmsNormMnnvl_v2", "RMSNormFusedCudaVector8_v4"),
        "ScaledMmFp8BlockCoordinateGivenScale_v2": ("ScaledMmFp8BlockSharedScale_v2",),
        "Xor2_v1": ("DsaButterflyTagBits_v1", "DsaRadixTies_v1", "MhcFusedSplit_v2"),
    }


def row_families() -> set[str]:
    from verity.definitions.numerics import rows

    return set(rows.ROWS)


def boolean_table() -> tuple[dict[str, list], dict[str, list], list[str]]:
    from verity.definitions.boolean import _load

    return _load()


def made_on_demand(fid: str) -> Any:
    from verity.definitions.boolean import made_on_demand

    return made_on_demand(fid)


def boolean_version() -> tuple[Any, type[Exception]]:
    from verity.definitions.boolean import NoBooleanVersion, boolean_version

    return boolean_version, NoBooleanVersion


def lift_composite(defn: Any) -> Any:
    from verity.definitions.registry.lifted import lift_composite

    return lift_composite(defn)
