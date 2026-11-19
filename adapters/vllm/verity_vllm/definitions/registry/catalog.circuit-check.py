"""What circuit-check reads of the catalog (`circuit_check.targets.HOOKS`), handed by the module that loads it, so circuit-check
names no module of the catalog's.  circuit-check reads these before it imports any module with a sidecar, so this file
imports nothing at its top level: each function imports what it hands."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def enumerators() -> set[tuple[str, str]]:
    """The functions (module, qualified name) that list a package only to import each module listed (`cache.enumerators`)."""
    return {("verity_vllm.definitions.registry.catalog", "load"), ("verity.circuits.ops.boolean.versions", "_load"),
            (__name__, "load_catalog")}


def handed_packages() -> tuple[str, ...]:
    """The packages whose circuits circuit-check is handed and names none of: the served Definitions' package."""
    return ("verity_vllm.definitions",)


def load_catalog() -> None:
    """Import every module of the registry and the kernels of the replay's rows and twins, which register with core."""
    import verity_vllm.definitions.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity_vllm.definitions.numerics import rows, twins  # noqa: F401


def preload() -> dict[str, Any]:
    from verity_vllm.definitions.numerics import twins

    return twins.preload()


def kernels() -> list[Any]:
    from verity_vllm.definitions.numerics import kernel_registry

    return kernel_registry.registered()


def table_probes() -> dict[str, tuple[int, Any, int, tuple[tuple[int, int], ...], Any]]:
    """`tanh_rn` (2^27 entries of two words) as parameter 1 of the Definitions that read it, probed at its ends and its
    regions' edges (`TANH_RN_REGIONS`); `TanhF32Rn_v2` reads entry i through the word of its region whose low 27 bits are i
    (past HI where no region word has them)."""
    from verity_vllm.definitions.registry import tables

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


def row_families() -> set[str]:
    from verity_vllm.definitions.numerics import rows

    return set(rows.ROWS)


def boolean_table() -> tuple[dict[str, list], dict[str, list], list[str]]:
    from verity.circuits.ops.boolean.versions import _load

    import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)

    return _load()


def made_on_demand(fid: str) -> Any:
    from verity.circuits.ops.boolean.versions import made_on_demand

    import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)

    return made_on_demand(fid)


def boolean_version() -> tuple[Any, type[Exception]]:
    from verity.circuits.ops.boolean.versions import NoBooleanVersion, boolean_version

    import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)

    return boolean_version, NoBooleanVersion


def lift_composite(defn: Any) -> Any:
    from verity_vllm.definitions.registry.lifted import lift_composite

    return lift_composite(defn)
