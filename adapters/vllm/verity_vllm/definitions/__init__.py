"""verity_vllm.definitions: kernel Definitions (`registry`), their vectorised twins and row evaluators (`numerics`), their
Boolean versions, the word replay (`replay`), the proof-block cut (`word`, `cross_call`, `call_scope`), a
Definition's encoding and digest (`encoding`), and the evidence ids their conformance texts and provenance records cite
(`evidence`).  It builds on the catalog (`verity.circuits.ops`, `.silicon`) and the kernels (`verity.circuits.kernels`); no
verifier reads it, so it is off the soundness path.

It is a catalog package, and names itself one as it loads: a window takes its templates from it by that name, wherever
the package is.  As it loads it also registers its registry with the Boolean-versions lookup
(`verity.circuits.ops.boolean.versions.catalog_loader`): every module `registry.catalog.load` imports, and the gathers
and `BitAtx` made on demand (`registry.boolean_gather` and `registry.boolean_fp8_moe`'s E4M3 gather) and the top-p keep word
(`registry.boolean_topp_mask_word`), so a lookup made in
a process that imported this package scans them.

It is public for that import alone: a caller outside the adapter imports it to register the catalog, and needs no name
from it, so `__all__` is empty.
"""
import re

from verity.circuits.defs import catalog_package
from verity.circuits.ops.boolean.versions import catalog_loader

__all__ = []

catalog_package(__name__)


def _load_registry() -> list[str]:
    from verity_vllm.definitions.registry.catalog import load

    return list(load()[1])


def _families() -> tuple:
    from verity_vllm.definitions.registry import boolean_dsa_topk_mask as BDM
    from verity_vllm.definitions.registry import boolean_fp8_moe as BFM
    from verity_vllm.definitions.registry import boolean_gather as BGA
    from verity_vllm.definitions.registry import boolean_topp_mask_word as BTM

    return ((re.compile(r"^GatherE4m3x(\d+)_v1$"), lambda m: BFM.gather_e4m3(int(m[1]))),
            (re.compile(r"^GatherBf16x(\d+)_v1$"), lambda m: BGA.gather_bf16(int(m[1]))),
            (re.compile(r"^GatherI32x(\d+)_v1$"), lambda m: BGA.gather_i32(int(m[1]))),
            (re.compile(r"^GatherF32x(\d+)_v1$"), lambda m: BGA.gather_f32(int(m[1]))),
            (re.compile(r"^BitAtx(\d+)_v1$"), lambda m: BGA.bit_at(int(m[1]))),
            (re.compile(r"^TopPMaskWordx(\d+)_v1$"), lambda m: BTM.topp_mask_word(int(m[1]))),
            (re.compile(r"^DsaTopKSlot(\d+)x(\d+)_v1$"), lambda m: BDM.slot(int(m[1]), int(m[2]))),
            (re.compile(r"^DsaTopKLongRowCheck(\d+)x(\d+)_v1$"), lambda m: BDM.long_row_check(int(m[1]), int(m[2]))),
            (re.compile(r"^DsaTopK(Counted)?Threshold(Key|Index)(\d+)x(\d+)_v1$"),
             lambda m: BDM.threshold(int(m[3]), int(m[4]), bool(m[1]))[m[2] == "Index"]),
            (re.compile(r"^TableRead(\d+)x(\d+)_v1$"), _table_read))


def _table_read(m: re.Match) -> object:
    """`TableRead{IB}x{EW}_v1`'s Boolean form (`verity.circuits.boolean.table`), for a table of at most `MAX_WORDS` words; none
    past it (`gelu_tanh_bf16`, `tanh_rn`)."""
    from verity.circuits.boolean import table as BT

    from verity_vllm.definitions.registry import tables

    ib, ew = int(m[1]), int(m[2])
    return BT.table_read(tables.table_read(ib, ew), ib) if ew << ib <= BT.MAX_WORDS else None


catalog_loader(_load_registry, _families)
