"""verity_vllm.definitions: kernel Definitions (`registry`), their vectorised twins and row evaluators (`numerics`), their
Boolean versions, the word replay (`replay`), the proof-block cut (`word`, `cross_call`, `call_scope`), a
Definition's encoding and digest (`encoding`), and the evidence ids their conformance texts and provenance records cite
(`evidence`).  It builds on the catalog (`verity.circuits.ops`, `.silicon`) and the kernels (`verity.circuits.kernels`); no
verifier reads it, so it is off the soundness path.

It is a catalog package, and names itself one as it loads: a window takes its templates from it by that name, wherever
the package is.  As it loads it also registers its registry with the Boolean-versions lookup
(`verity.circuits.ops.boolean.versions.catalog_loader`): every module `registry.catalog.load` imports, and the gathers
and `BitAtx` made on demand (`registry.boolean_gather`), so a lookup made in a process that imported this package scans
them.

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
    from verity_vllm.definitions.registry import boolean_gather as BGA

    return ((re.compile(r"^GatherBf16x(\d+)_v1$"), lambda m: BGA.gather_bf16(int(m[1]))),
            (re.compile(r"^GatherI32x(\d+)_v1$"), lambda m: BGA.gather_i32(int(m[1]))),
            (re.compile(r"^GatherF32x(\d+)_v1$"), lambda m: BGA.gather_f32(int(m[1]))),
            (re.compile(r"^BitAtx(\d+)_v1$"), lambda m: BGA.bit_at(int(m[1]))))


catalog_loader(_load_registry, _families)
