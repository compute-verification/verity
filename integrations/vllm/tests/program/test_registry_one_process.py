"""Core's `verity.ml` Definitions and every integration registry module load into one process, in either order, and
the integration's names for the core Definitions are core's objects (the registry refuses a second object under an id)."""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys

import pytest

import verity_vllm.program.registry as registry

CORE = ("verity.ml.prims", "verity.ml.gemm", "verity.ml.kernels")

CHECK = """
import importlib, sys
for m in sys.argv[1:]:
    importlib.import_module(m)
from verity.ir.defs import REGISTRY
from verity.ml import gemm as G, prims as C
from verity_vllm.program.registry import b1, hopper as H, prims as P, targets as T
core = [C.Bf16ToF32, C.F32ToBf16Rn, C.F2fpBf16, C.HopperBF16WgmmaDot16, C.AmpereBF16TcDot16, C.ZERO32, G.DotBf16,
        G.GemmCoordinate, G.Gemm]
assert all(REGISTRY[d.id] is d for d in core), [d.id for d in core if REGISTRY[d.id] is not d]
assert (P.Bf16ToF32, P.F32ToBf16Rn, P.F2fpBf16) == (C.Bf16ToF32, C.F32ToBf16Rn, C.F2fpBf16)
assert H.HopperBF16WgmmaDot16 is C.HopperBF16WgmmaDot16 is T.DOTS["HopperBF16WgmmaDot16"]
assert (b1.DotBf16V2, b1.GemmCoordinateV2, b1.GemmV2) == (G.DotBf16, G.GemmCoordinate, G.Gemm)
assert b1.const is C.const and b1.ZERO32 is C.ZERO32 and b1.const(16, 0) is C.const(16, 0)
from verity.ir.codec import decode_program, encode_program, program_digest
from verity.ir.program import Program
prog = Program(b1.GemmV2.bind(K=32, N=2, DOT=H.HopperBF16WgmmaDot16))
desc = encode_program(prog)
assert program_digest(encode_program(decode_program(desc, REGISTRY))) == program_digest(desc)
print("loaded", len(sys.argv) - 1, "modules;", len(REGISTRY.defs), "registered definitions")
"""


def _registry_modules() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(registry.__path__, registry.__name__ + "."))


@pytest.mark.parametrize("core_first", [True, False], ids=["core-first", "integration-first"])
def test_core_and_every_integration_registry_load_in_one_process(core_first: bool) -> None:
    mods = [*CORE, *_registry_modules()] if core_first else [*_registry_modules(), *CORE]
    r = subprocess.run([sys.executable, "-c", CHECK, *mods], capture_output=True, text=True, env=dict(os.environ), timeout=900)
    assert r.returncode == 0, r.stderr[-6000:]
    assert r.stdout.startswith("loaded"), r.stdout
