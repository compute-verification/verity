"""The stock FP8 libraries an H200 serves with beyond cuBLASLt, through their own builds (`bench.py --bridge`): vLLM's CUTLASS
`cutlass_scaled_mm` (sm_90: per-tensor, rowwise and 128-block scaling) and DeepGEMM's `fp8_gemm_nt` (128-block, through
`vllm.utils.deep_gemm` as vLLM calls it), from torch, on the harness's own operand buffers (A m x k and B n x k, K-major; D
m x n BF16) and its timing stream. Each candidate's scales are its own tensors of ones in the layout its library reads (the
harness's scale buffers are cuBLASLt's), so the gate's exact ternary products hold; their bytes are read as usual. A library
that won't import, or refuses a shape, is recorded and has no candidate there.

The bridge imports torch and vLLM only when a run asks for it, after every arm's binaries passed the SASS gate: what it maps
then is a stock library, the divisor timed beside the arms, never a binary a verifier recomputes (`exempt`)."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field

#: what `bench.py` and the tests read
__all__ = ["Bridge", "deep_gemm", "exempt"]

#: each family the bridge serves, and its sources: vLLM's CUTLASS for every FP8 scaling, DeepGEMM for 128-block only
SOURCES = {"fp8-e4m3": ("cutlass-vllm",), "fp8-e4m3-rowwise": ("cutlass-vllm",), "fp8-blk128": ("cutlass-vllm", "deepgemm")}
BLOCK = 128
#: where vLLM finds DeepGEMM, in its order (`vllm.utils.import_utils.has_deep_gemm`): an installed `deep_gemm`, else the copy
#: its wheel vendors
DEEP_GEMM = ("deep_gemm", "vllm.third_party.deep_gemm")


class _Raw:
    """A device pointer as torch reads one (`__cuda_array_interface__` v3, no stream to wait on)."""

    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": typestr, "data": (ptr, False), "version": 3,
                                         "strides": None, "stream": None}


@dataclass
class Bridge:
    """torch, vLLM's ops and DeepGEMM, each loaded or the reason it isn't (`errors`), and their versions."""

    torch: object = None
    ops: object = None
    dg: object = None
    errors: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> Bridge:
        b = cls()
        try:
            import torch

            b.torch, b.versions["torch"] = torch, torch.__version__
            b.versions["torch_cuda"] = torch.version.cuda
        except Exception as ex:  # noqa: BLE001 - recorded; no bridge candidate
            b.errors["torch"] = f"{type(ex).__name__}: {ex}"
            return b
        try:
            import vllm
            from vllm import _custom_ops  # noqa: F401 - registers torch.ops._C

            b.ops, b.versions["vllm"] = torch.ops._C, vllm.__version__
        except Exception as ex:  # noqa: BLE001
            b.errors["cutlass-vllm"] = f"{type(ex).__name__}: {ex}"
        try:
            from vllm.utils import deep_gemm as DG

            if not DG.is_deep_gemm_supported():
                raise RuntimeError("vllm.utils.deep_gemm.is_deep_gemm_supported() is False on this card")
            b.versions["deep_gemm_module"], b.versions["deep_gemm"] = deep_gemm()
            b.dg = DG
        except Exception as ex:  # noqa: BLE001
            b.errors["deepgemm"] = f"{type(ex).__name__}: {ex}"
        return b

    def record(self) -> dict:
        return {"versions": self.versions, "errors": self.errors}

    def candidates(self, fam, shape) -> tuple[list, list[str]]:
        """This family's bridge candidates at the shape, and why any source has none."""
        import baselines as B

        out, refused = [], []
        for src in SOURCES.get(fam.name, ()):
            lib = self.ops if src == "cutlass-vllm" else self.dg
            if lib is None:
                refused.append(f"{src}: {self.errors.get(src) or self.errors.get('torch')}")
                continue
            if fam.name == "fp8-blk128" and (shape.k % BLOCK or shape.n % BLOCK):
                refused.append(f"{src}: 128-block scaling needs n and k multiples of {BLOCK}")
                continue
            if src == "cutlass-vllm" and (shape.n % 16 or shape.k % 16):
                refused.append(f"{src}: vLLM sends n or k off 16 to Triton, not CUTLASS")
                continue
            cfg = {"library": src} | {k: v for k, v in self.versions.items() if k in ("vllm", "deep_gemm", "torch")}
            out.append(B.Candidate(fam.name, src, f"{src}_{fam.name}", cfg, bridge=self))
        return out, refused

    def prepare(self, cand, shape, ops):
        """The call's launcher (stream -> None): torch tensors over A, B and D, scales of ones in the library's layout."""
        torch = self.torch
        m, n, k = shape.m, shape.n, shape.k
        a = torch.as_tensor(_Raw(ops.a.ptr, (m, k), "|u1"), device="cuda").view(torch.float8_e4m3fn)
        b = torch.as_tensor(_Raw(ops.b.ptr, (n, k), "|u1"), device="cuda").view(torch.float8_e4m3fn)
        d = torch.as_tensor(_Raw(ops.d.ptr, (m, n), "<i2"), device="cuda").view(torch.bfloat16)
        f32 = {"dtype": torch.float32, "device": "cuda"}
        kb, nb = -(-k // BLOCK), -(-n // BLOCK)
        if cand.source == "deepgemm":
            sfa = self.dg.get_col_major_tma_aligned_tensor(torch.ones(m, kb, **f32))
            sfb = torch.ones(nb, kb, **f32)
            gemm = self.dg.fp8_gemm_nt

            def run():
                gemm((a, sfa), (b, sfb), d)
        else:
            if cand.family == "fp8-e4m3":
                sa, sb = torch.ones(1, **f32), torch.ones(1, **f32)
            elif cand.family == "fp8-e4m3-rowwise":
                sa, sb = torch.ones(m, 1, **f32), torch.ones(1, n, **f32)
            else:
                sa, sb = torch.ones(kb, m, **f32).t(), torch.ones(nb, kb, **f32).t()
            mm, bt = self.ops.cutlass_scaled_mm, b.t()

            def run():
                mm(d, a, bt, sa, sb, None)
        streams: dict[int, object] = {}

        def launch(stream: int) -> None:
            s = streams.get(stream) or streams.setdefault(stream, torch.cuda.ExternalStream(stream))
            with torch.cuda.stream(s):
                run()
        torch.cuda.synchronize()
        return launch


def deep_gemm() -> tuple[str, str | None]:
    """The DeepGEMM module vLLM calls and its version."""
    for name in DEEP_GEMM:
        if importlib.util.find_spec(name) is not None:
            return name, getattr(importlib.import_module(name), "__version__", None)
    raise ImportError(f"no DeepGEMM: none of {', '.join(DEEP_GEMM)} is installed")


def exempt(gate, paths: list[str]) -> None:
    """Records each library the bridge mapped as not scanned by the SASS gate: a stock divisor, never verified."""
    for p in paths:
        gate.not_scanned[p] = "a stock library bench.py --bridge loads after the arms' gates (a divisor, never verified)"
