"""``same_runtime`` relations for the compiled profile: launch the *Inductor-generated Triton kernel*
of the node standalone on the authenticated operands, M = 1 row.

Every fused Inductor region of the compiled piece is one node (``compiled_fx.program``); its relation
``inductor.<op_type>.same_runtime.v1`` carries in the description constants the kernel's identity
(``kernel.sha256_16`` = hash of the name-normalised source, the source itself lives in the bundle at
``kernels/<hash>.py``) and the launch signature (``launch``: for each positional argument of the
wrapper's ``kernel.run(...)`` call, whether it is an input arg of the node, an output port, a positions
tensor, a token, a literal, or a symbolic size in the row count ``M``). The checker compiles the source
with Inductor's own ``AsyncCompile.triton`` (the same loader the wrapper uses), allocates the outputs
for one row, evaluates the sizes at ``M = 1`` and launches. Per-row kernels (pointwise, per-row
persistent reductions, rope with an explicit positions tensor) give the row's bytes independently of
the batch they were computed in, which is what makes an M = 1 re-execution a check of a row captured
inside an arbitrary batch (same argument as the eager adapter's batch-invariance).

The user-defined Triton GEMM (vLLM's ``matmul_kernel_persistent`` traced through
``triton_kernel_wrapper``) is re-run through vLLM's own Python launcher of the same kernel
(``matmul_persistent``), i.e. the eager adapter's ``gemm.same_runtime.v1`` body: the compiled graph
contains the identical Triton program with the identical constexpr configuration (the kernel source
comparison in ``kernel_identity.json`` lists it as ``matmul_kernel_persistent``).

The checker needs the bundle directory for the kernel sources: ``VERITY_RUN_DIR`` (set by the verifier)
or :func:`configure`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from verity_vllm import relations as rel
from verity_vllm.capture import Value
from verity_vllm.description import NodeInstance
from verity_vllm.relations import SAME_RUNTIME, Inputs, RelationError, need
from verity_vllm_adapter.relations_gpu import _runtime, consts, tensor_bytes, to_tensor

_RUN_DIR: dict[str, Path | None] = {"dir": None}
_KERNELS: dict[str, Any] = {}
_WORK = {"launches": 0, "gemm_rows": 0, "bytes_out": 0}
_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "int32": torch.int32, "int64": torch.int64}


def configure(run_dir: str | os.PathLike[str]) -> None:
    _RUN_DIR["dir"] = Path(run_dir)


def run_dir() -> Path:
    d = _RUN_DIR["dir"] or (Path(os.environ["VERITY_RUN_DIR"]) if os.environ.get("VERITY_RUN_DIR") else None)
    if d is None:
        raise RelationError("compiled_fx relations: set VERITY_RUN_DIR (bundle directory with kernels/)")
    return d


def load_kernel(name: str, sha: str) -> Any:
    """Compile ``kernels/<sha>.py`` with Inductor's loader. The file is keyed by the *name-normalised*
    source, so identical kernels of different pieces (``triton_poi_fused_1`` in piece 0, ``triton_poi_fused_3``
    in the middle pieces) share one file; the function name defined in the file is the one to load."""
    k = _KERNELS.get(sha)
    if k is not None:
        return k
    path = run_dir() / "kernels" / f"{sha}.py"
    if not path.exists():
        raise RelationError(f"kernel source {path} missing from the bundle")
    src = path.read_text()
    m = re.search(r"^def (triton_[A-Za-z0-9_]+)\(", src, re.M)
    defined = m.group(1) if m else name
    from torch._inductor.async_compile import AsyncCompile

    ac = _KERNELS.setdefault("_ac", AsyncCompile())
    k = ac.triton(defined, src, device_str="cuda")
    if hasattr(k, "result"):
        k = k.result()
    _KERNELS[sha] = k
    return k


_SAFE = re.compile(r"^[\sM0-9+\-*/()%,minax]+$")


def eval_sym(expr: str, M: int) -> int:
    e = expr.strip()
    if not _SAFE.match(e):
        raise RelationError(f"unsafe size expression {expr!r}")
    return int(eval(e, {"__builtins__": {}}, {"M": M, "min": min, "max": max}))  # noqa: S307  (whitelisted charset)


def _input_tensor(inputs: Inputs, inst: NodeInstance, name: str, desc: dict[str, Any], M: int) -> torch.Tensor:
    v = inputs.get(name)
    if not isinstance(v, Value):
        raise RelationError(f"{name}: missing opened input")
    t = to_tensor(v)
    if _arg_source(inst, name) != "weight":
        t = t.reshape(1, *t.shape) if t.dim() >= 1 else t.reshape(1)
        if M != 1:
            raise RelationError("M must be 1 for opened rows")
    ri = desc.get("reinterpret")
    if ri:
        shape = [eval_sym(str(s), M) for s in ri["shape"]]
        stride = [eval_sym(str(s), M) for s in ri["stride"]]
        off = eval_sym(str(ri["offset"]), M)
        t = torch.as_strided(t.contiguous(), shape, stride, off)
    return t


def _arg_source(inst: NodeInstance, name: str) -> str:
    for a in inst.node.args:
        if a.name == name:
            return a.source
    return ""


def _launch_kernel(inst: NodeInstance, inputs: Inputs) -> bytes:
    c = consts(inst)
    kinfo = c["kernel"]
    launch = c["launch"]
    outs = {o["port"]: o for o in c["outputs"]}
    M = 1
    args: list[Any] = []
    out_tensors: dict[int, torch.Tensor] = {}
    for d in launch:
        kind = d["kind"]
        if kind == "input":
            args.append(_input_tensor(inputs, inst, d["arg"], d, M))
        elif kind == "output":
            o = outs[d["port"]]
            t = torch.empty((M, *o["row_shape"]), dtype=_TORCH_DTYPES[o["dtype"]], device="cuda")
            out_tensors[d["port"]] = t
            args.append(t)
        elif kind == "positions":
            args.append(torch.tensor([inst.t], dtype=torch.int64, device="cuda"))
        elif kind == "token":
            tok = rel.scalar_int(need(inputs, d["arg"]))
            args.append(torch.tensor([tok], dtype=_TORCH_DTYPES.get(d.get("dtype", "int32"), torch.int32), device="cuda"))
        elif kind == "sym":
            args.append(eval_sym(d["expr"], M))
        elif kind == "const":
            args.append(int(d["value"]))
        else:
            raise RelationError(f"unknown launch arg kind {kind}")
    k = load_kernel(kinfo["name"], kinfo["sha256_16"])
    k.run(*args, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    _WORK["launches"] += 1
    out = out_tensors[inst.port]
    _WORK["bytes_out"] += out.numel() * out.element_size()
    return tensor_bytes(out[0])


def _gemm_user_triton(inst: NodeInstance, inputs: Inputs) -> bytes:
    """matmul_kernel_persistent through vLLM's launcher (matmul_persistent) on the opened row."""
    c = consts(inst)
    rt = _runtime()
    x = None
    w = None
    for d in c["launch"]:
        if d["kind"] != "input":
            continue
        src = _arg_source(inst, d["arg"])
        t = _input_tensor(inputs, inst, d["arg"], d, 1)
        if src == "weight":
            w = t
        elif x is None:
            x = t
    if x is None or w is None:
        raise RelationError("gemm: need one activation row and one weight")
    x = x.reshape(1, -1)
    out = rt["matmul_persistent"](x, w)  # w already carries the wrapper's transposed view (reinterpret_tensor)
    _WORK["gemm_rows"] += 1
    return tensor_bytes(out[0])


def _register() -> list[str]:
    ids = []
    for op_type in ("rmsnorm", "silu_mul", "rope", "embedding", "other"):
        rid = f"inductor.{op_type}.same_runtime.v1"
        if rid not in rel.REGISTRY or not rel.REGISTRY[rid].implemented:
            rel.register(rid, klass=SAME_RUNTIME, op_type=op_type, includes_full_op=True, owner="W1d",
                         doc=f"Inductor-generated Triton kernel of the compiled piece ({op_type} fusion group), compiled from the bundle's kernels/<sha>.py "
                             "with AsyncCompile.triton and launched for the one opened row (M=1) with the wrapper's argument signature.")(_launch_kernel)
        ids.append(rid)
    rid = "inductor.gemm.same_runtime.v1"
    if rid not in rel.REGISTRY or not rel.REGISTRY[rid].implemented:
        rel.register(rid, klass=SAME_RUNTIME, op_type="gemm", includes_full_op=True, owner="W1d",
                     doc="vLLM batch-invariant Triton GEMM (matmul_kernel_persistent, the same user-defined Triton kernel the compiled graph launches "
                         "through triton_kernel_wrapper) re-run via vllm ...batch_invariant.matmul_persistent on the opened row (M=1) and the weight view.")(_gemm_user_triton)
    ids.append(rid)
    return ids


REGISTERED = _register()


def install() -> list[str]:
    return list(REGISTERED)


def work() -> dict[str, int]:
    return dict(_WORK)
