"""``inductor.fused.same_runtime.v1``: re-launch a generated Inductor Triton kernel standalone.

A fused Inductor region is one description node whose identity is the sha256 of the generated
kernel source (``node.extra["kernel_source_sha256"]``; the source itself ships in the bundle as
``kernels/<sha256>.py``, capture-format §11 revision) and whose ``launch`` spec lists every launch
argument in order: output buffers (allocated here for one row), inputs (authenticated opened rows),
weights (opened whole), the boundary token / position, and integers as ``a * rows + b`` fitted at
capture time (``rows`` = 1 here). The kernel is loaded through Inductor's own loader
(``AsyncCompile.triton`` -> ``CachingAutotuner``), pinned to the launch configuration the replay
actually used (``node.extra["launch_config"]``: the autotuner picked it once per kernel; for the
persistent-reduction kernels there is a single config, for pointwise kernels the choice cannot
change the arithmetic), and launched for the one row. Agreement with the row captured inside a
batch of arbitrary composition is the row-independence of the kernel (every row is one ``xnumel``
element or one program of the persistent reduction).

``includes_full_op`` is True: the whole generated kernel runs for the row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from verity_vllm import relations as rel
from verity_vllm.capture import Value
from verity_vllm.description import NodeInstance
from verity_vllm.relations import SAME_RUNTIME, Inputs, RelationError, need
from verity_vllm_adapter.launchspec import int_value
from verity_vllm_adapter.relations_gpu import _TORCH_DTYPES, row, tensor_bytes, to_tensor

SOURCES: dict[str, str] = {}  # sha256 -> generated kernel source (loaded from the bundle)
_KERNELS: dict[tuple[str, str, str], Any] = {}  # (sha256, fn name, pinned config repr) -> CachingAutotuner
LOADED_FROM: list[str] = []
# W1cF F8: per description node, which launch config the standalone launch was pinned to and where it
# came from ("production" = production's recorded autotune best_config carried in the node as
# `launch_config_production`; "single-config" = the kernel has one config, nothing to choose;
# "replay" = only the replay's choice is recorded — said so in the Table 2 row, never silent).
LAUNCH_CONFIG_USED: dict[str, dict[str, Any]] = {}


def choose_launch_config(extra: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """(config, source) for a fused node: production's recorded best_config when the description
    carries it, else the replay's, else none."""
    prod = extra.get("launch_config_production")
    if isinstance(prod, dict) and prod:
        return prod, "production"
    rep = extra.get("launch_config")
    if isinstance(rep, dict) and rep:
        return rep, "replay"
    return None, "none"


def load_sources(kernels_dir: Path) -> dict[str, Any]:
    """Load ``kernels/<sha256>.py`` from a bundle; every file must hash to its name."""
    d = Path(kernels_dir)
    out: dict[str, Any] = {"dir": str(d), "loaded": 0, "bad_hash": []}
    if not d.is_dir():
        out["missing"] = True
        return out
    for p in sorted(d.glob("*.py")):
        src = p.read_text()
        sha = hashlib.sha256(src.encode()).hexdigest()
        if sha != p.stem:
            out["bad_hash"].append(p.name)
            continue
        SOURCES[sha] = src
        out["loaded"] += 1
    LOADED_FROM.append(str(d))
    return out


def compile_source(sha: str, fn_name: str) -> Any:
    """Load and precompile (every launch config) a bundled kernel source without pinning: used by the
    verifier's PTX identity step (F10); the PTX lands in this process's ``TRITON_CACHE_DIR``."""
    return _kernel(sha, fn_name, None)


def _kernel(sha: str, fn_name: str, launch_config: dict[str, Any] | None) -> Any:
    key = (sha, fn_name, repr(sorted((launch_config or {}).items())))
    k = _KERNELS.get(key)
    if k is not None:
        return k
    src = SOURCES.get(sha)
    if src is None:
        raise RelationError(f"generated kernel source sha256:{sha[:16]} not in the bundle (kernels/ loaded from {LOADED_FROM or 'nowhere'})")
    if hashlib.sha256(src.encode()).hexdigest() != sha:
        raise RelationError("kernel source hash mismatch")
    from torch._inductor.async_compile import AsyncCompile

    ac = AsyncCompile()
    k = ac.triton(fn_name, src, device_str="cuda")
    if hasattr(k, "result") and not hasattr(k, "run"):
        k = k.result()
    try:
        k.precompile()
    except TypeError:
        k.precompile(warm_cache_only=False)
    k.verity_n_configs = len(k.launchers)  # before pinning: how many configs the autotuner could pick from
    if launch_config:
        want = (dict(launch_config.get("kwargs", {})), int(launch_config.get("num_warps", -1)), int(launch_config.get("num_stages", -1)))
        matching = [l for l in k.launchers if (dict(l.config.kwargs), int(l.config.num_warps), int(l.config.num_stages)) == want]
        if not matching:
            have = [(dict(l.config.kwargs), l.config.num_warps, l.config.num_stages) for l in k.launchers]
            raise RelationError(f"recorded launch config {want} is not among the kernel's configs {have}")
        k.launchers = matching[:1]
        k.configs = None
    _KERNELS[key] = k
    return k


@rel.register("inductor.fused.same_runtime.v1", klass=SAME_RUNTIME, op_type="other", includes_full_op=True, owner="W1c",
              doc="CAPTURE-LEVEL relation (GUIDANCE-2026-09-07 §10.2: not a Verity primitive, not a VU): generated Inductor "
                  "Triton kernel (bundle kernels/<sha256>.py, identity = source sha256) launched standalone through Inductor's "
                  "loader for the one row, pinned to production's recorded autotune best_config (launch_config_production; "
                  "else the replay's, said so in the Table 2 row); outputs by launch-argument order. ASSUMES the generated "
                  "kernel is row-independent: the 1-row launch must equal the row computed inside production's batch.")
def inductor_fused_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    ex = inst.node.extra
    sha = str(ex.get("kernel_source_sha256", ""))
    fn_name = str(ex.get("kernel_fn") or ex.get("kernel_name") or "")
    launch = ex.get("launch") or {}
    slots = launch.get("slots") or []
    if not sha or not fn_name or not slots:
        raise RelationError("node lacks kernel identity / launch spec")
    cfg, source = choose_launch_config(ex)
    k = _kernel(sha, fn_name, cfg)
    n_configs = int(getattr(k, "verity_n_configs", len(k.launchers)))
    if n_configs == 1:
        source = "single-config"
    elif cfg is None:
        raise RelationError(f"kernel {fn_name} has {n_configs} launch configs and neither production's nor the replay's choice is recorded")
    LAUNCH_CONFIG_USED.setdefault(inst.node.op, {"source": source, "config": cfg, "n_configs": n_configs, "kernel": fn_name})
    n = int(launch.get("n_args", len(slots)))
    args: list[Any] = [None] * n
    out_for_port: dict[int, int] = {}
    dev = "cuda"
    for s in slots:
        i = int(s["index"])
        kind = s["kind"]
        if kind == "out":
            dt = _TORCH_DTYPES.get(s["dtype"])
            if dt is None:
                raise RelationError(f"unsupported output dtype {s['dtype']}")
            args[i] = torch.empty((1, *[int(d) for d in s["row_shape"]]), dtype=dt, device=dev)
            out_for_port[int(s["port"])] = i
        elif kind == "tensor":
            role = s.get("role")
            name = s["name"]
            if role == "in":
                args[i] = row(inputs, inst, name).reshape(1, *[int(d) for d in s["row_shape"]]).contiguous()
            elif role == "weight":
                v = need(inputs, name)
                args[i] = to_tensor(v)
            elif role == "token":
                tok = rel.scalar_int(need(inputs, name))
                args[i] = torch.tensor([tok], dtype=_TORCH_DTYPES.get(s["dtype"], torch.int64), device=dev)
            elif role == "positions":
                args[i] = torch.tensor([inst.t], dtype=_TORCH_DTYPES.get(s["dtype"], torch.int64), device=dev)
            else:
                raise RelationError(f"tensor launch arg {name!r} has role {role!r} (unbound at capture)")
        elif kind == "int":
            try:
                args[i] = int_value(s, 1)
            except ValueError as e:
                raise RelationError(str(e)) from None
        elif kind == "const":
            args[i] = s.get("value")
        else:
            raise RelationError(f"unknown launch slot kind {kind!r}")
    if any(a is None for a in args):
        missing = [j for j, a in enumerate(args) if a is None]
        raise RelationError(f"launch args {missing} not reconstructed")
    if inst.port not in out_for_port:
        raise RelationError(f"port {inst.port} is not an output of the launch")
    stream = torch.cuda.current_stream().cuda_stream
    k.run(*args, stream=stream)
    torch.cuda.synchronize()
    return tensor_bytes(args[out_for_port[inst.port]][0])


def install() -> None:
    """Importing registers the relation; kept for symmetry."""
    return None
