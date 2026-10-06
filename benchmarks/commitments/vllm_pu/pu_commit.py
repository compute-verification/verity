"""Commit to a model's proof-unit outputs inside stock vLLM (torch.compile + CUDA graphs), for ``pu_bench.py``.

Each site (a PU output: a layer's residual stream, or every linear and op output in the fine arms) hashes its tensor's
bytes, one zero-padded row of whole 256-byte chunks per token, into SHA-256 leaves written into a device digest ring at a
static offset in the step; the last layer's ``finalize`` folds the step's leaves into a Merkle root and writes it, with
the step's counters, into pinned host memory (``pu_hash.cu`` has the formats).  Both are ``torch.library`` custom ops that
mutate one 1-element token tensor; the rings live behind the op, so torch.compile sees a tiny mutation and CUDA graphs
capture the kernels, whose ring positions come from device counters.  The ``retain`` arm's leaf kernel also writes every
hashed byte into a pinned host ring (mapped stores from the same kernel: a graph-captured memcpy has frozen addresses, and
a side stream would have to rejoin inside each piecewise graph).  The ``staged`` arm commits the same leaves and roots as
``fine``, but a site of fewer than ``DIRECT_MIN`` chunks in a step of at most ``STAGE_TOKENS`` tokens is only copied (in
stream) to its leaf positions in a staging buffer, and the step's staged sites are hashed in one launch before the fold: a
leaf costs five dependent SHA-256 compressions in one thread, so a small site's launch is latency-bound (~9 us) however
few bytes it has, and a decode step has hundreds of them.

``install(arm)`` wraps the model loader (``pu_ext`` runs it in the engine's worker) so that each loaded model's decoder
layers are instrumented before torch.compile traces them, whatever the architecture: ``coarse`` hashes the residual stream
entering each layer (the input of layer 0, else the residual its ``input_layernorm`` returns); the fine arms hash the output
of every linear, norm, attention, activation and fused-MoE module in a layer, and the residual each later norm returns.  A
site exists only where Dynamo traces it: the step layout is the sites in trace order, fixed at the last layer's
``finalize``.  A module that runs only inside an opaque op (an expert's linears inside fused MoE, MLA's ``kv_b_proj``) has
no site and is listed in ``stats()["untraced"]``.  ``fold`` and ``leaf`` are the hashlib reference ``verify`` uses.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

CHUNK = 256
ARMS = ("control", "coarse", "fine", "retain", "staged")
KEEP_BYTES = int(os.environ.get("VLLM_PU_KEEP_GB", "16")) << 30
STEP_RECORDS = 1 << 20
STAGE_TOKENS = int(os.environ.get("VLLM_PU_STAGE_TOKENS", "512"))
DIRECT_MIN = int(os.environ.get("VLLM_PU_DIRECT_MIN", "8192"))

_EXT = None
_STATE = None
_TOKEN: torch.Tensor | None = None
ARM = "control"
MAX_TOKENS = 0
SITES: dict[str, int] = {}                  # "module:role" -> site index, in trace order
SITE_LIST: list[tuple[str, int]] = []       # (name, bytes per token)
OFFS: list[int] = [0]                       # per-token chunk offset of each site, then the total
FROZEN = False
WRAPPED: list[str] = []                     # the qualified name of every module given sites
LAYOUT: "Layout | None" = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        src = Path(__file__).with_name("pu_hash.cu")
        _EXT = load(name="verity_pu_hash", sources=[str(src)], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def leaf(site: int, ci: int, n: int, chunk: bytes) -> bytes:
    return hashlib.sha256(chunk + b"vcb1" + struct.pack(">IQQ", site, ci, n)).digest()


def fold(nodes: list[bytes]) -> bytes:
    while len(nodes) > 1:
        nodes = [hashlib.sha256(nodes[i] + (nodes[i + 1] if i + 1 < len(nodes) else bytes(32))).digest()
                 for i in range(0, len(nodes), 2)]
    return nodes[0]


class Layout:
    """A step's sites in order: site ``s`` (``bytes_per_token[s]`` bytes per token, zero-padded to ``site_chunks[s]`` whole
    chunks) has id ``s`` and its leaves start ``tokens * chunk_offset[s]`` leaves into the step."""

    def __init__(self, sites: list[tuple[str, int]], max_tokens: int):
        self.names = tuple(n for n, _ in sites)
        self.bytes_per_token = tuple(w for _, w in sites)
        self.site_chunks = tuple(-(-w // CHUNK) for w in self.bytes_per_token)
        self.chunk_offset = tuple(sum(self.site_chunks[:s]) for s in range(len(sites) + 1))
        self.chunks_per_token = self.chunk_offset[-1]
        self.max_tokens = max_tokens

    def as_dict(self) -> dict:
        return {"sites": len(self.names), "max_tokens": self.max_tokens, "bytes_per_token_step": sum(self.bytes_per_token),
                "padded_bytes_per_token_step": self.chunks_per_token * CHUNK, "chunks_per_token": self.chunks_per_token,
                "padded_sites": sum(w % CHUNK != 0 for w in self.bytes_per_token),
                "first_sites": dict(zip(self.names[:24], self.bytes_per_token[:24]))}


class State:
    """The device digest ring (sized for the largest step), the tree scratch, the counters and the pinned host records."""

    def __init__(self, ring_leaves: int, keep_bytes: int, stage: "Layout | None" = None):
        e = ext()
        dev = torch.device("cuda", torch.cuda.current_device())
        self.ring = torch.zeros(ring_leaves * 32, dtype=torch.uint8, device=dev)
        self.scratch = torch.empty((2, ring_leaves // 512 + 2, 32), dtype=torch.uint8, device=dev)
        self.ctr = torch.zeros(8, dtype=torch.int64, device=dev)
        self.rec = torch.zeros(STEP_RECORDS * 64, dtype=torch.uint8).pin_memory()
        self.mirror = torch.zeros(8, dtype=torch.int64).pin_memory()
        self.keep = torch.empty(keep_bytes, dtype=torch.uint8, pin_memory=True) if keep_bytes else None
        self.RH = keep_bytes // CHUNK
        self.rec_ptr, self.mirror_ptr = e.device_pointer(self.rec), e.device_pointer(self.mirror)
        self.keep_ptr = e.device_pointer(self.keep) if keep_bytes else 0
        self.leaf_calls = self.finalize_calls = 0
        self.stage_layout = stage
        self.stage = (torch.zeros(STAGE_TOKENS * stage.chunks_per_token * CHUNK, dtype=torch.uint8, device=dev)
                      if stage else None)
        self.stage_offsets = torch.tensor(stage.chunk_offset, dtype=torch.int64, device=dev) if stage else None

    def staged(self, tokens: int, n_chunks: int) -> bool:
        return self.stage is not None and tokens <= STAGE_TOKENS and n_chunks < DIRECT_MIN

    def leaf(self, x: torch.Tensor, site: int, offset: int) -> None:
        self.leaf_calls += 1
        t = x.shape[0]
        rows = x.contiguous().view(torch.uint8).reshape(t, -1)
        pad = -rows.shape[1] % CHUNK
        if pad:
            rows = F.pad(rows, (0, pad))
        if self.staged(t, rows.numel() // CHUNK):
            self.stage.narrow(0, offset * CHUNK, rows.numel()).copy_(rows.reshape(-1))
            return
        ext().leaf_hash(rows, site, offset, self.ctr, self.ring, self.keep_ptr, self.RH)

    def finalize(self, n: int, nbytes: int) -> None:
        self.finalize_calls += 1
        if self.stage is not None:
            lay = self.stage_layout
            tokens = n // lay.chunks_per_token
            if tokens <= STAGE_TOKENS:
                ext().stage_hash(self.stage, tokens, self.stage_offsets, lay.chunks_per_token, DIRECT_MIN, self.ctr, self.ring)
        ext().finalize(n, nbytes, self.ctr, self.ring, self.scratch, self.RH, self.rec_ptr, STEP_RECORDS, self.mirror_ptr)

    def record(self, step: int) -> tuple[bytes, int, int, int, int]:
        r = self.rec[(step % STEP_RECORDS) * 64:][:64].numpy().tobytes()
        return (r[:32], *struct.unpack("<qqqq", r[32:]))


def state() -> State:
    global _STATE, LAYOUT
    if _STATE is None:
        if not FROZEN:
            raise RuntimeError("a commitment op ran before the step layout was fixed (was the model traced?)")
        LAYOUT = Layout(SITE_LIST, MAX_TOKENS)
        _STATE = State(LAYOUT.max_tokens * LAYOUT.chunks_per_token + (1 << 20), KEEP_BYTES if ARM == "retain" else 0,
                       LAYOUT if ARM == "staged" else None)
    return _STATE


@torch.library.custom_op("verity_pu::leaf_hash", mutates_args=("token",))
def leaf_hash(x: torch.Tensor, token: torch.Tensor, site: int, offset: int) -> None:
    state().leaf(x, site, offset)


@leaf_hash.register_fake
def _(x, token, site, offset):
    return None


@torch.library.custom_op("verity_pu::finalize", mutates_args=("token",))
def finalize(token: torch.Tensor, n_leaves: int, nbytes: int) -> None:
    state().finalize(n_leaves, nbytes)


@finalize.register_fake
def _(token, n_leaves, nbytes):
    return None


# The bookkeeping runs in Python while Dynamo traces (assume_constant_result calls it and keeps its result as a constant),
# so the sites are numbered in trace order and their offsets are static in the compiled graph.
@torch._dynamo.assume_constant_result
def _site(name: str, w: int) -> int:
    s = SITES.get(name)
    if s is None:
        if FROZEN:
            raise RuntimeError(f"site {name} first traced after the step layout was fixed")
        s = SITES[name] = len(SITE_LIST)
        SITE_LIST.append((name, w))
        OFFS.append(OFFS[-1] + -(-w // CHUNK))
    return s


@torch._dynamo.assume_constant_result
def _offset(s: int) -> int:
    return OFFS[s]


@torch._dynamo.assume_constant_result
def _freeze() -> int:
    global FROZEN
    FROZEN = True
    return OFFS[-1]


def _hash(mod, role: str, x) -> None:
    if not isinstance(x, torch.Tensor) or x.dim() < 1:
        return
    w = x.element_size()
    for d in x.shape[1:]:
        w *= d
    s = _site(f"{mod._pu_name}:{role}", w)
    torch.ops.verity_pu.leaf_hash(x, _TOKEN, s, x.shape[0] * _offset(s))


def _observe(mod, args, kwargs, out) -> None:
    for role in mod._pu_roles:
        if role == "out":
            _hash(mod, role, out[0] if isinstance(out, tuple) else out)
        elif role == "resid":
            if isinstance(out, tuple) and len(out) > 1:
                _hash(mod, role, out[1])
        elif role == "resid_in":
            residual = args[1] if len(args) > 1 else kwargs.get("residual")
            _hash(mod, role, out[1] if residual is not None else (args[0] if args else kwargs["x"]))
        elif role == "all":
            for i, t in enumerate(out if isinstance(out, tuple) else (out,)):
                _hash(mod, f"out{i}", t)


def _finalize_after(mod, out) -> None:
    total = _freeze()
    if total:
        t = (out[0] if isinstance(out, tuple) else out).shape[0]
        torch.ops.verity_pu.finalize(_TOKEN, t * total, t * total * CHUNK)


_SUBCLASSES: dict[type, type] = {}


def _wrap(mod, name: str, roles: tuple[str, ...] = (), last: bool = False) -> None:
    """Give ``mod`` a subclass of its class whose forward hashes ``roles`` of its output while Dynamo traces it (and, for
    the last decoder layer, folds the step after it)."""
    cls = type(mod)
    if cls.__dict__.get("_pu_wrapped"):
        cls = cls.__mro__[1]
    sub = _SUBCLASSES.get(cls)
    if sub is None:
        def forward(self, *args, **kwargs):
            out = cls.forward(self, *args, **kwargs)
            if torch.compiler.is_compiling():
                _observe(self, args, kwargs, out)
                if self._pu_last:
                    _finalize_after(self, out)
            return out
        sub = _SUBCLASSES[cls] = type(cls.__name__, (cls,), {"forward": forward, "_pu_wrapped": True,
                                                              "__module__": cls.__module__, "__qualname__": cls.__qualname__})
    mod.__class__ = sub
    mod._pu_roles = tuple(getattr(mod, "_pu_roles", ())) + roles
    mod._pu_last = getattr(mod, "_pu_last", False) or last
    mod._pu_name = name
    if roles:
        WRAPPED.append(name)


def _site_types() -> tuple[tuple[type, ...], ...]:
    from vllm.model_executor.layers import activation, layernorm
    from vllm.model_executor.layers.attention import Attention, MLAAttention
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import MoERunnerInterface as FusedMoE
    except ImportError:
        from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.custom_op import CustomOp
    acts = tuple(c for c in vars(activation).values() if isinstance(c, type) and issubclass(c, CustomOp) and c is not CustomOp)
    norms = tuple(c for c in vars(layernorm).values() if isinstance(c, type) and issubclass(c, torch.nn.Module)
                  and c.__module__ == layernorm.__name__)
    return (LinearBase, Attention, MLAAttention, *acts), norms, (FusedMoE,)


def _decoder_layers(model) -> tuple[str, torch.nn.ModuleList]:
    lists = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and n.split(".")[-1] == "layers"]
    if not lists:
        raise RuntimeError(f"no decoder layer list in {type(model).__name__}")
    return max(lists, key=lambda nm: len(nm[1]))


def instrument(model, arm: str, max_tokens: int) -> dict:
    global ARM, MAX_TOKENS, _TOKEN
    ARM, MAX_TOKENS = arm, max_tokens
    if arm == "control":
        return {}
    _TOKEN = torch.zeros(1, dtype=torch.int64, device=torch.device("cuda", torch.cuda.current_device()))
    plain, norms, moe = _site_types()
    lname, layers = _decoder_layers(model)
    real = [(i, l) for i, l in enumerate(layers) if hasattr(l, "input_layernorm")]

    def walk(mod, prefix):
        for n, c in mod.named_children():
            q = f"{prefix}.{n}"
            if isinstance(c, moe):
                _wrap(c, q, ("all",))
            elif isinstance(c, norms):
                _wrap(c, q, ("out",) if n == "input_layernorm" else ("out", "resid"))
            elif isinstance(c, plain):
                _wrap(c, q, ("out",))
            else:
                walk(c, q)

    for i, layer in real:
        if arm == "coarse":
            _wrap(layer.input_layernorm, f"{lname}.{i}.input_layernorm", ("resid_in",))
        else:
            walk(layer, f"{lname}.{i}")
    i, last = real[-1]
    _wrap(last, f"{lname}.{i}", last=True)
    return {"layers": len(real), "wrapped": len(WRAPPED)}


def install(arm: str) -> None:
    """Wrap vLLM's model loader so the loaded model is instrumented for ``arm`` (the control arm instruments nothing)."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    load0 = BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, *args, **kwargs):
        model = load0(self, vllm_config, model_config, *args, **kwargs)
        instrument(model, arm, vllm_config.scheduler_config.max_num_batched_tokens)
        return model

    BaseModelLoader.load_model = load_model


def stats(since_step: int | None = None) -> dict:
    """Counters (from the host-mapped mirror the last finalize wrote) and, given ``since_step``, the token counts of the
    steps since then (from their records)."""
    traced = {k.rsplit(":", 1)[0] for k in SITES}
    untraced: dict[str, int] = {}
    for n in WRAPPED:
        if n not in traced:
            p = re.sub(r"\.\d+\.", ".*.", n)
            untraced[p] = untraced.get(p, 0) + 1
    if _STATE is None:
        return {"installed": bool(WRAPPED), "sites": len(SITE_LIST), "untraced": untraced}
    torch.cuda.synchronize()
    m = _STATE.mirror.tolist()
    out = {"installed": True, "steps": m[2], "bytes": m[3], "leaves": m[4], "leaf_calls": _STATE.leaf_calls,
           "finalize_calls": _STATE.finalize_calls, "ring_bytes": _STATE.ring.numel(), "keep_bytes": _STATE.RH * CHUNK,
           "layout": LAYOUT.as_dict(), "sites": len(SITE_LIST), "untraced": untraced,
           "stage_bytes": _STATE.stage.numel() if _STATE.stage is not None else 0}
    if since_step is not None:
        tokens: dict[int, int] = {}
        for s in range(max(since_step, m[2] - STEP_RECORDS), m[2]):
            t = _STATE.record(s)[1] // LAYOUT.chunks_per_token
            tokens[t] = tokens.get(t, 0) + 1
        out["step_tokens"] = dict(sorted(tokens.items()))
    return out


def _ring_slice(ring: torch.Tensor, start: int, n: int, width: int) -> bytes:
    """``n`` records of ``width`` bytes from a ring of records, from record ``start``, wrapping."""
    total = ring.numel() // width
    start %= total
    first = min(n, total - start)
    parts = [ring[start * width:(start + first) * width]]
    if first < n:
        parts.append(ring[:(n - first) * width])
    return b"".join(p.cpu().numpy().tobytes() for p in parts)


def verify(step: int) -> dict:
    """Recompute step ``step`` with hashlib: its root from the device ring's leaves, (staged arm, last step) every staged
    leaf from the staging buffer, and (retain arm) every leaf from the retained bytes."""
    st = state()
    torch.cuda.synchronize()
    root, n, base, hbase, idx = st.record(step)
    assert idx == step, f"step record {step} was overwritten (holds {idx})"
    leaves_b = _ring_slice(st.ring, base, n, 32)
    leaves = [leaves_b[i * 32:(i + 1) * 32] for i in range(n)]
    out = {"step": step, "n_leaves": n, "tokens": n // LAYOUT.chunks_per_token, "root": root.hex(),
           "root_from_device_leaves_matches": fold(leaves) == root, "nonzero_leaves": sum(l != bytes(32) for l in leaves)}
    t = n // LAYOUT.chunks_per_token

    def recompute(data: bytes, only_staged: bool) -> tuple[int, int]:
        done = bad = 0
        for s, c in enumerate(LAYOUT.site_chunks):
            off, cnt = t * LAYOUT.chunk_offset[s], t * c
            if only_staged and not st.staged(t, cnt):
                continue
            for ci in range(cnt):
                done += 1
                bad += leaf(s, ci, cnt, data[(off + ci) * CHUNK:(off + ci + 1) * CHUNK]) != leaves[off + ci]
        return done, bad

    if st.stage is not None and step == int(st.mirror[2]) - 1:
        done, bad = recompute(st.stage[:n * CHUNK].cpu().numpy().tobytes(), True)
        out.update(staged_leaves_recomputed=done, staged_leaves_mismatched=bad,
                   root_from_staged_bytes_matches=done > 0 and bad == 0 and out["root_from_device_leaves_matches"])
    if st.keep is not None:
        done, bad = recompute(_ring_slice(st.keep, hbase, n, CHUNK), False)
        out.update(leaves_recomputed=done, leaves_mismatched=bad,
                   root_from_retained_bytes_matches=bad == 0 and out["root_from_device_leaves_matches"])
    return out
