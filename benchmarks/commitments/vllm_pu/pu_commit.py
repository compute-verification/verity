"""Commit to Llama's proof-unit outputs inside stock vLLM (torch.compile + CUDA graphs), for ``pu_bench.py``.

Each site (a PU output: a layer's residual stream, or every linear and op output in the fine arms) hashes its 16-bit
tensor into SHA-256 leaves of 256-byte chunks, written into a device digest ring at a static offset in the step;
the last layer's ``finalize`` folds the step's leaves into a Merkle root and writes it, with the step's counters, into
pinned host memory (``pu_hash.cu`` has the formats).  Both are ``torch.library`` custom ops that mutate one
1-element token tensor; the rings live behind the op, so torch.compile sees a tiny mutation and CUDA graphs capture the
kernels, whose ring positions come from device counters.  The ``retain`` arm's leaf kernel also writes every hashed
byte into a pinned host ring (mapped stores from the same kernel: a graph-captured memcpy has frozen addresses, and a
side stream would have to rejoin inside each piecewise graph).

``install(arm)`` patches ``LlamaDecoderLayer``, ``LlamaAttention`` and ``LlamaMLP`` before the engine builds its
model (``pu_ext`` runs it in the engine's worker).  ``fold`` and ``leaf`` are the hashlib reference ``verify`` uses.
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import torch

CHUNK = 256
FINE = ("in_norm", "qkv", "attn", "o_proj", "resid_post", "post_norm", "gate_up", "act", "down")
ARMS = {"control": (), "coarse": ("resid_in",), "fine": FINE, "retain": FINE}
KEEP_BYTES = int(os.environ.get("VLLM_PU_KEEP_GB", "16")) << 30
STEP_RECORDS = 1 << 20

_EXT = None
_STATE = None
_TOKEN: torch.Tensor | None = None
SITES: dict[str, int] = {}          # site name -> index within a layer, for the installed arm
LAYOUT: "Layout | None" = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        src = Path(__file__).with_name("pu_hash.cu")
        _EXT = load(name="verity_pu_hash", sources=[str(src)], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def leaf(site: int, ci: int, n: int, chunk: bytes) -> bytes:
    return hashlib.sha256(b"vcb1" + struct.pack(">IQQ", site, ci, n) + bytes(40) + chunk).digest()


def fold(nodes: list[bytes]) -> bytes:
    while len(nodes) > 1:
        nodes = [hashlib.sha256(nodes[i] + (nodes[i + 1] if i + 1 < len(nodes) else bytes(32))).digest()
                 for i in range(0, len(nodes), 2)]
    return nodes[0]


def site_widths(hf) -> dict[str, int]:
    """Elements per token of each site's tensor (all 16-bit)."""
    h = hf.hidden_size
    hd = getattr(hf, "head_dim", None) or h // hf.num_attention_heads
    q, kv, i = hf.num_attention_heads * hd, hf.num_key_value_heads * hd, hf.intermediate_size
    return {"resid_in": h, "in_norm": h, "qkv": q + 2 * kv, "attn": q, "o_proj": h, "resid_post": h, "post_norm": h,
            "gate_up": 2 * i, "act": i, "down": h}


class Layout:
    """Where each site's leaves sit in a step: site (layer, k) has id ``layer * len(names) + k`` and starts at
    ``tokens * chunk_offset[layer][k]`` leaves into the step."""

    def __init__(self, names: tuple[str, ...], widths: dict[str, int], layers: int, max_tokens: int):
        self.names, self.layers, self.max_tokens = names, layers, max_tokens
        self.widths = tuple(widths[n] for n in names)
        assert all(w % (CHUNK // 2) == 0 for w in self.widths), "every site must be whole 256-byte chunks per token"
        self.per_layer = sum(self.widths)
        self.chunks_per_token = self.per_layer * layers // (CHUNK // 2)
        self.chunk_offset = tuple(tuple((l * self.per_layer + sum(self.widths[:k])) // (CHUNK // 2) for k in range(len(names)))
                                  for l in range(layers))

    def as_dict(self) -> dict:
        return {"sites": dict(zip(self.names, self.widths)), "layers": self.layers, "max_tokens": self.max_tokens,
                "bytes_per_token_step": self.per_layer * self.layers * 2, "chunks_per_token": self.chunks_per_token}


class State:
    """The device digest ring (sized for the largest step), the tree scratch, the counters and the pinned host records."""

    def __init__(self, ring_leaves: int, keep_bytes: int):
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

    def leaf(self, x: torch.Tensor, site: int, offset: int) -> None:
        self.leaf_calls += 1
        ext().leaf_hash(x, site, offset, self.ctr, self.ring, self.keep_ptr, self.RH)

    def finalize(self, n: int, nbytes: int) -> None:
        self.finalize_calls += 1
        ext().finalize(n, nbytes, self.ctr, self.ring, self.scratch, self.RH, self.rec_ptr, STEP_RECORDS, self.mirror_ptr)

    def record(self, step: int) -> tuple[bytes, int, int, int, int]:
        r = self.rec[(step % STEP_RECORDS) * 64:][:64].numpy().tobytes()
        return (r[:32], *struct.unpack("<qqqq", r[32:]))


def state() -> State:
    global _STATE
    if _STATE is None:
        keep = KEEP_BYTES if os.environ.get("VLLM_PU_ARM") == "retain" else 0
        _STATE = State(LAYOUT.max_tokens * LAYOUT.chunks_per_token + (1 << 20), keep)
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


def _hash(mod, name: str, x: torch.Tensor) -> None:
    k = SITES.get(name)
    if k is None:
        return
    layer = mod._pu_layer
    torch.ops.verity_pu.leaf_hash(x, _TOKEN, layer * len(LAYOUT.names) + k, x.shape[0] * LAYOUT.chunk_offset[layer][k])


def _finalize(mod, x: torch.Tensor) -> None:
    if SITES and mod._pu_layer == LAYOUT.layers - 1:
        t = x.shape[0]
        torch.ops.verity_pu.finalize(_TOKEN, t * LAYOUT.chunks_per_token, t * LAYOUT.chunks_per_token * CHUNK)


def install(arm: str) -> None:
    """Patch vLLM's Llama modules to hash the arm's sites (the control arm patches nothing)."""
    global SITES
    names = ARMS[arm]
    if not names:
        return
    SITES = {n: k for k, n in enumerate(names)}
    from vllm.model_executor.models import llama as L

    init0 = L.LlamaDecoderLayer.__init__

    def init(self, *args, **kwargs):
        global LAYOUT, _TOKEN
        init0(self, *args, **kwargs)
        vc = kwargs["vllm_config"] if "vllm_config" in kwargs else args[0]
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        if LAYOUT is None:
            hf = vc.model_config.hf_config
            LAYOUT = Layout(names, site_widths(hf), hf.num_hidden_layers, vc.scheduler_config.max_num_batched_tokens)
            _TOKEN = torch.zeros(1, dtype=torch.int64, device=torch.device("cuda", torch.cuda.current_device()))
        self._pu_layer = self.self_attn._pu_layer = self.mlp._pu_layer = L.extract_layer_index(prefix)

    def layer_forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        _hash(self, "resid_in", residual)
        _hash(self, "in_norm", hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        _hash(self, "resid_post", residual)
        _hash(self, "post_norm", hidden_states)
        hidden_states = self.mlp(hidden_states)
        _finalize(self, hidden_states)
        return hidden_states, residual

    def attn_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        _hash(self, "qkv", qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        _hash(self, "attn", attn_output)
        output, _ = self.o_proj(attn_output)
        _hash(self, "o_proj", output)
        return output

    def mlp_forward(self, x):
        x, _ = self.gate_up_proj(x)
        _hash(self, "gate_up", x)
        x = L.maybe_fused_act_quant(self.act_fn, x, self.down_proj)
        _hash(self, "act", x)
        x, _ = self.down_proj(x)
        _hash(self, "down", x)
        return x

    L.LlamaDecoderLayer.__init__ = init
    L.LlamaDecoderLayer.forward = layer_forward
    L.LlamaAttention.forward = attn_forward
    L.LlamaMLP.forward = mlp_forward


def stats(since_step: int | None = None) -> dict:
    """Counters (from the host-mapped mirror the last finalize wrote) and, given ``since_step``, the token counts of the
    steps since then (from their records)."""
    if _STATE is None:
        return {"installed": bool(SITES)}
    torch.cuda.synchronize()
    m = _STATE.mirror.tolist()
    out = {"installed": True, "steps": m[2], "bytes": m[3], "leaves": m[4], "leaf_calls": _STATE.leaf_calls,
           "finalize_calls": _STATE.finalize_calls, "ring_bytes": _STATE.ring.numel(), "keep_bytes": _STATE.RH * CHUNK,
           "layout": LAYOUT.as_dict()}
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
    """Recompute step ``step`` with hashlib: its root from the device ring's leaves, and (retain arm) every leaf from the
    retained bytes; plus a check that the retained bytes are the model's: each layer's ``act`` = silu(gate) * up of its
    ``gate_up``, rounded to bf16."""
    st = state()
    torch.cuda.synchronize()
    root, n, base, hbase, idx = st.record(step)
    assert idx == step, f"step record {step} was overwritten (holds {idx})"
    leaves_b = _ring_slice(st.ring, base, n, 32)
    leaves = [leaves_b[i * 32:(i + 1) * 32] for i in range(n)]
    out = {"step": step, "n_leaves": n, "tokens": n // LAYOUT.chunks_per_token, "root": root.hex(),
           "root_from_device_leaves_matches": fold(leaves) == root, "nonzero_leaves": sum(l != bytes(32) for l in leaves)}
    if st.keep is None:
        return out
    t = n // LAYOUT.chunks_per_token
    data = _ring_slice(st.keep, hbase, n, CHUNK)
    bad = 0
    for layer in range(LAYOUT.layers):
        for k, w in enumerate(LAYOUT.widths):
            site, off, cnt = layer * len(LAYOUT.names) + k, t * LAYOUT.chunk_offset[layer][k], t * w // (CHUNK // 2)
            for ci in range(cnt):
                if leaf(site, ci, cnt, data[(off + ci) * CHUNK:(off + ci + 1) * CHUNK]) != leaves[off + ci]:
                    bad += 1
    out["leaves_recomputed"] = n
    out["leaves_mismatched"] = bad
    out["root_from_retained_bytes_matches"] = bad == 0 and out["root_from_device_leaves_matches"]
    if "gate_up" in SITES and "act" in SITES:
        def site_tensor(layer: int, name: str) -> torch.Tensor:
            k = SITES[name]
            off, w = t * LAYOUT.chunk_offset[layer][k], LAYOUT.widths[k]
            raw = bytearray(data[off * CHUNK:(off + t * w // (CHUNK // 2)) * CHUNK])
            return torch.frombuffer(raw, dtype=torch.bfloat16).view(t, w)
        equal = total = 0
        worst = 0.0
        for layer in range(LAYOUT.layers):
            gu, act = site_tensor(layer, "gate_up").float(), site_tensor(layer, "act")
            g, u = gu.chunk(2, dim=-1)
            ref = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
            equal += int((ref == act).sum())
            total += act.numel()
            worst = max(worst, float((ref.float() - act.float()).abs().max()))
        out["act_eq_silu_gate_times_up"] = {"bf16_equal_fraction": equal / total, "max_abs_diff": worst, "elements": total}
    return out
