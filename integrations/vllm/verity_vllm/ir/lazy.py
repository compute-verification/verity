"""Lazy `Serve_v1` programs: an O(query) root layout and a streamed program digest (LZ, 2026-09-10; additive).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

Why this module exists.  `Program(bind(Serve, C, LP, STEPS))` (`program.py::_lower_root`) runs the `Serve_v1` body
builder, whose root body holds, for every (layer, position) attention call, the KV-cache argument `kc[0:T]` as a
concatenation of `T` per-position reference parts.  Summed over positions and layers that is `NL * P * (P + 1)`
`Affine` objects with `P = LP + STEPS` (plus the nested `Concat`s the decode loop builds, one level per step, which is
where the `RecursionError` at 2048x1023 comes from: `Concat.slice` recurses through the nesting).  The rest of the
program -- every non-root specialization -- is small: `AttentionHead_v1{T}` has ceil(T/64) block nodes, everything
else is constant-size.  A3 measured 11.7 GB at 1024x511 and an OOM kill at 124.7 GB for 6500x1599 (`data/contract/
a3-sequence/conformance-rows.md`, last addendum).

What it provides, without touching `verity.ir` semantics, digests or existing signatures:

* :class:`ServeLayout` -- the root body of `Serve_v1{C, LP, STEPS}` as arithmetic: node index k <-> structural role
  (`("input", pi)`, `("embed", t)`, `("pre"|"attn"|"post", t, L)`, `("final", t)`), gate offset of every node,
  node containing any gate, the callee specialization, argument/output types and the argument reference sequences
  of any node -- each in O(1) or O(log P) time and O(query) memory.  Gate counts of the per-`T` attention
  specializations come from a closed form over the (at most BN + 1) distinct `AttnBlock` specializations; the
  streamed encoder asserts that form against every materialised `AttentionHead_v1{T}` body it encodes.
* :class:`LazyProgram` -- duck-type of :class:`verity.ir.program.Program` (`fn`, `root`, `circuit`, `gates`,
  `input_gates`, `input_coll`, `annotations`) whose root `SpecializedDefinition` carries a :class:`LazyRootBody`:
  `nodes[k]` materialises root node k on demand (an attention node at position t costs O(t) references, cached in a
  small LRU), `offsets[k]` and `node_at(g)` are arithmetic.  `Circuit`, `FunctionCall`, `Scope`, `resolve`,
  `param_leaf` and `Circuit.gate` from `layout.py` run on it unchanged, so a query touches only the root calls it
  addresses (route (ii) of A3 §(f')).  Materialised attention arguments are *flat* `Concat`s of the same per-position
  `Affine` parts the full build nests -- the same reference sequence (`Refs.__getitem__` agrees leaf by leaf), and
  the codec flattens nested concatenations before encoding, so nothing observable differs.
* :func:`encode_program_stream` / :func:`lazy_program_digest` -- the descriptor bytes `canonical_json(encode_program(
  Program(sp)) minus annotations)` produced as a stream without materialising the root body (route (i)): every
  non-root definition is encoded by the real codec (`codec._encode_definition`, same interning tables, same
  dependency walk order as `codec.encode_program`), and the root body's nodes are emitted one at a time with the
  v1 alias rule applied by construction (an attention KV argument at position t >= 2 is the previous position's
  argument plus one part -- exactly the `_BodyEncoder` alias hit, verified byte-for-byte against the real encoder at
  every shape where the full build exists).  The digest of the stream is `program_digest(encode_program(Program(sp)))`.

Only `Serve_v1` (`registry.b1.Serve`) and `Serve_v4` (`registry.b1.ServeV4`, the D54 kernel-geometry attention chain;
same root layout, `Attention_v3` at the attention nodes) are laid out; any other program function raises :class:`LazyUnsupported`.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

from verity.ir.codec import (
    SCHEMA,
    _as_specialized,
    _encode_definition,
    _spec_id,
    _static_functions,
    _Tables,
    canonical_json,
)
from verity.ir.defs import (
    CompositeDefinition,
    FunctionDefinition,
    IdentityConflict,
    Node,
    PrimitiveDefinition,
    SpecializedDefinition,
    bind,
)
from verity.ir.layout import Circuit
from verity.ir.program import input_prim
from verity.ir.refs import Affine, Coll, Concat, Refs
from verity.ir.types import Array, Type, Value

__all__ = [
    "LazyProgram",
    "LazyRootBody",
    "LazyUnsupported",
    "ServeLayout",
    "attention_head_gates",
    "encode_program_stream",
    "lazy_descriptor_size",
    "lazy_program_digest",
]


class LazyUnsupported(TypeError):
    """The program function is not one this module lays out (only `Serve_v1` / `Serve_v4` from `registry.b1`)."""


# ---------------------------------------------------------------------------------------------------------
# gate counts of the per-T attention specializations (closed form over the AttnBlock specializations)
# ---------------------------------------------------------------------------------------------------------


def attention_head_gates(T: int, D: int, block_gates, BN: int | None = None) -> int:
    """`bind(AttentionHead, T=T, D=D).gates` without building the body (`registry.b1.AttentionHead`): ceil(T/BN) blocks
    visited last -> first; the last-visited block is `FIRST` with `NVIS = T - BN*(nb-1)` and, when partial, one shared
    `Const16[0]` pad gate; the others are `FIRST=False, NVIS=BN`; then `Const32[0]`, three `F32AddFtz`, `Fa2InvSum`, and
    two `D`-wide batches.  `block_gates(nvis, first)` -> gates of `AttnBlock{D, NVIS=nvis, FIRST=first}`.

    ``BN`` is the block size of the chain being counted: None / `b1.BN` (64) for `AttentionHead_v1` (Serve_v1);
    `b1.fa2_kblock_n(D)` for `AttentionHead_v3` (Serve_v4) [E7 D54]."""
    if BN is None:
        from verity_vllm.registry.b1 import BN as BN_V1
        BN = BN_V1
    nb = (T + BN - 1) // BN
    nvis_last = T - BN * (nb - 1)
    pad = 1 if nvis_last < BN else 0
    return pad + 1 + block_gates(nvis_last, True) + (nb - 1) * block_gates(BN, False) + 3 + 1 + 2 * D


def b1_mod():
    from verity_vllm.registry import b1
    return b1


class _LRU:
    __slots__ = ("cap", "d")

    def __init__(self, cap: int) -> None:
        self.cap, self.d = cap, OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v) -> None:
        self.d[k] = v
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


# ---------------------------------------------------------------------------------------------------------
# the root layout of Serve_v1{C, LP, STEPS}
# ---------------------------------------------------------------------------------------------------------


class ServeLayout:
    """Root body of `Serve_v1{C, LP, STEPS}` as arithmetic (see the module docstring).

    Root node order (`program._lower_root` + `registry.b1.Serve` / `_forward_positions`):
        0: `batch(Input32, LP)` (prompt)          1: `batch(Input16, W)` (weights)
        prefill:  Embedding(t) for t < LP;  for L < NL: LayerPre(t, L) for t < LP, then (Attention(t, L), LayerPost(t, L))
                  for t < LP;  Final(LP-1)
        decode step s (t = LP+s):  Embedding(t);  for L < NL: LayerPre(t, L), Attention(t, L), LayerPost(t, L);  Final(t)
    Attention(t, L) is `Attention_v1{T=t+1, NH, KVH, D}`; LayerPre(t, 0) is `LayerPre_v1{C, L0=True}`, the other layers
    `L0=False`; every other callee is one specialization per role."""

    def __init__(self, sp: SpecializedDefinition) -> None:
        from verity_vllm.registry import b1
        # [E7 D54] the two serving families share the root layout; they differ in the attention chain (and its block size)
        if not (isinstance(sp, SpecializedDefinition) and sp.defn in (b1.Serve, b1.ServeV4)):
            raise LazyUnsupported(f"lazy layout knows Serve_v1 / Serve_v4 only, not {getattr(sp, 'id', sp)!r}")
        if not sp.complete:
            raise TypeError(f"program function {sp.id} is not fully bound")
        self.sp = sp
        C = sp.bindings["C"]
        self.C = C
        self.LP, self.STEPS = int(sp.bindings["LP"]), int(sp.bindings["STEPS"])
        if self.LP < 1 or self.STEPS < 0:
            raise ValueError("Serve needs LP >= 1 and STEPS >= 0")
        self.P = self.LP + self.STEPS
        self.NL, self.H, self.NH, self.KVH, self.D, self.V = C["NL"], C["H"], C["NH"], C["KVH"], C["D"], C["V"]
        params, ret_t = sp.signature()
        (self.prompt_name, self.prompt_t), (self.weights_name, self.weights_t) = params
        self.ret_type = ret_t
        for name, t in params:
            if t.homogeneous_width() is None or t.leaves == 0:
                raise LazyUnsupported(f"parameter {name!r} of type {t!r} is not one homogeneous Input batch")
        self.W = self.weights_t.leaves
        # callee specializations (bodies of the constant-shape ones are built once; they are small)
        self.emb = bind(b1.Embedding, V=self.V, H=self.H)
        self.pre0 = bind(b1.LayerPre, C=C, L0=True)
        self.pre1 = bind(b1.LayerPre, C=C, L0=False)
        self.post = bind(b1.LayerPost, C=C)
        self.fin = bind(b1.Final, C=C)
        if sp.defn is b1.ServeV4:
            self.family = "Serve_v4"
            self._attn_defn, self._ah_defn, self._ab_defn = b1.AttentionV3, b1.AttentionHeadV3, b1.AttnBlockV3
            self.BN = int(sp.bindings["BN"])  # the declared static (D57)
        else:
            self.family = "Serve_v1"
            self._attn_defn, self._ah_defn, self._ab_defn = b1.Attention, b1.AttentionHead, b1.AttnBlock
            self.BN = b1.BN
        self.g_emb, self.g_pre0, self.g_pre1, self.g_post, self.g_fin = self.emb.gates, self.pre0.gates, self.pre1.gates, self.post.gates, self.fin.gates
        # per-T attention gates from the (<= 65) AttnBlock specializations
        self._ab_gates: dict[tuple[int, bool], int] = {}
        self._A = [0] * (self.P + 1)  # A[T] = sum_{T'=1..T} g_attn(T')
        for T in range(1, self.P + 1):
            self._A[T] = self._A[T - 1] + self.attn_gates(T)
        # typed parameter collections over the two Input batches, in body-scoped refs (node 0 / node 1)
        self.prompt_coll = Coll(self.prompt_t, Affine("n", 0, 0, self.LP))
        self.weights_coll = Coll(self.weights_t, Affine("n", 1, 0, self.W))
        self._emb_w = self.weights_coll["embed"]
        self._norm_w = self.weights_coll["norm"]
        self._layer_w = [self.weights_coll["layers"][L] for L in range(self.NL)]
        self._cos_sin = self.weights_coll["cos_sin"]
        # output types (the real ones: `fn.signature()`)
        self.emb_out = self.emb.signature()[1]
        self.pre_out = self.pre0.signature()[1]
        self.post_out = self.post.signature()[1]
        self.fin_out = self.fin.signature()[1]
        self.attn_out = Array(self.NH, Array(self.D, Value(16)))
        self.kvd = self.KVH * self.D
        self.q_off = self.pre_out.field_offset(1)
        self.k_off = self.pre_out.field_offset(2)
        self.v_off = self.pre_out.field_offset(3)
        # node-index geometry
        self.k_embed0 = 2
        self.k_layer0 = 2 + self.LP
        self.k_final0 = self.k_layer0 + 3 * self.LP * self.NL
        self.step_nodes = 2 + 3 * self.NL
        self.n_nodes = self.k_final0 + 1 + self.STEPS * self.step_nodes
        # gate geometry
        self.G0 = self.LP + self.W + self.LP * self.g_emb  # start of prefill layer 0
        self._blk0 = self.LP * self.g_pre0 + self._A[self.LP] + self.LP * self.g_post
        self._blk1 = self.LP * self.g_pre1 + self._A[self.LP] + self.LP * self.g_post
        self.F0 = self.G0 + self._blk0 + (self.NL - 1) * self._blk1  # start of Final(LP-1)
        self.S0 = self.F0 + self.g_fin  # start of decode step 0
        self._c0 = self.g_emb + self.g_fin + self._PRE(self.NL) + self.NL * self.g_post  # per-step gates without attention
        self.gates = self.S0 + self._D(self.STEPS)
        self._nodes = _LRU(32)

    # -- gate counts ----------------------------------------------------------------------------------------
    def block_gates(self, nvis: int, first: bool) -> int:
        key = (nvis, first)
        g = self._ab_gates.get(key)
        if g is None:
            extra = {"BN": self.BN} if self._ab_defn is b1_mod().AttnBlockV3 else {}
            g = self._ab_gates[key] = bind(self._ab_defn, D=self.D, NVIS=nvis, FIRST=first, **extra).gates
        return g

    def head_gates(self, T: int) -> int:
        return attention_head_gates(T, self.D, self.block_gates, self.BN)

    def attn_gates(self, T: int) -> int:
        """`Attention_v1{T, NH, KVH, D}.gates` = NH heads x `AttentionHead_v1{T, D}.gates` (KVH batches of NH/KVH members)."""
        return self.NH * self.head_gates(T)

    def g_pre(self, L: int) -> int:
        return self.g_pre0 if L == 0 else self.g_pre1

    def _PRE(self, L: int) -> int:
        """Gates of LayerPre over layers 0..L-1."""
        return 0 if L <= 0 else self.g_pre0 + (L - 1) * self.g_pre1

    def _D(self, s: int) -> int:
        """Gates of decode steps 0..s-1."""
        return s * self._c0 + self.NL * (self._A[self.LP + s] - self._A[self.LP])

    def _G(self, L: int) -> int:
        """Start gate of prefill layer block L."""
        return self.G0 + (self._blk0 + (L - 1) * self._blk1 if L >= 1 else 0)

    # -- specializations ----------------------------------------------------------------------------------
    def attn_spec(self, T: int) -> SpecializedDefinition:
        extra = {"BN": self.BN} if self.family == "Serve_v4" else {}
        return bind(self._attn_defn, T=T, NH=self.NH, KVH=self.KVH, D=self.D, **extra)

    def head_spec(self, T: int) -> SpecializedDefinition:
        extra = {"BN": self.BN} if self.family == "Serve_v4" else {}
        return bind(self._ah_defn, T=T, D=self.D, **extra)

    # -- node index <-> role ------------------------------------------------------------------------------------
    def role(self, k: int) -> tuple:
        """`("input", pi)` | `("embed", t)` | `("pre"|"attn"|"post", t, L)` | `("final", t)` of root node k."""
        if not (0 <= k < self.n_nodes):
            raise IndexError(k)
        if k < 2:
            return ("input", k)
        if k < self.k_layer0:
            return ("embed", k - 2)
        if k < self.k_final0:
            r = k - self.k_layer0
            L, r = divmod(r, 3 * self.LP)
            if r < self.LP:
                return ("pre", r, L)
            r -= self.LP
            t, w = divmod(r, 2)
            return ("attn" if w == 0 else "post", t, L)
        if k == self.k_final0:
            return ("final", self.LP - 1)
        s, r = divmod(k - self.k_final0 - 1, self.step_nodes)
        t = self.LP + s
        if r == 0:
            return ("embed", t)
        if r == self.step_nodes - 1:
            return ("final", t)
        L, w = divmod(r - 1, 3)
        return (("pre", "attn", "post")[w], t, L)

    def index(self, role: tuple) -> int:
        """Inverse of :meth:`role`."""
        kind = role[0]
        if kind == "input":
            return role[1]
        if kind == "embed":
            t = role[1]
            return 2 + t if t < self.LP else self.k_final0 + 1 + (t - self.LP) * self.step_nodes
        if kind == "final":
            t = role[1]
            return self.k_final0 if t == self.LP - 1 else self.k_final0 + 1 + (t - self.LP) * self.step_nodes + self.step_nodes - 1
        _, t, L = role
        w = ("pre", "attn", "post").index(kind)
        if t < self.LP:
            base = self.k_layer0 + 3 * self.LP * L
            return base + t if kind == "pre" else base + self.LP + 2 * t + (w - 1)
        return self.k_final0 + 1 + (t - self.LP) * self.step_nodes + 1 + 3 * L + w

    def nth_root(self, kind: str, i: int) -> int:
        """Root node index of the i-th root call of `kind` in canonical (emission) order -- the per-kind enumeration rule A1's `RankLayout`
        lacks for the five root-call kinds: `"embed"` (`Embedding_v1`), `"pre"` (`LayerPre_v1`), `"attn"` (`Attention_v1{T}`), `"post"`
        (`LayerPost_v1`), `"final"` (`Final_v1`).  Prefill roots of `pre`/`attn`/`post` are layer-major (L outer, t inner), decode roots
        position-major (t outer, L inner); `role(nth_root(kind, i))` gives (t, L) and `offset` the gate interval."""
        if kind == "embed":
            if not 0 <= i < self.P:
                raise IndexError((kind, i))
            return self.index(("embed", i))
        if kind == "final":
            if not 0 <= i <= self.STEPS:
                raise IndexError((kind, i))
            return self.index(("final", self.LP - 1 + i))
        if kind not in ("pre", "attn", "post"):
            raise KeyError(kind)
        if not 0 <= i < self.P * self.NL:
            raise IndexError((kind, i))
        if i < self.LP * self.NL:
            L, t = divmod(i, self.LP)
        else:
            s, L = divmod(i - self.LP * self.NL, self.NL)
            t = self.LP + s
        return self.index((kind, t, L))

    def name(self, k: int) -> str | None:
        """The structural name (`StructuralCircuitMap` naming == the `Serve_v1` builder's annotations)."""
        r = self.role(k)
        if r[0] == "input":
            return (self.prompt_name, self.weights_name)[r[1]]
        if r[0] == "embed":
            return f"t={r[1]}/embed"
        if r[0] == "final":
            return f"t={r[1]}/final"
        return f"t={r[1]}/layers.{r[2]}/{r[0]}"

    # -- gate offsets -----------------------------------------------------------------------------------------
    def offset(self, k: int) -> int:
        """Root gate offset of node k (k == n_nodes gives the total gate count)."""
        if k == self.n_nodes:
            return self.gates
        r = self.role(k)
        kind = r[0]
        if kind == "input":
            return 0 if r[1] == 0 else self.LP
        if kind == "embed":
            t = r[1]
            if t < self.LP:
                return self.LP + self.W + t * self.g_emb
            return self.S0 + self._D(t - self.LP)
        if kind == "final":
            t = r[1]
            if t == self.LP - 1:
                return self.F0
            s = t - self.LP
            T = t + 1
            return self.S0 + self._D(s) + self.g_emb + self._PRE(self.NL) + self.NL * (self.attn_gates(T) + self.g_post)
        _, t, L = r
        T = t + 1
        if t < self.LP:
            base = self._G(L)
            if kind == "pre":
                return base + t * self.g_pre(L)
            a = base + self.LP * self.g_pre(L) + self._A[t] + t * self.g_post
            return a if kind == "attn" else a + self.attn_gates(T)
        s = t - self.LP
        p = self.S0 + self._D(s) + self.g_emb + self._PRE(L) + L * (self.attn_gates(T) + self.g_post)
        if kind == "pre":
            return p
        a = p + self.g_pre(L)
        return a if kind == "attn" else a + self.attn_gates(T)

    def node_at(self, g: int) -> int:
        """Index of the root node whose interval contains gate g."""
        if not (0 <= g < self.gates):
            raise IndexError(g)
        if g < self.LP:
            return 0
        if g < self.LP + self.W:
            return 1
        if g < self.G0:
            return 2 + (g - self.LP - self.W) // self.g_emb
        if g < self.F0:
            r = g - self.G0
            if r < self._blk0:
                L = 0
            else:
                L = 1 + (r - self._blk0) // self._blk1
                r -= self._blk0 + (L - 1) * self._blk1
            gp = self.g_pre(L)
            if r < self.LP * gp:
                return self.index(("pre", r // gp, L))
            r -= self.LP * gp
            # position t with A[t] + t*g_post <= r < A[t+1] + (t+1)*g_post
            lo, hi = 0, self.LP - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._A[mid] + mid * self.g_post <= r:
                    lo = mid
                else:
                    hi = mid - 1
            t = lo
            r -= self._A[t] + t * self.g_post
            return self.index(("attn" if r < self.attn_gates(t + 1) else "post", t, L))
        if g < self.S0:
            return self.k_final0
        r = g - self.S0
        lo, hi = 0, self.STEPS - 1
        while lo < hi:  # step s with D(s) <= r < D(s+1)
            mid = (lo + hi + 1) // 2
            if self._D(mid) <= r:
                lo = mid
            else:
                hi = mid - 1
        s = lo
        r -= self._D(s)
        t = self.LP + s
        T = t + 1
        if r < self.g_emb:
            return self.index(("embed", t))
        r -= self.g_emb
        ga = self.attn_gates(T)
        lay0 = self.g_pre0 + ga + self.g_post
        lay1 = self.g_pre1 + ga + self.g_post
        if r < lay0:
            L = 0
        else:
            L = 1 + (r - lay0) // lay1
            if L >= self.NL:
                return self.index(("final", t))
            r -= lay0 + (L - 1) * lay1
        if r < self.g_pre(L):
            return self.index(("pre", t, L))
        r -= self.g_pre(L)
        return self.index(("attn" if r < ga else "post", t, L))

    # -- node contents -----------------------------------------------------------------------------------------
    def fn(self, k: int):
        """Callee (primitive / specialization) of root node k."""
        r = self.role(k)
        kind = r[0]
        if kind == "input":
            return input_prim(self.prompt_t.homogeneous_width() if r[1] == 0 else self.weights_t.homogeneous_width())
        if kind == "embed":
            return self.emb
        if kind == "final":
            return self.fin
        if kind == "pre":
            return self.pre0 if r[2] == 0 else self.pre1
        if kind == "attn":
            return self.attn_spec(r[1] + 1)
        return self.post

    def form(self, k: int) -> str:
        return "batch" if k < 2 else "call"

    def arg_types(self, k: int) -> tuple[Type, ...]:
        r = self.role(k)
        kind = r[0]
        if kind == "input":
            return ()
        if kind == "embed":
            return (self.prompt_t.elem, self._emb_w.type)
        if kind == "pre":
            return (self.emb_out, self.emb_out, self._layer_w[r[2]].type, self._cos_sin.type.elem)
        if kind == "attn":
            T = r[1] + 1
            kv_t = Array(T, Array(self.KVH, Array(self.D, Value(16))))
            return (self.attn_out, kv_t, kv_t)
        if kind == "post":
            return (self.attn_out, self.emb_out, self._layer_w[r[2]].type)
        return (self.emb_out, self.emb_out, self._norm_w.type, self._emb_w.type)

    def out_type(self, k: int) -> Type:
        r = self.role(k)
        kind = r[0]
        if kind == "input":
            return Array(self.LP if r[1] == 0 else self.W, Value(self.prompt_t.homogeneous_width() if r[1] == 0 else self.weights_t.homogeneous_width()))
        return {"embed": self.emb_out, "pre": self.pre_out, "attn": self.attn_out, "post": self.post_out, "final": self.fin_out}[kind]

    def kv_part(self, t: int, L: int, which: str) -> Affine:
        """The reference part contributed by position t to layer L's K (`which == "k"`) or V cache: leaves 2 / 3 of LayerPre(t, L)."""
        return Affine("n", self.index(("pre", t, L)), self.k_off if which == "k" else self.v_off, self.kvd)

    def arg_refs(self, k: int, i: int) -> Refs:
        """Reference sequence of argument i of root node k (attention KV arguments: a flat `Concat` of the per-position parts)."""
        r = self.role(k)
        kind = r[0]
        if kind == "embed":
            t = r[1]
            if i == 0:
                return Affine("n", 0, t, 1) if t < self.LP else Affine("n", self.index(("final", t - 1)), 0, 1)
            return self._emb_w.refs
        if kind == "pre":
            _, t, L = r
            if i == 0 or i == 1:
                if L == 0:
                    return Affine("n", self.index(("embed", t)), 0, self.H)
                return Affine("n", self.index(("post", t, L - 1)), 0 if i == 0 else self.H, self.H)
            if i == 2:
                return self._layer_w[L].refs
            return self._cos_sin[t].refs
        if kind == "attn":
            _, t, L = r
            if i == 0:
                return Affine("n", self.index(("pre", t, L)), self.q_off, self.NH * self.D)
            which = "k" if i == 1 else "v"
            if t == 0:
                return self.kv_part(0, L, which)
            return Concat([self.kv_part(tp, L, which) for tp in range(t + 1)])
        if kind == "post":
            _, t, L = r
            if i == 0:
                return Affine("n", self.index(("attn", t, L)), 0, self.NH * self.D)
            if i == 1:
                return Affine("n", self.index(("pre", t, L)), 0, self.H)
            return self._layer_w[L].refs
        if kind == "final":
            t = r[1]
            kp = self.index(("post", t, self.NL - 1))
            if i == 0:
                return Affine("n", kp, 0, self.H)
            if i == 1:
                return Affine("n", kp, self.H, self.H)
            return self._norm_w.refs if i == 2 else self._emb_w.refs
        raise IndexError((k, i))

    def node(self, k: int) -> Node:
        """Root node k as a `defs.Node` (cached in a small LRU)."""
        n = self._nodes.get(k)
        if n is not None:
            return n
        fn = self.fn(k)
        if k < 2:
            n = Node("batch", fn, (), self.out_type(k), axes=(), n=self.LP if k == 0 else self.W, name=self.name(k))
        else:
            types = self.arg_types(k)
            args = tuple(Coll(t, self.arg_refs(k, i)) for i, t in enumerate(types))
            n = Node("call", fn, args, self.out_type(k), name=self.name(k))
        self._nodes.put(k, n)
        return n

    def ret_refs(self) -> Refs:
        parts = [Affine("n", self.index(("final", self.LP - 1 + j)), 0, 1) for j in range(self.STEPS + 1)]
        return parts[0] if len(parts) == 1 else Concat(parts)

    # -- descriptor (v1) encoding of the root body, node by node -----------------------------------------------
    def _aff_json(self, a: Affine) -> dict:
        return {"affine": ["n", a.idx, a.base, a.count]}

    def encode_refs(self, k: int, i: int) -> Any:
        """`codec._BodyEncoder.refs` of argument i of node k under the v1 alias rule, by construction: single parts verbatim;
        the two-part KV argument of position 1 as a plain concat (position 0's argument is one part and is never recorded);
        position t >= 2's KV argument as `alias(Attention(t-1, L), i)` + its last part (the previous position's argument is
        exactly the (t)-part prefix, recorded first, and lies within the ALIAS_WINDOW)."""
        r = self.role(k)
        if r[0] == "attn" and i >= 1:
            _, t, L = r
            which = "k" if i == 1 else "v"
            if t == 0:
                return self._aff_json(self.kv_part(0, L, which))
            if t == 1:
                return {"concat": [self._aff_json(self.kv_part(0, L, which)), self._aff_json(self.kv_part(1, L, which))]}
            return {"concat": [{"alias": [self.index(("attn", t - 1, L)), i]}, self._aff_json(self.kv_part(t, L, which))]}
        refs = self.arg_refs(k, i)
        assert isinstance(refs, Affine), (k, i, refs)
        return {"affine": [refs.space, refs.idx, refs.base, refs.count]}

    def encode_node(self, k: int, tables: _Tables) -> dict:
        """`codec._encode_node` of root node k (interning through `tables` in the same order: callee, argument types, out)."""
        fn = self.fn(k)
        d: dict[str, Any] = {"form": self.form(k), "fn": tables.callee(_spec_id(fn))}
        d["args"] = [{"type": tables.type(t), "refs": self.encode_refs(k, i)} for i, t in enumerate(self.arg_types(k))]
        d["out"] = tables.type(self.out_type(k))
        if k < 2:
            d["axes"] = []
            d["n"] = self.LP if k == 0 else self.W
        return d

    def encode_ret(self) -> Any:
        r = self.ret_refs()
        if isinstance(r, Affine):
            return self._aff_json(r)
        return {"concat": [self._aff_json(p) for p in r.parts]}

    def intern_node(self, k: int, tables: _Tables) -> None:
        """The interning side effect of `encode_node` alone (pass 1 of the streamed encoder)."""
        tables.callee(_spec_id(self.fn(k)))
        for t in self.arg_types(k):
            tables.type(t)
        tables.type(self.out_type(k))


# ---------------------------------------------------------------------------------------------------------
# a lazy root Body and the LazyProgram
# ---------------------------------------------------------------------------------------------------------


class _LazyNodes:
    """`body.nodes` of the lazy root: `[k]` materialises node k; iteration materialises one node at a time."""

    __slots__ = ("layout",)

    def __init__(self, layout: ServeLayout) -> None:
        self.layout = layout

    def __len__(self) -> int:
        return self.layout.n_nodes

    def __getitem__(self, k: int) -> Node:
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(len(self)))]
        if k < 0:
            k += len(self)
        return self.layout.node(k)

    def __iter__(self) -> Iterator[Node]:
        for k in range(self.layout.n_nodes):
            yield self.layout.node(k)


class _LazyOffsets:
    """`body.offsets` of the lazy root: `len == n_nodes + 1`, `[k]` arithmetic, `[-1]` the gate count."""

    __slots__ = ("layout",)

    def __init__(self, layout: ServeLayout) -> None:
        self.layout = layout

    def __len__(self) -> int:
        return self.layout.n_nodes + 1

    def __getitem__(self, k: int) -> int:
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(len(self)))]
        if k < 0:
            k += len(self)
        if not (0 <= k < len(self)):
            raise IndexError(k)
        return self.layout.offset(k)

    def __iter__(self) -> Iterator[int]:
        for k in range(len(self)):
            yield self.layout.offset(k)


class LazyRootBody:
    """Duck-type of `defs.Body` for the root of a lazy `Serve_v1` program (`params`, `ret_type`, `nodes`, `ret`, `offsets`,
    `gates`, `node_at`)."""

    def __init__(self, layout: ServeLayout) -> None:
        self.layout = layout
        self.params: tuple = ()
        self.ret_type = layout.ret_type
        self.nodes = _LazyNodes(layout)
        self.offsets = _LazyOffsets(layout)
        self.ret = layout.ret_refs()

    @property
    def gates(self) -> int:
        return self.layout.gates

    def node_at(self, g: int) -> int:
        return self.layout.node_at(g)


class LazyProgram:
    """`Program(sp)` for `sp = bind(Serve, C, LP, STEPS)` without materialising the root body (see the module docstring).
    Same surface as :class:`verity.ir.program.Program`; `circuit` is a `layout.Circuit` over the lazy root."""

    def __init__(self, fn: SpecializedDefinition | CompositeDefinition, annotations: dict[str, Any] | None = None) -> None:
        if isinstance(fn, CompositeDefinition):
            fn = bind(fn)
        self.fn = fn
        self.annotations: dict[str, Any] = dict(annotations or {})
        self.layout = ServeLayout(fn)
        _params, ret = fn.signature()
        root_def = CompositeDefinition(f"Root[{fn.id}]", 1, (), lambda S: ((), ret), lambda *_: None, register=False)
        root = SpecializedDefinition(root_def, {}, ())
        root.set_body(LazyRootBody(self.layout))
        root.program_fn = fn  # type: ignore[attr-defined]
        lay = self.layout
        root.input_colls = [Coll(lay.prompt_t, Affine("g", 0, 0, lay.LP)),  # type: ignore[attr-defined]
                            Coll(lay.weights_t, Affine("g", 0, lay.LP, lay.W))]
        self.root = root
        self._circuit: Circuit | None = None

    @property
    def circuit(self) -> Circuit:
        if self._circuit is None:
            self._circuit = Circuit(self.root)
        return self._circuit

    @property
    def gates(self) -> int:
        return self.layout.gates

    @property
    def input_gates(self) -> int:
        return self.layout.LP + self.layout.W

    def input_coll(self, pi: int) -> Coll:
        return self.root.input_colls[pi]  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"LazyProgram({self.fn.id}, gates={self.gates})"


# ---------------------------------------------------------------------------------------------------------
# streamed descriptor / digest
# ---------------------------------------------------------------------------------------------------------


def _release(spec: SpecializedDefinition) -> None:
    """Drop the cached body of a per-T specialization after it was encoded (rebuilt on the next access; the binding
    record, id and signature are unaffected)."""
    spec._body = None


def encode_program_stream(program: LazyProgram, *, release_bodies: bool = True, check_gates: bool = True) -> Iterator[bytes]:
    """The bytes `canonical_json({k: v for k, v in encode_program(Program(sp)).items() if k != "annotations"})` for the
    program of `program.fn`, as a stream, without materialising the root body.

    Pass 1 walks the dependency graph exactly as `codec.encode_program` does (root node callees in root order, each
    callee's body nodes, then function-valued statics; post-order encoding fills the interning tables) using the
    real `codec._encode_definition` for every non-root definition and `ServeLayout.intern_node` for the root.  Pass 2
    emits the descriptor in canonical key order: `callees`, `definitions` (sorted ids; the root's body streamed node by
    node), `program_fn`, `root`, `schema`, `types`.  With `release_bodies` the `Attention_v1{T}` / `AttentionHead_v1{T}`
    bodies are dropped after each use (memory O(P) instead of O(P^2/128) nodes); with `check_gates` every materialised
    `AttentionHead_v1{T}` body's gate count is asserted equal to :func:`attention_head_gates`."""
    lay = program.layout
    tables = _Tables()
    seen: dict[str, FunctionDefinition] = {}
    duplicates: list[tuple[str, FunctionDefinition]] = []
    per_t: set[str] = set()

    def is_per_t(fn: FunctionDefinition) -> bool:
        return isinstance(fn, SpecializedDefinition) and fn.defn in (lay._attn_defn, lay._ah_defn)

    def visit(fn: FunctionDefinition) -> None:
        fid = _spec_id(fn)
        prev = seen.get(fid)
        if prev is not None:
            if prev is not fn:
                duplicates.append((fid, fn))
            return
        seen[fid] = fn
        if isinstance(fn, PrimitiveDefinition):
            return
        assert isinstance(fn, SpecializedDefinition)
        for n in fn.body.nodes:
            visit(n.fn)
        for k in fn.defn.statics:
            for f in _static_functions(fn.bindings[k]):
                visit(_as_specialized(f, fid))
        if check_gates and fn.defn is lay._ah_defn:
            got, want = fn.body.gates, lay.head_gates(fn.bindings["T"])
            if got != want:
                raise AssertionError(f"{fn.id}: built body has {got} gates, the layout's closed form says {want}")
        _encode_definition(fn, SCHEMA, tables)  # interning side effect only (pass 1)
        if release_bodies and is_per_t(fn):
            per_t.add(fid)
            _release(fn)

    root = program.root
    root_fid = _spec_id(root)
    seen[root_fid] = root
    for k in range(lay.n_nodes):
        visit(lay.fn(k))
    for k in root.defn.statics:  # (none for the root; kept parallel to encode_program)
        for f in _static_functions(root.bindings[k]):
            visit(_as_specialized(f, root_fid))
    for k in range(lay.n_nodes):
        lay.intern_node(k, tables)
    pfn = program.fn
    for k in pfn.defn.statics:  # the program function's own binding record (`_program_fn_functions`)
        for f in _static_functions(pfn.bindings[k]):
            visit(_as_specialized(f, _spec_id(pfn)))
    for fid, fn in duplicates:
        if _encode_definition(fn) != _encode_definition(seen[fid]):
            raise IdentityConflict(f"E_IDENTITY: two different definitions claim the specialization {fid!r}")
    n_callees, n_types = len(tables.callees), len(tables.type_json)

    # pass 2: emit
    yield b'{"callees":' + canonical_json(list(tables.callees)) + b',"definitions":{'
    first = True
    for fid in sorted(seen):
        if not first:
            yield b","
        first = False
        yield canonical_json(fid) + b":"
        if fid == root_fid:
            yield b'{"bindings":{},"body":{"nodes":['
            for k in range(lay.n_nodes):
                if k:
                    yield b","
                yield canonical_json(lay.encode_node(k, tables))
            yield (b'],"params":[],"ret":' + canonical_json(lay.encode_ret()) + b',"ret_type":' + canonical_json(lay.ret_type.to_json())
                   + b'},"form":"composite","id":' + canonical_json(root.base_id) + b"}")
        else:
            fn = seen[fid]
            yield canonical_json(_encode_definition(fn, SCHEMA, tables))
            if release_bodies and fid in per_t:
                _release(fn)  # type: ignore[arg-type]
    if (len(tables.callees), len(tables.type_json)) != (n_callees, n_types):
        raise AssertionError("interning tables grew during emission: pass 1 missed a callee or type")
    yield (b'},"program_fn":' + canonical_json(_spec_id(pfn)) + b',"root":' + canonical_json(root_fid) + b',"schema":' + canonical_json(SCHEMA)
           + b',"types":' + canonical_json(tables.type_json) + b"}")


def lazy_program_digest(program: LazyProgram, **kw: Any) -> str:
    """`program_digest(Program(program.fn))` computed from :func:`encode_program_stream`."""
    h = hashlib.sha256()
    for chunk in encode_program_stream(program, **kw):
        h.update(chunk)
    return h.hexdigest()


def lazy_descriptor_size(program: LazyProgram, **kw: Any) -> int:
    return sum(len(c) for c in encode_program_stream(program, **kw))


# ---------------------------------------------------------------------------------------------------------
# measurement CLI
# ---------------------------------------------------------------------------------------------------------


def _rss_gb() -> float:
    import resource
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import random
    import time

    from verity_vllm.registry import b1

    ap = argparse.ArgumentParser(description="lazy Serve_v1 program: streamed digest and O(query) locate, with stage timings and max RSS")
    ap.add_argument("--case", default="B0", help="registry config name (B0, B1) or TINY (tiny_serve3 config)")
    ap.add_argument("--lp", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--family", default="Serve_v1", help="serving program family: Serve_v1 (legacy geometry) or Serve_v4 (declared geometry, --attention)")
    ap.add_argument("--attention", default=None, help='[D57] JSON geometry for Serve_v4, e.g. {"kernel":"fa2-splitkv-align","kBlockM":64,"kBlockN":128,"num_splits":1}')
    ap.add_argument("--digest", action="store_true", help="stream the descriptor and report its sha256 (MEASURED over the full structure)")
    ap.add_argument("--keep-bodies", action="store_true", help="do not release per-T bodies during the streamed encode")
    ap.add_argument("--gates", type=int, default=0, help="locate N uniformly random gates (Circuit.gate + root role) and report per-query time")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expect-digest", default=None)
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    a = ap.parse_args(argv)
    if a.case == "TINY":
        C = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64, "GU": 64, "EPS": 1e-5,
             "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
    else:
        C = dict(getattr(b1, f"{a.case}_CONFIG"))
    out: dict[str, Any] = {"case": a.case, "LP": a.lp, "STEPS": a.steps, "stages": [], "label": "MEASURED"}

    def stage(name: str, t0: float) -> None:
        rec = {"stage": name, "seconds": round(time.perf_counter() - t0, 3), "max_rss_gb": round(_rss_gb(), 3)}
        out["stages"].append(rec)
        print(json.dumps(rec), flush=True)

    t0 = time.perf_counter()
    geom = json.loads(a.attention) if getattr(a, "attention", None) else None
    sp = b1.serve_program(C, a.lp, a.steps, family=getattr(a, "family", None), attention=geom)
    prog = LazyProgram(sp)
    out.update(function=sp.id, gates=prog.gates, input_gates=prog.input_gates, root_nodes=prog.layout.n_nodes)
    stage("LazyProgram (layout)", t0)
    if a.digest:
        t0 = time.perf_counter()
        d = lazy_program_digest(prog, release_bodies=not a.keep_bodies)
        out["program_digest"] = d
        out["program_digest_how"] = ("streamed v1 descriptor over the full structure: every non-root definition built and encoded by "
                                     "verity.ir.codec._encode_definition, every root node generated by verity_vllm.ir.lazy.ServeLayout; sha256 of the stream")
        stage("streamed encode + sha256", t0)
        if a.expect_digest is not None:
            out["digest_matches_expected"] = d == a.expect_digest
            print("digest", d, "expected", a.expect_digest, "EQUAL" if d == a.expect_digest else "DIFFERENT", flush=True)
    if a.gates:
        rng = random.Random(a.seed)
        circ = prog.circuit
        lay = prog.layout
        t0 = time.perf_counter()
        roles: dict[str, int] = {}
        for _ in range(a.gates):
            g = rng.randrange(prog.gates)
            k = lay.node_at(g)
            roles[lay.role(k)[0]] = roles.get(lay.role(k)[0], 0) + 1
            gate = circ.gate(g)
            _ = gate.prim.id, len(gate.operands)
        dt = time.perf_counter() - t0
        out["random_gates"] = {"n": a.gates, "seconds": round(dt, 3), "ms_per_gate": round(1000 * dt / a.gates, 4), "roles": roles, "seed": a.seed}
        stage(f"{a.gates} random Circuit.gate lookups", t0)
    out["max_rss_gb"] = round(_rss_gb(), 3)
    print("DONE", json.dumps({k: v for k, v in out.items() if k != "stages"}), flush=True)
    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    if a.expect_digest is not None and out.get("program_digest") != a.expect_digest:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main())
