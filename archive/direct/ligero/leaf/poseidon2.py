"""``poseidon2``: the Poseidon2-BabyBear sponge leaf (width 24, rate 16, overwrite mode; Plonky3 constants) -- the scheme
statement v5 / ``leaf/v2h`` were defined with, moved here from ``hashchain.py`` / ``poseidon2.py`` as the first
:class:`~archive.direct.ligero.leaf.base.LeafScheme` plugin (lane leaf-iface).  A move, not a change: ``compose(rel)``
with this scheme is byte-identical to the pre-registry ``compose(rel)`` (same rows in the same order with the same names,
same constraints, same ``sys_id``; asserted by ``leaf/conformance_test.py``).

Per chain column and operand: the column's 16-bit lanes are packed from the ``k`` operand words (``LANE_BITS / word_bits``
words per lane, little-endian), the 8 capacity lanes come from the carried rows ``<nm>.cap[x]`` plus the IV
(``IV(role, word_bits, row_words)``: the rows hold ``capacity - IV`` so a chain start's 0 is the IV), one permutation
(213 S-boxes x 4 product rows: x^2, x^3, x^4, x^7; the linear layers are affine, no rows), the digest is the first 8 rate
lanes of the output, the carry is ``output capacity - IV``.

Native (committer): :func:`archive.direct.ligero.poseidon2.hash_rows_np` == ``verity.ml.commitments.poseidon2_babybear.hash_row``
row by row.  Chain witness: :func:`sponge_states` (torch, host or device, all VUs in lockstep).  Device hook:
:class:`Poseidon2Witness` writes the 1704 S-box rows per unit with the state in registers (``LIGERO_DEVICE_SPONGE=0`` keeps
the fused program path).
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from verity.ml.commitments.rowleaf import SCHEMA_ROW, row_leaf_value

from .. import poseidon2 as h2
from ..compile import Expr, LigeroCtx, P, Quadratic
from . import registry

__all__ = ["DEVICE_SPONGE", "LaneFormat", "Poseidon2Leaf", "Poseidon2Witness", "lane_geometry", "permute_torch",
           "sponge_kernel_source", "sponge_lanes", "sponge_states"]

#: The sponges' 1704 S-box rows per unit are written by a dedicated CUDA kernel (:class:`Poseidon2Witness`, a witness
#: hook of witness_device.py) with the 24-lane state in registers, instead of the fused program evaluating the linear
#: layers as ~74k-term affine forms over W (the table-driven kernel, ~0.11 s per 4096 fp8-ada VUs).  Same rows, same
#: values; ``LIGERO_DEVICE_SPONGE=0`` keeps the program path.  CPU / torch witnesses never use hooks.
DEVICE_SPONGE = os.environ.get("LIGERO_DEVICE_SPONGE", "1") != "0"


def _reduce(e):
    if isinstance(e, Expr):
        return Expr(e.ctx, {k: v % P for k, v in e.lin.items()}, e.c % P)
    return e % P


def _prod(ctx: LigeroCtx, a: Expr, b: Expr) -> Expr:
    """A product row without ``LigeroCtx.product``'s interval analysis: the sponge lanes are field elements with no
    integer interval (they enter no linear constraint), and the analysis is quadratic in the number of learned
    facts, which the operand bit rows make large."""
    z = ctx._row("prod", f"prod{len(ctx._prod_memo)}", None, None, (a, b))
    idx = next(iter(z.lin))
    ctx._prod_memo[(a.key(), b.key(), idx)] = idx      # a unique key: only the counter matters
    ctx.sys.program.append(("prod", idx, a, b))
    ctx.sys.quadratic.append(Quadratic(f"hash.prod{idx}", a, b, z))
    return z


def _sbox_gadget(ctx: LigeroCtx, rows: list | None = None):
    """``x -> x^7`` as four product rows; ``rows`` (if given) collects their indices as (x2, x3, x4, x7) per S-box, in
    permutation order -- the order the device kernel (:class:`Poseidon2Witness`) stores them in."""
    def sbox(x: Expr) -> Expr:
        x2 = _prod(ctx, x, x)
        x3 = _prod(ctx, x2, x)
        x4 = _prod(ctx, x2, x2)
        x7 = _prod(ctx, x4, x3)
        if rows is not None:
            rows.append(tuple(next(iter(e.lin)) for e in (x2, x3, x4, x7)))
        return x7
    return sbox


class _Sym:
    """A u64 CUDA expression with an upper bound on its value, for emitting the permutation as straight-line code
    through :func:`poseidon2_babybear.permute_ring` (the same traversal that compiled the rows, so the S-boxes come in
    the same order)."""
    __slots__ = ("em", "var", "bound")

    def __init__(self, em, var: str, bound: int):
        self.em, self.var, self.bound = em, var, bound

    def _new(self, expr: str, bound: int) -> "_Sym":
        if bound >= 2**64:
            raise OverflowError("u64 sponge emitter: bound exceeded")
        v = f"t{len(self.em)}"
        self.em.append(f"    u64 {v} = {expr};")
        return _Sym(self.em, v, bound)

    def __add__(self, o):
        if isinstance(o, _Sym):
            return self._new(f"{self.var} + {o.var}", self.bound + o.bound)
        o = int(o)
        return self._new(f"{self.var} + {o}ULL", self.bound + o)

    __radd__ = __add__

    def __mul__(self, c):
        c = int(c)
        return self._new(f"{self.var} * {c}ULL", self.bound * c)

    __rmul__ = __mul__


def sponge_kernel_source(nb: int, per: int, n_sbox: int, n_lanes: int = h2.RATE) -> str:
    """CUDA source of ``poseidon2_rows``: one thread per column; for each sponge, the ``n_lanes`` used rate lanes packed
    from the word bit rows (``per`` elements of ``nb`` bits per lane, little-endian; lanes ``n_lanes..15`` are 0), the
    capacity from the cap rows plus the IV, then the permutation with every S-box's x^2, x^3, x^4, x^7 stored to its row."""
    em: list[str] = []
    state = [_Sym(em, f"s[{i}]", P - 1) for i in range(h2.WIDTH)]
    count = [0]

    def reduce(x):
        if isinstance(x, _Sym):
            return x if x.bound < P else x._new(f"{x.var} % P", P - 1)
        return int(x) % P

    def sbox(x: _Sym) -> _Sym:
        i = count[0]
        count[0] += 1
        x2 = x._new(f"({x.var} * {x.var}) % P", P - 1)
        x3 = x._new(f"({x2.var} * {x.var}) % P", P - 1)
        x4 = x._new(f"({x2.var} * {x2.var}) % P", P - 1)
        x7 = x._new(f"({x4.var} * {x3.var}) % P", P - 1)
        for q, v in enumerate((x2, x3, x4, x7)):
            em.append(f"    W[(size_t)rows[{4 * i + q}] * l + col] = (unsigned){v.var};")
        return x7

    h2.permute_ring(state, sbox, reduce)
    if count[0] != n_sbox:
        raise ValueError(f"emitted {count[0]} S-boxes, the system has {n_sbox}")
    body = "\n".join(em)
    zero_lanes = f"\n        for (int j = {n_lanes}; j < {h2.RATE}; ++j) s[j] = 0;" if n_lanes < h2.RATE else ""
    return f"""
#define P {P}ULL
typedef unsigned long long u64;

__device__ __forceinline__ void permute_store(u64* s, unsigned* __restrict__ W, int l, int col, const int* __restrict__ rows)
{{
{body}
}}

extern "C" __global__ void __launch_bounds__(256)
poseidon2_rows(unsigned* __restrict__ W, int l, int n_sponges, const int* __restrict__ bitrows,
               const int* __restrict__ caprows, const unsigned* __restrict__ iv, const int* __restrict__ sboxrows,
               int rolerow, const unsigned* __restrict__ ivd)
{{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= l) return;
    // the hash side (hashchain.row_hash_system): IV(role) = IV_x + role (IV_w - IV_x), role the unit's public pin row
    const u64 role = rolerow >= 0 ? (u64)W[(size_t)rolerow * l + col] : 0ULL;
    for (int sp = 0; sp < n_sponges; ++sp) {{
        u64 s[{h2.WIDTH}];
        const int* br = bitrows + (size_t)sp * {n_lanes * per * nb};
        for (int j = 0; j < {n_lanes}; ++j) {{
            u64 v = 0;
            for (int q = 0; q < {per}; ++q)
                for (int b = 0; b < {nb}; ++b)
                    v |= (u64)W[(size_t)br[(j * {per} + q) * {nb} + b] * l + col] << ({nb} * q + b);
            s[j] = v;
        }}{zero_lanes}
        for (int x = 0; x < {h2.CAPACITY}; ++x)
            s[{h2.RATE} + x] = ((u64)W[(size_t)caprows[sp * {h2.CAPACITY} + x] * l + col] + iv[sp * {h2.CAPACITY} + x]
                                + (role * ivd[sp * {h2.CAPACITY} + x]) % P) % P;
        permute_store(s, W, l, col, sboxrows + (size_t)sp * {4 * n_sbox});
    }}
}}
"""


def row_sponge_kernel_source() -> str:
    """CUDA source of ``row_sponge``: one thread per committed row, the sponge along its ``steps`` rate blocks with the
    state in registers (the permutation as :func:`poseidon2_babybear.permute_ring`'s straight-line code, as
    :func:`sponge_kernel_source`), writing the linked rows ``cap - IV`` before every block and the digest after the last:
    :func:`sponge_lanes` / ``hash_rows_np`` for all rows at once."""
    em: list[str] = []
    state = [_Sym(em, f"s[{i}]", P - 1) for i in range(h2.WIDTH)]

    def reduce(x):
        if isinstance(x, _Sym):
            return x if x.bound < P else x._new(f"{x.var} % P", P - 1)
        return int(x) % P

    def sbox(x: _Sym) -> _Sym:
        x2 = x._new(f"({x.var} * {x.var}) % P", P - 1)
        x3 = x._new(f"({x2.var} * {x.var}) % P", P - 1)
        x4 = x._new(f"({x2.var} * {x2.var}) % P", P - 1)
        return x._new(f"({x4.var} * {x3.var}) % P", P - 1)

    out = h2.permute_ring(state, sbox, reduce)
    em += [f"    s[{i}] = {o.var};" for i, o in enumerate(out)]
    body = "\n".join(em)
    return f"""
#define P {P}ULL
typedef unsigned long long u64;

__device__ __forceinline__ void permute(u64* s)
{{
{body}
}}

extern "C" __global__ void __launch_bounds__(32)
row_sponge(const long long* __restrict__ el, int n, int steps, const unsigned* __restrict__ iv, long long* __restrict__ cap,
           long long* __restrict__ dg)
{{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    u64 s[{h2.WIDTH}];
#pragma unroll
    for (int x = 0; x < {h2.CAPACITY}; ++x) s[{h2.RATE} + x] = iv[x];
    for (int st = 0; st < steps; ++st) {{
        const size_t base = (size_t)r * steps + st;
#pragma unroll
        for (int x = 0; x < {h2.CAPACITY}; ++x) cap[base * {h2.CAPACITY} + x] = (long long)((s[{h2.RATE} + x] + P - iv[x]) % P);
#pragma unroll
        for (int j = 0; j < {h2.RATE}; ++j) s[j] = (u64)el[base * {h2.RATE} + j] % P;
        permute(s);
    }}
#pragma unroll
    for (int x = 0; x < {h2.OUT}; ++x) dg[(size_t)r * {h2.OUT} + x] = (long long)s[x];
}}
"""


_ROW_SPONGE: dict = {}


def row_sponge_kernel():
    """The compiled ``row_sponge`` of the current device (cupy RawModule, once per process and device), or ``None`` without
    cupy.  Compiling is a function of the Poseidon2 constants alone: :meth:`Poseidon2Leaf.prepare` does it before any clock."""
    try:
        import cupy as cp
    except ImportError:
        return None
    dev = cp.cuda.Device().id
    fn = _ROW_SPONGE.get(dev)
    if fn is None:
        fn = _ROW_SPONGE[dev] = cp.RawModule(code=row_sponge_kernel_source(), options=("-std=c++17",)).get_function("row_sponge")
    return fn


def row_sponges_lanes(el, iv: list[int], n: int, steps: int):
    """``el`` (n * steps, 16) int64 CUDA torch rate lanes (row-major: row r's block s at r * steps + s) from the capacity
    ``iv`` -> ``(cap (n, steps, 8): cap - IV before every block, digests (n, 8))`` int64, by :func:`row_sponge_kernel`;
    ``None`` when it is unavailable.  Equal to :func:`sponge_lanes` (``cap.T.reshape(n, steps, 8)``, digests)."""
    import torch
    fn = row_sponge_kernel()
    if fn is None or not el.is_cuda:
        return None
    import cupy as cp
    el = el.to(torch.int64).contiguous()
    cap = torch.empty((n, steps, h2.CAPACITY), dtype=torch.int64, device=el.device)
    dg = torch.empty((n, h2.OUT), dtype=torch.int64, device=el.device)
    if n:
        with cp.cuda.ExternalStream(torch.cuda.current_stream(el.device).cuda_stream):
            ivc = cp.asarray(np.asarray([int(v) for v in iv[h2.RATE:]], dtype=np.uint32))
            fn((-(-n // 32),), (32,), (cp.asarray(el), np.int32(n), np.int32(steps), ivc, cp.asarray(cap), cp.asarray(dg)))
    return cap, dg


class Poseidon2Witness:
    """The witness hook (witness_device.py ``sys.witness_hooks``) writing the sponges' S-box rows on the device.
    ``sponges``: per sponge ``(bit_rows word-major, cap_rows, iv_capacity, sbox_rows [(x2, x3, x4, x7), ...])``.
    ``role_row`` / ``ivd`` (the hash side of row sharing, ``hashchain.row_hash_system``): the capacity IV is
    ``iv + role * ivd`` with ``role`` the value of that row in the column.  ``n_lanes``: the rate lanes the bit rows fill
    (a :class:`LaneFormat` with fewer elements than a rate block leaves the rest 0)."""

    def __init__(self, nb: int, per: int, sponges: list, role_row: int = -1, ivd: list | None = None, n_lanes: int = h2.RATE):
        self.nb, self.per, self.sponges, self.n_lanes = nb, per, sponges, n_lanes
        self.role_row = role_row
        self.ivd = list(ivd) if ivd is not None else [0] * h2.CAPACITY
        self.n_sbox = len(sponges[0][3])
        if any(len(s[0]) != n_lanes * per * nb for s in sponges):
            raise ValueError(f"sponge bit rows are not {n_lanes} lanes x {per} x {nb} bits")
        if any(len(s[3]) != self.n_sbox for s in sponges):
            raise ValueError("sponges with different S-box counts")
        self.rows = frozenset(r for s in sponges for quad in s[3] for r in quad)
        self._dev: dict = {}

    def _arrays(self, cp):
        dev = cp.cuda.Device().id
        hit = self._dev.get(dev)
        if hit is None:
            src = sponge_kernel_source(self.nb, self.per, self.n_sbox, self.n_lanes)
            mod = cp.RawModule(code=src, options=("-std=c++17",))
            hit = (mod.get_function("poseidon2_rows"),
                   cp.asarray(np.array([r for s in self.sponges for r in s[0]], dtype=np.int32)),
                   cp.asarray(np.array([r for s in self.sponges for r in s[1]], dtype=np.int32)),
                   cp.asarray(np.array([v for s in self.sponges for v in s[2]], dtype=np.uint32)),
                   cp.asarray(np.array([r for s in self.sponges for quad in s[3] for r in quad], dtype=np.int32)),
                   cp.asarray(np.array(self.ivd * len(self.sponges), dtype=np.uint32)))
            self._dev[dev] = hit
        return hit

    def launch(self, cp, W_u32, l: int) -> None:
        kernel, bitrows, caprows, iv, sboxrows, ivd = self._arrays(cp)
        kernel((-(-l // 256),), (256,), (W_u32, np.int32(l), np.int32(len(self.sponges)), bitrows, caprows, iv, sboxrows,
                                         np.int32(self.role_row), ivd))


# ---------------------------------------------------------------------------------------------------------------------
# Witness side (torch): the permutation on whole states, the sponge along the chain
# ---------------------------------------------------------------------------------------------------------------------

def _sbox_t(x):
    x2 = (x * x) % P
    x4 = (x2 * x2) % P
    x3 = (x2 * x) % P
    return (x4 * x3) % P


def _external_matrix() -> list[list[int]]:
    """circ(2 M4, M4, ..., M4) as a 24 x 24 integer matrix (column i = the layer on unit vector i; entries <= 6)."""
    cols = []
    for i in range(h2.WIDTH):
        e = [0] * h2.WIDTH
        e[i] = 1
        cols.append(h2._external_layer(e, lambda v: v))
    return [[cols[j][i] for j in range(h2.WIDTH)] for i in range(h2.WIDTH)]


_MEXT = _external_matrix()
_TORCH_CONST: dict = {}


def _torch_consts(device):
    """(M_ext as float64 (24, 24), DIAG (24, 1) int64, initial/final round constants (4, 24) int64, internal (21,))."""
    import torch
    key = str(device)
    hit = _TORCH_CONST.get(key)
    if hit is None:
        hit = (torch.tensor(_MEXT, dtype=torch.float64, device=device),
               torch.tensor(h2.DIAG, dtype=torch.int64, device=device)[:, None],
               torch.tensor(h2.RC_EXTERNAL_INITIAL, dtype=torch.int64, device=device),
               torch.tensor(h2.RC_EXTERNAL_FINAL, dtype=torch.int64, device=device),
               torch.tensor(h2.RC_INTERNAL, dtype=torch.int64, device=device))
        _TORCH_CONST[key] = hit
    return hit


def permute_torch(st):
    """(24, N) int64 states in [0, p) -> (24, N): ``poseidon2_babybear.permute_np`` on a torch tensor, whole-state ops
    (the external layer as one exact float64 matmul: entries <= 6, residues < 2^31, 24 terms < 2^39 < 2^53)."""
    import torch
    mext, diag, rc_i, rc_f, rc_int = _torch_consts(st.device)

    def ext(s):
        return torch.round(mext @ s.to(torch.float64)).to(torch.int64) % P

    def internal(s):
        return (s * diag + s.sum(0, keepdim=True)) % P

    s = ext(st % P)
    for r in range(h2.HALF_FULL_ROUNDS):
        s = ext(_sbox_t((s + rc_i[r][:, None]) % P))
    for r in range(h2.PARTIAL_ROUNDS):
        s = s.clone()
        s[0] = _sbox_t((s[0] + rc_int[r]) % P)
        s = internal(s)
    for r in range(h2.HALF_FULL_ROUNDS):
        s = ext(_sbox_t((s + rc_f[r][:, None]) % P))
    return s


def sponge_states(words, word_bits: int, role: int, n_words: int, n_vus: int, steps: int, l: int):
    """``words`` (l, k) int64 torch tensor (pad columns 0) -> (cap_in (8, l): the linked rows ``cap - IV`` per column,
    digests (n_vus, 8)) by running the sponge of every VU along its ``steps`` columns, all VUs in lockstep."""
    import torch
    el = torch.as_tensor(h2.pack_words_np(words.cpu().numpy(), word_bits), dtype=torch.int64, device=words.device)   # (l, 16)
    return sponge_lanes(el, h2.IV(role, word_bits, n_words), n_vus, steps, l)


def sponge_lanes(el, iv: list[int], n_vus: int, steps: int, l: int):
    """:func:`sponge_states` on the rate lanes ``el`` (l, 16) int64 torch (pad columns 0) from the capacity ``iv``."""
    import torch
    dev = el.device
    cap_in = torch.zeros((h2.CAPACITY, l), dtype=torch.int64, device=dev)
    st = torch.zeros((h2.WIDTH, n_vus), dtype=torch.int64, device=dev)
    for x in range(h2.CAPACITY):
        st[h2.RATE + x] = iv[h2.RATE + x]
    cols0 = torch.arange(n_vus, device=dev) * steps
    for s in range(steps):
        cols = cols0 + s
        for x in range(h2.CAPACITY):
            cap_in[x, cols] = (st[h2.RATE + x] - iv[h2.RATE + x]) % P
        st = torch.cat([el[cols].T, st[h2.RATE:]], 0)
        st = permute_torch(st)
    return cap_in, st[:h2.OUT].T.contiguous()


# ---------------------------------------------------------------------------------------------------------------------
# The scheme
# ---------------------------------------------------------------------------------------------------------------------

def _bit_rows(word: Expr, nb: int) -> list[int]:
    """The committed bit rows behind an operand word expression ``sum_j b_j 2^j``, in bit order."""
    rows = sorted(word.lin.items(), key=lambda kv: kv[1])
    if word.c != 0 or len(rows) != nb or [c for _, c in rows] != [2**j for j in range(nb)]:
        raise ValueError("poseidon2 gadget: operand words must be the plain little-endian sums of their committed bit rows")
    return [r for r, _ in rows]


class LaneFormat(Protocol):
    """An operand row format whose committed words are NOT the sponge's lane elements (``fp4/hashed.py`` ``FP4Format``:
    64 E2M1 codes + 4 UE4M3 scale bytes per column -> 72 nibbles).  Per column, the ``k`` words map to ``n_elems``
    elements of ``elem_bits`` bits, packed ``lane_bits // elem_bits`` per rate lane (little-endian); the lanes past
    ``n_elems / per`` are 0.  The IV is ``IV(role, elem_bits, steps * n_elems)``: a domain of its own."""

    k: int
    n_elems: int
    elem_bits: int
    lane_bits: int

    def elements_np(self, words) -> np.ndarray:
        """(N, k) words -> (N, n_elems) elements (numpy, int64)."""
        ...

    def elements_torch(self, words):
        """(l, k) int64 torch -> (l, n_elems) int64 torch on the same device."""
        ...


def lane_geometry(lanes: LaneFormat) -> tuple[int, int]:
    """``(per, n_lanes)`` of a lane format: elements per lane and rate lanes used (checked: whole lanes, at most one rate
    block, a lane below 2^30 < p so the packing is injective)."""
    per = lanes.lane_bits // lanes.elem_bits
    n_lanes = lanes.n_elems // per if per else 0
    if lanes.lane_bits % lanes.elem_bits or lanes.n_elems % per or not 0 < n_lanes <= h2.RATE or lanes.lane_bits > 30:
        raise ValueError(f"lane format: {lanes.n_elems} elements of {lanes.elem_bits} bits on {lanes.lane_bits}-bit lanes "
                         f"are not whole lanes of one rate block")
    return per, n_lanes


def _pack_elements_np(el: np.ndarray, lanes: LaneFormat) -> np.ndarray:
    """(N, n_elems) elements -> (N, 16) rate lanes (``per`` little-endian elements each, unused lanes 0)."""
    per, n_lanes = lane_geometry(lanes)
    el = np.asarray(el, dtype=np.int64)
    if ((el < 0) | (el >= 2**lanes.elem_bits)).any():
        raise ValueError("lane element out of range")
    g = el.reshape(el.shape[0], n_lanes, per)
    out = np.zeros((el.shape[0], h2.RATE), dtype=np.int64)
    out[:, :n_lanes] = (g * 2 ** (lanes.elem_bits * np.arange(per, dtype=np.int64))).sum(-1)
    return out


def _arange_like(n: int, like):
    import torch
    return torch.arange(n, dtype=torch.int64, device=like.device)


def _pack_elements_torch(el, lanes: LaneFormat):
    import torch
    per, n_lanes = lane_geometry(lanes)
    g = el.to(torch.int64).reshape(el.shape[0], n_lanes, per)
    out = torch.zeros((el.shape[0], h2.RATE), dtype=torch.int64, device=el.device)
    out[:, :n_lanes] = (g * 2 ** (lanes.elem_bits * torch.arange(per, dtype=torch.int64, device=el.device))).sum(-1)
    return out


class Poseidon2Leaf:
    """``lanes``: ``None`` (the operand words ARE the lane elements, ``LANE_BITS / word_bits`` per 16-bit lane, one rate
    block per column) or a :class:`LaneFormat` (:meth:`with_lanes`).  Under a format, :meth:`gadget` takes the column's
    ``n_elems`` element expressions and ``row_words`` counts elements; :meth:`native` / :meth:`chain_witness` take the
    committed words and map them through the format."""
    name = "poseidon2"
    assumption = "Poseidon2-BabyBear (width 24, rate 16, Plonky3 constants) collision resistance: algebraic hash, 2^-124 birthday on the 8-lane digest"
    digest_elems = h2.OUT
    carry_elems = h2.CAPACITY
    schema = SCHEMA_ROW
    params_sha256 = h2.CONSTANTS_SHA256
    hash_name = "poseidon2-babybear-w24"
    privacy_note = "deterministic, unsalted: hides only up to preimage search on a row"

    def __init__(self, lanes: LaneFormat | None = None):
        self._sponges: list = []       # per gadget call: (bit_rows word-major, cap rows, iv capacity, sbox rows)
        self._nb: int | None = None
        self._per: int | None = None
        self._n_lanes = h2.RATE
        self.lanes = lanes
        if lanes is not None:
            lane_geometry(lanes)

    def with_lanes(self, lanes: LaneFormat) -> "Poseidon2Leaf":
        """A fresh instance of this scheme over the operand format ``lanes``."""
        s = type(self)(lanes)
        s.test_only = getattr(self, "test_only", False)   # type: ignore[attr-defined]
        return s

    def _row_elems(self, row_words: int) -> int:
        """Elements of a row of ``row_words`` committed words under the lane format."""
        k = self.lanes.k
        if row_words % k:
            raise ValueError(f"poseidon2 lanes: a row of {row_words} words is not whole columns of {k}")
        return row_words // k * self.lanes.n_elems

    # --- native ------------------------------------------------------------------------------------------------------
    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        if self.lanes is None:
            return h2.hash_rows_np(np.asarray(rows, dtype=np.int64), word_bits, role)
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 2:
            raise ValueError("rows must be (N, n_words)")
        n, k = rows.shape[0], self.lanes.k
        steps = self._row_elems(rows.shape[1]) // self.lanes.n_elems
        el = _pack_elements_np(self.lanes.elements_np(rows.reshape(n * steps, k)), self.lanes).reshape(n, steps, h2.RATE)
        iv = h2.IV(role, self.lanes.elem_bits, steps * self.lanes.n_elems)
        st = np.zeros((h2.WIDTH, n), dtype=np.int64)
        for x in range(h2.CAPACITY):
            st[h2.RATE + x] = iv[h2.RATE + x]
        for s in range(steps):
            st[:h2.RATE] = el[:, s, :].T
            st = h2.permute_np(st)
        return st[:h2.OUT].T.copy()

    def leaf_bytes(self, digest_row) -> bytes:
        return row_leaf_value(digest_row)

    def leaf_bytes_many(self, digests: np.ndarray) -> list[bytes]:
        """:meth:`leaf_bytes` of every row of ``digests`` (count, 8) in [0, p)."""
        d = np.asarray(digests, dtype=np.int64)
        if d.ndim != 2 or d.shape[1] != h2.OUT or ((d < 0) | (d >= h2.P)).any():
            raise ValueError("a row digest is 8 BabyBear lanes")
        b = d.astype(">u4").tobytes()
        return [b[32 * i:32 * i + 32] for i in range(d.shape[0])]

    # --- gadget --------------------------------------------------------------------------------------------------------
    def gadget(self, ctx: LigeroCtx, words: Sequence[Expr], *, word_bits: int, role: int, row_words: int,
               carry_in: Sequence[Expr], first: bool = False, last: bool = True) -> tuple[list[Expr], list[Expr]]:
        nb = word_bits
        if self.lanes is None:
            per, n_lanes = h2.LANE_BITS // nb, h2.RATE
            if h2.LANE_BITS % nb or len(words) != h2.RATE * per:
                raise ValueError(f"poseidon2: k = {len(words)} words of {nb} bits are not one rate block of {h2.RATE} lanes")
        else:
            per, n_lanes = lane_geometry(self.lanes)
            if nb != self.lanes.elem_bits or len(words) != self.lanes.n_elems:
                raise ValueError(f"poseidon2 lanes: the gadget takes {self.lanes.n_elems} elements of {self.lanes.elem_bits} bits "
                                 f"per column, got {len(words)} of {nb}")
        if len(carry_in) != h2.CAPACITY:
            raise ValueError("poseidon2: the carried state is the 8 capacity lanes")
        if self._nb not in (None, nb):
            raise ValueError("poseidon2: one word width per compose")
        self._nb, self._per, self._n_lanes = nb, per, n_lanes
        iv = h2.IV(role, nb, row_words)
        lanes = []
        for j in range(n_lanes):
            e = Expr(ctx)
            for q in range(per):
                e = e + words[per * j + q] * 2 ** (nb * q)
            lanes.append(_reduce(e))
        lanes += [Expr(ctx) for _ in range(h2.RATE - n_lanes)]
        state = lanes + [_reduce(c + iv[h2.RATE + x]) for x, c in enumerate(carry_in)]
        sbox_rows: list[tuple] = []
        out = h2.permute_ring(state, _sbox_gadget(ctx, sbox_rows), _reduce)
        bit_rows = [r for w in words for r in _bit_rows(w, nb)]
        self._sponges.append((bit_rows, [next(iter(c.lin)) for c in carry_in], list(iv[h2.RATE:]), sbox_rows))
        carry_out = [_reduce(out[h2.RATE + x] - iv[h2.RATE + x]) for x in range(h2.CAPACITY)]
        return list(out[:h2.OUT]), carry_out

    def witness(self):
        if not DEVICE_SPONGE or not self._sponges:
            return None
        return Poseidon2Witness(self._nb, self._per, self._sponges, n_lanes=self._n_lanes)

    # --- chain witness -------------------------------------------------------------------------------------------------
    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        if self.lanes is None:
            return sponge_states(words, word_bits, role, row_words, n_vus, steps, l)
        el = _pack_elements_torch(self.lanes.elements_torch(words), self.lanes)
        return sponge_lanes(el, h2.IV(role, self.lanes.elem_bits, self._row_elems(row_words)), n_vus, steps, l)

    def row_sponges(self, words, *, word_bits: int, role: int, row_words: int, n: int, steps: int):
        """The committer's sponges of ``n`` whole rows at once: ``words`` (n * steps, k) int64 CUDA torch, row-major ->
        ``(cap (n, steps, 8), digests (n, 8))`` = :meth:`chain_witness` reshaped per row, whose digests are :meth:`native`'s
        (the tree leaves); ``None`` off CUDA or without cupy (the caller falls back to :meth:`chain_witness`)."""
        if not words.is_cuda:
            return None
        if self.lanes is None:
            per = h2.LANE_BITS // word_bits
            if bool(((words < 0) | (words >= 2**word_bits)).any()):
                raise ValueError("word out of range")
            w = words.reshape(words.shape[0], h2.RATE, per)
            el = (w * (2 ** (word_bits * _arange_like(per, words)))[None, None, :]).sum(2)
            iv = h2.IV(role, word_bits, row_words)
        else:
            e = self.lanes.elements_torch(words)
            if bool(((e < 0) | (e >= 2**self.lanes.elem_bits)).any()):
                raise ValueError("lane element out of range")
            el = _pack_elements_torch(e, self.lanes)
            iv = h2.IV(role, self.lanes.elem_bits, self._row_elems(row_words))
        return row_sponges_lanes(el, iv, n, steps)

    def prepare(self) -> None:
        """Compile the committer's device kernel (a function of the constants only; outside every clock)."""
        row_sponge_kernel()

    def census(self) -> dict:
        out = {"permutations_per_unit": 2,
               "sbox_rows_per_permutation": 4 * (h2.WIDTH * 2 * h2.HALF_FULL_ROUNDS + h2.PARTIAL_ROUNDS)}
        if self.lanes is not None:
            out.update({"lane_bits": self.lanes.lane_bits, "rate_lanes_used": lane_geometry(self.lanes)[1],
                        "elements_per_unit": self.lanes.n_elems})
        return out


registry.register(Poseidon2Leaf)
