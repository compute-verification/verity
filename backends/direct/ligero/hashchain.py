"""``authentication = included-hash``: the operand rows hashed INSIDE the B-Ligero relation with Poseidon2 (lane hash-relation).

``compose(rel)`` takes a registered relation's chain-compiled system (``relations.py``: bf16-hopper, fp8-hopper, fp8-ada;
their operand words enter as *pinned* decodes ``a[i].s / .m / .e``, public per column) and turns it into the private,
self-hashing relation of the same GEMM step -- without editing the unit compilers (``unit.py``, ``fp8/relation.py``):

1. **Private operands.**  Every operand pin becomes a hint row; the word behind it is ``word_bits`` boolean hint rows
   ``hash.a[i].b<j>`` and the decode is re-established by constraints on those rows: ``s = b_msb``, ``m = f + nz 2^frac``,
   ``e = t - nz - (bias - 1)`` with ``nz = [t != 0]`` (``t * inv = nz``, ``(1 - nz) t = 0``, ``nz`` boolean) and the
   finiteness the verifier used to check on the public word (``t != 2^exp - 1`` for BF16; ``(t, f) != (15, 7)`` for
   E4M3) as one ``inverse_nonzero``.  The ranges the pins declared (``Row.lo / hi``) are now theorems of these
   constraints, so every downstream audit of the unit compiler stands.  18 rows per BF16 word, 13 per E4M3 byte.
2. **Two sponges along the chain** (``poseidon2.py``: width 24, rate 16, overwrite mode).  Unit ``i`` of a VU absorbs its
   16 lanes of ``a`` (the x row's words ``[16 i, 16 i + 16)``) into sponge X and its lanes of ``b`` (the W column) into
   sponge W, one permutation each; the 8 capacity lanes of each sponge are *linked* rows (``sys.chain["c"] / ["y"]``:
   ``cap_in[j + 1] = cap_out[j]``, 0 at a chain start; the rows hold ``cap - IV`` so the start value is 0) -- 16 more
   linked rows than the accumulator's three (``chain.py``: monomial coefficients for the extras).  852 product rows per
   permutation (213 S-boxes x 4 rows: x^2, x^3, x^4, x^7), the linear layers as affine expressions (no rows).
3. **The digests at the chain end.**  Public pin ``hash.is_end`` (1 at the last unit of a VU, 0 elsewhere -- the verifier
   knows it from the layout) and 16 public pins ``hash.d[0..15]`` (the x digest, then the W digest; 0 off the ends), tied
   by ``is_end * (state_out[x] - d[x]) = 0``: at an end column the sponge's first 8 rate lanes equal the published digest;
   elsewhere nothing is revealed (the ZK rows hide the states).  The relation's own public word (``y16``) stays a chain
   end as before.

The statement (``serialize.py`` v5) then carries, instead of 27 MB of operand words, the three roots, each VU's leaf
indices, the published digests and SHA-256 Merkle multiproofs from the Poseidon2 leaves to the roots
(``leaf/v2h``: ``verity.commitments.leaf_v2h``); the verifier never learns an operand word.  Binding: Poseidon2 collision
resistance (2^-124 birthday on 8 lanes) + SHA-256 above; privacy: the Ligero ZK rows + the digest of a 1536-word row (a
low-entropy row would need a salt: not done, see the lane note).
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from . import poseidon2 as h2
from .compile import Expr, LigeroCtx, Linear, P, Quadratic, System
from .relations import Relation

__all__ = ["HASH_PIN_END", "N_DIGEST", "HashedRelation", "compose", "hash_pin", "hash_public_vectors", "sponge_states"]

HASH_PIN_END = "hash.is_end"
N_DIGEST = 2 * h2.OUT          # x digest (8) then W digest (8)
ROLES = ((0, "a", h2.ROLE_X), (1, "b", h2.ROLE_W))


def hash_pin(x: int) -> str:
    return f"hash.d[{x}]"


@dataclass
class HashedRelation:
    rel: Relation
    sys: System
    word_bits: int
    k: int                      # operand words per unit
    n_base_hints: int           # hint rows the relation's own generator supplies (a prefix of sys.hints)
    extra_hint_names: list      # the appended hint rows, in sys.hints order
    census: dict
    block_layout: bool | None = None    # memo of _block_layout (hash_hints)
    fmt: object = None          # the operand format (``_FloatFormat`` / ``fp4.hashed.FP4Format``): decode + lane packing

    @property
    def steps(self) -> int:
        return self.rel.steps

    def iv(self, role: int) -> list[int]:
        """The sponge IV of operand ``role``: (role, the format's word width, the row's element count)."""
        return h2.IV(role, self.fmt.iv_word_bits, self.steps * self.fmt.n_elems)

    def native_digests(self, rows, role: int) -> np.ndarray:
        """The committer's row digests (N, 8) of ``rows`` (N, steps * k) words -- ``poseidon2.hash_rows_np`` for the
        16-bit-lane formats (byte-identical to the leaf reference), the format's lane packing otherwise."""
        rows = np.asarray(rows, dtype=np.int64)
        if self.fmt.lane_bits == h2.LANE_BITS and self.fmt.n_lanes == h2.RATE:
            return h2.hash_rows_np(rows, self.word_bits, role)
        n, steps = rows.shape[0], self.steps
        lanes = self.fmt.lanes_np(rows.reshape(n * steps, self.k)).reshape(n, steps, h2.RATE)
        iv = self.iv(role)
        st = np.zeros((h2.WIDTH, n), dtype=np.int64)
        for x in range(h2.CAPACITY):
            st[h2.RATE + x] = iv[h2.RATE + x]
        for s in range(steps):
            st[:h2.RATE] = lanes[:, s, :].T
            st = h2.permute_np(st)
        return st[:h2.OUT].T.copy()


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


#: The sponges' 1704 S-box rows per unit are written by a dedicated CUDA kernel (:class:`Poseidon2Witness`, a witness
#: hook of witness_device.py) with the 24-lane state in registers, instead of the fused program evaluating the linear
#: layers as ~74k-term affine forms over W (the table-driven kernel, ~0.11 s per 4096 fp8-ada VUs).  Same rows, same
#: values; ``LIGERO_DEVICE_SPONGE=0`` keeps the program path.  CPU / torch witnesses never use hooks.
DEVICE_SPONGE = os.environ.get("LIGERO_DEVICE_SPONGE", "1") != "0"


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
    return f"""
#define P {P}ULL
typedef unsigned long long u64;

__device__ __forceinline__ void permute_store(u64* s, unsigned* __restrict__ W, int l, int col, const int* __restrict__ rows)
{{
{body}
}}

extern "C" __global__ void __launch_bounds__(256)
poseidon2_rows(unsigned* __restrict__ W, int l, int n_sponges, const int* __restrict__ bitrows,
               const int* __restrict__ caprows, const unsigned* __restrict__ iv, const int* __restrict__ sboxrows)
{{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= l) return;
    for (int sp = 0; sp < n_sponges; ++sp) {{
        u64 s[{h2.WIDTH}];
        const int* br = bitrows + (size_t)sp * {n_lanes * per * nb};
        for (int j = 0; j < {n_lanes}; ++j) {{
            u64 v = 0;
            for (int q = 0; q < {per}; ++q)
                for (int b = 0; b < {nb}; ++b)
                    v |= (u64)W[(size_t)br[(j * {per} + q) * {nb} + b] * l + col] << ({nb} * q + b);
            s[j] = v;
        }}
        for (int j = {n_lanes}; j < {h2.RATE}; ++j) s[j] = 0;
        for (int x = 0; x < {h2.CAPACITY}; ++x)
            s[{h2.RATE} + x] = ((u64)W[(size_t)caprows[sp * {h2.CAPACITY} + x] * l + col] + iv[sp * {h2.CAPACITY} + x]) % P;
        permute_store(s, W, l, col, sboxrows + (size_t)sp * {4 * n_sbox});
    }}
}}
"""


class Poseidon2Witness:
    """The witness hook (witness_device.py ``sys.witness_hooks``) writing the sponges' S-box rows on the device.
    ``sponges``: per sponge ``(bit_rows word-major, cap_rows, iv_capacity, sbox_rows [(x2, x3, x4, x7), ...])``."""

    def __init__(self, nb: int, per: int, sponges: list, n_lanes: int = h2.RATE):
        self.nb, self.per, self.sponges, self.n_lanes = nb, per, sponges, n_lanes
        self.n_sbox = len(sponges[0][3])
        if any(len(s[0]) != n_lanes * per * nb for s in sponges):
            raise ValueError(f"sponge bit rows are not {n_lanes} lanes x {per} x {nb} bits")
        if any(len(s[3]) != self.n_sbox for s in sponges):
            raise ValueError("sponges with different S-box counts")
        self.rows = frozenset(r for s in sponges for quad in s[3] for r in quad)
        self._dev: dict = {}

    def _arrays(self, cp):
        import numpy as np
        dev = cp.cuda.Device().id
        hit = self._dev.get(dev)
        if hit is None:
            src = sponge_kernel_source(self.nb, self.per, self.n_sbox, self.n_lanes)
            mod = cp.RawModule(code=src, options=("-std=c++17",))
            hit = (mod.get_function("poseidon2_rows"),
                   cp.asarray(np.array([r for s in self.sponges for r in s[0]], dtype=np.int32)),
                   cp.asarray(np.array([r for s in self.sponges for r in s[1]], dtype=np.int32)),
                   cp.asarray(np.array([v for s in self.sponges for v in s[2]], dtype=np.uint32)),
                   cp.asarray(np.array([r for s in self.sponges for quad in s[3] for r in quad], dtype=np.int32)))
            self._dev[dev] = hit
        return hit

    def launch(self, cp, W_u32, l: int) -> None:
        import numpy as np
        kernel, bitrows, caprows, iv, sboxrows = self._arrays(cp)
        kernel((-(-l // 256),), (256,), (W_u32, np.int32(l), np.int32(len(self.sponges)), bitrows, caprows, iv, sboxrows))


def _format(rel: Relation) -> dict:
    p = rel.params
    if rel.word_dtype == "<u2":            # BF16: s | t(8) | f(7)
        return {"bits": 16, "frac": p.frac, "exp": p.exp, "bias": p.bias, "kind": "bf16", "sig": p.sig}
    if rel.word_dtype == "<u1":            # E4M3: s | t(4) | f(3)
        return {"bits": 8, "frac": p.op_frac, "exp": p.op_exp, "bias": p.op_bias, "kind": "e4m3", "sig": p.sig}
    raise ValueError(f"{rel.name}: no private decode for operand dtype {rel.word_dtype}")


class _FloatFormat:
    """The operand format ``compose`` was written for: BF16 words (18 rows each) / E4M3 bytes (13 rows each) whose pins are
    the decode triples ``<nm>[i].s / .m / .e``, packed ``LANE_BITS // word_bits`` per 16-bit lane, one rate block per unit.
    The seam another format fills (``fp4/hashed.py``: E2M1 nibbles + UE4M3 scale bytes on 24-bit lanes): ``pin_names``,
    ``pin_order``, ``decode`` (one operand: the bit rows, the lane elements, the re-derived pins), ``finish`` (constraints
    joining the two operands), ``lanes_np`` / ``lanes_torch`` / ``bit_rows_torch`` (the witness side)."""

    def __init__(self, rel: Relation):
        f = _format(rel)
        self.kind, self.nb, self.frac, self.exp, self.bias = f["kind"], f["bits"], f["frac"], f["exp"], f["bias"]
        self.k = rel.k
        self.elem_bits = self.nb                  # bits per packed element (a word)
        self.n_elems = self.k                     # elements per operand per unit
        self.lane_bits = h2.LANE_BITS
        self.per = h2.LANE_BITS // self.nb        # elements per lane
        self.n_lanes = h2.RATE                    # rate lanes used
        self.iv_word_bits = self.nb
        self.bit_layout = [self.nb] * self.k      # hint bit rows per word, word-major
        if self.k != self.n_lanes * self.per:
            raise ValueError(f"{rel.name}: k = {self.k} words of {self.nb} bits are not one rate block of 16 lanes")

    def pin_names(self) -> set:
        return {f"{nm}[{i}].{c}" for nm in ("a", "b") for i in range(self.k) for c in ("s", "m", "e")}

    def pin_order(self, want: set) -> list:
        return sorted(want, key=lambda n: (n[0], int(n.split("[")[1].split("]")[0]), n[-1]))

    def decode(self, ctx: LigeroCtx, sys: System, pins: dict, nm: str) -> tuple[list, list, int, object]:
        """Operand ``nm``: (lane elements, bit row indices word-major, rows added, aux for ``finish``)."""
        nb, frac, exp_bits, bias = self.nb, self.frac, self.exp, self.bias
        one = Expr(ctx, {}, 1)
        E = lambda idx: Expr(ctx, {idx: 1}, 0)  # noqa: E731
        words, bit_rows, rows = [], [], 0
        for i in range(self.k):
            tag = f"{nm}[{i}]"
            bits = [ctx.hint(f"{tag}.b{j}") for j in range(nb)]
            bit_rows += [next(iter(b.lin)) for b in bits]
            for j, b in enumerate(bits):
                ctx.boolean(f"{tag}.bool{j}", b)
            words.append(_reduce(sum((b * 2**j for j, b in enumerate(bits)), Expr(ctx))))
            s = bits[nb - 1]
            t = sum((bits[frac + j] * 2**j for j in range(exp_bits)), Expr(ctx))
            f = sum((bits[j] * 2**j for j in range(frac)), Expr(ctx))
            # nz = [t != 0]:  t * inv = nz, (1 - nz) t = 0, nz boolean
            inv = ctx._row("inv", f"{tag}.tinv", spec=(t,))
            sys.program.append(("inv", next(iter(inv.lin)), t))
            nz = ctx.product(t, inv)
            ctx.boolean(f"{tag}.nz", nz)
            sys.quadratic.append(Quadratic(f"hash.{tag}.nz0", one - nz, t, Expr(ctx)))
            # finiteness (what the verifier's decode rejected on the public word)
            if self.kind == "bf16":
                ctx.inverse_nonzero(f"{tag}.finite", (2**exp_bits - 1) - t)
            else:
                ctx.inverse_nonzero(f"{tag}.finite", (2 ** (nb - 1) - 1) - (t * 2**frac + f))
            ctx.eq(f"{tag}.s", E(pins[f"{tag}.s"]), s)
            ctx.eq(f"{tag}.m", E(pins[f"{tag}.m"]), f + nz * 2**frac)
            ctx.eq(f"{tag}.e", E(pins[f"{tag}.e"]), t - nz - (bias - 1))
            rows += nb + 3
        return words, bit_rows, rows, None

    def finish(self, ctx: LigeroCtx, sys: System, pins: dict, aux_a, aux_b) -> int:
        return 0

    def lanes_np(self, words) -> np.ndarray:
        """(N, k) words -> (N, 16) lane elements (``poseidon2.pack_words_np``)."""
        return h2.pack_words_np(np.asarray(words, dtype=np.int64), self.nb)

    def lanes_torch(self, words):
        """(l, k) int64 torch -> (l, 16) int64 torch on the same device."""
        import torch
        g = words.reshape(words.shape[0], self.n_lanes, self.per)
        weights = (2 ** (self.nb * torch.arange(self.per, dtype=torch.int64, device=words.device)))
        return (g * weights).sum(-1)

    def bit_rows_torch(self, w):
        """(l, k) int64 torch -> (k * nb, l): the values of the hint rows ``<nm>[i].b<j>`` word-major."""
        import torch
        shifts = torch.arange(self.nb, dtype=torch.int64, device=w.device)
        bits = (w[:, :, None] >> shifts) & 1                          # (l, k, nb)
        return bits.permute(1, 2, 0).reshape(self.k * self.nb, w.shape[0])


def compose(rel: Relation, fmt=None) -> HashedRelation:
    """The hashed, private relation of ``rel`` (see the module doc).  Compiles ``rel`` afresh.  ``fmt`` (default: the
    relation's ``hash_format`` attribute, else the BF16 / E4M3 :class:`_FloatFormat`) is the operand format: which pins
    are converted, how they are re-derived from the committed bits, how the bits pack into the sponge's rate lanes."""
    p = rel.params
    sys = rel.compile(p, True)
    fmt = getattr(rel, "hash_format", None) if fmt is None else fmt
    fmt = _FloatFormat(rel) if fmt is None else fmt
    k = rel.k
    nb, per, n_lanes = fmt.elem_bits, fmt.per, fmt.n_lanes
    m0, l0, q0, h0 = sys.m, len(sys.linear), len(sys.quadratic), len(sys.hints)
    # --- 1. operand pins -> hint rows ------------------------------------------------------------------------------
    want = fmt.pin_names()
    pins = {name: idx for idx, name in sys.pins}
    if set(pins) != want:
        raise ValueError(f"{rel.name}: operand pins {sorted(set(pins) ^ want)} differ from the expected decode pins")
    sys.pins = []
    sys.linear = [c for c in sys.linear if c.pub is None]
    for i, op in enumerate(sys.program):
        if op[0] == "pin":
            sys.program[i] = ("hint", op[1], op[2])
    for name in fmt.pin_order(want):
        idx = pins[name]
        sys.rows[idx].kind = "hint"
        sys.rows[idx].spec = (name,)
        sys.hints.append((idx, name))
    ctx = LigeroCtx(p)
    ctx.sys = sys
    ctx.scope = "hash"
    rows_private = 0
    elems = {}
    bit_rows = {}       # row index of every word bit, word-major (the device kernel packs lanes from them)
    aux = {}
    for nm in ("a", "b"):
        elems[nm], bit_rows[nm], r, aux[nm] = fmt.decode(ctx, sys, pins, nm)
        rows_private += r
        if len(elems[nm]) != n_lanes * per or len(bit_rows[nm]) != n_lanes * per * nb:
            raise ValueError(f"{rel.name}: operand {nm} decodes to {len(elems[nm])} elements / {len(bit_rows[nm])} bits, "
                             f"not {n_lanes} lanes x {per} x {nb} bits")
    rows_private += fmt.finish(ctx, sys, pins, aux["a"], aux["b"])
    m1 = sys.m
    # --- 2. the sponges ------------------------------------------------------------------------------------------------
    sbox_rows: list[tuple] = []
    sbox = _sbox_gadget(ctx, sbox_rows)
    ends: list[Expr] = []
    sponges = []
    for _, nm, role in ROLES:
        iv = h2.IV(role, fmt.iv_word_bits, rel.steps * fmt.n_elems)
        lanes = []
        for j in range(n_lanes):
            e = Expr(ctx)
            for q in range(per):
                e = e + elems[nm][per * j + q] * 2 ** (nb * q)
            lanes.append(_reduce(e))
        lanes += [Expr(ctx) for _ in range(h2.RATE - n_lanes)]
        caps = [ctx.hint(f"{nm}.cap[{x}]") for x in range(h2.CAPACITY)]
        state = lanes + [_reduce(c + iv[h2.RATE + x]) for x, c in enumerate(caps)]
        n0 = len(sbox_rows)
        out = h2.permute_ring(state, sbox, _reduce)
        sponges.append((bit_rows[nm], [next(iter(c.lin)) for c in caps], list(iv[h2.RATE:]), sbox_rows[n0:]))
        sys.chain["c"] += [next(iter(c.lin)) for c in caps]
        sys.chain["y"] += [_reduce(out[h2.RATE + x] - iv[h2.RATE + x]) for x in range(h2.CAPACITY)]
        ends += out[:h2.OUT]
    m2 = sys.m
    if DEVICE_SPONGE:
        sys.witness_hooks = [Poseidon2Witness(nb, per, sponges, n_lanes)]
    # --- 3. the digests at the chain end ---------------------------------------------------------------------------
    ctx.scope = ""                      # the public names carry their own prefix
    is_end = ctx.pin(HASH_PIN_END, 0, 2)
    for x, st in enumerate(ends):
        d = ctx.pin(hash_pin(x), 0, P)
        sys.quadratic.append(Quadratic(f"hash.end[{x}]", is_end, _reduce(st - d), Expr(ctx)))
    census = {"rows_base": m0, "rows_private_operands": m1 - m0, "rows_hash": m2 - m1, "rows_digest_pins": sys.m - m2,
              "rows_total": sys.m, "rows_added": sys.m - m0, "linear_added": len(sys.linear) - l0,
              "quadratic_added": len(sys.quadratic) - q0, "hints_added": len(sys.hints) - h0,
              "permutations_per_unit": 2, "sbox_rows_per_permutation": 4 * (h2.WIDTH * 2 * h2.HALF_FULL_ROUNDS + h2.PARTIAL_ROUNDS),
              "linked_rows": len(sys.chain["c"]), "word_bits": nb, "k": k}
    if fmt.lane_bits != h2.LANE_BITS or n_lanes != h2.RATE:
        census.update({"lane_bits": fmt.lane_bits, "rate_lanes_used": n_lanes, "elements_per_unit": fmt.n_elems})
    extra = [n for _, n in sys.hints[h0:]]
    return HashedRelation(rel, sys, nb, k, h0, extra, census, fmt=fmt)


# ---------------------------------------------------------------------------------------------------------------------
# Witness side: the extra hint rows and the digests
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


def sponge_states(lanes, iv: list[int], n_vus: int, steps: int, l: int):
    """``lanes`` (l, 16) int64 torch tensor of rate-lane elements (pad columns 0) -> (cap_in (8, l): the linked rows
    ``cap - IV`` per column, digests (n_vus, 8)) by running the sponge of every VU along its ``steps`` columns, all VUs
    in lockstep, from the capacity ``iv``."""
    import torch
    dev = lanes.device
    el = lanes
    cap_in = torch.zeros((h2.CAPACITY, l), dtype=torch.int64, device=dev)
    st = torch.zeros((h2.WIDTH, n_vus), dtype=torch.int64, device=dev)
    for x in range(h2.CAPACITY):
        st[h2.RATE + x] = iv[h2.RATE + x]
    cols0 = torch.arange(n_vus, device=dev) * steps
    for s in range(steps):
        cols = cols0 + s
        for x in range(h2.CAPACITY):
            cap_in[h2.RATE - h2.RATE + x, cols] = (st[h2.RATE + x] - iv[h2.RATE + x]) % P
        st = torch.cat([el[cols].T, st[h2.RATE:]], 0)
        st = permute_torch(st)
    return cap_in, st[:h2.OUT].T.contiguous()


def row_sponges(hr: HashedRelation, rows, role: int, device) -> tuple:
    """The sponge of every committed row at once: ``rows`` (N, K) words -> (cap (N, steps, 8): the linked capacity rows
    of every step, digests (N, 8)).  The prover computes this once per committed set (``HashedRelationRunner.commit``)
    and gathers per sub-batch instead of re-running 2 x steps permutations per proof."""
    import torch
    rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)
    n, steps, k = rows.shape[0], hr.steps, hr.k
    words = rows.reshape(n * steps, k)
    cap, dg = sponge_states(hr.fmt.lanes_torch(words), hr.iv(role), n, steps, n * steps)
    return cap.T.reshape(n, steps, h2.CAPACITY).contiguous(), dg


def hash_hints(hr: HashedRelation, a, b, n_vus: int, l: int, device, sponges: dict | None = None) -> tuple:
    """The appended hint rows (len(extra_hint_names), l) for operand words ``a``, ``b`` (l, k) int64 (pad columns 0), and
    the digests (n_vus, 16) the statement publishes.  ``sponges``: ``{"a": (cap (n_vus, steps, 8), digests (n_vus, 8)),
    "b": ...}`` precomputed by :func:`row_sponges` for exactly these VUs' rows (the caller checked the words match)."""
    import torch
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    vals = hr.rel.public_vectors(hr.rel.params, None, a, b, None, device=device)
    fmt, k = hr.fmt, hr.k
    n_bits = sum(fmt.bit_layout)
    digests, bit_blocks, cap_blocks = [], [], []
    for _, nm, role in ROLES:
        w = a if nm == "a" else b
        bit_blocks.append(fmt.bit_rows_torch(w))                  # rows nm[i].b{j}, i major
        if sponges is not None and nm in sponges:
            cap_v, dg = sponges[nm]
            cap = torch.zeros((h2.CAPACITY, l), dtype=torch.int64, device=device)
            cap[:, :n_vus * hr.steps] = cap_v.to(device).reshape(n_vus * hr.steps, h2.CAPACITY).T
            dg = dg.to(device)
        else:
            cap, dg = sponge_states(fmt.lanes_torch(w), hr.iv(role), n_vus, hr.steps, l)
        cap_blocks.append(cap)
        digests.append(dg)
    n_dec = len(hr.extra_hint_names) - len(ROLES) * (n_bits + h2.CAPACITY)
    if _block_layout(hr, n_dec):
        # compose() appends the rows as [converted decode pins | a bits | b bits | a.cap | b.cap]: a handful of copies
        # instead of one per row (720 rows x 13 sub-batches was ~0.1 s per 4096 VUs)
        dec = torch.stack([vals[n].to(device) for n in hr.extra_hint_names[:n_dec]])
        out = torch.cat([dec] + bit_blocks + cap_blocks)
    else:
        for (_, nm, _), bits, cap in zip(ROLES, bit_blocks, cap_blocks):
            r = 0
            for i in range(k):
                for j in range(fmt.bit_layout[i]):
                    vals[f"{nm}[{i}].b{j}"] = bits[r]
                    r += 1
            for x in range(h2.CAPACITY):
                vals[f"{nm}.cap[{x}]"] = cap[x]
        out = torch.stack([vals[n].to(device) for n in hr.extra_hint_names])
    return out, torch.cat(digests, 1)


def _block_layout(hr: HashedRelation, n_dec: int) -> bool:
    """Whether ``hr.extra_hint_names`` is ``[decode rows | a bits | b bits | a.cap | b.cap]`` (memoised on ``hr``)."""
    if hr.block_layout is None:
        names = hr.extra_hint_names
        tail = [f"{nm}[{i}].b{j}" for _, nm, _ in ROLES for i in range(hr.k) for j in range(hr.fmt.bit_layout[i])]
        tail += [f"{nm}.cap[{x}]" for _, nm, _ in ROLES for x in range(h2.CAPACITY)]
        hr.block_layout = n_dec >= 0 and names[n_dec:] == tail and all("].b" not in n for n in names[:n_dec])
    return hr.block_layout


def hash_public_vectors(p, c, a, b, y, device="cpu"):
    """The verifier's public vectors of a hashed statement: ``a`` is the (l, 17) public table ``[is_end | d[0..15]]``
    (zeros off the chain ends), ``b`` the per-VU leaf indices (not per-column: only hashed into the statement)."""
    import torch
    t = torch.as_tensor(a, dtype=torch.int64, device=device)
    if t.ndim != 2 or t.shape[1] != 1 + N_DIGEST:
        raise ValueError("hashed statement: public table must be (l, 17)")
    if ((t < 0) | (t >= P)).any() or ((t[:, 0] != 0) & (t[:, 0] != 1)).any():
        raise ValueError("hashed statement: public table out of range")
    out = {HASH_PIN_END: t[:, 0]}
    for x in range(N_DIGEST):
        out[hash_pin(x)] = t[:, 1 + x]
    return out


def public_table(digests, n_vus: int, steps: int, l: int) -> np.ndarray:
    """(l, 17) int64: ``is_end`` and the 16 digest lanes at every VU's last column, 0 elsewhere."""
    d = np.asarray(digests, dtype=np.int64).reshape(n_vus, N_DIGEST)
    tab = np.zeros((l, 1 + N_DIGEST), dtype=np.int64)
    ends = np.arange(n_vus) * steps + steps - 1
    tab[ends, 0] = 1
    tab[ends, 1:] = d
    return tab
