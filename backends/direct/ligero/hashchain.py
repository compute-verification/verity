"""``authentication = included-hash``: the operand rows hashed INSIDE the B-Ligero relation (lane hash-relation), with the
row hash a pluggable :class:`~backends.direct.ligero.leaf.base.LeafScheme` (lane leaf-iface; Poseidon2 is the first
plugin, ``leaf/poseidon2.py``).

``compose(rel, leaf="poseidon2")`` takes a registered relation's chain-compiled system (``relations.py``: bf16-hopper,
fp8-hopper, fp8-ada, bf16-ampere; their operand words enter as *pinned* decodes ``a[i].s / .m / .e``, public per column)
and turns it into the private, self-hashing relation of the same GEMM step -- without editing the unit compilers
(``unit.py``, ``fp8/relation.py``):

1. **Private operands.**  Every operand pin becomes a hint row; the word behind it is ``word_bits`` boolean hint rows
   ``hash.a[i].b<j>`` and the decode is re-established by constraints on those rows: ``s = b_msb``, ``m = f + nz 2^frac``,
   ``e = t - nz - (bias - 1)`` with ``nz = [t != 0]`` (``t * inv = nz``, ``(1 - nz) t = 0``, ``nz`` boolean) and the
   finiteness the verifier used to check on the public word (``t != 2^exp - 1`` for BF16; ``(t, f) != (15, 7)`` for
   E4M3) as one ``inverse_nonzero``.  The ranges the pins declared (``Row.lo / hi``) are now theorems of these
   constraints, so every downstream audit of the unit compiler stands.  18 rows per BF16 word, 13 per E4M3 byte.
2. **Two leaf chains along the VU** (one per operand: unit ``i`` absorbs its ``k`` words of ``a`` -- the x row's words
   ``[k i, k i + k)`` -- into chain X and its words of ``b`` (the W column) into chain W): the scheme's
   :meth:`~LeafScheme.gadget` per column, its ``carry_elems`` carried elements per operand as *linked* hint rows
   ``<nm>.cap[x]`` (``sys.chain["c"] / ["y"]``: ``carry_in[j + 1] = carry_out[j]``, 0 at a chain start) -- ``2 x carry_elems``
   more linked rows than the accumulator's three (``chain.py``: monomial coefficients for the extras).  Poseidon2: 852
   product rows per permutation, 8 capacity lanes carried as ``capacity - IV``.
3. **The digests at the chain end.**  Public pin ``hash.is_end`` (1 at the last unit of a VU, 0 elsewhere -- the verifier
   knows it from the layout) and ``2 x digest_elems`` public pins ``hash.d[..]`` (the x digest, then the W digest; 0 off
   the ends), tied by ``is_end * (digest_x - d[x]) = 0``: at an end column the chain's digest expressions equal the
   published digest; elsewhere nothing is revealed (the ZK rows hide the states).  The relation's own public word
   (``y16``) stays a chain end as before.

The statement (``serialize.py`` v5) then carries, instead of 27 MB of operand words, the three roots, each VU's leaf
indices, the published digests and SHA-256 Merkle multiproofs from the scheme's leaves to the roots (``hashauth.py``);
the verifier never learns an operand word.  Binding: the scheme's collision resistance (``LeafScheme.assumption``) +
SHA-256 above; privacy: the Ligero ZK rows + the digest of a 1536-word row (a low-entropy row would need a salt: not done,
see the lane notes).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import poseidon2 as h2
from .compile import Expr, LigeroCtx, P, Quadratic, System
from .leaf import base as leaf_base
from .leaf import registry as leaf_registry
from .leaf.poseidon2 import (  # noqa: F401  (re-exported: hashchain_test / witness_device_test address them here)
    DEVICE_SPONGE,
    Poseidon2Witness,
    _sbox_gadget,
    _Sym,
    permute_torch,
    sponge_kernel_source,
    sponge_states,
)
from .relations import Relation

__all__ = ["HASH_PIN_END", "N_DIGEST", "N_ROW_DIGEST", "PIN_ROLE", "ROLES", "ROWS_TAG", "SHARED_TAG", "HashedRelation",
           "RowHashSystem", "compose", "compose_shared", "hash_hints", "hash_pin", "hash_public_vectors", "pack_lanes",
           "public_table", "row_chain", "row_hash_hints", "row_hash_public_vectors", "row_hash_system", "row_sponges",
           "shared_hints", "shared_public_vectors", "shared_units", "sponge_states_roles"]

#: ``row_chain`` runs a scheme's ``row_sponges`` device kernel (Poseidon2: one thread per committed row) when it has one, and
#: the committer takes the tree leaves' digests from it instead of hashing the rows again on the host (``hashauth``).  Same
#: states, digests, leaves and roots; ``LIGERO_ROW_SPONGE_KERNEL=0`` keeps the torch sponge and the host digests.
ROW_SPONGE_KERNEL = os.environ.get("LIGERO_ROW_SPONGE_KERNEL", "1") != "0"

HASH_PIN_END = "hash.is_end"
N_DIGEST = 2 * h2.OUT          # the default (Poseidon2) digest table width: x digest (8) then W digest (8)
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
    leaf: leaf_base.LeafScheme  # the scheme (one instance per compose; its witness hook is on ``sys``)
    block_layout: bool | None = None    # memo of _block_layout (hash_hints)
    dec_index: dict | None = None       # memo of private_decode's row gather per name list
    fmt: object = None          # the relation's ``hash_format`` (``fp4/hashed.py`` FP4Format), None for the BF16 / E4M3 words

    @property
    def bit_layout(self) -> list[int]:
        """Bit rows per operand word (``<nm>[i].b<j>``), word-major."""
        return list(self.fmt.bit_layout) if self.fmt is not None else [self.word_bits] * self.k

    @property
    def steps(self) -> int:
        return self.rel.steps

    @property
    def n_digest(self) -> int:
        """Width of the public digest table per VU: the x digest then the W digest."""
        return 2 * self.leaf.digest_elems

    @property
    def name(self) -> str:
        """The relation string of the statement: ``<rel>`` for the default scheme, ``<rel>+<leaf>`` otherwise."""
        return leaf_registry.relation_name(self.rel.name, self.leaf)


def _reduce(e):
    if isinstance(e, Expr):
        return Expr(e.ctx, {k: v % P for k, v in e.lin.items()}, e.c % P)
    return e % P


def _format(rel: Relation) -> dict:
    if getattr(rel, "hash_format", None) is not None:
        raise ValueError(f"{rel.name}: its operand format ({type(rel.hash_format).__name__}) composes through compose() only "
                         f"(no row sharing, no word-level private decode)")
    p = rel.params
    if rel.word_dtype == "<u2":            # BF16: s | t(8) | f(7)
        return {"bits": 16, "frac": p.frac, "exp": p.exp, "bias": p.bias, "kind": "bf16", "sig": p.sig}
    if rel.word_dtype == "<u1":            # E4M3: s | t(4) | f(3)
        return {"bits": 8, "frac": p.op_frac, "exp": p.op_exp, "bias": p.op_bias, "kind": "e4m3", "sig": p.sig}
    raise ValueError(f"{rel.name}: no private decode for operand dtype {rel.word_dtype}")


def _private_operands(rel: Relation, sys: System, fmt: dict):
    """``compose`` step 1 on ``sys`` (the relation's chain system): the operand pins become hint rows, the words behind them
    ``nb`` boolean rows each, the decode re-established by constraints.  Returns ``(ctx, words, bit_rows)``: the context
    (scope ``hash``), the word expressions per operand and the bit rows word-major."""
    p = rel.params
    k = rel.k
    nb, frac, exp_bits, bias = fmt["bits"], fmt["frac"], fmt["exp"], fmt["bias"]
    want = {f"{nm}[{i}].{c}" for nm in ("a", "b") for i in range(k) for c in ("s", "m", "e")}
    pins = {name: idx for idx, name in sys.pins}
    if set(pins) != want:
        raise ValueError(f"{rel.name}: operand pins {sorted(set(pins) ^ want)} differ from the expected decode triples")
    sys.pins = []
    sys.linear = [c for c in sys.linear if c.pub is None]
    for i, op in enumerate(sys.program):
        if op[0] == "pin":
            sys.program[i] = ("hint", op[1], op[2])
    for name in sorted(want, key=lambda n: (n[0], int(n.split("[")[1].split("]")[0]), n[-1])):
        idx = pins[name]
        sys.rows[idx].kind = "hint"
        sys.rows[idx].spec = (name,)
        sys.hints.append((idx, name))
    ctx = LigeroCtx(p)
    ctx.sys = sys
    ctx.scope = "hash"
    one = Expr(ctx, {}, 1)
    E = lambda idx: Expr(ctx, {idx: 1}, 0)  # noqa: E731
    rows_private = 0
    words = {"a": [], "b": []}
    bit_rows = {"a": [], "b": []}       # row index of every word bit, word-major (the device kernel packs lanes from them)
    for nm in ("a", "b"):
        for i in range(k):
            tag = f"{nm}[{i}]"
            bits = [ctx.hint(f"{tag}.b{j}") for j in range(nb)]
            bit_rows[nm] += [next(iter(b.lin)) for b in bits]
            for j, b in enumerate(bits):
                ctx.boolean(f"{tag}.bool{j}", b)
            w = _reduce(sum((b * 2**j for j, b in enumerate(bits)), Expr(ctx)))
            words[nm].append(w)
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
            if fmt["kind"] == "bf16":
                ctx.inverse_nonzero(f"{tag}.finite", (2**exp_bits - 1) - t)
            else:
                ctx.inverse_nonzero(f"{tag}.finite", (2 ** (nb - 1) - 1) - (t * 2**frac + f))
            ctx.eq(f"{tag}.s", E(pins[f"{tag}.s"]), s)
            ctx.eq(f"{tag}.m", E(pins[f"{tag}.m"]), f + nz * 2**frac)
            ctx.eq(f"{tag}.e", E(pins[f"{tag}.e"]), t - nz - (bias - 1))
            rows_private += nb + 3
    return ctx, words, bit_rows


def _formatted_operands(rel: Relation, sys: System, fmt) -> tuple[LigeroCtx, dict]:
    """``compose`` step 1 for a relation with its own operand format (``rel.hash_format``: ``fp4/hashed.py`` FP4Format):
    the format's pins become hint rows in its order, ``fmt.decode`` re-derives each operand's from its committed bit rows,
    ``fmt.finish`` the pins joining both operands.  Returns ``(ctx, elements)``: the lane element expressions per operand
    (what the leaf's :class:`~backends.direct.ligero.leaf.poseidon2.LaneFormat` packs)."""
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
    ctx = LigeroCtx(rel.params)
    ctx.sys = sys
    ctx.scope = "hash"
    elems, aux = {}, {}
    for nm in ("a", "b"):
        elems[nm], bit_rows, _, aux[nm] = fmt.decode(ctx, sys, pins, nm)
        if len(elems[nm]) != fmt.n_elems or len(bit_rows) != sum(fmt.bit_layout):
            raise ValueError(f"{rel.name}: operand {nm} decodes to {len(elems[nm])} elements / {len(bit_rows)} bits, "
                             f"not {fmt.n_elems} / {sum(fmt.bit_layout)}")
    fmt.finish(ctx, sys, pins, aux["a"], aux["b"])
    return ctx, elems


def _lanes_of(ctx: LigeroCtx, words: list, nb: int) -> list[Expr]:
    """The 16 rate lanes of a column as affine expressions of its word expressions (``per`` words of ``nb`` bits each,
    little-endian: ``poseidon2.pack_words_np``)."""
    per = h2.LANE_BITS // nb
    if len(words) != h2.RATE * per:
        raise ValueError(f"k = {len(words)} words of {nb} bits are not one rate block of 16 lanes")
    lanes = []
    for j in range(h2.RATE):
        e = Expr(ctx)
        for q in range(per):
            e = e + words[per * j + q] * 2 ** (nb * q)
        lanes.append(_reduce(e))
    return lanes


def _lane_rows(ctx: LigeroCtx, nm: str, lanes: list[Expr]) -> list[int]:
    """The lanes as their own hint rows ``<nm>.lane[j]`` constrained equal to the packed words (the fingerprint constraint
    of chain.py touches 16 rows per operand instead of the 128 / 256 bit rows behind them)."""
    rows = []
    for j, e in enumerate(lanes):
        r = ctx.hint(f"{nm}.lane[{j}]")
        ctx.learn(r, 0, 2**h2.LANE_BITS)          # a theorem of the equality below: e is the packed bits, in [0, 2^16)
        ctx.eq(f"{nm}.lane{j}", r, e)
        rows.append(next(iter(r.lin)))
    return rows



def compose(rel: Relation, leaf: str | leaf_base.LeafScheme | None = None) -> HashedRelation:
    """The hashed, private relation of ``rel`` under the leaf scheme ``leaf`` (a registry name -- ``"poseidon2"`` /
    ``"hash"`` / ``None`` are the default -- or a scheme instance).  Compiles ``rel`` afresh.  A relation with a
    ``hash_format`` (``fp4/hashed.py``) decodes its operands through it and hands the leaf the format's lane elements
    (the scheme's ``with_lanes``; Poseidon2 only)."""
    scheme = leaf_registry.get(leaf) if leaf is None or isinstance(leaf, str) else leaf
    leaf_base.check_scheme(scheme)
    p = rel.params
    sys = rel.compile(p, True)
    hfmt = getattr(rel, "hash_format", None)
    k = rel.k
    m0, l0, q0, h0 = sys.m, len(sys.linear), len(sys.quadratic), len(sys.hints)
    # --- 1. operand pins -> hint rows ------------------------------------------------------------------------------
    if hfmt is None:
        nb = _format(rel)["bits"]
        ctx, words, _ = _private_operands(rel, sys, _format(rel))
        row_words = rel.steps * k
    else:
        if getattr(scheme, "with_lanes", None) is None:
            raise ValueError(f"{rel.name}: leaf {scheme.name} does not take a lane format (its operands are {hfmt.n_elems} "
                             f"{hfmt.elem_bits}-bit elements on {hfmt.lane_bits}-bit lanes)")
        scheme = scheme.with_lanes(hfmt)
        nb = hfmt.elem_bits
        ctx, words = _formatted_operands(rel, sys, hfmt)
        row_words = rel.steps * hfmt.n_elems
    m1 = sys.m
    # --- 2. the leaf chains: the scheme's gadget per operand, its carried state as linked hint rows -------------------
    ends: list[Expr] = []
    for _, nm, role in ROLES:
        carry = [ctx.hint(f"{nm}.cap[{x}]") for x in range(scheme.carry_elems)]
        digest, carry_out = scheme.gadget(ctx, words[nm], word_bits=nb, role=role, row_words=row_words, carry_in=carry,
                                          first=False, last=True)
        if len(digest) != scheme.digest_elems or len(carry_out) != scheme.carry_elems:
            raise ValueError(f"leaf {scheme.name}: gadget returned {len(digest)} digest / {len(carry_out)} carry expressions, "
                             f"declared {scheme.digest_elems} / {scheme.carry_elems}")
        sys.chain["c"] += [next(iter(c.lin)) for c in carry]
        sys.chain["y"] += [_reduce(e) for e in carry_out]
        ends += [_reduce(e) for e in digest]
    m2 = sys.m
    hook = scheme.witness()
    if hook is not None:
        sys.witness_hooks = [hook]
    # --- 3. the digests at the chain end ---------------------------------------------------------------------------
    ctx.scope = ""                      # the public names carry their own prefix
    is_end = ctx.pin(HASH_PIN_END, 0, 2)
    for x, st in enumerate(ends):
        d = ctx.pin(hash_pin(x), 0, P)
        sys.quadratic.append(Quadratic(f"hash.end[{x}]", is_end, _reduce(st - d), Expr(ctx)))
    census = {"rows_base": m0, "rows_private_operands": m1 - m0, "rows_hash": m2 - m1, "rows_digest_pins": sys.m - m2,
              "rows_total": sys.m, "rows_added": sys.m - m0, "linear_added": len(sys.linear) - l0,
              "quadratic_added": len(sys.quadratic) - q0, "hints_added": len(sys.hints) - h0,
              **scheme.census(),
              "linked_rows": len(sys.chain["c"]), "word_bits": nb, "k": k, "leaf": scheme.name,
              "digest_elems": scheme.digest_elems, "carry_elems": scheme.carry_elems}
    extra = [n for _, n in sys.hints[h0:]]
    return HashedRelation(rel, sys, nb, k, h0, extra, census, scheme, fmt=hfmt)


# ---------------------------------------------------------------------------------------------------------------------

SHARED_TAG = b"share/v1|"          # the GEMM side's statement tag prefix (protocol.RelationHooks.tag)
ROWS_TAG = b"rows/v1|"             # the hash side's
PIN_ROLE = "hash.role"             # hash side: 0 = an x row (ROLE_X), 1 = a W column (ROLE_W), constant over a unit
N_ROW_DIGEST = h2.OUT              # the hash side publishes one digest (8 lanes) per unit


def compose_shared(rel: Relation, coords: int | None = None) -> HashedRelation:
    """The GEMM side of row sharing (module comment of chain.py, "Fingerprint constraints"): ``compose`` step 1 (private
    operands) + 16 lane rows per operand, NO sponges; the chain carries the fingerprint spec ``sys.chain["fp"]`` in the
    "coin" mode (rho from challenge 1, F a prover message).  One public pin, ``hash.is_end`` (the layout's end mask),
    keeps the device witness path.  ~4505 rows per unit for fp8-ada against 6210 hashed and 3769 bare."""
    from .chain import FP_COORDS
    C = FP_COORDS if coords is None else int(coords)
    p = rel.params
    sys = rel.compile(p, True)
    fmt = _format(rel)
    k, nb = rel.k, fmt["bits"]
    m0, l0, q0, h0 = sys.m, len(sys.linear), len(sys.quadratic), len(sys.hints)
    ctx, words, bit_rows = _private_operands(rel, sys, fmt)
    m1 = sys.m
    lanes = {nm: _lane_rows(ctx, nm, _lanes_of(ctx, words[nm], nb)) for _, nm, _ in ROLES}
    m2 = sys.m
    ctx.scope = ""
    ctx.pin(HASH_PIN_END, 0, 2)
    sys.chain["fp"] = {"lanes": [lanes[nm] for _, nm, _ in ROLES], "coords": C, "mode": "coin"}
    census = {"rows_base": m0, "rows_private_operands": m1 - m0, "rows_lanes": m2 - m1, "rows_pins": sys.m - m2,
              "rows_hash": 0, "rows_total": sys.m, "rows_added": sys.m - m0, "linear_added": len(sys.linear) - l0,
              "quadratic_added": len(sys.quadratic) - q0, "hints_added": len(sys.hints) - h0, "permutations_per_unit": 0,
              "linked_rows": len(sys.chain["c"]), "fp_coords": C, "fp_constraints_per_vu": 2 * C,
              "word_bits": nb, "k": k}
    extra = [n for _, n in sys.hints[h0:]]
    return HashedRelation(rel, sys, nb, k, h0, extra, census, leaf_registry.get("poseidon2"))


@dataclass
class RowHashSystem:
    """The hash side: ``sys`` hashes ONE operand row per unit over ``steps`` columns (``k`` words of ``word_bits`` per
    column) with one Poseidon2 sponge whose IV the ``hash.role`` pin selects, publishes the digest at the chain end and
    proves the row's fingerprint (``sys.chain["fp"]``, "public" mode: rho and F are statement values)."""
    rel: Relation
    sys: System
    word_bits: int
    k: int
    steps: int
    hint_names: list
    census: dict
    leaf: leaf_base.LeafScheme = None      # Poseidon2 (row_chain / row_sponges gather the committed rows' sponges through it)


def row_hash_system(rel: Relation, coords: int | None = None) -> RowHashSystem:
    from .chain import FP_COORDS
    C = FP_COORDS if coords is None else int(coords)
    p = rel.params
    fmt = _format(rel)
    k, nb, steps = rel.k, fmt["bits"], rel.steps
    ctx = LigeroCtx(p)
    sys = ctx.sys
    ctx.scope = ""
    is_end = ctx.pin(HASH_PIN_END, 0, 2)
    role = ctx.pin(PIN_ROLE, 0, 2)
    dpins = [ctx.pin(hash_pin(x), 0, P) for x in range(N_ROW_DIGEST)]
    ctx.scope = "hash"
    bits_all: list[int] = []
    words: list[Expr] = []
    for i in range(k):
        bits = [ctx.hint(f"r[{i}].b{j}") for j in range(nb)]
        bits_all += [next(iter(b.lin)) for b in bits]
        for j, b in enumerate(bits):
            ctx.boolean(f"r[{i}].bool{j}", b)
        words.append(_reduce(sum((b * 2**j for j, b in enumerate(bits)), Expr(ctx))))
    m1 = sys.m
    lane_rows = _lane_rows(ctx, "r", _lanes_of(ctx, words, nb))
    lanes = [Expr(ctx, {r: 1}, 0) for r in lane_rows]
    m2 = sys.m
    # the sponge: IV(role) = IV_x + role (IV_w - IV_x) on the capacity, the linked rows hold cap - IV(role)
    iv_x, iv_w = h2.IV(h2.ROLE_X, nb, steps * k), h2.IV(h2.ROLE_W, nb, steps * k)
    ivd = [(iv_w[h2.RATE + x] - iv_x[h2.RATE + x]) % P for x in range(h2.CAPACITY)]
    caps = [ctx.hint(f"cap[{x}]") for x in range(h2.CAPACITY)]
    iv_expr = [_reduce(role * ivd[x] + iv_x[h2.RATE + x]) for x in range(h2.CAPACITY)]
    state = lanes + [_reduce(c + iv_expr[x]) for x, c in enumerate(caps)]
    sbox_rows: list[tuple] = []
    out = h2.permute_ring(state, _sbox_gadget(ctx, sbox_rows), _reduce)
    m3 = sys.m
    sys.chain = {"c": [next(iter(c.lin)) for c in caps], "y": [_reduce(out[h2.RATE + x] - iv_expr[x]) for x in range(h2.CAPACITY)],
                 "y16": Expr(ctx), "fp": {"lanes": [lane_rows], "coords": C, "mode": "public"}}
    for x in range(N_ROW_DIGEST):
        sys.quadratic.append(Quadratic(f"hash.end[{x}]", is_end, _reduce(out[x] - dpins[x]), Expr(ctx)))
    if DEVICE_SPONGE:
        role_row = next(iter(role.lin))
        sys.witness_hooks = [Poseidon2Witness(nb, h2.LANE_BITS // nb, [(bits_all, sys.chain["c"], list(iv_x[h2.RATE:]), sbox_rows)],
                                              role_row=role_row, ivd=ivd)]
    census = {"rows_pins": 2 + N_ROW_DIGEST, "rows_bits": m1 - 2 - N_ROW_DIGEST, "rows_lanes": m2 - m1, "rows_hash": m3 - m2,
              "rows_total": sys.m, "linear": len(sys.linear), "quadratic": len(sys.quadratic), "linked_rows": h2.CAPACITY,
              "permutations_per_unit": 1, "fp_coords": C, "word_bits": nb, "k": k, "steps": steps}
    return RowHashSystem(rel, sys, nb, k, steps, [n for _, n in sys.hints], census, leaf_registry.get("poseidon2"))


def shared_hints(hr: HashedRelation, a, b, l: int, device) -> "torch.Tensor":
    """The GEMM side's appended hint rows (len(extra_hint_names), l): the converted decode pins, the bit rows of ``a`` and
    ``b`` (l, k) and their 16 lane rows each (``poseidon2.pack_words_np``)."""
    import torch
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    nb, k = hr.word_bits, hr.k
    shifts = torch.arange(nb, dtype=torch.int64, device=device)
    n_dec = len(hr.extra_hint_names) - len(ROLES) * (k * nb + h2.RATE)
    dec = private_decode(hr, a, b, hr.extra_hint_names[:n_dec])
    blocks = [dec]
    for _, nm, _ in ROLES:
        w = a if nm == "a" else b
        blocks.append(((w[:, :, None] >> shifts) & 1).permute(1, 2, 0).reshape(k * nb, l))
    for _, nm, _ in ROLES:
        w = a if nm == "a" else b
        blocks.append(pack_lanes(w, nb).T.contiguous())
    return torch.cat(blocks)


def private_decode(hr: HashedRelation, a, b, names: list) -> "torch.Tensor":
    """The converted decode pins ``names`` (``<op>[<i>].<s|m|e>``) as hint rows (len(names), l) from the operand words
    ``a``, ``b`` (l, k) on the device, by the decode the private-operand constraints of :func:`_private_operands` fix
    (s = top bit, m = f + nz 2^frac, e = t - nz - (bias - 1)) -- no domain checks, so no host round trip (the relation's
    ``public_vectors`` validates with ``.any()``: a device sync per call).  Equal to ``public_vectors`` on the finite
    domain (``hashchain_test.test_private_decode_is_the_public_decode``); off it the constraints reject the witness."""
    import torch
    fmt = _format(hr.rel)
    nb, frac, exp_bits, bias = fmt["bits"], fmt["frac"], fmt["exp"], fmt["bias"]
    k = a.shape[1]
    comps = []
    for w in (a, b):
        s = w >> (nb - 1)
        t = (w >> frac) & (2**exp_bits - 1)
        f = w & (2**frac - 1)
        nz = (t != 0).to(torch.int64)
        comps += [s, f + nz * 2**frac, t - nz - (bias - 1)]
    allrows = torch.stack(comps).permute(0, 2, 1).reshape(6 * k, -1)      # (op, comp, i) major
    key = (tuple(names), str(a.device))
    if hr.dec_index is None:
        hr.dec_index = {}
    idx = hr.dec_index.get(key)
    if idx is None:
        where = {}
        for o, nm in enumerate(("a", "b")):
            for c, comp in enumerate(("s", "m", "e")):
                for i in range(k):
                    where[f"{nm}[{i}].{comp}"] = (3 * o + c) * k + i
        idx = torch.as_tensor([where[n] for n in names], dtype=torch.int64).to(a.device)
        hr.dec_index[key] = idx
    return allrows.index_select(0, idx)


def pack_lanes(words, nb: int):
    """(l, k) words of ``nb`` bits -> (l, 16) lanes (``poseidon2.pack_words_np`` on the device)."""
    import torch
    per = h2.LANE_BITS // nb
    l = words.shape[0]
    w = words.reshape(l, h2.RATE, per)
    mult = (2 ** (nb * torch.arange(per, dtype=torch.int64, device=words.device)))[None, None, :]
    return (w * mult).sum(2) % P


def row_hash_hints(rh: RowHashSystem, rows, roles, n_units: int, l: int, device, sponges: tuple | None = None) -> tuple:
    """The hash side's witness for ``rows`` (n_units, steps * k) words with ``roles`` (n_units,) in {0, 1}: the hint rows
    (bits | lanes | caps) (n_hints, l), the public table (l, 2 + 8) ``[is_end | role | d0..d7]`` and the digests
    (n_units, 8).  ``sponges``: ``(cap (n_units, steps, 8), digests (n_units, 8))`` precomputed by :func:`row_sponges` for
    exactly these rows (the committed set's, once at commit time; the caller checked the words match) -- else the
    sponges run here (steps permutations on the device)."""
    import torch
    rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device).reshape(n_units, rh.steps, rh.k)
    roles = torch.as_tensor(np.asarray(roles, dtype=np.int64), device=device)
    nb, k, steps = rh.word_bits, rh.k, rh.steps
    if sponges is None:
        words = torch.zeros((l, k), dtype=torch.int64, device=device)
        words[:n_units * steps] = rows.reshape(n_units * steps, k)
        cap, dg = sponge_states_roles(words, nb, roles, steps * k, n_units, steps, l)
        hints = row_hash_witness(rh, rows, n_units, l, cap=cap)
    else:
        cap_u, dg = sponges
        hints = row_hash_witness(rh, rows, n_units, l, cap_units=torch.as_tensor(cap_u, device=device))
        dg = torch.as_tensor(dg, device=device)
    table = torch.zeros((l, 2 + N_ROW_DIGEST), dtype=torch.int64, device=device)
    ends = torch.arange(n_units, device=device) * steps + steps - 1
    table[ends, 0] = 1
    table[:n_units * steps, 1] = roles.repeat_interleave(steps)
    table[ends, 2:] = dg
    return hints, table, dg


def row_hash_witness(rh: RowHashSystem, rows, n_units: int, l: int, cap_units=None, cap=None) -> "torch.Tensor":
    """The hash side's hint rows (bits | lanes | caps) (n_hints, l) from device tensors alone (no host round trip):
    ``rows`` (n_units, steps * k) words, the linked capacity rows either per unit ``cap_units`` (n_units, steps, 8)
    (the committed rows' sponges, gathered) or already laid out ``cap`` (8, l)."""
    import torch
    nb, k, steps = rh.word_bits, rh.k, rh.steps
    device = rows.device
    words = torch.zeros((l, k), dtype=torch.int64, device=device)
    words[:n_units * steps] = rows.reshape(n_units * steps, k)
    shifts = torch.arange(nb, dtype=torch.int64, device=device)
    bits = ((words[:, :, None] >> shifts) & 1).permute(1, 2, 0).reshape(k * nb, l)
    lanes = pack_lanes(words, nb).T.contiguous()
    if cap is None:
        cap = torch.zeros((h2.CAPACITY, l), dtype=torch.int64, device=device)
        cap[:, :n_units * steps] = cap_units.reshape(n_units * steps, h2.CAPACITY).T
    return torch.cat([bits, lanes, cap])


def row_hash_public_vectors(p, c, a, b, y, device="cpu"):
    """The hash side's public vectors: ``a`` the (l, 10) table ``[is_end | role | d0..d7]``; ``b`` (n_units, 2) (role, leaf
    index) only hashed into the statement."""
    import torch
    t = torch.as_tensor(a, dtype=torch.int64, device=device)
    if t.ndim != 2 or t.shape[1] != 2 + N_ROW_DIGEST:
        raise ValueError("row-hash statement: public table must be (l, 10)")
    if ((t < 0) | (t >= P)).any() or ((t[:, 0] != 0) & (t[:, 0] != 1)).any() or ((t[:, 1] != 0) & (t[:, 1] != 1)).any():
        raise ValueError("row-hash statement: public table out of range")
    out = {HASH_PIN_END: t[:, 0], PIN_ROLE: t[:, 1]}
    for x in range(N_ROW_DIGEST):
        out[hash_pin(x)] = t[:, 2 + x]
    return out


def shared_public_vectors(p, c, a, b, y, device="cpu"):
    """The GEMM side's public vectors (``compose_shared``): ``a`` the (l, 1) table ``[is_end]``; ``b`` the (n_vus, 3)
    ``(vu, x_index, w_index)`` leaf triples, only hashed into the statement (``hashauth.HashAuth.indices``)."""
    import torch
    t = torch.as_tensor(a, dtype=torch.int64, device=device)
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError("shared statement: public table must be (l, 1)")
    if ((t != 0) & (t != 1)).any():
        raise ValueError("shared statement: is_end must be 0 / 1")
    return {HASH_PIN_END: t[:, 0]}


def shared_units(auth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The hash side's units of a sub-batch from its authentication block: the distinct x rows (role 0, ascending leaf
    rank) then the distinct W columns (role 1); returns ``(units (n_units, 2) [role, rank], unit_of_x (n_vus,),
    unit_of_w (n_vus,))`` -- the unit each VU's operand row is hashed in.  Deterministic in ``auth``, so prover and
    verifier derive the same table."""
    xi = np.asarray(auth.x_index, dtype=np.int64)
    wi = np.asarray(auth.w_index, dtype=np.int64)
    xr, ux = np.unique(xi, return_inverse=True)
    wr, uw = np.unique(wi, return_inverse=True)
    units = np.concatenate([np.stack([np.zeros_like(xr), xr], 1), np.stack([np.ones_like(wr), wr], 1)], 0)
    return units, ux.reshape(-1), (uw.reshape(-1) + len(xr))


def sponge_states_roles(words, word_bits: int, roles, n_words: int, n_units: int, steps: int, l: int):
    """:func:`sponge_states` with a per-unit role (IV): -> (cap_in (8, l), digests (n_units, 8))."""
    import torch
    dev = words.device
    el = torch.as_tensor(h2.pack_words_np(words.cpu().numpy(), word_bits), dtype=torch.int64, device=dev)   # (l, 16)
    ivs = torch.tensor([h2.IV(r, word_bits, n_words)[h2.RATE:] for r in (h2.ROLE_X, h2.ROLE_W)], dtype=torch.int64, device=dev)
    iv = ivs[roles.to(dev)].T                                                     # (8, n_units)
    cap_in = torch.zeros((h2.CAPACITY, l), dtype=torch.int64, device=dev)
    st = torch.zeros((h2.WIDTH, n_units), dtype=torch.int64, device=dev)
    st[h2.RATE:] = iv
    cols0 = torch.arange(n_units, device=dev) * steps
    for s in range(steps):
        cols = cols0 + s
        cap_in[:, cols] = (st[h2.RATE:] - iv) % P
        st = torch.cat([el[cols].T, st[h2.RATE:]], 0)
        st = permute_torch(st)
    return cap_in, st[:h2.OUT].T.contiguous()


# ---------------------------------------------------------------------------------------------------------------------
# Witness side: the extra hint rows and the digests
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Witness side: the extra hint rows and the digests
# ---------------------------------------------------------------------------------------------------------------------

def row_chain(hr: HashedRelation, rows, role: int, device) -> tuple:
    """The leaf chain of every committed row at once: ``rows`` (N, K) words -> (carry (N, steps, carry_elems): the linked
    rows of every step, digests (N, digest_elems)[, extras {hint name: (N, steps)}]).  The prover computes this once per
    committed set (``HashedRelationRunner.commit``) and gathers per sub-batch instead of re-running 2 x steps chains per
    proof.  The third item is present only when the scheme's ``chain_witness`` returns extra hint rows."""
    import torch
    rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)
    n, steps, k = rows.shape[0], hr.steps, hr.k
    words = rows.reshape(n * steps, k)
    fast = getattr(hr.leaf, "row_sponges", None) if ROW_SPONGE_KERNEL else None
    if fast is not None:
        # one thread per row, the whole sponge in registers (leaf/poseidon2.py row_sponge): the same states and digests
        res = fast(words, word_bits=hr.word_bits, role=role, row_words=steps * k, n=n, steps=steps)
        if res is not None:
            return res
    out = hr.leaf.chain_witness(words, word_bits=hr.word_bits, role=role, row_words=steps * k, n_vus=n, steps=steps, l=n * steps)
    cap, dg = out[0], out[1]
    res = (cap.T.reshape(n, steps, hr.leaf.carry_elems).contiguous(), dg)
    if len(out) > 2 and out[2]:
        res += ({nm: torch.as_tensor(v, dtype=torch.int64, device=device).reshape(n, steps) for nm, v in out[2].items()},)
    return res


row_sponges = row_chain     # the pre-registry name (Poseidon2's carried state is the sponge capacity)


def call_with_slot(fn, slot, *args, **kw):
    """``fn(*args, **kw)``, plus ``slot=slot.index`` when a pipeline slot is given and ``fn`` takes one (a relation whose
    device hint generator keeps per-stream static buffers: ``fp4/hashed.py``)."""
    import inspect
    if slot is not None and "slot" in inspect.signature(fn).parameters:
        kw["slot"] = slot.index
    return fn(*args, **kw)


def hash_hints(hr: HashedRelation, a, b, n_vus: int, l: int, device, sponges: dict | None = None, slot=None) -> tuple:
    """The appended hint rows (len(extra_hint_names), l) for operand words ``a``, ``b`` (l, k) int64 (pad columns 0), and
    the digests (n_vus, 2 x digest_elems) the statement publishes.  ``sponges``: ``{"a": (carry (n_vus, steps, ce),
    digests (n_vus, de)[, extras {name: (n_vus, steps)}]), "b": ...}`` precomputed by :func:`row_chain` and gathered for
    exactly these VUs' rows (the caller checked the words match).  ``slot``: the pipeline slot of this sub-batch,
    forwarded to the relation's decode (``call_with_slot``)."""
    import torch
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    nb, k, ce = hr.word_bits, hr.k, hr.leaf.carry_elems
    layout = hr.bit_layout
    shifts = torch.arange(nb, dtype=torch.int64, device=device)
    digests, bit_blocks, cap_blocks, extras = [], [], [], {}
    for _, nm, role in ROLES:
        w = a if nm == "a" else b
        if hr.fmt is not None:
            bit_blocks.append(hr.fmt.bit_rows_torch(w))
        else:
            bits = (w[:, :, None] >> shifts) & 1                      # (l, k, nb)
            bit_blocks.append(bits.permute(1, 2, 0).reshape(k * nb, l))   # rows nm[i].b{j}, i major
        if sponges is not None and nm in sponges:
            cap_v, dg = sponges[nm][0], sponges[nm][1]
            cap = torch.zeros((ce, l), dtype=torch.int64, device=device)
            cap[:, :n_vus * hr.steps] = cap_v.to(device).reshape(n_vus * hr.steps, ce).T
            dg = dg.to(device)
            if len(sponges[nm]) > 2:
                for n_, v in sponges[nm][2].items():
                    row = torch.zeros(l, dtype=torch.int64, device=device)
                    row[:n_vus * hr.steps] = v.to(device).reshape(-1)
                    extras[n_] = row
        else:
            out = hr.leaf.chain_witness(w, word_bits=nb, role=role, row_words=hr.steps * k, n_vus=n_vus, steps=hr.steps, l=l)
            cap, dg = out[0], out[1]
            if len(out) > 2 and out[2]:
                extras.update({n: torch.as_tensor(v, dtype=torch.int64, device=device) for n, v in out[2].items()})
        cap_blocks.append(cap)
        digests.append(dg)
    n_dec = len(hr.extra_hint_names) - len(ROLES) * (sum(layout) + ce)
    if not extras and _block_layout(hr, n_dec):
        # compose() appends the rows as [converted decode pins | a bits | b bits | a.cap | b.cap]: a handful of copies
        # instead of one per row (720 rows x 13 sub-batches was ~0.1 s per 4096 VUs)
        if hr.fmt is None:
            dec = private_decode(hr, a, b, hr.extra_hint_names[:n_dec])
        else:
            vals = call_with_slot(hr.rel.public_vectors, slot, hr.rel.params, None, a, b, None, device=device)
            dec = torch.stack([vals[n].to(device) for n in hr.extra_hint_names[:n_dec]])
        out = torch.cat([dec] + bit_blocks + cap_blocks)
    else:
        vals = call_with_slot(hr.rel.public_vectors, slot, hr.rel.params, None, a, b, None, device=device)
        for (_, nm, _), bits, cap in zip(ROLES, bit_blocks, cap_blocks):
            r = 0
            for i in range(k):
                for j in range(layout[i]):
                    vals[f"{nm}[{i}].b{j}"] = bits[r]
                    r += 1
            for x in range(ce):
                vals[f"{nm}.cap[{x}]"] = cap[x]
        vals.update(extras)
        try:
            out = torch.stack([vals[n].to(device) for n in hr.extra_hint_names])
        except KeyError as e:
            raise KeyError(f"leaf {hr.leaf.name}: no witness value for hint row {e.args[0]!r} (chain_witness must return it)") from None
    return out, torch.cat(digests, 1)


def _block_layout(hr: HashedRelation, n_dec: int) -> bool:
    """Whether ``hr.extra_hint_names`` is ``[decode rows | a bits | b bits | a.cap | b.cap]`` (memoised on ``hr``)."""
    if hr.block_layout is None:
        names = hr.extra_hint_names
        layout = hr.bit_layout
        tail = [f"{nm}[{i}].b{j}" for _, nm, _ in ROLES for i in range(hr.k) for j in range(layout[i])]
        tail += [f"{nm}.cap[{x}]" for _, nm, _ in ROLES for x in range(hr.leaf.carry_elems)]
        hr.block_layout = n_dec >= 0 and names[n_dec:] == tail and all("].b" not in n for n in names[:n_dec])
    return hr.block_layout


def hash_public_vectors(p, c, a, b, y, device="cpu", n_digest: int = N_DIGEST):
    """The verifier's public vectors of a hashed statement: ``a`` is the (l, 1 + n_digest) public table
    ``[is_end | d[0..n_digest)]`` (zeros off the chain ends), ``b`` the per-VU leaf indices (not per-column: only hashed
    into the statement).  ``HashedRelationRunner`` binds ``n_digest`` to its scheme's."""
    import torch
    t = torch.as_tensor(a, dtype=torch.int64, device=device)
    if t.ndim != 2 or t.shape[1] != 1 + n_digest:
        raise ValueError(f"hashed statement: public table must be (l, {1 + n_digest})")
    if ((t < 0) | (t >= P)).any() or ((t[:, 0] != 0) & (t[:, 0] != 1)).any():
        raise ValueError("hashed statement: public table out of range")
    out = {HASH_PIN_END: t[:, 0]}
    for x in range(n_digest):
        out[hash_pin(x)] = t[:, 1 + x]
    return out


def public_table(digests, n_vus: int, steps: int, l: int) -> np.ndarray:
    """(l, 1 + n_digest) int64: ``is_end`` and the digest lanes at every VU's last column, 0 elsewhere (``n_digest`` is
    read off ``digests``: (n_vus, n_digest))."""
    d = np.asarray(digests, dtype=np.int64).reshape(n_vus, -1)
    tab = np.zeros((l, 1 + d.shape[1]), dtype=np.int64)
    ends = np.arange(n_vus) * steps + steps - 1
    tab[ends, 0] = 1
    tab[ends, 1:] = d
    return tab
