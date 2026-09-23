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

from dataclasses import dataclass

import numpy as np

from . import poseidon2 as h2
from .compile import Expr, LigeroCtx, P, Quadratic, System
from .leaf import base as leaf_base
from .leaf import registry as leaf_registry
from .leaf.poseidon2 import (  # noqa: F401  (re-exported: hashchain_test / witness_device_test address them here)
    DEVICE_SPONGE,
    Poseidon2Witness,
    _Sym,
    permute_torch,
    sponge_kernel_source,
    sponge_states,
)
from .relations import Relation

__all__ = ["HASH_PIN_END", "N_DIGEST", "ROLES", "HashedRelation", "compose", "hash_hints", "hash_pin", "hash_public_vectors",
           "public_table", "row_chain", "row_sponges"]

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
        return leaf_registry.relation_name(self.rel.name, self.leaf.name)


def _reduce(e):
    if isinstance(e, Expr):
        return Expr(e.ctx, {k: v % P for k, v in e.lin.items()}, e.c % P)
    return e % P


def _format(rel: Relation) -> dict:
    p = rel.params
    if rel.word_dtype == "<u2":            # BF16: s | t(8) | f(7)
        return {"bits": 16, "frac": p.frac, "exp": p.exp, "bias": p.bias, "kind": "bf16", "sig": p.sig}
    if rel.word_dtype == "<u1":            # E4M3: s | t(4) | f(3)
        return {"bits": 8, "frac": p.op_frac, "exp": p.op_exp, "bias": p.op_bias, "kind": "e4m3", "sig": p.sig}
    raise ValueError(f"{rel.name}: no private decode for operand dtype {rel.word_dtype}")


def compose(rel: Relation, leaf: str | leaf_base.LeafScheme | None = None) -> HashedRelation:
    """The hashed, private relation of ``rel`` under the leaf scheme ``leaf`` (a registry name -- ``"poseidon2"`` /
    ``"hash"`` / ``None`` are the default -- or a scheme instance).  Compiles ``rel`` afresh."""
    scheme = leaf_registry.get(leaf) if leaf is None or isinstance(leaf, str) else leaf
    leaf_base.check_scheme(scheme)
    p = rel.params
    sys = rel.compile(p, True)
    fmt = _format(rel)
    k = rel.k
    nb, frac, exp_bits, bias = fmt["bits"], fmt["frac"], fmt["exp"], fmt["bias"]
    m0, l0, q0, h0 = sys.m, len(sys.linear), len(sys.quadratic), len(sys.hints)
    # --- 1. operand pins -> hint rows ------------------------------------------------------------------------------
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
    for nm in ("a", "b"):
        for i in range(k):
            tag = f"{nm}[{i}]"
            bits = [ctx.hint(f"{tag}.b{j}") for j in range(nb)]
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
    m1 = sys.m
    # --- 2. the leaf chains: the scheme's gadget per operand, its carried state as linked hint rows -------------------
    ends: list[Expr] = []
    row_words = rel.steps * k
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
    return HashedRelation(rel, sys, nb, k, h0, extra, census, scheme)


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
    out = hr.leaf.chain_witness(words, word_bits=hr.word_bits, role=role, row_words=steps * k, n_vus=n, steps=steps, l=n * steps)
    cap, dg = out[0], out[1]
    res = (cap.T.reshape(n, steps, hr.leaf.carry_elems).contiguous(), dg)
    if len(out) > 2 and out[2]:
        res += ({nm: torch.as_tensor(v, dtype=torch.int64, device=device).reshape(n, steps) for nm, v in out[2].items()},)
    return res


row_sponges = row_chain     # the pre-registry name (Poseidon2's carried state is the sponge capacity)


def hash_hints(hr: HashedRelation, a, b, n_vus: int, l: int, device, sponges: dict | None = None) -> tuple:
    """The appended hint rows (len(extra_hint_names), l) for operand words ``a``, ``b`` (l, k) int64 (pad columns 0), and
    the digests (n_vus, 2 x digest_elems) the statement publishes.  ``sponges``: ``{"a": (carry (n_vus, steps, ce),
    digests (n_vus, de)[, extras {name: (n_vus, steps)}]), "b": ...}`` precomputed by :func:`row_chain` and gathered for
    exactly these VUs' rows (the caller checked the words match)."""
    import torch
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    vals = hr.rel.public_vectors(hr.rel.params, None, a, b, None, device=device)
    nb, k, ce = hr.word_bits, hr.k, hr.leaf.carry_elems
    shifts = torch.arange(nb, dtype=torch.int64, device=device)
    digests, bit_blocks, cap_blocks, extras = [], [], [], {}
    for _, nm, role in ROLES:
        w = a if nm == "a" else b
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
    n_dec = len(hr.extra_hint_names) - len(ROLES) * (k * nb + ce)
    if not extras and _block_layout(hr, n_dec):
        # compose() appends the rows as [converted decode pins | a bits | b bits | a.cap | b.cap]: a handful of copies
        # instead of one per row (720 rows x 13 sub-batches was ~0.1 s per 4096 VUs)
        dec = torch.stack([vals[n].to(device) for n in hr.extra_hint_names[:n_dec]])
        out = torch.cat([dec] + bit_blocks + cap_blocks)
    else:
        for (_, nm, _), bits, cap in zip(ROLES, bit_blocks, cap_blocks):
            for i in range(k):
                for j in range(nb):
                    vals[f"{nm}[{i}].b{j}"] = bits[i * nb + j]
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
        tail = [f"{nm}[{i}].b{j}" for _, nm, _ in ROLES for i in range(hr.k) for j in range(hr.word_bits)]
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
