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

import numpy as np

from verity.commitments import poseidon2_babybear as h2
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

    @property
    def steps(self) -> int:
        return self.rel.steps


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


def _sbox_gadget(ctx: LigeroCtx):
    def sbox(x: Expr) -> Expr:
        x2 = _prod(ctx, x, x)
        x3 = _prod(ctx, x2, x)
        x4 = _prod(ctx, x2, x2)
        return _prod(ctx, x4, x3)
    return sbox


def _format(rel: Relation) -> dict:
    p = rel.params
    if rel.word_dtype == "<u2":            # BF16: s | t(8) | f(7)
        return {"bits": 16, "frac": p.frac, "exp": p.exp, "bias": p.bias, "kind": "bf16", "sig": p.sig}
    if rel.word_dtype == "<u1":            # E4M3: s | t(4) | f(3)
        return {"bits": 8, "frac": p.op_frac, "exp": p.op_exp, "bias": p.op_bias, "kind": "e4m3", "sig": p.sig}
    raise ValueError(f"{rel.name}: no private decode for operand dtype {rel.word_dtype}")


def compose(rel: Relation) -> HashedRelation:
    """The hashed, private relation of ``rel`` (see the module doc).  Compiles ``rel`` afresh."""
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
    # --- 2. the sponges ------------------------------------------------------------------------------------------------
    sbox = _sbox_gadget(ctx)
    per = h2.LANE_BITS // nb
    if k != h2.RATE * per:
        raise ValueError(f"{rel.name}: k = {k} words of {nb} bits are not one rate block of 16 lanes")
    ends: list[Expr] = []
    for _, nm, role in ROLES:
        iv = h2.IV(role, nb, rel.steps * k)
        lanes = []
        for j in range(h2.RATE):
            e = Expr(ctx)
            for q in range(per):
                e = e + words[nm][per * j + q] * 2 ** (nb * q)
            lanes.append(_reduce(e))
        caps = [ctx.hint(f"{nm}.cap[{x}]") for x in range(h2.CAPACITY)]
        state = lanes + [_reduce(c + iv[h2.RATE + x]) for x, c in enumerate(caps)]
        out = h2.permute_ring(state, sbox, _reduce)
        sys.chain["c"] += [next(iter(c.lin)) for c in caps]
        sys.chain["y"] += [_reduce(out[h2.RATE + x] - iv[h2.RATE + x]) for x in range(h2.CAPACITY)]
        ends += out[:h2.OUT]
    m2 = sys.m
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
    extra = [n for _, n in sys.hints[h0:]]
    return HashedRelation(rel, sys, nb, k, h0, extra, census)


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


def sponge_states(words, word_bits: int, role: int, n_words: int, n_vus: int, steps: int, l: int):
    """``words`` (l, k) int64 torch tensor (pad columns 0) -> (cap_in (8, l): the linked rows ``cap - IV`` per column,
    digests (n_vus, 8)) by running the sponge of every VU along its ``steps`` columns, all VUs in lockstep."""
    import torch
    dev = words.device
    el = torch.as_tensor(h2.pack_words(words.cpu().numpy(), word_bits), dtype=torch.int64, device=dev)   # (l, 16)
    iv = h2.IV(role, word_bits, n_words)
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
    cap, dg = sponge_states(words, hr.word_bits, role, steps * k, n, steps, n * steps)
    return cap.T.reshape(n, steps, h2.CAPACITY).contiguous(), dg


def hash_hints(hr: HashedRelation, a, b, n_vus: int, l: int, device, sponges: dict | None = None) -> tuple:
    """The appended hint rows (len(extra_hint_names), l) for operand words ``a``, ``b`` (l, k) int64 (pad columns 0), and
    the digests (n_vus, 16) the statement publishes.  ``sponges``: ``{"a": (cap (n_vus, steps, 8), digests (n_vus, 8)),
    "b": ...}`` precomputed by :func:`row_sponges` for exactly these VUs' rows (the caller checked the words match)."""
    import torch
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    vals = hr.rel.public_vectors(hr.rel.params, None, a, b, None, device=device)
    nb, k = hr.word_bits, hr.k
    shifts = torch.arange(nb, dtype=torch.int64, device=device)
    digests = []
    for _, nm, role in ROLES:
        w = a if nm == "a" else b
        bits = (w[:, :, None] >> shifts) & 1                      # (l, k, nb)
        for i in range(k):
            for j in range(nb):
                vals[f"{nm}[{i}].b{j}"] = bits[:, i, j]
        if sponges is not None and nm in sponges:
            cap_v, dg = sponges[nm]
            cap = torch.zeros((h2.CAPACITY, l), dtype=torch.int64, device=device)
            cap[:, :n_vus * hr.steps] = cap_v.to(device).reshape(n_vus * hr.steps, h2.CAPACITY).T
            dg = dg.to(device)
        else:
            cap, dg = sponge_states(w, nb, role, hr.steps * k, n_vus, hr.steps, l)
        for x in range(h2.CAPACITY):
            vals[f"{nm}.cap[{x}]"] = cap[x]
        digests.append(dg)
    out = torch.stack([vals[n].to(device) for n in hr.extra_hint_names])
    return out, torch.cat(digests, 1)


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
