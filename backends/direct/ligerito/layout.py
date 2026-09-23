"""``compile.System`` + sub-batches -> one multilinear polynomial on the boolean hypercube (lane ligerito-sumcheck).

The committed polynomial ``w`` of a batch has ``N = 2^(n_i + n_s + n_j)`` cells at flat index

    x = i * (S * l) + s * l + j        i = row (HIGH bits), s = sub-batch, j = column within the sub-batch (LOW bits)

with variable ``t`` = bit ``t`` of ``x`` (LSB first): ``eq(r, x) = prod_t (r_t x_t + (1 - r_t)(1 - x_t))``.  Rows ``i < m`` are
the System's rows WITHOUT the ``pin`` rows (a pin is ``W_row = pub(j)``; the sumcheck references the public vector directly, so the
row is neither committed nor constrained); rows ``[m, 2^n_i)`` are zero in the commitment and hold the prover's *virtual rows*:

    const              1 on every column
    pub:<name>         Ligero's public per-column vectors (``Linear.pub``; the former pins are the same vectors)
    link/start/end     the chain masks of ``chain.Layout`` per sub-batch (0/1; pad sub-batches use the same layout with 0 VUs)
    y16                the claimed chain-end words (0 off the ends)
    next:<x>           the shifted rows  Wnext_x(s, j) = W(c_x, s, j + 1 mod l)  (the chain's column -> next-column coupling)

so that EVERY constraint is  A_k(z) * B_k(z) = C_k(z)  with ``A, B, C`` linear (scalar coefficients) in the column vector ``z_c``
over the ``2^n_i`` real+virtual rows: ``Quadratic`` rows as they are; ``Linear`` ``L(W) + c = pub`` as ``A = L + c - pub_row,
B = const, C = 0``; chain link ``A = link_row, B = next_x - Y_x(W)``; start ``A = start_row, B = W[c_x]``; end ``A = end_row,
B = Y16(W) - y16_row``.  Lookups (one-hot selectors, key selection) and range bits are ``Quadratic`` / ``Linear`` rows already.
``System.wide`` is a hint-partition record only (no constraint of its own).

Constraint slots ``k in [0, 2^n_k)``: quadratics, then the non-pin linears, then the ``2 n_links + 1`` chain constraints; unused
slots are all-zero.  The zero-check runs over ``(k, c)`` with ``c = s * l + j`` (``layout.py`` fixes the bit order everywhere).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from backends.direct.ligero.chain import Layout as ChainLayout
from backends.direct.ligero.compile import P, Expr, System

__all__ = ["Constraints", "Layout", "SubBatch", "build_z", "constraints", "layout_for", "public_rows"]


def _log2(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


@dataclass(frozen=True)
class Layout:
    m: int                      # committed rows (System rows minus pins)
    n_i: int                    # row bits: 2^n_i >= m + virtual rows
    l: int                      # columns per sub-batch (a power of two)
    n_j: int
    n_sub: int                  # real sub-batches
    n_s: int                    # sub-batch bits; S = 2^n_s >= n_sub (pad sub-batches: 0 VUs, all pad units)
    steps: int                  # units per VU (48 fp8, 96 bf16)
    n_vus: tuple                # VUs per real sub-batch
    row_of_sys: dict = field(hash=False, compare=False)   # System row idx -> committed row (pins absent)
    virt: dict = field(hash=False, compare=False)         # virtual row name -> row index in [m, 2^n_i)
    c_rows: tuple = ()          # committed row index of the linked ``c`` rows (sys.chain["c"])
    n_k: int = 0                # constraint slot bits
    K: int = 0                  # real constraints

    @property
    def S(self) -> int:
        return 1 << self.n_s

    @property
    def C(self) -> int:
        """columns of the whole batch = S * l"""
        return self.S * self.l

    @property
    def R(self) -> int:
        return 1 << self.n_i

    @property
    def n_c(self) -> int:
        return self.n_s + self.n_j

    @property
    def n_vars(self) -> int:
        return self.n_i + self.n_c

    @property
    def N(self) -> int:
        return self.R * self.C

    @property
    def n_links(self) -> int:
        return len(self.c_rows)

    def chain_layout(self, s: int) -> ChainLayout:
        return ChainLayout(self.l, self.n_vus[s] if s < self.n_sub else 0, self.steps)

    def summary(self) -> dict:
        return {"m": self.m, "n_i": self.n_i, "l": self.l, "n_j": self.n_j, "n_sub": self.n_sub, "S": self.S, "n_k": self.n_k,
                "K": self.K, "R": self.R, "C": self.C, "N_log2": self.n_vars, "virtual_rows": len(self.virt), "n_links": self.n_links}


def end_row_names(sys: System) -> list[str]:
    """The claimed chain-end rows: ``y16`` (one field element), or ``y_end0..`` for a claim wider than the field whose end
    binds components (``sys.chain["y_end"] = [(expr, shift, width), ...]``, e.g. fp4-nvf4's FP32 word: sign, exponent,
    fraction)."""
    if sys.chain.get("y16") is not None:
        return ["y16"]
    return [f"y_end{e}" for e in range(len(sys.chain["y_end"]))]


def end_values(sys: System, ypub: torch.Tensor) -> dict[str, torch.Tensor]:
    """The end rows' values from the claimed words ``ypub`` (int64; 0 off the ends): the word itself, or its components."""
    if sys.chain.get("y16") is not None:
        return {"y16": ypub}
    return {f"y_end{e}": (ypub.to(torch.int64) >> sh) & ((1 << wd) - 1) for e, (_, sh, wd) in enumerate(sys.chain["y_end"])}


def layout_for(sys: System, l: int, n_vus: list[int], steps: int) -> Layout:
    """The layout of a batch of ``len(n_vus)`` sub-batches of ``l`` columns (``n_vus[s]`` VUs of ``steps`` units each)."""
    if sys.chain is None or (sys.chain.get("y16") is None and not sys.chain.get("y_end")):
        raise NotImplementedError("layout_for: a chain-compiled system with a y16 end or component ends (y_end)")
    pins = {idx for idx, _ in sys.pins}
    row_of_sys: dict[int, int] = {}
    for r in sys.rows:
        if r.idx not in pins:
            row_of_sys[r.idx] = len(row_of_sys)
    m = len(row_of_sys)
    pub_names: list[str] = []
    for idx, name in sys.pins:
        if name not in pub_names:
            pub_names.append(name)
    for c in sys.linear:
        if c.pub is not None and c.pub not in pub_names:
            pub_names.append(c.pub)
    virt: dict[str, int] = {}
    nxt = m
    for name in ["const", "link", "start", "end"] + end_row_names(sys) + [f"pub:{n}" for n in pub_names] + \
            [f"next:{x}" for x in range(len(sys.chain["c"]))]:
        virt[name] = nxt
        nxt += 1
    n_i = max(1, (nxt - 1).bit_length())
    n_links = len(sys.chain["c"])
    n_pin_lin = sum(1 for c in sys.linear if c.name.startswith("pin:"))
    K = len(sys.quadratic) + len(sys.linear) - n_pin_lin + 2 * n_links + len(end_row_names(sys))
    n_k = max(1, (K - 1).bit_length())
    n_sub = len(n_vus)
    n_s = max(0, (n_sub - 1).bit_length())
    for s, nv in enumerate(n_vus):
        if nv * steps > l:
            raise ValueError(f"sub-batch {s}: {nv} VUs x {steps} > l = {l}")
    c_rows = tuple(row_of_sys[idx] for idx in sys.chain["c"])
    return Layout(m, n_i, l, _log2(l), n_sub, n_s, steps, tuple(n_vus), row_of_sys, virt, c_rows, n_k, K)


# ---------------------------------------------------------------------------------------------------------------------
# Constraint matrices
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Constraints:
    """``A, B, C`` as COO triples ``(k, row, coef)`` over ``2^n_k`` slots x ``2^n_i`` rows (coef mod p); ``names[k]``."""
    K_pad: int
    R: int
    coo: dict                   # side -> (k (nnz,), row (nnz,), coef (nnz,)) int64 tensors
    names: list

    def nnz(self) -> dict:
        return {s: int(t[0].numel()) for s, t in self.coo.items()}


def _map_expr(sys: System, lay: Layout, e: Expr, pin_name: dict) -> dict[int, int]:
    """An affine Expr over System rows -> {z row: coef} (pins -> their pub row; the constant -> the const row)."""
    out: dict[int, int] = {}
    for idx, v in e.lin.items():
        if idx in pin_name:
            r = lay.virt[f"pub:{pin_name[idx]}"]
        else:
            r = lay.row_of_sys[idx]
        out[r] = (out.get(r, 0) + v) % P
    if e.c % P:
        r = lay.virt["const"]
        out[r] = (out.get(r, 0) + e.c) % P
    return {r: v for r, v in out.items() if v}


def constraints(sys: System, lay: Layout, device="cpu") -> Constraints:
    pin_name = dict(sys.pins)
    rows: dict[str, list] = {s: [] for s in "abc"}
    names: list[str] = []

    def emit(name: str, a: dict, b: dict, c: dict):
        k = len(names)
        names.append(name)
        for side, d in zip("abc", (a, b, c)):
            for r, v in d.items():
                rows[side].append((k, r, v % P))

    for q in sys.quadratic:
        emit(q.name, _map_expr(sys, lay, q.a, pin_name), _map_expr(sys, lay, q.b, pin_name), _map_expr(sys, lay, q.c, pin_name))
    one = {lay.virt["const"]: 1}
    for c in sys.linear:
        if c.name.startswith("pin:"):
            continue
        a = _map_expr(sys, lay, Expr(None, c.lin, c.c), pin_name)
        if c.pub is not None:
            r = lay.virt[f"pub:{c.pub}"]
            a[r] = (a.get(r, 0) - 1) % P
            if not a[r]:
                del a[r]
        emit(c.name, a, one, {})
    ch = sys.chain
    for x, (c_idx, Y) in enumerate(zip(ch["c"], ch["y"])):
        b = _map_expr(sys, lay, Y, pin_name)
        b = {r: (-v) % P for r, v in b.items()}
        r = lay.virt[f"next:{x}"]
        b[r] = (b.get(r, 0) + 1) % P
        emit(f"chain.link{x}", {lay.virt["link"]: 1}, {r: v for r, v in b.items() if v}, {})
    for x, c_idx in enumerate(ch["c"]):
        emit(f"chain.start{x}", {lay.virt["start"]: 1}, {lay.row_of_sys[c_idx]: 1}, {})
    ends = [(ch["y16"], "y16", "chain.end")] if ch.get("y16") is not None else \
        [(Y, f"y_end{e}", f"chain.end{e}") for e, (Y, _, _) in enumerate(ch["y_end"])]
    for Y, name, cname in ends:
        b = _map_expr(sys, lay, Y, pin_name)
        r = lay.virt[name]
        b[r] = (b.get(r, 0) - 1) % P
        emit(cname, {lay.virt["end"]: 1}, {r: v for r, v in b.items() if v}, {})
    assert len(names) == lay.K, (len(names), lay.K)
    coo = {}
    for side in "abc":
        t = rows[side]
        ks = torch.tensor([k for k, _, _ in t], dtype=torch.int64, device=device)
        rs = torch.tensor([r for _, r, _ in t], dtype=torch.int64, device=device)
        vs = torch.tensor([v for _, _, v in t], dtype=torch.int64, device=device)
        coo[side] = (ks, rs, vs)
    return Constraints(1 << lay.n_k, lay.R, coo, names)


# ---------------------------------------------------------------------------------------------------------------------
# The z matrix (rows x C) from Ligero's per-sub-batch witness
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class SubBatch:
    """One sub-batch as Ligero's prover materialises it: ``W`` (sys.m, l) int (mod p) incl. the pin rows (``None`` on the
    verifier's side), ``pub`` name -> (l,) public vectors (``rel.public_vectors``), ``ypub`` (l,) claimed chain-end words
    (0 off the ends), ``n_vus``."""
    W: torch.Tensor | None
    pub: dict
    ypub: torch.Tensor
    n_vus: int

    def public(self) -> "SubBatch":
        return SubBatch(None, self.pub, self.ypub, self.n_vus)


def _masks(lay: Layout, s: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return lay.chain_layout(s).masks(device)


def build_z(sys: System, lay: Layout, subs: list[SubBatch], device, dtype=torch.int32) -> torch.Tensor:
    """``z`` (2^n_i, C) with the committed rows, the virtual rows and the shifted rows filled; ``subs`` has ``S`` entries
    (the caller supplies the pad sub-batches: trivial-unit witnesses, ``ypub = 0``, ``n_vus = 0``)."""
    if len(subs) != lay.S:
        raise ValueError(f"build_z: {len(subs)} sub-batches, layout has S = {lay.S}")
    z = torch.zeros((lay.R, lay.C), dtype=dtype, device=device)
    for s, sb in enumerate(subs):
        fill_z(sys, lay, z, s, sb)
    return z


def fill_z(sys: System, lay: Layout, z: torch.Tensor, s: int, sb: SubBatch) -> None:
    """Write sub-batch ``s``'s columns of ``z`` (``build_z`` one sub-batch at a time: the caller may drop ``sb.W`` after)."""
    device, dtype = z.device, z.dtype
    sys_rows = torch.tensor(sorted(lay.row_of_sys, key=lay.row_of_sys.get), dtype=torch.int64, device=device)
    W = sb.W.to(device)
    if W.shape != (sys.m, lay.l):
        raise ValueError(f"sub-batch {s}: W {tuple(W.shape)} != {(sys.m, lay.l)}")
    cols = slice(s * lay.l, (s + 1) * lay.l)
    z[: lay.m, cols] = (W.index_select(0, sys_rows) % P).to(dtype)
    z[lay.virt["const"], cols] = 1
    start, link, end = _masks(lay, s, device)
    z[lay.virt["link"], cols] = link.to(dtype)
    z[lay.virt["start"], cols] = start.to(dtype)
    z[lay.virt["end"], cols] = end.to(dtype)
    for name, v in end_values(sys, sb.ypub.to(device)).items():
        z[lay.virt[name], cols] = (v % P).to(dtype)
    for name, r in lay.virt.items():
        if name.startswith("pub:"):
            z[r, cols] = (sb.pub[name[4:]].to(device) % P).to(dtype)
    for x, cr in enumerate(lay.c_rows):
        z[lay.virt[f"next:{x}"], cols] = torch.roll(z[cr, cols], -1)      # Wnext(j) = W(c_x, j + 1 mod l)


def public_rows(sys: System, lay: Layout, subs: list[SubBatch], device) -> dict[str, torch.Tensor]:
    """The verifier's side of the virtual rows: name -> (C,) int64 for every virtual row EXCEPT ``next:*`` (those are the
    prover's claims, proved by the shift sumcheck)."""
    out: dict[str, torch.Tensor] = {}
    for name, r in lay.virt.items():
        if name.startswith("next:"):
            continue
        v = torch.zeros((lay.C,), dtype=torch.int64, device=device)
        for s, sb in enumerate(subs):
            cols = slice(s * lay.l, (s + 1) * lay.l)
            if name == "const":
                v[cols] = 1
            elif name in ("link", "start", "end"):
                start, link, end = _masks(lay, s, device)
                v[cols] = {"link": link, "start": start, "end": end}[name]
            elif name == "y16" or name.startswith("y_end"):
                v[cols] = end_values(sys, sb.ypub.to(device))[name] % P
            else:
                v[cols] = sb.pub[name[4:]].to(device) % P
        out[name] = v
    return out
