"""The witness matrix W (m rows x l columns, one column per unit) from the public vectors and the checker's hints,
by running ``System.program`` in torch; and the constraint tables the protocol needs as tensors."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from verity_numerical.checker.params import Params
from verity_numerical.checker.step import gen_step

from .compile import Expr, P, System
from .field import compiled
from .unit import flatten_hints

__all__ = ["Tables", "build_witness", "check_constraints", "eval_expr", "hints_tensor", "tables"]


_EXPR_CACHE: dict = {}


def eval_expr(e: Expr, W: torch.Tensor) -> torch.Tensor:
    """(l,) values of an affine expression on the witness (mod p)."""
    if not e.lin:
        return torch.full((W.shape[1],), e.c % P, dtype=torch.int64, device=W.device)
    key = (id(e), str(W.device))
    hit = _EXPR_CACHE.get(key)
    if hit is None or hit[0] is not e:
        idx = torch.tensor(list(e.lin.keys()), dtype=torch.int64, device=W.device)
        coef = torch.tensor([v % P for v in e.lin.values()], dtype=torch.int64, device=W.device)
        hit = (e, idx, coef)
        _EXPR_CACHE[key] = hit
    _, idx, coef = hit
    acc = ((W.index_select(0, idx) * coef[:, None]) % P).sum(0)
    return (acc + e.c) % P


def _signed(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > P // 2, x - P, x)


def hints_tensor(p: Params, sys: System, units: list[tuple[int, list[int], list[int]]], device) -> torch.Tensor:
    """(n_hints, l) from ``gen_step`` on every unit (host Python; the reference hint generator).  A unit is
    ``(c, a, b)`` or, in chain mode, ``(c, a, b, y)`` with ``y`` the unit's output word (for the epilogue rows)."""
    names = [n for _, n in sys.hints]
    pos = {n: i for i, n in enumerate(names)}
    out = torch.zeros((len(names), len(units)), dtype=torch.int64)
    for col, u in enumerate(units):
        c, a, b = u[:3]
        flat = flatten_hints(p, gen_step(p, c, a, b), u[3] if len(u) > 3 else None)
        for n, v in flat.items():
            out[pos[n], col] = v
    return out.to(device)


def build_witness(sys: System, pub: dict[str, torch.Tensor], hints: torch.Tensor, device) -> torch.Tensor:
    """W (m, l) int64 in [0, p)."""
    l = hints.shape[1]
    W = torch.zeros((sys.m, l), dtype=torch.int64, device=device)
    hpos = {n: i for i, (_, n) in enumerate(sys.hints)}
    for op in sys.program:
        kind = op[0]
        if kind == "pin":
            _, idx, name = op
            W[idx] = pub[name].to(device) % P
        elif kind == "hint":
            _, idx, name = op
            W[idx] = hints[hpos[name]] % P
        elif kind == "prod":
            _, idx, a, b = op
            W[idx] = (eval_expr(a, W) * eval_expr(b, W)) % P
        elif kind == "bits":
            _, row0, nbits, e = op
            x = eval_expr(e, W)                      # honest: an integer in [0, 2^nbits)
            shifts = torch.arange(nbits, dtype=torch.int64, device=device)
            W[row0:row0 + nbits] = (x[None, :] >> shifts[:, None]) & 1
        elif kind == "sels":
            _, row0, k0s, k1s, key = op
            x = _signed(eval_expr(key, W))
            k0 = torch.tensor(k0s, dtype=torch.int64, device=device)[:, None]
            k1 = torch.tensor(k1s, dtype=torch.int64, device=device)[:, None]
            W[row0:row0 + len(k0s)] = ((x[None, :] >= k0) & (x[None, :] <= k1)).to(torch.int64)
        elif kind == "inv":
            from .field import batch_modinv
            _, idx, x = op
            W[idx] = batch_modinv(eval_expr(x, W))
        else:
            raise ValueError(kind)
    return W


@dataclass
class Tables:
    """Sparse constraint tables as index/coefficient tensors.

    linear: (L constraints)  rows ``lin_q, lin_j, lin_coef`` (triples), ``lin_c`` (L,), ``lin_pub`` list[str|None]
    quadratic: for each side s in (a, b, c): triples ``q_{s}_q, q_{s}_j, q_{s}_coef`` and constants ``q_{s}_c`` (Q,)
    """
    L: int
    Q: int
    lin_q: torch.Tensor
    lin_j: torch.Tensor
    lin_coef: torch.Tensor
    lin_c: torch.Tensor
    lin_pub: list
    quad: dict


def _triples(exprs: list[Expr], device):
    qs, js, cs = [], [], []
    consts = []
    for q, e in enumerate(exprs):
        for j, v in e.lin.items():
            qs.append(q)
            js.append(j)
            cs.append(v % P)
        consts.append(e.c % P)
    t = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
    return t(qs), t(js), t(cs), t(consts)


def tables(sys: System, device) -> Tables:
    lin_exprs = [Expr(None, c.lin, c.c) for c in sys.linear]
    lq, lj, lc, lconst = _triples(lin_exprs, device)
    quad = {}
    for side in ("a", "b", "c"):
        exprs = [getattr(q, side) for q in sys.quadratic]
        qq, qj, qc, qconst = _triples(exprs, device)
        quad[side] = (qq, qj, qc, qconst)
    return Tables(len(sys.linear), len(sys.quadratic), lq, lj, lc, lconst, [c.pub for c in sys.linear], quad)


def apply_sparse(triples, consts: torch.Tensor, n_out: int, M: torch.Tensor) -> torch.Tensor:
    """(n_out, cols) = sum_j coef * M[j] + const for each output row (M: (m, cols) mod p)."""
    qq, jj, cc = triples
    out = torch.zeros((n_out, M.shape[1]), dtype=torch.int64, device=M.device)
    if jj.numel():
        terms = compiled(_scaled_rows, M.device)(M, jj, cc)
        out.index_add_(0, qq, terms)
    return (out + consts[:, None]) % P


def _scaled_rows(M: torch.Tensor, jj: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    return (M.index_select(0, jj) * cc[:, None]) % P


def check_constraints(sys: System, tb: Tables, W: torch.Tensor, pub: dict[str, torch.Tensor]) -> list[str]:
    """Which constraints fail on W (the prover's own sanity check; the verifier never sees W)."""
    bad = []
    lin = apply_sparse((tb.lin_q, tb.lin_j, tb.lin_coef), tb.lin_c, tb.L, W)
    for q, name in enumerate(tb.lin_pub):
        rhs = pub[name].to(W.device) % P if name is not None else torch.zeros_like(lin[q])
        if not torch.equal(lin[q], rhs):
            bad.append(f"linear {sys.linear[q].name}")
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, W)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, W)
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, W)
    fail = ((A * B) % P != C).any(dim=1)
    for q in torch.nonzero(fail).flatten().tolist():
        bad.append(f"quadratic {sys.quadratic[q].name}")
    return bad
