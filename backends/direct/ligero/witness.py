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
        y = u[3] if len(u) > 3 and sys.chain is not None else None      # the epilogue rows exist in chain mode only
        flat = flatten_hints(p, gen_step(p, c, a, b), y, wide=sys.wide, unit=(c, a, b))
        for n, v in flat.items():
            out[pos[n], col] = v
    return out.to(device)


_CONST_CACHE: dict = {}


def _const(key, make):
    t = _CONST_CACHE.get(key)
    if t is None:
        t = make()
        _CONST_CACHE[key] = t
    return t


def _run_program(sys: System, W: torch.Tensor, pub_rows: torch.Tensor, hints: torch.Tensor, device) -> None:
    """The derived rows of W in place.  ``pub_rows``: (n_pins, l) the pinned values in program order (mod p);
    ``hints``: (n_hints, l).  Device-only work (graph-capturable): every constant is cached on first use."""
    hpos = {n: i for i, (_, n) in enumerate(sys.hints)}
    ppos = {n: i for i, (_, n) in enumerate(sys.pins)}
    dk = str(device)
    for op in sys.program:
        kind = op[0]
        if kind == "pin":
            _, idx, name = op
            W[idx] = pub_rows[ppos[name]]
        elif kind == "hint":
            _, idx, name = op
            W[idx] = hints[hpos[name]] % P
        elif kind == "prod":
            _, idx, a, b = op
            W[idx] = (eval_expr(a, W) * eval_expr(b, W)) % P
        elif kind == "bits":
            _, row0, nbits, e = op
            x = eval_expr(e, W)                      # honest: an integer in [0, 2^nbits)
            shifts = _const(("shifts", nbits, dk), lambda nb=nbits: torch.arange(nb, dtype=torch.int64, device=device)[:, None])
            W[row0:row0 + nbits] = (x[None, :] >> shifts) & 1
        elif kind == "sels":
            _, row0, k0s, k1s, key = op
            x = _signed(eval_expr(key, W))
            k0 = _const(("k0", id(op), dk), lambda v=k0s: torch.tensor(v, dtype=torch.int64, device=device)[:, None])
            k1 = _const(("k1", id(op), dk), lambda v=k1s: torch.tensor(v, dtype=torch.int64, device=device)[:, None])
            W[row0:row0 + len(k0s)] = ((x[None, :] >= k0) & (x[None, :] <= k1)).to(torch.int64)
        elif kind == "inv":
            from .field import batch_modinv
            _, idx, x = op
            W[idx] = batch_modinv(eval_expr(x, W))
        else:
            raise ValueError(kind)


class _Graph:
    """CUDA-graph replay of the witness program for one (system, l, device): the ~1000 small kernels are launched
    once at capture and replayed afterwards (the program is a fixed function of the system)."""

    def __init__(self, sys: System, l: int, device):
        self.W = torch.zeros((sys.m, l), dtype=torch.int64, device=device)
        self.pub = torch.zeros((len(sys.pins), l), dtype=torch.int64, device=device)
        self.hints = torch.zeros((len(sys.hints), l), dtype=torch.int64, device=device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                                   # warm-up: populate the constant caches
                _run_program(sys, self.W, self.pub, self.hints, device)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            _run_program(sys, self.W, self.pub, self.hints, device)

    def run(self, pub_rows: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        self.pub.copy_(pub_rows)
        self.hints.copy_(hints)
        self.graph.replay()
        return self.W.clone()


_GRAPHS: dict = {}
USE_CUDA_GRAPH = True

#: ``"device"``: the program as one fused CUDA kernel (witness_device.py; W int32, bit-identical values), the
#: default on CUDA; ``"legacy"``: the torch program (CUDA graph when capturable, else ~4000 eager kernels).
IMPL = "device"


def build_witness(sys: System, pub: dict[str, torch.Tensor], hints: torch.Tensor, device) -> torch.Tensor:
    """W (m, l) in [0, p): int32 from the fused kernel (``IMPL == "device"`` on CUDA), int64 from the torch program."""
    l = hints.shape[1]
    if IMPL == "device" and str(device).startswith("cuda") and sys.pins:
        from .witness_device import fused_for
        f = fused_for(sys, device)
        if f is not None:
            pub_rows = torch.stack([pub[name].to(device) for _, name in sys.pins]) % P   # one gather + one reduction
            return f.run(pub_rows, hints)
    pub_rows = torch.stack([pub[name].to(device) % P for _, name in sys.pins]) if sys.pins else \
        torch.zeros((0, l), dtype=torch.int64, device=device)
    if USE_CUDA_GRAPH and str(device).startswith("cuda"):
        key = (id(sys), l, str(device))
        g = _GRAPHS.get(key)
        if g is None:
            try:
                g = _Graph(sys, l, device)
            except Exception:  # noqa: BLE001 -- capture unsupported here: eager
                g = False
            _GRAPHS[key] = g
        if g:
            return g.run(pub_rows, hints.to(device))
    W = torch.zeros((sys.m, l), dtype=torch.int64, device=device)
    _run_program(sys, W, pub_rows, hints.to(device), device)
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
    tb = Tables(len(sys.linear), len(sys.quadratic), lq, lj, lc, lconst, [c.pub for c in sys.linear], quad)
    # the split evaluation (quad_p0): booleanity constraints W_j^2 = W_j are 93% of the quadratic constraints and
    # need no sparse tables -- one square per row; the rest go through apply_sparse on a Q_gen-row subset
    bool_q, bool_j, gen_q = [], [], []
    for q, qc in enumerate(sys.quadratic):
        a, b, c = qc.a, qc.b, qc.c
        if (a.c % P == 0 and b.c % P == 0 and c.c % P == 0 and len(a.lin) == len(b.lin) == len(c.lin) == 1):
            (ja, ka), = a.lin.items()
            (jb, kb), = b.lin.items()
            (jc, kc), = c.lin.items()
            if ja == jb == jc and ka % P == 1 and kb % P == 1 and kc % P == 1:
                bool_q.append(q)
                bool_j.append(ja)
                continue
        gen_q.append(q)
    gen = {}
    for side in ("a", "b", "c"):
        exprs = [getattr(sys.quadratic[q], side) for q in gen_q]
        gen[side] = _triples(exprs, device)
    t = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
    quad["split"] = {"bool_q": t(bool_q), "bool_j": t(bool_j), "gen_q": t(gen_q), "gen": gen, "Q_gen": len(gen_q)}
    return tb


USE_FUSED = True    # tests_fused.py kernels on CUDA (one pass); False = the torch.compile passes below


def _bool_row(c: torch.Tensor, Ub: torch.Tensor) -> torch.Tensor:
    """sum_j c_j (Ub_j^2 - Ub_j) mod p over the rows of Ub (compiled: one fused pass)."""
    t = (Ub * Ub - Ub) % P
    return ((c[:, None] * t) % P).sum(0) % P


def quad_p0(tb: Tables, rho: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(D, cols) = sum_q rho[d, q] (A_q(M) B_q(M) - C_q(M)) mod p, evaluated column-wise on M (m, cols) -- the batched
    quadratic test's polynomial.  Split form: booleanity constraints directly from the squared rows (no Q x cols
    intermediates), the general constraints through the sparse tables of their subset.  Equal to the naive
    apply_sparse form (checked by ``tests_bench.py`` and the gates)."""
    sp = tb.quad["split"]
    D = rho.shape[0]
    rb = rho.index_select(1, sp["bool_q"])
    if USE_FUSED:
        from . import tests_fused
        f = tests_fused.fused_for(D, M.device)
        if f is not None:                                         # one pass over the rows for all D combinations
            if "bool_j32" not in sp:
                sp["bool_j32"] = sp["bool_j"].to(torch.int32).contiguous()
            out = f.lincomb(rb, M, square_minus=True, rows=sp["bool_j32"])    # the rows read in place (no copy)
            if sp["Q_gen"]:
                if "csr" not in sp:
                    sp["csr"] = tests_fused.csr_tables(sp["gen"], sp["Q_gen"], M.device)
                out = (out + f.quad_general(rho.index_select(1, sp["gen_q"]), M, sp["csr"])) % P
            return out
    out = torch.empty((D, M.shape[1]), dtype=torch.int64, device=M.device)
    Ub = M.index_select(0, sp["bool_j"]).to(torch.int64)          # M may be the kernel's int32 codeword
    row = compiled(_bool_row, M.device)
    for d in range(D):
        out[d] = row(rb[d], Ub)
    del Ub
    if sp["Q_gen"]:
        g = sp["gen"]
        A = apply_sparse(g["a"][:3], g["a"][3], sp["Q_gen"], M)
        B = apply_sparse(g["b"][:3], g["b"][3], sp["Q_gen"], M)
        C = apply_sparse(g["c"][:3], g["c"][3], sp["Q_gen"], M)
        T = ((A * B) % P - C) % P
        del A, B, C
        rg = rho.index_select(1, sp["gen_q"])
        crow = compiled(_combine_row_w, M.device)
        for d in range(D):
            out[d] = (out[d] + crow(rg[d], T)) % P
    return out


def _combine_row_w(c: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return ((c[:, None] * M) % P).sum(0) % P


def apply_sparse(triples, consts: torch.Tensor, n_out: int, M: torch.Tensor) -> torch.Tensor:
    """(n_out, cols) = sum_j coef * M[j] + const for each output row (M: (m, cols) mod p)."""
    qq, jj, cc = triples
    out = torch.zeros((n_out, M.shape[1]), dtype=torch.int64, device=M.device)
    if jj.numel():
        terms = compiled(_scaled_rows, M.device)(M, jj, cc)
        out.index_add_(0, qq, terms)
    return (out + consts[:, None]) % P


def _scaled_rows(M: torch.Tensor, jj: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    return (M.index_select(0, jj).to(torch.int64) * cc[:, None]) % P      # M may be int32 (the SIMT codeword)


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
