"""B-Ligero at the checker-v2 shape, end to end (lane b-e2e-v2).  Torch, one Ligero column per unit, ``l`` units
(``l / 96`` VUs) per proof.  Transcript (``Proof``), sequential depth 3:

    stmt = H(domain | params | config | system id | n_vus, steps | claimed words)          (F1-F3: statement bound)
    P -> V   root1                      Merkle root of U1 = Enc(W1): circuit columns, product wires, multiplicities
    V -> P   z_T, beta_T (per table)    F_c, from H(stmt, root1); two slots per table, the first not in the public
                                        table image {t_v(beta_T)} is used (public pole redraw)
    P -> V   root2                      U2 = Enc(W2): lookup helpers h_j = 1/(z - w_j(beta)), table inverses h_v = m_v/(z - t_v(beta))
    V -> P   r, rho_lin, rho_sum, rho_quad, rc      D independent BabyBear combinations each, from H(stmt, root1, root2)
    P -> V   w = r W (D x l)            proximity (IRS) message over both tableaux
             v = s W  (D x l)           the per-table sums (constant coefficients s from rho_sum): sum_H v = 0
             h (D x (l-1))              quotient of the batched quadratic test by Z_H
             q (D x (2l-1))             the chain test's combined polynomial (state links, starts, ovf = 0, y16 = claimed)
    V -> P   cols                       t column indices from H(stmt, roots, w, v, h, q)
    P -> V   U1[:, cols], U2[:, cols], Merkle paths

Verifier: (i) Enc(w)[cols] = r U[cols]; (ii) linear asserts: (alpha U)[cols] = beta (constants); (iii) sums:
sum_H v = 0 and Enc(v)[cols] = s U[cols]; (iv) quadratic: F_t products, F_c helper triples with the public ACTIVE row,
table-side triples with the public coefficient rows (z - t_v(beta)) -- all at the opened columns -- equal h(eta) Z_H(eta);
(v) chain: l (q_0 + q_l) = r^T b and q(eta) = sum_i R_i(eta) U_i[eta]; (vi) Merkle paths; every size from the config.
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from verity_numerical.security import accounting as acc

from .. import merkle
from ..field import GEN, P, batch_modinv, encode, eval_on_coset, intt, modinv, zh_on_coset
from . import fc
from .system import STATE, Layout, V2System

__all__ = ["Config", "Proof", "config_for", "prove", "soundness", "verify"]

D = 6
DOMAIN = b"verity/b-ligero-v2/1"


@dataclass(frozen=True)
class Config:
    l: int
    n: int
    t: int
    target_log2: float
    steps: int = 96
    n_vus: int = 0                # VUs per proof (B_proof); 0 = as many as fit

    def __post_init__(self):
        if self.n_vus == 0:
            object.__setattr__(self, "n_vus", self.l // self.steps)
        if self.n_vus * self.steps > self.l:
            raise ValueError("layout does not fit")


def config_for(l: int, target_log2: float = -128.0, rate_log2: int = 2, steps: int = 96, n_vus: int = 0) -> Config:
    n = l << rate_log2
    t = acc.ligero_queries_for(target_log2, n, l, l)
    return Config(l, n, t, target_log2, steps, n_vus)


def soundness(cfg: Config, sysm: V2System, n_quadratic: int) -> dict:
    """Ligero Appendix-C terms (uniform combinations, k = l + 1 stands in for k = l) plus the logUp identity term per
    table (Haboeck Thm 4, conditioned challenge space, ``columns = C``), union-bounded."""
    Fb = acc.Field.prime("BabyBear", P)
    F = Fb.extension(D)
    lp = acc.LigeroParams(n=cfg.n, k=cfg.l + 1, l=cfg.l, t=cfg.t, zk=False)
    terms = {k: v for k, v in acc.ligero_error(lp, Fb, F, combination="uniform").items()}
    for name in sysm.table_order:
        td = sysm.tables[name]
        nq = sysm.n_lookups_of(name) * cfg.l
        terms[f"logup_{name}"] = acc.logup_identity_error(nq, td.size, F, "conditioned", columns=td.C)
    total = sum(terms.values())
    return {"terms_log2": {k: acc.log2(v) for k, v in terms.items()}, "total_log2": acc.log2(total), "field": F.name,
            "t": cfg.t, "n": cfg.n, "l": cfg.l, "n_quadratic_constraints": n_quadratic,
            "note": "uniform independent BabyBear coefficients per coordinate (D = 6); k = l + 1 stands in for the "
                    "non-ZK k = l in the accounting's validate(); the linear asserts and sums send at most the "
                    "Appendix-C linear query term; logUp per table with the public pole-redraw rule"}


# ---------------------------------------------------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------------------------------------------------

def statement_digest(sysm: V2System, cfg: Config, y16: np.ndarray) -> bytes:
    """F1-F3: everything the verifier holds is absorbed before any challenge.  ``y16``: (n_vus,) claimed words."""
    h = hashlib.sha256()
    h.update(DOMAIN)
    h.update(repr(sysm.p).encode())
    h.update(struct.pack("<QQQQQ", cfg.l, cfg.n, cfg.t, cfg.steps, len(y16)))
    h.update(struct.pack("<d", cfg.target_log2))
    h.update(sysm.system_id)
    h.update(np.asarray(y16, dtype="<u2").tobytes())
    return h.digest()


def _expand(seed: bytes, count: int, device) -> torch.Tensor:
    raw = hashlib.shake_256(seed).digest(4 * count)
    arr = np.frombuffer(raw, dtype="<u4").astype(np.int64) % P
    return torch.from_numpy(arr).to(device)


def _cols_from(seed: bytes, t: int, n: int) -> list[int]:
    cols: list[int] = []
    seen = set()
    ctr = 0
    while len(cols) < t:
        raw = hashlib.shake_256(seed + ctr.to_bytes(4, "little")).digest(4 * 2 * t)
        for v in np.frombuffer(raw, dtype="<u4"):
            c = int(v) % n
            if c not in seen:
                seen.add(c)
                cols.append(c)
                if len(cols) == t:
                    break
        ctr += 1
    return cols


@dataclass
class Proof:
    root1: bytes
    root2: bytes
    w: np.ndarray            # (D, l)
    v: np.ndarray            # (D, l)
    h: np.ndarray            # (D, l-1)
    q: np.ndarray            # (D, 2l-1)
    cols: list[int]
    opened1: np.ndarray      # (m1, t)
    opened2: np.ndarray      # (m2, t)
    paths1: list
    paths2: list
    hash_name: str = merkle.HASH_NAME
    timings: dict = field(default_factory=dict)

    def n_bytes(self) -> dict:
        b = {"roots": len(self.root1) + len(self.root2), "w": self.w.nbytes, "v": self.v.nbytes, "h": self.h.nbytes,
             "q": self.q.nbytes, "cols": 4 * len(self.cols), "opened_columns": self.opened1.nbytes + self.opened2.nbytes,
             "merkle_paths": sum(32 * len(p) for p in self.paths1) + sum(32 * len(p) for p in self.paths2)}
        b["total"] = sum(b.values())
        return b


# ---------------------------------------------------------------------------------------------------------------------
# the public data of a proof: tables at beta, coefficient rows, active masks
# ---------------------------------------------------------------------------------------------------------------------

class Public:
    """Everything both parties compute from the system, the config and the round-1 challenges."""

    def __init__(self, sysm: V2System, cfg: Config, lay: Layout, device):
        self.sysm, self.cfg, self.lay, self.device = sysm, cfg, lay, device
        l = cfg.l
        self.active = torch.zeros((l,), dtype=torch.int64, device=device)
        self.active[: cfg.n_vus * cfg.steps] = 1
        self.tables_dev = {name: torch.from_numpy(sysm.tables[name].entries).to(device) for name in sysm.table_order}
        # ACT_T rows: (rows_T, l) with entry v = r l + pos active iff v < size
        self.act = {}
        for name in sysm.table_order:
            size, rows = sysm.tables[name].size, lay.mult_rows[name]
            a = torch.zeros((rows * l,), dtype=torch.int64, device=device)
            a[:size] = 1
            self.act[name] = a.view(rows, l)

    def challenges1(self, stmt: bytes, root1: bytes) -> dict:
        """``z_T, beta_T`` per table (two z slots; the first outside the public image is used).  Also the compressed
        table values ``tval[T]`` ((rows, l, 6): z - t_v(beta), 0 on pads) and ``t_at_beta``."""
        out = {}
        for name in self.sysm.table_order:
            seed = merkle.hash_bytes(b"c1|" + stmt + root1 + name.encode())
            x = _expand(seed, 3 * D, self.device).view(3, D)
            beta = x[0]
            tb = self.tables_dev[name]                                     # (size, C)
            pw = fc.fc_pow_seq(beta, tb.shape[1])                          # (C, 6)
            tval = fc.fc_sum((tb[:, :, None] * pw[None, :, :]) % P, 1)     # (size, 6)
            z = None
            for slot in (1, 2):
                cand = x[slot]
                hit = bool((tval == cand[None, :]).all(1).any().item()) or not bool(cand.any().item())   # image + the pad tuple 0
                if not hit:
                    z = cand
                    break
            if z is None:
                raise ValueError(f"both z slots of {name} lie in the table image: public abort")
            rows, l = self.lay.mult_rows[name], self.cfg.l
            full = torch.zeros((rows * l, D), dtype=torch.int64, device=self.device)
            full[: tb.shape[0]] = (z[None, :] - tval) % P
            out[name] = {"z": z, "beta": beta, "beta_pow": pw, "coef": full.view(rows, l, D)}
        return out

    def public_rows(self, ch: dict) -> torch.Tensor:
        """The public position-varying rows (message form, (n_pub, l)): ACTIVE, then per table ACT_T (rows) and the
        coefficient rows z - t_v(beta) (6 x rows, coordinate-major)."""
        rows = [self.active[None, :]]
        self.pub_idx = {"ACTIVE": 0}
        r = 1
        for name in self.sysm.table_order:
            nr = self.lay.mult_rows[name]
            self.pub_idx[("ACT", name)] = r
            rows.append(self.act[name])
            r += nr
            self.pub_idx[("COEF", name)] = r
            rows.append(ch[name]["coef"].permute(2, 0, 1).reshape(D * nr, self.cfg.l))
            r += D * nr
        return torch.cat(rows, 0)


# ---------------------------------------------------------------------------------------------------------------------
# constraint evaluation on an evaluation matrix (codeword domain for the prover, opened columns for the verifier)
# ---------------------------------------------------------------------------------------------------------------------

def _lin_eval(lin: tuple, U: torch.Tensor, ncols: int) -> torch.Tensor:
    coef, c = lin
    out = torch.full((ncols,), c % P, dtype=torch.int64, device=U.device)
    if coef:
        idx = torch.tensor(list(coef.keys()), dtype=torch.int64, device=U.device)
        cf = torch.tensor(list(coef.values()), dtype=torch.int64, device=U.device)
        out = (out + ((U.index_select(0, idx) * cf[:, None]) % P).sum(0)) % P
    return out


def _lins_eval(lins: list, U: torch.Tensor) -> torch.Tensor:
    """(len(lins), cols) by one index_add over all terms."""
    ncols = U.shape[1]
    qs, js, cs, consts = [], [], [], []
    for q, (coef, c) in enumerate(lins):
        for j, v in coef.items():
            qs.append(q)
            js.append(j)
            cs.append(v)
        consts.append(c % P)
    dev = U.device
    out = torch.tensor(consts, dtype=torch.int64, device=dev)[:, None].repeat(1, ncols)
    if js:
        jj = torch.tensor(js, dtype=torch.int64, device=dev)
        cc = torch.tensor(cs, dtype=torch.int64, device=dev)
        qq = torch.tensor(qs, dtype=torch.int64, device=dev)
        out.index_add_(0, qq, (U.index_select(0, jj) * cc[:, None]) % P)
    return out % P


_FC_PAIRS = [[(i, (c - i) % D, 1 if i <= c else fc.NR) for i in range(D)] for c in range(D)]
# coordinate c of x * y over F_p[X]/(X^6 - 31): sum_{i + k = c} x_i y_k + 31 sum_{i + k = c + 6} x_i y_k


def _fc_conv(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """``X, Y`` (rows, 6, cols) -> (rows, 6, cols): coordinate-wise product over F_c."""
    out = []
    for c in range(D):
        s = None
        for i, k, wgt in _FC_PAIRS[c]:
            v = (X[:, i] * Y[:, k]) % P
            if wgt != 1:
                v = (v * wgt) % P
            s = v if s is None else s + v
        out.append(s % P)
    return torch.stack(out, 1)


class Constraints:
    """The constraint system of one proof as evaluators on any ``(m + n_pub, cols)`` evaluation matrix."""

    def __init__(self, sysm: V2System, cfg: Config, lay: Layout, pub: Public, ch: dict, device):
        self.sysm, self.cfg, self.lay, self.pub, self.ch, self.device = sysm, cfg, lay, pub, ch, device
        self.m = lay.m
        # linear asserts (per column, constant rhs)
        self.lin = [(coef, c) for _, coef, c in sysm.asserts]
        # F_t products
        self.prod_a = [a for _, a, _, _ in sysm.prods]
        self.prod_b = [b for _, _, b, _ in sysm.prods]
        self.prod_out = torch.tensor([o for _, _, _, o in sysm.prods], dtype=torch.int64, device=device)
        # lookup helper triples: Y_{j,k} = z_k - sum_q beta^q_k L_q  as Lins
        self.ylins = []
        for tname, lins in sysm.lookups:
            z, pw = ch[tname]["z"].tolist(), ch[tname]["beta_pow"].tolist()
            for k in range(D):
                coef: dict[int, int] = {}
                c = z[k]
                for q, (lc, lconst) in enumerate(lins):
                    bq = pw[q][k]
                    c = (c - bq * lconst) % P
                    for r, v in lc.items():
                        coef[r] = (coef.get(r, 0) - bq * v) % P
                self.ylins.append(({r: v for r, v in coef.items() if v}, c))
        nL = len(sysm.lookups)
        self.h_rows = torch.tensor([lay.helper_row(j, c) for j in range(nL) for c in range(D)], dtype=torch.int64, device=device)
        # table side: rows of h_v (6, rows_T) and of the public coefficient rows, per table
        self.tab = []
        for name in sysm.table_order:
            nr = lay.mult_rows[name]
            hv = torch.tensor([[lay.hv_row(name, c, r) for r in range(nr)] for c in range(D)], dtype=torch.int64, device=device)
            cf = self.m + pub.pub_idx[("COEF", name)] + torch.arange(D * nr, device=device).view(D, nr)
            mrow = torch.tensor([lay.mult_row(name, r) for r in range(nr)], dtype=torch.int64, device=device)
            act = self.m + pub.pub_idx[("ACT", name)] + torch.arange(nr, device=device)
            self.tab.append((name, nr, hv, cf, mrow, act))
        self.active_row = self.m + pub.pub_idx["ACTIVE"]
        self.Q = len(sysm.prods) + D * nL + sum(D * nr for _, nr, *_ in self.tab)
        self.L = len(self.lin)
        # the per-table sums: 6 x |tables| functionals with constant coefficients
        self.n_sum = D * len(sysm.table_order)

    # -- quadratic: T (Q, cols) streamed into P0 = rho @ T -----------------------------------------------------------
    def quadratic_terms(self, U: torch.Tensor, sink, chunk_rows: int = 96) -> None:
        """Streams the constraint evaluations ``T`` ((Qc, cols) chunks, in constraint order) into ``sink(T, label)``."""
        cols = U.shape[1]
        if self.prod_a:
            A = _lins_eval(self.prod_a, U)
            B = _lins_eval(self.prod_b, U)
            sink(((A * B) % P - U.index_select(0, self.prod_out)) % P, "prod")
        nL = len(self.sysm.lookups)
        if nL:
            Y = _lins_eval(self.ylins, U).view(nL, D, cols)
            H = U.index_select(0, self.h_rows).view(nL, D, cols)
            T = _fc_conv(H, Y)                                                    # (nL, 6, cols)
            T[:, 0] = (T[:, 0] - U[self.active_row][None, :]) % P
            sink(T.reshape(nL * D, cols), "lookup")
        for name, nr, hv, cf, mrow, act in self.tab:
            for r0 in range(0, nr, chunk_rows):
                r1 = min(nr, r0 + chunk_rows)
                Hv = U.index_select(0, hv[:, r0:r1].reshape(-1)).view(D, r1 - r0, cols).permute(1, 0, 2)
                Cf = U.index_select(0, cf[:, r0:r1].reshape(-1)).view(D, r1 - r0, cols).permute(1, 0, 2)
                T = _fc_conv(Hv, Cf)                                              # (rows, 6, cols)
                rhs = (U.index_select(0, mrow[r0:r1]) * U.index_select(0, act[r0:r1])) % P
                T[:, 0] = (T[:, 0] - rhs) % P
                sink(T.reshape((r1 - r0) * D, cols), f"table {name} rows {r0}..{r1}")

    def quadratic_combine(self, U: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """``rho`` (D, Q) -> ``P0`` (D, cols) = sum_q rho[:, q] T_q(U)."""
        cols = U.shape[1]
        P0 = torch.zeros((D, cols), dtype=torch.int64, device=U.device)
        state = {"q0": 0}

        def add(T, _label):
            Qc = T.shape[0]
            rr = rho[:, state["q0"]:state["q0"] + Qc]
            for d in range(D):
                P0[d] = (P0[d] + ((rr[d][:, None] * T) % P).sum(0)) % P
            state["q0"] += Qc

        self.quadratic_terms(U, add)
        assert state["q0"] == self.Q
        return P0

    # -- linear asserts ------------------------------------------------------------------------------------------------
    def alpha_beta(self, rho_lin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``alpha`` (D, m): the combined coefficient per row; ``beta`` (D,): the combined constant (negated)."""
        alpha = torch.zeros((D, self.m), dtype=torch.int64, device=self.device)
        beta = torch.zeros((D,), dtype=torch.int64, device=self.device)
        js, qs, cs, consts = [], [], [], []
        for q, (coef, c) in enumerate(self.lin):
            for j, v in coef.items():
                qs.append(q)
                js.append(j)
                cs.append(v)
            consts.append(c % P)
        dev = self.device
        qq, jj, cc = (torch.tensor(x, dtype=torch.int64, device=dev) for x in (qs, js, cs))
        terms = (rho_lin[:, qq] * cc[None, :]) % P
        alpha.index_add_(1, jj, terms)
        cst = torch.tensor(consts, dtype=torch.int64, device=dev)
        beta = ((rho_lin * ((-cst) % P)[None, :]) % P).sum(1) % P
        return alpha % P, beta

    # -- sums ----------------------------------------------------------------------------------------------------------
    def sum_coefs(self, rho_sum: torch.Tensor) -> torch.Tensor:
        """``s`` (D, m): +rho[e] on helper coordinate rows of table e's queries, -rho[e] on its h_v rows."""
        s = torch.zeros((D, self.m), dtype=torch.int64, device=self.device)
        tix = {name: i for i, name in enumerate(self.sysm.table_order)}
        for j, (tname, _) in enumerate(self.sysm.lookups):
            for c in range(D):
                s[:, self.lay.helper_row(j, c)] = rho_sum[:, D * tix[tname] + c]
        for name, nr, hv, *_ in self.tab:
            for c in range(D):
                s[:, hv[c]] = (-rho_sum[:, D * tix[name] + c])[:, None] % P
        return s

    # -- chain -----------------------------------------------------------------------------------------------------------
    def chain_coefs(self, rc: torch.Tensor, y16_pub: torch.Tensor):
        """Families: 4 links (x.*[j+1] = y.*[j]), 4 starts (x.* = zero state at VU starts and pads), G ovf = 0 on
        non-last units, 1 end (y16 = claimed).  Returns (rows (R,), coef (D, R, l), rhs (D,))."""
        sysm, cfg = self.sysm, self.cfg
        l, steps, dev = cfg.l, cfg.steps, self.device
        j = torch.arange(l, device=dev)
        in_vu = j < steps * cfg.n_vus
        i = j % steps
        start = ((~in_vu) | (i == 0)).to(torch.int64)
        link = (in_vu & (i != steps - 1)).to(torch.int64)
        end = (in_vu & (i == steps - 1)).to(torch.int64)
        z0 = [0, sysm.p.floor, 0, 1]
        rows = [sysm.col_idx[f"x.{n}"] for n in STATE] + [sysm.col_idx[f"y.{n}"] for n in STATE] + sysm.col_idx["__ovf"] + [sysm.col_idx["y16"]]
        pos = {r: k for k, r in enumerate(rows)}
        coef = torch.zeros((D, len(rows), l), dtype=torch.int64, device=dev)
        rhs = torch.zeros((D,), dtype=torch.int64, device=dev)
        fam = 0
        for x in range(4):
            xi, yi = pos[rows[x]], pos[rows[4 + x]]
            r_link = (rc[:, fam, :] * link[None, :]) % P
            fam += 1
            coef[:, xi, 1:] = (coef[:, xi, 1:] + r_link[:, :-1]) % P
            coef[:, yi, :] = (coef[:, yi, :] + r_link * (P - 1)) % P
            r_start = (rc[:, fam, :] * start[None, :]) % P
            fam += 1
            coef[:, xi, :] = (coef[:, xi, :] + r_start) % P
            rhs = (rhs + (r_start.sum(1) % P) * (z0[x] % P)) % P
        for g in range(len(sysm.col_idx["__ovf"])):
            oi = pos[rows[8 + g]]
            r_o = (rc[:, fam, :] * link[None, :]) % P
            fam += 1
            coef[:, oi, :] = (coef[:, oi, :] + r_o) % P
        yi = pos[rows[-1]]
        r_end = (rc[:, fam, :] * end[None, :]) % P
        fam += 1
        coef[:, yi, :] = (coef[:, yi, :] + r_end) % P
        yv = torch.zeros((l,), dtype=torch.int64, device=dev)
        yv[steps - 1: steps * cfg.n_vus: steps] = y16_pub.to(dev) % P
        rhs = (rhs + ((r_end * yv[None, :]) % P).sum(1)) % P
        assert fam == self.n_families
        return torch.tensor(rows, dtype=torch.int64, device=dev), coef % P, rhs % P

    @property
    def n_families(self) -> int:
        return 4 + 4 + len(self.sysm.col_idx["__ovf"]) + 1


def _challenges2(stmt: bytes, root1: bytes, root2: bytes, cons: Constraints, device):
    m, L, Q, S, l = cons.m, cons.L, cons.Q, cons.n_sum, cons.cfg.l
    F = cons.n_families
    seed = merkle.hash_bytes(b"c2|" + stmt + root1 + root2)
    x = _expand(seed, D * (m + L + S + Q + F * l), device).view(D, -1)
    o = 0
    r = x[:, o:o + m]; o += m
    rho_lin = x[:, o:o + L]; o += L
    rho_sum = x[:, o:o + S]; o += S
    rho_quad = x[:, o:o + Q]; o += Q
    rc = x[:, o:o + F * l].reshape(D, F, l)
    return r, rho_lin, rho_sum, rho_quad, rc


def _challenge3(stmt: bytes, root1: bytes, root2: bytes, w, v, h, q, t: int, n: int) -> list[int]:
    seed = merkle.hash_bytes(b"c3|" + stmt + root1 + root2 + w.tobytes() + v.tobytes() + h.tobytes() + q.tobytes())
    return _cols_from(seed, t, n)


# ---------------------------------------------------------------------------------------------------------------------
# prover
# ---------------------------------------------------------------------------------------------------------------------

def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class _Clock:
    def __init__(self, device):
        self.device = device
        self.marks: dict[str, float] = {}
        _sync(device)
        self.t0 = self.last = time.perf_counter()

    def lap(self, name):
        _sync(self.device)
        now = time.perf_counter()
        self.marks[name] = self.marks.get(name, 0.0) + now - self.last
        self.last = now

    def total(self):
        _sync(self.device)
        return time.perf_counter() - self.t0


_GINV: dict = {}


def _ginv_powers(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _GINV:
        gi = modinv(GEN)
        pw = [1]
        for _ in range(n - 1):
            pw.append(pw[-1] * gi % P)
        _GINV[key] = torch.tensor(pw, dtype=torch.int64, device=device)
    return _GINV[key]


def _commit(U: torch.Tensor, device):
    """Merkle root over the columns of U (GPU Blake3 when available; host fallback).  Returns (root, opener, name)."""
    if merkle.gpu_available() and str(device).startswith("cuda"):
        U32 = U.to(torch.int32).contiguous()
        root, tree = merkle.commit_device(U32)

        def opener(cols):
            ct = torch.tensor(cols, dtype=torch.int64, device=device)
            return U32.index_select(1, ct).cpu().numpy().astype(np.uint32), merkle.open_device(tree, cols)
        return root, opener, merkle.GPU_HASH_NAME
    U_cols = U.t().contiguous().cpu().numpy().astype(np.uint32)
    root, levels = merkle.commit(U_cols)

    def opener(cols):
        return U_cols[cols].T.copy(), merkle.open_columns(levels, cols)
    return root, opener, merkle.HASH_NAME


def build_tableau1(sysm: V2System, cfg: Config, lay: Layout, units, device) -> torch.Tensor:
    """W1 (m1, l) from the units (a list of column-value lists or a ``UnitArrays``; pads filled here): columns,
    product wires, multiplicities."""
    from .system import UnitArrays, dummy_unit_vals, unit_arrays

    l = cfg.l
    if not isinstance(units, UnitArrays):
        units = unit_arrays(sysm, units)
    n_active = units.n
    assert n_active <= l
    pad = unit_arrays(sysm, [sysm.pad if sysm.pad is not None else dummy_unit_vals(sysm)])
    cols_np = np.concatenate((units.cols, np.repeat(pad.cols, l - n_active, 0)), 0).T
    prods_np = np.concatenate((units.prods, np.repeat(pad.prods, l - n_active, 0)), 0).T
    W1 = torch.zeros((lay.m1, l), dtype=torch.int64, device=device)
    W1[:lay.n_cols] = torch.from_numpy(np.ascontiguousarray(cols_np)).to(device)
    W1[lay.n_cols:lay.n_cols + lay.n_prods] = torch.from_numpy(np.ascontiguousarray(prods_np)).to(device)
    # multiplicities: keys of the active units' queries (integer keys, before reduction) -> entry indices
    # the key Lin's coefficients are stored mod p, so the key is its residue; the signed representative in
    # (-p/2, p/2) is the integer key for every table (all keys are small)
    counts_of = {name: np.zeros((sysm.tables[name].size,), dtype=np.int64) for name in sysm.table_order}
    for j, (tname, lins) in enumerate(sysm.lookups):
        td = sysm.tables[tname]
        kk = units.keys[j] % P
        kk = np.where(kk > P // 2, kk - P, kk)
        idx = kk - td.lo
        ok = (idx >= 0) & (idx < len(td.index_of_key))
        ent = np.where(ok, td.index_of_key[np.clip(idx, 0, len(td.index_of_key) - 1)], -1)
        ent = ent[ent >= 0]
        counts_of[tname] += np.bincount(ent, minlength=td.size)          # every query into the table accumulates
    for tname in sysm.table_order:
        base, nr = lay.mult_base[tname], lay.mult_rows[tname]
        full = np.zeros((nr * l,), dtype=np.int64)
        full[:sysm.tables[tname].size] = counts_of[tname]
        W1[base:base + nr] = torch.from_numpy(full.reshape(nr, l)).to(device)
    return W1


def build_tableau2(sysm: V2System, cfg: Config, lay: Layout, W1: torch.Tensor, pub: Public, ch: dict, device) -> torch.Tensor:
    """W2 (m2, l): lookup helpers (active positions only) and table-side inverses."""
    l = cfg.l
    W2 = torch.zeros((lay.m2, l), dtype=torch.int64, device=device)
    # helpers: h_j = ACTIVE / (z - w_j(beta))
    for j, (tname, lins) in enumerate(sysm.lookups):
        vals = _lins_eval(lins, W1)                                           # (C, l)
        pw = ch[tname]["beta_pow"]                                            # (C, 6)
        wj = fc.fc_sum((vals[:, :, None] * pw[:, None, :]) % P, 0)           # (l, 6)
        den = (ch[tname]["z"][None, :] - wj) % P
        h = fc.fc_inv(den)
        h = (h * pub.active[:, None]) % P
        W2[lay.helper_row(j, 0) - lay.m1: lay.helper_row(j, 0) - lay.m1 + D] = h.t()
    # table side: h_v = m_v / (z - t_v(beta)), 0 on pads
    for name in sysm.table_order:
        nr = lay.mult_rows[name]
        coef = ch[name]["coef"].view(nr * l, D)
        m = W1[lay.mult_base[name]:lay.mult_base[name] + nr].reshape(-1)
        inv = fc.fc_inv(coef)
        hv = fc.fc_scalar_mul(m, inv)                                          # (nr l, 6); pads: coef = 0 -> inv 0
        base = lay.hv_base[name] - lay.m1
        W2[base:base + D * nr] = hv.view(nr, l, D).permute(2, 0, 1).reshape(D * nr, l)
    return W2


def prove(sysm: V2System, cfg: Config, units: list[list[int]], y16_pub: np.ndarray, device, check: bool = False,
          mutate=None) -> Proof:
    """``units``: the column values of ``n_vus * steps`` units (VU-major); ``y16_pub``: (n_vus,) the claimed words.
    ``mutate(W1)``: test hook editing tableau 1 before it is committed (never used by an honest prover)."""
    clk = _Clock(device)
    l, n, t = cfg.l, cfg.n, cfg.t
    lay = sysm.layout(l)
    n_units = units.n if hasattr(units, "n") else len(units)
    assert n_units == cfg.n_vus * cfg.steps and len(y16_pub) == cfg.n_vus
    stmt = statement_digest(sysm, cfg, y16_pub)
    W1 = build_tableau1(sysm, cfg, lay, units, device)
    if mutate is not None:
        mutate(W1)
    clk.lap("witness")
    U1 = encode(W1, n)
    clk.lap("encode1")
    root1, open1, hash_name = _commit(U1, device)
    clk.lap("merkle1")
    pub = Public(sysm, cfg, lay, device)
    ch = pub.challenges1(stmt, root1)
    Pm = pub.public_rows(ch)
    clk.lap("challenges1")
    W2 = build_tableau2(sysm, cfg, lay, W1, pub, ch, device)
    clk.lap("helpers")
    U2 = encode(W2, n)
    clk.lap("encode2")
    root2, open2, _ = _commit(U2, device)
    clk.lap("merkle2")
    cons = Constraints(sysm, cfg, lay, pub, ch, device)
    r, rho_lin, rho_sum, rho_quad, rc = _challenges2(stmt, root1, root2, cons, device)
    W = torch.cat((W1, W2), 0)
    if check:
        bad = self_check(cons, W, Pm, device)
        if bad:
            raise ValueError("witness violates constraints: " + "; ".join(bad[:8]))
        clk.lap("self_check")
    # IRS + sums on the message side
    w = torch.stack([((r[d][:, None] * W) % P).sum(0) % P for d in range(D)])
    s = cons.sum_coefs(rho_sum)
    v = torch.stack([((s[d][:, None] * W) % P).sum(0) % P for d in range(D)])
    clk.lap("irs_sum")
    # quadratic test on the codeword domain
    Uall = torch.cat((U1, U2, encode(Pm, n)), 0)
    P0 = cons.quadratic_combine(Uall, rho_quad)
    zinv = batch_modinv(zh_on_coset(n, l, device))
    hcoef = (intt((P0 * zinv) % P) * _ginv_powers(n, device)) % P
    if check:
        assert (hcoef[:, l - 1:] == 0).all(), "quotient degree too high"
    h = hcoef[:, :l - 1]
    clk.lap("quadratic")
    # chain test
    rows, coef, rhs = cons.chain_coefs(rc, torch.as_tensor(np.asarray(y16_pub, dtype=np.int64)))
    R = coef.shape[1]
    Renc = encode(coef.reshape(D * R, l), n).reshape(D, R, n)
    Qev = ((Renc * Uall.index_select(0, rows)[None, :, :]) % P).sum(1) % P
    qcoef = (intt(Qev) * _ginv_powers(n, device)) % P
    if check:
        assert (qcoef[:, 2 * l - 1:] == 0).all(), "chain polynomial degree too high"
        assert torch.equal((l * (qcoef[:, 0] + qcoef[:, l])) % P, rhs), "chain constraints violated"
    q = qcoef[:, :2 * l - 1]
    clk.lap("chain")
    w_np, v_np, h_np, q_np = (x.cpu().numpy().astype(np.uint32) for x in (w, v, h, q))
    cols = _challenge3(stmt, root1, root2, w_np, v_np, h_np, q_np, t, n)
    opened1, paths1 = open1(cols)
    opened2, paths2 = open2(cols)
    clk.lap("openings")
    timings = dict(clk.marks)
    timings["total"] = clk.total()
    timings["peak_device_bytes"] = torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0
    return Proof(root1, root2, w_np, v_np, h_np, q_np, cols, opened1, opened2, paths1, paths2, hash_name, timings)


def self_check(cons: Constraints, W: torch.Tensor, Pm: torch.Tensor, device) -> list[str]:
    """The prover's own check of every constraint on the message side (W, public rows)."""
    bad = []
    Wall = torch.cat((W, Pm), 0)
    lin = _lins_eval(cons.lin, Wall)
    for q in torch.nonzero(lin.any(dim=1)).flatten().tolist():
        bad.append(f"linear {cons.sysm.asserts[q][0]}")
    names = [f"prod {n}" for n, *_ in cons.sysm.prods] + [f"lookup {j}:{t}.c{c}" for j, (t, _) in enumerate(cons.sysm.lookups) for c in range(D)]
    state = {"q0": 0}

    def sink(T, label):
        fails = torch.nonzero(T.any(dim=1)).flatten().tolist()
        for q in fails[:5]:
            qa = state["q0"] + q
            bad.append(f"quadratic {names[qa] if qa < len(names) else label + ' #' + str(q)}")
        state["q0"] += T.shape[0]

    cons.quadratic_terms(Wall, sink)
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify(sysm: V2System, cfg: Config, y16_pub, proof: Proof, device) -> tuple[bool, str]:
    """Accept/reject the claim that the ``n_vus`` VUs cast to ``y16_pub`` -- every size from ``cfg``, never from the proof."""
    t0 = time.perf_counter()
    l, n, t = cfg.l, cfg.n, cfg.t
    lay = sysm.layout(l)
    y16_pub = np.asarray(y16_pub, dtype=np.int64)
    if len(y16_pub) != cfg.n_vus or ((y16_pub < 0) | (y16_pub >= 2**sysm.p.word_bits)).any():
        return False, "claimed words: count or range"
    if proof.w.shape != (D, l) or proof.v.shape != (D, l) or proof.h.shape != (D, l - 1) or proof.q.shape != (D, 2 * l - 1):
        return False, "malformed messages"
    if proof.opened1.shape != (lay.m1, t) or proof.opened2.shape != (lay.m2, t) or len(proof.cols) != t:
        return False, "malformed openings"
    for arr in (proof.w, proof.v, proof.h, proof.q, proof.opened1, proof.opened2):
        if (arr.astype(np.int64) >= P).any():
            return False, "transcript element out of the field"
    if len(set(proof.cols)) != t or any(not 0 <= c < n for c in proof.cols):
        return False, "bad column set"
    stmt = statement_digest(sysm, cfg, y16_pub)
    if proof.cols != _challenge3(stmt, proof.root1, proof.root2, proof.w, proof.v, proof.h, proof.q, t, n):
        return False, "column challenge mismatch"
    # Merkle
    for k, c in enumerate(proof.cols):
        if not merkle.verify_path(proof.root1, c, proof.opened1[:, k], proof.paths1[k]):
            return False, f"merkle path {k} (tableau 1) invalid"
        if not merkle.verify_path(proof.root2, c, proof.opened2[:, k], proof.paths2[k]):
            return False, f"merkle path {k} (tableau 2) invalid"
    pub = Public(sysm, cfg, lay, device)
    try:
        ch = pub.challenges1(stmt, proof.root1)
    except ValueError as e:
        return False, str(e)
    Pm = pub.public_rows(ch)
    cons = Constraints(sysm, cfg, lay, pub, ch, device)
    r, rho_lin, rho_sum, rho_quad, rc = _challenges2(stmt, proof.root1, proof.root2, cons, device)
    cols = torch.tensor(proof.cols, dtype=torch.int64, device=device)
    Uo = torch.cat((torch.from_numpy(proof.opened1.astype(np.int64)), torch.from_numpy(proof.opened2.astype(np.int64))), 0).to(device)
    Pmo = encode(Pm, n)[:, cols]
    Uall = torch.cat((Uo, Pmo), 0)                                              # (m + n_pub, t)
    # (i) proximity
    w = torch.from_numpy(proof.w.astype(np.int64)).to(device)
    lhs = torch.stack([((r[d][:, None] * Uo) % P).sum(0) % P for d in range(D)])
    if not torch.equal(encode(w, n)[:, cols], lhs):
        return False, "proximity test failed"
    # (ii) linear asserts
    alpha, beta = cons.alpha_beta(rho_lin)
    lhs = torch.stack([((alpha[d][:, None] * Uo) % P).sum(0) % P for d in range(D)])
    if not torch.equal(lhs, beta[:, None].expand(D, t)):
        return False, "linear constraints failed"
    # (iii) sums
    v = torch.from_numpy(proof.v.astype(np.int64)).to(device)
    if (v.sum(1) % P != 0).any():
        return False, "lookup sum identity failed"
    s = cons.sum_coefs(rho_sum)
    lhs = torch.stack([((s[d][:, None] * Uo) % P).sum(0) % P for d in range(D)])
    if not torch.equal(encode(v, n)[:, cols], lhs):
        return False, "lookup sum test failed at the opened columns"
    # (iv) quadratic
    P0 = cons.quadratic_combine(Uall, rho_quad)
    h = torch.from_numpy(proof.h.astype(np.int64)).to(device)
    rhs = (eval_on_coset(h, n)[:, cols] * zh_on_coset(n, l, device)[cols]) % P
    if not torch.equal(P0, rhs):
        return False, "quadratic constraints failed"
    # (v) chain
    rows, coef, rhs_c = cons.chain_coefs(rc, torch.as_tensor(y16_pub))
    q = torch.from_numpy(proof.q.astype(np.int64)).to(device)
    if not torch.equal((l * (q[:, 0] + q[:, l])) % P, rhs_c):
        return False, "chain constraints failed (sum over H)"
    R = coef.shape[1]
    Rc = encode(coef.reshape(D * R, l), n).reshape(D, R, n)[:, :, cols]
    expect = ((Rc * Uall.index_select(0, rows)[None, :, :]) % P).sum(1) % P
    if not torch.equal(eval_on_coset(q, n)[:, cols], expect):
        return False, "chain constraints failed (opened columns)"
    _sync(device)
    return True, f"accept ({time.perf_counter() - t0:.3f}s)"
