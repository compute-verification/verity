"""The chain test (PROTOCOL.md section "Chain"): Ligero's *general* linear test for the cross-column constraints of
the verification unit.  Columns j = 96 v + i hold unit i of VU v; the constraints

    link_j  : W[c.x][j + 1] - Y_x(W)[j] = 0     x in (s, t, f), for every j whose successor is in the same VU
    start_j : W[c.x][j] = 0                     for every j that starts a VU (and every pad column)
    end_j   : Y16(W)[j] = y16_pub[j]            for every j that ends a VU

are one linear system A w = b over the flattened witness.  The verifier draws one uniform coefficient per constraint
(D coordinates); r^T A is a coefficient vector per (row, column), interpolated per row to R_i(X) (degree < l); the
prover sends q(X) = sum_i R_i(X) p_i(X) (degree < 2l - 1); the verifier checks sum_{H} q = r^T b, i.e.
l (q_0 + q_l) = r^T b, and q(eta_c) = sum_i R_i(eta_c) U_i[c] at the opened columns.  Nothing is rotated or opened
beyond the t columns; the soundness term is the linear test's ((k + l)/n)^t with k = l (Appendix C).

Chain ends in components (``sys.chain["y_end"]``, the FP4 relation).  A public word of more than 30 bits cannot be
compared to a single field element (2^32 > p: the word w and w + p are the same residue), so a relation whose claimed
word is a full FP32 may end its chain on up to three bounded COMPONENTS instead: ``y_end = [(Y_x, lo_x, bits_x)]``,
one constraint ``Y_x(W)[j] = (y_pub[j] >> lo_x) & (2^bits_x - 1)`` per component at every end column j.  Each
component is a bit field of a 32-bit word of at most 30 bits (``check_component``: 1 <= bits <= 30, lo + bits <= 32;
2^30 < p = 15 * 2^27 + 1 < 2^31, so a 31-bit component in [p, 2^31) would alias w - p), hence an integer below p, so
equality mod p is integer equality once the relation range-constrains the row -- the same rule the Rust verifier's
parser enforces (``ligero-verify``: a component that is not such a field refuses the file).  The
coefficients: component 0 uses the end family (6) and components 1, 2 the link families (0, 1), which are unused at
end columns (an end column has no successor in its VU: the link mask is 0 there), so every constraint still has its
own uniform coefficient and nothing is sampled beyond the seven families.  Single-``y16`` systems (BF16, FP8) keep
their transcripts byte for byte.

More than three linked rows (``sys.chain["c"]`` / ``["y"]`` of length > 3: the hashed relations of ``hashchain.py`` carry
the two Poseidon2 sponges' 16 capacity lanes along the chain).  The verifier samples exactly ``N_FAMILIES = 7`` uniform
coefficients per column (``protocol._challenge1``); the extra constraints take *monomials* in them instead: extra
constraint ``e`` at column ``j`` (link extras first, then start extras, then end extras beyond the seven base families)
has coefficient ``u_{e mod 6} * u_6^(e div 6 + 1)`` where ``u_f = rc[f][j]``.  Every constraint at a column then has its
own monomial (the base ones are the degree-1 ``u_0 .. u_6``, the extras have ``u_6``-degree >= 1), so the combination
``r^T (A w - b)`` is a nonzero polynomial of degree <= 1 + ceil(E / 6) in the column's coefficients whenever ``A w != b``
(``E`` extras at that column) and vanishes with probability <= (1 + ceil(E / 6)) / p per challenge coordinate
(Schwartz-Zippel) instead of 1 / p -- for the 32 extras of a hashed chain start column: 7 / p, i.e. (7 / p)^D = 2^-169
at D = 6, below the field term ``(n + 3) / p^D`` the accounting already books.  Systems with three linked rows are
unchanged (no extras; byte-identical transcripts).

Fingerprint constraints (``sys.chain["fp"]``, lane share-logup: ``hashchain.compose_shared`` / ``row_hash_system``).  A
system whose operand rows are hashed ONCE per tile instead of once per VU binds every VU to its shared row by a
post-commitment fingerprint: with ``lanes[o][i]`` the 16 lane rows of operand slot ``o`` (the 16-bit words the sponge
absorbs) and ``rho_0 .. rho_{C-1}`` verifier coins drawn AFTER the root (``protocol._challenge1``, ``C = FP_COORDS``),
the constraint at the END column of VU ``v`` for every ``(o, x)`` is

    fp_{v,o,x} : sum_{s < steps} sum_{i < 16} rho_x^(16 s + i) lanes[o][i][steps v + s] = F[v][o][x]

-- linear in the witness with rho-dependent coefficients, so it costs no witness row: it is ``2 x C`` more constraints
per VU of the same general linear test, with the end column's monomials as their uniform coefficients (extra constraints
``2 n_extra_links + o C + x`` in the rule above).  ``F`` is a prover MESSAGE after the coin (``Proof.fp``; the "coin" mode,
the GEMM side) or a public input (the "public" mode, the hash side: the row unit proves the SAME fingerprint of the row
it hashes, and the verifier matches the two).  Two different rows agree on one coordinate with probability
<= (16 steps - 1) / p (a nonzero polynomial of degree < 16 steps in rho_x); C = 8 coordinates give (767 / p)^8 ~ 2^-171
for steps = 48 (``protocol.soundness`` books it as ``fingerprint_collision``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .compile import P, System

__all__ = ["FP_COORDS", "FP_COORDS_FS", "N_FAMILIES", "END_FAMILIES", "MAX_COMPONENT_BITS", "N_BASE_LINKS", "WORD_BITS", "ChainCoefs",
           "FpArgs", "Layout", "chain_coefs", "check_component", "end_terms", "extra_coef", "fp_lane_rows", "fp_values",
           "n_links", "rho_powers"]

N_FAMILIES = 7        # link x3, start x3, end
FP_COORDS = 8         # fingerprint coordinates, interactive mode (module comment): (767 / p)^8 ~ 2^-171 at 48 steps
FP_COORDS_FS = 10     # Fiat-Shamir mode: the term takes the 2^60 query factor (protocol.soundness): (767 / p)^10 ~ 2^-213
FP_LANES = 16         # lanes per operand slot per column (the sponge's rate)
END_FAMILIES = (6, 0, 1)   # coefficient family of end component x (see the module comment)
N_BASE_LINKS = 3      # linked rows with their own uniform link / start families; more take monomials (module comment)
WORD_BITS = 32             # the public word a component end is a bit field of
MAX_COMPONENT_BITS = 30    # 2^30 < p < 2^31: the widest bit field that is an integer below p (the Rust parser's cap)


def check_component(x: int, lo: int, bits: int) -> tuple[int, int]:
    """``(lo, bits)`` of end component ``x`` as ints, or ``ValueError`` unless it is a bit field of a 32-bit word that is
    an integer below p: ``1 <= bits <= 30``, ``0 <= lo``, ``lo + bits <= 32`` (the Rust verifier's rule, so both sides
    refuse the same files)."""
    lo, bits = int(lo), int(bits)
    if not (1 <= bits <= MAX_COMPONENT_BITS and 0 <= lo and lo + bits <= WORD_BITS):
        raise ValueError(f"chain end component {x}: bit field lo = {lo}, bits = {bits} is not a field of a "
                         f"{WORD_BITS}-bit word below p (1 <= bits <= {MAX_COMPONENT_BITS}, lo + bits <= {WORD_BITS})")
    return lo, bits

def n_links(ch: dict) -> int:
    c, y = ch["c"], ch["y"]
    if len(c) != len(y) or len(c) < N_BASE_LINKS:
        raise ValueError(f"a chain links {N_BASE_LINKS} or more rows ({len(c)} c rows, {len(y)} y expressions)")
    return len(c)


def extra_coef(rc: torch.Tensor, e: int) -> torch.Tensor:
    """The (D, l) coefficient of extra constraint ``e``: ``rc[e mod 6] * rc[6]^(e div 6 + 1)`` (module comment)."""
    f, k = e % 6, e // 6 + 1
    u6 = rc[:, 6, :]
    pw = u6
    for _ in range(k - 1):
        pw = (pw * u6) % P
    return (rc[:, f, :] * pw) % P


def end_terms(ch: dict) -> list[tuple[object, int, int, int]]:
    """The chain-end constraints of a compiled chain as ``(Expr, family, lo, bits)``: one ``(y16, 6, 0, 0)`` for a
    single-word end (``bits = 0``: the whole public word, compared as a residue), else one per component, each a
    valid bit field (``check_component``)."""
    if ch.get("y16") is not None:
        return [(ch["y16"], END_FAMILIES[0], 0, 0)]
    comps = ch["y_end"]
    if not 1 <= len(comps) <= len(END_FAMILIES):
        raise ValueError(f"a chain end has 1..{len(END_FAMILIES)} components, got {len(comps)}")
    return [(Y, END_FAMILIES[x], *check_component(x, lo, bits)) for x, (Y, lo, bits) in enumerate(comps)]


def end_values(ch: dict, y_pub: torch.Tensor) -> list[torch.Tensor]:
    """The public value each end constraint compares against, per column: ``y_pub`` (l,) as a residue for a
    single-word end, else the bit field ``(y_pub >> lo) & (2^bits - 1)`` of every component."""
    out = []
    for _, _, lo, bits in end_terms(ch):
        out.append(y_pub % P if bits == 0 else (y_pub >> lo) & (2 ** bits - 1))
    return out


@dataclass(frozen=True)
class Layout:
    """l columns; VU v occupies columns [steps v, steps (v + 1)); columns >= steps * n_vus are pad units (a start
    with no end: c_in = 0, operands 0, output unchecked)."""
    l: int
    n_vus: int
    steps: int

    def __post_init__(self):
        if self.steps * self.n_vus > self.l:
            raise ValueError("layout does not fit")

    def masks(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = torch.arange(self.l, device=device)
        in_vu = j < self.steps * self.n_vus
        i = j % self.steps
        start = (~in_vu) | (i == 0)
        end = in_vu & (i == self.steps - 1)
        link = in_vu & (i != self.steps - 1)                 # j + 1 is in the same VU
        return start.to(torch.int64), link.to(torch.int64), end.to(torch.int64)


@dataclass
class ChainCoefs:
    rows: torch.Tensor          # (R,) involved witness row indices
    coef: torch.Tensor          # (D, R, l) coefficient of W[rows[r]][j] in the combination
    rhs: torch.Tensor           # (D,) r^T b


@dataclass
class FpArgs:
    """The fingerprint constraints' public data (module comment): ``rho`` (C,) the coins, ``F`` (n_slots, n_op, C) the
    claimed fingerprints per VU slot of the layout (``n_slots = l // steps``; slots >= n_vus are pad and ignored)."""
    rho: torch.Tensor
    F: torch.Tensor


def fp_lane_rows(ch: dict) -> torch.Tensor | None:
    """(n_op * 16,) the lane rows of a fingerprinted chain (``ch["fp"]["lanes"]``, operand-major), else None."""
    fp = ch.get("fp")
    if fp is None:
        return None
    return torch.tensor([r for op in fp["lanes"] for r in op], dtype=torch.int64)


def rho_powers(rho: torch.Tensor, steps: int) -> torch.Tensor:
    """(C, steps, 16): ``rho_x^(16 s + i)`` by binary exponentiation on the device (no host sync: graph-capturable)."""
    C = rho.shape[0]
    e = torch.arange(steps * FP_LANES, dtype=torch.int64, device=rho.device)
    out = torch.ones((C, steps * FP_LANES), dtype=torch.int64, device=rho.device)
    base = rho % P
    for b in range(max(1, int(steps * FP_LANES - 1).bit_length())):
        bit = ((e >> b) & 1)[None, :]
        out = torch.where(bit == 1, (out * base[:, None]) % P, out)
        base = (base * base) % P
    return out.reshape(C, steps, FP_LANES)


def fp_values(lanes: torch.Tensor, rho: torch.Tensor, n_op: int, steps: int, l: int) -> torch.Tensor:
    """The honest fingerprints: ``lanes`` (n_op * 16, l) lane rows of the witness -> F (l // steps, n_op, C) with
    ``F[v][o][x] = sum_{s,i} rho_x^(16 s + i) lanes[o][i][steps v + s]``."""
    n_slots = l // steps
    C = rho.shape[0]
    pw = rho_powers(rho, steps)                                                        # (C, steps, 16)
    L = (lanes[:, :n_slots * steps] % P).reshape(n_op, FP_LANES, n_slots, steps)       # (o, i, v, s)
    # products < 2^62 reduced per term, then <= 768 terms of < 2^31: no overflow
    prod = (L[:, :, :, :, None] * pw.permute(2, 1, 0)[None, :, None, :, :]) % P        # (o, i, v, s, x)
    return (prod.sum((1, 3)) % P).permute(1, 0, 2).contiguous()                        # (v, o, x)


FUSED_FP = True     # the fingerprint lane coefficients through tests_fused.fp_lanes on CUDA (bit-exact; False: torch)


def _fused_tests(D: int, device):
    if not FUSED_FP or not str(device).startswith("cuda"):
        return None
    from . import tests_fused
    return tests_fused.fused_for(D, device)


def chain_coefs(sys: System, layout: Layout, rc: torch.Tensor, y16_pub: torch.Tensor, fp: FpArgs | None = None) -> ChainCoefs:
    """``rc``: (D, N_FAMILIES, l) uniform challenge coefficients (one per constraint; entries at masked-out
    positions are ignored); ``y16_pub``: (l,) the claimed public words at the chain ends (0 elsewhere); ``fp``: the
    fingerprint coins and values of a fingerprinted chain (``sys.chain["fp"]``; required then, refused otherwise)."""
    ch = sys.chain
    device = rc.device
    D, _, l = rc.shape
    tabs = _chain_tables(sys, layout, device)
    rows, T = tabs["rows"], tabs["terms"]
    nl = n_links(ch)
    if (ch.get("fp") is not None) != (fp is not None):
        raise ValueError("fingerprint data given for a chain without fingerprint constraints, or missing for one with them")
    # rm[:, v, :] = the coefficients of virtual family v masked to their constraint positions: v = 0..6 are the seven
    # families (link x3, start x3, end component 0), v = 6 + x (x >= 1) the end component x, which draws its
    # coefficients from END_FAMILIES[x] (a link family, unused at end columns) under the END mask; then, for chains with
    # more than three linked rows, the link extras and the start extras (monomials of the seven families, module comment).
    base = rc.index_select(1, tabs["famsel"])                                     # (D, V_base, l)
    if tabs["n_extra"]:
        extras = torch.stack([extra_coef(rc, e) for e in range(tabs["n_extra"])], 1)   # (D, n_extra, l)
        base = torch.cat([base, extras], 1)
    rm = (base * tabs["maskmat"][None, :, :]) % P                                # (D, V, l)
    # every (row, family, kappa) term of r^T A at once: coef[:, row, j] += kappa rc[f][j] (masked); the link terms'
    # W[c][j + 1] coefficient is the same sum shifted one column right.  Exact integer sums of < 2^31 residues (a few
    # hundred terms per entry: no overflow), reduced once -- the same residues as the per-term reduction, in ~10 kernels
    # instead of one per term (lane hostphase).
    coef = torch.zeros((D, len(rows), l), dtype=torch.int64, device=device)
    coef.index_add_(1, T["row"], (rm.index_select(1, T["fam"]) * T["kappa"][None, :, None]) % P)
    coef[:, :, 1:].index_add_(1, T["shift_row"], rm.index_select(1, tabs["link_fams"])[:, :, :-1])
    coef %= P
    rhs = torch.zeros((D,), dtype=torch.int64, device=device)
    link_fams = tabs["link_fams_list"]        # the host list (a .tolist() here is a device sync: not permitted inside the
    for x in range(nl):                        # pipelined prover's CUDA-graph capture of the tests stage; lane merge-val)
        Y = ch["y"][x]
        r_link = rm[:, link_fams[x], :]                                            # zero off the linked positions
        rhs = (rhs + (r_link.sum(1) % P) * (Y.c % P)) % P                          # b = +Y.c
    # end_j: sum_i kappa_i W[i][j] + Y.c = y_pub[j] (one constraint per end component)
    y_vals = end_values(ch, y16_pub.to(device))
    for x, ((Y, _, _, _), yv) in enumerate(zip(end_terms(ch), y_vals)):
        r_end = rm[:, 6 + x, :]
        rhs = (rhs + ((r_end * ((yv - Y.c) % P)[None, :]) % P).sum(1)) % P
    if fp is not None:
        # fp_{v,o,x} at the end column of VU v (module comment): coefficient rho_x^(16 s + i) u_{v,o,x} on lane row (o, i)
        # at column steps v + s, u the end column's monomial for extra constraint 2 n_extra_links + o C + x, masked to
        # the real VUs (a pad slot's "end column" carries no constraint); rhs += u_{v,o,x} F[v][o][x]
        n_op, C = len(ch["fp"]["lanes"]), int(ch["fp"]["coords"])
        steps, n_slots = layout.steps, l // layout.steps
        if tuple(fp.F.shape) != (n_slots, n_op, C) or fp.rho.shape != (C,):
            raise ValueError(f"fingerprint data: F {tuple(fp.F.shape)} != {(n_slots, n_op, C)} or rho {tuple(fp.rho.shape)} != {(C,)}")
        ends = torch.arange(n_slots, device=device) * steps + steps - 1
        real = (torch.arange(n_slots, device=device) < layout.n_vus).to(torch.int64)
        e0 = tabs["n_extra"]
        u = torch.stack([extra_coef(rc, e0 + e)[:, ends] for e in range(n_op * C)], 1)     # (D, n_op C, n_slots)
        u = (u * real[None, None, :]).reshape(D, n_op, C, n_slots) % P
        pw = rho_powers(fp.rho.to(device), steps)                                           # (C, steps, 16)
        lane_pos = tabs["lane_pos"]
        ft = _fused_tests(D, device)
        if ft is not None:
            ft.fp_lanes(u, pw, lane_pos, coef, steps)
        else:
            # coef_lane[d, o, i, v, s] = sum_x u[d, o, x, v] pw[x, s, i]: C products of < 2^62 reduced per term, C terms summed
            prod = (u[:, :, :, :, None, None] * pw[None, None, :, None, :, :]) % P            # (D, o, x, v, s, i)
            lane_coef = (prod.sum(2) % P).permute(0, 1, 4, 2, 3).reshape(D, n_op * FP_LANES, n_slots * steps)
            coef[:, lane_pos, :n_slots * steps] = (coef[:, lane_pos, :n_slots * steps] + lane_coef) % P
        F = fp.F.to(device) % P
        rhs = (rhs + ((u * F.permute(1, 2, 0)[None]) % P).sum((1, 2, 3))) % P
    return ChainCoefs(rows, coef, rhs)


_TABLES: dict = {}


def _chain_tables(sys: System, layout: Layout, device) -> dict:
    """The constant part of ``chain_coefs`` for (system, layout, device): the involved rows, the (row, virtual family,
    kappa) term table, the shifted link rows, the virtual-family -> rc-family selector and the (V, l) mask matrix."""
    key = (id(sys), layout.l, layout.n_vus, layout.steps, str(device))
    hit = _TABLES.get(key)
    if hit is not None and hit[0] is sys:
        return hit[1]
    ch = sys.chain
    ends = end_terms(ch)
    nl = n_links(ch)
    n_extra_links = nl - N_BASE_LINKS
    involved: dict[int, int] = {}
    for idx in ch["c"]:
        involved.setdefault(idx, len(involved))
    for e in ch["y"] + [Y for Y, _, _, _ in ends]:
        for idx in e.lin:
            involved.setdefault(idx, len(involved))
    lane_pos = []
    if ch.get("fp") is not None:
        for op in ch["fp"]["lanes"]:
            if len(op) != FP_LANES:
                raise ValueError(f"a fingerprinted operand slot has {FP_LANES} lane rows, got {len(op)}")
            for idx in op:
                lane_pos.append(involved.setdefault(idx, len(involved)))
    # virtual families: 0..6 base, 7 .. 6 + n_end the end components 1.., then the link extras, then the start extras
    V_base = N_FAMILIES + len(ends) - 1
    link_fams = list(range(N_BASE_LINKS)) + [V_base + e for e in range(n_extra_links)]
    start_fams = list(range(3, 6)) + [V_base + n_extra_links + e for e in range(n_extra_links)]
    row, fam, kappa, shift_row = [], [], [], []
    for x in range(nl):
        c_idx, Y = ch["c"][x], ch["y"][x]
        ci = involved[c_idx]
        shift_row.append(ci)                                   # link_j: + W[c][j+1]  (family x, shifted)
        for idx, kap in Y.lin.items():                         #         - sum_i kappa_i W[i][j]
            row.append(involved[idx]); fam.append(link_fams[x]); kappa.append((-kap) % P)
        row.append(ci); fam.append(start_fams[x]); kappa.append(1)     # start_j: W[c][j] = 0
    for x, (Y, _, _, _) in enumerate(ends):                    # end_j: sum_i kappa_i W[i][j] + Y.c = y_pub[j]
        for idx, kap in Y.lin.items():
            row.append(involved[idx]); fam.append(6 + x); kappa.append(kap % P)
    start, link, end = layout.masks(device)
    famsel = list(range(N_FAMILIES)) + [f for _, f, _, _ in ends[1:]]
    maskmat = torch.stack([link, link, link, start, start, start] + [end] * len(ends)
                          + [link] * n_extra_links + [start] * n_extra_links)   # (V, l) 0/1
    t = lambda v: torch.tensor(v, dtype=torch.int64, device=device)  # noqa: E731
    tabs = {"rows": t(list(involved.keys())), "famsel": t(famsel), "maskmat": maskmat, "n_extra": 2 * n_extra_links,
            "lane_pos": t(lane_pos), "link_fams": t(link_fams), "link_fams_list": list(link_fams),
            "terms": {"row": t(row), "fam": t(fam), "kappa": t(kappa), "shift_row": t(shift_row)}}
    _TABLES[key] = (sys, tabs)
    return tabs
