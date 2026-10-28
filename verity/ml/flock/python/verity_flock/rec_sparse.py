"""V*'s sparse phase (the opening-only candidate, deployment B): on one rep of an inner session whose unit circuits are hidden, the inner verifier's checks H1 to H4 at the
rep's coins, as a Boolean Definition and its plain-integer reference.

    U16 = Array<16, Value<1>>
    RecSparse_v1{NU, ELL, S_LOG, T_LOG}(h: Array<512, U16>, pre0 .. pre5: Array<64, U16>, v2: Array<64, U16>,
                                        sc0 .. sc{NU + S_LOG - 1}: Array<P(deg_j), U16>, u: Array<P(P'), U16>,
                                        [vt: Array<P(T), U16>,] claim: Array<64, U16>, [tau: Array<W(S), U16>,]
                                        v: Array<P(nv), U16>) -> Array<OUT(shape), U16>

**The shape.** A session has ``S = 2^S_LOG`` slots, slot σ of a hidden type ``τ_σ < T = 2^T_LOG``. The bucket registers its T
templates as one table ``R(t, p, k)``: ``P' = 2^⌈log₂(2 ELL + 2)⌉`` bit planes (``p < ELL`` the row bits, ``ELL <= p < 2 ELL`` the
column bits, ``2 ELL`` the A-side bit, ``2 ELL + 1`` the B-side bit, the rest zero) over ``2^NU`` entries ``k``, under one public
root. ``d = 2 ELL + 1``. ``Q(σ, p, k) = R(τ_σ, p, k)``, ``U_σ`` is slot σ's public layout factor ``Σ_{u ∈ units(σ)} eq(x_hi, u)
eq(y_hi, u)``, and with the lincheck's α, the row point ``x = (ζ_r, ρ_in)`` and the column point ``y = (ζ_c, t)``,

    Φ(q) = (α q_{2 ELL} + q_{2 ELL + 1}) · ∏_{j < ELL} (1 + x_j + q_j) · ∏_{j < ELL} (1 + y_j + q_{ELL + j}).

**The checks** (one rep; a coin is drawn after the message before it, and every sumcheck binds its lowest variable first and
sends a round's coefficients without ``c_1 = claim + Σ_{i >= 2} c_i``, characteristic 2):

* H1: ``claim = Σ_{s < 64} eq(ζ_c, s) h[s]``, ``h`` the 64 folded hidden comb values the lincheck's L4′ sent.
* H2: 6 rounds of degree 2 over the low row bits of ``Σ_s ŵ(s) V(s)`` (``pre_j = (c_0, c_2)``, coin ``ζ_r[j]``), then ``v2``,
  and ``claim = ŵ(ζ_r) · v2`` (``ŵ(ζ_r) = Σ_s eq(ζ_r, s) L_s(z)``).
* H3: from ``v2``, NU rounds of degree d over the entry bits, then S_LOG of degree d + 1 over the slot bits, of
  ``Σ_{k, σ} Ũ(σ) Φ(Q̂(σ, ·, k))`` (``sc_j = (c_0, c_2, .., c_deg)``, coins ``r_k`` then ``r_s``), then ``u`` (``u_p = Q̂(r_s, p,
  r_k)``), and ``claim = Ũ(r_s) · Φ(u)``.
* H4: with T > 1, ``vt`` (``vt_t = R̂(t, ζ_p, r_k)``) and ``Σ_p eq(ζ_p, p) u_p = Σ_t w_t vt_t``, ``w_t = Σ_{σ: τ_σ = t} eq(r_s, σ)``
  from the hidden τ (the type selects); the registered claim at ``(r_k, ζ_p, ρ_t)`` is ``claim = Σ_t eq(ρ_t, t) vt_t``. With
  T = 1 there is no ``vt`` and no τ, and ``claim = Σ_p eq(ζ_p, p) u_p``.

**Ports.** Every port is u16 words, an F element 8 of them (``lo || hi``) as `rec_residuals` has it, and a port of n elements
``P(n) = 64 ⌈n / 8⌉`` words (whole M0 rows), its words past the elements unread.

* ``h``, ``pre<j>``, ``v2``, ``sc<j>``, ``u``, ``vt``: the rep's sparse-phase messages, one port each, as the firewall committed
  them (each a row the statement reads under the coin server's record, as `rec_residuals`' ``m<i>``). ``h`` is the inner
  session's h round (`rec_algebra.h_round`, absorbed before r′ and ζ_c), which the rep's algebra (`rec_residuals.InnerRepCheck`)
  reads at the same row for its lincheck check L4′.
* ``claim``: the registered claim's value, the row the registered opening's algebra reads (row r rep r's) as a witness
  element at the coins' point (`registered_claim`), never as `rec_algebra`'s public extra claim.
* ``tau``: word σ is ``τ_σ``, ``W(S) = 64 ⌈S / 64⌉`` words, the slots' types as F2's coverage binds them: one row of the session,
  which every rep's instance reads (`reads`, `session_rows`).
* ``v``: the verifier's own row (`verifier_row`): every value that only the coins and the public layout decide (`v_layout`).

**Output.** The residuals, 8 words each, ``h2`` (H2's last check), ``h3`` (H3's), with T > 1 ``h4`` (the plane batch against the
selected sum), and ``claim`` (the registered claim); then with T > 1 one word a slot, ``τ_σ`` with its low T_LOG bits cleared;
then zero words to a whole M0 row. The statement registers it as zero, so an instance holds exactly when the rep's sparse
phase accepts at a τ inside the bucket.

**Cost.** One Gf128Mul (2,187 ANDs) for each product with a message or a computed value in it (`products`): 64 (H1), 2 a round
(H2), 1 (``ŵ · v2``), deg a round (H3), d + 1 (``α u_a``, Φ's 2 ELL factors and ``Ũ(r_s)``), P' and with T > 1 2T (H4), and
``128 (T - 1)`` ANDs a slot for the selects (a demultiplexer of ``eq(r_s, σ)`` by τ_σ's bits). Everything that only coins
decide is the verifier's (`verifier_row`), so no product has two public operands. The firewall commitments of the messages and
the reads of ``claim`` and ``tau`` are the statement's row reads, outside this Definition.

`sparse_ref` is the reference on words. The body computes the products of a message by a verifier value as one
``GfResiduals_v1`` (`products_structure`), then the sumchecks' claim chains, Φ's product tree, the selects and the residuals.

**The M0 unit** (`unit_circuit`, named `unit_name`) is the same steps over `forms`, as `rec_residuals`' unit: one input port
of u16 words per port, one output row. Its ANDs are ``ands`` and `copy_rows`' (leaf sums and committed product operands).
`stage` writes one statement over a session's reps (`session_rows`, read at `reads`' rows), as `rec_vstage` stages
InnerRepCheck's parts (`rec_residuals.write_rows`).

**The prover** (`prove`, T = 1: one hidden circuit, the slots' factor ``Ũ`` its range's layout) runs a rep's phase on coins it
draws one round at a time (``draw``: the coin server's, through the proxy), from the circuit's `entries`, which `table` lays
out as R's cells (cell ``k + 2^NU p``, byte ``i >> 3``, bit ``i & 7``, zero cells to fast100's least m). It checks every sum it
sends against the claim before it, and its registered claim is R's value at the coins' point (`table_value`).
"""
from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.ops.boolean import gf2k as G
from verity.ml.circuits.ops.boolean.gf2k import mul
from verity.ml.circuits.boolean.forms import ZERO
from verity.ml.commitments.gates.sha512 import commit

__all__ = ["Coins", "PAD_TO", "PRE_ROUNDS", "RecSparse", "RegisteredClaim", "Shape", "ands", "copy_rows", "definition", "entries",
           "inner_points", "layout_factors", "lowering", "message_ports", "out_words", "port_words", "ports", "products",
           "products_structure", "prove", "reads", "registered_claim", "registrations", "rows", "session_rows", "sparse_ref",
           "stage", "statement", "subcircuit", "table", "table_value", "unit_circuit", "unit_name", "v_layout", "verifier_row"]

BIT = BL.BIT
U16 = Array(16, BIT)
F_WORDS = 8
ROW_BLOCK = 64
#: H2's rounds: the univariate skip's 6 low row bits
PRE_ROUNDS = 6
SKIP = 1 << PRE_ROUNDS
#: fast100's least m (`rec_algebra.fast100`): `registered_claim` pads a shorter point to it (under --zk, 25: ``pad_to``)
PAD_TO = 22


@dataclass(frozen=True)
class Shape:
    """``RecSparse``'s statics: the entries' bits NU, the template's row and column bits ELL (at least the skip's 6), the slots'
    bits S_LOG and the stacked types' bits T_LOG."""

    NU: int
    ELL: int
    S_LOG: int
    T_LOG: int

    def __post_init__(self):
        for n in ("NU", "ELL", "S_LOG", "T_LOG"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"RecSparse's {n} is an int, not {v!r}")
        if self.NU < 1 or self.ELL < PRE_ROUNDS or self.S_LOG < 0 or not 0 <= self.T_LOG <= 16:
            raise ValueError(f"RecSparse's NU >= 1, ELL >= {PRE_ROUNDS}, S_LOG >= 0 and 0 <= T_LOG <= 16 (a type is one word), "
                             f"not {self}")

    @property
    def d(self) -> int:
        return 2 * self.ELL + 1

    @property
    def P(self) -> int:
        return 2 * self.ELL + 2

    @property
    def P_LOG(self) -> int:
        return (self.P - 1).bit_length()

    @property
    def PP(self) -> int:
        return 1 << self.P_LOG

    @property
    def S(self) -> int:
        return 1 << self.S_LOG

    @property
    def T(self) -> int:
        return 1 << self.T_LOG

    @property
    def degrees(self) -> list[int]:
        """H3's round degrees: d over the entry bits, d + 1 over the slot bits."""
        return [self.d] * self.NU + [self.d + 1] * self.S_LOG

    @property
    def stacked(self) -> bool:
        return self.T_LOG > 0


def _shape(S) -> Shape:
    return S if isinstance(S, Shape) else Shape(S.NU, S.ELL, S.S_LOG, S.T_LOG)


# --- layout ------------------------------------------------------------------------------------------------------------------

def port_words(n: int) -> int:
    return -(-F_WORDS * n // ROW_BLOCK) * ROW_BLOCK


def message_ports(sh: Shape) -> list[tuple[str, int]]:
    """The F-element ports in order, each with its elements: the messages, then ``claim``."""
    out = [("h", SKIP)] + [(f"pre{j}", 2) for j in range(PRE_ROUNDS)] + [("v2", 1)]
    out += [(f"sc{j}", deg) for j, deg in enumerate(sh.degrees)] + [("u", sh.PP)]
    if sh.stacked:
        out.append(("vt", sh.T))
    return out + [("claim", 1)]


def tau_words(sh: Shape) -> int:
    return -(-sh.S // ROW_BLOCK) * ROW_BLOCK


def v_layout(sh: Shape) -> dict[str, tuple[int, int]]:
    """The verifier row's parts, ``name: (first element, elements)``: ``eqc`` (eq(ζ_c, ·), 64), ``pre`` (per H2 round ``r``,
    ``r + r²``), ``what`` (ŵ(ζ_r)), ``sc`` (per H3 round ``r``, then ``r + r^i`` for i = 2 .. deg), ``alpha``, ``ax`` and ``ay``
    (``1 + x_j``, ``1 + y_j`` for j < ELL), ``ur`` (Ũ(r_s)), ``eqp`` (eq(ζ_p, ·)), and with T > 1 ``eqs`` (eq(r_s, ·)) and ``eqt``
    (eq(ρ_t, ·))."""
    parts = [("eqc", SKIP), ("pre", 2 * PRE_ROUNDS), ("what", 1), ("sc", sum(sh.degrees)), ("alpha", 1), ("ax", sh.ELL),
             ("ay", sh.ELL), ("ur", 1), ("eqp", sh.PP)]
    if sh.stacked:
        parts += [("eqs", sh.S), ("eqt", sh.T)]
    out, at = {}, 0
    for nm, n in parts:
        out[nm] = (at, n)
        at += n
    return out


def _nv(sh: Shape) -> int:
    return sum(n for _, n in v_layout(sh).values())


def residual_names(sh: Shape) -> list[str]:
    return ["h2", "h3"] + (["h4"] if sh.stacked else []) + ["claim"]


def out_words(sh: Shape) -> int:
    n = F_WORDS * len(residual_names(sh)) + (sh.S if sh.stacked else 0)
    return -(-n // ROW_BLOCK) * ROW_BLOCK


def ports(sh: Shape) -> list[tuple[str, int]]:
    """Every port in order with its words."""
    out = [(nm, port_words(n)) for nm, n in message_ports(sh)]
    if sh.stacked:
        out.append(("tau", tau_words(sh)))
    return out + [("v", port_words(_nv(sh)))]


def products(sh: Shape) -> int:
    """Gf128Mul calls of one rep."""
    return SKIP + 2 * PRE_ROUNDS + 1 + sum(sh.degrees) + sh.d + 1 + sh.PP + (2 * sh.T if sh.stacked else 0)


def ands(sh: Shape) -> int:
    """ANDs of one rep: the products and the selects."""
    return 3 ** 7 * products(sh) + (128 * (sh.T - 1) * sh.S if sh.stacked else 0)


# --- words -------------------------------------------------------------------------------------------------------------------

def words(elems: Sequence[int], n_words: int) -> list[int]:
    out = [(x >> (16 * j)) & 0xFFFF for x in elems for j in range(F_WORDS)]
    return out + [0] * (n_words - len(out))


def elements(ws: Sequence[int], n: int) -> list[int]:
    return [sum(int(ws[F_WORDS * i + j]) << (16 * j) for j in range(F_WORDS)) for i in range(n)]


def rows(sh: Shape, msgs: Mapping, v: Sequence[int]) -> dict[str, list[int]]:
    """Each port's words from the rep's messages (``h``, ``pre`` (6 pairs), ``v2``, ``sc`` (a list a round), ``u``, ``vt``,
    ``claim``, ``tau``) and the verifier row."""
    el = {"h": msgs["h"], "v2": [msgs["v2"]], "u": msgs["u"], "claim": [msgs["claim"]]}
    el.update({f"pre{j}": list(p) for j, p in enumerate(msgs["pre"])})
    el.update({f"sc{j}": list(p) for j, p in enumerate(msgs["sc"])})
    if sh.stacked:
        el["vt"] = msgs["vt"]
    out = {}
    for nm, n in message_ports(sh):
        if len(el[nm]) != n:
            raise ValueError(f"message {nm}: {len(el[nm])} elements, the shape wants {n}")
        out[nm] = words(el[nm], port_words(n))
    if sh.stacked:
        tau = [int(t) for t in msgs["tau"]]
        if len(tau) != sh.S or any(not 0 <= t < 1 << 16 for t in tau):
            raise ValueError(f"tau: {sh.S} words, not {msgs['tau']!r}")
        out["tau"] = tau + [0] * (tau_words(sh) - sh.S)
    if len(v) != _nv(sh):
        raise ValueError(f"v: {len(v)} elements, the shape wants {_nv(sh)}")
    out["v"] = words(v, port_words(_nv(sh)))
    return out


def reads(sh: Shape, reps: int) -> dict[str, list[int]]:
    """The row of its registered value each port reads in instance r: row r for the messages, ``claim`` and ``v`` (rep r's;
    ``h`` is the inner session's `rec_algebra.h_round` value, which `InnerRepCheck` reads at the same row), and row 0 for
    ``tau`` in every instance: the session's one type row, which F2's coverage check reads too."""
    out = {nm: list(range(reps)) for nm, _ in ports(sh)}
    if sh.stacked:
        out["tau"] = [0] * reps
    return out


def session_rows(sh: Shape, reps: Sequence[Mapping], vs: Sequence[Sequence[int]], tau: Sequence[int] | None = None
                 ) -> list[dict[str, list[int]]]:
    """Every rep's instance's ports (`rows`), from each rep's messages and verifier row and the session's one ``tau``: a rep's
    messages that carry a τ of their own are refused, so the reps can't select different types."""
    if any("tau" in m for m in reps):
        raise ValueError("τ is the session's one row, read by every rep's instance: give it once, not in a rep's messages")
    if sh.stacked != (tau is not None):
        raise ValueError(f"{sh}: tau {'missing' if sh.stacked else 'given for one template'}")
    out = [rows(sh, {**m, "tau": tau} if sh.stacked else m, v) for m, v in zip(reps, vs, strict=True)]
    if sh.stacked and any(o["tau"] != out[0]["tau"] for o in out):
        raise AssertionError("the reps' τ rows differ")
    return out


# --- the verifier's row ------------------------------------------------------------------------------------------------------

def eq_table(r: Sequence[int]) -> list[int]:
    """eq(r, b) for b < 2^len(r), bit j of b against r[j]."""
    t = [1]
    for rj in r:
        t = [mul(x, rj ^ 1) for x in t] + [mul(x, rj) for x in t]
    return t


def _dot(a: Sequence[int], b: Sequence[int]) -> int:
    return reduce(lambda x, y: x ^ y, (mul(p, q) for p, q in zip(a, b, strict=True)), 0)


@dataclass(frozen=True)
class Coins:
    """A rep's public values the sparse phase reads: the lincheck's ``alpha``, the zerocheck's skip coin ``z``, the row and column
    points' coordinates past the skip's (``x_rest`` = ρ_in's, ``y_rest`` = the column point in bit order, `rec_algebra`'s
    ``r_in``, the lincheck's coins t reversed; at least ELL - 6 each, beyond which they enter only ``U``; `inner_points`), H1's ``zeta_c`` (6), H2's ``pre`` (6, ζ_r), H3's ``rounds`` (r_k then r_s),
    H4's ``zeta_p`` (P_LOG) and ``rho`` (T_LOG), and the slots' layout factors ``U`` (U_σ for σ < S, from the public layout and
    the points' high coordinates)."""

    alpha: int
    z: int
    x_rest: tuple[int, ...]
    y_rest: tuple[int, ...]
    zeta_c: tuple[int, ...]
    pre: tuple[int, ...]
    rounds: tuple[int, ...]
    zeta_p: tuple[int, ...]
    rho: tuple[int, ...]
    U: tuple[int, ...]

    def x_lo(self, sh: Shape) -> list[int]:
        return list(self.pre) + list(self.x_rest[:sh.ELL - PRE_ROUNDS])

    def y_lo(self, sh: Shape) -> list[int]:
        return list(self.zeta_c) + list(self.y_rest[:sh.ELL - PRE_ROUNDS])


def inner_points(inner, rounds: Sequence[Sequence[int]]) -> dict:
    """``Coins``' ``alpha``, ``z``, ``x_rest`` and ``y_rest`` from the inner rep's coins by round (an inner
    `rec_algebra.Shape`), through `rec_algebra.run`: the row point's ρ_in and the column point in bit order (``r_in``), the
    points the comb is folded at and the inner claim on z has."""
    from . import rec_algebra as A

    info = A.run(inner, A.Coins(rounds), A.symbolic(inner), [], fold_only=True)[1]
    return {"alpha": info["fold"]["alpha"], "z": info["z"], "x_rest": tuple(info["fold"]["rho_in"]), "y_rest": tuple(info["r_in"])}


@dataclass(frozen=True)
class RegisteredClaim:
    """The claim rep r's instance hands the registered opening: its value is row ``row`` (= r) of the prover's ``claim``
    registration, which the opening reads as a witness element, never as a public value (it depends on the templates), so
    it has no ``value`` and `rec_algebra.run` refuses it as an extra claim; its point is the coins': skip weights
    ``eq(point[:6], ·)`` and ``x = point[6:]``, ``point = (r_k, ζ_p, ρ_t)``, the stacked table's variables in order, then
    ``pad`` coordinates the shape fixes at zero (`registered_claim`)."""

    row: int
    skip: tuple[int, ...]
    x: tuple[int, ...]
    port: str = "claim"
    pad: int = 0
    value = None


def registered_claim(sh: Shape, c: Coins, row: int, pad_to: int = PAD_TO) -> RegisteredClaim:
    """Rep ``row``'s claim at the coins' point. A point of fewer than ``pad_to`` coordinates (the stacked table of fewer than
    fast100's least m variables) is padded with zeros in new high variables, the table with zero rows: ``eq(x || 0, b || c)``
    vanishes off ``c = 0``, so the claim is the table's at the unpadded point. How many is the shape's, never the prover's,
    and the opening refuses a padded coordinate that is not zero (`rec_algebra._registered`)."""
    point = list(c.rounds[:sh.NU]) + list(c.zeta_p) + list(c.rho)
    if len(point) < PRE_ROUNDS:
        raise ValueError(f"{sh}: a claim point of {len(point)} coordinates, fewer than the skip's {PRE_ROUNDS}")
    pad = max(0, pad_to - len(point))
    point += [0] * pad
    return RegisteredClaim(row, tuple(eq_table(point[:PRE_ROUNDS])), tuple(point[PRE_ROUNDS:]), pad=pad)


def verifier_row(sh: Shape, c: Coins) -> list[int]:
    """``v``'s elements (`v_layout`) from the rep's coins and the slots' layout factors."""
    from .rec_algebra import skip_weights

    want = {"zeta_c": PRE_ROUNDS, "pre": PRE_ROUNDS, "rounds": sh.NU + sh.S_LOG, "zeta_p": sh.P_LOG, "rho": sh.T_LOG, "U": sh.S}
    if any(len(getattr(c, k)) != n for k, n in want.items()) or min(len(c.x_rest), len(c.y_rest)) < sh.ELL - PRE_ROUNDS:
        raise ValueError(f"coins of the wrong lengths for {sh}")
    r_s = c.rounds[sh.NU:]

    def pows(r: int, deg: int) -> list[int]:
        out, p = [r], r
        for _ in range(2, deg + 1):
            p = mul(p, r)
            out.append(p ^ r)
        return out

    v = eq_table(c.zeta_c)
    v += [x for r in c.pre for x in pows(r, 2)]
    v.append(_dot(eq_table(c.pre), skip_weights(c.z)))
    v += [x for r, deg in zip(c.rounds, sh.degrees, strict=True) for x in pows(r, deg)]
    v.append(c.alpha)
    v += [x ^ 1 for x in c.x_lo(sh)] + [y ^ 1 for y in c.y_lo(sh)]
    v.append(_dot(eq_table(r_s), c.U))
    v += eq_table(c.zeta_p)
    if sh.stacked:
        v += eq_table(r_s) + eq_table(c.rho)
    assert len(v) == _nv(sh)
    return v


# --- the reference -----------------------------------------------------------------------------------------------------------

def sparse_ref(sh: Shape, rows_: Mapping[str, Sequence[int]], v: Sequence[int] | None = None) -> list[int]:
    """The reference: the output words on the ports' words (``rows_`` by name; ``v`` the verifier row's elements, or its words
    under ``rows_["v"]``)."""
    E = {nm: elements(rows_[nm], n) for nm, n in message_ports(sh)}
    if v is None:
        v = elements(rows_["v"], _nv(sh))
    lay = v_layout(sh)

    def V(nm: str) -> list[int]:
        a, n = lay[nm]
        return list(v[a:a + n])

    claim = _dot(V("eqc"), E["h"])
    vp = V("pre")
    for j in range(PRE_ROUNDS):
        c0, c2 = E[f"pre{j}"]
        r, rsq = vp[2 * j], vp[2 * j + 1]
        claim = c0 ^ mul(claim, r) ^ mul(c2, rsq)
    h2 = claim ^ mul(V("what")[0], E["v2"][0])
    claim, vs = E["v2"][0], V("sc")
    for j, deg in enumerate(sh.degrees):
        cs, at = E[f"sc{j}"], sum(sh.degrees[:j])
        r = vs[at]
        claim = cs[0] ^ mul(claim, r) ^ _dot(cs[1:], vs[at + 1:at + deg])
    u = E["u"]
    phi = mul(V("alpha")[0], u[2 * sh.ELL]) ^ u[2 * sh.ELL + 1]
    for j, (ax, ay) in enumerate(zip(V("ax"), V("ay"), strict=True)):
        phi = mul(mul(phi, ax ^ u[j]), ay ^ u[sh.ELL + j])
    h3 = claim ^ mul(V("ur")[0], phi)
    batch = _dot(V("eqp"), u)
    res = [h2, h3]
    if sh.stacked:
        tau = [int(t) for t in rows_["tau"][:sh.S]]
        w = [0] * sh.T
        for s, e in enumerate(V("eqs")):
            w[tau[s] & (sh.T - 1)] ^= e
        res += [batch ^ _dot(w, E["vt"]), E["claim"][0] ^ _dot(V("eqt"), E["vt"])]
        tail = [t & ~(sh.T - 1) & 0xFFFF for t in tau]
    else:
        res.append(E["claim"][0] ^ batch)
        tail = []
    out = words(res, F_WORDS * len(res)) + tail
    return out + [0] * (out_words(sh) - len(out))


# --- the Definition ----------------------------------------------------------------------------------------------------------

def products_structure(sh: Shape) -> tuple[dict, dict[str, int]]:
    """``GfResiduals_v1``'s S for the products of a message by a verifier value, and the linear forms beside them, over w = the
    message ports' elements in order and v = the verifier row; and each residual's index by name: ``h1`` (H1's claim), ``pre<j>``
    (``c_0 + c_2 (r + r²)``), ``wv`` (``ŵ v2``), ``sc<j>`` (``c_0 + Σ_{i >= 2} c_i (r + r^i)``), ``f0`` (``α u_a + u_b``), ``fx<j>``
    and ``fy<j>`` (Φ's factors), ``bp`` (``Σ_p eq(ζ_p, p) u_p``, plus ``claim`` with T = 1) and with T > 1 ``claim`` (``claim + Σ_t
    eq(ρ_t, t) vt_t``)."""
    at, w0 = 0, {}
    for nm, n in message_ports(sh):
        w0[nm] = at
        at += n
    lay = v_layout(sh)
    ops: list[list] = []
    opix: dict[tuple, int] = {}

    def op(t: tuple) -> int:
        if t not in opix:
            opix[t] = len(ops)
            ops.append([list(t)])
        return opix[t]

    def W(nm: str, i: int = 0) -> tuple:
        return ("w", w0[nm] + i)

    def Vt(nm: str, i: int = 0) -> tuple:
        return ("v", lay[nm][0] + i)

    res: list[dict] = []
    names: dict[str, int] = {}

    def put(nm: str, mul_: list[tuple[tuple, tuple]], add: list[tuple]) -> None:
        names[nm] = len(res)
        res.append({"mul": [[op(a), op(b)] for a, b in mul_], "add": [list(t) for t in add]})

    put("h1", [(Vt("eqc", s), W("h", s)) for s in range(SKIP)], [])
    for j in range(PRE_ROUNDS):
        put(f"pre{j}", [(W(f"pre{j}", 1), Vt("pre", 2 * j + 1))], [W(f"pre{j}", 0)])
    put("wv", [(Vt("what"), W("v2"))], [])
    for j, deg in enumerate(sh.degrees):
        a = sum(sh.degrees[:j])
        put(f"sc{j}", [(W(f"sc{j}", i - 1), Vt("sc", a + i - 1)) for i in range(2, deg + 1)], [W(f"sc{j}", 0)])
    put("f0", [(Vt("alpha"), W("u", 2 * sh.ELL))], [W("u", 2 * sh.ELL + 1)])
    for j in range(sh.ELL):
        put(f"fx{j}", [], [Vt("ax", j), W("u", j)])
        put(f"fy{j}", [], [Vt("ay", j), W("u", sh.ELL + j)])
    put("bp", [(Vt("eqp", p), W("u", p)) for p in range(sh.PP)], [] if sh.stacked else [W("claim")])
    if sh.stacked:
        put("claim", [(Vt("eqt", t), W("vt", t)) for t in range(sh.T)], [W("claim")])
    S = {"nw": at, "nv": sum(n for _, n in lay.values()), "ops": ops, "res": res}
    G.check_structure(S)
    return S, names


class _Ir:
    def __init__(self, B):
        self.B = B
        self._zero = None

    @staticmethod
    def flat(c: Coll) -> list:
        return [c.refs[i] for i in range(len(c.refs))]

    def zero(self):
        if self._zero is None:
            self._zero = self.B.call(BL.constant(0)).refs[0]
        return self._zero

    def muls(self, pairs: list[tuple[list, list]]) -> list[list]:
        """One batch of Gf128Mul over the pairs of elements."""
        n = len(pairs)
        a = Coll(Array(n, G.F), TR._refs([x for p, _ in pairs for x in p]))
        b = Coll(Array(n, G.F), TR._refs([x for _, q in pairs for x in q]))
        out = self.flat(self.B.batch(G.Gf128Mul, a, b, axes=(0, 0)))
        return [out[128 * i:128 * i + 128] for i in range(n)]

    def _pairs(self, fn, a: list, b: list) -> list:
        return self.flat(self.B.batch(fn, Coll(Array(2, Array(len(a), BIT)), TR._refs(a + b)), axes=(1,)))

    def xor(self, a: list, b: list) -> list:
        return self._pairs(BL.XOR, a, b)

    def and_(self, a: list, b: list) -> list:
        return self._pairs(BL.AND, a, b)

    def sums(self, groups: list[list[list]]) -> list[list]:
        """Each group of equal-width vectors summed, level by level, one XOR batch a level."""
        cur = [list(g) for g in groups]
        while any(len(g) > 1 for g in cur):
            pairs = [(gi, k) for gi, g in enumerate(cur) for k in range(0, len(g) - 1, 2)]
            out = self.xor([x for gi, k in pairs for x in cur[gi][k]], [x for gi, k in pairs for x in cur[gi][k + 1]])
            nxt: list[list] = [[] for _ in cur]
            o = 0
            for gi, k in pairs:
                n = len(cur[gi][k])
                nxt[gi].append(out[o:o + n])
                o += n
            for gi, g in enumerate(cur):
                if len(g) % 2:
                    nxt[gi].append(g[-1])
            cur = nxt
        return [g[0] for g in cur]

    def residuals(self, S: dict, w: list, v: list) -> list[list]:
        out = self.flat(self.B.call(bind(G.GfResiduals, S=S), Coll(Array(S["nw"], G.F), TR._refs(w)),
                                    Coll(Array(S["nv"], G.F), TR._refs(v))))
        return [out[128 * i:128 * i + 128] for i in range(len(S["res"]))]


def _elems(bits: list, n: int) -> list[list]:
    return [bits[128 * i:128 * i + 128] for i in range(n)]


def _steps(ops: _Ir, sh: Shape, P: dict[str, list]) -> list:
    """The output bits from the ports' bits (by name)."""
    S, ix = products_structure(sh)
    msg = message_ports(sh)
    w = [b for nm, n in msg for b in P[nm][:128 * n]]
    v = P["v"][:128 * S["nv"]]
    R = ops.residuals(S, w, v)
    E = {nm: _elems(P[nm], n) for nm, n in msg}
    lay = v_layout(sh)

    def V(nm: str, i: int = 0) -> list:
        return v[128 * (lay[nm][0] + i):128 * (lay[nm][0] + i + 1)]

    claim = R[ix["h1"]]
    for j in range(PRE_ROUNDS):
        (p,) = ops.muls([(claim, V("pre", 2 * j))])
        claim = ops.xor(R[ix[f"pre{j}"]], p)
    h2 = ops.xor(claim, R[ix["wv"]])
    claim = E["v2"][0]
    for j in range(len(sh.degrees)):
        (p,) = ops.muls([(claim, V("sc", sum(sh.degrees[:j])))])
        claim = ops.xor(R[ix[f"sc{j}"]], p)
    fac = [R[ix["f0"]]] + [R[ix[f"{a}{j}"]] for j in range(sh.ELL) for a in ("fx", "fy")] + [V("ur")]
    while len(fac) > 1:
        prod = ops.muls([(fac[k], fac[k + 1]) for k in range(0, len(fac) - 1, 2)])
        fac = prod + ([fac[-1]] if len(fac) % 2 else [])
    h3 = ops.xor(claim, fac[0])
    out = [h2, h3]
    if sh.stacked:
        tau = P["tau"]
        lists = [[V("eqs", s)] for s in range(sh.S)]
        for j in reversed(range(sh.T_LOG)):
            src = [(s, x) for s in range(sh.S) for x in lists[s]]
            hi = ops.and_([b for s, _ in src for b in [tau[16 * s + j]] * 128], [b for _, x in src for b in x])
            lo = ops.xor([b for _, x in src for b in x], hi)
            nxt: list[list] = [[] for _ in range(sh.S)]
            for i, (s, _) in enumerate(src):
                nxt[s] += [lo[128 * i:128 * i + 128], hi[128 * i:128 * i + 128]]
            lists = nxt
        wt = ops.sums([[lists[s][t] for s in range(sh.S)] for t in range(sh.T)])
        sel = ops.muls(list(zip(wt, E["vt"], strict=True)))
        (h4,) = ops.sums([[R[ix["bp"]]] + sel])
        out += [h4, R[ix["claim"]]]
        tail = [b for s in range(sh.S) for b in [ops.zero()] * sh.T_LOG + tau[16 * s + sh.T_LOG:16 * s + 16]]
    else:
        out.append(R[ix["bp"]])
        tail = []
    bits = [b for e in out for b in e] + tail
    return bits + [ops.zero()] * (16 * out_words(sh) - len(bits))


def _signature(S):
    sh = _shape(S)
    return tuple((nm, Array(n, U16)) for nm, n in ports(sh)), Array(out_words(sh), U16)


def _body(B, S, *colls):
    sh = _shape(S)
    P = {nm: _Ir.flat(c) for (nm, _), c in zip(ports(sh), colls, strict=True)}
    return Coll(Array(out_words(sh), U16), TR._refs(_steps(_Ir(B), sh, P)))


RecSparse = CompositeDefinition(
    "RecSparse", 1, ("NU", "ELL", "S_LOG", "T_LOG"), _signature, _body, conformance="exact-model-tested",
    doc="One rep's sparse phase of F4 candidate 3 (H1 to H4 at the rep's coins, the verifier's values in v): its residuals "
        "(H2's last check, H3's, with T > 1 the plane batch against the type-selected sum, the registered claim), then with "
        "T > 1 each slot's type with its low T_LOG bits cleared, as u16 words, zero iff the phase accepts")


def definition(NU: int, ELL: int, S_LOG: int, T_LOG: int):
    Shape(NU, ELL, S_LOG, T_LOG)
    return bind(RecSparse, NU=NU, ELL=ELL, S_LOG=S_LOG, T_LOG=T_LOG)


# --- the M0 unit -------------------------------------------------------------------------------------------------------------

class _Forms:
    """`_steps` over `forms`: the products of messages by verifier values leaf-summed per residual
    (`rec_residuals.residual_forms`), and each other product's operands committed first unless they are already committed
    bits, so that no AND reads a chain's or a product tree's accumulated sum."""

    def __init__(self, C):
        self.C = C

    def zero(self):
        return ZERO

    def muls(self, pairs: list[tuple[list, list]]) -> list[list]:
        return [G.mul_forms(self.C, commit(self.C, a), commit(self.C, b)) for a, b in pairs]

    def xor(self, a: list, b: list) -> list:
        return [p ^ q for p, q in zip(a, b, strict=True)]

    def and_(self, a: list, b: list) -> list:
        return [self.C.AND(p, q) for p, q in zip(a, b, strict=True)]

    def sums(self, groups: list[list[list]]) -> list[list]:
        return [reduce(self.xor, g) for g in groups]

    def residuals(self, S: dict, w: list, v: list) -> list[list]:
        from .rec_residuals import residual_forms

        return _elems(residual_forms(self.C, S, w, v), len(S["res"]))


def copy_rows(sh: Shape) -> int:
    """The unit's rows beyond ``ands`` (each one AND with ``B = 1``, `sha512.commit`): 2,187 leaf sums for each of
    `products_structure`'s residuals of two products or more, and 128 for each product operand that is not already committed
    bits: H2's running claim each round, H3's from its second round, the operands of Φ's product tree but ``Ũ(r_s)``, and with
    T > 1 each ``w_t`` (the all-ones type's is one AND when S = 1)."""
    sh = _shape(sh)
    S, _ = products_structure(sh)
    leaf = 3 ** 7 * sum(1 for r in S["res"] if len(r["mul"]) > 1)
    flags, tree = [True] * (2 * sh.ELL + 1) + [False], 0
    while len(flags) > 1:
        k = len(flags) // 2
        tree += sum(flags[:2 * k])
        flags = [True] * k + flags[2 * k:]
    w = (sh.T - (sh.S == 1)) if sh.stacked else 0
    return leaf + 128 * (PRE_ROUNDS + len(sh.degrees) - 1 + tree + w)


def unit_name(sh: Shape) -> str:
    sh = _shape(sh)
    return f"rec-sparse/n{sh.NU}-e{sh.ELL}-s{sh.S_LOG}-t{sh.T_LOG}"


def unit_circuit(sh: Shape):
    """The unit over `forms`: inputs ``in0 ..`` (each port's words in `ports`' order, 16 bits each), outputs ``out0 ..
    out{out_words - 1}``."""
    from verity.ml.circuits.boolean.forms import Circuit

    sh = _shape(sh)
    C = Circuit()
    pw = [n for _, n in ports(sh)]
    ins = [C.input(f"in{j}", 16) for j in range(sum(pw))]
    flat = [b for wd in ins for b in wd]
    at = [16 * sum(pw[:i]) for i in range(len(pw) + 1)]
    P = {nm: flat[a:b] for (nm, _), a, b in zip(ports(sh), at, at[1:])}
    bits = _steps(_Forms(C), sh, P)
    outs = [f"out{j}" for j in range(out_words(sh))]
    for j, nm in enumerate(outs):
        C.output(nm, bits[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(len(ins))], outs


def lowering(sh: Shape):
    """`unit_circuit` as M0's unit, named `unit_name`: one u16 row per port, one output row."""
    from . import ir_lower as IL
    from .class_statement import ClassDefinition

    sh = _shape(sh)
    name = unit_name(sh)
    C, ins, outs = unit_circuit(sh)
    w_in, w_out = len(ins), len(outs)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    defn = ClassDefinition(name, tuple(n for _, n in ports(sh)), w_out)
    return IL.Lowering(name, defn, units, C, IL.layout(C, ins, outs))


def subcircuit(sh: Shape):
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``RecSparse_v1{shape}``."""
    from types import SimpleNamespace

    from verity.ml.circuits.codec import program_digest
    from verity.ml.circuits.program import Program

    sh = _shape(sh)
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    defn = definition(sh.NU, sh.ELL, sh.S_LOG, sh.T_LOG)
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=unit_name(sh), template=SimpleNamespace(id="rec-sparse"), inputs=[port(nm) for nm, _ in ports(sh)],
                           outputs=[port("out")], definition=lambda **_: defn, program_digests=lambda statics: {defn.id: digest})


# --- the statement -----------------------------------------------------------------------------------------------------------

V_VALUE = "rec-sparse-v"
ZERO_VALUE = "rec-sparse-zero"


def statement(sh: Shape, n: int):
    """``RecSparse_v1{shape}`` over ``n`` instances (a session's reps): its Program, partition, unit and circuit text, in
    `rec_residuals.write_rows`' form (row ports every port but ``v``, ``v`` the verifier's, one output row registered zero)."""
    import time
    from types import SimpleNamespace

    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    from . import circuit as FC

    sh = _shape(sh)
    t0 = time.perf_counter()
    defn = definition(sh.NU, sh.ELL, sh.S_LOG, sh.T_LOG)
    program = P.population_program(defn, n)
    part = P.partition(program, P.template_instance_query(defn))
    prog = P.program_sha512(program)
    low, sub = lowering(sh), subcircuit(sh)
    text = FC.compose(low, sub, partition=P.digest(part), program=prog)
    return SimpleNamespace(sh=sh, n=n, defn=defn, program=program, part=part, prog=prog, low=low, sub=sub, text=text,
                           names=[nm for nm, _ in ports(sh) if nm != "v"], v_words=dict(ports(sh))["v"], out_words=out_words(sh),
                           v_value=V_VALUE, zero_value=ZERO_VALUE, ref=lambda rows, vw: sparse_ref(sh, {**rows, "v": vw}),
                           build_s=round(time.perf_counter() - t0, 1))


def registrations(sh: Shape, program: str, inst: Sequence[Mapping[str, list[int]]], salts: Mapping | None = None) -> dict:
    """The prover's registered values of the row ports, from `session_rows`' instances: each port as ``rec-sparse-<port>``, row r
    rep r's, but ``tau``, the session's one row, and ``claim``, ``rec-claim`` (`rec_outer.CLAIM_VALUE`, which the registered
    opening reads too). ``salts``: per port, its rows' salts (default fresh)."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG
    from .rec_outer import CLAIM_VALUE

    sh = _shape(sh)
    out = {}
    for nm, _ in ports(sh):
        if nm == "v":
            continue
        rows = [list(inst[0][nm])] if nm == "tau" else [list(r[nm]) for r in inst]
        out[nm] = RG.Registered.commit(program, CLAIM_VALUE if nm == "claim" else f"rec-sparse-{nm}", rows, (salts or {}).get(nm))
    return out


def stage(sh: Shape, inst: Sequence[Mapping[str, list[int]]], out_dir, program: str | None = None, regs: Mapping | None = None,
          own: Mapping | None = None, evaluate: bool = True) -> dict:
    """One RecSparse statement over a session's reps, written to ``out_dir`` as `rec_vstage` stages InnerRepCheck's parts
    (`rec_residuals.write_rows`): ``inst`` is `session_rows`' instances, each row port read at its row of `reads` (τ at row 0,
    every other port at row r) of the prover's registration (``regs`` by port, the rest `registrations`' under ``program``,
    default the statement's), ``v`` the verifier's own registration of each rep's row, and ``out`` its zero rows. ``own``: per
    row port the verifier's own entry, which its reads file names (default the registration's). Returns what the statement
    costs and holds, and each instance's nonzero residuals (and ``tau`` for a type outside the bucket)."""
    from .rec_residuals import write_rows

    sh = _shape(sh)
    n = len(inst)
    st = statement(sh, n)
    rd = reads(sh, n)
    regs = {**registrations(sh, program or st.prog, inst), **(regs or {})}
    s = write_rows(st, {nm: (regs[nm], rd[nm]) for nm in st.names}, [list(r["v"]) for r in inst], out_dir, evaluate, own)
    names = residual_names(sh)
    hits = [[nm for i, nm in enumerate(names) if any(x[8 * i:8 * i + 8])] + (["tau"] if any(x[8 * len(names):]) else [])
            for x in s.pop("ref")]
    return {"template": st.defn.id, "unit": unit_name(sh), "instances": n, "products": products(sh), "ands": ands(sh),
            "nonzero_residuals": hits, **s}



# --- the prover at T = 1 -----------------------------------------------------------------------------------------------------

#: `entries` lays a circuit's entries out in this many runs, each followed by zero entries, so that `prove`'s workers (each a
#: contiguous share of the runs) share H3's work evenly
BLOCKS = 64


def entries(sh: Shape, a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], blocks: int = BLOCKS
            ) -> list[tuple[int, int, int, int]]:
    """A hidden circuit's 2^NU entries ``(row, column, a, b)`` from its A and B rows (`rec_session.hidden_circuits`' ``a`` and
    ``b``): one a cell of A or B, ``a`` and ``b`` its parity in each (a cell held an even number of times in both is none), in
    row then column order, in ``blocks`` runs each followed by zero entries (Φ is zero on them)."""
    par: dict[tuple[int, int], int] = {}
    for side, rows_ in enumerate((a, b)):
        for i, row in enumerate(rows_):
            for c in row:
                par[(i, c)] = par.get((i, c), 0) ^ (1 << side)
    ents = [(i, c, f & 1, f >> 1) for (i, c), f in sorted(par.items()) if f]
    n = 1 << sh.NU
    if len(ents) > n or any(max(i, c) >> sh.ELL for i, c, _, _ in ents):
        raise ValueError(f"{sh}: {len(ents)} entries, not at most 2^NU entries of 2^ELL rows and columns")
    blocks = max(1, min(blocks, n >> 1))
    size, per = n // blocks, -(-len(ents) // blocks)
    out: list[tuple[int, int, int, int]] = []
    for j in range(blocks):
        run = ents[j * per:(j + 1) * per]
        out += run + [(0, 0, 0, 0)] * (size - len(run))
    return out


def _planes(sh: Shape, e: tuple[int, int, int, int]) -> tuple[int, ...]:
    row, col, a, b = e
    return tuple((row >> j) & 1 for j in range(sh.ELL)) + tuple((col >> j) & 1 for j in range(sh.ELL)) + (a, b)


def table_m(sh: Shape, pad_to: int = PAD_TO) -> int:
    """R's variables: its cells' (``NU + P_LOG + T_LOG``), at least ``pad_to`` (`registered_claim`'s padded point)."""
    return max(pad_to, sh.NU + sh.P_LOG + sh.T_LOG)


def table(sh: Shape, ents: Sequence[tuple[int, int, int, int]], pad_to: int = PAD_TO) -> bytes:
    """R's bytes (``flock-circuit stacked --table``): cell ``k + 2^NU p`` is entry k's plane p (its row bits, column bits, a and
    b, then zero planes to P'), cell i at byte ``i >> 3``, bit ``i & 7``, then zero cells to ``2^pad_to`` (`registered_claim`'s
    padded point)."""
    import numpy as np

    n = 1 << sh.NU
    if sh.stacked or len(ents) != n:
        raise ValueError(f"{sh}: R of one template's 2^NU entries, not {len(ents)}")
    m = table_m(sh, pad_to)
    e = np.array(ents, dtype=np.int64).reshape(n, 4)
    bits = np.zeros((1 << (m - sh.NU), n), np.uint8)
    for j in range(sh.ELL):
        bits[j], bits[sh.ELL + j] = (e[:, 0] >> j) & 1, (e[:, 1] >> j) & 1
    bits[2 * sh.ELL], bits[2 * sh.ELL + 1] = e[:, 2], e[:, 3]
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def table_value(raw: bytes, point: Sequence[int]) -> int:
    """R's multilinear extension at ``point`` (one coordinate a bit of the cell index, low first): ``Σ_i eq(point, i) R_i``,
    the value the registered opening at `registered_claim`'s point proves."""
    import numpy as np

    m = len(point)
    if len(raw) * 8 != 1 << m:
        raise ValueError(f"{len(raw)} bytes of R, a point of {m} coordinates")
    lo = min(m, 12)
    cells = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little").reshape(1 << (m - lo), 1 << lo)
    e = np.frombuffer(b"".join(x.to_bytes(16, "little") for x in eq_table(point[:lo])), np.uint8)
    e = np.unpackbits(e, bitorder="little").reshape(1 << lo, 128).astype(np.float32)
    rows = np.packbits((cells.astype(np.float32) @ e).astype(np.int64).astype(np.uint8) & 1, axis=1, bitorder="little")
    return reduce(lambda x, y: x ^ y, (mul(w, int.from_bytes(r.tobytes(), "little")) for w, r in zip(eq_table(point[lo:]), rows)), 0)


def layout_factors(sh: Shape, x_rest: Sequence[int], y_rest: Sequence[int], rng: Mapping) -> list[int]:
    """``U``: slot σ's ``eq(x_hi, q) eq(y_hi, q)`` at its position ``q = pos0 + σ`` for σ below the hidden circuit's range's
    count (META's range, `rec_session.hidden_circuits`), zero past it; ``x_hi`` and ``y_hi`` the points' coordinates past
    ELL - 6. Their sum is `rec_session.hidden_fold`'s weight."""
    k = sh.ELL - PRE_ROUNDS
    ex, ey = eq_table(x_rest[k:]), eq_table(y_rest[k:])
    if rng["slot_log"] != sh.ELL or rng["count"] > sh.S or rng["pos0"] + rng["count"] > min(len(ex), len(ey)):
        raise ValueError(f"{sh}: the range {dict(rng)} is not at most S slots of 2^ELL rows inside the block")
    return [mul(ex[q], ey[q]) for q in range(rng["pos0"], rng["pos0"] + rng["count"])] + [0] * (sh.S - rng["count"])


def _xor(xs) -> int:
    return reduce(lambda x, y: x ^ y, xs, 0)


def _sc(alpha: int, q: int) -> int:
    return alpha if q == 1 else 0 if q == 0 else mul(alpha, q)


def _lin(P: list[int], a: int, d: int) -> list[int]:
    """``P · (a + d X)``, coefficients lowest first."""
    if d == 1:
        return [mul(a, P[0])] + [mul(a, P[i]) ^ P[i - 1] for i in range(1, len(P))] + [P[-1]]
    return [mul(a, P[0])] + [mul(a, P[i]) ^ mul(d, P[i - 1]) for i in range(1, len(P))] + [mul(d, P[-1])]


def _poly(T: Sequence[tuple[int, ...]], alpha: int, ax: Sequence[int], ay: Sequence[int]) -> list[int]:
    """``Σ_i Φ`` on the line through entries 2i and 2i + 1 of ``T`` (each its P planes), coefficients c_0 .. c_d: a pair's d
    linear factors multiplied out, those constant on its line multiplied in as one scalar."""
    E = len(ax)
    fa = [*ax, *ay]
    acc = [0] * (2 * E + 2)
    for i in range(0, len(T), 2):
        e0, e1 = T[i], T[i + 1]
        qa, qb = e0[2 * E], e0[2 * E + 1]
        f0, d0 = _sc(alpha, qa) ^ qb, _sc(alpha, qa ^ e1[2 * E]) ^ qb ^ e1[2 * E + 1]
        if not (f0 or d0):
            continue
        c, P = (1, [f0, d0]) if d0 else (f0, [1])
        for j in range(2 * E):
            x, y = e0[j], e1[j]
            f = fa[j] ^ x
            if x != y:
                P = _lin(P, f, x ^ y)
            elif not f:
                break
            else:
                c = f if c == 1 else mul(c, f)
        else:
            for k, p in enumerate(P):
                acc[k] ^= p if c == 1 else mul(c, p)
    return acc


def _fold(T: Sequence[tuple[int, ...]], r: int) -> list[tuple[int, ...]]:
    """``T`` with its lowest variable bound to ``r``: each pair's planes on their line at r."""
    out = []
    for i in range(0, len(T), 2):
        e0, e1 = T[i], T[i + 1]
        out.append(e0 if e0 == e1 else tuple(x if x == y else x ^ (r if x ^ y == 1 else mul(r, x ^ y)) for x, y in zip(e0, e1)))
    return out


def _slice(conn, T, alpha, ax, ay) -> None:
    """A worker of `prove`: its share of the entries' rounds until one entry is left, which it sends."""
    while len(T) > 1:
        conn.send(_poly(T, alpha, ax, ay))
        T = _fold(T, conn.recv())
    conn.send(T[0])


def _eval(poly: Sequence[int], r: int) -> int:
    out = 0
    for c in reversed(poly):
        out = mul(out, r) ^ c
    return out


def _entry_rounds(T, alpha, ax, ay, W, claim, draw, workers):
    nu = len(T).bit_length() - 1
    lw = max(0, min(workers.bit_length() - 1, nu - 2))
    msgs, rs, conns, procs = [], [], [], []
    try:
        if lw:
            ctx, size = multiprocessing.get_context("fork"), len(T) >> lw
            for i in range(1 << lw):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_slice, args=(b, T[i * size:(i + 1) * size], alpha, ax, ay), daemon=True)
                p.start()
                b.close()
                conns.append(a)
                procs.append(p)
        for j in range(nu):
            if conns and j == nu - lw:
                T, conns = [c.recv() for c in conns], []
            if conns:
                poly = reduce(lambda p, q: [x ^ y for x, y in zip(p, q)], (c.recv() for c in conns))
            else:
                poly = _poly(T, alpha, ax, ay)
            if claim != mul(W, _xor(poly[1:])):
                raise AssertionError(f"H3 entry round {j}: the entries' sum is not the claim")
            msg = [mul(W, poly[0])] + [mul(W, x) for x in poly[2:]]
            (r,) = draw(msg, 1)
            for c in conns:
                c.send(r)
            if not conns:
                T = _fold(T, r)
            claim = mul(W, _eval(poly, r))
            msgs.append(msg)
            rs.append(r)
        return T[0], msgs, rs, claim
    finally:
        for p in procs:
            p.kill()
            p.join()


def prove(sh: Shape, ents: Sequence[tuple[int, int, int, int]], pts: Mapping, U: Sequence[int], h: Sequence[int],
          draw: Callable[[list[int], int], list[int]], workers: int = 1) -> tuple[Coins, dict]:
    """One rep's sparse phase at T = 1 (the module docstring): ``ents`` the hidden circuit's `entries`, ``pts`` the rep's
    `inner_points` (``alpha``, ``z``, ``x_rest``, ``y_rest``), ``U`` its slots' `layout_factors`, ``h`` the 64 elements of the
    rep's h round. Each round's message goes to ``draw(elements, n)``, which answers its n coins (v2's and the claim's
    rounds none: the claim goes last, so the coin server records the ``c_k`` that ``rec-claim``'s rows must be);
    ``workers`` processes share H3's entry rounds. Returns the coins and the messages as `rows` takes them (``claim`` the
    registered claim's value). AssertionError if a sum the phase sends is not the claim before it (``ents`` are not the
    circuit h was folded from)."""
    from .rec_algebra import skip_weights

    if sh.stacked or len(ents) != 1 << sh.NU or len(U) != sh.S or len(h) != SKIP:
        raise ValueError(f"{sh}: the prover runs T = 1 on 2^NU entries, S layout factors and h's {SKIP} elements")
    E, k, alpha, W = sh.ELL, sh.ELL - PRE_ROUNDS, pts["alpha"], _xor(U)
    x_rest, y_rest = tuple(pts["x_rest"]), tuple(pts["y_rest"])
    zeta_c = tuple(draw([], PRE_ROUNDS))
    eqx, eqy = eq_table(x_rest[:k]), eq_table(list(zeta_c) + list(y_rest[:k]))
    side: dict[int, tuple[int, int]] = {}
    for row, col, a, b in ents:
        if a or b:
            sa, sb = side.get(row, (0, 0))
            side[row] = (sa ^ (eqy[col] if a else 0), sb ^ (eqy[col] if b else 0))
    V = [0] * SKIP
    for row, (sa, sb) in side.items():
        V[row % SKIP] ^= mul(eqx[row >> PRE_ROUNDS], mul(alpha, sa) ^ sb)
    V, L = [mul(W, v) for v in V], skip_weights(pts["z"])
    claim = _dot(eq_table(zeta_c), h)
    if claim != _dot(L, V):
        raise AssertionError("H1: h is not the entries' fold at the rep's points")
    pre, pre_msgs = [], []
    for j in range(PRE_ROUNDS):
        c0 = _xor(mul(L[i], V[i]) for i in range(0, len(V), 2))
        g1 = _xor(mul(L[i], V[i]) for i in range(1, len(V), 2))
        c2 = _xor(mul(L[i] ^ L[i + 1], V[i] ^ V[i + 1]) for i in range(0, len(V), 2))
        if claim != c0 ^ g1:
            raise AssertionError(f"H2 round {j}: the sum is not the claim")
        (r,) = draw([c0, c2], 1)
        claim = c0 ^ mul(claim, r) ^ mul(c2, r ^ mul(r, r))
        L = [x ^ mul(r, x ^ y) for x, y in zip(L[0::2], L[1::2])]
        V = [x ^ mul(r, x ^ y) for x, y in zip(V[0::2], V[1::2])]
        pre.append(r)
        pre_msgs.append((c0, c2))
    (v2,) = V
    if claim != mul(L[0], v2):
        raise AssertionError("H2: the claim is not ŵ(ζ_r) v2")
    draw([v2], 0)
    ax = [1 ^ x for x in pre + list(x_rest[:k])]
    ay = [1 ^ y for y in list(zeta_c) + list(y_rest[:k])]
    last, sc, rk, claim = _entry_rounds([_planes(sh, e) for e in ents], alpha, ax, ay, W, v2, draw, workers)
    u = list(last) + [0] * (sh.PP - sh.P)
    phi = mul(alpha, u[2 * E]) ^ u[2 * E + 1]
    for j in range(E):
        phi = mul(mul(phi, ax[j] ^ u[j]), ay[j] ^ u[E + j])
    Ut, rs = list(U), []
    for j in range(sh.S_LOG):
        a0, a1 = _xor(Ut[0::2]), _xor(Ut[1::2])
        if claim != mul(phi, a0 ^ a1):
            raise AssertionError(f"H3 slot round {j}: the sum is not the claim")
        msg = [mul(phi, a0)] + [0] * sh.d
        (r,) = draw(msg, 1)
        claim = mul(phi, a0 ^ mul(r, a0 ^ a1))
        Ut = [x ^ mul(r, x ^ y) for x, y in zip(Ut[0::2], Ut[1::2])]
        sc.append(msg)
        rs.append(r)
    if claim != mul(Ut[0], phi):
        raise AssertionError("H3: the claim is not Ũ(r_s) Φ(u)")
    zeta_p = tuple(draw(u, sh.P_LOG))
    claim = _dot(eq_table(zeta_p), u)
    draw([claim], 0)
    coins = Coins(alpha, pts["z"], x_rest, y_rest, zeta_c, tuple(pre), tuple(rk + rs), zeta_p, (), tuple(U))
    return coins, {"h": list(h), "pre": pre_msgs, "v2": v2, "sc": sc, "u": u, "claim": claim}
