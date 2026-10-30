"""The served rows' tie to the compute: a window's committed served rows (`served_rows`) held to the positions its steps'
Commit holds, so a prover can't register cheaper rows at serve time than the batch it ran. The six-pass run's rows
(user-1 never admitted) leave step 6's 29 prefill tokens of the honest batch uncovered, and the tie refuses them.

    ServedRowsTie_v2{R, P, G, IW, L, T, IB, PB} (pub, srv, pos_<g>_<k> …, sw_<g> …) -> verdict[16]
        G steps of a P-step window, block j of its P / G
    ServedTieStep_v1{R, L, T, IB, PB} (step, n, chunks, switches) -> (count, pos)
        ServedTieSlots_v1{R, L, T} → IoTieRoute_v1{W = TW + LW, D} → ServedTieScan_v1{R, L, T, IB, PB}
    ServedTieSelect_v1{P, G, SB} (j, srv) -> block j's G step records, and whether j < P / G

**The step-input port**, the Commit's layout the tie reads (vLLM's; `port_chunks`): each step's stream starts with its input
ids, then its positions, each T little-endian elements (IB and PB bytes) with zeros past the step's N tokens, T the engine's
token budget (``max_num_batched_tokens``). So the positions are the stream's bytes [T·IB, T·(IB + PB)), K chunks of the
step's rows-sha512 tree at fixed leaf indices (`geometry`), and the binding map's entries for the ids and the positions
(``step.input/ids`` and ``step.input/positions``) name those ranges. The ingress tie, ``IoTieRequest``, reads the ids
from the same chunks through the same layout (`positions` and `ids`, the shared read).

**The rows** (each zero-padded to whole 16-bit words):
- ``pub``, public: the unit's block j of the window (JW = bitlen(P / G − 1) bits, none when G = P), then each of its G
  steps' token count N (TW bits a step), the binding map's element count of the positions;
- ``srv``, the window's served rows, row w of the served-rows registration (`served_rows`: P steps of R slots of (valid,
  request, kind, start, n), `compile_slice.rows_bits`' packing), whole in every unit so that its b‖c is the
  registration's leaf, checked natively;
- ``pos_<g>_<k>``, the unit's step g's k-th positions chunk, a 4096-byte leaf of the step's tree, its b‖c checked natively
  against that leaf (`chunk_ok`) at its public index, and the tree's root through the step root to the run root;
- ``sw_<g>``, witness: the prover's switches of step g's routing (`route`).

**The statement** (`check`). Per step, the slots in batch order cover the step's tokens in turn: slot k's run is [lo_k,
lo_k + n_k), lo_k the sum of the earlier valid slots' n (an invalid slot's n is taken as 0). Bit 0 ``count``: the block
is one of the window's, and every step's runs end at its N, N ≤ T. Bit 1 ``pos``: every token t < N is in a valid slot's run
and at the position its start puts it, position[t] = start_k + (t − lo_k), the position's high bits zero. ``ACCEPT`` = 0x3.
Tokens from N on are padding and unread. The request ids are not the compute's to show: ``IoTieRequest`` holds each prefill
slot's ids, at its tokens, to its certified request's prompt ids at those positions, and the egress tie each decode slot's
input to the request's last sample.

**The routing** (`ServedTieStep`). ``ServedTieSlots`` makes each slot's record (v, start): v = lo + 1 for a live slot (n > 0
and lo below N and T), else 0. ``IoTieRoute``, a Beneš network over M = 2^D ≥ max(T, R) positions, takes the R
records and M − R zero ones by the prover's switches; any setting is a permutation. ``ServedTieScan`` holds each position t
to v ∈ {0, t + 1} (v = 0 from T on), so every live record lands at its lo and nothing else marks a run's start, and then
reads the tokens in order: token t's position is the start routed to t where a run starts there (always at t = 0), else
token t − 1's position plus one, compared on X = max(LW, TW) + 1 bits with the rest zero. No run can wrap X bits (start +
t − lo < 2^LW + 2^TW), so with the honest switches (`route`) the verdict is `check`'s, and with any switches it accepts
only what `check` accepts (`reference` is the circuit in integers).

**Selection: every unit, always proved** (`Selection` All, as a transfer-gate unit is). The tie is a binding, not a check
on work: a step left untied makes that step's served rows the prover's word, and one such step can lower a window's charge
(the six passes are one). The charge it protects feeds the network verdict, an AND over every window with no ε, so a tie
sampled at ε would leave each untied step's charge free with probability 1 − ε. A window is its P / G units, block j = 0 …
P / G − 1 each once, under one template.

**Cost** (`size`, `step_size`): a step is the routing, about (M / 2)(2D − 1)·(TW + LW) ANDs less the switches a zero record
folds away, and about 3·TW + 2·X + 8·PB ANDs a token for the scan; a unit is G steps and, when G < P, P·TB ANDs to select
its block (TB = R·(1 + LW + TW), the valid, start and n bits a step's tie reads of its served rows; the request ids and
kinds are wired past). N is public, as the Commit's binding map and leaf counts publish it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import _part
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf
from verity.core.protocols.certifier import head_check as HC, io_tie as IT, replay_chain as RC
from verity.core.protocols.certifier.head_check import BIT

STATICS = ("R", "P", "G", "IW", "L", "T", "IB", "PB")
STEP = ("R", "L", "T", "IB", "PB")
CHUNK = 4096
WORD = 16
FLAGS = ("count", "pos")
ACCEPT = (1 << len(FLAGS)) - 1
INST = hm96.SHA512


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def _sizes(S, *names: str) -> None:
    if min(getattr(S, k) for k in names) < 1:
        raise ValueError(f"statics {', '.join(f'{k}={getattr(S, k)!r}' for k in names)}: every size ≥ 1")


def _step_widths(S) -> SimpleNamespace:
    _sizes(S, "R", "L", "T")
    LW, TW = _bitlen(S.L), _bitlen(S.T)
    D = IT.route_depth(S.T, S.R)
    return SimpleNamespace(LW=LW, TW=TW, W=TW + _bitlen(S.R), X=max(LW, TW) + 1, V=TW + LW, D=D, M=1 << D,
                           SW=(2 * D - 1) << (D - 1))


def widths(S) -> SimpleNamespace:
    """The step's widths (`_step_widths`), the unit's blocks U = P / G and their index's bits JW, a step's served-row bits
    SB, and the TB of them its tie reads."""
    Z = _step_widths(S)
    if min(S.P, S.G) < 1 or S.P % S.G:
        raise ValueError(f"a unit is G ≥ 1 steps of a window of P, G dividing P, not P={S.P!r}, G={S.G!r}")
    Z.U = S.P // S.G
    Z.JW = (Z.U - 1).bit_length()
    Z.SB = RC._bits_of(_step_fields(S))
    Z.TB = RC._bits_of(_tie_fields(S))
    return Z


def geometry(S) -> SimpleNamespace:
    """The positions' chunks of a step's tree, [lo, hi), and their first byte within chunk lo."""
    _sizes(S, "T", "IB", "PB")
    o0, o1 = S.T * S.IB, S.T * (S.IB + S.PB)
    lo, hi = o0 // CHUNK, -(-o1 // CHUNK)
    return SimpleNamespace(lo=lo, hi=hi, K=hi - lo, off=o0 - lo * CHUNK)


def _step_fields(S) -> list[tuple[str, int, int]]:
    _sizes(S, "IW")
    Z = _step_widths(S)
    return [("valid", S.R, 1), ("id", S.R, S.IW), ("kind", S.R, 1), ("start", S.R, Z.LW), ("n", S.R, Z.TW)]


def _tie_fields(S) -> list[tuple[str, int, int]]:
    Z = _step_widths(S)
    return [("valid", S.R, 1), ("start", S.R, Z.LW), ("n", S.R, Z.TW)]


def _tie(S, step: Sequence) -> list:
    """A step's served-row bits (`_step_fields`) as the tie reads them (`_tie_fields`): its slots' valid, start and n."""
    return RC._flat(RC._split(list(step), _step_fields(S)), _tie_fields(S))


def _words(bits: int) -> int:
    return WORD * -(-bits // WORD)


def rows(S) -> list[tuple[str, int]]:
    """The unit's rows and their widths in bits."""
    Z, G = widths(S), geometry(S)
    return ([("pub", _words(Z.JW + S.G * Z.TW)), ("srv", _words(S.P * Z.SB))]
            + [(f"pos_{g}_{k}", 8 * CHUNK) for g in range(S.G) for k in range(G.K)]
            + [(f"sw_{g}", _words(Z.SW)) for g in range(S.G)])


# --- the shared read ---------------------------------------------------------------------------------------------------------

def positions(S, chunks: Sequence) -> list[list]:
    """Step's T positions (PB·8 bits each, least significant first) from its K positions chunks' bits: the port's read, any
    bits (circuit wires or ints)."""
    G = geometry(S)
    bits = [b for c in chunks for b in c]
    at = 8 * G.off
    return [list(bits[at + 8 * S.PB * t:at + 8 * S.PB * (t + 1)]) for t in range(S.T)]


def ids(S, chunks: Sequence) -> list[list]:
    """Step's T input ids (IB·8 bits each) from the chunks [0, ⌈T·IB / 4096⌉) of its tree: ``IoTieRequest``'s read of the
    same port."""
    bits = [b for c in chunks for b in c]
    return [list(bits[8 * S.IB * t:8 * S.IB * (t + 1)]) for t in range(S.T)]


# --- the circuit -------------------------------------------------------------------------------------------------------------

def _sig_slots(S):
    Z = _step_widths(S)
    return (("step", Array(RC._bits_of(_tie_fields(S)), BIT)), ("n", Array(Z.TW, BIT))), Array(2 + S.R * Z.V, BIT)


def _build_slots(C, S, step, n):
    Z = _step_widths(S)
    f = RC._split(step, _tie_fields(S))
    N, T = RC._pad(n, Z.W), const_bits(S.T, Z.W)
    lo, recs = [ZERO] * Z.W, []
    for k in range(S.R):
        nk = [C.AND(f["valid"][k][0], b) for b in f["n"][k]]
        live = and_reduce(C, [or_reduce(C, nk), RC._lt(C, lo, N), RC._lt(C, lo, T)])
        recs += [C.AND(live, b) for b in add(C, lo[:Z.TW], const_bits(1, Z.TW), width=Z.TW)] + list(f["start"][k])
        lo = add(C, lo, RC._pad(nk, Z.W), width=Z.W)
    return [[C.AND(RC._eq(C, lo, N), RC._lt(C, T, N).inv()), RC._lt(C, lo, N).inv(), *recs]]


ServedTieSlots = _part(
    "ServedTieSlots", ("R", "L", "T"), _sig_slots, _build_slots,
    "a step's served rows (its slots' valid, start and n) as (count, cover, records): count iff the slots' runs end at the step's N and N ≤ T, cover iff "
    "they reach N, and slot k's record (v, start), v = lo_k + 1 (TW bits) for a live slot (n > 0, lo_k below N and T) and "
    "0 for another, lo_k the earlier valid slots' n summed")


def _sig_scan(S):
    Z, G = _step_widths(S), geometry(S)
    return ((("routed", Array(Z.M, Array(Z.V, BIT))), ("n", Array(Z.TW, BIT)), ("chunks", Array(G.K, Array(8 * CHUNK, BIT))),
             ("cover", BIT)), BIT)


def _build_scan(C, S, *e):
    Z, G = _step_widths(S), geometry(S)
    M, TW, X = Z.M, Z.TW, Z.X
    routed, n, chunks, cover = e[:M], e[M], e[M + 1:M + 1 + G.K], e[-1][0]
    oks, here = [cover], []
    for t in range(M):
        v = list(routed[t][:TW])
        z = RC._zero(C, v)
        if t < S.T:
            here.append(RC._eq(C, v, const_bits(t + 1, TW)))
            oks.append(here[-1] ^ z)
        else:
            oks.append(z)
    prev: list = []
    for t, p in enumerate(positions(S, chunks)):
        low = RC._pad(p[:X], X)
        start = RC._pad(routed[t][TW:], X)
        want = start if t == 0 else mux(C, here[t], start, add(C, prev, const_bits(1, X), width=X))
        ok = C.AND(RC._eq(C, low, want), RC._zero(C, p[X:]))
        oks.append(_or(C, RC._lt(C, const_bits(t, TW), n).inv(), ok))
        prev = low
    return [[and_reduce(C, oks)]]


ServedTieScan = _part(
    "ServedTieScan", ("R", "L", "T", "IB", "PB"), _sig_scan, _build_scan,
    "1 iff cover, each of the M routed records' v is 0 or t + 1 at position t (0 from T on), and each token t below N is at "
    "the start routed to t where v = t + 1 or t = 0, else at token t − 1's position plus one, on X = max(LW, TW) + 1 bits, "
    "its high bits zero")


def _or(C, a, b):
    return C.AND(a.inv(), b.inv()).inv()


def _sig_step(S):
    Z, G = _step_widths(S), geometry(S)
    return ((("step", Array(RC._bits_of(_tie_fields(S)), BIT)), ("n", Array(Z.TW, BIT)),
             ("chunks", Array(G.K, Array(8 * CHUNK, BIT))), ("switches", Array(Z.SW, BIT))), Array(2, BIT))


def _body_step(B, S, step, n, chunks, switches):
    Z, K = _step_widths(S), HC._Consts(B)
    sl = HC._refs(B.call(bind(ServedTieSlots, R=S.R, L=S.L, T=S.T), step, n))
    recs = HC._coll(Array(Z.M, Array(Z.V, BIT)), sl[2:] + K.bits(0, (Z.M - S.R) * Z.V))
    routed = B.call(bind(IT.IoTieRoute, W=Z.V, D=Z.D), recs, switches)
    pos = B.call(bind(ServedTieScan, R=S.R, L=S.L, T=S.T, IB=S.IB, PB=S.PB), routed, n, chunks, HC._coll(BIT, [sl[1]]))
    return HC._coll(Array(2, BIT), [sl[0], *HC._refs(pos)])


ServedTieStep = CompositeDefinition(
    "ServedTieStep", 1, STEP, _sig_step, _body_step, conformance="exact-model-tested",
    doc="(count, pos) of one step's tie: its slots' records (`ServedTieSlots`) routed by the switches (`IoTieRoute`) to "
        "M = 2^D ≥ max(T, R) positions and scanned against its committed positions (`ServedTieScan`)")


def _check_select(S) -> None:
    if min(S.P, S.G, S.SB) < 1 or S.P % S.G or S.P // S.G < 2:
        raise ValueError(f"a selection is of a block of G steps among P ≥ 2·G, G dividing P, not P={S.P!r}, G={S.G!r}, "
                         f"SB={S.SB!r}")


def _sig_select(S):
    _check_select(S)
    return ((("j", Array((S.P // S.G - 1).bit_length(), BIT)), ("srv", Array(S.P * S.SB, BIT))), Array(S.G * S.SB + 1, BIT))


def _build_select(C, S, j, srv):
    U, width = S.P // S.G, S.G * S.SB
    hot = [RC._eq(C, j, const_bits(i, len(j))) for i in range(U)]
    out = []
    for b in range(width):
        acc = ZERO
        for i in range(U):
            acc = acc ^ C.AND(hot[i], srv[i * width + b])
        out.append(acc)
    inside = ZERO
    for h in hot:
        inside = inside ^ h
    return [out + [inside]]


ServedTieSelect = _part(
    "ServedTieSelect", ("P", "G", "SB"), _sig_select, _build_select,
    "block j's G step records of SB bits from a window's P, and 1 iff j < P / G (else the records are zero)")


def _sig(S):
    return tuple((r, Array(n, BIT)) for r, n in rows(S)), Array(16, BIT)


def _all(B, refs: list):
    if len(refs) == 1:
        return refs[0]
    return HC._refs(B.call(BL.and_gate(len(refs)), HC._coll(Array(len(refs), BIT), refs)))[0]


def _body(B, S, pub, srv, *rest):
    Z, G, K = widths(S), geometry(S), HC._Consts(B)
    pr, sr = HC._refs(pub), HC._refs(srv)
    pos, sw = rest[:S.G * G.K], rest[S.G * G.K:]
    counts, poss = [], []
    tie = [b for i in range(S.P) for b in _tie(S, sr[i * Z.SB:(i + 1) * Z.SB])]
    if Z.U > 1:
        sel = HC._refs(B.call(bind(ServedTieSelect, P=S.P, G=S.G, SB=Z.TB), HC._coll(Array(Z.JW, BIT), pr[:Z.JW]),
                              HC._coll(Array(S.P * Z.TB, BIT), tie)))
        recs = sel[:-1]
        counts.append(sel[-1])
    else:
        recs = tie
    step = bind(ServedTieStep, **{k: getattr(S, k) for k in STEP})
    for g in range(S.G):
        o = HC._refs(B.call(step, HC._coll(Array(Z.TB, BIT), recs[g * Z.TB:(g + 1) * Z.TB]),
                            HC._coll(Array(Z.TW, BIT), pr[Z.JW + g * Z.TW:Z.JW + (g + 1) * Z.TW]),
                            HC._coll(Array(G.K, Array(8 * CHUNK, BIT)), [r for c in pos[g * G.K:(g + 1) * G.K] for r in HC._refs(c)]),
                            HC._coll(Array(Z.SW, BIT), HC._refs(sw[g])[:Z.SW])))
        counts.append(o[0])
        poss.append(o[1])
    return HC._coll(Array(16, BIT), [_all(B, counts), _all(B, poss), *K.bits(0, 16 - len(FLAGS))])


ServedRowsTie = CompositeDefinition(
    "ServedRowsTie", 2, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="block j of a window's served rows tied to its G steps' committed positions (verdict bits count, pos): the block's "
        "step records (`ServedTieSelect` when G < P) and each step's tie (`ServedTieStep`), the steps' bits ANDed")


def statics(S, IB: int = 8, PB: int = 8, G: int | None = None) -> dict:
    """The tie's statics for a compile slice's statics `S` (its R, P, IW, and ReplayStep_v1's L and T), the port's element
    bytes IB and PB, and a unit's G steps (the whole window by default)."""
    return {"R": S.R, "P": S.P, "G": S.P if G is None else G, "IW": S.IW, "L": S.L, "T": S.T, "IB": IB, "PB": PB}


def unit(**st):
    return bind(ServedRowsTie, **st)


# --- the specification -------------------------------------------------------------------------------------------------------

def _int(bits: Sequence[int]) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def _fields(S, bits: Sequence[int]) -> dict[str, list[int]]:
    return {k: [_int(e) for e in v] for k, v in RC._split(list(bits), _step_fields(S)).items()}


def _runs(S, f: dict) -> tuple[list[tuple[int, int, int]], int]:
    """Each slot's run (lo, n, start), an invalid slot's n 0, and where the runs end."""
    lo, runs = 0, []
    for k in range(S.R):
        n = f["n"][k] if f["valid"][k] else 0
        runs.append((lo, n, f["start"][k]))
        lo += n
    return runs, lo


def _steps(S, unit_rows: dict):
    """Whether the unit's block is the window's, and each of its steps' fields, N, positions and switches."""
    Z, G = widths(S), geometry(S)
    j, w = _int(unit_rows["pub"][:Z.JW]), S.G * Z.SB
    block = list(unit_rows["srv"][j * w:(j + 1) * w]) if j < Z.U else [0] * w
    out = []
    for g in range(S.G):
        N = _int(unit_rows["pub"][Z.JW + g * Z.TW:Z.JW + (g + 1) * Z.TW])
        got = [_int(p) for p in positions(S, [unit_rows[f"pos_{g}_{k}"] for k in range(G.K)])]
        out.append((_fields(S, block[g * Z.SB:(g + 1) * Z.SB]), N, got, list(unit_rows[f"sw_{g}"][:Z.SW])))
    return j < Z.U, out


def check(S, unit_rows: dict) -> dict[str, bool]:
    """The statement's flags on a unit's rows (bits each), in integers: the switches unread."""
    inside, steps = _steps(S, unit_rows)
    count, pos = inside, True
    for f, N, got, _sw in steps:
        runs, end = _runs(S, f)
        count &= end == N and N <= S.T
        pos &= end >= N and all(any(a <= t < a + n and got[t] == s + t - a for a, n, s in runs) for t in range(min(N, S.T)))
    return {"count": count, "pos": pos}


def _live(S, a: int, n: int, N: int) -> bool:
    return n > 0 and a < N and a < S.T


def reference(S, unit_rows: dict) -> dict[str, bool]:
    """The circuit's flags in integers, the switches read: each step's records routed (`io_tie.benes_apply`) and scanned."""
    Z = widths(S)
    inside, steps = _steps(S, unit_rows)
    count, pos = inside, True
    for f, N, got, sw in steps:
        runs, end = _runs(S, f)
        count &= end == N and N <= S.T
        xs = [(a + 1 if _live(S, a, n, N) else 0, s) for a, n, s in runs] + [(0, 0)] * (Z.M - S.R)
        routed = IT.benes_apply(xs, sw)
        ok = end >= N and all(v in (0, t + 1) if t < S.T else v == 0 for t, (v, _s) in enumerate(routed))
        for t in range(min(N, S.T)):
            v, s = routed[t]
            ok &= got[t] == (s if t == 0 or v == t + 1 else (got[t - 1] + 1) % (1 << Z.X))
        pos &= ok
    return {"count": count, "pos": pos}


def route(S, f: dict, N: int) -> list[int]:
    """The prover's switches for a step: each live slot's record to its lo, the other records to the rest in order."""
    Z = _step_widths(S)
    perm = [-1] * Z.M
    for k, (a, n, _s) in enumerate(_runs(S, f)[0]):
        if _live(S, a, n, N):
            perm[k] = a
    rest = iter(sorted(set(range(Z.M)) - set(perm)))
    return IT.benes([p if p >= 0 else next(rest) for p in perm])


def verdict(flags: dict) -> int:
    return sum(int(flags[f]) << k for k, f in enumerate(FLAGS))


def _le(values: Sequence[int], width: int, count: int) -> bytes:
    if len(values) > count:
        raise ValueError(f"{len(values)} tokens over the budget T = {count}")
    return b"".join(int(v).to_bytes(width, "little") for v in values) + bytes(width * (count - len(values)))


def port_chunks(S, step_ids: Sequence[int], step_positions: Sequence[int]) -> list[bytes]:
    """The step-input port, the head of a step's stream as the tie reads it: the ids, then the positions, padded to T each
    and to whole chunks."""
    if len(step_ids) != len(step_positions):
        raise ValueError("one position a token")
    head = _le(step_ids, S.IB, S.T) + _le(step_positions, S.PB, S.T)
    head += bytes(-len(head) % CHUNK)
    return [head[k:k + CHUNK] for k in range(0, len(head), CHUNK)]


def _bits(row: bytes) -> list[int]:
    return [(x >> j) & 1 for x in row for j in range(8)]


@dataclass(frozen=True)
class Case:
    rows: dict
    flags: dict
    verdict: int


def unit_rows(S, srv_bits: Sequence[int], steps: Sequence[tuple[int, Sequence[bytes]] | None], j: int = 0) -> dict:
    """Block j's rows for a window: the served rows as registered (`served_rows.window_bits`), and each of the window's
    steps' N and the head of its stream (its chunks [0, `geometry`.hi)), an idle step (None, or past the list) N = 0 and
    zero chunks; the switches the prover's (`route`)."""
    Z, G = widths(S), geometry(S)
    steps = (list(steps) + [None] * S.P)[j * S.G:(j + 1) * S.G]
    Ns = [s[0] if s else 0 for s in steps]
    w = S.G * Z.SB
    block = (list(srv_bits[j * w:(j + 1) * w]) + [0] * w)[:w]
    out = {"pub": _bits_int(j, Z.JW) + [b for n in Ns for b in _bits_int(n, Z.TW)], "srv": list(srv_bits)}
    for g, s in enumerate(steps):
        for k in range(G.K):
            out[f"pos_{g}_{k}"] = _bits(s[1][G.lo + k]) if s else [0] * (8 * CHUNK)
        out[f"sw_{g}"] = route(S, _fields(S, block[g * Z.SB:(g + 1) * Z.SB]), Ns[g])
    for r, n in rows(S):
        out[r] = list(out[r]) + [0] * (n - len(out[r]))
    return out


def case(S, srv_bits: Sequence[int], steps: Sequence[tuple[Sequence[int], Sequence[int]] | None], j: int = 0) -> Case:
    """Block j's unit for a window: the served rows as registered and each executed step's committed (ids, positions)."""
    out = unit_rows(S, srv_bits, [(len(s[1]), port_chunks(S, *s)) if s else None for s in steps], j)
    flags = check(S, out)
    return Case(out, flags, verdict(flags))


def _bits_int(v: int, w: int) -> list[int]:
    if not 0 <= v < 1 << w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def evaluate(st: dict, cases: Sequence[Case]) -> list[int]:
    S = SimpleNamespace(**st)
    args = [[c.rows[r] for c in cases] for r, _n in rows(S)]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


def _build_route(C, S, xs: list, sw: list) -> list:
    """`IoTieRoute{W, D}` on C's wires as C-Flock lowers it, each `IoTieSwitch` activation by its builder, the switch bits
    in `IT.benes`' order: the input column, the upper half's, the lower half's, the output column."""
    it = iter(sw)

    def column(pairs: list) -> list:
        return [IT.IoTieSwitch.build(C, S, a, b, [next(it)]) for a, b in pairs]

    def run(v: list) -> list:
        if len(v) == 2:
            return column([(v[0], v[1])])[0]
        first = column(list(zip(v[0::2], v[1::2])))
        up, down = run([a for a, _ in first]), run([b for _, b in first])
        return [w for p in column(list(zip(up, down))) for w in p]

    return run(xs)


def _count(build, params) -> int:
    C = Circuit(cse=True)
    ins = [C.input(f"p{i}", w) for i, t in enumerate(params) for w in BL.elements(t)]
    out = build(C, *ins)
    C.output("out", [b for e in out for b in e])
    return C.counts()["ands"]


@lru_cache(maxsize=None)
def _step_ands(key: tuple) -> int:
    """The step's parts counted as C-Flock lowers them, each activation by its builder on its own wires: the slots, the
    routing with its M − R zero records constant, the scan."""
    S = SimpleNamespace(**dict(zip(STEP, key)))
    Z = _step_widths(S)

    def route(C, *e):
        return _build_route(C, SimpleNamespace(W=Z.V), [*e[:S.R], *[[ZERO] * Z.V] * (Z.M - S.R)], e[-1])

    return (_count(lambda C, *e: _build_slots(C, S, *e), [t for _n, t in _sig_slots(S)[0]])
            + _count(route, [Array(Z.V, BIT)] * S.R + [Array(Z.SW, BIT)])
            + _count(lambda C, *e: _build_scan(C, S, *e), [t for _n, t in _sig_scan(S)[0]]))


def step_size(**st) -> dict[str, int]:
    """One step's ANDs (`ServedTieStep`)."""
    return {"ands": _step_ands(tuple(st[k] for k in STEP))}


def size(**st) -> dict[str, int]:
    """A unit's ANDs: G steps, the block's selection when G < P, and the steps' flags ANDed."""
    S = SimpleNamespace(**st)
    Z = widths(S)
    Q = SimpleNamespace(P=S.P, G=S.G, SB=Z.TB)
    sel = _count(lambda C, *e: _build_select(C, Q, *e), [t for _n, t in _sig_select(Q)[0]]) if Z.U > 1 else 0
    G1 = S.G + (Z.U > 1)
    return {"ands": S.G * step_size(**st)["ands"] + sel + (G1 - 1) + (S.G - 1)}


# --- the verifier's native check ---------------------------------------------------------------------------------------------

def chunk_ok(domain: CommitmentDomain, index: int, bc: bytes, path: Sequence[bytes], tree_root: bytes) -> bool:
    """Whether a unit's chunk row, by its b‖c, is leaf ``index`` of the step tree over `domain` (the rows-sha512 step's) with
    root `tree_root`, `path` its siblings leaves first."""
    if not 0 <= index < domain.count or len(path) != merkle_depth(domain.count):
        return False
    digest, cursor = row_leaf(domain, index, INST.tree_leaf(INST.default_key, bytes(bc)), SCHEMA_HM96_SHA512_ROW_V2), index
    for d, sib in enumerate(path):
        digest = domain.node(d, cursor >> 1, *((digest, sib) if cursor % 2 == 0 else (sib, digest)))
        cursor >>= 1
    return digest == bytes(tree_root)
