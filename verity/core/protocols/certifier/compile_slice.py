"""A slice of the in-proof compile unit as one Boolean Definition: the pieces of the compile unit's statement (the
certifier PROTOCOL's "Advice v1") composed whole for one window of a few steps, so the composition is proved end to
end before the chain splits it into one unit a step.

    CompileSlice_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CN, CTW, CU}
        (pub, adv, cert, srv, cin, cout, wit) -> verdict[16]
    CompileSliceBlocks_v1{R, …, CU, G}
        (pub, adv, cert, srv0, …, srv{P/G − 1}, cin, cout, wit) -> verdict[16]

The second is the first with its served rows in P / G rows of G steps (G < P), as the served rows' registration commits a
window's blocks (`served_rows.register`'s G); at G = P the unit is the first, so its template id doesn't name G.

**The rows**, each a field list (elements least significant bit first) zero-padded to whole 16-bit words:
- ``pub``, the public input: the declared charge ``c`` (`arith_encode.CW` bits) and the registered book's tables
  (`charge_chain.book_fields(D)`: n_arrived's bins, its total, and the zero counts' frequencies and totals);
- ``adv``, the committed advice, a row under the advice's hiding root (`verity.ml.commitments.advice`): the window's raw
  decision bits ``dec`` (D), their ``count`` and the window's executed step count ``steps``;
- ``cert``, the window's certified arrivals (`replay_chain.arrival_fields`: K slots of valid, id, P, O) and ``tok``,
  each slot's first CU·B certified token ids (its prompt, then its outputs; zero past them), a row the verifier holds to
  the certified record's commitment (its b‖c);
- ``srv``, the served rows: the step rows the served window ran (P steps of `rows_bits`, idle ones zero; `srv_row`;
  each step's rows at the runner's batch positions), a row the verifier holds by its b‖c to the window's row of the
  served rows' registration (`served_rows`), committed on the Commit side before the draw. At G < P they are P / G
  rows ``srv0`` … (`srv_rows`), row j steps [j·G, (j + 1)·G) and held to window w's leaf w·(P / G) + j
  (`served_rows.leaf`); step u is checked against row u // G;
- ``cin`` and ``cout``, the charge's carry-in and carry-out (`charge_chain.encode`'s 96 bits): the window's reads of
  rows 0 and 1 of its `charge_chain.VALUE` value, which the verifier holds by their b‖c to the session's carry chain
  (`charge_chain.check_chain`);
- ``wit``, the witness: the state rows S_1 … S_{P+1} and the window row W_1 (`replay_chain`'s layouts), each step's
  arrivals ``arr`` and decision bits ``bits``, each step's prefix hits ``hit`` (its W + A candidates') and the cache rows
  C_0 … C_P (`cache_step.cache_fields`: the table and each running slot's walk entry), and at AS = 1 each step's
  ``map`` from batch positions to slots (R rows of R bits, bit s of row b set iff served row b is slot s's;
  `slot_map`).

**The circuit.** `replay_chain.ReplayLoad` from the session's empty rows (S_0 and W_0, constants) to S_1 and W_1 with
``cert``'s arrivals and ``adv``'s decisions. Then per step u = 1 … P: `replay_step.ReplayStep_v1` on S_u's state with
``arr[u]``, ``bits[u]`` and ``hit[u]``, and `replay_chain.ReplayAdvance` on the same ``arr``/``bits`` wires, which holds
them to W_1 at S_u's heads and S_{u+1} to the step's next state. The step's `cache_step.CacheStep_v1` calls
(`cache_step.build_scan_step`) run on C_u from the same wires: per row slot that admits, a lookup from ROOT whose hit is
``hit[u]``'s at the candidate's index, then the units the row completes from that lookup's entry or the slot's; then a
lookup of the first candidate left waiting. Each call's units are its request's tokens in ``cert``, selected by the
request's id, so neither the table's tokens nor a hit's are witness; C_0 is all zero and C_{u+1} is the calls' table and
next slots (C_u on an idle step). Each step's rows (valid, request, kind, start, n), zero on an idle step, packed in
`_out_fields`' order (`rows_bits`), are held equal to ``srv``'s: slot for slot at AS = 0, and by request at AS = 1,
where the runner's batch isn't the scheduler's slots (it can order a step's rows otherwise, and a request holds its
slot with no row while its last output is in flight). There ``map`` is held one-to-one from the batch positions onto
the slots with rows (at most one bit set in each of its rows and columns, and column s's set iff slot s has a row, so
none on an idle step), and each served row to its slot's row, zero where its row of ``map`` is. So
the map is unique, the identity where the batch is in slot order, and the served rows are the step's in some order.
The charge is `charge_chain.ChargeLink_v1{D}`
(`build_link`) on ``pub``'s book, ``cin`` and ``cout``, over ``adv``'s decisions with en_i = i < ``count``: the window
charges the decisions it commits, in commit order (advice v1). ReplayLoad appends them after the decisions the last
window left unread, and ReplayAdvance reads the window row's in turn, so the scan reads them in that order; it only
consumes them, and `replay_chain.ReplayClose` refuses a session that leaves one unread.

**The verdict** (16 bits, the unit's one output; `ACCEPT` = 0x3F): bit 0 ``load``, ReplayLoad's ok; bit 1 ``steps``,
every ReplayAdvance's ok (with each active step's ReplayStep ok); bit 2 ``code``, ChargeLink_v1's ok (the book's model
and the count in range, and ``cout`` the charge's carry' from ``cin``); bit 3 ``charge``, its bits ≤ c; bit 4 ``rows``,
the steps' rows are ``srv``'s (through ``map`` at AS = 1); bit 5 ``cache``, C_0 empty and every active step's calls ok with C_{u+1} their output.
The rest are 0.

**The reference** is `check`, the flags the circuit's pieces' references give any unit's rows; `case` builds the rows of
a prover's run (`scan`, with the choices free in a piece alone: the prefix hits and
the heads' moves, a made-up table or made-up tokens, the advice it commits and the carry it charges from) against the
served rows it puts in ``srv``, and `check`'s verdict on them. No witness field carries a context, a frequency, an
enable, a consumed count or a carry, and the hits are the cache's: ChargeLink_v1 reads its contexts and estimates from
``cin`` and the committed decisions, its enables are ``count``'s, the heads move by ReplayStep_v1's own ``taken`` and
``used``, and each hit ReplayStep_v1 reads is the one its lookup checks as the longest cached prefix of the request's
certified tokens at the table the step reads, which chains from the empty one. On corpus-smallkv's first seven steps,
with the served rows in ``srv``, a prefix hit is refused by ``cache`` and ``rows`` (0x0F); made-up tokens by ``rows``
(0x2F); a decision read again and an arrival left queued by ``steps`` and ``rows`` (0x2D). A made-up table that buys
request 1 a free 16-token hit at step 6 is refused by ``cache`` alone (0x1F) even where the server served that hit, its
own rows in ``srv``. Advice that drops a decision the scan asks (a cleared enable) is refused by ``load`` when the
window row keeps it and by ``steps`` when the scan reads past the count; a carry-out other than the charge's carry' by
``code``. Free returns aren't expressible on the slice: it has no stopping request. Advice that registers six passes
where the served window admitted user-1 at the sixth ask is refused by ``rows`` (0x2F) with the served rows in
``srv``; with its own rows there, the circuit accepts and the native
b‖c check refuses ``srv``. A restarted carry-in (the priors, or counts fitted to the window) is accepted by the unit,
which takes any carry-in, and refused by `charge_chain.check_chain` (``CH-carry``) on its b‖c.

**Stand-ins** (named, each a piece not ready):
- *the evictable and returned counts*: the derived evictable and returned-block counts are held at 0 (ReplayStep_v1's
  ``evict`` and ``ret``), true of a slice that stops or preempts nothing; on one that does, the rows differ from the
  clear recompile's and ``rows`` refuses. `compile_slice_v2.CompileSlice_v2` derives them from vLLM's block pool;
- *the still-sent fields*: ``adv`` holds only the decisions, the count and the step count, so ChargeLink_v1 codes the
  window's other counts as 0 (`arith_encode.ZERO_COUNTS`);
- *one window*: the slice is a window mid-session from the empty rows, so the decisions its steps leave unread carry
  (`replay_chain`'s load) and no `ReplayClose` follows it;
- *the binding of ``adv``, ``cert``, ``srv``, ``cin`` and ``cout``*: their rows' b‖c, which the verifier checks
  natively against the advice registration's leaf, the certified record's commitment, the served rows' leaf and the
  session's carry chain; no gate hashes them. That the served rows are the rows the served compute ran is the step
  tree's to bind (`served_rows`).

**Every advised window is proved**, never sampled: the carry chain binds only proved units (an unproved window's
carry-out is any the prover likes), and ``en`` is the slice's scan's (the certifier PROTOCOL's A3,
`protocol/compile-slice.md`).

**Statics**: ReplayStep_v1's (R, W, A and the engine's M, T, L, B, H, CH, WM, FULL, PC, NB, and IW), the chain's Q, D,
P, K (`replay_chain.Shape`), and the cache's CN entries, CTW a token id's width and CU the units of each arrival's tokens
``cert`` holds (`cache_step.scan_widths`). A row past CU units, or a table past CN entries, is refused: the slice's size
class. Without prefix caching (PC = 0) the engine caches nothing and ReplayStep_v1 reads no hits, so the slice has no
table, hits or tokens (``cert`` is the arrivals, ``wit`` has no ``hit`` or ``cache``), makes no CacheStep_v1 call, and
``cache`` is 1; CN, CTW and CU are then 0. A window smaller than the class is padded: empty arrival slots, decisions
past the count (en 0) and idle steps. G, the steps a ``srv`` row holds, is public and the served rows' registration's
(`served_rows.Served.record`); P unless given (`statics`, `with_block`), it divides P, and the statics hold it only
below P. The served bits and the unit's ANDs are the same at every G: a block's row is its steps' bits, padded to words.

**Cost** (`size`): the cache calls dominate where there is a cache. A unit walked costs about CN·(2·(B·CTW + EW) + 2·IW)
ANDs (`cache_step`'s cost; a lookup, which writes nothing, about half), and each step's three lookups walk min(CN + 1, CU)
units and its two computes ⌊(B − 1 + T) / B⌋. On corpus-smallkv's class (CN 24, CU 25, B 16) a step's calls are 1.06M
ANDs and the unit about 7.6M, past circuit-check's lowering: its circuit = reference tests, sidecar and ZK run
(`benchmarks/network_traces/compile_slice_zk.py`) are at the demonstration class (`demo`, 59,341 ANDs) and at the
served vLLM run's sizes under a sync scheduler (the sync class, no prefix caching: 857,866 ANDs, ChargeLink_v1{32}'s
97,159 and its 5,509 copies of the words it carries among them, `charge_chain.size`'s ``carry``), and at the corpus
class the run's staged verdict words are the reference's. That run's engine class is AS = 1, its AsyncScheduler, where
the ZK run proves w212. At AS = 1 ``rows`` is ``map``'s: R² products of a slot's row and 2·R·(R − 1) one-to-one ANDs a
step in place of the slot-for-slot comparison (`demo_async`, 32,236 ANDs; its async window 1 at its own statics,
314,936, and at its engine class 1,021,302, of them 123,120 the map's).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.core.protocols.certifier import arith_encode as AE, cache_step as CST, charge_chain as CC
from verity.core.protocols.certifier import replay_chain as RC, replay_step as RS
from verity.core.protocols.certifier.head_check import BIT, _part

STATICS = RS.STATICS + ("Q", "D", "P", "K", "CN", "CTW", "CU")
#: the statics of the slice whose ``srv`` is P / G rows (`CompileSliceBlocks`), G < P
BLOCK_STATICS = STATICS + ("G",)
CW = AE.CW
FLAGS = ("load", "steps", "code", "charge", "rows", "cache")
ACCEPT = (1 << len(FLAGS)) - 1
#: the rows at G = P (`row_names` at any G)
ROWS = ("pub", "adv", "cert", "srv", "cin", "cout", "wit")
ROW_FIELDS = ("row_valid", "row_id", "row_kind", "row_start", "row_n")
WORD = 16


def chain(S) -> RC.Shape:
    return RC.Shape(**{k: getattr(S, k) for k in RC.STATICS})


def step_statics(S) -> SimpleNamespace:
    return SimpleNamespace(**{k: getattr(S, k) for k in RS.STATICS})


def block(S) -> int:
    """G, the steps a ``srv`` row holds: P unless the statics give it, and dividing P."""
    G = getattr(S, "G", S.P)
    if not (isinstance(G, int) and 1 <= G <= S.P and S.P % G == 0):
        raise ValueError(f"a ``srv`` row is G steps, G dividing P = {S.P}, not G = {G!r}")
    return G


def srv_names(S) -> tuple[str, ...]:
    """The ``srv`` rows: ``srv`` at G = P, else ``srv0`` … ``srv{P/G − 1}``, row j steps [j·G, (j + 1)·G)."""
    U = S.P // block(S)
    return ("srv",) if U == 1 else tuple(f"srv{j}" for j in range(U))


def row_names(S) -> tuple[str, ...]:
    """The unit's rows in order: `ROWS`, with ``srv`` the P / G rows of `srv_names`."""
    return ROWS[:3] + srv_names(S) + ROWS[4:]


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    if not S.PC and (S.CN or S.CTW or S.CU):
        raise ValueError(f"statics {dict(S.__dict__)}: without prefix caching CN, CTW and CU are 0")
    Sc = chain(S)
    Z = RC.widths(Sc)
    cached = [("hit", S.P * (S.W + S.A), RS.widths(step_statics(S)).HW), ("cache", S.P + 1, CST.cache_width(S))]
    return {"pub": [("c", 1, CW)] + CC.book_fields(S.D),
            "adv": [("dec", S.D, 1), ("count", 1, Z.DW), ("steps", 1, Z.PW)],
            "cert": RC.arrival_fields(Sc) + ([("tok", S.K * S.CU * S.B, S.CTW)] if S.PC else []),
            **{r: [("rows", block(S), step_row_bits(S))] for r in srv_names(S)},
            "cin": [("carry", 1, CC.ROW_BITS)],
            "cout": [("carry", 1, CC.ROW_BITS)],
            "wit": [("state", S.P + 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)),
                    ("arr", S.P, RC._bits_of(RC.front_fields(Sc))), ("bits", S.P, S.A)] + (cached if S.PC else [])
                   + ([("map", S.P * S.R, S.R)] if S.AS else [])}


def row_bits(S) -> dict[str, int]:
    return {r: _words_of(RC._bits_of(fs)) for r, fs in fields(S).items()}


def _words_of(n: int) -> int:
    """`n` bits zero-padded to whole words."""
    return WORD * -(-n // WORD)


def step_row_bits(S) -> int:
    """One step's rows in bits: R × (valid, request, kind, start, n)."""
    Z = RS.widths(step_statics(S))
    return S.R * (1 + S.IW + 1 + Z.LW + Z.TW)


# --- the circuit ------------------------------------------------------------------------------------------------------------

def _consts(bits: Sequence[int]) -> list:
    return [ONE if b else ZERO for b in bits]


def _by_slot(S, flat: Sequence) -> list[list]:
    """One step's rows packed field by field (`rows_bits`' layout) as R rows (valid, request, kind, start, n)."""
    Z = RS.widths(step_statics(S))
    out, at = [[] for _ in range(S.R)], 0
    for wd in (1, S.IW, 1, Z.LW, Z.TW):
        for p in range(S.R):
            out[p] += list(flat[at:at + wd])
            at += wd
    return out


def _over_one(C, xs: Sequence) -> list:
    """Wires one of which is 1 iff more than one of `xs` is."""
    seen, bad = xs[0], []
    for x in xs[1:]:
        both = C.AND(seen, x)
        bad.append(both)
        seen = seen ^ x ^ both
    return bad


def _build_map(C, S, rep: Sequence, srv: Sequence, m: Sequence[Sequence]) -> list:
    """The mismatch wires of one step's served rows `srv` (by batch position) against the scan's rows `rep` (by slot)
    through the witnessed map `m` (m[b][s] = 1: served row b is slot s's): every wire 0 iff each row and each column of
    `m` holds at most one 1, column s holds one iff slot s has a row, and each served row is its slot's (zero where its
    row of `m` is)."""
    R = S.R
    bad = [x for b in range(R) for x in _over_one(C, m[b])]
    bad += [x for s in range(R) for x in _over_one(C, [m[b][s] for b in range(R)])]
    for s in range(R):
        col = ZERO
        for b in range(R):
            col = col ^ m[b][s]
        bad.append(col ^ rep[s][0])
    for b in range(R):
        sel, valid = [ZERO] * (len(rep[0]) - 1), ZERO
        for s in range(R):
            valid = valid ^ m[b][s]
            sel = [x ^ C.AND(m[b][s], y) for x, y in zip(sel, rep[s][1:])]
        bad += [x ^ y for x, y in zip(srv[b], [valid] + sel)]
    return bad


def _build(C, S, *ins):
    Sc, St = chain(S), step_statics(S)
    ZS = RS.widths(St)
    F = fields(S)
    split = {r: RC._split(x, F[r]) for x, r in zip(ins, row_names(S))}
    p, a, c, ci, co, w = (split[r] for r in ("pub", "adv", "cert", "cin", "cout", "wit"))
    served = [e for r in srv_names(S) for e in split[r]["rows"]]
    states, win = w["state"], w["window"][0]
    load_ok = RC._build_load(C, Sc, _consts(RC.encode(RC.empty(Sc), Sc)), _consts(RC.encode_window(RC.Window(), Sc)),
                             RC._flat(c, RC.arrival_fields(Sc)), [e[0] for e in a["dec"]], a["count"][0], a["steps"][0],
                             states[0], win)[0]
    CA = S.W + S.A
    if S.PC:
        tok = c["tok"]
        cert = {"valid": [e[0] for e in c["a_valid"]], "id": c["a_id"],
                "tok": [[[x for t in tok[(k * S.CU + j) * S.B:(k * S.CU + j + 1) * S.B] for x in t] for j in range(S.CU)]
                        for k in range(S.K)]}
        caches = w["cache"]
    cache_ok = [RC._zero(C, caches[0])] if S.PC else []

    steps_ok, rows, mism = [], [], []
    for u in range(S.P):
        cur, nxt, arr, bits = states[u], states[u + 1], w["arr"][u], w["bits"][u]
        r, fa = RC._split(cur, RC.row_fields(Sc)), RC._split(arr, RC.front_fields(Sc))
        hit = w["hit"][u * CA:(u + 1) * CA] if S.PC else [[ZERO] * ZS.HW for _ in range(CA)]
        elems = []
        for name, n, wd in RS._in_fields(St):
            if name in r:
                elems += r[name]
            elif name in fa:
                elems += fa[name]
            elif name == "bits":
                elems += [[b] for b in bits]
            elif name == "hit":
                elems += hit
            else:
                elems += [[ZERO] * wd for _ in range(n)]
        out = RS._split(RS._out_fields(St), RS._build(C, St, *elems))
        stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
        ok, active = RC._build_advance(C, Sc, out["ok"][0], cur, win, stepped, arr, bits, out["taken"][0],
                                       out["used"][0], nxt)
        act = active[0]
        steps_ok.append(ok[0])
        mine = [C.AND(act, x) for name in ROW_FIELDS for e in out[name] for x in e]
        if S.AS:
            mism += _build_map(C, S, _by_slot(S, mine), _by_slot(S, served[u]), w["map"][u * S.R:(u + 1) * S.R])
        else:
            rows += mine
        if S.PC:
            quiet = C.AND(or_reduce(C, [e[0] for e in r["run_valid"]]).inv(), r["wait_valid"][0][0].inv())
            forced = C.AND(quiet, fa["arr_valid"][0][0])
            c_ok, c_next = CST.build_scan_step(C, S, caches[u], r, fa, forced, out, hit, cert)
            cache_ok += [C.AND(act, c_ok.inv()).inv(), RC._eq(C, mux(C, act, c_next, caches[u]), caches[u + 1])]

    count = a["count"][0]
    en = [RC._lt(C, const_bits(k, len(count)), count) for k in range(S.D)]
    book = [e for name, _n, _w in CC.book_fields(S.D) for e in p[name]]
    code_ok, charged = CC.build_link(C, S.D, book, ci["carry"][0], co["carry"][0], en, [e[0] for e in a["dec"]])
    charge_ok = RC._lt(C, p["c"][0], charged).inv()

    rows_ok = RC._zero(C, mism) if S.AS else RC._eq(C, rows, [x for e in served for x in e])

    flags = [load_ok, and_reduce(C, steps_ok), code_ok, charge_ok, rows_ok, and_reduce(C, cache_ok)]
    return [flags + [ZERO] * (16 - len(flags))]


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in row_names(S)), Array(16, BIT)


_DOC = ("a slice of the compile unit, one window whole: ReplayLoad, then per step ReplayStep_v1 and ReplayAdvance on the "
        "same arrival and decision wires, CacheStep_v1's calls on the same wires and the certified tokens checking the "
        "hits ReplayStep_v1 reads, the committed decisions charged by ChargeLink_v1 from the window's carry-in to its "
        "carry-out and held to the declared charge, and the step rows held to the committed served rows (verdict bits "
        "load, steps, code, charge, rows, cache)")
CompileSlice = _part("CompileSlice", STATICS, _sig, _build, _DOC)
CompileSlice.build = _build  # type: ignore[attr-defined]
CompileSliceBlocks = _part("CompileSliceBlocks", BLOCK_STATICS, _sig, _build,
                           _DOC + "; its served rows are P / G rows of G steps each, step u held to row u // G")
CompileSliceBlocks.build = _build  # type: ignore[attr-defined]


def _refuse_blocks(b: dict) -> str | None:
    if "G" in b and "P" in b and (not 1 <= b["G"] < b["P"] or b["P"] % b["G"]):
        return f"G = {b['G']} isn't a proper divisor of P = {b['P']} (at G = P the unit is CompileSlice_v1)"
    return None


CompileSliceBlocks.refuse = _refuse_blocks


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, CN: int = 0, CTW: int = 0,
            CU: int = 0, G: int | None = None) -> dict:
    """The slice's statics; G, the steps a ``srv`` row holds, is P unless given, and is in them only below P."""
    st = {**RS.statics(R, W, A, engine, IW), "Q": Q, "D": D, "P": P, "K": K, "CN": CN, "CTW": CTW, "CU": CU}
    return with_block(st, G)


def with_block(st: dict, G: int | None) -> dict:
    """`st` with its ``srv`` rows G steps each (`block`): G dropped at G = P (None: P), so the unit is CompileSlice_v1."""
    st = {k: v for k, v in st.items() if k != "G"}
    G = st["P"] if G is None else G
    block(SimpleNamespace(**st, G=G))
    return st if G == st["P"] else {**st, "G": G}


def unit(**st):
    """The unit at statics `st`: CompileSlice_v1 at G = P (`st` without G), else CompileSliceBlocks_v1{…, G}."""
    return bind(CompileSliceBlocks if "G" in st else CompileSlice, **st)


#: a synthetic engine small enough to trace and to prove in about a minute: 4-token blocks, 16 tokens a step
DEMO_ENGINE = {"max_num_seqs": 4, "max_num_batched_tokens": 16, "max_model_len": 64, "num_gpu_blocks": 20, "block_size": 4,
               "enable_prefix_caching": True, "enable_chunked_prefill": True, "scheduler_reserve_full_isl": True}


def demo() -> tuple[dict, Slice]:
    """The demonstration class's statics and slice: four steps at (R 2, W 1, A 2) on `DEMO_ENGINE`, three arrivals, five
    decisions, a 6-entry prefix cache of 4-bit tokens and 4 units a request. Request 1's prompt shares request 0's first
    two blocks and request 2's its first; request 1 is admitted at step 2 and request 2 never in the window (its asks
    are 0), so every hit path is taken."""
    st = statics(2, 1, 2, DEMO_ENGINE, 3, Q=3, D=5, P=4, K=3, CN=6, CTW=4, CU=4)
    tokens = ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2), (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 3, 5, 7, 9),
              (1, 2, 3, 4, 9, 9, 6, 6, 6))
    return st, Slice(((0, 9, 8), (1, 11, 4), (2, 6, 3)), (0, 0, 1, 0, 0), 4, tokens)


def demo_async() -> tuple[dict, Slice]:
    """The asynchronous demonstration class's statics and slice: `DEMO_ENGINE` at two sequences under async scheduling
    without prefix caching (AS = 1, so ``wit`` has a ``map``), (R 2, W 1, A 2), three arrivals (id, P, O, cap), two
    decisions and six units. Units 0 to 3 run two rows each, so a served batch in another order than the slots is a
    map other than the identity."""
    engine = {**DEMO_ENGINE, "max_num_seqs": 2, "enable_prefix_caching": False, "async_scheduling": True}
    st = statics(2, 1, 2, engine, 3, Q=3, D=2, P=6, K=3)
    return st, Slice(((0, 9, 3, 0), (1, 6, 2, 0), (2, 5, 2, 1)), (1, 1), 6)


# --- the specification -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Slice:
    """One window's inputs: its certified arrivals (id, P, O) in order, its decisions, its executed step count and
    each arrival's certified tokens (its prompt, then its outputs)."""

    arrivals: tuple[RC.Arrival, ...]
    decisions: tuple[int, ...]
    steps: int
    tokens: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class Case:
    """A unit's rows (bits each, `fields`' layouts) and the verdict `check` gives them (None where it leaves a flag
    open), with the scan's rows (per step, `RS.Out.rows`), the decisions it asked in order, the charge of the advice's
    decisions from ``cin`` (`arith_encode.reference_charge`), ``cout``'s carry and the hits each step's candidates
    take."""

    rows: dict
    verdict: int | None
    flags: dict
    steps: tuple
    asks: tuple
    length: int
    carry: AE.Carry
    hits: tuple = ()


def _state(Sc: RC.Shape, row: RC.Row) -> RS.State:
    """A state row's entries as ReplayStep_v1's requests, by the chain's names at its shape (AS = 1 adds cap and fin)."""
    return RS.State(tuple(RS.Req(**dict(zip(RC.running_names(Sc), x))) for x in row.running),
                    tuple(RS.Req(**dict(zip(RC.waiting_names(Sc), x))) for x in row.waiting), row.free)


def _row(Sc: RC.Shape, st: RS.State) -> RC.Row:
    return RC.Row(tuple(tuple(getattr(r, k) for k in RC.running_names(Sc)) for r in st.running),
                  tuple(tuple(getattr(r, k) for k in RC.waiting_names(Sc)) for r in st.waiting), st.free)


def rows_bits(S, rows: Sequence[tuple[int, int, int, int, int]] | None) -> list[int]:
    """One step's rows as the circuit packs them: `_out_fields`' row fields in order, slot by slot (zero on an idle step,
    None)."""
    Z = RS.widths(step_statics(S))
    by = {x[0]: x for x in rows or ()}
    out = []
    for k, wd in zip(range(1, 5 + 1), (1, S.IW, 1, Z.LW, Z.TW)):
        for p in range(S.R):
            v = (1, *by[p][1:])[k - 1] if p in by else 0
            out += [(v >> j) & 1 for j in range(wd)]
    return out


def _steps_bits(S, steps: Sequence) -> list[list[int]]:
    if len(steps) > S.P:
        raise ValueError(f"a window of {len(steps)} steps is over P = {S.P}")
    return [rows_bits(S, s) for s in list(steps) + [None] * (S.P - len(steps))]


def srv_row(S, steps: Sequence) -> list[int]:
    """The rows of a window's steps whole: each step's `rows_bits` in turn (P steps, idle ones zero, None), zero-padded to
    whole words; at G = P the ``srv`` row."""
    bits = [b for s in _steps_bits(S, steps) for b in s]
    return bits + [0] * (_words_of(len(bits)) - len(bits))


def srv_rows(S, steps: Sequence) -> dict[str, list[int]]:
    """The ``srv`` rows of a window's step rows (`srv_names`): row j steps [j·G, (j + 1)·G) of `srv_row`'s, zero-padded
    to whole words, as `served_rows.block_bits` registers block j."""
    G, per = block(S), _steps_bits(S, steps)
    out = {}
    for j, r in enumerate(srv_names(S)):
        bits = [b for s in per[j * G:(j + 1) * G] for b in s]
        out[r] = bits + [0] * (_words_of(len(bits)) - len(bits))
    return out


def _served(S, rows: dict) -> list[int]:
    """The served steps' bits a unit's ``srv`` rows hold, in step order (each row's fields, its padding dropped)."""
    F = fields(S)
    return [int(b) for r in srv_names(S) for b in rows[r][:RC._bits_of(F[r])]]


def slot_map(S, rows: Sequence[tuple] | None, served: Sequence[tuple] | None) -> list[list[int]]:
    """The ``map`` a prover puts in for one step at AS = 1 (m[b][s] = 1: served row b is slot s's): each served row (b,
    request, …) at the slot of the scan's row (`rows`, by slot) for its request, and none for a request without one."""
    at = {x[1]: x[0] for x in rows or ()}
    m = [[0] * S.R for _ in range(S.R)]
    for x in served or ():
        if x[1] in at:
            m[x[0]][at[x[1]]] = 1
    return m


def _mapped(S, rows: Sequence[tuple] | None, m: Sequence[Sequence[int]]) -> tuple | None:
    """A step's rows (`RS.Out.rows`, by slot; None on an idle step) at the batch positions the map `m` gives them, or None
    where `m` isn't one-to-one from batch positions onto the slots with rows."""
    if any(sum(r) > 1 for r in m) or any(sum(c) > 1 for c in zip(*m)):
        return None
    by = {x[0]: x for x in rows or ()}
    if {s for s in range(S.R) if any(r[s] for r in m)} != set(by):
        return None
    return tuple((b, *by[s][1:]) for b, r in enumerate(m) for s in range(S.R) if r[s])


def _rows_ok(S, out_rows: Sequence, served: Sequence[int], maps: Sequence | None) -> bool:
    """``rows``: the active steps' rows are the served ones (`_served`: step u's in ``srv`` row u // G), slot for slot at
    AS = 0, and at AS = 1 at the batch positions the witnessed map gives them (`_mapped`)."""
    if S.AS:
        out_rows = [_mapped(S, r, [[int(x) for x in maps[u * S.R + b]] for b in range(S.R)])
                    for u, r in enumerate(out_rows)]
        if None in out_rows:
            return False
    return [b for s in _steps_bits(S, out_rows) for b in s] == list(served)


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def _pack(S, r: str, vals: dict) -> list[int]:
    out = [b for name, n, wd in fields(S)[r] for v in vals[name] for b in _bits(v, wd)]
    return out + [0] * (row_bits(S)[r] - len(out))


def _inputs(S, arr: Sequence[RC.Arrival], bits: Sequence[int], hit: Sequence[int] | None = None) -> RS.Inputs:
    """ReplayStep_v1's inputs at a step: the arrivals, bits and hits (0 unless `hit`), the evictable and returned
    counts 0 (the stand-in)."""
    zero = (0,) * (S.W + S.A)
    names = RC.arrival_names(chain(S))
    return RS.Inputs(tuple(RS.Req(**dict(zip(names, a))) for a in arr), tuple(bits),
                     tuple(hit) if hit is not None else zero, zero, (0,) * S.R)


def _fabricated(S, cache: CST.Cache, entries: Sequence[CST.Entry]) -> CST.Cache:
    if len(cache.table) + len(entries) > S.CN:
        raise ValueError("no room in the table for the made-up entries")
    return replace(cache, table=cache.table + tuple(entries))


def scan(S, sl: Slice, window: Sequence[int] | None = None, cert: Sequence[RC.Arrival] | None = None, *,
         hits: dict[int, Sequence[int]] | None = None, consume: dict[int, tuple[int, int]] | None = None,
         tokens: Sequence[Sequence[int]] | None = None,
         fabricate: dict[int, Sequence[CST.Entry]] | None = None) -> SimpleNamespace:
    """The window as a prover runs it from `sl`: ReplayLoad's rows, then each step by `replay_step.reference` on the state
    and the window row's arrivals and bits at the heads, and the cache through the step by `cache_step.scan_step` from
    the empty table, each candidate taking the hit it derives (its longest cached prefix, at the table it is looked up
    at: run to a fixed point, at most one round a candidate). The prover's choices: `window`, the decisions it puts in
    W_1 (default the advice's); `cert`, the arrivals it puts in ``cert`` (default `sl`'s); `tokens`, their tokens
    (default `sl`'s); `hits`, step u -> the prefix hits its step u takes in place of the derived ones
    (`replay_step.Inputs.hit`); `consume`, step u -> the (taken, used) it moves the heads by at step u (default the
    step's own); `fabricate`, u -> entries it adds to the table step u reads (0: the session's first, which is empty).
    Its next row is the step's next state with the heads moved by those, and its asks are each step's first ``used``
    bits."""
    Sc, St = chain(S), step_statics(S)
    arrivals = tuple(map(tuple, cert if cert is not None else sl.arrivals))
    toks = {a[0]: tuple(t) for a, t in zip(arrivals, tokens if tokens is not None else sl.tokens)}
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), arrivals, sl.decisions, sl.steps)
    if loaded is None:
        raise ValueError("the load refuses the slice")
    cur, win = loaded
    if window is not None:
        win = replace(win, decisions=tuple(window))
    fabricate = fabricate or {}
    cache = _fabricated(S, CST.Cache(), fabricate.get(0, ()))
    states, fronts, steps, asks, caches, hit_rows = [cur], [], [], [], [cache], []
    for u in range(S.P):
        arr, bits = RC.front(Sc, win, cur)
        fronts.append((arr, bits))
        h = (0,) * (S.W + S.A)
        if not cur.steps:
            states.append(cur)
            hit_rows.append(h)
            cache = _fabricated(S, cache, fabricate.get(u + 1, ()))
            caches.append(cache)
            continue
        for _round in range(S.W + S.A + 2):
            h = tuple((hits or {}).get(u, h))
            out = RS.reference(St, _state(Sc, cur), _inputs(S, arr, bits, h))
            if not out.ok:
                raise ValueError("the step refuses: the slice's class can't take it")
            if not S.PC:
                got = SimpleNamespace(cache=cache, hits=h)
                break
            got = CST.scan_step(S, cache, _state(Sc, cur), _inputs(S, arr, bits, h), out, toks, derive=True)
            if got is None:
                raise ValueError("the cache refuses: the slice's class can't take it")
            if u in (hits or {}) or got.hits == h:
                break
            h = got.hits
        else:
            raise ValueError(f"step {u}'s hits don't settle")
        cache = _fabricated(S, got.cache, fabricate.get(u + 1, ()))
        caches.append(cache)
        hit_rows.append(h)
        taken, used = (consume or {}).get(u, (out.taken, out.used))
        stepped = _row(Sc, out.state)
        nxt = RC.Row(stepped.running, stepped.waiting, stepped.free, cur.qh + taken, cur.dh + used, cur.steps - 1)
        asks += bits[:used]
        steps.append(out.rows)
        states.append(nxt)
        cur = nxt
    return SimpleNamespace(states=states, window=win, fronts=fronts, steps=steps, asks=asks, arrivals=arrivals,
                           tokens=toks, caches=caches, hits=hit_rows)


def _int(bits: Sequence[int]) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def _cert_tokens(S, bits: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """The ``cert`` row's tokens by request, its valid slots'."""
    f = RC._split(bits, fields(S)["cert"])
    n = S.CU * S.B
    return {_int(f["a_id"][k]): tuple(_int(t) for t in f["tok"][k * n:(k + 1) * n])
            for k in range(S.K) if f["a_valid"][k][0]}


def _book(S, p: dict) -> AE.ChargeBook:
    """The ``pub`` row's book (`charge_chain.book_fields`), as `arith_encode.charge_book` gives it."""
    v = {name: [_int(e) for e in p[name]] for name, _n, _w in CC.book_fields(S.D)}
    return AE.ChargeBook(tuple(v["arrived"]), v["total"][0], tuple(zip(v["zfreq"], v["ztotal"])))


def check(S, rows: dict) -> dict[str, bool | None]:
    """The circuit's flags on a unit's rows (`fields`' layouts), by its pieces' references: ReplayLoad's `load` from the
    session's empty rows; per active step `replay_step.reference` on the witness state row with the witness arrivals,
    bits and hits (the evictable and returned counts 0) and `advance` with that step's own taken and used, against the
    next witness row, and an idle step's row kept; `arith_encode.reference_charge` from ``cin``'s carry on ``adv``'s
    first ``count`` decisions under ``pub``'s book, ``cout`` its carry' and its bits within c; the active steps' rows
    against ``srv``'s; the first cache row empty, and per active step `cache_step.scan_step` on the witness cache row
    with the step's hits and rows and ``cert``'s tokens, against the next cache row (an idle step's kept).

    For well-formed rows (`replay_chain.decode`'s, each field in range, and `cache_step.decode_cache`'s); `MalformedRow`
    otherwise. A flag is None where a step's ReplayStep refuses and the circuit's later outputs are its own: ``steps``
    is then False; and ``charge`` is None where the charge refuses its carry-in or its count."""
    Sc, St = chain(S), step_statics(S)
    F = fields(S)
    p, a, w = (RC._split(rows[r], F[r]) for r in ("pub", "adv", "wit"))
    states = [RC.decode(e, Sc) for e in w["state"]]
    win = RC.decode_window(w["window"][0], Sc)
    cert = RC._items(RC._unflatten(rows["cert"], RC.arrival_fields(Sc))[0], "a", RC.arrival_names(Sc))
    toks = _cert_tokens(S, rows["cert"]) if S.PC else {}
    crow = [[int(b) for b in e] for e in w["cache"]] if S.PC else []
    hit = [_int(e) for e in w["hit"]] if S.PC else [0] * (S.P * (S.W + S.A))
    cache_ok = not any(crow[0]) if S.PC else True
    count = _int(a["count"][0])
    dec = [e[0] for e in a["dec"]]
    if count > S.D or any(dec[count:]):
        raise RC.MalformedRow(f"the advice's count {count} with a decision past it")
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), cert, dec[:count], _int(a["steps"][0]))
    flags: dict[str, bool | None] = {"load": loaded == (states[0], win)}
    ok, length, carried = AE.reference_charge(S.D, _book(S, p), CC.carry_of([int(b) for b in rows["cin"]]),
                                              [int(k < count) for k in range(S.D)], dec)
    flags.update(code=bool(ok) and tuple(int(b) for b in rows["cout"]) == CC.encode(carried),
                 charge=length <= _int(p["c"][0]) if ok else None)
    steps_ok, out_rows = True, []
    for u in range(S.P):
        cur, nxt = states[u], states[u + 1]
        arr = RC._items(RC._unflatten(w["arr"][u], RC.front_fields(Sc))[0], "arr", RC.arrival_names(Sc))
        bits = [int(b) for b in w["bits"][u]]
        if not cur.steps:
            steps_ok &= nxt == cur
            cache_ok &= not S.PC or crow[u + 1] == crow[u]
            out_rows.append(None)
            continue
        inp = _inputs(S, arr, bits, hit[u * (S.W + S.A):(u + 1) * (S.W + S.A)])
        out = RS.reference(St, _state(Sc, cur), inp)
        if not out.ok:
            flags.update(steps=False, rows=None, cache=None)
            return {f: flags[f] for f in FLAGS}
        want = RC.advance(Sc, win, cur, True, _row(Sc, out.state), arr, bits, out.taken, out.used)
        steps_ok &= want is not None and want == nxt
        if S.PC:
            try:
                held = CST.decode_cache(S, crow[u], len(cur.running))
            except ValueError as e:
                raise RC.MalformedRow(str(e)) from None
            got = CST.scan_step(S, held, _state(Sc, cur), inp, out, toks)
            cache_ok &= got is not None and CST.encode_cache(S, got.cache) == crow[u + 1]
        out_rows.append(out.rows)
    flags.update(steps=steps_ok, rows=_rows_ok(S, out_rows, _served(S, rows), w.get("map")), cache=cache_ok)
    return {f: flags[f] for f in FLAGS}


def verdict(flags: dict[str, bool | None]) -> int | None:
    """The verdict word of `check`'s flags; None while one is unspecified."""
    if any(flags[f] is None for f in FLAGS):
        return None
    return sum(int(flags[f]) << k for k, f in enumerate(FLAGS))


def pub(S, charge: int, book: dict) -> list[int]:
    """The verifier's own ``pub`` row (u16 words): the declared charge and the registered book's tables (its JSON,
    `code_v1.Book.to_json()`'s form)."""
    return words(_pack(S, "pub", {"c": [charge], **CC.book_values(AE.charge_book(book, S.D)[0])}))


def case(S, sl: Slice, book: dict, *, carry: AE.Carry | None = None, cout: AE.Carry | None = None,
         advice: Sequence[int] | None = None, served: Sequence | None = None, charge: int | None = None,
         window: Sequence[int] | None = None, cert: Sequence[RC.Arrival] | None = None,
         hits: dict[int, Sequence[int]] | None = None, consume: dict[int, tuple[int, int]] | None = None,
         tokens: Sequence[Sequence[int]] | None = None, fabricate: dict[int, Sequence[CST.Entry]] | None = None,
         maps: dict[int, Sequence[Sequence[int]]] | None = None) -> Case:
    """A unit's rows for the prover's run of `sl` (`scan`'s, with `window`, `cert`, `hits`, `consume`, `tokens` and
    `fabricate`) under the registered `book` (its JSON), and `check`'s verdict on them. `carry` is the carry-in it puts
    in ``cin`` (default the book's priors, the session's first window's), `advice` the decisions it commits in ``adv``
    (default `sl`'s), `cout` the carry-out it puts in ``cout`` (default the charge's carry' on the advice from `carry`),
    `served` the window's step rows it puts in ``srv`` (default the run's own), `charge` ``pub``'s (default the
    advice's charge, ``length``) and, at AS = 1, `maps` step u -> the map it puts in ``map`` for step u (default
    `slot_map`'s)."""
    Sc = chain(S)
    cb, first = AE.charge_book(book, S.D)
    carry = first if carry is None else carry
    dec = tuple(sl.decisions if advice is None else advice)
    if len(dec) > S.D:
        raise ValueError(f"{len(dec)} decisions over D = {S.D}")
    padded = list(dec) + [0] * (S.D - len(dec))
    ok, length, nxt = AE.reference_charge(S.D, cb, carry, [1] * len(dec) + [0] * (S.D - len(dec)), padded)
    if not ok:
        raise ValueError("the charge refuses the advice from the carry-in")
    cout = nxt if cout is None else cout
    sc = scan(S, sl, window, cert, hits=hits, consume=consume, tokens=tokens, fabricate=fabricate)
    served = sc.steps if served is None else served
    width = row_bits(S)
    rows = {"pub": _pack(S, "pub", {"c": [length if charge is None else charge], **CC.book_values(cb)}),
            "adv": _pack(S, "adv", {"dec": padded, "count": [len(dec)], "steps": [sl.steps]}),
            **srv_rows(S, served),
            "cin": list(CC.encode(carry)), "cout": list(CC.encode(cout))}
    cert_bits = RC.load_arguments(Sc, RC.empty(Sc), RC.Window(), sc.arrivals, sl.decisions, len(sl.decisions), sl.steps,
                                  RC.encode(sc.states[0], Sc), RC.encode_window(sc.window, Sc))[2]
    if S.PC:
        n = S.CU * S.B
        held = [tuple(sc.tokens[x[0]])[:n] for x in sc.arrivals] + [()] * (S.K - len(sc.arrivals))
        cert_bits += [b for t in held for v in list(t) + [0] * (n - len(t)) for b in _bits(v, S.CTW)]
    rows["cert"] = cert_bits + [0] * (width["cert"] - len(cert_bits))
    wit = [b for st in sc.states for b in RC.encode(st, Sc)] + list(RC.encode_window(sc.window, Sc))
    wit += [b for arr, _b in sc.fronts for b in RC.front_bits(Sc, arr)]
    wit += [b for _a, bits in sc.fronts for b in bits]
    if S.PC:
        wit += [b for h in sc.hits for v in h for b in _bits(v, RS.widths(step_statics(S)).HW)]
        wit += [b for c in sc.caches for b in CST.encode_cache(S, c)]
    if S.AS:
        mine, theirs = (list(x) + [None] * (S.P - len(x)) for x in (sc.steps, served))
        wit += [b for u in range(S.P) for r in (maps or {}).get(u, slot_map(S, mine[u], theirs[u])) for b in r]
    rows["wit"] = wit + [0] * (width["wit"] - len(wit))
    assert set(rows) == set(width) and all(len(rows[r]) == width[r] for r in width)
    flags = check(S, rows)
    return Case(rows, verdict(flags), flags, tuple(sc.steps), tuple(sc.asks), length, cout, tuple(sc.hits))


PHASE9 = ("hit", "reread", "untaken")


def phase9(S, sl: Slice) -> dict[str, dict]:
    """The interfaces free in a piece alone that the slice still has, as `case`'s keywords, each the prover's run of
    `sl` with that interface chosen to lower its charge or its schedule: ``hit``, the first step's first candidate
    takes a 2-block prefix hit; ``reread``, step 2 leaves the decision head in place, so step 3 reads its decision
    again; ``untaken``, step 0 leaves the arrival head in place. ``srv``'s rows are the caller's (the served ones).
    Its context and frequency choices have no input: ChargeLink_v1 computes both from ``cin`` and the committed
    decisions."""
    return {"hit": {"hits": {0: (2,) + (0,) * (S.W + S.A - 1)}},
            "reread": {"consume": {2: (0, 0)}},
            "untaken": {"consume": {0: (0, 0)}}}


def made_up(S, sl: Slice, request: int, units: int = 1) -> dict:
    """`case`'s keywords for a made-up first table: `request`'s first `units` units under ROOT, owned by request 0."""
    toks = sl.tokens[request]
    return {"fabricate": {0: tuple(CST.Entry(S.CN if j == 0 else j - 1, tuple(toks[j * S.B:(j + 1) * S.B]), 0)
                                   for j in range(units))}}


def words(bits: Sequence[int]) -> list[int]:
    """A row's u16 words, least significant bit first."""
    return [sum(int(b) << j for j, b in enumerate(bits[k:k + WORD])) for k in range(0, len(bits), WORD)]


def evaluate(st: dict, cases: Sequence[Case]) -> list[int]:
    """The unit's verdict word on each case, in one evaluation."""
    args = [[c.rows[r] for c in cases] for r in row_names(SimpleNamespace(**st))]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


@lru_cache(maxsize=None)
def _counted(key: tuple) -> dict[str, int]:
    S = SimpleNamespace(**dict(key))
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in _sig(S)[0]]
    C.output("verdict", _build(C, S, *ins)[0])
    return C.counts()


def size(**st) -> dict[str, int]:
    """The unit's ANDs and its rows' widths at statics `st`."""
    S = SimpleNamespace(**st)
    key = tuple((k, st[k]) for k in BLOCK_STATICS if k in st)
    return {"ands": _counted(key)["ands"], **{f"{r}_bits": b for r, b in row_bits(S).items()}}
