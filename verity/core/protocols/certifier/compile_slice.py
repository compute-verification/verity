"""A slice of the in-proof compile unit as one Boolean Definition: the pieces of PRF-143's statement (the certifier
PROTOCOL's "Advice v1") composed whole for one window of a few steps, so the composition is proved end to end before the
chain splits it into one unit a step.

    CompileSlice_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, Q, D, P, K, SEG} (pub, adv, cert, srv, wit)
        -> verdict[16]

**The rows**, each a field list (elements least significant bit first) zero-padded to whole 16-bit words:
- ``pub``, the public input: the declared charge ``c`` (16 bits);
- ``adv``, the committed advice, a row under the advice's hiding root (`verity.ml.commitments.advice`): the window's raw
  decision bits ``dec`` (D), their ``count`` and the window's executed step count ``steps``;
- ``cert``, the window's certified arrivals (`replay_chain.arrival_fields`: K slots of valid, id, P, O), a row the
  verifier holds to the certified record's commitment (its b‖c);
- ``srv``, the served rows: the step rows the served window ran (P steps of `rows_bits`, idle ones zero; `srv_row`), a
  row the verifier holds by its b‖c to the window's row of the served rows' registration (`served_rows`), committed on
  the Commit side before the draw;
- ``wit``, the witness: the state rows S_1 … S_{P+1} and the window row W_1 (`replay_chain`'s layouts), each step's
  arrivals ``arr`` and decision bits ``bits`` and the window's codeword ``seg`` (SEG bits).

**The circuit.** `replay_chain.ReplayLoad` from the session's empty rows (S_0 and W_0, constants) to S_1 and W_1 with
``cert``'s arrivals and ``adv``'s decisions. Then per step u = 1 … P: `replay_step.ReplayStep_v1` on S_u's state with
``arr[u]`` and ``bits[u]``, and `replay_chain.ReplayAdvance` on the same ``arr``/``bits`` wires, which holds them to W_1 at
S_u's heads and S_{u+1} to the step's next state. The step's asks (the first ``used`` of its bits) are coded in the scan's
order: per ask `arith_encode.KTContext_v1{C=3, NW=24}` under the plain context (the asked request's passes so far,
min(passes, 2): the queue head's ``pc`` for the first ask of a step that forced nothing, else 0) and
`arith_encode.ArithEncodeStep_v1{K=18, BIN=1}` against ``seg`` at the code's position; then
`arith_encode.ArithEncodeFinish_v1`. The position after the finish is the coded length, the charge. Each step's rows
(valid, request, kind, start, n), zero on an idle step, packed in `_out_fields`' order (`rows_bits`), are held equal
to ``srv``'s. ``pc`` moves as `replay.encode`'s ``passed`` does: after a step whose last ask was
a 0 it is min(pc + 1, 2) when that was the step's one ask and nothing was forced (the head passed once more), else 1 (a
new head passed once); otherwise it is 0 when the step took an arrival and unchanged when it took none.

**The verdict** (16 bits, the unit's one output; `ACCEPT` = 0x1F): bit 0 ``load``, ReplayLoad's ok; bit 1 ``steps``,
every ReplayAdvance's ok (with each active step's ReplayStep ok); bit 2 ``code``, every model's and coder's ok and the
finish's, the code within SEG bits; bit 3 ``charge``, the coded length ≤ c; bit 4 ``rows``, the steps' rows are
``srv``'s. The rest are 0.

**The reference** is `check`, the flags the circuit's pieces' references give any unit's rows; `case` builds the rows of
a prover's run (`scan`, with the choices REDTEAM's phase 9 found free in a piece alone: the codeword, the prefix hits
and the heads' moves) against the served rows it puts in ``srv``, and `check`'s verdict on them. No witness field
carries a context, a frequency, a cache count or a consumed count: the context is the plain counter's, the frequencies
KTContext_v1's outputs, and the heads move by ReplayStep_v1's own ``taken`` and ``used``. On corpus-smallkv's first
seven steps (60,574 ANDs) a context or frequencies of the prover's choosing are refused by ``code`` and ``charge``
(0x13); a prefix hit, a decision read again and an arrival left queued by ``steps`` and ``rows`` (0x0D), the hit by the
stand-in below until CacheStep_v1's derived hit is wired. A fabricated cache table and free returns aren't expressible
on the slice: it has no table input and no stopping request. REDTEAM phase 10's six passes (advice that registers six
passes where the served window admitted user-1 at the sixth ask, a 5-bit code against the honest 8) are refused by
``rows`` (0x0F) with the served rows in ``srv``; with its own rows there, the circuit accepts and the native b‖c check
refuses ``srv``.

**Stand-ins** (named, each a piece not ready):
- *the cache counts*: ENC-DERIVED's hit, evictable and returned-block counts are held at 0 (ReplayStep_v1's ``hit``,
  ``evict`` and ``ret``), true of a slice that hits no prefix and stops or preempts nothing; `cache_step.CacheStep_v1`
  isn't wired, so a claimed hit is refused by the 0 and an honest one can't be taken: on any slice that hits, the rows
  differ from the clear recompile's and ``rows`` refuses;
- *the model*: KT counts from a zero prior at each window's start, the per-window counts not coded (advice v1 under
  #3642 starts from the registered book's prior and codes them first);
- *the counting encoder*: ENC-CODE's encoder that counts its emitted bits is ArithEncodeStep_v1 checking a witness
  codeword, whose consumed positions are the count (a symbol's scalings depend only on the interval);
- *the still-sent fields*: ``adv`` holds only the decisions, the count and the step count;
- *sync scheduling*: ReplayStep_v1 models no async step;
- *the binding of ``adv``, ``cert`` and ``srv``*: their rows' b‖c, which the verifier checks natively against the
  advice registration's leaf, the certified record's commitment and the served rows' leaf; no gate hashes them. That
  the served rows are the rows the served compute ran is the step tree's to bind (`served_rows`).

**Statics**: ReplayStep_v1's (R, W, A and the engine's M, T, L, B, H, CH, WM, FULL, PC, NB, and IW), the chain's Q, D,
P, K (`replay_chain.Shape`) and SEG, the codeword's cap in bits.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.core.protocols.certifier import arith_encode as AE, replay_chain as RC, replay_step as RS
from verity.core.protocols.certifier.head_check import BIT, _part

STATICS = RS.STATICS + ("Q", "D", "P", "K", "SEG")
CW = 16
EK, EC, ENW = AE.K_MAX, 3, 24
FLAGS = ("load", "steps", "code", "charge", "rows")
ACCEPT = (1 << len(FLAGS)) - 1
ROWS = ("pub", "adv", "cert", "srv", "wit")
ROW_FIELDS = ("row_valid", "row_id", "row_kind", "row_start", "row_n")
WORD = 16


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def chain(S) -> RC.Shape:
    return RC.Shape(**{k: getattr(S, k) for k in RC.STATICS})


def step_statics(S) -> SimpleNamespace:
    return SimpleNamespace(**{k: getattr(S, k) for k in RS.STATICS})


def _kt():
    return SimpleNamespace(C=EC, NW=ENW)


def _coder():
    return SimpleNamespace(K=EK, BIN=1)


def fields(S) -> dict[str, list[tuple[str, int, int]]]:
    Sc = chain(S)
    Z = RC.widths(Sc)
    return {"pub": [("c", 1, CW)],
            "adv": [("dec", S.D, 1), ("count", 1, Z.DW), ("steps", 1, Z.PW)],
            "cert": RC.arrival_fields(Sc),
            "srv": [("rows", S.P, step_row_bits(S))],
            "wit": [("state", S.P + 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)),
                    ("arr", S.P, RC._bits_of(RC.front_fields(Sc))), ("bits", S.P, S.A), ("seg", 1, S.SEG)]}


def row_bits(S) -> dict[str, int]:
    return {r: WORD * -(-RC._bits_of(fs) // WORD) for r, fs in fields(S).items()}


def step_row_bits(S) -> int:
    """One step's rows in bits: R × (valid, request, kind, start, n)."""
    Z = RS.widths(step_statics(S))
    return S.R * (1 + S.IW + 1 + Z.LW + Z.TW)


def _pos_width(S) -> int:
    """The code position's width: every ask consumes at most EK bits and the finish 2, so the count can't wrap."""
    return _bitlen(max(S.SEG, S.P * S.A * EK + 2))


# --- the circuit ------------------------------------------------------------------------------------------------------------

def _consts(bits: Sequence[int]) -> list:
    return [ONE if b else ZERO for b in bits]


def _build(C, S, pub, adv, cert, srv, wit):
    Sc, St = chain(S), step_statics(S)
    ZS = RS.widths(St)
    F = fields(S)
    p, a, c, v, w = (RC._split(x, F[r]) for x, r in zip((pub, adv, cert, srv, wit), ROWS))
    states, win = w["state"], w["window"][0]
    load_ok = RC._build_load(C, Sc, _consts(RC.encode(RC.empty(Sc), Sc)), _consts(RC.encode_window(RC.Window(), Sc)),
                             RC._flat(c, F["cert"]), [e[0] for e in a["dec"]], a["count"][0], a["steps"][0], states[0],
                             win)[0]

    fresh = AE.fresh()
    coder = [const_bits(fresh.low, 32), const_bits(fresh.high, 32), [ZERO], [ZERO], [ZERO], [ONE]]
    n0, n1 = [[ZERO] * ENW for _ in range(EC)], [[ZERO] * ENW for _ in range(EC)]
    pc, PW = [ZERO, ZERO], _pos_width(S)
    pos = [ZERO] * PW
    seg = [[b] for b in w["seg"][0]]
    steps_ok, code_ok, rows = [], [], []
    for u in range(S.P):
        cur, nxt, arr, bits = states[u], states[u + 1], w["arr"][u], w["bits"][u]
        r, fa = RC._split(cur, RC.row_fields(Sc)), RC._split(arr, RC.front_fields(Sc))
        elems = []
        for name, n, wd in RS._in_fields(St):
            if name in r:
                elems += r[name]
            elif name in fa:
                elems += fa[name]
            elif name == "bits":
                elems += [[b] for b in bits]
            else:
                elems += [[ZERO] * wd for _ in range(n)]
        out = RS._split(RS._out_fields(St), RS._build(C, St, *elems))
        stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
        taken, used = out["taken"][0], out["used"][0]
        ok, active = RC._build_advance(C, Sc, out["ok"][0], cur, win, stepped, arr, bits, taken, used, nxt)
        act = active[0]
        steps_ok.append(ok[0])
        rows += [C.AND(act, x) for name in ROW_FIELDS for e in out[name] for x in e]

        quiet = C.AND(or_reduce(C, [e[0] for e in r["run_valid"]]).inv(), r["wait_valid"][0][0].inv())
        forced = C.AND(quiet, fa["arr_valid"][0][0])
        for k in range(S.A):
            en = C.AND(act, RC._lt(C, const_bits(k, ZS.AC), used))
            ctx = [C.AND(x, forced.inv()) for x in pc] if k == 0 else [ZERO, ZERO]
            kt = AE._build_kt(C, _kt(), *n0, *n1, [en], ctx, [bits[k]])
            code_ok.append(kt[0][0])
            f0, f1, n0, n1 = kt[1], kt[2], kt[3:3 + EC], kt[3 + EC:3 + 2 * EC]
            got = AE._build_step(C, _coder(), *coder, [en], [bits[k]], f0, f1,
                                 [e[0] for e in RC._front(C, seg, pos, EK)])
            coder = got[:6]
            pos = add(C, pos, RC._pad(got[6], PW), width=PW)

        last = or_reduce(C, [C.AND(RC._eq(C, used, const_bits(k + 1, ZS.AC)), bits[k]) for k in range(S.A)])
        last0 = C.AND(or_reduce(C, used), last.inv())
        once = C.AND(RC._eq(C, used, const_bits(1, ZS.AC)), forced.inv())
        inc = [C.AND(pc[0].inv(), pc[1].inv()), RS.OR2(C, pc[0], pc[1])]
        stay = [C.AND(RC._zero(C, taken), x) for x in pc]
        pc = mux(C, act, mux(C, last0, mux(C, once, inc, [ONE, ZERO]), stay), pc)

    fin = AE._build_finish(C, _coder(), *coder, [e[0] for e in RC._front(C, seg, pos, 2)])
    count = add(C, pos, RC._pad(fin[1], PW), width=PW)
    code_ok += [fin[0][0], RC._lt(C, const_bits(S.SEG, PW), count).inv()]
    charge_ok = RC._lt(C, p["c"][0], count).inv()

    rows_ok = RC._eq(C, rows, [x for e in v["rows"] for x in e])

    flags = [load_ok, and_reduce(C, steps_ok), and_reduce(C, code_ok), charge_ok, rows_ok]
    return [flags + [ZERO] * (16 - len(flags))]


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in ROWS), Array(16, BIT)


CompileSlice = _part(
    "CompileSlice", STATICS, _sig, _build,
    "a slice of the compile unit, one window whole: ReplayLoad, then per step ReplayStep_v1 and ReplayAdvance on the "
    "same arrival and decision wires, the asks coded under the plain contexts against the window's codeword, the coded "
    "length held to the declared charge and the step rows to the committed served rows (verdict bits load, steps, "
    "code, charge, rows)")
CompileSlice.build = _build  # type: ignore[attr-defined]


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, SEG: int) -> dict:
    return {**RS.statics(R, W, A, engine, IW), "Q": Q, "D": D, "P": P, "K": K, "SEG": SEG}


def unit(**st):
    return bind(CompileSlice, **st)


# --- the specification -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Slice:
    """One window's inputs: its certified arrivals (id, P, O) in order, its decisions and its executed step count."""

    arrivals: tuple[RC.Arrival, ...]
    decisions: tuple[int, ...]
    steps: int


@dataclass(frozen=True)
class Case:
    """A unit's rows (bits each, `fields`' layouts) and the verdict `check` gives them (None where it leaves a flag
    open), with the scan's rows (per step, `RS.Out.rows`), its symbols (bit, f0, f1) and the coded length."""

    rows: dict
    verdict: int | None
    flags: dict
    steps: tuple
    symbols: tuple
    length: int


def codeword(symbols: Sequence[tuple[int, int, int]]) -> list[int]:
    """`replay._Encoder`'s output on the symbols (bit, f0, f1), terminated (ENC-REPLAY's 32-bit WNC coder)."""
    lo, hi, pending, out = 0, AE.TOP, 0, []

    def emit(b):
        nonlocal pending
        out.append(b)
        out.extend([1 - b] * pending)
        pending = 0

    for bit, f0, f1 in symbols:
        split = lo + (hi - lo + 1) * f0 // (f0 + f1)
        lo, hi = (split, hi) if bit else (lo, split - 1)
        while True:
            if hi < AE.HALF:
                emit(0)
            elif lo >= AE.HALF:
                emit(1)
                lo, hi = lo - AE.HALF, hi - AE.HALF
            elif lo >= AE.Q1 and hi < AE.Q3:
                pending += 1
                lo, hi = lo - AE.Q1, hi - AE.Q1
            else:
                break
            lo, hi = 2 * lo, 2 * hi + 1
    if symbols:
        pending += 1
        emit(0 if lo < AE.Q1 else 1)
    return out


def _state(row: RC.Row) -> RS.State:
    return RS.State(tuple(RS.Req(*x) for x in row.running), tuple(RS.Req(*x) for x in row.waiting), row.free)


def _row(st: RS.State) -> RC.Row:
    return RC.Row(tuple((r.id, r.P, r.O, r.out, r.comp) for r in st.running), tuple((r.id, r.P, r.O, r.out) for r in st.waiting),
                  st.free)


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


def srv_row(S, steps: Sequence) -> list[int]:
    """The ``srv`` row of a window's step rows: each step's `rows_bits` in turn (P steps, idle ones zero, None)."""
    if len(steps) > S.P:
        raise ValueError(f"a window of {len(steps)} steps is over P = {S.P}")
    bits = [b for s in list(steps) + [None] * (S.P - len(steps)) for b in rows_bits(S, s)]
    return bits + [0] * (row_bits(S)["srv"] - len(bits))


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def _pack(S, r: str, vals: dict) -> list[int]:
    out = [b for name, n, wd in fields(S)[r] for v in vals[name] for b in _bits(v, wd)]
    return out + [0] * (row_bits(S)[r] - len(out))


def _plain(pc: int, used: int, taken: int, bits: Sequence[int], forced: bool) -> int:
    """The plain-context counter after a step, as `replay.encode`'s ``passed`` moves."""
    if used and not bits[used - 1]:
        return min(pc + 1, 2) if used == 1 and not forced else 1
    return 0 if taken else pc


def _forced(cur: RC.Row, arr: Sequence[RC.Arrival]) -> bool:
    return not cur.running and not cur.waiting and len(arr) > 0


def _inputs(S, arr: Sequence[RC.Arrival], bits: Sequence[int], hit: Sequence[int] | None = None) -> RS.Inputs:
    """ReplayStep_v1's inputs at a step: the arrivals and bits, the cache counts 0 (the stand-in) unless `hit`."""
    zero = (0,) * (S.W + S.A)
    return RS.Inputs(tuple(RS.Req(*a) for a in arr), tuple(bits), tuple(hit) if hit is not None else zero, zero,
                     (0,) * S.R)


def scan(S, sl: Slice, window: Sequence[int] | None = None, cert: Sequence[RC.Arrival] | None = None, *,
         hits: dict[int, Sequence[int]] | None = None,
         consume: dict[int, tuple[int, int]] | None = None) -> SimpleNamespace:
    """The window as a prover runs it from `sl`: ReplayLoad's rows, then each step by `replay_step.reference` on the state
    and the window row's arrivals and bits at the heads, the asks' symbols under the plain contexts. The prover's
    choices: `window`, the decisions it puts in W_1 (default the advice's); `cert`, the arrivals it puts in ``cert``
    (default `sl`'s); `hits`, step u -> the prefix hit its step u takes (`replay_step.Inputs.hit`; default none);
    `consume`, step u -> the (taken, used) it moves the heads by at step u (default the step's own). Its next row is
    the step's next state with the heads moved by those, and its symbols are the first ``used`` bits'."""
    Sc, St = chain(S), step_statics(S)
    arrivals = tuple(map(tuple, cert if cert is not None else sl.arrivals))
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), arrivals, sl.decisions, sl.steps)
    if loaded is None:
        raise ValueError("the load refuses the slice")
    cur, win = loaded
    if window is not None:
        win = replace(win, decisions=tuple(window))
    states, fronts, steps, symbols = [cur], [], [], []
    pc, n = 0, ((0, 0),) * EC
    for u in range(S.P):
        arr, bits = RC.front(Sc, win, cur)
        fronts.append((arr, bits))
        if not cur.steps:
            states.append(cur)
            continue
        out = RS.reference(St, _state(cur), _inputs(S, arr, bits, (hits or {}).get(u)))
        if not out.ok:
            raise ValueError("the step refuses: the slice's class can't take it")
        taken, used = (consume or {}).get(u, (out.taken, out.used))
        stepped = _row(out.state)
        nxt = RC.Row(stepped.running, stepped.waiting, stepped.free, cur.qh + taken, cur.dh + used, cur.steps - 1)
        forced = _forced(cur, arr)
        for k in range(used):
            _ok, f0, f1, n = AE.reference_kt(EC, ENW, n, 1, pc if k == 0 and not forced else 0, bits[k])
            symbols.append((bits[k], f0, f1))
        pc = _plain(pc, used, taken, bits, forced)
        steps.append(out.rows)
        states.append(nxt)
        cur = nxt
    return SimpleNamespace(states=states, window=win, fronts=fronts, steps=steps, symbols=symbols, arrivals=arrivals)


def _int(bits: Sequence[int]) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def check(S, rows: dict) -> dict[str, bool | None]:
    """The circuit's flags on a unit's rows (`fields`' layouts), by its pieces' references: ReplayLoad's `load` from the
    session's empty rows; per active step `replay_step.reference` on the witness state row with the witness arrivals and
    bits (the cache counts 0) and `advance` with that step's own taken and used, against the next witness row, and an
    idle step's row kept; the step's asks under the plain contexts (`reference_kt`) through `reference_step` against the
    witness codeword; `reference_finish`, the count within SEG and c; the active steps' rows against ``srv``'s.

    For well-formed rows (`replay_chain.decode`'s, each field in range); `MalformedRow` otherwise. A flag is None where
    a step's ReplayStep refuses and the circuit's later outputs are its own: ``steps`` is then False."""
    Sc, St = chain(S), step_statics(S)
    F = fields(S)
    p, a, w = (RC._split(rows[r], F[r]) for r in ("pub", "adv", "wit"))
    served = RC._bits_of(F["srv"])
    states = [RC.decode(e, Sc) for e in w["state"]]
    win = RC.decode_window(w["window"][0], Sc)
    cert = RC._items(RC._unflatten(rows["cert"], RC.arrival_fields(Sc))[0], "a", ("id", "P", "O"))
    count = _int(a["count"][0])
    dec = [e[0] for e in a["dec"]]
    if count > S.D or any(dec[count:]):
        raise RC.MalformedRow(f"the advice's count {count} with a decision past it")
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), cert, dec[:count], _int(a["steps"][0]))
    flags: dict[str, bool | None] = {"load": loaded == (states[0], win)}
    seg = [int(b) for b in w["seg"][0]]
    pc, n, coder, pos, steps_ok, code_ok, out_rows = 0, ((0, 0),) * EC, AE.fresh(), 0, True, True, []
    for u in range(S.P):
        cur, nxt = states[u], states[u + 1]
        arr = RC._items(RC._unflatten(w["arr"][u], RC.front_fields(Sc))[0], "arr", ("id", "P", "O"))
        bits = [int(b) for b in w["bits"][u]]
        if not cur.steps:
            steps_ok &= nxt == cur
            out_rows.append(None)
            continue
        out = RS.reference(St, _state(cur), _inputs(S, arr, bits))
        if not out.ok:
            return {**flags, "steps": False, "code": None, "charge": None, "rows": None}
        want = RC.advance(Sc, win, cur, True, _row(out.state), arr, bits, out.taken, out.used)
        steps_ok &= want is not None and want == nxt
        forced = _forced(cur, arr)
        for k in range(out.used):
            ok, f0, f1, n = AE.reference_kt(EC, ENW, n, 1, pc if k == 0 and not forced else 0, bits[k])
            code_ok &= bool(ok)
            coder, m = AE.reference_step(EK, 1, coder, 1, (bits[k], f0, f1), (seg[pos:pos + EK] + [0] * EK)[:EK])
            pos += m
        pc = _plain(pc, out.used, out.taken, bits, forced)
        out_rows.append(out.rows)
    ok, m = AE.reference_finish(coder, (seg[pos:pos + 2] + [0, 0])[:2])
    length = pos + m
    flags.update(steps=steps_ok, code=code_ok and bool(ok) and length <= S.SEG, charge=length <= _int(p["c"][0]),
                 rows=srv_row(S, out_rows)[:served] == [int(b) for b in rows["srv"][:served]])
    return flags


def verdict(flags: dict[str, bool | None]) -> int | None:
    """The verdict word of `check`'s flags; None while one is unspecified."""
    if any(flags[f] is None for f in FLAGS):
        return None
    return sum(int(flags[f]) << k for k, f in enumerate(FLAGS))


def pub(S, charge: int) -> list[int]:
    """The verifier's own ``pub`` row (u16 words): the declared charge."""
    return words(_pack(S, "pub", {"c": [charge]}))


def case(S, sl: Slice, *, served: Sequence | None = None, charge: int | None = None,
         window: Sequence[int] | None = None, cert: Sequence[RC.Arrival] | None = None,
         hits: dict[int, Sequence[int]] | None = None, consume: dict[int, tuple[int, int]] | None = None,
         seg: Sequence[int] | None = None) -> Case:
    """A unit's rows for the prover's run of `sl` (`scan`'s, with `window`, `cert`, `hits` and `consume`) and `check`'s
    verdict on them. `served` is the window's step rows it puts in ``srv`` (default the run's own) and `charge`
    ``pub``'s (default its coded length); `seg` is the codeword it puts in ``wit`` (default its symbols' `codeword`).
    ``length`` is the codeword's."""
    Sc = chain(S)
    Z = RC.widths(Sc)
    sc = scan(S, sl, window, cert, hits=hits, consume=consume)
    seg = list(codeword(sc.symbols) if seg is None else seg)
    length = len(seg)
    charge = length if charge is None else charge
    if length > S.SEG:
        seg = seg[:S.SEG]
    width = row_bits(S)
    rows = {"pub": _pack(S, "pub", {"c": [charge]}),
            "adv": _pack(S, "adv", {"dec": list(sl.decisions) + [0] * (S.D - len(sl.decisions)),
                                    "count": [len(sl.decisions)], "steps": [sl.steps]}),
            "srv": srv_row(S, sc.steps if served is None else served)}
    cert_bits = RC.load_arguments(Sc, RC.empty(Sc), RC.Window(), sc.arrivals, sl.decisions, len(sl.decisions), sl.steps,
                                  RC.encode(sc.states[0], Sc), RC.encode_window(sc.window, Sc))[2]
    rows["cert"] = cert_bits + [0] * (width["cert"] - len(cert_bits))
    wit = [b for st in sc.states for b in RC.encode(st, Sc)] + list(RC.encode_window(sc.window, Sc))
    wit += [b for arr, _b in sc.fronts for b in RC.front_bits(Sc, arr)]
    wit += [b for _a, bits in sc.fronts for b in bits]
    wit += seg + [0] * (S.SEG - len(seg))
    rows["wit"] = wit + [0] * (width["wit"] - len(wit))
    assert Z.DW and all(len(rows[r]) == width[r] for r in ROWS)
    flags = check(S, rows)
    return Case(rows, verdict(flags), flags, tuple(sc.steps), tuple(sc.symbols), length)


PHASE9 = ("context", "frequency", "hit", "reread", "untaken")


def phase9(S, sl: Slice) -> dict[str, dict]:
    """REDTEAM's phase 9 interfaces (`note:network-accounting/20261024T0558Z-finding-redteam-v3`, "PHASE 9 RESULTS") as
    `case`'s keywords, each the prover's run of `sl` with that interface chosen to lower its charge or its schedule:
    ``context``, each decision coded at the context that makes the codeword shortest, the charge that length;
    ``frequency``, each decision coded as near certain (2^16 − 1 to 1), the charge that length; ``hit``, the first
    step's first candidate takes a 2-block prefix hit; ``reread``, step 2 leaves the decision head in place, so step 3
    reads its decision again; ``untaken``, step 0 leaves the arrival head in place. ``srv``'s rows are the caller's
    (the served ones); in the first two the run's step rows are the honest ones too."""

    def coded(ctxs):
        n, syms = ((0, 0),) * EC, []
        for c, b in zip(ctxs, sl.decisions):
            _ok, f0, f1, n = AE.reference_kt(EC, ENW, n, 1, c, b)
            syms.append((b, f0, f1))
        return codeword(syms)

    cheapest = min(itertools.product(range(EC), repeat=len(sl.decisions)), key=lambda c: (len(coded(c)), c))
    certain = codeword([(b, 1, AE.MAX_TOTAL - 1) if b else (b, AE.MAX_TOTAL - 1, 1) for b in sl.decisions])
    return {"context": {"seg": coded(cheapest), "charge": len(coded(cheapest))},
            "frequency": {"seg": certain, "charge": len(certain)},
            "hit": {"hits": {0: (2,) + (0,) * (S.W + S.A - 1)}},
            "reread": {"consume": {2: (0, 0)}},
            "untaken": {"consume": {0: (0, 0)}}}


def words(bits: Sequence[int]) -> list[int]:
    """A row's u16 words, least significant bit first."""
    return [sum(int(b) << j for j, b in enumerate(bits[k:k + WORD])) for k in range(0, len(bits), WORD)]


def evaluate(st: dict, cases: Sequence[Case]) -> list[int]:
    """The unit's verdict word on each case, in one evaluation."""
    args = [[c.rows[r] for c in cases] for r in ROWS]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


@lru_cache(maxsize=None)
def _counted(key: tuple) -> dict[str, int]:
    S = SimpleNamespace(**dict(zip(STATICS, key)))
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in _sig(S)[0]]
    C.output("verdict", _build(C, S, *ins)[0])
    return C.counts()


def size(**st) -> dict[str, int]:
    """The unit's ANDs and its rows' widths at statics `st`."""
    S = SimpleNamespace(**st)
    return {"ands": _counted(tuple(st[k] for k in STATICS))["ands"], **{f"{r}_bits": b for r, b in row_bits(S).items()}}
