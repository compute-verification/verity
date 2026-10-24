"""A slice of the in-proof compile unit as one Boolean Definition: the pieces of PRF-143's statement (the certifier
PROTOCOL's "Advice v1") composed whole for one window of a few steps, so the composition is proved end to end before the
chain splits it into one unit a step.

    CompileSlice_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, Q, D, P, K, SEG} (pub, adv, cert, wit) -> verdict[16]

**The rows**, each a field list (elements least significant bit first) zero-padded to whole 16-bit words:
- ``pub``, the public inputs: the declared charge ``c`` (16 bits) and ``digest``, the step rows' hiding digest
  SHA-512(salt ‖ rows) (64 bytes), which the clear recompile gives (`rows_digest`);
- ``adv``, the committed advice, a row under the advice's hiding root (`verity.ml.commitments.advice`): the window's raw
  decision bits ``dec`` (D), their ``count`` and the window's executed step count ``steps``;
- ``cert``, the window's certified arrivals (`replay_chain.arrival_fields`: K slots of valid, id, P, O), a row the
  verifier holds to the certified record's commitment (its b‖c);
- ``wit``, the witness: the state rows S_1 … S_{P+1} and the window row W_1 (`replay_chain`'s layouts), each step's
  arrivals ``arr`` and decision bits ``bits``, the window's codeword ``seg`` (SEG bits) and the digest's salt (32 bytes).

**The circuit.** `replay_chain.ReplayLoad` from the session's empty rows (S_0 and W_0, constants) to S_1 and W_1 with
``cert``'s arrivals and ``adv``'s decisions. Then per step u = 1 … P: `replay_step.ReplayStep_v1` on S_u's state with
``arr[u]`` and ``bits[u]``, and `replay_chain.ReplayAdvance` on the same ``arr``/``bits`` wires, which holds them to W_1 at
S_u's heads and S_{u+1} to the step's next state. The step's asks (the first ``used`` of its bits) are coded in the scan's
order: per ask `arith_encode.KTContext_v1{C=3, NW=24}` under the plain context (the asked request's passes so far,
min(passes, 2): the queue head's ``pc`` for the first ask of a step that forced nothing, else 0) and
`arith_encode.ArithEncodeStep_v1{K=18, BIN=1}` against ``seg`` at the code's position; then
`arith_encode.ArithEncodeFinish_v1`. The position after the finish is the coded length, the charge. Each step's rows
(valid, request, kind, start, n), zero on an idle step, packed in `_out_fields`' order (`rows_bits`), are hashed under
the salt with one SHA-512 per 128 bytes. ``pc`` moves as `replay.encode`'s ``passed`` does: after a step whose last ask was
a 0 it is min(pc + 1, 2) when that was the step's one ask and nothing was forced (the head passed once more), else 1 (a
new head passed once); otherwise it is 0 when the step took an arrival and unchanged when it took none.

**The verdict** (16 bits, the unit's one output; `ACCEPT` = 0x1F): bit 0 ``load``, ReplayLoad's ok; bit 1 ``steps``,
every ReplayAdvance's ok (with each active step's ReplayStep ok); bit 2 ``code``, every model's and coder's ok and the
finish's, the code within SEG bits; bit 3 ``charge``, the coded length ≤ c; bit 4 ``rows``, the rows' digest is
``pub``'s. The rest are 0.

**Stand-ins** (named, each a piece not ready):
- *the cache counts*: ENC-DERIVED's hit, evictable and returned-block counts are held at 0 (ReplayStep_v1's ``hit``,
  ``evict`` and ``ret``), true of a slice that hits no prefix and stops or preempts nothing; `cache_step.CacheStep_v1`
  isn't wired, and on any other slice the rows differ from the clear recompile's and ``rows`` refuses;
- *the counting encoder*: ENC-CODE's encoder that counts its emitted bits is ArithEncodeStep_v1 checking a witness
  codeword, whose consumed positions are the count (a symbol's scalings depend only on the interval);
- *the still-sent fields*: ``adv`` holds only the decisions, the count and the step count;
- *sync scheduling*: ReplayStep_v1 models no async step;
- *the binding of ``adv`` and ``cert``*: their rows' b‖c, which the verifier checks natively against the advice
  registration's leaf and the certified record's commitment; no gate hashes them.

**Statics**: ReplayStep_v1's (R, W, A and the engine's M, T, L, B, H, CH, WM, FULL, PC, NB, and IW), the chain's Q, D,
P, K (`replay_chain.Shape`) and SEG, the codeword's cap in bits.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.sha512 import IV, compress_gadget, pad_tail
from verity.core.protocols.certifier import arith_encode as AE, replay_chain as RC, replay_step as RS
from verity.core.protocols.certifier.head_check import BIT, _part, _words as _sha_words

STATICS = RS.STATICS + ("Q", "D", "P", "K", "SEG")
CW = 16
SALT = 32
EK, EC, ENW = AE.K_MAX, 3, 24
FLAGS = ("load", "steps", "code", "charge", "rows")
ACCEPT = (1 << len(FLAGS)) - 1
ROWS = ("pub", "adv", "cert", "wit")
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
    return {"pub": [("c", 1, CW), ("digest", 64, 8)],
            "adv": [("dec", S.D, 1), ("count", 1, Z.DW), ("steps", 1, Z.PW)],
            "cert": RC.arrival_fields(Sc),
            "wit": [("state", S.P + 1, RC.width(Sc)), ("window", 1, RC.window_width(Sc)),
                    ("arr", S.P, RC._bits_of(RC.front_fields(Sc))), ("bits", S.P, S.A), ("seg", 1, S.SEG),
                    ("salt", SALT, 8)]}


def row_bits(S) -> dict[str, int]:
    return {r: WORD * -(-RC._bits_of(fs) // WORD) for r, fs in fields(S).items()}


def step_row_bits(S) -> int:
    """One step's rows in bits: R × (valid, request, kind, start, n)."""
    Z = RS.widths(step_statics(S))
    return S.R * (1 + S.IW + 1 + Z.LW + Z.TW)


def _pos_width(S) -> int:
    return _bitlen(S.SEG + EK + 2)


# --- the circuit ------------------------------------------------------------------------------------------------------------

def _consts(bits: Sequence[int]) -> list:
    return [ONE if b else ZERO for b in bits]


def _build(C, S, pub, adv, cert, wit):
    Sc, St = chain(S), step_statics(S)
    ZS = RS.widths(St)
    F = fields(S)
    p, a, c, w = (RC._split(x, F[r]) for x, r in zip((pub, adv, cert, wit), ROWS))
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

    rows += [ZERO] * (-len(rows) % 8)
    msg = list(w["salt"]) + [rows[k:k + 8] for k in range(0, len(rows), 8)]
    words = _sha_words(msg + [const_bits(b, 8) for b in pad_tail(len(msg))])
    cv = [const_bits(v, 64) for v in IV]
    for k in range(0, len(words), 16):
        cv = compress_gadget(C, cv, words[k:k + 16])
    digest = [x for k in range(8) for i in range(8) for x in cv[k][8 * (7 - i):8 * (8 - i)]]
    rows_ok = RC._eq(C, digest, [x for e in p["digest"] for x in e])

    flags = [load_ok, and_reduce(C, steps_ok), and_reduce(C, code_ok), charge_ok, rows_ok]
    return [flags + [ZERO] * (16 - len(flags))]


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BIT)) for r in ROWS), Array(16, BIT)


CompileSlice = _part(
    "CompileSlice", STATICS, _sig, _build,
    "a slice of the compile unit, one window whole: ReplayLoad, then per step ReplayStep_v1 and ReplayAdvance on the "
    "same arrival and decision wires, the asks coded under the plain contexts against the window's codeword, the coded "
    "length held to the declared charge and the step rows' salted SHA-512 to the public digest (verdict bits load, "
    "steps, code, charge, rows)")
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
    """A unit's rows (bits each, `fields`' layouts) and the verdict the reference gives them, with the scan's rows (per
    step, `RS.Out.rows`), its symbols (bit, f0, f1) and the coded length."""

    rows: dict
    verdict: int
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


def _bytes(bits: Sequence[int]) -> bytes:
    bits = list(bits) + [0] * (-len(bits) % 8)
    return bytes(sum(bits[8 * k + j] << j for j in range(8)) for k in range(len(bits) // 8))


def rows_digest(S, steps: Sequence, salt: bytes) -> bytes:
    """SHA-512(salt ‖ the window's step rows), each step's `rows_bits` in turn (P steps, idle ones zero)."""
    bits = [b for s in list(steps) + [None] * (S.P - len(steps)) for b in rows_bits(S, s)]
    return hashlib.sha512(bytes(salt) + _bytes(bits)).digest()


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def _pack(S, r: str, vals: dict) -> list[int]:
    out = [b for name, n, wd in fields(S)[r] for v in vals[name] for b in _bits(v, wd)]
    return out + [0] * (row_bits(S)[r] - len(out))


def scan(S, sl: Slice, window: Sequence[int] | None = None, cert: Sequence[RC.Arrival] | None = None) -> SimpleNamespace:
    """The window as a prover runs it from `sl`: ReplayLoad's rows, then each step by `replay_step.reference` on the state
    and the window row's arrivals and bits at the heads, the asks' symbols under the plain contexts. `window`: the
    decisions the prover puts in W_1 (default the advice's); `cert`: the arrivals it puts in ``cert`` (default `sl`'s).
    The rows past a refused advance are the step's next state with the heads moved, as the circuit computes them."""
    Sc, St = chain(S), step_statics(S)
    arrivals = tuple(map(tuple, cert if cert is not None else sl.arrivals))
    loaded = RC.load(Sc, RC.empty(Sc), RC.Window(), arrivals, sl.decisions, sl.steps)
    if loaded is None:
        raise ValueError("the load refuses the slice")
    cur, win = loaded
    if window is not None:
        win = replace(win, decisions=tuple(window))
    states, fronts, steps, ok, symbols = [cur], [], [], True, []
    pc, n = 0, {c: (0, 0) for c in range(EC)}
    for _u in range(S.P):
        arr, bits = RC.front(Sc, win, cur)
        fronts.append((arr, bits))
        if not cur.steps:
            states.append(cur)
            continue
        inp = RS.Inputs(tuple(RS.Req(i, p_, o) for i, p_, o in arr), tuple(bits), (0,) * (S.W + S.A), (0,) * (S.W + S.A),
                        (0,) * S.R)
        out = RS.reference(St, _state(cur), inp)
        if not out.ok:
            raise ValueError("the step refuses: the slice's class can't take it")
        stepped = _row(out.state)
        nxt = RC.advance(Sc, win, cur, out.ok, stepped, arr, bits, out.taken, out.used)
        if nxt is None:
            ok = False
            nxt = RC.Row(stepped.running, stepped.waiting, stepped.free, cur.qh + out.taken, cur.dh + out.used, cur.steps - 1)
        forced = not cur.running and not cur.waiting and len(arr) > 0
        for k in range(out.used):
            ctx = pc if k == 0 and not forced else 0
            f0, f1 = AE.scaled(2 * n[ctx][0] + 1, 2 * n[ctx][1] + 1)
            symbols.append((bits[k], f0, f1))
            n[ctx] = (n[ctx][0] + (not bits[k]), n[ctx][1] + bool(bits[k]))
        if out.used and not bits[out.used - 1]:
            pc = min(pc + 1, 2) if out.used == 1 and not forced else 1
        elif out.taken:
            pc = 0
        steps.append(out.rows)
        states.append(nxt)
        cur = nxt
    return SimpleNamespace(states=states, window=win, fronts=fronts, steps=steps, ok=ok, symbols=symbols, arrivals=arrivals)


def case(S, sl: Slice, salt: bytes, *, digest: bytes | None = None, charge: int | None = None,
         window: Sequence[int] | None = None, cert: Sequence[RC.Arrival] | None = None) -> Case:
    """A unit's rows for `sl` and the reference's verdict on them. `digest` and `charge` are ``pub``'s (default the
    scan's own: its rows' digest under `salt` and its coded length); `window` and `cert` are `scan`'s."""
    Sc = chain(S)
    Z = RC.widths(Sc)
    sc = scan(S, sl, window, cert)
    seg = codeword(sc.symbols)
    length = len(seg)
    own = rows_digest(S, sc.steps, salt)
    digest = own if digest is None else bytes(digest)
    charge = length if charge is None else charge
    honest = RC.load(Sc, RC.empty(Sc), RC.Window(), sc.arrivals, sl.decisions, sl.steps)
    flags = {"load": honest is not None and honest == (sc.states[0], sc.window), "steps": sc.ok,
             "code": length <= S.SEG and AE.check_window(EK, 1, sc.symbols, seg), "charge": length <= charge,
             "rows": own == digest}
    verdict = sum(int(flags[f]) << k for k, f in enumerate(FLAGS))
    if length > S.SEG:
        seg = seg[:S.SEG]
    width = row_bits(S)
    rows = {"pub": _pack(S, "pub", {"c": [charge], "digest": list(digest)}),
            "adv": _pack(S, "adv", {"dec": list(sl.decisions) + [0] * (S.D - len(sl.decisions)),
                                    "count": [len(sl.decisions)], "steps": [sl.steps]})}
    cert_bits = RC.load_arguments(Sc, RC.empty(Sc), RC.Window(), sc.arrivals, sl.decisions, len(sl.decisions), sl.steps,
                                  RC.encode(sc.states[0], Sc), RC.encode_window(sc.window, Sc))[2]
    rows["cert"] = cert_bits + [0] * (width["cert"] - len(cert_bits))
    wit = [b for st in sc.states for b in RC.encode(st, Sc)] + list(RC.encode_window(sc.window, Sc))
    wit += [b for arr, _b in sc.fronts for b in RC.front_bits(Sc, arr)]
    wit += [b for _a, bits in sc.fronts for b in bits]
    wit += seg + [0] * (S.SEG - len(seg)) + [b for x in salt for b in _bits(x, 8)]
    rows["wit"] = wit + [0] * (width["wit"] - len(wit))
    assert Z.DW and all(len(rows[r]) == width[r] for r in ROWS)
    return Case(rows, verdict, flags, tuple(sc.steps), tuple(sc.symbols), length)


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
