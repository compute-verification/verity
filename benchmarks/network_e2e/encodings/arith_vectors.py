"""The difftest of advice v1's in-circuit encoder (`verity.core.protocols.certifier.arith_encode`) against the advice
code's coders (`code.py`, `code_v1`), over what v1 actually codes.

Every stream is `code_v1`'s: a list of windows, each one `code.py` `ArithEncoder` segment with the symbols it took,
tagged `[1, bit, f0, f1]` for a bit `code_v1.Encoder.bit` coded at its KT estimate and `[0, cum, freq, total]` for an
integer's symbols, code.py's state after each (low, high, pending, used) and the segment. With it come each bit's (name,
context, bit) from the book's priors (`kt`) and each window's charge (`charge`). Two sources:
- `decisions(trace, stops, T=)`: advice v1's windows (`v1.residual`: the replay's decisions, the arrival order's
  deviations and the hits sent past the pool bound) under a book fit to them at v1's registered caps (`v1.TABLE_CAPS`; a
  skewed book: the difftest is the coder's, not the book's), with v1's charges (`v1._string`). It checks that they are
  `v1.encode`'s string;
- `fields(trace)`: the derived rules' still-sent fields (`derived.residual`: the hits past the pool bound, the resumes,
  the order misses and async's lag bits), each field cut into windows of 64 values, under a book fit to them.

    python3 arith_vectors.py [LABEL=DIR ...] [--steps] [--circuit] [--cut N] [--out OUT.json] [--fixture PATH]

checks `arith_encode`'s references against code.py on every stream (`check`), and with `--circuit` the Definitions too:
the bit-checking step along each segment, the counting step's tally (each window's count is its segment's length, and
its finish gives v1's charge), and the book's model (`BookKT_v1`) at each bit. On each decisions stream it holds
`ChargeCount_v1`, one window's charge from its decisions alone, to `code_v1` (`check_charges`): its reference on each
window's decisions and on the decisions cut again into windows of at most `--cut` slots with gaps, and with `--circuit`
the Definition at that N on the cut. It holds `ChargeCount_v2` (`window_charge`), the whole window's charge with its
hits and order entries, to `code_v1` too (`check_windows`): its reference on each window at its own sizes (v1's
charges), on the windows with `SEEDED` hits and order entries added, and on their cut into `WHOLE` slots, and with
`--circuit` `ChargeLink_v2` on the cut's committed rows. `--steps` takes the
replay's step traces in `tests/fixtures` (`test_v1`'s, at T = 20, so they span windows, their decisions cut at the compile
slice's P·A = 14 whatever `--cut` says) and the trace-corpus fixture's
fields, and `--fixture PATH` writes them as the vectors the certifier's test reads
(`verity/core/protocols/certifier/tests/fixtures/arith_encode_vectors.json`). LABEL=DIR are trace-corpus traces
(`replay.read_corpus_trace`; `derived.load_corpus_trace`), each read twice: with the warden's ingress, where the
arrivals are in release order, and with the client's send standing for the release, as v1's own table reads them,
where they deviate and v1 sends order entries. The fixtures' arrivals never deviate. The report has each stream's
shape: symbols, segment bits and scalings per window, which is what the circuit's cost reads.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import random
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "tests" / "fixtures"


def _load(name: str) -> Any:
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


V, R, C, D, CV = _load("v1"), _load("replay"), _load("code"), _load("derived"), _load("code_v1")

#: the step's scaling slots (`arith_encode.K_MAX`), the binding the vectors are cut for
K = 18
#: `ChargeCount_v1`'s slots a window in the corpus difftest (`--cut`), and in the fixture's streams, the compile slice's P·A
CUT, SLICE_CUT = 64, 14
#: `ChargeCount_v2`'s slots a window (decisions, hits, order entries) in the whole windows' cut, and the hits and order
#: entries seeded into each decisions stream, since the fixtures' traces send almost none
WHOLE, SEEDED = (4, 2, 2), (12, 10)
EOS = 2
FIXTURE = HERE.parents[2] / "verity" / "core" / "protocols" / "certifier" / "tests" / "fixtures" / "arith_encode_vectors.json"


def _ae() -> Any:
    """`arith_encode`, for the checks only: the streams are code.py's and v1's alone."""
    from verity.core.protocols.certifier import arith_encode as AE
    assert AE.K_MAX == K
    return AE


def _wc() -> Any:
    from verity.ml.protocols.certifier import window_charge as WC
    return WC


@contextmanager
def _coded() -> Iterator[tuple[list[dict[str, list]], list[list]]]:
    """Every `code.ArithEncoder` opened inside: its symbols, tagged, and its state after each; and each bit
    `code_v1.Encoder.bit` codes, as (name, context, bit)."""
    E, Enc = C.ArithEncoder, CV.Encoder
    init, enc, put_bit, flag = E.__init__, E.encode, Enc.bit, Enc._flag
    logs: list[dict[str, list]] = []
    kt: list[list] = []

    def __init__(self: Any, out: Any) -> None:
        init(self, out)
        logs.append({"symbols": [], "states": []})
        self._vectors, self._bit = logs[-1], None

    def encode(self: Any, cum: int, freq: int, total: int) -> None:
        enc(self, cum, freq, total)
        self._vectors["symbols"].append([1, *self._bit] if self._bit else [0, cum, freq, total])
        self._vectors["states"].append([self.low, self.high, self.pending, int(self.used)])
        self._bit = None

    def bit(self: Any, name: str, b: bool, ctx: Any = 0) -> bool:
        f0, f1 = self.kt.freqs(CV._key(name, ctx))
        self._open()._bit = (int(bool(b)), f0, f1)
        kt.append([name, ctx, int(bool(b))])
        return put_bit(self, name, b, ctx)

    E.__init__, E.encode, Enc.bit, Enc._flag = __init__, encode, bit, bit
    try:
        yield logs, kt
    finally:
        E.__init__, E.encode, Enc.bit, Enc._flag = init, enc, put_bit, flag


def _bits(b: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in b)


def _stream(wins: Sequence[Any], book: Any) -> dict[str, Any]:
    """`code_v1`'s stream for `wins` under `book` (the module's docstring)."""
    with _coded() as (logs, kt):
        enc = CV.encode(wins, book)
    if CV.decode(enc.bits, enc.nbits, book, len(wins)) != list(wins):
        raise AssertionError("code_v1 doesn't decode its own encoding")
    bits, at, windows = enc.bitlist(), 0, []
    for lg, ln in zip(logs, enc.lens, strict=True):
        windows.append({"symbols": lg["symbols"], "states": lg["states"], "segment": _bits(bits[at:at + ln])})
        at += ln
    priors = {name: [list(book.priors.get(CV._key(name, c), (0, 0))) for c in ctxs] for name, ctxs in CV.BITS.items()}
    return {"priors": priors, "kt": kt, "charge": list(enc.lens), "windows": windows}


def decisions(trace: Any, stops: Sequence[int], *, T: int, row_order: str = "batch", cut: int = CUT,
              rng: random.Random | None = None, seeded: tuple[int, int] = (0, 0)) -> dict[str, Any]:
    """Advice v1's stream for `trace` (the module's docstring), with its decisions' charges (`decision_charges`) and
    its whole windows' (`window_charges`)."""
    rec = V.record_of(trace, stops)
    adv = V.advice_of(trace, rec)
    reg = V.Registered(trace.engine, tuple(stops), T=T, row_order=row_order)
    wins = V.residual(adv, rec, reg)
    reg = dataclasses.replace(reg, book=CV.fit_book([wins], V.TABLE_CAPS))
    bits, charge = V._string(wins, reg)
    if CV.pack(bits)[0] != V.encode(adv, rec, reg, check=False):
        raise AssertionError("the windows aren't v1.encode's")
    s = _stream(wins, reg.book)
    if s["charge"] != charge:
        raise AssertionError("the charges aren't v1's")
    rng = rng or random.Random(0)
    s["decisions"] = decision_charges(wins, reg.book, cut, rng)
    s["window"] = window_charges(wins, reg.book, WHOLE, rng, seeded)
    return s


def decision_charges(wins: Sequence[Any], book: Any, N: int, rng: random.Random) -> dict[str, Any]:
    """What `ChargeCount_v1` is held to: the book (`Book.to_json()`); each window's decisions alone
    (`window_charge.Advice(arrived=…)`), `code_v1`'s charge for them, and whether the window sends only decisions
    (then that charge is v1's); and the run's decisions cut again into `code_v1` windows of at most N, each placed in
    N slots in order with disabled slots between (`en`, a disabled slot's bit random), with `code_v1`'s charge for
    each. The cut's sizes are 0, N and random, so its windows reach every bin."""
    alone = [CV.Advice(arrived=w.arrived) for w in wins]
    flat = [b for w in wins for b in w.arrived]
    sizes, at = [], 0
    while at < len(flat):
        sizes.append(min(rng.choice([0, N, rng.randrange(N + 1), rng.randrange(N + 1)]), len(flat) - at))
        at += sizes[-1]
    cut, at = [], 0
    for n in sizes:
        cut.append(CV.Advice(arrived=tuple(flat[at:at + n])))
        at += n
    rows = []
    for w, length in zip(cut, CV.encode(cut, book).lens, strict=True):
        slots = set(rng.sample(range(N), len(w.arrived)))
        it = iter(w.arrived)
        en = [int(j in slots) for j in range(N)]
        rows.append([_bits(en), _bits(int(next(it)) if e else rng.randrange(2) for e in en), length])
    return {"book": book.to_json(),
            "windows": [[_bits(w.arrived), n, w == a] for w, a, n in zip(wins, alone, CV.encode(alone, book).lens)],
            "cut": {"N": N, "windows": rows}}


def check_charges(dec: dict[str, Any], *, circuit: bool) -> dict[str, Any]:
    """`ChargeCount_v1`'s reference against `code_v1` on `decision_charges`' windows, the carry passed along: each
    window's decisions alone, at N its own count (all enabled), and the cut, at its N with its gaps; with `circuit`,
    the Definition at the cut's N against the reference on every cut window, its carry the reference's."""
    AE = _ae()
    book = dec["book"]
    bad = only = 0
    carry = AE.charge_book(book, 1)[1]
    for bits, length, alone in dec["windows"]:
        n = max(len(bits), 1)
        pad = [0] * (n - len(bits))
        ok, got, carry = AE.reference_charge(n, AE.charge_book(book, n)[0], carry, [1] * len(bits) + pad,
                                             [int(c) for c in bits] + pad)
        bad += not ok or got != length
        only += alone
    N = dec["cut"]["N"]
    cb, carry = AE.charge_book(book, N)
    cases, want, cbad = [], [], 0
    for en, bits, length in dec["cut"]["windows"]:
        case = (cb, carry, [int(c) for c in en], [int(c) for c in bits])
        w = AE.reference_charge(N, *case)
        cbad += not w[0] or w[1] != length
        cases.append(case)
        want.append(w)
        carry = w[2]
    out = {"charge_windows": len(dec["windows"]), "charge_decisions": sum(len(w[0]) for w in dec["windows"]),
           "decisions_only": only, "charge_mismatches": bad, "cut_N": N, "cut_windows": len(cases),
           "cut_mismatches": cbad}
    if circuit:
        got = [g for chunk in _chunks(cases) for g in AE.evaluate_charges(N, chunk)]
        out["charge_circuit_mismatches"] = sum(g != w for g, w in zip(got, want, strict=True))
    return out


def _seed(wins: Sequence[Any], rng: random.Random, n_hits: int, n_order: int) -> list[Any]:
    """`wins` with `n_hits` hits and `n_order` order entries added at random windows, after the field's own values: the
    hits at bin edges and between, up to hit_units; the entries' gaps and rank steps likewise, up to v1's gap and index
    caps."""
    hits = [rng.choice([0, 1, 2, 3, rng.randrange(4, 1 << 8), rng.randrange(1 << 8, 1 << 16), 1 << 15, (1 << 16) - 1,
                        1 << 16]) for _ in range(n_hits)]
    own = [(w, o) for w, win in enumerate(wins) for o in win.order]
    pos, rank = (own[-1][1][0], own[-1][1][1][0]) if own else (-1, -1)
    order = []
    for _ in range(n_order):
        pos += 1 + rng.choice([0, 1, rng.randrange(2, 1 << 10), rng.randrange(1 << 10, 1 << 17),
                               (1 << 16) + rng.randrange(1 << 8), rng.randrange(1 << 17, 1 << 24), 1 << 24])
        rank = rng.choice([0, 1 << 24, max(rank, 0), rng.randrange((1 << 24) + 1),
                           min(rank + 1 + rng.randrange(1 << 10), 1 << 24), max(rank - rng.randrange(1, 1 << 10), 0)])
        order.append((pos, (rank,)))
    out = [[list(w.hits), list(w.order)] for w in wins]
    for h, w in zip(hits, sorted(rng.randrange(len(wins)) for _ in hits)):
        out[w][0].append(h)
    first = own[-1][0] if own else 0
    for o, w in zip(order, sorted(rng.randrange(first, len(wins)) for _ in order)):
        out[w][1].append(o)
    return [dataclasses.replace(win, hits=tuple(h), order=tuple(o)) for win, (h, o) in zip(wins, out)]


def _whole(w: Any) -> list:
    return [_bits(w.arrived), list(w.hits), [p for p, _r in w.order], [r[0] for _p, r in w.order]]


def window_charges(wins: Sequence[Any], book: Any, cut: tuple[int, int, int], rng: random.Random,
                   seeded: tuple[int, int] = (0, 0)) -> dict[str, Any]:
    """What `ChargeCount_v2` is held to: the run's own windows, each (decisions, hits, order positions, ranks), whose
    charges are v1's (`own`); with `seeded`, the same windows with that many hits and order entries added (`_seed`)
    and `code_v1`'s charge for each (`seeded`); and those windows' values cut again into `code_v1` windows of at most
    `cut` = (N, H, O) decisions, hits and order entries, each field in its slots in order with disabled slots between,
    a disabled slot's values random, with `code_v1`'s charge for each (`cut`). The cut's sizes are 0, the slots and
    random, so its windows reach every bin of each count."""
    out: dict[str, Any] = {"own": [_whole(w) for w in wins]}
    if any(seeded):
        wins = _seed(wins, rng, *seeded)
        out["seeded"] = [_whole(w) + [n] for w, n in zip(wins, CV.encode(wins, book).lens, strict=True)]
    flat = ([b for w in wins for b in w.arrived], [h for w in wins for h in w.hits], [o for w in wins for o in w.order])
    at, cutwins = [0, 0, 0], []
    while any(a < len(f) for a, f in zip(at, flat)):
        n = [min(rng.choice([0, S, rng.randrange(S + 1), rng.randrange(S + 1)]), len(f) - a)
             for S, f, a in zip(cut, flat, at)]
        cutwins.append(CV.Advice(arrived=tuple(flat[0][at[0]:at[0] + n[0]]), hits=tuple(flat[1][at[1]:at[1] + n[1]]),
                                 order=tuple(flat[2][at[2]:at[2] + n[2]])))
        at = [a + k for a, k in zip(at, n)]

    def place(S: int, values: Sequence, filler: Callable[[], Any]) -> tuple[list[int], list]:
        slots = set(rng.sample(range(S), len(values)))
        it = iter(values)
        en = [int(j in slots) for j in range(S)]
        return en, [next(it) if e else filler() for e in en]

    N, H, O = cut
    rows = []
    for w, length in zip(cutwins, CV.encode(cutwins, book).lens, strict=True):
        en, bits = place(N, [int(b) for b in w.arrived], lambda: rng.randrange(2))
        hen, short = place(H, list(w.hits), lambda: rng.randrange(1 << 17))
        oen, entries = place(O, list(w.order), lambda: (rng.randrange(1 << 32), (rng.randrange(1 << 25),)))
        rows.append([_bits(en), _bits(bits), _bits(hen), short, _bits(oen), [p for p, _r in entries],
                     [r[0] for _p, r in entries], length])
    return {**out, "N": N, "H": H, "O": O, "cut": rows}


def check_windows(stream: dict[str, Any], *, circuit: bool) -> dict[str, Any]:
    """`ChargeCount_v2`'s reference against `code_v1` on `window_charges`' windows, the carry passed along: the own
    windows, each at its own sizes with every slot enabled, at v1's charges; the seeded ones at theirs; the cut at its
    (N, H, O) with its gaps. With `circuit`, `ChargeLink_v2` at the cut's statics against the reference on every cut
    window, on its committed rows."""
    WC = _wc()
    book, win = stream["decisions"]["book"], stream["window"]

    def chain(rows: Sequence[list], lens: Sequence[int]) -> int:
        carry, bad = WC.window_book(book, 0, 0, 0)[1], 0
        for (bits, hits, p, r, *_x), length in zip(rows, lens, strict=True):
            N, H, O = len(bits), len(hits), len(p)
            ok, got, carry = WC.reference_window(N, H, O, WC.window_book(book, N, H, O)[0], carry, [1] * N,
                                                 [int(c) for c in bits], [1] * H, hits, [1] * O, p, r)
            bad += not ok or got != length
        return bad

    out = {"whole_windows": len(win["own"]), "whole_hits": sum(len(w[1]) for w in win["own"]),
           "whole_order": sum(len(w[2]) for w in win["own"]), "whole_mismatches": chain(win["own"], stream["charge"])}
    if "seeded" in win:
        out["seeded_mismatches"] = chain(win["seeded"], [w[-1] for w in win["seeded"]])
    N, H, O = win["N"], win["H"], win["O"]
    slots = [([int(c) for c in en], [int(c) for c in bits], [int(c) for c in hen], short, [int(c) for c in oen], p, r)
             for en, bits, hen, short, oen, p, r, _l in win["cut"]]
    try:
        ses = WC.session(N, H, O, book, slots)
    except ValueError as e:
        return {**out, "cut_windows": len(slots), "cut_whole_mismatches": len(slots), "cut_refused": str(e)}
    out.update(cut_windows=len(slots), cut_whole_mismatches=sum(s[0] != w[-1] for s, w in zip(ses, win["cut"])))
    if circuit:
        wb = WC.window_book(book, N, H, O)[0]
        cases = [(wb, c_in, c_out, *s) for (_l, c_in, c_out), s in zip(ses, slots)]
        got = [g for chunk in _chunks(cases) for g in WC.evaluate_links(N, H, O, chunk)]
        out["window_circuit_mismatches"] = sum(g != (1, s[0]) for g, s in zip(got, ses, strict=True))
    return out


def fields(trace: dict, per: int = 64) -> dict[str, Any] | None:
    """`code_v1`'s stream of the derived rules' still-sent fields for `trace` (`derived.load_trace`'s format), or None
    when it sends none."""
    cfg = D.config_of(trace)
    adv, order = D.advice_of(trace)
    res = D.residual(adv, cfg, D.holdings_of(trace), order=order)
    cols = (res.hits, res.resumes or (), res.order, res.lag or ())
    n = max(-(-len(c) // per) for c in cols)
    if not n:
        return None
    wins = [CV.Advice(hits=cols[0][j * per:(j + 1) * per], resumes=cols[1][j * per:(j + 1) * per],
                      order=cols[2][j * per:(j + 1) * per], lag=cols[3][j * per:(j + 1) * per]) for j in range(n)]
    return _stream(wins, CV.fit_book([wins]))


# --- the checks --------------------------------------------------------------------------------------------------------


def _look(seg: Sequence[int], pos: int, k: int) -> list[int]:
    return (list(seg[pos:pos + k]) + [0] * k)[:k]


def general(sym: Sequence[int]) -> list[int]:
    """A tagged symbol as code.py's (cum, freq, total): a bit's is v1's `encode(f0 if bit else 0, f1 if bit else f0,
    f0 + f1)`."""
    if sym[0]:
        _, b, f0, f1 = sym
        return [f0 if b else 0, f1 if b else f0, f0 + f1]
    return list(sym[1:])


def _chunks(xs: Sequence, n: int = 4096) -> Iterator[Sequence]:
    for at in range(0, len(xs), n):
        yield xs[at:at + n]


def check(stream: dict, *, circuit: bool, rng: random.Random | None = None, flips: int = 8) -> dict[str, Any]:
    """The references against code.py on the stream: after each symbol low and high are code.py's, a run is open
    exactly when code.py has pending bits, `used` is code.py's, and the window's finish ends its segment; a sample of
    single-bit flips, a trailing bit and a dropped bit are refused (`check_window`). The counting step's tally after each
    symbol is code.py's interval and the bits consumed, and each window's counting finish gives v1's charge and refuses
    one bit less. The book's model gives each bit's estimate, and each decision's context is
    min(run, 2) of the decisions before it. With `circuit`, the Definitions against the references on every step and
    finish (on a flipped lookahead and a disabled step at each symbol too) and every bit's model."""
    AE = _ae()
    rng = rng or random.Random(0)
    bad: list[str] = []
    n_bits, scal = [], {0: [0], 1: [0]}
    steps: dict[int, list] = {0: [], 1: []}
    counts: dict[int, list] = {0: [], 1: []}
    fins, cfins = [], []
    for w, win in enumerate(stream["windows"]):
        seg = [int(c) for c in win["segment"]]
        st, t, pos = AE.fresh(), AE.Tally(), 0
        for j, (sym, (lo, hi, pend, used)) in enumerate(zip(win["symbols"], win["states"], strict=True)):
            BIN, s = sym[0], sym[1:]
            look = _look(seg, pos, K)
            nxt, n = AE.reference_step(K, BIN, st, 1, s, look)
            if (nxt.ok, nxt.low, nxt.high, nxt.open, nxt.used) != (1, lo, hi, int(pend > 0), used):
                bad.append(f"window {w} symbol {j}")
            steps[BIN].append((st, s, look, (nxt, n)))
            counts[BIN].append((t, 1, s))
            t = AE.reference_count_step(K, BIN, AE.CW, t, 1, s)
            if (t.ok, t.low, t.high, t.used, t.bits) != (1, lo, hi, used, pos + n):
                bad.append(f"window {w} symbol {j}: the tally")
            st, pos = nxt, pos + n
            scal[BIN].append(n)
        tail = _look(seg, pos, 2)
        ok, n = AE.reference_finish(st, tail)
        if not ok or pos + n != len(seg):
            bad.append(f"window {w} finish")
        fins.append((st, tail, (ok, n)))
        cuts = [seg + [0]] + ([seg[:-1]] if seg else [])
        for k in rng.sample(range(len(seg)), min(flips, len(seg))):
            cuts.append(seg[:k] + [1 - seg[k]] + seg[k + 1:])
        if any(AE.check_window(K, 0, [general(s) for s in win["symbols"]], c) for c in cuts):
            bad.append(f"window {w}: a flipped or cut segment accepted")
        c = stream["charge"][w]
        if AE.reference_count_finish(AE.CW, t, c) != (1, c) or (c and AE.reference_count_finish(AE.CW, t, c - 1)[0]):
            bad.append(f"window {w}: the counted charge isn't v1's {c}")
        cfins.extend((t, x) for x in (c, max(c - 1, 0)))
        n_bits.append(len(seg))
    tables = {name: [tuple(p) for p in ps] for name, ps in stream["priors"].items()}
    freqs = [tuple(s[2:]) for win in stream["windows"] for s in win["symbols"] if s[0]]
    kt_cases: dict[int, list] = {}
    kbad = cbad_ctx = run = 0
    for (name, ctx, bit), f in zip(stream["kt"], freqs, strict=True):
        nc = len(tables[name])
        kt_cases.setdefault(nc, []).append((tuple(tables[name]), 1, ctx, bit))
        ok_, f0, f1, tables[name] = AE.reference_book_kt(nc, tables[name], 1, ctx, bit)
        kbad += not ok_ or (f0, f1) != f
        if name == "arrived":
            cbad_ctx += ctx != min(run, 2)
            run = 0 if bit else run + 1
    out: dict[str, Any] = {
        "windows": len(stream["windows"]), "symbols": sum(len(w["symbols"]) for w in stream["windows"]),
        "bit_symbols": len(freqs), "max_symbols": max((len(w["symbols"]) for w in stream["windows"]), default=0),
        "bits": sum(n_bits), "max_bits": max(n_bits, default=0), "max_scalings_bit": max(scal[1]),
        "max_scalings_int": max(scal[0]), "charged": sum(stream["charge"]),
        "reference_mismatches": len(bad), "reference_first": bad[:5], "kt_mismatches": kbad, "context_mismatches": cbad_ctx}
    if circuit:
        cbad, n_cases = 0, 0
        for BIN in (0, 1):
            cases, want = [], []
            for st, s, look, nxt in steps[BIN]:
                flip = list(look)
                flip[rng.randrange(K)] ^= 1
                for cs in ((st, 1, s, look), (st, 0, s, look), (st, 1, s, flip)):
                    cases.append(cs)
                    want.append(nxt if cs[1] == 1 and cs[3] is look else AE.reference_step(K, BIN, *cs))
            for at, chunk in zip(range(0, len(cases), 4096), _chunks(cases)):
                for g, w_ in zip(AE.evaluate_steps(K, BIN, chunk), want[at:at + 4096]):
                    cbad += (g[0].ok != 0) if w_[0].ok == 0 else g != w_
            ccases = counts[BIN] + [(t, 0, s) for t, _en, s in counts[BIN]]
            for chunk in _chunks(ccases):
                for g, cs in zip(AE.evaluate_counts(K, BIN, AE.CW, chunk), chunk):
                    w_ = AE.reference_count_step(K, BIN, AE.CW, *cs)
                    cbad += (g.ok != 0) if w_.ok == 0 else g != w_
            n_cases += len(cases) + len(ccases)
        got = AE.evaluate_finishes([(st, tail) for st, tail, _ in fins]) if fins else []
        cbad += sum(g != w_ for g, (_s, _t, w_) in zip(got, fins))
        got = AE.evaluate_count_finishes(AE.CW, cfins) if cfins else []
        cbad += sum(g != AE.reference_count_finish(AE.CW, t, c) for g, (t, c) in zip(got, cfins))
        for nc, cs in kt_cases.items():
            for chunk in _chunks(cs):
                for g, c_ in zip(AE.evaluate_book_kt(nc, chunk), chunk):
                    w_ = AE.reference_book_kt(nc, *c_)
                    cbad += (g[0] != 0) if w_[0] == 0 else g != w_
        out.update(circuit_cases=n_cases + len(fins) + len(cfins) + sum(map(len, kt_cases.values())),
                   circuit_mismatches=cbad)
    if "decisions" in stream:
        out.update(check_charges(stream["decisions"], circuit=circuit))
    if "window" in stream:
        out.update(check_windows(stream, circuit=circuit))
    return out


# --- the sources ---------------------------------------------------------------------------------------------------------


def step_trace(name: str) -> Any:
    """A replay fixture as `test_v1` takes it: each request given tokens where the trace has none."""
    tok = FIXTURES / f"{name}.tokens.json"
    t = R.read_step_trace(FIXTURES / f"{name}.jsonl", tokens=json.loads(tok.read_text()) if tok.exists() else None)
    t.requests = [r if r.tokens is not None else dataclasses.replace(
        r, tokens=(1000 + i,) * r.prompt_len + (3,) * (r.output_len - r.eos) + (EOS,) * r.eos)
        for i, r in enumerate(t.requests)]
    return t


def sources(corpus: Sequence[str], steps: bool, stops: Sequence[int], T: int,
            cut: int = CUT) -> Iterator[tuple[str, Callable[[], dict | None]]]:
    if steps:
        for p in sorted(FIXTURES.glob("*.jsonl")):
            yield f"{p.stem} (decisions, T=20)", lambda p=p: decisions(
                step_trace(p.stem), (EOS,), T=20, row_order="scheduling", cut=SLICE_CUT, rng=random.Random(p.stem),
                seeded=SEEDED)
        yield "corpus-smallkv (fields)", lambda: fields(D.load_corpus_trace(
            FIXTURES / "corpus-smallkv", eos_token_id=128009, stop_token_ids=(128001, 128008)))
    for spec in corpus:
        label, d = spec.split("=", 1)
        for ingress in ("warden", "client"):
            name = f"{label} (decisions, T={T}, {ingress})"
            yield name, lambda d=d, ingress=ingress, name=name: decisions(
                R.read_corpus_trace(d, ingress=ingress), stops, T=T, cut=cut, rng=random.Random(name), seeded=SEEDED)
        yield f"{label} (fields)", lambda d=d: fields(D.load_corpus_trace(d, eos_token_id=stops[0],
                                                                          stop_token_ids=tuple(stops[1:])))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="advice v1's in-circuit encoder against code.py's, symbol for symbol.")
    ap.add_argument("traces", nargs="*", help="LABEL=DIR: trace-corpus traces")
    ap.add_argument("--steps", action="store_true", help="the replay's step-trace fixtures and the corpus fixture")
    ap.add_argument("--stops", default="128001", help="the corpus traces' stop ids, comma-separated, EOS first")
    ap.add_argument("--T", type=int, default=600, help="the corpus traces' window, in 100 ms buckets")
    ap.add_argument("--circuit", action="store_true", help="evaluate the Definitions too")
    ap.add_argument("--cut", type=int, default=CUT, help="ChargeCount_v1's slots a window, the decisions cut to it")
    ap.add_argument("--out", type=Path, help="the report's JSON")
    ap.add_argument("--fixture", type=Path, help="write the streams as the certifier test's vectors")
    a = ap.parse_args(argv)
    stops = tuple(int(s) for s in a.stops.split(","))
    report: dict[str, Any] = {"K": K, "streams": {}}
    kept = []
    ok = True
    for name, make in sources(a.traces, a.steps, stops, a.T, a.cut):
        try:
            stream = make()
        except (ValueError, RuntimeError) as e:
            report["streams"][name] = {"skipped": str(e)}
            print(f"{name}: skipped ({e})", flush=True)
            ok = False
            continue
        if stream is None:
            report["streams"][name] = {"none": "nothing still sent"}
            print(f"{name}: nothing still sent", flush=True)
            continue
        r = report["streams"][name] = check(stream, circuit=a.circuit, rng=random.Random(name))
        ok &= not (r["reference_mismatches"] or r["kt_mismatches"] or r["context_mismatches"]
                   or r.get("circuit_mismatches") or r.get("charge_mismatches") or r.get("cut_mismatches")
                   or r.get("charge_circuit_mismatches") or r.get("whole_mismatches") or r.get("seeded_mismatches")
                   or r.get("cut_whole_mismatches") or r.get("window_circuit_mismatches"))
        print(f"{name}: {json.dumps(r)}", flush=True)
        kept.append({"source": name, **stream})
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    if a.fixture:
        a.fixture.write_text(json.dumps({"K": K, "streams": kept}, separators=(",", ":")) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
