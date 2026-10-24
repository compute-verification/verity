"""The difftest of advice v1's in-circuit encoder (`verity.core.protocols.certifier.arith_encode`) against ENC-CODE's
coder (`code.py`), over what v1 and the residual's coder actually code.

Two kinds of stream, each a list of windows, and each window the symbols code.py's `ArithEncoder` took, its state after
each (low, high, pending, used) and the segment it wrote:
- `decisions(trace, stops, T=)`: advice v1's (`v1.encode`, contexts "plain"): the replay's decisions as `(bit, f0, f1)`
  at ENC-REPLAY's KT estimate, scaled as v1 scales it, one segment per window, with each decision's context and bit
  (`kt`) from the model's start (`priors`). It checks that the segments are `v1.encode`'s and that the decisions' windows
  never decrease along the replay, so the circuit runs one encoder at a time;
- `fields(trace)`: `code_v1`'s encoder over ENC-DERIVED's still-sent fields (`derived.residual`: the hits past the pool
  bound, the resumes, the order misses and async's lag bits), each field cut into windows of 64 values, as the general
  `(cum, freq, total)` symbols code.py receives, under a book fit to the fields (a skewed one: the difftest is the
  coder's, not the book's).

    python3 arith_vectors.py [LABEL=DIR ...] [--steps] [--circuit] [--out OUT.json] [--fixture PATH]

checks `arith_encode`'s references against code.py on every stream (`check`), and with `--circuit` the Definitions too;
`--steps` takes the replay's step traces in `tests/fixtures` (`test_v1`'s, at T = 20, so they span windows) and the
TRACE-CORPUS fixture's fields, and `--fixture PATH` writes them as the vectors the certifier's test reads
(`verity/core/protocols/certifier/tests/fixtures/arith_encode_vectors.json`). LABEL=DIR are TRACE-CORPUS traces
(`replay.read_corpus_trace`, the warden's ingress; `derived.load_corpus_trace`). The report has each stream's shape:
symbols, segment bits and scalings per window, which is what the circuit's cost reads.
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

from verity.core.protocols.certifier import arith_encode as AE  # noqa: E402

K = AE.K_MAX
NW = 24
EOS = 2


@contextmanager
def _coded() -> Iterator[list[dict[str, list]]]:
    """Every `code.ArithEncoder` opened inside: its symbols and its state after each."""
    E = C.ArithEncoder
    init, enc = E.__init__, E.encode
    logs: list[dict[str, list]] = []

    def __init__(self: Any, out: Any) -> None:
        init(self, out)
        logs.append({"symbols": [], "states": []})
        self._vectors = logs[-1]

    def encode(self: Any, cum: int, freq: int, total: int) -> None:
        enc(self, cum, freq, total)
        self._vectors["symbols"].append([cum, freq, total])
        self._vectors["states"].append([self.low, self.high, self.pending, int(self.used)])

    E.__init__, E.encode = __init__, encode
    try:
        yield logs
    finally:
        E.__init__, E.encode = init, enc


@contextmanager
def _logged_model(log: list[tuple[Any, int]], priors: dict) -> Iterator[None]:
    """ENC-REPLAY's model as v1's `_run` opens it, each update logged as (context, bit), its starting counts in
    `priors`."""
    orig = R._model

    def model(contexts: str) -> Any:
        m = orig(contexts)
        priors.update(m.n)
        update = m.update

        def logged(ctx: Any, bit: bool) -> None:
            log.append((ctx, int(bit)))
            update(ctx, bit)

        m.update = logged
        return m

    R._model = model
    try:
        yield
    finally:
        R._model = orig


def _bits(b: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in b)


def decisions(trace: Any, stops: Sequence[int], *, T: int, row_order: str = "batch") -> dict[str, Any]:
    """Advice v1's stream for `trace` under contexts "plain" (the module's docstring)."""
    rec = V.record_of(trace, stops)
    adv = V.advice_of(trace, rec)
    reg = V.Registered(trace.engine, tuple(stops), T=T, contexts="plain", row_order=row_order)
    order = V.engine_order(rec)
    index = {rec.requests[k].id: e for e, k in enumerate(order)}
    first = [-1] * len(order)
    for s, st in enumerate(adv.steps):
        for x in st.rows:
            e = index[adv.requests[x.request].id]
            if first[e] < 0:
                first[e] = s
    wins: list[int] = []

    def bit_of(i: int, k: int, w: int, f0: int, f1: int) -> int:
        wins.append(w)
        return int(k >= first[i])

    log: list[tuple[Any, int]] = []
    priors: dict = {}
    with _logged_model(log, priors):
        _rp, syms = V._run(rec, reg, reg.engine, order, V.windows(rec, T), bit_of)
    if any(b < a for a, b in zip(wins, wins[1:])):
        raise AssertionError("a decision's window below the one before it: the circuit's one encoder can't code it")
    with _coded() as logs:
        segs = [V._segment(s) for s in syms]
    if V._pack(None if reg.pinned else reg.engine.num_gpu_blocks, segs) != V.encode(adv, rec, reg, check=False):
        raise AssertionError("the segments aren't v1.encode's")
    ctxs = sorted({c for c, _ in log} | set(priors))
    if any(len(c) != 1 or not 0 <= c[0] < 3 for c in ctxs):
        raise AssertionError(f"plain contexts are (min(passes, 2),), not {ctxs[:4]}")
    windows = [{"symbols": [list(s) for s in ss], "states": lg["states"], "segment": _bits(seg)}
               for ss, lg, seg in zip(syms, logs, segs, strict=True)]
    return {"BIN": 1, "C": 3, "NW": NW, "priors": [list(priors.get((c,), (0, 0))) for c in range(3)],
            "kt": [[c[0], b] for c, b in log], "windows": windows}


def fields(trace: dict, per: int = 64) -> dict[str, Any] | None:
    """`code_v1`'s stream of ENC-DERIVED's still-sent fields for `trace` (`derived.load_trace`'s format), or None when
    it sends none."""
    cfg = D.config_of(trace)
    adv, order = D.advice_of(trace)
    res = D.residual(adv, cfg, D.holdings_of(trace), order=order)
    cols = (res.hits, res.resumes or (), res.order, res.lag or ())
    n = max(-(-len(c) // per) for c in cols)
    if not n:
        return None
    wins = [CV.Window(hits=cols[0][j * per:(j + 1) * per], resumes=cols[1][j * per:(j + 1) * per],
                      order=cols[2][j * per:(j + 1) * per], lag=cols[3][j * per:(j + 1) * per]) for j in range(n)]
    book = CV.fit_book([wins])
    with _coded() as logs:
        enc = CV.encode(wins, book)
    if CV.decode(enc.bits, enc.nbits, book, len(wins)) != wins:
        raise AssertionError("code_v1 doesn't decode its own encoding")
    bits, at, windows = enc.bitlist(), 0, []
    for lg, ln in zip(logs, enc.lens, strict=True):
        windows.append({"symbols": lg["symbols"], "states": lg["states"], "segment": _bits(bits[at:at + ln])})
        at += ln
    return {"BIN": 0, "windows": windows}


# --- the checks --------------------------------------------------------------------------------------------------------


def _look(seg: Sequence[int], pos: int, k: int) -> list[int]:
    return (list(seg[pos:pos + k]) + [0] * k)[:k]


def walk(stream: dict) -> Iterator[tuple[int, list[tuple[AE.Coder, list, list[int], tuple[AE.Coder, int]]], tuple]]:
    """Each window's reference run along its segment: (window, [(state, symbol, seg, (next, n))], (state, last 2 bits,
    (ok, n), end))."""
    for w, win in enumerate(stream["windows"]):
        seg = [int(c) for c in win["segment"]]
        st, pos, steps = AE.fresh(), 0, []
        for sym in win["symbols"]:
            look = _look(seg, pos, K)
            nxt = AE.reference_step(K, stream["BIN"], st, 1, sym, look)
            steps.append((st, sym, look, nxt))
            st, pos = nxt[0], pos + nxt[1]
        tail = _look(seg, pos, 2)
        yield w, steps, (st, tail, AE.reference_finish(st, tail), pos)


def check(stream: dict, *, circuit: bool, rng: random.Random | None = None, flips: int = 8) -> dict[str, Any]:
    """The references against code.py on the stream: after each symbol low and high are code.py's, a run is open
    exactly when code.py has pending bits, `used` is code.py's, and the window's finish ends its segment; a sample of
    single-bit flips, a trailing bit and a dropped bit are refused (`check_window`); the model's estimates are v1's.
    With `circuit`, the Definitions against the references on every step and finish, and on a flipped lookahead and a
    disabled step at each symbol."""
    rng = rng or random.Random(0)
    BIN, bad, n_sym, n_bits, scal = stream["BIN"], [], 0, [], []
    steps_all, fins_all = [], []
    for w, steps, (st, tail, (ok, n), pos) in walk(stream):
        win = stream["windows"][w]
        seg = [int(c) for c in win["segment"]]
        for j, ((_s, _sym, _look, (nxt, _n)), (lo, hi, pend, used)) in enumerate(zip(steps, win["states"], strict=True)):
            if (nxt.ok, nxt.low, nxt.high, nxt.open, nxt.used) != (1, lo, hi, int(pend > 0), used):
                bad.append(f"window {w} symbol {j}")
        if not ok or pos + n != len(seg):
            bad.append(f"window {w} finish")
        syms = win["symbols"]
        cuts = [seg + [0]] + ([seg[:-1]] if seg else [])
        for k in rng.sample(range(len(seg)), min(flips, len(seg))):
            cuts.append(seg[:k] + [1 - seg[k]] + seg[k + 1:])
        if any(AE.check_window(K, BIN, syms, c) for c in cuts):
            bad.append(f"window {w}: a flipped or cut segment accepted")
        n_sym += len(syms)
        n_bits.append(len(seg))
        scal += [x[3][1] for x in steps]
        steps_all += steps
        fins_all.append((st, tail, (ok, n)))
    out: dict[str, Any] = {"windows": len(stream["windows"]), "symbols": n_sym, "max_symbols": max(
        (len(w["symbols"]) for w in stream["windows"]), default=0), "bits": sum(n_bits), "max_bits": max(n_bits, default=0),
        "max_scalings": max(scal, default=0), "reference_mismatches": len(bad), "reference_first": bad[:5]}
    kt_cases = []
    if "kt" in stream:
        counts = [tuple(p) for p in stream["priors"]]
        f = [tuple(s[1:]) for win in stream["windows"] for s in win["symbols"]]
        kbad = 0
        for j, ((ctx, bit), fs) in enumerate(zip(stream["kt"], f, strict=True)):
            kt_cases.append((tuple(counts), 1, ctx, bit))
            ok_, f0, f1, counts = AE.reference_kt(stream["C"], stream["NW"], counts, 1, ctx, bit)
            kbad += not ok_ or (f0, f1) != fs
        out.update(kt_mismatches=kbad)
    if circuit:
        cases, want = [], []
        for st, sym, look, nxt in steps_all:
            flip = list(look)
            flip[rng.randrange(K)] ^= 1
            for c in ((st, 1, sym, look), (st, 0, sym, look), (st, 1, sym, flip)):
                cases.append(c)
                want.append(nxt if c[1] == 1 and c[3] is look else AE.reference_step(K, BIN, *c))
        cbad = 0
        for at in range(0, len(cases), 4096):
            got = AE.evaluate_steps(K, BIN, cases[at:at + 4096])
            for g, w_ in zip(got, want[at:at + 4096]):
                cbad += (g[0].ok != 0) if w_[0].ok == 0 else g != w_
        fin = AE.evaluate_finishes([(st, tail) for st, tail, _ in fins_all]) if fins_all else []
        cbad += sum(g != w_ for g, (_s, _t, w_) in zip(fin, fins_all))
        if kt_cases:
            for at in range(0, len(kt_cases), 4096):
                chunk = kt_cases[at:at + 4096]
                got = AE.evaluate_kt(stream["C"], stream["NW"], chunk)
                for g, c in zip(got, chunk):
                    w_ = AE.reference_kt(stream["C"], stream["NW"], *c)
                    cbad += (g[0] != 0) if w_[0] == 0 else g != w_
        out.update(circuit_cases=len(cases) + len(fins_all) + len(kt_cases), circuit_mismatches=cbad)
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


def sources(corpus: Sequence[str], steps: bool, stops: Sequence[int], T: int) -> Iterator[tuple[str, Callable[[], dict | None]]]:
    if steps:
        for p in sorted(FIXTURES.glob("*.jsonl")):
            yield f"{p.stem} (decisions, T=20)", lambda p=p: decisions(step_trace(p.stem), (EOS,), T=20,
                                                                       row_order="scheduling")
        yield "corpus-smallkv (fields)", lambda: fields(D.load_corpus_trace(
            FIXTURES / "corpus-smallkv", eos_token_id=128009, stop_token_ids=(128001, 128008)))
    for spec in corpus:
        label, d = spec.split("=", 1)
        yield f"{label} (decisions, T={T})", lambda d=d: decisions(R.read_corpus_trace(d, ingress="warden"), stops, T=T)
        yield f"{label} (fields)", lambda d=d: fields(D.load_corpus_trace(d, eos_token_id=stops[0],
                                                                          stop_token_ids=tuple(stops[1:])))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="advice v1's in-circuit encoder against code.py's, symbol for symbol.")
    ap.add_argument("traces", nargs="*", help="LABEL=DIR: TRACE-CORPUS traces")
    ap.add_argument("--steps", action="store_true", help="the replay's step-trace fixtures and the corpus fixture")
    ap.add_argument("--stops", default="128001", help="the corpus traces' stop ids, comma-separated, EOS first")
    ap.add_argument("--T", type=int, default=600, help="the corpus traces' window, in 100 ms buckets")
    ap.add_argument("--circuit", action="store_true", help="evaluate the Definitions too")
    ap.add_argument("--out", type=Path, help="the report's JSON")
    ap.add_argument("--fixture", type=Path, help="write the streams as the certifier test's vectors")
    a = ap.parse_args(argv)
    stops = tuple(int(s) for s in a.stops.split(","))
    report: dict[str, Any] = {"K": K, "streams": {}}
    kept = []
    ok = True
    for name, make in sources(a.traces, a.steps, stops, a.T):
        try:
            stream = make()
        except (ValueError, RuntimeError) as e:
            report["streams"][name] = {"skipped": str(e)}
            print(f"{name}: skipped ({e})", flush=True)
            continue
        if stream is None:
            report["streams"][name] = {"none": "nothing still sent"}
            print(f"{name}: nothing still sent", flush=True)
            continue
        r = report["streams"][name] = check(stream, circuit=a.circuit, rng=random.Random(name))
        ok &= not r["reference_mismatches"] and not r.get("kt_mismatches") and not r.get("circuit_mismatches")
        print(f"{name}: {json.dumps(r)}", flush=True)
        kept.append({"source": name, **stream})
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    if a.fixture:
        a.fixture.write_text(json.dumps({"K": K, "streams": kept}, separators=(",", ":")) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
