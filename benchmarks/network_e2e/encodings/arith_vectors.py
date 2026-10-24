"""The difftest of advice v1's in-circuit encoder (`verity.core.protocols.certifier.arith_encode`) against ENC-CODE's
coders (`code.py`, `code_v1`), over what v1 actually codes.

Every stream is `code_v1`'s: a list of windows, each one `code.py` `ArithEncoder` segment with the symbols it took,
tagged `[1, bit, f0, f1]` for a bit `code_v1.Encoder.bit` coded at its KT estimate and `[0, cum, freq, total]` for an
integer's symbols, code.py's state after each (low, high, pending, used) and the segment. With it come each bit's (name,
context, bit) from the book's priors (`kt`), the header (`NB` its width, 0 for none, and `pool`) and each window's charge
(`charge`). Two sources:
- `decisions(trace, stops, T=)`: advice v1's windows (`v1.residual`: the replay's decisions and the hits sent past the
  pool bound) under a book fit to them at v1's registered caps (`v1.TABLE_CAPS`; a skewed book: the difftest is the
  coder's, not the book's), with v1's header and charges (`v1._string`). It checks that they are `v1.encode`'s string;
- `fields(trace)`: ENC-DERIVED's still-sent fields (`derived.residual`: the hits past the pool bound, the resumes, the
  order misses and async's lag bits), each field cut into windows of 64 values, under a book fit to them.

    python3 arith_vectors.py [LABEL=DIR ...] [--steps] [--circuit] [--out OUT.json] [--fixture PATH]

checks `arith_encode`'s references against code.py on every stream (`check`), and with `--circuit` the Definitions too:
the bit-checking step along each segment and each window's finish. `--steps` takes the
replay's step traces in `tests/fixtures` (`test_v1`'s, at T = 20, so they span windows) and the TRACE-CORPUS fixture's
fields, and `--fixture PATH` writes them as the vectors the certifier's test reads
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

#: the step's scaling slots (`arith_encode.K_MAX`), the binding the vectors are cut for
K = 18
EOS = 2
FIXTURE = HERE.parents[2] / "verity" / "core" / "protocols" / "certifier" / "tests" / "fixtures" / "arith_encode_vectors.json"


def _ae() -> Any:
    """`arith_encode`, for the checks only: the streams are code.py's and v1's alone."""
    from verity.core.protocols.certifier import arith_encode as AE
    assert AE.K_MAX == K
    return AE


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
    """`code_v1`'s stream for `wins` under `book` (the module's docstring), with no header."""
    with _coded() as (logs, kt):
        enc = CV.encode(wins, book)
    if CV.decode(enc.bits, enc.nbits, book, len(wins)) != list(wins):
        raise AssertionError("code_v1 doesn't decode its own encoding")
    bits, at, windows = enc.bitlist(), 0, []
    for lg, ln in zip(logs, enc.lens, strict=True):
        windows.append({"symbols": lg["symbols"], "states": lg["states"], "segment": _bits(bits[at:at + ln])})
        at += ln
    priors = {name: [list(book.priors.get(CV._key(name, c), (0, 0))) for c in ctxs] for name, ctxs in CV.BITS.items()}
    return {"priors": priors, "kt": kt, "NB": 0, "pool": 0, "charge": list(enc.lens), "windows": windows}


def decisions(trace: Any, stops: Sequence[int], *, T: int, row_order: str = "batch") -> dict[str, Any]:
    """Advice v1's stream for `trace` (the module's docstring)."""
    rec = V.record_of(trace, stops)
    adv = V.advice_of(trace, rec)
    reg = V.Registered(trace.engine, tuple(stops), T=T, row_order=row_order)
    wins = V.residual(adv, rec, reg)
    reg = dataclasses.replace(reg, book=CV.fit_book([wins], V.TABLE_CAPS))
    blocks = None if reg.pinned else reg.engine.num_gpu_blocks
    bits, charge = V._string(blocks, wins, reg)
    if CV.pack(bits)[0] != V.encode(adv, rec, reg, check=False):
        raise AssertionError("the windows aren't v1.encode's")
    s = _stream(wins, reg.book)
    s.update(NB=0 if blocks is None else reg.max_gpu_blocks.bit_length(), pool=blocks or 0, charge=charge)
    return s


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
    single-bit flips, a trailing bit and a dropped bit are refused (`check_window`). Each decision's context is min(run,
    2) of the decisions before it, a function of the bits. With `circuit`, the Definitions against the references on
    every step and finish, and on a flipped lookahead and a disabled step at each symbol."""
    AE = _ae()
    rng = rng or random.Random(0)
    bad: list[str] = []
    n_bits, scal = [], {0: [0], 1: [0]}
    steps: dict[int, list] = {0: [], 1: []}
    fins = []
    for w, win in enumerate(stream["windows"]):
        seg = [int(c) for c in win["segment"]]
        st, pos = AE.fresh(), 0
        for j, (sym, (lo, hi, pend, used)) in enumerate(zip(win["symbols"], win["states"], strict=True)):
            BIN, s = sym[0], sym[1:]
            look = _look(seg, pos, K)
            nxt, n = AE.reference_step(K, BIN, st, 1, s, look)
            if (nxt.ok, nxt.low, nxt.high, nxt.open, nxt.used) != (1, lo, hi, int(pend > 0), used):
                bad.append(f"window {w} symbol {j}")
            steps[BIN].append((st, s, look, (nxt, n)))
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
        n_bits.append(len(seg))
    freqs = [s for win in stream["windows"] for s in win["symbols"] if s[0]]
    if len(freqs) != len(stream["kt"]):
        bad.append("the bits aren't the model's")
    cbad_ctx = run = 0
    for name, ctx, bit in stream["kt"]:
        if name == "arrived":
            cbad_ctx += ctx != min(run, 2)
            run = 0 if bit else run + 1
    out: dict[str, Any] = {
        "windows": len(stream["windows"]), "symbols": sum(len(w["symbols"]) for w in stream["windows"]),
        "bit_symbols": len(freqs), "max_symbols": max((len(w["symbols"]) for w in stream["windows"]), default=0),
        "bits": sum(n_bits), "max_bits": max(n_bits, default=0), "max_scalings_bit": max(scal[1]),
        "max_scalings_int": max(scal[0]), "charged": sum(stream["charge"]), "reference_mismatches": len(bad),
        "reference_first": bad[:5], "context_mismatches": cbad_ctx}
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
            n_cases += len(cases)
        got = AE.evaluate_finishes([(st, tail) for st, tail, _ in fins]) if fins else []
        cbad += sum(g != w_ for g, (_s, _t, w_) in zip(got, fins))
        out.update(circuit_cases=n_cases + len(fins), circuit_mismatches=cbad)
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
            ok = False
            continue
        if stream is None:
            report["streams"][name] = {"none": "nothing still sent"}
            print(f"{name}: nothing still sent", flush=True)
            continue
        r = report["streams"][name] = check(stream, circuit=a.circuit, rng=random.Random(name))
        ok &= not (r["reference_mismatches"] or r["context_mismatches"] or r.get("circuit_mismatches"))
        print(f"{name}: {json.dumps(r)}", flush=True)
        kept.append({"source": name, **stream})
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    if a.fixture:
        a.fixture.write_text(json.dumps({"K": K, "streams": kept}, separators=(",", ":")) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
