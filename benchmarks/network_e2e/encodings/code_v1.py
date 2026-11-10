"""The advice code's books, fit on history: `fit_book`, the book to register, fit on historical runs' windows and never
on the audited run (a book fit to the run would be advice), and `ideal_bits`, each window's model cost before the
coder's rounding and termination.

The window coder itself is `verity_vllm.pipeline.advice_code.window_code` (`Caps`, `Book`, `encode`, `decode`; its
docstring states the code and why it is prefix-free, canonical and self-delimiting), re-exported here for the modules
and tests that load this one by path.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

<<<<<<< 99c8914848ef788e008018a5dc84bd535d48d62c
from verity.core.protocols.certifier.window_charge import Advice
||||||| merged common ancestors
from verity.core.protocols.certifier.window_charge import Window
=======
from verity.ml.protocols.certifier.window_charge import Window
from verity_vllm.pipeline.advice_code.window_code import (
    Book, Caps, DEFAULT_CAPS, decode as decode, Decoder as Decoder, encode as encode, Encoded as Encoded,
    Encoder as Encoder, ints, context_key, KT, pack as pack, CoderState, unpack as unpack, code_window,
)
>>>>>>> fa1fc89f64f1a57fa1d2d62892c18eb3a29f7627

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if f"enc_{name}" in sys.modules:
        return sys.modules[f"enc_{name}"]
    spec = importlib.util.spec_from_file_location(f"enc_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("code")


# ------------------------------------------------------------------------------------------------------ the residual


#: the bit symbols and the contexts `encode` gives them
BITS = {"arrived": (0, 1, 2), "lag": (0,)}


# ------------------------------------------------------------------------------------------------------- the book


class _Ideal:
    """Each symbol's model cost, before the coder's rounding and termination."""

    def __init__(self, book: Book) -> None:
        self.book, self.kt, self.bits = book, KT(book.priors), 0.0

    def _sym(self, name: str, v: int, lo: int, hi: int) -> int:
        v = C.in_range(name, v, lo, hi)
        self.bits += self.book.models[name].cost(v, lo, hi)
        return v

    def _flag(self, name: str, b: bool, ctx: Any = 0) -> bool:
        key, b = context_key(name, ctx), bool(b)
        f0, f1 = self.kt.freqs(key)
        self.bits += math.log2((f0 + f1) / (f1 if b else f0))
        self.kt.update(key, b)
        return b


class _Count:
    """The symbols, by name, for fitting a book."""

    def __init__(self, caps: Caps) -> None:
        self.book = Book.uniform(caps)
        self.ints: dict[str, list[int]] = {}
        self.bits: dict[str, list[int]] = {}

    def _sym(self, name: str, v: int, lo: int, hi: int) -> int:
        v = C.in_range(name, v, lo, hi)
        self.ints.setdefault(name, []).append(v)
        return v

    def _flag(self, name: str, b: bool, ctx: Any = 0) -> bool:
        self.bits.setdefault(context_key(name, ctx), [0, 0])[int(bool(b))] += 1
        return bool(b)


# ------------------------------------------------------------------------------------------------------ a window


class _State:
    """What the windows carry from one to the next: the run of decisions since the last arrival, and the last resumed
    position, order step and ingress index."""

    def __init__(self) -> None:
        self.run, self.resume, self.step, self.index = 0, -1, -1, -1


def _window(c: Any, win: Advice | None, st: _State) -> Advice:
    """One window's symbols, written once for every direction: `win` is None when decoding."""
    caps = c.book.caps

    def count(name: str, xs: Sequence) -> int:
        return c._sym(name, None if win is None else len(xs), 0, getattr(caps, name[2:]))

    arrived = []
    for j in range(count("n_arrived", win.arrived if win else ())):
        b = c._flag("arrived", None if win is None else win.arrived[j], min(st.run, 2))
        st.run = 0 if b else st.run + 1
        arrived.append(b)
    hits = [c._sym("hit", None if win is None else win.hits[j], 0, caps.hit_units)
            for j in range(count("n_hits", win.hits if win else ()))]
    resumes = []
    for j in range(count("n_resumes", win.resumes if win else ())):
        st.resume += c._sym("resume_gap", None if win is None else win.resumes[j] - st.resume - 1, 0, caps.gap) + 1
        resumes.append(st.resume)
    order = []
    for j in range(count("n_order", win.order if win else ())):
        step, idx = (None, None) if win is None else win.order[j]
        st.step += c._sym("order_gap", None if win is None else step - st.step - 1, 0, caps.gap) + 1
        got = []
        for m in range(c._sym("order_len", None if win is None else len(idx), 1, caps.order_len)):
            if win is not None and not 0 <= idx[m] <= caps.index:
                raise ValueError(f"order index {idx[m]} outside [0, {caps.index}]")
            z = c._sym("order_index", None if win is None else C.zigzag(idx[m] - st.index - 1), 0, 2 * caps.index + 1)
            i = st.index + 1 + C.unzigzag(z)
            if not 0 <= i <= caps.index:
                raise ValueError(f"order index {i} outside [0, {caps.index}]")
            st.index = i
            got.append(i)
        order.append((st.step, tuple(got)))
    lag = [c._flag("lag", None if win is None else win.lag[j], 0) for j in range(count("n_lag", win.lag if win else ()))]
    return Advice(tuple(arrived), tuple(hits), tuple(resumes), tuple(order), tuple(lag))


# ------------------------------------------------------------------------------------------------------------ API


def encode(windows: Sequence[Advice], book: Book) -> Encoded:
    """The run's residual, window by window; `lens[w]` is window w's charge."""
    e, st = Encoder(book), _State()
    for win in windows:
        e.window()
        _window(e, win, st)
    return e.finish()


def decode(bits: bytes, nbits: int, book: Book, n_windows: int) -> list[Advice]:
    """The windows back, refusing (ValueError) any string `encode` wouldn't produce."""
    d, st = Decoder(bits, nbits, book), _State()
    out = []
    for _ in range(n_windows):
        d.window()
        out.append(_window(d, None, st))
    d.finish()
    return out


def ideal_bits(windows: Sequence[Advice], book: Book) -> list[float]:
    """Each window's model cost: its charge less the coder's termination and rounding."""
    c, st, out = _Ideal(book), CoderState(), []
    for win in windows:
        before = c.bits
        code_window(c, win, st)
        out.append(c.bits - before)
    return out


<<<<<<< 99c8914848ef788e008018a5dc84bd535d48d62c
def fit_book(history: Iterable[Sequence[Advice]], caps: Caps = Caps(), prior_weight: int = 32) -> Book:
||||||| merged common ancestors
def fit_book(history: Iterable[Sequence[Window]], caps: Caps = Caps(), prior_weight: int = 32) -> Book:
=======
def fit_book(history: Iterable[Sequence[Window]], caps: Caps = DEFAULT_CAPS, prior_weight: int = 32) -> Book:
>>>>>>> fa1fc89f64f1a57fa1d2d62892c18eb3a29f7627
    """The book to register, fit on historical runs (each a sequence of windows), never on the audited run: each integer
    symbol's bin table from its samples, and each bit context's prior from its counts, scaled to at most `prior_weight`
    observations."""
    c = _Count(caps)
    for windows in history:
        st = CoderState()
        for win in windows:
            code_window(c, win, st)
    models = {n: C.Binned(C.fit_table([C.bin_of(v) for v in c.ints.get(n, [])], 0, C.bin_of(hi)))
              for n, (_, hi) in ints(caps).items()}
    priors = {}
    for k, (n0, n1) in sorted(c.bits.items()):
        s = max(1, -(-(n0 + n1) // prior_weight))
        priors[k] = (n0 // s, n1 // s)
    return Book(caps, models, priors)
