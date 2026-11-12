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

from verity.protocols.certifier.prover.window_charge import Window
from verity_vllm.pipeline.advice_code.window_code import (
    Book, Caps, DEFAULT_CAPS, decode as decode, Decoder as Decoder, encode as encode, Encoded as Encoded,
    Encoder as Encoder, ints, context_key, KT, pack as pack, CoderState, unpack as unpack, code_window,
)

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


def ideal_bits(windows: Sequence[Window], book: Book) -> list[float]:
    """Each window's model cost: its charge less the coder's termination and rounding."""
    c, st, out = _Ideal(book), CoderState(), []
    for win in windows:
        before = c.bits
        code_window(c, win, st)
        out.append(c.bits - before)
    return out


def fit_book(history: Iterable[Sequence[Window]], caps: Caps = DEFAULT_CAPS, prior_weight: int = 32) -> Book:
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
