"""Advice v1's charge by class: each window's bits (`codec.bits_by_window`) split by the field that put them there.

A reader beside the codec. It parses a string as `codec.bits_by_window` does, codes its windows again
(`window_code.code_window`) through an encoder that notes what each symbol adds to its window's segment, and refuses
unless the string it wrote is the one it read. It changes no string, no codebook and no registration.

**The parts** (`FIELDS`) are integers and sum to the window's charge exactly.
- A symbol's bits are the shifts the arithmetic coder makes while it codes the symbol (`arith.ArithEncoder.encode`): each
  shift is one bit of the segment, written or pending. A symbol whose value is the only one possible leaves the
  interval as it was and makes none. The coder carries under 2 bits of interval from one symbol to the next, so a
  field's part is its symbols' cost under the book to within 2 bits; where one field alone codes anything, its part
  is the segment's shifts.
- `terminator` is what ends the segment: the 2 bits `arith.ArithEncoder.finish` adds to a segment that coded
  anything, and none to one that coded nothing.
- `output_len` and `finish` are the window's length terms (`lengths`), plain bits after the segment.

**The classes** (`CLASSES`) group the fields as the advice's charge is stated: the prefix-cache hits; the admission,
which is when each request arrived (`decisions`) and in what order (`order`); the terminator; the length terms; and
`resumes` and `lag`, the window's two other fields. Two classes have no field in the code (`NO_FIELD`): an abort,
which the code refuses rather than charges, and the release buckets, which are the certified record's.

**What is 0 by construction.** `closed` gives, for a registration, the fields no window of it can charge, each with
the rules (`RULES`) that hold it there; `where` finds a rule's line in this package's source, so a report cites the
code as it stands and a rule that moved is refused rather than cited stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from verity_vllm.pipeline.advice_code import codec as V
from verity_vllm.pipeline.advice_code import window_code as CV
from verity_vllm.pipeline.advice_code.codec import AdviceCodeError, Record, Registered

__all__ = ["bits_by_class", "CLASSES", "closed", "FIELDS", "NO_FIELD", "Rule", "RULES", "SYMBOLS", "where"]

#: a window's parts: its segment's symbols by field, what ends the segment, and its tail's two kinds of term
FIELDS = ("decisions", "order", "hits", "resumes", "lag", "terminator", "output_len", "finish")

#: the field of each symbol `window_code.code_window` codes
SYMBOLS = {"n_arrived": "decisions", "arrived": "decisions", "n_hits": "hits", "hit": "hits", "n_resumes": "resumes",
           "resume_gap": "resumes", "n_order": "order", "order_gap": "order", "order_len": "order",
           "order_index": "order", "n_lag": "lag", "lag": "lag"}

#: the classes a charge is stated in, and the fields of each
CLASSES = {"hits": ("hits",), "admission": ("decisions", "order"), "terminator": ("terminator",),
           "lengths": ("output_len", "finish"), "resumes and lag": ("resumes", "lag")}

#: the cap (`window_code.Caps`) on each segment field's count a window
_CAP = {"decisions": "arrived", "order": "order", "hits": "hits", "resumes": "resumes", "lag": "lag"}


class Rule(NamedTuple):
    """A rule of the code: what it holds, and the module of this package and the text of the one line that holds it."""

    why: str
    module: str
    line: str


RULES = {
    "one value": Rule("a symbol whose value is the only one possible codes nothing, and a count capped at 0 is one",
                      "arith", "if freq == total:  # the only possible symbol"),
    "no segment": Rule("a segment that coded nothing has no termination", "arith", "if self.used:"),
    "caps": Rule("v1's caps hold a window's hits, resumes and lag bits at 0: every hit is the replay's block pool's",
                 "codec", "CAPS = CV.Caps(hits=0, resumes=0, order_len=1, lag=0)"),
    "refused": Rule("decode refuses a window that sends a hit, a resume or a lag bit", "codec",
                    "if any(w.hits or w.resumes or w.lag for w in wins):"),
    "tick caps": Rule("under the certified tick v1's caps also hold a window's decisions and order entries at 0",
                      "codec", "TICK_CAPS = replace(CAPS, arrived=0, order=0)"),
    "tick book": Rule("under the certified tick a registered book whose caps aren't 0 is refused", "codec",
                      "if c is not None and (c.arrived or c.hits or c.resumes or c.order or c.lag):"),
    "tick replay": Rule("the certified tick's replay admits by the release buckets and asks no decision", "codec",
                        "rp = R.replay(engine, ios, tick=reg.tick"),
    "tick refused": Rule("under the certified tick decode refuses a window that sends a decision or an order entry",
                         "codec", "if reg.tick is not None and any(w.arrived or w.order for w in wins):"),
    "no lengths": Rule("a registration that sends no lengths has no length term in any window", "codec",
                       "if not reg.send_lengths:"),
    "aborts unmodelled": Rule("the replay refuses what it doesn't model, aborts among them", "replay",
                              "aborts and stop strings"),
    "aborts refused": Rule("a record with an output the stop rule doesn't end is refused", "codec",
                           "its output isn't one the stop rule ends"),
    "release certified": Rule("each output's release bucket is read from the certified record", "codec",
                              "release.append(reqs[first_of[x.request]].release[kept[x.request]])"),
}

#: the classes with no field in the code, and the rules that say what the code does with each instead
NO_FIELD = {"aborts": ("aborts unmodelled", "aborts refused"), "release buckets": ("release certified",)}


def where(rule: str) -> tuple[str, int]:
    """The file (in this package) and the 1-based line that hold `RULES[rule]`; refuses a rule whose text is no longer
    on exactly one line there."""
    r = RULES[rule]
    name = f"{r.module}.py"
    at = [n for n, text in enumerate((Path(__file__).parent / name).read_text().splitlines(), 1) if r.line in text]
    if len(at) != 1:
        raise AdviceCodeError(f"the rule {rule!r} is on {len(at)} lines of {name}, not one: {r.line!r} moved, so "
                              "by_class.RULES names it anew")
    return name, at[0]


def closed(reg: Registered) -> dict[str, tuple[str, ...]]:
    """The fields no window of the registration can charge, each with the rules (`RULES`' keys) that hold it at 0: a
    segment's field whose count the registered caps hold at 0, the terminator where they hold every count there, and
    the length terms where the registration sends none."""
    caps, tick = V.book_of(reg).caps, reg.tick is not None
    out: dict[str, tuple[str, ...]] = {}
    for field, cap in _CAP.items():
        if getattr(caps, cap):
            continue
        ours = ("tick caps",) if cap in ("arrived", "order") else ("caps",)
        held = ("tick book",) if reg.book is not None and tick else ours if reg.book is None else ()
        more = ("refused",) if cap not in ("arrived", "order") else ("tick replay", "tick refused") if tick else ()
        out[field] = (*held, "one value", *more)
    if len(out) == len(_CAP):
        out["terminator"] = ("no segment",)
    if not reg.send_lengths:
        out["output_len"] = out["finish"] = ("no lengths",)
    return {f: out[f] for f in FIELDS if f in out}


class _Meter:
    """`window_code.Encoder` behind `code_window`, noting per field the bits each symbol adds to the open segment: the
    coder's shifts while it codes the symbol, each one bit written or pending."""

    def __init__(self, book: CV.Book) -> None:
        self.e, self.book = CV.Encoder(book), book
        self.by: dict[str, int] = {}

    def _bits(self) -> int:
        seg = self.e.seg
        return len(self.e.w) + (0 if seg is None else seg.pending)

    def _note(self, name: str, at: int) -> None:
        if name not in SYMBOLS:
            raise AdviceCodeError(f"the window code's symbol {name!r} has no field in by_class.SYMBOLS")
        self.by[SYMBOLS[name]] += self._bits() - at

    def _sym(self, name: str, v: int, lo: int, hi: int) -> int:
        at = self._bits()
        out = self.e.value(name, v, lo, hi)
        self._note(name, at)
        return out

    def _flag(self, name: str, b: bool, ctx: Any = 0) -> bool:
        at = self._bits()
        out = self.e.bit(name, b, ctx)
        self._note(name, at)
        return out


def bits_by_class(data: bytes, record: Record, reg: Registered) -> dict[int, dict[str, int]]:
    """Each window's charge by field (`FIELDS`): the bits `codec.bits_by_window` charges the window, split, so each
    window's parts sum to its charge. Refuses what `codec.bits_by_window` refuses."""
    def run() -> dict[int, dict[str, int]]:
        wins, tails = V._parse(data, reg, record)
        m, st = _Meter(V.book_of(reg)), CV.CoderState()
        parts = []
        for win, values, widths in zip(wins, tails, V.length_widths(record, reg)):
            m.e.window()
            m.by = dict.fromkeys(FIELDS, 0)
            CV.code_window(m, win, st)
            for j, (v, n) in enumerate(zip(values, widths)):
                m.e.raw(v, n)
                m.by["finish" if j % 2 else "output_len"] += n
            parts.append(m.by)
        enc = m.e.finish()
        if enc.bits != bytes(data):
            raise AdviceCodeError("the windows the string decodes to code as another string")
        for by, n in zip(parts, enc.lens):
            by["terminator"] = n - sum(by.values())
            if by["terminator"] not in (0, 2):
                raise AdviceCodeError(f"a segment ends in {by['terminator']} bits, not the termination's 2 or none")
        return dict(enumerate(parts))
    return V._total(run)
