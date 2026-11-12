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
- `segment_end` is what ends the segment: the 2 bits `arith.ArithEncoder.finish` adds to a segment that coded
  anything, and none to one that coded nothing.
- `output_len` and `finish` are the window's length terms (`lengths`), plain bits after the segment: each request's
  output length, and the 1 bit that says how it ended, on a stop id or at its length.

**The classes** (`CLASSES`) group the fields as the advice's charge is stated: the prefix-cache hits; the admission,
which is when each request arrived (`decisions`) and in what order (`order`); the terminator, which is how each request
ended (`finish`); and, outside those, the output lengths, `resumes` and `lag`, and the segment's end. Two classes have
no field in the code (`NO_FIELD`): an abort, which the code refuses rather than charges, and the release buckets, which
are the certified record's.

**What is 0 by construction.** `closed` gives, for a registration, the fields no window of it can charge, each with
the rules (`RULES`) that hold it there; `where` finds a rule's line in this package's source, so a report cites the
code as it stands and a rule that moved is refused rather than cited stale. One rule says what is charged rather than
what is 0: `length width`, the bits a request's output length and its end are sent in.

**The events** (`observing`, `Seen`). A class's 0 bits say something only where its event happened, so a report
counts, beside each window's bits, what the replay's scheduler and block pool did at each executed step: the cached
blocks the pool took back for other tokens (evictions), the admissions whose prefix-cache lookup such an eviction cut
short, the requests a step left waiting and what held them, and its preemptions and resumptions. `observing` reads
them off the replay as it runs, as `codec.replay_record` reads its lookups, and changes nothing the replay computes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, NamedTuple

from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline.advice_code import codec as V
from verity_vllm.pipeline.advice_code import pool as P
from verity_vllm.pipeline.advice_code import replay as R
from verity_vllm.pipeline.advice_code import window_code as CV
from verity_vllm.pipeline.advice_code.codec import AdviceCodeError, Record, Registered

__all__ = ["bits_by_class", "CLASSES", "closed", "FIELDS", "NO_FIELD", "no_field", "observing", "Rule", "RULES", "Seen",
           "SYMBOLS", "WHY", "where"]

#: a window's parts: its segment's symbols by field, what ends the segment, and its tail's two kinds of term
FIELDS = ("decisions", "order", "hits", "resumes", "lag", "segment_end", "output_len", "finish")

#: the field of each symbol `window_code.code_window` codes
SYMBOLS = {"n_arrived": "decisions", "arrived": "decisions", "n_hits": "hits", "hit": "hits", "n_resumes": "resumes",
           "resume_gap": "resumes", "n_order": "order", "order_gap": "order", "order_len": "order",
           "order_index": "order", "n_lag": "lag", "lag": "lag"}

#: the classes a charge is stated in, and the fields of each: the three of the five named classes that have a field
#: (`NO_FIELD` has the other two), then the fields outside the five
CLASSES = {"hits": ("hits",), "admission": ("decisions", "order"), "terminator": ("finish",),
           "output lengths": ("output_len",), "resumes and lag": ("resumes", "lag"), "segment end": ("segment_end",)}

#: the field of `window_code.Caps` that bounds each segment field's count a window
_BOUND = {"decisions": "arrived", "order": "order", "hits": "hits", "resumes": "resumes", "lag": "lag"}


class Rule(NamedTuple):
    """A rule of the code: what it holds, and the module of this package and the text that holds it there: one line
    of code or, with `prose`, words of a docstring, which break over lines wherever the docstring is wrapped."""

    why: str
    module: str
    line: str
    prose: bool = False


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
    "length width": Rule("a request's output length is sent in ⌈log2(max_tokens + 1)⌉ bits and how it ended in 1",
                         "lengths", "C.width(requests[k].max_tokens + 1), 1)"),
    "aborts unmodelled": Rule("the replay refuses what it doesn't model, aborts among them", "replay",
                              "aborts and stop strings", prose=True),
    "aborts refused": Rule("a record with an output the stop rule doesn't end is refused", "codec",
                           "its output isn't one the stop rule ends"),
    "release certified": Rule("each output's release bucket is read from the certified record", "codec",
                              "release.append(reqs[first_of[x.request]].release[kept[x.request]])"),
}

#: the classes with no field in the code, and the rules that say what the code does with each instead
NO_FIELD = {"aborts": ("aborts unmodelled", "aborts refused"), "release buckets": ("release certified",)}


def no_field(reg: Registered) -> dict[str, tuple[str, ...]]:
    """`NO_FIELD` under a registration: under the certified tick a request's release bucket is also what admits it."""
    out = dict(NO_FIELD)
    if reg.tick is not None:
        out["release buckets"] = (*out["release buckets"], "tick replay")
    return out


def where(rule: str) -> tuple[str, int]:
    """The file (in this package) and the 1-based line that hold `RULES[rule]`, for prose the line its words begin on;
    refuses a rule whose text is no longer there exactly once: on one line or, for prose, as those words in that order
    with any white space between them, so a docstring wrapped anew keeps its rule."""
    r = RULES[rule]
    name = f"{r.module}.py"
    text = (Path(__file__).parent / name).read_text()
    if r.prose:
        words = r"\s+".join(map(re.escape, r.line.split()))
        at = [text.count("\n", 0, m.start()) + 1 for m in re.finditer(words, text)]
    else:
        at = [n for n, line in enumerate(text.splitlines(), 1) if r.line in line]
    if len(at) != 1:
        raise AdviceCodeError(f"the rule {rule!r} is on {len(at)} lines of {name}, not one: {r.line!r} moved, so "
                              "by_class.RULES names it anew")
    return name, at[0]


def closed(reg: Registered) -> dict[str, tuple[str, ...]]:
    """The fields no window of the registration can charge, each with the rules (`RULES`' keys) that hold it at 0: a
    segment's field whose count the registered bounds hold at 0, the segment's end where they hold every count there,
    and the length terms where the registration sends none."""
    bounds, tick = V.book_of(reg).caps, reg.tick is not None
    out: dict[str, tuple[str, ...]] = {}
    for field, name in _BOUND.items():
        if getattr(bounds, name):
            continue
        ours = ("tick caps",) if name in ("arrived", "order") else ("caps",)
        held = ("tick book",) if reg.book is not None and tick else ours if reg.book is None else ()
        more = ("refused",) if name not in ("arrived", "order") else ("tick replay", "tick refused") if tick else ()
        out[field] = (*held, "one value", *more)
    if len(out) == len(_BOUND):
        out["segment_end"] = ("no segment",)
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


#: what holds the requests a step leaves waiting (`Seen.why`): the batch is full, by its sequences (`max_num_seqs`
#: are running) or by its tokens (the step's `max_num_batched_tokens` are scheduled, or without chunked prefill the
#: prompt at the queue's head is over what is left of them); the pool has no block for the request at the queue's
#: head; the step preempted, and such a step admits nobody; or every one of them was preempted with a step in flight
#: and waits for it to come back
WHY = ("max_num_seqs", "tokens", "blocks", "preempted", "in flight")


@dataclass
class Seen:
    """What the replay's scheduler and block pool did at one executed step, beside the step's rows.

    `evicted`: the cached blocks the pool took from its free queue for other tokens, each one's hash dropped with it.
    `cut`: the admissions and resumptions whose prefix-cache lookup an eviction cut short, as (request, blocks): the
    lookup's hit ended at a block that had been cached and that the pool had since evicted from every block holding
    it, and `blocks` is how many more whole blocks, from that one on, the request's tokens share with what the pool
    has ever cached: what it would have hit had the pool dropped nothing.
    `waiting`: the requests in the engine that the step left unscheduled, and `why`, what held them (`WHY`; None where
    nobody waited). `preempted` and `resumed` are the step's own (`replay.Step`), with those of the `schedule()` calls
    since the last executed step that scheduled nothing."""

    evicted: int = 0
    cut: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    waiting: tuple[int, ...] = ()
    why: str | None = None
    preempted: tuple[int, ...] = ()
    resumed: tuple[int, ...] = ()


def _why(s: Any, step: Any, hit: Any) -> str:
    """What holds the requests `s` has left waiting after `step` (`WHY`), by the waiting loop's own order of tests."""
    e = s.e
    if step.preempted:
        return "preempted"
    head = next((r for r in (*s.skipped, *s.waiting) if not (r.stale > 0 and not r.drop_stale)), None)
    if head is None:
        return "in flight"
    left = e.max_num_batched_tokens - sum(n for _, _, n in step.rows)
    if left <= 0:
        return "tokens"
    if len(s.running) >= e.max_num_seqs:
        return "max_num_seqs"
    if not e.enable_chunked_prefill:
        n = head.num_tokens - (head.num_computed or len(hit(s.kv, head)) * e.block_size)
        if min(n, e.long_prefill_token_threshold or n) > left:
            return "tokens"
    return "blocks"


@contextmanager
def observing() -> Iterator[list[Seen]]:
    """What the replay's scheduler and pool do while this is open, one `Seen` an executed step, in the steps' order
    (`replay.Replay.steps`: a `schedule()` that scheduled nothing is no step, and what it did counts at the next that
    is). A what-if run on a fork is not the replay's own and is not counted. It wraps the scheduler's and the pool's
    methods while it is open (`engine.hooks.Hooks` puts the wrappers on and takes them off), as `codec.replay_record`
    wraps them for its lookups, reads what they compute and changes none of it."""
    steps: list[Seen] = []
    now: list[Seen] = []                            # what the `schedule()` calls since the last executed step did
    pools: dict[int, tuple[Any, set[int], set[int]]] = {}   # by pool: it, the hashes it has evicted, all it has cached
    cut: dict[int, int] = {}
    taking = [0]
    hit0, pop0, new0, cache0 = P.BlockPool.hit, P.BlockPool._pop_hash, P.BlockPool._get_new_blocks, P.BlockPool.cache_blocks
    schedule0 = R.Scheduler.schedule

    def of(kv: Any) -> tuple[Any, set[int], set[int]]:
        return pools.setdefault(id(kv), (kv, set(), set()))

    def new(kv: Any, k: int) -> list[int]:
        taking[0] += 1
        try:
            return new0(kv, k)
        finally:
            taking[0] -= 1

    def pop(kv: Any, b: int) -> None:
        h = kv.block_hash[b]
        pop0(kv, b)
        if taking[0] and h is not None and kv.mine is None:
            if now:
                now[-1].evicted += 1
            if h not in kv.by_hash:
                of(kv)[1].add(h)

    def cache(kv: Any, r: Any, num_tokens: int) -> None:
        done = kv.num_cached.get(r.i, 0)
        cache0(kv, r, num_tokens)
        if kv.mine is None:
            of(kv)[2].update(r.block_hashes[done:kv.num_cached.get(r.i, done)])

    def hit(kv: Any, r: Any) -> list[int]:
        got = hit0(kv, r)
        if kv.mine is None and now:
            _, gone, ever = of(kv)
            hashes = r.block_hashes[: (r.num_tokens - 1) // kv.B]
            k = len(got)
            cut.pop(r.i, None)
            if k < len(hashes) and hashes[k] in gone:
                more = 0
                while k + more < len(hashes) and hashes[k + more] in ever:
                    more += 1
                cut[r.i] = more
        return got

    def schedule(s: Any, on_open: Any = None) -> Any:
        if s.kv.mine is not None:
            return schedule0(s, on_open)
        if not now:
            now.append(Seen())
        cut.clear()
        step, samples = schedule0(s, on_open)
        seen = now[-1]
        seen.preempted += tuple(step.preempted)
        seen.resumed += tuple(step.resumed)
        seen.cut += [(i, cut[i]) for i in (*step.new, *step.resumed) if i in cut]
        if step.rows:
            now.pop()
            seen.waiting = tuple(r.i for r in (*s.skipped, *s.waiting))
            seen.why = _why(s, step, hit0) if seen.waiting else None
            steps.append(seen)
        return step, samples

    with Hooks("by_class.observing") as hooks:
        for name, wrapper in (("hit", hit), ("_pop_hash", pop), ("_get_new_blocks", new), ("cache_blocks", cache)):
            hooks.patch(P.BlockPool, name, wrapper)
        hooks.patch(R.Scheduler, "schedule", schedule)
        yield steps


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
            by["segment_end"] = n - sum(by.values())
            if by["segment_end"] not in (0, 2):
                raise AdviceCodeError(f"a segment ends in {by['segment_end']} bits, not the termination's 2 or none")
        return dict(enumerate(parts))
    return V._total(run)
