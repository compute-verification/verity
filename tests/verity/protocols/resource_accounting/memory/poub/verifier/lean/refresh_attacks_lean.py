"""PoUB's refresh attacks, each given its verdict by the compiled Lean verifier (`Poub.Session`, of record) on the slot
schedule: a session is driven by `ra_cases.Run`'s books with refresh off, as `test_refresh_lean.Refresh` drives one, and
sent to the Lean with refresh on, its appends naming their slots and origins and a re-key's openings their origin's leaf.
An accepted session's free receipts are the Lean ledger's credit on the schedule check, less the stored forms the
prover held and read. `python refresh_attacks_lean.py [TRIALS]` prints the table and the sampled rates."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction

import ra_cases as M
import test_refresh_lean as R
import verity_difftest as D

from verity.protocols.resource_accounting.memory.poub.verifier.receipts import Plan
from verity.protocols.resource_accounting.memory.poub.verifier.rota import Schedule

N, WINDOWS, LIFETIME, FRESH_SLOTS = 64, 6, 2, 4
"""The reference session's: 64 registered blocks, W = 6, L = 2, F = 4 fresh slots a window (T = 53)."""
SAMPLES = 16
"""The reference session's openings a window, which the rates draw; the pairs open every receipt."""
SCHEDULE_CAP = 4096
CLASS = N // LIFETIME


class Session(R.Refresh):
    """One session at the reference shape, its verifier's coins from `coins_label`. Blocks are named by the append's
    name, a run `(first, count)`, or `(name, m)` for block m of an append."""

    def __init__(self, *, samples: int = 10**6, coins_label: bytes = b"coins") -> None:
        M.Run.__init__(self, M.World(blocks=N), windows=WINDOWS, lifetime=LIFETIME, samples=samples)
        if coins_label != b"coins":
            reg = self.store.registration
            self.coins = M.coins.local(reg.digest, 0, WINDOWS, M.Entropy(coins_label))
            self.s = M.AuditRun(self.params, reg, None, self.coins, not_before_ns=0, clock=self.clock)
        self.refresh, self.fresh_slots = True, FRESH_SLOTS
        self.by_slot, self.origins = {}, {}
        self.schedule = Schedule(N, LIFETIME, WINDOWS, FRESH_SLOTS)
        self.named: dict[str, object] = {}
        self.skip: set[str] = set()
        """The control's moves a strategy leaves out."""
        self.forge: dict[str, Callable[[Session], list[bytes]]] = {}
        """The payloads a strategy sends in place of a control re-key's."""
        self.stopped = False
        """Set by a strategy whose last move the verifier refuses and the books can't follow."""

    def runs(self, what) -> tuple[int, int]:
        if type(what) is str:
            return self.named[what].first, self.named[what].count
        if type(what[0]) is str:
            a = self.named[what[0]]
            assert 0 <= what[1] < a.count
            return a.first + what[1], 1
        return what

    def stored(self, *what) -> list[bytes]:
        """The stored forms of `what`'s blocks, as a carrier's payloads."""
        out = []
        for w in what:
            first, count = self.runs(w)
            out += [self.store.blocks[j] for j in range(first, first + count)]
        return out

    def slot_of(self, kind: tuple) -> int:
        """The current window's slot of `kind`, `("registered",)` or `("rekey", k)`."""
        return next(k for k in self.schedule.slots(self.t) if self.schedule.slot_at(k)[1][:len(kind)] == kind)

    def fresh_at(self, name: str, i: int, payloads: list[bytes], slot: int | None = None) -> None:
        """A fresh append in the current window's i-th fresh slot (or `slot`)."""
        slot = self.schedule.fresh_slot(self.t, i) if slot is None else slot
        a = self.store.append(payloads, M.salts(b"salts of " + name.encode(), len(payloads)))
        self.send(a.root, a.count, slot)
        self.by_slot[slot], self.named[name] = a, a

    def registered_rekey(self, name: str) -> None:
        """The current window's registered re-key slot: its class run, its payloads unless forged."""
        lo, hi = self.schedule.class_run(self.t % LIFETIME)
        payloads = self.forge[name](self) if name in self.forge else list(self.store.payloads[lo:hi])
        self.named[name] = self._rekey(payloads, list(self.store.salts[lo:hi]), self.slot_of(("registered",)),
                                       {"registered": lo}, lo, None)

    def fresh_rekey(self, name: str, of: str, origin_slot: int | None = None) -> None:
        """The current window's re-key slot of `of`'s fresh slot, whole, naming `origin_slot` (default that slot)."""
        k = next(s for s, a in self.by_slot.items() if a is self.named[of])
        o = k if origin_slot is None else origin_slot
        b = self.by_slot[o]
        payloads = (self.forge[name](self) if name in self.forge
                    else [self.store.payload(j) for j in range(b.first, b.first + b.count)])
        self.named[name] = self._rekey(payloads, list(b.salts), self.slot_of(("rekey", k)), {"append": o}, b.first, b)

    def read(self, *what) -> object:
        return self.run(Plan(_canonical(self.runs(w) for w in what), self.store.extent))

    def held_reads(self) -> int:
        """Receipts backed by a stored form the prover holds: a pass's reads less the blocks whose stored form is the
        payload of another block of its plan (a carrier's)."""
        out = 0
        for ps in self.passes.values():
            blocks = list(ps.plan)
            forms = {self.store.blocks[j]: j for j in blocks}
            carried = {forms[m] for c in blocks if (m := self.store.payload(c)) in forms and forms[m] != c}
            out += len(blocks) - len(carried)
        return out


def _canonical(runs: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for f, c in sorted(runs):
        if out and out[-1][0] + out[-1][1] == f:
            out[-1][1] += c
        else:
            out.append([f, c])
    return tuple((f, c) for f, c in out)


def _key(t: int, c: int) -> int | None:
    """The window of the re-key a plan of t reads class c under, None for the registration's key."""
    w = t if t % LIFETIME == c else t - 1
    return w if w >= 1 else None


def control(s: Session) -> list:
    """The honest prover's moves in window t before its pass, and what the pass reads: a KV append F of 2 pages in
    window 0's first fresh slot; each window's registered re-key `R{c}@{t}`; F's re-key `F@{t}` in windows 2 and 4;
    each chunk read under the key its re-key gives it at t, F once out of probation. A name in `s.skip` is not sent."""
    t = s.t
    if t == 0 and "F" not in s.skip:
        s.fresh_at("F", 0, M.pages(1000, 2))
    if t >= 1 and (name := f"R{t % LIFETIME}@{t}") not in s.skip:
        s.registered_rekey(name)
    if t >= LIFETIME and t % LIFETIME == 0 and (name := f"F@{t}") not in s.skip and "F" in s.named:
        s.fresh_rekey(name, "F")
    reads: list = []
    for c in range(LIFETIME):
        w = _key(t, c)
        if w is None:
            reads.append(s.schedule.class_run(c)[:1] + (CLASS,))
        elif f"R{c}@{w}" in s.named:
            reads.append(f"R{c}@{w}")
    if t >= LIFETIME and f"F@{t - t % LIFETIME}" in s.named:
        reads.append(f"F@{t - t % LIFETIME}")
    return reads


Cheat = Callable[[Session], Iterable]


def drive(s: Session, cheat: Cheat = lambda s: ()) -> None:
    """Every window: the control's appends, the strategy's, one pass over both's reads, and the boundary; then every
    window closed and verified with the Lean's openings. A strategy's skips and forgeries are set at window 0, before
    the control's moves they change. A pass the books refuse ends the session there: the window can't end."""
    s.start()
    for t in range(WINDOWS):
        reads = control(s)
        extra = tuple(cheat(s))
        if s.stopped:
            return
        got = s.read(*reads, *extra)
        if got == M.REFUSED or getattr(got, "accepted", True) is False:
            return
        s.boundary()
    s.finish_lean()


def plain_stacking(s: Session) -> Iterable:
    if s.t == 0:
        s.fresh_at("A", 1, M.pages(2000, 1))
        s.fresh_at("B", 2, s.stored("A"))
        return ("A", "B")
    return ()


def registered_carrier(s: Session) -> Iterable:
    if s.t == 0:
        s.fresh_at("B", 1, s.stored((0, 1)))
    if s.t == 2:
        s.fresh_rekey("B@2", "B")
        return ((0, 1), "B@2")
    return ()


def forged_age(s: Session) -> Iterable:
    s.forge["F@2"] = lambda s: s.stored(("R0@2", 0), ("R0@2", 1))
    return ()


def laundered(m: int) -> Cheat:
    """The registered re-key of window 2 (class 0) with its first m chunks carrying R1@1's first m stored forms, read in
    place of the honest re-key in windows 2 and 3."""
    def cheat(s: Session) -> Iterable:
        s.forge["R0@2"] = lambda s: s.stored(*[("R1@1", u) for u in range(m)]) + list(s.store.payloads[m:CLASS])
        return ()
    return cheat


def rekey_of_rekey(s: Session) -> Iterable:
    s.skip.add("F@4")
    if s.t == 4:
        k = next(slot for slot in s.schedule.slots(2) if s.schedule.slot_at(slot)[1][0] == "rekey")
        s.by_slot[k] = s.named["F@2"]
        s.fresh_rekey("F@4", "F", origin_slot=k)
    return ()


def early_plan(s: Session) -> Iterable:
    if s.t == 1:
        s.fresh_at("B", 0, s.stored(("R1@1", 0)))
    return ("B",) if s.t == 2 else ()


def late_rekey(s: Session) -> Iterable:
    if s.t == 1:
        s.fresh_at("B", 0, s.stored(("R1@1", 0)))
    if s.t == 3:
        s.fresh_rekey("B@3", "B")
        return (("R1@1", 0), "B@3")
    return ()


def shifted_origin(s: Session) -> Iterable:
    """Fresh F and G in window 0; in window 2, after R0@2's coin, the re-key slot of F's slot names G's slot."""
    s.skip.add("F@2")
    if s.t == 0:
        s.fresh_at("G", 1, M.pages(3000, 2))
    if s.t == 2:
        s.fresh_rekey("F@2", "F", origin_slot=s.schedule.fresh_slot(0, 1))
    return ()


def empty_slot(s: Session) -> Iterable:
    """In window 2, the re-key slot of window 0's second fresh slot, which no append filled."""
    if s.t == 2:
        k = s.schedule.fresh_slot(0, 1)
        s.send(R.rekey_root(s.digest, s.store.extent, 2, {"append": k}), 2, s.slot_of(("rekey", k)), {"append": k})
        s.stopped = True
    return ()


def over_cap(s: Session) -> Iterable:
    if s.t == 0:
        s.fresh_at("Big", 1, M.pages(4000, 513))
    return ()


def skipped_registered(s: Session) -> Iterable:
    """Window 2's registered re-key left empty and class 0 read under the registration's key."""
    s.skip.add("R0@2")
    return ((0, CLASS),) if s.t == 2 else ()


def skipped_unread(s: Session) -> Iterable:
    """Window 2's registered re-key left empty and class 0 read no more."""
    s.skip.add("R0@2")
    return ()


@dataclass(frozen=True)
class Row:
    name: str
    rule: str
    failure: str | None
    """What the Lean names; None: accepted."""
    cheat: Cheat


ROWS = (
    Row("honest (control)", "-", None, lambda s: ()),
    Row("plain stacking", "probation", "probation", plain_stacking),
    Row("carrier of a registered chunk", "rule 1", "lifetime", registered_carrier),
    Row("forged age", "origin path", "origin-path", forged_age),
    Row("laundered origin: registered, m = 2", "origin path", "origin-path", laundered(2)),
    Row("laundered origin: re-key of a re-key", "slot origin", "origin", rekey_of_rekey),
    Row("early plan", "payload timing", "probation", early_plan),
    Row("late re-key", "key timing", "lifetime", late_rekey),
    Row("shifted origin (prefix_origin_reads_key)", "slot origin", "origin", shifted_origin),
    Row("re-key of an empty slot", "slot origin", "origin", empty_slot),
    Row("fresh append of 513 blocks", "slot cap", "origin", over_cap),
    Row("skipped registered re-key, class still read", "rule 1", "lifetime", skipped_registered),
    Row("skipped registered re-key, class not read", "-", None, skipped_unread),
)


@dataclass(frozen=True)
class Verdict:
    outcome: str
    """`accepted` or the failure the Lean names."""
    event: int | None
    """The event the Lean refused at, its kind beside: `append`, `receive` or `verify`."""
    kind: str | None
    free: int | None
    """Free receipts of an accepted session."""
    sample: int | None
    """The Lean ledger's sample term of an accepted session, in receipts."""


def ledger(server: D.Server, s: Session, record: dict) -> tuple[Fraction, Fraction]:
    """The Lean ledger's credit on the schedule check and its sample term, in receipts, at four times the least capacity
    the record's checks pass at (`test_refresh_lean.gamma`)."""
    b = M.codec.BLOCK_BYTES
    ws = [(sum(w["lengths"]), w["deadline_ns"] - w["released_ns"]) for w in record["windows"]]
    span = record["windows"][-1]["deadline_ns"] - record["windows"][0]["released_ns"]
    capacity = 4 * max(Fraction(u * b * 10 ** 9, t) for u, t in [(sum(u for u, _ in ws), span), *ws])
    kind, text = server.call("gamma", {
        "registration": s.case()["registration"], "record": record, "delta": "1/1099511627776", "block_bytes": b,
        "chip_bytes": 0, "capacity_bytes_per_s": f"{capacity.numerator}/{capacity.denominator}",
        "squarings_per_s": "0", "scheduled": SCHEDULE_CAP})
    assert kind == "output", text
    terms = json.loads(text)["terms"]
    whole = capacity * span / 10 ** 9 / b
    credited = (1 - Fraction(terms["timing"]) - Fraction(terms["carrier"])) * whole
    return credited, Fraction(terms["sample"]) * whole


def verdict(server: D.Server, s: Session) -> Verdict:
    kind, text = R.PORT.lean(server, s.case())
    assert kind == "output", text
    out = json.loads(text)
    if out["outcome"] != "accepted":
        k = next(k for k, x in enumerate(out["results"]) if "failure" in x)
        return Verdict(out["outcome"], k, s.events[k]["kind"], None, None)
    credited, sample = ledger(server, s, out["record"])
    assert credited.denominator == 1, credited
    return Verdict("accepted", None, None, max(0, int(credited) - s.held_reads()), math.ceil(sample))


def play(server: D.Server, cheat: Cheat, **kw) -> Verdict:
    s = Session(**kw)
    drive(s, cheat)
    return verdict(server, s)


@dataclass(frozen=True)
class Rate:
    trials: int
    caught: int
    free: tuple[int, ...]
    sample: tuple[int, ...]
    bound: float
    """The probability the sample misses every bad receipt, hypergeometric over the windows that read them."""


def rate(server: D.Server, cheat: Cheat, failure: str, bad: Callable[[Session], dict[int, int]], trials: int,
         label: bytes, samples: int = SAMPLES) -> Rate:
    """`cheat` under `trials` independent coins at `samples` openings a window: how many sessions the Lean refuses with
    `failure`, what each accepted one earns and is charged, and the sampling bound from `bad(s)`, window t's bad
    receipts out of its plan."""
    caught, free, sample, miss = 0, [], [], 1.0
    for trial in range(trials):
        s = Session(samples=samples, coins_label=b"%s, trial %d" % (label, trial))
        drive(s, cheat)
        v = verdict(server, s)
        if v.outcome == failure:
            caught += 1
        else:
            assert v.outcome == "accepted", v
            free.append(v.free)
            sample.append(v.sample)
        if trial == 0:
            for t, m in bad(s).items():
                u = sum(c for _, c in s.passes[t, 0].plan.runs)
                miss *= math.comb(u - m, min(samples, u)) / math.comb(u, min(samples, u))
    return Rate(trials, caught, tuple(free), tuple(sample), miss)


def laundering(server: D.Server, m: int, trials: int) -> Rate:
    return rate(server, laundered(m), "origin-path", lambda s: {2: m, 3: m}, trials, b"laundering m=%d" % m)


def skipping(server: D.Server, trials: int) -> Rate:
    return rate(server, skipped_registered, "lifetime", lambda s: {2: CLASS}, trials, b"skipped re-key")


def _grid(rows) -> str:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return "\n".join(" | ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip() for r in rows)


def table(server: D.Server) -> str:
    rows = [("attack", "rule that stops it", "Lean verdict (slot form, every rule on)", "refused at", "free receipts")]
    for r in ROWS:
        v = play(server, r.cheat)
        rows.append((r.name, r.rule, v.outcome, "-" if v.event is None else f"{v.kind} (event {v.event})",
                     "-" if v.free is None else str(v.free)))
        print(" | ".join(rows[-1]), file=sys.stderr, flush=True)
    return _grid(rows)


def rate_table(server: D.Server, trials: int, ms=(1, 2, 4, 8, 16)) -> str:
    rows = [("strategy", "caught", "free receipts, uncaught", "sample term, uncaught", "P(caught), bound")]
    for m in ms:
        r = laundering(server, m, trials)
        rows.append((f"laundered m = {m} of {CLASS}", f"{r.caught}/{r.trials}",
                     ",".join(map(str, sorted(set(r.free)))) or "-", ",".join(map(str, sorted(set(r.sample)))) or "-",
                     f"{1 - r.bound:.3f}"))
    r = skipping(server, trials)
    rows.append(("skipped registered re-key, class read", f"{r.caught}/{r.trials}",
                 ",".join(map(str, sorted(set(r.free)))) or "-", ",".join(map(str, sorted(set(r.sample)))) or "-",
                 "1 (refused at receive)"))
    return _grid(rows)


if __name__ == "__main__":
    with D.Server(R.PORT.argv()) as server:
        print(table(server), flush=True)
        print()
        print(rate_table(server, int(sys.argv[1]) if len(sys.argv) > 1 else 20), flush=True)
