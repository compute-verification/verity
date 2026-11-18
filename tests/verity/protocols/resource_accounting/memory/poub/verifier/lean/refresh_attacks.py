"""PoUB's refresh attacks on the slot schedule, each session given two verdicts: the Python books' (`audit.AuditRun`
with refresh on, `ra_cases.Run`), whose first refused event ends it, and the compiled Lean verifier's (`Poub.Session`,
of record) on the same events. Each attack also runs on the books with the one rule that stops it switched off here
(`rule_off`), for what it would earn. Free receipts are a ledger's credit on the schedule check less the stored forms
the prover held and read. `python refresh_attacks.py [TRIALS]` prints the table and the sampled rates.

Every attack stacks: a carrier, a chunk whose payload is another credited chunk's stored form, so that holding the
carrier's stored form answers for both. Its payload is chosen after the other chunk's key is drawn, which the refresh
cut forbids; each attack takes one route around one rule."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import math
import sys
import textwrap
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction

import ra_cases as M
import test_refresh_lean as R
import verity_difftest as D

from verity.protocols.resource_accounting.memory.poub.verifier import audit, schedule
from verity.protocols.resource_accounting.memory.poub.verifier.audit import AuditRun
from verity.protocols.resource_accounting.memory.poub.verifier.receipts import Plan

N, WINDOWS, LIFETIME, FRESH_SLOTS = 64, 6, 2, 4
"""The reference session's: 64 registered blocks, W = 6, L = 2, F = 4 fresh slots a window (T = 53)."""
SAMPLES = 16
"""The reference session's openings a window, which the rates draw; the pairs open every receipt."""
SCHEDULE_CAP = 4096
CAPACITY = float(1 << 50)
"""The Python ledger's bytes a second: no record's reads exceed it, a power of two so the credit reads back exactly."""
CLASS = N // LIFETIME


class Over(Exception):
    """The books refused an event: the session ends there."""


class _Store(M.RekeyStore):
    """The reference store, a re-key of a re-key opening with its first fresh origin's salt and path: the prover's
    best try, since a re-key has no payload tree to open under."""

    def committed(self, j: int) -> tuple[bytes, tuple[bytes, ...], bytes]:
        pos = self.origin_position(j)
        if pos is None:
            return super().committed(j)
        while (q := self.origin_position(pos)) is not None:
            pos = q
        salt, path, _ = M.audit_prover.Store.committed(self, pos)
        a = self._append_of(j)
        return salt, path, a.digests[j - a.first]


class Session(M.Run):
    """One session at the reference shape with refresh on, its verifier's coins from `coins_label`. Blocks are named
    by the append's name, a run `(first, count)`, or `(name, m)` for block m of an append."""

    def __init__(self, *, samples: int = 10**6, coins_label: bytes = b"coins") -> None:
        super().__init__(M.World(blocks=N), windows=WINDOWS, lifetime=LIFETIME, samples=samples, refresh=True,
                         fresh_slots=FRESH_SLOTS)
        reg = self.store.registration
        self.store = _Store.of(self.store)
        self.params = dataclasses.replace(self.params, schedule_cap=SCHEDULE_CAP)
        self.coins = M.coins.local(reg.digest, 0, WINDOWS, M.Entropy(coins_label))
        self.s = AuditRun(self.params, reg, None, self.coins, not_before_ns=0, clock=self.clock)
        self.prover = M.audit_prover.Prover(self.store, None, entropy=M.Entropy(b"receipt salts"))
        self.schedule = self.params.schedule
        self.named: dict[str, object] = {}
        self.skip: set[str] = set()
        """The control's moves a strategy leaves out."""
        self.forge: dict[str, Callable[[Session], list[bytes]]] = {}
        """The payloads a strategy sends in place of a control re-key's."""
        self.refused: tuple[str, int, str] | None = None
        """The books' first refusal: its failure, its event and the event's kind."""

    def _call(self, event: dict, fn: Callable[[], object]) -> object:
        out = super()._call(event, fn)
        if isinstance(out, str) or (isinstance(out, audit.Verdict) and not out.accepted):
            self.refused = (out if isinstance(out, str) else out.failure, len(self.events) - 1, event["kind"])
            raise Over
        return out

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

    def slot_held(self, name: str) -> int:
        return next(k for k, n in self.store.slotted.items() if self.store.appends[n] is self.named[name])

    def fresh_at(self, name: str, i: int, payloads: list[bytes], slot: int | None = None) -> None:
        """A fresh append in the current window's i-th fresh slot (or `slot`)."""
        a = self.store.append(payloads, M.salts(b"salts of " + name.encode(), len(payloads)))
        self.named[name] = a
        self.commit(a.root, a.count, slot=self.schedule.fresh_slot(self.t, i) if slot is None else slot)

    def _rekey(self, name: str, slot: int, origin: audit.Origin, count: int) -> None:
        a = self.store.rekey(origin, count)
        if name in self.forge:
            payloads = self.forge[name](self)
            assert len(payloads) == count
            a = dataclasses.replace(a, digests=tuple(M.codec.digest(m) for m in payloads))
            self.store.appends[-1], self.store.pending[a.first] = a, tuple(payloads)
        self.named[name] = a
        self.commit(a.root, a.count, origin, slot)

    def registered_rekey(self, name: str) -> None:
        """The current window's registered re-key slot: its class run, its payloads unless forged."""
        lo, hi = self.schedule.class_run(self.t % LIFETIME)
        self._rekey(name, self.slot_of(("registered",)), ("registered", lo), hi - lo)

    def fresh_rekey(self, name: str, of: str, origin_slot: int | None = None) -> None:
        """The current window's re-key slot of `of`'s fresh slot, whole, naming `origin_slot` (default that slot)."""
        k = self.slot_held(of)
        o = k if origin_slot is None else origin_slot
        self._rekey(name, self.slot_of(("rekey", k)), ("append", o), self.store.in_slot(o).count)

    def read(self, *what) -> None:
        self.run(Plan(_canonical(self.runs(w) for w in what), self.store.extent), step=len(self.passes) + 1)

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

    def layout(self) -> schedule.Schedule:
        """The schedule whose plan at each pass's step is the plan the pass read: every chunk resident exactly while
        planned, a registered one from the start."""
        steps = list(range(1, len(self.passes) + 1))
        planned: dict[int, list[int]] = {}
        for s, ps in zip(steps, self.passes.values()):
            for j in ps.plan:
                planned.setdefault(j, []).append(s)
        for j, ss in planned.items():
            if ss != list(range(ss[0], ss[-1] + 1)) or (j < N and ss[0] != steps[0]):
                raise ValueError(f"block {j} is planned at steps {ss}: no schedule frees it and brings it back")
        registered, last = [], steps[-1]
        for j in range(N):
            e = planned[j][-1]
            if e < last:
                if registered and registered[-1][0] + registered[-1][1] == j and registered[-1][2] == e:
                    registered[-1] = (registered[-1][0], registered[-1][1] + 1, e)
                else:
                    registered.append((j, 1, e))
        appends = [tuple(None if j not in planned else
                         (planned[j][0] - 1, None if planned[j][-1] == last else planned[j][-1])
                         for j in range(a.first, a.first + a.count)) for a in self.store.appends]
        return schedule.Schedule(tuple(registered), tuple(appends), tuple(steps))


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
    window closed and verified. A strategy's skips and forgeries are set at window 0, before the control's moves they
    change. The books' first refusal ends the session, kept in `s.refused`."""
    try:
        s.start()
        for t in range(WINDOWS):
            reads = control(s)
            s.read(*reads, *cheat(s))
            s.boundary()
        s.finish()
    except Over:
        assert s.refused is not None, "Over ends a session only at a refusal of the books, which `_call` records first"


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
    """F's re-key in window 4 naming F@2's slot, a re-key's, as its origin."""
    s.skip.add("F@4")
    if s.t == 4:
        k = next(slot for slot in s.schedule.slots(2) if s.schedule.slot_at(slot)[1][0] == "rekey")
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
        o = ("append", k)
        s.commit(audit.rekey_root(s.store.registration.digest, s.store.extent, 2, o), 2, o, s.slot_of(("rekey", k)))
    return ()


def over_cap(s: Session) -> Iterable:
    if s.t == 0:
        s.fresh_at("Big", 1, M.pages(4000, audit.REKEY_MAX + 1))
    return ()


def skipped_registered(s: Session) -> Iterable:
    """Window 2's registered re-key left empty and class 0 read under the registration's key."""
    s.skip.add("R0@2")
    return ((0, CLASS),) if s.t == 2 else ()


def skipped_unread(s: Session) -> Iterable:
    """Window 2's registered re-key left empty and class 0 read no more."""
    s.skip.add("R0@2")
    return ()


@contextlib.contextmanager
def _edited(name: str, old: str, new: str) -> Iterator[None]:
    """`AuditRun.<name>` rebuilt from its own source with the one text `old` replaced by `new`, for one run."""
    source = textwrap.dedent(inspect.getsource(AuditRun.__dict__[name]))
    if source.count(old) != 1:
        raise AssertionError(f"AuditRun.{name} no longer holds {old!r} once: update its switch in refresh_attacks.py")
    scope: dict = {}
    exec(compile(source.replace(old, new), f"<{name} switched off>", "exec"), vars(audit), scope)  # noqa: S102
    original = AuditRun.__dict__[name]
    setattr(AuditRun, name, scope[name])
    try:
        yield
    finally:
        setattr(AuditRun, name, original)


SWITCHES: dict[str, Callable[[], contextlib.AbstractContextManager]] = {
    "rule 1": lambda: _edited("receive", "t >= self.params.lifetime and any(f < N for f, _ in plan.runs)", "False"),
    "probation": lambda: _edited("receive", 'return self._fail(t, "probation")', "pass"),
    "origin path": lambda: _edited("_check", "if not verify_opening(", "if origin is None and not verify_opening("),
    "slot origin": lambda: _edited("_slot_ok", 'origin == ("append", kind[1])', 'origin[0] == "append"'),
    "payload timing": lambda: _edited("_seasoning", "b + L > t", "b + 1 > t"),
    "key timing": lambda: _edited("_stale_below", "self._slot_window(s, t) <= t - self.params.lifetime", "False"),
}
"""Each rule's test-only switch: the books' method rebuilt from its own source with that rule's one condition edited,
for the length of one run, here and nowhere in the verifier's code, which takes no parameter for it. The Lean has no
switch: a rule-off column is the books' alone."""


@dataclass(frozen=True)
class Row:
    name: str
    rule: str
    """The rule that stops it, a key of `SWITCHES` where its rule-off column is measured."""
    failure: str | None
    """What both verifiers name; None: accepted."""
    cheat: Cheat
    switched: bool = True
    """Whether the rule-off column is measured."""


ROWS = (
    Row("honest (control)", "-", None, lambda s: (), False),
    Row("plain stacking", "probation", "probation", plain_stacking),
    Row("carrier of a registered chunk", "rule 1", "lifetime", registered_carrier),
    Row("forged age", "origin path", "origin-path", forged_age),
    Row("laundered origin: registered, m = 2", "origin path", "origin-path", laundered(2)),
    Row("laundered origin: re-key of a re-key", "slot origin", "origin", rekey_of_rekey),
    Row("early plan", "payload timing", "probation", early_plan),
    Row("late re-key", "key timing", "lifetime", late_rekey),
    Row("shifted origin (prefix_origin_reads_key)", "slot origin", "origin", shifted_origin),
    Row("re-key of an empty slot", "slot origin", "origin", empty_slot, False),
    Row("fresh append of 513 blocks", "slot cap", "origin", over_cap, False),
    Row("skipped registered re-key, class still read", "rule 1", "lifetime", skipped_registered, False),
    Row("skipped registered re-key, class not read", "-", None, skipped_unread, False),
)


@dataclass(frozen=True)
class Verdict:
    outcome: str
    """`accepted` or the failure named."""
    event: int | None
    """The event refused, its kind beside: `append`, `receive` or `verify`."""
    kind: str | None
    free: int | None
    """Free receipts of an accepted session."""
    sample: int | None = None
    """The Lean ledger's sample term of an accepted session, in receipts."""


def python_verdict(s: Session) -> Verdict:
    if s.refused is not None:
        return Verdict(*s.refused, None)
    rec = s.s.record()
    g = audit.ledger(rec, capacity_bytes_per_s=CAPACITY, chip_bytes=0, rebuild=audit.Rebuild(rec.codec, 0.0),
                     schedule=s.s.scheduled(s.layout()))
    whole = CAPACITY * rec.span_ns / 1e9 / rec.block_bytes
    credited = round((1 - g.timing) * whole) - round(g.carrier * whole)
    return Verdict("accepted", None, None, max(0, credited - s.held_reads()))


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


def lean_verdict(server: D.Server, s: Session) -> Verdict:
    kind, text = R.PORT.lean(server, s.case())
    assert kind == "output", text
    out = json.loads(text)
    if out["outcome"] != "accepted":
        k = next(k for k, x in enumerate(out["results"]) if "failure" in x)
        return Verdict(out["outcome"], k, s.events[k]["kind"], None)
    if s.refused is not None:
        return Verdict("accepted (to the books' refusal)", None, None, None)
    credited, sample = ledger(server, s, out["record"])
    assert credited.denominator == 1, credited
    return Verdict("accepted", None, None, max(0, int(credited) - s.held_reads()), math.ceil(sample))


def play(server: D.Server | None, cheat: Cheat, **kw) -> tuple[Verdict, Verdict | None]:
    """The session's verdicts, the books' and (given a server) the Lean's."""
    s = Session(**kw)
    drive(s, cheat)
    return python_verdict(s), None if server is None else lean_verdict(server, s)


def rule_off(row: Row) -> Verdict:
    """The books' verdict with the row's rule switched off."""
    with SWITCHES[row.rule]():
        return play(None, row.cheat)[0]


def agree(py: Verdict, lean: Verdict) -> bool:
    return (py.outcome, py.event, py.free) == (lean.outcome, lean.event, lean.free)


@dataclass(frozen=True)
class Rate:
    trials: int
    caught: int
    """Sessions both verifiers refuse with the failure."""
    differ: int
    """Sessions whose two verdicts differ."""
    free: tuple[int, ...]
    sample: tuple[int, ...]
    bound: float
    """The probability the sample misses every bad receipt, hypergeometric over the windows that read them."""


def rate(server: D.Server, cheat: Cheat, failure: str, bad: dict[int, int], trials: int, label: bytes,
         samples: int = SAMPLES) -> Rate:
    """`cheat` under `trials` independent coins at `samples` openings a window: how many sessions both verifiers refuse
    with `failure`, how many they differ on, what each accepted one earns and is charged by the Lean, and the sampling
    bound from `bad`, window t's bad receipts out of its plan."""
    caught, differ, free, sample, miss = 0, 0, [], [], 1.0
    for trial in range(trials):
        s = Session(samples=samples, coins_label=b"%s, trial %d" % (label, trial))
        drive(s, cheat)
        py, lean = python_verdict(s), lean_verdict(server, s)
        differ += not agree(py, lean)
        if lean.outcome == failure and py.outcome == failure:
            caught += 1
        elif lean.outcome == "accepted":
            free.append(lean.free)
            sample.append(lean.sample)
        if trial == 0:
            for t, m in bad.items():
                u = sum(c for _, c in s.passes[t, 0].plan.runs)
                miss *= math.comb(u - m, min(samples, u)) / math.comb(u, min(samples, u))
    return Rate(trials, caught, differ, tuple(free), tuple(sample), miss)


def laundering(server: D.Server, m: int, trials: int) -> Rate:
    return rate(server, laundered(m), "origin-path", {2: m, 3: m}, trials, b"laundering m=%d" % m)


def skipping(server: D.Server, trials: int) -> Rate:
    return rate(server, skipped_registered, "lifetime", {2: CLASS}, trials, b"skipped re-key")


def _grid(rows) -> str:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return "\n".join(" | ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip() for r in rows)


def _said(v: Verdict) -> str:
    return v.outcome if v.event is None else f"{v.outcome} at {v.kind} (event {v.event})"


def table(server: D.Server) -> str:
    rows = [("attack", "rule that stops it", "rule off: books, free receipts", "rule on: books", "rule on: Lean",
             "free receipts", "verdicts agree")]
    for r in ROWS:
        off = rule_off(r) if r.switched else None
        py, lean = play(server, r.cheat)
        rows.append((r.name, r.rule, "-" if off is None else (str(off.free) if off.free is not None else
                                                            "refused " + off.outcome),
                     _said(py), _said(lean), "-" if lean.free is None else str(lean.free),
                     "yes" if agree(py, lean) else f"NO (books' free {py.free})"))
        print(" | ".join(rows[-1]), file=sys.stderr, flush=True)
    return _grid(rows)


def rate_table(server: D.Server, trials: int, ms=(1, 2, 4, 8, 16)) -> str:
    rows = [("strategy", "caught by both", "verdicts differ", "free receipts, uncaught", "sample term, uncaught",
             "P(caught), bound")]

    def row(name: str, r: Rate, bound: str) -> tuple:
        return (name, f"{r.caught}/{r.trials}", str(r.differ), ",".join(map(str, sorted(set(r.free)))) or "-",
                ",".join(map(str, sorted(set(r.sample)))) or "-", bound)
    for m in ms:
        r = laundering(server, m, trials)
        rows.append(row(f"laundered m = {m} of {CLASS}", r, f"{1 - r.bound:.3f}"))
        print(" | ".join(rows[-1]), file=sys.stderr, flush=True)
    rows.append(row("skipped registered re-key, class read", skipping(server, trials), "1 (refused at receive)"))
    return _grid(rows)


if __name__ == "__main__":
    with D.Server(R.PORT.argv()) as server:
        print(table(server), flush=True)
        print()
        print(rate_table(server, int(sys.argv[1]) if len(sys.argv) > 1 else 20), flush=True)
