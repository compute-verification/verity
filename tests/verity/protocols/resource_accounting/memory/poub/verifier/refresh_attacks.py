"""PoUB's refresh setting (`Params.refresh`) under attack: six prover strategies against the verifier's own session
(`audit.AuditRun`) over the reference prover's store (`audit_prover`) and its honest rota (`audit_prover.Refresh`),
each run with refresh as designed and with the one rule that stops it switched off here (`rule_off`), and the
receipts the ledger credits for stored forms the prover did not hold counted (`free_receipts`).
`python refresh_attacks.py` prints the table and the laundering rate (`laundering`).

Every attack stacks: a carrier, a chunk whose payload is another credited chunk's stored form, so that holding the
carrier's stored form answers for both (decoding is k squarings a word, encoding k square roots).  Its payload is
chosen after the other chunk's key is drawn, which the refresh cut forbids; each attack takes one route around one
rule."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import inspect
import math
import sys
import textwrap
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

from poub_fakes import Clock, Entropy
from test_appends import NODE, P512, SOURCE, pages, payload, salts

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.prover.rota import Rota
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, coins, schedule
from verity.protocols.resource_accounting.memory.poub.verifier.audit import AuditRun, Params
from verity.protocols.resource_accounting.memory.poub.verifier.receipts import Plan

N = 64
WINDOWS = 6
LIFETIME = 2
"""The reference session's: 64 registered blocks, W = 6, L = 2 (at 2 the next window boundary and a full period after
a commit differ, which the early plan needs)."""
SAMPLES = 16
"""The reference session's sample a window, which the laundering rate draws; the pairs sample every receipt."""
SCHEDULE_CAP = 4096
CAPACITY = float(1 << 50)
"""Bytes a second: large enough that no record's reads exceed it, a power of two so the credit reads back exactly."""
CLASS = N // LIFETIME
"""Blocks in a registered class: class c is blocks c·CLASS … (c + 1)·CLASS − 1."""


def refresh_mirrored() -> bool:
    """Whether the verifier this tree holds has the refresh setting (`Params.refresh`)."""
    return "refresh" in {f.name for f in dataclasses.fields(Params)}


@functools.cache
def _registered() -> audit_prover.Store:
    payloads = [payload(b"weights %d" % j) for j in range(N)]
    ys = salts(b"payload salts", N)
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    return audit_prover.Store(audit.register(SOURCE, NODE, N, root, P512, nonce=bytes(32)), payloads, ys)


def store() -> audit_prover.Store:
    """A store of N registered blocks under p512/k2 with no appends yet, its own to append to."""
    out = copy.copy(_registered())
    out.blocks, out.appends, out.pending, out.coin_of = list(out.blocks), [], {}, {}
    return out


class Rejected(Exception):
    def __init__(self, verdict: audit.Verdict) -> None:
        super().__init__(verdict.failure)
        self.verdict = verdict


class Session:
    """One session, the verifier's `AuditRun` and the reference prover in step, a pass a window at step t + 1, so
    that a block appended before the first pass is written at a step, 0.  Every receipt is sampled unless `samples`
    says otherwise, and `coins` labels the verifier's coins.  Appends and passes name blocks by the append's name, a
    run `(first, count)`, or `(name, m)` for block m of an append."""

    def __init__(self, *, refresh: bool = True, windows: int = WINDOWS, samples: int = 10**6,
                 coins_label: bytes = b"coins") -> None:
        self.store = store()
        reg = self.store.registration
        self.p = Params(blocks=N, deadline_ns=10**6, windows=windows, samples=samples, delta=1e-6, window_ns=10,
                        codec=reg.codec, lifetime=LIFETIME, schedule_cap=SCHEDULE_CAP,
                        **({"refresh": True} if refresh else {}))
        self.clock = Clock()
        self.s = AuditRun(self.p, reg, None, coins.local(reg.digest, 0, windows, Entropy(coins_label)),
                          not_before_ns=0, clock=self.clock)
        self.prover = audit_prover.Prover(self.store, None, entropy=Entropy(b"receipt salts"))
        self.rota = Rota(N, LIFETIME, windows)
        self.honest = audit_prover.Refresh(self.store, self.rota)
        self.challenges = [self.s.start()]
        self.t, self.i, self.step, self.previous = 0, 0, 0, reg.digest
        self.head: bytes | None = None
        self.named: dict[str, audit_prover.Append] = {}
        self.passes: list[audit_prover.Pass] = []
        self.by_window: dict[tuple[int, int], audit_prover.Pass] = {}
        self.keep: set[int] = set()
        """Indices the honest rota would release that a strategy keeps."""

    def runs(self, what) -> tuple[int, int]:
        if type(what) is str:
            return self.named[what].first, self.named[what].count
        if type(what[0]) is str:
            a = self.named[what[0]]
            assert 0 <= what[1] < a.count
            return a.first + what[1], 1
        return what

    def stored(self, what) -> list[bytes]:
        """The stored forms of `what`'s blocks, as a carrier's payloads."""
        first, count = self.runs(what)
        return [self.store.blocks[j] for j in range(first, first + count)]

    def index(self, name: str) -> int:
        """The append's index k in the store, as a re-key's origin `("append", k, o)` names it."""
        return self.store.appends.index(self.named[name])

    def _send(self, name: str, a: audit_prover.Append, origin=None) -> audit_prover.Append:
        got = self.s.append(a.root, a.count) if origin is None else self.s.append(a.root, a.count, origin=origin)
        if type(got) is audit.Verdict:
            raise Rejected(got)
        first, coin = got
        assert first == a.first
        self.store.encode(first, coin)
        self.head = audit.append_head(self.head, first, a.count, a.root)
        self.named[name] = a
        return a

    def fresh(self, name: str, payloads: Sequence[bytes]) -> audit_prover.Append:
        return self._send(name, self.store.append(list(payloads), salts(b"salts of " + name.encode(), len(payloads))))

    def rekey(self, name: str, origin, count: int, payloads: Sequence[bytes] | None = None) -> audit_prover.Append:
        """A re-key of `count` blocks from `origin`, carrying its origin's payloads unless the prover sends others,
        under digests of what it sends and its origin's salts and paths (`Store.committed`)."""
        a = self.store.rekey(origin, count)
        if payloads is not None:
            a = self._forge(a, payloads)
        return self._send(name, a, origin)

    def rekey_of_rekey(self, name: str, of: str, payloads: Sequence[bytes] | None = None) -> audit_prover.Append:
        """A re-key whose origin is the re-key `of`, which the reference store refuses to make: its blocks and
        declaration made here, its payloads `of`'s unless the prover sends others."""
        b = self.named[of]
        origin = ("append", self.index(of), 0)
        a = self.store.rekey(b.origin, b.count)
        a = dataclasses.replace(a, origin=origin, declaration=audit.rekey_root(
            self.store.registration.digest, a.first, a.count, origin))
        self.store.appends[-1] = a
        return self._send(name, a if payloads is None else self._forge(a, payloads), origin)

    def _forge(self, a: audit_prover.Append, payloads: Sequence[bytes]) -> audit_prover.Append:
        assert len(payloads) == a.count and self.store.appends[-1] is a
        forged = dataclasses.replace(a, digests=tuple(codec.digest(m) for m in payloads))
        self.store.appends[-1], self.store.pending[a.first] = forged, tuple(payloads)
        return forged

    def release(self, *what) -> None:
        if runs := _canonical([self.runs(w) for w in what]):
            if (v := self.s.release(runs)) is not None:
                raise Rejected(v)

    def control(self) -> list[tuple[int, int]]:
        """The honest prover's moves before window t's pass on the rota (`audit_prover.Refresh`): a KV append F of 2
        pages in window 0, the re-keys due at t, named `R{c}@{t}` (registered class c) and `F@{t}`, and the release
        of each index a re-key's encode frees but those in `keep`; the runs of its plan of t."""
        if self.t == 0:
            self.honest.appended(self.fresh("F", pages(N, 2)), 0)
        for origin, count in self.honest.due(self.t):
            name = (f"R{self.rota.registered_class(origin[1])}@{self.t}" if origin[0] == "registered"
                    else f"F@{self.t}")
            gone = self.honest.appended(self.rekey(name, origin, count), self.t)
            self.release(*[(j, 1) for j in gone if j not in self.keep])
        return list(self.honest.plan(self.t).runs)

    def read(self, *what) -> None:
        plan = Plan(_canonical([self.runs(w) for w in what]), self.store.extent)
        ps = self.prover.run(self.challenges[self.t], self.i, self.previous, plan, head=self.head)
        self.step += 1
        if (v := self.s.receive(self.t, self.i, ps.root, plan.runs, step=self.step)) is not None:
            raise Rejected(v)
        self.passes.append(ps)
        self.by_window[self.t, self.i] = ps
        self.previous, self.i = ps.root, self.i + 1

    def next_window(self) -> None:
        self.clock.now += self.p.window_ns
        if (coin := self.s.boundary()) is not None:
            self.challenges.append(coin)
        self.t, self.i = self.t + 1, 0

    def finish(self) -> None:
        while self.t < self.p.windows:
            self.next_window()
        self.clock.now = self.s.deadline(len(self.s.windows) - 1) + 1
        for t in range(self.p.windows):
            sampled = self.s.close(t)
            if type(sampled) is audit.Verdict:
                raise Rejected(sampled)
            v = self.s.verify(t, [self.prover.open(self.by_window[t, i], q) for i, q in sampled])
            if not v.accepted:
                raise Rejected(v)

    def schedule(self) -> schedule.Schedule:
        """The schedule whose plan at each pass's step is the plan the pass read: every chunk resident exactly while
        planned, a registered one from the start, so `schedule.check` takes it when the plans allow one."""
        steps = list(range(1, len(self.passes) + 1))
        planned: dict[int, list[int]] = {}
        for s, ps in zip(steps, self.passes):
            for j in ps.plan:
                planned.setdefault(j, []).append(s)
        for j, ss in planned.items():
            if ss != list(range(ss[0], ss[-1] + 1)) or (j < N and ss[0] != steps[0]):
                raise ValueError(f"block {j} is planned at steps {ss}: no schedule frees it and brings it back")
        registered, last = [], steps[-1]
        for j in range(N):
            if j not in planned:
                raise ValueError(f"registered block {j} is in no plan")
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

    def held_reads(self) -> int:
        """Receipts backed by a stored form the prover holds: a pass's reads less the blocks whose stored form is the
        payload of another block of its plan (a carrier's), which decoding the carrier gives."""
        out = 0
        for ps in self.passes:
            blocks = list(ps.plan)
            forms = {self.store.blocks[j]: j for j in blocks}
            carried = {forms[m] for c in blocks if (m := self.store.payload(c)) in forms and forms[m] != c}
            out += len(blocks) - len(carried)
        return out


def ledger(s: Session, verdict: audit.PostSeedVerdict | None = None) -> audit.GammaTerms:
    rec = s.s.record()
    return audit.ledger(rec, capacity_bytes_per_s=CAPACITY, chip_bytes=0, rebuild=audit.Rebuild(rec.codec, 0.0),
                        schedule=s.s.scheduled(s.schedule()), credit_post_seed=verdict)


def receipts(s: Session, fraction: float) -> int:
    """A ledger term, a fraction of the session's capacity over its span, in receipts."""
    rec = s.s.record()
    return round(fraction * CAPACITY * rec.span_ns / 1e9 / rec.block_bytes)


def credited_receipts(s: Session, verdict: audit.PostSeedVerdict | None = None) -> int:
    """The receipts the ledger credits on the session's record and schedule check (and `verdict`, for a record
    without refresh), less its carrier charge."""
    g = ledger(s, verdict)
    return receipts(s, 1 - g.timing) - receipts(s, g.carrier)


def free_receipts(s: Session, verdict: audit.PostSeedVerdict | None = None) -> int:
    """Credited receipts beyond the stored forms the prover held and read: `credited_receipts` less `held_reads`,
    none when the ledger credits fewer than those."""
    return max(0, credited_receipts(s, verdict) - s.held_reads())


def _canonical(runs: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for f, c in sorted(runs):
        if out and out[-1][0] + out[-1][1] == f:
            out[-1][1] += c
        else:
            out.append([f, c])
    return tuple((f, c) for f, c in out)


Cheat = Callable[[Session], Iterable]
"""A strategy's moves in window `s.t` after the honest rota's, returning what its pass reads beside the rota's."""


def drive(s: Session, cheat: Cheat = lambda s: ()) -> None:
    """Every window: the rota's moves, the strategy's, and one pass over both's reads."""
    for t in range(s.p.windows):
        if t:
            s.next_window()
        plan = s.control()
        s.read(*plan, *cheat(s))


def honest(s: Session) -> None:
    drive(s)


def plain_stacking(s: Session) -> Iterable:
    """Fresh A in window 0, then fresh B whose payload is A's stored form, chosen once A's coin is out; both read in
    windows 0 and 1, their key's lifetime."""
    if s.t == 0:
        s.fresh("A", pages(16, 1))
        s.fresh("B", s.stored("A"))
    return ("A", "B") if s.t < 2 else ()


def registered_carrier(s: Session) -> Iterable:
    """Registered block 0 kept past its class's re-key at window 2, beside fresh B of window 0 carrying its stored
    form, B re-keyed on the rota (fresh ordinal 1: windows 1, 3, 5) and read with block 0 from window 2 on."""
    if s.t == 0:
        s.keep.add(0)
        s.fresh("B", s.stored((0, 1)))
    if s.t % 2:
        s.rekey(f"B@{s.t}", ("append", s.index("B"), 0), 1)
    if s.t == 3:
        s.release("B@1")
    if s.t == 5:
        s.release("B@3")
    return ((0, 1), f"B@{s.t - 1 + s.t % 2}") if s.t >= 2 else ()


def forged_age(s: Session) -> Iterable:
    """X, a re-key of F (born 0) in window 2, carrying the stored forms of R0@2's first two blocks in place of F's
    payloads: a payload of window 2 credited as one of window 0; read in windows 2 and 3."""
    if s.t == 2:
        s.rekey("X", ("append", s.index("F"), 0), 2, s.stored(("R0@2", 0)) + s.stored(("R0@2", 1)))
    if s.t == 4:
        s.release("X")
    return ("X",) if s.t in (2, 3) else ()


def laundered_registered(s: Session) -> Iterable:
    """Y, a re-key of registered blocks 2 and 3 in window 2 (class 0's), carrying the stored forms of R0@2's first
    two blocks: a payload chosen after the seed credited as registered (born none); read in windows 2 and 3."""
    if s.t == 2:
        s.rekey("Y", ("registered", 2), 2, s.stored(("R0@2", 0)) + s.stored(("R0@2", 1)))
    if s.t == 4:
        s.release("Y")
    return ("Y",) if s.t in (2, 3) else ()


def laundered_rekey(s: Session) -> Iterable:
    """K2, a re-key of registered blocks 2 and 3 in window 2, then K3, a re-key of K2 in window 3, carrying the
    stored forms of R0@2's first two blocks, a payload of window 3 credited as K2's; read in window 3."""
    if s.t == 2:
        s.rekey("K2", ("registered", 2), 2)
    if s.t == 3:
        s.rekey_of_rekey("K3", "K2", s.stored(("R0@2", 0)) + s.stored(("R0@2", 1)))
        return ("K3",)
    if s.t == 4:
        s.release("K2", "K3")
    return ()


def early_plan(s: Session) -> Iterable:
    """Fresh D, then fresh B in window 1 (fresh ordinal 2: class 0) carrying R1@1's first block's stored form; B
    re-keyed in window 2 and read there, at the next window boundary after its commit, not a period after it."""
    if s.t == 1:
        s.fresh("D", pages(32, 1))
        s.fresh("B", s.stored(("R1@1", 0)))
    if s.t == 2:
        s.rekey("B@2", ("append", s.index("B"), 0), 1)
        return ("B@2",)
    if s.t == 3:
        s.release("B@2")
    return ()


def late_rekey(s: Session) -> Iterable:
    """Fresh B in window 1 (fresh ordinal 1: class 1) carrying R1@1's first block's stored form, re-keyed on the rota
    in window 3 and read there beside R1@1's first block, kept past its key's lifetime."""
    if s.t == 1:
        s.fresh("B", s.stored(("R1@1", 0)))
        s.keep.add(s.runs(("R1@1", 0))[0])
    if s.t == 3:
        s.rekey("B@3", ("append", s.index("B"), 0), 1)
        return (("R1@1", 0), "B@3")
    if s.t == 4:
        s.release(("R1@1", 0), "B@3")
    return ()


def chosen_origin(second: str) -> Cheat:
    """lean's `prefix_origin_reads_key` on the rota by fresh ordinal: fresh D and G after F in window 0 (ordinals 1
    and 2, so F and G are both class 0); in window 2, after the control's re-key of F has its coin, a second re-key K
    whose origin is `second`, G or F again, picked on that coin; K read in windows 2 and 3."""
    def cheat(s: Session) -> Iterable:
        if s.t == 0:
            s.fresh("D", pages(3000, 1))
            s.fresh("G", pages(3100, 2))
        if s.t == 2:
            s.rekey("K", ("append", s.index(second), 0), 2)
        if s.t == 4:
            s.release("K")
        return ("K",) if s.t in (2, 3) else ()
    return cheat


def origin_choice() -> dict[str, Outcome]:
    """Both of `chosen_origin`'s picks, with every rule of this mirror on."""
    return {second: play(chosen_origin(second)) for second in ("G", "F")}


@dataclass(frozen=True)
class Attack:
    name: str
    rule: str
    """The one rule that stops it, a key of `SWITCHES`."""
    failure: str
    """What the verifier names when the rule stops it."""
    cheat: Cheat
    free_off: int | None
    """The free receipts it earns with its rule off, a carried block a pass beside its carrier; None where another
    rule still rejects it (`off_failure`)."""
    gain: str
    off_failure: str | None = None


ATTACKS = (
    Attack("plain stacking", "probation", "probation", plain_stacking, 2,
           "B's stored form answers for A's: one held a pass for two credited"),
    Attack("carrier of a registered chunk", "rule 1", "lifetime", registered_carrier, 4,
           "B's stored form answers for registered block 0's, whose key is the seed's"),
    Attack("forged age", "origin path", "origin-path", forged_age, 4,
           "X's stored forms answer for R0@2's first two: a payload of window 2 credited as F's of window 0"),
    Attack("laundered origin: registered", "origin path", "origin-path", laundered_registered, 4,
           "Y's stored forms answer for R0@2's first two: a payload chosen after the seed credited as registered"),
    Attack("laundered origin: re-key of a re-key", "fresh origin", "append", laundered_rekey, None,
           "none: K3's birth is the window K2 joined, so probation holds it a period past K2's commit",
           off_failure="probation"),
    Attack("early plan", "payload timing", "probation", early_plan, 1,
           "B@2's stored form answers for R1@1's first: credited at the next boundary after its commit"),
    Attack("late re-key", "key timing", "lifetime", late_rekey, 1,
           "B@3's stored form answers for R1@1's first, kept past its key's lifetime"),
)


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
    "fresh origin": lambda: _edited("_origin_holds", " and self._origins[k] is None", ""),
    "payload timing": lambda: _edited("_seasoning", "b + L > t", "b + 1 > t"),
    "key timing": lambda: _edited("_stale_below", "k = bisect.bisect_right(self._joined, t - self.params.lifetime)",
                                  "return self.params.blocks"),
}
"""Each rule's test-only switch: the verifier's method rebuilt from its own source with that rule's one condition
edited, for the length of one run, here and nowhere in the verifier's code, which takes no parameter for it."""


def rule_off(rule: str) -> contextlib.AbstractContextManager:
    return SWITCHES[rule]()


@dataclass(frozen=True)
class Outcome:
    accepted: bool
    failure: str | None
    free: int | None
    """Free receipts on an accepted session, None on a rejected one, which credits nothing."""


def play(cheat: Cheat, **kw) -> Outcome:
    s = Session(**kw)
    try:
        drive(s, cheat)
        s.finish()
    except Rejected as e:
        return Outcome(False, e.verdict.failure, None)
    return Outcome(True, None, free_receipts(s))


def pair(a: Attack) -> tuple[Outcome, Outcome]:
    """The attack with its rule switched off, then with refresh as designed."""
    with rule_off(a.rule):
        off = play(a.cheat)
    return off, play(a.cheat)


def launder(m: int) -> Cheat:
    """Attack 4 at scale: Y, a re-key of registered class 0 (CLASS blocks) in window 2, its first m chunks carrying
    the stored forms of R0@2's first m blocks and the rest their registered payloads; read in windows 2 and 3."""
    def cheat(s: Session) -> Iterable:
        if s.t == 2:
            forged = [x for u in range(m) for x in s.stored(("R0@2", u))]
            s.rekey("Y", ("registered", 0), CLASS, forged + list(s.store.payloads[m:CLASS]))
        if s.t == 4:
            s.release("Y")
        return ("Y",) if s.t in (2, 3) else ()
    return cheat


@dataclass(frozen=True)
class Rate:
    m: int
    trials: int
    caught: int
    free: tuple[int, ...]
    """Free receipts of each uncaught run."""
    sample: tuple[int, ...]
    """The ledger's sample term of each uncaught run, in receipts: the wrong receipts it charges."""
    bound: float
    """The probability the sample misses every laundered receipt, the hypergeometric figure over the windows that
    read Y."""


def laundering(m: int, trials: int, samples: int = SAMPLES) -> Rate:
    """Attack 4 with every rule on and `samples` openings a window: of `trials` sessions under independent coins,
    how many the verifier catches (`origin-path`), what each uncaught one earns, and the sampling bound beside."""
    caught, free, sample, sizes = 0, [], [], None
    for trial in range(trials):
        s = Session(samples=samples, coins_label=b"coins of m=%d, trial %d" % (m, trial))
        try:
            drive(s, launder(m))
            s.finish()
        except Rejected as e:
            assert e.verdict.failure == "origin-path", e.verdict.failure
            caught += 1
        else:
            free.append(free_receipts(s))
            sample.append(receipts(s, ledger(s).sample))
        sizes = [sum(c for _, c in s.by_window[t, 0].plan.runs) for t in (2, 3)] if (2, 0) in s.by_window else sizes
    miss = math.prod(math.comb(u - m, min(samples, u)) / math.comb(u, min(samples, u)) for u in sizes)
    return Rate(m, trials, caught, tuple(free), tuple(sample), miss)


def table() -> str:
    rows = [("attack", "rule that stops it", "free receipts, rule off", "free receipts, rule on", "failure named")]
    control_run = play(lambda s: ())
    rows.append(("honest (control)", "-", "-", str(control_run.free), control_run.failure or "accepted"))
    for a in ATTACKS:
        off, on = pair(a)
        rows.append((a.name, a.rule, "rejected " + str(off.failure) if off.free is None else str(off.free),
                     "rejected" if on.free is None else str(on.free), on.failure or "accepted"))
    for second, o in origin_choice().items():
        rows.append((f"chosen origin, K of {second} (old rota)", "none in this mirror", "-",
                     "rejected" if o.free is None else str(o.free), o.failure or "accepted"))
    return _grid(rows)


def rate_table(ms: Sequence[int] = (1, 2, 4, 8, 16), trials: int = 20) -> str:
    rows = [("laundered m of 32", "caught", "free receipts, uncaught", "sample term, uncaught", "P(miss), bound")]
    for m in ms:
        r = laundering(m, trials)
        rows.append((str(m), f"{r.caught}/{r.trials}", ",".join(map(str, sorted(set(r.free)))) or "-",
                     ",".join(map(str, sorted(set(r.sample)))) or "-", f"{r.bound:.3f}"))
    return _grid(rows)


def _grid(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return "\n".join(" | ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip() for r in rows)


if __name__ == "__main__":
    if not refresh_mirrored():
        sys.exit("this tree's verifier has no refresh setting (Params.refresh): merge the Python mirror first")
    print(table())
    print()
    print(rate_table(trials=int(sys.argv[1]) if len(sys.argv) > 1 else 20))
