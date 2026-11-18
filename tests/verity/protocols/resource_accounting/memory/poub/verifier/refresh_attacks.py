"""PoUB's refresh setting (`Params.refresh`) under attack: six prover strategies against the verifier's own session
(`audit.AuditRun`) over the reference prover's store (`audit_prover`), each run with refresh as designed and with the
one rule that stops it switched off here (`rule_off`), and the receipts the ledger credits for stored forms the prover
did not hold counted (`free_receipts`).  `python refresh_attacks.py` prints the table.

Every attack stacks: a carrier, a chunk whose payload is another credited chunk's stored form, so that holding the
carrier's stored form answers for both (decoding is k squarings a word, encoding k square roots).  Its payload is
chosen after the other chunk's key is drawn, which the refresh cut forbids; each attack takes one route around one
rule."""
from __future__ import annotations

import contextlib
import dataclasses
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from poub_fakes import Clock, Entropy
from test_appends import N, pages, salts, store

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.prover.rota import REGISTRATION, Rota
from verity.protocols.resource_accounting.memory.poub.verifier import audit, coins, schedule
from verity.protocols.resource_accounting.memory.poub.verifier.audit import AuditRun, Params
from verity.protocols.resource_accounting.memory.poub.verifier.receipts import Plan

WINDOWS = 5
LIFETIME = 2
"""L: at 2 the next window boundary and a full period after a commit differ, which the early plan needs."""
SCHEDULE_CAP = 4096
CAPACITY = float(1 << 50)
"""Bytes a second: large enough that no record's reads exceed it, a power of two so the credit reads back exactly."""
REG = "REG"
"""The registered store, as a name `Session.read` and `Session.release` take."""


def refresh_mirrored() -> bool:
    """Whether the verifier this tree holds has the refresh setting (`Params.refresh`)."""
    return "refresh" in {f.name for f in dataclasses.fields(Params)}


class Rejected(Exception):
    def __init__(self, verdict: audit.Verdict) -> None:
        super().__init__(verdict.failure)
        self.verdict = verdict


class Pending(Exception):
    """A rule this tree's verifier doesn't have yet, so its switch has nothing to patch."""


class Session:
    """One session, the verifier's `AuditRun` and the reference prover in step, every receipt sampled, a pass a step.
    Appends and passes name blocks by the append's name (`REG` the registered store), a run `(first, count)`, or
    `(name, m)` for block m of an append."""

    def __init__(self, *, refresh: bool = True, windows: int = WINDOWS, lifetime: int = LIFETIME) -> None:
        self.store = store()
        reg = self.store.registration
        self.refresh = refresh
        self.p = Params(blocks=N, deadline_ns=10**6, windows=windows, samples=10**6, delta=1e-6, window_ns=10,
                        codec=reg.codec, lifetime=lifetime, schedule_cap=SCHEDULE_CAP,
                        **({"refresh": True} if refresh else {}))
        self.clock = Clock()
        self.s = AuditRun(self.p, reg, None, coins.local(reg.digest, 0, windows, Entropy(b"coins")), not_before_ns=0,
                          clock=self.clock)
        self.prover = audit_prover.Prover(self.store, None, entropy=Entropy(b"receipt salts"))
        self.challenges = [self.s.start()]
        self.t, self.i, self.step, self.previous = 0, 0, -1, reg.digest
        self.head: bytes | None = None
        self.named: dict[str, tuple[int, int, int]] = {REG: (-1, 0, N)}
        self.origins: list[tuple[str, int] | None] = []
        self.passes: list[audit_prover.Pass] = []
        self.by_window: dict[tuple[int, int], audit_prover.Pass] = {}
        self.controlled: list = []

    def runs(self, what) -> tuple[int, int]:
        if type(what) is str:
            _, first, count = self.named[what]
            return first, count
        if type(what[0]) is str:
            _, first, count = self.named[what[0]]
            assert 0 <= what[1] < count
            return first + what[1], 1
        return what

    def stored(self, what) -> list[bytes]:
        """The stored forms of `what`'s blocks, as a carrier's payloads."""
        first, count = self.runs(what)
        return [self.store.blocks[j] for j in range(first, first + count)]

    def _append(self, name: str, payloads: Sequence[bytes], origin: tuple[str, int] | None) -> int:
        a = self.store.append(list(payloads), salts(b"salts of " + name.encode(), len(payloads)))
        got = self.s.append(a.root, a.count) if origin is None else self.s.append(a.root, a.count, origin=origin)
        if type(got) is audit.Verdict:
            raise Rejected(got)
        first, coin = got
        assert first == a.first
        self.store.encode(first, coin)
        self.head = audit.append_head(self.head, first, a.count, a.root)
        self.named[name] = (len(self.origins), first, a.count)
        self.origins.append(origin)
        return first

    def fresh(self, name: str, payloads: Sequence[bytes]) -> int:
        return self._append(name, payloads, None)

    def rekey(self, name: str, of: str | tuple[str, int], payloads: Sequence[bytes] | None = None,
              count: int | None = None) -> int:
        """A re-key of registered blocks from `of = (REG, a)` (`count` of them, N by default) or of the append named
        `of`, carrying its origin's payloads unless the prover sends others."""
        if type(of) is tuple:
            origin, first, count = ("registered", of[1]), of[1], N - of[1] if count is None else count
        else:
            k, first, count = self.named[of]
            origin = ("append", k)
        if payloads is None:
            payloads = [self.store.payload(j) for j in range(first, first + count)]
        return self._append(name, payloads, origin)

    def release(self, *what) -> None:
        runs = _canonical([self.runs(w) for w in what])
        if (v := self.s.release(runs)) is not None:
            raise Rejected(v)

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

    def _open(self, ps: audit_prover.Pass, q: int) -> audit.Opening:
        """The reference prover's opening, with a re-key's origin fields as the prover can make them: the origin's
        position for the block, and the salt and path of that position's leaf, whatever that leaf commits."""
        o = self.prover.open(ps, q)
        j = ps.plan.block(q)
        if j < N:
            return o
        k, first = next((k, first) for k, first, count in self.named.values() if k >= 0 and first <= j < first + count)
        origin = self.origins[k]
        if origin is None:
            return o
        if origin[0] == "registered":
            at = origin[1] + j - first
            salt, path = self.store.salts[at], self.store.payload_tree.open(at).path
        else:
            a = self.store.appends[origin[1]]
            at = a.first + j - first
            salt, path = a.salts[at - a.first], a.tree.open(at).path
        return dataclasses.replace(o, origin_position=at, origin_salt=salt, origin_path=path)

    def finish(self) -> None:
        while self.t < self.p.windows:
            self.next_window()
        self.clock.now = self.s.deadline(len(self.s.windows) - 1) + 1
        for t in range(self.p.windows):
            sampled = self.s.close(t)
            if type(sampled) is audit.Verdict:
                raise Rejected(sampled)
            v = self.s.verify(t, [self._open(self.by_window[t, i], q) for i, q in sampled])
            if not v.accepted:
                raise Rejected(v)

    def schedule(self) -> schedule.Schedule:
        """The schedule whose plan at each pass's step is the plan the pass read: every chunk resident exactly while
        planned, a registered one from the start, so `schedule.check` takes it when the plans allow one."""
        steps = list(range(len(self.passes)))
        planned: dict[int, list[int]] = {}
        for s, ps in zip(steps, self.passes):
            for j in ps.plan:
                planned.setdefault(j, []).append(s)
        for j, ss in planned.items():
            if ss != list(range(ss[0], ss[-1] + 1)) or (j < N and ss[0] != 0):
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
        appends = []
        for _, first, count in sorted(v for v in self.named.values() if v[0] >= 0):
            appends.append(tuple(None if j not in planned else
                                 (planned[j][0] - 1, None if planned[j][-1] == last else planned[j][-1])
                                 for j in range(first, first + count)))
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


def credited_receipts(s: Session, verdict: audit.PostSeedVerdict | None = None) -> int:
    """The receipts the ledger credits on the session's record and schedule check (and `verdict`, for a record
    without refresh), less its carrier charge."""
    rec = s.s.record()
    g = audit.ledger(rec, capacity_bytes_per_s=CAPACITY, chip_bytes=0, rebuild=audit.Rebuild(rec.codec, 0.0),
                     schedule=s.s.scheduled(s.schedule()), credit_post_seed=verdict)
    whole = CAPACITY * rec.span_ns / 1e9
    credited = round((1 - g.timing) * whole / rec.block_bytes)
    return credited - round(g.carrier * whole / rec.block_bytes)


def free_receipts(s: Session, verdict: audit.PostSeedVerdict | None = None) -> int:
    """Credited receipts beyond the stored forms the prover held and read: `credited_receipts` less `held_reads`,
    none when the ledger credits fewer than those."""
    return max(0, credited_receipts(s, verdict) - s.held_reads())


def _canonical(runs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for f, c in sorted(runs):
        if out and out[-1][0] + out[-1][1] == f:
            out[-1][1] += c
        else:
            out.append([f, c])
    return tuple((f, c) for f, c in out)


R1 = "R1@1"
"""The rota's re-key of registered class 1 (blocks 4 … 7) joining window 1, the key it reads them under in windows 1
and 2."""


def control(s: Session, t: int, kept: Sequence[int] = (), keep: Sequence[str] = ()) -> list:
    """The honest prover's moves in window t on the refresh rota (`rota.Rota`), returning what its pass reads: a KV
    append F of 2 pages in window 0; the re-keys due at t, `R{c}@{t}` of registered class c's run and `F@{t}` of F;
    the release of what the window before read under a key this one doesn't, but for `kept` registered blocks and
    `keep` appends; and every chunk read under the key the rota gives it at t, F only once its probation is over."""
    r = Rota(N, s.p.lifetime, s.p.windows)
    if t == 0:
        s.fresh("F", pages(N, 2))
    reads: list = []
    for c in range(r.lifetime):
        first, count = r.class_runs(c)
        if count:
            key = r.key(0, c, t, registered=True)
            reads.append((first, count) if key == REGISTRATION else f"R{c}@{key}")
    if (key := r.key(0, r.append_class(0), t, registered=False)) is not None:
        reads.append(f"F@{key}")
    gone = []
    for w in s.controlled:
        if w in reads:
            continue
        if type(w) is str:
            gone += [] if w in keep else [w]
        else:
            gone += [(j, 1) for j in range(w[0], w[0] + w[1]) if j not in kept]
    if gone:
        s.release(*gone)
    s.controlled = reads
    for kind, a in r.due(t, {0: 0}):
        if kind == "registered":
            s.rekey(f"R{r.registered_class(a)}@{t}", (REG, a), count=dict(r.registered_due(t))[a])
        else:
            s.rekey(f"F@{t}", "F")
    return reads


def honest(s: Session) -> None:
    for t in range(s.p.windows):
        s.read(*control(s, t))
        if t < s.p.windows - 1:
            s.next_window()


def plain_stacking(s: Session) -> None:
    """Two fresh appends in window 0, B's payload A's stored form, chosen once A's coin is out; both read in windows
    0 and 1, their key's lifetime."""
    reads = control(s, 0)
    s.read(*reads)
    s.fresh("A", pages(16, 1))
    s.fresh("B", s.stored("A"))
    s.read(*reads, "A", "B")
    s.next_window()
    s.read(*control(s, 1), "A", "B")
    for t in range(2, s.p.windows):
        s.next_window()
        s.read(*control(s, t))


def registered_carrier(s: Session) -> None:
    """A fresh append B in window 0 carrying registered block 0's stored form under the seed's key, re-keyed each
    window from 1 (B{t}), aged L by window 2, read from window 2 with registered block 0 kept past its key's
    lifetime."""
    s.read(*control(s, 0))
    s.fresh("B", s.stored((0, 1)))
    for t in range(1, s.p.windows):
        s.next_window()
        reads = control(s, t, kept=(0,))
        if t >= 3:
            s.release(f"B{t - 2}")
        if t >= 2:
            reads += [(0, 1), f"B{t - 1}"]
        if t < s.p.windows - 1:
            s.rekey(f"B{t}", "B")
        s.read(*reads)


def forged_age(s: Session) -> None:
    """In window 1, once R1's coin is out, a re-key naming F (born in window 0) whose payloads are R1's first two
    stored forms: under F's age it is plannable in window 2, where R1 is read."""
    s.read(*control(s, 0))
    s.next_window()
    s.read(*control(s, 1))
    s.rekey("X", "F", payloads=s.stored((R1, 0)) + s.stored((R1, 1)))
    s.next_window()
    s.read(*control(s, 2), "X")
    for t in range(3, s.p.windows):
        s.next_window()
        s.read(*control(s, t))


def laundered_registered(s: Session) -> None:
    """In window 0 a re-key naming registered blocks 2 and 3 whose payloads are their stored forms, so born before
    the session and read at once beside them."""
    reads = control(s, 0)
    s.rekey("Y", (REG, 2), payloads=s.stored((2, 2)), count=2)
    s.read(*reads, "Y")
    for t in range(1, s.p.windows):
        s.next_window()
        if t == 2:
            s.release("Y")
        s.read(*control(s, t), *(["Y"] if t == 1 else []))


def laundered_rekey(s: Session) -> None:
    """In window 0 a re-key K2 naming registered blocks 2 and 3 whose payloads are their stored forms, never read,
    so never opened, then a re-key K3 of K2, which carries them under K2's origin, read beside them: credited only
    were K3 born before the session, as K2 claims, rather than in the window K2 joined (`State.born`)."""
    reads = control(s, 0)
    s.rekey("K2", (REG, 2), payloads=s.stored((2, 2)), count=2)
    s.rekey("K3", "K2")
    s.read(*reads, "K3")
    for t in range(1, s.p.windows):
        s.next_window()
        if t == 2:
            s.release("K3")
        s.read(*control(s, t), *(["K3"] if t == 1 else []))


def early_plan(s: Session) -> None:
    """In window 1, once R1's coin is out, a fresh append B carrying R1's first stored form, re-keyed (B1) and read
    in window 2 beside R1: the next window boundary after its commit, a window short of its full period."""
    s.read(*control(s, 0))
    s.next_window()
    s.read(*control(s, 1))
    s.fresh("B", s.stored((R1, 0)))
    s.rekey("B1", "B")
    s.next_window()
    s.read(*control(s, 2), "B1")
    for t in range(3, s.p.windows):
        s.next_window()
        s.read(*control(s, t))


def late_rekey(s: Session) -> None:
    """In window 1, once R1's coin is out, a fresh append B carrying R1's first stored form, re-keyed in window 2
    (B2) and read in window 3 when its payload has aged L, beside R1's first block kept past R1's key's lifetime."""
    s.read(*control(s, 0))
    s.next_window()
    s.read(*control(s, 1))
    s.fresh("B", s.stored((R1, 0)))
    s.next_window()
    s.read(*control(s, 2))
    s.rekey("B2", "B")
    s.next_window()
    reads = control(s, 3, keep=(R1,))
    s.release(*[(R1, m) for m in range(1, s.runs(R1)[1])])
    s.read(*reads, (R1, 0), "B2")
    s.next_window()
    s.release((R1, 0), "B2")
    s.read(*control(s, 4))


@dataclass(frozen=True)
class Attack:
    name: str
    rule: str
    """The one rule that stops it, a key of `SWITCHES`."""
    failure: str
    """What the verifier names when the rule stops it."""
    play: Callable[[Session], None]
    free_off: int | None
    """The free receipts it earns with its rule off, a carried block a pass beside its carrier; None where another
    rule still rejects it (`off_failure`)."""
    gain: str
    off_failure: str | None = None


ATTACKS = (
    Attack("plain stacking", "probation", "probation", plain_stacking, 2,
           "B's stored form answers for A's: one held a pass for two credited"),
    Attack("carrier of a registered chunk", "rule 1", "lifetime", registered_carrier, 3,
           "B's stored form answers for registered block 0's, whose key is the seed's"),
    Attack("forged age", "origin path", "origin-path", forged_age, 2,
           "X's stored forms answer for R1's first two: a payload of window 1 credited as F's of window 0"),
    Attack("laundered origin: registered", "origin path", "origin-path", laundered_registered, 4,
           "Y's stored forms answer for registered blocks 2 and 3: a payload chosen after the seed credited as "
           "registered"),
    Attack("laundered origin: re-key of a re-key", "fresh origin", "append", laundered_rekey, None,
           "none: K3's birth is the window K2 joined, so probation holds it a full period past K2's commit",
           off_failure="probation"),
    Attack("early plan", "payload timing", "probation", early_plan, 1,
           "B1's stored form answers for R1's first: credited at the next boundary after its commit"),
    Attack("late re-key", "key timing", "lifetime", late_rekey, 1,
           "B2's stored form answers for R1's first, kept past R1's key's lifetime"),
)


def _stale_below_off(self: AuditRun, t: int) -> int:
    return self.params.blocks


def _pending(rule: str) -> Callable[[], Iterator[None]]:
    @contextlib.contextmanager
    def patch() -> Iterator[None]:
        raise Pending(f"{rule}: this tree's verifier has no such check to switch off")
        yield
    return patch


@contextlib.contextmanager
def _key_timing() -> Iterator[None]:
    with _patched(AuditRun, "_stale_below", _stale_below_off):
        yield


@contextlib.contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = owner.__dict__[name]
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


SWITCHES: dict[str, Callable[[], contextlib.AbstractContextManager]] = {
    "rule 1": _pending("rule 1"),
    "probation": _pending("probation"),
    "origin path": _pending("origin path"),
    "fresh origin": _pending("fresh origin"),
    "payload timing": _pending("payload timing"),
    "key timing": _key_timing,
}
"""Each rule's test-only switch: a patch of the verifier's class for the length of one run, here and nowhere in the
verifier's code, which takes no parameter for it."""


def rule_off(rule: str) -> contextlib.AbstractContextManager:
    return SWITCHES[rule]()


@dataclass(frozen=True)
class Outcome:
    accepted: bool
    failure: str | None
    free: int | None
    """Free receipts on an accepted session, None on a rejected one, which credits nothing."""


def play(attack: Callable[[Session], None], **kw) -> Outcome:
    s = Session(**kw)
    try:
        attack(s)
        s.finish()
    except Rejected as e:
        return Outcome(False, e.verdict.failure, None)
    return Outcome(True, None, free_receipts(s))


def pair(a: Attack) -> tuple[Outcome, Outcome]:
    """The attack with its rule switched off, then with refresh as designed."""
    with rule_off(a.rule):
        off = play(a.play)
    return off, play(a.play)


def table() -> str:
    rows = [("attack", "rule that stops it", "free receipts, rule off", "free receipts, rule on", "failure named")]
    control_run = play(honest)
    rows.append(("honest (control)", "-", "-", str(control_run.free), control_run.failure or "accepted"))
    for a in ATTACKS:
        try:
            off, on = pair(a)
        except Pending as e:
            rows.append((a.name, a.rule, f"pending: {e}", "", ""))
            continue
        rows.append((a.name, a.rule, "rejected " + str(off.failure) if off.free is None else str(off.free),
                     "rejected" if on.free is None else str(on.free), on.failure or "accepted"))
    widths = [max(len(r[c]) for r in rows) for c in range(5)]
    return "\n".join(" | ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip() for r in rows)


if __name__ == "__main__":
    if not refresh_mirrored():
        sys.exit("this tree's verifier has no refresh setting (Params.refresh): merge the Python mirror first")
    print(table())
