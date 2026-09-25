"""Three ways to hook the protocols into the engine.  Each produces the same epoch; they differ in who decides what
runs where, and in what the protocols must know about the engine.

    registry       the application writes one table: hook -> protocol calls, in order.  Protocols keep their own
                   method signatures and know nothing about hooks.  (Recommended.)
    phases         each protocol gets an adapter with per-phase methods named after the engine's hooks (`on_commit`,
                   ...); a dispatcher calls every adapter at every hook, in one fixed protocol order.
    subscriptions  each adapter subscribes its own handlers; order is whatever order they subscribed in.

In all three the adapters live in the application: a protocol cannot import the engine, and hook vocabulary is not
core.  So `phases` and `subscriptions` are the registry with the table split across per-protocol classes.
"""

from __future__ import annotations

from verity_pous import respond
from verity_pouw import noise


def _sectors(world, words: int) -> int:
    return -(-words // world.memory.sector_words)


def _retain(world, now):
    world.memory.allocate("sp/retained", world.cfg.retention, _sectors(world, world.prover.retained_words),
                          "sampled-proofs")


def _audit(world, index: int):
    def handler(scheduled: float) -> None:
        ch, clock = world.challenger, world.clock
        issued = max(scheduled, world.audits[-1][4] if world.audits else 0.0)  # the challenger audits one at a time
        waited = clock.now - issued
        c = ch.challenge(index, issued)
        r = respond(world.pous, world.memory.read, c, waited)
        clock.advance(r.elapsed - waited)
        owned = len(world.memory.regions["kv"].words) - len(world.engine.free_pages)
        world.audits.append((ch.check(c, r), waited, r.elapsed - waited, owned, clock.now))
    return handler


def _schedule_audits(world):
    for i, t in enumerate(world.challenger.times(world.cfg.epoch)):
        world.clock.at(t, _audit(world, i))


def registry(world) -> None:
    h, w, K = world.engine.hooks, world, world.engine.cfg.K
    h.on("admit", lambda a: w.log.register(f"pouw/header/{a.request.id}", a.request.header, a.now))
    h.on("admit", lambda a: w.pouw.issue(a.request.id, w.device, a.request.header, w.log))
    h.on("alloc", lambda a: w.challenger.tap("kv", a.page, a.words))
    h.on("step_start", lambda s: s.noise.update({r: noise(w.pouw.issued[r], s.id, K) for r in s.placement}))
    h.on("commit", lambda c: w.prover.commit(c.step.id, c.step.program, c.values, c.now))
    h.on("commit", lambda c: [w.challenger.tap("kv", page, words) for page, words in c.kv_writes])
    h.on("commit", lambda c: _retain(w, c.now))
    h.on("step_end", lambda s: w.tickets.extend(w.pouw.tickets(s.id, s.program, s.placement, s.values)))
    h.on("step_end", lambda s: (w.prover.poll(w.clock.now), _retain(w, w.clock.now)))
    h.on("idle", lambda i: (w.prover.poll(i.until), _retain(w, i.until)))
    _schedule_audits(w)


class _PoUW:
    def __init__(self, w):
        self.w = w

    def on_admit(self, a):
        self.w.log.register(f"pouw/header/{a.request.id}", a.request.header, a.now)
        self.w.pouw.issue(a.request.id, self.w.device, a.request.header, self.w.log)

    def on_step_start(self, s):
        s.noise.update({r: noise(self.w.pouw.issued[r], s.id, self.w.engine.cfg.K) for r in s.placement})

    def on_step_end(self, s):
        self.w.tickets.extend(self.w.pouw.tickets(s.id, s.program, s.placement, s.values))


class _SampledProofs:
    def __init__(self, w):
        self.w = w

    def on_commit(self, c):
        self.w.prover.commit(c.step.id, c.step.program, c.values, c.now)
        _retain(self.w, c.now)

    def on_step_end(self, s):
        self.w.prover.poll(self.w.clock.now)
        _retain(self.w, self.w.clock.now)

    def on_idle(self, i):
        self.w.prover.poll(i.until)
        _retain(self.w, i.until)


class _PoUS:
    def __init__(self, w):
        self.w = w
        _schedule_audits(w)

    def on_alloc(self, a):
        self.w.challenger.tap("kv", a.page, a.words)

    def on_commit(self, c):
        for page, words in c.kv_writes:
            self.w.challenger.tap("kv", page, words)


ADAPTERS = (_PoUW, _SampledProofs, _PoUS)


def phases(world, order=ADAPTERS) -> None:
    adapters = [cls(world) for cls in order]
    for name in world.engine.hooks.NAMES:
        world.engine.hooks.on(name, lambda ev, name=name: [getattr(a, f"on_{name}")(ev) for a in adapters
                                                           if hasattr(a, f"on_{name}")])


def subscriptions(world, order=ADAPTERS) -> None:
    for a in (cls(world) for cls in order):
        for name in world.engine.hooks.NAMES:
            if hasattr(a, f"on_{name}"):
                world.engine.hooks.on(name, getattr(a, f"on_{name}"))


SHAPES = {"registry": registry, "phases": phases, "subscriptions": subscriptions}
