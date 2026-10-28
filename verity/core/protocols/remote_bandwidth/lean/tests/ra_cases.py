"""The Python side of the remote audit's difftests, which hold `audit.py`, `receipts.py` and `codec.py` to the Lean
(`RemoteAudit`, the verifier's definition: `../../README.md`): the Python behind the Lean's JSON formats
(`RemoteAudit.Difftest`), and the sessions the tests record. Where the two differ, the Python is what is wrong.

The decoders mirror `RemoteAudit.Difftest`'s: byte strings lowercase hex, indices integers, a registration checked as
`audit.register` checks one, and `bind` refused (the Lean covers receipts of blocks alone), as `audit.Registry.register`
refuses a registration that would bind (counted, for a node with one live). Past decoding each side is
the Python's own code on the decoded objects. A session's events run through `audit.AuditRun` itself, on a clock each
event sets, its coins those of a challenger that replays the case's (`Replay`).

A recorded session is one `Run` drives through a real `AuditRun`, the prover's side computed by `audit.Prover` from a
`Store`, honest or with one deviation, every call kept as an event with the clock it read. `SCENARIOS` are the recorded
ones; each names the outcome the definition gives it, which `test_session.py` holds the Lean's answer to, so a scenario
that stops reaching its failure fails there, not silently in the difftest."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import verity_difftest as D
from verity_difftest import protocol as P
from verity.core.protocols import registration as log
from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts, wire
from verity.core.protocols.remote_bandwidth.audit import AuditRun, Params, SessionError
from verity.core.protocols.remote_bandwidth.receipts import Plan
from verity.core.protocols.sampled_proofs.challenger import COIN_BYTES as LIVE_COIN_BYTES, HM96, LiveChallenger

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
EXE = "remote-audit-difftest"
#: The ports' side of record (`verity_difftest`): the Lean, the verifier `benchmarks/remote_session/verifier.py` runs.
OF_RECORD = "lean"
#: What the Python raises where the Lean refuses: a decoder's `Reject`, and the Python's own refusals, `ValueError`
#: (`verity.errors.ProtocolError` and `InvalidArtifact` among them), `TypeError` and `IndexError`.
REFUSALS = (D.Reject, ValueError, TypeError, IndexError)

P512 = codec.Codec(codec.P512, 2, False)
P512_K3 = codec.Codec(codec.P512, 3, True)
NODE = b"node-under-test"
SOURCE = b"the verifier's secret"


# --- decoding, as `RemoteAudit.Difftest` decodes ----------------------------------------------------------------------

def _get(j: Any, k: str) -> Any:
    if not isinstance(j, dict) or k not in j:
        raise D.Reject(f"{k} missing")
    return j[k]


def _str(j: Any, k: str) -> str:
    v = _get(j, k)
    if not isinstance(v, str):
        raise D.Reject(f"{k}: not a string")
    return v


def _bool(j: Any, k: str) -> bool:
    v = _get(j, k)
    if not isinstance(v, bool):
        raise D.Reject(f"{k}: not a boolean")
    return v


def _list(j: Any, k: str) -> list:
    v = _get(j, k)
    if not isinstance(v, list):
        raise D.Reject(f"{k}: not a list")
    return v


def _int(v: Any, k: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise D.Reject(f"{k}: not an integer")
    return v


def int_(j: Any, k: str) -> int:
    return _int(_get(j, k), k)


def nat(j: Any, k: str) -> int:
    _get(j, k)
    return P.nat(j, k)


def _unhex(v: Any, k: str) -> bytes:
    if not isinstance(v, str):
        raise D.Reject(f"{k}: not a string")
    return P.unhex(v)


def hex_(j: Any, k: str) -> bytes:
    return _unhex(_get(j, k), k)


def hex_or_none(j: Any, k: str) -> bytes | None:
    v = _get(j, k)
    return None if v is None else _unhex(v, k)


def hexes(j: Any, k: str) -> list[bytes]:
    return [_unhex(x, k) for x in _list(j, k)]


def runs_of(v: Any, k: str) -> tuple[tuple[int, int], ...]:
    """`runsOf`: a list of `[first, count]` integer pairs."""
    if not isinstance(v, list):
        raise D.Reject(f"{k}: not a list")
    out = []
    for r in v:
        if not isinstance(r, list) or len(r) != 2:
            raise D.Reject(f"{k}: a run is [first, count]")
        out.append((_int(r[0], k), _int(r[1], k)))
    return tuple(out)


@contextlib.contextmanager
def decoding():
    """The Python's refusals while it decodes its input, as `Reject`: past decoding, a `ValueError` is a difference."""
    try:
        yield
    except (ValueError, TypeError) as e:
        raise D.Reject(str(e)) from None


def codec_json(c: codec.Codec) -> dict:
    return {"field": c.field.name, "masked": c.masked, "rounds": c.rounds}


def codec_of(v: Any) -> codec.Codec:
    name, rounds, masked = _str(v, "field"), nat(v, "rounds"), _bool(v, "masked")
    if name not in codec.FIELDS:
        raise D.Reject(f"no field {name}")
    return codec.Codec(codec.FIELDS[name], rounds, masked)


def registration_json(r: audit.Registration) -> dict:
    return {"bind": r.bind, "blocks": r.blocks, "codec": codec_json(r.codec), "codec_seed": r.codec_seed.hex(),
            "layout": r.layout, "node": r.node.hex(), "payload_root": r.payload_root.hex(), "receipt": r.receipt,
            "rows": r.rows}


def registration_of(v: Any) -> audit.Registration:
    """`Registration.make?`: `audit.register`'s checks on a registration whose seed is given, `bind` refused."""
    kind = _str(v, "receipt")
    node, blocks, root, seed = hex_(v, "node"), nat(v, "blocks"), hex_(v, "payload_root"), hex_(v, "codec_seed")
    c = codec_of(_get(v, "codec"))
    bind, layout, rows = _bool(v, "bind"), _str(v, "layout"), nat(v, "rows")
    receipts.receipt_kind(kind)
    audit._node(node)
    receipts.store_blocks(blocks)
    if blocks > c.max_blocks:
        raise ValueError(f"a store under {c.name} is at most {c.max_blocks} blocks")
    if len(root) != receipts.DIGEST_BYTES:
        raise ValueError("the payload root is 64 bytes")
    if audit._bind(bind):
        raise D.Reject("a registration that binds PoUW's tile outputs is not covered: the port refuses it")
    receipts.block_layout(layout)
    receipts.sketch_rows(kind, rows)
    out = audit.Registration(node, blocks, root, seed, c, kind, False, layout, rows)
    out.key
    return out


def opening_json(o: audit.Opening) -> dict:
    return {"block": o.block.hex(), "digest": o.digest.hex(), "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "pass": o.pass_index, "payload_path": [x.hex() for x in o.payload_path],
            "payload_salt": o.payload_salt.hex(), "position": o.position,
            "receipt_path": [x.hex() for x in o.receipt_path], "tile": o.tile.hex()}


def opening_of(v: Any) -> audit.Opening:
    return audit.Opening(int_(v, "pass"), int_(v, "position"), hex_(v, "block"), hex_(v, "leaf"), hex_(v, "leaf_salt"),
                         tuple(hexes(v, "receipt_path")), hex_(v, "payload_salt"), tuple(hexes(v, "payload_path")),
                         hex_(v, "tile"), hex_(v, "digest"))


def event_of(v: Any) -> tuple:
    """`eventOf`: an event as a tuple, its kind first. A `verify` carries its openings, or the frames they came in
    (`frames`), read as the live verifiers read them (`wire.openings_of`)."""
    kind = _str(v, "kind")
    if kind in ("start", "boundary"):
        return kind, nat(v, "now")
    if kind == "append":
        return kind, hex_(v, "root"), int_(v, "count")
    if kind == "release":
        return kind, runs_of(_get(v, "runs"), "runs"), runs_of(_get(v, "withdrawn"), "withdrawn")
    if kind == "receive":
        runs = _get(v, "runs")
        change = _get(v, "change")
        if change is not None:
            if not isinstance(change, list) or len(change) != 2:
                raise D.Reject("a change is [added, removed]")
            change = (runs_of(change[0], "added"), runs_of(change[1], "removed"))
        step = _get(v, "step")
        return (kind, int_(v, "t"), int_(v, "i"), hex_(v, "root"), None if runs is None else runs_of(runs, "runs"),
                change, None if step is None else _int(step, "step"), nat(v, "now"))
    if kind == "close":
        return kind, int_(v, "t"), nat(v, "now")
    if kind == "verify":
        if "frames" in v:
            return kind, int_(v, "t"), wire.openings_of(hexes(v, "frames"))
        return kind, int_(v, "t"), [opening_of(o) for o in _list(v, "openings")]
    raise D.Reject(f"no event kind {kind}")


# --- the session ----------------------------------------------------------------------------------------------------

class Clock:
    """The suite's one injected clock: nanoseconds from `now`, each reading `step` after the last (at `step = 0` time moves
    only when a test sets `now`); waiting until `t` (`period.run`'s `sleep_until`) moves it there."""

    def __init__(self, now: int = 0, step: int = 0) -> None:
        self.now, self.step = now, step

    def __call__(self) -> int:
        self.now += self.step
        return self.now

    def sleep_until(self, _clock, t: int) -> None:
        self.now = max(self.now, t)


class Replay:
    """Session k's challenger replaying a case's coins: a `LiveChallenger` of the session's layout whose coins are the
    case's 2W, then its append coins in the order drawn, each the audit's 32 bytes padded to the challenger's 64 under a
    zero salt. Past the case's append coins it refuses the case, as the Lean does."""

    def __init__(self, k: int, windows: int, committed: list[bytes], appends: list[bytes], per_coin: int = 1) -> None:
        a = max(coins.APPENDS, len(appends))
        padded = [c + bytes(LIVE_COIN_BYTES - len(c)) for c in [*committed, *appends]]
        out = iter(x for c in padded + [bytes(LIVE_COIN_BYTES)] * (a - len(appends))
                   for x in (c, bytes(HM96.salt_bytes)))
        self.live = LiveChallenger(coins.SCHEME, k, coins.families(windows, a, per_coin), lambda n: next(out))
        self.commitment, self.appends = self.live.commitment, len(appends)

    def open(self, label: str, index: int, after: bytes) -> coins.LiveOpening:
        if label == coins.APPEND and index >= self.appends:
            raise D.Reject("the append coins ran out")
        return self.live.open(label, index, after)


def replayed(reg: audit.Registration, k: int, windows: int, committed: list[bytes], appends: list[bytes],
             per_coin: int = 1) -> coins.SessionCoins:
    return coins.SessionCoins(Replay(k, windows, committed, appends, per_coin), reg.digest, k, windows,
                              per_coin=per_coin)


def verdict_json(v: audit.Verdict) -> dict:
    return {"accepted": v.window} if v.accepted else {
        "at": None if v.at is None else list(v.at), "failure": v.failure, "window": v.window}


def _done(out: audit.Verdict | None) -> dict:
    return {"ok": True} if out is None else verdict_json(out)


def apply(s: AuditRun, clock: Clock, e: tuple) -> dict:
    """One event through the session: what the call returns, or `refused` where it raises `SessionError`."""
    kind = e[0]
    try:
        if kind == "start":
            clock.now = e[1]
            return {"released": s.start().index}
        if kind == "boundary":
            clock.now = e[1]
            c = s.boundary()
            return {"released": None if c is None else c.index}
        if kind == "append":
            out = s.append(e[1], e[2])
            return verdict_json(out) if isinstance(out, audit.Verdict) else {"coin": out[1].hex(), "first": out[0]}
        if kind == "release":
            return _done(s.release(e[1], e[2]))
        if kind == "receive":
            _, t, i, root, runs, change, step, now = e
            clock.now = now
            return _done(s.receive(t, i, root, runs, change=change, step=step))
        if kind == "close":
            clock.now = e[2]
            out = s.close(e[1])
            return verdict_json(out) if isinstance(out, audit.Verdict) else {"draws": [list(d) for d in out]}
        return verdict_json(s.verify(e[1], e[2]))
    except SessionError:
        return {"refused": True}


def summary(s: AuditRun) -> dict:
    """`summaryJ`: what `AuditRun.record` gives that the port computes, which is all of it but the coin commitment's
    root, the block size and the registration's fields its digest binds, and of each window's profile its draws; and
    `sessionFn`'s `groups_per_coin` when it isn't 1."""
    rec = s.record()
    shared = {} if rec.groups_per_coin == 1 else {"groups_per_coin": rec.groups_per_coin}
    return shared | {"appended": rec.appended, "appends": None if rec.appends is None else rec.appends.hex(),
            "blocks": rec.blocks, "lifetime": rec.lifetime, "pace": None if rec.pace is None else rec.pace.to_json(),
            "registration": rec.registration.hex(), "released": rec.released,
            "windows": [{"arrived_ns": list(w.arrived_ns), "deadline_ns": w.deadline_ns,
                         "drawn": w.profile.drawn["receipt"], "heads": [None if h is None else h.hex() for h in w.heads],
                         "lengths": list(w.lengths), "opened_ns": w.opened_ns, "post_seed": list(w.post_seed),
                         "released_ns": w.released_ns, "stale": list(w.stale), "steps": list(w.steps)}
                        for w in rec.windows]}


def _positive(p: Any, k: str) -> int:
    n = nat(p, k)
    if n == 0:
        raise D.Reject(f"{k} is a positive integer")
    return n


def pace_of(v: Any) -> audit.Pace | None:
    """`sessionOf`'s pace: None, or `{"k", "window_ns"}`, each a positive integer, and `in_flight`, 1 or 2, if it names
    it (`flightOf`: 2 when it doesn't)."""
    if v is None:
        return None
    k, window_ns = _positive(v, "k"), _positive(v, "window_ns")
    f = nat(v, "in_flight") if "in_flight" in v else 2
    if f not in (1, 2):
        raise D.Reject("a pace's in_flight is 1 or 2")
    return audit.Pace(k, window_ns, f)


def session_of(v: Any) -> tuple[dict, list[bytes], list[bytes], int]:
    """`sessionOf`: a session's parameters (`Params`' keywords less the registration's, refused as `Params` refuses
    them), its W + ⌈W/g⌉ coins, its append coins and its `not_before_ns`. `groups_per_coin` (g) is 1 when absent."""
    p = _get(v, "params")
    kw = {k: _positive(p, k) for k in ("deadline_ns", "windows", "samples", "min_passes", "window_ns", "lifetime")}
    kw["deferred"] = _bool(p, "deferred")
    kw["pace"] = pace_of(_get(p, "pace"))
    kw["groups_per_coin"] = _positive(p, "groups_per_coin") if "groups_per_coin" in p else 1
    Params(1, **kw)
    committed = hexes(v, "coins")
    if len(committed) != Params(1, **kw).coins or any(len(c) != coins.COIN_BYTES for c in committed):
        raise D.Reject("W + ⌈W/g⌉ coins of 32 bytes")
    appends = hexes(v, "append_coins")
    if any(len(c) != coins.COIN_BYTES for c in appends):
        raise D.Reject("append coins are 32 bytes")
    return kw, committed, appends, nat(v, "not_before_ns")


def params_of(reg: audit.Registration, kw: dict) -> Params:
    return Params(reg.blocks, codec=reg.codec, receipt=reg.receipt, layout=reg.layout, rows=reg.rows, **kw)


def session(v: Any) -> dict:
    """`sessionFn`: a case's events through `audit.AuditRun`, every event's result, the outcome and the record."""
    with decoding():
        reg = registration_of(_get(v, "registration"))
        kw, committed, appends, not_before = session_of(v)
        events = [event_of(e) for e in _list(v, "events")]
        params = params_of(reg, kw)
    clock = Clock()
    s = AuditRun(params, reg, None, replayed(reg, 0, params.windows, committed, appends, params.groups_per_coin),
                 not_before_ns=not_before, clock=clock)
    results = [apply(s, clock, e) for e in events]
    return {"outcome": s.outcome, "record": summary(s) if s.outcome == "accepted" else None, "results": results}


# --- the registry ---------------------------------------------------------------------------------------------------

def _object(j: Any, k: str) -> dict:
    v = _get(j, k)
    if not isinstance(v, dict):
        raise D.Reject(f"{k}: not an object")
    return v


def registry_of(v: Any) -> audit.Registry:
    """`Registry.ofJson`: `audit.Registry.from_json` on a registry/v0 whose fields are there and whose byte strings are
    lowercase hex, as the Lean reads them; `None`, an empty registry."""
    if v is None:
        return audit.Registry()
    for n, x in _object(v, "live").items():
        _unhex(n, "live")
        _unhex(_get(x, "registration"), "registration")
        _get(x, "sessions")
    for c in _list(v, "closed"):
        hex_(c, "node"), hex_(c, "registration"), _get(c, "sessions")
    for n in _object(v, "refused"):
        _unhex(n, "refused")
    return audit.Registry.from_json(v)


STORE = "store"


def store_log(node: bytes, blocks: int, payload_root: bytes, store: str = STORE, item: dict | None = None,
              store_bytes: int | None = None) -> dict:
    """A remote site's registration log holding a node's store, with the auditor's parameters, as a registry's `register`
    takes them (`{"params", "log"}`): the site, the node as its host, the store as a memory leaf of the node's N blocks
    (or `store_bytes`), and θ's value for its payload (`audit.payload_item`, or `item`)."""
    n = node.decode()
    site = log.Hardware(locs=(log.Loc("site", "site"), log.Loc(n, "host", "site"),
                              log.Loc(store, "memory", n, tier="dram",
                                      bytes=blocks * codec.BLOCK_BYTES if store_bytes is None else store_bytes)))
    theta = log.Information((audit.payload_item(store, node, blocks, payload_root) if item is None else item,))
    return {"params": log.params("remote", log.Auditor()),
            "log": {"composite": "remote", "entries": [e.to_json() for e in log.chain("remote", [site, theta])]}}


def logged_view(v: Any) -> log.View:
    """`loggedView`: the view of an accepted registration log, `{"params", "log"}`; a log of another composite, or one
    the check refuses, is refused."""
    composite, auditor = log.params_of(_get(v, "params"))
    theirs, entries = log.log_of(_get(v, "log"))
    if theirs != composite:
        raise D.Reject("the log is another composite's")
    try:
        return log.check(auditor, composite, entries).view()
    except log.Refused as e:
        raise D.Reject(str(e)) from None


def registry_op(v: Any) -> tuple:
    """`opOf`: a call on the registry, its kind first. A session names its registration by its place among those drawn
    in the case, or in full."""
    op = _str(v, "op")
    if op == "register":
        bind, kind = _bool(v, "bind"), _str(v, "receipt")
        receipts.receipt_kind(kind)
        return (op, logged_view(v), _str(v, "store"), codec_of(_get(v, "codec")), kind, bind, _str(v, "layout"),
                nat(v, "rows"), hex_(v, "nonce"))
    if op == "session":
        which = _get(v, "registration")
        which = nat(v, "registration") if isinstance(which, int) and not isinstance(which, bool) else registration_of(which)
        index = _get(v, "index")
        return op, which, *session_of(v), None if index is None else _int(index, "index")
    if op == "event":
        return op, hex_(v, "node"), nat(v, "session"), event_of(_get(v, "event"))
    if op in ("close", "live", "drawn"):
        return op, hex_(v, "node")
    raise D.Reject(f"no registry op {op}")


def _registry_call(r: audit.Registry, source: bytes, drawn: list[audit.Registration], op: str, a: tuple) -> Any:
    if op == "register":
        view, store, c, kind, bind, layout, rows, nonce = a
        reg = r.register(source, view, store, c, kind, bind=bind, layout=layout, rows=rows, nonce=nonce)
        drawn.append(reg)
        return {"codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()}
    if op == "session":
        which, kw, committed, appends, not_before, index = a
        reg = drawn[which] if isinstance(which, int) else which
        s = r.session(params_of(reg, kw), reg, None, not_before_ns=not_before, clock=Clock(), index=index,
                      challenger=lambda k: Replay(k, kw["windows"], committed, appends, kw["groups_per_coin"]))
        return {"session": s.commitment.epoch}
    if op == "event":
        node, k, e = a
        x = r._live.get(audit._node(node))
        s = x.sessions[k] if x is not None and k < len(x.sessions) else None
        return apply(s, s.clock, e) if isinstance(s, AuditRun) else {"refused": True}
    if op == "close":
        return r.close(a[0]).to_json()
    if op == "live":
        d = r.live(a[0])
        return None if d is None else d.hex()
    return r.drawn(a[0])


def registry(v: Any) -> dict:
    """`registryFn`: a case's calls on `audit.Registry` under its source, from the registry read back (`registry_of`),
    each call's result (`refused` where it raises `RegistryError`) and the registry after them. A session's events
    go through the `AuditRun` the registry holds, on the clock it was started with."""
    with decoding():
        source = hex_(v, "source")
        r = registry_of(_get(v, "registry"))
        ops = [registry_op(o) for o in _list(v, "ops")]
    drawn: list[audit.Registration] = []
    results = []
    for op, *a in ops:
        try:
            results.append(_registry_call(r, source, drawn, op, tuple(a)))
        except audit.RegistryError:
            results.append({"refused": True})
    return {"registry": r.to_json(), "results": results}


# --- recorded sessions ----------------------------------------------------------------------------------------------

class Entropy:
    """The suite's one entropy source: SHA-512 in counter mode under `label`, in place of the operating system's
    generator."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(label: bytes) -> bytes:
    """64-byte pieces with their top byte cleared, so every word is below p at either width."""
    return b"".join(hashlib.sha512(label + b"/%d" % u).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


def salts(label: bytes, n: int) -> list[bytes]:
    entropy = Entropy(label)
    return [entropy(receipts.SALT_BYTES) for _ in range(n)]


def pages(first: int, count: int, label: bytes = b"kv page") -> list[bytes]:
    return [payload(label + b" %d" % j) for j in range(first, first + count)]


@dataclasses.dataclass(frozen=True)
class World:
    """A registered store: its codec, receipt kind, sketch rows and size."""

    codec: codec.Codec = P512
    receipt: str = receipts.DEFAULT
    rows: int | None = None
    blocks: int = 6


@functools.cache
def _encoded(c: codec.Codec, blocks: int) -> audit.Store:
    payloads = [payload(b"weights %d" % j) for j in range(blocks)]
    ys = salts(b"payload salts", blocks)
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    return audit.Store(audit.register(SOURCE, audit.site_log(NODE, blocks, root), audit.STORE, c,
                                      nonce=bytes(32)), payloads, ys)


def store(world: World) -> audit.Store:
    """A fresh store of the world's registration with no appends yet. The codec seed is drawn without the receipt kind
    (`audit.register`), so worlds of one codec and size share their encoded blocks."""
    base = _encoded(world.codec, world.blocks)
    r = base.registration
    out = copy.copy(base)
    out.registration = audit.Registration(r.node, r.blocks, r.payload_root, r.codec_seed, r.codec, world.receipt, False,
                                          r.layout, receipts.sketch_rows(world.receipt, world.rows))
    out.blocks, out.appends, out.pending = list(base.blocks), [], {}
    return out


REFUSED = "refused"


def _runs_json(runs) -> list | None:
    return None if runs is None else [list(r) for r in runs]


class Run:
    """One session over a world's store, the verifier's side (a real `AuditRun`) and the prover's (`audit.Prover`, from
    `cheat` if given, else the store) in step, every call to the session kept as an event with the clock it read. The
    clock moves one tick an event unless a call needs more. Every receipt is drawn unless `samples` says fewer."""

    def __init__(self, world: World = World(), *, windows: int = 2, samples: int = 10**6, min_passes: int = 1,
                 lifetime: int | None = None, deferred: bool = False, deadline_ns: int = 100, window_ns: int = 10,
                 not_before_ns: int = 0, cheat: Callable[[audit.Store], audit.Store] | None = None,
                 pace: audit.Pace | None = None, groups_per_coin: int = 1) -> None:
        self.store = store(world)
        reg = self.store.registration
        self.params = Params(reg.blocks, deadline_ns, windows, samples=samples, min_passes=min_passes,
                             window_ns=window_ns, codec=reg.codec, receipt=reg.receipt, rows=reg.rows, lifetime=lifetime,
                             deferred=deferred, pace=pace, groups_per_coin=groups_per_coin)
        self.clock = Clock()
        self.clock.now = not_before_ns
        self.not_before_ns = not_before_ns
        self.coins = coins.local(reg.digest, 0, windows, Entropy(b"coins"), per_coin=groups_per_coin)
        self.s = AuditRun(self.params, reg, None, self.coins, not_before_ns=not_before_ns, clock=self.clock)
        self.prover = audit.Prover(cheat(self.store) if cheat else self.store, None, entropy=Entropy(b"receipt salts"))
        self.events: list[dict] = []
        self.append_coins: list[bytes] = []
        self.challenges: dict[int, coins.LiveOpening] = {}
        self.t, self.i, self.previous, self.plan, self.step = 0, 0, reg.digest, Plan.whole(reg.blocks), -1
        self.head: bytes | None = None
        self.passes: dict[tuple[int, int], audit.Pass] = {}
        self.draws: dict[int, tuple] = {}

    def _call(self, event: dict, fn: Callable[[], Any]) -> Any:
        self.events.append(event)
        try:
            return fn()
        except SessionError:
            return REFUSED

    def tick(self, at: int | None = None) -> int:
        self.clock.now = self.clock.now + 1 if at is None else at
        return self.clock.now

    def start(self, at: int | None = None) -> Any:
        out = self._call({"kind": "start", "now": self.tick(at)}, self.s.start)
        if out is not REFUSED:
            self.challenges[0] = out
        return out

    def boundary(self, at: int | None = None) -> Any:
        """The current window's end, at its earliest unless `at`; the prover moves to the next window."""
        t = next((t for t, w in enumerate(self.s.windows) if w.boundary_ns is None), None)
        if at is None and t is not None:
            at = max(self.clock.now + 1, self.s.windows[t].released_ns + self.params.window_ns)
        out = self._call({"kind": "boundary", "now": self.tick(at)}, self.s.boundary)
        if out is not REFUSED and out is not None:
            self.challenges[out.index] = out
        if out is not REFUSED:
            self.t, self.i = self.t + 1, 0
        return out

    def commit(self, root: bytes, count: int) -> Any:
        """An append's root sent: its coin kept and, if the store awaits it, the append's blocks encoded under it."""
        out = self._call({"kind": "append", "root": root.hex(), "count": count}, lambda: self.s.append(root, count))
        if isinstance(out, tuple):
            first, coin = out
            self.append_coins.append(coin)
            self.head = audit.append_head(self.head, first, count, root)
            if first in self.store.pending:
                self.store.encode(first, coin)
        return out

    def append(self, count: int, label: bytes = b"kv page") -> audit.Append:
        a = self.store.append(pages(self.store.extent, count, label),
                              salts(b"page salts %d" % self.store.extent, count))
        self.commit(a.root, a.count)
        return a

    def release(self, runs, withdrawn=()) -> Any:
        return self._call({"kind": "release", "runs": _runs_json(runs), "withdrawn": _runs_json(withdrawn)},
                          lambda: self.s.release(runs, withdrawn))

    def receive(self, t: int, i: int, root: bytes, runs=None, change=None, step=None, at: int | None = None) -> Any:
        event = {"kind": "receive", "t": t, "i": i, "root": root.hex(), "runs": _runs_json(runs),
                 "change": None if change is None else [_runs_json(change[0]), _runs_json(change[1])], "step": step,
                 "now": self.tick(at)}
        return self._call(event, lambda: self.s.receive(t, i, root, runs, change=change, step=step))

    def run(self, plan: Plan | None = None, change=None, *, step: int | None = None, send: dict | None = None,
            challenge: coins.LiveOpening | None = None, previous: bytes | None = None, head=False,
            at: int | None = None) -> Any:
        """One honest pass of the current window, its plan sent whole (None: the registered store) or as its `change`,
        covering `step` (None: the one after the last). `send` replaces what is sent; `challenge`, `previous` and
        `head` (False: the store's), what the prover's seed is computed from."""
        if change is not None:
            plan = self.plan.changed(*change, self.store.extent)
        ps = self.prover.run(challenge or self.challenges[self.t], self.i, self.previous if previous is None else previous,
                             plan, head=self.head if head is False else head)
        self.passes[self.t, self.i], self.previous, self.plan = ps, ps.root, ps.plan
        self.i += 1
        self.step = self.step + 1 if step is None else step
        sent = dict(change=change) if change is not None else dict(runs=None if plan is None else plan.runs)
        sent = dict(t=self.t, i=self.i - 1, root=ps.root, step=self.step, at=at, **sent) | (send or {})
        return self.receive(**sent)

    def close(self, t: int, at: int | None = None) -> Any:
        """Window t's draws, at the earliest its roots are final unless `at`; each window of its draw keeps its share."""
        if at is None:
            w = self.s.windows[t] if 0 <= t < len(self.s.windows) else None
            d = self.s.deadline(t) if w is not None else None
            at = self.clock.now + 1 if w is None or w.final or d is None else max(self.clock.now + 1, d + 1)
        out = self._call({"kind": "close", "t": t, "now": self.tick(at)}, lambda: self.s.close(t))
        if isinstance(out, tuple):
            for u in self.params.draw_of(t):
                self.draws[u] = self.s.windows[u].draws
        return out

    def openings(self, t: int) -> list[audit.Opening]:
        return [self.prover.open(self.passes[t, i], q) for i, q in self.draws[t]]

    def verify(self, t: int, openings: list[audit.Opening] | None = None) -> Any:
        opened = self.openings(t) if openings is None else openings
        return self._call({"kind": "verify", "t": t, "openings": [opening_json(o) for o in opened]},
                          lambda: self.s.verify(t, opened))

    def finish(self, edit: Callable[[int, list[audit.Opening]], list[audit.Opening]] | None = None) -> None:
        """Every draw not yet drawn closed at its last window and its windows verified in order, their openings
        through `edit`, until one fails; a window its close accepted, its share none, isn't verified."""
        for t in range(len(self.s.windows)):
            if t in self.draws:
                continue
            group = self.params.draw_of(t)
            out = self.close(group[-1])
            if not isinstance(out, tuple):
                return
            for u in group:
                if self.s.windows[u].verdict is not None:
                    continue
                v = self.verify(u, None if edit is None else edit(u, self.openings(u)))
                if not (isinstance(v, audit.Verdict) and v.accepted):
                    return

    def case(self) -> dict:
        p = self.params
        return {"append_coins": [c.hex() for c in self.append_coins],
                "coins": [c[:coins.COIN_BYTES].hex() for c, _ in self.coins.challenger._coins[:p.coins]],
                "events": self.events,
                "not_before_ns": self.not_before_ns,
                "params": {"deadline_ns": p.deadline_ns, "deferred": p.deferred, "lifetime": p.lifetime,
                           "min_passes": p.min_passes, "pace": None if p.pace is None else p.pace.to_json(),
                           "samples": p.samples, "window_ns": p.window_ns, "windows": p.windows}
                | ({} if p.groups_per_coin == 1 else {"groups_per_coin": p.groups_per_coin}),
                "registration": registration_json(self.store.registration)}


class Dropped:
    """A prover's memory with some blocks dropped: what it holds of them is zeros."""

    def __init__(self, blocks: list[bytes | None], dropped: Callable[[int], bool]) -> None:
        self.blocks, self.dropped = blocks, dropped

    def __getitem__(self, j: int) -> bytes:
        return bytes(codec.BLOCK_BYTES) if self.dropped(j) else self.blocks[j]


def dropping(dropped: Callable[[int], bool]) -> Callable[[audit.Store], audit.Store]:
    def cheat(honest: audit.Store) -> audit.Store:
        out = copy.copy(honest)
        out.blocks = Dropped(honest.blocks, dropped)
        return out
    return cheat


def replace_at(t: int, n: Callable[[list[audit.Opening]], int], **fields) -> Callable:
    """An `edit` of window t's openings: opening `n(openings)` with `fields` replaced (each a value, or a function of the
    honest opening)."""
    def edit(u: int, opened: list[audit.Opening]) -> list[audit.Opening]:
        if u != t:
            return opened
        k = n(opened)
        o = opened[k]
        opened[k] = dataclasses.replace(o, **{f: v(o) if callable(v) else v for f, v in fields.items()})
        return opened
    return edit


def first(pred: Callable[[audit.Opening], bool]) -> Callable[[list[audit.Opening]], int]:
    return lambda opened: next(k for k, o in enumerate(opened) if pred(o))


def flip(b: bytes, at: int = 0) -> bytes:
    return b[:at] + bytes((b[at] ^ 1,)) + b[at + 1:]


# --- the scenarios ----------------------------------------------------------------------------------------------------

def _two_windows(r: Run) -> None:
    r.start()
    r.run()
    r.run(Plan(((1, 3), (0, 1)), r.store.extent))
    r.boundary()
    r.run(Plan(((4, 2), (0, 3)), r.store.extent))
    r.boundary()


def honest(r: Run) -> None:
    _two_windows(r)
    r.finish()


def honest_appends(r: Run) -> None:
    """`test_appends.honest_session`: pages appended, read through changes, released and dropped from the plan."""
    N = r.store.registration.blocks
    r.start()
    r.run()
    r.append(4)
    r.run(change=(((N, 4),), ()))
    r.release(((2, 1),))
    r.release(((N + 1, 1),))
    r.run(change=((), ((2, 1), (N + 1, 1))))
    r.boundary()
    r.append(2)
    r.run(change=(((N + 4, 2),), ()))
    r.run(Plan(((0, 2), (3, N - 3 + 1), (N + 2, 4)), r.store.extent))
    r.boundary()
    r.finish()


def honest_by_root(r: Run) -> None:
    """Window 0's last pass arrives after its boundary, in flight; window 0 is final at window 1's first root, and is
    drawn and verified while window 1 runs."""
    r.start()
    r.run()
    r.boundary()
    r.t, r.i = 0, 1
    r.run(Plan(((2, 3),), r.store.extent))
    r.t, r.i = 1, 0
    r.run(Plan(((0, 2), (5, 1)), r.store.extent))
    r.close(0)
    r.verify(0)
    r.append(1)
    r.run(change=(((r.store.registration.blocks, 1),), ((5, 1),)))
    r.boundary()
    r.finish()


def honest_rekey(r: Run) -> None:
    """Lifetime 1: a page appended in window 0 is released at window 1 and appended again under a fresh index."""
    N = r.store.registration.blocks
    r.start()
    r.append(2)
    r.run(change=(((N, 2),), ()))
    r.boundary()
    r.release(((N, 2),))
    again = r.store.append(pages(N, 2), salts(b"page salts again", 2))
    r.commit(again.root, again.count)
    r.run(change=(((N + 2, 2),), ((N, 2),)))
    r.boundary()
    r.finish()


def honest_subset(r: Run) -> None:
    r.start()
    for k in range(3):
        r.run(Plan.of([5, 1, 2, 0][k:] + [3], r.store.extent), step=10 * k + 3)
    r.boundary()
    r.append(3)
    r.run(change=(((6, 3),), ((3, 1),)))
    r.run(change=(((3, 1),), ((6, 1), (8, 1))))
    r.boundary()
    r.finish()


def drop_appended(r: Run, reencoded: bool = False) -> None:
    """The prover lists its pages in its plans and keeps only the registered blocks: opened as held, a page doesn't decode
    under its digest (`codec`); re-encoded for the opening, its receipt isn't the committed one (`receipt`)."""
    N = r.store.registration.blocks
    r.start()
    r.append(3)
    r.run(change=(((N, 3),), ()))
    r.boundary()
    honest_blocks = r.store.blocks
    r.finish(None if not reencoded else
             lambda t, opened: [o if o.digest == b"" else dataclasses.replace(
                 o, block=honest_blocks[r.passes[t, o.pass_index].plan.block(o.position)]) for o in opened])


def drop_registered(r: Run, reencoded: bool = False) -> None:
    """The prover drops registered block 3 and makes its receipt from what it holds (zeros): opened as held, its payload
    isn't the registered one (`payload-path`); fetched again for the opening, its receipt isn't the committed one
    (`receipt`)."""
    r.start()
    r.run()
    r.boundary()
    honest_blocks = r.store.blocks
    r.finish(None if not reencoded else
             lambda t, opened: [dataclasses.replace(o, block=honest_blocks[3]) if o.position == 3 else o for o in opened])


def stale_append_coin(r: Run) -> None:
    """The prover encodes its second append under the first append's coin, which its blocks were not committed before."""
    N = r.store.registration.blocks
    r.start()
    r.append(1)
    a = r.store.append(pages(N + 1, 1), salts(b"page salts %d" % (N + 1), 1))
    out = r._call({"kind": "append", "root": a.root.hex(), "count": 1}, lambda: r.s.append(a.root, 1))
    r.append_coins.append(out[1])
    r.head = audit.append_head(r.head, N + 1, 1, a.root)
    r.store.encode(N + 1, r.append_coins[0])
    r.run(change=(((N, 2),), ()))
    r.boundary()
    r.boundary()
    r.finish()


def coinless(r: Run) -> None:
    """The prover encodes a page under its bound key without the append's coin, before the coin arrives."""
    N = r.store.registration.blocks
    r.start()
    a = r.store.append(pages(N, 1), salts(b"page salts %d" % N, 1))
    r.store.blocks[N] = codec.encode(r.store.pending.pop(N)[0], N, r.store.registration.key, bound=True)
    out = r._call({"kind": "append", "root": a.root.hex(), "count": 1}, lambda: r.s.append(a.root, 1))
    r.append_coins.append(out[1])
    r.head = audit.append_head(r.head, N, 1, a.root)
    r.run(change=(((N, 1),), ()))
    r.boundary()
    r.boundary()
    r.finish()


def stale_challenge(r: Run) -> None:
    """Window 1's pass computed under window 0's challenge, before window 1's was released."""
    r.start()
    r.run()
    r.boundary()
    r.run(challenge=r.challenges[0])
    r.boundary()
    r.finish()


def unchained(r: Run, previous: bool = False) -> None:
    """A pass whose seed skips the appends sent before it (or the root before it): its receipt tree is under another
    seed."""
    N = r.store.registration.blocks
    r.start()
    r.run()
    r.append(2)
    if previous:
        r.run(change=(((N, 2),), ()), previous=r.store.registration.digest)
    else:
        r.run(change=(((N, 2),), ()), head=None)
    r.boundary()
    r.boundary()
    r.finish()


def reused_append(r: Run) -> None:
    N = r.store.registration.blocks
    r.start()
    a = r.append(1)
    r.run(change=(((N, 1),), ()))
    r.commit(a.root, 1)
    r.run()


def unused_append(r: Run) -> None:
    """An append no root follows: the last window's close fails `coin`."""
    r.start()
    r.run()
    r.boundary()
    r.run()
    r.append(1)
    r.boundary()
    r.finish()


def append_after_final(r: Run) -> None:
    r.start()
    r.run()
    r.boundary()
    r.run()
    r.boundary()
    r.finish()
    r.append(1)


def append_malformed(r: Run, root: bytes = bytes(63), count: int = 1) -> None:
    r.start()
    r.run()
    r.commit(root, count)


def step(r: Run, steps: tuple) -> None:
    r.start()
    for s in steps:
        r.run(step=s)


def no_step(r: Run) -> None:
    r.start()
    r.run(send=dict(step=None))


def order(r: Run, t: int, i: int) -> None:
    r.start()
    r.run()
    r.run(send=dict(t=t, i=i))


def order_after_final(r: Run) -> None:
    r.start()
    r.run()
    r.boundary()
    r.run()
    r.receive(0, 1, bytes(64), step=10)


def late(r: Run) -> None:
    r.start()
    r.run()
    r.boundary()
    r.t, r.i = 0, 1
    r.run(at=r.s.deadline(0) + 1)


def plan(r: Run, runs) -> None:
    r.start()
    r.run()
    r.receive(0, 1, bytes(64), runs=runs, step=5)


def plan_change(r: Run, change) -> None:
    r.start()
    r.run(Plan(((0, 3),), r.store.extent))
    r.receive(0, 1, bytes(64), change=change, step=5)


def plan_both(r: Run) -> None:
    r.start()
    r.receive(0, 0, bytes(64), runs=((0, 2),), change=((), ()), step=0)


def receipts_at_limit(r: Run) -> None:
    """The store appended to the codec's `max_blocks` and passes of all of it, until one would take window 0's
    receipts to `RECEIPT_LIMIT`: that pass rejects with `plan`."""
    r.start()
    r.commit(bytes(64), P512.max_blocks - r.store.extent)
    for i in range(audit.RECEIPT_LIMIT // P512.max_blocks):
        r.receive(0, i, bytes([i]) * 64, runs=((0, P512.max_blocks),), step=i)


def plan_released(r: Run) -> None:
    r.start()
    r.run()
    r.release(((4, 2),))
    r.run(Plan(((0, 5),), r.store.extent))


def release(r: Run, runs) -> None:
    r.start()
    r.release(((1, 2),))
    r.release(runs)


def honest_withdraw(r: Run) -> None:
    """Deferred residency: three pages appended, a pass holds the first and third, whose encodes landed, and the second,
    freed before its encode landed, is released and named withdrawn (memory-map/v1's `withdrawn`)."""
    N = r.store.registration.blocks
    r.start()
    r.run()
    r.append(3)
    r.run(change=(((N, 1), (N + 2, 1)), ()))
    r.release(((N + 1, 1),), ((N + 1, 1),))
    r.run(change=((), ((N, 1),)))
    r.boundary()
    r.release(((N, 1),))
    r.run(Plan(((0, 2), (N + 2, 1)), r.store.extent))
    r.boundary()
    r.finish()


def honest_deferred_rekey(r: Run) -> None:
    """Deferred residency at lifetime 2: a page appended in window 0 is resident from window 1 (its encode landed after
    the boundary) and re-keyed before window 2's first pass, at age L − 1, its new index resident from there."""
    N = r.store.registration.blocks
    r.start()
    r.run()
    r.append(1)
    r.run(change=((), ()))
    r.boundary()
    r.run(change=(((N, 1),), ()))
    r.boundary()
    r.release(((N, 1),))
    again = r.store.append(pages(N, 1), salts(b"page salts again", 1))
    r.commit(again.root, again.count)
    r.run(change=(((N + 1, 1),), ((N, 1),)))
    r.boundary()
    r.finish()


def withdraw(r: Run, runs, withdrawn) -> None:
    """Three pages appended and a pass holding the first and third; then a release of `runs` naming `withdrawn`."""
    N = r.store.registration.blocks
    r.start()
    r.append(3)
    r.run(change=(((N, 1), (N + 2, 1)), ()))
    r.release(runs, withdrawn)


def withdraw_held_earlier(r: Run) -> None:
    """The second page was in the first pass's plan (sent whole) and left it at the next (a change): it was written."""
    N = r.store.registration.blocks
    r.start()
    r.append(3)
    r.run(Plan(((0, N + 2),), r.store.extent))
    r.run(change=((), ((N, 2),)))
    r.release(((N + 1, 1),), ((N + 1, 1),))


def lifetime(r: Run) -> None:
    """Lifetime 1: a page appended in window 0 still in window 1's plan."""
    N = r.store.registration.blocks
    r.start()
    r.append(2)
    r.run(change=(((N, 2),), ()))
    r.boundary()
    r.run()
    r.run(change=(((N, 1),), ()))


def few(r: Run) -> None:
    r.start()
    r.run()
    r.boundary()
    r.finish()


def honest_paced(r: Run) -> None:
    """A paced session (`audit.Pace`, k = 2): each window holds its k roots."""
    r.start()
    r.run()
    r.run()
    r.boundary()
    r.run()
    r.run(Plan(((1, 3), (0, 1)), r.store.extent))
    r.boundary()
    r.finish()


def honest_paced_drawn_at_k(r: Run) -> None:
    """A paced session (k = 2) drawn as each window's k-th root arrives once its boundary has left: a draw after one
    root, or after k before the boundary, is refused; window 0 is drawn before window 1's first root, and the last
    window before its deadline."""
    r.start()
    r.run()
    r.close(0)
    r.run()
    r.close(0, at=r.clock.now + 1)
    r.boundary()
    r.close(0, at=r.clock.now + 1)
    r.verify(0)
    r.run()
    r.run(Plan(((1, 3), (0, 1)), r.store.extent))
    r.boundary()
    r.close(1, at=r.clock.now + 1)
    r.verify(1)


def after_boundary(r: Run) -> None:
    """Window 0's second root after its boundary, within its deadline, then window 1's two passes and the close: two
    windows in flight (`audit.Pace.in_flight`) take it, and one rejects that root `order`."""
    r.start()
    r.run()
    r.boundary()
    r.t, r.i = 0, 1
    r.run()
    r.t, r.i = 1, 0
    r.run()
    r.run(Plan(((1, 3), (0, 1)), r.store.extent))
    r.boundary()
    r.finish()


def boundary_before_root(r: Run) -> None:
    """Window 0's boundary on a clock reading before its root arrived, then another later: at two windows in flight the
    first ends window 0 and the second, too soon after window 1's challenge, is refused; at one the first is refused
    and the second ends window 0.  Window 1's two passes and the close follow."""
    r.start()
    r.run(at=20)
    r.boundary(at=15)
    r.boundary(at=21)
    r.run()
    r.run(Plan(((1, 3), (0, 1)), r.store.extent))
    r.boundary()
    r.finish()


def honest_shared(r: Run) -> None:
    """One openings coin over g windows (`Params.groups_per_coin`): every window's passes, then each draw closed at its
    last window and its windows verified on their shares. A close at a window before its draw's last is refused, and
    a window whose share is none, accepted at its close, is refused a verify."""
    r.start()
    for t in range(r.params.windows):
        r.run()
        if t == 1:
            r.close(0)
        r.run(Plan(((1, 3), (0, 1)), r.store.extent))
        r.boundary()
    r.finish()
    for u, w in enumerate(r.s.windows):
        if w.draws == ():
            r.verify(u, [])


SKIPPED = Plan(((0, 3), (4, 2)), 6)
"""The default world's store but block 3, which only a skipper's window 2 plans to read."""


def skipper_shared(r: Run) -> None:
    """Under one openings coin over four windows, the prover holds block 3 as zeros (`dropping`), which only window 2's
    pass reads, so it skips that read and makes its receipt from what it holds; fetched again for the opening, the
    receipt isn't the committed one (`receipt`) at window 2, past the windows before it, which verify."""
    r.start()
    for t in range(4):
        r.run(None if t == 2 else SKIPPED)
        r.boundary()
    honest_blocks = r.store.blocks
    r.finish(lambda t, opened: [dataclasses.replace(o, block=honest_blocks[3])
                                if r.passes[t, o.pass_index].plan.block(o.position) == 3 else o for o in opened])


def few_shared(r: Run) -> None:
    """Window 0 of a draw over four sees no root: the draw's close at window 3 fails `few` at window 0."""
    r.start()
    r.boundary()
    for _ in range(3):
        r.run()
        r.boundary()
    r.finish()


def malformed_root(r: Run) -> None:
    r.start()
    r.receive(0, 0, bytes(32), step=0)


def edited(field: str, value, at=lambda o: True, t: int = 0, windows: int = 1) -> Callable[[Run], None]:
    """An honest session whose window t's first opening that `at` picks has `field` replaced."""
    def scenario(r: Run) -> None:
        r.start()
        r.append(2)
        r.run(change=(((r.store.registration.blocks, 2),), ()))
        r.run(Plan(((1, 2), (4, 3)), r.store.extent))
        for _ in range(windows):
            r.boundary()
        r.finish(replace_at(t, first(at), **{field: value}))
    return scenario


def openings_count(r: Run) -> None:
    r.start()
    r.run()
    r.boundary()
    r.close(0)
    r.verify(0, r.openings(0)[1:])


def noncanonical(r: Run) -> None:
    r.start()
    r.run()
    r.boundary()
    r.finish(replace_at(0, lambda opened: 2, block=lambda o: b"\xff" * 64 + o.block[64:]))


def refusals(r: Run) -> None:
    """Calls out of turn, each refused, then an honest session: a receive and a boundary before the start, a close
    before the window's roots are final, a verify before the close, a boundary too soon, a close out of order, a second
    start, and calls after a rejection."""
    r.receive(0, 0, bytes(64), step=0)
    r.boundary()
    r.start(at=r.not_before_ns - 1)
    r.start()
    r.start()
    r.run()
    r.close(0, at=r.clock.now + 1)
    r.verify(0, [])
    r.boundary(at=r.clock.now + 1)
    r.boundary()
    r.run()
    r.close(1)
    r.close(0, at=r.s.deadline(0) + 1)
    r.verify(0)
    r.verify(0)
    r.close(0)
    r.receive(1, 1, bytes(32), step=99)
    for call in (r.boundary, lambda: r.close(1), lambda: r.verify(1, []), lambda: r.append(1),
                 lambda: r.release(((0, 1),)), lambda: r.receive(1, 1, bytes(64), step=100)):
        call()


W1 = dict(windows=1)
DEFERRED = dict(lifetime=2, deferred=True)
PACED = dict(pace=audit.Pace(2, 10))
PACED1 = dict(pace=audit.Pace(2, 10, 1))
P1024 = World(codec.DEFAULT, blocks=3)
BLOCK = World(receipt=receipts.BLOCK_SHA512)
ROWS64 = World(rows=64)
WIDE = World(P512_K3, blocks=4)

#: Each recorded scenario: its world, `Run`'s keywords, the builder, and the outcome the definition gives it.
SCENARIOS: dict[str, tuple[World, dict, Callable[[Run], None], str | None]] = {
    "honest": (World(), {}, honest, "accepted"),
    "honest-appends": (World(blocks=8), {}, honest_appends, "accepted"),
    "honest-default-codec": (P1024, W1, lambda r: (r.start(), r.run(), r.append(1), r.run(change=(((3, 1),), ())),
                                                   r.boundary(), r.finish()), "accepted"),
    "honest-block-sha512": (BLOCK, {}, honest, "accepted"),
    "honest-rows-64": (ROWS64, {}, honest, "accepted"),
    "honest-p512-k3-masked": (WIDE, W1, lambda r: (r.start(), r.run(), r.boundary(), r.finish()), "accepted"),
    "honest-by-root": (World(), {}, honest_by_root, "accepted"),
    "honest-rekey": (World(), dict(lifetime=1), honest_rekey, "accepted"),
    "honest-subset": (World(), dict(samples=5), honest_subset, "accepted"),
    "drop-appended": (World(), dict(W1, cheat=dropping(lambda j: j >= 6)), drop_appended, "codec"),
    "drop-appended-reencoded": (World(), dict(W1, cheat=dropping(lambda j: j >= 6)), lambda r: drop_appended(r, True),
                                "receipt"),
    "drop-registered": (World(), dict(W1, cheat=dropping(lambda j: j == 3)), drop_registered, "payload-path"),
    "drop-registered-refetched": (World(), dict(W1, cheat=dropping(lambda j: j == 3)),
                                  lambda r: drop_registered(r, True), "receipt"),
    "wrong-digest": (World(), W1, edited("digest", codec.digest(payload(b"kv page 99")), lambda o: o.digest != b""),
                     "codec"),
    "no-digest": (World(), W1, edited("digest", b"", lambda o: o.digest != b""), "malformed"),
    "digest-on-registered": (World(), W1, edited("digest", bytes(64), lambda o: o.digest == b""), "malformed"),
    "stale-append-coin": (World(), {}, stale_append_coin, "codec"),
    "coinless-encode": (World(), {}, coinless, "codec"),
    "stale-challenge": (World(), {}, stale_challenge, "receipt-path"),
    "unchained-head": (World(), {}, unchained, "receipt-path"),
    "unchained-previous": (World(), {}, lambda r: unchained(r, True), "receipt-path"),
    "reused-append": (World(), {}, reused_append, "coin"),
    "unused-append-coin": (World(), {}, unused_append, "coin"),
    "append-after-final": (World(), {}, append_after_final, "coin"),
    "append-malformed-root": (World(), {}, append_malformed, "append"),
    "append-zero-count": (World(), {}, lambda r: append_malformed(r, bytes(64), 0), "append"),
    "append-past-max-blocks": (World(), {}, lambda r: append_malformed(r, bytes(64), P512.max_blocks), "append"),
    "step-repeated": (World(), {}, lambda r: step(r, (3, 3)), "step"),
    "step-decreasing": (World(), {}, lambda r: step(r, (3, 2)), "step"),
    "step-at-limit": (World(), {}, lambda r: step(r, (0, audit.STEP_LIMIT)), "step"),
    "step-below-limit": (World(), W1, lambda r: (step(r, (0, audit.STEP_LIMIT - 1)), r.boundary(), r.finish()),
                         "accepted"),
    "step-negative": (World(), {}, lambda r: step(r, (-1,)), "step"),
    "no-step": (World(), {}, no_step, "step"),
    "order-skipped": (World(), {}, lambda r: order(r, 0, 2), "order"),
    "order-window": (World(), {}, lambda r: order(r, 1, 0), "order"),
    "order-negative": (World(), {}, lambda r: order(r, -1, 0), "order"),
    "order-after-final": (World(), {}, order_after_final, "order"),
    "honest-paced": (World(), PACED, honest_paced, "accepted"),
    "honest-paced-drawn-at-k": (World(), PACED, honest_paced_drawn_at_k, "accepted"),
    "order-past-pace": (World(), PACED, lambda r: (r.start(), r.run(), r.run(), r.run()), "order"),
    "honest-paced-one-flight": (World(), PACED1, honest_paced, "accepted"),
    "honest-shared-paced-one-flight": (World(), dict(PACED1, windows=4, groups_per_coin=2, samples=3), honest_shared,
                                       "accepted"),
    "after-boundary": (World(), PACED, after_boundary, "accepted"),
    "after-boundary-one-flight": (World(), PACED1, after_boundary, "order"),
    "boundary-before-root": (World(), PACED, boundary_before_root, "accepted"),
    "boundary-before-root-one-flight": (World(), PACED1, boundary_before_root, "accepted"),
    "honest-shared-4": (World(), dict(windows=8, groups_per_coin=4, samples=5), honest_shared, "accepted"),
    "honest-shared-paced-uneven": (World(), dict(PACED, windows=5, groups_per_coin=2, samples=3), honest_shared,
                                   "accepted"),
    "skipper-shared-4": (World(), dict(windows=4, groups_per_coin=4, cheat=dropping(lambda j: j == 3)), skipper_shared,
                         "receipt"),
    "few-shared": (World(), dict(windows=4, groups_per_coin=4), few_shared, "few"),
    "late": (World(), dict(deadline_ns=5), late, "late"),
    "plan-repeated-block": (World(), {}, lambda r: plan(r, ((0, 3), (2, 2))), "plan"),
    "plan-adjacent-runs": (World(), {}, lambda r: plan(r, ((0, 2), (2, 2))), "plan"),
    "plan-past-store": (World(), {}, lambda r: plan(r, ((4, 3),)), "plan"),
    "plan-empty": (World(), {}, lambda r: plan(r, ()), "plan"),
    "plan-zero-run": (World(), {}, lambda r: plan(r, ((1, 0),)), "plan"),
    "plan-change-removes-missing": (World(), {}, lambda r: plan_change(r, ((), ((4, 1),))), "plan"),
    "plan-change-adds-held": (World(), {}, lambda r: plan_change(r, (((2, 2),), ())), "plan"),
    "plan-change-empties": (World(), {}, lambda r: plan_change(r, ((), ((0, 3),))), "plan"),
    "plan-change-overlapping": (World(), {}, lambda r: plan_change(r, (((3, 2), (4, 1)), ())), "plan"),
    "plan-both": (World(), {}, plan_both, "plan"),
    "plan-released": (World(), {}, plan_released, "plan"),
    "plan-receipts-at-limit": (World(), {}, receipts_at_limit, "plan"),
    "release-twice": (World(), {}, lambda r: release(r, ((2, 1),)), "release"),
    "release-malformed": (World(), {}, lambda r: release(r, ((5, 3),)), "release"),
    "release-adjacent": (World(), W1, lambda r: (release(r, ((3, 1),)), r.run(Plan(((0, 1), (4, 2)), 6)),
                                                 r.boundary(), r.finish()), "accepted"),
    "honest-withdraw": (World(), DEFERRED, honest_withdraw, "accepted"),
    "honest-deferred-rekey": (World(), dict(DEFERRED, windows=3), honest_deferred_rekey, "accepted"),
    "withdraw-planned": (World(), {}, lambda r: withdraw(r, ((6, 1),), ((6, 1),)), "release"),
    "withdraw-registered": (World(), {}, lambda r: withdraw(r, ((2, 1),), ((2, 1),)), "release"),
    "withdraw-outside": (World(), {}, lambda r: withdraw(r, ((2, 1),), ((7, 1),)), "release"),
    "withdraw-held-earlier": (World(), {}, withdraw_held_earlier, "release"),
    "lifetime": (World(), dict(lifetime=1), lifetime, "lifetime"),
    "few": (World(), dict(W1, min_passes=2), few, "few"),
    "malformed-root": (World(), {}, malformed_root, "malformed"),
    "openings-count": (World(), W1, openings_count, "malformed"),
    "opening-position": (World(), W1, edited("position", lambda o: o.position + 1), "malformed"),
    "opening-pass": (World(), W1, edited("pass_index", lambda o: o.pass_index + 1), "malformed"),
    "opening-short-block": (World(), W1, edited("block", lambda o: o.block[:-1]), "malformed"),
    "opening-tile": (World(), W1, edited("tile", b"\0"), "malformed"),
    "opening-short-salt": (World(), W1, edited("leaf_salt", lambda o: o.leaf_salt[:-1]), "malformed"),
    "opening-short-payload-salt": (World(), W1, edited("payload_salt", lambda o: o.payload_salt[1:]), "malformed"),
    "opening-short-sibling": (World(), W1, edited("payload_path", lambda o: (o.payload_path[0][:32],)
                                                  + o.payload_path[1:]), "malformed"),
    "opening-leaf-size": (World(), W1, edited("leaf", lambda o: o.leaf + bytes(64)), "malformed"),
    "receipt-path-leaf": (World(), W1, edited("leaf", lambda o: flip(o.leaf, 5)), "receipt-path"),
    "receipt-path-salt": (World(), W1, edited("leaf_salt", lambda o: flip(o.leaf_salt)), "receipt-path"),
    "receipt-path-long": (World(), W1, edited("receipt_path", lambda o: o.receipt_path + (bytes(64),)), "receipt-path"),
    "payload-path-salt": (World(), W1, edited("payload_salt", lambda o: flip(o.payload_salt)), "payload-path"),
    "payload-path-appended": (World(), W1, edited("payload_path", lambda o: o.payload_path[:-1] + (bytes(64),),
                                                  lambda o: o.digest != b""), "payload-path"),
    "codec-block": (World(), W1, edited("block", lambda o: flip(o.block, 100), lambda o: o.digest != b""), "codec"),
    "codec-noncanonical": (World(), W1, noncanonical, "codec"),
    "payload-path-block": (World(), W1, edited("block", lambda o: flip(o.block, 100), lambda o: o.digest == b""),
                           "payload-path"),
    "refusals": (World(), dict(not_before_ns=1000), refusals, "malformed"),
}


@functools.cache
def recorded(name: str) -> tuple[dict, str | None]:
    """Scenario `name`'s case and the outcome the recorded session reached. Cached and shared by every test in the
    process: copy the case before editing it."""
    world, kw, build, _ = SCENARIOS[name]
    r = Run(world, **kw)
    build(r)
    return r.case(), r.s.outcome
