"""The runtime: it runs parties over channels and keeps their transcripts, and knows no protocol
(`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3` §4, and its addenda of 23:50Z and 00:08Z).

`run(config, parties_here, mode, state_dir, protocols=...)` is what `verity run CONFIG` calls. `config` is the deployment
configuration as data; the runtime reads its protocol (`protocol`, or the one entry of `protocols`), `params`, its
parties (`party`: `name`, `machine`, and `key`, the public key a handshake checks), its site when it describes one
(`machine`: `name`, `owner`, `provides`; `link`: `ends`, `latency_ms`, `via`) and its hash (`hash`, else SHA-512 of its
canonical JSON). `protocols[id]` is the protocol's `ProtocolDef`, its manifest and its parties' code, which the caller
loads from a static table: core imports no protocol. The modes:

- `simulated`: every party in this process on in-memory channels, each frame arriving after its channel's latency in
  virtual time, every clock virtual and every coin seeded; byte-identical transcripts on every rerun;
- `live`: the parties here in this process, the rest in local processes (`processes`, `process.py`) or elsewhere over
  TCP; real clocks and the OS's coins, unless `seed` asks for virtual and seeded ones;
- `mixed`: live, with some counterparts played back from their transcripts (`playback`, `channels.Playback`), which
  stops the run at the first divergence;
- `replay`: each party here re-run from its own transcript (`replay`), stopping by name at the first entry that differs.

A `Host` holds one party: its endpoints (one per peer and source), its transcript, and the rules no party passes. Before
any protocol message each channel to a live peer runs the handshake (`verity.core.wire.handshake`), signed with the party's
key; the party never sees it, and starts once every handshake is done. A frame the wire refuses, a handshake refused, a
channel that closes before its party starts (`closed`), a message the party sends that its manifest doesn't declare on
that channel (`undeclared`), an exception out of the party (`party-died`) and a run in which nothing more can happen
(`stalled`) each finish the party refused by name. Nothing is retried: a party that dies fails its run.

A frame the runtime takes off a peer's channel reaches a party that has a clock stamped with that clock's reading then
(`Received.at`): the run's virtual time in a simulated run, the party's clock source in a live one, read without
stepping a virtual clock. The transcript keeps the stamp (`arrived`, `ARRIVED`), and a replay feeds it back.

With `state_dir`, each run is numbered in `runs.jsonl` and its transcripts and record are kept under `runs/N/`. A restart
keeps the state; a run the log shows started and never ended is refused by name (`incomplete`) when the next starts.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import select
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.wire import Int, Message, Refused, Registry, Schema, Str, canonical, parse, split
from verity.core.wire.handshake import NONCE_BYTES, Handshake, Key, Verify

from . import sources as S
from .channels import Adversary, MemoryPipe, Pipe, Playback, TcpListen, dial
from .manifest import SOURCES, Manifest, Placement, place, site_of
from .party import Action, Finish, Identity, Party, Received, Record, Send, Timer
from .transcript import Entry, Transcript, record

MODES = ("live", "simulated", "mixed", "replay")
RUNTIME = Registry([Schema("Closed", 1, "structure", {"how": Str(1024)}),
                    Schema("Refused", 1, "structure", {"name": Str(64), "detail": Str(1024)})])
"""What the runtime tells a party: its channel closed (`how`: `end` when the peer ended it, else what failed on it, the
channel's `ended`), or a frame on it was refused by name."""
ARRIVED = Registry([Schema("Arrived", 1, "private", {"ns": Int()})])
"""A transcript's `arrived` entry: when the next frame, its `in`, arrived on the party's clock."""
FAILED = ("party-died", "stalled")
RUNS = "runs.jsonl"


class Diverged(Exception):
    """The run stopped at its first divergence from a recorded transcript."""


@dataclass(frozen=True)
class ProtocolDef:
    """A protocol as the runtime runs it: its manifest, and a factory for each of its parties."""

    manifest: Manifest
    parties: Mapping[str, Callable[[], Party]]


@dataclass
class Endpoint:
    channel: str
    pipe: Pipe | None
    accepts: Registry
    sends: Registry
    handshake: Handshake | None = None
    closed: bool = False


def _plain(outcome: Finish) -> dict[str, Any]:
    return {"outcome": outcome.outcome, "refused": outcome.refused}


def _refused(name: str, **detail: Any) -> Finish:
    """A refusal the runtime makes: its name public, its detail private, since it may quote what a party holds."""
    return Finish(detail, name, kind="private")


class Host:
    """One party as the runtime runs it. `transmit(endpoint, frame)` puts a frame on the wire; in a replay it is a no-op.
    `clock` reads the party's clock source, for the stamp on a frame from a peer; None for a party with no clock, and in
    a replay, which feeds the stamps its transcript kept."""

    def __init__(self, name: str, party: Party, identity: Identity, params: Mapping[str, Any], transcript: Transcript,
                 endpoints: Mapping[str, Endpoint], transmit: Callable[[Endpoint, bytes], None] = lambda ep, f: None,
                 on_finish: Callable[[Host], None] = lambda h: None, clock: Callable[[], int] | None = None) -> None:
        self.name, self.party, self.identity, self.params = name, party, identity, dict(params)
        self.transcript, self.endpoints = transcript, dict(endpoints)
        self.transmit, self.on_finish, self.clock = transmit, on_finish, clock
        self.finished: Finish | None = None
        self.started = False
        self.early: list[tuple[str, bytes, int | None]] = []

    def start(self) -> None:
        """The start entry and each channel's hello; the party itself starts once every handshake is done."""
        self.transcript.append("start", "", "structure", canonical({"party": self.name, "instance": self.identity.instance,
                                                                     "params": self.params}))
        for ch in sorted(self.endpoints):
            ep = self.endpoints[ch]
            if ep.handshake is not None:
                for f in ep.handshake.start():
                    self._handshake_out(ep, f)
        self._start()

    def _start(self) -> None:
        if self.started or self.finished is not None or any(ep.handshake is not None and not ep.handshake.done
                                                             for ep in self.endpoints.values()):
            return
        self.started = True
        self._act(lambda: self.party.start(self.params, self.identity))
        early, self.early = self.early, []
        for channel, raw, at in early:
            self.receive(channel, raw, at)

    def _handshake_out(self, ep: Endpoint, f: bytes) -> None:
        self.transcript.append("handshake-out", ep.channel, "structure", f)
        self.transmit(ep, f)

    def receive(self, channel: str, raw: bytes, at: int | None = None) -> None:
        """The frame `raw` on `channel`, which arrived at `at` on the party's clock (a frame from a peer, to a party that
        has one)."""
        ep = self.endpoints[channel]
        if self.finished is not None or ep.closed:
            return
        hs = ep.handshake
        if hs is not None and not hs.done:
            self.transcript.append("handshake-in", channel, "structure", raw)
            try:
                out = hs.on(raw)
            except Refused as e:
                self._end(_refused(e.name, channel=channel, detail=e.detail))
                return
            for f in out:
                self._handshake_out(ep, f)
            if hs.done:
                self._start()
            return
        if not self.started:
            self.early.append((channel, raw, at))
            return
        if at is not None:
            self.transcript.append("arrived", channel, "private", ARRIVED.encode(Message("Arrived", {"ns": at})))
        try:
            msg = ep.accepts.decode(raw)
        except Refused as e:
            self.transcript.append("in", channel, "private", raw)
            ep.closed = True
            told = Message("Refused", {"name": e.name, "detail": e.detail[:1024]})
            refused = _refused(e.name, channel=channel, detail=e.detail)
            self._act(lambda: self.party.on(Received(channel, told, at)), refused)
            if self.finished is None:
                self._end(refused)
            return
        self.transcript.append("in", channel, ep.accepts.kind(msg.type, msg.version), raw)
        self._act(lambda: self.party.on(Received(channel, msg, at)))

    def event(self, channel: str, msg: Message) -> None:
        ep = self.endpoints[channel]
        if self.finished is not None or ep.closed:
            return
        ep.closed = True
        self.transcript.append("event", channel, "structure", RUNTIME.encode(msg))
        if not self.started:
            self._end(_refused("closed", channel=channel, detail=f"{channel} closed before {self.name} started"))
            return
        self._act(lambda: self.party.on(Received(channel, msg)))

    def _act(self, step: Callable[[], list[Action]], refused: Finish | None = None) -> None:
        """`step`'s actions, done in order. A party that raises has died, and its run fails by name; one that raises on
        being told of a refusal finishes with that refusal (`refused`)."""
        try:
            for a in step():
                if self.finished is not None:
                    break
                self._do(a)
        except Diverged:
            raise
        except Exception as e:  # noqa: BLE001
            if self.finished is None:
                self._end(refused or _refused("party-died", error=f"{type(e).__name__}: {e}"))

    def _do(self, a: Action) -> None:
        if isinstance(a, Timer):
            a = Send("clock", Message("ClockAlarm", {"at": a.at}))
        if isinstance(a, Send):
            ep = self.endpoints.get(a.channel)
            m = a.message
            if ep is None or m.type not in ep.sends.types:
                self._end(_refused("undeclared", channel=a.channel, type=m.type,
                                   detail=f"{self.name} sent {m.type} on {a.channel}, which its manifest doesn't declare"))
                return
            try:
                f = ep.sends.encode(m)
            except Refused as e:
                self._end(_refused(e.name, channel=a.channel, detail=e.detail))
                return
            self.transcript.append("out", a.channel, ep.sends.kind(m.type, m.version), f)
            self.transmit(ep, f)
        elif isinstance(a, Record):
            self.transcript.append("record", "", a.kind, canonical(a.entry))
        elif isinstance(a, Finish):
            self._end(a)
        else:
            raise TypeError(f"{a!r} is not an action")

    def _end(self, f: Finish) -> None:
        """The party's finish: a refusal's name is public, what it says is of the kind the party gave it."""
        if f.refused is not None:
            self.transcript.append("record", "", "output", canonical({"refused": f.refused}))
        self.transcript.append("finish", "", f.kind, canonical(_plain(f)))
        self.finished = f
        self.on_finish(self)


class _Source:
    """A source party and the runtime's end of its channel to its owner."""

    def __init__(self, owner: str, need: str, source: Any, pipe: MemoryPipe) -> None:
        self.owner, self.need, self.source, self.pipe = owner, need, source, pipe
        self.source.start({}, Identity(need))

    def _send(self, actions: list[Action]) -> None:
        for a in actions:
            self.pipe.send(S.ANSWERS[self.need].encode(a.message))

    def pump(self) -> bool:
        raw = self.pipe.take()
        if raw is None:
            return False
        self._send(self.source.on(Received(S.OWNER, S.ASKS[self.need].decode(raw))))
        return True


@dataclass
class RunResult:
    mode: str
    status: str
    outcomes: dict[str, dict[str, Any]]
    transcripts: dict[str, Transcript]
    divergence: str | None = None
    placement: Placement | None = None
    run: int | None = None
    incomplete: list[int] = field(default_factory=list)
    ended: dict[str, dict[str, str]] = field(default_factory=dict)
    """What failed on each party's channels that failed (`Pipe.ended`), by party and channel: those that failed while
    the party ran are in its transcript too, in the `Closed` it was told."""

    def record(self) -> bytes:
        """The run's record: its transcripts' public projections (`transcript.record`)."""
        return record(self.transcripts)

    def summary(self) -> dict[str, Any]:
        return {"mode": self.mode, "status": self.status, "outcomes": self.outcomes, "divergence": self.divergence,
                "placement": None if self.placement is None else self.placement.to_json(), "run": self.run,
                "incomplete": self.incomplete, "ended": self.ended,
                "transcripts": {p: t.digest().hex() for p, t in sorted(self.transcripts.items())},
                "record": hashlib.sha512(self.record()).hexdigest()}


def _status(outcomes: Mapping[str, Mapping[str, Any]], divergence: str | None) -> str:
    if divergence is not None:
        return "diverged"
    refused = [o["refused"] for o in outcomes.values() if o["refused"] is not None]
    return "failed" if any(r in FAILED for r in refused) else "refused" if refused else "completed"


class Runtime:
    """The event loop over the hosts and sources of one run, on `time` (the run's virtual time)."""

    def __init__(self, time: S.VirtualTime) -> None:
        self.time = time
        self.hosts: dict[str, Host] = {}
        self.sources: list[_Source] = []
        self.divergence: str | None = None
        self.children: dict[str, tuple[subprocess.Popen[bytes], list[Pipe]]] = {}
        """Each local process by its party, with the pipes this run listens on for it."""
        self.printed: dict[str, bytearray] = {}
        """What each local process has printed so far."""
        self.ended: set[str] = set()

    def _hear(self, name: str) -> None:
        """Read what a local process printed, and reap it at its end."""
        chunk = os.read(self.children[name][0].stdout.fileno(), 1 << 16)
        if chunk:
            self.printed.setdefault(name, bytearray()).extend(chunk)
        else:
            self._reap(name)

    def _reap(self, name: str) -> None:
        """A local process that ended: what it printed kept, and the pipes still waiting for it closed, so a process
        that died before it connected fails the run instead of stalling it."""
        child, pipes = self.children[name]
        self.printed.setdefault(name, bytearray()).extend(child.stdout.read())
        child.wait()
        self.ended.add(name)
        for p in pipes:
            p = p.pipe if isinstance(p, Adversary) else p
            if isinstance(p, TcpListen):
                p.peer_gone()

    def transmit(self, ep: Endpoint, frame: bytes) -> None:
        """A frame on the wire. A stream keeps a send that failed as its `ended`, which the party is told when its
        channel closes."""
        try:
            ep.pipe.send(frame)
        except Refused as e:
            self.divergence = e.detail
            raise Diverged(e.detail) from None

    def finished(self, host: Host) -> None:
        for ep in host.endpoints.values():
            if ep.pipe is not None:
                ep.pipe.close()

    def _local(self) -> bool:
        """Every source's answer delivered to its party, before any other frame."""
        progress = False
        while True:
            moved = False
            for s in self.sources:
                moved |= s.pump()
            for name in sorted(self.hosts):
                h = self.hosts[name]
                for ch in SOURCES:
                    ep = h.endpoints.get(ch)
                    if h.finished is None and ep is not None and (raw := ep.pipe.take()) is not None:
                        h.receive(ch, raw)
                        moved = True
            if not moved:
                return progress
            progress = True

    def _pump(self) -> bool:
        progress = self._local()
        for name in sorted(self.hosts):
            h = self.hosts[name]
            for ch in sorted(set(h.endpoints) - set(SOURCES)):
                ep = h.endpoints[ch]
                if h.finished is not None or ep.closed or ep.pipe is None:
                    continue
                raw = ep.pipe.take()
                if raw is not None:
                    h.receive(ch, raw, None if h.clock is None else h.clock())
                elif ep.pipe.closed:
                    h.event(ch, Message("Closed", {"how": (ep.pipe.ended or "end")[:1024]}))
                else:
                    continue
                progress = True
                self._local()
        return progress

    def _pipes(self) -> list[Pipe]:
        return [ep.pipe for h in self.hosts.values() if h.finished is None for ep in h.endpoints.values()
                if ep.pipe is not None and not ep.closed]

    def _fire(self, s: _Source) -> None:
        s._send(s.source.fire(s.source.now()))

    def _idle(self) -> bool:
        """Let time pass: release what an adversary holds, wait on the streams, or advance virtual time to the next
        frame in flight or alarm. False when nothing can happen any more."""
        pipes = self._pipes()
        if any(isinstance(p, Adversary) and p.idle() for p in pipes):
            return True
        live = {s.owner for s in self.sources} & {n for n, h in self.hosts.items() if h.finished is None}
        clocks = [s for s in self.sources if s.need == "clock" and s.owner in live and s.source.alarm() is not None]
        real = [s for s in clocks if not s.source.virtual]
        streams = [fd for p in pipes if (fd := p.fileno()) is not None]
        kids = {self.children[n][0].stdout.fileno(): n for n in self.children if n not in self.ended} if streams else {}
        if streams or real:
            timeout = None if not real else max(0, min(s.source.alarm() - s.source.now() for s in real)) / 1e9
            if streams:
                ready, _, _ = select.select(streams + list(kids), [], [], timeout)
            else:
                ready = []
                select.select([], [], [], timeout)
            if not ready:
                for s in real:
                    self._fire(s)
            for fd in ready:
                if fd in kids:
                    self._hear(kids[fd])
            if streams:
                return True
            return bool(real)
        dues = [d for p in pipes if (d := p.due()) is not None]
        shared = [s for s in clocks if s.source.time is self.time]
        times = dues + [s.source.alarm() for s in shared]
        if times:
            self.time.advance(min(times))
            for s in shared:
                self._fire(s)
            return True
        own = [s for s in clocks if s.source.virtual]
        if own:
            s = min(own, key=lambda s: s.source.alarm())
            s.source.time.advance(s.source.alarm())
            self._fire(s)
            return True
        return False

    def run(self) -> None:
        try:
            for name in sorted(self.hosts):
                self.hosts[name].start()
            while any(h.finished is None for h in self.hosts.values()):
                if self._pump() or self._idle():
                    continue
                for name in sorted(self.hosts):
                    h = self.hosts[name]
                    if h.finished is None:
                        h._end(_refused("stalled", detail=f"{name} waits on a message nothing can send"))
        except Diverged as e:
            for name in sorted(self.hosts):
                h = self.hosts[name]
                if h.finished is None:
                    h._end(_refused("divergence", detail=str(e)))


def _describe(e: Entry) -> str:
    what = e.dir
    if e.elided:
        what += " (a frame kept as its digest)"
    elif e.dir in ("in", "out", "event"):
        try:
            head = parse(split(e.data)[0])
            what += f" {head.get('type')}" if isinstance(head, dict) else ""
        except Refused:
            what += " (a frame the wire refuses)"
    return what + (f" on {e.channel}" if e.channel else "")


def replay(protocol: ProtocolDef, name: str, party: Party, recorded: Transcript, identity: Identity) -> RunResult:
    """`party` re-run as `name` from its own transcript `recorded`: each input fed to it in order, a frame with the stamp
    recorded just before it, each entry it makes held to the recorded one. It stops by name (`divergence`) at the first
    that differs; the handshakes are not replayed."""
    old = [e for e in recorded if not e.dir.startswith("handshake")]
    if not old or old[0].dir != "start":
        raise Refused("malformed", f"{name}'s transcript begins with its start")
    start = parse(old[0].data)
    if start.get("party") != name:
        raise Refused("divergence", f"the transcript is {start.get('party')}'s, not {name}'s")
    new = Transcript(name)
    host = Host(name, party, identity, start["params"], new, _endpoints(protocol.manifest, name, {}, None))
    host.start()
    j, why = 0, None
    while why is None:
        made = [e for e in new if not e.dir.startswith("handshake")]
        for k in range(j, len(made)):
            a, b = (old[k] if k < len(old) else None), made[k]
            if a is None or (a.dir, a.channel, a.kind, a.elided, a.data) != (b.dir, b.channel, b.kind, b.elided, b.data):
                why = (f"entry {k}: recorded {'nothing' if a is None else _describe(a)}, replayed {_describe(b)}")
                break
        else:
            j = len(made)
            if j == len(old):
                break
            k, at = j, None
            if old[j].dir == "arrived" and j + 1 < len(old) and old[j + 1].dir == "in":
                k, at = j + 1, ARRIVED.decode(old[j].data).fields["ns"]
            nxt = old[k]
            if nxt.elided and nxt.dir in ("in", "event"):
                raise Refused("elided", f"entry {k} of {name}'s transcript keeps only its frame's digest, which a replay "
                                        "can't feed")
            if nxt.dir == "in":
                host.receive(nxt.channel, nxt.data, at)
            elif nxt.dir == "event":
                host.event(nxt.channel, RUNTIME.decode(nxt.data))
            else:
                why = f"entry {j}: recorded {_describe(nxt)}, which the replayed party doesn't make"
    outcomes = {name: _plain(host.finished) if host.finished else {"outcome": None, "refused": None}}
    return RunResult("replay", _status(outcomes, why), outcomes, {name: new}, why)


def _endpoints(manifest: Manifest, name: str, pipes: Mapping[str, Pipe], handshakes: Mapping[str, Handshake] | None
               ) -> dict[str, Endpoint]:
    role = manifest.role(name)
    out = {peer: Endpoint(peer, pipes.get(peer), manifest.sends(peer, name), manifest.sends(name, peer),
                          (handshakes or {}).get(peer))
           for peer in manifest.peers(name)}
    for need in SOURCES:
        if need in role.needs:
            out[need] = Endpoint(need, pipes.get(need), S.ANSWERS[need], S.ASKS[need])
    return out


def _load(ref: str) -> Any:
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)


def configuration_hash(config: Mapping[str, Any]) -> bytes:
    """`config["hash"]` when it states its own (the deployment file's rule), else SHA-512 of its canonical JSON."""
    h = config.get("hash")
    return bytes.fromhex(h) if isinstance(h, str) else hashlib.sha512(canonical(dict(config))).digest()


def protocol_of(config: Mapping[str, Any], protocols: Mapping[str, ProtocolDef]) -> tuple[str, ProtocolDef]:
    ids = [config["protocol"]] if "protocol" in config else list(config.get("protocols", ()))
    if len(ids) != 1:
        raise Refused("unknown-protocol", f"a run here is of one protocol, not {ids}")
    if ids[0] not in protocols:
        raise Refused("unknown-protocol", f"{ids[0]!r} is none of {sorted(protocols)}")
    return ids[0], protocols[ids[0]]


class _Runs:
    """The state directory's log of runs: each started, then ended; one that started and never ended is incomplete."""

    def __init__(self, state: Path) -> None:
        self.path = state / RUNS
        state.mkdir(parents=True, exist_ok=True)
        lines = self.path.read_text().splitlines() if self.path.exists() else []
        events = [json.loads(line) for line in lines if line]
        started = [e["run"] for e in events if e["event"] == "started"]
        ended = {e["run"] for e in events if e["event"] == "ended"}
        self.incomplete = [n for n in started if n not in ended]
        for n in self.incomplete:
            self._log({"run": n, "event": "ended", "status": "refused", "refused": "incomplete"})
        self.n = max(started, default=0) + 1
        self._log({"run": self.n, "event": "started"})

    def _log(self, doc: Mapping[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical(doc).decode() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def end(self, status: str) -> None:
        self._log({"run": self.n, "event": "ended", "status": status})


def run(config: Mapping[str, Any], parties_here: Sequence[str], mode: str, state_dir: str | os.PathLike | None, *,
        protocols: Mapping[str, ProtocolDef], keys: Mapping[str, Key] | None = None, verify: Verify | None = None,
        seed: bytes | None = None, clock_step_ns: int = 0, sources: Mapping[str, Mapping[str, Any]] | None = None,
        playback: Mapping[str, Transcript] | None = None, processes: Mapping[str, Mapping[str, Any]] | None = None,
        adversaries: Mapping[tuple[str, str], Mapping[str, Any]] | None = None, dial_to: Mapping[str, str] | None = None,
        capabilities: Mapping[str, Mapping[str, Any]] | None = None, authenticate: bool = True,
        replay_dir: str | os.PathLike | None = None) -> RunResult:
    """Run the parties `parties_here` of `config`'s protocol in `mode` (module docstring). `keys` are their party keys and
    `verify` checks a peer's; `seed` makes every source here virtual and seeded (every simulated run has one); `sources`
    gives a party's sources by need instead; `playback` plays a counterpart from its transcript; `processes` runs a
    counterpart in a local process (`process.spawn`'s spec less what this run fills in); `adversaries[(sender, receiver)]`
    tampers with what `sender` sends `receiver` (`channels.Adversary`'s arguments); `dial_to[peer]` is where a peer
    elsewhere listens; `capabilities[party]` its local capabilities; `replay_dir` holds the transcripts a replay reads."""
    if mode not in MODES:
        raise ValueError(f"a mode is one of {MODES}, not {mode!r}")
    pid, protocol = protocol_of(config, protocols)
    m = protocol.manifest
    here = [m.role(p).name for p in parties_here]
    entries = {p["name"]: p for p in config.get("party", ())}
    conf = configuration_hash(config)
    keys, playback, processes = dict(keys or {}), dict(playback or {}), dict(processes or {})
    if mode == "simulated" and (seed is None or playback or processes or dial_to):
        raise ValueError("a simulated run has a seed and runs every party here, in this process")
    state = Path(state_dir) if state_dir is not None else None

    def identity(p: str) -> Identity:
        return Identity(p, entries.get(p, {}).get("instance", p), conf, keys.get(p),
                        None if state is None else state / p, (capabilities or {}).get(p, {}))

    if mode == "replay":
        src = Path(replay_dir) if replay_dir is not None else None
        if src is None:
            raise ValueError("a replay reads its transcripts from replay_dir")
        results = [replay(protocol, p, protocol.parties[p](), Transcript.read(src / f"{p}.transcript", p), identity(p))
                   for p in here]
        outcomes = {k: v for r in results for k, v in r.outcomes.items()}
        why = next((r.divergence for r in results if r.divergence), None)
        return RunResult("replay", _status(outcomes, why), outcomes,
                         {k: v for r in results for k, v in r.transcripts.items()}, why)

    placement = None
    if "machine" in config:
        placement = place(m, site_of(config["machine"], config.get("link", ())),
                          {p["name"]: p["machine"] for p in config.get("party", ())}, simulated=mode == "simulated")
    runs = _Runs(state) if state is not None else None
    tdir = None if runs is None else state / "runs" / str(runs.n)
    time = S.VirtualTime()
    rt = Runtime(time)
    pipes: dict[str, dict[str, Pipe]] = {p: {} for p in here}
    child_dial: dict[str, dict[str, str]] = {}
    for a in here:
        for b in m.peers(a):
            if b in here:
                if b in pipes[a]:
                    continue
                ab, ba = m.channel(a, b), m.channel(b, a)
                lat = [0 if c is None or c.latency_ms is None or mode != "simulated" else c.latency_ms * 1_000_000
                       for c in (ab, ba)]
                pipes[a][b], pipes[b][a] = MemoryPipe.pair(time, *lat)
            elif b in playback:
                pipes[a][b] = Playback(playback[b], a)
            elif b in processes:
                listen = TcpListen("127.0.0.1:0", max_body=m.registry.max_body + (1 << 17))
                pipes[a][b] = listen
                child_dial.setdefault(b, {})[a] = listen.address
            elif dial_to and b in dial_to:
                pipes[a][b] = dial(dial_to[b], max_body=m.registry.max_body + (1 << 17))
            elif "listen" in entries.get(a, {}):
                pipes[a][b] = TcpListen(entries[a]["listen"], max_body=m.registry.max_body + (1 << 17))
            else:
                raise Refused("no-address", f"{a} runs here and its peer {b} elsewhere, and neither says where to meet")
    for (snd, rcv), spec in (adversaries or {}).items():
        if snd in pipes and rcv in pipes[snd]:
            pipes[snd][rcv] = Adversary(pipes[snd][rcv], **spec)
    for a in here:
        role = m.role(a)
        given = (sources or {}).get(a, {})
        clock = None
        for need in SOURCES:
            if need not in role.needs:
                continue
            src = given.get(need)
            if src is None and need == "clock":
                src = (S.VirtualClock(time if mode == "simulated" else S.VirtualTime(), clock_step_ns)
                       if seed is not None else S.RealClock())
            elif src is None:
                src = S.seeded(seed, a, "coins") if seed is not None else S.OsCoins()
            if need == "clock":
                clock = src.now
            mine, theirs = MemoryPipe.pair(S.VirtualTime())
            pipes[a][need] = mine
            rt.sources.append(_Source(a, need, src, theirs))
        handshakes = {}
        for b in m.peers(a):
            if not authenticate or isinstance(pipes[a][b], Playback):
                continue
            if keys.get(a) is None or verify is None:
                raise Refused("no-key", f"{a}'s channels authenticate with its party key, and it has none here")
            nonce = S.seeded(seed, a, b, "nonce").draw(NONCE_BYTES) if seed is not None else os.urandom(NONCE_BYTES)
            peer_key = entries.get(b, {}).get("key")
            handshakes[b] = Handshake(party=a, peer=b, protocol=pid, version=m.version, configuration=conf, nonce=nonce,
                                      key=keys[a], verify=verify, peer_key=None if peer_key is None else bytes.fromhex(peer_key))
        t = Transcript(a, None if tdir is None else tdir / f"{a}.transcript")
        rt.hosts[a] = Host(a, protocol.parties[a](), identity(a), config.get("params", {}), t,
                           _endpoints(m, a, pipes[a], handshakes), rt.transmit, rt.finished, clock)
    from .process import spawn  # here, since process imports this module

    for b, spec in sorted(processes.items()):
        child = spawn(dict(spec, config=dict(config), parties_here=[b], mode=mode,
                           state_dir=None if tdir is None else str(tdir / f"process-{b}"),
                           seed=None if seed is None else seed.hex(), clock_step_ns=clock_step_ns,
                           dial=child_dial.get(b, {}), authenticate=authenticate))
        rt.children[b] = (child, [pipes[a][b] for a in here if b in pipes[a]])
    try:
        rt.run()
    except BaseException:
        for child, _ in rt.children.values():
            child.kill()
        raise
    outcomes = {n: _plain(h.finished) for n, h in sorted(rt.hosts.items())}
    for b, (child, _) in rt.children.items():
        if b not in rt.ended:
            rt._reap(b)
        out = bytes(rt.printed[b])
        try:
            got = json.loads(out.decode().strip().splitlines()[-1])
            outcomes.update(got["outcomes"])
        except (ValueError, IndexError, KeyError):
            outcomes[b] = {"outcome": {"exit": child.returncode}, "refused": "party-died"}
        if child.returncode != 0 and outcomes.get(b, {}).get("refused") is None:
            outcomes[b] = {"outcome": {"exit": child.returncode}, "refused": "party-died"}
    for h in rt.hosts.values():
        h.transcript.close()
    ended = {n: e for n, h in sorted(rt.hosts.items())
             if (e := {ch: ep.pipe.ended for ch, ep in sorted(h.endpoints.items())
                       if ep.pipe is not None and ep.pipe.ended is not None})}
    result = RunResult(mode if not playback else "mixed", _status(outcomes, rt.divergence), outcomes,
                       {n: h.transcript for n, h in rt.hosts.items()}, rt.divergence, placement,
                       None if runs is None else runs.n, [] if runs is None else runs.incomplete, ended)
    if runs is not None:
        (tdir / "record").write_bytes(result.record())
        runs.end(result.status)
    return result


__all__ = ["ARRIVED", "FAILED", "MODES", "RUNTIME", "Diverged", "Endpoint", "Host", "ProtocolDef", "RunResult", "Runtime",
           "configuration_hash", "protocol_of", "replay", "run"]
