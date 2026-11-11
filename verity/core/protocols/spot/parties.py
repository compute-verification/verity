"""Sampled proofs' parties on the deployment layer (`verity.core.deploy`), as roles and channels a deployment configuration
declares, in three deployments, which the deployment layer's runtime runs (`verity.core.deploy.runtime.run`; `PROTOCOLS` by
id):

- **SPOT[Flock] on its own** (`SPOT_ID`: `SPOT_MANIFEST`, `SPOT_PROTOCOL`), three roles. The inference server serves and
  commits its values. The proof server replays the sampled proof blocks and proves them (`flock-circuit prove --zk`). The
  verifier holds the challenger's live coins and signed log, with the challenger on its side, and runs the Lean verifier.
  There is no ZK server and no recursion: the proof server talks only to the verifier.
- **SPOT[Flock] in two stages** (`SPOT_TWO_ID`: `SPOT_TWO_MANIFEST`, `SPOT_TWO_PROTOCOL`), the same three roles under a
  two-stage sampling design. The draw samples replay blocks; the proof server replays them and registers their interiors
  (the interior registration), and the verifier then draws the second stage on fresh coins and audits both stages.
- **The core deployment** (`ID`: `MANIFEST`, `PROTOCOL`), which wraps SPOT[Flock] for both composites (remote and onsite).
  Its workers are the inference server and the proof server on one machine, proving behind the ZK server, whose pin the
  verifier holds. Its verifier composes each block's audit with the composite's other subprotocols.

The network certifier is a layer of its own. It stamps each request's release order and bucket at ingress, and SPOT reads
that stamp as a certified input. Neither imports the other (`tests/test_boundaries.py`); they compose only above both.

~~~text
SPOT[Flock]
challenger       -> verifier          Challenger {address, pin}: where its host answers the open and finish of a session
inference server -> verifier          Registration {block}, the execution phase's registration record its body
inference server -> proof server      Replay {block}, the values the block's proof blocks read its body; one way
verifier         -> proof server      Draw {block, address}, the block's sample its body, once its registration is in
proof server     -> verifier          Proof {block}, the proofs of the sample's proof blocks its body
verifier         -> challenger        Done, once every execution phase is decided

SPOT[Flock] in two stages: SPOT[Flock]'s messages, with two between Draw and Proof
proof server     -> verifier          Interior {block}, the interior registration of the drawn replay blocks its body
verifier         -> proof server      Second {block, address}, the second stage's sample its body, once its interior is in
proof server     -> verifier          Proof {block}, the proofs of the second stage's proof blocks its body

the core deployment
challenger -> verifier    Challenger {address, pin}
ZK server  -> verifier    Firewall {sha512, sessions}: the Lean ZK server every outer proof goes through, one at a time
workers    -> verifier    Registration {block}
verifier   -> challenger  Done
~~~

A party does no I/O: what its machine does is a capability it is given (`Identity.capabilities`), by name.

- the challenger's ``host``, with ``start() -> {"address", "pin"}`` and ``stop()``: the challenger's host on its machine,
  whose coins are its own, never the verifier's or the prover's, answering a session's open and finish at ``address``
  and pinned on main's first-parent commit ``pin``;
- SPOT[Flock]'s:
  - the inference server's ``serve(block) -> bytes``, execution phase ``block`` served and its registration record, and
    ``replay(block) -> bytes``, the block's replay inputs, which only the proof server reads: every value its proof
    blocks read, each a proof block's output, a certified input (the network certifier's stamp among them), a registered
    value or advice charged at its width, with the salts that open their commitments. A value the served hardware
    decides that no block computes is never a free witness among them: serving makes it canonical, or it is advice;
  - the proof server's ``prove(block, replay, sample, address) -> bytes``: the sample's proof blocks replayed and proved,
    each session's coins live from the challenger's host at ``address``;
  - the verifier's ``draw(block, registration, challenger) -> bytes``, the block's sample on that challenger's coins, and
    ``audit(block, registration, challenger, sample, proof) -> answer``: the Lean verifier over the proofs and the
    challenger's signed log, its answer (``verdict``, its ``commitment`` and ``reason``);
- SPOT[Flock] in two stages', SPOT[Flock]'s, except:
  - the proof server's ``interior(block, replay, sample) -> bytes``: the sample's replay blocks replayed and their
    interiors committed, the interior registration (following the block's registration), and ``prove(block, replay,
    sample, interior, second, address) -> bytes``, the second stage's proof blocks proved, one session a template;
  - the verifier's ``second(block, registration, challenger, sample, interior) -> bytes``, the second stage's sample on
    that challenger's coins opened after the interior registration's digest, and ``audit(block, registration,
    challenger, sample, proof, interior, second) -> answer``, both stages' audit, whose answer also names the interior
    registration it audited (``interior``, its digest);
- the core deployment's:
  - the workers' ``serve(block) -> bytes``, as the inference server's;
  - the ZK server's ``pin() -> {"sha512", "sessions"}``: the ZK server the prover's sessions run, by its SHA-512;
  - the verifier's ``audit(block, registration, challenger, firewall) -> (answer, case)``: the block's one-stage audit
    sampled on that challenger's coins, its answer and its case for the product (`COMPOSED`'s, in its one-stage form), or
    None; ``component(block, name) -> (answer, case) | None``, another listed subprotocol's; and ``product(ids, cases,
    digests) -> {"product", "verdict", "reason"}``, the composite of exactly those subprotocols (`Core.Composites.ofIds`:
    remote or onsite).

A capability that can't do its step raises `Unavailable` with its reason, and the party finishes refused by its step's
name with that reason recorded (an audit's or a product's is that block's ``error``). Anything else a capability raises
is a fault of the machine: it leaves the party, and the run fails by name (the runtime's ``party-died``).

The run's parameters are its execution phases (``blocks``) and what its coins are (``coins``: ``simulated`` when the
challenger's host is a machine the prover's parties share, else ``development``, a key made for the session); the core
deployment's (`params_of`) also list the configuration's subprotocols (``components``: exactly one composite's, by name,
`SUBPROTOCOLS`, `verity.core.deploy.config.COMPOSITES`), SPOT[Flock]'s, in one stage or two (`spot_params_of`), none.

Every verifier takes block ``i``'s coins only once its registration and the challenger's address are in, in order of the
blocks, so no coin is flipped for a block before its registration (live coins, after the commitment). SPOT[Flock]'s draws
block ``i``'s sample then and sends it to the proof server, and audits the block once its proof is in. In two stages it
draws block ``i``'s second stage only once the block's interior registration is in, so no second-stage coin is flipped
before the interiors are committed, and audits the block once its proof is in. The core deployment's audits it once the
ZK server's pin is in too. Each holds the audit's commitment to the registration it was sent, and in two stages the
answer's ``interior`` to the interior registration the proof server sent: an accepting audit of another record rejects.
Each block's line is an output entry of the record and the verifier's outcome:

    SPOT[Flock]:  {"block": I, "coins": C, "registration": DIGEST, "verdict": V, "reason": R}
    two stages:   {"block": I, "coins": C, "registration": DIGEST, "interior": DIGEST, "verdict": V, "reason": R}
    core:         {"block": I, "coins": C, "components": {NAME: accept|reject|undecided|error|absent}, "product": ID,
                   "registration": DIGEST, "verdict": V, "reason": R}

In the core deployment a listed subprotocol with no answer is ``absent``, its case `NO_CASE`, which its own decoders
refuse, so the composite never accepts over it, and an accepting audit of another record has its case withheld from the
composite. An audit or a product that is `Unavailable` is that block's ``error``, never a verdict (an audit's reason a
private entry of the record); a draw or a second-stage draw that is `Unavailable` ends SPOT[Flock]'s verifier refused
``draw`` or ``second``. A verifier refuses by name a block out of range or registered twice (``schema``,
``out-of-phase``), a second address or pin, an interior registration that isn't a JSON object (``schema``) or is of a
block not drawn or already registered, or a proof of a block not drawn (in two stages, not drawn a second time) or
already proved (``out-of-phase``), and a peer that leaves before it said what the verifier waits on (``closed``); it
tells the challenger ``Done`` before it finishes, however it finishes, and the challenger stops its host on ``Done`` or
when the verifier leaves. The proof server refuses a draw, a second-stage draw or a replay of a block out of range
(``schema``), sent twice, or, for a second-stage draw, of a block whose interior it hasn't registered (``out-of-phase``),
and a peer that leaves before it sent what the proof server waits on (``closed``).

What stays the machine's (the capabilities above): the serving, the replay and the prover's sessions, the challenger's
host (its LMS key, ledger and pins log) and the Lean verifier; in the core deployment also the records audit that runs
the prover's sessions behind the ZK server on the sampled members and the Lean product over the record. There, until the
audit's sample, its witnesses and its proofs are messages of their own (the configuration's ``draw``, ``witness`` and
``proof`` channels, as SPOT[Flock]'s ``Draw`` and ``Proof`` are), the verifier and the workers share the machine the
block's rows are on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from verity.core.deploy.config import COMPOSITES
from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Identity, Received, LogEntry, Send
from verity.core.deploy.runtime import ProtocolDef
from verity.core.wire import Body, Hex, Int, Message, Schema, Str

from .one_stage import registration as R

ID = R.PROTOCOL
#: SPOT as both composites compose it (`VStar.Recursive.Sampled`): one-stage v2's audit (`ID`'s record, the
#: verdicts the verifier's own run of the proof check) under the ``form`` ``one-stage``; `ID`'s own audit, whose
#: verdicts are the evidence's, and the recursive audit are refused
COMPOSED = "verity/sampled-proofs/v1"
VERSION = 1
#: SPOT[Flock] on its own: no ZK server, no recursion and no composite
SPOT_ID = "verity/sampled-proofs/spot/v1"
#: SPOT[Flock] on its own under a two-stage sampling design
SPOT_TWO_ID = "verity/sampled-proofs/spot/two-stage/v1"
CHALLENGER, FIREWALL, WORKERS, VERIFIER = "challenger", "firewall", "workers", "verifier"
INFERENCE_SERVER, PROOF_SERVER = "inference-server", "proof-server"
#: each composite's subprotocols, by a configuration's names, and their protocol ids (`Core.Onsite.components`,
#: `Core.Remote.components`): each one's PoUW is its period check, onsite's bandwidth the P2 session on PoUS's store,
#: and both compose sampled proofs as `COMPOSED`
SUBPROTOCOLS = {
    "verity/onsite/v0": {"bandwidth": "verity/remote-bandwidth/p2/v0", "certifier": "verity/certifier/v0",
                         "inference-only": "verity/inference-only/v0", "pous": "verity/pous/p2/v4/run",
                         "pouw": "verity/pouw/period/v1", "sampled-proofs": COMPOSED,
                         "wipe": "verity/wipe/v0"},
    "verity/remote/v0": {"inference-only": "verity/inference-only/v0", "pouw": "verity/pouw/period/v1",
                         "sampled-proofs": COMPOSED},
}
#: a configuration's subprotocol names and the protocol id each has outside a composite
IDS = {"bandwidth": "verity/remote-bandwidth/v0", "certifier": "verity/certifier/v0",
       "inference-only": "verity/inference-only/v0", "pous": "verity/pous/p2/v4/run", "pouw": "verity/pouw/v1",
       "sampled-proofs": ID}
COMPONENT = "sampled-proofs"
#: what a run's coins are (module docstring)
COINS = ("simulated", "development")
#: a listed subprotocol whose party gave no case: its own decoders refuse it
NO_CASE = {"commitment": "", "evidence": "", "params": {}}
ABSENT = "absent"
MAX_BLOCKS = 1 << 16
MAX_REGISTRATION = 1 << 24
MAX_SAMPLE = 1 << 24
MAX_REPLAY = MAX_PROOF = (1 << 32) - 1
CHALLENGER_SCHEMA = Schema("Challenger", 1, "structure", {"address": Str(255), "pin": Str(64, r"[0-9a-f]{40}|[0-9a-f]{64}")})
DONE_SCHEMA = Schema("Done", 1, "structure")
REGISTRATION_SCHEMA = Schema("Registration", 1, "commitment", {"block": Int(0, MAX_BLOCKS - 1)}, Body(max=MAX_REGISTRATION))
MANIFEST = Manifest(
    ID, VERSION, "verity.core.protocols.spot.parties:params_of",
    parties=(Role(CHALLENGER, "auditor", needs=("host",)), Role(FIREWALL, "developer", needs=("pin",)),
             Role(WORKERS, "developer", needs=("serve",)),
             Role(VERIFIER, "auditor", needs=("audit", "component", "product"))),
    channels=(Channel(CHALLENGER, VERIFIER, ("Challenger",), carries_coins=True),
              Channel(VERIFIER, CHALLENGER, ("Done",)),
              Channel(FIREWALL, VERIFIER, ("Firewall",)),
              Channel(WORKERS, VERIFIER, ("Registration",))),
    messages=(CHALLENGER_SCHEMA, DONE_SCHEMA,
              Schema("Firewall", 1, "structure", {"sha512": Hex(64), "sessions": Int(1, 1 << 10)}),
              REGISTRATION_SCHEMA))
SPOT_SCHEMAS = (CHALLENGER_SCHEMA, DONE_SCHEMA, REGISTRATION_SCHEMA,
                Schema("Replay", 1, "private", {"block": Int(0, MAX_BLOCKS - 1)}, Body(max=MAX_REPLAY)),
                Schema("Draw", 1, "coins", {"block": Int(0, MAX_BLOCKS - 1), "address": Str(255)}, Body(max=MAX_SAMPLE)),
                Schema("Proof", 1, "private", {"block": Int(0, MAX_BLOCKS - 1)}, Body(max=MAX_PROOF)))


def _spot_manifest(protocol: str, proof_server: tuple[str, ...], verifier: tuple[str, ...], draws: tuple[str, ...],
                   proofs: tuple[str, ...], schemas: tuple[Schema, ...] = ()) -> Manifest:
    return Manifest(
        protocol, VERSION, "verity.core.protocols.spot.parties:spot_params_of",
        parties=(Role(CHALLENGER, "auditor", needs=("host",)),
                 Role(INFERENCE_SERVER, "developer", needs=("serve", "replay")),
                 Role(PROOF_SERVER, "developer", needs=proof_server), Role(VERIFIER, "auditor", needs=verifier)),
        channels=(Channel(CHALLENGER, VERIFIER, ("Challenger",), carries_coins=True),
                  Channel(VERIFIER, CHALLENGER, ("Done",)),
                  Channel(INFERENCE_SERVER, VERIFIER, ("Registration",)),
                  Channel(INFERENCE_SERVER, PROOF_SERVER, ("Replay",)),
                  Channel(VERIFIER, PROOF_SERVER, draws, carries_coins=True, after=("Registration", "Challenger")),
                  Channel(PROOF_SERVER, VERIFIER, proofs, after=("Draw", "Replay"))),
        messages=SPOT_SCHEMAS + schemas)


SPOT_MANIFEST = _spot_manifest(SPOT_ID, ("prove",), ("draw", "audit"), ("Draw",), ("Proof",))
SPOT_TWO_MANIFEST = _spot_manifest(
    SPOT_TWO_ID, ("interior", "prove"), ("draw", "second", "audit"), ("Draw", "Second"), ("Interior", "Proof"),
    (Schema("Interior", 1, "commitment", {"block": Int(0, MAX_BLOCKS - 1)}, Body(max=MAX_REGISTRATION)),
     Schema("Second", 1, "coins", {"block": Int(0, MAX_BLOCKS - 1), "address": Str(255)}, Body(max=MAX_SAMPLE))))


def ids_of(names: Sequence[str]) -> dict[str, str]:
    """The protocol id of each of ``names`` in the composite they make (`SUBPROTOCOLS`), or a ValueError."""
    for subs in SUBPROTOCOLS.values():
        if sorted(names) == sorted(subs) and len(set(names)) == len(names):
            return {n: subs[n] for n in names}
    raise ValueError(f"components lists one composite's subprotocols, each once ({COMPOSITES}), not {list(names)}")


def _blocks(blocks: int) -> None:
    if type(blocks) is not int or not 1 <= blocks <= MAX_BLOCKS:
        raise ValueError(f"blocks is an integer in [1, {MAX_BLOCKS}], not {blocks!r}")


def _coins(coins: str) -> None:
    if coins not in COINS:
        raise ValueError(f"coins is one of {COINS}, not {coins!r}")


def params_of(blocks: int, components: Sequence[str], coins: str) -> dict[str, Any]:
    """The core deployment's parameters (module docstring), or a ValueError saying which is wrong."""
    _blocks(blocks)
    listed = list(components)
    ids_of(listed)
    _coins(coins)
    return {"blocks": blocks, "components": listed, "coins": coins}


def spot_params_of(blocks: int, coins: str) -> dict[str, Any]:
    """SPOT[Flock]'s parameters (module docstring), or a ValueError saying which is wrong."""
    _blocks(blocks)
    _coins(coins)
    return {"blocks": blocks, "coins": coins}


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    if set(params) != {"blocks", "components", "coins"}:
        raise ValueError(f"the parameters are blocks, components and coins, not {sorted(params)}")
    return params_of(params["blocks"], params["components"], params["coins"])


def _spot_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if set(params) != {"blocks", "coins"}:
        raise ValueError(f"the parameters are blocks and coins, not {sorted(params)}")
    return spot_params_of(params["blocks"], params["coins"])


class Unavailable(Exception):
    """A capability's machine can't do its step; its message is the reason (module docstring)."""


def _why(e: Unavailable) -> str:
    return str(e)[:2000]


class Challenger:
    """The challenger: its host started, its address told to the verifier, and stopped once the verifier is done."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.host = identity.capabilities["host"]
        try:
            at = self.host.start()
        except Unavailable as e:
            return [Finish({"error": _why(e)}, "host", kind="private")]
        self.at = {"address": at["address"], "pin": at["pin"]}
        return [Send(VERIFIER, Message("Challenger", self.at))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        self.host.stop()
        if m.type == "Done":
            return [Finish(self.at)]
        return [Finish(self.at, m.fields["name"] if m.type == "Refused" else "closed")]


class Firewall:
    """The core deployment's ZK server: its pin told to the verifier."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        try:
            pin = identity.capabilities["pin"]()
        except Unavailable as e:
            return [Finish({"error": _why(e)}, "firewall", kind="private")]
        msg = Message("Firewall", {"sha512": pin["sha512"], "sessions": pin["sessions"]})
        return [Send(VERIFIER, msg), Finish(dict(msg.fields))]

    def on(self, r: Received) -> list[Any]:
        return []


#: a serving party's step: its capability's name, and the party and message type its bytes go to
type Step = tuple[str, str, str]
REGISTER: Step = ("serve", VERIFIER, "Registration")


def _serve(params: Mapping[str, Any], identity: Identity, parse: Callable[[Mapping[str, Any]], dict[str, Any]],
           steps: Sequence[Step]) -> list[Any]:
    """Each execution phase served in order, each of ``steps`` sending its capability's bytes for the block; refused by a
    step's name when its capability is `Unavailable`, and ``schema`` on parameters out of range."""
    try:
        blocks = parse(params)["blocks"]
    except ValueError:
        return [Finish({"served": 0}, "schema")]
    caps, out = identity.capabilities, []
    for i in range(blocks):
        for name, to, kind in steps:
            try:
                raw = bytes(caps[name](i))
            except Unavailable as e:
                return [*out, Finish({"served": i, "error": _why(e)}, name, kind="private")]
            out.append(Send(to, Message(kind, {"block": i}, raw)))
    return [*out, Finish({"served": blocks})]


class Workers:
    """The core deployment's workers: each execution phase served, and its registration sent to the verifier."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        return _serve(params, identity, _params, [REGISTER])

    def on(self, r: Received) -> list[Any]:
        return []


class InferenceServer:
    """SPOT[Flock]'s inference server: each execution phase served, its registration sent to the verifier, then its replay
    inputs to the proof server, which never answers."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        return _serve(params, identity, _spot_params, [REGISTER, ("replay", PROOF_SERVER, "Replay")])

    def on(self, r: Received) -> list[Any]:
        return []


class ProofServer:
    """SPOT[Flock]'s proof server: each block proved once its replay inputs and its draw are in, and its proof sent to the
    verifier (module docstring)."""

    #: whether each block's interior is registered on its draw and the block proved on its second-stage draw
    TWO = False

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        caps = identity.capabilities
        self.prove = caps["prove"]
        self.interior = caps["interior"] if self.TWO else None
        self.replays: dict[int, bytes] = {}
        self.draws: dict[int, tuple[bytes, str]] = {}
        self.interiors: dict[int, bytes] = {}
        self.seconds: dict[int, tuple[bytes, str]] = {}
        self.registered = self.proved = 0
        try:
            self.blocks = _spot_params(params)["blocks"]
        except ValueError:
            return [Finish({"proved": 0}, "schema")]
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type == "Refused":
            return [Finish({"proved": self.proved}, m.fields["name"])]
        last = self.seconds if self.TWO else self.draws
        if m.type == "Closed":
            waits = self.replays if r.channel == INFERENCE_SERVER else last
            return [] if self.proved + len(waits) == self.blocks else [Finish({"proved": self.proved}, "closed")]
        held = {"Replay": self.replays, "Draw": self.draws, "Second": self.seconds}[m.type]
        i = m.fields["block"]
        if i >= self.blocks:
            return [Finish({"proved": self.proved}, "schema")]
        if i in held or i < self.proved or (m.type == "Second" and i >= self.registered):
            return [Finish({"proved": self.proved}, "out-of-phase")]
        held[i] = m.body if m.type == "Replay" else (m.body, m.fields["address"])
        out: list[Any] = []
        while self.TWO and self.registered in self.replays and self.registered in self.draws:
            i = self.registered
            try:
                interior = bytes(self.interior(i, self.replays[i], self.draws[i][0]))
            except Unavailable as e:
                return [*out, Finish({"proved": self.proved, "error": _why(e)}, "interior", kind="private")]
            self.interiors[i] = interior
            self.registered += 1
            out.append(Send(VERIFIER, Message("Interior", {"block": i}, interior)))
        while self.proved in self.replays and self.proved in last:
            i = self.proved
            replay, (sample, address) = self.replays.pop(i), self.draws.pop(i)
            if self.TWO:
                second, address = self.seconds.pop(i)
                args: tuple[Any, ...] = (i, replay, sample, self.interiors.pop(i), second, address)
            else:
                args = (i, replay, sample, address)
            try:
                proof = bytes(self.prove(*args))
            except Unavailable as e:
                return [*out, Finish({"proved": i, "error": _why(e)}, "prove", kind="private")]
            self.proved += 1
            out.append(Send(VERIFIER, Message("Proof", {"block": i}, proof)))
        if self.proved == self.blocks:
            out.append(Finish({"proved": self.proved}))
        return out


class TwoStageProofServer(ProofServer):
    """SPOT[Flock]'s proof server in two stages: each block's interior registered once its replay inputs and its draw are
    in, and the block proved once its second-stage draw is in (module docstring)."""

    TWO = True


class _Verifier:
    """What both deployments' verifiers share: their parameters, the challenger's address and each block's registration
    held, a refusal by name, an accepting audit held to the registration it was sent, and ``Done`` told to the challenger
    however it finishes (module docstring). A deployment's verifier adds its other messages (`_take`), what it waits on
    from its other peers (`_said`) and its decisions (`_advance`)."""

    #: the party whose registrations it holds
    SERVER = WORKERS

    def _start(self, params: Mapping[str, Any], parse: Callable[[Mapping[str, Any]], dict[str, Any]]) -> list[Any]:
        self.lines: list[dict[str, Any]] = []
        self.registrations: dict[int, tuple[bytes, dict[str, Any]]] = {}
        self.challenger: dict[str, Any] | None = None
        self.done = False
        try:
            self.params = parse(params)
        except ValueError:
            return self._finish("schema")
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type == "Refused":
            return self._finish(m.fields["name"])
        if m.type == "Closed":
            if r.channel == CHALLENGER:
                self.done = True
            return [] if self._said(r.channel) else self._finish("closed")
        refused = self._take(m)
        return self._advance() if refused is None else self._finish(refused)

    def _take(self, m: Message) -> str | None:
        """Hold ``m``, or say what it is refused by."""
        if m.type == "Challenger":
            if self.challenger is not None:
                return "out-of-phase"
            self.challenger = dict(m.fields)
            return None
        i = m.fields["block"]
        if i >= self.params["blocks"]:
            return "schema"
        if i in self.registrations:
            return "out-of-phase"
        if isinstance(record := _record(m.body), str):
            return record
        self.registrations[i] = (m.body, record)
        return None

    def _said(self, peer: str) -> bool:
        """Whether ``peer`` sent everything the verifier waits on from it."""
        if peer == self.SERVER:
            return len(self.registrations) == self.params["blocks"]
        return self.challenger is not None

    def _advance(self) -> list[Any]:
        raise NotImplementedError

    def _held(self, answer: dict[str, Any], digest: str, interior: str | None = None) -> dict[str, Any]:
        """``answer``, or a rejection when it accepts a commitment other than ``digest``, the registration's, or, given
        ``interior``, the interior registration's digest, an interior other than it."""
        if answer.get("verdict") != "accept":
            return answer
        if answer.get("commitment") != digest:
            return {"verdict": "reject", "commitment": answer.get("commitment"),
                    "reason": f"the audit's commitment is not the registration the {self.SERVER.replace('-', ' ')} sent"}
        if interior is not None and answer.get("interior") != interior:
            return {"verdict": "reject", "commitment": answer.get("commitment"),
                    "reason": "the audit's interior is not the interior registration the proof server sent"}
        return answer

    def _finish(self, refused: str | None) -> list[Any]:
        out: list[Any] = [] if self.done else [Send(CHALLENGER, Message("Done"))]
        self.done = True
        return [*out, Finish({"verdicts": list(self.lines)}, refused)]


def _record(body: bytes) -> dict[str, Any] | str:
    """A registration's body as its record, or the refusal `schema` when it isn't a JSON object."""
    try:
        record = json.loads(body)
    except ValueError:
        return "schema"
    return record if isinstance(record, dict) else "schema"


def _unavailable(i: int, e: Unavailable) -> LogEntry:
    return LogEntry({"block": i, "audit": "unavailable", "error": _why(e)}, "private")


class Verifier(_Verifier):
    """The core deployment's verifier: each block audited on the challenger's coins behind the ZK server's pin, then the
    composite of the listed subprotocols (module docstring)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.firewall: dict[str, Any] | None = None
        caps = identity.capabilities
        self.audit, self.component, self.product = caps["audit"], caps["component"], caps["product"]
        return self._start(params, _params)

    def _take(self, m: Message) -> str | None:
        if m.type != "Firewall":
            return super()._take(m)
        if self.firewall is not None:
            return "out-of-phase"
        self.firewall = dict(m.fields)
        return None

    def _said(self, peer: str) -> bool:
        return self.firewall is not None if peer == FIREWALL else super()._said(peer)

    def _advance(self) -> list[Any]:
        out: list[Any] = []
        while (len(self.lines) < self.params["blocks"] and len(self.lines) in self.registrations
               and self.challenger is not None and self.firewall is not None):
            out += self._decide(len(self.lines))
        if len(self.lines) == self.params["blocks"]:
            out += self._finish(None)
        return out

    def _decide(self, i: int) -> list[LogEntry]:
        """Block ``i``'s line as an output entry, after a private one saying why its audit was unavailable."""
        raw, record = self.registrations[i]
        digest = R.digest(record)
        out = []
        try:
            answer, case = self.audit(i, raw, dict(self.challenger), dict(self.firewall))
        except Unavailable as e:
            answer, case = {"verdict": "error"}, None
            out.append(_unavailable(i, e))
        if (held := self._held(answer, digest)) is not answer:
            answer, case = held, None
        status, cases, digests = {}, {}, {}
        ids = ids_of(self.params["components"])
        for c in self.params["components"]:
            got = (answer, case) if c == COMPONENT else self.component(i, c)
            ans, cs = got if got is not None else (None, None)
            status[c] = ABSENT if ans is None else str(ans.get("verdict", "error"))
            cases[ids[c]] = cs or NO_CASE
            digests[ids[c]] = (ans or {}).get("commitment") or ""
        try:
            got = self.product([ids[c] for c in self.params["components"]], cases, digests)
            verdict = {"product": str(got["product"]), "verdict": str(got["verdict"]),
                       "reason": None if got.get("reason") is None else str(got["reason"])}
        except Unavailable as e:
            verdict = {"product": None, "verdict": "error", "reason": _why(e)}
        line = {"block": i, "coins": self.params["coins"], "components": status, "registration": digest, **verdict}
        self.lines.append(line)
        return [*out, LogEntry(line, "output")]


class SpotVerifier(_Verifier):
    """SPOT[Flock]'s verifier: each block's sample drawn on the challenger's coins once its registration is in and sent to
    the proof server, and the block audited once its proof is in (module docstring)."""

    SERVER = INFERENCE_SERVER
    #: whether each block is proved on a second-stage draw over its interior registration
    TWO = False

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.samples: list[bytes] = []
        self.interiors: dict[int, tuple[bytes, dict[str, Any]]] = {}
        self.seconds: list[bytes] = []
        self.proofs: dict[int, bytes] = {}
        caps = identity.capabilities
        self.draw, self.audit = caps["draw"], caps["audit"]
        self.second = caps["second"] if self.TWO else None
        return self._start(params, _spot_params)

    def _take(self, m: Message) -> str | None:
        if m.type not in ("Interior", "Proof"):
            return super()._take(m)
        i = m.fields["block"]
        if i >= self.params["blocks"]:
            return "schema"
        if m.type == "Interior":
            if i >= len(self.samples) or i in self.interiors or i < len(self.lines):
                return "out-of-phase"
            if isinstance(record := _record(m.body), str):
                return record
            self.interiors[i] = (m.body, record)
            return None
        if i >= len(self.seconds if self.TWO else self.samples) or i in self.proofs or i < len(self.lines):
            return "out-of-phase"
        self.proofs[i] = m.body
        return None

    def _said(self, peer: str) -> bool:
        if peer == PROOF_SERVER:
            return len(self.lines) + len(self.proofs) == self.params["blocks"]
        return super()._said(peer)

    def _advance(self) -> list[Any]:
        out: list[Any] = []
        blocks = self.params["blocks"]
        while len(self.samples) < blocks and len(self.samples) in self.registrations and self.challenger is not None:
            i = len(self.samples)
            try:
                sample = bytes(self.draw(i, self.registrations[i][0], dict(self.challenger)))
            except Unavailable as e:
                return [*out, LogEntry({"block": i, "draw": "unavailable", "error": _why(e)}, "private"),
                        *self._finish("draw")]
            self.samples.append(sample)
            out.append(Send(PROOF_SERVER, Message("Draw", {"block": i, "address": self.challenger["address"]}, sample)))
        while self.TWO and len(self.seconds) < len(self.samples) and len(self.seconds) in self.interiors:
            i = len(self.seconds)
            try:
                second = bytes(self.second(i, self.registrations[i][0], dict(self.challenger), self.samples[i],
                                           self.interiors[i][0]))
            except Unavailable as e:
                return [*out, LogEntry({"block": i, "second": "unavailable", "error": _why(e)}, "private"),
                        *self._finish("second")]
            self.seconds.append(second)
            out.append(Send(PROOF_SERVER, Message("Second", {"block": i, "address": self.challenger["address"]}, second)))
        while len(self.lines) < blocks and len(self.lines) in self.proofs:
            out += self._decide(len(self.lines))
        if len(self.lines) == blocks:
            out += self._finish(None)
        return out

    def _decide(self, i: int) -> list[LogEntry]:
        """Block ``i``'s line as an output entry, after a private one saying why its audit was unavailable."""
        raw, record = self.registrations[i]
        digest = R.digest(record)
        args: tuple[Any, ...] = (i, raw, dict(self.challenger), self.samples[i], self.proofs.pop(i))
        interior: dict[str, str] = {}
        if self.TWO:
            body, inner = self.interiors.pop(i)
            interior = {"interior": R.digest(inner)}
            args = (*args, body, self.seconds[i])
        out = []
        try:
            answer = self._held(self.audit(*args), digest, interior.get("interior"))
        except Unavailable as e:
            answer = {"verdict": "error"}
            out.append(_unavailable(i, e))
        reason = answer.get("reason")
        line = {"block": i, "coins": self.params["coins"], "registration": digest, **interior,
                "verdict": str(answer.get("verdict", "error")), "reason": None if reason is None else str(reason)}
        self.lines.append(line)
        return [*out, LogEntry(line, "output")]


class TwoStageVerifier(SpotVerifier):
    """SPOT[Flock]'s verifier in two stages: each block's sample drawn as SPOT[Flock]'s is, its second stage drawn on the
    challenger's coins once its interior registration is in and sent to the proof server, and the block audited on both
    stages once its proof is in (module docstring)."""

    TWO = True


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: Challenger, FIREWALL: Firewall, WORKERS: Workers, VERIFIER: Verifier})
SPOT_PROTOCOL = ProtocolDef(SPOT_MANIFEST, {CHALLENGER: Challenger, INFERENCE_SERVER: InferenceServer,
                                            PROOF_SERVER: ProofServer, VERIFIER: SpotVerifier})
SPOT_TWO_PROTOCOL = ProtocolDef(SPOT_TWO_MANIFEST, {CHALLENGER: Challenger, INFERENCE_SERVER: InferenceServer,
                                                    PROOF_SERVER: TwoStageProofServer, VERIFIER: TwoStageVerifier})
PROTOCOLS = {ID: PROTOCOL, SPOT_ID: SPOT_PROTOCOL, SPOT_TWO_ID: SPOT_TWO_PROTOCOL}


__all__ = ["ABSENT", "CHALLENGER", "COINS", "COMPONENT", "COMPOSED", "FIREWALL", "ID", "IDS", "INFERENCE_SERVER", "MANIFEST",
           "NO_CASE", "PROOF_SERVER", "PROTOCOL", "PROTOCOLS", "SPOT_ID", "SPOT_MANIFEST", "SPOT_PROTOCOL", "SPOT_SCHEMAS",
           "SPOT_TWO_ID", "SPOT_TWO_MANIFEST", "SPOT_TWO_PROTOCOL", "SUBPROTOCOLS", "VERIFIER", "VERSION", "WORKERS",
           "Challenger", "Firewall", "InferenceServer", "ProofServer", "SpotVerifier", "TwoStageProofServer",
           "TwoStageVerifier", "Unavailable", "Verifier", "Workers", "ids_of", "params_of", "spot_params_of"]
