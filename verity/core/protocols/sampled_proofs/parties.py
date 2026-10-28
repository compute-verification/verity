"""Sampled proofs' parties on the deployment layer (`verity.core.deploy`): the challenger, the firewall, the workers and the
verifier, the roles `verity run CONFIG --party ROLE` runs, and their table (`PROTOCOL`; `PROTOCOLS` by id).

~~~text
challenger -> verifier    Challenger {address, pin}: where its host answers the verifier's open and finish of a session
ZK server   -> verifier    Firewall {sha512, sessions}: the Lean ZK server every outer proof goes through, one at a time
workers    -> verifier    Registration {block}, the verification phase's registration record its body (canonical JSON)
verifier   -> challenger  Done, once every verification phase is decided
~~~

A party does no I/O: what its machine does is a capability it is given (`Identity.capabilities`), by name.

- the workers' ``serve(block) -> bytes``: serve verification phase ``block`` of the run, and give its registration record;
- the ZK server's ``pin() -> {"sha512", "sessions"}``: the ZK server the prover's sessions run, by its SHA-512;
- the challenger's ``host``, with ``start() -> {"address", "pin"}`` and ``stop()``: the challenger's host on its machine,
  whose coins are its own, never the verifier's or the prover's, answering a session's open and finish at ``address``
  and pinned on main's first-parent commit ``pin``;
- the verifier's ``audit(block, registration, challenger, firewall) -> (answer, case)``: the block's one-stage audit
  drawn on that challenger's coins, its answer (``verdict``, and its ``commitment``) and its case for the product, or
  None; ``component(block, name) -> (answer, case) | None``, another listed component's; and ``product(ids, cases,
  digests) -> {"product", "verdict", "reason"}``, `Core.Composition.product` over exactly those components.

A capability that can't do its step raises `Unavailable` with its reason, and the party finishes refused by its step's
name with that reason recorded (an audit's or a product's is that block's ``error``). Anything else a capability raises
is a fault of the machine: it leaves the party, and the run fails by name (the runtime's ``party-died``).

The run's parameters (`params_of`) are its verification phases (``blocks``), the configuration's listed components
(``components``, `IDS`' names, sampled proofs among them) and what its coins are (``coins``: ``simulated`` when the
challenger's host is a machine the prover's parties share, else ``development``, a key made for the session).

The verifier audits block ``i`` once its registration, the ZK server's pin and the challenger's address are all in, in
order of the blocks, so no coin is flipped for a block before its registration (live coins, after the commitment). It
holds the audit's commitment to the registration the workers sent: an accepting audit of another record rejects, its
case withheld from the product. A
listed component with no answer is ``absent``, its case `NO_CASE`, which its own decoders refuse, so the product never
accepts over it. Each block's line is an output entry of the record and the verifier's outcome:

    {"block": I, "coins": C, "components": {NAME: accept|reject|undecided|error|absent}, "product": ID,
     "registration": DIGEST, "verdict": V, "reason": R}

An audit or a product that is `Unavailable` is that block's ``error``, never a verdict (an audit's reason a private entry
of the record). The verifier refuses by name a block out of
range or registered twice (``schema``, ``out-of-phase``), a second address or pin (``out-of-phase``), and a peer that
leaves before it said what the verifier waits on (``closed``); it tells the challenger ``Done`` before it finishes,
however it finishes, and the challenger stops its host on ``Done`` or when the verifier leaves.

What stays the machine's (the capabilities above): the toy's serving, the records audit that runs the prover's sessions
behind the ZK server on the drawn members, the challenger's host (its LMS key, ledger and pins log) and the Lean product
over the record. Until the audit's sample, its witnesses and its proofs are messages of their own (the configuration's
``draw``, ``witness`` and ``proof`` channels), the verifier and the workers share the machine the block's rows are on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Identity, Received, Record, Send
from verity.core.deploy.runtime import ProtocolDef
from verity.core.wire import Body, Hex, Int, Message, Schema, Str

from .one_stage import registration as R

ID = R.PROTOCOL
VERSION = 1
CHALLENGER, FIREWALL, WORKERS, VERIFIER = "challenger", "firewall", "workers", "verifier"
#: a configuration's component names and their protocol ids, the ones `Core.Onsite.components` holds
IDS = {"bandwidth": "verity/remote-bandwidth/v0", "certifier": "verity/certifier/v0", "pous": "verity/pous/p2/v4/run",
       "pouw": "verity/pouw/v1", "sampled-proofs": ID}
COMPONENT = "sampled-proofs"
#: what a run's coins are (module docstring)
COINS = ("simulated", "development")
#: a listed component whose party gave no case: its own decoders refuse it
NO_CASE = {"commitment": "", "evidence": "", "params": {}}
ABSENT = "absent"
MAX_BLOCKS = 1 << 16
MAX_REGISTRATION = 1 << 24
MANIFEST = Manifest(
    ID, VERSION, "verity.core.protocols.sampled_proofs.parties:params_of",
    parties=(Role(CHALLENGER, "auditor", needs=("host",)), Role(FIREWALL, "developer", needs=("pin",)),
             Role(WORKERS, "developer", needs=("serve",)),
             Role(VERIFIER, "auditor", needs=("audit", "component", "product"))),
    channels=(Channel(CHALLENGER, VERIFIER, ("Challenger",), carries_coins=True),
              Channel(VERIFIER, CHALLENGER, ("Done",)),
              Channel(FIREWALL, VERIFIER, ("Firewall",)),
              Channel(WORKERS, VERIFIER, ("Registration",))),
    messages=(Schema("Challenger", 1, "structure", {"address": Str(255), "pin": Str(64, r"[0-9a-f]{40}|[0-9a-f]{64}")}),
              Schema("Done", 1, "structure"),
              Schema("Firewall", 1, "structure", {"sha512": Hex(64), "sessions": Int(1, 1 << 10)}),
              Schema("Registration", 1, "commitment", {"block": Int(0, MAX_BLOCKS - 1)}, Body(max=MAX_REGISTRATION))))


def params_of(blocks: int, components: Sequence[str], coins: str) -> dict[str, Any]:
    """The run's parameters (module docstring), or a ValueError saying which is wrong."""
    if type(blocks) is not int or not 1 <= blocks <= MAX_BLOCKS:
        raise ValueError(f"blocks is an integer in [1, {MAX_BLOCKS}], not {blocks!r}")
    listed = list(components)
    if COMPONENT not in listed or len(set(listed)) != len(listed) or not set(listed) <= set(IDS):
        raise ValueError(f"components lists {COMPONENT} and others of {sorted(IDS)}, each once, not {listed}")
    if coins not in COINS:
        raise ValueError(f"coins is one of {COINS}, not {coins!r}")
    return {"blocks": blocks, "components": listed, "coins": coins}


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    if set(params) != {"blocks", "components", "coins"}:
        raise ValueError(f"the parameters are blocks, components and coins, not {sorted(params)}")
    return params_of(params["blocks"], params["components"], params["coins"])


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
    """The ZK server: its pin told to the verifier."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        try:
            pin = identity.capabilities["pin"]()
        except Unavailable as e:
            return [Finish({"error": _why(e)}, "firewall", kind="private")]
        msg = Message("Firewall", {"sha512": pin["sha512"], "sessions": pin["sessions"]})
        return [Send(VERIFIER, msg), Finish(dict(msg.fields))]

    def on(self, r: Received) -> list[Any]:
        return []


class Workers:
    """The workers: each verification phase served, and its registration sent to the verifier."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        try:
            blocks = _params(params)["blocks"]
        except ValueError:
            return [Finish({"served": 0}, "schema")]
        serve, out = identity.capabilities["serve"], []
        for i in range(blocks):
            try:
                registration = bytes(serve(i))
            except Unavailable as e:
                return [*out, Finish({"served": i, "error": _why(e)}, "serve", kind="private")]
            out.append(Send(VERIFIER, Message("Registration", {"block": i}, registration)))
        return [*out, Finish({"served": blocks})]

    def on(self, r: Received) -> list[Any]:
        return []


class Verifier:
    """The verifier: each block audited on the challenger's coins, then the product over the listed components (module
    docstring)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.lines: list[dict[str, Any]] = []
        self.registrations: dict[int, tuple[bytes, dict[str, Any]]] = {}
        self.challenger: dict[str, Any] | None = None
        self.firewall: dict[str, Any] | None = None
        self.done = False
        caps = identity.capabilities
        self.audit, self.component, self.product = caps["audit"], caps["component"], caps["product"]
        try:
            self.params = _params(params)
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
        if m.type == "Challenger" or m.type == "Firewall":
            if (self.challenger if m.type == "Challenger" else self.firewall) is not None:
                return self._finish("out-of-phase")
            if m.type == "Challenger":
                self.challenger = dict(m.fields)
            else:
                self.firewall = dict(m.fields)
        else:
            i = m.fields["block"]
            if i >= self.params["blocks"]:
                return self._finish("schema")
            if i in self.registrations:
                return self._finish("out-of-phase")
            try:
                record = json.loads(m.body)
            except ValueError:
                return self._finish("schema")
            if not isinstance(record, dict):
                return self._finish("schema")
            self.registrations[i] = (m.body, record)
        return self._advance()

    def _said(self, peer: str) -> bool:
        """Whether ``peer`` sent everything the verifier waits on from it."""
        if peer == WORKERS:
            return len(self.registrations) == self.params["blocks"]
        return (self.firewall if peer == FIREWALL else self.challenger) is not None

    def _advance(self) -> list[Any]:
        out: list[Any] = []
        while (len(self.lines) < self.params["blocks"] and len(self.lines) in self.registrations
               and self.challenger is not None and self.firewall is not None):
            out += self._decide(len(self.lines))
        if len(self.lines) == self.params["blocks"]:
            out += self._finish(None)
        return out

    def _decide(self, i: int) -> list[Record]:
        """Block ``i``'s line as an output entry, after a private one saying why its audit was unavailable."""
        raw, record = self.registrations[i]
        digest = R.digest(record)
        out = []
        try:
            answer, case = self.audit(i, raw, dict(self.challenger), dict(self.firewall))
        except Unavailable as e:
            answer, case = {"verdict": "error"}, None
            out.append(Record({"block": i, "audit": "unavailable", "error": _why(e)}, "private"))
        if answer.get("verdict") == "accept" and answer.get("commitment") != digest:
            answer, case = {"verdict": "reject", "commitment": answer.get("commitment"),
                            "reason": "the audit's commitment is not the registration the workers sent"}, None
        status, cases, digests = {}, {}, {}
        for c in self.params["components"]:
            got = (answer, case) if c == COMPONENT else self.component(i, c)
            ans, cs = got if got is not None else (None, None)
            status[c] = ABSENT if ans is None else str(ans.get("verdict", "error"))
            cases[IDS[c]] = cs or NO_CASE
            digests[IDS[c]] = (ans or {}).get("commitment") or ""
        try:
            got = self.product([IDS[c] for c in self.params["components"]], cases, digests)
            verdict = {"product": str(got["product"]), "verdict": str(got["verdict"]),
                       "reason": None if got.get("reason") is None else str(got["reason"])}
        except Unavailable as e:
            verdict = {"product": None, "verdict": "error", "reason": _why(e)}
        line = {"block": i, "coins": self.params["coins"], "components": status, "registration": digest, **verdict}
        self.lines.append(line)
        return [*out, Record(line, "output")]

    def _finish(self, refused: str | None) -> list[Any]:
        out: list[Any] = [] if self.done else [Send(CHALLENGER, Message("Done"))]
        self.done = True
        return [*out, Finish({"verdicts": list(self.lines)}, refused)]


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: Challenger, FIREWALL: Firewall, WORKERS: Workers, VERIFIER: Verifier})
PROTOCOLS = {ID: PROTOCOL}


__all__ = ["ABSENT", "CHALLENGER", "COINS", "COMPONENT", "FIREWALL", "ID", "IDS", "MANIFEST", "NO_CASE", "PROTOCOL",
           "PROTOCOLS", "VERIFIER", "VERSION", "WORKERS", "Challenger", "Firewall", "Unavailable", "Verifier", "Workers",
           "params_of"]
