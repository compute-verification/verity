"""PoUW's execution phases as three parties of the deployment layer (`verity.runtime`): the challenger
(`ChallengerParty`) and the verifier (`Verifier`) on the auditor's devices, and the prover (`Prover`) on the developer's
machine, one a server.

~~~text
challenger -> verifier    Commitment {phase} + the challenge commitment's root, for every phase before the first is
                          begun; Coin {phase, opening}, a challenge opened once, after what the verifier recorded
verifier   -> challenger  Open {phase, label} + what the challenge follows (64 bytes); Done, once every phase is decided
verifier   -> prover      Begin {phase} + the phase's challenge commitment root; Salt {phase, opening}, the salt
                          challenge; Sample {phase, opening} + the sample, its sampling challenge and the proof blocks
                          it samples; End
prover     -> verifier    Root {phase} + the root the salt follows; Registration {phase} + the registration;
                          Proof {phase} + the sampled proof blocks' proofs (private)
~~~

The challenger holds each phase's challenges: `coins.Challenger`, its salt challenge and its sampling challenge and
their hm96 salts, sampled from what its `coins` need gives (in `coins.Challenger`'s order) and committed before the
phase is begun. It opens a challenge only on the verifier's `Open`, after what the verifier recorded
(`coins.CoinOpening.after`), and never twice, so a retry never resamples; a second `Open` of one challenge is refused
`out-of-phase`. It finishes on `Done` with each challenge it opened, ``{"phase", "label", "after"}``.

The verifier begins phase ``i`` once phase ``i − 1`` is decided and the challenger's commitment for ``i`` is in. It
asks for the salt challenge only once the prover's root is recorded, and for the sampling challenge only once the
registration is, so no challenge is opened before what it follows is committed. It holds each opening to the phase's
commitment, its label and what it recorded (`coins.coin_reason`), and refuses the run ``coin`` when one doesn't open.
Its machine gives it:

- ``registered(phase, registration) -> bytes``: the registration's digest, the one the sampling challenge follows
  (`core_check.digest` of `core_check.commitment`), or a ValueError when it isn't PoUW's, which rejects the phase;
- ``sample(phase, registration, opening) -> bytes``: the proof blocks the sampling challenge samples, a uniform subset
  of K′ of the registered N, as `core_check` holds a sample to;
- ``check(phase, registration, salt, opening, sample, proof) -> {"verdict", "reason"}``: the Lean verifier over the
  sampled proof blocks' proofs and `pouw-check`'s `check` of the phase (`PouwCheck.check`), on the challenges it was
  given.

Each phase's line is an output entry of the record, and the verifier finishes with every line:

    {"phase": I, "registration": DIGEST | None, "verdict": accept|reject|undecided|error, "reason": R}

A phase whose registration isn't PoUW's is rejected with no sampling challenge opened, and the next phase is begun: a
phase credits only itself. A capability that is `Unavailable` makes that phase's line ``error``. The prover's machine
gives it:

- ``open(phase) -> bytes``: the 64-byte root the salt follows: the weights' root a credit period's open signs, or a live
  phase's open root (`salt.window_open_root`);
- ``serve(phase, salt) -> bytes``: the phase served on the salt challenge's noise, its values committed, and its
  registration;
- ``prove(phase, opening, sample) -> bytes``: the sampled proof blocks proved in zero knowledge.

The prover holds the salt challenge to the commitment it was sent with `Begin` and to the root it sent
(`coins.coin_reason`), and refuses the run ``coin`` when it doesn't open. It finishes on `End` with how many phases it
proved. Either side refuses by name a frame out of phase (``out-of-phase``), a phase out of range (``schema``) and a
peer that leaves before it sent what the party waits on (``closed``).

A certified phase's salt is the network certifier's (`salt.revealed`), a layer of its own that these parties don't hold;
a live phase's round challenges (`window.live_challenger`) aren't in them yet either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from verity.channel import Body, Int, Json, Message, Schema, Str
from verity.commitments.scheme import DIGEST_BYTES
from verity.party import Finish, Identity, LogEntry, Received, Send
from verity.protocol import Channel, Manifest, ProtocolDef, Role
from verity.protocols.proofs.spot.verifier.parties import MAX_PROOF, MAX_REGISTRATION, MAX_SAMPLE, Unavailable

from . import coins as Co
from .protocol import DomainError

ID = "verity/pouw/execution-phases"
VERSION = 1
CHALLENGER, PROVER, VERIFIER = "challenger", "prover", "verifier"
SALT, SAMPLING = Co.LABELS
MAX_PHASES = 1 << 10
PHASE_COINS = len(Co.LABELS) * (Co.COIN_BYTES + Co.HM96.salt_bytes)
"""The bytes of the `coins` need one phase's `coins.Challenger` reads."""
ROOT = Body(size=DIGEST_BYTES["sha512"], max=DIGEST_BYTES["sha512"])
PHASE = Int(0, MAX_PHASES - 1)
OPENING = Json(1 << 12)
MANIFEST = Manifest(
    ID, VERSION, "verity.protocols.accounting.pouw.verifier.parties:params_of",
    parties=(Role(CHALLENGER, "auditor", needs=("coins",)),
             Role(PROVER, "developer", per="server", needs=("open", "serve", "prove")),
             Role(VERIFIER, "auditor", needs=("registered", "sample", "check"))),
    channels=(Channel(CHALLENGER, VERIFIER, ("Commitment", "Coin"), carries_coins=True),
              Channel(VERIFIER, CHALLENGER, ("Open", "Done"), after=("Commitment",)),
              Channel(VERIFIER, PROVER, ("Begin", "Salt", "Sample", "End"), carries_coins=True, after=("Commitment",)),
              Channel(PROVER, VERIFIER, ("Root", "Registration", "Proof"), after=("Begin",))),
    messages=(Schema("Commitment", 1, "commitment", {"phase": PHASE}, ROOT),
              Schema("Open", 1, "commitment", {"phase": PHASE, "label": Str(16)}, ROOT),
              Schema("Coin", 1, "coins", {"phase": PHASE, "opening": OPENING}),
              Schema("Done", 1, "structure"),
              Schema("Begin", 1, "commitment", {"phase": PHASE}, ROOT),
              Schema("Root", 1, "commitment", {"phase": PHASE}, ROOT),
              Schema("Salt", 1, "coins", {"phase": PHASE, "opening": OPENING}),
              Schema("Registration", 1, "commitment", {"phase": PHASE}, Body(max=MAX_REGISTRATION)),
              Schema("Sample", 1, "coins", {"phase": PHASE, "opening": OPENING}, Body(max=MAX_SAMPLE)),
              Schema("Proof", 1, "private", {"phase": PHASE}, Body(max=MAX_PROOF)),
              Schema("End", 1, "structure")))


def params_of(phases: int) -> dict[str, Any]:
    """The run's parameters, its execution phases' count, or a ValueError."""
    if type(phases) is not int or not 1 <= phases <= MAX_PHASES:
        raise ValueError(f"phases is an integer in [1, {MAX_PHASES}], not {phases!r}")
    return {"phases": phases}


def _params(params: Mapping[str, Any]) -> int:
    if set(params) != {"phases"}:
        raise ValueError(f"the parameters are phases, not {sorted(params)}")
    return params_of(params["phases"])["phases"]


def _why(e: Exception) -> str:
    return str(e)[:2000]


def _ended(m: Message) -> str:
    return m.fields["name"] if m.type == "Refused" else "closed"


class _Stream:
    """`coins.Challenger`'s entropy: the bytes the challenger's `coins` need gave, read in order, each once."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __call__(self, n: int) -> bytes:
        if n > len(self.data):
            raise ValueError(f"the phase's challenge bytes are {len(self.data)}, not {n}")
        out, self.data = self.data[:n], self.data[n:]
        return out


class ChallengerParty:
    """The challenger: every phase's challenges committed before the first is begun, each challenge opened once on the
    verifier's `Open` (module docstring). `ask` is the type of the message that asks its `coins` need for bytes, which
    the program that starts the run hands in (`protocol`)."""

    def __init__(self, ask: str) -> None:
        self.ask = ask

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        try:
            self.phases = _params(params)
        except ValueError:
            return [Finish({"opened": []}, "schema")]
        self.c: list[Co.Challenger] = []
        self.opened: list[dict[str, Any]] = []
        return [Send("coins", Message(self.ask, {"n": PHASE_COINS}))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish({"opened": self.opened}, _ended(m))]
        if m.type == "Coins":
            c = Co.Challenger(len(self.c), _Stream(m.body))
            self.c.append(c)
            out = [Send(VERIFIER, Message("Commitment", {"phase": c.epoch}, c.commitment.root))]
            more = [Send("coins", Message(self.ask, {"n": PHASE_COINS}))] if len(self.c) < self.phases else []
            return out + more
        if m.type == "Done":
            return [Finish({"opened": self.opened})]
        i, label = m.fields["phase"], m.fields["label"]
        if i >= len(self.c) or label not in Co.LABELS:
            return [Finish({"opened": self.opened}, "schema")]
        try:
            opening = self.c[i].open(label, m.body)
        except DomainError:
            return [Finish({"opened": self.opened}, "out-of-phase")]
        self.opened.append({"phase": i, "label": label, "after": m.body.hex()})
        return [Send(VERIFIER, Message("Coin", {"phase": i, "opening": opening.to_json()}))]


class Verifier:
    """The verifier: each phase begun, its challenges asked for once what they follow is recorded, and decided on its
    proofs (module docstring)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.given = identity.capabilities
        self.commitments: dict[int, Co.CoinCommitment] = {}
        self.lines: list[dict[str, Any]] = []
        self.phase, self.stage = 0, "begin"
        try:
            self.phases = _params(params)
        except ValueError:
            return self._finish("schema")
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return self._finish(_ended(m))
        i = m.fields.get("phase")
        if m.type == "Commitment":
            if i != len(self.commitments) or i >= self.phases:
                return self._finish("out-of-phase")
            self.commitments[i] = Co.CoinCommitment(i, m.body)
            return self._begin() if self.stage == "begin" else []
        if i != self.phase:
            return self._finish("out-of-phase")
        step = {("Root", "root"): self._root, ("Coin", "opening"): self._coin,
                ("Registration", "registration"): self._registration, ("Proof", "proof"): self._proof}
        handler = step.get((m.type, self.stage))
        return self._finish("out-of-phase") if handler is None else handler(m)

    def _begin(self) -> list[Any]:
        if self.phase == self.phases:
            return self._finish()
        c = self.commitments.get(self.phase)
        if c is None:
            self.stage = "begin"
            return []
        self.stage, self.registration, self.digest = "root", None, None
        return [Send(PROVER, Message("Begin", {"phase": self.phase}, c.root))]

    def _open(self, label: str, after: bytes) -> list[Any]:
        self.stage, self.label, self.after = "opening", label, after
        return [Send(CHALLENGER, Message("Open", {"phase": self.phase, "label": label}, after))]

    def _root(self, m: Message) -> list[Any]:
        return self._open(SALT, m.body)

    def _coin(self, m: Message) -> list[Any]:
        try:
            opening = Co.CoinOpening.from_json(m.fields["opening"])
            why = Co.coin_reason(self.commitments[self.phase], opening, self.label, self.after)
        except (DomainError, KeyError, TypeError, ValueError) as e:
            why = _why(e)
        if why is not None:
            return self._finish("coin")
        i = self.phase
        if self.label == SALT:
            self.salt, self.stage = opening, "registration"
            return [Send(PROVER, Message("Salt", {"phase": i, "opening": opening.to_json()}))]
        try:
            self.sample = bytes(self.given["sample"](i, self.registration, opening))
        except Unavailable as e:
            return self._decide("error", _why(e))
        self.opening, self.stage = opening, "proof"
        return [Send(PROVER, Message("Sample", {"phase": i, "opening": opening.to_json()}, self.sample))]

    def _registration(self, m: Message) -> list[Any]:
        self.registration = m.body
        try:
            self.digest = bytes(self.given["registered"](self.phase, m.body))
        except Unavailable as e:
            return self._decide("error", _why(e))
        except ValueError as e:
            return self._decide("reject", f"registration: {_why(e)}")
        return self._open(SAMPLING, self.digest)

    def _proof(self, m: Message) -> list[Any]:
        try:
            answer = self.given["check"](self.phase, self.registration, self.salt, self.opening, self.sample, m.body)
        except Unavailable as e:
            return self._decide("error", _why(e))
        return self._decide(answer["verdict"], answer.get("reason"))

    def _decide(self, verdict: str, reason: str | None) -> list[Any]:
        line = {"phase": self.phase, "registration": None if self.digest is None else self.digest.hex(),
                "verdict": verdict, "reason": reason}
        self.lines.append(line)
        self.phase += 1
        return [LogEntry(line, "output"), *self._begin()]

    def _finish(self, refused: str | None = None) -> list[Any]:
        end = [] if refused is not None else [Send(PROVER, Message("End"))]
        return [Send(CHALLENGER, Message("Done")), *end, Finish({"phases": self.lines}, refused)]


class Prover:
    """The prover: each phase opened on `Begin`, served on its salt challenge, and its sample proved (module
    docstring)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.given: Mapping[str, Callable[..., Any]] = identity.capabilities
        self.proved, self.phase, self.expect = 0, -1, ("Begin", "End")
        try:
            self.phases = _params(params)
        except ValueError:
            return [Finish({"proved": 0}, "schema")]
        return []

    def _stop(self, refused: str, e: Exception | None = None) -> list[Any]:
        if e is None:
            return [Finish({"proved": self.proved}, refused)]
        return [Finish({"proved": self.proved, "error": _why(e)}, refused, kind="private")]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return self._stop(_ended(m))
        if m.type not in self.expect:
            return self._stop("out-of-phase")
        if m.type == "End":
            return [Finish({"proved": self.proved})]
        i = m.fields["phase"]
        if m.type == "Begin":
            if i <= self.phase or i >= self.phases:
                return self._stop("out-of-phase")
            self.phase, self.commitment = i, Co.CoinCommitment(i, m.body)
            try:
                self.root = bytes(self.given["open"](i))
            except Unavailable as e:
                return self._stop("open", e)
            self.expect = ("Salt",)
            return [Send(VERIFIER, Message("Root", {"phase": i}, self.root))]
        if i != self.phase:
            return self._stop("out-of-phase")
        try:
            opening = Co.CoinOpening.from_json(m.fields["opening"])
        except (DomainError, KeyError, TypeError, ValueError):
            return self._stop("coin")
        if m.type == "Salt":
            if Co.coin_reason(self.commitment, opening, SALT, self.root) is not None:
                return self._stop("coin")
            try:
                registration = bytes(self.given["serve"](i, opening))
            except Unavailable as e:
                return self._stop("serve", e)
            self.expect = ("Sample", "Begin", "End")
            return [Send(VERIFIER, Message("Registration", {"phase": i}, registration))]
        try:
            proof = bytes(self.given["prove"](i, opening, m.body))
        except Unavailable as e:
            return self._stop("prove", e)
        self.proved += 1
        self.expect = ("Begin", "End")
        return [Send(VERIFIER, Message("Proof", {"phase": i}, proof))]


def protocol(ask: str) -> ProtocolDef:
    """PoUW's three parties, the challenger handed `ask`, the type of the message that asks its `coins` need for bytes."""
    return ProtocolDef(MANIFEST, {CHALLENGER: lambda: ChallengerParty(ask), PROVER: Prover, VERIFIER: Verifier})


__all__ = ["CHALLENGER", "ID", "MANIFEST", "MAX_PHASES", "PHASE_COINS", "PROVER", "SALT", "SAMPLING", "VERIFIER", "VERSION",
           "ChallengerParty", "Prover", "Verifier", "params_of", "protocol"]
