"""The proof service, the challenger's side: a protocol's `Spec`, the public-items rule (`check_public`), and a `Stream` of
registrations, draws and outcomes over the one-stage audit (`one_stage`).

A protocol states what it commits (`Spec.values`), which proof units need proofs (`Spec.selection`, its selection rule),
the check each drawn unit passes, as a public Program (`Spec.check`), and what it makes public, each item with its reason
(`Spec.public`). What it does with each `Outcome` is its own code; the service does the rest. The parties: the workers
compute and prove with zero knowledge off, trusted by neither party; the firewall, trusted by the developer, is their only
path to the auditor, and commits, registers and releases the proofs; the challenger, trusted by the auditor, receives
registrations, draws from its own randomness and keeps the record (this module); the verifier, the auditor's Lean
verifier, verifies offline. The design and each protocol's call sequence are in
`note:proofs/20261006T0307Z-draft-proof-service-architecture`.

**The public-items rule.** Everything the auditor observes is a function of the Spec's public items. A registration is
one-stage's record (`one_stage.registration`, v1 or v2), and `check_public` holds each of its observable items to the
rule: it is the service's own, a salted root of a declared Value, or an item the Spec declares in `public`.

* The service's own: `format`, `protocol` and `previous` (one-stage's tags and the log's chain), `scheme` (`SCHEME`: hm96
  rows in frame-v3-sha512 trees, the one hash), `query` (the check's) and `law` (the selection rule's).
* A salted root: one per declared Value, a `roots` entry for a served value and a `registered` entry for a registered one.
* Declared, or refused: `population` (the record's `population`, `program` and `partition`, which are functions of the
  check and N); `population:<template>` (a stratified or work law's strata, one per template); `window` (its provenance);
  `leaf_layer` and `anchors` when not empty; and a value's layout, `leaves:<value>` and `domain:<value>`, with
  `words:<value>` and `word_bits:<value>` for a registered value.

One-stage's record always shows N, each root's leaf count and domain, and the window's provenance, so a Spec over it
declares them; the joint note's §6 step 0 allows that on test prompts, and hiding them is the hidden layout (step 8).

**The draw.** The challenger draws from its own bytes as the Lean verifier's `Flock.Draw.drawWith` does
(`two_stage.draw_first` over `two_stage.OwnRandomness`), with no hash and no derivation, after the receipt. A work draw
carries the challenger's closure map's closure (`two_stage.closure_of`), which the proof proves with the drawn units.
From the selection rule to the draw object it is `selection_law`, `draw.derive` and `select_draw`, which the Lean
verifier's `SampledProofs.Select` ports, held to them by its difftest on fixed coins.

**The two-stage rule** (`TwoStage(replay, check="all")`): the Spec's boundary values (committed by serve time) are
registered first, and the stream draws replay units (RUs) by `replay`; the drawn RUs' interiors (phase `interior`,
committed after the draw) are then registered once (`register_interiors`), as one-stage's record over the proved RUs in
ascending order, all of them drawn (`subset:K` at K = their count), logged after the boundary registration, before any
proof. Every proof unit of a drawn RU is
checked, so there is no second draw. The outcome accepts only when both registrations' audit records do, and its profile
is the first stage's law over the RUs (one-stage's profile of the boundary registration).

**The salt and the registration deadline.** `Stream.salt` issues the epoch's salt coin once, from the challenger's own
coins, and reads its release on the challenger's clock (`salted_at`). `Stream.salt_released` instead records a salt
released elsewhere (PoUW's epoch coin, shared by one stream per level) with its release on the same clock, which can't
be later than the clock's reading now (`released-later`); either is the epoch's one salt. Under `Check.registration_deadline` (seconds on
that clock from the release), `register` reads the receipt (`received_at`) first and refuses a registration received
later than the deadline, or before any salt, as `late-registration`, whatever it holds. That is one visible failure,
never retried, and it ends the epoch: the stream receives no registration after it (`epoch-ended`) and never reissues
the salt (`salted`). It is the rule of `Pouw.AcceptsBy` (`Security/Definitions/Pouw/Game/Deadline.lean`), with `ts` =
`salted_at`, `tr` = `received_at` and `td` = `ts` + the deadline. The readings are the challenger's own: the stream's
epoch log (`Stream.epoch_log`) and each `Registration` carry them, and neither the receipt nor an outcome's record does.

**No retries.** The salt is issued or recorded once (`salt`, `salt_released`). A registration is drawn once, after its receipt (`select`); its
interiors are registered once; its proof is delivered once (`deliver`); its outcome is final (`outcome`), and nothing
redraws. A proof delivered after `Check.proof_deadline` (seconds on the challenger's clock, from the draw) is a reject,
and so is a draw whose proof is not delivered by then: a `Stream` refuses a check without a proof deadline, so every draw
ends. Each failure is one visible symbol, `reject` or `late`, or a late registration's refusal, `late-registration`.

**Sessions over several rounds.** A stream of `rounds` R (1 to `ROUNDS`) holds up to R registrations open at once, so
one proof session can cover the draws of up to R consecutive registrations (rounds), each round's registration its own
tables of the session's statement. `Stream.session` opens it once, after the last covered draw (no covered round is
undrawn), over draws no other session covers and none delivered or final. Each round is committed before its own draw,
and so before every later round's. The first covered round's `Check.proof_deadline` spans the session. `deliver`
delivers the session once, and every covered draw's outcome binds the same session records. Each draw's `outcome`
runs its own audit record, on its own units and its own table's roots, and its record names the session's rounds
(`session`). A late session is `late` for every round it covers.

**The failure budget.** A stream counts its failures, and at the `budget`-th it stops and refuses further registrations.
It holds at most `budget` minus its failures open registrations (and at most `rounds`), so it never ends with more than
`budget` failures, though one session that fails is a failure for each round it covers.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import SpecializedDefinition
from verity.ml.circuits.program import Program
from verity.core.protocols.profile import IntegrityProfile

from .one_stage import audit as A, draw as D, partition as P, registered as G, registration as R, verifier as V
from .one_stage.verdict import Verdict, _Collect
from .two_stage import OwnRandomness, closure_of, draw_first

SCHEME = "frame-v3-sha512/hm96-sha512"
HIDING = "hm96-sha512/"
KINDS = ("structure", "output", "coins", "commitment")
SOURCES = ("parameter", "window")
PHASES = ("epoch", "serve", "interior", "in-deadline")
STREAM = "verity/proof-service/stream/v0"
OUTCOME = "verity/proof-service/outcome/v0"
ACCEPT, REJECT, LATE = "accept", "reject", "late"
#: The most rounds (registrations) one proof session covers.
ROUNDS = 4

_RECORD_KEYS = {"format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots", "leaf_layer",
                "anchors", "window", "previous", "registered"}
_LAYOUT = {"roots": ("leaves", "domain"), "registered": ("leaves", "words", "domain", "word_bits")}
_DIGEST = re.compile(r"^[0-9a-f]{128}$")
_UNIFORM = re.compile(r"^uniform:[1-9][0-9]*$")


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class Public:
    """An item the auditor may see, and why. `kind`: protocol structure the verifier must know (`structure`), the
    protocol's output (`output`), the challenger's coins and what they draw (`coins`), or a declared Value's salted root
    (`commitment`); never a committed value. `source`: fixed before the run (`parameter`), or a public output of the
    check's window statement (`window`)."""

    item: str
    kind: str
    reason: str
    source: str = "parameter"

    def __post_init__(self) -> None:
        if not (isinstance(self.item, str) and self.item):
            raise ValueError("a public item has a name")
        if self.kind not in KINDS:
            raise ValueError(f"public item {self.item!r}: its kind is one of {', '.join(KINDS)}, not {self.kind!r}")
        if self.source not in SOURCES:
            raise ValueError(f"public item {self.item!r}: its source is one of {', '.join(SOURCES)}, not {self.source!r}")
        if not (isinstance(self.reason, str) and self.reason.strip()):
            raise ValueError(f"public item {self.item!r} needs its reason")

    def to_json(self) -> dict[str, str]:
        return {"item": self.item, "kind": self.kind, "reason": self.reason, "source": self.source}


@dataclass(frozen=True)
class Value:
    """A committed value. `name`: its root's port, or the registered value's name. `schema`: its leaves' schema, a hiding
    one (`hm96-sha512/...`); None for a registered value, whose schema the service fixes. `phase`: when it is committed
    (`epoch`, `serve`, `interior` after the draw, or `in-deadline`). `registered`: committed once per deployment and read
    by position. `fresh`: nonce-bound inside a deadline (PoUS's answers)."""

    name: str
    schema: str | None
    phase: str
    registered: bool = False
    fresh: bool = False

    def __post_init__(self) -> None:
        if not (isinstance(self.name, str) and self.name):
            raise ValueError("a value has a name")
        if self.phase not in PHASES:
            raise ValueError(f"value {self.name!r}: its phase is one of {', '.join(PHASES)}, not {self.phase!r}")
        if self.registered:
            if self.schema is not None:
                raise ValueError(f"value {self.name!r}: a registered value's schema is the service's, not {self.schema!r}")
        elif not (isinstance(self.schema, str) and self.schema.startswith(HIDING)):
            raise ValueError(f"value {self.name!r}: its leaves hide it ({HIDING}...), not {self.schema!r}")

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "schema": self.schema, "phase": self.phase, "registered": self.registered,
                "fresh": self.fresh}


@dataclass(frozen=True)
class All:
    """Every proof unit is proved."""

    def to_json(self) -> dict[str, Any]:
        return {"rule": "all"}


@dataclass(frozen=True)
class Law:
    """A one-stage law, in one-stage's syntax (`draw.parse_law`): `subset:K`, `bernoulli:NUM/DEN`, `stratified:K` or
    `work:K`."""

    text: str

    def __post_init__(self) -> None:
        D.parse_law(self.text)

    def to_json(self) -> dict[str, Any]:
        return {"rule": "law", "law": self.text}


@dataclass(frozen=True)
class TwoStage:
    """The two-stage law: replay units drawn by `replay`, then `check` proof units in each drawn one (`"all"`, or a count).
    A `Stream` runs `"all"`, each RU one instance of the check's unit; it refuses a count, which needs RUs of several proof
    units and a second draw after the interiors are registered."""

    replay: Law
    check: str | int

    def __post_init__(self) -> None:
        if not isinstance(self.replay, Law):
            raise ValueError("a two-stage rule's replay draw is a Law")
        if self.check != "all" and not (type(self.check) is int and self.check >= 1):
            raise ValueError(f"a two-stage rule checks 'all' or a count of proof units, not {self.check!r}")

    def to_json(self) -> dict[str, Any]:
        return {"rule": "two-stage", "replay": self.replay.text, "check": self.check}


@dataclass(frozen=True)
class Sequential:
    """A timed session's draw, one index a round (PoUS's audit): `law` is `uniform:k`, over the public item named by
    `population`. A `Stream` refuses it until the memory challenger's timed mode lands (§6 step 5)."""

    law: str
    population: str

    def __post_init__(self) -> None:
        if not _UNIFORM.match(str(self.law)):
            raise ValueError(f"a sequential rule's law is uniform:k, not {self.law!r}")
        if not (isinstance(self.population, str) and self.population):
            raise ValueError("a sequential rule names its population's public item")

    def to_json(self) -> dict[str, Any]:
        return {"rule": "sequential", "law": self.law, "population": self.population}


Selection = All | Law | TwoStage | Sequential


@dataclass(frozen=True)
class Check:
    """The check each drawn unit passes. `program`: the per-unit check Program's digest (`partition.program_digest`).
    `query`: its proof units, one-stage's template query over that Program's Definition. `window`: the window statement's
    Program's SHA-512, proved once per window over the committed layout. `public_outputs`: what the verifier reports that
    the auditor reads. `deadline`: a timed session's (PoUS's rounds). `proof_deadline`: seconds on the challenger's clock
    from the draw; a proof delivered later, or none by then, is a reject (`late`), and a `Stream` refuses a check
    without one. `registration_deadline`: seconds on the challenger's clock from
    the salt coin's release (`Stream.salt`, or `Stream.salt_released`'s `released_at`); a registration received later,
    or before any salt, is refused (`late-registration`) and ends the epoch. The interiors' registration is not held to
    it: it follows the draw, which `proof_deadline` bounds."""

    program: str
    query: Mapping[str, Any]
    window: str | None = None
    public_outputs: tuple[str, ...] = ()
    deadline: Mapping[str, Any] | None = None
    proof_deadline: float | None = None
    registration_deadline: float | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.program, str) and _DIGEST.match(self.program)):
            raise ValueError("a check's program is its SHA-512, 128 hex digits")
        if not isinstance(self.query, Mapping):
            raise ValueError("a check's query is a partition query object")
        if self.window is not None and not (isinstance(self.window, str) and _DIGEST.match(self.window)):
            raise ValueError("a check's window statement is its Program's SHA-512, 128 hex digits")
        if not (isinstance(self.public_outputs, tuple) and all(isinstance(o, str) and o for o in self.public_outputs)
                and len(set(self.public_outputs)) == len(self.public_outputs)):
            raise ValueError("a check's public outputs are distinct names")
        for name, d in (("proof", self.proof_deadline), ("registration", self.registration_deadline)):
            if d is not None and not (isinstance(d, (int, float)) and not isinstance(d, bool) and d > 0):
                raise ValueError(f"a {name} deadline is a positive number of seconds, not {d!r}")

    def to_json(self) -> dict[str, Any]:
        return {"program": self.program, "query": dict(self.query), "window": self.window,
                "public_outputs": list(self.public_outputs), "deadline": None if self.deadline is None else dict(self.deadline),
                "proof_deadline": self.proof_deadline, "registration_deadline": self.registration_deadline}


@dataclass(frozen=True)
class Spec:
    """What a protocol states: `values` (what is committed), `selection` (its selection rule), `check`, `public` (what the
    auditor may see beyond the service's own items, each with its reason) and `context` (the statement context the
    stream's digest binds). A two-stage Spec commits its RUs' boundaries by serve time and their interiors (phase
    `interior`, never registered) after the draw."""

    values: tuple[Value, ...]
    selection: Selection
    check: Check
    public: tuple[Public, ...]
    context: bytes

    def __post_init__(self) -> None:
        if not (isinstance(self.values, tuple) and self.values and all(isinstance(v, Value) for v in self.values)):
            raise ValueError("a spec commits a tuple of one or more Values")
        names = [v.name for v in self.values]
        if len(set(names)) != len(names):
            raise ValueError("a spec's values have distinct names")
        if not isinstance(self.selection, (All, Law, TwoStage, Sequential)):
            raise ValueError(f"a selection rule is All, Law, TwoStage or Sequential, not {self.selection!r}")
        if not isinstance(self.check, Check):
            raise ValueError("a spec's check is a Check")
        if not (isinstance(self.public, tuple) and all(isinstance(p, Public) for p in self.public)):
            raise ValueError("a spec's public items are a tuple of Public")
        items = {p.item: p for p in self.public}
        if len(items) != len(self.public):
            raise ValueError("a spec's public items have distinct names")
        for p in self.public:
            if p.kind == "commitment" and p.item not in names:
                raise ValueError(f"public item {p.item!r}: a commitment item names a declared value")
            if p.source == "window" and self.check.window is None:
                raise ValueError(f"public item {p.item!r}: a window item needs the check's window statement")
        for o in self.check.public_outputs:
            if o not in items or items[o].kind != "output":
                raise ValueError(f"public output {o!r} needs a public item of kind 'output', with its reason")
        if isinstance(self.selection, Sequential) and self.selection.population not in items:
            raise ValueError(f"a sequential rule's population {self.selection.population!r} is a declared public item")
        if isinstance(self.selection, TwoStage):
            interior = [v for v in self.values if v.phase == "interior"]
            if not interior or len(interior) == len(self.values) or any(v.phase == "in-deadline" for v in self.values):
                raise ValueError("a two-stage spec commits its RUs' boundaries by serve time and their interiors "
                                 "(phase 'interior') after the draw")
            if any(v.registered for v in interior):
                raise ValueError("a registered value is committed once per deployment, never as an RU's interior")
        if not isinstance(self.context, bytes):
            raise ValueError("a spec's context is bytes")

    def to_json(self) -> dict[str, Any]:
        return {"values": [v.to_json() for v in self.values], "selection": self.selection.to_json(),
                "check": self.check.to_json(), "public": [p.to_json() for p in self.public], "context": self.context.hex()}

    def digest(self) -> str:
        return _sha512(self.to_json())


@dataclass(frozen=True)
class Exhaustive:
    """Every one of `units` proof units was proved, and the verifier accepted them."""

    units: int


@dataclass(frozen=True)
class Terms:
    """The outcome's named error terms: `sampling`, the profile's delta (None for an exhaustive selection). The proof's and
    binding's terms come with the soundness composition (the joint note's §6 step 13)."""

    sampling: float | None


@dataclass(frozen=True)
class Outcome:
    """A registration's final outcome. `verdict.codes` is empty when accepted, and otherwise the one visible symbol
    (`reject` or `late`). `profile`: the integrity profile of an accepted sampled selection; `exhaustive`: the fact of an
    accepted `All`. `public`: the check's public outputs, as the verifier of record reported them. `record`: the SHA-512
    of the outcome's record, which binds the registration, the draw, the delivered sessions, the audit record and, for a
    draw a `Session` covers, the session's rounds."""

    accepted: bool
    verdict: Verdict
    profile: IntegrityProfile | None
    exhaustive: Exhaustive | None
    terms: Terms
    public: Mapping[str, Any]
    record: str


@dataclass(frozen=True)
class Registration:
    """A registration the challenger accepted: one-stage's `record`, its `receipt` and `digest`, and `order`, its audit's
    index in the stream (a two-stage audit's interiors share it). The receipt's `received` is the record's position in
    the stream's log; the challenger's clock stays its own. Under `Check.registration_deadline`, `salted_at` and
    `received_at` are its readings of the salt's release and of this registration's receipt (None otherwise)."""

    record: Mapping[str, Any]
    receipt: Mapping[str, Any]
    digest: str
    order: int
    salted_at: float | None = None
    received_at: float | None = None


@dataclass(frozen=True)
class Draw:
    """The challenger's one draw for a registration: `law` (its law, expanded over the challenger's own program) and
    `draw`, one-stage's draw object (with a work draw's `closure`), which goes to the firewall."""

    registration: Registration
    law: Mapping[str, Any]
    draw: Mapping[str, Any]

    @property
    def units(self) -> list[int]:
        return list(self.draw["units"])

    @property
    def proved(self) -> list[int]:
        """The units the proof proves: the drawn ones and their closure, ascending (`draw.proved`). Under `TwoStage`, the
        RUs whose interiors are registered, unit i of the interior registration being `proved[i]`."""
        return D.proved(self.draw)


@dataclass(frozen=True)
class Session:
    """One proof session over the draws of consecutive registrations of the stream `stream` (its digest), in order: each
    covered round's registration is its own tables of the session's statement. The challenger opens it once
    (`Stream.session`), after the last covered draw; the first covered draw's proof deadline spans it."""

    stream: str
    draws: tuple[Draw, ...]

    @property
    def orders(self) -> list[int]:
        return [d.registration.order for d in self.draws]


class Refused(Exception):
    """The challenger refused a call, and nothing in the stream changed, except that a late registration
    (`late-registration`) is logged and ends the epoch. `verdict` says why."""

    def __init__(self, verdict: Verdict) -> None:
        super().__init__(", ".join(verdict.codes))
        self.verdict = verdict


class Pending(Exception):
    """The outcome is not final yet: no proof has arrived and its deadline hasn't passed, or the verifier of record hasn't
    run."""


def _refuse(code: str, detail: Any) -> Refused:
    return Refused(Verdict(False, (code,), {code: detail}))


def check_public(spec: Spec, record: Any, values: Sequence[Value] | None = None) -> Verdict:
    """The public-items rule (the module docstring) on a one-stage registration record whose roots are of `values` (by
    default every declared Value; under `TwoStage`, a `Stream` passes the boundary values or the interior ones). Codes:
    `keys` (a field one-stage doesn't define), `roots` (a root or registered entry that is not one of those Values' one
    salted root, such as a per-call root), `unsalted` (a scheme other than `SCHEME`), `query` (not the check's),
    `undeclared` (observable items the Spec doesn't declare, such as a leaf count or a template's population). An entry's
    fields are read in key order, so the detail is a function of the record's canonical JSON, which is all the Lean port
    (`verifier.check_public`, which a `Stream` runs) reads."""
    values = spec.values if values is None else tuple(values)
    c = _Collect()
    if not isinstance(record, Mapping):
        c.fail("keys", "not a record")
        return c.verdict()
    extra = sorted(str(k) for k in set(record) - _RECORD_KEYS)
    if record.get("scheme") != SCHEME:
        c.fail("unsalted", {"scheme": record.get("scheme"), "service": SCHEME})
    if record.get("query") != R.query_digest(spec.check.query):
        c.fail("query", {"registered": record.get("query"), "check": R.query_digest(spec.check.query)})
    need: list[str] = []
    if {"population", "program", "partition"} & set(record):
        need.append("population")
    law = record.get("law")
    if isinstance(law, Mapping) and isinstance(law.get("strata"), list):
        need += [f"population:{s.get('template') if isinstance(s, Mapping) else s}" for s in law["strata"]]
    if "window" in record:
        need.append("window")
    if record.get("leaf_layer") is not None:
        need.append("leaf_layer")
    if record.get("anchors"):
        need.append("anchors")
    wrong: list[Any] = []
    for field_, registered in (("roots", False), ("registered", True)):
        if field_ not in record:
            continue
        entries = record[field_] if isinstance(record[field_], list) else [record[field_]]
        names = {v.name for v in values if v.registered == registered}
        seen: set[Any] = set()
        for e in entries:
            name = e.get("value" if registered else "port") if isinstance(e, Mapping) else None
            if name not in names or name in seen:
                wrong.append(name if name is not None else e)
                continue
            seen.add(name)
            for k in sorted(e):
                if k in _LAYOUT[field_]:
                    need.append(f"{k}:{name}")
                elif k not in ("value" if registered else "port", "root"):
                    extra.append(f"{field_}.{name}.{k}")
    if extra:
        c.fail("keys", extra)
    if wrong:
        c.fail("roots", wrong)
    declared = {p.item for p in spec.public}
    missing = [n for n in dict.fromkeys(need) if n not in declared]
    if missing:
        c.fail("undeclared", missing)
    return c.verdict()


def check_verdicts(verdicts: Mapping[str, Mapping[str, Any]], sessions: Sequence[str], outputs: Sequence[str]) -> Verdict:
    """What an outcome holds the verifiers' verdicts to before the audit record, on the verdicts in key order, the first
    failure only: `pending` (no verdict of record, or one that hasn't run: `accepted` null), `sessions` (verdicts of
    record about other sessions than the delivered `sessions`) and `public` (the check's public `outputs` not reported
    alike by every verdict of record)."""
    of_record = [(k, verdicts[k]) for k in sorted(verdicts) if verdicts[k].get("of_record")]
    if not of_record or any(v.get("accepted") is None for _, v in of_record):
        return Verdict(False, ("pending",), {"pending": "the verifier of record hasn't run"})
    wrong = [k for k, v in of_record if list(v.get("sessions") or ()) != list(sessions)]
    if wrong:
        return Verdict(False, ("sessions",), {"sessions": {"verdicts": wrong, "delivered": list(sessions)}})
    if outputs:
        reported = [v.get("public") for _, v in of_record]
        if not all(isinstance(r, Mapping) and set(r) == set(outputs) for r in reported) or \
                any(_sha512(dict(r)) != _sha512(dict(reported[0])) for r in reported):
            return Verdict(False, ("public",), {"public": {"outputs": list(outputs), "reported": reported}})
    return Verdict(True)


Sampler = Callable[[Mapping[str, Any], int], Sequence[int]]


def _window_key(window: Any) -> bytes | None:
    """The key under which R6 looks for an earlier window equal to `window`. Two plain JSON windows (dicts with string
    keys, lists, strings, integers, booleans and null) that Python's `==` or the Lean's `jeq` finds equal share it: a
    boolean keys as its integer, as `==` reads it. Any other window (a float, a tuple, another type) has none, and
    `Stream._earlier` compares it with the whole log."""
    def plain(x: Any) -> Any:
        if x is None or type(x) in (str, int):
            return x
        if type(x) is bool:
            return int(x)
        if type(x) is list:
            return [plain(v) for v in x]
        if type(x) is dict and all(type(k) is str for k in x):
            return {k: plain(v) for k, v in x.items()}
        raise TypeError(type(x).__name__)

    try:
        return canonical_json(plain(window))
    except (TypeError, RecursionError):
        return None


def _reads_positions(supplied: Mapping[str, Any] | None) -> bool:
    """Whether a supplied value reads a position (`{"position": K}`); a form `registration.check` refuses reads none."""
    try:
        return any(k == "position" for rows in G.sources(supplied or {}).values() for row in rows for k, _ in row)
    except ValueError:
        return False


def selection_law(rule: Selection, n: int) -> dict[str, Any]:
    """The law a selection rule names over `n` proof units, before `draw.derive` expands it over the check's program and
    query: `subset:n` for `All`, a `Law`'s own, and a two-stage rule's `replay` (over its RUs)."""
    if isinstance(rule, All):
        return {"law": "subset", "k": n}
    return D.parse_law((rule.replay if isinstance(rule, TwoStage) else rule).text)


def count_varies(law: Mapping[str, Any]) -> bool:
    """Whether `law` (a `selection_law`) draws a count that varies from draw to draw, so the sessions of one registration
    set up tables of different shapes: `bernoulli` at 0 < p < 1. A work law's closure varies too, but the work draw with
    a producer closure is a planned design, so it is not refused. The Lean verifier's `Run.countVaries` is this."""
    if law.get("law") != "bernoulli":
        return False
    num, _, den = str(law.get("p")).partition("/")
    return D.digits(num) and D.digits(den) and 0 < int(num) < int(den)


def select_draw(rule: Selection, law: Mapping[str, Any], n: int, coins: Callable[[int], bytes], *,
                closure: Mapping[int, Sequence[int]] | None = None, sampler: Sampler | None = None) -> dict[str, Any]:
    """The challenger's draw object over [0, n) under its expanded `law`: every unit for `All`, reading no coins;
    otherwise `sampler`'s draw, or `two_stage.draw_first` on `coins` read in order. With `closure`, the challenger's
    closure map, a work draw carries the drawn units' closure (`two_stage.closure_of`) when it is not empty. Raises
    `Refused` (`closure`) for a closure map on another law or outside [0, n), before any coin is read, and `ValueError`
    where `draw_first` does. The Lean verifier's `Select.select` is this, from the rule on."""
    if closure is not None and not (law["law"] == "work" and isinstance(closure, Mapping) and all(
            type(u) is int and 0 <= u < n and isinstance(cs, Sequence) and all(type(c) is int and 0 <= c < n for c in cs)
            for u, cs in closure.items())):
        raise _refuse("closure", f"a closure map {{unit: [units]}} over [0, {n}) widens a work draw only")
    if isinstance(rule, All):
        units = list(range(n))
    else:
        units = sorted(int(u) for u in (sampler(dict(law), n) if sampler is not None
                                        else draw_first(OwnRandomness(coins), law, n)["units"]))
    obj = {**law, "population": n, "units": units}
    if closure is not None and closure_of(closure, units):
        obj["closure"] = closure_of(closure, units)
    return obj


@dataclass
class _Entry:
    registration: Registration
    law: Mapping[str, Any]
    program: Program
    work_table: Mapping[str, Any] | None = None
    #: what `register` handed R1-R7 besides the record: (ports, leaf_layer, anchors, the log's records, registered,
    #: supplied, pinned)
    copies: tuple[Any, ...] = ()
    read: bytes | None = None
    closure: Mapping[int, Sequence[int]] | None = None
    draw: Draw | None = None
    drawn_at: float | None = None
    interior: Registration | None = None
    interior_draw: dict[str, Any] | None = None
    #: what `register_interiors` handed R1-R7 besides the record: (program, ports, leaf_layer, anchors, the log's records)
    interior_copies: tuple[Any, ...] = ()
    sessions: list[str] | None = None
    outcome: Outcome | None = None
    session: Session | None = None


class Stream:
    """One protocol's stream for one `Spec`, held by the challenger.

    `unit` is the challenger's own copy of the per-unit check's Definition: its Program's digest is the check's, and the
    challenger builds each registration's population program from it and the registration's N (under `TwoStage`, each
    RU is one instance of it). `budget` (b), `delta` (the profile's) and `rounds` (R, the most registrations open at
    once, and so the most one `Session` covers; 1 under `TwoStage`) are public parameters of the stream. `coins` is
    the challenger's own randomness (n bytes a call, read in order by `OwnRandomness`), and `clock` its clock in seconds;
    tests inject both. `sampler`, when given, draws instead of the coins (`flock-verify draw` in a process of its own,
    say): it takes the challenger's law and N, and its draw is held to them like any other.

    A stream runs one epoch with one salt: `salt` issues it from the coins, or `salt_released` records one released
    elsewhere with its release, and under `Check.registration_deadline` every registration is received within the
    deadline of that release, or the epoch ends (`ended`) at the first that isn't. `epoch_log` holds the challenger's
    clock readings.

    As its log grows, a stream keeps what one-stage's R6 and R7 read of it: where each window was logged, and each
    registered value's first registration. It also counts its failures as they come. So a registration's checks read
    the same few records whatever the stream's history (`_earlier`), with the verdicts the whole log gives.
    """

    def __init__(self, spec: Spec, unit: SpecializedDefinition, *, budget: int, delta: float, challenger: str,
                 coins: Callable[[int], bytes] = os.urandom, clock: Callable[[], float] = time.monotonic,
                 sampler: Sampler | None = None, rounds: int = 1) -> None:
        if isinstance(spec.selection, Sequential):
            raise ValueError("the service doesn't run a Sequential selection yet")
        if isinstance(spec.selection, TwoStage) and spec.selection.check != "all":
            raise ValueError("the service runs a TwoStage selection that checks every proof unit of a drawn RU "
                             "(check='all'); a count needs a second draw after the interiors are registered")
        if spec.check.window is not None or spec.check.deadline is not None:
            raise ValueError("the service doesn't prove a window statement or run a timed session yet")
        if spec.check.proof_deadline is None:
            raise ValueError("a stream's check has a proof deadline (Check.proof_deadline), so that a draw whose proof "
                             "never arrives ends as a reject (late)")
        if P.program_digest(Program(unit)) != spec.check.program:
            raise ValueError("the challenger's unit is not the check's program")
        if dict(spec.check.query) != P.template_instance_query(unit):
            raise ValueError("the check's query is not the template query over the challenger's unit")
        if not (type(budget) is int and budget >= 1):
            raise ValueError(f"a failure budget is a positive integer, not {budget!r}")
        if not (isinstance(delta, float) and 0 < delta < 1):
            raise ValueError(f"delta is a float in (0, 1), not {delta!r}")
        if not (type(rounds) is int and 1 <= rounds <= ROUNDS):
            raise ValueError(f"a session covers 1 to {ROUNDS} rounds, not {rounds!r}")
        if rounds > 1 and isinstance(spec.selection, TwoStage):
            raise ValueError("a two-stage stream's session covers one round: its interiors are registered between its "
                             "draw and its proof")
        self.spec, self.unit, self.budget, self.delta, self.challenger = spec, unit, budget, delta, challenger
        self.rounds = rounds
        self._coins, self._clock, self._sampler = coins, clock, sampler
        self.digest = _sha512({"format": STREAM, "spec": spec.digest(), "budget": budget, "delta": delta,
                               "challenger": challenger, **({"rounds": rounds} if rounds > 1 else {})})
        self._entries: list[_Entry] = []
        self._open: dict[int, None] = {}
        self._log: list[dict[str, Any]] = []
        self._windows: dict[bytes, list[int]] = {}
        self._irregular: list[int] = []
        self._first: dict[Any, int] = {}
        self._records: list[dict[str, Any]] = []
        self._failures = 0
        self._epoch_log: list[dict[str, Any]] = []
        self._salted_at: float | None = None
        self._ended: dict[str, Any] | None = None

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def stopped(self) -> bool:
        return self.failures >= self.budget

    @property
    def ended(self) -> bool:
        """The epoch ended: a registration was late (`late-registration`), and the stream receives none from then on."""
        return self._ended is not None

    @property
    def epoch_log(self) -> tuple[dict[str, Any], ...]:
        """The epoch's clock readings, in order: the salt's release, `{"event": "salt", "salt": <hex>, "salted_at"}`
        (with `"external": true` from `salt_released`, `salted_at` being its `released_at`), and under `Check.registration_deadline` each registration received, `{"event": "registration", "order",
        "registration": <digest>, "salted_at", "received_at"}`, or refused as late, `{"event": "late-registration",
        "salted_at", "received_at"}` (`salted_at` null when no salt was issued). A registration refused on what it holds
        is not received, and not logged."""
        return tuple(self._epoch_log)

    @property
    def symbols(self) -> tuple[str, ...]:
        """One symbol per final outcome, in order: `accept`, `reject` or `late`."""
        return tuple(r["symbol"] for r in self._records)

    @property
    def log(self) -> tuple[dict[str, Any], ...]:
        """The registrations received, in order: the log `registration.check`'s R6 reads."""
        return tuple(self._log)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """The outcomes' records, in order; `Outcome.record` is each one's SHA-512."""
        return tuple(self._records)

    def ready(self) -> None:
        """Raises `Refused` when the stream receives no registration now: it has stopped, its epoch ended
        (`epoch-ended`, with the late registration's readings), or it holds as many registrations without a final
        outcome as it may (`open`): `rounds`, and never more than `budget` minus its failures."""
        if self.stopped:
            raise _refuse("stopped", {"failures": self.failures, "budget": self.budget})
        if self._ended is not None:
            raise _refuse("epoch-ended", dict(self._ended))
        if len(self._open) >= min(self.rounds, self.budget - self.failures):
            raise _refuse("open", {"order": next(reversed(self._open))})

    def _unsalted(self) -> None:
        if self._salted_at is not None:
            raise _refuse("salted", {"salted_at": self._salted_at})
        if self._ended is not None:
            raise _refuse("epoch-ended", dict(self._ended))

    def salt(self, n: int = 64) -> bytes:
        """The epoch's salt coin: `n` bytes of the challenger's own coins (`OwnRandomness`, in order with the draws'
        reads), issued once. Its release is read on the challenger's clock (`salted_at`) and logged (`epoch_log`);
        `Check.registration_deadline` counts from it. Raises `Refused`: `salted` (the epoch has a salt, from `salt` or
        `salt_released`: nothing redraws it) or `epoch-ended` (the epoch ended before any salt)."""
        if not (type(n) is int and n >= 1):
            raise ValueError(f"a salt is a positive number of bytes, not {n!r}")
        self._unsalted()
        salt = OwnRandomness(self._coins).read(n)
        self._salted_at = self._clock()
        self._epoch_log.append({"event": "salt", "salt": salt.hex(), "salted_at": self._salted_at})
        return salt

    def salt_released(self, salt: bytes, released_at: float) -> None:
        """The epoch's salt when it was released elsewhere (PoUW's epoch coin, shared by one stream per level):
        `released_at` is its release, read on this stream's clock. It is then the epoch's salt as `salt`'s is, and
        `Check.registration_deadline` counts from `released_at`, so a stream created after the release (a level's, at
        its first registration) still measures from it. Logged with `"external": true`. Raises `ValueError` for a salt
        that isn't non-empty bytes or a release that isn't a finite number, and `Refused`: `salted` (the epoch has a
        salt), `epoch-ended`, or `released-later` (`released_at` after the clock's reading now: a release is never in
        the future)."""
        if not (isinstance(salt, bytes) and salt):
            raise ValueError(f"a released salt is non-empty bytes, not {salt!r}")
        if not (isinstance(released_at, (int, float)) and not isinstance(released_at, bool) and math.isfinite(released_at)):
            raise ValueError(f"a salt's release is a finite reading of the clock, not {released_at!r}")
        self._unsalted()
        now = self._clock()
        if released_at > now:
            raise _refuse("released-later", {"released_at": released_at, "now": now})
        self._salted_at = released_at
        self._epoch_log.append({"event": "salt", "salt": salt.hex(), "salted_at": released_at, "external": True})

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], *, leaf_layer: str | None = None,
                 anchors: Sequence[Any] = (), registered: Mapping[str, int] | None = None,
                 work_table: Mapping[str, Any] | None = None, supplied: Mapping[str, Any] | None = None,
                 pinned: Mapping[str, str] | None = None) -> Registration:
        """Receive a registration: the public-items rule (`check_public`), then one-stage's R1-R7 (`registration.check`),
        each by its verifier of record (`verifier.check_public`, `verifier.check`), against the challenger's own copies:
        the population program it builds from `unit` and the record's N, the check's query, the selection rule's law (a
        two-stage rule's `replay`), `SCHEME`, the stream's log (the records of it R6 and R7 read, `_earlier`), and the
        per-registration copies given here (`ports`, `leaf_layer`, `anchors`, `registered`, `work_table`, `supplied`,
        `pinned`, as `registration.check` takes them). Under `TwoStage` its roots are the boundary values'. Then the receipt. Under
        `Check.registration_deadline` the challenger first reads its clock (`received_at`): a registration received more
        than the deadline after the salt's release, or before any salt, is refused (`late-registration`, with both
        readings) whatever it holds, logged, and ends the epoch; an on-time one carries both readings
        (`Registration.salted_at`, `received_at`) and is logged with them. Raises
        `Refused`: when the stream has stopped or its epoch ended (`epoch-ended`), while it holds as many registrations
        without a final outcome as it may (`open`, `ready`), when it is late, when either check fails, or, after both,
        when the law draws a count that varies (`shapes`, `count_varies`)."""
        self.ready()
        d = self.spec.check.registration_deadline
        received_at = None
        if d is not None:
            received_at = self._clock()
            if self._salted_at is None or not received_at - self._salted_at <= d:
                self._ended = {"salted_at": self._salted_at, "received_at": received_at, "registration_deadline": d}
                self._epoch_log.append({"event": "late-registration", "salted_at": self._salted_at,
                                        "received_at": received_at})
                raise _refuse("late-registration", dict(self._ended))
        v = self._public(record, interior=False)
        if not v.ok:
            raise Refused(v)
        n = record.get("population")
        if not (type(n) is int and n >= 1):
            raise _refuse("population", n)
        program = P.population_program(self.unit, n)
        law = selection_law(self.spec.selection, n)
        earlier = self._earlier(record, supplied)
        v = V.check(record, program, self.spec.check.query, law, SCHEME, ports, leaf_layer, anchors, earlier, None,
                    work_table, registered, supplied, pinned)
        if not v.ok:
            raise Refused(v)
        if count_varies(law):
            raise _refuse("shapes", {"law": law})
        if earlier is self._log:
            earlier = self._log[:]
        salted_at = None if d is None else self._salted_at
        reg = self._log_record(record, len(self._entries), salted_at, received_at)
        self._entries.append(_Entry(reg, D.derive(program, self.spec.check.query, law, work_table), program, work_table,
                                    (ports, leaf_layer, anchors, earlier, registered, supplied, pinned)))
        self._open[reg.order] = None
        if d is not None:
            self._epoch_log.append({"event": "registration", "order": reg.order, "registration": reg.digest,
                                    "salted_at": salted_at, "received_at": received_at})
        return reg

    def _values(self, *, interior: bool) -> tuple[Value, ...]:
        if not isinstance(self.spec.selection, TwoStage):
            return self.spec.values
        return tuple(v for v in self.spec.values if (v.phase == "interior") == interior)

    def _public(self, record: Mapping[str, Any], *, interior: bool) -> Verdict:
        """`check_public` on the boundary values or the interior ones, by the verifier of record
        (`verifier.check_public`)."""
        values = self._values(interior=interior)
        return V.check_public(check_public(self.spec, record, values), record, SCHEME, self.spec.check.query,
                              [(v.name, v.registered) for v in values], [p.item for p in self.spec.public])

    def _log_record(self, record: Mapping[str, Any], order: int, salted_at: float | None = None,
                    received_at: float | None = None) -> Registration:
        reg = Registration(dict(record), R.receipt(record, self.challenger, received=str(len(self._log))), R.digest(record),
                           order, salted_at, received_at)
        at = len(self._log)
        self._log.append(logged := dict(record))
        key = _window_key(logged.get("window"))
        if key is None:
            self._irregular.append(at)
        else:
            self._windows.setdefault(key, []).append(at)
        for e in logged.get("registered") or ():
            self._first.setdefault(e["value"], at)
        return reg

    def _earlier(self, record: Mapping[str, Any], supplied: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """The records of the log that one-stage's R6 and R7 read for `record`, in log order: the last (R6's `previous`),
        every one whose window may equal `record`'s (R6's window: the same `_window_key`, or none), and each registered
        value's first registration (R7's held roots, `registration.held_roots`). Both verifiers of record give the same
        verdict on them as on the whole log. A record whose window has no key, or that has no window, gets the whole
        log, and so does one whose `supplied` values read a position, which follows every record of the log
        (`registered.log_base`)."""
        key = _window_key(record["window"]) if isinstance(record, Mapping) and "window" in record else None
        if key is None or _reads_positions(supplied):
            return self._log
        at = {*self._first.values(), *self._irregular, *self._windows.get(key, ())}
        if self._log:
            at.add(len(self._log) - 1)
        return [self._log[i] for i in sorted(at)]

    def _entry(self, registration: Registration) -> _Entry:
        o = registration.order
        if not (type(o) is int and 0 <= o < len(self._entries)) or self._entries[o].registration != registration:
            raise _refuse("unknown", {"order": o})
        return self._entries[o]

    def select(self, registration: Registration, *, closure: Mapping[int, Sequence[int]] | None = None) -> Draw:
        """The draw, once per registration, from the challenger's own randomness after the receipt: indices in [0, N) by
        the selection rule's law (`select_draw`; `All` draws every unit and reads no coins). `closure` is the challenger's closure map
        (`{unit: [units proved with it]}` over [0, N), from its own layout), for a work law only: the draw carries the
        drawn units' closure (`two_stage.closure_of`), which `draw.proved` and `draw.check_proved` count. The draw of
        record is the verifier of record's (`verifier.select`: the Lean's `Select.select` on the bytes of the coins this
        draw read), unless a `sampler` draws. Raises `Refused` for a registration drawn before (nothing redraws), a
        closure map on another law or outside [0, N) (before any coin is read), the two verifiers drawing differently
        (`disagree`), and a draw that isn't well formed under the law (`draw.check_draw`, by `verifier.check_draw`)."""
        e = self._entry(registration)
        if e.draw is not None:
            raise _refuse("drawn", {"order": registration.order})
        read = bytearray()

        def coins(n: int) -> bytes:
            b = self._coins(n)
            read.extend(b)
            return b
        obj = select_draw(self.spec.selection, e.law, registration.record["population"], coins, closure=closure,
                          sampler=self._sampler)
        if self._sampler is None:
            obj, v = V.select(obj, bytes(read), self.spec.selection.to_json(), e.program, self.spec.check.query,
                              e.work_table, closure)
            if not v.ok:
                raise Refused(v)
        v = V.check_draw(obj, registration.record, e.law)
        if not v.ok:
            raise Refused(v)
        e.draw, e.drawn_at = Draw(registration, e.law, obj), self._clock()
        e.read, e.closure = (None if self._sampler is not None else bytes(read)), closure
        return e.draw

    def register_interiors(self, draw: Draw, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], *,
                           leaf_layer: str | None = None, anchors: Sequence[Any] = ()) -> Registration:
        """Under `TwoStage`: the drawn RUs' interiors, registered once, after the draw and before any proof. `record` is
        one-stage's record over the K = `len(draw.proved)` proved RUs, unit i being RU `draw.proved[i]`, under `subset:K`
        (every unit drawn), with the interior values' roots, `previous` the boundary registration's digest, and a window
        other than the boundary's (R6). It is held to the public-items rule and
        to one-stage's R1-R7 as `register` holds a registration (`ports`, `leaf_layer` and `anchors` are the challenger's
        copies), but not to `Check.registration_deadline` (it follows the draw, which `proof_deadline` bounds), logged
        after the boundary registration, and receipted. Raises `Refused`: `rule` (not a two-stage
        stream), `not-drawn`, `final`, `delivered`, `interior-registered` (a second time), `interior-empty` (no RU was
        drawn), `interior-population` (not the proved RUs), or either check's codes."""
        if not isinstance(self.spec.selection, TwoStage):
            raise _refuse("rule", "interiors are registered under a two-stage selection rule")
        e = self._entry(draw.registration)
        order = draw.registration.order
        if e.draw != draw:
            raise _refuse("not-drawn", {"order": order})
        if e.outcome is not None:
            raise _refuse("final", {"order": order})
        if e.sessions is not None:
            raise _refuse("delivered", {"order": order})
        if e.interior is not None:
            raise _refuse("interior-registered", {"order": order})
        n = len(draw.proved)
        if not n:
            raise _refuse("interior-empty", "a draw of no RUs has no interiors")
        v = self._public(record, interior=True)
        if not v.ok:
            raise Refused(v)
        if not (type(record.get("population")) is int and record["population"] == n):
            raise _refuse("interior-population", {"registered": record.get("population"), "proved": n})
        law = {"law": "subset", "k": n}
        program, earlier = P.population_program(self.unit, n), self._earlier(record)
        v = V.check(record, program, self.spec.check.query, law, SCHEME, ports, leaf_layer, anchors, earlier)
        if not v.ok:
            raise Refused(v)
        if earlier is self._log:
            earlier = self._log[:]
        e.interior, e.interior_draw = self._log_record(record, order), {**law, "population": n, "units": list(range(n))}
        e.interior_copies = (program, ports, leaf_layer, anchors, earlier)
        return e.interior

    def session(self, draws: Sequence[Draw]) -> Session:
        """Open one proof session over `draws`, the draws of 1 to `rounds` consecutive registrations in order, each
        round's registration its own tables of the session's statement: once, after the last covered draw. Its proof
        deadline is the first covered draw's (`Check.proof_deadline` from that draw), for every round it covers. Raises
        `Refused`, changing nothing: `rule` (a two-stage stream), `rounds` (no draw, or more than `rounds`), `unknown`,
        `not-drawn` (a covered registration not drawn yet, or another draw), `final`, `delivered`, `covered` (a draw
        another session covers: nothing reopens it) or `order` (not consecutive registrations in order)."""
        if isinstance(self.spec.selection, TwoStage):
            raise _refuse("rule", "a two-stage stream delivers each draw's proof itself, after its interiors")
        draws = tuple(draws)
        if not 1 <= len(draws) <= self.rounds:
            raise _refuse("rounds", {"rounds": len(draws), "most": self.rounds})
        for d in draws:
            e = self._entry(d.registration)
            order = d.registration.order
            if e.draw != d:
                raise _refuse("not-drawn", {"order": order})
            if e.outcome is not None:
                raise _refuse("final", {"order": order})
            if e.sessions is not None:
                raise _refuse("delivered", {"order": order})
            if e.session is not None:
                raise _refuse("covered", {"order": order, "rounds": e.session.orders})
        s = Session(self.digest, draws)
        if s.orders != list(range(s.orders[0], s.orders[0] + len(draws))):
            raise _refuse("order", {"rounds": s.orders})
        for d in draws:
            self._entries[d.registration.order].session = s
        return s

    def _late(self, e: _Entry) -> bool:
        first = e if e.session is None else self._entries[e.session.orders[0]]
        assert first.drawn_at is not None and self.spec.check.proof_deadline is not None
        return self._clock() - first.drawn_at > self.spec.check.proof_deadline

    @staticmethod
    def _digests_ok(sessions: Any) -> None:
        if not (isinstance(sessions, Sequence) and not isinstance(sessions, str) and sessions
                and all(isinstance(s, str) and _DIGEST.match(s) for s in sessions)):
            raise _refuse("sessions", "one or more session records' SHA-512")

    def _deliver_session(self, session: Session, sessions: Sequence[str]) -> tuple[Outcome, ...] | None:
        if session.stream != self.digest or not session.draws:
            raise _refuse("not-session", {"stream": session.stream})
        entries = []
        for d in session.draws:
            e = self._entry(d.registration)
            order = d.registration.order
            if e.session != session:
                raise _refuse("not-session", {"order": order})
            if e.outcome is not None:
                raise _refuse("final", {"order": order})
            if e.sessions is not None:
                raise _refuse("delivered", {"order": order})
            entries.append(e)
        self._digests_ok(sessions)
        late = self._late(entries[0])
        for e in entries:
            e.sessions = list(sessions)
        return tuple(self._finish(e, LATE, None, {}) for e in entries) if late else None

    def deliver(self, draw: Draw | Session, sessions: Sequence[str]) -> Outcome | tuple[Outcome, ...] | None:
        """The firewall delivers the draw's proof, once: `sessions` are its session records' SHA-512, which the outcome
        binds. A proof after the proof deadline is a reject (`late`), returned at once; otherwise None, and the outcome waits
        on the verifier (`outcome`). Under `TwoStage`, a proof before the interiors are registered is refused
        (`interior`), unless no RU was drawn. A `Session`'s proof is delivered as the session, once, for every draw it
        covers (`covered`, delivering one of them alone): each binds the same `sessions`, and past the first covered
        draw's deadline each is `late`, returned in order. Raises `Refused` (`not-session`) for a session the stream
        didn't open."""
        if isinstance(draw, Session):
            return self._deliver_session(draw, sessions)
        e = self._entry(draw.registration)
        if e.draw != draw:
            raise _refuse("not-drawn", {"order": draw.registration.order})
        if e.outcome is not None:
            raise _refuse("final", {"order": draw.registration.order})
        if e.sessions is not None:
            raise _refuse("delivered", {"order": draw.registration.order})
        if e.session is not None:
            raise _refuse("covered", {"order": draw.registration.order, "rounds": e.session.orders})
        self._digests_ok(sessions)
        late = self._late(e)
        if not late and isinstance(self.spec.selection, TwoStage) and e.interior is None and draw.proved:
            raise _refuse("interior", "the interiors are registered before any proof")
        e.sessions = list(sessions)
        return self._finish(e, LATE, None, {}) if late else None

    def outcome(self, draw: Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                proved_units: Sequence[int] | None = None, instances: int | None = None, served: Sequence[int] | None = None,
                session_roots: Mapping[str, str] | None = None) -> Outcome:
        """The draw's final outcome, once. With no proof delivered: a reject (`late`) once the deadline has passed, and
        `Pending` before. With one: `Pending` until every verdict of record has run (`accepted` not null); each verdict of
        record names the delivered `sessions` and reports the public outputs alike (`check_verdicts`, by its verifier of
        record, `verifier.check_verdicts`); then one-stage's audit record by its verifier of record
        (`verifier.audit_record`; `proved_units` and `instances` default to the draw's, `served` and `session_roots` are
        its, and the registered roots it reads are the verdicts' own) decides accept or reject. Under `TwoStage` with
        interiors, the interior registration's audit record (all of its units proved, by the same statement and
        verdicts) must accept too, and `session_roots` (`{port: root}` from the sessions' statements) is split between
        the two by value. A draw a `Session` covers is decided alone, on the session's delivered records and its own
        audit record: its `proved_units`, `instances`, `served` and `session_roots` are its round's (its own tables'),
        never another round's. On a draw from the coins and audit records its two verifiers agree on, the whole run,
        interiors included, is held to the verifier of record's (`verifier.run`, the Lean's `Run.run` from the
        registration on): where they differ the outcome is a reject, its audit record failing `run`. Raises `Refused` for
        a second outcome, a verdict of record about other sessions, or public outputs the verdicts of record don't report
        alike."""
        e = self._entry(draw.registration)
        if e.draw != draw:
            raise _refuse("not-drawn", {"order": draw.registration.order})
        if e.outcome is not None:
            raise _refuse("final", {"order": draw.registration.order})
        if e.sessions is None:
            if self._late(e):
                return self._finish(e, LATE, None, {})
            raise Pending("no proof delivered")
        outputs = self.spec.check.public_outputs
        v = V.check_verdicts(check_verdicts(verdicts, e.sessions, outputs), verdicts, e.sessions, outputs)
        if v.codes == ("pending",):
            raise Pending(v.detail["pending"])
        if not v.ok:
            raise Refused(v)
        of_record = sorted(k for k in verdicts if verdicts[k].get("of_record"))
        public = dict(verdicts[of_record[0]]["public"]) if outputs else {}
        asked = (proved_units, instances)
        proved = list(proved_units) if proved_units is not None else D.proved(draw.draw)
        instances = len(proved) if instances is None else instances
        roots, interior_roots = session_roots, None
        if session_roots is not None and isinstance(self.spec.selection, TwoStage):
            inner = {v.name for v in self._values(interior=True)}
            roots = {p: r for p, r in session_roots.items() if p not in inner}
            interior_roots = {p: r for p, r in session_roots.items() if p in inner}
        rec = V.audit_record(draw.registration.record, draw.registration.receipt, draw.draw, draw.law, proved, instances,
                             statement, verdicts, self.delta, e.sessions, served, roots)
        inner_rec = None
        if e.interior is not None:
            assert e.interior_draw is not None
            inner_rec = V.audit_record(e.interior.record, e.interior.receipt, e.interior_draw, D.law_of(e.interior_draw),
                                       list(e.interior_draw["units"]), instances, statement, verdicts, self.delta,
                                       e.sessions, None, interior_roots)
        accepted = rec["accepted"] and (inner_rec is None or inner_rec["accepted"])
        if e.read is not None and rec["verifiers"]["agree"] and (inner_rec is None or inner_rec["verifiers"]["agree"]):
            reg = draw.registration
            ports, leaf_layer, anchors, earlier, registered, supplied, pinned = e.copies
            staged = isinstance(self.spec.selection, TwoStage)
            interior = None
            if e.interior is not None:
                assert e.interior_draw is not None
                program, i_ports, i_leaf_layer, i_anchors, i_earlier = e.interior_copies
                interior = {"program": program, "law": D.law_of(e.interior_draw), "ports": i_ports,
                            "leaf_layer": i_leaf_layer, "anchors": i_anchors, "log": i_earlier,
                            "record": e.interior.record, "received": e.interior.receipt["received"]}
            v = V.run({"receipt": dict(reg.receipt), "draw": dict(draw.draw), "read": len(e.read),
                       "interior": None if inner_rec is None else {
                           "receipt": dict(e.interior.receipt),
                           "audit": {k: x for k, x in inner_rec.items() if k != "verifiers"}},
                       "symbol": ACCEPT if accepted else REJECT, "public": public if accepted else {},
                       "audit": {k: x for k, x in rec.items() if k != "verifiers"}},
                      reg.record, e.program, self.spec.check.query,
                      selection_law(self.spec.selection, reg.record["population"]), SCHEME, ports, leaf_layer, anchors,
                      earlier, e.work_table, registered, self.spec.selection.to_json(),
                      [(x.name, x.registered) for x in self._values(interior=False)], [p.item for p in self.spec.public],
                      self.challenger, reg.receipt["received"], e.read, e.closure, e.sessions, verdicts, outputs,
                      statement, self.delta, *asked, served, session_roots,
                      interior_values=[(x.name, x.registered) for x in self._values(interior=True)] if staged else (),
                      interior=interior, supplied=supplied, pinned=pinned)
            if not v.ok:
                accepted = False
                rec = {**rec, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None,
                       "checks": {**rec["checks"], "run": {"ok": False, "codes": list(v.codes), "detail": v.detail}}}
        return self._finish(e, ACCEPT if accepted else REJECT, rec, public if accepted else {}, inner_rec)

    def _finish(self, e: _Entry, symbol: str, audit: Mapping[str, Any] | None, public: Mapping[str, Any],
                interior: Mapping[str, Any] | None = None) -> Outcome:
        reg, draw = e.registration, e.draw
        assert draw is not None
        body = {"format": OUTCOME, "stream": self.digest, "order": reg.order, "registration": reg.digest,
                "receipt": _sha512(dict(reg.receipt)), "draw": _sha512(dict(draw.draw)), "sessions": e.sessions or [],
                "symbol": symbol, "public": dict(public), "audit": None if audit is None else dict(audit),
                "interior": None if e.interior is None else {
                    "registration": e.interior.digest, "receipt": _sha512(dict(e.interior.receipt)),
                    "audit": None if interior is None else dict(interior)}}
        if e.session is not None:
            body["session"] = {"rounds": [{"order": d.registration.order, "registration": d.registration.digest,
                                           "draw": _sha512(dict(d.draw))} for d in e.session.draws]}
        accepted = symbol == ACCEPT
        exhaustive = isinstance(self.spec.selection, All)
        e.outcome = Outcome(
            accepted=accepted,
            verdict=Verdict(True) if accepted else Verdict(False, (symbol,), {symbol: _why(symbol, audit, interior)}),
            profile=A.profile(reg.record, draw.draw, self.delta) if accepted and not exhaustive else None,
            exhaustive=Exhaustive(len(draw.draw["units"])) if accepted and exhaustive else None,
            terms=Terms(None if exhaustive else self.delta),
            public=dict(public),
            record=_sha512(body))
        self._failures += not accepted
        self._open.pop(reg.order, None)
        self._records.append(body)
        return e.outcome


def _why(symbol: str, audit: Mapping[str, Any] | None, interior: Mapping[str, Any] | None = None) -> Any:
    if symbol == LATE:
        return "the proof was not delivered by the proof deadline"
    assert audit is not None
    why = {"checks": [k for k, c in audit["checks"].items() if not c["ok"]],
           "verdicts": {k: v.get("accepted") for k, v in audit["verdicts"].items()}}
    if interior is not None:
        why["interior"] = {"accepted": interior["accepted"],
                           "checks": [k for k, c in interior["checks"].items() if not c["ok"]]}
    return why


__all__ = ["ACCEPT", "All", "Check", "Draw", "Exhaustive", "HIDING", "KINDS", "LATE", "Law", "Outcome", "PHASES", "Pending",
           "Public", "REJECT", "ROUNDS", "Refused", "Registration", "SCHEME", "SOURCES", "Selection", "Sequential", "Session",
           "Spec", "Stream", "Terms", "TwoStage", "Value", "check_public", "check_verdicts", "select_draw", "selection_law"]
