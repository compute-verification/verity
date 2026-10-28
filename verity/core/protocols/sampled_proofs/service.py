"""The proof system, the challenger's side: a protocol's `Spec`, the public-items rule (`check_public`), and a `Stream` of
registrations, draws and outcomes over the one-stage audit (`one_stage`).

A protocol states what it commits (`Spec.values`), which proof blocks need proofs (`Spec.selection`, its selection rule),
the check each drawn proof block passes, as a public Program (`Spec.check`), and what it makes public, each item with its reason
(`Spec.public`). What it does with each `Outcome` is its own code; the service does the rest. The parties: the workers
compute and prove with zero knowledge off, trusted by neither party; the firewall, trusted by the developer, is their only
path to the auditor, and commits, registers and releases the proofs; the challenger, trusted by the auditor, receives
registrations, draws from its own randomness and keeps the record (this module); the verifier, the auditor's Lean
verifier, verifies offline.

**The public-items rule.** Everything the auditor observes is a function of the Spec's public items. A registration is
one-stage's record (`one_stage.registration`, v1 or v2), and `check_public` holds each of its observable items to the
rule: it is the service's own, a salted root of a declared Value, or an item the Spec declares in `public`.

* The service's own: `format`, `protocol` and `previous` (one-stage's tags and the log's chain), `scheme` (`SCHEME`: hm96
  rows in frame-v3-sha512 trees, the one hash), `query` (the check's) and `law` (the selection rule's).
* A salted root: one per declared Value, a `roots` entry for a served value and a `registered` entry for a registered one.
  A tree over roots (`ROOTS`, a row-table window's `row-table` port) is salted through its trees' hiding leaves, the
  schema its Value names (`over`).
* Declared, or refused: `population` (the record's `population`, `program` and `partition`, which are functions of the
  check and N); `population:<template>` (a stratified or work sampling design's strata, one per template); `window` (its provenance);
  `leaf_layer` and `anchors` when not empty; and a value's layout, `leaves:<value>` and `domain:<value>`, with
  `words:<value>` and `word_bits:<value>` for a registered value.

One-stage's record always shows N, each root's leaf count and domain, and the window's provenance, so a Spec over it
declares them; hiding them waits on the hidden layout (`one_stage.hidden_draw`).
The `Stream` runs `check_public` by the verifier of record before one-stage's R1 to R7, reading an entry's fields in key
order; its codes are `keys`, `roots`, `unsalted`, `query` and `undeclared`.

**A window of templates.** A `Stream` takes one template, or a window's several (a TP rank's served window, every
provable family of its step): the check's program is each template's digest and its query `Q_template_instances`, in
the window's order, and each registration's population program is the templates' mixed population at the challenger's
own counts (`register(counts=)`), drawn once by the Spec's sampling design, `stratified:K` with one stratum per template. The
verifier reads one program and one query either way, so a window's registration, draw and run are its
`check`, `Select.select` and `Run.run` as they are for one template's.

**The draw.** `register` logs the record and its receipt, whose `received` is the record's position in the stream's log.
The challenger then draws once (`select`) from its coins as the Lean verifier's `Flock.Draw.drawWith` does
(`coins.draw_first` over `coins.OwnRandomness`), with no hash and no derivation, or by the stream's `sampler`;
either draw is held to `draw.check_draw`. The coins are the operating system's bytes as the Lean verifier
reads them (`verifier.coins`: `IO.getRandomBytes`, assumption A3, `uniform/io-getrandombytes`); the Python never reads
them from the OS itself. A work draw carries the challenger's closure map's closure (`select(closure=)`, a map over
`[0, N)`, refused under another sampling design or out of range before any coin is read; `coins.closure_of`), which the proof
proves with the drawn proof blocks.
From the selection rule to the draw object it is `selection_law`, `draw.derive` and `select_draw`, which the Lean
verifier's `SampledProofs.Select` ports, held to them by its difftest on fixed coins.

**The salt and the registration deadline.** `Stream.salt` issues the epoch's salt coin once, from the challenger's own
coins, and reads its release on the challenger's clock (`salted_at`). `Stream.salt_released` instead records a salt
released elsewhere (PoUW's epoch coin, shared by one stream per level) with its release on the same clock, which can't
be later than the clock's reading now (`released-later`); either is the epoch's one salt. Under `Check.registration_deadline` (seconds on
that clock from the release), `register` reads the receipt (`received_at`) first and refuses a registration received
later than the deadline, or before any salt, as `late-registration`, whatever it holds. That is one visible failure,
never retried, and it ends the epoch: the stream receives no registration after it (`epoch-ended`) and never reissues
the salt (`salted`). It is the rule of `Pouw.AcceptsBy` (`Security/Definitions/Pouw/Game/Deadline.lean`), with `ts` =
`salted_at`, `tr` = `received_at` and `td` = `ts` + the deadline. The readings are the challenger's own: the stream's
epoch log (`Stream.epoch_log`) and each `WindowCommitmentRecord` carry them, and neither the receipt nor an outcome's
record does.
`salt(n=64)` is `n` bytes of the challenger's coins, read in order with the draws'. A level's stream created at its
first registration still measures from the epoch's release. A salt that isn't non-empty bytes, or a release that isn't a
finite number, raises `ValueError`, and after the epoch ends a salt of either kind is refused (`epoch-ended`). The epoch
log holds `{"event": "salt", "salt", "salted_at"}` (with `"external": true` from `salt_released`), then each
`{"event": "registration", "order", "registration", "salted_at", "received_at"}` or `{"event": "late-registration",
"salted_at", "received_at"}`. With no registration deadline `register` reads no clock and logs nothing to the epoch log.

**No retries.** The salt is issued or recorded once (`salt`, `salt_released`). A registration is drawn once, after its receipt (`select`); its
proof is delivered once (`deliver`); its outcome is final (`outcome`), and nothing
redraws. A proof delivered after `Check.proof_deadline` (seconds on the challenger's clock, from the draw) is a reject,
and so is a draw whose proof is not delivered by then: a `Stream` refuses a check without a proof deadline, so every draw
ends. Each failure is one visible symbol, `reject` or `late`, or a late registration's refusal, `late-registration`;
an outcome the fleet's halt ends is `halted`.
A second call of any of these is refused. An outcome is pending until every verifier has run, and each verdict
of record names the delivered sessions and reports the public outputs alike (`check_verdicts`: `pending`, `sessions`,
`public`, by the verifier).

**Sessions over several rounds.** A stream of `rounds` R (1 to `ROUNDS`) holds up to R registrations
open at once, so one proof session can cover the draws of up to R consecutive registrations (rounds), each round's
registration its own tables of the session's statement. `Stream.session` opens it once, after the last covered draw
(no covered round is undrawn), over draws no other session covers and none delivered or final. Each round is committed
before its own draw, and so before every later round's. The first covered round's `Check.proof_deadline` spans the
session. `deliver` delivers the session once, and every covered draw's outcome binds the same session records. Each
draw's `outcome` runs its own audit record, on its own proof blocks and its own table's roots, and its record names the
session's rounds (`session`). A late session is `late` for every round it covers. C-Flock's verifier refuses
such a session (`verity/ml/flock/verifier/PROTOCOL.md` §16.12); `Stream.session` has its refusals.

**The failure budget.** A stream counts its failures, and at the `budget`-th it stops and refuses further registrations.
It holds at most `budget` minus its failures open registrations (and at most `rounds`), so it never ends with more than
`budget` failures, though one session that fails is a failure for each round it covers.

**The fleet's halt (R5-all).** A `Fleet` holds the challenger's streams across protocols and levels, each attached to
its operator before its first registration. At the failure that brings any stream of an operator to its budget, the
fleet halts the operator: every stream attached to it halts at once (`Stream.halted`), each of its registrations
without a final outcome ends as `halted` (with no failure of its own; outcomes already final stay), and each refuses
every call from then on (`halted`), so nothing of the operator is accepted or registered after its halt. A `late` counts
toward the budget as a `reject` does. The fleet's rules (`_FleetControl`) decide by the verifier, the Lean's
`SampledProofs.Fleet`, and a halted stream's by `SampledProofs.Stream`'s `halt`.

**The log's history.** A registration's checks get the log's last record (R6's `previous`), the records whose window may
equal the registration's (R6's window) and each registered value's first registration (R7's held roots)
(`Stream._earlier`), so both verifiers give the whole log's verdict and the work doesn't grow with the stream.
The exceptions are windows that aren't plain JSON: a record whose window holds a float or a tuple is read by every check,
and one whose own window does gets the whole log.

**The records.** The stream's digest is the SHA-512 of `{"format": "verity/proof-service/stream/v0", "spec": <the Spec's
digest>, "budget", "delta", "challenger"}`, plus `"rounds"` when it is more than 1 and `"prover"`, the identity the
challenger registered the stream's prover under, when the stream has one (`Stream(prover=)`); such a stream binds each
registration to itself and the prover (`registration.check`'s `binding`). An outcome's record is
`{"format": "verity/proof-service/outcome/v0", "stream", "order", "registration", "receipt", "draw", "sessions",
"symbol", "public", "audit"}`: the receipt and the draw as the SHA-512 of their canonical JSON, `audit`
one-stage's audit record (null for `late` and `halted`). A `halted` outcome adds `"halt"`, the halt's cause, and
its `draw` is null when the registration was not drawn. `Outcome.record` is its SHA-512.

**Not built yet.** A stream refuses a window statement, a timed session and a check with no proof deadline. After R1 to R7, `register` refuses a sampling design whose drawn count varies (`shapes`: `bernoulli`
at `0 < p < 1`); a work draw's closure varies too and is not refused (a planned design).
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import SpecializedDefinition
from verity.ml.circuits.program import Program
from verity.core.protocols.profile import IntegrityProfile

from .one_stage import audit as A, draw as D, partition as P, registered as G, registration as R, verifier as V
from .one_stage.verdict import Verdict, _Collect
from .coins import ChallengerCoins, OwnRandomness, closure_of, draw_first

SCHEME = "frame-v3-sha512/hm96-sha512"
HIDING = "hm96-sha512/"
#: A tree over the roots of hiding trees (`row_table.ROOTS_SCHEMA`); a Value of it names its leaves' schema (`over`).
ROOTS = "frame-v3-sha512/root/v1"
KINDS = ("structure", "output", "coins", "commitment")
SOURCES = ("parameter", "window")
PHASES = ("epoch", "serve", "in-deadline")
STREAM = "verity/proof-service/stream/v0"
OUTCOME = "verity/proof-service/outcome/v0"
ACCEPT, REJECT, LATE, HALTED = "accept", "reject", "late", "halted"
#: The most rounds (registrations) one proof session covers.
ROUNDS = 4

_RECORD_KEYS = {"format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots", "leaf_layer",
                "anchors", "window", "previous", "registered"}
_LAYOUT = {"roots": ("leaves", "domain"), "registered": ("leaves", "words", "domain", "word_bits")}
_DIGEST = re.compile(r"^[0-9a-f]{128}$")


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
    one (`hm96-sha512/...`), or `ROOTS` for a tree over the roots of trees whose leaves hide it (a row-table window's
    `row-table` port: one root per step, each over its tables' roots); None for a registered value, whose schema the
    service fixes. `over`: a `ROOTS` value's hiding leaf schema, the one its inner trees' leaves have, and only a `ROOTS`
    value's; the Spec's digest binds it. `phase`: when it is committed (`epoch`, `serve` or `in-deadline`). `registered`:
    committed once per deployment and read by position. `fresh`: nonce-bound inside a deadline (PoUS's answers)."""

    name: str
    schema: str | None
    phase: str
    registered: bool = False
    fresh: bool = False
    over: str | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.name, str) and self.name):
            raise ValueError("a value has a name")
        if self.phase not in PHASES:
            raise ValueError(f"value {self.name!r}: its phase is one of {', '.join(PHASES)}, not {self.phase!r}")
        if self.registered:
            if self.schema is not None:
                raise ValueError(f"value {self.name!r}: a registered value's schema is the service's, not {self.schema!r}")
        elif self.schema == ROOTS:
            if not (isinstance(self.over, str) and self.over.startswith(HIDING)):
                raise ValueError(f"value {self.name!r}: a tree over roots ({ROOTS}) names the hiding leaf schema its "
                                 f"trees' leaves have (over={HIDING}...), not {self.over!r}")
        elif not (isinstance(self.schema, str) and self.schema.startswith(HIDING)):
            raise ValueError(f"value {self.name!r}: its leaves hide it ({HIDING}...), or it is a tree over roots "
                             f"({ROOTS}, over={HIDING}...), not {self.schema!r}")
        if self.over is not None and self.schema != ROOTS:
            raise ValueError(f"value {self.name!r}: only a tree over roots ({ROOTS}) has an over schema, not "
                             f"{self.schema!r}")

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "schema": self.schema, "phase": self.phase, "registered": self.registered,
                "fresh": self.fresh, **({} if self.over is None else {"over": self.over})}


@dataclass(frozen=True)
class All:
    """Every proof block is proved."""

    def to_json(self) -> dict[str, Any]:
        return {"rule": "all"}


@dataclass(frozen=True)
class Law:
    """A one-stage sampling design: the rate sampling design `rate:NUM/DEN` or `rate:NUM/DEN:T` (`draw.parse_rate`), what a new Spec states, or
    one in one-stage's syntax (`draw.parse_law`): `subset:K`, `bernoulli:NUM/DEN`, `stratified:K` or `work:K`. A rate
    draws `subset:K` at K = ceil(NUM N / DEN) rounded up to a multiple of the T tables (`draw.rate_count`) over the N
    proof blocks each registration brings, so every proof block is drawn with probability at least NUM/DEN however large the prover
    makes N; a fixed `subset:K` draws each with K/N."""

    text: str

    def __post_init__(self) -> None:
        if D.parse_rate(self.text) is None:
            try:
                D.parse_law(self.text)
            except ValueError:
                raise ValueError("a law is rate:NUM/DEN or rate:NUM/DEN:T (0 < NUM <= DEN, T >= 1), subset:K, stratified:K, "
                                 f"work:K or bernoulli:NUM/DEN, not {self.text!r}") from None

    def to_json(self) -> dict[str, Any]:
        return {"rule": "law", "law": self.text}


Selection = All | Law

RETIRED_TWO_STAGE = ("two-stage sampling is retired (ASK-25: everything is committed at serve time); a selection rule is "
                     "all or a law")
"""The refusal of a selection rule that names two-stage sampling, here and in the Lean verifier (`Select.retiredTwoStage`)."""


@dataclass(frozen=True)
class Check:
    """The check each drawn proof block passes. `program`: the proof-block check Program's digest (`partition.program_digest`), or,
    for a window of templates (`Stream`'s), the tuple of each template's in the window's order. `query`: its proof blocks,
    one-stage's template query over that Program's Definition (`Q_template_instance`), or over the window's templates in
    its order (`Q_template_instances`). `window`: the window statement's
    Program's SHA-512, proved once per window over the committed layout. `public_outputs`: what the verifier reports that
    the auditor reads. `deadline`: a timed session's (PoUS's rounds). `proof_deadline`: seconds on the challenger's clock
    from the draw; a proof delivered later, or none by then, is a reject (`late`), and a `Stream` refuses a check
    without one. `registration_deadline`: seconds on the challenger's clock from
    the salt coin's release (`Stream.salt`, or `Stream.salt_released`'s `released_at`); a registration received later,
    or before any salt, is refused (`late-registration`) and ends the epoch."""

    program: str | tuple[str, ...]
    query: Mapping[str, Any]
    window: str | None = None
    public_outputs: tuple[str, ...] = ()
    deadline: Mapping[str, Any] | None = None
    proof_deadline: float | None = None
    registration_deadline: float | None = None

    def __post_init__(self) -> None:
        digests = self.program if isinstance(self.program, tuple) else (self.program,)
        if not (all(isinstance(d, str) and _DIGEST.match(d) for d in digests)
                and (isinstance(self.program, str) or len(set(digests)) == len(digests) >= 1)):
            raise ValueError("a check's program is its SHA-512, 128 hex digits, or a window's: a non-empty tuple of "
                             "distinct ones, one a template")
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
        return {"program": list(self.program) if isinstance(self.program, tuple) else self.program,
                "query": dict(self.query), "window": self.window,
                "public_outputs": list(self.public_outputs), "deadline": None if self.deadline is None else dict(self.deadline),
                "proof_deadline": self.proof_deadline, "registration_deadline": self.registration_deadline}


def _names_two_stage(rule: Any) -> bool:
    if isinstance(rule, Mapping):
        return rule.get("rule") == "two-stage"
    return isinstance(rule, str) and rule.partition(":")[0] == "two-stage"


@dataclass(frozen=True)
class Spec:
    """What a protocol states: `values` (what is committed), `selection` (its selection rule), `check`, `public` (what the
    auditor may see beyond the service's own items, each with its reason) and `context` (the statement context the
    stream's digest binds)."""

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
        if _names_two_stage(self.selection):
            raise ValueError(RETIRED_TWO_STAGE)
        if not isinstance(self.selection, (All, Law)):
            raise ValueError(f"a selection rule is All or Law, not {self.selection!r}")
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
        if not isinstance(self.context, bytes):
            raise ValueError("a spec's context is bytes")

    def to_json(self) -> dict[str, Any]:
        return {"values": [v.to_json() for v in self.values], "selection": self.selection.to_json(),
                "check": self.check.to_json(), "public": [p.to_json() for p in self.public], "context": self.context.hex()}

    def digest(self) -> str:
        return _sha512(self.to_json())


@dataclass(frozen=True)
class Exhaustive:
    """Every one of `units` proof blocks was proved, and the verifier accepted them."""

    units: int


@dataclass(frozen=True)
class Terms:
    """The outcome's named error terms: `sampling`, the profile's delta (None for an exhaustive selection). The proof's and
    binding's terms are not here: the soundness composition states them."""

    sampling: float | None


@dataclass(frozen=True)
class Outcome:
    """A registration's final outcome. `verdict.codes` is empty when accepted, and otherwise the one visible symbol
    (`reject`, `late`, or `halted`, whose detail is the halt's cause). `profile`: the integrity profile of an accepted
    sampled selection; `exhaustive`: the fact of an accepted `All`. `public`: the check's public outputs, as the verifier
    of record reported them. `record`: the SHA-512 of the outcome's record, which binds the registration, the draw, the
    delivered sessions, the audit record and, for a draw a `ProofRun` covers, the session's rounds."""

    accepted: bool
    verdict: Verdict
    profile: IntegrityProfile | None
    exhaustive: Exhaustive | None
    terms: Terms
    public: Mapping[str, Any]
    record: str


_O = TypeVar("_O", Outcome, tuple[Outcome, ...])


@dataclass(frozen=True)
class WindowCommitmentRecord:
    """A window's commitment record the challenger admitted: one-stage's `record`, its `receipt` and `digest`, and
    `order`, its audit's index in the stream. It is Verification's, not an entry of the registration log, which registers
    only the site, θ and the builder. The receipt's `received` is the record's position in
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
    """The challenger's one draw for a registration: `law` (its sampling design, expanded over the challenger's own program) and
    `draw`, one-stage's draw object (with a work draw's `closure`), which goes to the firewall."""

    registration: WindowCommitmentRecord
    law: Mapping[str, Any]
    draw: Mapping[str, Any]

    @property
    def units(self) -> list[int]:
        return list(self.draw["units"])

    @property
    def proved(self) -> list[int]:
        """The proof blocks the proof proves: the drawn ones and their closure, ascending (`draw.proved`)."""
        return D.proved(self.draw)


@dataclass(frozen=True)
class ProofRun:
    """One proof run (a session) over the draws of consecutive registrations of the stream `stream` (its digest), in order:
    each covered round's registration is its own tables of the session's statement. The challenger opens it once
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
    """The outcome is not final yet: no proof has arrived and its deadline hasn't passed, or the verifier hasn't
    run."""


def _refuse(code: str, detail: Any) -> Refused:
    return Refused(Verdict(False, (code,), {code: detail}))


def check_public(spec: Spec, record: Any, values: Sequence[Value] | None = None) -> Verdict:
    """The public-items rule (the module docstring) on a one-stage registration record whose roots are of `values` (by
    default every declared Value). Codes:
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
    """The sampling design a selection rule names over `n` proof blocks, before `draw.derive` expands it over the check's program and
    query: `subset:n` for `All` and a `Law`'s own. A rate sampling design is `subset:K` at K = `draw.rate_count` over `n`, and raises
    `Refused` (`rate`) unless 1 <= K <= n. The Lean verifier's `Select.Rule.over`, then `Select.Rule.lawOver`, is this."""
    if isinstance(rule, All):
        return {"law": "subset", "k": n}
    text = rule.text
    rate = D.parse_rate(text)
    if rate is None:
        return D.parse_law(text)
    k = D.rate_count(*rate, n)
    if not 1 <= k <= n:
        raise _refuse("rate", {"law": text, "population": n, "k": k})
    return {"law": "subset", "k": k}


def count_varies(law: Mapping[str, Any]) -> bool:
    """Whether `law` (a `selection_law`) draws a count that varies from draw to draw, so the sessions of one registration
    set up tables of different shapes: `bernoulli` at 0 < p < 1. A work sampling design's closure varies too, but the work draw with
    a producer closure is a planned design, so it is not refused. The Lean verifier's `Run.countVaries` is this. A Spec
    that wants Bernoulli's public rate at a fixed count states the rate sampling design, `rate:NUM/DEN[:T]`."""
    if law.get("law") != "bernoulli":
        return False
    num, _, den = str(law.get("p")).partition("/")
    return D.digits(num) and D.digits(den) and 0 < int(num) < int(den)


def select_draw(rule: Selection, law: Mapping[str, Any], n: int, coins: Callable[[int], bytes], *,
                closure: Mapping[int, Sequence[int]] | None = None, sampler: Sampler | None = None) -> dict[str, Any]:
    """The challenger's draw object over [0, n) under its expanded `law`: every proof block for `All`, reading no coins;
    otherwise `sampler`'s draw, or `coins.draw_first` on `coins` read in order. With `closure`, the challenger's
    closure map, a work draw carries the drawn proof blocks' closure (`coins.closure_of`) when it is not empty. Raises
    `Refused` (`closure`) for a closure map on another sampling design or outside [0, n), before any coin is read, and `ValueError`
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
    registration: WindowCommitmentRecord
    law: Mapping[str, Any]
    program: Program
    work_table: Mapping[str, Any] | None = None
    #: what `register` handed R1-R7 besides the record: (ports, leaf_layer, anchors, the log's records, registered,
    #: supplied, pinned, schemas, window_root, held)
    copies: tuple[Any, ...] = ()
    read: bytes | None = None
    closure: Mapping[int, Sequence[int]] | None = None
    draw: Draw | None = None
    drawn_at: float | None = None
    sessions: list[str] | None = None
    outcome: Outcome | None = None
    session: ProofRun | None = None


def _reading(x: Any) -> float | None:
    return None if x is None else float(x)


def _nat(x: Any) -> bool:
    return type(x) is int and x >= 0


def _float(x: Any, k: str) -> float:
    """A reading as the verifier reads it: a JSON number, as the double nearest it."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{k} is a reading of the clock")
    try:
        return float(x)
    except OverflowError:
        raise ValueError(f"{k} is a reading of the clock") from None


def _token(x: Any, k: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{k} is a token")
    return x


def _order(r: WindowCommitmentRecord) -> int | None:
    return r.order if _nat(r.order) else None


def _registration_token(r: WindowCommitmentRecord) -> str:
    """The token `_Control` and `SampledProofs.Stream` know a registration by: equal exactly when the registrations are."""
    return _sha512({"record": dict(r.record), "receipt": dict(r.receipt), "digest": r.digest, "order": r.order,
                    "salted_at": r.salted_at, "received_at": r.received_at})


def _draw_token(d: Draw) -> str:
    return _sha512({"registration": _registration_token(d.registration), "law": dict(d.law), "draw": dict(d.draw)})


def _covered(d: Draw) -> list[Any]:
    return [_order(d.registration), _registration_token(d.registration), _draw_token(d)]


def _session_digests(sessions: Any) -> list[str | None] | None:
    if not isinstance(sessions, Sequence) or isinstance(sessions, str):
        return None
    return [s if isinstance(s, str) else None for s in sessions]


class _Stop(Exception):
    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision


def _no(code: str, detail: Any = None) -> NoReturn:
    raise _Stop({"refused": [code], "detail": detail})


def _need(op: Mapping[str, Any], k: str) -> Any:
    if k not in op:
        raise _Stop({"unchecked": k})
    return op[k]


#: The ops of a stream's rules, each `_Control`'s method of that name.
_OPS = ("ready", "salt", "salt_released", "register", "select", "session", "deliver", "deliver_session", "outcome",
        "halt")
#: What the stream's rules hold of an empty stream.
_EMPTY = {"size": 0, "opened": [], "failures": 0, "salted_at": None, "ended": None, "halted": None, "entries": []}


class _Control:
    """A stream's own rules, the reference for the verifier's (`SampledProofs.Stream`, whose docstring has the
    state and the ops): the failure budget, the registrations open at once, the epoch's salt and registration deadline,
    and each registration's once-only orders (drawn, session, delivered, final) and proof deadline, on what
    they hold of the stream that one op reads (`_step`). A refusal's or a pending outcome's `detail` is the stream's
    own, which the Lean doesn't write. `ValueError` is an op or a state the Lean doesn't read."""

    def __init__(self, params: Mapping[str, Any], state: Mapping[str, Any]) -> None:
        self.params = params
        self.deadline = float(params["proof_deadline"])
        rd = params["registration_deadline"]
        self.registration_deadline = None if rd is None else float(rd)
        self.s = copy.deepcopy(dict(state))
        self.at = {e["order"]: e for e in self.s.pop("entries")}

    def state(self) -> dict[str, Any]:
        return {**self.s, "entries": list(self.at.values())}

    def _ended(self) -> dict[str, Any]:
        return {**self.s["ended"], "registration_deadline": self.params["registration_deadline"]}

    def _ready(self, op: Mapping[str, Any] | None = None) -> dict[str, Any]:
        s, budget = self.s, self.params["budget"]
        if s["failures"] >= budget:
            _no("stopped", {"failures": s["failures"], "budget": budget})
        if s["ended"] is not None:
            _no("epoch-ended", self._ended())
        if len(s["opened"]) >= min(self.params["rounds"], budget - s["failures"]):
            _no("open", {"order": s["opened"][-1]})
        return {"ok": None}

    def _unsalted(self) -> None:
        if self.s["salted_at"] is not None:
            _no("salted", {"salted_at": self.s["salted_at"]})
        if self.s["ended"] is not None:
            _no("epoch-ended", self._ended())

    def _salt(self, op: Mapping[str, Any]) -> dict[str, Any]:
        self._unsalted()
        at = _need(op, "at")
        _float(at, "at")
        self.s["salted_at"] = at
        return {"ok": None}

    def _salt_released(self, op: Mapping[str, Any]) -> dict[str, Any]:
        self._unsalted()
        released = _need(op, "released_at")
        r = _float(released, "released_at")
        now = _float(_need(op, "now"), "now")
        if r > now:
            _no("released-later", {"released_at": released, "now": now})
        self.s["salted_at"] = released
        return {"ok": None}

    def _held(self, order: int) -> dict[str, Any]:
        if order not in self.at:
            raise ValueError(f"the state holds no entry {order}")
        return self.at[order]

    def _entry(self, order: Any, registration: str) -> dict[str, Any]:
        if not (_nat(order) and order < self.s["size"]):
            _no("unknown", {"order": order})
        e = self._held(order)
        if e["registration"] != registration:
            _no("unknown", {"order": order})
        return e

    def _entry_of(self, op: Mapping[str, Any]) -> dict[str, Any]:
        return self._entry(op.get("order"), _token(op.get("registration"), "registration"))

    def _drawn(self, op: Mapping[str, Any]) -> dict[str, Any]:
        e = self._entry_of(op)
        if e["draw"] != _token(op.get("draw"), "draw"):
            _no("not-drawn", {"order": op["order"]})
        return e

    @staticmethod
    def _codes(op: Mapping[str, Any], k: str) -> list[str] | None:
        cs = _need(op, k)
        if cs is not None and not (isinstance(cs, list) and all(isinstance(c, str) for c in cs)):
            raise ValueError(f"{k} are null or codes")
        return cs

    def _checks(self, op: Mapping[str, Any]) -> None:
        if (cs := self._codes(op, "checks")) is not None:
            raise _Stop({"refused": list(cs)})

    def _register(self, op: Mapping[str, Any]) -> dict[str, Any]:
        self._ready()
        d = self.registration_deadline
        if d is not None:
            received = _need(op, "received_at")
            t, at = _float(received, "received_at"), self.s["salted_at"]
            if at is None or not t - float(at) <= d:
                self.s["ended"] = {"received_at": received, "salted_at": at}
                return {"refused": ["late-registration"], "detail": self._ended()}
        self._checks(op)
        registration = _token(_need(op, "registration"), "registration")
        order = self.s["size"]
        self.at[order] = {"order": order, "registration": registration, "draw": None, "proved": 0, "drawn_at": None,
                          "delivered": False, "session": None, "final": False}
        self.s["size"] = order + 1
        self.s["opened"].append(order)
        return {"ok": order}

    def _select(self, op: Mapping[str, Any]) -> dict[str, Any]:
        e = self._entry_of(op)
        if e["draw"] is not None:
            _no("drawn", {"order": op["order"]})
        self._checks(op)
        draw = _token(_need(op, "draw"), "draw")
        if not _nat(proved := _need(op, "proved")):
            raise ValueError("proved is a count")
        at = _need(op, "at")
        _float(at, "at")
        e.update(draw=draw, proved=proved, drawn_at=at)
        return {"ok": None}

    @staticmethod
    def _draws(op: Mapping[str, Any]) -> list[list[Any]]:
        ds = op.get("draws")
        if not isinstance(ds, list):
            raise ValueError("draws are a list")
        if not all(isinstance(d, list) and len(d) == 3 and isinstance(d[1], str) and isinstance(d[2], str) for d in ds):
            raise ValueError("a covered draw is [order, registration, draw]")
        return [[d[0] if _nat(d[0]) else None, d[1], d[2]] for d in ds]

    def _open_entry(self, e: Mapping[str, Any], order: Any) -> None:
        if e["final"]:
            _no("final", {"order": order})
        if e["delivered"]:
            _no("delivered", {"order": order})
        if e["session"] is not None:
            _no("covered", {"order": order, "rounds": [d[0] for d in e["session"]]})

    def _session(self, op: Mapping[str, Any]) -> dict[str, Any]:
        draws = self._draws(op)
        if not 1 <= len(draws) <= self.params["rounds"]:
            _no("rounds", {"rounds": len(draws), "most": self.params["rounds"]})
        for order, registration, draw in draws:
            e = self._entry(order, registration)
            if e["draw"] != draw:
                _no("not-drawn", {"order": order})
            self._open_entry(e, order)
        orders = [d[0] for d in draws]
        if orders != list(range(orders[0], orders[0] + len(orders))):
            _no("order", {"rounds": orders})
        for o in orders:
            self.at[o]["session"] = [list(d) for d in draws]
        return {"ok": orders}

    @staticmethod
    def _sessions(op: Mapping[str, Any]) -> None:
        ss = op.get("sessions")
        if not (isinstance(ss, list) and ss and all(isinstance(s, str) and _DIGEST.match(s) for s in ss)):
            _no("sessions", "one or more session records' SHA-512")

    def _late(self, e: Mapping[str, Any], now: float) -> bool:
        s = e["session"]
        first = self._held(s[0][0]) if s and _nat(s[0][0]) else e
        if first["drawn_at"] is None:
            raise ValueError("a drawn entry has its draw's reading")
        return now - float(first["drawn_at"]) > self.deadline

    def _finish(self, e: dict[str, Any], symbol: str) -> str:
        e["final"] = True
        self.s["failures"] += int(symbol != ACCEPT)
        if e["order"] in self.s["opened"]:
            self.s["opened"].remove(e["order"])
        return symbol

    def _deliver(self, op: Mapping[str, Any]) -> dict[str, Any]:
        e = self._drawn(op)
        self._open_entry(e, op["order"])
        self._sessions(op)
        late = self._late(e, _float(_need(op, "now"), "now"))
        e["delivered"] = True
        return {"ok": self._finish(e, LATE) if late else None}

    def _deliver_session(self, op: Mapping[str, Any]) -> dict[str, Any]:
        stream = _token(op.get("stream"), "stream")
        draws = self._draws(op)
        if stream != self.params["stream"] or not draws:
            _no("not-session", {"stream": stream})
        entries = []
        for order, registration, _ in draws:
            e = self._entry(order, registration)
            if e["session"] != draws:
                _no("not-session", {"order": order})
            if e["final"]:
                _no("final", {"order": order})
            if e["delivered"]:
                _no("delivered", {"order": order})
            entries.append(e)
        self._sessions(op)
        late = self._late(entries[0], _float(_need(op, "now"), "now"))
        for e in entries:
            e["delivered"] = True
        return {"ok": [self._finish(e, LATE) for e in entries] if late else None}

    def _outcome(self, op: Mapping[str, Any]) -> dict[str, Any]:
        e = self._drawn(op)
        if e["final"]:
            _no("final", {"order": op["order"]})
        if not e["delivered"]:
            if self._late(e, _float(_need(op, "now"), "now")):
                return {"ok": self._finish(e, LATE)}
            raise _Stop({"pending": True, "detail": "no proof delivered"})
        if (cs := self._codes(op, "verdicts")) is not None:
            raise _Stop({"pending": True} if cs == ["pending"] else {"refused": list(cs)})
        if not isinstance(accepted := _need(op, "accepted"), bool):
            raise ValueError("accepted is a boolean")
        return {"ok": self._finish(e, ACCEPT if accepted else REJECT)}

    def _halt(self, op: Mapping[str, Any]) -> dict[str, Any]:
        cause = op.get("cause")
        if not isinstance(cause, dict):
            raise ValueError("a halt's cause is an object")
        opened = list(self.s["opened"])
        for o in opened:
            self._held(o)["final"] = True
        self.s.update(opened=[], halted=cause)
        return {"ok": opened}


def _step(params: Mapping[str, Any], state: Mapping[str, Any], op: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`op`'s decision on `state` by the stream's rules (`_Control`), and what they hold after it: `state` itself where
    the op ends early, as the verifier's `Stream.run` gives them. A halted stream refuses every op."""
    name = op.get("op") if isinstance(op, Mapping) else None
    if name not in _OPS:
        raise ValueError("an op is one of " + ", ".join(_OPS))
    if state["halted"] is not None:
        return {"refused": [HALTED], "detail": state["halted"]}, dict(state)
    c = _Control(params, state)
    try:
        return getattr(c, "_" + name)(op), c.state()
    except _Stop as s:
        return s.decision, dict(state)


#: The ops of a fleet's rules, each `_FleetControl`'s method of that name.
_FLEET_OPS = ("attach", "register", "failures")
#: What the fleet's rules hold of an empty fleet.
_FLEET_EMPTY = {"size": 0, "streams": [], "halted": []}


class _FleetControl:
    """A fleet's own rules, the reference for the verifier's (`SampledProofs.Fleet`, whose docstring has the
    state and the ops): R5-all, each stream attached to its operator with its failure budget, and at the failure that
    brings one to its budget the operator's halt, which names every stream attached to it. A refusal's `detail` is the
    fleet's own, which the Lean doesn't write. `ValueError` is an op the Lean doesn't read."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.s = copy.deepcopy(dict(state))

    def _halt_of(self, operator: str) -> dict[str, Any] | None:
        return next((h["cause"] for h in self.s["halted"] if h["operator"] == operator), None)

    def _attached(self, op: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        i = op.get("stream")
        if not (_nat(i) and i < len(self.s["streams"])):
            _no("unattached", {"stream": i})
        a = self.s["streams"][i]
        if (cause := self._halt_of(a["operator"])) is not None:
            _no(HALTED, cause)
        return i, a

    def _attach(self, op: Mapping[str, Any]) -> dict[str, Any]:
        operator, budget = op.get("operator"), op.get("budget")
        if not isinstance(operator, str):
            raise ValueError("an operator is a string")
        if not (_nat(budget) and budget >= 1):
            raise ValueError("a failure budget is a positive integer")
        if (cause := self._halt_of(operator)) is not None:
            _no(HALTED, cause)
        self.s["streams"].append({"operator": operator, "budget": budget})
        return {"ok": len(self.s["streams"]) - 1}

    def _register(self, op: Mapping[str, Any]) -> dict[str, Any]:
        self._attached(op)
        self.s["size"] += 1
        return {"ok": self.s["size"] - 1}

    def _failures(self, op: Mapping[str, Any]) -> dict[str, Any]:
        i, a = self._attached(op)
        if not _nat(f := op.get("failures")):
            raise ValueError("failures is a count")
        if f < a["budget"]:
            return {"ok": None}
        cause = {"operator": a["operator"], "stream": i, "at": self.s["size"]}
        streams = [j for j, b in enumerate(self.s["streams"]) if b["operator"] == a["operator"]]
        self.s["halted"].append({"operator": a["operator"], "cause": cause})
        return {"ok": {"cause": cause, "streams": streams}}


def _fleet_step(state: Mapping[str, Any], op: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`op`'s decision on `state` by the fleet's rules (`_FleetControl`), and what they hold after it: `state` itself
    where the op ends early, as the verifier's `Fleet.run` gives them."""
    name = op.get("op") if isinstance(op, Mapping) else None
    if name not in _FLEET_OPS:
        raise ValueError("an op is one of " + ", ".join(_FLEET_OPS))
    c = _FleetControl(state)
    try:
        return getattr(c, "_" + name)(op), c.s
    except _Stop as s:
        return s.decision, dict(state)


class Fleet:
    """The challenger's streams across protocols and levels, each attached to its operator, the identity it registers
    for, with the failure budget it runs under. Its rule is R5-all: at the failure that brings any stream of an operator
    to its budget (its b-th), the operator halts. Every stream attached to it halts at once (`Stream.halted`): each
    outcome of it not yet final ends as `halted`, never `accept`, outcomes already final stay, and each refuses every
    call from then on. The fleet refuses a registration to a stream of a halted operator, and refuses to attach another
    stream to it (`halted`).

    Each registration the fleet admits has a fleet position, the number it admitted before; a halt's cause
    (`halted`) names the operator, the stream whose failure halted it (its index, in the order attached) and the fleet
    position then. The fleet's rules decide each op by the verifier (`verifier.fleet`: the Lean's
    `SampledProofs.Fleet` against `_FleetControl`); where the two disagree the fleet stops, refusing every op after it
    (`disagree`)."""

    def __init__(self) -> None:
        self._rules: dict[str, Any] = copy.deepcopy(_FLEET_EMPTY)
        self._streams: list[Stream] = []
        self._broken: Verdict | None = None

    @property
    def size(self) -> int:
        """The registrations the fleet admitted: the next one's fleet position."""
        return self._rules["size"]

    @property
    def halted(self) -> dict[str, dict[str, Any]]:
        """Each halted operator's halt's cause, `{"operator", "stream", "at"}`."""
        return {h["operator"]: h["cause"] for h in self._rules["halted"]}

    def attach(self, stream: Stream, operator: str) -> int:
        """Attach `stream`, before its first registration, to `operator` with its failure budget: its index. Raises
        `ValueError` for a stream attached already or one that has registered, and `Refused` (`halted`) when the
        operator halted."""
        if stream._fleet is not None:
            raise ValueError("the stream is attached to a fleet already")
        if stream._rules["size"] or stream._entries:
            raise ValueError("a stream is attached before its first registration")
        i = Stream._held(self._decide({"op": "attach", "operator": operator, "budget": stream.budget}))["ok"]
        if i != len(self._streams):
            raise RuntimeError(f"the fleet's rules attach stream {i!r}, not the fleet's next")
        self._streams.append(stream)
        stream._fleet, stream._fleet_index = self, i
        return i

    def _decide(self, op: dict[str, Any]) -> dict[str, Any]:
        if self._broken is not None:
            raise Refused(self._broken)
        d, after, v = V.fleet(self._rules, op, *_fleet_step(self._rules, op))
        if not v.ok:
            self._broken = v
            raise Refused(v)
        self._rules = after
        return d

    def _admit(self, stream: Stream) -> int:
        """The registration's fleet position, or `Refused` (`halted`)."""
        return Stream._held(self._decide({"op": "register", "stream": stream._fleet_index}))["ok"]

    def _failed(self, stream: Stream) -> None:
        """`stream` ended an outcome as a failure: at its budget, the operator halts, and so does every stream of it."""
        d = Stream._held(self._decide({"op": "failures", "stream": stream._fleet_index, "failures": stream.failures}))
        if d["ok"] is not None:
            for j in d["ok"]["streams"]:
                self._streams[j]._halt(d["ok"]["cause"])


class Stream:
    """One protocol's stream for one `Spec`, held by the challenger.

    `unit` is the challenger's own copy of the proof-block check's Definition: its Program's digest is the check's, and the
    challenger builds each registration's population program from it and the registration's N. Or `unit` is a window's
    templates, a sequence of Definitions (a TP rank's served window: each provable family of the step): the check's
    `program` is their digests and its query `Q_template_instances` over them, both in the window's order, and each
    registration's population program is their mixed population (`partition.mixed_population_program`) at the counts
    `register` is given, under one draw by the Spec's sampling design (`stratified:K`: a stratum per template, each drawn at least
    once; `work:K`: at least the work table's floor). It is one program and one query, as for one template, so the
    verifier of record's draw and run (`Select.select`, `Run.run`) read it in the same form. `budget` (b), `delta`
    (the profile's) and `rounds` (R, the most registrations open at once, and so the most one `ProofRun` covers) are
    public parameters of the stream. `coins` is
    the challenger's randomness (n bytes a call, read in order by `OwnRandomness`): by default the operating system's
    bytes as the Lean verifier reads them (`verifier.coins`, `IO.getRandomBytes`, A3), never the Python's own
    read of the OS. `clock` is its clock in seconds; tests inject both. Coins from the auditor's challenger are a
    `coins.ChallengerCoins`: the stream's `coin_source` is then `"challenger"` (else `"verifier"`, the verifier's own
    coins or a test's stand-in for them), its digest names it, and every outcome's run by the verifier needs
    the coins' evidence (`outcome`'s `coins`), refusing a draw without it. `sampler`, when given, draws instead of the
    coins (`flock-verify draw` in a process of its own, say), never beside the challenger's: it takes the challenger's
    sampling design and N, and its draw is held to them like any other. `prover`, when given, is the identity the challenger registered the stream's prover under
    (lowercase hex): the stream's digest names it, and every registration is bound to the stream and the prover
    (`binding`, `registration.check`'s): its window names both, and a window salt is bound to them, so a tile seeded for
    one prover's stream is refused in any other.

    A stream runs one epoch with one salt: `salt` issues it from the coins, or `salt_released` records one released
    elsewhere with its release, and under `Check.registration_deadline` every registration is received within the
    deadline of that release, or the epoch ends (`ended`) at the first that isn't. `epoch_log` holds the challenger's
    clock readings.

    The stream's own rules (the budget, the open registrations, the salt and both deadlines, and each registration's
    once-only orders) decide each op by the verifier (`verifier.stream`: the Lean's `SampledProofs.Stream`
    against `_Control`), the clock's readings as floats, on what they hold of the stream that the op reads (`_slice`: its
    counts and the entries it names). What they hold is the verifier's state after each op, from the empty
    stream, so an op is decided on a state of a bounded size whatever the stream's length. Where the two disagree the
    stream stops: that op and every one after it is refused (`disagree`). A stream attached to a `Fleet`
    (`Fleet.attach`) is admitted by it at each registration and reports each failure to it, and halts with its operator.

    As its log grows, a stream keeps what one-stage's R6 and R7 read of it: where each window was logged, and each
    registered value's first registration. So a registration's checks read the same few records whatever the stream's
    history (`_earlier`), with the verdicts the whole log gives.
    """

    def __init__(self, spec: Spec, unit: SpecializedDefinition | Sequence[SpecializedDefinition], *, budget: int, delta: float, challenger: str,
                 coins: Callable[[int], bytes] = V.coins, clock: Callable[[], float] = time.monotonic,
                 sampler: Sampler | None = None, rounds: int = 1, prover: str | None = None) -> None:
        if spec.check.window is not None or spec.check.deadline is not None:
            raise ValueError("the service doesn't prove a window statement or run a timed session yet")
        if spec.check.proof_deadline is None:
            raise ValueError("a stream's check has a proof deadline (Check.proof_deadline), so that a draw whose proof "
                             "never arrives ends as a reject (late)")
        units = (unit,) if isinstance(unit, SpecializedDefinition) else tuple(unit)
        if not (units and all(isinstance(u, SpecializedDefinition) for u in units)):
            raise ValueError("a stream's unit is a Definition, or a window's: a sequence of them")
        digests = tuple(P.program_digest(Program(u)) for u in units)
        if (digests[0] if isinstance(unit, SpecializedDefinition) else digests) != spec.check.program:
            raise ValueError("the challenger's unit is not the check's program (a window's: one digest a template, in "
                             "its order)")
        query = P.template_instance_query(unit) if isinstance(unit, SpecializedDefinition) else \
            P.template_instances_query(units)
        if dict(spec.check.query) != query:
            raise ValueError("the check's query is not the template query over the challenger's unit (a window's: "
                             "Q_template_instances over its templates, in its order)")
        if not (type(budget) is int and budget >= 1):
            raise ValueError(f"a failure budget is a positive integer, not {budget!r}")
        if not (isinstance(delta, float) and 0 < delta < 1):
            raise ValueError(f"delta is a float in (0, 1), not {delta!r}")
        if not (type(rounds) is int and 1 <= rounds <= ROUNDS):
            raise ValueError(f"a session covers 1 to {ROUNDS} rounds, not {rounds!r}")
        self.coin_source = "challenger" if isinstance(coins, ChallengerCoins) else "verifier"
        if self.coin_source == "challenger" and sampler is not None:
            raise ValueError("a stream drawing from the challenger's coins takes no sampler, which would draw instead")
        self.spec, self.unit, self.budget, self.delta, self.challenger = spec, unit, budget, delta, challenger
        self.templates = None if isinstance(unit, SpecializedDefinition) else units
        self.rounds = rounds
        self._coins, self._clock, self._sampler = coins, clock, sampler
        self.digest = _sha512({"format": STREAM, "spec": spec.digest(), "budget": budget, "delta": delta,
                               "challenger": challenger, **({"rounds": rounds} if rounds > 1 else {}),
                               **({"prover": prover} if prover is not None else {}),
                               **({"coins": "challenger"} if self.coin_source == "challenger" else {})})
        self.binding = None if prover is None else R._binding({"stream": self.digest, "prover": prover})
        self._params = {"budget": budget, "rounds": rounds,
                        "registration_deadline": _reading(spec.check.registration_deadline),
                        "proof_deadline": float(spec.check.proof_deadline), "stream": self.digest}
        self._rules: dict[str, Any] = {k: v for k, v in _EMPTY.items() if k != "entries"}
        self._rule_entries: list[dict[str, Any]] = []
        self._broken: Verdict | None = None
        self._entries: list[_Entry] = []
        self._log: list[dict[str, Any]] = []
        self._windows: dict[bytes, list[int]] = {}
        self._irregular: list[int] = []
        self._first: dict[Any, int] = {}
        self._records: list[dict[str, Any]] = []
        self._epoch_log: list[dict[str, Any]] = []
        self._salted_at: float | None = None
        self._fleet: Fleet | None = None
        self._fleet_index: int | None = None

    @property
    def failures(self) -> int:
        return self._rules["failures"]

    @property
    def halted(self) -> dict[str, Any] | None:
        """The halt's cause once the stream's fleet halted its operator (`Fleet`), else None. A halted stream refuses
        every call (`halted`)."""
        return self._rules["halted"]

    @property
    def stopped(self) -> bool:
        return self.failures >= self.budget

    @property
    def ended(self) -> bool:
        """The epoch ended: a registration was late (`late-registration`), and the stream receives none from then on."""
        return self._rules["ended"] is not None

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
        """One symbol per final outcome, in order: `accept`, `reject`, `late` or `halted`."""
        return tuple(r["symbol"] for r in self._records)

    @property
    def log(self) -> tuple[dict[str, Any], ...]:
        """The registrations received, in order: the log `registration.check`'s R6 reads."""
        return tuple(self._log)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """The outcomes' records, in order; `Outcome.record` is each one's SHA-512."""
        return tuple(self._records)

    def _decide(self, op: dict[str, Any], fill: Mapping[str, Callable[[], Any]] | None = None) -> dict[str, Any]:
        """`op`'s decision of record by the stream's rules (`verifier.stream`), filling in what it asks for first: while
        it is `unchecked`, `fill[field]()` reads the clock or runs the checks, and the op is asked again. The state of
        record after it is what the stream's rules hold from then on; a disagreement stops the stream."""
        if self._broken is not None:
            raise Refused(self._broken)
        while True:
            state = self._slice(op)
            d, after, v = V.stream(self._params, state, op, *_step(self._params, state, op))
            if not v.ok:
                self._broken = v
                raise Refused(v)
            field = d.get("unchecked")
            if field is None:
                break
            if field in op or field not in (fill or {}):
                raise RuntimeError(f"the stream's rules ask for {field!r}, which this op has or can't give")
            op[field] = fill[field]()
        self._merge(after)
        return d

    def _slice(self, op: Mapping[str, Any]) -> dict[str, Any]:
        """What the stream's rules hold that `op` reads: the counts, and the entries it names, with the first round of a
        session that covers one; a halt names every open one."""
        entries = self._rule_entries
        named = list(self._rules["opened"]) if op.get("op") == "halt" else \
            [op.get("order")] if "order" in op else [d[0] for d in op.get("draws") or ()]
        at: set[int] = set()
        for o in named:
            if _nat(o) and o < len(entries):
                at.add(o)
                s = entries[o]["session"]
                if s and _nat(s[0][0]):
                    at.add(s[0][0])
        return {**self._rules, "entries": [entries[o] for o in sorted(at)]}

    def _merge(self, after: Mapping[str, Any]) -> None:
        entries = self._rule_entries
        for e in after["entries"]:
            if not (_nat(e["order"]) and e["order"] <= len(entries)):
                raise RuntimeError(f"the stream's rules hold an entry the stream doesn't: {e['order']!r}")
            if e["order"] == len(entries):
                entries.append(e)
            else:
                entries[e["order"]] = e
        self._rules = {k: v for k, v in after.items() if k != "entries"}
        if self._rules["size"] != len(entries):
            raise RuntimeError("the stream's rules count registrations the stream doesn't hold")

    @staticmethod
    def _held(d: Mapping[str, Any], checks: Verdict | None = None) -> Mapping[str, Any]:
        """`d` when it decides the op, else raises: `Refused` with the checks' own verdict when it is theirs, `Pending`."""
        if "refused" in d:
            if checks is not None and not checks.ok and list(checks.codes) == d["refused"]:
                raise Refused(checks)
            code = d["refused"][0]
            raise Refused(Verdict(False, tuple(d["refused"]), {code: d.get("detail")}))
        if "pending" in d:
            raise Pending(checks.detail["pending"] if checks is not None and checks.codes == ("pending",) else
                          d.get("detail"))
        return d

    def ready(self) -> None:
        """Raises `Refused` when the stream receives no registration now: it has stopped, its epoch ended
        (`epoch-ended`, with the late registration's readings), or it holds as many registrations without a final
        outcome as it may (`open`): `rounds`, and never more than `budget` minus its failures."""
        self._held(self._decide({"op": "ready"}))

    def salt(self, n: int = 64) -> bytes:
        """The epoch's salt coin: `n` bytes of the challenger's own coins (`OwnRandomness`, in order with the draws'
        reads), issued once. Its release is read on the challenger's clock (`salted_at`) and logged (`epoch_log`);
        `Check.registration_deadline` counts from it. Raises `Refused`: `salted` (the epoch has a salt, from `salt` or
        `salt_released`: nothing redraws it) or `epoch-ended` (the epoch ended before any salt)."""
        if not (type(n) is int and n >= 1):
            raise ValueError(f"a salt is a positive number of bytes, not {n!r}")
        got: dict[str, Any] = {}

        def at() -> float:
            got["salt"] = OwnRandomness(self._coins).read(n)
            got["salted_at"] = self._clock()
            return float(got["salted_at"])
        self._held(self._decide({"op": "salt"}, {"at": at}))
        self._salted_at = got["salted_at"]
        self._epoch_log.append({"event": "salt", "salt": got["salt"].hex(), "salted_at": self._salted_at})
        return got["salt"]

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
        self._held(self._decide({"op": "salt_released", "released_at": float(released_at)},
                                {"now": lambda: float(self._clock())}))
        self._salted_at = released_at
        self._epoch_log.append({"event": "salt", "salt": salt.hex(), "salted_at": released_at, "external": True})

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], *, leaf_layer: str | None = None,
                 anchors: Sequence[Any] = (), registered: Mapping[str, int] | None = None,
                 work_table: Mapping[str, Any] | None = None, supplied: Mapping[str, Any] | None = None,
                 pinned: Mapping[str, str] | None = None, schemas: Mapping[str, str] | None = None,
                 window_root: str | None = None, held: Sequence[str] | None = None,
                 counts: Mapping[str, int] | None = None) -> WindowCommitmentRecord:
        """Receive a registration: the public-items rule (`check_public`), then one-stage's R1-R7 (`registration.check`),
        each by the verifier (`verifier.check_public`, `verifier.check`), against the challenger's own copies:
        the population program it builds from `unit` and the record's N (a window's: the mixed population of its
        templates, in its order, at `counts`, the challenger's own `{template's descriptor id: instances}` of the window,
        every template's at least one), the check's query, the selection rule's sampling design,
        `SCHEME`, the stream's log (the records of it R6 and R7 read, `_earlier`), and the per-registration copies given
        here (`ports`, `leaf_layer`, `anchors`, `registered`, `work_table`, `supplied`, `pinned`, `schemas`,
        `window_root` and `held`, as `registration.check` takes them). Then the receipt. Under
        `Check.registration_deadline` the challenger first reads its clock (`received_at`): a registration received more
        than the deadline after the salt's release, or before any salt, is refused (`late-registration`, with both
        readings) whatever it holds, logged, and ends the epoch; an on-time one carries both readings
        (`WindowCommitmentRecord.salted_at`, `received_at`) and is logged with them. Raises
        `Refused`: when the stream has stopped or halted (`halted`) or its epoch ended (`epoch-ended`), while it holds as
        many registrations without a final outcome as it may (`open`, `ready`), when it is late, when a window's `counts`
        aren't at least one instance for each of its templates and N in all, or a single proof block's stream is given any
        (`population`), when the rule's rate sampling design
        draws none or more than N proof blocks (`rate`, `selection_law`, before R1-R7), when either check fails, or, after both,
        when the sampling design draws a count that varies (`shapes`, `count_varies`), when the program's proof blocks read stratum by stratum
        are not its N proof blocks in order (`population-order`, `partition.in_order`), or its fleet refuses it (`halted`)."""
        deadline = self.spec.check.registration_deadline
        op: dict[str, Any] = {"op": "register"}
        got: dict[str, Any] = {}

        def received() -> float:
            got["received_at"] = self._clock()
            return float(got["received_at"])

        def checks() -> list[str] | None:
            v = self._public(record)
            n = record.get("population")
            if v.ok and not (type(n) is int and n >= 1):
                v = _refuse("population", n).verdict
            if v.ok:
                try:
                    law = selection_law(self.spec.selection, n)
                except Refused as r:
                    v = r.verdict
            if v.ok:
                program = self._program(n, counts)
                if program is None:
                    v = _refuse("population", {"counts": counts, "templates": None if self.templates is None else
                                               [P.template_id(t) for t in self.templates]}).verdict
            if v.ok:
                earlier = self._earlier(record, supplied)
                v = V.check(record, program, self.spec.check.query, law, SCHEME, ports, leaf_layer, anchors, earlier, None,
                            work_table, registered, supplied, pinned, self.binding, schemas, window_root, held)
                if v.ok and count_varies(law):
                    v = _refuse("shapes", {"law": law}).verdict
                if v.ok and not P.in_order(program, self.spec.check.query, n):
                    v = _refuse("population-order", {"population": n}).verdict
                got.update(program=program, law=law, earlier=earlier)
            if v.ok and self._fleet is not None:
                try:
                    self._fleet._admit(self)
                except Refused as r:
                    v = r.verdict
            got["checks"] = v
            if not v.ok:
                return list(v.codes)
            salted_at = None if deadline is None else self._salted_at
            got["registration"] = self._registration(record, len(self._entries), salted_at, got.get("received_at"))
            op["registration"] = _registration_token(got["registration"])
            return None
        d = self._decide(op, {"received_at": received, "checks": checks})
        if d.get("refused") == ["late-registration"]:
            self._epoch_log.append({"event": "late-registration", "salted_at": self._salted_at,
                                    "received_at": got["received_at"]})
        self._held(d, got.get("checks"))
        reg, earlier = got["registration"], got["earlier"]
        if earlier is self._log:
            earlier = self._log[:]
        self._append(reg)
        program, law = got["program"], got["law"]
        self._entries.append(_Entry(reg, D.derive(program, self.spec.check.query, law, work_table), program, work_table,
                                    (ports, leaf_layer, anchors, earlier, registered, supplied, pinned, schemas,
                                     window_root, held)))
        if deadline is not None:
            self._epoch_log.append({"event": "registration", "order": reg.order, "registration": reg.digest,
                                    "salted_at": reg.salted_at, "received_at": reg.received_at})
        return reg

    def _program(self, n: int, counts: Mapping[str, int] | None) -> Program | None:
        """The challenger's population program for a registration of N = `n`: `unit`'s N instances, or a window's mixed
        population at `counts` (`register`'s), which number N in all; None when `counts` don't fit the stream."""
        if self.templates is None:
            return P.population_program(self.unit, n) if counts is None else None
        ids = [P.template_id(t) for t in self.templates]
        if not (isinstance(counts, Mapping) and set(counts) == set(ids)
                and all(type(counts[i]) is int and counts[i] >= 1 for i in ids) and sum(counts.values()) == n):
            return None
        return P.mixed_population_program([(t, counts[i]) for t, i in zip(self.templates, ids)])

    def _public(self, record: Mapping[str, Any]) -> Verdict:
        """`check_public` by the verifier (`verifier.check_public`)."""
        values = self.spec.values
        return V.check_public(check_public(self.spec, record, values), record, SCHEME, self.spec.check.query,
                              [(v.name, v.registered) for v in values], [p.item for p in self.spec.public])

    def _registration(self, record: Mapping[str, Any], order: int, salted_at: float | None = None,
                      received_at: float | None = None) -> WindowCommitmentRecord:
        return WindowCommitmentRecord(dict(record), R.receipt(record, self.challenger, received=str(len(self._log))),
                                      R.digest(record), order, salted_at, received_at)

    def _append(self, reg: WindowCommitmentRecord) -> None:
        at = len(self._log)
        self._log.append(logged := dict(reg.record))
        key = _window_key(logged.get("window"))
        if key is None:
            self._irregular.append(at)
        else:
            self._windows.setdefault(key, []).append(at)
        for e in logged.get("registered") or ():
            self._first.setdefault((e["value"], e["domain"]), at)

    def _earlier(self, record: Mapping[str, Any], supplied: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """The records of the log that one-stage's R6 and R7 read for `record`, in log order: the last (R6's `previous`),
        every one whose window may equal `record`'s (R6's window: the same `_window_key`, or none), and each registered
        value's first registration under each domain (R7's held roots, `registration.held_roots`). Both verifiers give the same
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

    def select(self, registration: WindowCommitmentRecord, *, closure: Mapping[int, Sequence[int]] | None = None) -> Draw:
        """The draw, once per registration, from the challenger's own randomness after the receipt: indices in [0, N) by
        the selection rule's sampling design (`select_draw`; `All` draws every proof block and reads no coins). `closure` is the challenger's closure map
        (`{unit: [units proved with it]}` over [0, N), from its own layout), for a work sampling design only: the draw carries the
        drawn proof blocks' closure (`coins.closure_of`), which `draw.proved` and `draw.check_proved` count. The draw of
        record is the verifier's (`verifier.select`: the Lean's `Select.select` on the bytes of the coins this
        draw read), unless a `sampler` draws. Raises `Refused` for a registration drawn before (nothing redraws), a
        closure map on another sampling design or outside [0, N) (before any coin is read), the two verifiers drawing differently
        (`disagree`), and a draw that isn't well formed under the sampling design (`draw.check_draw`, by `verifier.check_draw`)."""
        op: dict[str, Any] = {"op": "select", "order": _order(registration),
                              "registration": _registration_token(registration)}
        got: dict[str, Any] = {}

        def checks() -> list[str] | None:
            e = self._entries[registration.order]
            read = bytearray()

            def coins(n: int) -> bytes:
                b = self._coins(n)
                read.extend(b)
                return b
            try:
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
            except Refused as r:
                got["checks"] = r.verdict
                return list(r.verdict.codes)
            draw = Draw(registration, e.law, obj)
            got.update(draw=draw, drawn_at=self._clock(), read=None if self._sampler is not None else bytes(read))
            op.update(draw=_draw_token(draw), proved=len(draw.proved), at=float(got["drawn_at"]))
            return None
        self._held(self._decide(op, {"checks": checks}), got.get("checks"))
        e = self._entries[registration.order]
        e.draw, e.drawn_at, e.read, e.closure = got["draw"], got["drawn_at"], got["read"], closure
        return e.draw

    def session(self, draws: Sequence[Draw]) -> ProofRun:
        """Open one proof session over `draws`, the draws of 1 to `rounds` consecutive registrations in order, each
        round's registration its own tables of the session's statement: once, after the last covered draw. Its proof
        deadline is the first covered draw's (`Check.proof_deadline` from that draw), for every round it covers. Raises
        `Refused`, changing nothing: `rounds` (no draw, or more than `rounds`), `unknown`,
        `not-drawn` (a covered registration not drawn yet, or another draw), `final`, `delivered`, `covered` (a draw
        another session covers: nothing reopens it) or `order` (not consecutive registrations in order)."""
        draws = tuple(draws)
        self._held(self._decide({"op": "session", "draws": [_covered(d) for d in draws]}))
        s = ProofRun(self.digest, draws)
        for d in draws:
            self._entries[d.registration.order].session = s
        return s

    def _deliver_session(self, session: ProofRun, sessions: Sequence[str]) -> tuple[Outcome, ...] | None:
        draws = tuple(session.draws) if isinstance(session.draws, Sequence) else ()
        d = self._held(self._decide({"op": "deliver_session",
                                     "stream": session.stream if isinstance(session.stream, str) else "",
                                     "draws": [_covered(x) for x in draws], "sessions": _session_digests(sessions)},
                                    {"now": lambda: float(self._clock())}))
        entries = [self._entries[x.registration.order] for x in draws]
        for e in entries:
            e.sessions = list(sessions)
        if d["ok"] is None:
            return None
        return self._reported(tuple(self._finish(e, LATE, None, {}) for e in entries))

    def deliver(self, draw: Draw | ProofRun, sessions: Sequence[str]) -> Outcome | tuple[Outcome, ...] | None:
        """The firewall delivers the draw's proof, once: `sessions` are its session records' SHA-512, which the outcome
        binds. A proof after the proof deadline is a reject (`late`), returned at once; otherwise None, and the outcome waits
        on the verifier (`outcome`). A `ProofRun`'s proof is delivered as the session, once, for every draw it
        covers (`covered`, delivering one of them alone): each binds the same `sessions`, and past the first covered
        draw's deadline each is `late`, returned in order. Raises `Refused` (`not-session`) for a session the stream
        didn't open."""
        if isinstance(draw, ProofRun):
            return self._deliver_session(draw, sessions)
        d = self._held(self._decide({"op": "deliver", "order": _order(draw.registration),
                                     "registration": _registration_token(draw.registration), "draw": _draw_token(draw),
                                     "sessions": _session_digests(sessions)}, {"now": lambda: float(self._clock())}))
        e = self._entries[draw.registration.order]
        e.sessions = list(sessions)
        return self._reported(self._finish(e, LATE, None, {})) if d["ok"] == LATE else None

    def outcome(self, draw: Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                proved_units: Sequence[int] | None = None, instances: int | None = None, served: Sequence[int] | None = None,
                session_roots: Mapping[str, str] | None = None, coins: Mapping[str, Any] | None = None) -> Outcome:
        """The draw's final outcome, once. With no proof delivered: a reject (`late`) once the deadline has passed, and
        `Pending` before. With one: `Pending` until every verdict of record has run (`accepted` not null); each verdict of
        record names the delivered `sessions` and reports the public outputs alike (`check_verdicts`, by the verifier,
        `verifier.check_verdicts`); then one-stage's audit record by the verifier
        (`verifier.audit_record`; `proved_units` and `instances` default to the draw's, `served` and `session_roots` are
        its, and the registered roots it reads are the verdicts' own) decides accept or reject. A draw a `ProofRun` covers is decided alone, on the session's delivered records and its own
        audit record: its `proved_units`, `instances`, `served` and `session_roots` are its round's (its own tables'),
        never another round's. On a draw from the coins and an audit record its two verifiers agree on, the whole run is
        held to the verifier's (`verifier.run`, the Lean's `Run.run` from the
        registration on): where they differ the outcome is a reject, its audit record failing `run`. The run takes the
        stream's `coin_source`, so on a stream drawing from the auditor's challenger (`coins.ChallengerCoins`) the
        Lean's run refuses every draw unless `coins` is its evidence (`coins.coins_reason`'s `{key, ledger, main, pins,
        segment, session}`) and the bytes it read are exactly the coins the session the ledger opens for the
        registration's window (`register`'s `window_root`), from main's code (`challenger.pinned_session`), released
        after the registration's digest, each checked against its
        commitment and every one read; missing evidence, or a registration without its window root, is a reject failing
        `run` too, never a skip. Raises `Refused` for a second outcome, a verdict of record about
        other sessions, public outputs the verdicts of record don't report alike, or `coins` for a draw that read no
        coins."""
        op: dict[str, Any] = {"op": "outcome", "order": _order(draw.registration),
                              "registration": _registration_token(draw.registration), "draw": _draw_token(draw)}
        got: dict[str, Any] = {}
        o = op["order"]
        if coins is not None and o is not None and o < len(self._entries) and self._entries[o].read is None:
            raise _refuse("coins",
                          "coin evidence is for a draw read from the challenger's coins, and this draw read none")

        def held() -> list[str] | None:
            e = self._entries[draw.registration.order]
            assert e.sessions is not None
            outputs = self.spec.check.public_outputs
            v = V.check_verdicts(check_verdicts(verdicts, e.sessions, outputs), verdicts, e.sessions, outputs)
            got["verdicts"] = v
            if not v.ok:
                return list(v.codes)
            got.update(zip(("audit", "public"), self._audit(e, draw, verdicts, statement, proved_units, instances, served,
                                                            session_roots, coins)))
            op["accepted"] = got["audit"]["accepted"]
            return None
        d = self._held(self._decide(op, {"now": lambda: float(self._clock()), "verdicts": held}), got.get("verdicts"))
        e = self._entries[draw.registration.order]
        if d["ok"] == LATE:
            return self._reported(self._finish(e, LATE, None, {}))
        return self._reported(self._finish(e, d["ok"], got["audit"], got["public"] if d["ok"] == ACCEPT else {}))

    def _reported(self, outcome: _O) -> _O:
        """`outcome`, one or a session's, once the stream's fleet has its failures (`Fleet`): at the stream's budget the
        fleet halts the operator, and every stream of it."""
        if self._fleet is not None and not all(o.accepted for o in (outcome if isinstance(outcome, tuple) else (outcome,))):
            self._fleet._failed(self)
        return outcome

    def _halt(self, cause: Mapping[str, Any]) -> tuple[Outcome, ...]:
        """The fleet halted the stream's operator, for `cause`: every registration without a final outcome ends now, in
        order, as `halted`, with no failure of its own, and the stream refuses every call from then on."""
        d = self._held(self._decide({"op": "halt", "cause": dict(cause)}))
        return tuple(self._finish(self._entries[o], HALTED, None, {}, halt=cause) for o in d["ok"])

    def _audit(self, e: _Entry, draw: Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any],
               proved_units: Sequence[int] | None, instances: int | None, served: Sequence[int] | None,
               session_roots: Mapping[str, str] | None, coins: Mapping[str, Any] | None = None
               ) -> tuple[dict[str, Any], dict[str, Any]]:
        """`outcome`'s audit record, once the verdicts hold (failing `run` where the whole run isn't the verifier's),
        and the public outputs."""
        assert e.sessions is not None
        outputs = self.spec.check.public_outputs
        of_record = sorted(k for k in verdicts if verdicts[k].get("of_record"))
        public = dict(verdicts[of_record[0]]["public"]) if outputs else {}
        asked = (proved_units, instances)
        proved = list(proved_units) if proved_units is not None else D.proved(draw.draw)
        instances = len(proved) if instances is None else instances
        rec = V.audit_record(draw.registration.record, draw.registration.receipt, draw.draw, draw.law, proved, instances,
                             statement, verdicts, self.delta, e.sessions, served, session_roots)
        accepted = rec["accepted"]
        if e.read is not None and rec["verifiers"]["agree"]:
            reg = draw.registration
            ports, leaf_layer, anchors, earlier, registered, supplied, pinned, schemas, window_root, held = e.copies
            v = V.run({"receipt": dict(reg.receipt), "draw": dict(draw.draw), "read": len(e.read),
                       "symbol": ACCEPT if accepted else REJECT, "public": public if accepted else {},
                       "audit": {k: x for k, x in rec.items() if k != "verifiers"}},
                      reg.record, e.program, self.spec.check.query,
                      selection_law(self.spec.selection, reg.record["population"]), SCHEME, ports, leaf_layer, anchors,
                      earlier, e.work_table, registered, self.spec.selection.to_json(),
                      [(x.name, x.registered) for x in self.spec.values], [p.item for p in self.spec.public],
                      self.challenger, reg.receipt["received"], e.read, e.closure, e.sessions, verdicts, outputs,
                      statement, self.delta, *asked, served, session_roots, supplied=supplied, pinned=pinned,
                      binding=self.binding, schemas=schemas, window_root=window_root, held=held,
                      coin_source=self.coin_source, coins=coins)
            if not v.ok:
                rec = {**rec, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None,
                       "checks": {**rec["checks"], "run": {"ok": False, "codes": list(v.codes), "detail": v.detail}}}
        return rec, public

    def _finish(self, e: _Entry, symbol: str, audit: Mapping[str, Any] | None, public: Mapping[str, Any], *,
                halt: Mapping[str, Any] | None = None) -> Outcome:
        reg, draw = e.registration, e.draw
        assert draw is not None or symbol == HALTED
        body = {"format": OUTCOME, "stream": self.digest, "order": reg.order, "registration": reg.digest,
                "receipt": _sha512(dict(reg.receipt)), "draw": None if draw is None else _sha512(dict(draw.draw)),
                "sessions": e.sessions or [],
                "symbol": symbol, "public": dict(public), "audit": None if audit is None else dict(audit)}
        if e.session is not None:
            body["session"] = {"rounds": [{"order": d.registration.order, "registration": d.registration.digest,
                                           "draw": _sha512(dict(d.draw))} for d in e.session.draws]}
        if halt is not None:
            body["halt"] = dict(halt)
        accepted = symbol == ACCEPT
        exhaustive = isinstance(self.spec.selection, All)
        e.outcome = Outcome(
            accepted=accepted,
            verdict=Verdict(True) if accepted else Verdict(False, (symbol,), {symbol: _why(symbol, audit, halt)}),
            profile=A.profile(reg.record, draw.draw, self.delta) if accepted and not exhaustive else None,
            exhaustive=Exhaustive(len(draw.draw["units"])) if accepted and exhaustive else None,
            terms=Terms(None if exhaustive else self.delta),
            public=dict(public),
            record=_sha512(body))
        self._records.append(body)
        return e.outcome


def _why(symbol: str, audit: Mapping[str, Any] | None,
         halt: Mapping[str, Any] | None = None) -> Any:
    if symbol == HALTED:
        return dict(halt or {})
    if symbol == LATE:
        return "the proof was not delivered by the proof deadline"
    assert audit is not None
    return {"checks": [k for k, c in audit["checks"].items() if not c["ok"]],
            "verdicts": {k: v.get("accepted") for k, v in audit["verdicts"].items()}}


__all__ = ["ACCEPT", "All", "Check", "Draw", "Exhaustive", "Fleet", "HALTED", "HIDING", "KINDS", "LATE", "Law", "Outcome",
           "PHASES", "Pending", "ProofRun", "Public", "REJECT", "ROOTS", "ROUNDS", "Refused", "SCHEME", "SOURCES",
           "Selection", "Spec", "Stream", "Terms", "Value", "WindowCommitmentRecord", "check_public", "check_verdicts",
           "select_draw", "selection_law"]
