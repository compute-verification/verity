"""The proof system, the challenger's side: a protocol's `Spec`, the public-items rule (`check_public`), and a `Stream` of
registrations, samples and outcomes over the one-stage audit (`one_stage`).

A protocol states what it commits (`Spec.values`), which proof blocks need proofs (`Spec.selection`, its selection rule),
the check each sampled proof block passes, as a public Program (`Spec.check`), and what it makes public, each item with its reason
(`Spec.public`). What it does with each `Outcome` is its own code; the service does the rest. The parties: the workers
compute and prove with zero knowledge off, trusted by neither party; the ZK server, trusted by the developer, is their only
path to the auditor, and commits, registers and releases the proofs; the challenger, trusted by the auditor, receives
registrations, samples from its own randomness and keeps the record (this module); the verifier, the auditor's Lean
verifier, verifies offline, and reads coins only from the challenger's log.

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
  `seed_salt` when the window names one (the challenger's coin, so the Spec declares it of kind `coins`);
  `leaf_layer` and `anchors` when not empty; and a value's layout, `leaves:<value>` and `domain:<value>`, with
  `words:<value>` and `word_bits:<value>` for a registered value.

One-stage's record always shows N, each root's leaf count and domain, and the window's provenance, so a Spec over it
declares them; hiding them waits on the hidden layout (`one_stage.hidden_draw`).
The `Stream` runs `check_public` by the verifier before one-stage's R1 to R7, reading an entry's fields in key
order; its codes are `keys`, `roots`, `unsalted`, `query` and `undeclared`.

**A window of templates.** A `Stream` takes one template, or a window's several (a TP rank's served window, every
provable family of its step): the check's program is each template's digest and its query `Q_template_instances`, in
the window's order, and each registration's population program is the templates' mixed population at the challenger's
own counts (`register(counts=)`), sampled once by the Spec's sampling design, `stratified-rate:NUM/DEN` or `work-rate:NUM/DEN`
with one stratum per template. The verifier reads one program and one query either way, so a window's registration, sample
and run are its `check`, `Select.select` and `Run.run` as they are for one template's.

**The sample.** `register` logs the record and its receipt, whose `received` is the record's position in the stream's log.
The challenger then samples once (`select`) from its coins (`coins.ChallengerCoins`, opened after the registration's
digest) as the Lean verifier's `Flock.Draw.drawWith` does (`coins.draw_first` over `coins.OwnRandomness`), with no hash
and no derivation, or by the stream's `sampler`; either sample is held to `draw.check_draw`. Neither the stream nor the
verifier has coins of its own: a stream whose selection reads coins refuses any source but the challenger's
(`NO_OWN_COINS`), and the verifier's run refuses a sample that read coins without the challenger's log to show they are
its (`outcome`'s `coins`). A closure map (`select(closure=)`) is refused before any coin is read (`closure`): a work
sample widened by one proves a set whose size varies with the sample, which the sampled audit's soundness statement
doesn't cover.
From the selection rule to the sample object it is `selection_law`, `draw.derive` and `select_draw`, which the Lean
verifier's `Spot.Select` ports, held to them by its difftest on fixed coins.

**A two-stage sampling design** (`TwoStage(grain, replay, k)`, `Stream(replay=)`). The registration is the boundary
commitment: its population is the replay blocks, the instances of the replay block's Definition under the replay grain's
query, and its roots are the boundary values (phase `serve`). The first stage (`select`) samples replay blocks by the
`replay` law, a rate law: the replay sample. The prover then replays each sampled replay block and commits its
interior (phase `interior`) in the interior registration (`register_interiors`), over the Q·n_rho proof blocks of the Q
sampled replay blocks, registered after the registration (`previous`) and before the second stage's coins. A replay
block holds n_t instances of each template t of the check (`replay_cells`, the check naming them in the order of their
first instance), and the interior is template-major (`Stream.interior_program`): template t's Q·n_t proof blocks from
off_t = Q·Σ_{t'<t} n_t', replay block `sample[i]`'s instances of it [off_t + i·n_t, off_t + (i+1)·n_t), one cell. The
second stage (the second `select`, on the interior registration) samples k_t proof blocks in each cell of template t on
fresh coins (`second_law`: one stratum per cell; `second_counts`), and its sample is proved. The outcome holds both
audit records and accepts when both do; its profile counts incorrect proof blocks at each replay block's footprint,
which the boundary commitment fixes before any coin (`IntegrityProfile`'s two levels). One-stage is `Law`: a two-stage design whose replay blocks are single proof blocks is
refused, so each design has one form. Both stages' samples are the Lean verifier's (`Select.select`,
`Select.selectSecond`), and its knowledge soundness at Flock's masked record is `SamplingDeploy.ksTwo_code`'s, under
the premises that theorem takes.

**The circuit files.** A two-stage run's second-stage sessions each verify against their template's circuit file: the
flat Flock circuit file of one instance of it (`verity_flock.class_statement.UnitStager`, `unit_lowering` at
`structural=False`, as one-stage's sessions verify against theirs), which binds the interior registration's partition
and program digests. The auditor names each file by its SHA-512 in the run's interior params, `proofs.templates`
(`template_circuits`), beside the header of the template's registered reference statement and the registered reads it
reads, which round 2 takes as given and holds each statement's header and registered reads to but for the run's fields
(`second_stage.RUN_FIELDS`, `second_stage.blank_reads`), and reproduces the file from public data
alone: the template's Definition and rows, the interior's
partition and program digests, the values it reads (`reads`), its AND gates per stage (`stage_ands`) and the leaf
scheme, in a process where `FLOCK_PACK_WORDS`, `FLOCK_CORE_ROWS`, `FLOCK_PLAIN_LEAVES`, `FLOCK_OWN_ROWS` and
`FLOCK_GEMM_TILE` are unset, since each changes the file. The interior's digests depend on Q, which a rate law draws, so
the names are each run's and are handed to `outcome`.

**The salt and the registration deadline.** `Stream.salt` issues the epoch's salt coin once, from the challenger's own
coins, and reads its release on the challenger's clock (`salted_at`). `Stream.salt_released` instead records a salt
released elsewhere (PoUW's epoch coin, shared by one stream per level) with its release on the same clock, which can't
be later than the clock's reading now (`released-later`); either is the epoch's one salt. Under `Check.registration_deadline` (seconds on
that clock from the release), `register` reads the receipt (`received_at`) first and refuses a registration received
later than the deadline, or before any salt, as `late-registration`, whatever it holds. That is one visible failure,
never retried, and it ends the epoch: the stream receives no registration after it (`epoch-ended`) and never reissues
the salt (`salted`). It is the rule of `Pouw.AcceptsBy` (`security/Definitions/Pouw/Game/Deadline.lean`), with `ts` =
`salted_at`, `tr` = `received_at` and `td` = `ts` + the deadline. The readings are the challenger's own: the stream's
epoch log (`Stream.epoch_log`) and each `WindowCommitmentRecord` carry them, and neither the receipt nor an outcome's
record does.
`salt(n=64)` is `n` bytes of the challenger's coins, read in order with the samples'. A level's stream created at its
first registration still measures from the epoch's release. A salt that isn't non-empty bytes, or a release that isn't a
finite number, raises `ValueError`, and after the epoch ends a salt of either kind is refused (`epoch-ended`). The epoch
log holds `{"event": "salt", "salt", "salted_at"}` (with `"external": true` from `salt_released`), then each
`{"event": "registration", "order", "registration", "salted_at", "received_at"}` or `{"event": "late-registration",
"salted_at", "received_at"}`. With no registration deadline `register` reads no clock and logs nothing to the epoch log.

**The window's seed salt.** `Stream.seed_salt` draws a window's seed salt at its open, before it serves: the one coin of
the challenger's seed-salt family released after the window's open root (`coins.draw_seed_salt`), a family no sample
reads, drawn once a window and before any registration of it (`seed-salted`, `seed-salt-late`), and logged
(`{"event": "seed-salt", "window_root", "seed_salt", "seeded_at"}`). Each registration of that window is held to it by
both verifiers (`registration.check`'s `seed_salt`, R6-seed-salt): its window names it, a certified input the window's
circuits read. A verifier that didn't draw it reads it from the challenger's log (`coins.seed_salt_of`).

**No retries.** The salt is issued or recorded once (`salt`, `salt_released`). A registration is sampled once, after its receipt (`select`); its
proof is delivered once (`deliver`); its outcome is final (`outcome`), and nothing
resamples. A proof delivered after `Check.proof_deadline` (seconds on the challenger's clock, from the sample) is a reject,
and so is a sample whose proof is not delivered by then: a `Stream` refuses a check without a proof deadline, so every sample
ends. Each failure is one visible symbol, `reject` or `late`, or a late registration's refusal, `late-registration`;
an outcome the fleet's halt ends is `halted`.
A second call of any of these is refused. An outcome is pending until every verifier has run, and each verdict
of record names the delivered sessions and reports the public outputs alike (`check_verdicts`: `pending`, `sessions`,
`public`, by the verifier).

**Sessions over several rounds.** A stream of `rounds` R (1 to `ROUNDS`) holds up to R registrations
open at once, so one proof session can cover the samples of up to R consecutive registrations (rounds), each round's
registration its own tables of the session's statement. `Stream.session` opens it once, after the last covered sample
(no covered round is unsampled), over samples no other session covers and none delivered or final. Each round is committed
before its own sample, and so before every later round's. The first covered round's `Check.proof_deadline` spans the
session. `deliver` delivers the session once, and every covered sample's outcome binds the same session records. Each
sample's `outcome` runs its own audit record, on its own proof blocks and its own table's roots, and its record names the
session's rounds (`session`). A late session is `late` for every round it covers. Flock's verifier refuses
such a session (`Flock.ProvedScope`, `Draw.tablesWhy`); `Stream.session` has its refusals.

**The failure budget.** A stream counts its failures, and at the `budget`-th it stops and refuses further registrations.
It holds at most `budget` minus its failures open registrations (and at most `rounds`), so it never ends with more than
`budget` failures, though one session that fails is a failure for each round it covers.

**The fleet's halt (R5-all).** A `Fleet` holds the challenger's streams across protocols and levels, each attached to
its operator before its first registration. At the failure that brings any stream of an operator to its budget, the
fleet halts the operator: every stream attached to it halts at once (`Stream.halted`), each of its registrations
without a final outcome ends as `halted` (with no failure of its own; outcomes already final stay), and each refuses
every call from then on (`halted`), so nothing of the operator is accepted or registered after its halt. A `late` counts
toward the budget as a `reject` does. The fleet's rules (`_FleetControl`) decide by the verifier, the Lean's
`Spot.Fleet`, and a halted stream's by `Spot.Stream`'s `halt`.

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
"symbol", "public", "audit"}`: the receipt and the sample as the SHA-512 of their canonical JSON, `audit`
one-stage's audit record (null for `late` and `halted`). A `halted` outcome adds `"halt"`, the halt's cause, and
its `draw` is null when the registration was not sampled. `Outcome.record` is its SHA-512.

**Not built yet.** A stream refuses the `Sequential` rule, a window statement, a timed session and a
check with no proof deadline, and under a two-stage design more than one round. After R1 to R7, `register` refuses a `bernoulli` sampling design at any `p` (`bernoulli`), and `select` refuses a
closure map (`closure`), which the sampled audit's soundness statement doesn't cover either.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, NoReturn, Sequence, TypeVar

from verity.circuits.codec import canonical_json, descriptor_id
from verity.circuits.defs import SpecializedDefinition
from verity.circuits.partition import Level
from verity.circuits.program import Program
from verity.commitments import history as HT
from verity.protocols.spot.verifier.profile import Bernoulli, IntegrityProfile, Subset

from . import second_stage as SS
from .one_stage import audit as A, draw as D, partition as P, registered as G, registration as R, verifier as V
from .one_stage.verdict import Verdict, _Collect
from verity.protocols.spot.device.coins import ChallengerCoins, OwnRandomness, closure_of, draw_first, draw_seed_salt

SCHEME = "frame-v3-sha512/hm96-sha512"
HIDING = "hm96-sha512/"
#: A tree over the roots of hiding trees (`row_table.ROOTS_SCHEMA`); a Value of it names its leaves' schema (`over`).
ROOTS = "frame-v3-sha512/root/v1"
KINDS = ("structure", "output", "coins", "commitment")
SOURCES = ("parameter", "window")
PHASES = ("epoch", "serve", "interior", "in-deadline")
#: A two-stage outcome's profile's level of replay blocks, above one-stage's `audit.LEVEL` of proof blocks.
REPLAY = "replay"
STREAM = "verity/proof-service/stream/v0"
OUTCOME = "verity/proof-service/outcome/v0"
ACCEPT, REJECT, LATE, HALTED = "accept", "reject", "late", "halted"
#: The most rounds (registrations) one proof session covers.
ROUNDS = 4
NO_OWN_COINS = ("the verifier draws no coins of its own: a stream whose selection reads coins takes the auditor's "
                "challenger's, a coins.ChallengerCoins over its live record (a test's over a challenger of its own)")

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
    protocol's output (`output`), the challenger's coins and what they sample (`coins`), or a declared Value's salted root
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
    value's; the Spec's digest binds it. `phase`: when it is committed (`epoch`, `serve`, `interior`, a two-stage
    design's interior commitment after its first stage, or `in-deadline`). `registered`:
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


def fixed(text: Any) -> str:
    """The refusal of a law that is no rate (`rate`): a fixed count `subset:K`, `stratified:K` or `work:K` samples each proof
    block with K/N, which falls as the prover pads the N it registers. The Lean verifier's `Select.FIXED` is this."""
    return ("a law is rate:NUM/DEN or rate:NUM/DEN:T (0 < NUM <= DEN, T >= 1), stratified-rate:NUM/DEN or work-rate:NUM/DEN "
            f"(0 < NUM <= DEN) or bernoulli:NUM/DEN, not {text!r}: a fixed count samples each proof block with K/N, which "
            "falls as the prover pads the N it registers")


def rate(text: Any) -> bool:
    """Whether a law text is a rate: a rate sampling design (`draw.parse_rate`), a stratified or work rate sampling design
    (`draw.parse_strata_rate`), or `bernoulli:NUM/DEN`, which `Stream.register` refuses (`bernoulli`). The Lean verifier's
    `Select.rate?` is this."""
    if D.parse_rate(text) is not None or D.parse_strata_rate(text) is not None:
        return True
    try:
        return D.parse_design(text)["law"] == "bernoulli"
    except (ValueError, TypeError, AttributeError):
        return False


@dataclass(frozen=True)
class Law:
    """A one-stage sampling design, a rate (`rate`): the rate sampling design `rate:NUM/DEN` or `rate:NUM/DEN:T`
    (`draw.parse_rate`), the stratified rate `stratified-rate:NUM/DEN` or the work rate `work-rate:NUM/DEN`
    (`draw.parse_strata_rate`), or `bernoulli:NUM/DEN`, which `Stream.register` refuses. A registration's N proof blocks
    are the prover's to size, so a fixed count, `subset:K`, `stratified:K` or `work:K`, which samples each proof block with
    K/N as the prover pads N, is refused (`fixed`). A rate samples `subset:K` at K = ceil(NUM N / DEN) rounded up to a
    multiple of the T tables (`draw.rate_count`) over the N proof blocks each registration brings, so every proof block is
    sampled with probability at least NUM/DEN however large the prover makes N. The stratified rate samples the stratified
    sampling design at the K its strata need to draw ceil(NUM n_s / DEN) of each template's n_s proof blocks (`draw.strat_k`),
    and the work rate the work sampling design at K = ceil(NUM N / DEN) with each stratum's floor at the rate
    (`selection_table`, `draw.rate_floor`), so each stratum's proof blocks are sampled with probability at least NUM/DEN
    however the prover spreads N over templates."""

    text: str

    def __post_init__(self) -> None:
        if not rate(self.text):
            raise ValueError(fixed(self.text))

    def to_json(self) -> dict[str, Any]:
        return {"rule": "law", "law": self.text}


@dataclass(frozen=True)
class Sequential:
    """A timed session's sample, one index a round (PoUS's audit): `law` is `uniform:k`, over the public item named by
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


def _grain(g: Any) -> bool:
    """`Select.grain?`: `partition.template_instance_query`'s form, exactly `{name, version, params: {template}}`."""
    return (isinstance(g, Mapping) and set(g) == {"name", "version", "params"} and g["name"] == "Q_template_instance"
            and type(g["version"]) is int and g["version"] == 0 and isinstance(g["params"], Mapping)
            and set(g["params"]) == {"template"} and isinstance(g["params"]["template"], str))


def _replay_law(text: Any) -> bool:
    """`Select.replayLaw?`: a rate law, a uniform sample of replay blocks at the rate however many the prover registers."""
    return D.parse_rate(text) is not None


@dataclass(frozen=True)
class TwoStage:
    """A two-stage sampling design (the module docstring). `grain`: the replay grain, the replay partition's query,
    `Q_template_instance` over the replay block's Definition (`partition.template_instance_query`), whose instances in the
    registration's program are its replay blocks. `replay`: the first stage's law over them, a rate law.
    `k`: the proof blocks the second stage samples in each cell (a sampled replay block's instances of one template), on
    fresh coins after the interior registration: one count for every template, or `{template's descriptor id: k_t}`
    naming each of the check's templates once. A cell of fewer proof blocks than its count is proved whole
    (`second_counts`); a `Stream` refuses a design that proves every cell whole, which would prove whole replay blocks,
    and replay blocks of one proof block (`Law` is that design)."""

    grain: Mapping[str, Any]
    replay: str
    k: int | Mapping[str, int]

    def __post_init__(self) -> None:
        if not _grain(self.grain):
            raise ValueError("a two-stage design's grain is the replay partition's query, Q_template_instance v0 over the "
                             f"replay block's Definition (partition.template_instance_query), not {self.grain!r}")
        if not _replay_law(self.replay):
            raise ValueError(f"a two-stage rule's replay law is a rate law, rate:NUM/DEN or rate:NUM/DEN:T, not {self.replay!r}")
        if not (type(self.k) is int and self.k >= 1 or isinstance(self.k, Mapping) and self.k and all(
                isinstance(t, str) and type(k) is int and k >= 1 for t, k in self.k.items())):
            raise ValueError("a two-stage rule samples k >= 1 proof blocks in each cell: one count, or one per template "
                             "{descriptor id: k_t}")

    def to_json(self) -> dict[str, Any]:
        return {"rule": "two-stage", "grain": dict(self.grain), "replay": self.replay,
                "k": self.k if type(self.k) is int else dict(self.k)}


Selection = All | Law | Sequential | TwoStage


def boundary_values(values: Sequence[Value]) -> tuple[Value, ...]:
    """The values a registration commits: under a two-stage design, the boundary commitment's, every phase but
    `interior`."""
    return tuple(v for v in values if v.phase != "interior")


def interior_values(values: Sequence[Value]) -> tuple[Value, ...]:
    """The values a two-stage design's interior registration commits: phase `interior`, and the registered ones."""
    return tuple(v for v in values if v.phase == "interior" or v.registered)


@dataclass(frozen=True)
class Check:
    """The check each sampled proof block passes. `program`: the proof-block check Program's digest (`partition.program_digest`), or,
    for a window of templates (`Stream`'s), the tuple of each template's in the window's order. `query`: its proof blocks,
    one-stage's template query over that Program's Definition (`Q_template_instance`), or over the window's templates in
    its order (`Q_template_instances`). `window`: the window statement's
    Program's SHA-512, proved once per window over the committed layout. `public_outputs`: what the verifier reports that
    the auditor reads. `deadline`: a timed session's (PoUS's rounds). `proof_deadline`: seconds on the challenger's clock
    from the sample; a proof delivered later, or none by then, is a reject (`late`), and a `Stream` refuses a check
    without one. `registration_deadline`: seconds on the challenger's clock from
    the salt coin's release (`Stream.salt`, or `Stream.salt_released`'s `released_at`); a registration received later,
    or before any salt, is refused (`late-registration`) and ends the epoch. `tied`: the window Program's parameters, run
    Inputs, whose rows a later check ties to the registrations that computed them, which R8 lets through though several
    proof blocks read them (`registration.check`'s `tied`); the stream's digest binds them."""

    program: str | tuple[str, ...]
    query: Mapping[str, Any]
    window: str | None = None
    public_outputs: tuple[str, ...] = ()
    deadline: Mapping[str, Any] | None = None
    proof_deadline: float | None = None
    registration_deadline: float | None = None
    tied: tuple[str, ...] = ()

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
        if not (isinstance(self.tied, tuple) and all(isinstance(t, str) and t for t in self.tied)
                and len(set(self.tied)) == len(self.tied)):
            raise ValueError("a check's tied values are distinct parameter names")
        for name, d in (("proof", self.proof_deadline), ("registration", self.registration_deadline)):
            if d is not None and not (isinstance(d, (int, float)) and not isinstance(d, bool) and d > 0):
                raise ValueError(f"a {name} deadline is a positive number of seconds, not {d!r}")

    def to_json(self) -> dict[str, Any]:
        return {"program": list(self.program) if isinstance(self.program, tuple) else self.program,
                "query": dict(self.query), "window": self.window,
                "public_outputs": list(self.public_outputs), "deadline": None if self.deadline is None else dict(self.deadline),
                "proof_deadline": self.proof_deadline, "registration_deadline": self.registration_deadline,
                **({"tied": list(self.tied)} if self.tied else {})}


@dataclass(frozen=True)
class Spec:
    """What a protocol states: `values` (what is committed), `selection` (its selection rule), `check`, `public` (what the
    auditor may see beyond the service's own items, each with its reason) and `context` (the statement context the
    stream's digest binds). Under a two-stage design the values are the boundary values (the registration's: every
    phase but `interior`) and the interior values (the interior registration's: phase `interior`, and the registered
    ones)."""

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
        if not isinstance(self.selection, (All, Law, Sequential, TwoStage)):
            raise ValueError(f"a selection rule is All, Law, Sequential or TwoStage, not {self.selection!r}")
        interior = [v.name for v in self.values if v.phase == "interior"]
        if isinstance(self.selection, TwoStage):
            if not interior or not any(v.phase == "serve" and not v.registered for v in self.values):
                raise ValueError("a two-stage spec commits boundary values (phase serve) and interior values (phase "
                                 "interior)")
            if any(v.phase == "in-deadline" for v in self.values):
                raise ValueError("a two-stage spec commits no in-deadline values")
        elif interior:
            raise ValueError(f"values {interior!r}: phase interior is a two-stage design's interior commitment, after "
                             "its first stage; a one-stage selection has none")
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
        if R.SEED_SALT in items and items[R.SEED_SALT].kind != "coins":
            raise ValueError(f"public item {R.SEED_SALT!r}: the window's seed salt is the challenger's coin, of kind "
                             "'coins'")
        for o in self.check.public_outputs:
            if o not in items or items[o].kind != "output":
                raise ValueError(f"public output {o!r} needs a public item of kind 'output', with its reason")
        if isinstance(self.selection, Sequential) and self.selection.population not in items:
            raise ValueError(f"a sequential rule's population {self.selection.population!r} is a declared public item")
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
    sampled selection; `exhaustive`: the fact of an accepted `All`. `public`: the check's public outputs, as the
    verifier reported them. `record`: the SHA-512 of the outcome's record, which binds the registration, the sample, the
    delivered sessions, the audit record and, for a sample a `Session` covers, the session's rounds."""

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
    only the site, θ and the registered compiler. The receipt's `received` is the record's position in
    the stream's log; the challenger's clock stays its own. Under `Check.registration_deadline`, `salted_at` and
    `received_at` are its readings of the salt's release and of this registration's receipt (None otherwise). `fanout`
    is the fan-out F its registration verdict published on the challenger's program (R8, `Verdict.fanout`): a two-stage
    registration's is its replay grain's, the N at which `plancheck.two` prices it in the capped form.
    `rests_on_period_check` is the tied values its verdict let through (`Verdict.rests_on_period_check`): its acceptance
    rests on the network period check. `returned` is the bits one replay block of a two-stage registration returns,
    counted on the challenger's own program (`second_stage.replay_returned`): no served cap S below it prices the
    registration (`plancheck.two`'s `returned`). None for a one-stage registration, an interior one, and a replay
    block whose links can't be derived."""

    record: Mapping[str, Any]
    receipt: Mapping[str, Any]
    digest: str
    order: int
    salted_at: float | None = None
    received_at: float | None = None
    fanout: int | None = field(default=None, compare=False)
    rests_on_period_check: tuple[str, ...] = field(default=(), compare=False)
    returned: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Draw:
    """The challenger's one sample for a registration: `law` (its sampling design, expanded over the challenger's own program) and
    `draw`, one-stage's sample object (with a work sample's `closure`), which goes to the ZK server. A two-stage design's
    second stage is the sample of its interior registration, and `replay_sample` is the first stage's, the sample of
    the registration it follows."""

    registration: WindowCommitmentRecord
    law: Mapping[str, Any]
    draw: Mapping[str, Any]
    replay_sample: Draw | None = None

    @property
    def units(self) -> list[int]:
        return list(self.draw["units"])

    @property
    def proved(self) -> list[int]:
        """The proof blocks the proof proves: the sampled ones and their closure, ascending (`draw.proved`)."""
        return D.proved(self.draw)


@dataclass(frozen=True)
class Session:
    """One proof session over the samples of consecutive registrations of the stream `stream` (its digest), in order: each
    covered round's registration is its own tables of the session's statement. The challenger opens it once
    (`Stream.session`), after the last covered sample; the first covered sample's proof deadline spans it."""

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


def check_public(spec: Spec, record: Any, values: Sequence[Value] | None = None,
                 query: Mapping[str, Any] | None = None) -> Verdict:
    """The public-items rule (the module docstring) on a one-stage registration record whose roots are of `values` (by
    default every declared Value) under `query` (by default the check's; a two-stage design's registration's is its
    grain). Codes:
    `keys` (a field one-stage doesn't define), `roots` (a root or registered entry that is not one of those Values' one
    salted root, such as a per-call root), `unsalted` (a scheme other than `SCHEME`), `query` (not `query`),
    `undeclared` (observable items the Spec doesn't declare, such as a leaf count or a template's population). An entry's
    fields are read in key order, so the detail is a function of the record's canonical JSON, which is all the Lean port
    (`verifier.check_public`, which a `Stream` runs) reads."""
    values = spec.values if values is None else tuple(values)
    query = spec.check.query if query is None else query
    c = _Collect()
    if not isinstance(record, Mapping):
        c.fail("keys", "not a record")
        return c.verdict()
    extra = sorted(str(k) for k in set(record) - _RECORD_KEYS)
    if record.get("scheme") != SCHEME:
        c.fail("unsalted", {"scheme": record.get("scheme"), "service": SCHEME})
    if record.get("query") != R.query_digest(query):
        c.fail("query", {"registered": record.get("query"), "check": R.query_digest(query)})
    need: list[str] = []
    if {"population", "program", "partition"} & set(record):
        need.append("population")
    law = record.get("law")
    if isinstance(law, Mapping) and isinstance(law.get("strata"), list):
        need += [f"population:{s.get('template') if isinstance(s, Mapping) else s}" for s in law["strata"]]
    if "window" in record:
        need.append("window")
    if isinstance(record.get("window"), Mapping) and R.SEED_SALT in record["window"]:
        need.append(R.SEED_SALT)
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


def selection_sizes(rule: Selection, program: Program, query: Mapping[str, Any]) -> list[int]:
    """The strata sizes a stratified rate's K reads (`draw.strata_sizes` of `program` under `query`), none for any other rule.
    Raises `Refused` (`rate`) where the query doesn't apply to a stratified rate, as `selection_table` does for a work rate.
    The Lean verifier's `Select.strataSizes`, which `Rule.over` reads only for a stratified rate, is this."""
    strata = D.parse_strata_rate(rule.text) if isinstance(rule, Law) else None
    if strata is None or strata[0]:
        return []
    try:
        return D.strata_sizes(program, query)
    except P.QueryRefused as e:
        raise _refuse("rate", {"law": rule.text, "reason": str(e)}) from None


def selection_law(rule: Selection, n: int, sizes: Sequence[int] = ()) -> dict[str, Any]:
    """The sampling design a selection rule names over `n` proof blocks in strata of `sizes` (`selection_sizes` of the check's
    program and query), before `draw.derive` expands it over them: `subset:n` for `All` and a `Law`'s own. A rate sampling
    design is `subset:K` at K = `draw.rate_count` over `n`, a stratified rate `stratified:K` at K = `draw.strat_k` over
    `sizes`, and a work rate `work:K` at K = ceil(NUM n / DEN) over `selection_table`'s work table, each raising `Refused`
    (`rate`) unless 1 <= K <= n, and a stratified rate without strata too. A two-stage design's is its first stage's, over
    `n` replay blocks. The Lean verifier's `Select.Rule.over`, then `Select.Rule.lawOver`, is this."""
    if isinstance(rule, All):
        return {"law": "subset", "k": n}
    text = rule.replay if isinstance(rule, TwoStage) else rule.text
    rate, strata = D.parse_rate(text), D.parse_strata_rate(text)
    if rate is None and strata is None:
        return D.parse_design(text)
    if rate is not None:
        kind, k = "subset", D.rate_count(*rate, n)
    elif strata[0]:
        kind, k = "work", D.rate_count(strata[1], strata[2], 1, n)
    else:
        kind, k = "stratified", D.strat_k(strata[1], strata[2], sizes, n)
    if not 1 <= k <= n or (kind == "stratified" and not sizes):
        raise _refuse("rate", {"law": text, "population": n, "k": k})
    return {"law": kind, "k": k}


def selection_table(rule: Selection, program: Program, query: Mapping[str, Any],
                    work_table: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The work table a selection rule's sampling design (`selection_law`'s) is expanded over, from the challenger's own
    `work_table`: a work rate's, each stratum of `program` under `query` with its floor at the rate (`draw.rate_table`), else
    `work_table` itself. Raises `Refused` (`rate`) where the query doesn't apply, a work rate has no work table or it names
    no work for a stratum's template. The Lean verifier's `Select.Rule.tableOver` is this."""
    strata = D.parse_strata_rate(rule.text) if isinstance(rule, Law) else None
    if strata is None or not strata[0]:
        return work_table
    try:
        return D.rate_table(program, query, strata[1], strata[2], work_table)
    except (P.QueryRefused, ValueError) as e:
        raise _refuse("rate", {"law": rule.text, "reason": str(e)}) from None


def params_law(law: Any, program: Program, query: Mapping[str, Any],
               work_table: Mapping[str, Any] | None) -> tuple[Any, Mapping[str, Any] | None]:
    """One-stage's parameters' agreed sampling design and the work table it is expanded over: a law object as given, with
    `work_table`, or a rate as a selection rule states it (`draw.parse_rate`, `draw.parse_strata_rate`), its sampling design
    over the population of `program` under `query` and its strata (`selection_law`) and its work table (`selection_table`). Raises
    `ValueError` for any other text, a query that doesn't apply, or a rate `selection_law` or `selection_table` refuses.
    The Lean verifier's `OneStage.design` is this."""
    if not isinstance(law, str):
        return law, work_table
    if D.parse_rate(law) is None and D.parse_strata_rate(law) is None:
        raise ValueError("a law stated as text is a rate, rate:NUM/DEN[:T], stratified-rate:NUM/DEN or work-rate:NUM/DEN, "
                         f"not {law!r}")
    rule = Law(law)
    try:
        law = selection_law(rule, P.units(program, query), selection_sizes(rule, program, query))
        return law, selection_table(rule, program, query, work_table)
    except (P.QueryRefused, Refused) as e:
        raise ValueError(str(e)) from None


#: The refusal of a `bernoulli` sampling design (`bernoulli`), the Lean verifier's `Draw.BERNOULLI`.
BERNOULLI = "the law is bernoulli, which the sampled audit's soundness statement doesn't cover"


def bernoulli(law: Mapping[str, Any]) -> bool:
    """Whether `law` (a `selection_law`) is `bernoulli`, at any p, which the sampled audit's soundness statement doesn't
    cover: at 0 < p < 1 its sampled count varies, and p = 0 (nothing sampled) and p = 1 aren't covered either. A work
    sample's closure varies too, and `Stream.select` refuses a closure map (`closure`). The Lean verifier's
    `Draw.bernoulli` is this. A Spec that wants Bernoulli's public rate at a fixed count states the rate sampling design,
    `rate:NUM/DEN[:T]`, and one that wants every proof block, `All`."""
    return law.get("law") == "bernoulli"


def select_draw(rule: Selection, law: Mapping[str, Any], n: int, coins: Callable[[int], bytes], *,
                closure: Mapping[int, Sequence[int]] | None = None, sampler: Sampler | None = None) -> dict[str, Any]:
    """The challenger's sample object over [0, n) under its expanded `law`: every proof block for `All`, reading no coins;
    otherwise `sampler`'s sample, or `coins.draw_first` on `coins` read in order. With `closure`, the challenger's
    closure map, a work sample carries the sampled proof blocks' closure (`coins.closure_of`) when it is not empty. Raises
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


def replay_cells(replay: Program, query: Mapping[str, Any]) -> tuple[int, ...]:
    """Each template's proof blocks in one replay block (n_t), in the order the check's template query names them: the
    query applied to the replay block's Program, counted by template. Raises `ValueError` where the query doesn't
    apply, and where the query doesn't name each template of the replay block once, in the order of its first instance
    in it, so that template t's cells are the statement's stratum t. The Lean verifier's `Select.replayCells` is this."""
    try:
        ranges = P.ranges(replay, query)
    except (P.QueryRefused, ValueError) as e:
        raise ValueError(f"the check's query doesn't apply to the replay block: {e}") from None
    ids = _check_ids(query)
    first = list(dict.fromkeys(t for t, _lo, _n in ranges))
    if ids is None or first != ids:
        raise ValueError(f"the check's templates {ids} are not the replay block's, each once, in the order of its first "
                         f"instance in it ({first})")
    return tuple(sum(n for t, _lo, n in ranges if t == i) for i in ids)


def _check_ids(query: Any) -> list[str] | None:
    """The templates the check's template query names, in its order, its parameters checked as core's `validate` does;
    None for any other query. The Lean verifier's `Select.templateIds` (`Flock.TemplateQuery.templatesOf`) is this."""
    if not isinstance(query, Mapping):
        return None
    name = query.get("name") if isinstance(query.get("name"), str) else ""
    version = query.get("version") if _nat(query.get("version")) else 0
    params = query.get("params")
    keys = sorted(params) if isinstance(params, Mapping) else []
    if (name, version) == ("Q_template_instance", 0):
        return [params["template"]] if keys == ["template"] and isinstance(params["template"], str) else None
    ts = params.get("templates") if (name, version) == ("Q_template_instances", 0) and keys == ["templates"] else None
    ok = isinstance(ts, list) and ts and all(isinstance(t, str) for t in ts) and len(set(ts)) == len(ts)
    return list(ts) if ok else None


def template_circuits(proofs: Mapping[str, Any] | None, query: Mapping[str, Any]
                      ) -> tuple[tuple[str, str, dict[str, Any], dict[str, Any] | None], ...]:
    """The auditor's params naming each template's circuit file and its registered reference statement's header and
    registered reads (the module docstring), a two-stage run's `interior.params.proofs`: `{"templates": [{"template":
    <descriptor id>, "circuit": <SHA-512, lowercase hex>, "header": <the registered reference statement's header>,
    "registered": <the registered reads it reads, null for none>}, ...]}`, as (template, circuit, header,
    registered) in the order the check's template query `query` names the templates, which is REPLAY_c's first-instance
    order (`replay_cells`). None names no file and gives none. Raises `ValueError` for any other form, a circuit that
    isn't 128 lowercase hex digits, a header that isn't an object, registered reads neither an object nor null, and a
    list that doesn't name each of the check's templates once, in its order. The Lean verifier's `Run.templateCircuits`
    is this."""
    if proofs is None:
        return ()
    ids = _check_ids(query)
    if ids is None:
        raise ValueError("a two-stage design's check is a template query")
    form = ('the params\' proofs are {"templates": [{"template": <descriptor id>, "circuit": <SHA-512>, '
            '"header": <object>, "registered": <object or null>}, ...]}')
    if not isinstance(proofs, Mapping) or set(proofs) != {"templates"} or not isinstance(proofs["templates"], (list, tuple)):
        raise ValueError(form)
    named = []
    for e in proofs["templates"]:
        if not (isinstance(e, Mapping) and set(e) == {"template", "circuit", "header", "registered"}
                and isinstance(e["template"], str) and isinstance(e["circuit"], str) and _DIGEST.fullmatch(e["circuit"])
                and isinstance(e["header"], dict) and (e["registered"] is None or isinstance(e["registered"], dict))):
            raise ValueError(f"{form}, each circuit a SHA-512 in lowercase hex, each header an object and each "
                             f"registered an object or null: {e!r}")
        named.append((e["template"], e["circuit"], e["header"], e["registered"]))
    if [t for t, *_ in named] != ids:
        raise ValueError(f"the params name the circuits of {[t for t, *_ in named]}, not the check's templates {ids}, "
                         "each once, in its order")
    return tuple(named)


def tables_served(record: Any, tables: Iterable[tuple[str, bytes, bytes]]) -> bool:
    """Whether each history table the interior reads is one the boundary registration served (`interior-table`): every
    table of `tables`, `(name, root, domain)` with the root and domain id `history.roots_of` computed for it, is a port
    of the boundary registration's record `record` of that name with that root and domain, at the port's last root
    (`Record.byPort`, the one round 1's checks hold). The interior's copies name the tables round 2 recomputes from the
    writers it opens; this holds each one to the table round 1 registered under its name. The Lean verifier's
    `Run.tablesServed` is this."""
    roots = record.get("roots") if isinstance(record, Mapping) else None
    ports = {r["port"]: r for r in (roots if isinstance(roots, list) else ())
             if isinstance(r, Mapping) and isinstance(r.get("port"), str)}
    return all((r := ports.get(name)) is not None and r.get("root") == root.hex() and r.get("domain") == domain.hex()
               for name, root, domain in tables)


def second_counts(k: int | Mapping[str, Any], ids: Sequence[str], ns: Sequence[int]) -> tuple[int, ...]:
    """The second stage's count in each cell of template t, in the check's order: the rule's `k` (or its k_t), and n_t for
    a cell of fewer proof blocks (proved whole). Raises `ValueError` for a map that doesn't name each of the check's
    templates `ids` once. The Lean verifier's `Select.Counts.over` is this."""
    if isinstance(k, Mapping):
        if set(k) != set(ids):
            raise ValueError(f"a two-stage rule's counts name each of the check's templates once: {sorted(k)} vs {list(ids)}")
        return tuple(min(k[t], n) for t, n in zip(ids, ns))
    return tuple(min(k, n) for n in ns)


def _offsets(q: int, ns: Sequence[int]) -> list[int]:
    """Each template's first interior proof block, over a replay sample of `q`: Q·Σ_{t'<t} n_t'."""
    return [q * sum(ns[:t]) for t in range(len(ns))]


def second_law(sample: Sequence[int], ns: Sequence[int], ks: Sequence[int]) -> dict[str, Any]:
    """The second stage's sampling design over the interior's len(sample)·Σ n_t proof blocks, template-major: template
    t's being [off_t, off_t + Q·n_t), off_t = Q·Σ_{t'<t} n_t', and sampled replay block `sample[i]`'s instances of it
    [off_t + i·n_t, off_t + (i+1)·n_t), one cell. One stratum per cell, template by template and in each by sampled
    replay block, named `r/t` (replay block r of the registration, template t of the check's), `ks[t]` of the cell's
    `ns[t]` proof blocks in each; the strata tile [0, Q·Σ n_t) in order. The stratified law sizes each stratum's count
    in proportion to its size, which cells of several templates' sizes don't follow, so the law is the work law at
    k = 1 and unit work: its rule (`draw.work_k`) gives each stratum max(floor, 1) capped at its size, and each cell's
    floor is its count. The Lean verifier's `Select.secondLaw` is this."""
    off = _offsets(len(sample), ns)
    return {"law": "work", "k": 1,
            "strata": [{"template": f"{r}/{t}", "units": [[off[t] + i * n, off[t] + (i + 1) * n]], "work": 1,
                        "floor": ks[t], "k": ks[t]}
                       for t, n in enumerate(ns) for i, r in enumerate(sample)]}


def select_second(sample: Sequence[int], ns: Sequence[int], ks: Sequence[int], coins: Callable[[int], bytes], *,
                  sampler: Sampler | None = None) -> dict[str, Any]:
    """The second stage: `second_law`'s sample object over the replay sample's proof blocks, `sampler`'s sample or
    `coins.draw_first` on `coins` (the challenger's, opened after the interior registration's digest), held to
    `draw.check_draw`. Raises `ValueError` for a replay sample that isn't ascending replay blocks, each once, cells that
    aren't one or more templates' of at least one proof block each, a replay block of fewer than two proof blocks (its
    design is `Law`'s), a count not in [1, n_t], counts that prove every cell whole (whole replay blocks), a sample not
    well formed under its sampling design, and where `draw_first` does. The Lean verifier's `Select.selectSecond` is
    this."""
    if not (sample and all(type(r) is int and r >= 0 for r in sample) and all(a < b for a, b in zip(sample, sample[1:]))):
        raise ValueError("a replay sample is one or more replay blocks, ascending, each once")
    if not (ns and len(ks) == len(ns) and all(type(n) is int and n >= 1 for n in ns)):
        raise ValueError("a replay block's cells are one or more templates', each of at least one proof block, with a "
                         "count each")
    if sum(ns) < 2:
        raise ValueError(f"a replay block of {sum(ns)} proof blocks is a proof block: its rule is 'law'")
    if not all(type(k) is int and 1 <= k <= n for k, n in zip(ks, ns)):
        raise ValueError(f"the second stage samples 1 to n_t of each cell's n_t proof blocks: counts {list(ks)} over "
                         f"cells of {list(ns)}")
    if list(ks) == list(ns):
        raise ValueError(f"counts {list(ks)} prove every cell of {list(ns)} whole, and so whole replay blocks")
    law, m = second_law(sample, ns, ks), len(sample) * sum(ns)
    units = sorted(int(u) for u in (sampler(dict(law), m) if sampler is not None
                                    else draw_first(OwnRandomness(coins), law, m)["units"]))
    obj = {**law, "population": m, "units": units}
    v = D.check_draw(obj, {"population": m}, law)
    if not v.ok:
        raise ValueError(f"the second stage's draw is not well formed under its law: {list(v.codes)}")
    return obj


@dataclass
class _Entry:
    registration: WindowCommitmentRecord
    law: Mapping[str, Any]
    program: Program
    work_table: Mapping[str, Any] | None = None
    #: what `register` handed R1-R8 besides the record: (ports, leaf_layer, anchors, the log's records, registered,
    #: supplied, pinned, schemas, window_root, held, outputs), its window's seed salt and its history tables
    copies: tuple[Any, ...] = ()
    seed_salt: str | None = None
    history: Mapping[str, Any] | None = None
    read: bytes | None = None
    closure: Mapping[int, Sequence[int]] | None = None
    draw: Draw | None = None
    drawn_at: float | None = None
    sessions: list[str] | None = None
    outcome: Outcome | None = None
    session: Session | None = None
    #: under a two-stage design: the interior registration, its population program and what `register_interiors`
    #: handed R1-R8 (as `copies`, `seed_salt` and `history`), then the second stage's sample and the coins it read
    interior: WindowCommitmentRecord | None = None
    interior_program: Program | None = None
    interior_copies: tuple[Any, ...] = ()
    interior_seed_salt: str | None = None
    interior_history: Mapping[str, Any] | None = None
    second: Draw | None = None
    second_read: bytes | None = None


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
    """The token `_Control` and `Spot.Stream` know a registration by: equal exactly when the registrations are."""
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
_OPS = ("ready", "salt", "salt_released", "register", "select", "interiors", "second", "session", "deliver",
        "deliver_session", "outcome", "halt")
#: What the stream's rules hold of an empty stream.
_EMPTY = {"size": 0, "opened": [], "failures": 0, "salted_at": None, "ended": None, "halted": None, "entries": []}


class _Control:
    """A stream's own rules, the reference for the verifier's (`Spot.Stream`, whose docstring has the
    state and the ops): the failure budget, the registrations open at once, the epoch's salt and registration deadline,
    and each registration's once-only orders (sampled, session, delivered, final) and proof deadline, on what
    they hold of the stream that one op reads (`_step`). A refusal's or a pending outcome's `detail` is the stream's
    own, which the Lean doesn't write. `ValueError` is an op or a state the Lean doesn't read."""

    def __init__(self, params: Mapping[str, Any], state: Mapping[str, Any]) -> None:
        self.params = params
        self.two_stage = params.get("two_stage") is True
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

    def _staged(self, op: Mapping[str, Any]) -> dict[str, Any]:
        if not self.two_stage:
            _no("rule", "a one-stage stream has no interior registration or second stage")
        e = self._drawn(op)
        if e["final"]:
            _no("final", {"order": op["order"]})
        if e["delivered"]:
            _no("delivered", {"order": op["order"]})
        return e

    def _interiors(self, op: Mapping[str, Any]) -> dict[str, Any]:
        e = self._staged(op)
        if e.get("interior") is not None:
            _no("interior-registered", {"order": op["order"]})
        self._checks(op)
        e["interior"] = _token(_need(op, "interior"), "interior")
        return {"ok": None}

    def _second(self, op: Mapping[str, Any]) -> dict[str, Any]:
        e = self._staged(op)
        if e.get("interior") != _token(op.get("interior"), "interior"):
            _no("no-interior", {"order": op["order"]})
        if e.get("second") is not None:
            _no("drawn", {"order": op["order"]})
        self._checks(op)
        e["second"] = _token(_need(op, "second"), "second")
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
        if self.two_stage:
            _no("rule", "each sample of a two-stage stream is proved after its own interior registration")
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
        if self.two_stage and not late:
            if e.get("second") is None:
                _no("second", {"order": op["order"]})
            if op.get("second") != e["second"]:
                _no("not-drawn", {"order": op["order"]})
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
    """A fleet's own rules, the reference for the verifier's (`Spot.Fleet`, whose docstring has the
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
    `Spot.Fleet` against `_FleetControl`); where the two disagree the fleet stops, refusing every op after it
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


def _stream_values(values: Sequence[Any]) -> tuple[Any, ...]:
    """A stream's values as tuples, each width a tuple where it is a leaf type's shape."""
    return tuple(tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, (list, tuple)) else v
                 for v in values)


class Stream:
    """One protocol's stream for one `Spec`, held by the challenger.

    `unit` is the challenger's own copy of the proof-block check's Definition: its Program's digest is the check's, and the
    challenger builds each registration's population program from it and the registration's N. Or `unit` is a window's
    templates, a sequence of Definitions (a TP rank's served window: each provable family of the step): the check's
    `program` is their digests and its query `Q_template_instances` over them, both in the window's order, and each
    registration's population program is their mixed population (`partition.mixed_population_program`) at the counts
    `register` is given, under one sample by the Spec's sampling design (`stratified:K`: a stratum per template, each sampled at least
    once; `work:K`: at least the work table's floor). It is one program and one query, as for one template, so the
    verifier's sample and run (`Select.select`, `Run.run`) read it in the same form. `budget` (b), `delta`
    (the profile's) and `rounds` (R, the most registrations open at once, and so the most one `Session` covers) are
    public parameters of the stream. `coins` is
    the auditor's challenger's coins, a `coins.ChallengerCoins` (n bytes a call, read in order by `OwnRandomness`), which
    the stream's digest names: the verifier has no coins of its own, so there is no default, and a stream whose
    selection reads coins refuses any other source (`NO_OWN_COINS`). Every outcome's run by the verifier needs the
    coins' evidence (`outcome`'s `coins`), refusing a sample without it. A stream that reads no coins (`All`, or one a
    `sampler` samples for) needs none and reads none it is given. `clock` is its clock in seconds; tests inject both, the coins from a challenger of
    their own. `sampler`, when given, samples instead of the coins (`flock-verify draw` in a process of its own, say), never
    beside the challenger's: it takes the challenger's sampling design and N, and its sample is held to them like any
    other. `prover`, when given, is the identity the challenger registered the stream's prover under
    (lowercase hex): the stream's digest names it, and every registration is bound to the stream and the prover
    (`binding`, `registration.check`'s): its window names both, and a window salt is bound to them, so a tile seeded for
    one prover's stream is refused in any other.

    A stream runs one epoch with one salt: `salt` issues it from the coins, or `salt_released` records one released
    elsewhere with its release, and under `Check.registration_deadline` every registration is received within the
    deadline of that release, or the epoch ends (`ended`) at the first that isn't. A window's seed salt is
    `seed_salt`'s, drawn at the window's open, and both verifiers hold every registration of the window to name it
    (R6). That it is drawn once a window and before the window's first registration is the stream's own bookkeeping,
    outside the rules below: a verifier that didn't draw it reads it from the challenger's log, which shows the one coin
    (`coins.seed_salt_of`). `epoch_log` holds the challenger's clock readings.

    The stream's own rules (the budget, the open registrations, the salt and both deadlines, and each registration's
    once-only orders) decide each op by the verifier (`verifier.stream`: the Lean's `Spot.Stream`
    against `_Control`), the clock's readings as floats, on what they hold of the stream that the op reads (`_slice`: its
    counts and the entries it names). What they hold is the verifier's state after each op, from the empty
    stream, so an op is decided on a state of a bounded size whatever the stream's length. Where the two disagree the
    stream stops: that op and every one after it is refused (`disagree`). A stream attached to a `Fleet`
    (`Fleet.attach`) is admitted by it at each registration and reports each failure to it, and halts with its operator.

    As its log grows, a stream keeps what one-stage's R6 and R7 read of it: where each window was logged, and each
    registered value's first registration. So a registration's checks read the same few records whatever the stream's
    history (`_earlier`), with the verdicts the whole log gives.

    A window's stream may take `values`, the values its mixed population takes ahead of its templates' parameters
    (`partition.mixed_population_program`'s, `(name, leaves, width)` each, `width` an integer or a leaf type's shape, a
    `Digest`'s `(8, 64, 1)`), which a registration registers as its program's (R7, which reads each by its type: a
    supplied value of `Digest`s in Digest order); the stream's digest names them. Or a window's stream takes `program`, the challenger's own copy of
    the window's Program where it is another form than the mixed population (the network certifier's scan in its
    generator form, one batch a template reading a shared input, say): its proof blocks under the check's query are the
    templates', one range each, in the window's order, and it is each registration's population program, a registration
    whose `counts` aren't its ranges' refused (`population`); the stream's digest names it. `log` is the prover's records the challenger logged before the
    stream, a registration of another Program whose values this stream's registrations reuse (`register`'s
    `outputs`), say: the stream's log starts with them, in order, so R6 and R7 read them, and none of them is sampled.

    Under a two-stage design, `replay` is the challenger's own copy of the replay block's Definition: the grain is its
    template query, each registration's population program is its N instances, and its `n_rho` proof blocks are the
    instances of `unit` (a window's templates) in its own Program under the check's query, `cells` of them each
    template's (n_t, `replay_cells`) and `counts` the second stage's in each of its cells (k_t, `second_counts`). A
    registration is then the boundary commitment and its `select` the first stage; `register_interiors` takes the
    interior registration (`interior_program`), a second `select` on it is the second stage, and `deliver` and
    `outcome` take that sample (the module docstring). A window's `values` are then the interior's, and so are its
    `span_values`, `(name, leaves, width)` each in `values`' form with `leaves` a sampled span's: the interior program
    over a replay sample of Q takes each at `leaves * Q`, after `values`, since under a rate sampling design Q is what the
    first stage's coins chose (a table read's supplied root, one `Digest` a sampled span, say); the stream's digest names
    them.
    """

    def __init__(self, spec: Spec, unit: SpecializedDefinition | Sequence[SpecializedDefinition], *, budget: int, delta: float, challenger: str,
                 coins: ChallengerCoins | None = None, clock: Callable[[], float] = time.monotonic,
                 sampler: Sampler | None = None, rounds: int = 1, prover: str | None = None,
                 values: Sequence[tuple[str, int, int | Sequence[int]]] = (), log: Sequence[Mapping[str, Any]] = (),
                 replay: SpecializedDefinition | None = None, program: Program | None = None,
                 span_values: Sequence[tuple[str, int, int | Sequence[int]]] = ()) -> None:
        if isinstance(spec.selection, Sequential):
            raise ValueError("the service doesn't run a Sequential selection yet")
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
        if isinstance(coins, ChallengerCoins) and sampler is not None:
            raise ValueError("a stream drawing from the challenger's coins takes no sampler, which would draw instead")
        if not (isinstance(coins, ChallengerCoins) or sampler is not None or isinstance(spec.selection, All)):
            raise ValueError(NO_OWN_COINS)
        values, span_values = _stream_values(values), _stream_values(span_values)
        if values and isinstance(unit, SpecializedDefinition):
            raise ValueError("a stream of one template's instances takes no values: a window with values is a mixed "
                             "population of its templates (give its unit as a sequence of them)")
        if span_values and (isinstance(unit, SpecializedDefinition) or replay is None):
            raise ValueError("a stream's per-span values are a two-stage window's interior values: give its unit as a "
                             "sequence of templates and its replay block")
        if values or span_values:
            P.mixed_population_program([(u, 1) for u in units], values=values + span_values)
        if program is not None:
            if isinstance(unit, SpecializedDefinition) or values or replay is not None:
                raise ValueError("a stream's program is a window's, of its templates, given in place of their mixed "
                                 "population: it takes no values and no replay block")
            if not isinstance(program, Program):
                raise ValueError(f"a window's program is a Program, not {type(program).__name__}")
            try:
                got = [tid for tid, _lo, _n in P.ranges(program, query)]
            except (P.QueryRefused, ValueError) as e:
                raise ValueError(f"the check's query does not apply to the window's program: {e}") from None
            if got != [P.template_id(u) for u in units]:
                raise ValueError(f"the window's program's proof blocks are not one range a template, in the window's "
                                 f"order: its ranges are {got}")
        try:
            earlier = [(dict(r), R.digest(r)) for r in log]
            R.held_roots(r for r, _d in earlier)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise ValueError("a stream's earlier log is the prover's records, each a JSON object whose registered "
                             f"entries name their value, domain and root: {type(e).__name__}: {e}") from None
        self.replay, self.n_rho, self.cells, self.counts = replay, None, (), ()
        if isinstance(spec.selection, TwoStage):
            self.cells, self.counts = self._replay_blocks(spec, units, replay, rounds)
            self.n_rho = sum(self.cells)
        elif replay is not None:
            raise ValueError("only a two-stage stream has a replay block (Stream(replay=))")
        self.spec, self.unit, self.budget, self.delta, self.challenger = spec, unit, budget, delta, challenger
        self.query = spec.selection.grain if isinstance(spec.selection, TwoStage) else spec.check.query
        self.templates = None if isinstance(unit, SpecializedDefinition) else units
        self.rounds, self.values, self.span_values, self.program = rounds, values, span_values, program
        self._coins, self._clock, self._sampler = coins, clock, sampler
        self.digest = _sha512({"format": STREAM, "spec": spec.digest(), "budget": budget, "delta": delta,
                               "challenger": challenger, **({"rounds": rounds} if rounds > 1 else {}),
                               **({"prover": prover} if prover is not None else {}),
                               **({"coins": "challenger"} if isinstance(coins, ChallengerCoins) else {}),
                               **({"values": [list(v) for v in values]} if values else {}),
                               **({"span_values": [list(v) for v in span_values]} if span_values else {}),
                               **({"program": P.program_digest(program)} if program is not None else {})})
        self.binding = None if prover is None else R._binding({"stream": self.digest, "prover": prover})
        self._params = {"budget": budget, "rounds": rounds,
                        "registration_deadline": _reading(spec.check.registration_deadline),
                        "proof_deadline": float(spec.check.proof_deadline), "stream": self.digest,
                        **({"two_stage": True} if self.n_rho is not None else {})}
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
        self._seed_salts: dict[str, bytes] = {}
        self._fleet: Fleet | None = None
        self._fleet_index: int | None = None
        self._digests: dict[str, int] = {}
        for r, d in earlier:
            self._logged(r, d)

    @staticmethod
    def _replay_blocks(spec: Spec, units: Sequence[SpecializedDefinition], replay: Any,
                       rounds: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """A two-stage stream's cells, each template's proof blocks in one replay block (`replay_cells`, n_t in the
        check's order), and the second stage's count in each (`second_counts`), once its copies fit the design; raises
        `ValueError` otherwise."""
        rule = spec.selection
        assert isinstance(rule, TwoStage)
        if not isinstance(replay, SpecializedDefinition):
            raise ValueError("a two-stage stream takes the challenger's copy of the replay block's Definition (replay=)")
        if dict(rule.grain) != P.template_instance_query(replay):
            raise ValueError("the two-stage design's grain is not the template query over the challenger's replay block")
        program = Program(replay)
        ns = replay_cells(program, spec.check.query)
        if sum(ns) < 2:
            raise ValueError(f"a replay block of {sum(ns)} proof blocks is a proof block: its rule is 'law'")
        try:
            gates = P.unit_gates(program, spec.check.query)
        except (P.QueryRefused, ValueError) as e:
            raise ValueError("a proof block of the replay block has no gates of its own") from e
        if len(gates) != sum(ns):
            raise ValueError("a proof block of the replay block has no gates of its own")
        ks = second_counts(rule.k, [P.template_id(u) for u in units], ns)
        if ks == ns:
            raise ValueError(f"the second stage's counts {list(rule.k.values()) if isinstance(rule.k, Mapping) else rule.k}"
                             f" prove every cell of {list(ns)} whole, and so whole replay blocks")
        if rounds != 1:
            raise ValueError("each sample of a two-stage stream is proved after its own interior registration: one round")
        return ns, ks

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
        (with `"external": true` from `salt_released`, `salted_at` being its `released_at`), each window's seed salt,
        `{"event": "seed-salt", "window_root", "seed_salt": <hex>, "seeded_at"}` (`seed_salt`), and under `Check.registration_deadline` each registration received, `{"event": "registration", "order",
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
        """The epoch's salt coin: `n` bytes of the challenger's own coins (`OwnRandomness`, in order with the samples'
        reads), issued once, and only from a `coins.ChallengerCoins` (`ValueError`, `NO_OWN_COINS`, before any byte is
        read). Its release is read on the challenger's clock (`salted_at`) and logged (`epoch_log`);
        `Check.registration_deadline` counts from it. Raises `Refused`: `salted` (the epoch has a salt, from `salt` or
        `salt_released`: nothing resamples it) or `epoch-ended` (the epoch ended before any salt)."""
        if not (type(n) is int and n >= 1):
            raise ValueError(f"a salt is a positive number of bytes, not {n!r}")
        if not isinstance(self._coins, ChallengerCoins):
            raise ValueError(NO_OWN_COINS)
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

    def seed_salt(self, coins: ChallengerCoins, window_root: str) -> bytes:
        """The window's seed salt, drawn at the window's open, before it serves (`coins.draw_seed_salt`): the first coin
        of the challenger's seed-salt family (`coins`, a `coins.ChallengerCoins` over that family of the window's
        session), released after the window's open root `window_root`. Its release is read on the challenger's clock
        (`seeded_at`) and logged (`epoch_log`). Each registration of the window (`register`'s and `register_interiors`'
        `window_root`) is then held to it by both verifiers: its window names it (`registration.SEED_SALT`), and a
        window that names another, or none, is refused (`R6-seed-salt`), as is a seed salt where the stream drew none.
        Raises `Refused`: `seed-salted` (the window has its salt: nothing redraws it) or `seed-salt-late` (the stream
        received a registration of the window already); and `CoinError` where the draw fails."""
        if isinstance(window_root, str) and window_root in self._seed_salts:
            raise _refuse("seed-salted", {"window_root": window_root})
        if isinstance(window_root, str) and any(window_root in (*e.copies[8:9], *e.interior_copies[8:9])
                                                for e in self._entries):
            raise _refuse("seed-salt-late", {"window_root": window_root})
        salt = draw_seed_salt(coins, window_root)
        seeded_at = self._clock()
        self._seed_salts[window_root] = salt
        self._epoch_log.append({"event": "seed-salt", "window_root": window_root, "seed_salt": salt.hex(),
                                "seeded_at": seeded_at})
        return salt

    def _seed_salt_of(self, window_root: str | None) -> str | None:
        salt = self._seed_salts.get(window_root) if isinstance(window_root, str) else None
        return None if salt is None else salt.hex()

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], *, leaf_layer: str | None = None,
                 anchors: Sequence[Any] = (), registered: Mapping[str, int] | None = None,
                 work_table: Mapping[str, Any] | None = None, supplied: Mapping[str, Any] | None = None,
                 pinned: Mapping[str, str] | None = None, schemas: Mapping[str, str] | None = None,
                 window_root: str | None = None, held: Sequence[str] | None = None,
                 counts: Mapping[str, int] | None = None,
                 outputs: Mapping[str, Mapping[str, str]] | None = None,
                 history: Mapping[str, Any] | None = None) -> WindowCommitmentRecord:
        """Receive a registration: the public-items rule (`check_public`), then one-stage's R1-R8 (`registration.check`),
        each by the verifier (`verifier.check_public`, `verifier.check`), against the challenger's own copies:
        the population program it builds from `unit` and the record's N (a window's: the mixed population of its
        templates, in its order, at `counts`, the challenger's own `{template's descriptor id: instances}` of the window,
        every template's at least one, taking the stream's `values`; or the stream's own `program`, its ranges `counts`),
        the check's query, the selection rule's sampling
        design, `SCHEME`, the stream's log (the records of it R6 and R7 read, `_earlier`), and the per-registration
        copies given here (`ports`, `leaf_layer`, `anchors`, `registered`, `work_table`, `supplied`, `pinned`,
        `schemas`, `window_root`, `held`, `outputs` and `history`, as `registration.check` takes them). Then the receipt. Under
        `Check.registration_deadline` the challenger first reads its clock (`received_at`): a registration received more
        than the deadline after the salt's release, or before any salt, is refused (`late-registration`, with both
        readings) whatever it holds, logged, and ends the epoch; an on-time one carries both readings
        (`WindowCommitmentRecord.salted_at`, `received_at`) and is logged with them. Raises
        `Refused`: when the stream has stopped or halted (`halted`) or its epoch ended (`epoch-ended`), while it holds as
        many registrations without a final outcome as it may (`open`, `ready`), when it is late, when a window's `counts`
        aren't at least one instance for each of its templates and N in all, or a single proof block's stream is given any
        (`population`), when the rule's rate sampling design
        samples none or more than N proof blocks or a stratified rate's strata don't apply (`rate`, `selection_law`, before
        R1-R8) or a work rate's strata or work table don't apply (`rate`, `selection_table`, R1-R8 then reading its table),
        when either check fails, or, after both,
        when the sampling design is `bernoulli` at any p (`bernoulli`, `BERNOULLI`), when the program's proof blocks read stratum by stratum
        are not its N proof blocks in order (`population-order`, `partition.in_order`), or its fleet refuses it (`halted`)."""
        deadline = self.spec.check.registration_deadline
        op: dict[str, Any] = {"op": "register"}
        got: dict[str, Any] = {}
        seed_salt = self._seed_salt_of(window_root)

        def received() -> float:
            got["received_at"] = self._clock()
            return float(got["received_at"])

        def checks() -> list[str] | None:
            v = self._public(record)
            n = record.get("population")
            if v.ok and not (type(n) is int and n >= 1):
                v = _refuse("population", n).verdict
            if v.ok:
                program = self._program(n, counts)
                if program is None:
                    v = _refuse("population", {"counts": counts, "templates": None if self.templates is None else
                                               [P.template_id(t) for t in self.templates]}).verdict
            if v.ok:
                try:
                    law = selection_law(self.spec.selection, n, selection_sizes(self.spec.selection, program, self.query))
                except Refused as r:
                    v = r.verdict
            if v.ok:
                try:
                    table = selection_table(self.spec.selection, program, self.query, work_table)
                except Refused as r:
                    v = r.verdict
            if v.ok:
                earlier = self._earlier(record, supplied, outputs)
                v = V.check(record, program, self.query, law, SCHEME, ports, leaf_layer, anchors, earlier, None,
                            table, registered, supplied, pinned, self.binding, schemas, window_root, held,
                            outputs=outputs, seed_salt=seed_salt, tied=self.spec.check.tied or None,
                            history=history)
                if v.ok and bernoulli(law):
                    v = _refuse("bernoulli", {"law": law, "reason": BERNOULLI}).verdict
                if v.ok and not P.in_order(program, self.query, n):
                    v = _refuse("population-order", {"population": n}).verdict
                got.update(program=program, law=law, table=table, earlier=earlier)
            if v.ok and self._fleet is not None:
                try:
                    self._fleet._admit(self)
                except Refused as r:
                    v = r.verdict
            got["checks"] = v
            if not v.ok:
                return list(v.codes)
            salted_at = None if deadline is None else self._salted_at
            got["registration"] = self._registration(record, len(self._entries), salted_at, got.get("received_at"),
                                                     v.fanout, v.rests_on_period_check, self._returned(got["program"]))
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
        self._entries.append(_Entry(reg, D.derive(program, self.query, law, got["table"]), program, work_table,
                                    (ports, leaf_layer, anchors, earlier, registered, supplied, pinned, schemas,
                                     window_root, held, outputs), seed_salt, history))
        if deadline is not None:
            self._epoch_log.append({"event": "registration", "order": reg.order, "registration": reg.digest,
                                    "salted_at": reg.salted_at, "received_at": reg.received_at})
        return reg

    def _program(self, n: int, counts: Mapping[str, int] | None) -> Program | None:
        """The challenger's population program for a registration of N = `n`: `unit`'s N instances, or a window's mixed
        population at `counts` (`register`'s), which number N in all, taking the stream's `values`, or the stream's own
        `program` where its ranges are `counts`; None when `counts` don't fit the stream. A two-stage stream's is
        `replay`'s N instances, its replay blocks."""
        if self.templates is None or self.replay is not None:
            return P.population_program(self.replay or self.unit, n) if counts is None else None
        ids = [P.template_id(t) for t in self.templates]
        if not (isinstance(counts, Mapping) and set(counts) == set(ids)
                and all(type(counts[i]) is int and counts[i] >= 1 for i in ids) and sum(counts.values()) == n):
            return None
        if self.program is not None:
            ranges = {tid: m for tid, _lo, m in P.ranges(self.program, self.query)}
            return self.program if ranges == dict(counts) else None
        return P.mixed_population_program([(t, counts[i]) for t, i in zip(self.templates, ids)], values=self.values)

    def _public(self, record: Mapping[str, Any], interior: bool = False) -> Verdict:
        """`check_public` by the verifier (`verifier.check_public`): on a registration's values under its query, or on
        a two-stage design's interior values under the check's."""
        values = self._values(interior)
        query = self.spec.check.query if interior else self.query
        return V.check_public(check_public(self.spec, record, values, query), record, SCHEME, query,
                              [(v.name, v.registered) for v in values], [p.item for p in self.spec.public])

    def _values(self, interior: bool = False) -> tuple[Value, ...]:
        if interior:
            return interior_values(self.spec.values)
        return boundary_values(self.spec.values) if self.n_rho is not None else self.spec.values

    def _registration(self, record: Mapping[str, Any], order: int, salted_at: float | None = None,
                      received_at: float | None = None, fanout: int | None = None,
                      rests_on_period_check: tuple[str, ...] = (), returned: int | None = None) -> WindowCommitmentRecord:
        return WindowCommitmentRecord(dict(record), R.receipt(record, self.challenger, received=str(len(self._log))),
                                      R.digest(record), order, salted_at, received_at, fanout, rests_on_period_check,
                                      returned)

    def _returned(self, program: Program) -> int | None:
        """The bits one replay block of a two-stage stream's registration returns (`second_stage.replay_returned`, on
        the challenger's own `program`, its replay blocks' population); None for a one-stage stream's."""
        return None if self.replay is None else SS.replay_returned(program, descriptor_id(self.replay))

    def _append(self, reg: WindowCommitmentRecord) -> None:
        self._logged(dict(reg.record), reg.digest)

    def _logged(self, logged: dict[str, Any], digest: str) -> None:
        """Append `logged`, whose digest is `digest`, to the log, keeping what `_earlier` reads of it."""
        at = len(self._log)
        self._log.append(logged)
        key = _window_key(logged.get("window"))
        if key is None:
            self._irregular.append(at)
        else:
            self._windows.setdefault(key, []).append(at)
        for e in logged.get("registered") or ():
            self._first.setdefault((e["value"], e["domain"]), at)
        self._digests.setdefault(digest, at)

    def _earlier(self, record: Mapping[str, Any], supplied: Mapping[str, Any] | None = None,
                 outputs: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """The records of the log that one-stage's R6 and R7 read for `record`, in log order: the last (R6's `previous`),
        every one whose window may equal `record`'s (R6's window: the same `_window_key`, or none), each registered
        value's first registration under each domain (R7's held roots, `registration.held_roots`), and the first record
        of each digest its `outputs` name (R7's output values). Both verifiers give the same
        verdict on them as on the whole log. A record whose window has no key, or that has no window, gets the whole
        log, and so does one whose `supplied` values read a position, which follows every record of the log
        (`registered.log_base`)."""
        key = _window_key(record["window"]) if isinstance(record, Mapping) and "window" in record else None
        if key is None or _reads_positions(supplied):
            return self._log
        at = {*self._first.values(), *self._irregular, *self._windows.get(key, ())}
        named = (o.get("record") for o in (outputs or {}).values() if isinstance(o, Mapping)) \
            if isinstance(outputs, Mapping) else ()
        at |= {self._digests[d] for d in named if isinstance(d, str) and d in self._digests}
        if self._log:
            at.add(len(self._log) - 1)
        return [self._log[i] for i in sorted(at)]

    def register_interiors(self, draw: Draw, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], *,
                           leaf_layer: str | None = None, anchors: Sequence[Any] = (),
                           registered: Mapping[str, int] | None = None, supplied: Mapping[str, Any] | None = None,
                           pinned: Mapping[str, str] | None = None, schemas: Mapping[str, str] | None = None,
                           window_root: str | None = None, held: Sequence[str] | None = None,
                           outputs: Mapping[str, Mapping[str, str]] | None = None,
                           history: Mapping[str, Any] | None = None) -> WindowCommitmentRecord:
        """Receive a two-stage design's interior registration, once, after the first stage `draw` and before its second
        stage: the interiors of the replay sample's Q replay blocks, over their Q·n_rho proof blocks, template-major
        (`interior_program`): template t's Q·n_t from off_t = Q·Σ_{t'<t} n_t', replay block `draw.units[i]`'s instances
        of it [off_t + i·n_t, off_t + (i+1)·n_t). The public-items rule on the interior values under the check's
        query, its population (`interior-population`: Q·n_rho), its window (`interior-window`: `window_root` is the
        boundary registration's, or the root the verifier computes from the boundary registration's digest and the
        replay sample, `second_stage.interior_root`, so the second stage is sampled on one session with the first or
        on the session on that one root), then one-stage's R1-R8 against the challenger's own copies
        (`interior_program`, the check's query, `subset:Q·n_rho`, every one committed, `SCHEME`, the log, whose last
        record is the registration it follows, and the copies given here, as `register` takes them),
        `population-order`, and each history table `history` names being a port of the boundary registration's record
        at the root and domain it computed for it (`tables_served`: `interior-table`). Its receipt follows the log as a
        registration's does, and it is logged; it reads no clock.
        It names no links, wires or rows: the verifier derives them from the replay block's Definition
        (`second_stage.replay_links`, `second_stage.replay_wires`, `second_stage.replay_rows`).
        Raises `Refused`: `rule` (a one-stage stream), `unknown`, `not-drawn`, `final`, `delivered`,
        `interior-registered` (it has one), or either check failing."""
        first = draw.replay_sample or draw
        op: dict[str, Any] = {"op": "interiors", **self._drawn_op(first)}
        got: dict[str, Any] = {}
        seed_salt = self._seed_salt_of(window_root)

        def checks() -> list[str] | None:
            m = len(first.units) * self.n_rho
            v = self._public(record, interior=True)
            if v.ok and not (type(n := record.get("population")) is int and n == m):
                v = _refuse("interior-population", {"population": n, "replay_sample": len(first.units),
                                                    "n_rho": self.n_rho}).verdict
            boundary = self._entries[first.registration.order].copies[8]
            if v.ok and window_root != boundary and (window_root is None or window_root != SS.interior_root(
                    first.registration.digest, D.proved(first.draw))):
                v = _refuse("interior-window", {"window_root": window_root, "boundary": boundary}).verdict
            if v.ok:
                program, query = self.interior_program(len(first.units)), self.spec.check.query
                earlier = self._earlier(record, supplied, outputs)
                v = V.check(record, program, query, {"law": "subset", "k": m}, SCHEME, ports, leaf_layer, anchors,
                            earlier, None, None, registered, supplied, pinned, self.binding, schemas, window_root, held,
                            outputs=outputs, seed_salt=seed_salt, history=history)
                if v.ok and not P.in_order(program, query, m):
                    v = _refuse("population-order", {"population": m}).verdict
                tables = {} if history is None or not v.ok else HT.roots_of(leaf_layer, history)
                served = first.registration.record
                unserved = [n for n, rd in sorted(tables.items()) if not tables_served(served, [(n, *rd)])]
                if unserved:
                    v = _refuse("interior-table", {"tables": unserved}).verdict
                got.update(program=program, earlier=earlier)
            got["checks"] = v
            if not v.ok:
                return list(v.codes)
            got["registration"] = self._registration(record, first.registration.order, fanout=v.fanout)
            op["interior"] = _registration_token(got["registration"])
            return None
        self._held(self._decide(op, {"checks": checks}), got.get("checks"))
        reg, earlier = got["registration"], got["earlier"]
        if earlier is self._log:
            earlier = self._log[:]
        self._append(reg)
        e = self._entries[first.registration.order]
        e.interior, e.interior_program = reg, got["program"]
        e.interior_copies = (ports, leaf_layer, anchors, earlier, registered, supplied, pinned, schemas, window_root, held,
                             outputs)
        e.interior_seed_salt, e.interior_history = seed_salt, history
        return reg

    def interior_program(self, q: int) -> Program:
        """The challenger's interior program over a replay sample of `q`: `unit`'s Q·n_rho instances, or a window's mixed
        population of its templates at Q·n_t each, in the check's order, taking the stream's `values` (which its proof
        blocks read as registered values, an earlier record's among them: `register_interiors`' `outputs`), then its
        `span_values`, each at its leaves times Q."""
        assert self.n_rho is not None
        if self.templates is None:
            return P.population_program(self.unit, q * self.n_rho)
        spans = tuple((name, leaves * q, width) for name, leaves, width in self.span_values)
        return P.mixed_population_program([(t, q * n) for t, n in zip(self.templates, self.cells)],
                                          values=self.values + spans)

    def _drawn_op(self, draw: Draw) -> dict[str, Any]:
        """What an op names a sample by: its registration's order and token and its token; for a two-stage design's
        second stage, the first stage's, with `second`, its own token."""
        first = draw.replay_sample or draw
        op = {"order": _order(first.registration), "registration": _registration_token(first.registration),
              "draw": _draw_token(first)}
        if draw.replay_sample is not None:
            op["second"] = _draw_token(draw)
        return op

    def select(self, registration: WindowCommitmentRecord, *, closure: Mapping[int, Sequence[int]] | None = None) -> Draw:
        """The sample, once per registration, from the challenger's own randomness after the receipt: indices in [0, N) by
        the selection rule's sampling design (`select_draw`; `All` samples every proof block and reads no coins). A `closure`,
        the challenger's closure map, is refused before any coin is read (`closure`, as the Lean's `Run.run` refuses it): a
        work sample it widens proves a set whose size varies with the sample, which the sampled audit's soundness
        statement doesn't cover. The sample of record is the verifier's (`verifier.select`: the Lean's `Select.select` on the
        bytes of the coins this sample read), unless a `sampler` samples. Raises `Refused` for a registration sampled before
        (nothing resamples), a closure map (before any coin is read), the two verifiers sampling differently
        (`disagree`), and a sample that isn't well formed under the sampling design (`draw.check_draw`, by `verifier.check_draw`).

        Under a two-stage design, `select` of a registration is the first stage, and of its interior registration
        (`register_interiors`) the second (`select_second`, by `verifier.select_second`): `k` proof blocks in each
        sampled replay block, on the coins opened after the interior registration's digest, once (`drawn`), and only
        on the entry's own interior registration (`no-interior`)."""
        o = registration.order
        if self.n_rho is not None and _nat(o) and o < len(self._entries) and self._entries[o].interior is not None \
                and self._entries[o].interior == registration:
            return self._select_second(self._entries[o])
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
                if closure is not None:
                    raise _refuse("closure", "a closure map widens a work sample, which the sampled audit's soundness "
                                             "statement doesn't cover")
                obj = select_draw(self.spec.selection, e.law, registration.record["population"], coins, closure=closure,
                                  sampler=self._sampler)
                if self._sampler is None:
                    obj, v = V.select(obj, bytes(read), self.spec.selection.to_json(), e.program, self.query,
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

    def _select_second(self, e: _Entry) -> Draw:
        first, reg = e.draw, e.interior
        assert first is not None and reg is not None and isinstance(self.spec.selection, TwoStage)
        op: dict[str, Any] = {"op": "second", **self._drawn_op(first), "interior": _registration_token(reg)}
        ns, ks, got = self.cells, self.counts, {}

        def checks() -> list[str] | None:
            read = bytearray()

            def coins(n: int) -> bytes:
                b = self._coins(n)
                read.extend(b)
                return b
            law = second_law(first.units, ns, ks)
            try:
                obj = select_second(first.units, ns, ks, coins, sampler=self._sampler)
                if self._sampler is None:
                    obj, v = V.select_second(obj, bytes(read), first.units, ns, ks)
                    if not v.ok:
                        raise Refused(v)
                v = V.check_draw(obj, reg.record, law)
                if not v.ok:
                    raise Refused(v)
            except Refused as r:
                got["checks"] = r.verdict
                return list(r.verdict.codes)
            got.update(draw=Draw(reg, law, obj, first), read=None if self._sampler is not None else bytes(read))
            op["second"] = _draw_token(got["draw"])
            return None
        self._held(self._decide(op, {"checks": checks}), got.get("checks"))
        e.second, e.second_read = got["draw"], got["read"]
        return e.second

    def session(self, draws: Sequence[Draw]) -> Session:
        """Open one proof session over `draws`, the samples of 1 to `rounds` consecutive registrations in order, each
        round's registration its own tables of the session's statement: once, after the last covered sample. Its proof
        deadline is the first covered sample's (`Check.proof_deadline` from that sample), for every round it covers. Raises
        `Refused`, changing nothing: `rounds` (no sample, or more than `rounds`), `unknown`,
        `not-drawn` (a covered registration not sampled yet, or another sample), `final`, `delivered`, `covered` (a sample
        another session covers: nothing reopens it) or `order` (not consecutive registrations in order)."""
        draws = tuple(draws)
        self._held(self._decide({"op": "session", "draws": [_covered(d) for d in draws]}))
        s = Session(self.digest, draws)
        for d in draws:
            self._entries[d.registration.order].session = s
        return s

    def _deliver_session(self, session: Session, sessions: Sequence[str]) -> tuple[Outcome, ...] | None:
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

    def deliver(self, draw: Draw | Session, sessions: Sequence[str]) -> Outcome | tuple[Outcome, ...] | None:
        """The ZK server delivers the sample's proof, once: `sessions` are its session records' SHA-512, which the outcome
        binds. A proof after the proof deadline is a reject (`late`), returned at once; otherwise None, and the outcome waits
        on the verifier (`outcome`). A `Session`'s proof is delivered as the session, once, for every sample it
        covers (`covered`, delivering one of them alone): each binds the same `sessions`, and past the first covered
        sample's deadline each is `late`, returned in order. Raises `Refused` (`not-session`) for a session the stream
        didn't open. Under a two-stage design the proof is the second stage's sample's, and the deadline counts from the
        first stage: before it, the first stage's sample is refused (`second` before the second stage, `not-drawn`
        after)."""
        if isinstance(draw, Session):
            return self._deliver_session(draw, sessions)
        d = self._held(self._decide({"op": "deliver", **self._drawn_op(draw), "sessions": _session_digests(sessions)},
                                    {"now": lambda: float(self._clock())}))
        e = self._entries[draw.registration.order]
        e.sessions = list(sessions)
        return self._reported(self._finish(e, LATE, None, {})) if d["ok"] == LATE else None

    def outcome(self, draw: Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                proved_units: Sequence[int] | None = None, instances: int | None = None, served: Sequence[int] | None = None,
                session_roots: Mapping[str, str] | None = None, coins: Mapping[str, Any] | None = None,
                interior_coins: Mapping[str, Any] | None = None,
                statements: Sequence[Mapping[str, Any]] | None = None, proofs: Mapping[str, Any] | None = None) -> Outcome:
        """The sample's final outcome, once. With no proof delivered: a reject (`late`) once the deadline has passed, and
        `Pending` before. With one: `Pending` until every verdict of record has run (`accepted` not null); each verdict of
        record names the delivered `sessions` and reports the public outputs alike (`check_verdicts`, by the verifier,
        `verifier.check_verdicts`); then one-stage's audit record by the verifier
        (`verifier.audit_record`; `proved_units` and `instances` default to the sample's, `served` and `session_roots` are
        its, and the registered roots it reads are the verdicts' own) decides accept or reject. A sample a `Session` covers is decided alone, on the session's delivered records and its own
        audit record: its `proved_units`, `instances`, `served` and `session_roots` are its round's (its own tables'),
        never another round's. On a sample from the coins and an audit record its two verifiers agree on, the whole run is
        held to the verifier's (`verifier.run`, the Lean's `Run.run` from the
        registration on): where they differ the outcome is a reject, its audit record failing `run`. The verifier has
        no coins of its own, so the Lean's run refuses every sample that read coins unless `coins` is their evidence
        (`coins.coins_reason`'s `{key, ledger, main, pins, segment, session}`) and the bytes it read are exactly the
        coins the session the ledger opens for the
        registration's window (`register`'s `window_root`), from main's code (`challenger.pinned_session`), released
        after the registration's digest, each checked against its
        commitment and every one read; missing evidence, or a registration without its window root, is a reject failing
        `run` too, never a skip. Raises `Refused` for a second outcome, a verdict of record about
        other sessions, public outputs the verdicts of record don't report alike, or `coins` for a sample that read no
        coins.

        Under a two-stage design `draw` is the second stage's sample (or, once the proof is late, the first stage's):
        `proved_units`, `instances` and `served` are its, `session_roots` is held port by port to the registration
        whose port it names (the boundary values the sessions opened to the registration's audit record, when they
        opened any, and the rest to the interior registration's), `coins` is the first stage's evidence and
        `interior_coins` the second stage's. `statements` are the second-stage sessions' (`second_stage`), which round 2
        holds to the verdicts of record that checked them (each one's header, registered reads and sample, by the digests
        and sample a verdict names), the interior registration and the second stage's cells as the interior
        registration's audit record's `statements` check: a two-stage outcome without them fails it, and so does every
        one while `second_stage.FORM` refuses the two-stage form. `proofs` is the auditor's params naming each template's
        circuit file and the reference header and registered reads round 2 holds its statements to (`template_circuits`;
        without them every statement fails ``second-header`` and ``second-values``), which the run of record reads from
        its interior params. The outcome accepts
        when the registration's audit record (over the replay sample) and the interior registration's (over the sample)
        both do, and the run of record checks both. Raises `Refused` for `statements` on a one-stage outcome, and
        (`proofs`) for `proofs` on a one-stage stream or in any form `template_circuits` refuses, before anything is
        decided."""
        if statements is not None and self.n_rho is None:
            raise _refuse("statements", "second-stage statements are a two-stage outcome's")
        if proofs is not None:
            if self.n_rho is None:
                raise _refuse("proofs", "a one-stage run's params name no template circuit files")
            try:
                template_circuits(proofs, self.spec.check.query)
            except ValueError as e:
                raise _refuse("proofs", str(e)) from None
        op: dict[str, Any] = {"op": "outcome", **{k: v for k, v in self._drawn_op(draw).items() if k != "second"}}
        got: dict[str, Any] = {}
        o = op["order"]
        held_entry = self._entries[o] if o is not None and o < len(self._entries) else None
        for evidence, read in ((coins, held_entry and held_entry.read), (interior_coins, held_entry and held_entry.second_read)):
            if evidence is not None and held_entry is not None and read is None:
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
            if self.n_rho is not None:
                got.update(zip(("audit", "public", "replay"), self._audit_two(e, verdicts, statement, proved_units,
                                                                              instances, served, session_roots, coins,
                                                                              interior_coins, statements, proofs)))
                op["accepted"] = got["audit"]["accepted"] and got["replay"]["accepted"]
                return None
            got.update(zip(("audit", "public"), self._audit(e, draw, verdicts, statement, proved_units, instances, served,
                                                            session_roots, coins)))
            op["accepted"] = got["audit"]["accepted"]
            return None
        d = self._held(self._decide(op, {"now": lambda: float(self._clock()), "verdicts": held}), got.get("verdicts"))
        e = self._entries[draw.registration.order]
        if d["ok"] == LATE:
            return self._reported(self._finish(e, LATE, None, {}))
        return self._reported(self._finish(e, d["ok"], got["audit"], got["public"] if d["ok"] == ACCEPT else {},
                                           replay_audit=got.get("replay")))

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
            ports, leaf_layer, anchors, earlier, registered, supplied, pinned, schemas, window_root, held, given = e.copies
            v = V.run({"receipt": dict(reg.receipt), "draw": dict(draw.draw), "read": len(e.read),
                       "symbol": ACCEPT if accepted else REJECT, "public": public if accepted else {},
                       "audit": {k: x for k, x in rec.items() if k != "verifiers"}},
                      reg.record, e.program, self.spec.check.query,
                      selection_law(self.spec.selection, reg.record["population"],
                                    selection_sizes(self.spec.selection, e.program, self.spec.check.query)),
                      SCHEME, ports, leaf_layer, anchors,
                      earlier, e.work_table, registered, self.spec.selection.to_json(),
                      [(x.name, x.registered) for x in self.spec.values], [p.item for p in self.spec.public],
                      self.challenger, reg.receipt["received"], e.read, e.closure, e.sessions, verdicts, outputs,
                      statement, self.delta, *asked, served, session_roots, supplied=supplied, pinned=pinned,
                      binding=self.binding, schemas=schemas, window_root=window_root, held=held, coins=coins,
                      value_outputs=given, seed_salt=e.seed_salt, tied=self.spec.check.tied or None,
                      history=e.history)
            if not v.ok:
                rec = _run_failed(rec, v)
        return rec, public

    def _audit_two(self, e: _Entry, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any],
                   proved_units: Sequence[int] | None, instances: int | None, served: Sequence[int] | None,
                   session_roots: Mapping[str, str] | None, coins: Mapping[str, Any] | None,
                   interior_coins: Mapping[str, Any] | None, statements: Sequence[Mapping[str, Any]] | None,
                   proofs: Mapping[str, Any] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """`outcome`'s audit records under a two-stage design, once the verdicts hold: the interior registration's over
        the second stage's sample, with round 2's check of `statements` (`second_stage.check`, over the parts the
        interior program's strata make, the design's links and wires and their rows, which it derives from the replay
        block's Definition (`second_stage.replay_links`, `second_stage.replay_wires`, `second_stage.replay_rows`), the
        reads the verdicts report, what they
        name of each session they accepted
        (`second_stage.verified_of`), and the reference header and registered reads the params `proofs` give each
        template, `template_circuits`), and the registration's
        over the replay sample, each taking the sessions' reads of its own ports at their own roots (`own_reads`), both
        failing `run` where the whole run isn't the verifier's; and the public outputs."""
        assert e.sessions is not None and e.draw is not None and e.interior is not None and e.second is not None
        reg, first, ireg, second = e.registration, e.draw, e.interior, e.second
        outputs = self.spec.check.public_outputs
        of_record = sorted(k for k in verdicts if verdicts[k].get("of_record"))
        public = dict(verdicts[of_record[0]]["public"]) if outputs else {}
        boundary_ports = {r["port"] for r in reg.record["roots"]}
        replay_session_roots = None if session_roots is None else \
            {p: r for p, r in session_roots.items() if p in boundary_ports} or None
        if session_roots is not None:
            session_roots = {p: r for p, r in session_roots.items() if p not in boundary_ports}
        proved = list(proved_units) if proved_units is not None else D.proved(second.draw)
        replayed = D.proved(first.draw)
        rec = V.audit_record(ireg.record, ireg.receipt, second.draw, second.law, proved,
                             len(proved) if instances is None else instances, statement, verdicts, self.delta,
                             e.sessions, served, session_roots, own_reads=True)
        named = template_circuits(proofs, self.spec.check.query)
        own = SS.Own(tuple(replayed), SS.parts(len(replayed), P.strata(e.interior_program, self.spec.check.query)),
                     ireg.record, reg.record, tuple(D.proved(second.draw)),
                     SS.replay_links(e.program, descriptor_id(self.replay), reg.record),
                     A.session_registered(verdicts), SS.verified_of(verdicts),
                     tuple((t, h) for t, _c, h, _r in named), tuple((t, r) for t, _c, _h, r in named),
                     wires=SS.replay_wires(e.program, descriptor_id(self.replay)),
                     rows=SS.replay_rows(e.program, descriptor_id(self.replay)))
        rec = SS.attach(rec, SS.check(SS.FORM, own, statements), SS.interior_reads(own, statements or ()))
        boundary = V.audit_record(reg.record, reg.receipt, first.draw, first.law, replayed, len(replayed), statement,
                                  verdicts, self.delta, e.sessions, None, replay_session_roots, own_reads=True)
        accepted = rec["accepted"] and boundary["accepted"]
        if e.read is not None and e.second_read is not None and rec["verifiers"]["agree"] and \
                boundary["verifiers"]["agree"]:
            ports, leaf_layer, anchors, earlier, registered, supplied, pinned, schemas, window_root, held, given = e.copies
            within = {k: x for k, x in rec.items() if k != "verifiers"}
            v = V.run({"receipt": dict(reg.receipt), "draw": dict(first.draw), "read": len(e.read),
                       "symbol": ACCEPT if accepted else REJECT, "public": public if accepted else {},
                       "audit": {k: x for k, x in boundary.items() if k != "verifiers"},
                       "interior": {"receipt": dict(ireg.receipt), "draw": dict(second.draw),
                                    "read": len(e.second_read), "audit": within}},
                      reg.record, e.program, self.query,
                      selection_law(self.spec.selection, reg.record["population"],
                                    selection_sizes(self.spec.selection, e.program, self.query)),
                      SCHEME, ports, leaf_layer, anchors, earlier, e.work_table, registered,
                      self.spec.selection.to_json(), [(x.name, x.registered) for x in self._values()],
                      [p.item for p in self.spec.public], self.challenger, reg.receipt["received"], e.read, e.closure,
                      e.sessions, verdicts, outputs, statement, self.delta, None, None, None, replay_session_roots,
                      supplied=supplied, pinned=pinned, binding=self.binding, schemas=schemas, window_root=window_root,
                      held=held, coins=coins, value_outputs=given, seed_salt=e.seed_salt, history=e.history,
                      tied=self.spec.check.tied or None,
                      interior={"program": e.interior_program, "query": self.spec.check.query,
                                "law": {"law": "subset", "k": ireg.record["population"]}, "copies": e.interior_copies,
                                "seed_salt": e.interior_seed_salt, "history": e.interior_history,
                                "replay": Program(self.replay),
                                "values": [(x.name, x.registered) for x in self._values(interior=True)],
                                "record": ireg.record, "received": ireg.receipt["received"], "read": e.second_read,
                                "coins": interior_coins, "proved_units": proved_units, "instances": instances,
                                "served": served, "session_roots": session_roots, "statements": statements,
                                "proofs": proofs})
            if not v.ok:
                rec, boundary = _run_failed(rec, v), _run_failed(boundary, v)
        return rec, public, boundary

    def _finish(self, e: _Entry, symbol: str, audit: Mapping[str, Any] | None, public: Mapping[str, Any], *,
                halt: Mapping[str, Any] | None = None, replay_audit: Mapping[str, Any] | None = None) -> Outcome:
        reg, draw = e.registration, e.draw
        assert draw is not None or symbol == HALTED
        sample = draw if self.n_rho is None else e.second
        body = {"format": OUTCOME, "stream": self.digest, "order": reg.order, "registration": reg.digest,
                "receipt": _sha512(dict(reg.receipt)), "draw": None if sample is None else _sha512(dict(sample.draw)),
                "sessions": e.sessions or [],
                "symbol": symbol, "public": dict(public), "audit": None if audit is None else dict(audit)}
        if self.n_rho is not None:
            body["replay_sample"] = None if draw is None else {
                "draw": _sha512(dict(draw.draw)), "audit": None if replay_audit is None else dict(replay_audit)}
            body["interior"] = None if e.interior is None else {
                "registration": e.interior.digest, "receipt": _sha512(dict(e.interior.receipt))}
        if e.session is not None:
            body["session"] = {"rounds": [{"order": d.registration.order, "registration": d.registration.digest,
                                           "draw": _sha512(dict(d.draw))} for d in e.session.draws]}
        if halt is not None:
            body["halt"] = dict(halt)
        accepted = symbol == ACCEPT
        exhaustive = isinstance(self.spec.selection, All)
        profile = None
        if accepted and not exhaustive:
            profile = A.profile(reg.record, draw.draw, self.delta) if self.n_rho is None else self._footprint(e)
        e.outcome = Outcome(
            accepted=accepted,
            verdict=Verdict(True) if accepted else Verdict(False, (symbol,), {symbol: _why(symbol, audit, halt,
                                                                                         replay_audit)}),
            profile=profile,
            exhaustive=Exhaustive(len(draw.draw["units"])) if accepted and exhaustive else None,
            terms=Terms(None if exhaustive else self.delta),
            public=dict(public),
            record=_sha512(body))
        self._records.append(body)
        return e.outcome

    def _footprint(self, e: _Entry) -> IntegrityProfile:
        """An accepted two-stage outcome's integrity profile: two levels, the replay blocks (`replay`, the registration's
        partition) and their proof blocks (`A.LEVEL`, the interior registration's). The first stage's
        uniform Q-subset of the N replay blocks is bounded by `Bernoulli(Q/N)` (`Design.Two.Pa_le_bernoulli_pop`: it
        misses a set T with probability C(N − |T|, Q)/C(N, Q) ≤ (1 − Q/N)^|T|). The second stage samples k_t of each
        cell's n_t, and a replay block of T templates misses m incorrect proof blocks with probability at most
        `Subset(K)`'s over T·n_max at K = min_t k_t: ∏_t C(n_t − w_t, k_t)/C(n_t, k_t) ≤ C(T·n_max − m, K)/C(T·n_max, K)
        for Σ_t w_t = m, factor by factor, since ∏_t (1 − w_t/(n_t − j)) ≤ 1 − max_t w_t/(n_max − j)
        ≤ 1 − m/(T·n_max − j). With one template that is `Subset(k)` over n_rho, exactly. Its `accept(m)` is a replay
        block's at footprint m, the fewest incorrect proof blocks of any interior consistent with its boundary
        commitment, which is fixed before any coin; the prover's choice of interior after the first stage only raises
        it."""
        assert e.draw is not None and e.interior is not None and e.second is not None
        reg, ireg = e.registration.record, e.interior.record
        assert self.n_rho is not None
        q, n = len(e.draw.units), reg["population"]
        return IntegrityProfile(levels=(Level(REPLAY, reg["partition"]), Level(A.LEVEL, ireg["partition"])),
                                sizes=(n, len(self.cells) * max(self.cells)),
                                draws={REPLAY: Bernoulli(Fraction(q, n)), A.LEVEL: Subset(min(self.counts))},
                                delta=self.delta, programs=(reg["program"], ireg["program"]),
                                drawn={REPLAY: q, A.LEVEL: len(e.second.units)})


def _run_failed(rec: Mapping[str, Any], v: Verdict) -> dict[str, Any]:
    """An audit record failing `run`: the whole run is not the verifier's."""
    return {**rec, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None,
            "checks": {**rec["checks"], "run": {"ok": False, "codes": list(v.codes), "detail": v.detail}}}


def _why(symbol: str, audit: Mapping[str, Any] | None,
         halt: Mapping[str, Any] | None = None, replay_audit: Mapping[str, Any] | None = None) -> Any:
    if symbol == HALTED:
        return dict(halt or {})
    if symbol == LATE:
        return "the proof was not delivered by the proof deadline"
    assert audit is not None
    why = {"checks": [k for k, c in audit["checks"].items() if not c["ok"]],
           "verdicts": {k: v.get("accepted") for k, v in audit["verdicts"].items()}}
    if replay_audit is not None:
        why["replay_sample"] = [k for k, c in replay_audit["checks"].items() if not c["ok"]]
    return why


__all__ = ["ACCEPT", "All", "Check", "Draw", "Exhaustive", "Fleet", "HALTED", "HIDING", "KINDS", "LATE", "Law", "Outcome",
           "PHASES", "Pending", "Public", "REJECT", "REPLAY", "ROOTS", "ROUNDS", "Refused", "SCHEME", "SOURCES",
           "Selection", "Sequential", "Session", "Spec", "Stream", "Terms", "TwoStage", "Value",
           "WindowCommitmentRecord", "boundary_values", "check_public", "check_verdicts", "interior_values",
           "replay_cells", "second_counts", "second_law", "select_draw", "select_second", "selection_law"]
