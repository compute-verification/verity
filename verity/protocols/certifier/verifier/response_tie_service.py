"""The response tie's verifier: one class of one execution phase's content rows (`content_rows`), whose response
rows a population of the response tie's proof blocks returns, held to the network certifier's signed head
(`connection_rows`' "The signed head"). It reads the head, builds the tie's Program at the count of rows the head
states for the class, gives the sampled-proofs verifier the entries the head signs, and reads one-stage's answer.

**The tie's record** is one-stage's registration record of a population of n proof blocks of the tie's Definition
(`ResponseTie.unit`: one of the response tie's two forms at the class's O ids), n the rows the head counts in the
class, proof block j the class's j-th release. Its Program's parameters are the proof block's:

- ``end``, the network certifier's end row: proof block j reads row j of the class's `end` table, a registered port
  the verifier builds from the head (`ResponseTie.reads`);
- every other parameter (the compute side's committed ids, and in a stop class its stop word), a registered value
  that is a port of an earlier record of the tie's own log (`registration.check`'s `outputs`): the compute side's
  records the tie reads, which its stream's log starts with.

Its second output, ``o1``, is committed as row j of the class's `response` table through the other registered port
(`ResponseTie.reads`), so a proof block's proof opens the signed root only where the row its gates return is the
row the network certifier signed. Its first output, ``o0``, the proof blocks' verdict words, is the record's one
port (`PORT`), under a domain that binds the Program, the count and the head's digest (`ResponseTie.ports`). The
record registers the class's two tables as certified values, under the names the head's registration gives them
(`connection_rows.value_name`), beside the compute side's.

**What the verifier holds** (`ResponseTie.check`), in order:

- the head's registration reads under the network certifier's key in the session, as the execution phase's of the
  link under the schedule (`connection_rows.check_registration`; `HEAD`);
- the class's entry counts a row (`NO_ROWS`): for a class in which the execution phase filed no release no tie is
  owed, and none is accepted;
- the record's population is the entry's rows (`POPULATION`). A stream builds its population program from the
  record's own population, so nothing else compares the two where a record is registered on one (`register`);
- one-stage's R1 to R8 against the verifier's own copies (`verifier.check`): the Program at the count, the
  sampling design the tie's rate gives over it, the port's leaves and domain, the compute side's values as output
  values of the log's records, and the certified entries built from the head, each table's root, rows, words and
  domain (`ResponseTie.certified`). R7 holds the record's entry of each to the head's: another root is refused
  `R7-certified`, another domain `R7-domain`, and this module compares neither again.

**The verdict** (`ResponseTie.verdict`, `TieVerdict`) carries two bits and is never one: `accepted`, the record
passes the checks above and one-stage's answer is an accept of that record (its `commitment` the record's digest);
and `complete`, the accept's public ``complete`` is true. An answer without the field is accepted and not
complete. The answer is one-stage's own (`Core.Protocol.answer`'s form): this module reads it and runs no audit.

**What it leaves.** To the period check: that a tie's verdict exists for every class the head counts a row in,
that the records the compute side's values name are that class's records of the same execution phase, that the
answer is its own run's, and whether an accept that is not complete counts. To the compute audit: the response
tie's open premise, that the committed ids below the count are the Program's outputs. To the caller: the tie's
Definition, which the verifier takes as the registered one and holds only to the class's interface, since the
Definitions are the prover's (`verity.protocols.certifier.prover.response_tie`); the Spec's check names its digest.
A schedule under the lengths rows' form 1 has no tie here: its head states a class's size, not a count.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

from verity.circuits.codec import canonical_json
from verity.circuits.defs import SpecializedDefinition
from verity.circuits.program import Program
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity.protocols.certifier.verifier import connection_rows as CR, content_rows as CT
from verity.protocols.certifier.verifier.certificate import PublicKey
from verity.protocols.certifier.verifier.warden import WardenParams
from verity.protocols.certifier.verifier.warden_rows import WORD_BITS
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P, registration as R, verifier as V
from verity.protocols.spot.verifier.one_stage.verdict import Verdict

SPEC = "verity-network-certifier/response-tie-spec/v1"
VERDICT = "verity-network-certifier/response-tie-verdict/v1"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
#: the record's one port: the proof blocks' first output, their verdict words
PORT = "o0"
#: the proof block's ports that read the network certifier's tables, each with the class's table it reads
READS = {"end": CT.END, "o1": CT.RESPONSE}
#: the proof block's parameter among `READS`: the others are its outputs
END = "end"
#: this verifier's own refusals: the head's registration doesn't read; the class's entry counts no row; the record's
#: population is not the entry's rows; one-stage's answer is no accept of the record
HEAD, NO_ROWS, POPULATION, ANSWER = "head", "no-rows", "population", "answer"


def _refused(code: str, detail: Any) -> S.Refused:
    return S.Refused(Verdict(False, (code,), {code: detail}))


@dataclass(frozen=True)
class TieVerdict:
    """The response tie's verdict on one class of one execution phase of a link: the head's digest (None where its
    registration didn't read) and the record's (None where the record was refused), `accepted`, `complete`, which
    only an accepted one is, and the refusal's codes, none where it is accepted."""

    link: str
    window: int
    cls: int
    head: str | None
    registration: str | None
    accepted: bool
    complete: bool
    codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool or type(self.complete) is not bool:
            raise ValueError("a tie's verdict says accepted and complete, a bit each")
        if self.complete and not self.accepted:
            raise ValueError("only an accepted tie is complete")
        if self.accepted == bool(self.codes) or (self.accepted and None in (self.head, self.registration)):
            raise ValueError("an accepted tie names its head and its record and no refusal, and a refused one says why")

    def to_json(self) -> dict[str, Any]:
        return {"format": VERDICT, "link": self.link, "window": self.window, "class": self.cls, "head": self.head,
                "registration": self.registration, "accepted": self.accepted, "complete": self.complete,
                "codes": list(self.codes)}


@dataclass(frozen=True)
class ResponseTie:
    """The response tie on one class of one execution phase of a link, every field public: the network certifier's
    key and session, the schedule's parameters, the link, the execution phase (`window`), the class's place in the
    schedule's list (`cls`), the caller's copy of the tie's registered Definition (`unit`), the rate its proof
    blocks are sampled at (`rate`, `rate:NUM/DEN`: a tie's selection is a rate, never `All`), and the proof
    deadline in seconds."""

    key: PublicKey
    session: bytes
    params: WardenParams
    link: str
    window: int
    cls: int
    unit: SpecializedDefinition
    rate: str
    proof_deadline: float | None = None

    def __post_init__(self) -> None:
        if type(self.params) is not WardenParams or not (isinstance(self.link, str) and self.link) or \
                type(self.session) is not bytes or type(self.window) is not int or self.window < 0:
            raise ValueError("a response tie is on a named link's execution phase, of a warden.WardenParams and a session")
        if self.params.content is None or not CT.counted(self.params):
            raise ValueError("a response tie's population is the head's count of its class's rows, which a schedule "
                             "states under the lengths rows' form 2 (lengths is 2, with content classes)")
        if type(self.cls) is not int or not 0 <= self.cls < len(self.params.content):
            raise ValueError(f"the schedule lists {len(self.params.content)} classes, from 0, not a class {self.cls!r}")
        if D.parse_rate(self.rate) is None:
            raise ValueError(f"a tie's proof blocks are sampled at a rate, rate:NUM/DEN, not {self.rate!r}")
        if not isinstance(self.unit, SpecializedDefinition):
            raise ValueError("a response tie's unit is its Definition, bound at the class's ids")
        params, ret = self.unit.signature()
        bits = {name: t.leaves for name, t in params}
        fields = tuple(getattr(ret, "fields", ()))
        rows = (WORD_BITS * CT.TIE_WORDS, 8 * CT.response_bytes(self.ids))
        if any(t.width_set() != {1} for t in (*dict(params).values(), *fields)) or bits.get(END) != 8 * CT.END_BYTES \
                or tuple(f.leaves for f in fields) != rows or len(bits) < 2 or any(b % WORD_BITS for b in bits.values()):
            raise ValueError(f"{self.unit.id} is no response tie's proof block of class {self.cls}: one reads an end "
                             f"row of {8 * CT.END_BYTES} bits ({END}) and the compute side's rows of whole words, and "
                             f"returns a verdict word and the class's response row, {rows[1]} bits at {self.ids} ids")

    @property
    def ids(self) -> int:
        """O, the ids a response row of the class holds."""
        return self.params.content[self.cls][1]

    def table(self, kind: str) -> str:
        """The class's table of `kind` (`content_rows.END`, `content_rows.RESPONSE`) as the head names it: by its
        name among the class's tables, whatever others the class has."""
        name = f"{kind}.{self.cls}"
        if name not in CT.names(self.cls):
            raise ValueError(f"class {self.cls} has the tables {CT.names(self.cls)}, not a table {name!r}")
        return name

    @cached_property
    def registered(self) -> dict[str, int]:
        """The compute side's values, every parameter of the proof block but its end row: each one's leaves a row,
        of `WORD_BITS` bits."""
        return {name: t.leaves // WORD_BITS for name, t in self.unit.signature()[0] if name != END}

    @property
    def schemas(self) -> dict[str, str]:
        return dict.fromkeys(self.registered, SCHEMA)

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instance_query(self.unit)

    def program(self, rows: int) -> Program:
        """The tie's Program at a population of `rows` proof blocks."""
        return P.population_program(self.unit, rows)

    def law(self, rows: int) -> dict[str, Any]:
        """The sampling design the tie's rate gives over `rows` proof blocks (`service.selection_law`); raises
        `service.Refused` (``rate``) where it samples none of them."""
        return S.selection_law(S.Law(self.rate), rows)

    @cached_property
    def values(self) -> dict[str, str]:
        """The certified value each of the proof block's `READS` ports reads: the class's table's registered value's
        name in the execution phase's registration."""
        return {port: CR.value_name(self.link, self.table(kind), self.window) for port, kind in READS.items()}

    @cached_property
    def spec(self) -> S.Spec:
        """The Spec of the tie's stream, one stream a tied class an execution phase: the port, the compute side's
        values and the certified ones, sampled at the tie's rate."""
        named = (*self.registered, *self.values.values())
        context = canonical_json({"format": SPEC, "link": self.link, "window": self.window, "class": self.cls,
                                  "session": self.session.hex(), "params": self.params.digest.hex()})
        check = S.Check(P.program_digest(Program(self.unit)), self.query, proof_deadline=self.proof_deadline)
        public = (
            S.Public("population", "structure", "the rows the signed head counts in the class, a proof block each"),
            S.Public("window", "structure", "provenance: the tie's own window of its stream's log"),
            S.Public(f"leaves:{PORT}", "structure", "one verdict word a proof block"),
            S.Public(f"domain:{PORT}", "structure", "the port's domain, which binds the Program, the count and the "
                     "signed head"),
            S.Public(PORT, "commitment", "the salted root over the proof blocks' verdict words"),
            *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}: a registered value's layout, the earlier record's or "
                       "the signed head's") for v in named for k in ("leaves", "words", "domain")),
            *(S.Public(v, "commitment", f"{v}'s salted root, an earlier record's port's or the signed head's entry's")
              for v in named))
        values = (S.Value(PORT, SCHEMA, "serve"), *(S.Value(v, None, "serve", registered=True) for v in named))
        return S.Spec(values, S.Law(self.rate), check, public, context)

    def stream(self, log: Sequence[Mapping[str, Any]], **kwargs: Any) -> S.Stream:
        """The tie's stream, whose log starts with `log`: the compute side's records the tie reads, in their order."""
        return S.Stream(self.spec, self.unit, log=log, **kwargs)

    # the head

    def _head(self, doc: Mapping[str, Any]) -> CR.Head:
        if not isinstance(doc, Mapping):
            raise _refused(HEAD, f"a registration is a JSON object, not {type(doc).__name__}")
        try:
            return CR.check_registration(doc, self.key, self.session, self.params, self.link, self.window)[0]
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise _refused(HEAD, str(e)) from None

    def _owed(self, head: CR.Head) -> None:
        if not self.rows(head):
            raise _refused(NO_ROWS, f"the head counts no row in class {self.cls} ({self.table(CT.END)}): the execution "
                                    f"phase filed no release in it, so no tie is owed for it and none is accepted")

    def _population(self, record: Any, head: CR.Head) -> None:
        n = record.get("population") if isinstance(record, Mapping) else None
        if type(n) is not int or n != self.rows(head):
            raise _refused(POPULATION, {"record": n, "head": self.rows(head), "table": self.table(CT.END)})

    def read(self, doc: Mapping[str, Any]) -> CR.Head:
        """The head of the execution phase's registration `doc`, read under the network certifier's key
        (`connection_rows.check_registration`). Raises `service.Refused`: `HEAD` where the registration doesn't
        read, with what fails; `NO_ROWS` where it counts no row in the class."""
        head = self._head(doc)
        self._owed(head)
        return head

    def rows(self, head: CR.Head) -> int:
        """The rows `head` counts in the class: the tie's population."""
        return head.table(self.table(CT.END)).rows

    def ports(self, head: CR.Head) -> dict[str, dict[str, Any]]:
        """The record's port's leaves and domain, as `registration.check` takes them: one verdict word a proof
        block, under the window domain of the Program at the count that names `head`'s digest as its run."""
        n = self.rows(head)
        part = P.partition(self.program(n), self.query)
        return {PORT: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), PORT, SCHEMA, n,
                                                              head.digest().hex())}}

    def certified(self, head: CR.Head) -> dict[str, dict[str, Any]]:
        """The certified entries one-stage holds the record to (`registration.check`'s `certified`): the class's end
        and response tables as `head` signs them, each its root, rows, a row's words and domain."""
        out = {}
        for kind in READS.values():
            e = head.table(self.table(kind))
            out[CR.value_name(head.link, e.name, head.window)] = {
                "root": e.root.hex(), "leaves": e.rows, "words": e.row_bytes * 8 // WORD_BITS, "domain": e.domain.hex()}
        return out

    def reads(self, head: CR.Head, reads_from: int = 0) -> dict[str, dict[str, Any]]:
        """The verifier's own ``--registered`` ports of the proof block's `READS`, by circuit port
        (`connection_rows.port`): proof block `reads_from + j` reads row j of the class's end table and commits its
        response row as row j of the class's response table, each under `head`'s root."""
        return {port: CR.port(head, self.table(kind), reads_from) for port, kind in READS.items()}

    # the record

    def held(self, record: Any, doc: Mapping[str, Any]) -> CR.Head:
        """`read`, then the record's population held to the rows the head counts in the class. Raises
        `service.Refused`: `read`'s, or `POPULATION`."""
        head = self.read(doc)
        self._population(record, head)
        return head

    def _outputs(self, outputs: Any) -> dict[str, dict[str, str]]:
        if not isinstance(outputs, Mapping) or set(outputs) != set(self.registered) or \
                not all(isinstance(o, Mapping) and set(o) == {"record", "port"} for o in outputs.values()):
            raise ValueError(f"the compute side's values are {sorted(self.registered)}, each a port of a record of "
                             f"the tie's log ({{\"record\": digest, \"port\": name}}), not {outputs!r}")
        return {v: dict(o) for v, o in outputs.items()}

    def _checked(self, record: Any, doc: Mapping[str, Any], outputs: Any, log: Sequence[Mapping[str, Any]],
                 binding: Mapping[str, str] | None) -> tuple[CR.Head | None, Verdict]:
        outputs, head = self._outputs(outputs), None
        try:
            head = self._head(doc)
            self._owed(head)
            self._population(record, head)
            n = self.rows(head)
            law = self.law(n)
        except S.Refused as r:
            return head, r.verdict
        return head, V.check(record, self.program(n), self.query, law, S.SCHEME, self.ports(head), None, (), log,
                             registered=self.registered, binding=binding, schemas=self.schemas, outputs=outputs,
                             certified=self.certified(head))

    def check(self, record: Any, doc: Mapping[str, Any], *, outputs: Mapping[str, Mapping[str, str]],
              log: Sequence[Mapping[str, Any]], binding: Mapping[str, str] | None = None) -> Verdict:
        """The tie's record `record` held to the execution phase's registration `doc` (the module docstring's
        "What the verifier holds"): this verifier's refusal, or one-stage's R1 to R8 against its own copies.
        `outputs` names, for each of the compute side's values, the record of the tie's log and its port that the
        value is (`registration.check`'s `outputs`); `log` is the tie's stream's log before the record, the compute
        side's records; `binding` the stream and prover the record is registered under, where the stream names a
        prover. ValueError where `outputs` aren't the compute side's values."""
        return self._checked(record, doc, outputs, log, binding)[1]

    def register(self, stream: S.Stream, record: Mapping[str, Any], doc: Mapping[str, Any], *,
                 outputs: Mapping[str, Mapping[str, str]]) -> S.WindowCommitmentRecord:
        """`stream.register` of the tie's record on the tie's stream (`stream`), after `held`, with the verifier's
        own port, the compute side's values, and the certified entries built from `doc`'s head. Raises
        `service.Refused` with this verifier's refusal or the stream's; ValueError for another Spec's stream."""
        if stream.spec.digest() != self.spec.digest():
            raise ValueError("the stream is not this tie's: its Spec is another's")
        outputs, head = self._outputs(outputs), self.held(record, doc)
        return stream.register(record, self.ports(head), registered=self.registered, schemas=self.schemas,
                               outputs=outputs, certified=self.certified(head))

    # the verdict

    def verdict(self, record: Any, doc: Mapping[str, Any], answer: Any, *,
                outputs: Mapping[str, Mapping[str, str]], log: Sequence[Mapping[str, Any]],
                binding: Mapping[str, str] | None = None) -> TieVerdict:
        """The tie's verdict (the module docstring's "The verdict") on `record` under `doc` and one-stage's answer
        `answer` of the record's audit: refused with `check`'s codes; refused `ANSWER` where the record passes and
        the answer is not one-stage's accept of it; else accepted, and complete where the accept says so."""
        head, checked = self._checked(record, doc, outputs, log, binding)
        registration = R.digest(record) if checked.ok else None
        accepted = checked.ok and accepts(answer, registration)
        codes = checked.codes if not checked.ok else () if accepted else (ANSWER,)
        return TieVerdict(self.link, self.window, self.cls, None if head is None else head.digest().hex(),
                          registration, accepted, accepted and complete(answer), codes)


def accepts(answer: Any, registration: str | None) -> bool:
    """Whether `answer` is one-stage's accept of the record whose digest is `registration`
    (`Core.Protocol.answer`'s form: its protocol one-stage's, its verdict ``accept``, its commitment the record's)."""
    return isinstance(answer, Mapping) and registration is not None and answer.get("protocol") == R.PROTOCOL and \
        answer.get("verdict") == "accept" and answer.get("commitment") == registration


def complete(answer: Any) -> bool:
    """Whether one-stage's accept `answer` says every verdict of its audit ran: its public ``complete`` is true. An
    answer without the field, or with anything else there, is not complete."""
    public = answer.get("public") if isinstance(answer, Mapping) else None
    return isinstance(public, Mapping) and public.get("complete") is True


__all__ = ["ANSWER", "END", "HEAD", "NO_ROWS", "POPULATION", "PORT", "READS", "ResponseTie", "SCHEMA", "SPEC",
           "TieVerdict", "VERDICT", "accepts", "complete"]
