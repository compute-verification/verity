"""The ingress tie through the proof system: each bucket row of an ingress link-window of the warden's cells proved as
`io_request.IoTieRequest`, and the verifier of record that builds `advised.IoTieVerdict`, the tie of an advised link's
requests to its certified ingress, only from IoTieRequest proofs it verified on lanes it computed itself (PROTOCOL.md,
"The ingress tie").

**What is committed** is the warden's own (`warden_rows`): an ingress link-window's T bucket rows, row t bucket t's
S = L·R slots of B bytes (a slot no connection holds B zero bytes), each an `hm96-sha512/row/v2` row under the
warden's fresh salt in its `frame-v3-sha512` tree under the warden's domain (`warden_rows.domain`: the session, the
link, ingress and the window), whose root the warden's signed head names (`warden_rows.CellsHead`). The tie registers
that root with the challenger (`IngressTie.registration`, on the head `warden_rows.check_registration` reads off the
warden's signed registration); the rows and their salts are the prover's, which the warden hands it privately, and
the challenger and the verifier see the root. A registration is one link-window, its `window` provenance
`{"kind": "served", "window": w}` held to the registration's order (`IngressTie.register`), and T proof blocks, unit t
bucket row t. Selection is `All`.

**The check** is `IngressTie.unit`, `NetworkIngressTieBucket` v1{B, L, R}: one gate naming
`io_request.IoTieRequest_v1{B, L, R}` and its circuit digest, pinned in `io_request.circuit-check.json` and held to the
Definition by circuit-check (`gate`, as `head_check.gate` holds the head check), with one output bit, IoTieRequest's.
Its witness row is the bucket row itself, which the proof system's registered read holds to the warden's root
(`warden_rows.bucket_port`: unit t reads row t, the verifier's own `--registered` port, `IngressTie.registered`), so
no gate hashes it. A size class no sidecar pins has a gate that names no digest, and no proof of its unit is accepted
(`proved`).

**The statement is the verifier's own** (`io_request.lanes`): unit t's public row is bucket t's lanes, from the
advice's v0 prompt lengths and the warden's ingress release (`io_request.Release`: each connection's first bucket and
its request cells a bucket), which the verifier takes in the clear as it takes the ingress record's wiring (the named
gap), never from the prover. The proof's ``pub`` row is a unit's lanes, its ``wit`` row the unit's bucket row under the
warden's salt, and its ``o0`` row one u16 word holding the output bit; the ``pub`` and ``o0`` rows are each committed
under a hiding salt (`IngressTie.layout`).

**The verifier of record** is `IngressTie.verify(…, advice=A, releases=…, head=…)`, which decides on the delivered
proof and public inputs alone. It accepts the window when the verifier's own run of Flock's Lean verifier, with its
own registered reads at the warden's root (`registered`), accepts every session, they prove this tie's registered unit
(`proved`: its gate names the pinned digest, every session the registered Program) and every drawn unit, and each
unit's ``pub`` row opens, under the salt delivered, to the verifier's own lanes and its ``o0`` row to 1. Its public
outputs are `verdict` and `advice`, A's `advised.digest`. A proof on another advice's lanes or another release opens
to another ``pub`` row and is refused, and a row that isn't the warden's fails the registered read.

**The tie** (`IngressTie.tie`) is `advised.IoTieVerdict(link, registrations, advised.digest(A), accepted)`: the link,
the stream's registrations (each one's `registration.digest`, in order), A's digest and one bit a registration, read
from the stream's outcomes of record alone. Window w is accepted when the stream's w-th registration is window w's and
its outcome of record accepted, decided on the proof, naming A's digest, and when the release is the warden's schedule
for A (`io_request.lanes` takes it on every bucket: ordinals admitted in order, each connection one advice request,
its cells its request's) and gives A's `received`, each window's request cells. Every window is refused when they
don't, when the stream holds a registration past A's windows or misses one, and when the advised profile's framing
isn't the warden's cut of A's prompt lengths (`framing`) or names an own frame, which the warden's ingress has none
of. A connection's lanes chain by construction from (0, 0) into its first bucket to (0, n) out of its `END`'s, so an
accepted verdict on every window says what IoTieRequest's units say together: connection i + 1 is an id request of
v0's `prompt_len` prompt fields (a carried field count is held by the bucket before). That is
the premise `advised.ends` reads, from the verifier's own tie source alone (`Ties`, the tie, its stream and the
release as `service.LinkWindows.ties`), never from what the prover delivers, and `advised.view`
holds the verdict to the link and the registrations the source records, so a verdict on another link or another
ingress record isn't reused.

**What it doesn't establish.** The prompt ids' values (the ingress-ids tie, `io_ids`). The release is taken in the
clear (the named gap). That the warden's cells are the frames the ingress certifier counted is the device's: both are
the auditor's device's record of the link, and the advised unit holds `received` to the certified ingress head.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity.core.protocols.certifier import advised as AD, head_check as HC, io_request as Q
from verity.core.protocols.certifier import service as SVC, warden_rows as WR
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, proofs as PF, registration as R
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict

SPEC = "verity-network-certifier/ingress-tie-spec/v1"
STATEMENT = "verity-network-certifier/ingress-tie-statement/v1"
VERDICT = "verity-network-certifier/ingress-tie-verdict/v1"
VALUE = "cells"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "ingress-tie"
OUTPUTS = ("verdict", "advice")
ROWS, PORTS, OUT_BITS = SVC.ROWS, SVC.PORTS, SVC.OUT_BITS
#: the unit's witness parameter: the bucket row, which the registered read holds to the warden's root
WITNESS = ("row",)
#: the circuit port the warden's rows are read on: each unit's witness row is its bucket row
REGISTERED = "wit"
_STATICS = ("B", "L", "R")

PUBLIC = (
    S.Public("population", "structure", "T proof units per registration, unit t the link-window's bucket row t"),
    S.Public("window", "structure", "provenance: the ingress link-window's index, held to the registration's order"),
    S.Public(f"leaves:{VALUE}", "structure", "T leaves, one bucket row of L·R cells each"),
    S.Public(f"domain:{VALUE}", "structure", "the warden's domain (`warden_rows.domain`), which binds the session, the "
             "link, ingress and the window"),
    S.Public(VALUE, "commitment", "the warden's salted root over the window's bucket rows, its signed head's"),
    S.Public("shape", "structure", "B, L, R and T: the cells' bytes, the warden's lanes, its r and its buckets a "
             "window, fixed before the run"),
    S.Public("verdict", "output", "the window's IoTieRequest verdict, every bucket row's, accept or reject"),
    S.Public("advice", "output", "the advice the window's statement is computed from (`advised.digest`): its v0 "
             "prompt lengths and the warden's release give each bucket's lanes (`io_request.lanes`), a unit's public "
             "row in the named gap; `IngressTie.tie` holds every window to one advice"),
)


@lru_cache(maxsize=None)
def gate(B: int, L: int, R: int) -> PrimitiveDefinition:
    """`IoTieRequest_v1{B, L, R}` as one gate that names it and its pinned circuit digest (`io_request.PINS`),
    `Gate[IoTieRequest_v1{…}@<digest>]_v2`, as `head_check.gate` holds the head check."""
    d = bind(Q.IoTieRequest, B=B, L=L, R=R)
    return HC.gate_of(d, HC.pinned_digest(d, Q.PINS), "the ingress tie")


@lru_cache(maxsize=None)
def _expanded(B: int, L: int, R: int):
    return HC.expand(gate(B, L, R), bind(Q.IoTieRequest, B=B, L=L, R=R))


def _unit(check) -> CompositeDefinition:
    def sig(S_):
        return gate(S_.B, S_.L, S_.R).signature()

    def body(B, S_, *args):
        return B.call(check(S_.B, S_.L, S_.R), *args)

    return CompositeDefinition(
        "NetworkIngressTieBucket", 1, _STATICS, sig, body, register=False,
        doc="the proof block of the ingress tie: one bucket row of the warden's ingress link-window, its "
            "`io_request.IoTieRequest` on the row the registered read holds to the warden's root and the verifier's "
            "lanes")


_UNIT = _unit(gate)
#: the same unit with the gate replaced by the Definition it names, as a proof's circuit lowers it
_PROOF_UNIT = _unit(_expanded)


def request_cells(prompt_len: int, B: int) -> int:
    """The warden's request cells of an id request of a `prompt_len`-token prompt (`warden.cells`' cut, which
    `io_request.carries` reads)."""
    return len(Q.carries(prompt_len, B)) - 1


def framing(B: int, n_max: int) -> tuple[int, ...]:
    """The link's framing as the warden cuts requests, for `advised.AdvisedProfile.request_frames`: entry n − 1 is the
    request cells an id request of an n-token prompt takes at B (`request_cells`), n = 1 … n_max."""
    return tuple(request_cells(n, B) for n in range(1, n_max + 1))


def prompt_lens(advice: AD.LinkAdvice) -> tuple[int, ...]:
    """The advice's v0 prompt lengths, in its requests' order."""
    return tuple(q.prompt_len for q in advice.advice.requests)


@dataclass(frozen=True)
class Delivered:
    """An ingress link-window's proof as the verifier receives it: the statement's proof files (`proofs.Proof`, its
    circuit file the auditor's own lowering of the unit), and for each drawn unit, by its index, the salts of its public
    row and its output row (``pub``, ``o0``) and its output word, which the verifier opens against the public file's
    commitments to those rows. Never a witness row's salt: the warden's, which the registered read holds."""

    proof: PF.Proof
    salts: Mapping[int, Mapping[str, bytes]]
    outputs: Mapping[int, int]

    def sessions(self) -> list[str]:
        """The delivered sessions' SHA-512s, their `session.json`s', in order."""
        return [PF.file_sha512(Path(s) / "session.json") for s in self.proof.sessions]


def _opened(delivered: Delivered, unit: int, public: bytes | None, bits: int) -> int | None:
    """Unit `unit`'s output bit, once the proof's public file holds its public row as `public` (the verifier's own
    copy, `bits` long) and its output row as the word delivered, each under the salt delivered; else None."""
    bcs, word = SVC._commitments(delivered.proof.public, unit), delivered.outputs.get(unit)
    salts = delivered.salts.get(unit) or {}
    if public is None or bcs is None or type(word) is not int or word not in (0, 1):
        return None
    if not (SVC._opens(public, bits, salts.get("pub"), bcs["pub"])
            and SVC._opens(SVC.output_row(word), OUT_BITS, salts.get("o0"), bcs["o0"])):
        return None
    return word


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def window_cells(releases: Sequence[Q.Release], T: int) -> dict[int, int]:
    """Each window's request cells under the release, buckets of T, where there are any."""
    out: dict[int, int] = {}
    for rel in releases:
        for k, c in enumerate(rel.cells):
            if c:
                out[(rel.first + k) // T] = out.get((rel.first + k) // T, 0) + c
    return out


@dataclass(frozen=True)
class IngressTie:
    """The ingress tie on one link, every field public: the link, the warden's schedule (`warden.WardenParams`: B, its
    lanes L, its r and its buckets a window T), the session that names the warden's domains, and the proof deadline in
    seconds."""

    link: str
    params: WardenParams
    session: bytes
    proof_deadline: float | None = None

    def __post_init__(self) -> None:
        if type(self.params) is not WardenParams or not (isinstance(self.link, str) and self.link) or \
                type(self.session) is not bytes:
            raise ValueError("an ingress tie is on a named link, of a warden.WardenParams and a session")
        Q.Shape(self.params.cell_bytes, self.params.lanes, self.params.slots)

    @property
    def shape(self) -> Q.Shape:
        p = self.params
        return Q.Shape(p.cell_bytes, p.lanes, p.slots)

    @property
    def T(self) -> int:
        return WR.rows_of(self.params)

    @property
    def key(self) -> tuple[int, int, int]:
        sh = self.shape
        return sh.B, sh.L, sh.R

    @cached_property
    def unit(self):
        return bind(_UNIT, **dict(zip(_STATICS, self.key)))

    @cached_property
    def proof_unit(self):
        """The unit a proof's circuit lowers: `unit` with its gate the Definition it names (`head_check.expand`, a
        `ValueError` for an unpinned size class)."""
        return bind(_PROOF_UNIT, **dict(zip(_STATICS, self.key)))

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instance_query(self.unit)

    @cached_property
    def _program(self) -> Program:
        return P.population_program(self.unit, self.T)

    @cached_property
    def _partition(self) -> dict[str, Any]:
        return P.partition(self._program, self.query)

    @property
    def partition(self) -> dict[str, Any]:
        return self._partition

    @cached_property
    def program(self) -> bytes:
        """The registered Program's descriptor, the verifier's own copy, whose SHA-512 the partition object names."""
        return canonical_json(encode_program(self._program))

    @cached_property
    def spec(self) -> S.Spec:
        context = canonical_json({"format": SPEC, "link": self.link, "session": self.session.hex(),
                                  "params": self.params.digest.hex()})
        check = S.Check(P.program_sha512(Program(self.unit)), self.query, public_outputs=OUTPUTS,
                        proof_deadline=self.proof_deadline)
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, PUBLIC, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        return S.Stream(self.spec, self.unit, **kwargs)

    def domain(self, window: int) -> CommitmentDomain:
        """The warden's domain of the ingress link-window's T rows (`warden_rows.domain`)."""
        return WR.domain(self.params, self.session, self.link, Direction.INGRESS, window)

    def ports(self, window: int) -> dict[str, dict[str, Any]]:
        return {VALUE: {"leaves": self.T, "domain": self.domain(window).domain_id.hex()}}

    @staticmethod
    def provenance(window: int) -> dict[str, Any]:
        return {"kind": "served", "window": window}

    def _held(self, head: WR.CellsHead) -> None:
        p = self.params
        if type(head) is not WR.CellsHead or head.link != self.link or head.direction is not Direction.INGRESS or \
                (head.parameters, head.cell_bytes, head.slots, head.rows) != \
                (p.digest, p.cell_bytes, p.grid.allocation, self.T) or \
                head.domain != self.domain(head.window).domain_id:
            raise ValueError("the head isn't the warden's of this link's ingress rows under this schedule and session")

    # the challenger

    def registration(self, head: WR.CellsHead, previous: str | None = None) -> dict[str, Any]:
        """One-stage's record of the link-window: the warden's root of its rows, from its head
        (`warden_rows.check_registration`'s, the warden's signed registration as the challenger reads it), and nothing
        of the rows. Raises `ValueError` on a head of another link, direction, schedule or session."""
        self._held(head)
        roots = [{"port": VALUE, "root": head.root.hex(), **self.ports(head.window)[VALUE]}]
        return R.record(self._program, self.query, {"law": "subset", "k": self.T}, S.SCHEME, roots,
                        self.provenance(head.window), previous=previous)

    def register(self, stream: S.Stream, record: Mapping[str, Any]) -> S.Registration:
        """`stream.register` with the window's own ports, after holding its `window` provenance to exactly
        `{"kind": "served", "window": order}`."""
        window = len(stream.records)
        if not isinstance(record, Mapping) or record.get("window") != self.provenance(window):
            raise S.Refused(Verdict(False, ("window",), {"window": {"expected": self.provenance(window)}}))
        return stream.register(record, self.ports(window))

    def registered(self, head: WR.CellsHead, reads_from: int = 0) -> dict[str, Any]:
        """The verifier's own `--registered` file: the witness row of unit `reads_from + t` reads row t of the warden's
        root (`warden_rows.bucket_port`)."""
        self._held(head)
        return WR.reads_file({REGISTERED: WR.bucket_port(head, self.params, reads_from)})

    # the statement and the unit's arguments

    def lanes(self, advice: Any, releases: Sequence[Q.Release], window: int) -> tuple[tuple[Q.Lane, ...], ...] | None:
        """Each of the window's buckets' lanes from `advice`'s v0 prompt lengths and the warden's `releases`
        (`io_request.lanes`), None without an `advised.LinkAdvice`, past its windows, or where the release isn't the
        warden's schedule for its requests."""
        if type(advice) is not AD.LinkAdvice or not 0 <= window < len(advice.received):
            return None
        try:
            return tuple(Q.lanes(self.shape, prompt_lens(advice), releases, window * self.T + t)
                         for t in range(self.T))
        except ValueError:
            return None

    def arguments(self, row: bytes, lanes: Sequence[Q.Lane]) -> dict[str, list[int]] | None:
        """One unit's arguments on its bucket `row` and `lanes`, by parameter, each its leaves' bits: the prover's side.
        None where the row isn't the unit's N cells of B bytes or a lane isn't five u32s."""
        sh = self.shape
        if len(row) != sh.N * sh.B:
            return None
        try:
            args = Q.arguments(sh, Q.Instance(tuple(row[i * sh.B:(i + 1) * sh.B] for i in range(sh.N)), tuple(lanes)))
        except ValueError:
            return None
        return dict(zip((n for n, _ in self.unit.signature()[0]), args))

    def public_arguments(self, lanes: Sequence[Q.Lane]) -> dict[str, list[int]] | None:
        """A unit's public arguments (`layout`'s ``pub``) as the verifier derives them: its lanes."""
        args = self.arguments(bytes(self.shape.N * self.shape.B), lanes)
        return None if args is None else {n: args[n] for n in self.layout["pub"]}

    def outputs(self, row: bytes, lanes: Sequence[Q.Lane]) -> int:
        """A unit's output bit on its row by its gate's reference (`io_request.reference`), which the tests hold to the
        circuit: what the prover claims."""
        sh = self.shape
        if len(row) != sh.N * sh.B:
            return 0
        return Q.reference(sh, Q.Instance(tuple(row[i * sh.B:(i + 1) * sh.B] for i in range(sh.N)), tuple(lanes)))

    @cached_property
    def layout(self) -> dict[str, tuple[str, ...]]:
        """The statement's two input rows (`ROWS`), each its parameters in the unit's signature's order: ``pub`` the
        lanes, ``wit`` the bucket row (`WITNESS`), each the little-endian concatenation of its parameters' bits in whole
        u16 words, so a unit's ``wit`` row is its bucket row's bytes."""
        params, _ = self.unit.signature()
        out = {r: tuple(n for n, _ in params if (n in WITNESS) == (r == "wit")) for r in ROWS}
        for r, names in out.items():
            bits = sum(t.leaves for n, t in params if n in names)
            if bits % 16:
                raise ValueError(f"the unit's {r} row is {bits} bits, not whole u16 words")
        return out

    def bits(self, row: str) -> int:
        if row == "o0":
            return OUT_BITS
        params, _ = self.unit.signature()
        return sum(t.leaves for n, t in params if n in self.layout[row])

    def row(self, args: Mapping[str, Sequence[int]], row: str) -> bytes:
        bits = [b for n in self.layout[row] for b in args[n]]
        return bytes(sum(b << j for j, b in enumerate(bits[k:k + 8])) for k in range(0, len(bits), 8))

    # the verifier of record

    def gates(self) -> list[PrimitiveDefinition]:
        return [gate(*self.key)]

    def proved(self, findings: Sequence[Finding]) -> bool:
        """Whether `findings` prove this tie's unit: its gate names IoTieRequest's pinned circuit digest, and one or
        more sessions each accepted and named the registered Program."""
        want = self._partition["program"]
        return all(HC.pinned(g) for g in self.gates()) and bool(findings) \
            and all(f.accepted and f.program == want for f in findings)

    def clear(self, cells: WR.Cells, draw: S.Draw, advice: Any, releases: Sequence[Q.Release]) -> dict[str, Any]:
        """The in-the-clear check on the warden's opening of the rows, the witness: the debug comparison `verify` makes
        on request, never a verdict of record. Accepted when the opening reproduces the registered root and
        IoTieRequest's reference accepts every row on the verifier's lanes from `advice` and `releases`."""
        reg = draw.registration.record
        root, w = reg["roots"][0]["root"], reg["window"]["window"]
        lanes = self.lanes(advice, releases, w)
        accepted = cells.window == w and cells.root.hex() == root and lanes is not None and \
            all(self.outputs(r, ln) == 1 for r, ln in zip(cells.rows, lanes))
        return {"of_record": False, "accepted": accepted}

    def verify(self, draw: S.Draw, delivered: Delivered, *, circuit_sha512: str, advice: Any,
               releases: Sequence[Q.Release], head: WR.CellsHead, opening: WR.Cells | None = None) -> dict[str, Any]:
        """The verdict of record on the window's delivered proof, from public inputs and the proof alone: accepted when
        the verifier's own run of Flock's Lean verifier (`proofs.verify`, on the circuit file the auditor built,
        `circuit_sha512`, and the verifier's own registered reads at `head`, the warden's head of the registered root,
        `registered`) accepts every session, they prove this tie's registered unit (`proved`) and every drawn unit, and
        each drawn unit's public row opens to the verifier's own copy (`public_arguments` on its bucket's lanes from
        `advice` and `releases`, `lanes`) and its output row to 1. It reports `advice`'s digest (`advised.digest`),
        None without an `advised.LinkAdvice`. A head that isn't the registered root's refuses. `opening` (the witness)
        is for a debug comparison only: given, the verdict also carries `clear`'s accept bit, which decides nothing."""
        reg = draw.registration.record
        w, root = reg["window"]["window"], reg["roots"][0]
        try:
            self._held(head)
            ok_head = head.window == w and head.root.hex() == root["root"] and head.domain.hex() == root["domain"]
        except ValueError:
            ok_head = False
        accepted = False
        if ok_head:
            findings, refused = PF.verify(delivered.proof, circuit_sha512=circuit_sha512, program=self.program,
                                          partition=self._partition, registered=self.registered(head))
            units = list(draw.units)
            proved = refused is None and self.proved(findings) and all(list(f.units) == units for f in findings)
            lanes = self.lanes(advice, releases, w) if proved else None
            accepted = lanes is not None and len(units) == self.T and all(
                _opened(delivered, u, None if (a := self.public_arguments(lanes[u])) is None else self.row(a, "pub"),
                        self.bits("pub")) == 1 for u in units)
        public = {"verdict": int(accepted), "advice": AD.digest(advice) if type(advice) is AD.LinkAdvice else None}
        sessions = delivered.sessions()
        digest = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": sessions,
                          "accepted": accepted, "public": public})
        verdict = {"of_record": True, "decided_by": "proof", "accepted": accepted, "digest": digest,
                   "sessions": sessions, "public": public}
        if opening is not None:
            verdict["clear"] = {"accepted": self.clear(opening, draw, advice, releases)["accepted"]}
        return verdict

    def statement(self, draw: S.Draw) -> dict[str, Any]:
        reg = draw.registration.record
        return {"format": STATEMENT, "check": self.spec.check.program, "link": self.link,
                "window": reg["window"]["window"], "root": reg["roots"][0]["root"]}

    def outcome(self, stream: S.Stream, draw: S.Draw, delivered: Delivered, *, circuit_sha512: str, advice: Any,
                releases: Sequence[Q.Release], head: WR.CellsHead, opening: WR.Cells | None = None) -> S.Outcome:
        """`stream.outcome` on the verifier of record's verdict (`verify`), with the registered root as the session's."""
        verdict = self.verify(draw, delivered, circuit_sha512=circuit_sha512, advice=advice, releases=releases,
                              head=head, opening=opening)
        return stream.outcome(draw, {VERIFIER: verdict}, self.statement(draw),
                              session_roots={VALUE: draw.registration.record["roots"][0]["root"]})

    def tie(self, stream: S.Stream, advice: AD.LinkAdvice, profile: AD.AdvisedProfile,
            releases: Sequence[Q.Release]) -> AD.IoTieVerdict:
        """The tie of `advice` on this link (the module docstring), `advised.IoTieVerdict` on this link and the stream's
        registrations (each one's `registration.digest`, in order), one bit a registration: window w accepted when the
        stream's w-th registration is window w's and its outcome of record accepted, on the proof, naming `advice`'s
        digest, and `releases` is the warden's schedule for `advice`'s requests on every bucket of its windows
        (`lanes`) and gives its `received`; every window refused when they don't, when the stream has more or fewer
        registrations than `advice` has windows, or when `profile`'s framing isn't the warden's cut at B for `advice`'s
        prompt lengths (`framing`) or names an own frame. Raises `ValueError` on a stream that isn't this tie's, an
        advice that isn't an `advised.LinkAdvice` or a profile that isn't an `advised.AdvisedProfile`."""
        if stream.spec.digest() != self.spec.digest():
            raise ValueError("the stream is not this tie's: its spec isn't IngressTie.spec")
        if type(advice) is not AD.LinkAdvice or type(profile) is not AD.AdvisedProfile:
            raise ValueError("a tie is of an advised.LinkAdvice under an advised.AdvisedProfile")
        name, lens, n = AD.digest(advice), prompt_lens(advice), len(advice.received)
        frames, B = profile.request_frames, self.shape.B
        framed = profile.own == 0 and \
            all(k <= len(frames) and frames[k - 1] == request_cells(k, B) for k in set(lens))
        released = all(self.lanes(advice, releases, w) is not None for w in range(n)) and \
            window_cells(releases, self.T) == {w: c for w, c in enumerate(advice.received) if c}
        outcomes = {r["registration"]: r for r in stream.records}
        log, ok = stream.log, []
        for w in range(n):
            reg = log[w] if w < len(log) else None
            if reg is None or reg.get("window") != self.provenance(w):
                break
            body = outcomes.get(R.digest(reg))
            verdict = ((body or {}).get("audit") or {}).get("verdicts", {}).get(VERIFIER) or {}
            ok.append(body is not None and body["symbol"] == S.ACCEPT and body["public"].get("advice") == name
                      and verdict.get("decided_by") == "proof")
        held = len(log) == len(ok) == n and framed and released
        accepted = tuple(held and x for x in ok) + (False,) * (len(log) - len(ok))
        return AD.IoTieVerdict(self.link, tuple(R.digest(r) for r in log), name, accepted)


@dataclass(frozen=True)
class Ties:
    """The verifier's own tie source on one link (`advised.TieSource`, `service.LinkWindows.ties`): the ingress tie, its
    stream of record, the challenger's and the verifier of record's, never the prover's, and the warden's ingress
    release as the verifier read it off the ingress record (`releases`, the named gap). `registrations` is the stream's
    registrations (each one's `registration.digest`, in order) and `verdict` is `IngressTie.tie` on the stream, so the
    advised link reads IoTieRequest's verdict only from outcomes the verifier of record decided on proofs. Each raises
    `ValueError` on another link than the tie's, which the advised link reads as a refusal."""

    ingress: IngressTie
    stream: S.Stream
    releases: tuple[Q.Release, ...]

    def __post_init__(self) -> None:
        if type(self.ingress) is not IngressTie or type(self.stream) is not S.Stream or \
                self.stream.spec.digest() != self.ingress.spec.digest():
            raise ValueError("a tie source is an IngressTie and its own stream")
        object.__setattr__(self, "releases", tuple(self.releases))

    def _on(self, link: str) -> None:
        if link != self.ingress.link:
            raise ValueError(f"the tie source is on link {self.ingress.link!r:.80}, not {link!r:.80}")

    def registrations(self, link: str) -> tuple[str, ...]:
        self._on(link)
        return tuple(R.digest(r) for r in self.stream.log)

    def verdict(self, link: str, advice: AD.LinkAdvice, profile: AD.AdvisedProfile) -> AD.IoTieVerdict:
        self._on(link)
        return self.ingress.tie(self.stream, advice, profile, self.releases)


__all__ = ["Delivered", "IngressTie", "OUTPUTS", "PORTS", "PUBLIC", "REGISTERED", "ROWS", "Ties", "VALUE", "VERIFIER",
           "WITNESS", "framing", "gate", "prompt_lens", "request_cells", "window_cells"]
