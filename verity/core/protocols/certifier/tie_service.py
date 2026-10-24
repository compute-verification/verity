"""The ingress tie through the proof service: each ingress link-window's `io_request.IoTieRequest` proved, and the
verifier of record that builds `advised.IoTieVerdict`, the tie of an advised link's requests to its certified ingress,
only from IoTieRequest proofs it verified on statements it computed itself (PROTOCOL.md, "The ingress tie").

**What is committed.** One `Value`, `cells`: an ingress link-window's N = T·R cells of B bytes (`io_request`'s module
docstring), one `hm96-sha512/row/v2` row a bucket, each under a fresh 192-byte salt, in a dense `frame-v3-sha512` tree
of T rows (`io_tie.rows_root`) under one-stage's served domain, whose run is the link and the window
(`IngressTie.domain`, an `io_tie.domain`). The warden, the ingress certifier's device and so the auditor's, commits the
cells (`IngressTie.commit`), registers their root with the challenger (`IngressTie.registration`) and hands the prover
the salts, privately. The cells, the salts and the prover's switches are witness: the challenger and the verifier see
the root. A registration is one link-window, one proof unit; its `window` provenance is `{"kind": "served", "window":
w}`, held to the registration's order (`IngressTie.register`). Selection is `All`.

**The check** is `IngressTie.unit`, `NetworkIngressTieWindow` v1{B, T, R, OWN}: one gate naming
`io_request.IoTieRequest_v1{B, T, R, OWN}` and its circuit digest, pinned in `io_request.circuit-check.json` and held
to the Definition by circuit-check (`gate`, as `head_check.gate` holds the head check), and `pad`, the witness bits
that fill the witness row to whole u16 words (IoTieRequest's switches are (2D − 1)·2^(D − 1) bits, 20 at N = 8),
held to zero. Its one output bit is IoTieRequest's AND the pad's. A size class no sidecar pins has a gate that names
no digest, and no proof of its unit is accepted (`proved`): at the served size (B = 4,096, T = 600, R = 9) a window is
about 1.14 × 10^10 ANDs and none is pinned, so there every window, and every advised link, is refused.

**The statement is the verifier's own** (`io_request.statements`): window w's root and domain id are the warden's
registration, and its view (`io_request.request_view`: the carry-in, `ended`, `received` and the carry-out) comes from
the advice's v0 prompt lengths and `received` (`advised.LinkAdvice`), never from the prover. The proof's ``pub`` row is
IoTieRequest's public parameters on that statement (`IngressTie.public_arguments`), its ``wit`` row the cells, their
salts, the switches and the pad, and its ``o0`` row one u16 word holding the output bit, each committed under a hiding
salt (`IngressTie.layout`).

**The verifier of record** is `IngressTie.verify(…, advice=A)`, which decides on the delivered proof and public inputs
alone. It accepts the window when the verifier's own run of C-Flock's Lean verifier accepts every session, they prove
this tie's registered unit (`proved`: its gate names the pinned digest, every session the registered Program) and the
drawn unit, and the proof's ``pub`` row opens, under the salt delivered, to the verifier's own copy and its ``o0`` row
to 1. Its public outputs are `verdict` and `advice`, A's `advised.digest`. A proof on another advice's view, another
carry or another window's statement opens to another ``pub`` row and is refused.

**The tie** (`IngressTie.tie`) is `advised.IoTieVerdict(link, registrations, advised.digest(A), accepted)`: the link,
the stream's registrations (each one's `registration.digest`, in order), A's digest and one bit a registration, read
from the stream's outcomes of record alone. Window w is accepted when the stream's w-th registration is window w's, its
outcome of record accepted, decided on the proof, naming A's digest, and when the link's statements chain from window
0 (`io_request.chained`: window 0's carry-in is (0, 0, 0), each carry-out the next window's carry-in, the last between
requests). Every window is refused when they don't, when the stream holds a registration past A's windows or misses
one, and when the advised profile's framing isn't IoTieRequest's for A's prompt lengths (`active.id_request_frames`
at B, which `framing` gives) or its OWN isn't the tie's. So an accepted verdict on every window says what IoTieRequest's
statements say together: each window's cells take the request in progress from its carry-in to its carry-out, so the
link's j-th certified request is advice request j and its prompt v0's `prompt_len` prompt fields long (REDTEAM phase 9
(b): a carried field count is held by the previous window's carry-out). That is the premise `advised.ends` reads, from
the verifier's own tie source alone (`Ties`, the tie and its stream as `service.LinkWindows.ties`), never from what
the prover delivers (REDTEAM phase 10), and `advised.view` holds the verdict to the link and the registrations the
source records, so a verdict on another link or another ingress record isn't reused.

**What it doesn't establish.** The prompt ids' values (the ingress-ids tie to the Commit's `c00.input/ids`,
PROTOCOL.md, not built). That the warden's cells are the frames the ingress certifier counted is the device's: both are
the auditor's device's record of the link, and the advised unit holds `received` to the certified ingress head.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity.core.protocols.certifier import advised as AD, head_check as HC, io_request as Q, io_tie as IT
from verity.core.protocols.certifier import service as SVC
from verity.core.protocols.certifier.active import id_request_frames
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, proofs as PF, registration as R
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict

SPEC = "verity-network-certifier/ingress-tie-spec/v0"
STATEMENT = "verity-network-certifier/ingress-tie-statement/v0"
VERDICT = "verity-network-certifier/ingress-tie-verdict/v0"
VALUE = "cells"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "ingress-tie"
OUTPUTS = ("verdict", "advice")
ROWS, PORTS, OUT_BITS = SVC.ROWS, SVC.PORTS, SVC.OUT_BITS
#: the unit's witness parameters: IoTieRequest's, and the pad
WITNESS = ("cells", "salts", "switches", "pad")
_INST = hm96.SHA512
_STATICS = ("B", "T", "R", "OWN")

PUBLIC = (
    S.Public("population", "structure", "one proof unit per registration: the ingress link-window"),
    S.Public("window", "structure", "provenance: the ingress link-window's index, held to the registration's order"),
    S.Public(f"leaves:{VALUE}", "structure", "T leaves, one bucket row of R cells each"),
    S.Public(f"domain:{VALUE}", "structure", "the served domain, which binds the check, the link and the window"),
    S.Public(VALUE, "commitment", "the warden's salted root over the window's ingress cells"),
    S.Public("shape", "structure", "B, T, R and OWN: the cells' bytes, the buckets, the cells a bucket and the "
             "certifier's own frame, fixed before the run"),
    S.Public("verdict", "output", "the window's IoTieRequest verdict, accept or reject"),
    S.Public("advice", "output", "the advice the window's statement is computed from (`advised.digest`): its v0 "
             "prompt lengths and its windows' received request frames give the window's request view "
             "(`io_request.request_view`), the unit's public row in the named gap; `IngressTie.tie` holds every "
             "window to one advice"),
)


def _shape(S_) -> Q.Shape:
    return Q.Shape(S_.B, S_.T, S_.R, S_.OWN)


def _pad(shape: Q.Shape) -> int:
    return -shape.switches % 16


@lru_cache(maxsize=None)
def gate(B: int, T: int, R: int, OWN: int) -> PrimitiveDefinition:
    """`IoTieRequest_v1{B, T, R, OWN}` as one gate that names it and its pinned circuit digest (`io_request.PINS`),
    `Gate[IoTieRequest_v1{…}@<digest>]_v2`, as `head_check.gate` holds the head check: the unit's Program holds the gate
    rather than the Definition's body, whose row commitments encode to about 120 MB at B = 512, T = 4, R = 2."""
    d = bind(Q.IoTieRequest, B=B, T=T, R=R, OWN=OWN)
    return HC.gate_of(d, HC.pinned_digest(d, Q.PINS), "the ingress tie")


@lru_cache(maxsize=None)
def _expanded(B: int, T: int, R: int, OWN: int):
    return HC.expand(gate(B, T, R, OWN), bind(Q.IoTieRequest, B=B, T=T, R=R, OWN=OWN))


def _unit(check) -> CompositeDefinition:
    def sig(S_):
        params, _ = gate(S_.B, S_.T, S_.R, S_.OWN).signature()
        n = _pad(_shape(S_))
        return tuple(params) + ((("pad", Array(n, BL.BIT)),) if n else ()), BL.BIT

    def body(B, S_, *args):
        g = check(S_.B, S_.T, S_.R, S_.OWN)
        n = len(g.signature()[0])
        ok = B.call(g, *args[:n])
        if len(args) == n:
            return ok
        zero = [B.call(BL.NOT, HC._coll(BL.BIT, [r])).refs[0] for r in HC._refs(args[n])]
        bits = HC._refs(ok) + zero
        return B.call(BL.and_gate(len(bits)), HC._coll(Array(len(bits), BL.BIT), bits))

    return CompositeDefinition(
        "NetworkIngressTieWindow", 1, _STATICS, sig, body, register=False,
        doc="the proof unit of the ingress tie: one ingress link-window's `io_request.IoTieRequest` on the warden's "
            "cells, and the pad that fills its witness row to whole u16 words, held to zero")


_UNIT = _unit(gate)
#: the same unit with the gate replaced by the Definition it names, as a proof's circuit lowers it
_PROOF_UNIT = _unit(_expanded)


def framing(B: int, n_max: int) -> tuple[int, ...]:
    """The link's framing as IoTieRequest cuts it, for `advised.AdvisedProfile.request_frames`: entry n − 1 is the
    request frames an id request of an n-token prompt takes at B (`active.id_request_frames`), n = 1 … n_max."""
    return tuple(id_request_frames(n, B) for n in range(1, n_max + 1))


def prompt_lens(advice: AD.LinkAdvice) -> tuple[int, ...]:
    """The advice's v0 prompt lengths, in its requests' order."""
    return tuple(q.prompt_len for q in advice.advice.requests)


@dataclass(frozen=True)
class Cells:
    """The warden's side of one ingress link-window's commitment, all witness: its N cells, bucket by bucket, and one
    salt a bucket row."""

    window: int
    cells: tuple[bytes, ...]
    salts: tuple[bytes, ...]


def _opened(delivered: SVC.Delivered, unit: int, public: bytes | None, bits: int) -> int | None:
    """The unit's output bit, once the proof's public file holds its public row as `public` (the verifier's own copy,
    `bits` long) and its output row as the bit delivered, each under the salt delivered; else None."""
    bcs, word, salts = SVC._commitments(delivered.proof.public, unit), delivered.output, delivered.salts
    if public is None or bcs is None or type(word) is not int or word not in (0, 1):
        return None
    if not (SVC._opens(public, bits, salts.get("pub"), bcs["pub"])
            and SVC._opens(SVC.output_row(word), OUT_BITS, salts.get("o0"), bcs["o0"])):
        return None
    return word


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class IngressTie:
    """The ingress tie on one link, every field public: the link, its ingress windows' shape (`io_request.Shape`: B, T,
    R and OWN, the link's ingress frame size, buckets, allocation and own frame), the run that names the served domains,
    and the proof deadline in seconds."""

    link: str
    shape: Q.Shape
    run: str
    proof_deadline: float | None = None

    def __post_init__(self) -> None:
        if type(self.shape) is not Q.Shape or not (isinstance(self.link, str) and self.link):
            raise ValueError("an ingress tie is on a named link, of an io_request.Shape")

    @property
    def key(self) -> tuple[int, int, int, int]:
        sh = self.shape
        return sh.B, sh.T, sh.R, sh.OWN

    @property
    def pad(self) -> int:
        return _pad(self.shape)

    @cached_property
    def unit(self):
        return bind(_UNIT, **dict(zip(_STATICS, self.key)))

    @cached_property
    def proof_unit(self):
        """The unit a proof's circuit lowers: `unit` with its gate the Definition it names (`head_check.expand`, which
        traces it: about 75 s at B = 512, T = 4, R = 2, and a `ValueError` for an unpinned size class)."""
        return bind(_PROOF_UNIT, **dict(zip(_STATICS, self.key)))

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instance_query(self.unit)

    @cached_property
    def _program(self) -> Program:
        return P.population_program(self.unit, 1)

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
        context = canonical_json({"format": SPEC, "link": self.link, "run": self.run, "shape": list(self.key)})
        check = S.Check(P.program_sha512(Program(self.unit)), self.query, public_outputs=OUTPUTS,
                        proof_deadline=self.proof_deadline)
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, PUBLIC, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        return S.Stream(self.spec, self.unit, **kwargs)

    def domain(self, window: int) -> CommitmentDomain:
        """One-stage's served domain (`registration.served_domain`) for the ingress link-window, T leaves: run
        `<run>/<link>/<window>`."""
        binding = identity_digest_sha512(R.SERVED_DOMAIN, {
            "program": self._partition["program"], "partition": P.digest(self._partition), "port": VALUE,
            "schema": SCHEMA, "leaves": self.shape.T, "run": f"{self.run}/{self.link}/{window}"})
        return IT.domain(binding, self.shape.T)

    def ports(self, window: int) -> dict[str, dict[str, Any]]:
        return {VALUE: {"leaves": self.shape.T, "domain": self.domain(window).domain_id.hex()}}

    @staticmethod
    def provenance(window: int) -> dict[str, Any]:
        return {"kind": "served", "window": window}

    # the warden

    def commit(self, window: int, cells: Sequence[bytes], salts: Sequence[bytes] | None = None) -> Cells:
        """The window's commitment: its N cells and T distinct salts, one a bucket row (default: fresh)."""
        sh = self.shape
        salts = tuple(salts) if salts is not None else tuple(_INST.fresh_salt() for _ in range(sh.T))
        if len(cells) != sh.N or any(len(c) != sh.B for c in cells):
            raise ValueError(f"an ingress window is {sh.N} cells of {sh.B} bytes")
        if len(salts) != sh.T or len(set(salts)) != sh.T or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"{sh.T} distinct {_INST.salt_bytes}-byte salts")
        return Cells(window, tuple(bytes(c) for c in cells), tuple(bytes(s) for s in salts))

    def root(self, c: Cells) -> str:
        return IT.rows_root(self.domain(c.window).domain_id, Q.rows(self.shape, c.cells), c.salts).hex()

    def registration(self, c: Cells, previous: str | None = None) -> dict[str, Any]:
        """One-stage's record of the window: the cells' salted root and nothing of the cells."""
        roots = [{"port": VALUE, "root": self.root(c), **self.ports(c.window)[VALUE]}]
        return R.record(self._program, self.query, {"law": "subset", "k": 1}, S.SCHEME, roots,
                        self.provenance(c.window), previous=previous)

    # the challenger

    def register(self, stream: S.Stream, record: Mapping[str, Any]) -> S.Registration:
        """`stream.register` with the window's own ports, after holding its `window` provenance to exactly
        `{"kind": "served", "window": order}`."""
        window = len(stream.records)
        if not isinstance(record, Mapping) or record.get("window") != self.provenance(window):
            raise S.Refused(Verdict(False, ("window",), {"window": {"expected": self.provenance(window)}}))
        return stream.register(record, self.ports(window))

    # the statement and the unit's arguments

    def view(self, advice: Any, window: int) -> Q.RequestView | None:
        """The window's request view from `advice` (`io_request.request_view` of its v0 prompt lengths and `received`),
        None without an `advised.LinkAdvice`, past its windows, or where its requests' frames aren't its windows'
        total."""
        if type(advice) is not AD.LinkAdvice or not 0 <= window < len(advice.received):
            return None
        try:
            return Q.request_view(prompt_lens(advice), advice.received, window, self.shape.B)
        except ValueError:
            return None

    def _arguments(self, inst: Q.Instance, switches: Sequence[int]) -> dict[str, list[int]] | None:
        try:
            args = Q.arguments(self.shape, inst, switches)
        except ValueError:
            return None
        out = dict(zip((n for n, _ in self.unit.signature()[0]), args))
        if self.pad:
            out["pad"] = [0] * self.pad
        return out

    def _instance(self, c: Cells, root: bytes, view: Q.RequestView) -> Q.Instance:
        return Q.Instance(c.cells, c.salts, root, self.domain(c.window).domain_id, view, c.window)

    def arguments(self, c: Cells, root: bytes, view: Q.RequestView) -> dict[str, list[int]] | None:
        """The unit's arguments on the cells under `root`, the registered root, and `view`, by parameter, each its
        leaves' bits, the switches the prover's routing (`io_request.switches`): the prover's side. None where the view
        has more prompt lengths than N or a value past a u32."""
        inst = self._instance(c, root, view)
        return self._arguments(inst, Q.switches(self.shape, inst))

    def public_arguments(self, window: int, root: bytes, view: Q.RequestView) -> dict[str, list[int]] | None:
        """The unit's public arguments (`layout`'s ``pub``) as the verifier derives them: the registered `root`, the
        window's domain id, the window and `view`. None where the unit takes no such view."""
        sh = self.shape
        blank = Cells(window, (bytes(sh.B),) * sh.N, (bytes(_INST.salt_bytes),) * sh.T)
        args = self._arguments(self._instance(blank, root, view), [0] * sh.switches)
        return None if args is None else {n: args[n] for n in self.layout["pub"]}

    def outputs(self, c: Cells, root: bytes, view: Q.RequestView) -> int:
        """The unit's output bit on the cells by its gate's reference (`io_request.reference`), which the tests hold to
        the circuit: what the prover claims."""
        return Q.reference(self.shape, self._instance(c, root, view))

    @cached_property
    def layout(self) -> dict[str, tuple[str, ...]]:
        """The statement's two input rows (`ROWS`), each its parameters in the unit's signature's order: ``pub`` the
        public ones, ``wit`` the cells, their salts, the switches and the pad (`WITNESS`), each the little-endian
        concatenation of its parameters' bits in whole u16 words."""
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

    def clear(self, c: Cells, draw: S.Draw, advice: Any) -> dict[str, Any]:
        """The in-the-clear check on the cells, the witness: the debug comparison `verify` makes on request, never a
        verdict of record. Accepted when the cells reproduce the registered root and IoTieRequest's reference accepts
        them on the verifier's view of `advice`."""
        reg = draw.registration.record
        root, w = reg["roots"][0]["root"], reg["window"]["window"]
        view = self.view(advice, w)
        accepted = c.window == w and self.root(c) == root and view is not None and \
            self.outputs(c, bytes.fromhex(root), view) == 1
        return {"of_record": False, "accepted": accepted}

    def verify(self, draw: S.Draw, delivered: SVC.Delivered, *, circuit_sha512: str, advice: Any,
               opening: Cells | None = None) -> dict[str, Any]:
        """The verdict of record on the window's delivered proof, from public inputs and the proof alone: accepted when
        the verifier's own run of C-Flock's Lean verifier (`proofs.verify`, on the circuit file the auditor built,
        `circuit_sha512`) accepts every session, they prove this tie's registered unit (`proved`) and the drawn unit,
        and the proof's public row opens to the verifier's own copy (`public_arguments` on the registered root and the
        window's view of `advice`, `view`) and its output row to 1. It reports `advice`'s digest (`advised.digest`),
        None without an `advised.LinkAdvice`. `opening` (the witness) is for a debug comparison only: given, the
        verdict also carries `clear`'s accept bit, which decides nothing."""
        reg = draw.registration.record
        w, root = reg["window"]["window"], bytes.fromhex(reg["roots"][0]["root"])
        findings, refused = PF.verify(delivered.proof, circuit_sha512=circuit_sha512, program=self.program,
                                      partition=self._partition)
        proved = refused is None and self.proved(findings) and all(list(f.units) == list(draw.units) for f in findings)
        view = self.view(advice, w)
        args = None if view is None else self.public_arguments(w, root, view)
        word = _opened(delivered, draw.units[0], None if args is None else self.row(args, "pub"),
                       self.bits("pub")) if proved else None
        accepted = word == 1
        public = {"verdict": int(accepted), "advice": AD.digest(advice) if type(advice) is AD.LinkAdvice else None}
        sessions = delivered.sessions()
        digest = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": sessions,
                          "accepted": accepted, "public": public})
        verdict = {"of_record": True, "decided_by": "proof", "accepted": accepted, "digest": digest,
                   "sessions": sessions, "public": public}
        if opening is not None:
            verdict["clear"] = {"accepted": self.clear(opening, draw, advice)["accepted"]}
        return verdict

    def statement(self, draw: S.Draw) -> dict[str, Any]:
        reg = draw.registration.record
        return {"format": STATEMENT, "check": self.spec.check.program, "link": self.link,
                "window": reg["window"]["window"], "root": reg["roots"][0]["root"]}

    def outcome(self, stream: S.Stream, draw: S.Draw, delivered: SVC.Delivered, *, circuit_sha512: str, advice: Any,
                opening: Cells | None = None) -> S.Outcome:
        """`stream.outcome` on the verifier of record's verdict (`verify`), with the registered root as the session's."""
        verdict = self.verify(draw, delivered, circuit_sha512=circuit_sha512, advice=advice, opening=opening)
        return stream.outcome(draw, {VERIFIER: verdict}, self.statement(draw),
                              session_roots={VALUE: draw.registration.record["roots"][0]["root"]})

    def tie(self, stream: S.Stream, advice: AD.LinkAdvice, profile: AD.AdvisedProfile) -> AD.IoTieVerdict:
        """The tie of `advice` on this link (the module docstring), `advised.IoTieVerdict` on this link and the stream's
        registrations (each one's `registration.digest`, in order), one bit a registration: window w accepted when the
        stream's w-th registration is window w's and its outcome of record accepted, on the proof, naming `advice`'s
        digest, and the link's statements (`io_request.statements` on the registered roots and domains) chain from
        window 0; every window refused when they don't, when the stream has more or fewer registrations than `advice`
        has windows, or when `profile`'s framing isn't IoTieRequest's at B for `advice`'s prompt lengths or its OWN
        isn't the tie's. Raises `ValueError` on a stream that isn't this tie's, an advice that isn't an
        `advised.LinkAdvice` or a profile that isn't an `advised.AdvisedProfile`."""
        if stream.spec.digest() != self.spec.digest():
            raise ValueError("the stream is not this tie's: its spec isn't IngressTie.spec")
        if type(advice) is not AD.LinkAdvice or type(profile) is not AD.AdvisedProfile:
            raise ValueError("a tie is of an advised.LinkAdvice under an advised.AdvisedProfile")
        name, lens, n = AD.digest(advice), prompt_lens(advice), len(advice.received)
        frames = profile.request_frames
        framed = profile.own == self.shape.OWN and \
            all(k <= len(frames) and frames[k - 1] == id_request_frames(k, self.shape.B) for k in set(lens))
        outcomes = {r["registration"]: r for r in stream.records}
        log, ok, registered = stream.log, [], []
        for w in range(n):
            reg = log[w] if w < len(log) else None
            if reg is None or reg.get("window") != self.provenance(w):
                break
            root = reg["roots"][0]
            registered.append((bytes.fromhex(root["root"]), bytes.fromhex(root["domain"])))
            body = outcomes.get(R.digest(reg))
            verdict = ((body or {}).get("audit") or {}).get("verdicts", {}).get(VERIFIER) or {}
            ok.append(body is not None and body["symbol"] == S.ACCEPT and body["public"].get("advice") == name
                      and verdict.get("decided_by") == "proof")
        try:
            chained = len(log) == len(registered) == n and Q.chained(Q.statements(self.shape, lens, advice.received,
                                                                                  registered))
        except ValueError:
            chained = False
        held = chained and framed
        accepted = tuple(held and x for x in ok) + (False,) * (len(log) - len(ok))
        return AD.IoTieVerdict(self.link, tuple(R.digest(r) for r in log), name, accepted)


@dataclass(frozen=True)
class Ties:
    """The verifier's own tie source on one link (`advised.TieSource`, `service.LinkWindows.ties`): the ingress tie and
    its stream of record, the challenger's and the verifier of record's, never the prover's. `registrations` is the
    stream's registrations (each one's `registration.digest`, in order) and `verdict` is `IngressTie.tie` on the
    stream, so the advised link reads IoTieRequest's verdict only from outcomes the verifier of record decided on
    proofs. Each raises `ValueError` on another link than the tie's, which the advised link reads as a refusal."""

    ingress: IngressTie
    stream: S.Stream

    def __post_init__(self) -> None:
        if type(self.ingress) is not IngressTie or type(self.stream) is not S.Stream or \
                self.stream.spec.digest() != self.ingress.spec.digest():
            raise ValueError("a tie source is an IngressTie and its own stream")

    def _on(self, link: str) -> None:
        if link != self.ingress.link:
            raise ValueError(f"the tie source is on link {self.ingress.link!r:.80}, not {link!r:.80}")

    def registrations(self, link: str) -> tuple[str, ...]:
        self._on(link)
        return tuple(R.digest(r) for r in self.stream.log)

    def verdict(self, link: str, advice: AD.LinkAdvice, profile: AD.AdvisedProfile) -> AD.IoTieVerdict:
        self._on(link)
        return self.ingress.tie(self.stream, advice, profile)


__all__ = ["Cells", "IngressTie", "OUTPUTS", "PORTS", "PUBLIC", "ROWS", "Ties", "VALUE", "VERIFIER", "WITNESS",
           "framing", "gate", "prompt_lens"]
