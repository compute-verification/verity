"""The egress tie through the proof system: each execution phase's step-indexed scan on a network certifier link
(`ingress_scan`, `egress`) as one registration of the proof system, every bucket proved, and its verifier, which builds
`EgressTieVerdict`, the tie of a link's certified egress to the committed trace, only from proofs it verified on reads
it computed itself.

**What is committed.** The network certifier's own (`warden_rows`): an execution phase's T ingress bucket rows and its T
egress bucket rows, each link's rows an `hm96-sha512/row/v2` row under the network certifier's fresh salt in its
`frame-v3-sha512` tree under its domain (`warden_rows.domain`: the session, the link, the direction and the phase's
index), whose root its signed head names (`warden_rows.CellsHead`, which `warden_rows.check_registration` reads off its
signed registration). The committed trace (`Trace`), registered by the compute audit before this registration: its
`SCAN_STATE` (`ingress_scan.window_state`, T leaves, leaf t the scan's S_{t−1} and leaf 0 the empty state), which the
anchored Programs read, and each lane j's slot entries (nb = T / EVERY leaves, leaf b slot j's entry at boundary b),
which they write; each a tree at that registration's root and domain. And the prover's own, before any coin: the scan's
views (``scan/view``, T leaves, leaf t S_t's egress view) and S_{T−1} (``scan/last``, one leaf), which no state leaf of
the phase holds, each a hiding tree whose domain binds the registered Program and the ingress root
(`EgressTie.view_tree`, `EgressTie.last_tree`). The registration (`EgressTie.registration`) names every root:
``rows/ingress``, ``rows/egress``, ``scan/state``, ``scan/prog<j>``, ``scan/view`` and ``scan/last``. Its `window`
provenance, `{"kind": "served", "window": w}`, is the phase's index, held to the registration's order
(`EgressTie.register`).

**The check** is one Program of three templates in the generator form (`scan_root`, the form `NetworkCheck.Period`'s
`scanOf` reads), its 2T proof blocks the templates' instances in order: block 0 `IngressScanFirst` (bucket 0, from the
empty state), blocks 1 … T − 1 `IngressScanBucket` (bucket t), blocks T … 2T − 1 `EgressBucket` (egress bucket t), each
template one batch with its row, state or view inputs its own and every ingress block's ``prog<j>`` row ⌊t / EVERY⌋ of
lane j's slot entries. Selection is `All`: every proof block is proved, every bucket of every execution phase. The
stream's population program is the phase's (`service.Stream(program=…)`).

**The statement.** An execution phase is 2T proof blocks in four statements (`members`), each one Flock proof batching
its blocks: ``first`` (block 0), ``ingress`` (blocks 1 … T − 2), ``last`` (block T − 1) and ``egress``
(blocks T … 2T − 1). Every port is one of the verifier's own registered reads (`EgressTie.registered`; Flock's
`Registered.checkComplete`), so no row a block reads or outputs is committed afresh: ingress block t's ``row`` reads the
network certifier's ingress row t, ``sin`` state leaf t (S_{t−1}), ``prog<j>`` lane j's leaf ⌊t / EVERY⌋ (one table row
a boundary), ``o1`` (S_t) state leaf t + 1 and ``o2`` view leaf t; egress block t's ``state`` reads view leaf t and
``o1``, the row it renders, the network certifier's egress row t; and every block's ``o0``, its ok bit, reads a value
the verifier registers at known content (`EgressTie.known_tree`), row u the word 1 under a public salt. Block T − 1's
``o1``, S_{T−1}, reads ``scan/last`` leaf 0 instead, which is why it is a statement of its own: a statement's port reads
one tree. So S_t is the state leaf one block computes and the next reads, and the trace's, the view ingress block t
computes and egress block t reads, and the egress row the one the network certifier signed: the chains are wires through
the registered roots, which Flock's Lean verifier checks, and an output 0 fails its read.

**The verifier** is `EgressTie.verify(…, heads=…, trace=…)`, a Python verifier until its Lean one replaces it, which
decides on the delivered proofs and public inputs alone. It accepts the execution phase when the registration's roots
are the heads' and the trace's, every proof block is sampled, and its own run of Flock's Lean verifier, with its own
registered reads, accepts every session of every member, on the circuit file the auditor built of the member's template,
naming the registered Program and proving the member's blocks. Its public output is `verdict`.

**The tie** (`EgressTie.tie`) is `EgressTieVerdict(link, registrations, accepted)`: phase w accepted when the stream's
w-th registration is phase w's and its outcome accepted, decided on the proof. The verifier reads it only from its own
source (`Scans`: the tie and its stream), never from what the prover delivers. `service.LinkWindows.network_verdict`
counts a network certifier link's execution phase only where it is accepted, so a run whose egress differs from the
committed trace by one token is refused.

**What it doesn't establish.** Each registration's scan starts from the empty state, so an execution phase whose first
bucket finds a connection live from the phase before is refused (fail closed): S isn't carried across registrations, and
the trace's state leaf 0, the empty state the compute audit's boundary 0 reads, is read by no block of the scan. That
the slot entries are the anchored Programs' outputs on the `SCAN_STATE` they read is the compute audit's, whose
registration the trace's roots are read off. That the network certifier's rows are the link's bytes is the device's
(`trusted-host`).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, coll_concat
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import partition as P, produced, proofs as PF
from verity.core.protocols.spot.one_stage import registered as G, registration as R
from verity.core.protocols.spot.one_stage.verdict import Verdict
from verity.ml.protocols.certifier import egress as EG, ingress_scan as IS

SPEC = "verity-network-certifier/egress-tie-spec/v1"
STATEMENT = "verity-network-certifier/egress-tie-statement/v1"
VERDICT = "verity-network-certifier/egress-tie-verdict/v1"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "egress-tie"
OUTPUTS = ("verdict",)
#: the execution phase's templates in its Program's order
TEMPLATES = ("first", "ingress", "egress")
#: each template's parameters before its slot-entry ports, its inputs of its own in the phase's Program (`scan_root`)
FIXED = {"first": 1, "ingress": 2, "egress": 1}
#: the registration's ports of the network certifier's roots, by direction
ROWS = {Direction.INGRESS: "rows/ingress", Direction.EGRESS: "rows/egress"}
STATE, VIEW, LAST = "scan/state", "scan/view", "scan/last"
#: a ``--registered`` port's known content (Flock's `Registered.KNOWN_KEY`)
KNOWN_KEY = "known"
HEX = re.compile("(?:[0-9a-f]{2})+")


def prog(j: int) -> str:
    """The registration's port of lane j's slot entries."""
    return f"scan/prog{j}"


def value(kind: str) -> str:
    """A scan tree's value: ``state`` (`blocks.SCAN_STATE`), ``view`` or ``last``."""
    return f"verity/certifier/scan-{kind}/v1"


def words(bits: Sequence[int]) -> list[int]:
    """A bit row of whole u16 words as its words, low bit first."""
    if len(bits) % 16:
        raise ValueError(f"a row of {len(bits)} bits is no whole number of u16 words")
    return [sum(int(b) << i for i, b in enumerate(bits[w:w + 16])) for w in range(0, len(bits), 16)]


def templates(cls: IS.Class) -> dict[str, Any]:
    """The execution phase's templates at `cls`: `IngressScanFirst`, `IngressScanBucket` and `EgressBucket` at its egress shape."""
    st, sh = {k: getattr(cls, k) for k in IS.STATICS}, cls.egress
    return {"first": bind(IS.IngressScanFirst, **st), "ingress": bind(IS.IngressScanBucket, **st),
            "egress": bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D)}


def scan_root(members: Sequence[tuple[Any, int, int]], nb: int, every: int) -> Program:
    """The scan's root in the generator form (`NetworkCheck.Period`'s `scanOf`): each `(template, n, fixed)` of
    `members` one batch over its n units, in order, its first `fixed` parameters inputs of its own (``<i>.<name>``),
    and each later one, lane j's ``prog<j>``, at bucket t (the first template's first unit bucket 0, each next unit the
    next bucket) row ⌊t / every⌋ of lane j's input ``prog<j>``, its slot entries of nb boundaries."""
    sigs = [t.signature() for t, _, _ in members]
    lanes = [len(ps) - fixed for (ps, _), (_, _, fixed) in zip(sigs, members)]
    entry = next(ps[fixed][1] for (ps, _), (_, _, fixed), L in zip(sigs, members, lanes) if L)
    L = max(lanes)

    def signature(S_):
        params = tuple((f"{i}.{name}", Array(n, ty)) for i, ((ps, _), (_, n, fixed)) in enumerate(zip(sigs, members))
                       for name, ty in ps[:fixed])
        return (*params, *((f"prog{j}", Array(nb, entry)) for j in range(L))), \
            Tuple(*(Array(n, ret) for (_, ret), (_, n, _) in zip(sigs, members)))

    def body(B, S_, *args):
        entries, outs, at, start = args[len(args) - L:], [], 0, 0
        for (t, n, fixed), (ps, _) in zip(members, sigs):
            gathered = [array_of([entries[j][(start + m) // every] for m in range(n)])
                        for j in range(len(ps) - fixed)]
            outs.append(B.batch(t, *args[at:at + fixed], *gathered, axes=(0,) * len(ps)))
            at, start = at + fixed, start + n
        return coll_concat(signature(S_)[1], outs)

    defn = CompositeDefinition(
        "NetworkScanPhase", 1, ("templates", "counts", "nb", "every"), signature, body, register=False,
        doc="the step-indexed scan's execution phase in the generator form: each template one batch over its proof "
            "blocks, every ingress block's prog<j> its boundary's row of lane j's slot entries")
    return Program(bind(defn, templates=tuple(P.template_id(t) for t, _, _ in members),
                        counts=tuple(n for _, n, _ in members), nb=nb, every=every))


@dataclass(frozen=True)
class Member:
    """One statement of an execution phase's proof, one Flock proof with its own sessions: its name, its template, its first unit
    ``base`` and the scan's buckets its units compute, unit ``base + k`` bucket ``buckets[k]``."""

    name: str
    template: str
    base: int
    buckets: tuple[int, ...]

    @property
    def units(self) -> tuple[int, ...]:
        return tuple(range(self.base, self.base + len(self.buckets)))


@dataclass(frozen=True)
class Tree:
    """A registered tree the scan reads: its value, its root and its domain's id (of any owner) and its leaves, as the
    registration that committed it names them."""

    value: str
    root: bytes
    domain: bytes
    leaves: int

    def __post_init__(self) -> None:
        if not (isinstance(self.value, str) and self.value) or type(self.root) is not bytes or len(self.root) != 64 \
                or type(self.domain) is not bytes or len(self.domain) != 64 or type(self.leaves) is not int \
                or self.leaves < 1:
            raise ValueError("a registered tree names its value, a 64-byte root and domain id and one or more leaves")

    @classmethod
    def of(cls, reg: G.Registered) -> Tree:
        return cls(reg.value, reg.root, reg.domain.domain_id, reg.leaves)


@dataclass(frozen=True)
class Trace:
    """The committed trace an execution phase is held to, as the compute audit's registration names it: `state`, its
    `SCAN_STATE` (T leaves, leaf t S_{t−1}), and `progs`, lane j's slot entries (nb leaves) at j."""

    state: Tree
    progs: tuple[Tree, ...]

    def __post_init__(self) -> None:
        if type(self.state) is not Tree or not isinstance(self.progs, (tuple, list)) or \
                not all(type(p) is Tree for p in self.progs):
            raise ValueError("a trace is its state tree and one slot-entries tree a lane")
        object.__setattr__(self, "progs", tuple(self.progs))


class Heads(NamedTuple):
    """The network certifier's heads of one execution phase's rows, each read off its signed registration
    (`warden_rows.check_registration`)."""

    ingress: WR.CellsHead
    egress: WR.CellsHead


@dataclass(frozen=True)
class EgressTieVerdict:
    """The tie of a link's certified egress to the committed trace (`EgressTie.tie`): `accepted[w]` on the link named
    `link`, whose execution phases the challenger registered as `registrations` (each one's `registration.digest`, in
    the phases' order). Only `EgressTie.tie` builds one, from proofs the verifier verified itself."""

    link: str
    registrations: tuple[str, ...]
    accepted: tuple[bool, ...]

    def __post_init__(self) -> None:
        regs, acc = self.registrations, self.accepted
        if type(self.link) is not str or not self.link or not isinstance(regs, tuple) or \
                not all(type(r) is str and len(r) == 128 for r in regs) or not isinstance(acc, tuple) or \
                not all(type(x) is bool for x in acc) or len(acc) != len(regs):
            raise ValueError("the egress tie's verdict names its link and one bool a registration's digest")


@dataclass(frozen=True)
class Delivered:
    """An execution phase's proof as the verifier receives it: each member's statement's proof files (`proofs.Proof`, its
    circuit file the auditor's own lowering of the member's template), by member name."""

    proofs: Mapping[str, PF.Proof]

    def sessions(self, members: Sequence[Member]) -> list[str]:
        """The delivered sessions' SHA-512s, their `session.json`s', member by member in the phase's order."""
        return [PF.file_sha512(Path(s) / "session.json") for m in members if m.name in self.proofs
                for s in self.proofs[m.name].sessions]


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class EgressTie:
    """The egress tie on one link, every field public: the link, the network certifier's schedule
    (`warden.WardenParams`, its buckets an execution phase T), the session that names its domains, the scan's class (which must
    serve the schedule, `ingress_scan.Class.serves`) and the proof deadline in seconds."""

    link: str
    params: WardenParams
    session: bytes
    cls: IS.Class
    proof_deadline: float | None = None

    def __post_init__(self) -> None:
        if type(self.params) is not WardenParams or type(self.cls) is not IS.Class or \
                not (isinstance(self.link, str) and self.link) or type(self.session) is not bytes:
            raise ValueError("an egress tie is on a named link, of a warden.WardenParams, an ingress_scan.Class and a "
                             "session")
        self.cls.serves(self.params)
        if self.T < 3:
            raise ValueError(f"an execution phase of the scan is three or more buckets (first, ingress, last), not {self.T}")

    @property
    def T(self) -> int:
        return WR.rows_of(self.params)

    @property
    def N(self) -> int:
        """A registration's proof blocks: T ingress units and T egress units."""
        return 2 * self.T

    @property
    def nb(self) -> int:
        return self.T // self.cls.EVERY

    @cached_property
    def templates(self) -> dict[str, Any]:
        return templates(self.cls)

    @property
    def units(self) -> list[Any]:
        return [self.templates[tp] for tp in TEMPLATES]

    @cached_property
    def members(self) -> tuple[Member, ...]:
        T = self.T
        return (Member("first", "first", 0, (0,)), Member("ingress", "ingress", 1, tuple(range(1, T - 1))),
                Member("last", "ingress", T - 1, (T - 1,)), Member("egress", "egress", T, tuple(range(T))))

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instances_query(self.units)

    @cached_property
    def _program(self) -> Program:
        counts = {"first": 1, "ingress": self.T - 1, "egress": self.T}
        return scan_root([(self.templates[tp], counts[tp], FIXED[tp]) for tp in TEMPLATES], self.nb, self.cls.EVERY)

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

    @property
    def lanes(self) -> range:
        return range(self.cls.NR)

    @cached_property
    def values(self) -> tuple[str, ...]:
        return (ROWS[Direction.INGRESS], ROWS[Direction.EGRESS], STATE, *(prog(j) for j in self.lanes), VIEW, LAST)

    @cached_property
    def public(self) -> tuple[S.Public, ...]:
        why = {ROWS[Direction.INGRESS]: "the network certifier's salted root over the execution phase's ingress rows, "
                                        "its signed head's",
               ROWS[Direction.EGRESS]: "the network certifier's salted root over the execution phase's egress rows, its "
                                       "signed head's",
               STATE: "the committed trace's SCAN_STATE, the compute audit's salted root",
               **{prog(j): f"lane {j}'s slot entries, the compute audit's salted root" for j in self.lanes},
               VIEW: "the prover's salted root over the scan's egress views, registered before any coin",
               LAST: "the prover's salted root over S_{T-1}, registered before any coin"}
        return (S.Public("population", "structure", "2T proof blocks per registration in four statements: block t "
                                                    "ingress bucket t (block 0 from the empty state, blocks 1 ... T-2 "
                                                    "one statement, block T-1 one), block T + t egress bucket t"),
                S.Public("window", "structure", "provenance: the execution phase's index, held to the registration's "
                                                "order"),
                *(S.Public(f"{x}:{v}", "structure", f"the {v} tree's layout") for v in self.values
                  for x in ("leaves", "domain")),
                *(S.Public(v, "commitment", why[v]) for v in self.values),
                S.Public("shape", "structure", "the scan's class and the network certifier's schedule, fixed before the "
                                               "run"),
                S.Public("verdict", "output", "the execution phase's egress tie, every proof block's, accept or reject"))

    @cached_property
    def spec(self) -> S.Spec:
        context = canonical_json({"format": SPEC, "link": self.link, "session": self.session.hex(),
                                  "params": self.params.digest.hex(),
                                  "class": {k: getattr(self.cls, k) for k in IS.STATICS}})
        check = S.Check(tuple(P.program_sha512(Program(u)) for u in self.units), self.query, public_outputs=OUTPUTS,
                        proof_deadline=self.proof_deadline)
        return S.Spec(tuple(S.Value(v, SCHEMA, "serve") for v in self.values), S.All(), check, self.public, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        return S.Stream(self.spec, self.units, program=self._program, **kwargs)

    @staticmethod
    def provenance(window: int) -> dict[str, Any]:
        return {"kind": "served", "window": window}

    # the trees

    def _held(self, heads: Heads) -> int:
        """The execution phase both heads are of, once they are the network certifier's of this link's rows under this schedule
        and session; else `ValueError`."""
        p = self.params
        if type(heads) is not Heads:
            raise ValueError("the heads are a scan_service.Heads of the two links' rows")
        for d, head in zip((Direction.INGRESS, Direction.EGRESS), heads):
            if type(head) is not WR.CellsHead or head.link != self.link or head.direction is not d or \
                    (head.parameters, head.cell_bytes, head.slots, head.rows) != \
                    (p.digest, p.cell_bytes, p.grid.allocation, self.T) or \
                    head.domain != WR.domain(p, self.session, self.link, d, head.window).domain_id:
                raise ValueError(f"the {d.value} head isn't the network certifier's of this link's rows under this "
                                 "schedule and session")
        if heads.ingress.window != heads.egress.window:
            raise ValueError("the two heads are of different execution phases")
        return heads.ingress.window

    def _traced(self, trace: Trace) -> None:
        if type(trace) is not Trace or trace.state.leaves != self.T or len(trace.progs) != self.cls.NR or \
                any(p.leaves != self.nb for p in trace.progs):
            raise ValueError(f"the trace is a state tree of T = {self.T} leaves and {self.cls.NR} slot-entries trees of "
                             f"nb = {self.nb}")

    def _domain(self, heads: Heads, kind: str) -> CommitmentDomain:
        bits, leaves = (IS.view_bits(self.cls), self.T) if kind == "view" else (IS.state_bits(self.cls), 1)
        return G.domain(self._partition["program"], value(kind), bits // 16, leaves,
                        window=heads.ingress.root.hex())

    def _commit(self, heads: Heads, kind: str, rows: Sequence[Sequence[int]],
                salts: Sequence[bytes] | None) -> G.Registered:
        self._held(heads)
        return G.Registered.commit(self._partition["program"], value(kind), [words(r) for r in rows], salts,
                                   G.WORD_BITS, SCHEMA, window=heads.ingress.root.hex())

    def view_tree(self, heads: Heads, views: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None
                  ) -> G.Registered:
        """The prover's ``scan/view``: row t S_t's egress view (its bits), under fresh salts (or `salts`), in a tree
        whose domain binds the registered Program and the ingress root."""
        if len(views) != self.T or any(len(v) != IS.view_bits(self.cls) for v in views):
            raise ValueError(f"the views are T = {self.T} rows of {IS.view_bits(self.cls)} bits")
        return self._commit(heads, "view", views, salts)

    def last_tree(self, heads: Heads, state: Sequence[int], salts: Sequence[bytes] | None = None) -> G.Registered:
        """The prover's ``scan/last``: S_{T−1}, the phase's last ingress block's o1, which no state leaf holds."""
        if len(state) != IS.state_bits(self.cls):
            raise ValueError(f"S_(T-1) is a row of {IS.state_bits(self.cls)} bits")
        return self._commit(heads, "last", [state], salts)

    def known_tree(self, heads: Heads) -> G.Registered:
        """The value the verifier registers at known content on the execution phase: row u block u's ``o0``, the word 1, under its
        public salt (`registered.supplied_salts`), in a tree whose domain binds the registered Program and the ingress
        root. The prover commits its own ``o0`` rows the same way, so they open the verifier's root only where they
        are 1."""
        program, name = self._partition["program"], f"{heads.ingress.value}/{VERIFIER}/o0"
        return G.Registered.commit(program, name, [[1]] * self.N, G.supplied_salts(program, name, self.N),
                                   G.WORD_BITS, SCHEMA, window=heads.ingress.root.hex())

    def _layout(self, heads: Heads, trace: Trace) -> dict[str, tuple[int, bytes]]:
        """Each registered port's leaves and domain id, the verifier's own."""
        return {ROWS[Direction.INGRESS]: (self.T, heads.ingress.domain), ROWS[Direction.EGRESS]: (self.T, heads.egress.domain),
                STATE: (trace.state.leaves, trace.state.domain),
                **{prog(j): (p.leaves, p.domain) for j, p in enumerate(trace.progs)},
                VIEW: (self.T, self._domain(heads, "view").domain_id), LAST: (1, self._domain(heads, "last").domain_id)}

    def ports(self, heads: Heads, trace: Trace) -> dict[str, dict[str, Any]]:
        """`stream.register`'s ports: each registered port's leaves and domain, the verifier's own."""
        return {p: {"leaves": n, "domain": d.hex()} for p, (n, d) in self._layout(heads, trace).items()}

    def _fixed(self, heads: Heads, trace: Trace) -> dict[str, bytes]:
        """The roots the registration must name: the network certifier's and the trace's."""
        return {ROWS[Direction.INGRESS]: heads.ingress.root, ROWS[Direction.EGRESS]: heads.egress.root,
                STATE: trace.state.root, **{prog(j): p.root for j, p in enumerate(trace.progs)}}

    # the challenger

    def registration(self, heads: Heads, trace: Trace, view: bytes, last: bytes,
                     previous: str | None = None) -> dict[str, Any]:
        """One-stage's record of the execution phase: the network certifier's roots (from `heads`), the trace's and the
        prover's ``view`` and ``last`` roots, and nothing of the rows. Raises `ValueError` on heads of another link,
        schedule or session, or a trace of another shape."""
        w = self._held(heads)
        self._traced(trace)
        roots = {**self._fixed(heads, trace), VIEW: view, LAST: last}
        layout = self._layout(heads, trace)
        return R.record(self._program, self.query, {"law": "subset", "k": self.N}, S.SCHEME,
                        [{"port": p, "root": roots[p].hex(), "leaves": layout[p][0], "domain": layout[p][1].hex()}
                         for p in self.values], self.provenance(w), previous=previous, partition=self._partition)

    def register(self, stream: S.Stream, record: Mapping[str, Any], heads: Heads,
                 trace: Trace) -> S.WindowCommitmentRecord:
        """`stream.register` with the phase's own ports, after holding its `window` provenance to exactly
        `{"kind": "served", "window": order}` and its network certifier's and trace roots to `heads`' and `trace`'s."""
        window = len(stream.records)
        if not isinstance(record, Mapping) or record.get("window") != self.provenance(window):
            raise S.Refused(Verdict(False, ("window",), {"window": {"expected": self.provenance(window)}}))
        self._traced(trace)
        if self._held(heads) != window or self._roots(record) is None or any(
                self._roots(record).get(p) != r for p, r in self._fixed(heads, trace).items()):
            raise S.Refused(Verdict(False, ("roots",), {"roots": "the registration's network certifier and trace roots "
                                                                 "are the phase's heads' and the trace's"}))
        return stream.register(record, self.ports(heads, trace))

    @staticmethod
    def _roots(record: Mapping[str, Any]) -> dict[str, bytes] | None:
        rows = record.get("roots") if isinstance(record, Mapping) else None
        if not (isinstance(rows, list) and all(isinstance(r, Mapping) and isinstance(r.get("port"), str)
                                               and isinstance(r.get("root"), str) and HEX.fullmatch(r["root"])
                                               for r in rows)):
            return None
        return {r["port"]: bytes.fromhex(r["root"]) for r in rows}

    # the verifier

    def registered(self, member: Member, heads: Heads, trace: Trace, view: bytes, last: bytes) -> dict[str, Any]:
        """Member `member`'s `--registered` file, the verifier's own, which names every port of its statement
        (Flock's `Registered.checkComplete`): the module docstring's reads, at the network certifier's roots of
        `heads`, the trace's, the prover's registered ``view`` and ``last`` roots and the known ``o0`` words."""
        base, bs, every = member.base, list(member.buckets), self.cls.EVERY
        layout, known = self._layout(heads, trace), self.known_tree(heads)
        at = {} if not base else {G.READS_FROM: base}

        def tree(port: str, name: str, root: bytes, bits: int, reads: Sequence[int]) -> dict[str, Any]:
            n, d = layout[port]
            e = {"leaves": n, "words": bits // 16, "word_bits": G.WORD_BITS, "domain": d.hex()}
            return produced.port(name, e, root.hex(), list(reads), SCHEMA) | at

        def rows(head: WR.CellsHead) -> dict[str, Any]:
            return {**WR.bucket_port(head, self.params, base), "reads": bs}

        sb, vb, eb = IS.state_bits(self.cls), IS.view_bits(self.cls), IS.entry_bits(self.cls)
        views = tree(VIEW, value("view"), view, vb, bs)
        o0 = produced.port(known.value, known.entry(), known.root.hex(), list(member.units), SCHEMA) | at | \
            {KNOWN_KEY: {"rows": [list(r) for r in known.rows], "salts": [s.hex() for s in known.salts]}}
        if member.template == "egress":
            return WR.reads_file({"state": views, "o0": o0, "o1": rows(heads.egress)})
        state = trace.state
        ports = {"row": rows(heads.ingress)}
        if member.template == "ingress":
            ports["sin"] = tree(STATE, state.value, state.root, sb, bs)
        for j, p in enumerate(trace.progs):
            ports[f"prog{j}"] = tree(prog(j), p.value, p.root, eb, [b // every for b in bs])
        ports["o0"] = o0
        ports["o1"] = tree(LAST, value("last"), last, sb, [0]) if member.name == "last" else \
            tree(STATE, state.value, state.root, sb, [b + 1 for b in bs])
        ports["o2"] = views
        return WR.reads_file(ports)

    def _proved(self, member: Member, proof: PF.Proof | None, circuit: Any, reads: Mapping[str, Any]) -> str | None:
        """Why the member's proof doesn't prove its units on `reads`, or None when it does."""
        if type(proof) is not PF.Proof or not isinstance(circuit, str):
            return "no proof of the member, or no circuit file of its template"
        findings, refused = PF.verify(proof, circuit_sha512=circuit, program=self.program, partition=self._partition,
                                      registered=reads)
        if refused is not None:
            return f"{refused[0]}: {refused[1]}"
        bad = next((f for f in findings or () if not f.accepted), None)
        if not findings or bad is not None:
            return bad.why if bad is not None else "no session"
        if any(f.program != self._partition["program"] for f in findings):
            return "a session names another Program than the registered one"
        if sorted(u for f in findings for u in f.units) != list(member.units):
            return f"the sessions prove units other than the member's {member.units[0]} … {member.units[-1]}"
        return None

    def verify(self, draw: S.Draw, delivered: Delivered, *, circuits: Mapping[str, str], heads: Heads,
               trace: Trace) -> dict[str, Any]:
        """The verifier's verdict on the execution phase's delivered proofs, from public inputs and the proofs alone: accepted when
        the registration's roots are `heads`' and `trace`'s, every proof block is sampled, and every member is proved
        (`_proved`: the verifier's own run of Flock's Lean verifier, `proofs.verify`, on the circuit file the auditor
        built of the member's template, `circuits[template]`, and the verifier's own registered reads, `registered`,
        accepts every session, each naming the registered Program, and they prove the member's units). `refused` names
        each member not proved and why."""
        rec = draw.registration.record
        roots, refused = self._roots(rec), {}
        try:
            held = self._held(heads) == rec["window"]["window"]
            self._traced(trace)
            layout = self._layout(heads, trace)
        except (ValueError, KeyError, TypeError):
            held, layout = False, {}
        held = held and roots is not None and all(roots.get(p) == r for p, r in self._fixed(heads, trace).items()) \
            and {r["port"]: (r["leaves"], r["domain"]) for r in rec["roots"]} == \
            {p: (n, d.hex()) for p, (n, d) in layout.items()}
        if not held:
            refused["registration"] = "the registration's roots aren't the phase's heads' and the trace's"
        elif draw.units != list(range(self.N)):
            refused["draw"] = "every proof block of the execution phase is sampled"
        else:
            for m in self.members:
                why = self._proved(m, delivered.proofs.get(m.name), circuits.get(m.template),
                                   self.registered(m, heads, trace, roots[VIEW], roots[LAST]))
                if why is not None:
                    refused[m.name] = why
        accepted = not refused
        public = {"verdict": int(accepted)}
        sessions = delivered.sessions(self.members)
        bound = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": sessions,
                         "accepted": accepted, "public": public})
        return {"of_record": True, "decided_by": "proof", "accepted": accepted, "digest": bound, "sessions": sessions,
                "public": public, "refused": refused}

    def statement(self, draw: S.Draw) -> dict[str, Any]:
        reg = draw.registration.record
        return {"format": STATEMENT, "check": list(self.spec.check.program), "link": self.link,
                "window": reg["window"]["window"], "roots": {r["port"]: r["root"] for r in reg["roots"]}}

    def outcome(self, stream: S.Stream, draw: S.Draw, delivered: Delivered, *, circuits: Mapping[str, str],
                heads: Heads, trace: Trace) -> S.Outcome:
        """`stream.outcome` on the verifier's verdict (`verify`), with the registered roots as the
        sessions'."""
        verdict = self.verify(draw, delivered, circuits=circuits, heads=heads, trace=trace)
        return stream.outcome(draw, {VERIFIER: verdict}, self.statement(draw),
                              session_roots={r["port"]: r["root"] for r in draw.registration.record["roots"]})

    def tie(self, stream: S.Stream) -> EgressTieVerdict:
        """`EgressTieVerdict` on this link and the stream's registrations (each one's `registration.digest`, in order),
        one bit a registration: phase w accepted when the stream's w-th registration is phase w's and its outcome
        accepted, decided on the proof. Raises `ValueError` on a stream that isn't this tie's."""
        if stream.spec.digest() != self.spec.digest() or stream.program is None or \
                P.program_digest(stream.program) != P.program_digest(self._program):
            raise ValueError("the stream is not this tie's: its spec isn't EgressTie.spec or its population program "
                             "isn't the phase's")
        outcomes = {r["registration"]: r for r in stream.records}
        ok = []
        for w, reg in enumerate(stream.log):
            body = outcomes.get(R.digest(reg))
            verdict = ((body or {}).get("audit") or {}).get("verdicts", {}).get(VERIFIER) or {}
            ok.append(reg.get("window") == self.provenance(w) and body is not None and body["symbol"] == S.ACCEPT
                      and verdict.get("decided_by") == "proof")
        return EgressTieVerdict(self.link, tuple(R.digest(r) for r in stream.log), tuple(ok))


@dataclass(frozen=True)
class Scans:
    """The verifier's own egress tie source on one link: the tie and its stream, the challenger's and the verifier's,
    never the prover's. `verdict` is `EgressTie.tie` on the stream, so the egress tie's verdict
    comes only from outcomes the verifier decided on proofs. It raises `ValueError` on another link than the
    tie's."""

    egress: EgressTie
    stream: S.Stream

    def __post_init__(self) -> None:
        if type(self.egress) is not EgressTie or type(self.stream) is not S.Stream or \
                self.stream.spec.digest() != self.egress.spec.digest():
            raise ValueError("an egress tie source is an EgressTie and its own stream")

    def verdict(self, link: str) -> EgressTieVerdict:
        if link != self.egress.link:
            raise ValueError(f"the egress tie source is on link {self.egress.link!r:.80}, not {link!r:.80}")
        return self.egress.tie(self.stream)


__all__ = ["Delivered", "EgressTie", "EgressTieVerdict", "FIXED", "Heads", "KNOWN_KEY", "LAST", "Member", "OUTPUTS",
           "ROWS", "STATE", "Scans", "TEMPLATES", "Trace", "Tree", "VERIFIER", "VIEW", "prog", "scan_root", "templates",
           "value", "words"]
