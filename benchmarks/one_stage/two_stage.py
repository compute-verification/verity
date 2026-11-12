"""A served window's two-stage audit, as the challenger runs it. The first stage (``first``): each size class's boundary
registration (`verity_vllm.pipeline.boundary_rows`, committed inside the engine core while the window was served) on a
two-stage `service.Stream` of its own (`stream_of`), registered, then sampled once on the auditor's challenger's coins
(`first`): the replay blocks, which are the class's spans the replay side replays. The second stage (``second``, one
class): `resume` gives the replay side the same stream at the same point; the sampled spans' proof blocks are committed
as the interior, registered after the boundary registration and sampled (`replayed`); and the one-stage audit proves the
sample, a ``--zk`` Flock session per member under Lean's verdict, round 2 checks the sessions' statements and the stream
gives its outcome (`prove`, `outcome_of`).

* **The Spec** (`spec_of`): the class's boundary values (`Layout.schemas`: ``b.ingress`` and ``b.egress``, or under a
  serve's stop rule their parts ``b.request``, ``b.prompt`` and ``b.egress``, then its history tables; phase ``serve``),
  the interior values (phase ``interior``: the check template's ports as the replay side commits them,
  `interior_values`) and the replay block's
  parameters the interior registration registers as values (`Rule.span_values`: those the verifier supplies, a table
  read's root and domain or an index, and those the replay gives, a picking unit's scores or a read's opening, each
  with its rows a sampled span and a row's leaf type, which the stream takes as its per-span values); the selection
  `TwoStage(grain, replay, k)` over the class's replay block; the check, its templates' programs and their query
  (`Layout.templates`, `Layout.query`: the check template's instances, or, inlined, the templates its body calls) with
  its proof deadline; and the public items both registrations show.
* **The stop rule** (``--stop``): the verifier's own copy of the rule the serve registered, as ``--check`` is its own
  copy of the check template. A class is laid out under it (`boundary_rows.layout_of`), and a ``commit.json`` that
  names another rule, or names one where the verifier holds none, is refused: the serve chooses neither.
* **The stop blocks** (a stop class's, `stop_first`): a population of their own, one a span, on a one-stage stream of
  the stop block (`stop_spec`) whose log starts with the class's boundary record. After the class's first stage its
  stop record (`boundary_stop.stop_record`) is registered, its inputs that record's ``b.request`` and ``b.egress`` as
  output values, and sampled once at ``--stop-design`` (a rate over the class's stop blocks; the first stage's unless
  given) on a rank of its own in the same session. The sample is the hand-off's ``stop``; no proof of a sampled stop
  block is built here. A stop class whose stop blocks cannot plan their sample refuses the window's whole first stage,
  as any rank that cannot; one whose stop blocks then register or sample nothing is refused itself (``stop-stage``):
  a stop class is audited with its stop blocks or not at all.
* **The order** (`stop_order`, once the session has closed): read from the challenger's signed log as the session
  published it, never from a prover's file. Every coin a stop sample read was released after the challenger received
  the stop registration, and the class's boundary registration's coins come before them: another family's lines of
  the same log, since a family's coins follow one commitment. A window where one stop
  class's order does not hold is refused whole, after its samples (``cross-stream-order``): no class of it hands a
  sample on, and the root's one session is spent, so nothing samples it again.
* **The coins.** `first` takes a class's `ChallengerCoins` (a test's in-process challenger). The command opens one
  challenger session on the window's run (`a5_coins.start`: ``local``, ``loopback:HOST:PORT`` or ``gate:USER@HOST``),
  each class a rank of it, sampling on a thread of its own in its turn, and each stop class's stop blocks a rank more;
  the session's evidence goes into every class's ``stage1.json``.
* **The hand-off** (``stage1.json``, a class's: its ``commit.json``, the rule and the Spec's digest, the record, its
  registration's digest and receipt, the sample, the bytes it read, the coins' segment and evidence and the window root
  its registration was held to; a stop class's ``stop``, the same of its stop record, with the order check's result).
  A `Stream` holds
  code and doesn't persist, so `resume` builds the class's stream again from them, registers the record and samples it
  again on the bytes read (`Staged`, which then reads the second stage's coins from a live session), holding each to
  ``stage1.json``: the same Spec digest, registration digest and sample.

* **The design** (`design`, from the class's layout alone, so the replay side and the verifier derive one): what the
  replay block makes of a sampled span's interior, as round 2 derives it: one member a template, the block's
  parameters its templates' ports read (links, each registered by the interior as a value under the link's name), its
  wires and its return links. Without ``--interior`` a class's rule is its design's (`rule_of`). A block round 2
  derives no links or wires from has no second stage, and `design` says so by name: an inlined token select today,
  whose block gathers each token's chunks. ``second --unlinked`` runs such a class as far as it goes: its sampled
  proof blocks committed, registered, sampled and proved with no link and no wire, each port at the interior's own
  rows. That measures the sessions and ties nothing: round 2 refuses the statements (``second-wires``), and the run
  exits 1.
* **The second stage's coins.** The first stage's session closed with its sample and a challenger opens a root once, so
  ``second`` opens a session of its own on the interior window's run (`interior_rows.interior_run` of the boundary
  registration and the sample), whose root the interior registration is held to and its replay-given values bind.
* **What the one-stage audit is handed** (`prove`): the interior window written as the audit reads one
  (`interior_rows.write`); by template, the composer and Definition (`composer`: Flock's `member_stager` on the
  window's partition and program); by member key, the reads each port is verified against (`verifier_reads`: a link's
  port the value the interior registers for it, a wired port its writer's output, an output that writes a return
  round 1 commits that port of round 1's record, any other its own); the second stage's draw, law and receipt; and the
  auditor's params (``--proofs``; without them `references` builds a development run's, after the sample).
* **The block's return.** The interior never registers the block's return, and round 2 reads it only at a port round
  1's record commits. Where the record commits one (`boundary_rows.Layout.ids`), the replay side opens that port from
  the class's files (`boundary_rows.Committed.port`) and commits the output that writes it under the serve's salts at
  the rows its units write, so each row is round 1's leaf and its header opens it at round 1's root
  (`interior_rows.interior`'s ``returns``); the verifier's reads name round 1's entry for it, unit j of the i-th
  sampled block reading row ``sample[i]·per + at[j]``, and so do the params. Where the record commits none, the
  return's port is verified at the interior's own root and round 2 refuses that read (``second-read``): the run
  reports it and exits 1. The serve's salts of those rows go to the prover's commitment alone: round 2 is handed none.

Until the class's request Definition lands, the replay block is `boundary_rows`' stand-in (one check template's
instances, or their bodies inlined, as the class's ``commit.json`` says), and so is every sample this draws. An inlined
check whose body calls several templates needs core's two-stage over a window of templates. ``second`` replays a token
select's spans from a capture of the sampler's inputs (`engine.sampler_capture`); a block whose ports read a table or a
value the verifier supplies is not its yet.

Usage::

    two_stage.py first --boundary DIR [DIR ...] --check TEMPLATE --replay rate:NUM/DEN --k K|COUNTS.json \\
        --challenger local --main FILE --out OUT [--interior FILE.json [--span-values VALUES.json]] [--advice REGISTRATION.json] \\
        [--proof-deadline S] [--stop RULE [--stop-design rate:NUM/DEN]]
    two_stage.py second --stage1 OUT/<class>/stage1.json --boundary DIR --capture CAPTURE --challenger local --main FILE \\
        --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --out OUT2 [--proofs PARAMS.json] [--rids IDS.json] [--firewall F] \\
        [--unlinked] [--stop RULE]

Each DIR a class's (``w<window>/p<P>-o<O>/``), every one of one window (``second``: one class's, with its private
files); COUNTS.json each check template's own count in place of one for all, ``{descriptor id: k_t}`` (`counts`);
FILE.json the interior values, ``[[name, schema], ...]``; VALUES.json the replay block's parameters the interior
registration registers as values, ``[[name, rows a sampled span, leaf type], ...]`` (`interior_rows.span_values`);
REGISTRATION.json the window's advice registration, whose ``binding`` the records' windows name; CAPTURE the replay's
capture of the sampled spans; PARAMS.json the auditor's params; IDS.json the capture's request id for a span's engine
id, where they differ; RULE the stop rule as JSON, ``{"ids": [...], "max_model_len": n, "outputs": T}``
(`boundary_stop.Stop.of`). ``second`` writes under OUT2 the interior window (``interior/``, its private files too), the
params (``proofs.json``, ``references/``), a5's records (``audit/``), the sessions' statements (``statements.json``),
its challenger session's records (``challenger/``), both stages' coin evidence (``coins.json``) and ``summary.json``;
it exits 0 when the second stage held (`held`). A ``local`` session's challenger reads its coins through
``sampled-proofs-verifier``, which must be built first (``lake build sampled-proofs-verifier`` in
``verity/protocols/spot/verifier/lean``): building it on first use outlasts the local gate's start.
``two_stage_small.sh`` runs both commands on a CPU node at small statics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from verity.protocols.spot.device import challenger as Ch, coins as CC
from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import audit as A, draw as D, partition as P, registered as REG
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import SpecializedDefinition
from verity.circuits.program import Program
from verity.commitments.windows import links as LK, rows as CS
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS, boundary_tables as BT, interior_rows as IR

HERE = Path(__file__).resolve().parent
FORMAT = "verity-vllm/two-stage/stage1/v0"
CHALLENGER = "two_stage.py on the auditor's challenger's coins"
STAGE1 = "stage1.json"
#: what round 2 refuses every two-stage run by while the form is off (`second_stage.FORM`)
FORM_REFUSED = "two-stage-form"
#: a window refused after its samples: a stop class's coins don't follow its registrations in the challenger's log
ORDER_REFUSED = "cross-stream-order"
#: a stop class whose stop blocks registered or sampled nothing
STOP_REFUSED = "stop-stage"
#: the salts' key of a reference statement (`references`): its rows are stand-ins, so its salts hide nothing
REFERENCE_KEY = hashlib.sha512(b"verity-vllm/two-stage/reference/v0").digest()
#: a design's counts as a record (`benchmarks/spot_figures`' ``total_cost.py counts``): its ``k`` is the rule's
COUNTS = "spot-two-stage-counts/v1"
#: seconds on the challenger's clock from the sample to the proofs: the replay, the interior registration and the second
#: stage's proofs all come inside it
PROOF_DEADLINE_S = 86400.0


def interior_values(member: CS.Member) -> tuple[tuple[str, str], ...]:
    """The interior values of a replay side that commits the check template's units as one window member
    (`ml.windows.rows`): each port's name and schema, in the member's order."""
    return tuple((member.port(p), member.schema(p)) for p in CS.port_leaves(member))


@dataclass(frozen=True)
class Rule:
    """A class's two-stage rule: the first stage's law over its spans (``replay``, a rate law ``rate:NUM/DEN[:T]``), the
    proof blocks ``k`` the second stage samples of each check template in each sampled span (one count for every
    template, or ``{descriptor id: k_t}``, `service.TwoStage.k`), the interior values the replay side commits, the
    proof deadline, and the replay block's parameters the interior registration registers as values
    (``span_values``: as round 2 names them, each ``(name, rows a sampled span, leaf type)``, the stream's per-span
    values, `interior_rows.span_values`: those the verifier supplies and those the replay gives; none where a rule's
    JSON leaves them out). A fixed count ``subset:Q`` is refused: the serve sizes a class's N, and Q/N falls as it pads
    N."""

    replay: str
    k: int | Mapping[str, int]
    interior: tuple[tuple[str, str], ...]
    proof_deadline: float = PROOF_DEADLINE_S
    span_values: tuple[tuple[str, int, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if D.parse_rate(self.replay) is None:
            raise ValueError("a two-stage rule's replay law is a rate law, rate:NUM/DEN or rate:NUM/DEN:T, "
                             f"not {self.replay!r}")

    def to_json(self) -> dict[str, Any]:
        return {"replay": self.replay, "k": dict(self.k) if isinstance(self.k, Mapping) else self.k,
                "interior": [list(v) for v in self.interior],
                "proof_deadline": self.proof_deadline,
                "span_values": [[n, rows, list(leaf)] for n, rows, leaf in self.span_values]}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Rule:
        k = d["k"]
        return cls(d["replay"], {str(t): int(n) for t, n in k.items()} if isinstance(k, Mapping) else int(k),
                   tuple((str(n), str(s)) for n, s in d["interior"]), float(d["proof_deadline"]),
                   _span_values(d.get("span_values", ())))


def _span_values(values: Any) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    return tuple((str(n), int(rows), tuple(int(x) for x in leaf)) for n, rows, leaf in values)


def counts(text: str) -> int | dict[str, int]:
    """``--k``'s value: a whole number, the count of every check template, or a JSON file of each one's,
    ``{descriptor id: k_t}`` (`service.TwoStage.k`), alone or as the ``k`` of a counts record (`COUNTS`). ValueError
    for a file that names a template twice or gives one a count that is no whole number of at least one."""
    if text.isdecimal():
        return int(text)

    def once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        twice = sorted({n for i, (n, _) in enumerate(pairs) if any(n == m for m, _ in pairs[:i])})
        if twice:
            raise ValueError(f"{text} names {', '.join(twice)} twice: a counts file names each of the check's "
                             "templates once")
        return dict(pairs)
    doc = json.loads(Path(text).read_text(), object_pairs_hook=once)
    if isinstance(doc, dict) and "schema" in doc:
        if doc["schema"] != COUNTS:
            raise ValueError(f"{text} is a record of schema {doc['schema']!r}, not a {COUNTS} one")
        doc = doc.get("k")
    if not (isinstance(doc, dict) and doc):
        raise ValueError(f"{text} is no counts file: {{descriptor id: k_t}}, a count for each of the check's templates")
    bad = {t: k for t, k in doc.items() if not (type(k) is int and k >= 1)}
    if bad:
        raise ValueError(f"{text}: a template's count is a whole number of at least one, not {bad}")
    return doc


def counts_refused(k: int | Mapping[str, int], templates: Sequence[SpecializedDefinition]) -> str | None:
    """Why counts by template aren't a check's of ``templates``, which they name each once (`service.second_counts`
    refuses them): the templates they miss and the names that are none of its templates. None for one count, which is
    every template's."""
    if not isinstance(k, Mapping):
        return None
    ids = [P.template_id(t) for t in templates]
    missing, other = [t for t in ids if t not in k], [t for t in k if t not in ids]
    if not (missing or other):
        return None
    return "the counts name each of the check's templates once: " + "; ".join(
        [f"missing {t}" for t in missing] + [f"{t} is no template of the check" for t in other])


def spec_of(layout: BR.Layout, rule: Rule) -> S.Spec:
    served = layout.schemas()
    linked = [n for n, _rows, _leaf in rule.span_values]
    names = [*served, *(n for n, _ in rule.interior)]
    values = (tuple(S.Value(p, s, "serve") for p, s in served.items())
              + tuple(S.Value(n, s, "interior") for n, s in rule.interior)
              + tuple(S.Value(n, None, "interior", registered=True) for n in linked))
    public = (S.Public("population", "structure", "the class's spans N; then the sampled spans' proof blocks Q n_rho"),
              S.Public("window", "structure", "provenance: the serve's run, the window, the class, the check, the index's leaf"),
              S.Public("leaf_layer", "structure", "the interior registration's public bodies' digest"),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}, from the class and its spans") for v in names
                for k in ("leaves", "domain")),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}: a replay-block parameter's rows, from the sample")
                for v in linked for k in ("leaves", "words", "domain")))
    digests = tuple(P.program_digest(Program(t)) for t in layout.templates)
    window = len(digests) > 1 or bool(linked)
    check = S.Check(digests if window else digests[0],
                    P.template_instances_query(layout.templates) if window else layout.query,
                    proof_deadline=rule.proof_deadline)
    return S.Spec(values, S.TwoStage(layout.grain, rule.replay, rule.k), check, public, f"{BR.FORMAT}\0{layout.run}".encode())


def stream_of(layout: BR.Layout, rule: Rule, coins: CC.ChallengerCoins, *, delta: float = 0.5, **kwargs: Any) -> S.Stream:
    """The class's two-stage stream on ``coins``: its units the check's templates (`Layout.templates`), its replay block
    the layout's, its per-span values the rule's (a window of its templates then, even of one)."""
    ts = layout.templates
    return S.Stream(spec_of(layout, rule), ts[0] if len(ts) == 1 and not rule.span_values else ts, budget=1, delta=delta,
                    challenger=CHALLENGER, coins=coins, replay=layout.replay, span_values=rule.span_values, **kwargs)


class Recorded(CC.ChallengerCoins):
    """``coins`` (a `ChallengerCoins`) with every byte read kept (`read`): what `resume` samples again."""

    def __init__(self, coins: CC.ChallengerCoins) -> None:
        self.coins, self.read = coins, bytearray()

    @property
    def label(self) -> str | None:
        return self.coins.label

    def commit(self, after: bytes) -> Any:
        return self.coins.commit(after)

    def __call__(self, n: int) -> bytes:
        b = self.coins(n)
        self.read.extend(b)
        return b


class Staged(CC.ChallengerCoins):
    """The first stage's bytes read again, in order, as coins of the family the first stage read them from
    (``family``, set by `resume`); then, once committed (`commit`, to the interior registration's digest), ``then``'s:
    the second stage's live coins, of its own family."""

    family: str | None = None

    def __init__(self, read: bytes, then: CC.ChallengerCoins | None = None) -> None:
        self._read, self._at, self.then, self.live = bytes(read), 0, then, False

    @property
    def label(self) -> str | None:
        return self.then.label if self.live else self.family

    def commit(self, after: bytes) -> Any:
        if self._at != len(self._read):
            raise Ch.CoinError(f"the second stage's coins follow the first stage's {len(self._read)} bytes, of which "
                               f"{self._at} were read again")
        if self.then is None:
            raise Ch.CoinError("no live coins for the second stage")
        self.live = True
        return self.then.commit(after)

    def __call__(self, n: int) -> bytes:
        if self.live:
            return self.then(n)
        if self._at + n > len(self._read):
            raise Ch.CoinError("the first stage read again reads past its bytes")
        self._at += n
        return self._read[self._at - n:self._at]


@dataclass
class First:
    """A class's first stage: its layout and rule, the stream, the record, its registration and the sample."""

    pub: dict[str, Any]
    rule: Rule
    layout: BR.Layout
    stream: S.Stream
    record: dict[str, Any]
    registration: S.WindowCommitmentRecord
    sample: S.Draw
    coins: CC.ChallengerCoins
    read: bytes = b""
    extra: dict[str, Any] = field(default_factory=dict)

    def stage1(self) -> dict[str, Any]:
        """``stage1.json``: what `resume` and the replay side read."""
        return {"format": FORMAT, "commit": self.pub, "rule": self.rule.to_json(), "spec": self.stream.spec.digest(),
                "record": self.record, "registration": self.registration.digest, "receipt": dict(self.registration.receipt),
                "draw": self.sample.draw, "units": list(self.sample.units), "read": self.read.hex(),
                "family": self.stream.family(self.sample), **self.extra}

    def second(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]],
               **kwargs: Any) -> tuple[S.WindowCommitmentRecord, S.Draw]:
        """The replay side's interior registration over the sampled spans' proof blocks (`Stream.register_interiors`,
        taking its copies as it does), the coins committed to its digest, and the second stage's sample: ``k`` proof
        blocks in each sampled span."""
        ireg = self.stream.register_interiors(self.sample, record, ports, **kwargs)
        self.coins.commit(bytes.fromhex(ireg.digest))
        return ireg, self.stream.select(ireg)

    def history(self, layer: int, published: Mapping[str, Sequence[str]],
                bodies: Mapping[str, bytes]) -> dict[str, Any]:
        """The challenger's `history` for an interior whose reads open layer ``layer``'s tables (`interior_rows.history`,
        `second`'s ``history``): each sampled span's table named from this stage's own sample, under the class record's
        root for it, of its class's C strings, the prover's ``published`` ones; ``bodies`` the interior's released public
        bodies by member key."""
        L = self.layout
        names = [BT.name(L.run, L.prompt, L.outputs, int(u), layer) for u in self.sample.units]
        return IR.history(self.record, names, BT.WORDS, BT.class_rows(L.prompt, L.outputs, L.capacity), published, bodies)


def first(pub: Mapping[str, Any], check: SpecializedDefinition, rule: Rule, coins: CC.ChallengerCoins, *,
          previous: str | None = None, window: Mapping[str, Any] | None = None, delta: float = 0.5,
          window_root: str | None = None, stop: BS.Stop | None = None, **kwargs: Any) -> First:
    """One class's first stage on ``coins``, laid out under the verifier's own check template and stop rule (``stop``):
    its stream registers the boundary record (`boundary_rows.record`, at the rule's first-stage rate over its spans,
    chained after ``previous``, its window naming ``window``'s items), the coins commit to the registration's digest,
    and the stream samples once. Raises `service.Refused` as the stream does."""
    layout = BR.layout_of(pub, check, stop)
    coins = coins if isinstance(coins, Recorded) else Recorded(coins)
    stream = stream_of(layout, rule, coins, delta=delta, **kwargs)
    rec = BR.record(pub, check, S.selection_law(stream.spec.selection, layout.spans), previous, window, stop=stop)
    reg = stream.register(rec, layout.ports(), **({} if window_root is None else {"window_root": window_root}))
    coins.commit(bytes.fromhex(reg.digest))
    sample = stream.select(reg)
    return First(dict(pub), rule, layout, stream, rec, reg, sample, coins, bytes(coins.read))


def resume(stage1: Mapping[str, Any], check: SpecializedDefinition, then: CC.ChallengerCoins | None = None, *,
           delta: float = 0.5, window_root: str | None = None, stop: BS.Stop | None = None, **kwargs: Any) -> First:
    """A class's stream again at the end of its first stage, from its ``stage1.json``, under the verifier's own check
    template and stop rule: the record registered again and sampled again on the bytes the sample read, then on ``then``
    once the interior registration's digest is committed (`Staged.commit`). ValueError when the Spec, the registration
    or the sample isn't ``stage1.json``'s."""
    if stage1.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} hand-off")
    pub, rule = stage1["commit"], Rule.from_json(stage1["rule"])
    layout = BR.layout_of(pub, check, stop)
    coins = Staged(bytes.fromhex(stage1["read"]), then)
    stream = stream_of(layout, rule, coins, delta=delta, **kwargs)
    coins.family = stage1.get("family")
    if stream.spec.digest() != stage1["spec"]:
        raise ValueError("the class's Spec isn't the one its first stage ran under")
    reg = stream.register(stage1["record"], layout.ports(), **({} if window_root is None else {"window_root": window_root}))
    if reg.digest != stage1["registration"]:
        raise ValueError("the record registers to another digest than its first stage's")
    sample = stream.select(reg)
    if list(sample.units) != list(stage1["units"]) or sample.draw != stage1["draw"]:
        raise ValueError("the first stage samples other spans read again than it did")
    return First(dict(pub), rule, layout, stream, dict(stage1["record"]), reg, sample, coins,
                 bytes.fromhex(stage1["read"]))


def stop_spec(layout: BR.Layout, design: str, proof_deadline: float = PROOF_DEADLINE_S) -> S.Spec:
    """A stop class's stop population's Spec: its one committed value `b.stop`, the stop blocks' outputs; the block's
    parameters (`boundary_stop.STOP_READS`) as registered values, the class's boundary record's rows; the one-stage
    sampling design ``design``, a rate over the class's stop blocks; and the check, the verifier's stop block's program
    and its instances."""
    block = layout.stop.template
    public = (S.Public("population", "structure", "the stop class's spans N, one stop block each"),
              S.Public("window", "structure", "provenance: the class's boundary record's window, named the stop population's"),
              *(S.Public(f"{k}:{BR.STOP}", "structure", f"{BR.STOP}'s {k}, from the class and its spans")
                for k in ("leaves", "domain")),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}: the boundary record's {port} rows, from the class and its spans")
                for v, port in BS.STOP_READS.items() for k in ("leaves", "words", "domain", "word_bits")))
    values = (S.Value(BR.STOP, BR.SCHEMA, "serve"), *(S.Value(v, None, "serve", registered=True) for v in BS.STOP_READS))
    check = S.Check(P.program_digest(Program(block)), P.template_instance_query(block), proof_deadline=proof_deadline)
    return S.Spec(values, S.Law(design), check, public, f"{BR.FORMAT}\0{layout.run}\0{BR.STOP}".encode())


def stop_first(pub: Mapping[str, Any], check: SpecializedDefinition, stop: BS.Stop, design: str,
               coins: CC.ChallengerCoins, boundary: Mapping[str, Any], *, delta: float = 0.5,
               window_root: str | None = None, proof_deadline: float = PROOF_DEADLINE_S) -> dict[str, Any]:
    """A stop class's stop blocks, registered and sampled once on ``coins`` after the class's first stage: a one-stage
    stream of the verifier's stop block (`stop_spec`) whose log starts with the class's boundary record ``boundary``,
    as the challenger logged it; the stop record (`boundary_stop.stop_record`, at ``design`` over the class's spans)
    registered on it, its inputs that record's ports as output values; the coins committed to the registration's
    digest; and the sample. The hand-off's ``stop``: the Spec's digest, the sampling design, the record, its
    registration's digest and receipt, the sample and the bytes it read. Raises `service.Refused` as the stream does,
    and ValueError where the class is no stop class of ``stop`` or ``boundary`` isn't its record."""
    layout = BR.layout_of(pub, check, stop)
    coins = coins if isinstance(coins, Recorded) else Recorded(coins)
    stream = S.Stream(stop_spec(layout, design, proof_deadline), stop.template, budget=1, delta=delta,
                      challenger=CHALLENGER, coins=coins, log=[boundary])
    held = BS.stop_registration(pub, check, stop, boundary)
    rec = BS.stop_record(pub, check, stop, S.selection_law(stream.spec.selection, layout.spans), boundary)
    reg = stream.register(rec, layout.stop_ports(), registered=held.values, outputs=held.outputs, schemas=held.schemas,
                          **({} if window_root is None else {"window_root": window_root}))
    coins.commit(bytes.fromhex(reg.digest))
    sample = stream.select(reg)
    return {"spec": stream.spec.digest(), "design": design, "record": rec, "registration": reg.digest,
            "receipt": dict(reg.receipt), "sample": sample.draw, "units": list(sample.units),
            "read": bytes(coins.read).hex(), "family": stream.family(sample)}


def stop_order(evidence: Mapping[str, Any] | None, window_root: str, family: str | None, boundary: str,
               registration: str, read: bytes, verify: Callable[..., bool]) -> str | None:
    """Why a stop class's sample doesn't hold the order its stop record rests on, or None when it does. The order is
    read from the challenger's signed log as its session published it at the close (``evidence``, the stop sample's
    `a5_coins.evidence`: the session's files under the verifier's challenger key, the auditor's ledger and the host's
    pins, `challenger.trusted_session_of`), never from a prover's file. That log holds, for each coin of each family
    in the order released, the root the challenger had received when it released it. It holds:

    * every coin of the stop sample's family ``family`` (the stop stream's own, `stop_first`'s hand-off's) is one the
      log shows released after the stop registration ``registration`` (its digest, hex), and the sample read all of
      those and no other (`coins.coins_reason` on the bytes ``read``): no coin that sampled a stop block was released
      before the challenger received what fixes `b.stop` and the rows the blocks read;
    * the log shows a coin released after the class's boundary registration ``boundary``, and the last of those comes
      before the first released after the stop registration: the record whose ports the stop record takes as output
      values was received first. A family's coins follow one commitment, so the boundary registration's coins are
      another family's, and the order is the log's own, over every family's lines.

    The stop population's stream is the one reader of `b.stop` in the session: a class's own stream commits and reads
    its `Layout.values`, which never hold it. So the stop sample's coins are every reader's."""
    if evidence is None:
        return "the challenger's session published no signed log to read the order from"
    if family is None:
        return "the stop sample read no coins of a challenger's family"
    after, before = bytes.fromhex(registration), bytes.fromhex(boundary)
    why = CC.coins_reason(evidence, window_root, family, after, read, verify)
    if why is not None:
        return f"the stop sample's coins: {why}"
    try:
        Ch.trusted_session_of(evidence["ledger"], {"root": window_root}, Ch.ChallengerKey.from_json(evidence["key"]),
                              evidence["pins"], evidence["main"], evidence["session"], verify)
    except Ch.CoinError as e:
        return str(e)
    received = [bytes.fromhex(e["root"]) for e in map(json.loads, evidence["session"]["log"][1:])
                if e.get("event") == "root"]
    first_, then = ([i for i, r in enumerate(received) if r == x] for x in (before, after))
    for name, coins in (("class's boundary", first_), ("stop", then)):
        if not coins:
            return (f"the challenger's log shows no coin released after the {name} registration: its receipt is not in "
                    f"the log")
    if first_[-1] >= then[0]:
        return (f"the challenger's log shows coin {then[0]} released after the stop registration and coin {first_[-1]} "
                f"after the class's boundary registration: the boundary registration's coins come first")
    return None


def template_fn(t: str) -> SpecializedDefinition:
    return BR.check_template(t)


@dataclass(frozen=True)
class Design:
    """What a class's replay block makes of a sampled span's interior, from the class's layout alone, so the replay side
    and the verifier derive the same one: the block's descriptor id ``key``; one unit of each of the check's templates
    as an interior window lays it out (``members``, `interior_rows.member`) and each one's proof blocks in a replay
    block (``cells``, the stream's); the block's parameters its templates' ports read, as the interior registers their
    values (``links``, `interior_rows.links`); its wires (`second_stage.replay_wires`); its returns (``returns``,
    `interior_rows.returns`: a template's output port, the name round 2 gives the block's return it writes, the row
    each unit writes, and whether round 1's record commits it); and the rows its links' and wires' ports read
    (``rows``, `second_stage.replay_rows`), which say which of a writer's units a block each unit of a wired port
    reads. The interior never registers a return: round 2 reads one only at a port round 1's record commits."""

    key: str
    members: tuple[CS.Member, ...]
    cells: tuple[int, ...]
    links: tuple[IR.Link, ...]
    wires: tuple[SS.Wire, ...]
    returns: tuple[IR.Return, ...]
    rows: tuple[SS.Rows, ...] = ()

    def window(self, boundary: str, sample: Sequence[int]) -> CS.Binding:
        """The interior window of the replay sample ``sample`` after the boundary registration ``boundary``
        (`interior_rows.interior_window`): public items alone."""
        return IR.interior_window(self.members, self.cells, boundary, sample, self.wires,
                                  IR.link_values(self.links, len(sample)), self.rows)


def design(layout: BR.Layout, record: Mapping[str, Any] | None = None, *, unlinked: bool = False) -> Design:
    """The class's `Design`, its links named as round 1's record ``record`` names its ports (the layout's own ports
    where none is given). ValueError where round 2 derives no links or no wires from the replay block (its
    ``second-links`` and ``second-wires``): such a class has no second stage to run. ``unlinked``: such a class's
    members and cells alone, with no link and no wire: its sampled proof blocks can be committed and proved, each port
    at its own rows, but nothing ties one to another or to round 1, and round 2 refuses its statements by that code."""
    key = descriptor_id(layout.replay)
    record = {"roots": [{"port": p} for p in layout.ports()]} if record is None else record
    members = tuple(IR.member(P.template_id(t)) for t in layout.templates)
    cells = S.replay_cells(Program(layout.replay), P.template_instances_query(layout.templates))
    wires = SS.replay_wires(layout.program, key)
    if wires is None and unlinked:
        return Design(key, members, cells, (), (), ())
    if wires is None:
        raise ValueError(f"replay block {key} makes no wires round 2 derives (`second_stage.replay_wires`): "
                         f"{SS.WIRES_WHY}")
    links = IR.links(layout.program, key, record, members, cells)
    return Design(key, members, cells, links, tuple(wires), IR.returns(layout.program, key, record, members, cells),
                  SS.replay_rows(layout.program, key))


def rule_of(layout: BR.Layout, replay: str, k: int | Mapping[str, int],
            proof_deadline: float = PROOF_DEADLINE_S) -> Rule:
    """The class's rule from its `design`: the interior values the replay side commits (its templates' ports as an
    interior window lays them out) and the replay block's parameters its interior registers as values
    (`interior_rows.span_values`, sized by the class: its rows a sampled span). A class round 2 derives no wires from
    has its templates' ports alone (`design`'s ``unlinked``): its first stage samples as any class's does."""
    d = design(layout, unlinked=True)
    w = d.window("00", [0])
    return Rule(replay, k, tuple(v for m in w.members for v in interior_values(m)), proof_deadline,
                IR.span_values(d.links))


def second_need(stream: S.Stream, q: int) -> int:
    """The most coins the second stage's sample of ``q`` replay blocks reads, known before its session opens
    (`a5_coins.need`'s rule): its cells' draws on a source of zeros, with the widest read again."""
    reads: list[int] = []

    def zeros(k: int) -> bytes:
        reads.append(k)
        return bytes(k)
    S.select_second(list(range(q)), stream.cells, stream.counts, zeros)
    return -(-(sum(reads) + max(reads, default=0)) // Ch.COIN_BYTES)


def spans_of(c: BR.Committed, capture: Path, sample: Sequence[int],
             rids: Mapping[str, str] | None = None) -> tuple[list[tuple[IR.Span, ...]], dict[str, str]]:
    """Each sampled span's proof blocks from the replay's capture of the sampler's inputs (``capture``,
    `engine.sampler_capture.write`), as the class's replay block lays a span out: the check template's instances whole
    (`interior_rows.whole`) or, inlined, its body's units (`interior_rows.pieces`). The capture's tokens must be the
    ids the serve committed for the span (`interior_rows.captured`), or the replay didn't reproduce it and nothing is
    committed. ``rids``: the capture's request id for a span's engine id, where the replay named its requests otherwise.
    Every span is a span of the class's one replay Definition, whatever launch the replay ran it from; -> the spans, and
    each request's captured launch."""
    L, per, launches = c.layout, [], {}
    for u in sample:
        b = c.spans[int(u)]
        rid = (rids or {}).get(b.engine_id, b.engine_id)
        try:
            s = IR.captured(capture, rid, b.served)
        except KeyError:
            raise ValueError(f"the capture holds no span of request {rid!r}, the class's sampled span {u}") from None
        if s.member.template != P.template_id(L.check):
            raise ValueError(f"the capture's units are {s.member.template}'s, not the class's check {P.template_id(L.check)}'s")
        launches[rid] = s.program
        s = replace(s, program=L.partition["program"])
        per.append(IR.pieces(s) if L.inline else (IR.whole(s),))
    return per, launches


@dataclass
class Replayed:
    """A class's interior, registered on its stream and sampled: its first stage (`resume`), the class's design, the
    interior and the directory its window is written in, the record registered (the written one), its registration,
    the second stage's sample, the challenger's copies the registration was held to (``tie``) and each replayed
    request's captured launch."""

    first: First
    design: Design
    interior: IR.Interior
    directory: Path
    record: dict[str, Any]
    registration: S.WindowCommitmentRecord
    draw: S.Draw
    tie: dict[str, Any]
    launches: dict[str, str]

    @property
    def sample(self) -> list[int]:
        return [int(u) for u in self.first.sample.units]


def replayed(again: First, c: BR.Committed, capture: Path, directory: Path, *, window_root: str | None = None,
             rids: Mapping[str, str] | None = None, salt_key: bytes | None = None, unlinked: bool = False) -> Replayed:
    """The replay side's half of the second stage, on the class's stream at the end of its first stage (``again``,
    `resume`): the sampled spans' proof blocks from ``capture`` (`spans_of`), committed as the interior
    (`interior_rows.interior`: each replay-block parameter a port reads registered as a value, under ``window_root``;
    each output that writes a return round 1's record commits under the serve's salts at round 1's rows, opened from
    ``c``), written under ``directory`` as the one-stage audit reads a window (`interior_rows.write`), registered on the stream
    after the boundary registration, and sampled: ``k`` proof blocks in each (replay block, template) cell, on the
    stream's coins committed to the registration's digest. ``c``: the class's commitment, reopened from its files
    (`boundary_rows.reopen`), whose spans say which requests were sampled and what the serve emitted for them, and
    whose salts open the return's rows.
    ``window_root``: the open root of the challenger session the second stage's coins come from. ``salt_key``: tests
    only. ``unlinked``: a class round 2 derives no wires from is committed with no link and no wire (`design`).
    ValueError where the class has no second stage (`design`), the capture isn't the sampled spans', or a replayed
    unit returns another row than round 1 committed for it. `service.Refused` as the stream refuses the registration."""
    if {k: v for k, v in c.public().items() if k != "seconds"} != {k: v for k, v in again.pub.items() if k != "seconds"}:
        raise ValueError("the class's files are not the commitment its first stage registered")
    d = design(again.layout, again.record, unlinked=unlinked)
    sample = [int(u) for u in again.sample.units]
    per, launches = spans_of(c, capture, sample, rids)
    it = IR.interior(per, d.cells, again.registration.digest, sample, row=again.pub["window"], links=d.links,
                     wires=d.wires, rows=d.rows, served=again.record, window_root=window_root, salt_key=salt_key,
                     returns=[(x, c.port(x.value)) for x in d.returns if x.held])
    record = IR.write(it, directory)
    tie = {**({"registered": IR.link_words(d.links), "schemas": IR.link_schemas(d.members, d.links)} if d.links else {}),
           **({} if window_root is None else {"window_root": window_root})}
    ireg, draw = again.second(record, it.ports, leaf_layer=it.leaf_layer, **tie)
    return Replayed(again, d, it, Path(directory), record, ireg, draw, tie, launches)


def verifier_reads(d: Design, window: CS.Binding, interior: Mapping[str, Any], boundary: Mapping[str, Any],
                   sample: Sequence[int]) -> dict[str, dict[str, Any]]:
    """The verifier's own ``--registered`` file of each member (by member key), from the two registrations' records
    (``interior`` and ``boundary``), the class's design, the verifier's window and its own replay sample alone: a port
    a link names reads the value the interior registers under the link's name, at the root, leaves and domain the
    record gives it, unit j of sampled span s its row ``s·per + at[j]``; an output that writes a return round 1's
    record commits reads that port of ``boundary``, at the root, leaves and domain that record gives it, unit j of the
    i-th sampled block its row ``sample[i]·per + at[j]`` (`interior_rows.Return.rows`); a wired port its writer's
    output port of the interior record, and any other its own, each unit its own leaf (`rows.Binding.held`). A return
    round 1's record doesn't commit is read at its own port too: round 2 refuses that read (``second-read``), and
    nothing here stands in for the port."""
    values = {e["value"]: e for e in interior.get("registered", ())}
    linked = {(x.template, x.port): x for x in d.links}
    back = {(x.template, x.port): x for x in d.returns if x.held}
    q, out = window.population // sum(d.cells), {}
    if len(sample) != q:
        raise ValueError(f"a replay sample of {len(sample)} for a window of {q} sampled spans")
    for m in window.members:
        ports = {}
        for p, words in m.all_rows:
            x = linked.get((m.template, p))
            if (m.template, p) in back:
                y = back[m.template, p]
                ports[p] = LK.reads(boundary, y.value, words, m.schema(p), y.rows(sample), m.base, registered=REG)
                continue
            if x is None:
                ports[p] = LK.reads(interior, window.held(m, p), words, m.schema(p), window.held_leaves(m, p), m.base,
                                    registered=REG)
                continue
            if x.value not in values:
                raise ValueError(f"{m.template}'s {p!r} reads {x.value}, which the interior registration doesn't register")
            e = values[x.value]
            ports[p] = {"value": x.value, "root": e["root"], "leaves": int(e["leaves"]), "words": int(words),
                        "word_bits": REG.WORD_BITS, "domain": e["domain"], REG.SCHEMA_KEY: m.schema(p),
                        "reads": [s * x.per + a for s in range(q) for a in x.at], REG.READS_FROM: m.base}
        out[m.key] = LK.reads_file(ports, registered=REG)
    return out


def references(d: Design, q: int, out: Path) -> dict[str, Any]:
    """Params for a development run (`service.template_circuits`' ``proofs``): by template, in the check's order, its
    circuit file's SHA-512 and its reference statement's header and registered reads. The reference is
    `member_stager`'s statement of the template's member in the interior window of ``q`` replay blocks (`Design.window`,
    under a run of its own, whose fields round 2 places or blanks), bound to that window's partition and program as the
    sessions' circuits are, on zero rows, salted under `REFERENCE_KEY`: its public file is ``out``/m<k>/pub.bin, its
    ``circuit.txt`` beside it, and its prover file is deleted. Its registered reads name none of the run's fields: each
    port the value it reads (`verifier_reads`: a link's value, a return's that round 1 commits, else the port holding
    it), its words and schema. Both are
    a function of the class's layout and ``q`` alone, so an auditor fixes them before the run, one a sample size its
    first-stage law can draw; this command builds them after the sample, which is why its run is a development run."""
    from verity_flock.class_statement import member_stager

    window = d.window("00" * 64, range(q))
    bind = {"partition": window.partition_digest, "program": window.partition["program"]}
    linked = {(x.template, x.port): x.value for x in (*d.links, *(x for x in d.returns if x.held))}
    named = []
    for k, m in enumerate(window.members):
        at, rows = Path(out) / f"m{k}", {r: np.zeros((m.n, w), dtype=np.uint16) for r, w in m.rows}
        member_stager(template_fn(m.template), m, bind).stage(
            rows, at, REFERENCE_KEY, indices=range(m.base, m.base + m.n),
            tables={r: np.zeros((dict(m.tables)[r], w), dtype=np.uint16) for r, w in m.rows},
            refs={r: np.asarray(window.held_leaves(m, r)) for r in rows})
        (at / "inst.bin").unlink()
        with open(at / "pub.bin", "rb") as f:
            header = json.loads(f.readline())
        reads = LK.reads_file({p: {"value": linked.get((m.template, p), window.held(m, p)), "words": int(words),
                                   "word_bits": REG.WORD_BITS, REG.SCHEMA_KEY: m.schema(p)}
                               for p, words in m.all_rows}, registered=REG)
        named.append({"template": m.template, "circuit": header["circuit_sha512"], "header": header, "registered": reads})
    return {"templates": named}


def composer(defn: SpecializedDefinition, member: CS.Member) -> Callable[[Path, str, str], dict[str, Any]]:
    """``a5.audit``'s composer of a word-row member's template circuit: Flock's `member_stager`, bound to the window's
    partition and program."""
    def compose(circuit: Path, partition: str, program: str) -> dict[str, Any]:
        from verity_flock.class_statement import member_stager

        st = member_stager(defn, member, {"partition": partition, "program": program}, circuit)
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}
    return compose


def interior_source(window: CS.Binding, sample: Sequence[int]) -> Callable[[dict], dict]:
    """The verifier's check of a served interior registration's window: served, under the verifier's own interior run
    (`interior_rows.interior_run` of the boundary registration and the sample), an interior naming the sample in order."""
    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "served", "run": w.get("run") == window.run, "interior": w.get("interior") is True,
              "replay_sample": w.get("replay_sample") == [int(u) for u in sample]}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v],
                "detail": {"kind": w.get("kind"), "run": w.get("run")}}
    return source


def statements(audit_dir: Path, window: CS.Binding) -> list[dict]:
    """Each member session's statement as round 2 takes it (``{"template", "header", "draw", "registered"}``): its
    template, its public file's header, its session record's ``unit_draw`` and the reads it was verified against. A
    member the second stage drew nothing of has no session and no statement."""
    import a0

    out = []
    for k, m in enumerate(window.members):
        st, rec = audit_dir / f"m{k}", next((audit_dir / f"session-m{k}" / "sessions").glob("*/session.json"), None)
        if rec is None:
            continue
        reads = st / "registered.json"
        out.append({"template": m.template, "header": a0.public_header(st, m.n),
                    "draw": json.loads(rec.read_text()).get("unit_draw"),
                    "registered": json.loads(reads.read_text()) if reads.exists() else None})
    return out


def pins(proofs: Mapping[str, Any], stmts: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Each session's circuit file against the one the auditor's params ``proofs`` name (`member_stager`'s file for the
    template, built afresh from the template and its member alone, bound to the interior's partition and program): by
    template, whether the SHA-512 its header names (``circuit_sha512``) is the params'."""
    named = {e["template"]: e["circuit"] for e in proofs["templates"]}
    return {x["template"]: named.get(x["template"]) == x["header"].get("circuit_sha512") for x in stmts}


def round_two(replay: Program, key: str, sample: Sequence[int], window: CS.Binding, irec: Mapping[str, Any],
              brec: Mapping[str, Any], drawn: Sequence[int], stmts: Sequence[Mapping[str, Any]],
              record: Mapping[str, Any], proofs: Mapping[str, Any]) -> dict[str, Any]:
    """Round 2's statement check of the sessions (`second_stage.check`, the form allowed): over the links and wires the
    replay block ``replay`` defines under descriptor id ``key`` makes (`second_stage.replay_links`, at the boundary
    record's ports, and `second_stage.replay_wires`, as the stream derives them), their rows
    (`second_stage.replay_rows`) and the names of what the block returns (`second_stage.replay_returns`), the reads
    the audit record's verdicts of record report, what they name of each session they accepted
    (`second_stage.verified_of`), and each template's reference header and registered reads the params ``proofs``
    give."""
    verdicts = record.get("verdicts") or {}
    own = SS.Own(tuple(sample), SS.parts(len(sample), P.strata(window.program, window.query)), irec, brec, tuple(drawn),
                 SS.replay_links(replay, key, brec), A.session_registered(verdicts),
                 verified=SS.verified_of(verdicts),
                 refs=tuple((e["template"], e["header"]) for e in proofs["templates"]),
                 ref_reads=tuple((e["template"], e["registered"]) for e in proofs["templates"]),
                 wires=SS.replay_wires(replay, key), rows=SS.replay_rows(replay, key),
                 returns=SS.replay_returns(replay, key, brec))
    return SS.check(True, own, stmts).to_json()


def second_stage(a: argparse.Namespace, sets: dict, window: CS.Binding, window_dir: Path, source: Callable[[dict], dict],
                 brec: dict, irec: dict, sample: list[int], second: tuple[dict, dict, dict], reads: dict[str, dict],
                 proofs: dict, out: Path, provenance: dict, *, replay: tuple[Program, str],
                 tie: Mapping[str, Any] | None = None) -> tuple[dict, bool, dict, dict, list[dict] | None]:
    """The second stage over the interior registration ``irec`` of ``window`` (written at ``window_dir``), chained after
    the boundary registration ``brec`` and its replay sample: `a5.audit` of the stream's second sample ``second`` (its
    draw, law and the interior registration's receipt), each member's session verified against ``reads`` (by member
    key, the verifier's own ``--registered`` files), the audit record, and each session's statement held to round 2
    (`round_two`, over the replay block ``replay``, its Program and descriptor id) at the params ``proofs``. ``tie``:
    the verifier's other copies the interior registration's check takes beside the boundary record (`a5.tie`: the
    values it registers, their schemas, its window root). -> (a5's summary, its acceptance, the audit record, what the
    run reports, the sessions' statements: None without a draw). The report names what the proof server refused, in
    flock-circuit's words, where a session never ran."""
    for d in (HERE.parent / "prove_path", HERE):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import a5
    import records_audit as RA

    a.coins, a.rule, a.window_root, a.compose_cache, a.transfer_gate = None, None, None, None, []
    a.tie = {"log": [brec], **(tie or {})}
    port = a.port or RA.free_ports(a5.RANK_PORTS)
    s, ok = a5.audit(a, sets, out / "audit", window, window_dir, source, port, provenance, registered=reads, drawn=second)
    rec_file = out / "audit" / "audit.json"
    record = json.loads(rec_file.read_text()) if rec_file.exists() else {}
    got = {"audit": {x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives",
                                           "circuits", "timing")},
           "accepted": ok, "record": {"failed": sorted(k for k, v in (record.get("checks") or {}).items() if not v.get("ok"))}}
    refusals = sorted({line.split("\t", 1)[1] for f in sorted((out / "audit" / "proof-server").glob("*.log"))
                       for line in f.read_text(errors="replace").splitlines() if line.startswith("REFUSED\t")})
    if refusals:  # why a session never ran: flock-circuit's own words (a statement outside its shape, say)
        got["proof_server_refused"] = refusals
    stmts = None
    if s.get("draw") is not None:
        stmts = statements(out / "audit", window)
        (out / "statements.json").write_text(json.dumps(stmts, indent=1, sort_keys=True))
        got["pins"] = pins(proofs, stmts)
        got["round_two"] = round_two(*replay, sample, window, irec, brec, s["draw"], stmts, record, proofs)
    return s, ok, record, got, stmts


def outcome_of(stream: S.Stream, second: S.Draw, handed: dict, evidence: Sequence[dict | None], statements: list[dict],
               proofs: dict) -> dict:
    """The stream's outcome of the second stage's sample: the sessions delivered (`Stream.deliver`), then
    `Stream.outcome` on what the interior's audit record took (``handed``, a5's), each verdict of record naming the
    sessions, with both stages' coin evidence, the sessions' ``statements`` and the params ``proofs``: accepted, the
    checks each stage's audit record failed (each record taking the sessions' reads of its registration's own ports,
    `verifier.audit_record`'s ``own_reads``), and the codes of round 2's check in the interior's (``statements``)."""
    sessions = handed["sessions"]
    verdicts = {x: {**y, "sessions": sessions} if y.get("of_record") else y for x, y in handed["verdicts"].items()}
    given = {x: handed[x] for x in ("proved_units", "instances", "served", "session_roots")}
    try:
        o = stream.deliver(second, sessions) or stream.outcome(second, verdicts, handed["statement"], coins=evidence[0],
                                                               interior_coins=evidence[1], statements=statements,
                                                               proofs=proofs, **given)
    except (S.Refused, S.Pending) as e:
        return {"accepted": False, "refused": e.verdict.to_json() if isinstance(e, S.Refused) else str(e)}
    body = stream.records[-1]
    audits = {"interior": body["audit"], "replay_sample": (body.get("replay_sample") or {}).get("audit")}
    return {"accepted": o.accepted, "record": o.record, "symbol": body["symbol"],
            **{k: None if r is None else sorted(x for x, c in r["checks"].items() if not c["ok"]) for k, r in audits.items()},
            "registered": {k: ((r or {}).get("checks") or {}).get("registered") for k, r in audits.items()},
            "statements": ((body["audit"].get("checks") or {}).get("statements") or {}).get("codes")}


def held(s: Mapping[str, Any], r2: Mapping[str, Any] | None, o: Mapping[str, Any] | None,
         files: Mapping[str, bool] | None) -> bool:
    """Whether an honest second stage held: every check before the sample passed, every session's circuit file is the
    one its template's params name (`pins`, ``files``), every Lean verdict accepted, the sessions' registered reads
    the verifier's own, every negative refused, round 2 ``r2`` accepting the statements, and the stream's outcome ``o``
    accepted or refused by round 2's two-stage form alone (`FORM_REFUSED`, while `second_stage.FORM` is off): the
    interior's audit record failing no check but round 2's, by that code at most, and the replay sample's none."""
    audit = s.get("audit") or {}
    refused = [v if isinstance(v, bool) else bool(v.get("refused")) for v in (s.get("negatives") or {}).values()]
    return (all((s.get("checks_before_draw") or {}).values()) and bool(files) and all((files or {}).values())
            and audit.get("complete") is True and audit.get("registered_reads") is True
            and all(v is True for v in (audit.get("verdicts") or {}).values()) and bool(refused) and all(refused)
            and (r2 or {}).get("ok") is True and o is not None
            and (o.get("accepted") is True
                 or (o.get("interior") in ([], ["statements"]) and o.get("statements") in ([], [FORM_REFUSED])
                     and o.get("replay_sample") == [])))


def prove(a: argparse.Namespace, r: Replayed, out: Path,
          proofs: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict] | None, dict[str, Any]]:
    """The verifier's half of the second stage on a registered interior ``r`` (`replayed`): the one-stage audit of the
    stream's second sample over the written window (`second_stage`). It hands over, by template, the composer and
    Definition (`composer`, on the check's own templates); by member key, the reads each port is verified against
    (`verifier_reads`); the second stage's draw, law and receipt; and the verifier's own window (`Design.window`) and
    copies (``r.tie``). ``proofs``: the auditor's params, else a development run's built here (`references`).
    -> (a5's summary, what the run reports, the sessions' statements, the params)."""
    d, sample, layout = r.design, r.sample, r.first.layout
    mine = d.window(r.first.registration.digest, sample)
    if proofs is None:
        proofs = references(d, len(sample), out / "references")
    S.template_circuits(proofs, r.first.stream.spec.check.query)
    (out / "proofs.json").write_text(json.dumps(proofs, indent=1, sort_keys=True))
    sets = {m.template: (composer(template_fn(m.template), m), template_fn(m.template)) for m in mine.members}
    s, _ok, _record, got, stmts = second_stage(
        a, sets, mine, r.directory, interior_source(mine, sample), r.first.record, r.record, sample,
        (r.draw.draw, r.draw.law, r.registration.receipt), verifier_reads(d, mine, r.record, r.first.record, sample),
        dict(proofs), out,
        {"two_stage": "interior", "class": [layout.prompt, layout.outputs], "window": r.first.pub["window"]},
        replay=(layout.program, d.key), tie=r.tie)
    return s, got, stmts, dict(proofs)


def _refusal(pub: Mapping[str, Any], e: Exception) -> dict[str, Any]:
    return {"format": FORMAT, "commit": pub, "refused": getattr(getattr(e, "verdict", None), "codes", None),
            "error": f"{type(e).__name__}: {e}"}


def _first_all(a: argparse.Namespace, pubs: list[dict], check: SpecializedDefinition, rule: Rule | None,
               window: Mapping[str, Any] | None, stop: BS.Stop | None = None) -> list[dict]:
    """Every class's first stage on one challenger session over the window's run, each class a rank on a thread of its own
    (`a5_coins.Draws`: the session opens once every rank has planned): each class's ``stage1.json`` contents, or why it
    sampled nothing. ``rule``: every class's, or None for each class's own from its replay block (`rule_of`). ``stop``:
    the verifier's own stop rule. A stop class's stop blocks are a rank more of the session (`stop_first`), planned
    with the class's own and sampled on the class's thread after its first stage: its ``stop``. Once the session has
    closed, each stop sample's order is read from the challenger's signed log (`stop_order`), and a window where one
    does not hold is refused whole."""
    import a5_coins as AC

    root = pubs[0]["run"]
    stops = {k: len(pubs) + j for j, k in enumerate(k for k, p in enumerate(pubs) if p.get("stopped") is True)}
    manager, turns, finisher = AC.start(a.challenger, root, list(range(len(pubs) + len(stops))),
                                        Path(a.out) / "challenger", a.main, ssh_key=a.ssh_key, peer=a.peer,
                                        key_record=a.challenger_key)
    out: list[dict] = [{} for _ in pubs]
    stop_design = a.stop_design or a.replay
    caught = (S.Refused, ValueError, KeyError, TypeError, OSError)

    def run(k: int) -> None:
        coins, blocks = AC.RankCoins(turns, k), AC.RankCoins(turns, stops[k]) if k in stops else None
        planner, unplanned = None, []

        def plan_blocks(need: int) -> None:
            try:
                blocks.plan(need)
            except Ch.CoinError as e:
                unplanned.append(e)

        try:
            layout = BR.layout_of(pubs[k], check, stop)
            mine = rule if rule is not None else rule_of(layout, a.replay, a.k, a.proof_deadline)
            need = AC.need(mine.replay, layout.program, layout.grain, layout.spans)
            if blocks is not None:
                planner = threading.Thread(target=plan_blocks, name=f"class-{k}-stop", args=(
                    AC.need(stop_design, layout.stop_program, layout.stop_query, layout.spans),))
                planner.start()
            coins.plan(need)
            got = first(pubs[k], check, mine, coins, window=window, delta=a.delta, window_root=root, stop=stop)
            out[k] = {**got.stage1(), "window_root": root}
        except caught as e:  # its stage1.json says why; the others go on
            out[k] = _refusal(pubs[k], e)
        finally:
            out[k]["segment"] = coins.drawn()
            turns.skip(k)
        if blocks is None:
            return
        mine_stop: dict[str, Any] = {}
        try:
            if planner is None:
                raise ValueError("the class's stop blocks planned no sample")
            planner.join()
            if unplanned:
                raise unplanned[0]
            if "units" not in out[k]:
                raise ValueError(f"the class's first stage sampled nothing: {out[k].get('error')}")
            mine_stop = stop_first(pubs[k], check, stop, stop_design, blocks, out[k]["record"], delta=a.delta,
                                   window_root=root, proof_deadline=a.proof_deadline)
        except caught as e:
            mine_stop = {"refused": getattr(getattr(e, "verdict", None), "codes", None), "error": f"{type(e).__name__}: {e}"}
        finally:
            mine_stop["segment"] = blocks.drawn()
            turns.skip(stops[k])
        if "units" in out[k] and "units" not in mine_stop:
            out[k] = {"format": FORMAT, "commit": pubs[k], "refused": [STOP_REFUSED], "segment": out[k]["segment"],
                      "error": f"{STOP_REFUSED}: the class's stop blocks registered or sampled nothing "
                               f"({mine_stop['error']}): a stop class is audited with its stop blocks or not at all"}
        out[k]["stop"] = mine_stop

    threads = [threading.Thread(target=run, args=(k,), name=f"class-{k}") for k in range(len(pubs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finisher.join()
    published = turns.evidence()
    wrong: list[str] = []
    for k, o in enumerate(out):
        o["evidence"] = AC.evidence(published, o["segment"]) if o.get("segment") else None
        o["session"] = {x: published.get(x) for x in ("plan", "trusted", "error") if x in published}
        s = o.get("stop")
        if s is None:
            continue
        s["evidence"] = AC.evidence(published, s["segment"]) if s.get("segment") else None
        if "units" in s and "units" in o:
            why = stop_order(s["evidence"], root, s.get("family"), o["registration"], s["registration"],
                             bytes.fromhex(s["read"]), AC.lms_verify)
            s["order"] = {"held": why is None, "reason": why}
            if why is not None:
                wrong.append(f"class {k} (prompt {pubs[k].get('prompt')}, outputs {pubs[k].get('outputs')}): {why}")
    if wrong:
        error = (f"{ORDER_REFUSED}: {'; '.join(wrong)}: the window is refused after its samples, and nothing samples "
                 f"it again")
        out = [{"format": FORMAT, "commit": pubs[k], "refused": [ORDER_REFUSED], "error": error,
                **{x: o[x] for x in ("segment", "evidence", "session", "stop") if x in o}} for k, o in enumerate(out)]
    manager.shutdown()
    return out


def first_root(stage1: Mapping[str, Any]) -> str:
    """The open root a class's first stage registered under: its ``stage1.json``'s ``window_root``, or, where the
    hand-off names none, its commitment's ``run``, the root `_first_all` opens the window's session on."""
    return stage1.get("window_root") or stage1["commit"]["run"]


def _second(a: argparse.Namespace) -> int:
    """One class's second stage: its stream again from ``stage1.json`` (`resume`), a challenger session of its own on the
    interior window's run (the first stage's session closed with its sample, and a challenger opens a root once), the
    interior registered and sampled on that session's coins (`replayed`), the sessions (`prove`), and the stream's
    outcome on both sessions' evidence (`outcome_of`). ``summary.json`` under ``--out`` says what happened; exit 0 when
    the second stage held (`held`)."""
    import a5_coins as AC

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage1 = json.loads(Path(a.stage1).read_text())
    summary: dict[str, Any] = {"stage1": str(a.stage1), "held": False, "unlinked": bool(a.unlinked)}

    def done(code: int, **more: Any) -> int:
        summary.update(more)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        r2, o = summary.get("round_two"), summary.get("outcome") or {}
        print("TWO " + json.dumps({**{x: summary.get(x) for x in ("refused", "sample", "second_stage", "pins", "held",
                                                                  "proof_server_refused")},
                                   "round_two": r2 and {"ok": r2.get("ok"), "codes": r2.get("codes")},
                                   "outcome": {x: o.get(x) for x in ("accepted", "interior", "replay_sample", "statements",
                                                                     "refused")}}, default=str))
        return code
    if "units" not in stage1:
        return done(1, refused=f"the first stage sampled nothing: {stage1.get('error')}")
    try:
        check, stop = template_fn(stage1["commit"]["check"]), _stop(a.stop)
        c = BR.reopen(Path(a.boundary), check, stop=stop)
        again = resume(stage1, check, delta=a.delta, window_root=first_root(stage1), stop=stop)
    except (S.Refused, ValueError, KeyError, OSError) as e:
        return done(1, refused=f"{type(e).__name__}: {e}")
    sample = [int(u) for u in again.sample.units]
    summary.update(sample=sample, first_stage={"registration": again.registration.digest, "units": sample},
                   stream={"digest": again.stream.digest, "cells": list(again.stream.cells),
                           "counts": list(again.stream.counts)})
    if not sample:
        return done(0, refused="the first stage sampled no span: there is no interior to register")
    root = IR.interior_run(again.registration.digest, sample)
    manager, draws, finisher = AC.start(a.challenger, root, [0], out / "challenger", a.main, ssh_key=a.ssh_key,
                                        peer=a.peer, key_record=a.challenger_key)
    coins = AC.RankCoins(draws, 0)
    try:
        try:
            coins.plan(second_need(again.stream, len(sample)))
            again.coins.then = coins
            r = replayed(again, c, Path(a.capture), out / "interior", window_root=root,
                         rids=json.loads(Path(a.rids).read_text()) if a.rids else None, unlinked=a.unlinked)
        except (S.Refused, Ch.CoinError, ValueError, OSError) as e:
            return done(1, refused=e.verdict.to_json() if isinstance(e, S.Refused) else f"{type(e).__name__}: {e}")
        finally:
            segment = coins.drawn()
            draws.skip(0)
        w = r.interior.committed.window
        summary.update(second_stage={"registration": r.registration.digest, "law": dict(r.draw.law), "draw": r.draw.units,
                                     "window_root": root},
                       interior={"population": w.population, "run": w.run, "launches": r.launches,
                                 "members": [{"key": m.key, "n": m.n, "base": m.base} for m in w.members],
                                 "registered": [e["value"] for e in r.record.get("registered", ())],
                                 "links": [[x.template, x.port, x.value] for x in r.design.links],
                                 "returns": [[x.template, x.port, x.value] for x in r.design.returns]})
        s, got, stmts, proofs = prove(a, r, out, json.loads(Path(a.proofs).read_text()) if a.proofs else None)
        summary.update(got)
        published = draws.evidence()
        summary["challenger"] = {x: published.get(x) for x in ("trusted", "error", "plan")}
        evidence = [stage1.get("evidence"), AC.evidence(published, segment)]
        (out / "coins.json").write_text(json.dumps({"evidence": evidence, "read": [stage1["read"], coins.read.hex()]},
                                                   indent=1, sort_keys=True))
        if "handed" in s:
            summary["outcome"] = outcome_of(again.stream, r.draw, s["handed"], evidence, stmts, proofs)
        ok = held(s, summary.get("round_two"), summary.get("outcome"), summary.get("pins"))
        return done(0 if ok else 1, held=ok)
    finally:
        draws.skip(0)
        finisher.join()
        manager.shutdown()


STOP_HELP = ("the verifier's own copy of the serve's stop rule, as JSON: {\"ids\": [...], \"max_model_len\": n, "
             "\"outputs\": T} (boundary_stop.Stop.of); without it a class committed under a stop rule is refused")


def _stop(text: str | None) -> BS.Stop | None:
    return None if text is None else BS.Stop.of(json.loads(text))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("first", help="every class's boundary registration and first stage")
    f.add_argument("--boundary", nargs="+", required=True, help="each class's directory (w<window>/p<P>-o<O>/), one window's")
    f.add_argument("--check", required=True, help="the check template's descriptor id (serving_templates.template)")
    f.add_argument("--replay", required=True, help="the first stage's law over a class's spans, a rate law: rate:NUM/DEN or rate:NUM/DEN:T")
    f.add_argument("--k", required=True, help="proof blocks the second stage samples of each check template in each sampled span: "
                                              "a whole number, or a JSON file of each template's, {descriptor id: k_t}, "
                                              "which names each template of every class's check once")
    f.add_argument("--interior", help="the interior values, a JSON list of [name, schema]; without it each class's "
                                      "rule is its replay block's (rule_of), and --span-values is not taken")
    f.add_argument("--advice", help="the window's advice registration, whose binding the records' windows name")
    f.add_argument("--proof-deadline", type=float, default=PROOF_DEADLINE_S)
    f.add_argument("--span-values", help="the replay block's parameters the interior registration registers as values, "
                                         "a JSON list of [name, rows a sampled span, leaf type]")
    f.add_argument("--stop", help=STOP_HELP)
    f.add_argument("--stop-design", help="the sampling design over a stop class's stop blocks, one a span: a rate, as "
                                         "--replay is; without it, --replay's")
    f.add_argument("--delta", type=float, default=0.5)
    f.add_argument("--challenger", required=True, help="local, loopback:HOST:PORT or gate:USER@HOST (a5_coins.start)")
    f.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`, a file")
    f.add_argument("--peer")
    f.add_argument("--challenger-key")
    f.add_argument("--ssh-key")
    f.add_argument("--out", required=True)
    g = sub.add_parser("second", help="one class's interior registration, second stage, proofs and outcome")
    g.add_argument("--stage1", required=True, help="the class's stage1.json (first)")
    g.add_argument("--boundary", required=True, help="the class's directory (w<window>/p<P>-o<O>/), its private files too")
    g.add_argument("--capture", required=True, help="the replay's capture of the sampled spans' sampler inputs "
                                                    "(engine.sampler_capture.write)")
    g.add_argument("--rids", help="a JSON object: the capture's request id for a span's engine id, where they differ")
    g.add_argument("--unlinked", action="store_true",
                   help="a class whose replay block gives round 2 no wires (refused without this): commit and prove its "
                        "sampled proof blocks with no link and no wire, each port at its own rows. Nothing then ties a "
                        "block to another or to round 1: round 2 refuses the statements (second-wires) and this exits 1")
    g.add_argument("--proofs", help="the auditor's params (service.template_circuits); without them a development run's "
                                    "are built after the sample (references)")
    g.add_argument("--m0-pythonpath", required=True)
    g.add_argument("--bin", required=True, help="flock-circuit")
    g.add_argument("--lean", required=True, help="flock-verify")
    g.add_argument("--firewall", help="Lean's flock-firewall: every session behind the ZK server; without it each runs "
                                      "prove --zk itself, as SPOT[Flock]'s proof server does")
    g.add_argument("--stop", help=STOP_HELP)
    g.add_argument("--delta", type=float, default=0.5)
    g.add_argument("--port", type=int, default=0, help="the sessions' first port (0: a free block)")
    g.add_argument("--session-timeout", type=float, default=3600)
    g.add_argument("--challenger", required=True, help="local, loopback:HOST:PORT or gate:USER@HOST (a5_coins.start)")
    g.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`, a file")
    g.add_argument("--peer")
    g.add_argument("--challenger-key")
    g.add_argument("--ssh-key")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "second":
        return _second(a)
    if a.interior is None and a.span_values:
        ap.error("--span-values is the rule's that --interior gives: without --interior each class's rule is its replay block's")
    if a.stop_design and not a.stop:
        ap.error("--stop-design is the sampling design over the stop blocks of the rule --stop gives")
    pubs = [json.loads((Path(d) / BR.COMMIT).read_text()) for d in a.boundary]
    if len({(p["run"], p["window"]) for p in pubs}) != 1:
        ap.error("--boundary: every class of one window (one run and window index)")
    window = {"binding": json.loads(Path(a.advice).read_text())["binding"]} if a.advice else None
    check, stop = template_fn(a.check), _stop(a.stop)
    try:
        a.k = counts(a.k)
    except (ValueError, OSError) as e:
        ap.error(f"--k: {e}")
    for d, pub in zip(a.boundary, pubs, strict=True) if isinstance(a.k, Mapping) else ():
        try:
            why = counts_refused(a.k, BR.layout_of(pub, check, stop).templates)
        except ValueError as e:  # no class of this check and stop rule: the counts name no template of it
            why = str(e)
        if why:
            ap.error(f"--k: class {Path(d).name}: {why}")
    rule = None if a.interior is None else Rule(
        a.replay, a.k, tuple((str(n), str(s)) for n, s in json.loads(Path(a.interior).read_text())),
        a.proof_deadline, _span_values(json.loads(Path(a.span_values).read_text())) if a.span_values else ())
    out = Path(a.out)
    for d, s in zip(a.boundary, _first_all(a, pubs, check, rule, window, stop), strict=True):
        dest = out / Path(d).name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / STAGE1).write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps({"class": Path(d).name, "spans": s["commit"]["spans"], "units": s.get("units"),
                          "error": s.get("error"), **({"stop": {x: s["stop"].get(x) for x in ("units", "order", "error")}}
                                                      if "stop" in s else {})}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
