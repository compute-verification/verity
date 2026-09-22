"""[SAMPLER] Two-stage Bernoulli sampling against the VALIDATED PARTITION interface (GUIDANCE §57.15; S1/S3 approved).

    validated partition -> selected RU identities (stage 1, rate q) -> independently sampled VU identities inside the
    selected RUs (stage 2, rate s) -> required proof predicates (``e2e.predicates``) -> the same partition's Bound.

This module is ``e2e.security_sample``'s law over a *population view* instead of the dense ``ru_of`` list: the population may be
``veritor/population/v1`` (dense, ``e2e.population``) or ``veritor/population/v2`` (compact, ``e2e.population_v2``).  Both give
the same law -- stage 2 ranks the VUs of the selected RUs in J order, ascending VU index inside each RU, and draws
``veritor.protocol.challenge.bernoulli_subset`` over that ranking (``draw_Q`` here == ``security_sample.draw_Q``; tested on TINY
for the same seed).  The chronology rules are the existing ones: the seed is committed before the boundary commitment, ``r_q``
and ``r_s`` are derived as before, every challenge is recorded, a second draw is refused, realised sizes are recorded but never
enter the law.  What is new:

* the population is **bound to the instance's partition**: ``partition_digest``, ``plan_digest``, ``program_digest`` and the counts
  must all agree, and a ``population/v2`` file is self-checked (partition digest re-derived from its rows) before stage 1;
* the population is **pinned at stage 1** (its content sha256 is written into the chronology) and stage 2 / verify refuse a
  different file: a population substituted between the draws -- another plan, another representation, a tampered ``ru_of`` --
  is a REFUSAL, not a silently different sample (S9);
* stage 2 writes ``selected.json``: every selected VU as ``{vu_index, ru_index, kind, gate_interval, partition_digest}`` -- the
  identities the proof obligations must cover (``e2e.predicates`` maps each to its required predicate or REFUSES).

The committed relation store is not a population here: nothing in this module reads a capture, and the adapter's
``adapter-transcript/v1`` path (``security_sample.from_transcript``) is untouched.  Nothing in ``run_composed`` is routed through
this module (no default changed); it is callable from the recorder once REC decides to (REC-13).

[A3 contract 2026-09-10] (docs/PROTOCOL.md steps 3-4; conformance R6/R7/R8/R16):

* **exemptions**: stage 2 takes the prover's exemption request ``F`` (RU indices, one message with the interior commitment); it is
  REFUSED unless ``F ⊆ J`` and ``|F| <= B_ex`` (the instance's RU-count budget, ``inst.exemption_budget``); a request after ``r_s``
  exists is refused (late); ``r_s`` is drawn over the VUs of J minus F only; the request is a chronology event ``exemption_request``
  between ``interior`` and ``r_s`` and its digest is bound into ``phase_s`` (``step3_digest = H(STEP3, interior_digest, F digest)``);
* **seed custody**: ``stage1`` keeps the seed in the verifier process (``_HELD_SEEDS``; the library DEFAULT since A10's WR-2) instead of
  writing ``seed.hex`` next to the prover-readable record; the seed appears only in the ``seed_reveal`` event.  The legacy file path
  must be asked for (``legacy_seed_file=True`` / ``--legacy-seed-file``), prints a LEGACY line and is labelled ``record_dir_file``;
* **epoch barrier** (R16): stage 1 records the declared epoch and refuses a boundary commitment that does not name every declared
  request (``boundary_request_ids``); stage 2 refuses a step-3 message that does not cover every RU of ``J`` (``committed_rus`` ∪ ``F``).

    python -m e2e.partition_sample stage1 --instance I.json --population POP --boundary-digest HEX --out DIR [--seed HEX]
    python -m e2e.partition_sample stage2 --instance I.json --population POP --interior-digest HEX --out DIR [--exempt R ...]
    python -m e2e.partition_sample reveal --out DIR
    python -m e2e.partition_sample verify --instance I.json --population POP --out DIR
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from bisect import bisect_right
from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from collections.abc import Sequence
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e.population_v2 import POPULATION_V1, POPULATION_V2, CompactPopulation, PopulationError, canonical_bytes, sha256_hex  # noqa: E402
from e2e.protocol_instance import PARTITION_DIGEST_KIND, InstanceError, instance_digest, load_instance  # noqa: E402
from e2e.security_sample import (  # noqa: E402
    PHASE_DOMAIN,  # [A3 2026-09-10] was not imported: the redraw path (--condition-nonempty, empty r_q) raised NameError on mainline
    S_STAGE,
    SCHEMA,
    SEED_DOMAIN,
    ChronologyError,
    _event,
    _h,
    _limits,
    _read,
    _write,
    challenge_digest,
    draw_J,
    phase_digest_q,
    phase_digest_s,
)

SELECTED_SCHEMA = "veritor/selected-vus/v1"
SAMPLER = "e2e.partition_sample/v1"
STEP3_DOMAIN = b"veritor/contract/step3/v1"  # H(STEP3, interior_digest, exemption_digest): what phase_s binds when F exists
EXEMPTION_DOMAIN = b"veritor/contract/exemption-request/v1"
SEED_CUSTODY_MEMORY = "verifier_process_memory"
SEED_CUSTODY_FILE = "record_dir_file"
# seeds held by the verifier process between stage 1 and stage 2 (keyed by the resolved chronology directory); never written to
# the record directory unless stage1(..., legacy_seed_file=True) -- the legacy behaviour a co-located prover could read (R7)
_HELD_SEEDS: dict[str, bytes] = {}


# -- population views ----------------------------------------------------------------------------------------------------------


class PopulationView:
    """What the two-stage law reads: counts, digests, RU membership as *ranking* primitives.  Subclasses: dense v1, compact v2."""

    schema: str
    vu_count: int
    ru_count: int
    plan_digest: str
    program_digest: str | None
    partition_digest: str | None
    instances: int
    content_sha256: str
    path: Path | None

    def ru_of(self, vu: int) -> int:
        raise NotImplementedError

    def ru_size(self, ru: int) -> int:
        raise NotImplementedError

    def vu_at(self, ru: int, k: int) -> int:
        raise NotImplementedError

    def kind_of(self, vu: int) -> dict[str, Any] | None:  # v2 only
        return None

    def gate_interval(self, vu: int) -> tuple[int, int] | None:  # v2 only
        return None

    def describe(self) -> dict[str, Any]:
        return {"schema": self.schema, "path": str(self.path) if self.path else None, "content_sha256": self.content_sha256,
                "vu_count": self.vu_count, "ru_count": self.ru_count, "plan_digest": self.plan_digest, "program_digest": self.program_digest,
                "partition_digest": self.partition_digest, "instances": self.instances}


class DensePopulation(PopulationView):
    """``population/v1``: the ``ru_of`` list (kept readable; O(N_V) memory, as before)."""

    schema = POPULATION_V1

    def __init__(self, doc: dict[str, Any], *, path: Path | None = None, raw: bytes | None = None) -> None:
        for k in ("vu_count", "ru_count", "ru_of", "plan_digest", "program_digest"):
            if k not in doc:
                raise PopulationError(f"population/v1 missing {k!r}")
        if doc.get("schema") not in (POPULATION_V1, None):
            raise PopulationError(f"schema {doc.get('schema')!r} is not population/v1")
        if len(doc["ru_of"]) != doc["vu_count"]:
            raise PopulationError("population ru_of length != vu_count")
        self.doc, self.path = doc, path
        self.vu_count, self.ru_count = int(doc["vu_count"]), int(doc["ru_count"])
        self.plan_digest, self.program_digest = doc["plan_digest"], doc["program_digest"]
        self.partition_digest = doc.get("partition_digest")
        self.instances = int(doc.get("instances", 1))
        self.content_sha256 = sha256_hex(raw if raw is not None else canonical_bytes(doc))
        self._ru_of = doc["ru_of"]
        groups: list[list[int]] = [[] for _ in range(self.ru_count)]
        for vu, ru in enumerate(self._ru_of):
            if not 0 <= ru < self.ru_count:
                raise PopulationError(f"VU {vu} names RU {ru} outside [0, {self.ru_count})")
            groups[ru].append(vu)
        self._groups = groups

    def ru_of(self, vu: int) -> int:
        return self._ru_of[vu]

    def ru_size(self, ru: int) -> int:
        return len(self._groups[ru])

    def vu_at(self, ru: int, k: int) -> int:
        return self._groups[ru][k]


class CompactView(PopulationView):
    """``population/v2`` (:class:`e2e.population_v2.CompactPopulation`)."""

    schema = POPULATION_V2

    def __init__(self, pop: CompactPopulation) -> None:
        self.pop, self.path = pop, pop.path
        self.vu_count, self.ru_count = pop.vu_count, pop.ru_count
        self.plan_digest, self.program_digest, self.partition_digest = pop.plan_digest, pop.program_digest, pop.partition_digest
        self.instances, self.content_sha256 = pop.instances, pop.content_sha256

    def ru_of(self, vu: int) -> int:
        return self.pop.ru_of(vu)

    def ru_size(self, ru: int) -> int:
        return self.pop.ru_size(ru)

    def vu_at(self, ru: int, k: int) -> int:
        return self.pop.vu_at(ru, k)

    def kind_of(self, vu: int) -> dict[str, Any]:
        return self.pop.kind_of(vu).to_json()

    def gate_interval(self, vu: int) -> tuple[int, int]:
        return self.pop.gate_interval(vu)


class ReplicatedView(PopulationView):
    """[L1 RECORD 2026-09-10] The record-level population for ``instances`` program instances (requests) sharing ONE validated
    partition -- ``e2e.population.replicate_population``'s tiling (global VU = r * N_V + i, global RU = r * N_R + j) as a view over a
    per-instance population/v2 file instead of a dense copy.  Every instance carries the whole validated population, input units
    included (the per-RU-copy formulation, DIRECTIVE v3 A1); nothing is shared across instances and nothing is added.  Digests
    (program, plan, partition) are the base file's; ``content_sha256`` is the base file's, so the pin is the validated artifact."""

    def __init__(self, base: PopulationView, instances: int) -> None:
        if instances < 1:
            raise PopulationError("instances must be >= 1")
        if base.instances != 1:
            raise PopulationError("replicate a per-instance population (instances == 1)")
        self.base, self.schema, self.path = base, base.schema, base.path
        self.n_v, self.n_r = base.vu_count, base.ru_count
        self.vu_count, self.ru_count = instances * self.n_v, instances * self.n_r
        self.plan_digest, self.program_digest, self.partition_digest = base.plan_digest, base.program_digest, base.partition_digest
        self.instances, self.content_sha256 = instances, base.content_sha256

    def ru_of(self, vu: int) -> int:
        r, i = divmod(vu, self.n_v)
        return r * self.n_r + self.base.ru_of(i)

    def ru_size(self, ru: int) -> int:
        return self.base.ru_size(ru % self.n_r)

    def vu_at(self, ru: int, k: int) -> int:
        r, j = divmod(ru, self.n_r)
        return r * self.n_v + self.base.vu_at(j, k)

    def kind_of(self, vu: int) -> dict[str, Any] | None:
        return self.base.kind_of(vu % self.n_v)

    def gate_interval(self, vu: int) -> tuple[int, int] | None:
        return self.base.gate_interval(vu % self.n_v)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "per_instance": {"vu_count": self.n_v, "ru_count": self.n_r},
                "replication": f"view: {self.instances} program instance(s) x the per-instance file (global VU = r*N_V + i, global RU = r*N_R + j)"}


def load_population(path: Path, *, check: str = "fast", instances: int | None = None) -> PopulationView:
    """Either schema.  ``check`` applies to v2: ``"full"`` re-derives the partition digest (stage 1), ``"fast"`` the O(rows) checks.
    ``instances`` (> 1) wraps a per-instance file in a :class:`ReplicatedView` (the record's requests)."""

    path = Path(path)
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        raise PopulationError(f"{path}: {e}") from e
    schema = doc.get("schema")
    if schema == POPULATION_V2:
        pop = CompactPopulation(doc, path=path)
        if check != "none":
            pop.check(digest=check == "full")
        view: PopulationView = CompactView(pop)
    elif schema in (POPULATION_V1, None):
        view = DensePopulation(doc, path=path, raw=raw)
    else:
        raise PopulationError(f"{path}: unknown population schema {schema!r} (readable: {POPULATION_V1}, {POPULATION_V2})")
    if instances is not None and instances != view.instances:
        view = ReplicatedView(view, instances)
    return view


# -- the law -------------------------------------------------------------------------------------------------------------------


def draw_Q(seed: bytes, phase: str, J: tuple[int, ...], pop: PopulationView, s: Fraction) -> tuple[int, ...]:
    """Bernoulli(s) over the VUs of the selected RUs, ranked block by block in J order and ascending VU index inside a block --
    the ranking ``security_sample.draw_Q`` uses over ``vus_by_ru`` -- mapped back to VU indices.  ``O(|Q| log)``: never a list of
    the candidate VUs."""

    from veritor.protocol.challenge import bernoulli_subset

    sizes = [pop.ru_size(r) for r in J]
    ends = list(accumulate(sizes))
    count = ends[-1] if ends else 0
    ranks = bernoulli_subset(seed, S_STAGE, bytes.fromhex(phase), count, s, _limits())

    def vu_at(rank: int) -> int:
        b = bisect_right(ends, rank)
        start = ends[b] - sizes[b]
        return pop.vu_at(J[b], rank - start)

    return tuple(sorted(vu_at(r) for r in ranks))


def candidates(J: tuple[int, ...], pop: PopulationView) -> int:
    return sum(pop.ru_size(r) for r in J)


# -- binding the population to the instance ------------------------------------------------------------------------------------


def bind(inst, pop: PopulationView) -> None:
    """REFUSE unless the population IS the instance's validated partition (S3/S7/S9): law, counts, program, plan, partition."""

    if inst.sampler.law.kind != "bernoulli":
        raise ChronologyError(f"instance law is {inst.sampler.law.kind!r}; this sampler implements 'bernoulli' only (fixed-size stays on run_composed --k)")
    n = inst.partition.instances
    if inst.partition.vu_count.value * n != pop.vu_count or inst.partition.ru_count.value * n != pop.ru_count:
        raise ChronologyError(f"population {pop.vu_count}/{pop.ru_count} != instance partition {n} x {inst.partition.vu_count.value}/{inst.partition.ru_count.value}")
    if pop.instances != n:
        raise ChronologyError(f"population file instances {pop.instances} != instance partition.instances {n}")
    # [E9, D24 2026-09-12] every digest disagreement names the check and carries BOTH FULL digests: a byte flip deep in the digest made the
    # 12-hex prefixes read equal on E8's CORE-live-neg-mispin-01 ("bc1a72682890 differs from bc1a72682890")
    if inst.program.digest != pop.program_digest:
        raise ChronologyError(f"V2.program_digest: population file's program_digest {pop.program_digest} differs from the instance's {inst.program.digest} "
                              "(a population of another program)")
    if inst.partition.plan_descriptor_digest != pop.plan_digest:
        raise ChronologyError(f"V2.plan_digest: population file's plan_digest {pop.plan_digest} differs from the instance's {inst.partition.plan_descriptor_digest} "
                              "(a population of another plan / representation; the instance names exactly one)")
    if pop.partition_digest != inst.partition.partition_digest:
        raise ChronologyError(f"V2.partition_digest: population file's partition_digest {pop.partition_digest} differs from the instance's "
                              f"{inst.partition.partition_digest} (the population must be THE validated partition, S3/S7)")


def _pinned(doc: dict[str, Any], pop: PopulationView) -> None:
    """Stage 2 / verify: the population must be the one pinned at stage 1 (same content, same partition)."""

    pin = doc.get("population") or {}
    if pin.get("content_sha256") and pin["content_sha256"] != pop.content_sha256:
        raise ChronologyError(f"population {pop.path} (content {pop.content_sha256[:12]}) is not the file pinned at stage 1 ({pin['content_sha256'][:12]}, "
                              f"schema {pin.get('schema')}): a population substituted between the draws is refused")
    for k in ("vu_count", "ru_count", "plan_digest", "program_digest", "partition_digest"):
        if k in pin and pin[k] != getattr(pop, k):
            raise ChronologyError(f"population {k} {getattr(pop, k)!r} != the value pinned at stage 1 {pin[k]!r}")


# -- stages --------------------------------------------------------------------------------------------------------------------------


REDRAW_DOMAIN = b"q/replay-unit/redraw"
REDRAW_NOTE = "RU draw conditioned on >= 1 selected (demonstration)"


def phase_digest_q_redraw(phase_hex: str, attempt: int) -> str:
    """[L1 RECORD] The phase of the k-th logged redraw of r_q (DIRECTIVE §1.5.10: a seed that selects >= 1 RU): derived from the
    previous phase, never from a re-chosen seed; the empty draw stays in the chronology."""

    return _h(PHASE_DOMAIN, REDRAW_DOMAIN, bytes.fromhex(phase_hex), attempt.to_bytes(4, "big"))


def exemption_digest(F: tuple[int, ...]) -> str:
    """Digest of the exemption request ``F`` (sorted RU indices) -- the ``exemption_request`` event's digest."""

    return _h(EXEMPTION_DOMAIN, json.dumps(sorted(F), separators=(",", ":")).encode())


def step3_digest(interior_digest_hex: str, exemption_digest_hex: str) -> str:
    """What ``phase_s`` binds under the contract: the interior commitment AND the complete exemption request."""

    return _h(STEP3_DOMAIN, bytes.fromhex(interior_digest_hex), bytes.fromhex(exemption_digest_hex))


def _seed_for(out: Path, doc: dict[str, Any]) -> bytes:
    """The seed of this chronology: from the verifier process when stage 1 held it there, else the legacy ``seed.hex``."""

    key = str(out.resolve())
    if key in _HELD_SEEDS:
        return _HELD_SEEDS[key]
    f = out / "seed.hex"
    if not f.exists():
        raise ChronologyError("seed is neither held by this verifier process nor in seed.hex: stage 2 must run in the process that ran stage 1 "
                              "(seed_custody = verifier_process_memory)")
    return bytes.fromhex(f.read_text().strip())


# --- pinned instance/population format selection (integrator 23:18Z item 3; release gate) ---------------------------------------------
#
# The shipped entry points (verifier_v3 driver / reverify) select the instance and population loaders EXPLICITLY, by a name that is
# recorded in the chronology and validated against the instance -- never by a process-wide reassignment of ``load_instance`` /
# ``load_population``.  ``"contract"`` is this module's own pair; another format (A9's ``record_v5``: allowlisted instance violations +
# the ``a9/legacy-minus-inputs/v1`` population schema) registers its pair under its name and the caller names it.  When ``instance_format``
# is None the module-level names are used (the pre-existing behaviour, monkeypatch-compatible for legacy callers).

INSTANCE_FORMATS: dict[str, tuple[Any, Any, tuple[str, ...] | None]] = {}


def register_instance_format(name: str, load_instance_fn: Any, load_population_fn: Any, *, partition_digest_kind: str | Sequence[str] | None) -> None:
    """Register ``name`` -> (instance loader, population loader, the ADMITTED ``partition.partition_digest_kind`` values an instance of this
    format may declare -- one string or a sequence of them, kept as a tuple BY NAME; the verifier validates the CLI choice against the
    instance by membership and refuses anything else; None = no constraint).  [E9, D19 2026-09-11] a format may admit several kinds
    (``record_v5``: ``a9/legacy-minus-inputs/v1`` and ``/v2``); an unlisted kind is a refusal, never a silent reinterpretation."""

    kinds = None if partition_digest_kind is None else ((partition_digest_kind,) if isinstance(partition_digest_kind, str) else tuple(str(k) for k in partition_digest_kind))
    if kinds is not None and (not kinds or len(set(kinds)) != len(kinds) or any(not k for k in kinds)):
        raise ValueError(f"instance format {name!r}: admitted partition_digest_kinds must be non-empty, distinct, non-blank names: {kinds!r}")
    INSTANCE_FORMATS[name] = (load_instance_fn, load_population_fn, kinds)


def admitted_partition_digest_kinds(instance_format: str) -> tuple[str, ...] | None:
    """The registry of admitted ``partition_digest_kind`` names for a format (``None`` = unconstrained); ``KeyError`` for an unknown format."""

    return INSTANCE_FORMATS[instance_format][2]


def loaders(instance_format: str | None) -> tuple[Any, Any]:
    if instance_format is None:
        return load_instance, load_population  # module-level names (legacy callers may have reassigned them)
    if instance_format not in INSTANCE_FORMATS:
        raise InstanceError(f"unknown instance format {instance_format!r} (registered: {sorted(INSTANCE_FORMATS)})")
    li, lp, _ = INSTANCE_FORMATS[instance_format]
    return li, lp


def check_instance_format(instance_format: str, inst_doc: dict[str, Any]) -> list[str]:
    """The pinned instance must declare the format's ``partition_digest_kind``: a ``record_v5`` CLI choice over a contract instance (or the
    reverse) is a refusal by name, never a silent reinterpretation."""

    if instance_format not in INSTANCE_FORMATS:
        return [f"unknown instance format {instance_format!r} (registered: {sorted(INSTANCE_FORMATS)})"]
    want = INSTANCE_FORMATS[instance_format][2]
    got = ((inst_doc.get("partition") or {}).get("partition_digest_kind"))
    if want is not None and (not isinstance(got, str) or got not in want):
        return [f"instance format {instance_format!r} admits partition.partition_digest_kind in {list(want)!r}; the instance declares {got!r} -> refused by name"]
    return []


register_instance_format("contract", load_instance, load_population, partition_digest_kind=PARTITION_DIGEST_KIND)


def stage1(instance: Path, population: Path, boundary_digest: str, out: Path, seed_hex: str | None = None, *,
           instances: int | None = None, condition_nonempty: bool = False, max_redraws: int = 8, legacy_seed_file: bool = False,
           boundary_request_ids: list[str] | None = None, seed_file: bool | None = None, instance_format: str | None = None) -> dict[str, Any]:
    # [A10 WR-2, R7] the LIBRARY DEFAULT is in-process custody: the seed never touches the record directory.  The legacy file
    # (readable by a co-located prover) must be asked for explicitly with legacy_seed_file=True and is announced on stderr.
    # `seed_file` is the pre-WR-2 keyword, kept as an alias so no caller silently changes meaning: True == legacy_seed_file=True.
    if seed_file is not None:
        legacy_seed_file = bool(seed_file)
    if legacy_seed_file:
        print(f"[partition_sample] LEGACY seed custody: seed.hex will be written into {out} (record_dir_file, prover-readable; "
              f"docs/PROTOCOL-CONFORMANCE.md R7).  Not the contract path -- pass legacy_seed_file=False / drop --legacy-seed-file.", file=sys.stderr)
    seed_file = legacy_seed_file
    _li, _lp = loaders(instance_format)
    inst = _li(instance)
    pop = _lp(population, check="full", instances=instances)  # v2: the partition digest is re-derived from the rows before anything is drawn
    bind(inst, pop)
    idig = instance_digest(inst)
    existing = _read(out)
    if existing is not None:
        if any(e["kind"] == "challenge" for e in existing["events"]):
            raise ChronologyError("chronology already holds a challenge for this instance: a second stage-1 draw is a reroll and is refused")
        raise ChronologyError("chronology exists without a challenge: remove it explicitly (an aborted audit's chronology is evidence, not scratch)")
    if len(boundary_digest) != 64:
        raise ChronologyError("boundary commitment digest must be a sha256 hex")
    seed = bytes.fromhex(seed_hex) if seed_hex else secrets.token_bytes(32)
    if len(seed) != 32:
        raise ChronologyError("seed must be 32 bytes")
    q = Fraction(inst.sampler.law.q)
    # [A3 R16] the epoch barrier at step 2: the boundary commitment must cover EVERY request the declared epoch names before r_q is derived
    if inst.epoch is not None:
        if boundary_request_ids is None:
            raise ChronologyError("the instance declares an epoch: stage 1 needs the request ids the boundary commitment covers (boundary_request_ids) "
                                  "to enforce the barrier (r_q only after every RU of the epoch has its step-1 values committed)")
        declared, covered = set(inst.epoch.request_ids), set(boundary_request_ids)
        if len(covered) != len(boundary_request_ids):
            raise ChronologyError("boundary commitment names a request twice")
        if covered != declared:
            raise ChronologyError(f"boundary commitment covers {sorted(covered)} but the epoch {inst.epoch.epoch_id!r} declares {sorted(declared)}: "
                                  "r_q is derived only after every declared request's first-boundary values are committed (no streaming)")
    elif boundary_request_ids is not None:
        raise ChronologyError("boundary_request_ids given but the instance declares no epoch (legacy instance): declare the epoch (R16) or drop the ids")
    doc: dict[str, Any] = {
        "schema": SCHEMA, "sampler": SAMPLER, "instance_digest": idig, "instance_path": str(instance),
        "seed_custody": SEED_CUSTODY_FILE if seed_file else SEED_CUSTODY_MEMORY,
        "instance_format": instance_format,  # the explicitly selected loader pair (None = module-level names, legacy callers)
        "epoch": ({"epoch_id": inst.epoch.epoch_id, "request_ids": list(inst.epoch.request_ids), "ru_count": pop.ru_count, "B_ex": inst.epoch.B_ex,
                   "B_ex_unit": inst.epoch.B_ex_unit, "boundary_request_ids": list(boundary_request_ids or [])} if inst.epoch is not None else None),
        "law": {"kind": "bernoulli", "q": str(q), "s": inst.sampler.law.s},
        "population": {**pop.describe(), "note": "the instance's validated VU partition (S3): RU identities then VU identities of the selected RUs; "
                                                  "NOT committed leaves, NOT the adapter's relation units; pinned here by content sha256 for stage 2"},
        "seed_derivation": "seed = 32 random bytes (secrets.token_bytes) held by the verifier; seed_commitment = H(domain, seed); "
                           "r_q = bernoulli_subset(seed, 'q/replay-unit', phase_q, N_R, q) with phase_q = H(PHASE_DOMAIN, 'q', instance_digest, boundary_digest); "
                           "r_s = bernoulli_subset(seed, 's/verification-unit', phase_s, |VUs of J|, s) with phase_s = H(PHASE_DOMAIN, 's', instance_digest, r_q_digest, interior_digest); "
                           "|VUs of J| ranked in J order, ascending VU index inside each RU (identical for population/v1 and /v2)",
        "events": [], "challenges_drawn": [], "rerolls": 0, "realized": {},
    }
    _event(doc, "declaration", "protocol_instance", idig, [], note="status = " + inst.status)
    _event(doc, "commitment", "seed_commitment", _h(SEED_DOMAIN, seed), ["protocol_instance"])
    _event(doc, "commitment", "boundary", boundary_digest, ["protocol_instance"], note="prover's first-boundary commitment (all RUs' boundaries + designated outputs)")
    phase = phase_digest_q(idig, boundary_digest)
    J = draw_J(seed, phase, pop.ru_count, q)
    cd = challenge_digest("q", phase, pop.ru_count, q, J)
    redraws = 0
    if condition_nonempty:
        # [L1 RECORD] DIRECTIVE §1.5.10 ("a seed that selects at least one"): an empty r_q is RECORDED (event r_q_empty[k], its digest in
        # challenges_drawn) and r_q is drawn again under a derived phase; the certificate states REDRAW_NOTE and the Bound is
        # reported for the unconditioned law (orchestrator ruling 5).  Never a re-chosen seed; `verify` re-derives every attempt.
        while not J and redraws < max_redraws:
            _event(doc, "challenge", f"r_q_empty[{redraws}]", cd, ["seed_commitment", "boundary"], phase_digest=phase, selection=[],
                   note=f"empty RU draw #{redraws}, kept in the chronology; {REDRAW_NOTE}")
            doc["challenges_drawn"].append(cd)
            redraws += 1
            phase = phase_digest_q_redraw(phase, redraws)
            J = draw_J(seed, phase, pop.ru_count, q)
            cd = challenge_digest("q", phase, pop.ru_count, q, J)
        doc["conditioning"] = {"rule": REDRAW_NOTE, "redraws": redraws, "phase_rule": "phase_k = H(PHASE_DOMAIN, 'q/replay-unit/redraw', phase_{k-1}, k)",
                               "bound_law": "the Bound is reported for the unconditioned law (q, s)"}
    _event(doc, "challenge", "r_q", cd, ["seed_commitment", "boundary"], phase_digest=phase, selection=list(J),
           **({"redraws": redraws, "note": REDRAW_NOTE} if redraws else {}))
    doc["challenges_drawn"].append(cd)
    doc["realized"]["J_size"] = {"value": len(J), "label": "MEASURED", "source": "chronology.json events[r_q].selection (realized, not the rate q)"}
    if redraws:
        doc["realized"]["r_q_redraws"] = {"value": redraws, "label": "MEASURED", "source": "empty r_q draws kept as events r_q_empty[k] before r_q"}
    doc["realized"]["Q_candidates_expected"] = {"value": candidates(J, pop), "label": "DERIVED", "source": "sum of |VUs of r| over r in J (the stage-2 candidate count, fixed by J)"}
    _write(out, doc)
    if seed_file:
        # legacy custody: the seed lies in the record directory from now until the reveal (R7: readable by a co-located prover)
        (out / "seed.hex").write_text(seed.hex() + "\n")
    else:
        _HELD_SEEDS[str(out.resolve())] = seed
    return doc


def stage2(instance: Path, population: Path, interior_digest: str, out: Path, *, instances: int | None = None,
           exemptions: list[int] | tuple[int, ...] | None = None, committed_rus: list[int] | tuple[int, ...] | None = None,
           instance_format: str | None = None) -> dict[str, Any]:
    """Step 4 of the contract.  ``exemptions`` is the prover's request F (RU indices of J it cannot or will not supply);
    ``committed_rus`` the RUs whose step-3 values the interior commitment holds (the barrier check: every RU of J is committed or
    exempted).  Both are optional for a legacy instance (F = ∅, no barrier check); REQUIRED when the instance declares an epoch."""

    _li, _lp = loaders(instance_format)
    inst = _li(instance)
    pop = _lp(population, check="fast", instances=instances)
    bind(inst, pop)
    doc = _read(out)
    if doc is not None and instance_format is not None and doc.get("instance_format") not in (None, instance_format):
        raise ChronologyError(f"instance format {instance_format!r} != the chronology's {doc.get('instance_format')!r}")
    if doc is None:
        raise ChronologyError("no chronology: stage 1 has not run")
    if doc["instance_digest"] != instance_digest(inst):
        raise ChronologyError("chronology belongs to a different instance")
    _pinned(doc, pop)
    if any(e["name"] == "r_s" for e in doc["events"]):
        raise ChronologyError("r_s already drawn for this instance: a second inner draw is a reroll and is refused")
    rq = next((e for e in doc["events"] if e["name"] == "r_q"), None)
    if rq is None:
        raise ChronologyError("no r_q in the chronology")
    if len(interior_digest) != 64:
        raise ChronologyError("interior commitment digest must be a sha256 hex")
    if any(e["name"] == "exemption_request" for e in doc["events"]):
        raise ChronologyError("an exemption request is already recorded for this instance: F is submitted once, with the interior commitment")
    seed = _seed_for(out, doc)
    if _h(SEED_DOMAIN, seed) != next(e["digest"] for e in doc["events"] if e["name"] == "seed_commitment"):
        raise ChronologyError("seed does not open the recorded seed commitment")
    s = Fraction(inst.sampler.law.s)
    J = tuple(rq["selection"])
    Jset = set(J)
    # ---- [A3 R8] the exemption request F: selected RUs only, within the RU-count budget, complete before r_s ----
    contract = inst.epoch is not None
    if contract and (exemptions is None or committed_rus is None):
        raise ChronologyError("the instance declares an epoch: stage 2 needs the prover's exemption request F (exemptions; [] if none) and the RUs the "
                              "interior commitment covers (committed_rus) -- the step-4 checks are not optional")
    F = tuple(sorted(set(int(r) for r in (exemptions or ()))))
    if len(F) != len(list(exemptions or ())):
        raise ChronologyError("exemption request names an RU twice")
    if not set(F) <= Jset:
        raise ChronologyError(f"exemption request names RU(s) {sorted(set(F) - Jset)} outside J = {list(J)}: an exemption is a SELECTED RU (F ⊆ J)")
    budget = inst.exemption_budget
    if len(F) > budget:
        raise ChronologyError(f"exemption request |F| = {len(F)} exceeds the epoch's budget B_ex = {budget} (an RU count, one budget per epoch)")
    retained = tuple(r for r in J if r not in set(F))
    if committed_rus is not None:
        committed = set(int(r) for r in committed_rus)
        if not committed <= Jset:
            raise ChronologyError(f"interior commitment holds RU(s) {sorted(committed - Jset)} outside J (the prover replays exactly J)")
        missing = [r for r in retained if r not in committed]
        if missing:
            raise ChronologyError(f"retained RU(s) {missing} have no step-3 commitment and are not in F: r_s is derived only after every RU of J \\ F has its "
                                  "interior values committed (no streaming relaxation)")
    fd = exemption_digest(F)
    _event(doc, "commitment", "interior", interior_digest, ["r_q"],
           note=(f"prover's commitment to the retained interiors of J \\ F ({len(retained)} RU(s))" if contract else
                 "prover's commitment to the retained interiors of J (legacy instance: no exemption request; F = ∅)"),
           **({"committed_rus": sorted(committed)} if committed_rus is not None else {}))
    if contract or F:
        _event(doc, "message", "exemption_request", fd, ["r_q", "interior"], exemptions=list(F), retained=list(retained), B_ex=budget,
               note="F ⊆ J, |F| <= B_ex checked; r_s is drawn over the VUs of J \\ F only")
        s3 = step3_digest(interior_digest, fd)
    else:
        s3 = interior_digest  # legacy phase_s (no exemption request exists in the chronology)
    phase = phase_digest_s(doc["instance_digest"], rq["digest"], s3)
    Q = draw_Q(seed, phase, retained, pop, s)
    count = candidates(retained, pop)
    cd = challenge_digest("s", phase, count, s, Q)
    _event(doc, "challenge", "r_s", cd, ["r_q", "interior"] + (["exemption_request"] if (contract or F) else []), phase_digest=phase, selection=list(Q),
           candidates=count, **({"step3_digest": s3} if (contract or F) else {}))
    doc["challenges_drawn"].append(cd)
    doc["exemptions"] = {"F": list(F), "retained": list(retained), "B_ex": budget, "B_ex_unit": "replay_units",
                         "note": "contract R8: F ⊆ J, |F| <= B_ex; Q over J \\ F" if contract else "legacy instance: no exemption request; F = ∅"}
    doc["realized"]["Q_size"] = {"value": len(Q), "label": "MEASURED", "source": "chronology.json events[r_s].selection (realized, not the rate s)"}
    doc["realized"]["Q_candidates"] = {"value": count, "label": "MEASURED", "source": "sum of |VUs of r| over r in J \\ F"}
    doc["realized"]["F_size"] = {"value": len(F), "label": "MEASURED", "source": "chronology.json events[exemption_request].exemptions"}
    _write(out, doc)
    write_selected(out, doc, pop, Q)
    return doc


def write_selected(out: Path, doc: dict[str, Any], pop: PopulationView, Q: tuple[int, ...]) -> dict[str, Any]:
    """``selected.json``: the selected VU identities of the validated partition, each with its RU, kind and gate interval."""

    rs = next(e for e in doc["events"] if e["name"] == "r_s")
    kinds: dict[str, int] = {}
    rows = []
    for vu in Q:
        k = pop.kind_of(vu)
        iv = pop.gate_interval(vu)
        name = None if k is None else f"{k['fn']}|{k['form']}"
        if name is not None:
            kinds[name] = kinds.get(name, 0) + 1
        rows.append({"vu_index": vu, "ru_index": pop.ru_of(vu), "kind": k, "gate_interval": list(iv) if iv else None, "partition_digest": pop.partition_digest})
    sel = {"schema": SELECTED_SCHEMA, "instance_digest": doc["instance_digest"], "partition_digest": pop.partition_digest, "plan_digest": pop.plan_digest,
           "program_digest": pop.program_digest, "population": {"schema": pop.schema, "content_sha256": pop.content_sha256},
           "r_s": rs["digest"], "r_q": next(e["digest"] for e in doc["events"] if e["name"] == "r_q"),
           "count": len(Q), "candidates": rs.get("candidates"), "kinds": dict(sorted(kinds.items())),
           "note": ("kind / gate_interval are null for a population/v1 file (no kind table); population/v2 carries them" if pop.schema == POPULATION_V1 else
                    "each row is one VU of the validated partition; e2e.predicates maps it to its required proof predicate or REFUSES"),
           "selected": rows}
    (out / "selected.json").write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n")
    return sel


def reveal(out: Path) -> dict[str, Any]:
    doc = _read(out)
    if doc is None or not any(e["name"] == "r_s" for e in doc["events"]):
        raise ChronologyError("seed is revealed only after r_s (the second challenge) exists")
    if any(e["name"] == "seed_reveal" for e in doc["events"]):
        return doc
    _event(doc, "message", "seed_reveal", None, ["r_s"], seed=_seed_for(out, doc).hex())
    _write(out, doc)
    _HELD_SEEDS.pop(str(out.resolve()), None)
    return doc


def verify(instance: Path, population: Path, out: Path, *, instances: int | None = None, instance_format: str | None = None) -> list[str]:
    """Re-derive J and Q from the revealed seed and the recorded commitments over the given population; list every disagreement
    (population not the pinned one, not the instance's partition, selection outside J, missing draws, rerolls)."""

    problems: list[str] = []
    _li, _lp = loaders(instance_format)
    inst = _li(instance)
    try:
        pop = _lp(population, check="full", instances=instances)
        bind(inst, pop)
    except (PopulationError, ChronologyError) as e:
        return [f"population: {e}"]
    doc = _read(out)
    if doc is None:
        return ["no chronology.json"]
    try:
        _pinned(doc, pop)
    except ChronologyError as e:
        problems.append(str(e))
    if doc["instance_digest"] != instance_digest(inst):
        problems.append("chronology instance digest != instance")
    if instance_format is not None and doc.get("instance_format") not in (None, instance_format):
        problems.append(f"instance format {instance_format!r} != the chronology's {doc.get('instance_format')!r}")
    ev = {e["name"]: e for e in doc["events"]}
    for name in ("protocol_instance", "seed_commitment", "boundary", "r_q", "interior", "r_s", "seed_reveal"):
        if name not in ev:
            problems.append(f"chronology lacks event {name!r}")
    if problems:
        return problems
    orders = [e["order"] for e in doc["events"]]
    if orders != sorted(orders):
        problems.append("events out of order")
    for a, b in (("seed_commitment", "r_q"), ("boundary", "r_q"), ("r_q", "interior"), ("interior", "r_s"), ("r_s", "seed_reveal")):
        if ev[a]["order"] >= ev[b]["order"]:
            problems.append(f"{a} does not precede {b}")
    seed = bytes.fromhex(ev["seed_reveal"]["seed"])
    if _h(SEED_DOMAIN, seed) != ev["seed_commitment"]["digest"]:
        problems.append("revealed seed does not open the seed commitment")
    # [A3 R16] the epoch barrier at step 2: the chronology's epoch record must be the instance's and the boundary must cover every declared request
    if inst.epoch is not None:
        epr = doc.get("epoch") or {}
        if epr.get("epoch_id") != inst.epoch.epoch_id or epr.get("request_ids") != list(inst.epoch.request_ids):
            problems.append("chronology epoch record != the instance's epoch declaration")
        if sorted(epr.get("boundary_request_ids", [])) != sorted(inst.epoch.request_ids):
            problems.append("boundary commitment did not cover every request of the declared epoch before r_q")
        if epr.get("B_ex") != inst.epoch.B_ex:
            problems.append("chronology epoch B_ex != the instance's")
    q, s = Fraction(inst.sampler.law.q), Fraction(inst.sampler.law.s)
    phase_q = phase_digest_q(doc["instance_digest"], ev["boundary"]["digest"])
    # [L1 RECORD] logged redraws (conditioning on >= 1 selected RU): every empty attempt must re-derive as empty under its phase
    empties = sorted((e for e in doc["events"] if e["name"].startswith("r_q_empty[")), key=lambda e: e["order"])
    for k, e in enumerate(empties):
        if e["phase_digest"] != phase_q:
            problems.append(f"{e['name']} phase digest does not re-derive")
        Jk = draw_J(seed, phase_q, pop.ru_count, q)
        if Jk or e["selection"] or challenge_digest("q", phase_q, pop.ru_count, q, Jk) != e["digest"]:
            problems.append(f"{e['name']} is not an empty draw under its phase")
        if e["order"] >= ev["r_q"]["order"] or e["order"] <= ev["boundary"]["order"]:
            problems.append(f"{e['name']} is not between boundary and r_q")
        phase_q = phase_digest_q_redraw(phase_q, k + 1)
    if int(ev["r_q"].get("redraws", 0)) != len(empties):
        problems.append(f"r_q records {ev['r_q'].get('redraws', 0)} redraws but the chronology holds {len(empties)} empty draws")
    if phase_q != ev["r_q"]["phase_digest"]:
        problems.append("r_q phase digest != H(instance, boundary)" + (" (after the logged redraws)" if empties else ""))
    J = draw_J(seed, phase_q, pop.ru_count, q)
    if list(J) != ev["r_q"]["selection"]:
        problems.append("r_q selection does not re-derive")
    if challenge_digest("q", phase_q, pop.ru_count, q, J) != ev["r_q"]["digest"]:
        problems.append("r_q digest does not re-derive")
    # [A3 R8/R16] the exemption request: F ⊆ J, |F| <= B_ex, recorded between interior and r_s, bound into phase_s; Q over J \\ F
    Jset = set(J)
    F: tuple[int, ...] = ()
    ex = ev.get("exemption_request")
    if inst.epoch is not None and ex is None:
        problems.append("the instance declares an epoch but the chronology holds no exemption_request event (F must be submitted, possibly empty, before r_s)")
    if ex is not None:
        F = tuple(sorted(set(ex.get("exemptions", []))))
        if list(F) != list(ex.get("exemptions", [])):
            problems.append("exemption_request.exemptions is not a sorted set of RU indices")
        if not set(F) <= Jset:
            problems.append(f"exemption request names RU(s) {sorted(set(F) - Jset)} outside J")
        if len(F) > inst.exemption_budget:
            problems.append(f"|F| = {len(F)} exceeds the instance's B_ex = {inst.exemption_budget}")
        if exemption_digest(F) != ex["digest"]:
            problems.append("exemption_request digest does not re-derive from F")
        if not (ev["interior"]["order"] < ex["order"] < ev["r_s"]["order"]):
            problems.append("exemption_request is not between interior and r_s (a late or premature exemption request)")
        s3 = step3_digest(ev["interior"]["digest"], ex["digest"])
        if ev["r_s"].get("step3_digest") != s3:
            problems.append("r_s.step3_digest != H(STEP3, interior, exemption_request)")
        if inst.epoch is not None:
            committed = set(ev["interior"].get("committed_rus", []))
            missing = [r for r in J if r not in set(F) and r not in committed]
            if missing:
                problems.append(f"retained RU(s) {missing} have no recorded step-3 commitment (epoch barrier before r_s)")
    else:
        s3 = ev["interior"]["digest"]
    retained = tuple(r for r in J if r not in set(F))
    phase_s = phase_digest_s(doc["instance_digest"], ev["r_q"]["digest"], s3)
    if phase_s != ev["r_s"]["phase_digest"]:
        problems.append("r_s phase digest != H(instance, r_q, " + ("H(STEP3, interior, exemption_request))" if ex is not None else "interior)"))
    Q = draw_Q(seed, phase_s, retained, pop, s)
    if list(Q) != ev["r_s"]["selection"]:
        problems.append("r_s selection does not re-derive")
    if challenge_digest("s", phase_s, candidates(retained, pop), s, Q) != ev["r_s"]["digest"]:
        problems.append("r_s digest does not re-derive")
    Rset = set(retained)
    for vu in ev["r_s"]["selection"]:
        if not 0 <= vu < pop.vu_count:
            problems.append(f"sampled VU {vu} outside the population [0, {pop.vu_count})")
        elif pop.ru_of(vu) not in Jset:
            problems.append(f"sampled VU {vu} lies in RU {pop.ru_of(vu)} which is not in J")
        elif pop.ru_of(vu) not in Rset:
            problems.append(f"sampled VU {vu} lies in exempted RU {pop.ru_of(vu)} (Q must be drawn over J \\ F)")
    drawn = [e["digest"] for e in doc["events"] if e["kind"] == "challenge"]
    if drawn != doc["challenges_drawn"]:
        problems.append("challenges_drawn != the challenge events (a draw is missing from the record)")
    if doc.get("rerolls", 0) != 0:
        problems.append("rerolls != 0")
    sel_path = out / "selected.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        if [r["vu_index"] for r in sel["selected"]] != ev["r_s"]["selection"] or sel.get("r_s") != ev["r_s"]["digest"]:
            problems.append("selected.json does not list exactly the r_s selection")
        if sel.get("partition_digest") != pop.partition_digest:
            problems.append("selected.json partition_digest != population")
        for r in sel["selected"]:
            if r["ru_index"] != pop.ru_of(r["vu_index"]):
                problems.append(f"selected.json: VU {r['vu_index']} names RU {r['ru_index']} but the population says {pop.ru_of(r['vu_index'])}")
                break
            # [A3, integrator 19:50Z] the located unit is derived from these two fields by the verifier (e2e.verifier_v3.locator): they must
            # be the population's, not a relabelling in the file
            iv = pop.gate_interval(r["vu_index"])
            if iv is not None and list(iv) != list(r.get("gate_interval") or []):
                problems.append(f"selected.json: VU {r['vu_index']} gate_interval {r.get('gate_interval')} != population {list(iv)}")
                break
            k = pop.kind_of(r["vu_index"])
            if k is not None and (r.get("kind") or {}).get("fn") is not None and (k.get("fn"), k.get("form")) != (r["kind"].get("fn"), r["kind"].get("form")):
                problems.append(f"selected.json: VU {r['vu_index']} kind {r['kind'].get('fn')}|{r['kind'].get('form')} != population {k.get('fn')}|{k.get('form')}")
                break
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage1", "stage2", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--instance", required=True)
        sp.add_argument("--population", required=True, help="population.json (v1) or population.compact.json (v2)")
        sp.add_argument("--out", required=True)
        sp.add_argument("--instances", type=int, default=None, help="[L1 RECORD] program instances (requests): a per-instance population/v2 file is viewed tiled "
                                                                     "(e2e.population replicate's law) -- must equal the instance's partition.instances")
        if name == "stage1":
            sp.add_argument("--boundary-digest", required=True)
            sp.add_argument("--seed", default=None, help="32-byte hex (tests only; default: fresh OS randomness)")
            sp.add_argument("--condition-nonempty", action="store_true", help="[L1 RECORD] DIRECTIVE §1.5.10: log empty r_q draws and redraw under a derived phase "
                                                                              "until >= 1 RU is selected (certificate: 'RU draw conditioned on >= 1 selected (demonstration)')")
        if name == "stage1":
            sp.add_argument("--legacy-seed-file", action="store_true", help="[A3 R7 / A10 WR-2] LEGACY: write seed.hex into the record directory (prover-readable). Default: the seed is held in this process (stage2 must then run here)")
            sp.add_argument("--no-seed-file", action="store_true", help="(pre-WR-2 spelling of the default; accepted, no effect)")
            sp.add_argument("--boundary-request-id", action="append", default=None, help="[A3 R16] a request id the boundary commitment covers (repeat)")
        if name == "stage2":
            sp.add_argument("--interior-digest", required=True)
            sp.add_argument("--exempt", type=int, action="append", default=None, help="[A3 R8] RU index of J the prover requests exempt (repeat; F)")
            sp.add_argument("--committed-ru", type=int, action="append", default=None, help="[A3 R16] RU index whose interior the commitment holds (repeat)")
    rv = sub.add_parser("reveal")
    rv.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stage1":
            doc = stage1(Path(a.instance), Path(a.population), a.boundary_digest, Path(a.out), a.seed, instances=a.instances, condition_nonempty=a.condition_nonempty,
                         legacy_seed_file=a.legacy_seed_file, boundary_request_ids=a.boundary_request_id)
            print(f"[partition_sample] r_q drawn: |J| = {doc['realized']['J_size']['value']} of N_R = {doc['population']['ru_count']} (q = {doc['law']['q']}); "
                  f"stage-2 candidates {doc['realized']['Q_candidates_expected']['value']}; population {doc['population']['schema']} {doc['population']['content_sha256'][:12]}"
                  + (f"; redraws {doc['realized']['r_q_redraws']['value']} ({REDRAW_NOTE})" if doc["realized"].get("r_q_redraws") else ""))
        elif a.cmd == "stage2":
            doc = stage2(Path(a.instance), Path(a.population), a.interior_digest, Path(a.out), instances=a.instances, exemptions=a.exempt, committed_rus=a.committed_ru)
            print(f"[partition_sample] r_s drawn: |Q| = {doc['realized']['Q_size']['value']} of {doc['realized']['Q_candidates']['value']} candidates (s = {doc['law']['s']}); "
                  f"F = {doc['exemptions']['F']} (B_ex = {doc['exemptions']['B_ex']} RUs) -> selected.json")
        elif a.cmd == "reveal":
            reveal(Path(a.out))
            print("[partition_sample] seed revealed")
        else:
            problems = verify(Path(a.instance), Path(a.population), Path(a.out), instances=a.instances)
            for pr in problems:
                print(f"[partition_sample]   problem: {pr}")
            print(f"[partition_sample] {'REDERIVED' if not problems else 'MISMATCH'} ({len(problems)} problem(s))")
            return 0 if not problems else 1
    except (ChronologyError, InstanceError, PopulationError) as e:
        print(f"[partition_sample] REFUSED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
