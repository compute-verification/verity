"""Sampling population and address maps from the VALIDATED VU/RU partition (brief A.2/A.3; lane PROTO-A, A2).

The population the security sampler draws from is the verification-unit set of one validated plan of the pinned
program -- never the capture's physical committed leaves (GUIDANCE §50: 34,092 leaves vs 33,968 program values) and
never a rounded analytical census.  This module:

* :func:`plan_descriptor_digest` -- the digest that pins a plan to its program (``sha256`` of the bridged program
  name + ``VerificationPlan.to_json()``), the ``partition.plan_descriptor_digest`` of the protocol instance;
* :func:`population` -- exact ``N_V`` / ``N_R`` from the plan + verdict (``len`` of the unit sequences; the RU of
  every VU is the validator's ``Obligation.commitment_unit``);
* :class:`Addresser` -- ``index -> Address`` for a sampled VU: program value(s) as compact gate runs
  (``Circuit.resolve_run`` / ``GateRun``: no per-gate objects), the function call (identity + root-scoped gate
  interval), the RU index, the committed positions (a stable position key per output gate run: what a capture's
  leaf table must map to), and the proof obligation (``Obligation`` index + the relation the unit check is);
* :func:`write_maps` -- the explicit maps as documented TSV tables (``value_map.tsv``, ``vu_map.tsv``,
  ``ru_map.tsv``) + a JSON header, generated for TINY as the test fixture (``tests/vectors/proto-a-tiny/``).

Nothing here materialises a Python object per VU beyond what the existing ``VerificationPlan`` already holds
(one ``Description`` per unit -- SUMMARY-C's run-aware ``Plan`` replaces that; this interface is kept small so it
can be re-implemented over it: ``population(plan, verdict)`` and ``Addresser.resolve(i)``).  Per-VU output values
are resolved lazily and returned as ``GateRun`` s.

    python -m e2e.population tiny --out tests/vectors/proto-a-tiny        # TINY fixture (LP=4, STEPS=8, root + P3/cu=block + P1/cu=operator)
    python -m e2e.population digest --case B0 --lp 32 --steps 15 --plan P3/cu=block   # plan digest only (no validation)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritor.plans.summary import PARTITION_DIGEST, partition_digest, plan_digest  # noqa: E402  (SUMMARY-C, agreement 2: one plan digest convention across lanes)

MAPS_SCHEMA = "veritor/population-maps/v1"
POPULATION_SCHEMA = "veritor/population/v1"
TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32,
        "V": 64, "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
        "tie_word_embeddings": True, "profile": {}}
TINY_LP, TINY_STEPS = 4, 8


def sha256_json(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# -- building ---------------------------------------------------------------------------------------------------------------


LEGACY_KBLOCK_N = 64
"""[E6 CURVE 2026-09-13, D57] the FA2 key block of the LEGACY program (`Serve_v1`, `b1.BN`): what an identity WITHOUT `attention` means."""


def attention_kblock_n(attention: dict[str, Any] | None) -> int:
    """kBlockN of a pinned identity's declared geometry (`program.identity.attention`, E7's `Serve_v4{C,LP,STEPS,BN}`); the legacy 64 when the
    identity carries none.  The population RULE's basis (`census_v2.small_scaling.terms`) is evaluated at this block."""
    return int(attention["kBlockN"]) if attention else LEGACY_KBLOCK_N


def build(config: dict[str, Any], lp: int, steps: int, *, digest: bool = True, family: str | None = None, attention: dict[str, Any] | None = None):
    """verity.ir ``<family>{C,LP,STEPS[,BN]}`` -> compact plan IR.  Returns ``(program, adapted, circuit, program_digest)``.
    ``family`` / ``attention``: the identity's serving family and declared kernel geometry (`b1.serve_program`): None -> `Serve_v1`
    (legacy kBlockN 64); "Serve_v4" + {kernel, kBlockM, kBlockN, num_splits} -> the declared geometry [E7 D54/D57 2026-09-13]."""

    from verity.ir import Program, bind
    from verity_vllm.registry import b1
    from veritor.plans.verity_ir_adapter import adapt

    program = Program(b1.serve_program(config, lp, steps, family=family, attention=attention))
    pdigest = None
    if digest:
        from verity.ir import encode_program, program_digest

        pdigest = program_digest(encode_program(program))
    adapted = adapt(program, compact=True)
    return program, adapted, adapted.program.circuit(), pdigest


COMPACT_LEVELS = ("gate", "coordinate")
"""Canonical levels whose plans are built as ``UnitRuns`` by default (REC-15, lane COMPACT-P1): P0/P1 have one
unit per scalar coordinate (whole B0: 166 M units, the 105 GB wall as ``Parts`` objects); P2/P3/block plans stay
dense so their recorded ``plan_descriptor_digest`` pins (``plan-json/v1``, e.g. P3/cu=block ``2ba7a48c…``) are unchanged.
The ``partition_digest`` is representation-independent either way."""


def make_plan(spec: str, adapted, circuit, *, compact: bool | None = None):
    """``root`` (bridge_diag's P1: one VU per root node, CU(inputs)+CU(calls)) or a canonical ``P<k>/cu=<strategy>``
    -- the same constructors as ``scripts/compact_whole_b0.py`` so digests agree with COMPACT-2's runs.

    ``compact``: ``None`` (default) builds the levels in :data:`COMPACT_LEVELS` as ``UnitRuns`` and the others dense;
    ``True``/``False`` force the representation.  Dense and compact plans of one spec are the same partition
    (``partition_digest`` equal; ``tests/test_population_compact.py``); only ``plan_digest_kind`` differs."""

    from veritor.plans.boundary import Part, Parts
    from veritor.plans.plan import VerificationPlan

    Q = adapted.program
    if spec == "root":
        root = Q.root
        inputs = [k for k, n in enumerate(root.nodes) if n.form == "batch" and n.fn.name == "Input"]
        calls = [k for k in range(len(root.nodes)) if k not in inputs]
        vus = [Parts(circuit, [Part((), k, 0, root.nodes[k].n)]) for k in inputs + calls]
        cus = [Parts(circuit, [Part((), k, 0, root.nodes[k].n) for k in g]) for g in (inputs, calls) if g]
        return VerificationPlan("root", vus, cus), {"vu_level": "root-node", "cu_strategy": "inputs|calls"}
    from veritor.plans.canonical import PLAN_NAMES, PRESCRIBED_SUFFIX, Classifier, canonical_plan

    # [A1 2026-09-10, contract R2] ``P<k>/cu=<strategy>/inputs=prescribed``: the root Input batches in no VU and no CU
    inputs = "units"
    if spec.endswith(PRESCRIBED_SUFFIX):
        inputs, spec_core = "prescribed", spec[: -len(PRESCRIBED_SUFFIX)]
    else:
        spec_core = spec
    head, _, cu = spec_core.partition("/cu=")
    level = PLAN_NAMES[head]
    if compact is None:
        compact = level in COMPACT_LEVELS
    runs: dict[str, list[tuple[int, int]]] = {}
    for fid, original, members in adapted.unrolled:
        runs.setdefault(fid, []).append((adapted.node_index[(fid, original)], members))
    cp = canonical_plan(circuit, level, cu or "whole", Classifier({k: tuple(sorted(v)) for k, v in runs.items()}), name=spec, compact=compact, inputs=inputs)
    meta = {"vu_level": cp.vu_level, "cu_strategy": cp.cu_strategy, "representation": cp.plan.representation + (" (UnitRuns)" if cp.plan.compact else ""),
            "inputs": cp.inputs}
    if inputs == "prescribed":
        meta["prescribed_outside"] = [list(iv) for iv in cp.prescribed_outside]
        meta["prescribed_outside_sides"] = ["verification", "commitment"]
    return cp.plan, meta


def validate_kwargs(meta: dict) -> dict:
    """The ``validate_plan`` keywords a plan from :func:`make_plan` needs: the prescribed-outside intervals and sides of an
    ``inputs=prescribed`` plan, nothing for the historical construction."""

    if meta.get("inputs") != "prescribed":
        return {}
    return {"prescribed_outside": [tuple(iv) for iv in meta["prescribed_outside"]], "prescribed_outside_sides": tuple(meta["prescribed_outside_sides"])}


def plan_descriptor_digest(Q, plan) -> str:
    """The plan's digest, SUMMARY-C's convention (agreement 2, program-summary-format.md §2): ``veritor.plans.summary.plan_digest`` --
    tagged over ``plan.to_json()`` (``plan-json/v1``) or over the constructor rule of a compact plan (``plan-rule/v1``).  The plan is
    pinned to its program by the artifact's separate ``program_digest`` (the plan JSON names gates of that program only)."""

    return plan_digest(plan)[1]


def plan_digest_kind(plan) -> str:
    return plan_digest(plan)[0]


# -- population ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    vu_count: int
    ru_count: int
    plan_name: str
    plan_digest: str
    program_digest: str | None
    status: str
    partition_digest: str = ""
    """``N_V``, ``N_R`` counted on the validated plan (MEASURED); ``status`` is the validator's verdict.  ``plan_digest`` pins the
    plan's *constructor/representation* (``plan_digest_kind``); ``partition_digest`` (``partition-intervals/v1``, SUMMARY-C
    objection 1) pins the *partition itself* -- equal for a dense plan and its ``UnitRuns`` form -- and is what the instance's
    population refers to."""


def population(Q, plan, verdict, program_digest: str | None) -> Population:
    if verdict.status != "valid":
        raise ValueError(f"plan {plan.name!r} is {verdict.status}: an unvalidated partition defines no population")
    if getattr(verdict, "retained", True):
        if len(verdict.obligations) != len(plan.verification_units):
            raise ValueError("verdict obligations do not cover every verification unit")
    elif verdict.vu_range is not None or verdict.vu_stats.units != len(plan.verification_units):
        raise ValueError("streamed verdict did not compute every verification unit's boundary")
    return Population(len(plan.verification_units), len(plan.commitment_units), plan.name, plan_descriptor_digest(Q, plan), program_digest,
                      verdict.status, partition_digest(plan))


# -- addressing ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueRun:
    """``count`` program values at root gates ``gate + j*stride`` (one ``GateRun``), each ``width`` bits, produced by ``prim``."""

    gate: int
    count: int
    stride: int
    width: int
    prim: str

    @property
    def position_key(self) -> str:
        """The committed-position key: the canonical root gate run.  A capture's leaf table maps its leaves to these keys
        (L2/L3 correspondence); the protocol never addresses a leaf the program does not produce."""

        return f"g={self.gate}+{self.stride}x{self.count}"


@dataclass(frozen=True)
class Address:
    index: int
    ru: int
    kind: str
    function: str
    path: tuple[tuple[int, int], ...]
    interval: tuple[int, int]
    size: int
    out_bits: int
    in_bits: int
    outputs: tuple[ValueRun, ...]
    obligation: int
    relation: str

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "ru": self.ru, "kind": self.kind, "function": self.function, "path": [list(s) for s in self.path],
                "interval": list(self.interval), "size": self.size, "out_bits": self.out_bits, "in_bits": self.in_bits,
                "outputs": [{"gate": v.gate, "count": v.count, "stride": v.stride, "width": v.width, "prim": v.prim, "position": v.position_key}
                            for v in self.outputs],
                "obligation": self.obligation, "relation": self.relation}


class Addresser:
    """``resolve(i)`` for a sampled VU index of a validated plan; ``O(parts + output runs)`` per call, no per-gate objects."""

    def __init__(self, Q, circuit, plan, verdict) -> None:
        if verdict.status != "valid":
            raise ValueError("cannot address units of an unvalidated plan")
        self.Q, self.circuit, self.plan, self.verdict = Q, circuit, plan, verdict
        self.count = len(plan.verification_units)

    def _call(self, desc):
        """The single function call a unit is, when it is exactly one (root-scoped path + node); else None."""

        parts = getattr(desc, "parts", None)
        if not parts or len(parts) != 1:
            return None
        part = parts[0]
        if part.b - part.a != 1:
            scope = self.circuit.scope_at(part.path)
            node = scope.fn.nodes[part.node_idx]
            return ("batch", part.path, part.node_idx, node, part.a)
        # member ``part.a`` of the node, not member 0 (SUMMARY-C objection 2, 2026-09-09: every ``Part(path, k, a, a+1)`` with a > 0 --
        # all P1/coordinate units of a batch node -- was resolved to member 0's call, so gate_lo/gate_hi/path named the wrong unit)
        return ("call", part.path, part.node_idx, None, part.a)

    def resolve(self, i: int) -> Address:
        if not 0 <= i < self.count:
            raise IndexError(f"sampled index {i} outside the population [0, {self.count})")
        desc = self.plan.verification_units[i]
        ob = self.verdict.obligations[i]
        b = ob.boundary
        outs = tuple(ValueRun(g, n, s, w, "") for g, n, s, w in _runs_with_width(b.outputs))
        c = self._call(desc)
        if c is None:
            kind, function, path = "gates", "(explicit gate set)", ()
            lo, hi = desc.intervals.intervals()[0][0], desc.intervals.intervals()[-1][1]
        elif c[0] == "call":
            fc = self.circuit.call((*c[1], (c[2], c[4])))
            kind = "primitive" if fc.is_primitive else "call"
            function, path = fc.name, fc.path()
            lo, hi = fc.interval
            span = (desc.intervals.intervals()[0][0], desc.intervals.intervals()[-1][1])
            if (lo, hi) != span:  # the unit's own gate span is the ground truth; the call object must describe exactly it
                raise ValueError(f"VU {i}: call {path} spans {lo, hi} but the unit's intervals span {span}")
        else:
            _, path, node_idx, node, member = c
            kind, function = f"batch[{node.n}]", node.fn.name
            lo, hi = desc.intervals.intervals()[0][0], desc.intervals.intervals()[-1][1]
            path = (*path, (node_idx, member))
        relation = "input-constant" if function.startswith("Input") else f"unit-check f_V({function})"
        return Address(i, ob.commitment_unit, kind, function, tuple(path), (lo, hi), desc.size, b.out_bits, b.in_bits, outs, ob.index, relation)

    def __iter__(self) -> Iterator[Address]:
        for i in range(self.count):
            yield self.resolve(i)


def ru_membership(verdict, plan=None) -> list[int]:
    """``ru_of[vu]`` for every VU of a validated plan (one int per VU: the validator's ``Obligation.commitment_unit``).
    The inner Bernoulli draw needs, per selected RU, its VU indices; :func:`vus_by_ru` groups this list.
    A streamed verdict (``validate_plan(..., retain=False)``) keeps no obligations: ``plan`` is then required and the
    owner is looked up arithmetically (the same ``Obligation.commitment_unit`` rule: the CU owning the VU's first gate)."""

    if verdict.status != "valid":
        raise ValueError("no RU membership for an unvalidated plan")
    if getattr(verdict, "retained", True):
        return [ob.commitment_unit for ob in verdict.obligations]
    if plan is None:
        raise ValueError("ru_membership of a streamed verdict needs the plan")
    from veritor.plans.runs import UnitRuns
    from veritor.plans.validate import _Owners

    owners = _Owners(plan.commitment_units)
    vus = plan.verification_units
    if isinstance(vus, UnitRuns):
        return [owners.owner(run.interval(j)[0]) or 0 for run in vus.runs for j in range(run.count)]
    return [owners.owner(unit.intervals.los[0]) or 0 for unit in vus]


def vus_by_ru(ru_of: list[int], ru_count: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(ru_count)]
    for vu, ru in enumerate(ru_of):
        if not 0 <= ru < ru_count:
            raise ValueError(f"VU {vu} names RU {ru} outside [0, {ru_count})")
        groups[ru].append(vu)
    return groups


def _runs_with_width(gm) -> Iterator[tuple[int, int, int, int]]:
    """``(gate, count, stride, width)`` for every run of a ``GateMap`` (compact: runs, not gates)."""

    for lo, hi, width in gm.runs():
        yield lo, hi - lo, 1, width


# -- the explicit maps ---------------------------------------------------------------------------------------------------------


VALUE_COLUMNS = ("position", "gate", "count", "stride", "width_bits", "vu", "ru", "obligation")
VU_COLUMNS = ("vu", "ru", "kind", "function", "path", "gate_lo", "gate_hi", "size", "out_bits", "in_bits", "n_output_runs", "obligation", "relation")
RU_COLUMNS = ("ru", "n_vus", "first_vu", "last_vu", "size")


class _NullWriter:
    def writerow(self, row) -> None:  # noqa: D401 - csv.writer duck type
        pass


def write_maps(out: Path, Q, circuit, plan, verdict, program_digest: str | None, meta: dict[str, Any], *, tables: bool = True) -> dict[str, Any]:
    """``value_map.tsv`` (committed position -> VU/RU/obligation), ``vu_map.tsv``, ``ru_map.tsv`` + ``maps.json`` header.
    Table format documented in protocol-instance.md §A2.  ``tables=False``: every VU is still resolved and counted (the header's
    ``counts`` are exact) but the three TSVs are not materialized (fixtures with N_V in the tens of thousands)."""

    out.mkdir(parents=True, exist_ok=True)
    pop = population(Q, plan, verdict, program_digest)
    if not getattr(verdict, "retained", True):
        raise ValueError("write_maps needs a retained verdict (per-VU boundaries/obligations); a streamed verdict gives counts and digests only "
                         "(e2e.population maps --stream writes compact_validation.json + index.json)")
    addr = Addresser(Q, circuit, plan, verdict)
    ru_rows: dict[int, dict[str, Any]] = {}
    n_values = n_runs = 0
    t = time.perf_counter()
    names = ("value_map.tsv", "vu_map.tsv", "ru_map.tsv") if tables else ()
    with contextlib.ExitStack() as stack:
        if tables:
            fv = stack.enter_context((out / "value_map.tsv").open("w", newline=""))
            fu = stack.enter_context((out / "vu_map.tsv").open("w", newline=""))
            wv, wu = csv.writer(fv, delimiter="\t"), csv.writer(fu, delimiter="\t")
        else:
            wv = wu = _NullWriter()
        wv.writerow(VALUE_COLUMNS)
        wu.writerow(VU_COLUMNS)
        for a in addr:
            for v in a.outputs:
                wv.writerow([v.position_key, v.gate, v.count, v.stride, v.width, a.index, a.ru, a.obligation])
                n_values += v.count
                n_runs += 1
            wu.writerow([a.index, a.ru, a.kind, a.function, "/".join(f"{n}.{m}" for n, m in a.path), a.interval[0], a.interval[1], a.size,
                         a.out_bits, a.in_bits, len(a.outputs), a.obligation, a.relation])
            r = ru_rows.setdefault(a.ru, {"ru": a.ru, "n_vus": 0, "first_vu": a.index, "last_vu": a.index, "size": 0})
            r["n_vus"] += 1
            r["last_vu"] = a.index
            r["size"] += a.size
    if tables:
        with (out / "ru_map.tsv").open("w", newline="") as fr:
            wr = csv.writer(fr, delimiter="\t")
            wr.writerow(RU_COLUMNS)
            for ru in range(pop.ru_count):
                r = ru_rows.get(ru, {"ru": ru, "n_vus": 0, "first_vu": None, "last_vu": None, "size": 0})
                wr.writerow([r[c] for c in RU_COLUMNS])
    header = {
        "schema": MAPS_SCHEMA, "program": {"name": Q.name, "digest": program_digest, "gates": circuit.size}, **meta,
        "plan": {"name": plan.name, "descriptor_digest": pop.plan_digest, "digest_kind": plan_digest_kind(plan),
                 "partition_digest": pop.partition_digest, "partition_digest_kind": PARTITION_DIGEST,
                 "vus": pop.vu_count, "cus": pop.ru_count, "status": pop.status},
        "counts": {"program_values": n_values, "value_runs": n_runs, "vus": pop.vu_count, "rus": pop.ru_count,
                   "label": "MEASURED (counted on the validated plan: VU outgoing boundaries)"},
        "tables": ({"value_map.tsv": list(VALUE_COLUMNS), "vu_map.tsv": list(VU_COLUMNS), "ru_map.tsv": list(RU_COLUMNS)} if tables
                   else f"not materialized (N_V = {pop.vu_count}); regenerate with `e2e.population tiny --plans {plan.name} --tables-max-vus 0`"),
        "seconds": round(time.perf_counter() - t, 3),
        "sha256": {},
    }
    for name in names:
        header["sha256"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / "maps.json").write_text(json.dumps(header, indent=1, sort_keys=True) + "\n")
    # the sampler's population file (e2e.security_sample): counts + ru_of, pinned to program and plan digests
    popdoc = {"schema": POPULATION_SCHEMA, "vu_count": pop.vu_count, "ru_count": pop.ru_count, "ru_of": ru_membership(verdict),
              "plan_digest": pop.plan_digest, "plan_digest_kind": plan_digest_kind(plan),
              "partition_digest": pop.partition_digest, "partition_digest_kind": PARTITION_DIGEST,
              "program_digest": program_digest, "plan_name": plan.name, "instances": 1,
              "per_instance": {"vu_count": pop.vu_count, "ru_count": pop.ru_count},
              "label": "MEASURED (validated plan: one RU index per VU, Obligation.commitment_unit)"}
    (out / "population.json").write_text(json.dumps(popdoc, sort_keys=True, separators=(",", ":")) + "\n")
    return header


def replicate_population(pop: dict[str, Any], instances: int) -> dict[str, Any]:
    """The RECORD-level population for ``instances`` program instances (requests) sharing one validated partition: VU/RU indices
    tiled per instance (global VU = r * N_V + i, global RU = r * N_R + j).  Each instance is the same program, so the same plan."""

    if pop.get("instances", 1) != 1:
        raise ValueError("replicate a per-instance population (instances == 1)")
    if instances < 1:
        raise ValueError("instances must be >= 1")
    n_v, n_r = pop["vu_count"], pop["ru_count"]
    ru_of = [r * n_r + ru for r in range(instances) for ru in pop["ru_of"]]
    return {**pop, "vu_count": instances * n_v, "ru_count": instances * n_r, "ru_of": ru_of, "instances": instances,
            "per_instance": {"vu_count": n_v, "ru_count": n_r},
            "label": pop["label"] + f"; tiled over {instances} program instance(s) (requests)"}


# -- the compact-validation artifact -------------------------------------------------------------------------------------------


def validation_artifact(Q, plan, verdict, program_digest: str, meta: dict[str, Any], *, seconds: float | None, maxrss_mb: float | None,
                        verdict_source: str) -> dict[str, Any]:
    """``veritor/compact-validation/v1``: the verdict pinned to program digest + plan descriptor digest + partition digest (what certificate.py v3 requires)."""

    art = {"schema": "veritor/compact-validation/v1", "program_digest": program_digest, "program": Q.name,
           "plan_name": plan.name, "plan_digest": plan_descriptor_digest(Q, plan), "plan_digest_kind": plan_digest_kind(plan),
           "partition_digest": partition_digest(plan), "partition_digest_kind": PARTITION_DIGEST, **meta,
           "status": verdict.status, "codes": [str(x) for x in verdict.codes], "vus": len(plan.verification_units), "cus": len(plan.commitment_units),
           "vu_edges": verdict.vu_edges, "cu_edges": verdict.cu_edges, "seconds": seconds, "maxrss_mb": maxrss_mb,
           "verdict_source": verdict_source, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not getattr(verdict, "retained", True):  # streamed verdict (COMPACT-P1): the per-side boundary totals stand in for the boundary lists
        art["validator"] = "validate_plan(retain=False)"
        art["vu_stats"], art["cu_stats"] = verdict.vu_stats.to_json(), verdict.cu_stats.to_json()
    return art


# -- CLI ---------------------------------------------------------------------------------------------------------------------


def _tiny(out: Path, plans: list[str], tables_max_vus: int) -> int:
    from veritor.plans.validate import validate_plan

    program, adapted, circuit, pdigest = build(TINY, TINY_LP, TINY_STEPS)
    Q = adapted.program
    index: dict[str, Any] = {"schema": "veritor/proto-a-tiny-fixture/v1", "config": "TINY (tiny_serve3.py)", "LP": TINY_LP, "STEPS": TINY_STEPS,
                             "program": {"name": Q.name, "digest": pdigest, "gates": program.gates, "input_gates": program.input_gates}, "plans": {}}
    for spec in plans:
        plan, meta = make_plan(spec, adapted, circuit)
        t = time.perf_counter()
        v = validate_plan(circuit, plan, **validate_kwargs(meta))
        secs = round(time.perf_counter() - t, 3)
        sub = out / spec.replace("/", "_").replace("=", "-")
        header = write_maps(sub, Q, circuit, plan, v, pdigest, {"vu_level": meta["vu_level"], "cu_strategy": meta["cu_strategy"]},
                            tables=tables_max_vus <= 0 or len(plan.verification_units) <= tables_max_vus)
        art = validation_artifact(Q, plan, v, pdigest, meta, seconds=secs, maxrss_mb=None, verdict_source="validate_plan in this process (TINY, laptop)")
        (sub / "compact_validation.json").write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
        index["plans"][spec] = {"dir": sub.name, "vus": header["plan"]["vus"], "cus": header["plan"]["cus"], "status": v.status,
                                "plan_digest": header["plan"]["descriptor_digest"], "partition_digest": header["plan"]["partition_digest"],
                                "program_values": header["counts"]["program_values"], "tables": isinstance(header["tables"], dict)}
        print(f"[population] TINY {spec}: {v.status}, N_V={header['plan']['vus']} N_R={header['plan']['cus']} values={header['counts']['program_values']} "
              f"plan digest {header['plan']['descriptor_digest'][:16]} ({secs}s validate)")
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny", help="generate the TINY fixture maps")
    t.add_argument("--out", default=str(POC / "tests" / "vectors" / "proto-a-tiny"))
    t.add_argument("--plans", default="root,P3/cu=block,P1/cu=operator")
    t.add_argument("--tables-max-vus", type=int, default=10000, help="write the TSV tables only for plans with at most this many VUs (0 = always)")
    d = sub.add_parser("digest", help="plan descriptor digest for a case (build + plan only; no validation)")
    d.add_argument("--case", default="B0")
    d.add_argument("--lp", type=int, default=32)
    d.add_argument("--steps", type=int, default=15)
    d.add_argument("--plan", default="P3/cu=block")
    d.add_argument("--out", default=None, help="write {program_digest, plan_digest, vus, cus, plan_name} JSON here")
    f = sub.add_parser("from-compact2", help="turn a COMPACT-2 whole-B0 run record (scripts/compact_whole_b0.py <out>.json) into a "
                                             "veritor/compact-validation/v1 artifact: program + plan rebuilt and digested HERE, verdict/counts "
                                             "taken from the pod run and required to agree")
    f.add_argument("--compact2", required=True, help="e.g. docs/data/compact/whole-b0/P3-cu-block.json")
    f.add_argument("--plan", required=True, help="the plan spec inside it, e.g. P3/cu=block or root")
    f.add_argument("--out", required=True)
    f.add_argument("--case", default="B0")
    m = sub.add_parser("maps", help="build + plan + VALIDATE a case's whole program and write maps, population.json and compact_validation.json "
                                    "(whole B0: 311-340 s / 1.9-2.9 GB MEASURED on a pod -- not a laptop command)")
    m.add_argument("--case", default="B0")
    m.add_argument("--lp", type=int, default=32)
    m.add_argument("--steps", type=int, default=15)
    m.add_argument("--plan", default="P3/cu=block")
    m.add_argument("--out", required=True)
    m.add_argument("--summary", action="store_true", help="[REC] also write summary.json (veritor.plans.summary.summarize) + summary_meta.json")
    m.add_argument("--stream", action="store_true", help="[COMPACT-P1] validate_plan(retain=False): no per-VU boundaries/obligations are kept, so only "
                                                        "compact_validation.json + index.json are written (no maps/population.json); O(runs) memory")
    m.add_argument("--dense", action="store_true", help="force the dense (one Parts per unit) plan representation (default: P0/P1 compact, others dense)")
    m.add_argument("--population-v2", action="store_true", help="[SAMPLER] also write population.compact.json (veritor/population/v2, O(runs): the sampler's "
                                                               "population for e2e.partition_sample; the only population a --stream run can write)")
    r = sub.add_parser("replicate", help="record-level population.json for R program instances (requests) sharing one validated partition")
    r.add_argument("--population", required=True, help="a per-instance population.json (e2e.population tiny / maps)")
    r.add_argument("--instances", type=int, required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tiny":
        return _tiny(Path(a.out), a.plans.split(","), a.tables_max_vus)
    if a.cmd == "replicate":
        doc = replicate_population(json.loads(Path(a.population).read_text()), a.instances)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
        print(f"[population] record population -> {a.out}: {a.instances} x ({doc['per_instance']['vu_count']} VUs, {doc['per_instance']['ru_count']} RUs) "
              f"= {doc['vu_count']} VUs / {doc['ru_count']} RUs")
        return 0
    from verity_vllm.registry import b1

    if a.cmd == "maps":
        import resource

        from veritor.plans.validate import validate_plan

        if a.case == "TINY":
            config, a.lp, a.steps = dict(TINY), TINY_LP, TINY_STEPS
        else:
            config = dict(getattr(b1, f"{a.case}_CONFIG"))
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
        program, adapted, circuit, pdigest = build(config, a.lp, a.steps)
        plan, meta = make_plan(a.plan, adapted, circuit, compact=False if a.dense else None)
        t = time.perf_counter()
        v = validate_plan(circuit, plan, retain=not a.stream, **validate_kwargs(meta))
        secs = round(time.perf_counter() - t, 3)
        rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        if a.stream:
            pop = population(adapted.program, plan, v, pdigest) if v.status == "valid" else None
            header = {"plan": {"name": plan.name, "descriptor_digest": plan_descriptor_digest(adapted.program, plan), "digest_kind": plan_digest_kind(plan),
                               "partition_digest": partition_digest(plan), "partition_digest_kind": PARTITION_DIGEST,
                               "vus": len(plan.verification_units), "cus": len(plan.commitment_units), "status": v.status},
                      "counts": {"program_values": None, "vus": len(plan.verification_units), "rus": len(plan.commitment_units),
                                 "label": "MEASURED (streamed validation: counts and digests only; no maps/population.json)"}}
        else:
            header = write_maps(out, adapted.program, circuit, plan, v, pdigest, {"vu_level": meta["vu_level"], "cu_strategy": meta["cu_strategy"]})
        art = validation_artifact(adapted.program, plan, v, pdigest, meta, seconds=secs, maxrss_mb=rss_mb,
                                  verdict_source=f"validate_plan{'(retain=False)' if a.stream else ''} in this process ({a.case} LP={a.lp} STEPS={a.steps}; MEASURED)")
        art.update({"gates": program.gates, "input_gates": program.input_gates})
        (out / "compact_validation.json").write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
        (out / "index.json").write_text(json.dumps({"case": a.case, "LP": a.lp, "STEPS": a.steps, "program": {"name": adapted.program.name, "digest": pdigest,
                                                     "gates": program.gates, "input_gates": program.input_gates}, "plan": header["plan"]}, indent=1, sort_keys=True) + "\n")
        print(f"[population] {a.case} {a.plan}: {v.status}, N_V={header['plan']['vus']} N_R={header['plan']['cus']} values={header['counts']['program_values']} "
              f"plan digest {header['plan']['descriptor_digest'][:16]} partition {header['plan']['partition_digest'][:16]} ({secs} s validate, maxrss {rss_mb} MB) -> {out}")
        if a.population_v2 and v.status == "valid":
            # [SAMPLER] the O(runs) population of the validated partition (population/v2); with --stream this is the only population written
            from e2e.population_v2 import build as build_v2
            from e2e.population_v2 import outside_record_from_meta

            t = time.perf_counter()
            # [A1, contract R2] an ``/inputs=prescribed`` plan records its root Input intervals as ``prescribed_outside_partition`` (both sides)
            v2 = build_v2(plan, circuit, program_digest=pdigest, prescribed_outside=outside_record_from_meta(meta),
                          label=f"MEASURED (e2e.population maps --population-v2, {a.case} LP={a.lp} STEPS={a.steps}: rows = the validated plan's UnitRuns split at CU boundaries"
                                + ("; input gates prescribed, in no row" if meta.get("inputs") == "prescribed" else "") + ")")
            (out / "population.compact.json").write_text(json.dumps(v2, sort_keys=True, separators=(",", ":")) + "\n")
            print(f"[population] population/v2: {len(v2['rows'])} rows, {len(v2['kinds']['entries'])} kinds, content {v2['content_sha256'][:16]} "
                  f"({round(time.perf_counter() - t, 1)} s) -> {out / 'population.compact.json'}")
        if a.summary and v.status == "valid":
            # [REC] the exact Summary of the validated partition (SUMMARY-C agreement 2) beside the maps: the Bound is computed from it
            from veritor.plans.summary import summarize

            sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))  # hierarchical_table recurses through the program's leaves (B0 needs > 1000; as scripts/summary_probe.py)
            t = time.perf_counter()
            s = summarize(circuit, plan, v, program_digest=("verity-ir/program_digest", pdigest))
            s.save(out / "summary.json")
            sp = s.sampler_populations()
            (out / "summary_meta.json").write_text(json.dumps({"summary_digest": str(s.digest), "sampler_populations": sp, "seconds": round(time.perf_counter() - t, 3),
                                                              "partition_digest": s.plan.get("partition_digest")}, indent=1, sort_keys=True, default=str) + "\n")
            print(f"[population] summary: digest {str(s.digest)[:16]} N_V={sp['N_V']} N_R={sp['N_R']} ({round(time.perf_counter() - t, 1)} s) -> {out / 'summary.json'}")
        return 0 if v.status == "valid" else 1

    if a.cmd == "from-compact2":
        rec = json.loads(Path(a.compact2).read_text())
        res = (rec.get("plan_results") or {}).get(a.plan)
        if res is None:
            print(f"[population] {a.compact2} has no plan_results[{a.plan!r}]")
            return 1
        prow = next((r for r in rec.get("stages", []) if r.get("stage") == "program"), {})
        config = dict(getattr(b1, f"{a.case}_CONFIG"))
        program, adapted, circuit, pdigest = build(config, int(rec["lp"]), int(rec["steps"]))
        plan, meta = make_plan(a.plan, adapted, circuit)
        problems = []
        if prow.get("digest") and prow["digest"] != pdigest:
            problems.append(f"program digest here {pdigest[:12]} != COMPACT-2 run {prow['digest'][:12]}")
        if res.get("vus") != len(plan.verification_units) or res.get("cus") != len(plan.commitment_units):
            problems.append(f"plan here {len(plan.verification_units)}/{len(plan.commitment_units)} != COMPACT-2 run {res.get('vus')}/{res.get('cus')}")
        if res.get("plan") != plan.name and not (a.plan == "root" and res.get("plan") == "root"):
            problems.append(f"plan name here {plan.name!r} != run {res.get('plan')!r}")
        if problems:
            for pr in problems:
                print(f"[population]   problem: {pr}")
            return 1
        vstage = next((r for r in rec.get("stages", []) if r.get("stage") == f"validate[{a.plan}]"), {})
        art = {"schema": "veritor/compact-validation/v1", "program_digest": pdigest, "program": adapted.program.name, "gates": program.gates,
               "input_gates": program.input_gates, "plan_name": plan.name, "plan_digest": plan_descriptor_digest(adapted.program, plan), "plan_digest_kind": plan_digest_kind(plan),
               "partition_digest": partition_digest(plan), "partition_digest_kind": PARTITION_DIGEST, **meta,
               "status": res.get("status"), "codes": res.get("codes"), "vus": res.get("vus"), "cus": res.get("cus"),
               "vu_edges": vstage.get("vu_edges"), "cu_edges": vstage.get("cu_edges"), "seconds": res.get("validate_seconds"),
               "maxrss_mb": res.get("validate_maxrss_mb"),
               "verdict_source": f"COMPACT-2 pod run ({rec.get('machine', {}).get('hostname')}, {rec.get('started_utc')}): {a.compact2} "
                                 f"plan_results[{a.plan!r}] -- MEASURED on the pod; program + plan rebuilt and digested in this process, "
                                 f"digest/counts required to agree; the verdict itself is NOT recomputed here",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
        print(f"[population] compact-validation artifact -> {a.out}: {art['status']} N_V={art['vus']} N_R={art['cus']} "
              f"program {pdigest[:12]} plan {art['plan_digest'][:12]} ({art['seconds']} s / {art['maxrss_mb']} MB on the pod)")
        return 0

    config = dict(getattr(b1, f"{a.case}_CONFIG"))
    t0 = time.perf_counter()
    program, adapted, circuit, pdigest = build(config, a.lp, a.steps)
    plan, meta = make_plan(a.plan, adapted, circuit)
    doc = {"case": a.case, "LP": a.lp, "STEPS": a.steps, "program": adapted.program.name, "program_digest": pdigest, "gates": program.gates,
           "input_gates": program.input_gates, "plan_name": plan.name, "plan_digest": plan_descriptor_digest(adapted.program, plan), "plan_digest_kind": plan_digest_kind(plan),
           "partition_digest": partition_digest(plan), "partition_digest_kind": PARTITION_DIGEST,
           "vus": len(plan.verification_units), "cus": len(plan.commitment_units), **meta, "seconds": round(time.perf_counter() - t0, 2),
           "note": "plan built and digested only; NOT validated here (whole-B0 validate_plan is a pod job: 311-340 s / 1.9-2.9 GB)"}
    print(json.dumps(doc, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
