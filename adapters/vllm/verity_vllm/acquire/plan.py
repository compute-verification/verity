"""The ACQUISITION PLAN: for every class of the row's capture plan (`capture_plan.of_build`), the site where the runtime observes
it, by its capture class (`records.capture`), over the runtime correspondence (`correspondence.AcquireCorrespondence`).

    plan_of_capture(classes, correspondence) -> AcquisitionPlan       serialised as `acquisition-plan/v1`, digested

The declared by-name tables (the lifetime leaves, the analytic tables) are consulted only to mark a site's lifetime and to name
RESIDUALS (`plan.residuals`): facts the plan cannot derive from structure yet, each with its table.

Sites (`SITE_KINDS`):
  producer_output   forward hook on the owning module, return slot = the Value's port (`out` = the whole return, `k` = tuple slot k)
  consumer_input    forward PRE-hook on a consumer module, arg slot i; legal only while the Value is unmutated between its producer and the
                    consumer's read -- alias / in-place facts decide when the annotations carry them, else `lifetime: unknown`
  native_tap        an existing kernel-level source (the attention hidden-stream tap, the fused-MoE block source, the TP partial source)
  runner_return     the runner's `sample()` return (`runner.sampler/sampled_token_ids`); runner_state = a buffer the runner consumed
                    (`seed`)
  registered_input  a weight of record: hashed once at registration (`register_weights`); analytic_table = a derived table the Match phase
                    dumps and checks analytically (`analytic_table_of_buffer`)
  prescribed        no runtime producer: the Value is materialised by rule (a padding step's identities)

Policy (`POLICY`): a Value with one legal site takes it; with several, producer output first; a Value with no legal site is
UNSUPPORTED and the plan fails closed (`plan.unsupported`, listed by Value; `install` refuses such a plan).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.records import capture as CP
from verity_vllm.commit.committer.flush_points import (
    GATHER_FLUSH_LEAVES,
    PRE_FLUSH_LEAVES,
)
from verity_vllm.correspondence.reader_for_acquire import (
    AcquireCorrespondence,
    ReturnSlot,
    RuntimeTree,
)

SCHEMA = "acquisition-plan/v1"
SITE_KINDS = ("producer_output", "consumer_input", "native_tap", "runner_return", "runner_state", "registered_input", "analytic_table", "prescribed")
HOOK_KINDS = ("producer_output", "consumer_input", "runner_return")      # the kinds `install_from_plan` installs as hooks / wraps
POLICY = ("one legal site: chosen. several: producer output first (the owning module's forward hook, return slot = the Value's port); consumer input "
          "when the producer is a promoted interior Call (the Value is no module's return, so the first consumer's pre-hook is the only module "
          "boundary that carries it); a native tap / runner return / registered input / prescribed materialisation when the Value's family names "
          "one. none: UNSUPPORTED -- the plan fails closed by Value, nothing is guessed from a module class.")
WEIGHTS_FAMILY = "weights"


def analytic_table_of_buffer(name: str) -> str | None:
    """The analytic table (`cos_sin`, checked by `analytic.check_cos_sin`, or `normalizer`, by `analytic.check_normalizer`) a runtime
    BUFFER (`named_buffers()` path) is the observation of, or None: the one place the two derived tables are recognised by name
    (`match_capture.dump_cos_sin` / `dump_normalizer` dump them; `sites_of` and `residuals_of` name them).  vLLM naming facts: the rotary table lives on every layer's `rotary_emb.cos_sin_cache` (one shared table), Gemma-2's normalizer on
    `model.normalizer`."""
    if name.endswith("rotary_emb.cos_sin_cache"):
        return "cos_sin"
    if name == "model.normalizer" or name.endswith(".normalizer"):
        return "normalizer"
    return None


@dataclass(frozen=True)
class AcquisitionSite:
    kind: str
    module: str                  # module path / tap path / runner path / weight name
    slot: str                    # `out` | `0` | `1` | `in[0]` | `.sampled_token_ids` | plane | `weight` | `*`
    lifetime: str = "n/a"        # unmutated | unknown | mutated-after-hook (<table>) | n/a
    derived_from: str = ""       # the structural fact that made the site legal

    def key(self) -> tuple[str, str, str]:
        return (self.module, self.kind, self.slot)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValuePlan:
    op_path: str
    output_member: str
    family: str
    identities: int
    promoted: bool = False
    consumers: list[str] = field(default_factory=list)
    spec: list[str] = field(default_factory=list)
    anchor: str | None = None                                # the runtime module a tap Value is anchored on (`attention_module` / `moe_module`)
    prescribed_identities: int = 0                           # identities materialised by rule (padding steps: no runtime producer)
    capture: str = ""                                        # the capture plan's class (`capture_plan.CLASSES`); "" for a manifest's Value
    sites: list[AcquisitionSite] = field(default_factory=list)  # the LEGAL sites
    rejected: list[dict[str, str]] = field(default_factory=list)   # candidates the structure refuses, with the reason
    chosen: AcquisitionSite | None = None
    status: str = "unsupported"                              # chosen | unsupported
    why: str = ""

    @property
    def value_id(self) -> str:
        return f"{self.op_path}/{self.output_member}"

    @property
    def prescribed(self) -> bool:
        """Every identity of the Value is a padding-step materialisation: no runtime site is needed."""
        return self.identities > 0 and self.prescribed_identities == self.identities

    def to_json(self) -> dict[str, Any]:
        return {"value": self.value_id, "op_path": self.op_path, "output_member": self.output_member, "family": self.family, "identities": self.identities,
                "promoted": self.promoted, "consumers": list(self.consumers), "spec": list(self.spec), "anchor": self.anchor, "prescribed": self.prescribed,
                "prescribed_identities": self.prescribed_identities, "sites": [s.to_json() for s in self.sites],
                "rejected": list(self.rejected), "chosen": self.chosen.to_json() if self.chosen else None, "status": self.status, "why": self.why}


@dataclass
class Residual:
    table: str
    module: str
    cls: str
    reason: str
    values: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcquisitionPlan:
    program_digest: str
    manifest_digest: str
    runtime_source: str
    values: list[ValuePlan]
    residuals: list[Residual] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    correspondence: dict[str, Any] = field(default_factory=dict)   # provenance: which reader gave the owning modules (`AcquireCorrespondence.provenance()`)

    @property
    def unsupported(self) -> list[ValuePlan]:
        return [v for v in self.values if v.status != "chosen"]

    @property
    def complete(self) -> bool:
        return not self.unsupported

    def chosen_sites(self) -> list[AcquisitionSite]:
        seen: dict[tuple, AcquisitionSite] = {}
        for v in self.values:
            if v.chosen is not None:
                seen.setdefault(v.chosen.key(), v.chosen)
        return [seen[k] for k in sorted(seen)]

    def hooks(self) -> list[AcquisitionSite]:
        """The sites `install_from_plan` installs: (module path -> hook kind -> slot), sorted, unique."""
        return [s for s in self.chosen_sites() if s.kind in HOOK_KINDS]

    def sites_by_kind(self) -> dict[str, list[tuple[str, str, str]]]:
        out: dict[str, list[tuple[str, str, str]]] = {}
        for s in self.chosen_sites():
            out.setdefault(s.kind, []).append(s.key())
        return out

    def values_at(self, site: AcquisitionSite) -> list[str]:
        return [v.value_id for v in self.values if v.chosen is not None and v.chosen.key() == site.key()]

    def _values_by_site(self) -> dict[tuple[str, str, str], list[str]]:
        """`values_at` for every chosen site in one pass (a call-boundary manifest has one site per Value: tens of thousands)."""
        out: dict[tuple[str, str, str], list[str]] = {}
        for v in self.values:
            if v.chosen is not None:
                out.setdefault(v.chosen.key(), []).append(v.value_id)
        return out

    def body(self) -> dict[str, Any]:
        sites, unsupported, at = self.chosen_sites(), self.unsupported, self._values_by_site()
        hooks = [s for s in sites if s.kind in HOOK_KINDS]
        by_kind: dict[str, int] = {}
        for s in sites:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        return {"schema": SCHEMA, "program_digest": self.program_digest, "manifest_digest": self.manifest_digest, "runtime_source": self.runtime_source,
                "correspondence": dict(self.correspondence), "policy": POLICY, "site_kinds": list(SITE_KINDS), "hook_kinds": list(HOOK_KINDS), "complete": not unsupported,
                "counts": {"values": len(self.values), "identities": sum(v.identities for v in self.values), "chosen": len(self.values) - len(unsupported),
                           "unsupported": len(unsupported), "sites": len(sites), "hooks": len(hooks), "residuals": len(self.residuals),
                           "sites_by_kind": dict(sorted(by_kind.items()))},
                "hooks": [{"module": s.module, "kind": s.kind, "slot": s.slot, "lifetime": s.lifetime, "values": at[s.key()]} for s in hooks],
                "sites": [{"module": s.module, "kind": s.kind, "slot": s.slot, "lifetime": s.lifetime, "values": at[s.key()]} for s in sites],
                "values": [v.to_json() for v in self.values],
                "unsupported": [{"value": v.value_id, "family": v.family, "identities": v.identities, "why": v.why, "rejected": v.rejected} for v in unsupported],
                "residuals": [r.to_json() for r in self.residuals], "notes": list(self.notes)}

    def to_json(self) -> dict[str, Any]:
        b = self.body()
        b["plan_digest"] = plan_digest(b)
        return b

    @property
    def digest(self) -> str:
        return plan_digest(self.body())

    def save(self, path: str | Path) -> str:
        d = self.to_json()
        Path(path).write_text(json.dumps(d, indent=1, sort_keys=False) + "\n")
        return d["plan_digest"]


#: not content: where the runtime tree came from (meta instantiation vs a live engine) and which reader gave the owning modules (the
#: `runtime-correspondence/v1` record vs the v1 annotations): the same Program, manifest and tree must give the same digest under both
PROVENANCE_KEYS = ("plan_digest", "runtime_source", "correspondence")


def plan_digest(body: Mapping[str, Any]) -> str:
    """SHA-512's whole output over the canonical JSON of the plan body (sorted keys, compact separators), the provenance keys excluded: the same Program,
    manifest and runtime module tree give the same digest whether the tree was read off the meta-instantiated model or a live engine
    (the two agreed Value for Value on the reference row; only `runtime_source` differed), and whether the owning modules were read off
    the correspondence record or the v1 annotations (`correspondence.source`)."""
    doc = {k: v for k, v in body.items() if k not in PROVENANCE_KEYS}
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------
# the manifest's Values (one class per (op_path, output_member, family); hidden-stream geometries fold to `*`)
# ---------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------------------------------------------------------------

def values_of_plan(classes: Iterable[Mapping[str, Any]]) -> list[ValuePlan]:
    """One Value class per plan class (`capture_plan.of_build`), in its order."""
    return [ValuePlan(op_path=str(c["op_path"]), output_member=str(c["member"]), family="", identities=1, spec=list(c["definitions"]),
                      capture=str(c["capture"])) for c in classes]


def _capture_sites(v: ValuePlan, corr: AcquireCorrespondence) -> tuple[list[AcquisitionSite], list[dict[str, str]]]:
    """The legal sites of a plan class, by its capture class."""
    rt: RuntimeTree = corr.runtime
    call = {"op_path": v.op_path, "output_member": v.output_member}
    module = corr.owning_module(call)
    legal: list[AcquisitionSite] = []
    refused: list[dict[str, str]] = []
    shared = f" (a shared instance's call: {v.op_path} is {module}'s)" if module != v.op_path else ""

    def at(kind: str, mod: str, slot: str, lifetime: str, why: str) -> None:
        if rt.has_module(mod):
            legal.append(AcquisitionSite(kind, mod, slot, lifetime, why + shared))
        else:
            refused.append({"kind": kind, "module": mod, "slot": slot, "why": f"{mod!r} is not a module of the runtime tree"})

    if v.capture == CP.MODULE_OUTPUT:
        rs = corr.return_slot(call)
        at("producer_output", module, rs.label(), corr.output_lifetime(module), f"return slot `{rs.label()}` of the module's forward")
    elif v.capture == CP.INPUT_IDS:
        at("consumer_input", module, corr.arg_slot(call, 0).label(), corr.input_lifetime(module), "positional argument 0 of the module's forward")
    elif v.capture in (CP.CALL_BOUNDARY, CP.SCALE_PRODUCTS, CP.RUN_VALUE):
        owner = module.rsplit("/", 1)[0] if v.capture == CP.CALL_BOUNDARY else module
        at("native_tap", owner, v.output_member, "n/a", f"the {v.capture} source inside module {owner}")
    elif v.capture == CP.SAMPLED:
        if rt.runner.get("sample"):
            legal.append(AcquisitionSite("runner_return", module, _slot_label(corr.return_slot(call)), "n/a", "the runner's sample()"))
        else:
            refused.append({"kind": "runner_return", "module": module, "slot": v.output_member, "why": "the runtime has no runner with sample()"})
    else:
        refused.append({"kind": "?", "module": module, "slot": v.output_member, "why": f"capture class {v.capture!r} has no site rule"})
    return legal, refused


def _slot_label(rs: ReturnSlot | None) -> str:
    return "in" if rs is None else rs.label()


def choose(sites: list[AcquisitionSite], v: ValuePlan) -> tuple[AcquisitionSite | None, str]:
    """POLICY.  Returns (chosen, why)."""
    if not sites:
        return None, "no legal site"
    if len(sites) == 1:
        return sites[0], "one legal site"
    order = ["producer_output", "consumer_input", "native_tap", "runner_return", "runner_state", "registered_input", "prescribed", "analytic_table"]
    if v.promoted:
        order.remove("consumer_input")
        order.insert(0, "consumer_input")
    ranked = sorted(sites, key=lambda s: (order.index(s.kind) if s.kind in order else len(order), s.module, s.slot))
    return ranked[0], f"several legal sites ({', '.join(s.kind for s in ranked)}): policy chose {ranked[0].kind}"


# ---------------------------------------------------------------------------------------------------------------------------------
# residuals: the declared by-name facts the structure does not carry yet, by table
# ---------------------------------------------------------------------------------------------------------------------------------

def _lifetime_tables() -> dict[str, Any]:
    from verity_vllm.commit.committer.native_host import NativeHostCommitter
    return {"HAZARD_LEAVES": set(NativeHostCommitter.HAZARD_LEAVES), "GATHER_FLUSH_LEAVES": set(GATHER_FLUSH_LEAVES), "PRE_FLUSH_LEAVES": set(PRE_FLUSH_LEAVES)}


def hazard_by_table(module: str, slot: str, hazard_leaves: set[str]) -> bool:
    """The committer's `_hazard` decision for a module-output site (the in-place facts the table encodes) -- its own function, not a restatement."""
    from verity_vllm.commit.committer.native_host import hazard_of_record

    return hazard_of_record(f"{module}/{slot}", hazard_leaves)


def residuals_of(values: list[ValuePlan], rt: RuntimeTree, tables: Mapping[str, Any] | None = None) -> list[Residual]:
    t = dict(tables or _lifetime_tables())
    out: list[Residual] = []
    weights = {v.op_path for v in values if v.family == WEIGHTS_FAMILY}
    # lifetime tables: the plan has no alias / in-place facts from the annotations; the committer encodes them by leaf name
    haz = [v for v in values if v.chosen is not None and v.chosen.kind == "producer_output" and hazard_by_table(v.chosen.module, v.chosen.slot, t["HAZARD_LEAVES"])]
    if haz:
        out.append(Residual("HAZARD_LEAVES", "*", "*", "the committer snapshots these module outputs AT the hook (overwritten in place afterwards: RoPE in the qkv buffer, fused add-norm "
                            "into its inputs, the residual slot); the plan marks the site's lifetime from the table because the annotations carry no in-place fact",
                            values=[v.value_id for v in haz]))
    for table in ("GATHER_FLUSH_LEAVES", "PRE_FLUSH_LEAVES"):
        hits = [v for v in values if v.chosen is not None and v.chosen.kind == "producer_output" and v.chosen.module.rsplit(".", 1)[-1] in t[table]]
        if hits and t[table]:
            out.append(Residual(table, "*", "*", f"the gather collector flushes staged bytes at these module boundaries ({table}); the flush points are lifetime facts "
                                "the structure does not carry yet", values=[v.value_id for v in hits]))
    for b in sorted(rt.buffers):
        analytic = analytic_table_of_buffer(b)
        if analytic is not None and b not in weights:
            out.append(Residual("ANALYTIC_TABLES", b, analytic, "the Match stage dumps this buffer by name and checks it analytically (root_policy.ANALYTIC_TABLES); "
                                "the manifest names no weight Value for it"))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------------------------------------------------------------


def plan_of_capture(classes: Iterable[Mapping[str, Any]], correspondence: AcquireCorrespondence, *,
                    tables: Mapping[str, Any] | None = None) -> AcquisitionPlan:
    """The plan of a capture plan's classes (`capture_plan.of_build`) over `correspondence`; its source digest is the classes'
    (SHA-512 of their canonical JSON)."""
    classes = [dict(c) for c in classes]
    digest = hashlib.sha512(json.dumps(classes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan_of_values(values_of_plan(classes), correspondence, program_digest="", source_digest=digest, tables=tables)


def plan_of_values(values: list[ValuePlan], correspondence: AcquireCorrespondence, *, program_digest: str, source_digest: str,
                   tables: Mapping[str, Any] | None = None) -> AcquisitionPlan:
    """The plan of `values` over `correspondence` (`acquisition_plan`'s)."""
    for v in values:
        legal, refused = _capture_sites(v, correspondence)
        v.sites, v.rejected = legal, refused
        chosen, why = choose(legal, v)
        v.chosen = chosen
        v.status = "chosen" if chosen is not None else "unsupported"
        v.why = why if chosen is not None else ("no legal site: " + "; ".join(f"{r['kind']} at {r['module']}/{r['slot']}: {r['why']}" for r in refused))
    prov = correspondence.provenance()
    rt = correspondence.runtime
    tabs = tables if tables is not None else _lifetime_tables()
    for v in values:                                         # lifetime of a module-output site from the table when the annotations carry none
        if v.chosen is not None and v.chosen.kind == "producer_output" and v.chosen.lifetime == "unknown" and hazard_by_table(v.chosen.module, v.chosen.slot, tabs["HAZARD_LEAVES"]):
            v.chosen = AcquisitionSite(v.chosen.kind, v.chosen.module, v.chosen.slot, "mutated-after-hook (HAZARD_LEAVES)", v.chosen.derived_from)
            v.sites = [v.chosen if s.key() == v.chosen.key() else s for s in v.sites]
    residuals = residuals_of(values, rt, tabs)
    notes = [("every identity of a Value at a step >= its request's executed prefix is a PRESCRIBED materialisation (enablement, no runtime producer); "
              "the plan's runtime sites are for the executed steps")]
    return AcquisitionPlan(program_digest=program_digest, manifest_digest=source_digest, runtime_source=rt.source, values=values,
                           residuals=residuals, notes=notes, correspondence=prov)


__all__ = ["HOOK_KINDS", "AcquisitionPlan", "AcquisitionSite", "analytic_table_of_buffer", "plan_of_capture", "values_of_plan"]
