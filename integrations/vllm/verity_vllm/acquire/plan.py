"""The ACQUISITION PLAN: for every required Value, the legal sites where the runtime can observe it, one chosen by a documented policy.

    acquisition_plan(required_value_manifest, correspondence) -> AcquisitionPlan       serialised as `acquisition-plan/v1`, digested

Inputs are STRUCTURE only: the required-value manifest (`verity_vllm.pipeline.manifest`: one identity per required Value,
`op_path` / `output_member` / `family` / `promoted` / `consumers`) and the runtime correspondence (`correspondence.AcquireCorrespondence`:
`owning_module`, `return_slot`, `arg_slot` over the `runtime-correspondence/v1` record when the Build carries it, else the manifest's
v1-annotation reading -- `source` recorded in the plan's `correspondence` block + the runtime module tree).  Nothing here reads a module
CLASS table to decide a site; the declared by-name tables (the lifetime leaves, the analytic tables) are consulted only to mark a site's
lifetime and to name RESIDUALS (`plan.residuals`): facts the plan cannot derive from structure yet, each with its table.

Sites (`SITE_KINDS`):
  producer_output   forward hook on the owning module, return slot = the Value's port (`out` = the whole return, `k` = tuple slot k)
  consumer_input    forward PRE-hook on a consumer module, arg slot i; legal only while the Value is unmutated between its producer and the
                    consumer's read -- alias / in-place facts decide when the annotations carry them, else `lifetime: unknown`
  native_tap        an existing kernel-level source (the attention hidden-stream tap, the fused-MoE block source, the TP partial source)
  runner_return     the runner's `sample()` return (`runner.sampler/sampled_token_ids`); runner_state = a buffer the runner consumed
                    (`seed`); launch_proxy = a host constant read from a kernel launch (`splits`)
  registered_input  a weight of record: hashed once at registration (`register_weights`); analytic_table = a derived table the Match phase
                    dumps and checks analytically (`analytic_table_of_buffer`)
  prescribed        no runtime producer: the Value is materialised by rule (a padding step's identities, `padding_steps`)

Policy (`POLICY`): a Value with one legal site takes it; with several, producer output first; consumer input when the producer is a
PROMOTED interior Call (a pre-flip v1 manifest's row: its output is no module's return, `manifest_format.promoted_observation`); a Value with no legal site is
UNSUPPORTED and the plan fails closed (`plan.unsupported`, listed by Value; `install` refuses such a plan).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from verity_vllm.correspondence.reader_for_acquire import RUNNER_PATH, AcquireCorrespondence, ReturnSlot, RuntimeTree

AnnotationCorrespondence = AcquireCorrespondence

SCHEMA = "acquisition-plan/v1"
SITE_KINDS = ("producer_output", "consumer_input", "native_tap", "runner_return", "runner_state", "launch_proxy", "registered_input", "analytic_table", "prescribed")
HOOK_KINDS = ("producer_output", "consumer_input", "runner_return")      # the kinds `install_from_plan` installs as hooks / wraps
POLICY = ("one legal site: chosen. several: producer output first (the owning module's forward hook, return slot = the Value's port); consumer input "
          "when the producer is a promoted interior Call (the Value is no module's return, so the first consumer's pre-hook is the only module "
          "boundary that carries it); a native tap / runner return / registered input / prescribed materialisation when the Value's family names "
          "one. none: UNSUPPORTED -- the plan fails closed by Value, nothing is guessed from a module class.")
#: The required-value manifest's `family` vocabulary (`manifest_format.FAMILIES`, one entry each) -> the ROUTE the plan derives a site by.
#: This is the manifest's own schema read from the row's `family` field -- not a module class, a module path or a Definition family -- so
#: the dispatch below is a schema lookup (`ROUTE_OF_FAMILY.get(fam)`), never a name predicate (by-name lint, Draft 3.1 §12.4).
ROUTE_OF_FAMILY = {"instance_outputs": "module_output", "input_token_ids": "module_input",
                   "fa2_hidden_m1_stream": "hidden_tap", "fa3_hidden_m1_stream": "hidden_tap",
                   "moe_block_stream": "moe_tap", "tp_rank_partials": "tp_tap",
                   "sampled_token_ids": "runner_return", "sampler_seed": "runner_state", "sampler_splits": "launch_proxy",
                   "weights": "registered_input", "norm_scales": "norm_tap", "router_softmax": "router_tap", "vocab_range": "vocab_tap",
                   "scale_products": "scale_tap", "call_boundaries": "call_boundary_tap"}
MODULE_ROUTES = ("module_output", "module_input")                        # the routes whose sites are module boundaries (confirm_modules)
HIDDEN_TAP_OF_FAMILY = {"fa2_hidden_m1_stream": "fa2.m1", "fa3_hidden_m1_stream": "fa3.m1"}   # hidden_tap route -> the kernel tap
MOE_FAMILY = "moe_block_stream"
TP_FAMILY = "tp_rank_partials"
WEIGHTS_FAMILY = "weights"
#: manifest `boundary_class` / `enablement` values that mean "materialised by enablement, no runtime producer" (padding_steps)
PRESCRIBED_ENABLEMENT = frozenset({"padding", "inactive"})


def analytic_table_of_buffer(name: str) -> str | None:
    """The analytic table (`cos_sin`, checked by `analytic.check_cos_sin`, or `normalizer`, by `analytic.check_normalizer`) a runtime
    BUFFER (`named_buffers()` path) is the observation of, or None: the one place the two derived tables are recognised by name
    (`m1_capture.dump_cos_sin` / `dump_normalizer` dump them; `sites_of` and `residuals_of` name them).  vLLM naming facts: the rotary table lives on every layer's `rotary_emb.cos_sin_cache` (one shared table), Gemma-2's normalizer on
    `model.normalizer`."""
    if name.endswith("rotary_emb.cos_sin_cache"):
        return "cos_sin"
    if name == "model.normalizer" or name.endswith(".normalizer"):
        return "normalizer"
    return None


@dataclass(frozen=True)
class Site:
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
    sites: list[Site] = field(default_factory=list)          # the LEGAL sites
    rejected: list[dict[str, str]] = field(default_factory=list)   # candidates the structure refuses, with the reason
    chosen: Site | None = None
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

    def chosen_sites(self) -> list[Site]:
        seen: dict[tuple, Site] = {}
        for v in self.values:
            if v.chosen is not None:
                seen.setdefault(v.chosen.key(), v.chosen)
        return [seen[k] for k in sorted(seen)]

    def hooks(self) -> list[Site]:
        """The sites `install_from_plan` installs: (module path -> hook kind -> slot), sorted, unique."""
        return [s for s in self.chosen_sites() if s.kind in HOOK_KINDS]

    def sites_by_kind(self) -> dict[str, list[tuple[str, str, str]]]:
        out: dict[str, list[tuple[str, str, str]]] = {}
        for s in self.chosen_sites():
            out.setdefault(s.kind, []).append(s.key())
        return out

    def values_at(self, site: Site) -> list[str]:
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
    """sha256 over the canonical JSON of the plan body (sorted keys, compact separators), the provenance keys excluded: the same Program,
    manifest and runtime module tree give the same digest whether the tree was read off the meta-instantiated model or a live engine
    (the two agreed Value for Value on the reference row; only `runtime_source` differed), and whether the owning modules were read off
    the correspondence record or the v1 annotations (`correspondence.source`; a Value whose site differs between the two is a finding
    `gate.compare_plans` lists by Value)."""
    doc = {k: v for k, v in body.items() if k not in PROVENANCE_KEYS}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------
# the manifest's Values (one class per (op_path, output_member, family); hidden-stream geometries fold to `*`)
# ---------------------------------------------------------------------------------------------------------------------------------

def _identity_key(r: Mapping[str, Any]) -> tuple:
    return (str(r.get("request_id")), str(r.get("step")), int(r.get("invocation") or 0), str(r.get("op_path")), str(r.get("output_member", "out")), int(r.get("rank") or 0))


def value_classes(manifest: Mapping[str, Any], padding_identities: Iterable[Mapping[str, Any]] = ()) -> list[ValuePlan]:
    """One Value class per (op_path, output_member, family).  `padding_identities`: the manifest identities OUTSIDE their request's executed
    prefix of record (`padding_steps.padding_steps_of_record`, a runtime fact known at finalize): those are PRESCRIBED -- materialised by
    enablement, no runtime producer.  A row may also declare it (`boundary_class: padding`)."""
    rows = manifest.get("identities") or manifest.get("entries") or manifest.get("rows") or []
    padded = {_identity_key(p) for p in padding_identities}
    out: dict[tuple[str, str, str], ValuePlan] = {}
    for r in rows:
        fam = str(r.get("family") or "")
        op = str(r.get("op_path"))
        member = str(r.get("output_member", "out"))
        if ROUTE_OF_FAMILY.get(fam) == "hidden_tap":
            member = "*"                                          # the geometry name is per (layer, step); one Value class per tap site
        key = (op, member, fam)
        v = out.get(key)
        if v is None:
            v = out[key] = ValuePlan(op_path=op, output_member=member, family=fam, identities=0, promoted=bool(r.get("promoted")))
        v.identities += 1
        if r.get("promoted"):
            v.promoted = True
        for c in (r.get("consumers") or []):
            if str(c) not in v.consumers:
                v.consumers.append(str(c))
        sp = r.get("spec")
        if sp and str(sp) not in v.spec:
            v.spec.append(str(sp))
        for extra in ("attention_module", "moe_module"):
            if r.get(extra) and v.anchor is None:
                v.anchor = str(r[extra])
        bc = str(r.get("boundary_class") or "")
        if bc in PRESCRIBED_ENABLEMENT or str(r.get("enablement") or "") in PRESCRIBED_ENABLEMENT or _identity_key(r) in padded:
            v.prescribed_identities += 1
    return [out[k] for k in sorted(out)]


# ---------------------------------------------------------------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------------------------------------------------------------

def _slot_label(rs: ReturnSlot | None) -> str:
    return "in" if rs is None else rs.label()


def sites_of(v: ValuePlan, corr: AcquireCorrespondence) -> tuple[list[Site], list[dict[str, str]]]:
    """The legal sites of one Value class and the refused candidates (each with its reason).  Structure only."""
    rt: RuntimeTree = corr.runtime
    call = {"op_path": v.op_path, "output_member": v.output_member}
    module = corr.owning_module(call)
    legal: list[Site] = []
    refused: list[dict[str, str]] = []
    fam = v.family

    def refuse(kind: str, mod: str, slot: str, why: str) -> None:
        refused.append({"kind": kind, "module": mod, "slot": slot, "why": why})

    if v.prescribed_identities:
        n = v.prescribed_identities
        legal.append(Site("prescribed", module, v.output_member, "n/a", f"{n} of {v.identities} identities are padding-step materialisations (enablement, no runtime producer)"))
        if v.prescribed:
            return legal, refused                                # every identity is prescribed: no runtime site needed; else the executed steps' site follows

    route = ROUTE_OF_FAMILY.get(fam)
    if route in MODULE_ROUTES or (fam == "" and module != RUNNER_PATH):
        rs = corr.return_slot(call)
        if rs is not None and not v.promoted:
            if rt.has_module(module):
                legal.append(Site("producer_output", module, rs.label(), corr.output_lifetime(module),
                                  f"the Value is return slot `{rs.label()}` of the owning module's forward (manifest op_path, member)"))
            else:
                refuse("producer_output", module, rs.label(), "the owning module is not a module of the runtime tree")
        elif rs is not None and v.promoted:
            refuse("producer_output", module, rs.label(), "promoted interior Call: the Value is no return of the owning module (PROMOTED_OBSERVATION_RULE)")
        if rs is None:
            # the Value is an INPUT of the module (`<module>/in`: the ids the embedding reads): the module's own pre-hook, arg 0
            a = corr.arg_slot(call, 0)
            if rt.has_module(module):
                legal.append(Site("consumer_input", module, a.label(), corr.input_lifetime(module), "the Value is positional argument 0 of the module's forward (manifest member `in`)"))
            else:
                refuse("consumer_input", module, a.label(), "the consuming module is not a module of the runtime tree")
        for i, c in enumerate(v.consumers):
            a = corr.arg_slot(call, 0)
            if not rt.has_module(c):
                refuse("consumer_input", c, a.label(), f"consumer {i} is not a module of the runtime tree")
                continue
            legal.append(Site("consumer_input", c, a.label(), corr.input_lifetime(c),
                              f"the Program's dataflow reads the Value as positional argument 0 of consumer {i} (manifest `consumers`)"))
        return legal, refused

    if route == "hidden_tap":
        anchor = v.anchor
        if anchor and rt.has_module(anchor):
            legal.append(Site("native_tap", module, "*", "n/a", f"the attention hidden stream of module {anchor}: the {HIDDEN_TAP_OF_FAMILY[fam]} kernel tap (manifest `attention_module`)"))
        elif anchor:
            refuse("native_tap", module, "*", f"attention module {anchor} is not a module of the runtime tree")
        else:
            refuse("native_tap", module, "*", "the manifest row names no `attention_module` to anchor the tap on")
        return legal, refused

    if fam == MOE_FAMILY:
        anchor = v.anchor
        if anchor and rt.has_module(anchor):
            legal.append(Site("native_tap", module, v.output_member, "n/a", f"a fused-MoE block-stream plane of module {anchor} (manifest `moe_module`)"))
        elif anchor:
            refuse("native_tap", module, v.output_member, f"MoE module {anchor} is not a module of the runtime tree")
        else:
            refuse("native_tap", module, v.output_member, "the manifest row names no `moe_module` to anchor the block source on")
        return legal, refused

    if fam == TP_FAMILY:
        if rt.has_module(module):
            legal.append(Site("native_tap", module, v.output_member, "n/a", "a rank-local collective operand: the TP partial source at the module's collective"))
        else:
            refuse("native_tap", module, v.output_member, "the owning module is not a module of the runtime tree")
        return legal, refused

    if route == "norm_tap":
        if rt.has_module(module):
            legal.append(Site("native_tap", module, v.output_member, "n/a", "the scale of the module's norm: the norm-scale source inside the module "
                              "(a tapped kernel, or the rsqrt of an ATen-chain norm)"))
        else:
            refuse("native_tap", module, v.output_member, "the norm module is not a module of the runtime tree")
        return legal, refused

    if route == "router_tap":
        anchor = v.anchor
        if anchor and rt.has_module(anchor):
            legal.append(Site("native_tap", module, v.output_member, "n/a", f"the router softmax of MoE module {anchor}: the router-softmax tap "
                              "(tapped topk_softmax inside the module; manifest `moe_module`)"))
        elif anchor:
            refuse("native_tap", module, v.output_member, f"MoE module {anchor} is not a module of the runtime tree")
        else:
            refuse("native_tap", module, v.output_member, "the manifest row names no `moe_module` to anchor the router-softmax tap on")
        return legal, refused

    if route == "vocab_tap":
        if rt.has_module(module):
            legal.append(Site("native_tap", module, v.output_member, "n/a", "the rank's vocabulary-range values: the vocabulary-range source at the "
                              "module's fused embedding launch (computed from its token ids)"))
        else:
            refuse("native_tap", module, v.output_member, "the embedding site is not a module of the runtime tree")
        return legal, refused

    if route == "scale_tap":
        if rt.has_module(module):
            legal.append(Site("native_tap", module, v.output_member, "n/a", "the module's FP8 GEMM scale products: the scale-products source at "
                              "the module's cutlass_scaled_mm launch (computed from its scales)"))
        else:
            refuse("native_tap", module, v.output_member, "the linear module is not a module of the runtime tree")
    if route == "call_boundary_tap":
        owner = module.rsplit("/", 1)[0]
        if rt.has_module(owner):
            legal.append(Site("native_tap", module, v.output_member, "n/a", f"a Value two Calls of module body {owner} exchange: the call-boundary "
                              "source, evaluated on the host from the module's forward arguments"))
        else:
            refuse("native_tap", module, v.output_member, f"the module body {owner!r} is not a module of the runtime tree")
        return legal, refused

    if route in ("runner_return", "runner_state", "launch_proxy"):
        kind = route
        rs = corr.return_slot(call)
        if rt.runner.get("sample"):
            legal.append(Site(kind, module, _slot_label(rs), "n/a", f"the runner's sample() ({fam})"))
        else:
            refuse(kind, module, _slot_label(rs), "the runtime has no runner with sample()")
        return legal, refused

    if fam == WEIGHTS_FAMILY:
        owner = module.rsplit(".", 1)[0] if "." in module else module
        leaf = module.rsplit(".", 1)[-1]
        if rt.has_module(owner) or module in rt.buffers:
            legal.append(Site("registered_input", module, "weight", "n/a", "a weight of record: hashed once at registration"))
        else:
            refuse("registered_input", module, "weight", f"neither the weight's owning module {owner!r} nor a buffer of that name is in the runtime tree")
        table = analytic_table_of_buffer(module)
        if table is not None:
            legal.append(Site("analytic_table", module, table, "n/a", f"a derived table the Match stage dumps and checks analytically ({table})"))
        return legal, refused

    refuse("?", module, v.output_member, f"family {fam!r} has no site rule")
    return legal, refused


def choose(sites: list[Site], v: ValuePlan) -> tuple[Site | None, str]:
    """POLICY.  Returns (chosen, why)."""
    if not sites:
        return None, "no legal site"
    if len(sites) == 1:
        return sites[0], "one legal site"
    order = ["producer_output", "consumer_input", "native_tap", "runner_return", "runner_state", "launch_proxy", "registered_input", "prescribed", "analytic_table"]
    if v.promoted:
        order.remove("consumer_input"); order.insert(0, "consumer_input")
    ranked = sorted(sites, key=lambda s: (order.index(s.kind) if s.kind in order else len(order), s.module, s.slot))
    return ranked[0], f"several legal sites ({', '.join(s.kind for s in ranked)}): policy chose {ranked[0].kind}"


# ---------------------------------------------------------------------------------------------------------------------------------
# residuals: the declared by-name facts the structure does not carry yet, by table
# ---------------------------------------------------------------------------------------------------------------------------------

def _lifetime_tables() -> dict[str, Any]:
    from verity_vllm.commit.committer.native_host import NativeHostCommitter
    from verity_vllm.commit.committer.flush_points import GATHER_FLUSH_LEAVES, PRE_FLUSH_LEAVES
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
        table = analytic_table_of_buffer(b)
        if table is not None and b not in weights:
            out.append(Residual("ANALYTIC_TABLES", b, table, "the Match stage dumps this buffer by name and checks it analytically (root_policy.ANALYTIC_TABLES); "
                                "the manifest names no weight Value for it"))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------------------------------------------------------------

def confirm_modules(values: Iterable[ValuePlan], corr: AcquireCorrespondence) -> dict[str, Any]:
    """Under `source = correspondence`: every module-site Value (a module's output / input, its consumers) checked against the record --
    its owning module is a module the record owns Calls under (`confirmed`) or not (`unconfirmed`, listed by Value: a finding, the
    manifest's op_path stands).  Under the fallback: nothing to confirm against (`{}` counts, `applicable: False`)."""
    if corr.source != "correspondence":
        return {"applicable": False, "confirmed": 0, "unconfirmed": []}
    confirmed = 0
    unconfirmed: list[dict[str, str]] = []
    for v in values:
        if ROUTE_OF_FAMILY.get(v.family) not in MODULE_ROUTES:
            continue
        module = corr.owning_module({"op_path": v.op_path, "output_member": v.output_member})
        for m, role in [(module, "owning module")] + [(c, f"consumer {i}") for i, c in enumerate(v.consumers)]:
            if corr.confirmed(m):
                confirmed += 1
            else:
                unconfirmed.append({"value": v.value_id, "module": m, "role": role, "why": "not a module the correspondence record owns Calls under"})
    return {"applicable": True, "confirmed": confirmed, "unconfirmed": unconfirmed}


def acquisition_plan(manifest: Mapping[str, Any], correspondence: AcquireCorrespondence, *, tables: Mapping[str, Any] | None = None,
                     with_residuals: bool = True, padding_identities: Iterable[Mapping[str, Any]] = ()) -> AcquisitionPlan:
    values = value_classes(manifest, padding_identities)
    for v in values:
        legal, refused = sites_of(v, correspondence)
        v.sites, v.rejected = legal, refused
        chosen, why = choose(legal, v)
        v.chosen = chosen
        v.status = "chosen" if chosen is not None else "unsupported"
        v.why = why if chosen is not None else ("no legal site: " + "; ".join(f"{r['kind']} at {r['module']}/{r['slot']}: {r['why']}" for r in refused))
    prov = correspondence.provenance()
    prov["modules"] = confirm_modules(values, correspondence)
    rt = correspondence.runtime
    if with_residuals:
        tabs = tables if tables is not None else _lifetime_tables()
        for v in values:                                     # lifetime of a module-output site from the table when the annotations carry none
            if v.chosen is not None and v.chosen.kind == "producer_output" and v.chosen.lifetime == "unknown" and hazard_by_table(v.chosen.module, v.chosen.slot, tabs["HAZARD_LEAVES"]):
                v.chosen = Site(v.chosen.kind, v.chosen.module, v.chosen.slot, "mutated-after-hook (HAZARD_LEAVES)", v.chosen.derived_from)
                v.sites = [v.chosen if s.key() == v.chosen.key() else s for s in v.sites]
        residuals = residuals_of(values, rt, tabs)
    else:
        residuals = []
    notes = ["every identity of a Value at a step >= its request's executed prefix is a PRESCRIBED materialisation (padding_steps: enablement, no runtime producer); "
             "the plan's runtime sites are for the executed steps"]
    pdig = str(manifest.get("program_digest") or "")
    mdig = str(manifest.get("manifest_digest") or "")
    return AcquisitionPlan(program_digest=pdig, manifest_digest=mdig, runtime_source=rt.source, values=values, residuals=residuals, notes=notes, correspondence=prov)


def load_plan(path: str | Path) -> dict[str, Any]:
    d = json.loads(Path(path).read_text())
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r} != {SCHEMA}")
    if plan_digest(d) != d.get("plan_digest"):
        raise ValueError(f"{path}: plan_digest does not match the body")
    return d


def hooks_of_doc(doc: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [(h["module"], h["kind"], h["slot"]) for h in doc.get("hooks") or []]
