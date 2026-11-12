"""Derivation mode (design §4; GUIDANCE §43): an ordinary `nn.Module` / callable with NO Verity annotations ->
a `Program`, or an `Unsupported` outcome. Never a partial program, never an opaque gate.

    derive(source, example_inputs, *, mode, profile, config, rules, allowed_dims, name) -> Program | Unsupported

How it works
------------
1. `torch.export.export(strict=False)` on meta tensors (values are never read; shapes/dtypes are). The exported
   graph is pre-dispatch ATen: `nn.Linear` stays `aten.linear`, custom kernels stay their own ops, in-place ops
   stay in place (no functionalisation), lifted parameters/buffers/constants are placeholders.
2. Parameters and buffers become ONE prescribed `weights` record input (one field per *distinct tensor*: tied
   parameters are one field referenced from every placeholder that shares the tensor — boundary 3). Lifted
   tensor constants are refused (`hidden_state`) unless a rule accounts for them. A value no circuit computes (a training
   survey's hole, `rules.training_survey`) is a field of a `run` input a rule declares (`Rule.run_inputs`). A rule's check in
   gates is a bit output after the source's (`Ctx.check`), which the verifier requires to be 1.
3. Every graph node is translated by exactly one applicable rule (`rules/`): views are reference rearrangements;
   ops/in-place ops/state patterns emit calls of REGISTERED definitions bound to the mode's kind vocabulary
   (profile mode: the profile's kernels; reference mode: exact CPU-f32 kinds). Rules decide structurally
   (`applies(node, ctx)`), never by op name alone. A node no rule accepts is an `Unsupported` with a reason from
   design §4.3 and the reasons every candidate rule gave.
4. The result is a flat program over leaf kinds with `nn_module_stack` names as annotations only. Optionally a
   `FamilyRule` (rules/family.py) claims a registered program identity — only after verifying that the derived
   leaf dataflow is node-for-node isomorphic to that program's expanded body, with the module's parameters bound
   to the program's weight roles (identity comes from rule + registry, never from names).
5. Provenance (`Program.annotations["derive"]`) records mode, profile, kind vocabulary + version, rule set +
   versions, every rule application (rule, node, module path hint, preconditions checked, emitted kind), the
   weights map (module fqn -> record field, aliases), mutation effects, and the family match report.

Fixed computation (boundary 2): nothing here reads tensor values; every shape is static; loops are whatever the
source unrolled at export time; data-dependent control flow is `Unsupported(dynamic_control_flow)`.
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import torch
from verity.circuits.codec import (
    canonical_json,
    descriptor_id,
    encode_program,
    program_digest,
)
from verity.circuits.defs import Builder, CompositeDefinition, SpecializedDefinition
from verity.circuits.defs import bind as ir_bind
from verity.circuits.liveness import dead_gates as _dead_gates
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Coll, concat
from verity.circuits.types import Array, RecordType, Type, Value
from verity.circuits.types import Tuple as TupleT
from verity.definitions.replay import replayer

from verity_vllm.program.frontend import seed_rule as SEED
from verity_vllm.program.frontend import stop_rule as SR
from verity_vllm.program.frontend.derive_context import (
    Ctx,
    StaticIndex,
    WeightLeaf,
    _runs,
    fx_arg_nodes,
)
from verity_vllm.program.frontend.derive_context import (
    module_registrations as module_registrations,  # read through this module
)
from verity_vllm.program.frontend.outcomes import (
    _CODE_TO_REASON,
    REASONS,
    Refuse,
    Unsupported,
    _jsonable,
    refuse,
)
from verity_vllm.program.frontend.provenance import registry_version
from verity_vllm.program.frontend.registration import RegisteredInputs, RunInputs
from verity_vllm.program.frontend.report import DerivationTail, DeriveReport
from verity_vllm.program.frontend.rules.base import InPlaceRule, RuleSet, StateRule
from verity_vllm.program.frontend.torch_frontend import (
    FRONTEND_VERSION,
    FrontendError,
    check_identity,
    shape_of,
    type_of_tensor,
)
from verity_vllm.program.frontend.value_types import Alloc

DERIVE_VERSION = "verity_vllm.program.frontend.derive/2026-09-09.d4"  # d4: rotary table without terminal cast, 0-d buffer operand in Elementwise; d3: embedding padding_idx, bf16 inv_freq rotary table; d2: HF 5.x plumbing

aten = torch.ops.aten


# ---------------------------------------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------------------------------------


def derive(source, example_inputs: Sequence[Any], *, mode: str, profile: str | None = None, config: dict | None = None,
           rules: RuleSet | None = None, allowed_dims: dict | None = None, name: str | None = None,
           check_registry: bool = True, rank: tuple[int, int] | None = None, registered_values: bool = False) -> Program | Unsupported:
    """Derive a `Program` from an unannotated module/callable, or return `Unsupported`. See module docstring."""
    return derive_report(source, example_inputs, mode=mode, profile=profile, config=config, rules=rules, allowed_dims=allowed_dims,
                         name=name, check_registry=check_registry, rank=rank, registered_values=registered_values).result


def derive_report(source, example_inputs: Sequence[Any], *, mode: str, profile: str | None = None, config: dict | None = None,
                  rules: RuleSet | None = None, allowed_dims: dict | None = None, name: str | None = None,
                  check_registry: bool = True, rank: tuple[int, int] | None = None, exported=None, theta: Sequence[str] = (),
                  registered_values: bool = False, session: list | None = None, dead_free: bool | str = False, correspondence: bool = True) -> DeriveReport:
    """`registered_values`: the program's values (the norms' eps, the attention scale) are fields of a registered `values` parameter
    (before `weights`), read by the Definitions that take them as inputs (`TargetProfile.registered_values`); the provenance's
    `supplied` states every field's bits.  `rank=(world, rank)`: the rank coordinate of a tensor-parallel rank Program.  It is written into the runtime correspondence
    (`annotations["correspondence"].rank`, explicit integers) and NOWHERE else: not a static, not a config -- the Program's identity
    (`program_digest`) does not depend on it.  `exported`: the caller's export of `source` (same inputs and shims), derived from.  `theta`:
    the inputs that are weights (`Program.theta`).  `session`: a list that receives the derivation's `DerivationTail` when it succeeds
    (`relabel.rank_report` reads it).  `dead_free`: the flat body in the registered compiler's form (`program.prune.dead_free`: without the
    units no output reads, each Definition with dead gates swapped for its version without them; `prune.BITS`: without the swaps;
    `prune.BATCHED`: then batched, class G's form), and the correspondence's Calls with it.
    `correspondence=False`: no runtime correspondence (for a graph no serving runtime executes, a training step's); the Program is unchanged."""
    if mode not in ("profile", "reference"):
        raise ValueError("mode must be 'profile' or 'reference'")
    if not correspondence and (rank is not None or session is not None):
        raise ValueError("rank and session are the runtime correspondence's: they need correspondence=True")
    if rank is not None and not (len(rank) == 2 and 0 <= int(rank[1]) < int(rank[0])):
        raise ValueError(f"rank must be (world, rank) with 0 <= rank < world, got {rank!r}")
    from verity_vllm.program.frontend.rules import DEFAULT_RULES, vocabulary_for
    ruleset = rules if rules is not None else DEFAULT_RULES
    vocab = vocabulary_for(mode, profile)
    allowed_dims, config, theta = dict(allowed_dims or {}), dict(config or {}), tuple(theta)
    source_name: str = type(source).__name__ if isinstance(source, torch.nn.Module) else getattr(source, "__name__", "Derived")
    name = name or source_name
    source_desc = _describe_source(source, example_inputs, mode, profile, config, allowed_dims)
    prov: dict[str, Any] = {"schema": "verity_ir/derive-provenance/v1", "frontend": FRONTEND_VERSION, "derive": DERIVE_VERSION,
                            "torch": torch.__version__, "mode": mode, "profile": profile, "vocabulary": vocab.to_json(),
                            "ruleset": ruleset.to_json(), "registry": registry_version(), "source": source_desc,
                            "naming_hints": "nn_module_stack / node names are annotations only; identity comes from rule + registry"}
    holder: list = []
    try:
        flat_sd, ctx = _derive_flat(source, example_inputs, mode=mode, profile=profile, config=config, allowed_dims=allowed_dims,
                                    name=name, ruleset=ruleset, vocab=vocab, _ctx_holder=holder, exported=exported,
                                    registered_values=registered_values)
        head = (dict(prov), dict(ctx.unresolved_module_paths))
        rep = _finish_report(flat_sd, ctx, prov, mode=mode, ruleset=ruleset, rank=rank, check_registry=check_registry, dead_free=dead_free, correspondence=correspondence, theta=theta)
        if session is not None:
            session.append(DerivationTail(flat_sd, ctx, *head, name, mode, ruleset, check_registry, rank, rep.family, dead_free, theta))
        return rep
    except Refuse as e:
        prov["unsupported"] = e.u.to_json()
        _record_partial(prov, holder)
        return DeriveReport(None, e.u, prov)
    except FrontendError as e:
        u = Unsupported(_CODE_TO_REASON.get(e.code, "shape"), None, e.where, f"{e.code}: {e.message}")
        prov["unsupported"] = u.to_json()
        _record_partial(prov, holder)
        return DeriveReport(None, u, prov)


def _finish_report(flat_sd: SpecializedDefinition, ctx: Ctx, prov: dict, *, mode: str, ruleset: RuleSet, rank: tuple[int, int] | None,
                   check_registry: bool, dead_free: bool | str = False, correspondence: bool = True, theta: tuple[str, ...] = ()) -> DeriveReport:
    """The derivation's tail over the finished flat body: provenance, runtime correspondence, the flat Program and its digest, the
    family claims, `prov["result"]`.  Raises `Refuse` / `FrontendError` (the caller reports them)."""
    prov["weights"] = [w.to_json() for w in ctx.weights]
    prov["weight_roles"] = ctx.weight_roles
    prov["unresolved_module_paths"] = dict(sorted(ctx.unresolved_module_paths.items()))   # GATE-PATH residual
    prov["applications"] = ctx.applications
    prov["mutation_effects"] = ctx.mutation_effects
    prov["returned_inputs"] = ctx.returned_inputs
    prov["rules_applied"] = sorted({a["rule"] for a in ctx.applications})
    if ctx.registered_inputs is not None:
        prov["supplied"] = ctx.registered_inputs.layout()
    if ctx.run_inputs is not None:
        prov["run_inputs"] = ctx.run_inputs.layout()
    if ctx.checks:
        prov["checks"] = {"output": ctx.checks_output, "accept": 1, "of": [w for _, w in ctx.checks]}
    # runtime correspondence (`runtime-correspondence/v1`): per Call the owning module, the source occurrence and the return /
    # argument slots -- data beside the Program, under ONE annotation namespace, digested apart (annotations never enter program_digest)
    from verity_vllm.program.frontend.emit import NAMESPACE as _CORR
    from verity_vllm.program.frontend.emit import kept_correspondence
    body, corr_doc, corr_digest, hints = kept_correspondence(ctx, flat_sd.body, rank=rank, registered=dead_free, emit=correspondence)
    flat_sd = flat_sd if body is flat_sd.body else _flat_definition(flat_sd.defn.name[len("Derived["):-1], body, flat_sd.bindings["SRC"], mode)
    holes = {"survey_holes": ctx.state["survey_holes"]} if ctx.state.get("survey_holes") else {}
    flat_program = Program(flat_sd, annotations={"derive": prov, "node_hints": {str(k): v for k, v in hints.items()},
                                                 **({} if corr_doc is None else {_CORR: corr_doc}), **holes},
                           registered=ctx.registered_inputs.names if ctx.registered_inputs is not None else (),
                           theta=(() if ctx.weights_param is None else (flat_sd.signature()[0][ctx.weights_param][0],)) + theta)
    check_identity(flat_program, check_registry=check_registry)
    flat_digest = program_digest(flat_desc := encode_program(flat_program))
    program, family_report = flat_program, None
    for fam in ruleset.families:
        if fam.modes is not None and mode not in fam.modes:
            continue
        a = ("a family claim keeps no registered parameter" if ctx.registered_values else
             "a family claim keeps no run parameter or checks" if ctx.run_inputs is not None or ctx.checks else
             fam.applies_program(flat_sd, ctx))
        if a is not True:
            prov.setdefault("families_not_applied", []).append({"rule": fam.rule_id, "why": a})
            continue
        family_report = fam.claim(flat_sd, ctx)  # verified isomorphism or Refuse(identity)/no-claim
        if family_report is not None and family_report.get("matched"):
            program = Program(family_report.pop("program"), annotations={"derive": prov})
            check_identity(program, check_registry=check_registry)
            prov["family"] = family_report
            break
        elif family_report is not None:
            prov.setdefault("families_not_matched", []).append(family_report)
            family_report = None
    digest = flat_digest if program is flat_program else program_digest(program)
    dead = _dead_gates(program, limit=32)
    # a family claim replaces the flat body by the family's: the correspondence describes the FLAT Calls and stays with the flat root
    corr_applies = program is flat_program and corr_doc is not None
    prov["result"] = {"root": descriptor_id(program.fn), "digest": digest, "gates": program.gates, "flat_root": descriptor_id(flat_sd),
                      "flat_digest": flat_digest, "dead_gates": dead.total,
                      "correspondence_digest": corr_digest if corr_applies else None,
                      "correspondence": corr_doc and {"schema": corr_doc["schema"], "calls": len(corr_doc["calls"]), "occurrences": len(corr_doc["occurrences"]),
                                                      "applies_to": "program_fn" if corr_applies else f"flat body only ({descriptor_id(flat_sd)}); not attached", "digest": corr_digest}}
    return DeriveReport(program, None, prov, digest, descriptor_id(program.fn), program.gates, dead, flat_digest, descriptor_id(flat_sd),
                        prov.get("family"), flat_sd, corr_digest if corr_applies else None, flat_desc if program is flat_program else None)


def _record_partial(prov: dict, holder: list) -> None:
    """On refusal, the provenance still reports what fired before it (XFER-G8) — clearly marked as belonging to NO program."""
    if not holder:
        return
    ctx = holder[0]
    prov["weights"] = [w.to_json() for w in ctx.weights]
    prov["applications_before_refusal"] = ctx.applications
    prov["rules_applied_before_refusal"] = sorted({a["rule"] for a in ctx.applications})
    prov["note"] = "refused: no program exists; the applications listed fired before the refusing node and are evidence for the report only"


def _describe_source(source, example_inputs, mode, profile, config, allowed_dims) -> dict:
    if isinstance(source, torch.nn.Module):
        cls = type(source)                               # `DERIVE_SOURCE`: the source's declared, versioned id; SRC's class when present
        d = {"kind": "nn.Module", "class": getattr(source, "DERIVE_SOURCE", None) or f"{cls.__module__}.{cls.__qualname__}", "python_class": f"{cls.__module__}.{cls.__qualname__}",
             "submodules": [f"{n}: {type(m).__qualname__}" for n, m in source.named_modules() if n]}
    else:
        d = {"kind": "callable", "name": getattr(source, "__qualname__", repr(source))}
    d["example_inputs"] = [(str(getattr(x, "dtype", type(x).__name__)), list(getattr(x, "shape", ()))) for x in example_inputs]
    d["config"] = _jsonable(config)
    d["allowed_dims"] = _jsonable({k: sorted(v) if isinstance(v, (set, frozenset)) else v for k, v in allowed_dims.items()})
    d["mode"], d["profile"] = mode, profile
    return d


def _export(source, example_inputs):
    # inputs are shape/dtype carriers only: export traces with fake tensors and never reads a value. They live on
    # the module's device (meta for meta-constructed modules) so that export sees one device.
    dev = torch.device("meta")
    if isinstance(source, torch.nn.Module):
        for p in list(source.parameters()) + list(source.buffers()):
            dev = p.device
            break
    metas = []
    for x in example_inputs:
        if not isinstance(x, torch.Tensor):
            refuse("side_input", None, None, f"example input of type {type(x).__name__}: only tensors are prescribed inputs")
        metas.append(torch.empty(x.shape, dtype=x.dtype, device=dev))
    if isinstance(source, torch.nn.Module):
        mod = source
    else:
        class _Wrap(torch.nn.Module):
            def forward(self, *a):
                return source(*a)
        mod = _Wrap()
    try:
        # factory calls inside forward (`torch.ones(1, dtype=bool)`, a step's zero) land on the module's device
        with torch.no_grad(), torch.device(dev):
            return torch.export.export(mod, tuple(metas), strict=False)
    except Refuse:
        raise
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        if "data-dependent" in msg or "GuardOnDataDependentSymNode" in type(e).__name__ or "cond" in msg:
            refuse("dynamic_control_flow", None, None, f"torch.export: {type(e).__name__}: {msg}")
        refuse("export", None, None, f"torch.export failed: {type(e).__name__}: {msg}")


def _weights_of(module: torch.nn.Module | None) -> tuple[list[WeightLeaf], dict[str, int]]:
    """Distinct parameter/buffer tensors in module order; tied tensors collapse to one leaf with aliases."""
    if module is None:
        return [], {}
    leaves: list[WeightLeaf] = []
    by_id: dict[int, int] = {}
    fqn_to_leaf: dict[str, int] = {}
    items: list[tuple[str, torch.Tensor, str]] = [(n, p, "parameter") for n, p in module.named_parameters(remove_duplicate=False)]
    items += [(n, b, "buffer") for n, b in module.named_buffers(remove_duplicate=False)]
    for fqn, t, kind in items:
        key = id(t)
        if key in by_id:
            i = by_id[key]
            leaves[i].aliases = tuple(leaves[i].aliases) + (fqn,)
            fqn_to_leaf[fqn] = i
            continue
        i = len(leaves)
        by_id[key] = i
        fqn_to_leaf[fqn] = i
        leaves.append(WeightLeaf(fqn, (), kind, type_of_tensor(t), tuple(int(s) for s in t.shape), str(t.dtype).removeprefix("torch."), i))
    return leaves, fqn_to_leaf


def _derive_flat(source, example_inputs, *, mode, profile, config, allowed_dims, name, ruleset, vocab,
                 _ctx_holder: list | None = None, exported=None, registered_values: bool = False) -> tuple[SpecializedDefinition, Ctx]:
    module = source if isinstance(source, torch.nn.Module) else None
    ep = exported if exported is not None else _export(source, example_inputs)
    sig = ep.graph_signature
    leaves, fqn_to_leaf = _weights_of(module)
    placeholder_to_weight: dict[str, int] = {}
    for ph, fqn in list(sig.inputs_to_parameters.items()) + list(sig.inputs_to_buffers.items()):
        if fqn not in fqn_to_leaf:
            refuse("hidden_state", ph, None, f"export lifted {fqn!r} but the module does not own it as a parameter/buffer")
        placeholder_to_weight[ph] = fqn_to_leaf[fqn]
    lifted = dict(sig.inputs_to_lifted_tensor_constants)
    if getattr(sig, "inputs_to_lifted_custom_objs", None):
        refuse("side_input", None, None, f"export lifted custom objects {dict(sig.inputs_to_lifted_custom_objs)}")
    # derived tables a rule declares (HF RoPE cos/sin): extra fields of the weights record, derivation recorded
    for r in ruleset.rules:
        if r.modes is not None and mode not in r.modes:
            continue
        for fname, ftype, derivation in r.declare_inputs(ep, module, mode):
            if fname in fqn_to_leaf:
                refuse("hidden_state", None, None, f"rule {r.rule_id} declares input {fname!r} but the module owns a tensor of that name")
            shp, w = shape_of(ftype)
            fqn_to_leaf[fname] = len(leaves)
            leaves.append(WeightLeaf(fname, (), "derived_table", ftype, tuple(shp), derivation.get("dtype", f"{w}-bit"), len(leaves),
                                     {"rule": r.rule_id, **derivation}))

    # parameters of the flat function: the user inputs (by placeholder name), the `run` values a rule declares (`Rule.run_inputs`,
    # finalised after the translation), the registered `values` (finalised likewise: its fields are the values the Definitions
    # read), the registered tables the rules declare, and the weights record
    user_names = list(sig.user_inputs)
    if len(user_names) != len(example_inputs):
        refuse("shape", None, None, f"{len(example_inputs)} example inputs vs {len(user_names)} graph inputs")
    ph_nodes = {n.name: n for n in ep.graph.nodes if n.op == "placeholder"}
    params: list[tuple[str, Type]] = []
    for un in user_names:
        v = ph_nodes[un].meta.get("val")
        if not isinstance(v, torch.Tensor):
            refuse("dynamic_control_flow", ph_nodes[un], None, f"input {un} is not a tensor (symbolic scalar inputs are not static structure)")
        params.append((un, type_of_tensor(v)))
    run_inputs = (RunInputs.declare(params) if any(r.run_inputs(ep, mode) for r in ruleset.rules if r.modes is None or mode in r.modes)
                  else None)
    registered_inputs = RegisteredInputs.declare(params, ruleset, ep, mode) if registered_values else None
    wrec_t = RecordType(*[(w.name, w.type) for w in leaves]) if leaves else None
    if wrec_t is not None:
        params.append(("weights", wrec_t))

    B = Builder(tuple(params), Value(1))  # ret type is set once the outputs are known
    pcolls = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    user_inputs = {un: pcolls[i] for i, un in enumerate(user_names)}
    weights_coll = pcolls[-1] if wrec_t is not None else None
    ctx = Ctx(mode=mode, profile=profile, config=config, allowed_dims=allowed_dims, name=name, ruleset=ruleset, vocab=vocab, module=module,
              ep=ep, B=B, user_inputs=user_inputs, weights_coll=weights_coll, weights=leaves, placeholder_to_weight=placeholder_to_weight,
              lifted_constants=lifted, registered_inputs=registered_inputs, weights_param=len(params) - 1 if wrec_t is not None else None,
              run_inputs=run_inputs)
    if _ctx_holder is not None:
        _ctx_holder.append(ctx)
    out = _translate(ctx)
    if run_inputs is not None:
        try:
            run_inputs.finish(params)
        except ValueError as e:
            refuse("side_input", None, None, str(e))
    if registered_inputs is not None:
        try:
            registered_inputs.finish(params)
        except ValueError as e:
            refuse("numerics_unregistered", None, None, str(e))
    if run_inputs is not None or registered_inputs is not None:
        B.params = tuple(params)
    B.ret_type = out.type
    body = B.finish(out)
    src_static = {"class": _describe_source(source, example_inputs, mode, profile, config, allowed_dims).get("class", name), "mode": mode,
                  "profile": profile or "", "inputs": [list(shape_of(t)[0]) for _, t in params[:len(user_names)]],
                  "config_sha256": hashlib.sha256(canonical_json(_jsonable(config))).hexdigest()[:16]}
    return _flat_definition(name, body, src_static, mode), ctx


def _flat_definition(name: str, body, src_static: dict, mode: str) -> SpecializedDefinition:
    """The flat program's definition over `body`.  It never carries a registered name: identity is claimed only through a verified family match."""
    params, ret = tuple(body.params), body.ret_type
    defn = CompositeDefinition(f"Derived[{name}]", 1, ("SRC",), lambda S: (params, ret), replayer(body),
                               doc=f"derived ({mode}) from {src_static['class']}", conformance="derived: see provenance", register=False)
    sd = ir_bind(defn, SRC=src_static)
    sd.set_body(body)
    return sd


def _translate(ctx: Ctx) -> Coll:
    ep = ctx.ep
    sig = ep.graph_signature
    out_node = None
    # control flow is refused up front (whatever node would fail first, the program is not a fixed computation)
    for node in ep.graph.nodes:
        t = node.target if node.op == "call_function" else None
        if isinstance(t, torch._ops.HigherOrderOperator) and t._name in _CONTROL_FLOW_HOPS:
            ctx.current = node
            refuse("dynamic_control_flow", node, ctx.module_path(node),
                   f"higher-order op {t} (torch.cond / while_loop / map): control flow must be bounded and static — unroll it within "
                   f"`allowed_dims` or derive a fixed computation")
    for node in ep.graph.nodes:
        ctx.current = node
        if node.op == "placeholder":
            if node.name in ctx.user_inputs:
                ctx.env[node] = ctx.user_inputs[node.name]
            elif node.name in ctx.placeholder_to_weight:
                assert ctx.weights_coll is not None
                w = ctx.weights[ctx.placeholder_to_weight[node.name]]
                ctx.env[node] = ctx.weights_coll[w.index]
            elif node.name in ctx.lifted_constants:
                # a constant only a rule may account for: bind a marker; reading it without a rule is refused
                ctx.env[node] = _LiftedConstant(node, ctx.lifted_constants[node.name], node.meta.get("val"))
            else:
                refuse("side_input", node, None, f"placeholder {node.name} is neither a user input nor a parameter/buffer")
        elif node.op == "output":
            out_node = node
        elif node.op == "get_attr":  # a tensor constant of a traced graph (`make_fx`), as export lifts one
            if not isinstance(getattr(ep, "constants", {}).get(node.target), torch.Tensor):
                refuse("dynamic_control_flow", node, ctx.module_path(node), f"subgraph attribute {node.target} (higher-order op body)")
            ctx.env[node] = _LiftedConstant(node, node.target, node.meta.get("val", ep.constants[node.target]))
        elif node.op == "call_function":
            if node in ctx.env:
                continue  # absorbed by a pattern rule that fired at an earlier node of its chain
            _translate_node(ctx, node)
        else:
            refuse("decomposition", node, ctx.module_path(node), f"graph node of kind {node.op} has no derivation rule")
    assert out_node is not None
    ctx.current = out_node
    for spec in sig.output_specs:
        if spec.kind.name != "USER_OUTPUT":
            refuse("mutation", out_node, None, f"export recorded a {spec.kind.name} output ({spec.arg}); mutation of inputs/buffers must be owned by a rule")
    outs = []
    for a in out_node.args[0]:
        if a is None:
            refuse("shape", out_node, None, "a returned value is None")
        v = ctx.val(a)
        if isinstance(v, (int, float, bool)):
            refuse("side_input", out_node, None, f"returned Python scalar {v!r}; outputs must be values")
        outs.append(v.returned(ctx, a) if hasattr(v, "returned") else ctx.coll(a, "returned value"))
    if not outs:
        refuse("shape", out_node, None, "the source returns nothing")
    # returned inputs / repeated returns are representable (a return is a reference sequence) — record them explicitly
    for i, o in enumerate(outs):
        for space, idx, lo, hi in _runs(o.refs):
            if space == "p":
                ctx.returned_inputs.append({"output": i, "param": idx, "leaves": [lo, hi],
                                            "weights": idx == ctx.weights_param})
    SEED.in_derive(ctx, out_node)        # a stochastic request's seed rule (its `seed_rule.PARAMS`), then the length rule under
    SR.in_derive(ctx, outs, out_node)    # the config's `stop`: the Program's last Calls and checks
    if ctx.checks:
        ctx.checks_output = len(outs)
        outs.append(Coll(Array(len(ctx.checks), Value(1)), _concat_refs([c.refs for c, _ in ctx.checks])))
    ret = outs[0] if len(outs) == 1 else Coll(TupleT(*[o.type for o in outs]), _concat_refs([o.refs for o in outs]))
    return ret


def _concat_refs(parts):
    return concat(list(parts))


@dataclass(eq=False)
class _LiftedConstant:
    """A tensor constant torch.export lifted out of the forward. Not a value: reading it is `hidden_state`
    unless a rule (e.g. one that turns a RoPE table into an explicit Input) claims it."""
    node: Any
    name: str
    meta: Any


def _translate_node(ctx: Ctx, node) -> None:
    t = node.target
    if t is operator.getitem:
        base = ctx.val(node.args[0])
        if isinstance(base, _LiftedConstant):
            _refuse_constant(ctx, node, base)
        if isinstance(base, (list, tuple, Coll)):
            ctx.define(node, base[node.args[1]], view_of=node.args[0])
        elif isinstance(base, StaticIndex):
            ctx.define(node, StaticIndex(base.tensor[node.args[1]]), view_of=node.args[0])
        else:
            refuse("shape", node, ctx.module_path(node), f"getitem on {type(base).__name__}")
        return
    for a in node.args:
        if isinstance(a, torch.fx.Node) and isinstance(ctx.env.get(a), _LiftedConstant):
            cands = [r for r in ctx.ruleset.for_target(t, ctx.mode) if getattr(r, "accepts_constants", False)]
            if not cands:
                if _mutates(t) and not any(isinstance(r, (InPlaceRule, StateRule)) for r in ctx.ruleset.for_target(t, ctx.mode)):
                    refuse("mutation", node, ctx.module_path(node),
                           f"{t} mutates an operand and no InPlaceRule/StateRule accepts it (it also reads the lifted constant "
                           f"{ctx.env[a].name!r} — a literal written into a tensor is a mutation first)")
                _refuse_if_write_only(ctx, node)
                _refuse_constant(ctx, node, ctx.env[a])
    if isinstance(t, torch._ops.HigherOrderOperator):
        refuse("dynamic_control_flow", node, ctx.module_path(node),
               f"higher-order op {t} (torch.cond / while_loop / map): control flow must be bounded and static — unroll it within "
               f"`allowed_dims` or derive a fixed computation")
    reasons: list[str] = []
    for rule in ctx.ruleset.for_target(t, ctx.mode):
        verdict = rule.applies(node, ctx)
        if verdict is True:
            k0, before = ctx.begin_emit(node)
            rule.emit(node, ctx)
            ctx.note_emitted(node, k0, before)
            if node not in ctx.env and not isinstance(rule, (InPlaceRule, StateRule)):
                refuse("shape", node, ctx.module_path(node), f"{rule.rule_id} accepted the node but bound no value (internal)")
            if node not in ctx.env:
                ctx.env[node] = None  # an in-place op with no tensor result
            return
        reasons.append(f"{rule.rule_id}: {verdict}")
    _no_rule(ctx, node, reasons)


def _refuse_constant(ctx: Ctx, node, c: _LiftedConstant) -> NoReturn:
    meta = c.meta
    desc = f"{getattr(meta, 'dtype', '?')}{list(getattr(meta, 'shape', []))}"
    refuse("hidden_state", node, ctx.module_path(node),
           f"tensor constant {c.name!r} ({desc}) was created inside forward and lifted by export; it is hidden state — make it a "
           f"registered buffer / parameter (an explicit Input with recorded derivation) or a rule must account for it")


def _refuse_if_write_only(ctx: Ctx, node) -> None:
    """A value whose only consumers are unaccepted mutating ops (`y[0] = 0.0`: scalar_tensor / lift_fresh_copy -> fill_ / copy_)
    exists only to be written into a tensor: the mutation is the reason (XFER-G6), not the literal."""
    def unaccepted_writer(u) -> bool:
        t = getattr(u, "target", None)
        return (isinstance(t, torch._ops.OpOverload) and t._schema.is_mutable and t is not aten.detach_.default
                and not any(isinstance(r, (InPlaceRule, StateRule)) for r in ctx.ruleset.for_target(t, ctx.mode)))

    def sinks(n, depth=0):
        """The non-view consumers reached from n through view-like ops (view/expand/select/...), or None if any is not a writer."""
        out = []
        for u in n.users:
            if unaccepted_writer(u):
                out.append(u)
            elif depth < 4 and isinstance(getattr(u, "target", None), torch._ops.OpOverload) and not u.target._schema.is_mutable \
                    and u.target.namespace == "aten" and u.target.__name__.split(".")[0] in _VIEW_LIKE and list(u.users):
                sub = sinks(u, depth + 1)
                if sub is None:
                    return None
                out += sub
            else:
                return None
        return out

    writers = sinks(node)
    if writers:
        w0 = writers[0]
        refuse("mutation", w0, ctx.module_path(w0), f"{w0.target} mutates an operand and no InPlaceRule/StateRule accepts it "
                                                    f"(its operand {node.name} = {node.target} exists only for that write)")


_VIEW_LIKE = {"view", "reshape", "expand", "select", "slice", "unsqueeze", "squeeze", "permute", "transpose", "t", "alias", "detach",
              "lift_fresh_copy", "lift_fresh", "_to_copy", "to", "clone", "contiguous"}


def _mutates(t) -> bool:
    schema = getattr(t, "_schema", None)
    if schema is None:
        return False
    return any(arg.alias_info is not None and arg.alias_info.is_write for arg in schema.arguments)


_CONTROL_FLOW_HOPS = {"cond", "while_loop", "map", "map_impl", "scan", "associative_scan", "flat_apply"}
_SIDE_INPUT_OPS = {"randn", "rand", "randint", "normal", "bernoulli", "multinomial", "nonzero", "_local_scalar_dense", "item", "masked_select",
                   "unique", "unique_consecutive", "dropout", "native_dropout", "rand_like", "randn_like"}
_DECOMPOSED_HINTS = {
    "softmax": "attention appears decomposed (softmax); the profile binds fused attention only — use the fused attention op or file a kind",
    "_softmax": "attention appears decomposed (softmax); the profile binds fused attention only",
    "bmm": "batched matmul outside a bound pattern (decomposed attention?)",
    "scaled_dot_product_attention": "SDPA without a StateRule binding: derive attention through the vLLM-style attention op or file a kind",
    "_scaled_dot_product_flash_attention": "SDPA kernel without binding",
    "rsqrt": "RMSNorm appears decomposed (rsqrt/mean/pow): the profile binds the fused norm kernels only",
    "mean": "RMSNorm/LayerNorm appears decomposed (mean)",
    "pow": "RMSNorm appears decomposed (pow)",
    "silu": "SiLU alone: the profile binds fused silu_and_mul only",
    "gelu": "GELU has no registered kind (file to NUMERIC)",
    "native_layer_norm": "LayerNorm has no registered kind (file to NUMERIC)",
    "layer_norm": "LayerNorm has no registered kind (file to NUMERIC)",
}


def _no_rule(ctx: Ctx, node, reasons: list[str]) -> NoReturn:
    t = node.target
    mp = ctx.module_path(node)
    why = ("; ".join(reasons)) if reasons else "no rule considers this op"
    if isinstance(t, torch._ops.HigherOrderOperator):
        refuse("dynamic_control_flow", node, mp, f"higher-order op {t}: data-dependent control flow / functionalised region has no fixed Verity meaning")
    if isinstance(t, torch._ops.OpOverload):
        opname = t.__name__.split(".")[0]
        if t.namespace == "aten" and opname in _SIDE_INPUT_OPS:
            refuse("side_input", node, mp, f"{t} reads host/random/data-dependent state; the computation would not be fixed (boundary 2)")
        if t._schema.is_mutable:
            refuse("mutation", node, mp, f"{t} mutates an operand and no InPlaceRule/StateRule accepted it ({why})")
        # a value that exists only to be written into a tensor (`y[0] = 0.0`: scalar_tensor -> copy_) is a mutation first (XFER-G6)
        _refuse_if_write_only(ctx, node)
        if t.namespace != "aten":
            refuse("custom_op_without_binding", node, mp, f"custom op {t} has no rule binding it to a registered kind ({why})")
        # a value that only exists to steer control flow (torch.cond predicate, while_loop carry) is control flow
        if any(isinstance(getattr(u, "target", None), torch._ops.HigherOrderOperator) for u in node.users):
            hop = next(u.target for u in node.users if isinstance(getattr(u, "target", None), torch._ops.HigherOrderOperator))
            refuse("dynamic_control_flow", node, mp, f"{t} feeds higher-order op {hop} (data-dependent control flow); no derivation rule for it")
        # a declining rule may name the reason it would report ("numerics_unregistered: ...", "dynamic_control_flow: ...")
        for r in reasons:
            msg = r.split(": ", 1)[-1]
            for reason in REASONS:
                if msg.startswith(reason + ":"):
                    refuse(reason, node, mp, f"{t}: {why}")
        hint = _DECOMPOSED_HINTS.get(opname)
        if hint is not None:
            refuse("decomposition", node, mp, f"{t}: {hint}" + (f" ({why})" if reasons else ""))
        if reasons:
            refuse("decomposition", node, mp, f"{t}: {why}")
        refuse("numerics_unregistered", node, mp, f"ATen op {t} has no derivation rule and no registered kind in {ctx.mode} mode "
                                                 f"(vocabulary {ctx.vocab.name!r})")
    refuse("custom_op_without_binding", node, mp, f"Python callable {t!r} in the graph")


__all__ = ["derive", "derive_report", "DeriveReport", "DerivationTail", "Unsupported", "Refuse", "refuse", "Ctx", "StaticIndex", "Alloc", "WeightLeaf",
           "REASONS", "DERIVE_VERSION", "fx_arg_nodes"]
