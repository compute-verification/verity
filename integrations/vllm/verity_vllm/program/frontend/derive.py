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
   tensor constants are refused (`hidden_state`) unless a rule accounts for them.
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
from dataclasses import dataclass
from typing import Any, Callable, NoReturn, Sequence

import torch

from verity.ml.circuits.codec import _spec_id, canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, Builder, CompositeDefinition, PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.defs import bind as ir_bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll
from verity.ml.circuits.refs import runs as _refs_runs
from verity.ml.circuits.liveness import DeadGateReport, dead_gates as _dead_gates
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type, Value
from verity_vllm.program.frontend.outcomes import _CODE_TO_REASON, REASONS, Refuse, Unsupported, _jsonable, refuse
from verity_vllm.program.frontend.provenance import registry_version
from verity_vllm.program.frontend.registration import Registration
from verity_vllm.program.frontend.report import DeriveReport, DeriveSession
from verity_vllm.program.replay import replayer
from verity_vllm.program.frontend.torch_frontend import FRONTEND_VERSION, FrontendError, check_identity, shape_of, type_of, type_of_tensor, width_of
from verity_vllm.program.frontend.rules.base import ANY, InPlaceRule, Rule, RuleSet, StateRule, ViewRule

DERIVE_VERSION = "verity_vllm.program.frontend.derive/2026-09-09.d4"  # d4: G-RH-16 (rotary table without terminal cast), G-RH-17 (0-d buffer operand in Elementwise) (lane DERIVE-4); d2: HF 5.x plumbing closures G-RH-1/-2/-4/-6/-8/-9/-10 (lane DERIVE-2); d3: G-RH-14 (embedding padding_idx), G-RH-7 (bf16 inv_freq rotary table) (lane DERIVE-3)

aten = torch.ops.aten


# ---------------------------------------------------------------------------------------------------------
# non-value objects the translation carries between nodes
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticIndex:
    """An integer tensor whose *values* are statically known (positions, slot tables, `arange`s). Not a Verity
    value: it is static structure consumed by rules (which table row, which cache slot). It never becomes a
    gate, is never returned, and rules that consume it record the values they used."""
    tensor: torch.Tensor  # a real CPU integer tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    def flat(self) -> list[int]:
        return [int(x) for x in self.tensor.reshape(-1).tolist()]

    def __repr__(self) -> str:
        return f"StaticIndex({self.tensor.dtype}, {list(self.shape)})"


@dataclass(eq=False)
class Alloc:
    """An uninitialised allocation (`empty`, `empty_like`, `new_empty`): the write target of an in-place rule.
    Reading it before a rule writes it is refused."""
    type: Type
    node: Any


@dataclass
class WeightLeaf:
    """One distinct parameter/buffer tensor of the source module -> one field of the `weights` record."""
    name: str  # canonical fqn (first in module order)
    aliases: tuple[str, ...]  # other fqns bound to the same tensor (tied parameters)
    kind: str  # "parameter" | "buffer"
    type: Type
    shape: tuple[int, ...]
    dtype: str
    index: int
    derivation: dict | None = None  # kind == "derived_table": how the source computes it (rule, formula, source buffer, positions)

    def to_json(self) -> dict:
        d = {"name": self.name, "aliases": list(self.aliases), "kind": self.kind, "shape": list(self.shape), "dtype": self.dtype,
             "field": self.index}
        if self.derivation is not None:
            d["derivation"] = self.derivation
        return d


# ---------------------------------------------------------------------------------------------------------
# reference runs: (space, idx, lo, hi) intervals of a reference sequence — for alias overlap and matching
# ---------------------------------------------------------------------------------------------------------


_runs = _refs_runs  # `verity.ml.circuits.refs.runs` (pure; moved there so `rules.family.match_program` imports without torch)


def _runs_intersect(a: list[tuple[str, int, int, int]], b: list[tuple[str, int, int, int]]) -> bool:
    by: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for space, idx, lo, hi in a:
        by.setdefault((space, idx), []).append((lo, hi))
    for space, idx, lo, hi in b:
        for lo2, hi2 in by.get((space, idx), ()):
            if lo < hi2 and lo2 < hi:
                return True
    return False


# ---------------------------------------------------------------------------------------------------------
# the derivation context: what rules see and how they emit
# ---------------------------------------------------------------------------------------------------------


def module_registrations(module) -> tuple[set[str], dict[str, str]]:
    """[R15 moe GATE-PATH] (every registered path of every submodule, duplicate path -> first registration).  `named_modules()` (the
    implementation's vocabulary, what a hook-based collector names its tensors by) yields each module ONCE under its first path;
    `remove_duplicate=False` yields the other registrations of the same object (vLLM's MoERunner re-registers the block's router as
    its own `gate`; `get_rope` shares one RotaryEmbedding across layers).  Not a torch module: nothing known."""
    if not isinstance(module, torch.nn.Module):
        return set(), {}
    paths: set[str] = set()
    first: dict[int, str] = {}
    canon: dict[str, str] = {}
    for name, m in module.named_modules(remove_duplicate=False):
        paths.add(name)
        k = id(m)
        if k in first:
            if name != first[k]:
                canon[name] = first[k]
        else:
            first[k] = name
    return paths, canon


class _Storage(dict):
    """node -> storage base node, and `aliases[base]`: the nodes currently mapped to `base` (the group `Ctx.rebind` reads)."""

    def __init__(self) -> None:
        super().__init__()
        self.aliases: dict[Any, dict[Any, None]] = {}

    def __setitem__(self, node, base) -> None:
        self.pop(node, None)
        super().__setitem__(node, base)
        self.aliases.setdefault(base, {})[node] = None

    def setdefault(self, node, base=None):
        if node not in self:
            self[node] = base
        return self[node]

    def pop(self, node, *default):
        if node in self:
            self.aliases[dict.__getitem__(self, node)].pop(node, None)
        return super().pop(node, *default)


class Ctx:
    """State of one derivation. Rules read the graph through it and emit through `call` / `define` / `rebind`."""

    def __init__(self, *, mode: str, profile: str | None, config: dict | None, allowed_dims: dict, name: str, ruleset: RuleSet,
                 vocab, module, ep, B: Builder, user_inputs: dict[str, Coll], weights_coll: Coll | None, weights: list[WeightLeaf],
                 placeholder_to_weight: dict[str, int], lifted_constants: dict[str, str], registration: Registration | None = None,
                 weights_param: int | None = None, committed_inputs: dict[Any, Coll] | None = None,
                 committed_units: list[dict[str, Any]] | None = None,
                 committed_writes: dict[Any, list[Any]] | None = None) -> None:
        self.mode, self.profile, self.config, self.allowed_dims, self.name = mode, profile, dict(config or {}), dict(allowed_dims or {}), name
        self.ruleset, self.vocab, self.module, self.ep, self.B = ruleset, vocab, module, ep, B
        self.user_inputs, self.weights_coll, self.weights = user_inputs, weights_coll, weights
        self.placeholder_to_weight, self.lifted_constants = placeholder_to_weight, lifted_constants
        self.registration, self.weights_param = registration, weights_param   # the registered inputs (`registered_values`)
        self.committed_inputs = dict(committed_inputs or {})
        self.committed_units = list(committed_units or [])
        self.committed_writes = dict(committed_writes or {})
        self.env: dict[Any, Any] = {}
        self.storage: _Storage = _Storage()  # node -> storage base node (alias groups); absent: own storage
        self.stale: dict[Any, str] = {}  # node -> why its value may no longer be read (alias of a mutated tensor)
        self.written_allocs: set = set()
        self.mutation_effects: list[dict] = []
        self.returned_inputs: list[dict] = []
        self.applications: list[dict] = []
        self.node_hints: dict[int, str] = {}  # verity node index -> naming hint
        self.node_sources: dict[int, str] = {}  # verity node index -> fx node name
        self.node_fx: dict[int, Any] = {}  # verity node index -> the fx node it was lowered from (runtime correspondence; `emit_correspondence`)
        self.occurrence_bindings: dict[Any, dict] = {}  # fx node -> {"in", "out", "written"} bindings frozen at lowering (`note_emitted`)
        self.state: dict[str, Any] = {}  # StateRule scratch (cache ownership etc.)
        self.weight_roles: dict[str, dict] = {}  # weight field name -> {rule, role, statics} (first use)
        self.current: Any = None  # fx node being translated
        #: set while `relabel.rank_report` re-emits a node: `call` hands it each call instead of appending to `B` (the template's slot)
        self.emit_into: Callable[..., Coll] | None = None
        # [R15 moe GATE-PATH] every path a module of the source is registered under -> its first registration (`stack_path`)
        self._module_paths, self._canonical_path = module_registrations(module)
        #: unrooted stack entries that resolved under NO registered path (kept bare) -> bare path -> occurrences; recorded in the
        #: derive provenance (`unresolved_module_paths`) so the merged-name failure mode is visible at derive, not first at Commit coverage
        self.unresolved_module_paths: dict[str, int] = {}

    # -- reading ----------------------------------------------------------------------------------------
    def val(self, a):
        """Resolve a graph argument: fx nodes -> their translated value; lists recursively; scalars as is."""
        if isinstance(a, torch.fx.Node):
            if a in self.stale:
                refuse("mutation", self.current, self.module_path(self.current),
                       f"{a.name} is an alias of a tensor mutated in place after this alias was taken ({self.stale[a]}); "
                       f"the source reads a stale alias — restructure the source or add an InPlaceRule that covers the alias")
            if a not in self.env:
                refuse("shape", self.current, self.module_path(self.current), f"{a.name} has no translated value (internal ordering error)")
            return self.env[a]
        if isinstance(a, (list, tuple)):
            return [self.val(x) for x in a]
        return a

    def coll(self, a, what: str = "operand") -> Coll:
        v = self.val(a)
        if isinstance(v, Coll):
            return v
        if isinstance(v, Alloc):
            refuse("mutation", self.current, self.module_path(self.current),
                   f"{what} reads the uninitialised allocation {v.node.name} before any rule wrote it")
        if isinstance(v, StaticIndex):
            refuse("side_input", self.current, self.module_path(self.current),
                   f"{what} uses a statically known index tensor {v} as a value; only rules that take static structure may consume it")
        if type(v).__name__ == "Consumed":
            refuse("decomposition", self.current, self.module_path(self.current),
                   f"{what} reads an intermediate of a recognised pattern ({v}) from outside it; the pattern's kind has no such output")
        refuse("shape", self.current, self.module_path(self.current), f"{what} is not a tensor value ({type(v).__name__})")

    def static_index(self, a, what: str = "index") -> StaticIndex:
        v = self.val(a)
        if isinstance(v, StaticIndex):
            return v
        if isinstance(v, Coll):
            refuse("dynamic_control_flow", self.current, self.module_path(self.current),
                   f"{what} is a computed value, not a statically known index; the derived computation would depend on data (boundary 2)")
        refuse("shape", self.current, self.module_path(self.current), f"{what} is not an index tensor ({type(v).__name__})")

    def tensor_meta(self, node):
        return node.meta.get("val")

    def dtype_of(self, node) -> torch.dtype | None:
        v = self.tensor_meta(node)
        return getattr(v, "dtype", None)

    def shape_of_node(self, node) -> tuple[int, ...] | None:
        v = self.tensor_meta(node)
        if hasattr(v, "shape"):
            return tuple(int(s) for s in v.shape)
        return None

    def module_path(self, node) -> str | None:
        if node is None or not isinstance(node, torch.fx.Node):
            return None
        stack = node.meta.get("nn_module_stack") or {}
        if not stack:
            return None
        _path, cls = list(stack.values())[-1]
        path = self.stack_path(node)
        return f"{path or '<root>'} ({cls.__name__ if isinstance(cls, type) else str(cls).rsplit('.', 1)[-1]})"

    def module_class(self, node) -> str | None:
        stack = node.meta.get("nn_module_stack") or {}
        if not stack:
            return None
        _, cls = list(stack.values())[-1]
        return cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"

    def stack_path(self, node) -> str:
        """The producing module's path in the source module's OWN vocabulary (`named_modules()`), from the node's `nn_module_stack`.
        [R15 moe GATE-PATH] the export names a module by the attribute path the trace took; a module reached through a holder that is
        not its registering parent comes out UNROOTED (vLLM 0.28's MoERunner keeps the router as `self.gate` and calls it inside its own
        forward: the router Gemm's stack ends in a bare `gate`, one name for 48 layers) or under a DUPLICATE registration
        (`...mlp.experts.gate` where `named_modules()` yields `...mlp.gate`).  Rooted entries are kept verbatim (a shared RoPE instance
        stays under each layer's path, as the manifest's alias rule expects); an unrooted entry is resolved relative to its parent entry
        and, when that resolved path is a duplicate registration, replaced by the module's first registration -- the collector's name.
        Stated consequences (global M-0595): a ROOTED duplicate (an export that yields the full `...mlp.experts.gate` itself) stays under
        the duplicate name while the collector binds `...mlp.gate` -- C1 coverage then FAILS it as missing (fail-closed, never a silent
        merge); an unrooted entry that resolves under NO registered path keeps its bare name (as before) and is COUNTED in
        `unresolved_module_paths` (provenance) so a merged per-layer name is visible at derive, not first at Commit coverage."""
        stack = node.meta.get("nn_module_stack") or {}
        if not stack:
            return ""
        entries = [str(p or "") for p, _c in stack.values()]
        paths = self._module_paths
        if not paths:
            return entries[-1]
        parent = ""
        for fqn in entries:
            if fqn in paths or not fqn:
                parent = fqn
            elif parent and f"{parent}.{fqn}" in paths:
                parent = self._canonical_path.get(f"{parent}.{fqn}", f"{parent}.{fqn}")
            else:
                self.unresolved_module_paths[fqn] = self.unresolved_module_paths.get(fqn, 0) + 1
                parent = fqn
        return parent

    def hint_name(self, node) -> str:
        path = self.stack_path(node) if node.meta.get("nn_module_stack") else ""
        return f"{path}/{node.name}" if path else node.name

    def storage_base(self, node):
        return self.storage.get(node, node)

    def is_weight_storage(self, node) -> WeightLeaf | None:
        base = self.storage_base(node)
        if isinstance(base, torch.fx.Node) and base.op == "placeholder" and base.name in self.placeholder_to_weight:
            return self.weights[self.placeholder_to_weight[base.name]]
        return None

    def is_input_storage(self, node) -> str | None:
        base = self.storage_base(node)
        if isinstance(base, torch.fx.Node) and base.op == "placeholder" and base.name in self.user_inputs:
            return base.name
        return None

    def weight_of(self, a) -> WeightLeaf | None:
        """The weight leaf a graph argument is a (view of a) parameter/buffer of, if it is one."""
        if isinstance(a, torch.fx.Node):
            return self.is_weight_storage(a)
        return None

    # -- emitting ----------------------------------------------------------------------------------------
    def kind(self, role: str, node=None, **statics):
        """The mode's registered definition for a semantic role, bound to `statics`; `numerics_unregistered`
        if the vocabulary has no kind for it."""
        fn = self.vocab.kind(role, **statics)
        if fn is None:
            refuse("numerics_unregistered", node or self.current, self.module_path(node or self.current),
                   f"the {self.mode} vocabulary {self.vocab.name!r} has no registered kind for role {role!r} with {statics}; "
                   f"a NUMERIC kind is needed")
        return fn

    @property
    def registered_values(self) -> bool:
        return self.registration is not None

    def supplied(self, name: str, width: int, bits: int, node=None, distinct: bool = False) -> Coll:
        """The field `name` of the registered `values` parameter (`Registration.value`; `distinct`: one field per distinct value): a
        program value (eps, the attention scale, a scalar) that the Definition reading it takes as an input."""
        return self._registered(node, f"value {name!r}", lambda r: r.value(name, int(width), int(bits), distinct))

    def table(self, name: str, node=None) -> Coll:
        """The registered table `name` (`verity_vllm.program.registry.tables`), declared by a rule (`Rule.registered_tables`)."""
        return self._registered(node, f"table {name!r}", lambda r: r.table(name))

    def _registered(self, node, what: str, read) -> Coll:
        node = node or self.current
        try:
            if self.registration is None:
                raise ValueError(f"{what} read as an input outside a registered_values derivation")
            return read(self.registration)
        except ValueError as e:
            refuse("numerics_unregistered", node, self.module_path(node), str(e))

    def call(self, fn, *args: Coll, node=None, rule: Rule | None = None, name: str | None = None) -> Coll:
        node = node or self.current
        if self.emit_into is not None:
            return self.emit_into(fn, args, node, rule, name)
        try:
            out = self.B.call(fn, *args, name=name)
        except TypeError as e:
            refuse("shape", node, self.module_path(node), f"{rule.rule_id if rule else 'call'} -> {getattr(fn, 'id', fn)}: {e}")
        k = len(self.B.nodes) - 1
        self.node_hints[k] = name or self.hint_name(node)
        self.node_sources[k] = node.name
        self.node_fx[k] = node
        return out

    def begin_emit(self, node) -> tuple[int, dict[int, Any]]:
        """Before a rule lowers fx `node`: the Builder length and the bindings of its fx arguments AS READ (runtime correspondence)."""
        return len(self.B.nodes), {j: self.env.get(a) for j, a in fx_arg_nodes(node)}

    def note_emitted(self, node, k0: int, before: dict[int, Any] | None = None) -> None:
        """After a rule lowered fx `node`: every Builder node from index `k0` on was lowered from it (rules that emit through `self.B`
        directly, not `call`), and the occurrence's bindings are frozen for the runtime correspondence -- `in`: the arguments as read,
        `out`: the node's value as PRODUCED (a later in-place op may rebind `env[node]`), `written`: the arguments whose binding the
        lowering replaced (the storages an in-place op wrote)."""
        for k in range(k0, len(self.B.nodes)):
            self.node_fx.setdefault(k, node)
        if before is not None:
            written = {j: self.env.get(a) for j, a in fx_arg_nodes(node) if self.env.get(a) is not before.get(j)}
            self.occurrence_bindings[node] = {"in": before, "out": self.env.get(node), "written": written}

    def record(self, rule: Rule, node, preconditions: dict[str, Any], emitted: str | None = None, **extra) -> None:
        self.applications.append({"rule": rule.rule_id, "version": rule.info.version, "kind": rule.info.kind, "node": node.name, "op": str(node.target),
                                  "module_path": self.module_path(node), "preconditions_checked": _jsonable(preconditions),
                                  "emitted": emitted, **_jsonable(extra)})

    def define(self, node, value, *, view_of=None) -> None:
        """Bind a node's value. `view_of`: the node this value aliases (views share storage groups)."""
        self.env[node] = value
        if view_of is not None:
            self.storage[node] = self.storage_base(view_of)

    def rebind(self, node, new_value: Coll, rule: Rule, *, what: str = "operand") -> None:
        """An in-place rule wrote `node`'s tensor: the node now denotes `new_value`, and every other alias of the
        same storage becomes stale (boundary 3: mutation effects are explicit — reads through another alias are
        refused rather than silently reading the pre- or post-mutation value)."""
        if isinstance(node, torch.fx.Node):
            w = self.is_weight_storage(node)
            if w is not None:
                refuse("mutation", self.current, self.module_path(self.current),
                       f"{rule.rule_id} would mutate parameter/buffer {w.name!r} through {what}; weights are prescribed inputs and "
                       f"must not be written (a StateRule must own cache state explicitly)")
            inp = self.is_input_storage(node)
            old = self.env.get(node)
            if isinstance(old, Coll):
                # every other alias whose references overlap the written region can no longer be read
                base = self.storage_base(node)
                group = [n for n in self.storage.aliases.get(base, ()) if n is not node]
                if base is not node:
                    group.append(base)
                old_runs = _runs(old.refs)
                for other in group:
                    ov = self.env.get(other)
                    if isinstance(ov, Coll) and other not in self.stale and _runs_intersect(old_runs, _runs(ov.refs)):
                        self.stale[other] = f"{rule.rule_id} wrote {node.name} (overlapping alias) in {self.current.name}"
            if inp is not None:
                self.mutation_effects.append({"input": inp, "node": self.current.name, "rule": rule.rule_id, "what": what,
                                              "representation": "the mutated value is the pure result; the Input gates keep the pre-mutation value"})
            self.written_allocs.add(node)
        self.env[node] = new_value
        self.storage.pop(node, None)  # fresh storage

    def bind_weight_role(self, a, rule: Rule, role: str, **statics) -> None:
        """Record which semantic role (`ln1`, `qkv_w`, ...) a parameter played when a rule consumed it."""
        w = self.weight_of(a)
        if w is not None and w.name not in self.weight_roles:
            self.weight_roles[w.name] = {"rule": rule.rule_id, "role": role, **_jsonable(statics)}

    # -- allowed dims ------------------------------------------------------------------------------------
    def check_dim(self, dim_name: str, value: int, node=None) -> None:
        """Fail closed if `allowed_dims` pins `dim_name` and `value` is not among the allowed static bindings."""
        allowed = self.allowed_dims.get(dim_name)
        if allowed is None:
            return
        ok = value in allowed if isinstance(allowed, (set, frozenset, list, tuple)) else value == allowed
        if not ok:
            refuse("shape", node or self.current, self.module_path(node or self.current),
                   f"dimension {dim_name}={value} is not among the allowed static bindings {allowed}")


# ---------------------------------------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------------------------------------


def derive(source, example_inputs: Sequence[Any], *, mode: str, profile: str | None = None, config: dict | None = None,
           rules: RuleSet | None = None, allowed_dims: dict | None = None, name: str | None = None,
           check_registry: bool = True, rank: tuple[int, int] | None = None, registered_values: bool = False,
           committed_units: Sequence[dict[str, Any]] | None = None) -> Program | Unsupported:
    """Derive a `Program` from an unannotated module/callable, or return `Unsupported`. See module docstring."""
    return derive_report(source, example_inputs, mode=mode, profile=profile, config=config, rules=rules, allowed_dims=allowed_dims,
                         name=name, check_registry=check_registry, rank=rank, registered_values=registered_values,
                         committed_units=committed_units).result


def derive_report(source, example_inputs: Sequence[Any], *, mode: str, profile: str | None = None, config: dict | None = None,
                  rules: RuleSet | None = None, allowed_dims: dict | None = None, name: str | None = None,
                  check_registry: bool = True, rank: tuple[int, int] | None = None, exported=None,
                  registered_values: bool = False, session: list | None = None,
                  committed_units: Sequence[dict[str, Any]] | None = None) -> DeriveReport:
    """`registered_values`: the program's values (the norms' eps, the attention scale) are fields of a registered `values` parameter
    (before `weights`), read by the Definitions that take them as inputs (`TargetProfile.registered_values`); the provenance's
    `supplied` states every field's bits.  `rank=(world, rank)`: the rank coordinate of a tensor-parallel rank Program.  It is written into the runtime correspondence
    (`annotations["correspondence"].rank`, explicit integers) and NOWHERE else: not a static, not a config -- the Program's identity
    (`program_digest`) does not depend on it.  `exported`: the caller's export of `source` (same inputs and shims), derived from.
    `session`: a list that receives the derivation's `DeriveSession` when it succeeds (`relabel.rank_report` reads it)."""
    if mode not in ("profile", "reference"):
        raise ValueError("mode must be 'profile' or 'reference'")
    if rank is not None and not (len(rank) == 2 and 0 <= int(rank[1]) < int(rank[0])):
        raise ValueError(f"rank must be (world, rank) with 0 <= rank < world, got {rank!r}")
    from verity_vllm.program.frontend.rules import DEFAULT_RULES, vocabulary_for
    ruleset = rules if rules is not None else DEFAULT_RULES
    vocab = vocabulary_for(mode, profile)
    allowed_dims = dict(allowed_dims or {})
    config = dict(config or {})
    name = name or (type(source).__name__ if isinstance(source, torch.nn.Module) else getattr(source, "__name__", "Derived"))
    source_desc = _describe_source(source, example_inputs, mode, profile, config, allowed_dims)
    from verity_vllm.pipeline.committed_units import parse as parse_committed_units
    committed_entries = parse_committed_units(committed_units)
    prov: dict[str, Any] = {"schema": "verity_ir/derive-provenance/v1", "frontend": FRONTEND_VERSION, "derive": DERIVE_VERSION,
                            "torch": torch.__version__, "mode": mode, "profile": profile, "vocabulary": vocab.to_json(),
                            "ruleset": ruleset.to_json(), "registry": registry_version(), "source": source_desc,
                            "naming_hints": "nn_module_stack / node names are annotations only; identity comes from rule + registry"}
    holder: list = []
    try:
        flat_sd, ctx = _derive_flat(source, example_inputs, mode=mode, profile=profile, config=config, allowed_dims=allowed_dims,
                                    name=name, ruleset=ruleset, vocab=vocab, _ctx_holder=holder, exported=exported,
                                    registered_values=registered_values, committed_entries=committed_entries)
        head = (dict(prov), dict(ctx.unresolved_module_paths)) if session is not None else None
        rep = _finish_report(flat_sd, ctx, prov, mode=mode, ruleset=ruleset, rank=rank, check_registry=check_registry)
        if session is not None:
            session.append(DeriveSession(flat_sd, ctx, *head, name, mode, ruleset, check_registry, rank, rep.family))
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
                   check_registry: bool) -> DeriveReport:
    """The derivation's tail over the finished flat body: provenance, runtime correspondence, the flat Program and its digest, the
    family claims, `prov["result"]`.  Raises `Refuse` / `FrontendError` (the caller reports them)."""
    prov["weights"] = [w.to_json() for w in ctx.weights]
    prov["weight_roles"] = ctx.weight_roles
    prov["unresolved_module_paths"] = dict(sorted(ctx.unresolved_module_paths.items()))   # GATE-PATH residual (global M-0595 (3))
    prov["applications"] = ctx.applications
    prov["mutation_effects"] = ctx.mutation_effects
    prov["returned_inputs"] = ctx.returned_inputs
    if ctx.committed_units:
        prov["committed_units"] = ctx.committed_units
    prov["rules_applied"] = sorted({a["rule"] for a in ctx.applications})
    if ctx.registered_values:
        prov["supplied"] = ctx.registration.layout()
    # runtime correspondence (`runtime-correspondence/v1`): per Call the owning module, the source occurrence and the return /
    # argument slots -- data beside the Program, under ONE annotation namespace, digested apart (annotations never enter program_digest)
    from verity_vllm.program.frontend.emit import NAMESPACE as _CORR, emit_correspondence
    from verity_vllm.correspondence.runtime import correspondence_digest
    corr_doc = emit_correspondence(ctx, flat_sd.body, rank=rank)
    corr_digest = correspondence_digest(corr_doc)
    flat_program = Program(flat_sd, annotations={"derive": prov, "node_hints": {str(k): v for k, v in ctx.node_hints.items()}, _CORR: corr_doc},
                           registered=ctx.registration.names if ctx.registered_values else ())
    check_identity(flat_program, check_registry=check_registry)
    flat_digest = program_digest(flat_desc := encode_program(flat_program))
    program, family_report = flat_program, None
    for fam in (() if ctx.committed_units else ruleset.families):
        if fam.modes is not None and mode not in fam.modes:
            continue
        a = "a family claim keeps no registered parameter" if ctx.registered_values else fam.applies_program(flat_sd, ctx)
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
    corr_applies = program is flat_program
    prov["result"] = {"root": _spec_id(program.fn), "digest": digest, "gates": program.gates, "flat_root": _spec_id(flat_sd),
                      "flat_digest": flat_digest, "dead_gates": dead.total,
                      "correspondence_digest": corr_digest if corr_applies else None,
                      "correspondence": {"schema": corr_doc["schema"], "calls": len(corr_doc["calls"]), "occurrences": len(corr_doc["occurrences"]),
                                         "applies_to": "program_fn" if corr_applies else f"flat body only ({_spec_id(flat_sd)}); not attached", "digest": corr_digest}}
    return DeriveReport(program, None, prov, digest, _spec_id(program.fn), program.gates, dead, flat_digest, _spec_id(flat_sd),
                        prov.get("family"), flat_sd, corr_digest if corr_applies else None, flat_desc if corr_applies else None)


def _record_partial(prov: dict, holder: list) -> None:
    """On refusal, the provenance still reports what fired before it (XFER-G8) — clearly marked as belonging to NO program."""
    if not holder:
        return
    ctx = holder[0]
    prov["weights"] = [w.to_json() for w in ctx.weights]
    prov["applications_before_refusal"] = ctx.applications
    if ctx.committed_units:
        prov["committed_units_before_refusal"] = ctx.committed_units
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
    items = [(n, p, "parameter") for n, p in module.named_parameters(remove_duplicate=False)]
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
                 _ctx_holder: list | None = None, exported=None, registered_values: bool = False,
                 committed_entries: Sequence[dict[str, Any]] = ()) -> tuple[SpecializedDefinition, Ctx]:
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

    # parameters of the flat function: the user inputs (by placeholder name), the registered `values` (finalised after the
    # translation: its fields are the values the Definitions read), the registered tables the rules declare, and the weights record
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
    committed_nodes: list[tuple[Any, dict[str, Any], str]] = []
    committed_writes: dict[Any, list[Any]] = {}
    matched_entries: set[int] = set()
    if committed_entries:
        from verity_vllm.pipeline.committed_units import matches as committed_matches
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            op = str(node.target)
            stack = node.meta.get("nn_module_stack") or {}
            module_path = str(list(stack.values())[-1][0] or "") if stack else ""
            hit = next(((k, entry) for k, entry in enumerate(committed_entries) if committed_matches(entry, op, module_path)), None)
            if hit is None:
                continue
            k, entry = hit
            writes = _written_operands(node)
            try:
                returns = node.meta.get("val") is not None
                component_types = ([type_of(node.meta.get("val"), node.name)] if returns else []) + [
                    type_of(arg.meta.get("val"), arg.name) for arg in writes]
                if not component_types:
                    refuse("shape", node, module_path, "committed-only op has neither a returned value nor a written tensor operand")
                output_type = component_types[0] if len(component_types) == 1 else TupleT(*component_types)
            except FrontendError as e:
                refuse("shape", node, module_path, f"committed-only output cannot be typed: {e.message}")
            input_name = f"committed_{len(committed_nodes)}"
            params.append((input_name, output_type))
            committed_nodes.append((node, dict(entry), module_path))
            if writes:
                committed_writes[node] = writes
            matched_entries.add(k)
        missing = [dict(entry) for k, entry in enumerate(committed_entries) if k not in matched_entries]
        if missing:
            refuse("decomposition", None, None, f"committed-unit list entries matched no exported op/module occurrence: {missing}")
    registration = Registration.declare(params, ruleset, ep, mode) if registered_values else None
    wrec_t = Record(*[(w.name, w.type) for w in leaves]) if leaves else None
    if wrec_t is not None:
        params.append(("weights", wrec_t))

    B = Builder(tuple(params), Value(1))  # ret type is set once the outputs are known
    pcolls = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    user_inputs = {un: pcolls[i] for i, un in enumerate(user_names)}
    committed_offset = len(user_names)
    committed_inputs = {node: pcolls[committed_offset + k] for k, (node, _entry, _module) in enumerate(committed_nodes)}
    committed_provenance = [{"input": f"committed_{k}", "op": str(node.target), "module": module, "reason": entry["reason"],
                             "list_entry": entry, "node": node.name, "type": str(params[committed_offset + k][1]),
                             "returns": node.meta.get("val") is not None,
                             **({"writes": [arg.name for arg in committed_writes[node]]} if node in committed_writes else {})}
                            for k, (node, entry, module) in enumerate(committed_nodes)]
    weights_coll = pcolls[-1] if wrec_t is not None else None
    ctx = Ctx(mode=mode, profile=profile, config=config, allowed_dims=allowed_dims, name=name, ruleset=ruleset, vocab=vocab, module=module,
              ep=ep, B=B, user_inputs=user_inputs, weights_coll=weights_coll, weights=leaves, placeholder_to_weight=placeholder_to_weight,
              lifted_constants=lifted, registration=registration, weights_param=len(params) - 1 if wrec_t is not None else None,
              committed_inputs=committed_inputs, committed_units=committed_provenance, committed_writes=committed_writes)
    if _ctx_holder is not None:
        _ctx_holder.append(ctx)
    out = _translate(ctx)
    if registration is not None:
        try:
            registration.finish(params)
        except ValueError as e:
            refuse("numerics_unregistered", None, None, str(e))
        B.params = tuple(params)
    B.ret_type = out.type
    body = B.finish(out)
    src_static = {"class": _describe_source(source, example_inputs, mode, profile, config, allowed_dims).get("class", name), "mode": mode,
                  "profile": profile or "", "inputs": [list(shape_of(t)[0]) for _, t in params[:len(user_names)]],
                  "config_sha256": hashlib.sha256(canonical_json(_jsonable(config))).hexdigest()[:16]}
    return _flat_definition(name, body, src_static, mode), ctx


def _flat_definition(name: str, body, src_static: dict, mode: str) -> SpecializedDefinition:
    """The flat program's definition over `body`.  It never carries a registered name: identity is claimed only through a verified
    family match."""
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
                w = ctx.weights[ctx.placeholder_to_weight[node.name]]
                ctx.env[node] = ctx.weights_coll[w.index]
            elif node.name in ctx.lifted_constants:
                # a constant only a rule may account for: bind a marker; reading it without a rule is refused
                ctx.env[node] = _LiftedConstant(node, ctx.lifted_constants[node.name], node.meta.get("val"))
            else:
                refuse("side_input", node, None, f"placeholder {node.name} is neither a user input nor a parameter/buffer")
        elif node.op == "output":
            out_node = node
        elif node.op == "get_attr":
            refuse("dynamic_control_flow", node, ctx.module_path(node), f"subgraph attribute {node.target} (higher-order op body)")
        elif node.op == "call_function":
            if node in ctx.env:
                continue  # absorbed by a pattern rule that fired at an earlier node of its chain
            if node in ctx.committed_inputs:
                operands = []
                for j, arg in fx_arg_nodes(node):
                    value = ctx.env.get(arg)
                    operands.append({"index": j, "node": arg.name,
                                     "refs": [list(r) for r in _runs(value.refs)] if isinstance(value, Coll) else None})
                unit = ctx.committed_units[list(ctx.committed_inputs).index(node)]
                unit["module"] = ctx.stack_path(node)
                unit["operand_refs"] = operands
                committed = ctx.committed_inputs[node]
                writes = ctx.committed_writes.get(node, ())
                returns = bool(unit["returns"])
                if writes:
                    components = int(returns) + len(writes)
                    if returns:
                        ctx.define(node, committed if components == 1 else committed[0])
                    else:
                        ctx.env[node] = None
                    for k, arg in enumerate(writes):
                        ctx.env[arg] = committed if components == 1 else committed[int(returns) + k]
                else:
                    ctx.define(node, committed)
                continue
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
        outs.append(ctx.coll(a, "returned value"))
    if not outs:
        refuse("shape", out_node, None, "the source returns nothing")
    # returned inputs / repeated returns are representable (a return is a reference sequence) — record them explicitly
    for i, o in enumerate(outs):
        for space, idx, lo, hi in _runs(o.refs):
            if space == "p":
                ctx.returned_inputs.append({"output": i, "param": idx, "leaves": [lo, hi],
                                            "weights": idx == ctx.weights_param})
    ret = outs[0] if len(outs) == 1 else Coll(TupleT(*[o.type for o in outs]), _concat_refs([o.refs for o in outs]))
    return ret


def _concat_refs(parts):
    from verity.ml.circuits.refs import concat
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
        if isinstance(base, (list, tuple)):
            ctx.define(node, base[node.args[1]], view_of=node.args[0])
        elif isinstance(base, Coll):
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


def _written_operands(node) -> list[Any]:
    """FX tensor operands a mutable op writes, in schema order."""
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return []
    out = []
    for k, arg in enumerate(schema.arguments):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        value = node.args[k] if k < len(node.args) else node.kwargs.get(arg.name)
        if isinstance(value, torch.fx.Node):
            out.append(value)
    return out


def fx_arg_nodes(node) -> list[tuple[int, Any]]:
    """The fx-node arguments of `node`, flattened, as (flat index, fx node): positional arguments first (nodes inside a list / tuple
    argument in order), then keyword arguments in order.  The flat index is the runtime correspondence's `inputs[j]` coordinate."""
    out: list[Any] = []
    for a in node.args:
        if isinstance(a, torch.fx.Node):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(x for x in a if isinstance(x, torch.fx.Node))
    for a in node.kwargs.values():
        if isinstance(a, torch.fx.Node):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(x for x in a if isinstance(x, torch.fx.Node))
    return list(enumerate(out))


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


__all__ = ["derive", "derive_report", "DeriveReport", "DeriveSession", "Unsupported", "Refuse", "refuse", "Ctx", "StaticIndex", "Alloc", "WeightLeaf",
           "REASONS", "DERIVE_VERSION"]
