"""What a derivation carries between the nodes of its walk (`verity_vllm.program.frontend.derive` runs the walk and
hands these names on, so a rule file imports them from there).

`StaticIndex` and `WeightLeaf` are the non-value objects a translation carries, `_runs` and `_runs_intersect` read a
reference sequence as intervals, and `Ctx` is the state of one derivation: rules read the graph through it and emit
through its `call`, `define` and `rebind`. `fx_arg_nodes` gives a node's fx arguments their flat index, the runtime
correspondence's coordinate. Nothing here reads a tensor's values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from verity.circuits.defs import Builder
from verity.circuits.refs import Coll
from verity.circuits.refs import runs as _refs_runs
from verity.circuits.types import Type, Value

from verity_vllm.program import inputs
from verity_vllm.program.frontend.outcomes import _jsonable, refuse
from verity_vllm.program.frontend.registration import RegisteredInputs, RunInputs
from verity_vllm.program.frontend.rules.base import Rule, RuleSet
from verity_vllm.program.frontend.value_types import Alloc

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
        return d if self.derivation is None else {**d, "derivation": self.derivation}


# ---------------------------------------------------------------------------------------------------------
# reference runs: (space, idx, lo, hi) intervals of a reference sequence — for alias overlap and matching
# ---------------------------------------------------------------------------------------------------------


_runs = _refs_runs  # `verity.circuits.refs.runs` (pure; moved there so `rules.family.match_program` imports without torch)


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
    """[GATE-PATH] (every registered path of every submodule, duplicate path -> first registration).  `named_modules()` (the
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
                 placeholder_to_weight: dict[str, int], lifted_constants: dict[str, str], registered_inputs: RegisteredInputs | None = None,
                 weights_param: int | None = None, run_inputs: RunInputs | None = None) -> None:
        self.mode, self.profile, self.config, self.allowed_dims, self.name = mode, profile, dict(config or {}), dict(allowed_dims or {}), name
        self.ruleset, self.vocab, self.module, self.ep, self.B = ruleset, vocab, module, ep, B
        self.user_inputs, self.weights_coll, self.weights = user_inputs, weights_coll, weights
        self.placeholder_to_weight, self.lifted_constants = placeholder_to_weight, lifted_constants
        self.registered_inputs, self.weights_param = registered_inputs, weights_param   # the registered inputs (`registered_values`)
        self.run_inputs = run_inputs   # the `run` parameter a rule declares (`Rule.run_inputs`)
        self.checks: list[tuple[Coll, str]] = []   # (one bit, what it checks): outputs after the source's, required 1 (`check`)
        self.checks_output: int | None = None
        self.env: dict[Any, Any] = {}
        self.storage: _Storage = _Storage()  # node -> storage base node (alias groups); absent: own storage
        self.stale: dict[Any, str] = {}  # node -> why its value may no longer be read (alias of a mutated tensor)
        self.written_allocs: dict = {}  # node an in-place rule wrote -> the storage base it was written through
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
        # every path a module of the source is registered under -> its first registration (`stack_path`)
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
        stack = node.meta.get("nn_module_stack")
        if not stack:
            return None
        _path, cls = list(stack.values())[-1]
        path = self.stack_path(node)
        return f"{path or '<root>'} ({cls.__name__ if isinstance(cls, type) else str(cls).rsplit('.', 1)[-1]})"

    def module_class(self, node) -> str | None:
        stack = node.meta.get("nn_module_stack")
        if not stack:
            return None
        _, cls = list(stack.values())[-1]
        return cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"

    def stack_path(self, node) -> str:
        """The producing module's path in the source module's OWN vocabulary (`named_modules()`), from the node's `nn_module_stack`.
        [GATE-PATH] the export names a module by the attribute path the trace took; a module reached through a holder that is
        not its registering parent comes out UNROOTED (vLLM 0.28's MoERunner keeps the router as `self.gate` and calls it inside its own
        forward: the router Gemm's stack ends in a bare `gate`, one name for 48 layers) or under a DUPLICATE registration
        (`...mlp.experts.gate` where `named_modules()` yields `...mlp.gate`).  Rooted entries are kept verbatim (a shared RoPE instance
        stays under each layer's path, as the manifest's alias rule expects); an unrooted entry is resolved relative to its parent's
        registered path or, extending the previous unrooted entry, to that entry's holder (GLM-5.3's `_shared_experts._layer.act_fn`,
        from `...mlp.experts`), a duplicate registration replaced by the first (`...mlp.shared_experts.act_fn`, the collector's name).
        A ROOTED duplicate stays under the duplicate name (C1 coverage FAILS it as missing, never a silent merge); an unrooted
        entry under NO registered path keeps its bare name and is COUNTED in `unresolved_module_paths` (provenance)."""
        stack = node.meta.get("nn_module_stack")
        if not stack:
            return ""
        entries = [str(p or "") for p, _c in stack.values()]
        paths = self._module_paths
        if not paths:
            return entries[-1]
        parent = registered = holder = chain = ""      # resolved path, the registered one; an unrooted run's holder and last entry
        for fqn in entries:
            base = holder if chain and fqn.startswith(chain + ".") else registered
            if fqn and fqn not in paths and base and f"{base}.{fqn}" in paths:
                holder, chain, registered = base, fqn, f"{base}.{fqn}"
                parent = self._canonical_path.get(registered, registered)
                continue
            if fqn and fqn not in paths:
                self.unresolved_module_paths[fqn] = self.unresolved_module_paths.get(fqn, 0) + 1
            parent, registered, holder, chain = fqn, fqn, "", ""
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
        """The mode's registered definition for a semantic role, bound to `statics`; `numerics_unregistered` if the vocabulary has no kind for it."""
        fn = self.vocab.kind(role, **statics)
        if fn is None:
            refuse("numerics_unregistered", node or self.current, self.module_path(node or self.current),
                   f"the {self.mode} vocabulary {self.vocab.name!r} has no registered kind for role {role!r} with {statics}; "
                   f"a NUMERIC kind is needed")
        return fn

    @property
    def registered_values(self) -> bool:
        return self.registered_inputs is not None

    def supplied(self, name: str, width: int, bits: int, node=None, distinct: bool = False) -> Coll:
        """The field `name` of the registered `values` parameter (`RegisteredInputs.value`; `distinct`: one field per distinct value): a
        program value (eps, the attention scale, a scalar) that the Definition reading it takes as an input."""
        return self._registered(node, f"value {name!r}", lambda r: r.value(name, int(width), int(bits), distinct))

    def table(self, name: str, node=None) -> Coll:
        """The registered table `name` (`verity.definitions.registry.tables`), declared by a rule (`Rule.registered_tables`)."""
        return self._registered(node, f"table {name!r}", lambda r: r.table(name))

    def _registered(self, node, what: str, read) -> Coll:
        node = node or self.current
        try:
            if self.registered_inputs is None:
                raise ValueError(f"{what} read as an input outside a registered_values derivation")
            return read(self.registered_inputs)
        except ValueError as e:
            refuse("numerics_unregistered", node, self.module_path(node), str(e))

    def run_value(self, name: str, t: Type, declared: str, node=None) -> Coll:
        """The field `name` of the `run` parameter (`RunInputs.value`): a value the served run decides and the Program checks in
        gates rather than computes, declared by the rule's `Rule.run_inputs` and read under `program.inputs.RUN[declared]`."""
        node = node or self.current
        try:
            if self.run_inputs is None:
                raise ValueError(f"run value {name!r} read but no rule declared run inputs for this graph (`Rule.run_inputs`)")
            return self.run_inputs.value(name, t, declared)
        except ValueError as e:
            refuse("side_input", node, self.module_path(node), str(e))

    def check(self, ok: Coll, what: str) -> None:
        """Make the bit ``ok`` one of the Program's checks: an output after the source's, which the verifier requires to be 1
        (the provenance's `checks` names each)."""
        if ok.type != Value(1):
            raise TypeError(f"a check is one bit, not {ok.type!r} ({what})")
        try:
            inputs.check(self.B, ok, what)
        except inputs.Free as e:
            refuse("side_input", self.current, self.module_path(self.current), str(e))
        self.checks.append((ok, what))

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
            self.written_allocs[node] = self.storage_base(node)
        self.env[node] = new_value
        self.storage.pop(node, None)  # fresh storage

    def bind_weight_role(self, a, rule: Rule, role: str, **statics) -> None:
        """RecordType which semantic role (`ln1`, `qkv_w`, ...) a parameter played when a rule consumed it."""
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


__all__ = ["StaticIndex", "WeightLeaf", "module_registrations", "Ctx", "fx_arg_nodes"]
