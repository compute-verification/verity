"""Pure resolver: raw event subtrees -> {Compute(specialization), Transparent(view|move|alloc), Unsupported}.

All semantic classification of the raw observation log lives here and in the profile modules
(`profiles/*.py`), which register `Pattern`s and pin the constexpr / launch-geometry values they accept.
The resolver knows nothing about GPU memory contents and never reads tensor values; it only inspects the
event tree (ops, schemas, physical views, scalars, Triton constexprs) and the step context (rows, positions,
block tables, slot mapping, sampled tokens) and produces *declarative* resolutions that `fold.py` executes
against the versioned memory model.

Accepted outcomes (design §7; `Group` bundles several of the first two for one subtree):

    Compute      one registered Verity specialization (`SpecializedDefinition`), instantiated once per row,
                 with per-row operand/output *shapes* over physical views (see `Operand`);
    Transparent  a provenance-only event: alias view (no state change), move (bit-preserving copy of resolved
                 refs from `src` to `dst`, resolved *now*), or allocation;
    Unsupported  anything else.  Development mode reports it; the accepted profile requires 0.

A pattern consumes a whole subtree.  `resolve_step` checks that every event id under a matched node is in the
resolution's `consumed` set (no silent, unaccounted children) and that no id is consumed twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from verity_vllm.observe.events import Root, RunHeader, SampleEnd, StepLink, StepStart, ViewDesc
from verity_vllm.observe.fold.tree import StepEvents, TreeNode

# ---------------------------------------------------------------------------------------------------------
# operands: how row i of an instance reads / writes physical views
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shared:
    """Every row consumes the whole view (weights, tables)."""

    view: ViewDesc


@dataclass(frozen=True)
class PerRow:
    """Row i consumes/produces `view[i]` (index along the leading axis)."""

    view: ViewDesc


@dataclass(frozen=True)
class Gather:
    """Row i consumes/produces `view[indices[i]]` with *public* indices (positions -> cos/sin rows, KV slots,
    logits_indices, ...).  Indices are facts from the step context, never read from device memory."""

    view: ViewDesc
    indices: tuple[int, ...]


@dataclass(frozen=True)
class GatherMany:
    """Row i consumes `stack(view[j] for j in indices[i])` (a request's KV slots for positions 0..T_i-1)."""

    view: ViewDesc
    indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Const:
    """[R15 stoch] Every row consumes ONE Program constant: a `Const<width>[0x<bits>]` primitive instance the fold emits (once per
    Compute, immediately before it, in operand order) and wires in -- the derived Program's `ctx.kind("const")` of the same value.
    No device view: the value is a declared fact of the run header (the sampler's temperature / top_p / the row's position), never
    read from memory.  `name` is the instance-name tail (`gumbel_topp_token_select/top_p`)."""

    width: int
    bits: int
    name: str = ""


@dataclass(frozen=True)
class Param:
    """[R15 stoch] Row i consumes the whole of a declared runtime PARAMETER of the Program (`kind`: `"seed"` -> the request's seed
    parameter `Array<1,Value<64>>`); `keys[i]` is row i's request id.  No device view: the fold wires the parameter's leaves, exactly as
    the derived Program takes `seed` as an input (never a Const -- a wrong seed is a wrong INPUT, caught by the value check, not a
    different Program)."""

    kind: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Collective:
    """[R17 D90, COORD R17-4] Row i consumes the `parts` of a two-rank collective in RANK order: this rank's OWN partial row `own[i]`
    (an ordinary memory read of the rank's collective INPUT storage -- its in-log producer is the rank's `Gemm`/`EmbeddingShard` output;
    an unwritten row is Unresolved, never a root) and the PEER rank's partial row = a leaf range of the fold's `peers_<request>` root
    PARAMETER (root class `collective_received`: the other rank's committed partial, bound by the rank Program's `tp_links.json`).
    The fold numbers the collective (step, k) itself -- k = issue ordinal within the step -- and takes rank/world from
    `RunHeader.notes.tp2`; the pattern only names the kind and the issuing module (`site`, the locator of the cross-rank leg).
    The RECEIVED value is therefore never a free prescribed input: it is the Compute's OUTPUT, `AllReduce2_v1{N}` /
    `AllGather2_v1{N}(parts)`, written to the result storage the rank's next kernel reads."""

    own: ViewDesc
    kind: str            # "all_reduce" | "all_gather"
    site: str            # issuing module path (e.g. `model.layers.3.self_attn.o_proj`)
    #: one Compute row over the whole [M,N] partial as M*N words (`Array<W, Array<M*N>>`), the Build's one call per communicator
    #: all-reduce (`vllm_bindings.AllReduceRule`); the peers fields stay one per (request, peer rank)
    flat: bool = False


@dataclass(frozen=True)
class Supplied:
    """Every row consumes ONE input value of the Program, never a constant of it (a profile with `registered_values`):
    `binding` "registration" -> a field of the registered `values` parameter, one field per `name` (a norm's eps, the attention
    scale: what the program brings); "run" -> a field of `request`'s `run_<i>` parameter, one per (request, `name`) (the sampler's
    temperature and top_p, a row's Philox position).  `bits` is the fold's record of the value (`FoldResult.supplied`, the source
    of the registration record and the run's inputs); two operands of one field must agree on it."""

    binding: str
    name: str
    width: int
    bits: int
    request: str | None = None


Operand = Shared | PerRow | Gather | GatherMany | Const | Param | Collective | Supplied


def operand_rows(op: Operand) -> int | None:
    if isinstance(op, Param):
        return len(op.keys)
    if isinstance(op, (PerRow, Collective)):
        view = op.view if isinstance(op, PerRow) else op.own
        return view.shape[0] if view.shape else 1
    if isinstance(op, (Gather, GatherMany)):
        return len(op.indices)
    return None


# ---------------------------------------------------------------------------------------------------------
# resolutions
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Compute:
    """`rows` instances of `spec`.  `operands[k]` describes parameter k of the specialization; `outputs[k]`
    describes component k of its return (one entry for a non-tuple return).  Element order inside an operand
    view is the row-major leaf order of the parameter's type; the fold checks leaf counts."""

    spec: Any                                  # verity.ml.circuits.defs.SpecializedDefinition
    rows: int
    operands: tuple[Operand, ...]
    outputs: tuple[Operand, ...]
    consumed: tuple[int, ...]
    hint: str = ""                             # instance name hint (module path / op); never part of identity
    kernels: tuple[str, ...] = ()              # device kernel name substrings this compute accounts for (census)
    attribution: tuple[tuple[str, int], ...] = ()   # optional per-row (request_id, position)
    pattern: str = ""
    # [R15 fp8] optional per-operand parameter (Root) name -- the weight name of EACH weight operand when a Compute reads more than one
    # (the block-scaled FP8 GEMM reads `weight` and `weight_scale_inv`); the fold records it as `operand_roots` in instances.jsonl and
    # cb_a.compare names each weight range by it (the hint can only name one).  Empty = unchanged single-weight behaviour.
    roots: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class Transparent:
    """`effect`:
        "alias"  the outputs are views of the inputs' storage: no memory-model change (view/reshape/slice/...);
        "move"   bit-preserving copy: for each row, refs(dst row) := resolved refs(src row) *now*;
        "alloc"  a fresh allocation with undefined contents (empty/zeros before an overwrite);
        "noop"   no value effect (e.g. `aten._record_function_enter`, `aten.is_same_size`)."""

    effect: str
    consumed: tuple[int, ...]
    src: Operand | None = None
    dst: Operand | None = None
    rows: int = 1
    hint: str = ""
    kernels: tuple[str, ...] = ()
    pattern: str = ""


@dataclass(frozen=True)
class Unsupported:
    event_id: int
    op: str
    reason: str
    detail: dict = field(default_factory=dict)
    consumed: tuple[int, ...] = ()


@dataclass(frozen=True)
class Group:
    """Several Compute/Transparent resolutions explained by ONE raw subtree, e.g. `_C.rotary_embedding` (RoPE on q
    and on k), `_C.reshape_and_cache_flash` (K and V moves), or the attention op with rows grouped by distinct T.
    The fold executes the parts IN ORDER, each with the ordinary discipline (a Compute reads all its operands, emits
    its instances, then writes; a move reads its source now, then writes); a later part may therefore read what an
    earlier part wrote (a scope's internal move feeding its compute).  `consumed` covers the whole subtree; the
    parts' own `consumed` tuples are informational (may be empty).

    `patterns.py` reaches the same effect with `ComputeGroup`/`TransparentGroup` (a primary carrying `before`/`also`
    members); `parts_of()` is the one accessor the fold and the report use for both spellings."""

    parts: tuple[Compute | Transparent, ...]
    consumed: tuple[int, ...]
    hint: str = ""
    pattern: str = ""


Resolution = Compute | Transparent | Unsupported | Group


def parts_of(r: Resolution) -> tuple[Resolution, ...]:
    """The resolutions `r` stands for, in the fold's execution order: a Group's `parts`; a primary with `before`/`also`
    members (patterns.ComputeGroup / TransparentGroup) as `(*before, primary, *also)`; otherwise `(r,)`."""
    if isinstance(r, Group):
        return tuple(r.parts)
    before = tuple(getattr(r, "before", ()) or ())
    also = tuple(getattr(r, "also", ()) or ())
    return (*before, r, *also)


# ---------------------------------------------------------------------------------------------------------
# context and patterns
# ---------------------------------------------------------------------------------------------------------


@dataclass
class StepContext:
    """Facts available to patterns while resolving one step."""

    header: RunHeader
    start: StepStart
    sample: SampleEnd | None
    links: list[StepLink]
    roots: dict[int, list[Root]]              # alloc id -> roots living in that allocation
    config: dict                               # the model's static configuration C (H, NH, KVH, D, I, V, NL, EPS, ...)
    profile: Profile
    scratch: dict = field(default_factory=dict)  # pattern-private per-step memory (e.g. pending KV scatter)

    def root_named(self, name: str) -> Root | None:
        for rs in self.roots.values():
            for r in rs:
                if r.name == name:
                    return r
        return None

    def roots_of(self, view: ViewDesc) -> list[Root]:
        return self.roots.get(view.alloc, [])


@runtime_checkable
class Pattern(Protocol):
    name: str

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        """Return a resolution consuming `node`'s subtree (or a checked part of it), or None to decline."""


@dataclass
class Profile:
    """A pinned capture profile: the patterns that are allowed to explain events, the model configuration
    binding, and the accepted-profile switch."""

    id: str
    patterns: list[Pattern]
    config: dict
    weight_field: Any                           # callable(name: str) -> tuple[str | int, ...] | None  (weights Record path)
    accepted: bool = False
    kernel_allowlist: tuple[str, ...] = ()      # device kernels accounted for by Transparent/infra (census)
    notes: dict = field(default_factory=dict)
    # step passes: `pass(resolutions, ctx, roots) -> resolutions`, applied in order to a resolved step (one resolution per root kept)
    step_passes: tuple = ()
    # the program's values (eps, the attention scale) and the request's sampling values are inputs (`Supplied` operands of the
    # Definitions that read them as parameters), not constants; off: the legacy Definitions with the values as constants
    registered_values: bool = False


class ResolutionError(Exception):
    pass


def resolve_node(node: TreeNode, ctx: StepContext) -> Resolution:
    for p in ctx.profile.patterns:
        r = p.match(node, ctx)
        if r is not None:
            if not isinstance(r, Unsupported) and not r.pattern:
                r = _with_pattern(r, p.name)
            return r
    return Unsupported(node.id, node.op, "no pattern", {"phase": node.phase, "children": [c.op for c in node.children]})


def _with_pattern(r: Resolution, name: str) -> Resolution:
    import dataclasses
    return dataclasses.replace(r, pattern=name)


def resolve_step(step: StepEvents, ctx: StepContext) -> list[Resolution]:
    """Resolve every top-level node of the step in order.  Every event id in the step must be consumed by
    exactly one resolution (Unsupported resolutions consume their whole subtree)."""
    out: list[Resolution] = []
    seen: set[int] = set()
    for root in step.roots:
        r = resolve_node(root, ctx)
        ids = root.ids()
        if isinstance(r, Unsupported):
            r = Unsupported(r.event_id, r.op, r.reason, r.detail, tuple(sorted(ids)))
        consumed = set(r.consumed)
        missing = ids - consumed
        extra = consumed - ids
        if missing:
            raise ResolutionError(f"pattern {getattr(r, 'pattern', '?')} on {root.op} (id {root.id}) left events "
                                  f"unconsumed: {sorted(missing)[:8]}")
        if extra:
            raise ResolutionError(f"pattern {getattr(r, 'pattern', '?')} on {root.op} consumed ids outside its subtree: "
                                  f"{sorted(extra)[:8]}")
        dup = consumed & seen
        if dup:
            raise ResolutionError(f"events consumed twice: {sorted(dup)[:8]}")
        seen |= consumed
        out.append(r)
    for p in ctx.profile.step_passes:
        out = p(out, ctx, list(step.roots))
    return out


@dataclass
class ResolutionReport:
    steps: int = 0
    computes: int = 0
    transparent: int = 0
    unsupported: int = 0
    by_pattern: dict[str, int] = field(default_factory=dict)
    by_spec: dict[str, int] = field(default_factory=dict)
    unsupported_ops: dict[str, int] = field(default_factory=dict)
    unsupported_examples: list[dict] = field(default_factory=list)
    spec_modules: dict[str, str] = field(default_factory=dict)   # spec label -> module defining its body (quarantine attribution, G4)

    def add(self, r: Resolution) -> None:
        """One resolved root: `by_pattern` counts the root once (the Group's / primary's pattern); `computes`,
        `transparent` and `by_spec` count every part (`parts_of`), so grouped members are visible."""
        if isinstance(r, Unsupported):
            self.unsupported += 1
            self.unsupported_ops[r.op] = self.unsupported_ops.get(r.op, 0) + 1
            if len(self.unsupported_examples) < 32:
                self.unsupported_examples.append({"id": r.event_id, "op": r.op, "reason": r.reason, "detail": r.detail})
            return
        pat = getattr(r, "pattern", "")
        self.by_pattern[pat] = self.by_pattern.get(pat, 0) + 1
        for p in parts_of(r):
            if isinstance(p, Compute):
                self.computes += 1
                key = spec_label(p.spec)
                self.by_spec[key] = self.by_spec.get(key, 0) + p.rows
                if key not in self.spec_modules:
                    self.spec_modules[key] = spec_module(p.spec)
            elif isinstance(p, Transparent):
                self.transparent += 1
            elif isinstance(p, Unsupported):
                self.add(p)

    def to_dict(self) -> dict:
        return {"steps": self.steps, "computes": self.computes, "transparent": self.transparent,
                "unsupported": self.unsupported, "by_pattern": dict(sorted(self.by_pattern.items())),
                "by_spec": dict(sorted(self.by_spec.items())), "unsupported_ops": self.unsupported_ops,
                "unsupported_examples": self.unsupported_examples, "spec_modules": dict(sorted(self.spec_modules.items()))}


def spec_module(spec: Any) -> str:
    """The module that defines a specialization's body (`defn.body` of a composite, `evaluate` of a primitive): the
    registry a spec comes from — `verity_vllm.program.registry.quarantine.*` marks provisional numerical semantics (gate G4)."""
    defn = getattr(spec, "defn", spec)
    for attr in ("body", "evaluate"):
        fn = getattr(defn, attr, None)
        mod = getattr(fn, "__module__", None)
        if mod:
            return str(mod)
    return str(getattr(type(defn), "__module__", "?"))


def spec_label(spec: Any) -> str:
    """`Name_vN{K=..,N=..}` for a SpecializedDefinition (its `id`; falls back to repr)."""
    try:
        return spec.id
    except AttributeError:
        return repr(spec)
