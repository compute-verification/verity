"""General structural query API over Programs: `GateSet`, `Family`, the query AST, `evaluate`.

One general API, not a proof-block language: a query returns a `GateSet` or an ordered lazy `Family` of GateSets.  Results may
overlap or omit gates; nothing here assumes partitions and Definitions know nothing about proof blocks.  The Program hierarchy
(`Scope`, `Body.offsets`, specialization sharing) is used as a compact *index*: a family is a per-body **plan** (one
entry per body node it touches; entries are memoised per specialization, so every activation of a body shares them),
counts are structural (`n` x callee count, prefix sums), `by_index` is mixed-radix over the prefix sums -- the
`verity_vllm.check.replay.vu_query._Planner` method.  The root body of a folded Program has one node per runtime instance, so a
top-level plan costs O(root nodes) once; gates are never expanded.

Values
    GateSet(parts)      union of `parts.Part`s (each Part = items of one activation); algebra `union/minus/inter`
                        refines Parts along the hierarchy so results stay Parts; equality is by root gate intervals.
    Family              ordered lazy sequence: count(), by_index(i), index_of(S), iteration, tiling().

Query AST (frozen dataclasses; `evaluate(program, q)` is the only entry point; string syntax in `verity.ml.circuits.query`,
JSON codec in `verity.ml.circuits.query_codec`):
    instances(path)               one set per activation / batch member / scan iteration of the matched nodes
    members(path, sel, k=1)       inside each matched activation: one set per k adjacent members of the nodes `sel` selects
    members(path, sel, k=(r, c))  ... or one set per r x c rectangle: r members of a selected node by c members of the one
                                  batch node of its callee's body (PoUW's tiles: A rows by B rows), row blocks outermost
    nodes(path, sel)              inside each matched activation: each selected node whole (all members) as one set
    zip(path, [sel...])           inside each matched activation: set m = union of member m of every selected node (equal n)
    each(F, G)                    refinement: G evaluated inside every activation F selects
    concat(F1, F2, ...)           ordered union of families (in one body); `rest()` inside = complement of the siblings
    rest(F)                       every node/member of the bodies F refines that F does not cover, one set per member
    default(w)                    a node is one set per member iff its declared distinct return width <= w, else descended
    frontier(X)                   Q_call's rule in the activation, its returns read: a piece (the activation, a node, a
                                  member) whose w_out (computing gates read after it) <= X is one set, else its children;
                                  `each(F, frontier(X))` cuts each set of F so (a gate that computes nothing: no set)
    override(F, {path: G})        F, but nodes matching `path` (anywhere) are treated by G
    where(F, pred)                keep the sets whose node satisfies pred (spec pattern / form / width)
    table(path, name, of)         inside each matched activation: one set per label of the private table `name`, which
                                  labels the body's nodes (of="nodes") or the activation's gates (of="gates"); the rows are
                                  `evaluate(..., tables=...)`'s, one per distinct Definition, never one per copy
    group(F, key)                 F's sets in classes of a key of their node's Definition (`definition`, `name`, `base`,
                                  `static:K`), one set per class in first-seen order
    label(F, affine=(a, b, n))    F's sets through a label map, label l the union of the sets it receives: set i (its index
    label(F, table=name)          within the activation) to (a*i + b) mod n, or to the label a nodes table gives its node
    union(F, G) | minus | inter   GateSet algebra lifted elementwise (or broadcast from a one-set family); root()

A table, group, label or rectangle set, or frontier's whole activation, may span nodes or activations: `each` and `where`
refuse it, `rest` complements only a nodes table, a whole activation and rectangles, and a family holding one has no
`tiling()`.  `PlanFamily.label_of(i)` is the label a table,
group or label map gives set i.

Paths: `/Name` (direct children), `//Name` (descendants; a match stops the descent), `Name` may be a definition name,
`Name_vN`, a family, or `Name_vN{K=*,N=1536}` / `Name_vN{*}` (statics compared as `SpecializedDefinition.id` renders
them).  `@k`, `@call:Name`, `@batch:Prim`, `@scan:Name`, `@fn:Name`, `@form=batch`, `@name:label`, `@*` are node selectors
relative to a body; a path may be a single node selector (relative to the enclosing `each` body / the root) or `.`.
"""
from __future__ import annotations

import bisect
import functools
import itertools
import math
import re
import weakref
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterator, Mapping, Protocol, Sequence

from .codec import _split_record, descriptor_id
from .defs import PrimitiveDefinition, SpecializedDefinition, _fmt_static
from .intervals import complement, norm, total
from .layout import Scope
from .parts import Item, Part, n_members
from .intervals import Intervals, full, ref_intervals, restrict
from .liveness import Liveness, _members
from .refs import Refs

if TYPE_CHECKING:  # pragma: no cover
    from .program import Program


class QueryError(ValueError):
    """A query that cannot be parsed or planned on this Program: unknown constructor, malformed path / selector /
    document, a `zip` over nodes with unequal member counts, `each` over zip sets, ...  Raised by `parse` / `from_json`
    or at plan time (`evaluate`), never lazily from `by_index`: a family that exists is well-formed.  Selecting nothing
    is not an error (an empty family)."""


# ---------------------------------------------------------------------------------------------------------
# GateSet
# ---------------------------------------------------------------------------------------------------------


def _scope_key(s: Scope) -> tuple[int, int]:
    return (id(s.spec), s.offset)


def _chain_between(anc: Scope, desc: Scope) -> list[tuple[int, int]] | None:
    """(node_idx, member) steps from `anc` down to `desc` (empty when equal); None when `anc` is not an ancestor."""
    steps: list[tuple[int, int]] = []
    s: Scope | None = desc
    ak = _scope_key(anc)
    while s is not None:
        if _scope_key(s) == ak:
            return steps[::-1]
        steps.append((s.node_idx, s.member))
        s = s.parent
    return None


def _all_items(spec: SpecializedDefinition) -> frozenset[Item]:
    return frozenset((j, m) for j, node in enumerate(spec.body.nodes) for m in range(n_members(node)))


def _merge_parts(parts: list[Part]) -> tuple[Part, ...]:
    """Parts of the same activation are joined (items union); order of first appearance."""
    by_key: dict[tuple[int, int], tuple[Scope, set[Item]]] = {}
    order: list[tuple[int, int]] = []
    for p in parts:
        if not p.items:
            continue
        k = _scope_key(p.scope)
        if k not in by_key:
            by_key[k] = (p.scope, set(p.items))
            order.append(k)
        else:
            by_key[k][1].update(p.items)
    return tuple(Part(by_key[k][0], frozenset(by_key[k][1])) for k in order)


def _minus_part(pa: Part, pb: Part) -> list[Part]:
    if _scope_key(pa.scope) == _scope_key(pb.scope):
        rest = pa.items - pb.items
        return [Part(pa.scope, rest)] if rest else []
    down = _chain_between(pa.scope, pb.scope)
    if down is not None:  # pb lies inside one item of pa: refine that item along the path
        first = down[0]
        if first not in pa.items:
            return [pa]
        out: list[Part] = []
        rest = pa.items - {first}
        if rest:
            out.append(Part(pa.scope, rest))
        cur = pa.scope.child(*first)
        for step in down[1:]:
            items = _all_items(cur.spec) - {step}
            if items:
                out.append(Part(cur, items))
            cur = cur.child(*step)
        items = _all_items(cur.spec) - pb.items
        if items:
            out.append(Part(cur, items))
        return out
    up = _chain_between(pb.scope, pa.scope)
    if up is not None:  # pa lies inside pb's activation
        return [] if up[0] in pb.items else [pa]
    return [pa]


def _inter_part(pa: Part, pb: Part) -> Part | None:
    if _scope_key(pa.scope) == _scope_key(pb.scope):
        both = pa.items & pb.items
        return Part(pa.scope, both) if both else None
    down = _chain_between(pa.scope, pb.scope)
    if down is not None:
        return pb if down[0] in pa.items else None
    up = _chain_between(pb.scope, pa.scope)
    if up is not None:
        return pa if up[0] in pb.items else None
    return None


@dataclass(frozen=True, eq=False)
class GateSet:
    """A set of root gates as a union of `Part`s.  Structural queries produce one Part per set; algebra may produce
    several (possibly in different activations).  Equality and hashing are by gate content (root intervals)."""

    parts: tuple[Part, ...]

    @cached_property
    def _ivs(self) -> tuple[tuple[int, int], ...]:
        ivs: list[tuple[int, int]] = []
        for p in self.parts:
            ivs.extend(p.intervals())
        return norm(ivs)

    def intervals(self) -> list[tuple[int, int]]:
        """Half-open root gate intervals, ascending, merged."""
        return list(self._ivs)

    def size(self) -> int:
        return sum(hi - lo for lo, hi in self._ivs)

    def contains(self, g: int) -> bool:
        ivs = self._ivs
        i = bisect.bisect_right(ivs, (g, float("inf"))) - 1
        return i >= 0 and ivs[i][0] <= g < ivs[i][1]

    def __contains__(self, g: int) -> bool:
        return self.contains(g)

    def items(self) -> int:
        return sum(len(p.items) for p in self.parts)

    def union(self, o: "GateSet") -> "GateSet":
        return GateSet(_merge_parts(list(self.parts) + list(o.parts)))

    def minus(self, o: "GateSet") -> "GateSet":
        pieces = list(self.parts)
        for pb in o.parts:
            nxt: list[Part] = []
            for pa in pieces:
                nxt.extend(_minus_part(pa, pb))
            pieces = nxt
        return GateSet(_merge_parts(pieces))

    def inter(self, o: "GateSet") -> "GateSet":
        out: list[Part] = []
        for pa in self.parts:
            for pb in o.parts:
                r = _inter_part(pa, pb)
                if r is not None:
                    out.append(r)
        return GateSet(_merge_parts(out))

    def __eq__(self, other) -> bool:
        return isinstance(other, GateSet) and self._ivs == other._ivs

    def __hash__(self) -> int:
        return hash(self._ivs)

    def __repr__(self) -> str:
        ivs = self._ivs
        shown = ", ".join(f"[{lo}:{hi}]" for lo, hi in ivs[:4]) + (", ..." if len(ivs) > 4 else "")
        return f"GateSet({self.size()} gates; {shown})"


def gateset_of_scope(scope: Scope) -> GateSet:
    """The whole activation `scope` as one GateSet (every item of its body)."""
    return GateSet((Part(scope, _all_items(scope.spec)),))


# ---------------------------------------------------------------------------------------------------------
# Tiling (what partition.py's fast path consumes)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeTiling:
    """How one entry of a refined body treats node `node`: `atomic` (k adjacent members -> one set), `whole` (all
    members one set), `zip` (member m united with member m of the other nodes of `group`), `descend` (every member's
    body is refined by `Tiling.bodies[child]`)."""
    node: int
    kind: str
    members: tuple[tuple[int, int], ...]
    k: int = 1
    group: int | None = None
    child: str | None = None


@dataclass(frozen=True)
class BodyTiling:
    """The entries of one refined body in family order.  `key` is the plan key (`Tiling.bodies` key; a `descend`
    entry's `child` names another key), `spec` the body's specialization, `count` the number of sets one activation of
    this refined body contributes (zip group = `n`, atomic = sum of `ceil((hi-lo)/k)` over ranges, whole = 1, descend =
    `n` x `bodies[child].count`).  Node indices not mentioned by any entry are not covered by the family."""
    key: str
    spec_id: str
    spec: SpecializedDefinition
    n_nodes: int
    entries: tuple[NodeTiling, ...]
    count: int


@dataclass(frozen=True)
class Tiling:
    """Per refined body (keyed by plan key -- one body may be refined differently in different contexts), the node
    assignment in family order.  A description, not a claim: partially covered bodies are reported as they are."""
    root: str
    bodies: Mapping[str, BodyTiling]


@dataclass(frozen=True)
class SetClass:
    """All sets a structural family produces from one non-descend entry of one refined body are layout-identical up to
    the activation offset (and, for `atomic`/`zip`, the member index): one class.  `representative` is the first set of
    the class in family order; `per_activation` sets per activation of the body; `activations` activations of that
    refined body in the Program; `total = per_activation * activations` sets in the class.  A table's label or a group's
    key is a class of its own (`per_activation` 1)."""
    body: str
    entry: int
    kind: str
    representative: GateSet
    per_activation: int
    activations: int
    label: object = None  # a table's label or a group's key: each is a class of its own

    @property
    def total(self) -> int:
        return self.per_activation * self.activations


# ---------------------------------------------------------------------------------------------------------
# Family protocol
# ---------------------------------------------------------------------------------------------------------


class Family(Protocol):
    """An ordered, lazy sequence of GateSets.  `count()` and `by_index(i)` are structural (no gate expansion);
    `index_of(S)` is the inverse of `by_index` (None when S is not a set of the family); iteration yields the sets in
    index order; `tiling()` is the structural description for families built from structural constructors (`instances`,
    `members`, `nodes`, `zip`, `each`, `concat`, `rest`, `default`, `frontier`, `override`, `where`) and None otherwise
    (GateSet algebra results, families holding `table`, `group` or `label` sets, rectangles or frontier's whole activation,
    hand-made families)."""

    def count(self) -> int: ...
    def by_index(self, i: int) -> GateSet: ...
    def index_of(self, s: GateSet) -> int | None: ...
    def __iter__(self) -> Iterator[GateSet]: ...
    def tiling(self) -> Tiling | None: ...


# ---------------------------------------------------------------------------------------------------------
# Selectors and paths
# ---------------------------------------------------------------------------------------------------------


_NAME_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\[\]\.\-]*)(\{(.*)\})?$")


@dataclass(frozen=True)
class NameSel:
    """`Name`, `Name_vN`, family name, or `Name_vN{K=*,N=1536}` / `Name{*}`.  Without braces a definition matches by
    name, id or family (the legacy rule).  With braces the base matches by name or `Name_vN`, and every listed static
    must equal the rendered binding (`*` = any value); unlisted statics are unconstrained."""
    name: str
    statics: tuple[tuple[str, str], ...] | None = None
    any_statics: bool = False

    @staticmethod
    def parse(text: str) -> "NameSel":
        text = text.strip()
        m = _NAME_RE.match(text)
        if not m:
            raise QueryError(f"cannot parse selector {text!r}")
        name, braces, inner = m.group(1), m.group(2), m.group(3)
        if braces is None:
            return NameSel(name)
        inner = (inner or "").strip()
        if inner in ("", "*"):
            return NameSel(name, (), True)
        st: list[tuple[str, str]] = []
        any_ = False
        for piece in _split_top(inner):
            piece = piece.strip()
            if piece == "*":
                any_ = True
                continue
            if "=" not in piece:
                raise QueryError(f"static constraint {piece!r} needs K=value or *")
            k, v = piece.split("=", 1)
            st.append((k.strip(), v.strip()))
        return NameSel(name, tuple(st), any_)

    def __str__(self) -> str:
        if self.statics is None:
            return self.name
        parts = [f"{k}={v}" for k, v in self.statics]
        if self.any_statics and not parts:
            return f"{self.name}{{*}}"
        return f"{self.name}{{{','.join(parts)}}}"

    def matches(self, fn) -> bool:
        base = fn.id if isinstance(fn, PrimitiveDefinition) else fn.base_id
        if self.statics is None:
            if fn.name == self.name or base == self.name or getattr(fn, "family", None) == self.name:
                return True
            # a full specialization id (`Name_vN{...}`) -- computed last: rendering statics is the expensive part
            return "{" in self.name and fn.id == self.name
        if not (fn.name == self.name or base == self.name):
            return False
        bindings = fn.statics
        for k, v in self.statics:
            if k not in bindings:
                return False
            if v != "*" and _fmt_static(bindings[k]) != v:
                return False
        return True


def _split_top(s: str) -> list[str]:
    """Split on commas outside brackets/parentheses/braces."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur or out:
        out.append("".join(cur))
    return out


@dataclass(frozen=True)
class NodeSel:
    """`@k` | `@call:Name` | `@batch:Prim` | `@scan:Name` | `@fn:Name` | `@form=batch` | `@name:label` | `@*`."""
    kind: str
    arg: str | int | None = None

    @staticmethod
    def parse(text: str) -> "NodeSel":
        if not isinstance(text, str):
            raise QueryError(f"a node selector must be a string like '@batch:F', '@k' or '@*', got {text!r}")
        t = text.strip()
        if not t.startswith("@"):
            raise QueryError(f"node selector must start with '@': {text!r}")
        t = t[1:]
        if t == "*":
            return NodeSel("any")
        if t.isdigit():
            return NodeSel("index", int(t))
        if t.startswith("form="):
            form = t[5:]
            if form not in ("primitive", "call", "batch", "scan"):
                raise QueryError(f"unknown node form {form!r}")
            return NodeSel("form", form)
        for kind in ("call", "batch", "scan", "fn", "name"):
            if t.startswith(kind + ":"):
                arg = t[len(kind) + 1:]
                if kind != "name":
                    NameSel.parse(arg)  # validate
                return NodeSel(kind, arg)
        raise QueryError(f"cannot parse node selector {text!r}")

    def __str__(self) -> str:
        if self.kind == "any":
            return "@*"
        if self.kind == "index":
            return f"@{self.arg}"
        if self.kind == "form":
            return f"@form={self.arg}"
        return f"@{self.kind}:{self.arg}"

    @cached_property
    def _name(self) -> NameSel | None:
        return NameSel.parse(str(self.arg)) if self.kind in ("call", "batch", "scan", "fn") else None

    def matches(self, j: int, node) -> bool:
        k = self.kind
        if k == "any":
            return True
        if k == "index":
            return j == self.arg
        if k == "form":
            return node.form == self.arg
        if k == "name":
            return node.name == self.arg
        if k == "fn":
            return self._name.matches(node.fn)  # type: ignore[union-attr]
        if node.form != k:  # call / batch / scan
            return False
        return self._name.matches(node.fn)  # type: ignore[union-attr]


@dataclass(frozen=True)
class Step:
    axis: str  # "child" (/) | "desc" (//) | "node" (@) | "self" (.)
    sel: NameSel | NodeSel | None = None

    def __str__(self) -> str:
        if self.axis == "self":
            return "."
        if self.axis == "node":
            return str(self.sel)
        return ("/" if self.axis == "child" else "//") + str(self.sel)


_STEP_RE = re.compile(r"(//|/)([^/]*)")


@dataclass(frozen=True)
class Path:
    steps: tuple[Step, ...]

    @staticmethod
    def parse(text: str) -> "Path":
        if not isinstance(text, str):
            raise QueryError(f"a path must be a string like '//Gemm_v1{{*}}', '@batch:F' or '.', got {text!r}")
        t = text.strip()
        if t in (".", "/", ""):
            return Path((Step("self"),))
        if t.startswith("@"):
            # a node selector, optionally followed by more steps
            m = re.match(r"^(@[^/]*)(.*)$", t)
            head, tail = m.group(1), m.group(2)  # type: ignore[union-attr]
            steps = [Step("node", NodeSel.parse(head))]
            if tail:
                steps.extend(Path.parse(tail).steps)
            return Path(tuple(steps))
        steps: list[Step] = []
        pos = 0
        while pos < len(t):
            m = _STEP_RE.match(t, pos)
            if not m or not m.group(2).strip():
                raise QueryError(f"cannot parse path at {t[pos:]!r}")
            name = m.group(2).strip()
            pos = m.end()
            # a step may contain '{...}' with commas but not '/', fine; braces may contain '/'? not supported
            if name.startswith("@"):
                steps.append(Step("node", NodeSel.parse(name)))
            else:
                steps.append(Step("child" if m.group(1) == "/" else "desc", NameSel.parse(name)))
        return Path(tuple(steps))

    def __str__(self) -> str:
        out = []
        for i, s in enumerate(self.steps):
            out.append(("/" if i and s.axis == "node" else "") + str(s))
        return "".join(out)

    @property
    def is_self(self) -> bool:
        return len(self.steps) == 1 and self.steps[0].axis == "self"

    @property
    def is_absolute(self) -> bool:
        return self.steps[0].axis in ("child", "desc")


def _path(p) -> Path:
    return p if isinstance(p, Path) else Path.parse(p)


def _nodesel(s) -> NodeSel:
    return s if isinstance(s, NodeSel) else NodeSel.parse(s)


# ---------------------------------------------------------------------------------------------------------
# Query AST
# ---------------------------------------------------------------------------------------------------------


class Query:
    """Base of the frozen AST dataclasses."""
    __slots__ = ()


@dataclass(frozen=True)
class Instances(Query):
    path: Path


@dataclass(frozen=True)
class Members(Query):
    """`k` adjacent members of each selected node per set, or with `k=(r, c)` the r x c rectangles over two axes: r
    members of the selected node by c members of the one batch node of its callee's body."""
    path: Path
    sel: NodeSel
    k: int | tuple[int, int] = 1


@dataclass(frozen=True)
class Nodes(Query):
    path: Path
    sel: NodeSel


@dataclass(frozen=True)
class Zip(Query):
    path: Path
    sels: tuple[NodeSel, ...]


@dataclass(frozen=True)
class Each(Query):
    base: Query
    inner: Query


@dataclass(frozen=True)
class Concat(Query):
    parts: tuple[Query, ...]


@dataclass(frozen=True)
class Rest(Query):
    base: Query | None = None


@dataclass(frozen=True)
class Default(Query):
    w: int


@dataclass(frozen=True)
class Frontier(Query):
    """Q_call's rule inside the enclosing activation, its every return read: a piece (the activation, then a body node,
    a member or iteration of a batch or scan node, a call node's or composite member's body nodes) is one set when its
    w_out is at most `w`, else its children are; a gate is one set, and a piece without computing gates none.  w_out is
    the summed declared widths of the piece's computing gates (those whose primitive has parameters) read after it or
    returned, through any pass-through."""
    w: int


@dataclass(frozen=True)
class Override(Query):
    base: Query
    cases: tuple[tuple[Path, Query], ...]


@dataclass(frozen=True)
class Pred:
    spec: NameSel | None = None
    form: str | None = None
    w_max: int | None = None
    w_min: int | None = None


@dataclass(frozen=True)
class Where(Query):
    base: Query
    pred: Pred


@dataclass(frozen=True)
class Union(Query):
    a: Query
    b: Query


@dataclass(frozen=True)
class Minus(Query):
    a: Query
    b: Query


@dataclass(frozen=True)
class Inter(Query):
    a: Query
    b: Query


@dataclass(frozen=True)
class Root(Query):
    pass


TABLE_OF = ("nodes", "gates")


@dataclass(frozen=True)
class Table(Query):
    """Inside each activation `path` matches, the sets of the table `name` labels: label l's set is the body nodes (`of="nodes"`,
    each node whole) or the gates of the activation (`of="gates"`, in layout order) the table gives label l.  The term names
    the table; its rows are evaluation input (`evaluate(..., tables=...)`), one per distinct Definition, keyed by its
    descriptor id (`codec.descriptor_id`)."""
    path: Path
    name: str
    of: str = "nodes"


_KEY_PARTS = ("definition", "name", "base")


@dataclass(frozen=True)
class Group(Query):
    """The sets of `base` in classes of `key`, a tuple of parts of each set's node's Definition: `definition` (its descriptor
    id, statics included), `name` (`Name`, every version and specialization), `base` (`Name_vN`), `static:K` (static K's
    canonical JSON as the descriptor id writes it, "" when the Definition has none).  One set per class, in first-seen order
    of `base`; the empty key is one class."""
    base: Query
    key: tuple[str, ...]


@dataclass(frozen=True)
class Label(Query):
    """The sets of `base` sent through a label map, label l holding the union of the sets it receives.  `affine=(a, b, n)`:
    set i (its index in `base` within the enclosing activation) goes to (a*i + b) mod n, n labels; `table=name`: a set goes
    to the label table `name` gives its node of the body the term is evaluated in (rows as `table`'s, of="nodes")."""
    base: Query
    affine: tuple[int, int, int] | None = None
    table: str | None = None


# -- lowercase constructors (accept strings) ---------------------------------------------------------------


def instances(path) -> Instances:
    return Instances(_path(path))


def members(path, sel, k: int | tuple[int, int] = 1) -> Members:
    if isinstance(k, (tuple, list)):
        if len(k) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in k):
            raise QueryError(f"members(..., k=(r, c)) needs two positive integers, got {k!r}")
        return Members(_path(path), _nodesel(sel), (k[0], k[1]))
    if k <= 0:
        raise QueryError("members(...) needs a positive k")
    return Members(_path(path), _nodesel(sel), int(k))


def nodes(path, sel) -> Nodes:
    return Nodes(_path(path), _nodesel(sel))


def zip_(path, sels=None, **_ignored) -> Zip:
    """`zip(path, [sel...])`; `zip(sel)` alone means `zip(".", [sel])`."""
    if sels is None:
        sels, path = [path], "."
    if isinstance(sels, (str, NodeSel)):
        sels = [sels]
    return Zip(_path(path), tuple(_nodesel(s) for s in sels))


def each(base, inner) -> Each:
    return Each(_as_query(base), _as_query(inner))


def concat(*parts) -> Concat:
    return Concat(tuple(_as_query(p) for p in parts))


def rest(base=None) -> Rest:
    return Rest(None if base is None else _as_query(base))


def default(w: int) -> Default:
    return Default(int(w))


def frontier(w: int) -> Frontier:
    if isinstance(w, bool) or not isinstance(w, int) or w < 0:
        raise QueryError(f"frontier(X) needs an integer X >= 0 (bits), got {w!r}")
    return Frontier(w)


def override(base, cases) -> Override:
    items = cases.items() if isinstance(cases, dict) else cases
    return Override(_as_query(base), tuple((_path(p), _as_query(g)) for p, g in items))


def where(base, spec=None, form=None, w_max=None, w_min=None) -> Where:
    return Where(_as_query(base), Pred(None if spec is None else (spec if isinstance(spec, NameSel) else NameSel.parse(spec)),
                                       form, None if w_max is None else int(w_max), None if w_min is None else int(w_min)))


def union(a, b) -> Union:
    return Union(_as_query(a), _as_query(b))


def minus(a, b) -> Minus:
    return Minus(_as_query(a), _as_query(b))


def inter(a, b) -> Inter:
    return Inter(_as_query(a), _as_query(b))


def root() -> Root:
    return Root()


def table(path, name: str, of: str = "nodes") -> Table:
    if not isinstance(name, str) or not name:
        raise QueryError(f"table(...) needs a table name, got {name!r}")
    if of not in TABLE_OF:
        raise QueryError(f"table(...): of must be one of {TABLE_OF}, got {of!r}")
    return Table(_path(path), name, of)


def _base_name(base_id: str) -> str:
    """`Name` of `Name_vN`: the id without its last `_v<digits>`, or the id itself when it has none."""
    head, sep, version = base_id.rpartition("_v")
    return head if sep and all(c in "0123456789" for c in version) else base_id


def _key(key) -> tuple[str, ...]:
    parts = (key,) if isinstance(key, str) else tuple(key)
    for p in parts:
        if not isinstance(p, str) or not (p in _KEY_PARTS or (p.startswith("static:") and len(p) > 7)):
            raise QueryError(f"group(...): a key part is one of {_KEY_PARTS} or 'static:K', got {p!r}")
    return parts


def group(base, key) -> Group:
    return Group(_as_query(base), _key(key))


def label(base, affine=None, table: str | None = None) -> Label:
    if (affine is None) == (table is None):
        raise QueryError("label(F, affine=[a, b, n]) or label(F, table=NAME): exactly one label map")
    if table is not None:
        if not isinstance(table, str) or not table:
            raise QueryError(f"label(F, table=NAME) needs a table name, got {table!r}")
        return Label(_as_query(base), None, table)
    if (not isinstance(affine, (tuple, list)) or len(affine) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in affine) or affine[2] == 0):
        raise QueryError(f"label(F, affine=[a, b, n]): a, b >= 0 and n > 0, integers, got {affine!r}")
    return Label(_as_query(base), (affine[0], affine[1], affine[2]), None)


def _as_query(x) -> Query:
    """A Query, or a path string = `instances(path)` (so `each("//X", ...)` reads as in the design doc)."""
    if isinstance(x, Query):
        return x
    if isinstance(x, (str, Path)):
        return Instances(_path(x))
    raise QueryError(f"not a query: {x!r}")


# ---------------------------------------------------------------------------------------------------------
# Plans: per-body entries, memoised per specialization
# ---------------------------------------------------------------------------------------------------------


Chain = tuple[tuple[int, int], ...]
"""(node, member) steps from an activation down to one of its nested activations; () is the activation itself."""


class Entry:
    """One plan entry.  `atomic`: k adjacent members of node j per set; `whole`: node j (all members) as one set;
    `zip`: member m of every node in `nodes` as one set; `descend`: each covered member of j is refined by `sub`;
    `table`: set l is `sets[l]`, (chain, items) pairs relative to the activation, `nodes` the body nodes it reaches;
    `group` / `label`: one set, the union of every set of `sub` (one activation's sets with the key or label `value`);
    `body`: one set, the whole activation (`sub` its nodes whole), what `frontier` gives an activation that fits;
    `affine`: `rule` = (a, b, n), set l the union of the sets i of `sub` with (a*i + b) mod n == l;
    `grid`: `k` = (r, c), set (u, v) = members [u*r, u*r + r) of j by members [v*c, v*c + c) of node `inner` = (j', n')
    of j's callee body, u-major."""

    __slots__ = ("j", "kind", "members", "k", "nodes", "sub", "group", "count", "_groups", "sets", "value", "of", "_sigs",
                 "rule", "inner")

    def __init__(self, j: int, kind: str, members: tuple[tuple[int, int], ...], *, k: int | tuple[int, int] = 1,
                 nodes: tuple[int, ...] = (), sub: "Plan | None" = None, group: int | None = None,
                 sets: tuple[tuple[tuple[Chain, frozenset[Item]], ...], ...] = (), value=None, of: str | None = None,
                 rule: tuple[int, int, int] | None = None, inner: tuple[int, int] | None = None) -> None:
        self.j, self.kind, self.members, self.k, self.nodes, self.sub, self.group = j, kind, members, k, nodes, sub, group
        self.sets, self.value, self.of, self._sigs, self.rule, self.inner = sets, value, of, None, rule, inner
        if kind == "atomic":
            self._groups = tuple(-(-(hi - lo) // k) for lo, hi in members)  # type: ignore[operator]
            self.count = sum(self._groups)
        elif kind == "whole":
            self.count = 1 if total(members) else 0
        elif kind == "zip":
            self.count = total(members)
        elif kind == "table":
            self.count = len(sets)
        elif kind in ("group", "label", "body"):
            self.count = 1
        elif kind == "affine":
            self.count = rule[2]  # type: ignore[index]
        elif kind == "grid":
            (r, c), n_in = k, inner[1]  # type: ignore[misc, index]
            self.count = -(-total(members) // r) * -(-n_in // c)
        else:
            assert sub is not None
            self.count = total(members) * sub.total

    # -- members ---------------------------------------------------------------------------------------------
    def covers(self, m: int) -> bool:
        return any(lo <= m < hi for lo, hi in self.members)

    def member_at(self, pos: int) -> int:
        for lo, hi in self.members:
            if pos < hi - lo:
                return lo + pos
            pos -= hi - lo
        raise IndexError(pos)

    def member_pos(self, m: int) -> int:
        pos = 0
        for lo, hi in self.members:
            if lo <= m < hi:
                return pos + m - lo
            pos += hi - lo
        raise ValueError(m)

    def group_at(self, pos: int) -> tuple[int, int]:
        """atomic: the member range of the pos-th set."""
        for (lo, hi), g in zip(self.members, self._groups):
            if pos < g:
                a = lo + pos * self.k
                return a, min(a + self.k, hi)
            pos -= g
        raise IndexError(pos)

    def group_pos(self, a: int, b: int) -> int | None:
        """atomic: index of the set with member range [a, b), or None."""
        pos = 0
        for (lo, hi), g in zip(self.members, self._groups):
            if lo <= a < hi:
                if (a - lo) % self.k or b != min(a + self.k, hi):
                    return None
                return pos + (a - lo) // self.k
            pos += g
        return None

    def items_of(self, pos: int) -> frozenset[Item]:
        if self.kind == "atomic":
            a, b = self.group_at(pos)
            return frozenset((self.j, m) for m in range(a, b))
        if self.kind == "whole":
            return frozenset((self.j, m) for lo, hi in self.members for m in range(lo, hi))
        if self.kind == "zip":
            m = self.member_at(pos)
            return frozenset((j, m) for j in self.nodes)
        raise TypeError(f"{self.kind} entries have no items of one activation")

    def all_nodes(self) -> tuple[int, ...]:
        return self.nodes if self.kind in ("zip", "table", "group", "label", "affine", "body") else (self.j,)

    @property
    def multi(self) -> bool:
        """A table, group, label, grid or whole-activation set: it may span several nodes or several activations."""
        return self.kind in ("table", "group", "label", "affine", "grid", "body")


class Plan:
    __slots__ = ("key", "spec", "entries", "cum", "total", "_by_node", "query")

    def __init__(self, key: str, spec: SpecializedDefinition, entries: list[Entry]) -> None:
        self.key, self.spec, self.entries = key, spec, entries
        self.cum = [0]
        for e in entries:
            self.cum.append(self.cum[-1] + e.count)
        self.total = self.cum[-1]
        self._by_node = None
        self.query: Query | None = None  # the query this plan refines the body by, when built by `_Evaluator.plan`

    @property
    def by_node(self) -> dict[int, list[tuple[int, Entry]]]:
        if self._by_node is None:
            d: dict[int, list[tuple[int, Entry]]] = {}
            for t, e in enumerate(self.entries):
                for j in e.all_nodes():
                    d.setdefault(j, []).append((t, e))
            self._by_node = d
        return self._by_node


def _is_composite(node) -> bool:
    return not isinstance(node.fn, PrimitiveDefinition)


def _member_arg(node, pi: int, m: int) -> Refs:
    """Argument `pi` of member `m` of a batch node, as references in the enclosing body (an axis-None argument whole)."""
    a, ax = node.args[pi], node.axes[pi]
    if ax is None:
        return a.refs
    if ax == 0:
        L = a.type.elem.leaves
        return a.refs.slice(m * L, (m + 1) * L)
    return a.take(ax, m).refs


def _full(node) -> tuple[tuple[int, int], ...]:
    n = n_members(node)
    return ((0, n),) if n else ()


class _Evaluator:
    """Builds plans for one Program (or, without one, for the Definitions it is asked about: `frontier_pieces`); memoised
    by (spec id, query) and by (plan key, transformation)."""

    def __init__(self, program: "Program | None", tables: Mapping[str, Mapping[str, Sequence[int | None]]] | None = None) -> None:
        # (no strong reference to the Program itself: see `evaluator`)
        self.root: SpecializedDefinition = program.root if program is not None else None  # type: ignore[assignment]
        self.root_scope: Scope = program.circuit.root_scope if program is not None else None  # type: ignore[assignment]
        self.tables = dict(tables or {})
        self._plans: dict[tuple, Plan] = {}
        self._widths: dict[int, int] = {}
        self._contains: dict[tuple, bool] = {}
        self._group_keys: dict[tuple, tuple] = {}
        self._keys = itertools.count()
        self._live = Liveness(transitive=False, member_limit=None)
        self._produced_of: dict[tuple, tuple] = {}
        self._computes_of: dict[int, tuple] = {}

    def _new_plan(self, spec: SpecializedDefinition, entries: list[Entry]) -> Plan:
        return Plan(f"{spec.id}#{next(self._keys)}", spec, entries)

    def _memo(self, key: tuple, make) -> Plan:
        p = self._plans.get(key)
        if p is None:
            p = make()
            self._plans[key] = p
        return p

    # -- declared widths -------------------------------------------------------------------------------------

    def member_width(self, fn) -> int:
        """Declared distinct return width of one member/activation of `fn` (a primitive's `ret.w`; for a composite
        the sum of the widths of its body's distinct returned leaves)."""
        if isinstance(fn, PrimitiveDefinition):
            return fn.ret.w
        w = self._widths.get(id(fn))  # specializations are shared objects (bind cache); `fn.id` renders statics
        if w is None:
            widths = fn.body.ret_type.leaf_widths()
            seen: set = set()
            w = 0
            for leaf, ref in enumerate(fn.body.ret):
                if ref not in seen:
                    seen.add(ref)
                    w += widths[leaf]
            self._widths[id(fn)] = w
        return w

    # -- top-level dispatch ----------------------------------------------------------------------------------

    def plan(self, spec: SpecializedDefinition, q: Query) -> Plan:
        def make() -> Plan:
            p = self._build(spec, q)
            if p.query is None:
                p.query = q
            return p
        return self._memo((spec.id, q), make)

    @staticmethod
    def _root_match(spec: SpecializedDefinition, steps: tuple[Step, ...]) -> bool:
        """The root body is the inlined body of `program.fn` (program.py lowering), so it counts as an activation of
        that function: a leading `/Name` / `//Name` step that matches `program.fn` matches the root body itself."""
        pf = getattr(spec, "program_fn", None)
        return pf is not None and steps[0].axis in ("child", "desc") and steps[0].sel.matches(pf)  # type: ignore[union-attr]

    def _build(self, spec: SpecializedDefinition, q: Query) -> Plan:
        if isinstance(q, Instances):
            if q.path.is_self:
                raise QueryError("instances('.') inside a body is the body itself; use the enclosing family")
            return self._path_plan(spec, q.path.steps, ("instances",))
        if isinstance(q, Members):
            term = ("members", q.sel, q.k)
            return self._sel_plan(spec, term) if q.path.is_self else self._path_plan(spec, q.path.steps, term)
        if isinstance(q, Nodes):
            term = ("nodes", q.sel)
            return self._sel_plan(spec, term) if q.path.is_self else self._path_plan(spec, q.path.steps, term)
        if isinstance(q, Zip):
            term = ("zip", q.sels)
            return self._sel_plan(spec, term) if q.path.is_self else self._path_plan(spec, q.path.steps, term)
        if isinstance(q, Table):
            term = ("table", q.name, q.of)
            return self._sel_plan(spec, term) if q.path.is_self else self._path_plan(spec, q.path.steps, term)
        if isinstance(q, Group):
            return self._apply_group(self.plan(spec, q.base), q.key)
        if isinstance(q, Label):
            base = self.plan(spec, q.base)
            return self._apply_affine(base, q.affine) if q.affine is not None else self._apply_label_table(base, q.table)
        if isinstance(q, Each):
            if isinstance(q.base, Instances) and len(q.base.path.steps) == 1 and self._root_match(spec, q.base.path.steps):
                return self.plan(spec, q.inner)  # each(<the program's own function>, G) = G inside the root body
            return self._apply_each(self.plan(spec, q.base), q.inner)
        if isinstance(q, Concat):
            return self._concat(spec, q.parts)
        if isinstance(q, Rest):
            if q.base is None:
                raise QueryError("rest() without an argument is only meaningful inside concat(...)")
            return self._complement(self.plan(spec, q.base))
        if isinstance(q, Default):
            return self._default(spec, q.w)
        if isinstance(q, Frontier):
            return self._frontier(spec, q.w, full(spec.body.ret.count))
        if isinstance(q, Override):
            pf = getattr(spec, "program_fn", None)
            if pf is not None:
                g = self._case_for(pf, q.cases)
                if g is not None:  # the root body itself is an activation matched by a case
                    p = self.plan(spec, g)
                    if p.total:
                        # the root body = the lowering's Input nodes + the inlined body of program.fn; G owns the
                        # latter, the base keeps the Input nodes (they are not part of any Definition's body)
                        touched = set(p.by_node)
                        base = self.plan(spec, q.base)
                        keep = [e for e in base.entries if e.kind != "zip" and e.j not in touched
                                and getattr(spec.body.nodes[e.j].fn, "family", None) == "Input"]
                        return self._new_plan(spec, keep + list(p.entries))
            return self._apply_override(self.plan(spec, q.base), q.cases)
        if isinstance(q, Where):
            return self._apply_where(self.plan(spec, q.base), q.pred)
        if isinstance(q, (Union, Minus, Inter, Root)):
            raise QueryError(f"{type(q).__name__} is a set-algebra family, not a structural refinement of a body")
        raise QueryError(f"unknown query node {q!r}")

    # -- paths -------------------------------------------------------------------------------------------------

    def _terminal(self, spec: SpecializedDefinition, j: int, node, term: tuple) -> list[Entry]:
        kind = term[0]
        if kind == "instances":
            return [Entry(j, "atomic", _full(node))]
        if not _is_composite(node):
            return []  # members/nodes/zip need a body
        sub = self._sel_plan(node.fn, term)
        return [Entry(j, "descend", _full(node), sub=sub)] if sub.total else []

    def _path_plan(self, spec: SpecializedDefinition, steps: tuple[Step, ...], term: tuple) -> Plan:
        return self._memo((spec.id, "path", steps, term), lambda: self._build_path(spec, steps, term))

    def _build_path(self, spec: SpecializedDefinition, steps: tuple[Step, ...], term: tuple) -> Plan:
        step = steps[0]
        if step.axis == "self":
            if len(steps) == 1:
                raise QueryError("a path ending in '.' selects the enclosing body; use the enclosing family")
            return self._build_path(spec, steps[1:], term)
        if self._root_match(spec, steps):
            if len(steps) > 1:
                return self._build_path(spec, steps[1:], term)
            if term[0] != "instances":
                return self._sel_plan(spec, term)  # members/nodes/zip of the program's own body
            # instances(<program fn>) is the root itself: `evaluate` answers it; inside a plan it selects nothing
        entries: list[Entry] = []
        for j, node in enumerate(spec.body.nodes):
            if step.axis == "node":
                hit = step.sel.matches(j, node)  # type: ignore[union-attr]
            else:
                hit = step.sel.matches(node.fn)  # type: ignore[union-attr]
            if hit:
                if len(steps) == 1:
                    entries.extend(self._terminal(spec, j, node, term))
                elif _is_composite(node):
                    sub = self._path_plan(node.fn, steps[1:], term)
                    if sub.total:
                        entries.append(Entry(j, "descend", _full(node), sub=sub))
            elif step.axis == "desc" and _is_composite(node):
                sub = self._path_plan(node.fn, steps, term)
                if sub.total:
                    entries.append(Entry(j, "descend", _full(node), sub=sub))
        return self._new_plan(spec, entries)

    def _sel_plan(self, spec: SpecializedDefinition, term: tuple) -> Plan:
        return self._memo((spec.id, "sel", term), lambda: self._build_sel(spec, term))

    def _build_sel(self, spec: SpecializedDefinition, term: tuple) -> Plan:
        kind = term[0]
        body = spec.body
        entries: list[Entry] = []
        if kind == "members":
            sel, k = term[1], term[2]
            for j, node in enumerate(body.nodes):
                if sel.matches(j, node):
                    e = self._grid(spec, j, node, k) if isinstance(k, tuple) else Entry(j, "atomic", _full(node), k=k)
                    if e.count or not isinstance(k, tuple):
                        entries.append(e)
        elif kind == "nodes":
            sel = term[1]
            for j, node in enumerate(body.nodes):
                if sel.matches(j, node):
                    entries.append(Entry(j, "whole", _full(node)))
        elif kind == "zip":
            sels = term[1]
            js = [j for j, node in enumerate(body.nodes) if any(s.matches(j, node) for s in sels)]
            if js:
                ns = {n_members(body.nodes[j]) for j in js}
                if len(ns) != 1:
                    raise QueryError(f"zip in {spec.id}: selected nodes {js} have different member counts {sorted(ns)}")
                n = ns.pop()
                entries.append(Entry(js[0], "zip", ((0, n),) if n else (), nodes=tuple(js), group=0))
        elif kind == "table":
            entries.extend(self._table_entries(spec, term[1], term[2]))
        else:
            raise QueryError(kind)
        return self._new_plan(spec, entries)

    @staticmethod
    def _grid(spec: SpecializedDefinition, j: int, node, k: tuple[int, int]) -> Entry:
        """Node j's members by the members of the one batch node of its callee's body, in r x c rectangles."""
        if not _is_composite(node):
            raise QueryError(f"members(..., k={k}) in {spec.id}: node {j} calls the primitive {node.fn.id}, whose members "
                             f"are one axis; use members(..., k={k[0]})")
        inner = [i for i, n in enumerate(node.fn.body.nodes) if n.form == "batch"]
        if len(inner) != 1:
            raise QueryError(f"members(..., k={k}) in {spec.id}: node {j}'s callee {node.fn.id} has {len(inner)} batch nodes; "
                             f"the second axis is its one batch node")
        return Entry(j, "grid", _full(node), k=k, inner=(inner[0], n_members(node.fn.body.nodes[inner[0]])))

    # -- table -----------------------------------------------------------------------------------------------------

    def _table_row(self, spec: SpecializedDefinition, name: str, of: str) -> tuple[Sequence[int | None], int]:
        """The row of table `name` for `spec` and its label count L: labels are None (no set) or 0..L-1, each used."""
        rows = self.tables.get(name)
        key = descriptor_id(spec)
        if rows is None:
            raise QueryError(f"table {name!r} is not given: evaluate(program, query, tables={{{name!r}: {{descriptor id: labels}}}})")
        if not isinstance(rows, Mapping):
            raise QueryError(f"table {name!r}: rows are a mapping from a Definition's descriptor id to its labels, got {type(rows).__name__}")
        row = rows.get(key)
        if row is None:
            raise QueryError(f"table {name!r} has no row for {key}")
        n = len(spec.body.nodes) if of == "nodes" else spec.gates
        if len(row) != n:
            raise QueryError(f"table {name!r}: {key} has {n} {of}, its row {len(row)} labels")
        for x in row:
            if x is not None and (isinstance(x, bool) or not isinstance(x, int) or x < 0):
                raise QueryError(f"table {name!r}: a label is an int >= 0, or None for none, got {x!r} in {key}'s row")
        used = {x for x in row if x is not None}
        n_labels = max(used, default=-1) + 1
        if len(used) != n_labels:
            raise QueryError(f"table {name!r}: {key}'s labels are not 0..{n_labels - 1}, each used: "
                             f"{sorted(set(range(n_labels)) - used)[:4]} unused")
        return row, n_labels

    def _table_entries(self, spec: SpecializedDefinition, name: str, of: str) -> list[Entry]:
        """One `table` entry with a set per label: a nodes row labels each body node whole; a gates row labels each gate
        of the activation, resolved to the (chain, item) of the primitive member that is the gate."""
        row, n_labels = self._table_row(spec, name, of)
        if not n_labels:
            return []
        by_label: list[dict[Chain, set[Item]]] = [{} for _ in range(n_labels)]
        if of == "nodes":
            for j, node in enumerate(spec.body.nodes):
                if row[j] is not None and n_members(node):
                    by_label[row[j]].setdefault((), set()).update((j, m) for m in range(n_members(node)))
        else:
            def walk(s: SpecializedDefinition, base: int, chain: Chain) -> None:
                body = s.body
                for j, node in enumerate(body.nodes):
                    w = node.fn.gates
                    for m in range(n_members(node)):
                        at = base + body.offsets[j] + m * w
                        if isinstance(node.fn, PrimitiveDefinition):
                            if row[at] is not None:
                                by_label[row[at]].setdefault(chain, set()).add((j, m))
                        else:
                            walk(node.fn, at, chain + ((j, m),))
            walk(spec, 0, ())
        sets = tuple(tuple((c, frozenset(items)) for c, items in d.items()) for d in by_label)
        top = sorted({(c[0][0] if c else j) for s in sets for c, items in s for j, _ in items})
        return [Entry(top[0] if top else 0, "table", (), nodes=tuple(top), sets=sets, of=of)]

    # -- group -----------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_key(fn, key: tuple[str, ...]) -> tuple[str, ...]:
        out: list[str] = []
        did = descriptor_id(fn)
        for part in key:
            if part == "definition":
                out.append(did)
            elif part in ("name", "base"):
                base = fn.id if isinstance(fn, PrimitiveDefinition) else fn.base_id
                out.append(_base_name(base) if part == "name" else base)
            else:
                out.append(dict(_split_record(did)).get(part[7:], ""))
        return tuple(out)

    def _keys_of(self, plan: Plan, key: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        """The keys of `plan`'s sets in first-seen family order, memoised per plan."""
        memo = (plan.key, key)
        hit = self._group_keys.get(memo)
        if hit is not None:
            return hit
        seen: dict[tuple[str, ...], None] = {}
        body = plan.spec.body
        for e in plan.entries:
            if e.kind == "descend":
                seen.update(dict.fromkeys(self._keys_of(e.sub, key)))
            elif e.kind in ("atomic", "whole", "grid"):
                if e.count:
                    seen[self._set_key(body.nodes[e.j].fn, key)] = None
            else:
                raise QueryError(f"group(...) keys a set by its node's Definition; a {e.kind} set (node {e.j} of "
                                 f"{plan.spec.id}) is not inside one node")
        out = tuple(seen)
        self._group_keys[memo] = out
        return out

    def _filter(self, plan: Plan, key: tuple[str, ...], value: tuple[str, ...]) -> Plan:
        """`plan` restricted to the sets whose key is `value`."""
        def make() -> Plan:
            body = plan.spec.body
            entries: list[Entry] = []
            for e in plan.entries:
                if e.kind == "descend":
                    if value in self._keys_of(e.sub, key):
                        entries.append(Entry(e.j, "descend", e.members, sub=self._filter(e.sub, key, value)))
                elif e.count and self._set_key(body.nodes[e.j].fn, key) == value:
                    entries.append(e)
            return self._new_plan(plan.spec, entries)
        return self._memo((plan.key, "filter", key, value), make)

    def _apply_group(self, plan: Plan, key: tuple[str, ...]) -> Plan:
        def make() -> Plan:
            entries = []
            for v in self._keys_of(plan, key):
                sub = self._filter(plan, key, v)
                top = tuple(sorted({j for e in sub.entries for j in e.all_nodes()}))
                entries.append(Entry(top[0], "group", (), nodes=top, sub=sub, value=v))
            return self._new_plan(plan.spec, entries)
        return self._memo((plan.key, "group", key), make)

    # -- label -----------------------------------------------------------------------------------------------------

    def _apply_affine(self, plan: Plan, rule: tuple[int, int, int]) -> Plan:
        def make() -> Plan:
            top = tuple(sorted({j for e in plan.entries for j in e.all_nodes()}))
            return self._new_plan(plan.spec, [Entry(top[0] if top else 0, "affine", (), nodes=top, sub=plan, rule=rule)])
        return self._memo((plan.key, "affine", rule), make)

    def _apply_label_table(self, plan: Plan, name: str) -> Plan:
        """One `label` entry per label of the row: the entries of `plan` whose node the row gives that label."""
        def make() -> Plan:
            row, n_labels = self._table_row(plan.spec, name, "nodes")
            by_label: list[list[Entry]] = [[] for _ in range(n_labels)]
            for e in plan.entries:
                js = e.all_nodes()
                labels = {row[j] for j in js}
                if len(labels) > 1:
                    raise QueryError(f"label(..., table={name!r}) labels a set by its node; a {e.kind} set in {plan.spec.id} "
                                     f"spans nodes {list(js)} of labels {sorted(map(str, labels))}")
                if labels and (lab := labels.pop()) is not None:
                    by_label[lab].append(e)
            entries = []
            for lab, es in enumerate(by_label):
                top = tuple(sorted({j for e in es for j in e.all_nodes()}))
                entries.append(Entry(top[0] if top else 0, "label", (), nodes=top, sub=self._new_plan(plan.spec, es), value=lab))
            return self._new_plan(plan.spec, entries)
        return self._memo((plan.key, "label", name), make)

    def _inner_whole(self, fn: SpecializedDefinition, inner: tuple[int, int]) -> Plan:
        """The plan of `fn`'s body whose one set is its batch node `inner` whole: what a grid covers of an activation."""
        return self._memo((fn.id, "inner", inner), lambda: self._new_plan(fn, [Entry(inner[0], "whole", _full(fn.body.nodes[inner[0]]))]))

    # -- each ----------------------------------------------------------------------------------------------------

    def _apply_each(self, plan: Plan, inner: Query) -> Plan:
        return self._memo((plan.key, "each", inner), lambda: self._build_each(plan, inner))

    def _build_each(self, plan: Plan, inner: Query) -> Plan:
        body = plan.spec.body
        entries: list[Entry] = []
        for e in plan.entries:
            node = body.nodes[e.j]
            if e.kind == "descend":
                entries.append(Entry(e.j, "descend", e.members, sub=self._apply_each(e.sub, inner)))
            elif e.kind in ("atomic", "whole"):
                if _is_composite(node):
                    sub = self.plan(node.fn, inner)
                    if len(sub.entries) == 1 and sub.entries[0].kind == "body":
                        entries.append(Entry(e.j, "atomic", e.members))  # the whole activation: the member itself
                    elif sub.total or not isinstance(inner, Frontier):
                        entries.append(Entry(e.j, "descend", e.members, sub=sub))
                elif not isinstance(inner, Frontier) or self._computes(node.fn):
                    entries.append(Entry(e.j, "atomic", e.members))  # a gate cannot be refined; frontier drops one that computes nothing
            else:
                raise QueryError(f"each(...) over {e.kind} sets (node {e.j} of {plan.spec.id}) is not a refinement of one activation")
        return self._new_plan(plan.spec, entries)

    # -- concat / rest -------------------------------------------------------------------------------------------

    def _concat(self, spec: SpecializedDefinition, parts: tuple[Query, ...]) -> Plan:
        plans: list[Plan | None] = [None if (isinstance(p, Rest) and p.base is None) else self.plan(spec, p) for p in parts]
        if any(p is None for p in plans):
            covered = [p for p in plans if p is not None]
            comp = self._complement_of(spec, covered)
            plans = [comp if p is None else p for p in plans]
        entries: list[Entry] = []
        for p in plans:
            entries.extend(p.entries)  # type: ignore[union-attr]
        return self._new_plan(spec, entries)

    def _complement(self, plan: Plan) -> Plan:
        return self._memo((plan.key, "rest"), lambda: self._complement_of(plan.spec, [plan]))

    def _complement_of(self, spec: SpecializedDefinition, plans: list[Plan]) -> Plan:
        body = spec.body
        covered: dict[int, list[tuple[int, int]]] = {}
        descends: dict[int, list[Entry]] = {}
        for p in plans:
            for e in p.entries:
                if (e.kind == "table" and e.of == "nodes") or e.kind == "body":
                    for j in e.nodes:
                        covered.setdefault(j, []).extend(_full(body.nodes[j]))
                    continue
                if e.kind == "grid":  # the rectangles cover the inner batch node of every member
                    covered.setdefault(e.j, []).extend(e.members)
                    sub = self._inner_whole(body.nodes[e.j].fn, e.inner)  # type: ignore[arg-type]
                    descends.setdefault(e.j, []).append(Entry(e.j, "descend", e.members, sub=sub))
                    continue
                if e.multi:
                    raise QueryError(f"rest() beside a {e.kind} term in {spec.id}: only a nodes table and a grid have a structural "
                                     f"complement")
                for j in e.all_nodes():
                    covered.setdefault(j, []).extend(e.members)
                if e.kind == "descend":
                    descends.setdefault(e.j, []).append(e)
        entries: list[Entry] = []
        for j, node in enumerate(body.nodes):
            n = n_members(node)
            rs = covered.get(j)
            if rs is None:
                if n:
                    entries.append(Entry(j, "atomic", ((0, n),)))
                continue
            gaps = tuple(complement(norm(rs), n))
            if gaps:
                entries.append(Entry(j, "atomic", gaps))
            for e in descends.get(j, ()):
                sub = self._complement(e.sub)
                if sub.total:
                    entries.append(Entry(j, "descend", e.members, sub=sub))
        return self._new_plan(spec, entries)

    # -- default -------------------------------------------------------------------------------------------------

    def _default(self, spec: SpecializedDefinition, w: int) -> Plan:
        entries: list[Entry] = []
        for j, node in enumerate(spec.body.nodes):
            if not n_members(node):
                continue
            if self.member_width(node.fn) <= w or not _is_composite(node):
                entries.append(Entry(j, "atomic", _full(node)))
            else:
                entries.append(Entry(j, "descend", _full(node), sub=self.plan(node.fn, Default(w))))
        return self._new_plan(spec, entries)

    # -- frontier ------------------------------------------------------------------------------------------------

    def _computes(self, fn) -> bool:
        """Does an activation of `fn` hold a computing gate (one whose primitive has parameters)?"""
        if isinstance(fn, PrimitiveDefinition):
            return bool(fn.params)
        hit = self._computes_of.get(id(fn))
        if hit is None or hit[0] is not fn:
            hit = self._computes_of[id(fn)] = (fn, any(n_members(n) and self._computes(n.fn) for n in fn.body.nodes))
        return hit[1]

    def _produced(self, fn: SpecializedDefinition, lr: Intervals) -> tuple[int, tuple[Intervals, ...]]:
        """The summed declared widths of the distinct computing gates of one activation of `fn` that its return leaves
        `lr` resolve to, and per parameter the leaves `lr` passes through: a body walked backwards from its returns, each
        node's leaves resolved inside it once (so a value returned twice, or passed through twice, counts once)."""
        hit = self._produced_of.get((id(fn), lr))
        if hit is not None and hit[0] is fn:
            return hit[1]
        body = fn.body
        marks: list[list[tuple[int, int]]] = [[] for _ in body.nodes]
        pmarks: list[list[tuple[int, int]]] = [[] for _ in body.params]

        def mark(refs: Refs) -> None:
            t: dict[tuple[str, int], list] = {}
            ref_intervals(refs, t)
            for (space, idx), ivs in t.items():
                (marks if space == "n" else pmarks)[idx].extend(ivs)

        for lo, hi in lr:
            mark(body.ret.slice(lo, hi))
        w = 0
        for k in range(len(body.nodes) - 1, -1, -1):
            lv = norm(marks[k])
            if lv:
                w += self._node_produced(body.nodes[k], lv, mark)
        out = (w, tuple(norm(p) for p in pmarks))
        self._produced_of[(id(fn), lr)] = (fn, out)
        return out

    def _node_produced(self, node, lv: Intervals, mark=None) -> int:
        """`_produced` of a node's output leaves `lv`: the computing width they resolve to inside the node; `mark(refs)`,
        when given, receives the argument leaves they pass through (a scan's carry passed on is the previous iteration's)."""
        fn = node.fn
        if isinstance(fn, PrimitiveDefinition):
            return total(lv) * fn.ret.w if fn.params else 0
        if node.form == "call":
            w, reads = self._produced(fn, lv)
            for a, rd in zip(node.args, reads) if mark else ():
                for lo, hi in rd:
                    mark(a.refs.slice(lo, hi))
            return w
        if node.form == "batch":
            w = 0
            for m_lo, m_hi, sub in _members(lv, node.n, fn.out_leaves)[0]:
                sw, reads = self._produced(fn, sub)
                w += (m_hi - m_lo) * sw
                for pi, rd in enumerate(reads) if mark else ():
                    for m in range(m_lo, m_hi) if rd else ():
                        refs = _member_arg(node, pi, m)
                        for lo, hi in rd:
                            mark(refs.slice(lo, hi))
            return w
        c, y = node.out.fields[0].leaves, node.out.fields[1].elem.leaves
        ys: dict[int, Intervals] = {}
        for m_lo, m_hi, sub in (_members(restrict(lv, c, c + node.n * y), node.n, y)[0] if y else ()):
            ys.update(dict.fromkeys(range(m_lo, m_hi), tuple((c + lo, c + hi) for lo, hi in sub)))
        carry, w = restrict(lv, 0, c), 0
        for i in range(node.n - 1, -1, -1):
            lr = norm(list(carry) + list(ys.get(i, ())))
            if not lr:
                carry = ()
                continue
            sw, reads = self._produced(fn, lr)
            w += sw
            for pi in range(1, len(node.args)) if mark else ():
                a = node.args[pi]
                at = i * a.type.elem.leaves if pi <= node.n_xs else 0
                for lo, hi in reads[pi]:
                    mark(a.refs.slice(at + lo, at + hi))
            carry = reads[0]
        for lo, hi in carry if mark else ():
            mark(node.args[0].refs.slice(lo, hi))
        return w

    def _frontier(self, spec: SpecializedDefinition, w: int, lr: Intervals) -> Plan:
        """`frontier(w)` in one activation of `spec` whose return leaves `lr` are read after it."""
        return self._memo((spec.id, "frontier", w, lr), lambda: self._build_frontier(spec, w, lr))

    def _build_frontier(self, spec: SpecializedDefinition, w: int, lr: Intervals) -> Plan:
        body = spec.body
        if not self._computes(spec):
            return self._new_plan(spec, [])
        if self._produced(spec, lr)[0] <= w:
            js = tuple(j for j, node in enumerate(body.nodes) if n_members(node))
            whole = self._new_plan(spec, [Entry(j, "whole", _full(body.nodes[j])) for j in js])
            return self._new_plan(spec, [Entry(js[0], "body", (), nodes=js, sub=whole)])
        live = self._live.analyse(spec, lr)
        entries: list[Entry] = []
        for k, node in enumerate(body.nodes):
            fn = node.fn
            if not n_members(node) or not self._computes(fn):
                continue
            lv = live.node_live[k]
            if node.form == "primitive" or self._node_produced(node, lv) <= w:
                entries.append(Entry(k, "atomic" if node.form in ("primitive", "call") else "whole", _full(node)))
            elif node.form == "call":
                entries.append(Entry(k, "descend", _full(node), sub=self._frontier(fn, w, lv)))
            elif isinstance(fn, PrimitiveDefinition):
                entries.append(Entry(k, "atomic", _full(node)))
            else:
                if node.form == "batch":
                    runs = _members(lv, node.n, fn.out_leaves, include_dead=True)[0]
                else:
                    runs, at = [], 0
                    for sub, n in live.scan_runs[k]:
                        runs.append((at, at + n, sub))
                        at += n
                for m_lo, m_hi, sub in runs:
                    if self._produced(fn, sub)[0] <= w:
                        e = Entry(k, "atomic", ((m_lo, m_hi),))
                    else:
                        e = Entry(k, "descend", ((m_lo, m_hi),), sub=self._frontier(fn, w, sub))
                    last = entries[-1] if entries else None
                    if (last is not None and last.j == k and last.kind == e.kind and last.sub is e.sub
                            and last.members[-1][1] == m_lo):
                        entries[-1] = Entry(k, e.kind, ((last.members[0][0], m_hi),), sub=e.sub)
                    else:
                        entries.append(e)
        return self._new_plan(spec, entries)

    # -- override ------------------------------------------------------------------------------------------------

    def _apply_override(self, plan: Plan, cases: tuple[tuple[Path, Query], ...]) -> Plan:
        return self._memo((plan.key, "override", cases), lambda: self._build_override(plan, cases))

    @staticmethod
    def _case_for(fn, cases) -> Query | None:
        for path, g in cases:
            steps = path.steps
            if len(steps) != 1 or steps[0].axis not in ("child", "desc"):
                raise QueryError(f"override paths must be a single `//Name` step, got {path}")
            if steps[0].sel.matches(fn):  # type: ignore[union-attr]
                return g
        return None

    def _node_treatment(self, spec: SpecializedDefinition, j: int, node, g: Query) -> list[Entry]:
        """G's treatment of node j of `spec`: an absolute G is evaluated in the parent body and restricted to node j;
        a relative G is evaluated inside the node's body."""
        if _top_path_absolute(g):
            p = self.plan(spec, g)
            return [e for _, e in p.by_node.get(j, [])]
        if not _is_composite(node):
            return [Entry(j, "atomic", _full(node))]
        sub = self.plan(node.fn, g)
        return [Entry(j, "descend", _full(node), sub=sub)]

    def _contains_case(self, fn, cases) -> bool:
        """Does some node of `fn`'s body (transitively) match a case path?  Memoised per (spec id, cases)."""
        if isinstance(fn, PrimitiveDefinition):
            return False
        key = (id(fn), cases)
        hit = self._contains.get(key)
        if hit is not None:
            return hit
        self._contains[key] = False  # provisional (bodies are acyclic)
        found = any(self._case_for(n.fn, cases) is not None or self._contains_case(n.fn, cases) for n in fn.body.nodes)
        self._contains[key] = found
        return found

    def _build_override(self, plan: Plan, cases) -> Plan:
        body = plan.spec.body
        entries: list[Entry] = []
        done: set[int] = set()
        for e in plan.entries:
            node = body.nodes[e.j]
            g = self._case_for(node.fn, cases) if e.kind != "zip" and not e.multi else None
            if g is not None:
                if e.j in done:
                    continue
                repl = self._node_treatment(plan.spec, e.j, node, g)
                if repl:
                    done.add(e.j)
                    entries.extend(repl)
                    continue
            if e.kind == "descend":
                entries.append(Entry(e.j, "descend", e.members, sub=self._apply_override(e.sub, cases)))
            elif (e.kind in ("atomic", "whole") and plan.query is not None and _is_composite(node)
                  and self._contains_case(node.fn, cases)):
                # the base treats this activation as a whole but a matched body lies inside it: the override reaches
                # it by refining the activation with the base's own query (e.g. default(w) inside Final -> its Gemm)
                sub = self._apply_override(self.plan(node.fn, plan.query), cases)
                entries.append(Entry(e.j, "descend", e.members, sub=sub))
            else:
                entries.append(e)
        return self._new_plan(plan.spec, entries)

    # -- where ---------------------------------------------------------------------------------------------------

    def _apply_where(self, plan: Plan, pred: Pred) -> Plan:
        return self._memo((plan.key, "where", pred), lambda: self._build_where(plan, pred))

    def _pred_ok(self, node, pred: Pred, k: int = 1) -> bool:
        if pred.spec is not None and not pred.spec.matches(node.fn):
            return False
        if pred.form is not None and node.form != pred.form:
            return False
        if pred.w_max is not None or pred.w_min is not None:
            w = self.member_width(node.fn) * k
            if pred.w_max is not None and w > pred.w_max:
                return False
            if pred.w_min is not None and w < pred.w_min:
                return False
        return True

    def _build_where(self, plan: Plan, pred: Pred) -> Plan:
        body = plan.spec.body
        entries: list[Entry] = []
        for e in plan.entries:
            if e.multi:
                raise QueryError(f"where(...) keeps a set by its node; a {e.kind} set (in {plan.spec.id}) spans nodes, so filter its base")
            if e.kind == "descend":
                sub = self._apply_where(e.sub, pred)
                if sub.total:
                    entries.append(Entry(e.j, "descend", e.members, sub=sub))
            elif e.kind == "zip":
                if all(self._pred_ok(body.nodes[j], pred) for j in e.nodes):
                    entries.append(e)
            elif self._pred_ok(body.nodes[e.j], pred, e.k if e.kind == "atomic" else n_members(body.nodes[e.j])):
                entries.append(e)
        return self._new_plan(plan.spec, entries)


def _top_path_absolute(q: Query) -> bool:
    while isinstance(q, (Each, Where, Override, Group, Label)):
        q = q.base
    if isinstance(q, (Instances, Members, Nodes, Zip, Table)):
        return q.path.is_absolute
    return False


# ---------------------------------------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------------------------------------


def _entry_at(scope: Scope, plan: Plan, rem: int) -> tuple[Scope, Entry, int]:
    """(activation, entry, position in the entry) of set `rem` of `plan` applied to `scope`."""
    while True:
        t = bisect.bisect_right(plan.cum, rem) - 1
        e = plan.entries[t]
        rem -= plan.cum[t]
        if e.kind != "descend":
            return scope, e, rem
        q, rem = divmod(rem, e.sub.total)
        scope = scope.child(e.j, e.member_at(q))
        plan = e.sub


def _down(scope: Scope, chain: Chain) -> Scope:
    for j, m in chain:
        scope = scope.child(j, m)
    return scope


def _union_parts(scope: Scope, plan: Plan) -> list[Part]:
    """Every set of `plan` applied to `scope`: one Part per activation and atomic or whole entry, the parts of the rest."""
    out: list[Part] = []
    for e in plan.entries:
        if e.kind == "descend":
            for lo, hi in e.members:
                for m in range(lo, hi):
                    out.extend(_union_parts(scope.child(e.j, m), e.sub))
        elif e.kind in ("atomic", "whole"):
            out.append(Part(scope, frozenset((e.j, m) for lo, hi in e.members for m in range(lo, hi))))
        else:
            for pos in range(e.count):
                out.extend(_set_of(scope, e, pos).parts)
    return out


def _preimage(rule: tuple[int, int, int], label: int, n_sets: int) -> range:
    """The indices i < n_sets with (a*i + b) mod n == label, ascending: one residue class mod n / gcd(a, n), or none."""
    a, b, n = rule
    g = math.gcd(a, n)
    d = (label - b) % n
    if d % g:
        return range(0)
    step = n // g
    return range((d // g) * pow(a // g, -1, step) % step if step > 1 else 0, n_sets, step)


def _grid_cols(e: Entry) -> int:
    return -(-e.inner[1] // e.k[1])  # type: ignore[index]


def _grid_pos(scope: Scope, e: Entry, g: int) -> int | None:
    """The grid set of `e` holding root gate `g`, or None when `g` is outside the inner batch node."""
    body = scope.spec.body
    rel = g - scope.offset
    if not body.offsets[e.j] <= rel < body.offsets[e.j + 1]:
        return None
    fn = body.nodes[e.j].fn
    m, rel = divmod(rel - body.offsets[e.j], fn.gates)
    j_in = e.inner[0]  # type: ignore[index]
    inner = fn.body
    if not inner.offsets[j_in] <= rel < inner.offsets[j_in + 1]:
        return None
    x = (rel - inner.offsets[j_in]) // inner.nodes[j_in].fn.gates
    return (m // e.k[0]) * _grid_cols(e) + x // e.k[1]  # type: ignore[index]


def _set_of(scope: Scope, e: Entry, pos: int) -> GateSet:
    if e.kind == "table":
        return GateSet(_merge_parts([Part(_down(scope, c), items) for c, items in e.sets[pos]]))
    if e.kind in ("group", "label", "body"):
        return GateSet(_merge_parts(_union_parts(scope, e.sub)))
    if e.kind == "affine":
        sub = e.sub
        return GateSet(_merge_parts([p for i in _preimage(e.rule, pos, sub.total) for p in _set_of(*_entry_at(scope, sub, i)).parts]))
    if e.kind == "grid":
        (r, c), (j_in, n_in) = e.k, e.inner  # type: ignore[misc]
        u, v = divmod(pos, _grid_cols(e))
        items = frozenset((j_in, x) for x in range(v * c, min(v * c + c, n_in)))
        return GateSet(tuple(Part(scope.child(e.j, m), items) for m in range(u * r, min(u * r + r, total(e.members)))))
    return GateSet((Part(scope, e.items_of(pos)),))


class PlanFamily:
    """A structural family: `plan` applied to activation `scope`."""

    def __init__(self, ev: _Evaluator, scope: Scope, plan: Plan) -> None:
        self.ev, self.scope, self.plan = ev, scope, plan

    def count(self) -> int:
        return self.plan.total

    def __len__(self) -> int:
        return self.plan.total

    def by_index(self, i: int) -> GateSet:
        if not (0 <= i < self.plan.total):
            raise IndexError(f"index {i} outside the family of {self.plan.total}")
        return self._locate(self.scope, self.plan, i)

    def _locate(self, scope: Scope, plan: Plan, rem: int) -> GateSet:
        return _set_of(*_entry_at(scope, plan, rem))

    def label_of(self, i: int):
        """The label a table, group or label map gives set i (a label, or a group's key), None for a set of another entry."""
        if not (0 <= i < self.plan.total):
            raise IndexError(f"index {i} outside the family of {self.plan.total}")
        _, e, rem = _entry_at(self.scope, self.plan, i)
        return rem if e.kind in ("table", "affine") else e.value if e.kind in ("group", "label") else None

    def __iter__(self) -> Iterator[GateSet]:
        yield from self._walk(self.scope, self.plan)

    def _walk(self, scope: Scope, plan: Plan) -> Iterator[GateSet]:
        for e in plan.entries:
            if e.kind == "descend":
                if not e.sub.total:
                    continue
                for lo, hi in e.members:
                    for m in range(lo, hi):
                        yield from self._walk(scope.child(e.j, m), e.sub)
            else:
                for pos in range(e.count):
                    yield _set_of(scope, e, pos)

    def index_of(self, s: GateSet) -> int | None:
        parts = _merge_parts(list(s.parts))
        if not parts:
            return None
        chains = [_chain_between(self.scope, p.scope) for p in parts]
        if any(c is None for c in chains):
            return None
        if len(parts) == 1 and parts[0].items:
            r = self._index_in(self.plan, chains[0], parts[0].items)
            if r is not None:
                return r
        common = chains[0]
        for c in chains[1:]:
            n = 0
            while n < min(len(common), len(c)) and common[n] == c[n]:
                n += 1
            common = common[:n]
        return self._index_multi(self.scope, self.plan, common, s.intervals())

    def _index_multi(self, scope: Scope, plan: Plan, chain: list[tuple[int, int]], ivs: list[tuple[int, int]]) -> int | None:
        """A set of several parts: its activation is the family's scope or a scope on `chain` (toward every part of it).  A
        grid set is the one holding the set's first gate; another is matched on its intervals."""
        rel = tuple((lo - scope.offset, hi - scope.offset) for lo, hi in ivs)
        for t, e in enumerate(plan.entries):
            if e.kind == "grid":
                pos = _grid_pos(scope, e, ivs[0][0])
                if pos is not None and _set_of(scope, e, pos).intervals() == list(ivs):
                    return plan.cum[t] + pos
            elif e.multi:
                if e._sigs is None:
                    e._sigs = {}
                    for pos in range(e.count):
                        e._sigs.setdefault(tuple((lo - scope.offset, hi - scope.offset) for lo, hi in _set_of(scope, e, pos).intervals()), pos)
                pos = e._sigs.get(rel)
                if pos is not None:
                    return plan.cum[t] + pos
        if not chain:
            return None
        j, m = chain[0]
        for t, e in plan.by_node.get(j, []):
            if e.kind == "descend" and e.covers(m) and e.sub.total:
                r = self._index_multi(scope.child(j, m), e.sub, chain[1:], ivs)
                if r is not None:
                    return plan.cum[t] + e.member_pos(m) * e.sub.total + r
        return None

    def _index_in(self, plan: Plan, chain: list[tuple[int, int]], items: frozenset[Item]) -> int | None:
        if chain:
            j, m = chain[0]
            for t, e in plan.by_node.get(j, []):  # several entries may descend into the same node (search + rest)
                if e.kind == "descend" and e.covers(m) and e.sub.total:
                    r = self._index_in(e.sub, chain[1:], items)
                    if r is not None:
                        return plan.cum[t] + e.member_pos(m) * e.sub.total + r
            return None
        js = sorted({j for j, _ in items})
        for t, e in plan.by_node.get(js[0], []):
            if e.kind == "atomic":
                if len(js) != 1:
                    continue
                ms = sorted(m for _, m in items)
                if ms != list(range(ms[0], ms[-1] + 1)):
                    continue
                g = e.group_pos(ms[0], ms[-1] + 1)
                if g is not None:
                    return plan.cum[t] + g
            elif e.kind == "whole":
                if len(js) == 1 and e.items_of(0) == items:
                    return plan.cum[t]
            elif e.kind == "zip":
                ms = {m for _, m in items}
                if len(ms) == 1 and tuple(js) == tuple(sorted(e.nodes)):
                    m = ms.pop()
                    if e.covers(m):
                        return plan.cum[t] + e.member_pos(m)
        return None

    def tiling(self) -> Tiling | None:
        """The structural description, or None when a table or group set is reached (its sets are not one entry's)."""
        bodies: dict[str, BodyTiling] = {}

        def visit(plan: Plan) -> bool:
            if plan.key in bodies:
                return True
            if any(e.multi for e in plan.entries):
                return False
            ents: list[NodeTiling] = []
            for e in plan.entries:
                if e.kind == "zip":
                    for j in e.nodes:
                        ents.append(NodeTiling(j, "zip", e.members, group=e.group))
                elif e.kind == "descend":
                    ents.append(NodeTiling(e.j, "descend", e.members, child=e.sub.key))
                else:
                    ents.append(NodeTiling(e.j, e.kind, e.members, k=e.k))
            bodies[plan.key] = BodyTiling(plan.key, plan.spec.id, plan.spec, len(plan.spec.body.nodes), tuple(ents), plan.total)
            return all(visit(e.sub) for e in plan.entries if e.kind == "descend")

        return Tiling(self.plan.key, bodies) if visit(self.plan) else None

    def classes(self) -> list[SetClass]:
        """The layout classes of the family's sets (one per non-descend entry of every refined body reached from the
        root), each with a representative set from the first activation reached in family order.  O(plan entries),
        no gate expansion; `sum(c.total) == count()`."""
        # first activation scope per plan key + activation multiplicities, in a topological order of the plan DAG
        first: dict[str, tuple[Scope, Plan]] = {}
        order: list[Plan] = []

        def dfs(scope: Scope, plan: Plan) -> None:
            if plan.key in first:
                return
            first[plan.key] = (scope, plan)
            for e in plan.entries:
                if e.kind == "descend" and e.members:
                    dfs(scope.child(e.j, e.members[0][0]), e.sub)
            order.append(plan)

        dfs(self.scope, self.plan)
        acts: dict[str, int] = {p.key: 0 for p in order}
        acts[self.plan.key] = 1
        for plan in reversed(order):  # parents before children (reverse postorder)
            mult = acts[plan.key]
            for e in plan.entries:
                if e.kind == "descend":
                    acts[e.sub.key] += mult * total(e.members)
        out: list[SetClass] = []
        for plan in reversed(order):
            scope = first[plan.key][0]
            for t, e in enumerate(plan.entries):
                if e.kind == "descend" or not e.count:
                    continue
                if e.kind == "grid":  # one class per rectangle shape: full or edge rows by full or edge columns
                    (r, c), n, n_in = e.k, total(e.members), e.inner[1]  # type: ignore[misc, index]
                    rows = [(n // r, 0), (1 if n % r else 0, n // r)]
                    cols = [(n_in // c, 0), (1 if n_in % c else 0, n_in // c)]
                    for nu, u in rows:
                        for nv, v in cols:
                            if nu and nv:
                                rep = _set_of(scope, e, u * _grid_cols(e) + v)
                                out.append(SetClass(plan.key, t, e.kind, rep, nu * nv, acts[plan.key]))
                    continue
                if e.multi:  # each label's set has its own layout
                    for pos in range(e.count):
                        label = pos if e.kind in ("table", "affine") else e.value
                        out.append(SetClass(plan.key, t, e.kind, _set_of(scope, e, pos), 1, acts[plan.key], label))
                    continue
                rep = GateSet((Part(scope, e.items_of(0)),))
                out.append(SetClass(plan.key, t, e.kind, rep, e.count, acts[plan.key]))
        return out

    def __repr__(self) -> str:
        return f"PlanFamily({self.plan.spec.id}, {self.plan.total} sets)"


class ConstFamily:
    """A fixed list of GateSets (e.g. `root()`)."""

    def __init__(self, sets: list[GateSet]) -> None:
        self.sets = list(sets)

    def count(self) -> int:
        return len(self.sets)

    def __len__(self) -> int:
        return len(self.sets)

    def by_index(self, i: int) -> GateSet:
        if not (0 <= i < len(self.sets)):
            raise IndexError(f"index {i} outside the family of {len(self.sets)}")
        return self.sets[i]

    def __iter__(self) -> Iterator[GateSet]:
        return iter(self.sets)

    def index_of(self, s: GateSet) -> int | None:
        for i, x in enumerate(self.sets):
            if x == s:
                return i
        return None

    def tiling(self) -> Tiling | None:
        return None


class AlgebraFamily:
    """`union/minus/inter` of two families, elementwise; a one-set family broadcasts."""

    def __init__(self, op: str, a, b) -> None:
        self.op, self.a, self.b = op, a, b
        na, nb = a.count(), b.count()
        if na != nb and na != 1 and nb != 1:
            raise QueryError(f"{op}: families of {na} and {nb} sets cannot be combined elementwise")
        self.n = max(na, nb) if (na and nb) else 0

    def count(self) -> int:
        return self.n

    def __len__(self) -> int:
        return self.n

    def by_index(self, i: int) -> GateSet:
        if not (0 <= i < self.n):
            raise IndexError(f"index {i} outside the family of {self.n}")
        x = self.a.by_index(i if self.a.count() > 1 else 0)
        y = self.b.by_index(i if self.b.count() > 1 else 0)
        return getattr(x, self.op)(y)

    def __iter__(self) -> Iterator[GateSet]:
        for i in range(self.n):
            yield self.by_index(i)

    def index_of(self, s: GateSet) -> int | None:
        for i, x in enumerate(self):
            if x == s:
                return i
        return None

    def tiling(self) -> Tiling | None:
        return None


# ---------------------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------------------


_EVALUATORS: dict[int, tuple[object, _Evaluator]] = {}


def evaluator(program: "Program") -> _Evaluator:
    """The (cached per Program object, weakly) plan builder: plans are memoised per specialization across queries."""
    hit = _EVALUATORS.get(id(program))
    if hit is not None and hit[0]() is program:
        return hit[1]
    for k in [k for k, (ref, _) in _EVALUATORS.items() if ref() is None]:  # drop evaluators of collected Programs
        del _EVALUATORS[k]
    ev = _Evaluator(program)
    _EVALUATORS[id(program)] = (weakref.ref(program), ev)
    return ev


def evaluate(program: "Program", query, tables: Mapping[str, Mapping[str, Sequence[int | None]]] | None = None) -> Family:
    """Evaluate `query` (a Query or its string form) against `program`; the root body is the top-level context.  `tables`
    gives the rows of the query's `table` terms: table name -> descriptor id -> one label per body node (`of="nodes"`)
    or per gate of an activation (`of="gates"`), None for none, the labels 0..L-1 each used (`Flock.Annot`'s tables).
    Plans read the tables, so they are memoised per call."""
    from .query import parse  # string sugar lives with the legacy syntax
    q = parse(query) if isinstance(query, str) else query
    if not isinstance(q, Query):
        raise QueryError(f"not a query: {query!r}")
    ev = evaluator(program) if tables is None else _Evaluator(program, tables)
    if isinstance(q, Root) or (isinstance(q, Instances) and (q.path.is_self or (
            len(q.path.steps) == 1 and ev._root_match(program.root, q.path.steps)))):
        return ConstFamily([gateset_of_scope(ev.root_scope)])
    if isinstance(q, (Union, Minus, Inter)):
        op = {Union: "union", Minus: "minus", Inter: "inter"}[type(q)]
        return AlgebraFamily(op, evaluate(program, q.a, tables), evaluate(program, q.b, tables))
    return PlanFamily(ev, ev.root_scope, ev.plan(program.root, q))


Piece = tuple[int, int, tuple[tuple[int, ...], ...]]


def frontier_pieces(fn, w: int, memo: dict | None = None) -> list[Piece]:
    """`frontier(w)`'s sets in one activation of `fn`, its every return read (what `each(instances("@*"), frontier(w))`
    gives each root Call), as (lo, hi, path) over the activation's gates in family order: the path's steps are `(k,)`, body
    node k (a call or primitive node, or a batch or scan node whole), and `(k, m)`, member or iteration m of node k.
    `memo` (a dict) shares plans across calls."""
    if isinstance(fn, PrimitiveDefinition):
        return [(0, 1, ())] if fn.params else []
    ev = None if memo is None else memo.get(("query_ast/frontier",))
    if ev is None:
        ev = _Evaluator(None)
        if memo is not None:
            memo[("query_ast/frontier",)] = ev
    out: list[Piece] = []

    def walk(plan: Plan, off: int, path: tuple) -> None:
        body = plan.spec.body
        for e in plan.entries:
            if e.kind == "body":
                out.append((off, off + plan.spec.gates, path))
                continue
            node, at = body.nodes[e.j], off + body.offsets[e.j]
            if e.kind == "whole":
                out.append((at, off + body.offsets[e.j + 1], path + ((e.j,),)))
                continue
            g, one = node.fn.gates, node.form in ("call", "primitive")
            for lo, hi in e.members:
                for m in range(lo, hi):
                    step = (e.j,) if one else (e.j, m)
                    if e.kind == "atomic":
                        out.append((at + m * g, at + (m + 1) * g, path + (step,)))
                    else:
                        walk(e.sub, at + m * g, path + (step,))

    walk(ev.plan(fn, Frontier(w)), 0, ())
    return out


def refines(a, b) -> bool:
    """Does every set of `a` lie inside one set of `b` (`Verity.Annotation.Refines`), on every Program and in every
    activation where both evaluate?  Decided from the two terms (Query objects or their string form) alone, never from a
    Program: `each` and `where` stay inside their base's sets, `minus` and `inter` inside an operand's; `group` and an
    affine `label` merge their base's sets, `concat` holds its parts', and `root()` everything; `members`, `nodes`, `zip`
    and `table` lie inside `instances` of their path, `members` inside `nodes` of its selector and inside `members` whose
    block sizes are multiples of its own; `each(F, G)` refines `each(F, H)` when G refines H; `default` and `frontier`
    refine themselves at a larger width.  Sound, not complete: False says only that no rule shows it.  A `label` by
    table drops a set whose node has no label, and a `union` with an empty family has no sets, so neither is a coarsening."""
    from .query import parse  # string sugar lives with the legacy syntax
    qa, qb = (parse(q) if isinstance(q, str) else q for q in (a, b))
    for q in (qa, qb):
        if not isinstance(q, Query):
            raise QueryError(f"not a query: {q!r}")
    return _refines(qa, qb)


def _rows_cols(k: int | tuple[int, int]) -> tuple[int, int | None]:
    return (k, None) if isinstance(k, int) else k


@functools.lru_cache(maxsize=4096)
def _refines(a: Query, b: Query) -> bool:
    if a == b or isinstance(b, Root):
        return True
    if isinstance(a, (Each, Where)) and _refines(a.base, b):
        return True
    if isinstance(a, Minus) and _refines(a.a, b):
        return True
    if isinstance(a, Inter) and (_refines(a.a, b) or _refines(a.b, b)):
        return True
    if isinstance(a, Concat) and all(_refines(p, b) for p in a.parts):
        return True
    if isinstance(b, Group) or (isinstance(b, Label) and b.affine is not None):
        if _refines(a, b.base):
            return True
    if isinstance(b, Concat) and any(_refines(a, p) for p in b.parts):
        return True
    if isinstance(a, Each) and isinstance(b, Each) and a.base == b.base and _refines(a.inner, b.inner):
        return True
    if isinstance(a, (Default, Frontier)) and type(a) is type(b):
        return a.w <= b.w
    if isinstance(a, (Members, Nodes, Zip, Table)) and isinstance(b, Instances):
        return a.path == b.path
    if isinstance(a, Members) and isinstance(b, (Nodes, Members)) and (a.path, a.sel) == (b.path, b.sel):
        if isinstance(b, Nodes):
            return True
        (ra, ca), (rb, cb) = _rows_cols(a.k), _rows_cols(b.k)
        return rb % ra == 0 and (cb is None or (ca is not None and cb % ca == 0))
    return False


def parse(text: str) -> Query:
    from .query import parse as _parse
    return _parse(text)


def to_json(query: Query) -> dict:
    from .query_codec import to_json as _to
    return _to(query)


def from_json(doc: dict) -> Query:
    from .query_codec import from_json as _from
    return _from(doc)


def query_id(query: Query) -> str:
    from .query_codec import query_id as _qid
    return _qid(query)


__all__ = ["GateSet", "Family", "Tiling", "BodyTiling", "NodeTiling", "SetClass", "QueryError", "Part", "Item",
           "NameSel", "NodeSel", "Step", "Path", "Pred",
           "Query", "Instances", "Members", "Nodes", "Zip", "Each", "Concat", "Rest", "Default", "Frontier", "Override",
           "Where", "Union", "Minus", "Inter", "Root", "Table", "Group", "Label", "TABLE_OF",
           "instances", "members", "nodes", "zip_", "each", "concat", "rest", "default", "frontier", "override", "where",
           "union", "minus", "inter", "root", "table", "group", "label",
           "evaluate", "evaluator", "frontier_pieces", "refines", "parse", "to_json", "from_json", "query_id", "gateset_of_scope",
           "PlanFamily", "ConstFamily", "AlgebraFamily", "Plan", "Entry"]
