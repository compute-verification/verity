"""The certified rows a proof block reads or writes, from the Program's description: `Spot.CertifiedRows`, which the difftest's
`certified_rows` piece holds this to.

A template's port may be one row of a certified table (PROTOCOL.md §12, `ports.In(certified=label)` on a parameter,
`ports.Out(certified=label)` on its return). The label is the Program's, bound by its digest; the certified values it
stands for are the verifier's: `labels`, ``{label: [value, ...]}`` in order, and `certified`, ``{value: {"root",
"leaves", ...}}``. No check calls this yet.

* `marks`: the Program's marked ports, sorted by template and port. A marked parameter is read by the port of its own
  name, a marked field `j` of the return by the output port ``o<j>``. Refused (`CertifiedRowsError`, its codes sorted):
  a mark that is not one whole parameter or field (``certified-whole``), one label on two ports of a template
  (``certified-twice``), a mark on a Definition any node but a root call or batch calls (``certified-called``). The
  decoder and `program.Program` already refuse a mark on a primitive or the root and a label that is no name, which the
  Lean reader, over a descriptor the Lean decoder takes without reading its `ports`, refuses itself.
* `layout`: each label's rows from 0 over the root's nodes in order, each call or batch of a template marking it taking
  one row a member. One pass over the root's nodes.
* `rows_at`: unit `u`'s ``(port, label, row)`` for each mark of its template, from the template query's ranges.
* `value_at`: row `g` of a label as a certified value's ``(value, root, row)``, each value taking its entry's `leaves`.
* `labels_held`: the labels are the Program's, each value under one label once and held, every certified value under a
  label, and each label's values' leaves its rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from verity.circuits.codec import descriptor_id
from verity.circuits.ports import port_types
from verity.circuits.types import RecordType, Tuple


class CertifiedRowsError(ValueError):
    """`marks`' refusal: `codes`, each once, sorted."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = sorted(set(codes))
        super().__init__(", ".join(self.codes))


@dataclass(frozen=True)
class Mark:
    template: str
    port: str
    label: str
    output: bool


@dataclass(frozen=True)
class Layout:
    """A label's rows: `sites`, ``(root node, first row, mark)`` for each root call or batch of a template marking it, and
    its row count."""
    label: str
    sites: tuple[tuple[int, int, Mark], ...]
    rows: int


def _fields(t: Any) -> list[Any]:
    """`SecondStage.fields`: one per output port, a tuple's or a record's fields, else `t`."""
    if isinstance(t, RecordType):
        return [f for _n, f in t.fields]
    return list(t.fields) if isinstance(t, Tuple) else [t]


def _covers(rows: Sequence[tuple[int, int, str]], a: int, b: int) -> bool:
    """`CertifiedRows.covers`: `rows` are one label's and cover exactly leaves [a, b), in order."""
    rows = sorted(rows, key=lambda r: r[0])
    if not rows:
        return False
    at = a
    for lo, hi, label in rows:
        if lo != at or lo >= hi or label != rows[0][2]:
            return False
        at = hi
    return at == b


def _def_marks(key: str, fn: Any) -> list[Mark]:
    pt = port_types(fn)
    params, ret = fn.signature()
    out = []
    for (name, t), rs in zip(params, pt.params):
        mine = [(lo, hi, p.certified) for lo, hi, p, _w in rs if p.certified is not None]
        if mine:
            if not _covers(mine, 0, t.leaves):
                raise CertifiedRowsError(["certified-whole"])
            out.append(Mark(key, name, mine[0][2], False))
    fs = _fields(ret)
    offs = [0]
    for f in fs:
        offs.append(offs[-1] + f.leaves)
    rows = [(lo, hi, o.certified) for lo, hi, o, _w in pt.ret if o.certified is not None]
    field = [next((j for j in range(len(fs)) if offs[j] <= lo and hi <= offs[j + 1]), None) for lo, hi, _l in rows]
    if None in field:
        raise CertifiedRowsError(["certified-whole"])
    for j in range(len(fs)):
        mine = [r for r, f in zip(rows, field) if f == j]
        if mine:
            if not _covers(mine, offs[j], offs[j + 1]):
                raise CertifiedRowsError(["certified-whole"])
            out.append(Mark(key, f"o{j}", mine[0][2], True))
    if len({m.label for m in out}) != len(out):
        raise CertifiedRowsError(["certified-twice"])
    return out


def marks(program: Any) -> list[Mark]:
    """The marks of `program` (a `Program` or a decoded one), sorted by template and port; `CertifiedRowsError` with
    every Definition's refusal. Each Definition once and each node once."""
    root = program.root
    defs: dict[str, Any] = {}
    root_called: set[str] = set()
    other: set[str] = set()
    todo, seen = [root], {id(root)}
    while todo:
        fn = todo.pop()
        for node in fn.body.nodes:
            key = descriptor_id(node.fn)
            defs[key] = node.fn
            (root_called if fn is root and node.form in ("call", "batch") else other).add(key)
            if hasattr(node.fn, "body") and id(node.fn) not in seen:
                seen.add(id(node.fn))
                todo.append(node.fn)
    codes: list[str] = []
    out: list[Mark] = []
    for key, fn in defs.items():
        pt = port_types(fn)
        if pt is None or not any(r[-2].certified is not None for rs in (*pt.params, pt.ret) for r in rs):
            continue
        if key not in root_called or key in other:
            codes.append("certified-called")
        try:
            out += _def_marks(key, fn)
        except CertifiedRowsError as e:
            codes += e.codes
    if codes:
        raise CertifiedRowsError(codes)
    return sorted(out, key=lambda m: (m.template, m.port))


def layout(program: Any, ms: Sequence[Mark]) -> list[Layout]:
    """Each label of `ms`, sorted, with its rows over the root's nodes in order."""
    sites: dict[str, list] = {label: [] for label in sorted({m.label for m in ms})}
    rows = dict.fromkeys(sites, 0)
    for k, node in enumerate(program.root.body.nodes):
        if node.form not in ("call", "batch"):
            continue
        key = descriptor_id(node.fn)
        members = node.n if node.form == "batch" else 1
        for label in sites:
            for m in ms:
                if m.template == key and m.label == label:
                    sites[label].append((k, rows[label], m))
                    rows[label] += members
    return [Layout(label, tuple(s), rows[label]) for label, s in sites.items()]


def rows_at(ls: Sequence[Layout], ranges: Sequence[tuple[int, str, int, int]], u: int) -> list[tuple[str, str, int]]:
    """Unit `u`'s ``(port, label, row)`` for each mark of its template, by label; `ranges` are the template query's
    ``(root node, template, first unit, units)``. None for a unit of no range."""
    r = next((r for r in ranges if r[2] <= u < r[2] + r[3]), None)
    if r is None:
        return []
    return [(m.port, lay.label, first + u - r[2]) for lay in ls for k, first, m in lay.sites if k == r[0]]


def value_at(certified: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Sequence[str]], label: str,
             g: int) -> tuple[str, str, int] | None:
    """Row `g` of `label` as ``(value, root, row)``, its values in order each taking its entry's `leaves` rows; None past
    the label's rows, for a label with no values, or at a value with no entry."""
    at = 0
    for v in labels.get(label, ()):
        if v not in certified:
            return None
        e = certified[v]
        if g < at + e["leaves"]:
            return v, e["root"], g - at
        at += e["leaves"]
    return None


def labels_held(ls: Sequence[Layout], certified: Mapping[str, Mapping[str, Any]],
                labels: Mapping[str, Sequence[str]]) -> bool:
    """The labels are the Program's, no value is under two labels or twice, each is a certified value and each certified
    value is under a label, and each label's values' `leaves` sum to its rows."""
    values = [v for vs in labels.values() for v in vs]
    return (sorted(labels) == [lay.label for lay in ls] and len(set(values)) == len(values)
            and all(v in certified for v in values) and all(v in values for v in certified)
            and all(sum(certified[v]["leaves"] for v in labels[lay.label]) == lay.rows for lay in ls))
