"""Query language (IR §7): `/`, `/Name`, `//Name`, `//A/B`, `//A//B`, `//A ! groups(F, k)`.

Results are ordered collections of `FunctionCall`s (path queries) or `Region`s (`groups`), in canonical
call-preorder, zero-based. `groups(F, k)` is evaluated once per scope selected by the left side and
concatenated; a block whose calls are not adjacent is rejected (`GapError`), never absorbed or re-cut.

`parse(text)` turns the general structural query syntax of `verity.ml.circuits.query_ast` into its AST:
`members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1")`, `each(F, G)`, `concat(...)`, `rest()`, `default(32)`, `frontier(32)`,
`override(F, {"//Gemm_v1{*}": G})`, `zip(...)`, `where(F, spec=..., form=..., w_max=...)`, `union/minus/inter(F, G)`,
`root()`, `table(".", "owners")`, `table(path, name, of="gates")`, `group(F, "name")`, `group(F, ["base", "static:K"])`,
`members(path, sel, k=(r, c))`, `label(F, affine=(a, b, n))`, `label(F, table="dev")` (a tuple and a list are one value).  The legacy strings stay valid as sugar: `//A//B` -> `instances("//A//B")`, `/` -> `root()`,
`//A ! groups(F, k)` -> `members("//A", "@fn:F", k)` (k adjacent members of one node per set; the legacy `query()`
below is untouched and keeps its exact block semantics).
"""

from __future__ import annotations

import re
from typing import Iterator

from .query_ast import (Default, Each, Instances, Members, Nodes, NodeSel, Path, Query, QueryError, Rest, Root, Zip,
                      concat as _concat, frontier as _frontier, group as _group, inter as _inter, label as _label,
                      members as _members, minus as _minus, override as _override, table as _table, union as _union,
                      where as _where)
from .layout import FunctionCall, Circuit, Region


class GapError(QueryError):
    pass


_STEP = re.compile(r"(//|/)([A-Za-z_][A-Za-z0-9_]*)?")
_GROUPS = re.compile(r"^\s*groups\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*$")


def _parse_path(path: str) -> list[tuple[str, str | None]]:
    path = path.strip()
    if not path:
        raise QueryError("empty query")
    steps: list[tuple[str, str | None]] = []
    pos = 0
    while pos < len(path):
        m = _STEP.match(path, pos)
        if not m:
            raise QueryError(f"cannot parse query at {path[pos:]!r}")
        steps.append((m.group(1), m.group(2)))
        pos = m.end()
    if path == "/":
        return [("/", None)]
    for i, (op, name) in enumerate(steps):
        if name is None and not (i == 0 and len(steps) == 1):
            raise QueryError("a step needs a name")
    return steps


def _children(circuit: Circuit, scope_call: FunctionCall | None, name: str | None = None) -> Iterator[FunctionCall]:
    if scope_call is None:
        yield from circuit.root_calls(primitive_batches=name)
    else:
        yield from scope_call.children(primitive_batches=name)


def _descendants(circuit: Circuit, scope_call: FunctionCall | None, name: str | None = None) -> Iterator[FunctionCall]:
    for c in _children(circuit, scope_call, name):
        yield c
        if not c.is_primitive:
            yield from _descendants(circuit, c, name)


def query(circuit: Circuit, q: str) -> list:
    """Evaluate a query against the circuit. Returns a list of `FunctionCall` (path queries) or `Region`
    (`groups`) objects in canonical order."""
    if "!" in q:
        left, right = q.split("!", 1)
        m = _GROUPS.match(right)
        if not m:
            raise QueryError(f"unsupported right side {right.strip()!r} (only groups(F, k))")
        fname, k = m.group(1), int(m.group(2))
        if k <= 0:
            raise QueryError("groups(F, k) needs a positive k")
        scopes = _path(circuit, left)
        out: list[Region] = []
        for sc in scopes:
            out.extend(groups(circuit, sc, fname, k))
        return out
    return _path(circuit, q)


def _path(circuit: Circuit, path: str) -> list:
    steps = _parse_path(path)
    if steps == [("/", None)]:
        return [Region(circuit, 0, circuit.size)]
    # frontier: None stands for the root scope
    frontier: list[FunctionCall | None] = [None]
    for op, name in steps:
        nxt: list[FunctionCall] = []
        seen: set[tuple[int, int]] = set()
        for sc in frontier:
            it = _children(circuit, sc, name) if op == "/" else _descendants(circuit, sc, name)
            for c in it:
                if c.matches(name):  # type: ignore[arg-type]
                    key = (c.lo, c.hi)
                    if key in seen and op == "//":
                        continue  # the same call reached through two nested matching ancestors
                    seen.add(key)
                    nxt.append(c)
        nxt.sort(key=lambda c: (c.lo, -c.hi))
        frontier = list(nxt)
    return frontier  # type: ignore[return-value]


def groups(circuit: Circuit, scope: FunctionCall | Region | None, fname: str, k: int) -> list[Region]:
    """Within one scope: direct child calls to `fname` in canonical order, cut from the beginning into blocks
    of at most `k`; each block must be adjacent, nonempty, nonoverlapping; returns one region per block."""
    if k <= 0:
        raise QueryError("groups(F, k) needs a positive k")
    if isinstance(scope, Region) and not isinstance(scope, FunctionCall):
        if scope.lo == 0 and scope.hi == circuit.size:
            scope = None
        else:
            raise QueryError("groups scope must be a call (or the root)")
    calls = [c for c in _children(circuit, scope, fname) if c.matches(fname)]
    out: list[Region] = []
    for b in range(0, len(calls), k):
        block = calls[b:b + k]
        lo, hi = block[0].lo, block[0].hi
        for c in block[1:]:
            if c.lo != hi:
                raise GapError(f"groups({fname},{k}): calls {block[0]!r} .. {c!r} are not adjacent (gap at [{hi}:{c.lo}])")
            hi = c.hi
        out.append(Region(circuit, lo, hi))
    return out


# ---------------------------------------------------------------------------------------------------------
# `parse(text)`: the general structural query syntax -> `query_ast` AST (legacy strings are sugar)
# ---------------------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"""\s*(?:(?P<str>"[^"]*"|'[^']*')|(?P<num>-?\d+)|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<p>[(),:{}\[\]=]))""")


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if not m:
            raise QueryError(f"cannot tokenise query at {text[pos:pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)  # type: ignore[arg-type]
        if kind == "str":
            val = val[1:-1]
        out.append((kind, val))  # type: ignore[arg-type]
    return out


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = _tokens(text)
        self.i = 0

    def peek(self, k: int = 0):
        return self.toks[self.i + k] if self.i + k < len(self.toks) else (None, None)

    def take(self, kind: str | None = None, val: str | None = None):
        t = self.peek()
        if t[0] is None or (kind is not None and t[0] != kind) or (val is not None and t[1] != val):
            want = val or kind
            got = "end of query" if t[0] is None else repr(t[1])
            raise QueryError(f"expected {want!r}, got {got} (token {self.i})")
        self.i += 1
        return t

    def at(self, kind: str, val: str | None = None) -> bool:
        t = self.peek()
        return t[0] == kind and (val is None or t[1] == val)

    # -- values ----------------------------------------------------------------------------------------------
    def value(self):
        k, v = self.peek()
        if k == "str":
            self.i += 1
            return v
        if k == "num":
            self.i += 1
            return int(v)
        if k == "p" and v == "{":
            return self.dict()
        if k == "p" and v == "[":
            return self.list()
        if k == "p" and v == "(":
            return self.tuple_()
        if k == "id":
            if v in ("True", "False", "None") and not (self.peek(1)[0] == "p" and self.peek(1)[1] == "("):
                self.i += 1
                return {"True": True, "False": False, "None": None}[v]
            return self.call()
        if k is None:
            raise QueryError("unexpected end of query (unbalanced parentheses?)")
        raise QueryError(f"unexpected token {v!r} (token {self.i})")

    def dict(self) -> dict:
        self.take("p", "{")
        out: dict = {}
        while not self.at("p", "}"):
            key = self.take("str")[1]
            self.take("p", ":")
            out[key] = self.value()
            if self.at("p", ","):
                self.i += 1
        self.take("p", "}")
        return out

    def list(self) -> list:
        self.take("p", "[")
        out: list = []
        while not self.at("p", "]"):
            out.append(self.value())
            if self.at("p", ","):
                self.i += 1
        self.take("p", "]")
        return out

    def tuple_(self) -> tuple:
        self.take("p", "(")
        out: list = []
        while not self.at("p", ")"):
            out.append(self.value())
            if self.at("p", ","):
                self.i += 1
        self.take("p", ")")
        return tuple(out)

    def call(self) -> Query:
        name = self.take("id")[1]
        self.take("p", "(")
        args: list = []
        kwargs: dict = {}
        while not self.at("p", ")"):
            if self.at("id") and self.peek(1) == ("p", "="):
                key = self.take("id")[1]
                self.take("p", "=")
                kwargs[key] = self.value()
            else:
                if kwargs:
                    raise QueryError("positional argument after a keyword argument")
                args.append(self.value())
            if self.at("p", ","):
                self.i += 1
        self.take("p", ")")
        return _construct(name, args, kwargs)


def _q(x) -> Query:
    if isinstance(x, Query):
        return x
    if isinstance(x, str):
        return _legacy(x)
    raise QueryError(f"a query argument must be a query or a path string, got {x!r}")


def _int_arg(name: str, what: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise QueryError(f"{name}(...): {what} must be an integer, got {v!r}")
    return v


def _construct(name: str, args: list, kw: dict) -> Query:
    def need(n: int) -> None:
        if len(args) < n:
            raise QueryError(f"{name}(...) needs {n} positional argument(s), got {len(args)}")

    if name == "instances":
        need(1)
        return Instances(Path.parse(args[0]))
    if name == "members":
        need(2)
        k = args[2] if len(args) > 2 else kw.get("k", 1)
        return _members(Path.parse(args[0]), NodeSel.parse(args[1]), k if isinstance(k, (list, tuple)) else _int_arg(name, "k", k))
    if name == "nodes":
        need(2)
        return Nodes(Path.parse(args[0]), NodeSel.parse(args[1]))
    if name == "zip":
        need(1)
        if len(args) == 1:  # zip(sel) / zip(sel, by="member"): the enclosing body
            path, sels = ".", [args[0]]
        else:
            path, sels = args[0], (args[1] if isinstance(args[1], list) else [args[1]])
        return Zip(Path.parse(path), tuple(NodeSel.parse(s) for s in sels))
    if name == "each":
        need(2)
        return Each(_q(args[0]), _q(args[1]))
    if name == "concat":
        return _concat(*[_q(a) for a in args])
    if name == "rest":
        return Rest(_q(args[0]) if args else None)
    if name == "default":
        w = args[0] if args else kw.get("w")
        if w is None:
            raise QueryError("default(w) needs a width")
        return Default(_int_arg(name, "w", w))
    if name == "frontier":
        need(1)
        return _frontier(args[0])
    if name == "override":
        need(2)
        if not isinstance(args[1], dict):
            raise QueryError("override(F, {path: G}) needs a dict of cases")
        return _override(_q(args[0]), {p: _q(g) for p, g in args[1].items()})
    if name == "where":
        need(1)
        return _where(_q(args[0]), spec=kw.get("spec"), form=kw.get("form"), w_max=kw.get("w_max"), w_min=kw.get("w_min"))
    if name == "union":
        need(2)
        return _union(_q(args[0]), _q(args[1]))
    if name == "minus":
        need(2)
        return _minus(_q(args[0]), _q(args[1]))
    if name == "inter":
        need(2)
        return _inter(_q(args[0]), _q(args[1]))
    if name == "root":
        return Root()
    if name == "table":
        need(2)
        if not isinstance(args[1], str):
            raise QueryError(f"table(path, name): the name is a string, got {args[1]!r}")
        return _table(Path.parse(args[0]), args[1], args[2] if len(args) > 2 else kw.get("of", "nodes"))
    if name == "group":
        need(2)
        if not isinstance(args[1], (str, list)):
            raise QueryError(f"group(F, key): the key is a key part or a list of them, got {args[1]!r}")
        return _group(_q(args[0]), args[1])
    if name == "label":
        need(1)
        if len(args) > 1 or set(kw) - {"affine", "table"}:
            raise QueryError("label(F, affine=[a, b, n]) or label(F, table=NAME): the map is one keyword argument")
        return _label(_q(args[0]), affine=kw.get("affine"), table=kw.get("table"))
    raise QueryError(f"unknown query constructor {name!r}")


def _legacy(text: str) -> Query:
    """Legacy path strings as sugar: `/` -> root(); `//A//B` -> instances; `L ! groups(F, k)` -> members(L, @fn:F, k)."""
    t = text.strip()
    if "!" in t:
        left, right = t.split("!", 1)
        m = _GROUPS.match(right)
        if not m:
            raise QueryError(f"unsupported right side {right.strip()!r} (only groups(F, k))")
        fname, k = m.group(1), int(m.group(2))
        if k <= 0:
            raise QueryError("groups(F, k) needs a positive k")
        left = left.strip()
        path = Path.parse("." if left == "/" else left)
        return Members(path, NodeSel("fn", fname), k)
    if t == "/":
        return Root()
    return Instances(Path.parse(t))


def parse(text: str) -> Query:
    """Parse the general query syntax (or a legacy path string) into a `query_ast` Query."""
    t = text.strip()
    if not t:
        raise QueryError("empty query")
    if t[0] in "/@.":
        return _legacy(t)
    p = _Parser(t)
    q = p.call()
    if p.i != len(p.toks):
        raise QueryError(f"trailing tokens after the query: {p.toks[p.i:]!r}")
    return q
