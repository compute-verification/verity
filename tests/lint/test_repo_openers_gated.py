"""Every circuit template the circuit-check catalog reaches at its bindings that calls an opener (`OPENERS`) either gates
what it returns on the opener's verdict or returns a verdict alone, and each that does neither is an entry of
``allowlists/repo_openers_gated.json`` naming what consumes its verdict today; the list only shrinks.  With each opener's
verdict cut to 0 and every parameter unknown, a template's body is evaluated in three values (0, 1, unknown) with each
parameter's dependence carried beside its value: a returned element that comes out constant is gated, one that depends on
no parameter reaching an opener's opened value or path is a verdict, and any other is ungated.  A call that reaches no
opener on parameters all unknown is not entered: its returns are unknown and depend on all its parameters.  A template
the walk can't decide (a body that doesn't build or compile) fails.  The walk builds every body of the catalog, which takes
most of an hour: no cheaper test finds an opener, which is seen only in the bodies that call it."""
from __future__ import annotations

import heapq
import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / "repo_openers_gated.json"
KIND = "ungated"
FIELDS = ("file", "kind", "symbol", "detail")
OPENERS = {"FrameV3Read_v1": ("value", "path"), "FrameV3ReadAt_v1": ("value", "path"),
           "MerkleRead_v1": ("leaf", "path"), "LmsVerify_v1": ("message", "C", "y", "path")}
GATED, VERDICT, UNGATED = "gated", "verdict", "ungated"


def _imports() -> tuple[Any, Any, Any]:
    sys.path[:0] = [p for p in (str(ROOT / "tools/verity/circuit_check/src"),) if p not in sys.path]
    from circuit_check import bodies as BD
    from circuit_check import targets as T
    from verity.circuits import defs as D
    return BD, T, D


def _shape(v: Any, D: Any) -> str:
    if isinstance(v, D.SpecializedDefinition):
        return shape(v)
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return repr(v)
    if isinstance(v, int):
        return "0" if v == 0 else "n"
    if isinstance(v, float):
        return "f"
    if isinstance(v, (tuple, list)):
        return "(" + ",".join(_shape(x, D) for x in v) + ")"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_shape(x, D)}" for k, x in sorted(v.items())) + "}"
    return str(getattr(v, "id", repr(v)))


def shape(fn: Any) -> str:
    """A template's shape: its circuit template and its statics, each size taken only as zero or not."""
    D = _imports()[2]
    return fn.base_id + "{" + ",".join(f"{k}={_shape(v, D)}" for k, v in fn.bindings.items()) + "}"


def volume(v: Any) -> int:
    """The product of a binding's sizes, nested statics' included: the walk takes each shape at its least."""
    D = _imports()[2]
    if isinstance(v, D.SpecializedDefinition):
        return volume(list(v.bindings.values()))
    if isinstance(v, bool):
        return 1
    if isinstance(v, int):
        return max(v, 1)
    if isinstance(v, (tuple, list)):
        out = 1
        for x in v:
            out *= volume(x)
        return out
    if isinstance(v, dict):
        return volume(list(v.values()))
    return 1


def roots() -> list[Any]:
    """The catalog's roots as `targets.catalog` makes them, without building a model's body."""
    _BD, T, _D = _imports()
    out = list(T._registered())
    for units, make, _kind, _label in T.catalog_units():
        out += T._produce(units, make)
    return out


def walk(start: list[Any]) -> tuple[dict[str, list[tuple[int, Any]]], dict[str, set[str]], dict[str, str], list]:
    """Each shape reached from ``start``, with its bindings by size; each built shape's callee shapes; each that couldn't
    be opened, with why; and each body over 5 s to build.  Smallest binding first over the whole walk, one body per shape
    (an opener's none), each released once its callees are read."""
    _BD, _T, D = _imports()
    members: dict[str, list[tuple[int, Any]]] = {}
    callees: dict[str, set[str]] = {}
    undecided: dict[str, str] = {}
    slow: list[tuple[float, str]] = []
    seen: set[str] = set()
    heap: list[tuple[int, int, Any]] = []
    order = itertools.count()

    def push(fn: Any) -> None:
        if isinstance(fn, D.CompositeDefinition) and not fn.statics:
            fn = D.bind(fn)
        fid = getattr(fn, "id", None)
        if fid is None or fid in seen:
            return
        seen.add(fid)
        if isinstance(fn, D.SpecializedDefinition):
            heapq.heappush(heap, (volume(fn), next(order), fn))
        elif not isinstance(fn, D.PrimitiveDefinition) and type(fn).__name__ != "ScalarReference":
            undecided[fid] = f"a {type(fn).__name__}, neither a primitive gate nor a specialized template"

    for fn in start:
        push(fn)
    while heap:
        size, _, fn = heapq.heappop(heap)
        k = shape(fn)
        members.setdefault(k, []).append((size, fn))
        if k in callees or k in undecided:
            continue
        if fn.base_id in OPENERS:
            callees[k] = set()
            continue
        had, t = fn._body is not None, time.monotonic()
        try:
            kids = [n.fn for n in fn.body.nodes]
        except Exception as e:  # noqa: BLE001
            undecided[k] = f"{fn.id[:160]}: its body doesn't build: {type(e).__name__}: {str(e)[:160]}"
            continue
        finally:
            if not had:
                fn._body = None
        if (dt := time.monotonic() - t) > 5:
            slow.append((round(dt, 1), fn.id[:160]))
        callees[k] = {shape(c) for c in kids if isinstance(c, D.SpecializedDefinition)}
        for c in kids:
            push(c)
    return members, callees, undecided, slow


def reaching(members: dict[str, list[tuple[int, Any]]], callees: dict[str, set[str]]) -> set[str]:
    reach = {k for k, ms in members.items() if ms[0][1].base_id in OPENERS}
    changed = True
    while changed:
        changed = False
        for k, cs in callees.items():
            if k not in reach and cs & reach:
                reach.add(k)
                changed = True
    return reach


class Gating:
    """The three-valued run of one template over compiled bodies: a wire is (value, mask), value an int or None (unknown),
    mask the root's parameters it depends on, as a bitset."""

    def __init__(self, reach: set[str]) -> None:
        from verity.circuits import boolean as BL
        self.BD = _imports()[0]
        self.AND = BL.AND
        self.reach = reach
        self.memo: dict[tuple, tuple[tuple, tuple, int]] = {}
        self.raised: set[str] = set()

    def body(self, spec: Any) -> Any:
        return self.BD.compile_body(spec)

    def opener_mask(self, sub: Any, ws: list[int], M: list[int]) -> int:
        names = OPENERS[sub.spec.base_id]
        base, s = 0, 0
        for name, t in sub.spec.body.params:
            if name in names:
                for w in ws[base:base + t.leaves]:
                    s |= M[w]
            base += t.leaves
        return s

    def activation(self, sub: Any, vals: tuple, masks: tuple) -> tuple[tuple, tuple, int]:
        if sub.spec.base_id in OPENERS:
            return (0,) * len(sub.ret), (0,) * len(sub.ret), -1
        if shape(sub.spec) not in self.reach and sub.P and all(v is None for v in vals):
            m = 0
            for x in masks:
                m |= x
            return (None,) * len(sub.ret), (m,) * len(sub.ret), 0
        key = (sub.spec.id, vals, masks)
        hit = self.memo.get(key)
        if hit is None:
            hit = self.memo[key] = self.run(sub, list(vals), list(masks))
        return hit

    def run(self, body: Any, V: list, M: list) -> tuple[tuple, tuple, int]:
        BD = self.BD
        V.extend([None] * (body.size - body.P))
        M.extend([0] * (body.size - body.P))
        S = 0
        for plan in body.plans:
            kind = plan[0]
            if kind == "prim":
                _, prim, n, sa, _cols, _wide, out = plan
                fam = getattr(prim, "family", None)
                lazy = getattr(prim, "evaluate_lazy", None) is not None
                for m in range(n):
                    ws = BD._member(sa, m)
                    ops = [V[w] for w in ws]
                    mask = 0
                    for w in ws:
                        mask |= M[w]
                    val = None
                    if fam == "Input" or lazy:
                        val = None
                    elif all(o is not None for o in ops):
                        try:
                            val = prim.evaluate(*ops)
                        except Exception as e:  # noqa: BLE001
                            self.raised.add(f"{getattr(prim, 'id', prim)}: {type(e).__name__}")
                    elif prim is self.AND and any(o == 0 for o in ops):
                        val = 0
                    V[out + m] = val
                    M[out + m] = 0 if val is not None else mask
                continue
            if kind == "call":
                _, sub, ws, out = plan
                members = [(ws, out)]
            elif kind == "batch":
                _, sub, sa, n, out = plan
                R = len(sub.ret)
                members = [(BD._member(sa, m), out + m * R) for m in range(n)]
            else:
                _, sub, init, xs, shared, n, c, y, out = plan
                carry_v = [V[w] for w in init]
                carry_m = [M[w] for w in init]
                for m in range(n):
                    ws = [w for xw, L in xs for w in xw[m * L:(m + 1) * L]] + list(shared)
                    vals = tuple(carry_v + [V[w] for w in ws])
                    masks = tuple(carry_m + [M[w] for w in ws])
                    rv, rm, s = self.activation(sub, vals, masks)
                    S |= self.opener_mask(sub, list(init) + ws, M) if s == -1 else s
                    carry_v, carry_m = list(rv[:c]), list(rm[:c])
                    V[out + c + m * y:out + c + (m + 1) * y] = rv[c:]
                    M[out + c + m * y:out + c + (m + 1) * y] = rm[c:]
                V[out:out + c], M[out:out + c] = carry_v, carry_m
                continue
            for ws, o in members:
                rv, rm, s = self.activation(sub, tuple(V[w] for w in ws), tuple(M[w] for w in ws))
                S |= self.opener_mask(sub, ws, M) if s == -1 else s
                V[o:o + len(rv)] = rv
                M[o:o + len(rm)] = rm
        return tuple(V[w] for w in body.ret), tuple(M[w] for w in body.ret), S

    def classify(self, spec: Any) -> list[tuple[str, str]]:
        """(returned element, class) of each element of ``spec``'s return."""
        body = self.body(spec)
        params, ret = spec.signature()
        V: list = []
        M: list = []
        for i, (_, t) in enumerate(params):
            V += [None] * t.leaves
            M += [1 << i] * t.leaves
        rv, rm, S = self.run(body, V, M)
        fields = list(getattr(ret, "fields", (ret,)))
        out, lo = [], 0
        for k, t in enumerate(fields):
            vs, ms = rv[lo:lo + t.leaves], rm[lo:lo + t.leaves]
            lo += t.leaves
            if all(v is not None for v in vs):
                c = GATED
            elif not any(m & S for m in ms):
                c = VERDICT
            else:
                c = UNGATED
            out.append((str(k), c))
        return out


def parse(doc: object) -> dict[str, dict]:
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    out: dict[str, dict] = {}
    for e in doc["entries"]:
        assert isinstance(e, dict) and tuple(e) == FIELDS, f"{LIST.name}: an entry is {{{', '.join(FIELDS)}}}: {e}"
        assert e["kind"] == KIND, f"{LIST.name}: an entry's kind is {KIND!r}: {e}"
        assert all(isinstance(e[k], str) and e[k] for k in FIELDS), f"{LIST.name}: every field is a nonempty string: {e}"
        assert e["symbol"] not in out, f"{LIST.name}: duplicate entry {e}"
        out[e["symbol"]] = e
    return out


def allowed() -> dict[str, dict]:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def classes(start: list[Any] | None = None) -> tuple[dict[str, tuple[Any, list[tuple[str, str]]]], dict[str, str], dict]:
    """Each reaching shape's least binding with its classes, what couldn't be decided, and the walk's numbers.  A reaching
    shape's next binding, up to 64 times the least's size, is classified too and must agree."""
    BD = _imports()[0]
    t0 = time.monotonic()
    members, callees, undecided, slow = walk(roots() if start is None else start)
    built = time.monotonic() - t0
    reach = reaching(members, callees)
    G = Gating(reach)
    got: dict[str, tuple[Any, list[tuple[str, str]]]] = {}
    compared = 0
    for k in sorted(reach):
        ms = members[k]
        if ms[0][1].base_id in OPENERS:
            continue
        try:
            got[k] = (ms[0][1], G.classify(ms[0][1]))
            if len(ms) > 1 and ms[1][0] <= 64 * ms[0][0]:
                compared += 1
                if (other := G.classify(ms[1][1])) != got[k][1]:
                    undecided[k] = f"its classes change with its sizes: {ms[0][1].id[:120]} {got[k][1]}, {ms[1][1].id[:120]} {other}"
        except Exception as e:  # noqa: BLE001
            undecided[k] = f"its body doesn't compile: {type(e).__name__}: {str(e)[:160]}"
        finally:
            G.memo.clear()
            BD._BODIES.clear()
    stats = {"bindings": sum(len(ms) for ms in members.values()), "shapes": len(members), "built": len(callees),
             "reaching": len(reach), "openers": sorted(k for k in reach if members[k][0][1].base_id in OPENERS),
             "compared": compared, "build_s": round(built), "classify_s": round(time.monotonic() - t0 - built),
             "slowest_builds": sorted(slow, reverse=True)[:10], "unknown_where_evaluate_raised": sorted(G.raised)}
    return got, undecided, stats


def report(got: dict, undecided: dict, stats: dict) -> str:
    lines = [json.dumps(stats)]
    for _k, (fn, cs) in sorted(got.items()):
        worst = UNGATED if any(c == UNGATED for _, c in cs) else VERDICT if any(c == VERDICT for _, c in cs) else GATED
        lines.append(f"{worst:8} {fn.id[:220]}  [{', '.join(f'{k}:{c}' for k, c in cs)}]")
    lines += [f"UNDECIDED {k[:220]}: {v}" for k, v in sorted(undecided.items())]
    return "\n".join(lines)


@pytest.mark.slow
def test_no_new_ungated_opener() -> None:
    got, undecided, stats = classes()
    print(report(got, undecided, stats))
    listed = allowed()
    ungated = {fn.base_id for fn, cs in got.values() if any(c == UNGATED for _, c in cs)}
    new = sorted(ungated - set(listed))
    assert not undecided, ("templates the walk can't decide whether they reach an opener or gate its verdict: "
                           + "; ".join(f"{k[:160]}: {v}" for k, v in sorted(undecided.items())))
    assert not new, (f"circuit templates whose returned rows don't gate on their opener's verdict: {new}. Gate each returned "
                     f"row on the verdict (`EmbeddingRead_v4`'s shape), or return the verdict alone")
    stale = sorted(set(listed) - ungated)
    if stale:
        warnings.warn(f"{LIST.name}: no longer ungated, delete their entries: {stale}", stacklevel=1)


def test_the_list_is_well_formed() -> None:
    for e in allowed().values():
        assert (ROOT / e["file"]).is_file(), f"{LIST.name}: {e['file']} isn't a file"
    for bad in ({"entries": [{"file": "a.py", "kind": KIND, "symbol": "X_v1"}]},
                {"entries": [{"file": "a.py", "kind": "gated", "symbol": "X_v1", "detail": "d"}]}):
        with pytest.raises(AssertionError):
            parse(bad)


def test_a_gated_read_and_an_ungated_one() -> None:
    D = _imports()[2]
    from verity.commitments.gates import embedding as E
    v4, v2 = D.bind(E.EmbeddingRead_v4, V=2, H=1), D.bind(E.EmbeddingRead, V=2, H=1)
    got, undecided, stats = classes([v4, D.bind(E.EmbeddingRead, V=2, H=16), v2])
    assert not undecided and stats["compared"] == 1
    assert {fn.id: cs for fn, cs in got.values()} == {v4.id: [("0", GATED), ("1", GATED)], v2.id: [("0", UNGATED), ("1", VERDICT)]}
