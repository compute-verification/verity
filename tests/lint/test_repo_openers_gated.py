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


def walk() -> tuple[dict[str, Any], dict[str, list[str]], dict[str, str]]:
    """Every specialized template and primitive gate the catalog reaches, by id; each template's callees; and each id
    that couldn't be opened, with why."""
    _BD, T, D = _imports()
    found: dict[str, Any] = {}
    callees: dict[str, list[str]] = {}
    undecided: dict[str, str] = {}
    stack = list(T.catalog())
    while stack:
        fn = stack.pop()
        if isinstance(fn, D.CompositeDefinition) and not fn.statics:
            fn = D.bind(fn)
        fid = getattr(fn, "id", None)
        if fid is None or fid in found:
            continue
        found[fid] = fn
        if isinstance(fn, D.SpecializedDefinition):
            try:
                kids = [n.fn for n in fn.body.nodes]
            except Exception as e:  # noqa: BLE001
                undecided[fid] = f"its body doesn't build: {type(e).__name__}: {str(e)[:160]}"
                continue
            stack.extend(kids)
            callees[fid] = sorted({getattr(k, "id", repr(k)) for k in kids})
        elif not isinstance(fn, D.PrimitiveDefinition) and type(fn).__name__ != "ScalarReference":
            undecided[fid] = f"a {type(fn).__name__}, neither a primitive gate nor a specialized template"
    return found, callees, undecided


def reaching(found: dict[str, Any], callees: dict[str, list[str]]) -> set[str]:
    reach = {i for i, fn in found.items() if getattr(fn, "base_id", None) in OPENERS}
    changed = True
    while changed:
        changed = False
        for i, cs in callees.items():
            if i not in reach and any(c in reach for c in cs):
                reach.add(i)
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
        if sub.spec.id not in self.reach and sub.P and all(v is None for v in vals):
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


def classes() -> tuple[dict[str, list[tuple[str, str]]], dict[str, str], dict[str, Any], dict]:
    t0 = time.monotonic()
    found, callees, undecided = walk()
    built = time.monotonic() - t0
    reach = reaching(found, callees)
    G = Gating(reach)
    got: dict[str, list[tuple[str, str]]] = {}
    for fid in sorted(reach):
        fn = found[fid]
        if fn.base_id in OPENERS:
            continue
        try:
            got[fid] = G.classify(fn)
        except Exception as e:  # noqa: BLE001
            undecided[fid] = f"its body doesn't compile: {type(e).__name__}: {str(e)[:160]}"
    stats = {"walked": len(found), "templates": sum(1 for f in found.values() if hasattr(f, "base_id")),
             "reaching": len(reach), "openers": sorted(i for i in reach if found[i].base_id in OPENERS),
             "build_s": round(built), "classify_s": round(time.monotonic() - t0 - built),
             "unknown_where_evaluate_raised": sorted(G.raised)}
    return got, undecided, found, stats


def report(got: dict, undecided: dict, found: dict, stats: dict) -> str:
    lines = [json.dumps(stats)]
    for fid, cs in sorted(got.items()):
        worst = UNGATED if any(c == UNGATED for _, c in cs) else VERDICT if any(c == VERDICT for _, c in cs) else GATED
        lines.append(f"{worst:8} {fid[:220]}  [{', '.join(f'{k}:{c}' for k, c in cs)}]")
    lines += [f"UNDECIDED {k[:220]}: {v}" for k, v in sorted(undecided.items())]
    return "\n".join(lines)


@pytest.mark.slow
def test_no_new_ungated_opener() -> None:
    got, undecided, found, stats = classes()
    print(report(got, undecided, found, stats))
    listed = allowed()
    ungated = {found[f].base_id for f, cs in got.items() if any(c == UNGATED for _, c in cs)}
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
    found = {}
    stack = [v4, v2]
    callees: dict[str, list[str]] = {}
    while stack:
        fn = stack.pop()
        if fn.id in found:
            continue
        found[fn.id] = fn
        if isinstance(fn, D.SpecializedDefinition):
            callees[fn.id] = sorted({n.fn.id for n in fn.body.nodes})
            stack.extend(n.fn for n in fn.body.nodes)
    G = Gating(reaching(found, callees))
    assert G.classify(v4) == [("0", GATED), ("1", GATED)]
    assert G.classify(v2) == [("0", UNGATED), ("1", VERDICT)]
