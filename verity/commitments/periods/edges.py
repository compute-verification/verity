"""The wires a Program's root states, read off its nodes (`Spot.Edges.edgesOf`): each argument part of a root node that
reads an earlier non-input root node's outputs is one wire per row of the reader's port, in closed form, or the first
refusal by name (`edge-parts`, `edge-template`, `edge-form`, `not-topological`, `edge-rows`, `edge-fields` where a part reads more than one output field of one writer, one field a
part being the rule, `edge-width`). The program
is a descriptor (`circuits/PROTOCOL.md` §4), the ranges ``[node, template, first, count]`` and the wiring the record's
(its ``templates`` and ``cuts``). The work is the root's nodes and arguments, then each part times its port's rows."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

INPUT_ID = re.compile(r"^(Registered)?Input[1-9][0-9]*_v1$")


class Refusal(Exception):
    pass


def _leaves(t) -> int:
    k = t[0]
    if k == "Value":
        return 1
    if k == "Array":
        return t[1] * _leaves(t[2])
    if k == "Tuple":
        return sum(_leaves(f) for f in t[1])
    return sum(_leaves(f) for _, f in t[1])


def _bits(t) -> int:
    k = t[0]
    if k == "Value":
        return t[1]
    if k == "Array":
        return t[1] * _bits(t[2])
    if k == "Tuple":
        return sum(_bits(f) for f in t[1])
    return sum(_bits(f) for _, f in t[1])


def _fields(t) -> list:
    if t[0] == "Tuple":
        return list(t[1])
    if t[0] == "Record":
        return [f for _, f in t[1]]
    return [t]


def _axis_dims(t, ax: int):
    if t[0] != "Array":
        return None
    return (t[1], _leaves(t[2])) if ax == 0 else _axis_dims(t[2], ax - 1)


def _parts(j, args: list) -> list[tuple]:
    """A sequence's flat parts: ``("e", items)`` or ``("g", run, lanes, count, written)``, each lane
    ``(p, i, base, e, ((radix, step), ...))`` (an affine or strided part is the one-lane grid it spells, `written`
    false), a concat spliced and an alias resolved to the earlier argument's parts (``args[k][a]``)."""
    if "alias" in j:
        k, a = j["alias"]
        return list(args[k][a])
    if "concat" in j:
        return [q for x in j["concat"] for q in _parts(x, args)]
    if "affine" in j:
        s, i, b, c = j["affine"]
        return [("g", max(c, 1), ((s == "p", i if c else 0, b, 1, ()),), c, False)]
    if "strided" in j:
        s, i, b, inner, os, is_, c = j["strided"]
        T = max(1, -(-c // inner))
        return [("g", inner, ((s == "p", i if c else 0, b, is_, ((T, os),) if T > 1 else ()),), c, False)]
    if "grid" in j:
        run, lanes = j["grid"]
        lanes = tuple((s == "p", i, b, e, tuple((r, d) for r, d in lv)) for s, i, b, e, lv in lanes)
        return [("g", run, lanes, len(lanes) * run * _periods(lanes[0]), True)]
    if "explicit" in j:
        return [("e", tuple((s == "p", i, leaf) for s, i, leaf in j["explicit"]))]
    raise ValueError(f"edge-parts: references {json.dumps(j)}: not affine, strided, grid, explicit, concat or alias")


def _periods(lane) -> int:
    t = 1
    for r, _d in lane[4]:
        t *= r
    return t


def _count(q) -> int:
    return q[3] if q[0] == "g" else len(q[1])


def _keys(q) -> list:
    if q[0] == "g":
        return [(q[2][0][0], q[2][0][1])]
    return [(q[1][0][0], q[1][0][1]) if q[1] else None]


def _arg_parts_ok(k: int, np_: int, parts: list) -> bool:
    if len(parts) > k + np_ + 2:
        return False
    return len(parts) <= len({x for q in parts for x in _keys(q)}) + 1


def _field_at(t, o: int):
    at = 0
    for f, ty in enumerate(_fields(t)):
        if at == o:
            return f, _leaves(ty), _bits(ty)
        at += _leaves(ty)
    return None


def _field_ix(t, o: int):
    at = 0
    for f, ty in enumerate(_fields(t)):
        if at <= o < at + _leaves(ty):
            return f
        at += _leaves(ty)
    return None


def _leaf_at(q, k: int):
    _, run, lanes, _c, _w = q
    g = len(lanes)
    if run == 0 or g == 0:
        return None
    t, r = divmod(k, g * run)
    j, o = divmod(r, run)
    p, i, b, e, lv = lanes[j]
    leaf = b + o * e
    for x, (rad, d) in enumerate(lv):
        t, dg = (0, t) if x == len(lv) - 1 else divmod(t, rad)
        leaf += dg * d
    return p, i, leaf


def _crosses_fields(R, q, k0: int, L: int, ln: int, cnt: int) -> bool:
    """Whether the rows `k0 + m·L` (`m < cnt`), `ln` references each, of the grid part `q` read more than one output
    field of one writer node, at a row's last reference or the next member's row."""
    first = _leaf_at(q, k0)
    if first is None or first[0] or first[2] < 0 or first[1] >= len(R.nodes):
        return False
    _, w, l0 = first
    wn = R.nodes[w]
    lw = wn["out_leaves"]
    if lw == 0:
        return False
    f0 = _field_ix(wn["ret"], l0 % lw)
    ks = [k0 + ln - 1] + ([k0 + L, k0 + L + ln - 1] if cnt > 1 else [])
    for k in ks:
        if k >= _count(q):
            continue
        x = _leaf_at(q, k)
        if x is not None and not x[0] and x[1] == w and x[2] >= 0 and _field_ix(wn["ret"], x[2] % lw) != f0:
            return True
    return False


def _members_in(x0: int, L: int, ln: int, M: int, s0: int, c: int) -> tuple[int, int]:
    if L == 0:
        over = x0 < s0 + c and s0 < x0 + ln
        inside = s0 <= x0 and x0 + ln <= s0 + c
        if over and not inside:
            raise Refusal("edge-rows")
        return 0, (M if inside else 0)
    hi = M - 1
    clo = max(-((x0 - s0) // L), 0)
    chi = min((s0 + c - ln - x0) // L, hi)
    olo = max((s0 - ln - x0) // L + 1, 0)
    ohi = min((s0 + c - 1 - x0) // L, hi)
    cn = chi - clo + 1 if clo <= chi else 0
    on = ohi - olo + 1 if olo <= ohi else 0
    if cn != on:
        raise Refusal("edge-rows")
    return clo, cn


def _grid_ok(R, k: int, q) -> None:
    """A grid's lanes count one number of periods (else `edge-parts`), and each lane over an earlier root node lies
    inside that node's outputs at both its extremes (else `edge-rows`), in O(levels) a lane."""
    _, run, lanes, _c, _w = q
    if len({_periods(x) for x in lanes}) != 1:
        raise Refusal("edge-parts")
    for p, i, b, e, lv in lanes:
        lo = hi = b
        for rad, d in ((run, e), *lv):
            lo, hi = lo + (rad - 1) * min(d, 0), hi + (rad - 1) * max(d, 0)
        if not p and i < k and (lo < 0 or hi >= R.nodes[i]["out_leaves"] * R.nodes[i]["members"]):
            raise Refusal("edge-rows")


def _lane_wire(q, k0: int, L: int, ln: int, n: int):
    """The grid part `q`'s rows `k0 + m·L`, `m < n`, each `ln` references inside one lane's run: the lane's collection,
    its first leaf, and the members' mixed radix and steps in leaves (outermost first); refused `edge-rows` where a row
    leaves a run or the members' rows are no mixed radix of the lane's levels."""
    _, run, lanes, _c, _w = q
    g = len(lanes)
    if run == 0 or g == 0:
        raise Refusal("edge-rows")
    t0, r = divmod(k0, g * run)
    j, o0 = divmod(r, run)
    p, i, b, e, lv = lanes[j]
    if p or o0 + ln > run or not (ln == 1 or e == 1):
        raise Refusal("edge-rows")
    digits, t = [], t0
    for x, (rad, _d) in enumerate(lv):
        t, dg = (0, t) if x == len(lv) - 1 else divmod(t, rad)
        digits.append(dg)
    leaf = b + o0 * e + sum(dg * d for dg, (_r, d) in zip(digits, lv))
    if leaf < 0:
        raise Refusal("edge-rows")
    if n <= 1 or L == 0:
        return i, leaf, [max(n, 1)], [0]
    if L % (g * run) != 0:
        if g == 1 and o0 + (n - 1) * L + ln <= run:
            return i, leaf, [n], [L * e]
        raise Refusal("edge-rows")
    dt, pre = L // (g * run), [1]
    for r, _d in lv:
        pre.append(pre[-1] * r)
    if lv and dt % pre[-2] == 0:
        return i, leaf, [n], [dt // pre[-2] * lv[-1][1]]
    if dt not in pre[:-1]:
        raise Refusal("edge-rows")
    a = pre.index(dt)
    h, P = a, 1
    while h < len(lv) and (t0 // dt) % (P * lv[h][0]) == 0 and n % (P * lv[h][0]) == 0:
        P, h = P * lv[h][0], h + 1
    c = n // P
    inner = [lv[x] for x in reversed(range(a, h))]
    if c == 1:
        levels = inner
    elif h < len(lv) and digits[h] + c <= lv[h][0]:
        levels = [(c, lv[h][1])] + inner
    else:
        raise Refusal("edge-rows")
    return i, leaf, [r for r, _d in levels] or [1], [d for _r, d in levels] or [0]


class _Root:
    def __init__(self, desc: Mapping[str, Any]) -> None:
        defs, callees, types = desc["definitions"], desc["callees"], desc["types"]
        body = defs[desc["root"]]["body"]
        self.params = len(body["params"])
        self.nodes: list[dict] = []
        args: list[list[list[tuple]]] = []
        for x in body["nodes"]:
            d = defs[callees[x["fn"]]]
            prim = d["form"] == "primitive"
            ps = d["params"] if prim else d["body"]["params"]
            ret = d["ret"] if prim else d["body"]["ret_type"]
            axes = x.get("axes", [])
            node_args = []
            for a, g in enumerate(x["args"]):
                ax = axes[a] if x["form"] == "batch" and a < len(axes) else None
                node_args.append({"parts": _parts(g["refs"], args),
                                  "dims": None if ax is None else _axis_dims(types[g["type"]], ax)})
            args.append([g["parts"] for g in node_args])
            self.nodes.append({"key": callees[x["fn"]], "input": prim and bool(INPUT_ID.match(callees[x["fn"]])),
                               "form": x["form"], "members": x["n"] if x["form"] in ("batch", "scan") else 1,
                               "params": ps, "ret": ret, "out_leaves": _leaves(ret), "args": node_args})


def _part_wires(R: _Root, fu: dict, templates: Mapping, cuts_of, k: int, i: int, q, s0: int) -> list[dict]:
    node = R.nodes[k]
    if k not in fu:
        raise Refusal("edge-template")
    rt, rb = fu[k]
    port, ty = node["params"][i]
    a = node["args"][i]
    cuts = cuts_of(rt, port)
    n = _leaves(ty)
    if cuts == 0 or n % cuts != 0 or _bits(ty) % cuts != 0:
        raise Refusal("edge-rows")
    ln = n // cuts
    out = []
    for j in range(cuts):
        t0 = j * ln
        if a["dims"] is None:
            x0, L = t0, 0
        else:
            d, lb = a["dims"]
            if lb == 0 or t0 % lb + ln > lb:
                raise Refusal("edge-rows")
            x0, L = (t0 // lb * d) * lb + t0 % lb, lb
        lo, cnt = _members_in(x0, L, ln, node["members"], s0, _count(q))
        if cnt == 0:
            continue
        if q[0] == "e":
            if any(p or r >= k or not R.nodes[r]["input"] for p, r, _ in q[1]):
                if any(not p and r >= k for p, r, _ in q[1]):
                    raise Refusal("not-topological")
                raise Refusal("edge-rows")
            continue
        if q[4]:
            _grid_ok(R, k, q)
        k0 = max(x0 + lo * L - s0, 0)
        rows = "edge-fields" if _crosses_fields(R, q, k0, L, ln, cnt) else "edge-rows"
        try:
            w, l0, radix, steps = _lane_wire(q, k0, L, ln, cnt)
        except Refusal as x:
            raise Refusal(rows if str(x) == "edge-rows" else str(x)) from None
        if len(radix) != len(steps) or 0 in radix:
            raise Refusal("edge-rows")
        if w >= k:
            raise Refusal("not-topological")
        wn = R.nodes[w]
        if wn["input"]:
            continue
        if wn["form"] == "scan":
            raise Refusal("edge-form")
        if w not in fu:
            raise Refusal("edge-template")
        wt, wb = fu[w]
        if rt not in templates or wt not in templates:
            raise Refusal("edge-template")
        lw = wn["out_leaves"]
        if lw == 0:
            raise Refusal("edge-rows")
        fa = _field_at(wn["ret"], l0 % lw)
        if fa is None:
            raise Refusal(rows)
        f, fl, fb = fa
        if fl != ln or any(r > 1 and d % lw != 0 for r, d in zip(radix, steps)):
            raise Refusal(rows)
        if fb != _bits(ty) // cuts:
            raise Refusal("edge-width")
        out.append({"reader": rt, "port": port, "writer": wt, "out": f"o{f}", "at": rb + lo, "radix": radix,
                    "base": wb + l0 // lw, "step": [d // lw for d in steps],
                    "cut": [j, 1]})
    return out


def edges(desc: Mapping[str, Any], ranges: Sequence[Sequence[Any]], wiring: Mapping[str, Any]) -> dict:
    """``{"wires": [...]}`` in the root's node, argument and part order, each in the record's wire form with its
    ``cut``; or ``{"refusal": code}``, the first refusal."""
    R = _Root(desc)
    templates, cuts = wiring["templates"], wiring["cuts"]

    def cuts_of(t: str, p: str) -> int:
        return cuts.get(t, {}).get(p, 1)

    fu: dict[int, tuple[str, int]] = {}
    acc: dict[str, int] = {}
    for node, t, _first, count in ranges:
        b = acc.get(t, 0)
        fu[node] = (t, b)
        acc[t] = b + count
    try:
        if not all(_arg_parts_ok(k, R.params, g["parts"]) for k, x in enumerate(R.nodes) for g in x["args"]):
            raise Refusal("edge-parts")
        out = []
        for k, x in enumerate(R.nodes):
            if x["input"]:
                continue
            if k not in fu or fu[k][0] not in templates:
                raise Refusal("edge-template")
            if x["form"] == "scan":
                raise Refusal("edge-form")
            for i, g in enumerate(x["args"]):
                s0 = 0
                for q in g["parts"]:
                    out += _part_wires(R, fu, templates, cuts_of, k, i, q, s0)
                    s0 += _count(q)
        return {"wires": out}
    except Refusal as e:
        return {"refusal": str(e)}
