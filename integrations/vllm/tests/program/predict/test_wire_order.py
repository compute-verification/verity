"""The window circuit's wire-order certificate (`window_builder.order`, ``verity/wire-order/v1``), held to `check`: a Python
reference of the period check's closed form, refusing in the check's own words. A segment is template ``template``'s units
``[at, at + Π radix)``, ranked ``c + Σ_q r[q]·digit_q``; each record wire names its writer segment and its reader segment,
each side moves its segment's digits without a carry, and every writer unit ranks below its reader unit, so the wiring is
acyclic. Each class the tests lower is ordered on its own inference entries, a layer a digit of its own, and its function is
the one `FUNCTIONS` pins; a mutated certificate, a cyclic wiring and a wire the lowering made backward are refused by name.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import re
from itertools import product

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict import test_window_circuit as T
from tests.program.predict import test_window_circuit_fa3 as TF
from tests.program.predict import test_window_circuit_int8 as T8
from tests.program.predict import test_window_circuit_pearl as TP
from tests.program.predict import test_window_circuit_tables as TT
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import WindowClass
from verity_vllm.records import wire_order as WO

from verity.ml.windows import interface as IF

SEGMENT = ("template", "at", "radix", "c", "r")
#: each class the tests lower (its registered program, class, grain and tables: H1's, `test_window_circuit_tables`', or
#: Pearl+'s, `test_window_circuit_pearl`'s) and its function (`_function`): the digest of its description but its wires
#: and of its external I/O, with every wired reader row's writer row, and the wired rows. Placing wires in entries
#: (`window_builder.placed`) restates them and never changes what they wire.
FUNCTIONS = {
    "tiny": (S._registered, T.CLASS, T.GRAIN, None, 3264, "4348f7c1395f69e72ccb1896ddebabb786b40b144987ea6241a9f0e0e66fce7a130"
             "1cb291b201430cb7303872ce17425b8f39c1d6fb911ac96267151002b0586"),
    "slots": (S._registered, T.SLOTS, T.GRAIN, None, 13056, "a79b8d28bbdc4ee951ff66d294dad77a0b9e394818cafb203a19c5c769d11696e9"
              "3933dcee1d0ebb7c87982aae6fb74b1320ea7b850f9c7428613f4a87a625a4"),
    "int8": (T8._registered, T.CLASS, T8.GRAIN, None, 3824, "83d67739cb058f733b4c06aa6a2b4c93b4c6dca2d5b488261e92fa0b193c9ae019"
             "bbfc75926fc8c21bbbf01601df2e3520f1015f5cfecd99c87b6f65800a3b75"),
    "fa3": (TF._registered, T.CLASS, T.GRAIN, None, 3264, "72572d0bb227ea0abe3c239b7102d93715d9706d90852686c8836be541e808525822"
            "298406d0644916a5d65b9a974dc7b026636d0b3da757d38fd3262ba2441f"),
    "long": (S._registered, WindowClass(2, 8, 4), T.GRAIN, None, 8236, "4821b42fa44fb023c3385fa2dd763629515d4f364d29783256a85a"
             "2692fa82faf58796832d9ea007e36346e1ec34c8e1350a272d89573708953347746d02638c"),
    "h1-tables": (T8._registered, T.CLASS, T8.GRAIN, TT.TABLES, 3824, "1ddf3f3a38f117997bb721b8cd1e041d2609327340128f109cefece"
                  "2324d0fd3caae71d023c847cc443e676a8752833082eeb6f55844a77050d6abd5165cd43b"),
    "pearl": (TP._registered, T.CLASS, T.GRAIN, TP.PEARL, 1964, "a170c3dd445ed2b6fc062b1a89a2b838b39dfb7af5bd8fc3ee1130650b2c57"
              "9f24cad9ed469f8e02641576b4578968d8a483337773cd7e71ea72a6c3f06d7896"),
}


class Refused(ValueError):
    """The certificate doesn't order the record's wires: the period check's refusal, in its words."""


def _places(radix) -> list[int]:
    out, p = [], 1
    for r in reversed(radix):
        out.append(p)
        p *= r
    return out[::-1]


def _int(x, least: int | None = None) -> bool:
    return type(x) is int and (least is None or x >= least)


def check(record, order) -> list[int]:
    """Each of ``record``'s wires' greatest rank(writer) - rank(reader) under ``order``, every one below 0; else `Refused`,
    in the period check's order (`NetworkCheck.Circuit.orderOf`, then `orderWhy`).

    The certificate is read whole first: its keys and format, each segment a string template, a natural ``at``, an integer
    ``c``, and ``radix`` and ``r`` one integer each a digit with each radix at least 1, and each pair two naturals. Then
    the segments, in order: each names a template of the record and lies within its units; then one template's segments
    disjoint (the first segment overlapping a later one, with the first such); then one pair a record wire. Then wire by
    wire: both its segments are segments, of its writer's and its reader's templates, and each side lies in its segment,
    the writer's first: the writer side is offset ``base`` with digit k's stride its step, the reader side offset ``at``
    with digit k's stride the reader units' place of k, the cut digit's 0. A side's offset ``b`` lies in its segment, each
    moving digit (radix past 1, a stride t) takes the slowest position of radix past 1 whose place divides t, its multiple
    m = t / place there, and each position's digit of ``b`` moved by its digits' m·(radix - 1) stays in the position. Each
    side's rank is then ``K + Σ_k a_k·d_k`` over the wire's digits, so its greatest rank(writer) - rank(reader) is
    ``K_w - K_r + Σ_k max(0, (a_w,k - a_r,k)·(radix_k - 1))``."""
    units = {t: n for t, n in record["templates"]}
    wires = record["wires"]
    if not (isinstance(order, dict) and set(order) == {"format", "segments", "wires"} and order["format"] == WO.SCHEMA
            and isinstance(order["segments"], list) and isinstance(order["wires"], list)):
        raise Refused(f"not a {WO.SCHEMA} certificate")
    segs, pairs = order["segments"], order["wires"]
    for s, g in enumerate(segs):
        if not (isinstance(g, dict) and set(g) == set(SEGMENT) and isinstance(g["template"], str) and _int(g["at"], 0)
                and _int(g["c"])):
            raise Refused(f"segment {s}: not a segment of integers at and c")
        if not (isinstance(g["radix"], list) and isinstance(g["r"], list) and len(g["r"]) == len(g["radix"])
                and all(_int(x, 1) for x in g["radix"]) and all(_int(x) for x in g["r"])):
            raise Refused(f"segment {s}: radix and r are one integer each a digit, each radix at least 1")
    for i, pair in enumerate(pairs):
        if not (isinstance(pair, list) and len(pair) == 2 and all(_int(s, 0) for s in pair)):
            raise Refused(f"wire {i}: not a [writer segment, reader segment] pair")
    span = [(g["at"], g["at"] + math.prod(g["radix"])) for g in segs]
    for s, g in enumerate(segs):
        if g["template"] not in units:
            raise Refused(f"segment {s} names no template of the record")
        if span[s][1] > units[g["template"]]:
            raise Refused(f"segment {s}: units [{g['at']}, {span[s][1]}) outside {g['template']}'s {units[g['template']]}")
    by: dict[str, list[int]] = {}
    for s, g in enumerate(segs):
        by.setdefault(g["template"], []).append(s)
    first = []
    for ss in by.values():
        xs = sorted(span[s] for s in ss)
        if any(lo < hi for (_, hi), (lo, _) in zip(xs, xs[1:])):
            first.append(next((s, u) for k, s in enumerate(ss) for u in ss[k + 1:]
                              if span[s][0] < span[u][1] and span[u][0] < span[s][1]))
    if first:
        s, u = min(first)
        raise Refused(f"segments {s} and {u} of {segs[s]['template']} overlap")
    if len(pairs) != len(wires):
        raise Refused(f"{len(pairs)} wire pairs for {len(wires)} record wires")
    out = []
    for i, (w, pair) in enumerate(zip(wires, pairs)):
        if (bad := next((s for s in pair if s >= len(segs)), None)) is not None:
            raise Refused(f"wire {i}: segment {bad} is no segment")
        for side, tid, s in (("writer", w["writer"], pair[0]), ("reader", w["reader"], pair[1])):
            if segs[s]["template"] != tid:
                raise Refused(f"wire {i}: {side} segment {s} is template {segs[s]['template']}, not {tid}")
        radix, C = w["radix"], w.get("cut", [0, 1])[1]
        reader = _places(radix[:-1]) + [0] if C > 1 else _places(radix)
        K, a = [], []
        for (side, o, strides), s in zip((("writer", w["base"], w["step"]), ("reader", w["at"], reader)), pair):
            g = segs[s]
            R, P, b = g["radix"], _places(g["radix"]), o - g["at"]
            if not 0 <= b < math.prod(R):
                raise Refused(f"wire {i}: {side} offset {b} outside segment {s}")
            digits = [b // p % n for n, p in zip(R, P)]
            lo, hi, ak = list(digits), list(digits), []
            for k, (n, t) in enumerate(zip(radix, strides)):
                if n == 1 or t == 0:
                    ak.append(0)
                    continue
                q = next((q for q in range(len(R)) if R[q] > 1 and abs(t) % P[q] == 0), None)
                if q is None:
                    raise Refused(f"wire {i}: {side} digit {k} (stride {t}) on no position of segment {s}")
                m = t // P[q]
                lo[q], hi[q] = lo[q] + min(0, m * (n - 1)), hi[q] + max(0, m * (n - 1))
                ak.append(g["r"][q] * m)
            if (q := next((q for q in range(len(R)) if lo[q] < 0 or hi[q] > R[q] - 1), None)) is not None:
                raise Refused(f"wire {i}: {side} digit position {q} of segment {s} carries")
            K.append(g["c"] + sum(r * d for r, d in zip(g["r"], digits)))
            a.append(ak)
        reach = K[0] - K[1] + sum(max(0, (x - y) * (n - 1)) for x, y, n in zip(*a, radix))
        if reach >= 0:
            raise Refused(f"wire {i}: rank(writer) - rank(reader) reaches {reach} >= 0")
        out.append(reach)
    return out


def _units(w) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wire ``w`` unrolled: each wired row's reader unit, its row of the reader's port, and its writer unit."""
    c0, C = w.get("cut", [0, 1])
    i = np.arange(math.prod(w["radix"]), dtype=np.int64)
    src, j = np.full(i.shape, w["base"], dtype=np.int64), i.copy()
    for r, s in zip(reversed(w["radix"]), reversed(w["step"])):
        src, j = src + j % r * s, j // r
    return w["at"] + i // C, c0 + i % C, src


def unrolled(record, order) -> list[int]:
    """Each of ``record``'s wires' greatest rank(writer) - rank(reader) under ``order`` taken unit by unit, as `check`'s
    closed form must give it: every unit ranked ``c + Σ_q r[q]·digit_q`` in the segment holding it, and each wired row's
    reader unit and writer unit read off the wire (`_units`), each in some segment. ``order`` is one `check` has taken."""
    ranks = {t: np.zeros(n, dtype=np.int64) for t, n in record["templates"]}
    held = {t: np.zeros(n, dtype=bool) for t, n in record["templates"]}
    for g in order["segments"]:
        n = math.prod(g["radix"])
        i, r = np.arange(n, dtype=np.int64), np.full(n, g["c"], dtype=np.int64)
        for R, x in zip(reversed(g["radix"]), reversed(g["r"])):
            r, i = r + i % R * x, i // R
        ranks[g["template"]][g["at"]: g["at"] + n], held[g["template"]][g["at"]: g["at"] + n] = r, True
    out = []
    for k, w in enumerate(record["wires"]):
        unit, _, src = _units(w)
        assert held[w["writer"]][src].all() and held[w["reader"]][unit].all(), f"wire {k}: a unit in no segment"
        out.append(int((ranks[w["writer"]][src] - ranks[w["reader"]][unit]).max()))
    return out


def _function(desc, external) -> tuple[str, int]:
    """A circuit's function apart from how its wires state it: the SHA-512 of its description but its wires (its templates
    and whichever of cuts, registered reads and ingress it has) and of its external I/O ``external``, with every wired
    reader row ``(port, unit, row)``'s writer row ``(out, unit)`` in order, and the wired rows."""
    ports = sorted({(w["reader"], w["port"]) for w in desc["wires"]})
    outs = sorted({(w["writer"], w["out"]) for w in desc["wires"]})
    pi, oi, cols = {p: k for k, p in enumerate(ports)}, {o: k for k, o in enumerate(outs)}, []
    for w in desc["wires"]:
        unit, row, src = _units(w)
        cols.append(np.stack([np.full(unit.shape, pi[(w["reader"], w["port"])]), unit, row,
                              np.full(unit.shape, oi[(w["writer"], w["out"])]), src], axis=1))
    rows = np.concatenate(cols)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    assert not (np.diff(rows[:, :3], axis=0) == 0).all(axis=1).any(), "a reader row wired twice"
    h = hashlib.sha512(json.dumps({"external": list(external), **{k: v for k, v in desc.items() if k != "wires"}},
                                  sort_keys=True, separators=(",", ":")).encode())
    h.update(json.dumps([ports, outs], separators=(",", ":")).encode())
    h.update(np.ascontiguousarray(rows, dtype="<i8").tobytes())
    return h.hexdigest(), len(rows)


@functools.cache
def _lowered(name: str) -> tuple[WC.Lowered, dict]:
    registered, cls, grain, tables, _, _ = FUNCTIONS[name]
    low = WC.lower(registered(), cls, None, grain, provable=False, tables=tables)
    return low, json.loads(json.dumps(low.circuit.description()))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_each_classs_certificate_orders_its_wires_on_its_entries_its_function_unchanged(name):
    """Each class's certificate is its inference entries ranked, every wire inside one writer entry and one reader entry,
    and the check takes it; its function is the one `FUNCTIONS` pins."""
    (low, desc), (*_, rows, digest) = _lowered(name), FUNCTIONS[name]
    assert [{k: s[k] for k in ("template", "at", "radix")} for s in low.order["segments"]] == [
        {"template": e.template, "at": e.at, "radix": list(e.radix)} for e in low.circuit.blocks.inference]
    assert len(check(desc, json.loads(json.dumps(low.order)))) == len(desc["wires"])
    assert _function(desc, low.circuit.external) == (digest, rows)


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_unit_by_unit_each_writer_ranks_below_its_reader_so_a_pass_through_chain_ends(name):
    """Each class's certificate taken unit by unit (`unrolled`) ranks every wired row's writer unit below its reader unit,
    each wire's greatest difference the closed form's. Every edge between two units is a record wire, and a pass-through
    writer's output is a port of its own unit, so a chain through pass-through writers falls in rank at each wire it
    crosses: it never returns to a unit, and it ends."""
    low, desc = _lowered(name)
    order = json.loads(json.dumps(low.order))
    reach = check(desc, order)
    assert unrolled(desc, order) == reach and max(reach) < 0


def test_a_wire_read_one_writer_unit_on_is_another_function():
    """`_function` reads the wires and the external I/O: TINY's first wire read one writer unit on is another function of
    as many rows, and so is TINY with its external input and output swapped."""
    low, desc = _lowered("tiny")
    w = desc["wires"][0]
    moved = {**desc, "wires": [{**w, "base": w["base"] + 1}, *desc["wires"][1:]]}
    (digest, rows), (*_, n, pinned) = _function(moved, low.circuit.external), FUNCTIONS["tiny"]
    assert rows == n and digest != pinned
    assert _function(desc, low.circuit.external[::-1]) != (pinned, n)


def test_a_layer_is_a_digit_of_its_own_ranked_by_its_time():
    """An entry's rank is its units' time, affine in its digits: an int8 layer's two quantizers at K = 64 are one entry of
    the step, the layer, the quantizer and the slot, ranked a step, a layer and a quantizer apart and never a slot apart."""
    low, _ = _lowered("int8")
    (k,) = [k for k, e in enumerate(low.circuit.blocks.inference) if e.template == "Int8RowQuant_v3{K=64,R=1}"]
    e, s = low.circuit.blocks.inference[k], low.order["segments"][k]
    assert (e.radix, e.axes) == ((5, 2, 2, 2), ("step", "-", "-", "slot"))
    assert s["r"][3] == 0 and s["r"][0] > s["r"][1] * (2 - 1) + s["r"][2] * (2 - 1) and s["r"][1] > s["r"][2] > 0


def _mutated(name: str, how: str) -> tuple[dict, dict, str]:
    """``name``'s description, its certificate mutated ``how``, and the refusal it gets."""
    low, desc = _lowered(name)
    order, pairs = copy.deepcopy(low.order), low.order["wires"]
    reach = check(desc, order)
    if how == "rank":
        i = next(i for i, (s, u) in enumerate(pairs) if s != u)
        order["segments"][pairs[i][0]]["c"] += 10**6
        return desc, order, f"wire {i}: rank(writer) - rank(reader) reaches {reach[i] + 10**6} >= 0"
    if how == "reversed":
        i = next(i for i, w in enumerate(desc["wires"]) if w["writer"] != w["reader"])
        order["wires"][i] = pairs[i][::-1]
        return desc, order, (f"wire {i}: writer segment {pairs[i][1]} is template {desc['wires'][i]['reader']}, not "
                             f"{desc['wires'][i]['writer']}")
    if how == "dropped":
        order["wires"].pop()
        return desc, order, f"{len(pairs) - 1} wire pairs for {len(pairs)} record wires"
    raise ValueError(how)


@pytest.mark.parametrize("how", ["rank", "reversed", "dropped"])
def test_a_mutated_certificate_is_refused_by_name(how):
    """TINY's certificate with a writer segment ranked 10^6 later, a wire's segments swapped, or a wire's pair dropped."""
    desc, order, why = _mutated("tiny", how)
    with pytest.raises(Refused, match=f"^{re.escape(why)}$"):
        check(desc, order)


def _wire(reader: str, writer: str, at: int = 0, radix=(1,), base: int = 0, step=(0,), cut=(0, 1)) -> dict:
    return {"reader": reader, "port": "x", "writer": writer, "out": "o0", "at": at, "radix": list(radix), "base": base,
            "step": list(step), **({"cut": list(cut)} if cut != (0, 1) else {})}


def _order(*segments, wires=([0, 1],)) -> dict:
    return {"format": WO.SCHEMA, "segments": [dict(zip(SEGMENT, s)) for s in segments], "wires": [list(p) for p in wires]}


#: B's units 0..7 read A's, unit for unit
TOY = {"templates": [["A", 8], ["B", 8]], "wires": [_wire("B", "A", radix=(8,), step=(1,))]}
#: B's unit u reads A's units 2u and 2u + 1 as its port's two rows (the cut digit last)
CUT = {"templates": [["A", 8], ["B", 4]], "wires": [_wire("B", "A", radix=(4, 2), step=(2, 1), cut=(0, 2))]}
A, B = ("A", 0, [8], 0, [1]), ("B", 0, [8], 1, [1])


def test_a_toy_certificate_is_taken_at_the_tightest_ranks():
    """B's unit u ranked 1 + u after A's unit u ranked u reaches -1; with the cut digit, B's unit u reads A's 2u + 1, so B's
    ranks 2 + 2u reach -1 and 1 + 2u reaches 0, refused. Unit by unit (`unrolled`), the same."""
    assert check(TOY, _order(A, B)) == unrolled(TOY, _order(A, B)) == [-1]
    assert check(CUT, _order(A, ("B", 0, [4], 2, [2]))) == unrolled(CUT, _order(A, ("B", 0, [4], 2, [2]))) == [-1]
    assert unrolled(CUT, _order(A, ("B", 0, [4], 1, [2]))) == [0]
    with pytest.raises(Refused, match=re.escape("wire 0: rank(writer) - rank(reader) reaches 0 >= 0")):
        check(CUT, _order(A, ("B", 0, [4], 1, [2])))


@pytest.mark.parametrize("segments,wires,why", [
    ((("A", 0, [8], 1, [1]), B), ([0, 1],), "wire 0: rank(writer) - rank(reader) reaches 0 >= 0"),
    ((("A", 0, [8], 0, [2]), B), ([0, 1],), "wire 0: rank(writer) - rank(reader) reaches 6 >= 0"),
    ((("A", 0, [2, 4], 0, [4, 1]), B), ([0, 1],), "wire 0: writer digit position 1 of segment 0 carries"),
    ((("A", 0, [1], 0, [0]), B), ([0, 1],), "wire 0: writer digit 0 (stride 1) on no position of segment 0"),
    ((("A", 4, [4], 0, [1]), B), ([0, 1],), "wire 0: writer offset -4 outside segment 0"),
    ((A, ("B", 4, [4], 9, [1])), ([0, 1],), "wire 0: reader offset -4 outside segment 1"),
    ((A, B), ([0, 0],), "wire 0: reader segment 0 is template A, not B"),
    ((A, B), ([1, 0],), "wire 0: writer segment 1 is template B, not A"),
    ((A, B), ([0, 2],), "wire 0: segment 2 is no segment"),
    ((A, B), (), "0 wire pairs for 1 record wires"),
    ((("C", 0, [8], 0, [1]), B), ([0, 1],), "segment 0 names no template of the record"),
    ((("A", 0, [8], 0, [1, 0]), B), ([0, 1],), "segment 0: radix and r are one integer each a digit, each radix at least 1"),
    ((("A", 0, [8, 0], 0, [1, 0]), B), ([0, 1],), "segment 0: radix and r are one integer each a digit, each radix at least 1"),
    ((("A", 4, [8], 0, [1]), B), ([0, 1],), "segment 0: units [4, 12) outside A's 8"),
    ((A, B, ("A", 4, [2], 9, [0])), ([0, 1],), "segments 0 and 2 of A overlap"),
    ((("A", -1, [8], 0, [1]), B), ([0, 1],), "segment 0: not a segment of integers at and c"),
    (((5, 0, [8], 0, [1]), B), ([0, 1],), "segment 0: not a segment of integers at and c"),
    ((A, B), ([0, -1],), "wire 0: not a [writer segment, reader segment] pair"),
    ((("C", 0, [8], 0, [1]), B), ([0, 1, 1],), "wire 0: not a [writer segment, reader segment] pair"),
    ((("A", 4, [4], 0, [1]), ("A", 0, [2], 0, [1]), ("A", 1, [2], 0, [1]), ("A", 6, [1], 0, [0]), B), ([0, 4],),
     "segments 0 and 3 of A overlap"),
    ((("A", 4, [4], 0, [1]), B), ([0, 0],), "wire 0: reader segment 0 is template A, not B"),
])
def test_each_refusal_names_its_wire_or_segment(segments, wires, why):
    """Each refusal in the period check's words and order: the certificate read whole before a segment's template or a
    pair's count is checked, the first segment overlapping a later one, and both of a wire's templates before its sides."""
    with pytest.raises(Refused, match=f"^{re.escape(why)}$"):
        check(TOY, _order(*segments, wires=wires))


def test_a_certificate_of_other_keys_format_or_lists_is_none():
    for bad in ({}, {"format": WO.SCHEMA, "segments": {}, "wires": []}, {**_order(A, B), "format": "verity/wire-order/v0"},
                {**_order(A, B), "cut": 1}):
        with pytest.raises(Refused, match=f"^{re.escape(f'not a {WO.SCHEMA} certificate')}$"):
            check(TOY, bad)


def _looped(name: str) -> tuple[dict, dict, int]:
    """``name``'s description with its first uncut wire whose reader has an output as wide as a row of the wire's port
    made to read that output of its own reader unit, unit for unit: a cyclic wiring that `interface.check_wires` takes.
    With it, the class's certificate with the reader's segment as the wire's writer segment too, and the wire's index."""
    low, desc = _lowered(name)
    lays = low.circuit.layouts
    for i, w in enumerate(desc["wires"]):
        lay, k = lays[w["reader"]][0], low.circuit.cuts.get(w["reader"], {}).get(w["port"], 1)
        out = next((o for o, words in lay.outs if words == dict(lay.rows)[w["port"]] // k), None)
        if w["writer"] != w["reader"] and "cut" not in w and out is not None:
            wires = list(desc["wires"])
            wires[i] = {**w, "writer": w["reader"], "out": out, "base": w["at"], "step": _places(w["radix"])}
            IF.check_wires(lays, [IF.Wire.of(x) for x in wires], low.circuit.cuts)
            pairs = [list(p) for p in low.order["wires"]]
            pairs[i] = [pairs[i][1], pairs[i][1]]
            return {**desc, "wires": wires}, {**low.order, "wires": pairs}, i
    raise AssertionError(f"{name}: no wire to loop")


def test_a_cyclic_wiring_has_no_certificate():
    """A's unit reads B's and B's reads A's: the two wires' rank differences sum to 0, so one reaches 0 whatever the ranks
    (here every c of each in [-4, 4]); a unit reading its own row reaches 0 at any rank. TINY's record with one wire made
    to read its reader unit's own output (`_looped`), which every check of the wires but the order takes, is refused."""
    cyclic = {"templates": [["A", 1], ["B", 1]], "wires": [_wire("B", "A"), _wire("A", "B")]}
    for ca, cb in product(range(-4, 5), repeat=2):
        with pytest.raises(Refused, match=r"^wire [01]: rank\(writer\) - rank\(reader\) reaches \d+ >= 0$"):
            check(cyclic, _order(("A", 0, [1], ca, [0]), ("B", 0, [1], cb, [0]), wires=([0, 1], [1, 0])))
    for c in range(-4, 5):
        with pytest.raises(Refused, match=re.escape("wire 0: rank(writer) - rank(reader) reaches 0 >= 0")):
            check({"templates": [["A", 1]], "wires": [_wire("A", "A")]}, _order(("A", 0, [1], c, [0]), wires=([0, 0],)))
    looped, order, i = _looped("tiny")
    with pytest.raises(Refused, match=f"^{re.escape(f'wire {i}: rank(writer) - rank(reader) reaches 0 >= 0')}$"):
        check(looped, order)


def test_the_lowering_refuses_a_wire_whose_writer_is_made_after_its_reader():
    """`window_builder.order` refuses, naming it, a wire whose writer's units are timed after its reader's."""
    ents = [(IF.InferenceEntry("A", 0, (2,), ("slot",), 0), (1, 0), ((0, 0),)),
            (IF.InferenceEntry("B", 0, (2,), ("slot",), 0), (0, 0), ((0, 0),))]
    w = IF.Wire("B", "x", "A", "o0", at=0, radix=(2,), base=0, step=(1,))
    with pytest.raises(Unexpressible, match=re.escape(f"wire 0 {w.doc()}: rank(writer) - rank(reader) reaches 1 >= 0; its "
                                                      "writer isn't made before its reader")):
        WB.order([(w, 0, 1)], ents)


def _rows(wires) -> list[tuple]:
    return sorted((w.reader, w.port, w.at + i // w.cut[1], w.cut[0] + i % w.cut[1], w.writer, w.out, int(w.source(i)))
                  for w in wires for i in range(w.units))


def test_a_wire_is_cut_at_its_entries_and_refined_where_a_digit_spans_two_keeping_its_rows():
    """B's 12 units read A's 12, across A's entries [0, 8) (a slot digit under a ``-`` one) and [8, 12): the wire is cut at
    A's entries, and its part in the first refined from a digit of 8 into the entry's 2 and 4, the same rows."""
    ents = [IF.InferenceEntry("A", 0, (2, 4), ("-", "slot"), 0), IF.InferenceEntry("A", 8, (4,), ("slot",), 0),
            IF.InferenceEntry("B", 0, (12,), ("slot",), 0)]
    w = IF.Wire("B", "x", "A", "o0", at=0, radix=(12,), base=0, step=(1,))
    got = WB.placed([w], ents)
    assert got == [(IF.Wire("B", "x", "A", "o0", at=0, radix=(2, 4), base=0, step=(4, 1)), 0, 2),
                   (IF.Wire("B", "x", "A", "o0", at=8, radix=(4,), base=8, step=(1,)), 1, 2)]
    assert _rows([p for p, _, _ in got]) == _rows([w])
