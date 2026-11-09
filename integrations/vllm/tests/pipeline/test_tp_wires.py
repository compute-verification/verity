"""verity_vllm.pipeline.tp.wires: a tensor-parallel window's collective sites at node grain, each rank's combine parts cut W
ways and row p wired from rank p's producer output row.

The toy at TP2 (`test_steps`' TINY, `_mixed`'s first request: 45 collectives, each per row) is the compiled request. Each
unit's values are random words, a combine's parts rows its producers' outputs: the wiring check reads the committed strings
alone, never a unit's arithmetic."""
from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from tests.program.predict.test_steps import _mixed, _registered
from tests.program.test_tp_binding import N, rank_body
from verity_vllm.pipeline.tp import wires as TW
from verity_vllm.program import tp_binding as TB
from verity_vllm.program.predict.tp_steps import compile_tp

from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as I, wiring as WI

RUN, LAW, CAP, W = "ab" * 64, {"law": "stratified", "k": 2}, 4, 2


@pytest.fixture(scope="module")
def toy() -> list[TB.RequestBinding]:
    """The toy's first request at TP2: its binding."""
    return list(compile_tp(_registered(tp={"world": W, "rank": 0}), *_mixed()).bindings[:1])


def _fns(sites: TW.Sites) -> dict[str, object]:
    return {descriptor_id(fn): fn for fn, _ in sites.circuit.templates}


def _count(sites: TW.Sites) -> dict[str, int]:
    return I.input_count(I.descriptor_record(sites.circuit, CAP), _fns(sites).__getitem__)


def _values(tp: TW.Sites) -> dict[tuple, tuple[list[np.ndarray], list[np.ndarray]]]:
    """Each unit's input and output rows as u16 words, by unit key: random, a combine's parts its sites' producers' outputs in
    rank order (the toy's producers have one output row each)."""
    rng, fns, out = np.random.default_rng(0), _fns(tp), {}
    for key, (tid, _) in tp.units.items():
        lay = I.layout(fns[tid])
        rows = [rng.integers(0, 1 << 16, size=w, dtype=np.uint16) for _, w in lay.rows + lay.outs]
        out[key] = (rows[:len(lay.rows)], rows[len(lay.rows):])
    for (i, j, r, role) in tp.units:
        if role == TW.COMBINE:
            out[i, j, r, role][0][0] = np.concatenate([out[i, j, p, TW.PRODUCER][1][0] for p in range(W)])
    return out


def _window(sites: TW.Sites, values: dict, by_reference: bool) -> I.Window:
    """``sites``' window with every unit's ``values`` recorded, each combine's parts by reference (committed as its writers'
    rows: every row of them wired) with ``by_reference``, else as given."""
    fns, win = _fns(sites), I.Window(sites.circuit, RUN, LAW, CAP)
    order = {tid: [None] * n for tid, n in sites.circuit.units.items()}
    for key, (tid, u) in sites.units.items():
        order[tid][u] = key
    for tid, keys in order.items():
        lay = I.layout(fns[tid])

        def port(q: int, side: int, at: int, keys=keys, lay=lay) -> np.ndarray:
            w = lay.widths[q]
            return np.stack([values[k][side][at] for k in keys]).view(f"<u{w // 8}")

        combine = keys[0][3] == TW.COMBINE
        ins = [None if by_reference and combine and q == 0 else port(q, 0, q) for q in range(len(lay.rows))]
        win.record(fns[tid], ins, [port(len(lay.rows) + q, 1, q) for q in range(len(lay.outs))])
    return win


def _row(ms, publics: dict, tid: str, unit: int, row: str) -> tuple[object, int, slice]:
    """The member holding ``tid``'s unit ``unit``, that unit's global index, and where its row ``row``'s string sits in the
    member's public body."""
    base = min(m.base for m in ms if m.template == tid)
    m = next(m for m in ms if m.template == tid and m.base <= base + unit < m.base + m.n)
    names = [q for q, _ in m.all_rows]
    at = ((base + unit - m.base) * len(names) + names.index(row)) * WI.BC_BYTES
    assert len(publics[m.key]) == m.n * len(names) * WI.BC_BYTES
    return m, base + unit, slice(at, at + WI.BC_BYTES)


def test_every_peer_row_of_a_tp_window_is_wired_from_its_producers_output_row(toy):
    """The TP window holds both ranks' producers and combines, each combine's parts cut in two, and wires every parts row from
    its rank's producer: one wire covers a site's two reader ranks and two cuts (writer steps 0 and 1) where both ranks'
    producers share a template, one a cut where they don't (the embedding's shards). `input_count` counts every parts row
    wired; rank 0's and rank 1's windows alone wire only their own rows, each peer row free. The verifier derives the same
    members from the record."""
    tp = TW.sites_circuit(toy, "tp")
    combines = {tid for (*_, role), (tid, _) in tp.units.items() if role == TW.COMBINE}
    assert tp.circuit.cuts == {tid: {"parts": W} for tid in combines}
    docs = [w.doc() for w in tp.circuit.wires]
    assert any(d["writer"].startswith("Gemm_v2") and d["radix"][-2:] == [W, W] and d["step"][-2:] == [0, 1] for d in docs)
    assert {(d["writer"].split("START=")[1], tuple(d.get("cut", (0, 1)))) for d in docs if "Embedding" in d["writer"]} == {
        ("0}", (0, 1)), ("128}", (1, 1))}
    parts = sum(W * W * (s.hi - s.lo) * 16 for s in toy[0].sites)
    count, ranks = _count(tp), [_count(TW.sites_circuit(toy, f"rank{r}", ranks=[r])) for r in range(W)]
    assert count["wired"] == parts == 200704 and all(c["wired"] == parts // (W * W) for c in ranks)
    assert sum(c["free"] for c in ranks) - count["free"] == parts * (W - 1) // W
    rec = I.descriptor_record(tp.circuit, CAP)
    assert I.window_of_record({**rec, "run": RUN, "context": ""}, LAW, _fns(tp).__getitem__).members == I.members(tp.circuit, CAP)


def test_a_forged_peer_row_passes_the_per_rank_window_and_fails_the_tp_windows_wiring_check(toy):
    """The headline: a prover commits, as rank 0's peer row of an attention all-reduce, rank 1's partial with one bit flipped.
    In rank 0's window alone (today's per-rank form) the row is a free input: the window closes and its wiring check passes.
    In the TP window the honest prover refuses to commit it, and the same committed string in the TP window's public body is
    caught by the verifier's wiring check, and by `check_drawn` when the sample draws that combine, only then."""
    tp = TW.sites_circuit(toy, "tp")
    values = _values(tp)
    win = _window(tp, values, by_reference=True)
    got = win.close(salt_key=bytes(32))
    ms = win.window.members
    publics = {m.key: got.public_body(k) for k, m in enumerate(ms)}
    assert WI.check(ms, publics) == {} and WI.check_drawn(ms, publics, range(sum(m.n for m in ms))) == {}

    j = next(j for j, s in enumerate(toy[0].sites) if toy[0].fns[1][s.own[1][0][0]].id.startswith("Gemm_v2{K=32"))
    site = toy[0].sites[j]
    forged = dict(values)
    parts = values[0, j, 0, TW.COMBINE][0][0].copy()
    parts[site.hi - site.lo] ^= 1
    forged[0, j, 0, TW.COMBINE] = ([parts, *values[0, j, 0, TW.COMBINE][0][1:]], values[0, j, 0, TW.COMBINE][1])
    with pytest.raises(ValueError, match=r"a parts#1 row is not the o0 row of the Gemm_v2\{K=32,.* unit that wrote it"):
        _window(tp, forged, by_reference=False).close(salt_key=bytes(32))

    rank0 = TW.sites_circuit(toy, "rank0", ranks=[0])
    alone = _window(rank0, forged, by_reference=False)
    mine = alone.close(salt_key=bytes(32))
    ms0 = alone.window.members
    publics0 = {m.key: mine.public_body(k) for k, m in enumerate(ms0)}
    assert WI.check(ms0, publics0) == {}

    tid, u = tp.units[0, j, 0, TW.COMBINE]
    m, g, at = _row(ms, publics, tid, u, "parts#1")
    m0, _, at0 = _row(ms0, publics0, *rank0.units[0, j, 0, TW.COMBINE], "parts#1")
    body = bytearray(publics[m.key])
    body[at] = publics0[m0.key][at0]
    assert bytes(body[at]) != publics[m.key][at]
    bad = {**publics, m.key: bytes(body)}
    caught, drawn = WI.check(ms, bad), WI.check_drawn(ms, bad, [g])
    assert list(caught) == list(drawn) == [m.key] and "parts#1 strings are not the o0 strings" in caught[m.key]
    others = [x for x in range(sum(q.n for q in ms)) if not m.base <= x < m.base + m.n]
    assert WI.check_drawn(ms, bad, others) == {}
    print(json.dumps({"site": [site.k, site.lo, site.hi], "row": "rank 0's parts#1 (rank 1's partial, one bit flipped)",
                      "rank0_window": WI.check(ms0, publics0), "tp_window_check": caught, "tp_window_check_drawn": drawn,
                      "drawn_without_it": WI.check_drawn(ms, bad, others)}))


def test_a_collective_over_several_rows_is_refused_by_name():
    """A route all-reduce over both of a rank's rows (`_Peers.whole`, three ranks) binds, its own segment two producers' rows,
    and its window is refused by name: no one unit writes a rank's segment; the per-row collectives' window builds."""
    b = TB.derive([rank_body(3, r, whole=True) for r in range(3)])
    with pytest.raises(ValueError, match=r"w3: request 0's collective 1 \[0, 32\): rank 0's partial is 2 nodes' rows \[4, 5\], a "
                                         "collective over several rows, whose segment per rank no one unit writes at node grain"):
        TW.sites_circuit([b], "w3")
    one = TW.sites_circuit([TB.derive([rank_body(3, r) for r in range(3)])], "w3")
    assert _count(one)["wired"] == 2 * 9 * N * 16 and len(one.circuit.wires) == 1


def _closing(sites: TW.Sites) -> tuple[tuple[I.Wire, ...], str, str]:
    """``sites``' wires and one more closing a cycle: the first site whose rank 0 producer reads a row as wide as its combine's
    output reads its ``x`` from that output, which sums the producer's own partial. -> the wires, the producer's template and
    the combine's."""
    fns = _fns(sites)
    for (i, j, r, role), (ptid, pu) in sites.units.items():
        ctid, cu = sites.units[i, j, 0, TW.COMBINE]
        if role == TW.PRODUCER and r == 0 and dict(I.layout(fns[ptid]).rows).get("x") == dict(I.layout(fns[ctid]).outs)["o0"]:
            return (*sites.circuit.wires, I.Wire(ptid, "x", ctid, "o0", at=pu, radix=(1,), base=cu, step=(0,))), ptid, ctid
    raise AssertionError("no producer reads a row as wide as its combine's output")


@pytest.mark.parametrize("world", [W, 8])
def test_a_wire_closing_a_cycle_is_refused_and_the_tp_windows_wires_have_none(toy, world):
    """Units that read one another through wires pass every unit's check on a fixed point the prover chose (a combine
    summing a partial its producer computed from that combine's output), so `interface.check_wires` refuses wires whose
    dataflow between units has a cycle, naming a wire on it (`Spot.Wires.check_acyclic`). The TP window's wires,
    at TP2 (the toy) and TP8 (a Gemm partial a row, `rank_body`), have none and need no order. One wire more, rank 0's
    producer reading its ``x`` from its combine's output, is refused: with no order as a cycle of the two templates, under
    the honest wires' order as the first wire out of it, the forged one, and no order of single units holds it."""
    sites = TW.sites_circuit(toy, "tp") if world == W else \
        TW.sites_circuit([TB.derive([rank_body(world, r) for r in range(world)])], f"w{world}")
    c = sites.circuit
    lays = {descriptor_id(fn): (I.layout(fn), n) for fn, n in c.templates}
    assert c.order is None and len({r for _i, _j, r, _ in sites.units}) == world
    wires, ptid, ctid = _closing(sites)
    with pytest.raises(ValueError, match=r"cycle: wire \d+ .* is on a cycle of templates") as e:
        replace(c, wires=wires)
    named = wires[int(re.match(r"cycle: wire (\d+)", str(e.value))[1])]
    assert {named.reader, named.writer} == {ptid, ctid}
    honest = I.ordered(c.units, c.wires, {})
    I.check_wires(lays, c.wires, c.cuts, honest)
    with pytest.raises(ValueError, match=f"order: wire {len(wires) - 1} "):
        I.check_wires(lays, wires, c.cuts, honest)
    with pytest.raises(ValueError, match="cycle: no order of the ranges"):
        I.ordered(c.units, wires, {t: range(n) for t, n in c.units.items()})
