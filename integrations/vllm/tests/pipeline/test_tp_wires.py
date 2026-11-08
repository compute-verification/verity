"""verity_vllm.pipeline.tp.wires: a tensor-parallel window's collective sites at their producers' grain, each combine unit's
parts cut W ways and row p wired from rank p's producer unit, and each read of a combine's output wired from the units that
compute it.

The toy at TP2 (`test_steps`' TINY, `_mixed`'s first request: 45 collectives, each per row) is the compiled request: its
GEMM and embedding partials are coordinate units, so its all-reduces are coordinate combines, and its logits' all-gather
has none, the sampler reading every rank's LM-head coordinates. Each unit's values are random words, each wired row its
writer's output row: the wiring check reads the committed strings alone, never a unit's arithmetic."""
from __future__ import annotations

import bisect
import json

import numpy as np
import pytest

from tests.program.predict.test_steps import _mixed, _registered
from tests.program.test_tp_binding import N, ROWS, rank_body
from verity_vllm.pipeline.tp import wires as TW
from verity_vllm.program import tp_binding as TB
from verity_vllm.program.predict.tp_steps import compile_tp

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import moe
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


def _bits(sites: TW.Sites, wires=None) -> int:
    """The bits ``wires`` (every wire of ``sites``) fill, as `input_count` counts them."""
    cuts = sites.circuit.cuts
    return sum(w.units * 16 * dict(I.layout(_fns(sites)[w.reader]).rows)[w.port] // cuts.get(w.reader, {}).get(w.port, 1)
               for w in (sites.circuit.wires if wires is None else wires))


def _stage(sites: TW.Sites, stage: str) -> set[str]:
    return {tid for (_i, _j, s, _r), rs in sites.units.items() if s == stage for tid, _a, _n in rs}


def peer_bits(sites: TW.Sites) -> int:
    """The bits of the wires from producer units: every rank's partial into each combine unit's parts, or into a gather's
    readers."""
    return _bits(sites, [w for w in sites.circuit.wires if w.writer in _stage(sites, TW.PRODUCER)])


def in_order(sites: TW.Sites, wires: tuple[I.Wire, ...] | None = None) -> None:
    """Every wire of ``sites`` (or ``wires``) runs forward in ``sites.order``, as proofs' order check reads an order of unit
    ranges: the ranges tile each template's units, and for every range a wire's reader units meet, every range meeting the
    bounds on the writers it reads there (digit by digit, the fastest first) ranks below it. Refused (AssertionError) naming
    the wire."""
    ranges: dict[str, list[tuple[int, int, int]]] = {}
    for k, (tid, at, n) in enumerate(sites.order):
        ranges.setdefault(tid, []).append((at, n, k))
    for tid, n in sites.circuit.units.items():
        got = sorted(ranges[tid])
        assert [a for a, _, _ in got] == [0, *np.cumsum([m for _, m, _ in got])[:-1].tolist()] and sum(m for _, m, _ in got) == n
        ranges[tid] = got

    def box(w: I.Wire, lo: int, hi: int) -> tuple[int, int]:
        a = b = w.base
        for k in range(len(w.radix) - 1, -1, -1):
            r, s = w.radix[k], w.step[k]
            if lo // r == hi // r:
                q, rest = lo // r, 0
                for j in range(k - 1, -1, -1):
                    rest += q % w.radix[j] * w.step[j]
                    q //= w.radix[j]
                return a + min(lo % r * s, hi % r * s) + rest, b + max(lo % r * s, hi % r * s) + rest
            a, b, lo, hi = a + min(0, s * (r - 1)), b + max(0, s * (r - 1)), lo // r, hi // r
        return a, b

    def meeting(tid: str, lo: int, hi: int) -> list[tuple[int, int, int]]:
        got = ranges[tid]
        k = bisect.bisect_right([a for a, _, _ in got], lo) - 1
        out = []
        while k < len(got) and got[k][0] <= hi:
            out.append(got[k])
            k += 1
        return out

    for x, w in enumerate(sites.circuit.wires if wires is None else wires):
        C = w.cut[1]
        for at, n, k in meeting(w.reader, w.at, w.at + w.reader_units - 1):
            lo, hi = (max(at, w.at) - w.at) * C, (min(at + n, w.at + w.reader_units) - w.at) * C - 1
            a, b = box(w, lo, hi)
            late = [kk for _a, _n, kk in meeting(w.writer, a, b) if kk >= k]
            assert not late, f"wire {x} {w.doc()}: its reader range {k} reads writer ranges {late}, not below it"


def _values(sites: TW.Sites, seed: int = 0) -> dict[str, tuple[dict, dict]]:
    """Every unit's input and output rows as u16 words, by template (by port, by output): random, each wired row its writer's
    output row."""
    rng, fns, out = np.random.default_rng(seed), _fns(sites), {}
    for tid, n in sites.circuit.units.items():
        lay = I.layout(fns[tid])
        out[tid] = ({p: rng.integers(0, 1 << 16, (n, w), dtype=np.uint16) for p, w in lay.rows},
                    {o: rng.integers(0, 1 << 16, (n, w), dtype=np.uint16) for o, w in lay.outs})
    for w in sites.circuit.wires:
        dst, k = out[w.reader][0][w.port], sites.circuit.cuts.get(w.reader, {}).get(w.port, 1)
        i = np.arange(w.units)
        dst.reshape(len(dst), k, -1)[w.at + i // w.cut[1], w.cut[0] + i % w.cut[1]] = out[w.writer][1][w.out][w.source(i)]
    return out


def _window(sites: TW.Sites, values: dict) -> I.Window:
    """``sites``' window with every unit's ``values`` recorded."""
    win = I.Window(sites.circuit, RUN, LAW, CAP)
    for tid, fn in _fns(sites).items():
        lay, (ins, outs) = I.layout(fn), values[tid]
        view = [f"<u{b // 8}" for b in lay.widths]
        win.record(fn, [ins[p].view(view[q]) for q, (p, _) in enumerate(lay.rows)],
                   [outs[o].view(view[len(lay.rows) + q]) for q, (o, _) in enumerate(lay.outs)])
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


def test_every_peer_row_of_a_tp_window_is_wired_from_its_producer_units_output_row(toy):
    """The TP window holds both ranks' producer units, coordinate combines and the calls reading them. Every combine unit's
    parts are cut in two, each row its rank's producer unit's output word; every read of a combine's output (the norms' x
    and residual, the sampler's logits from both ranks' LM-head coordinates) is wired, its port cut a word a row, and none is
    left free. `input_count` counts every peer bit wired; rank 0's and rank 1's windows alone wire only their own rows, each
    peer row free. Every wire runs forward in the window's public order, and the verifier derives the same members from
    the record."""
    tp = TW.sites_circuit(toy, "tp")
    combines, fns = _stage(tp, TW.COMBINE), _fns(tp)
    assert tp.unwired == () and {fns[t].id for t in combines} == {"AllReduce2_v1{N=1}"}
    assert {t: tp.circuit.cuts[t] for t in combines} == {t: {"parts": W} for t in combines}
    assert {fns[t].id: c for t, c in tp.circuit.cuts.items() if t not in combines} == {
        "RMSNormTriton_v5{N=64,EPS=1e-05}": {"x": 64}, "RMSNormFusedCuda_v2{N=64,EPS=1e-05}": {"x": 64, "res": 64},
        "TokenSelect_v4{V=256}": {"logits": 256}}
    parts = sum(W * W * (s.hi - s.lo) * 16 for s in toy[0].sites)
    count, ranks = _count(tp), [TW.sites_circuit(toy, f"rank{r}", ranks=[r]) for r in range(W)]
    assert peer_bits(tp) == parts == 200704 and all(peer_bits(x) == parts // (W * W) for x in ranks)
    assert _bits(tp, [w for w in tp.circuit.wires if w.reader in combines]) == sum(
        tp.circuit.units[t] * 16 * W for t in combines)
    assert count["wired"] == _bits(tp) and sum(_count(x)["free"] for x in ranks) - count["free"] == parts * (W - 1) // W
    in_order(tp)
    rec = I.descriptor_record(tp.circuit, CAP)
    assert I.window_of_record({**rec, "run": RUN, "context": ""}, LAW, fns.__getitem__).members == I.members(tp.circuit, CAP)


def test_a_forged_peer_row_passes_the_per_rank_window_and_fails_the_tp_windows_wiring_check(toy):
    """The headline: a prover commits, as rank 0's peer row of one coordinate of an attention all-reduce, rank 1's partial
    there with one bit flipped. In rank 0's window alone (today's per-rank form) the row is a free input: the window closes
    and its wiring check passes. In the TP window the honest prover refuses to commit it, and the same committed string in
    the TP window's public body is caught by the verifier's wiring check, and by `check_drawn` when the sample draws that
    combine unit, only then."""
    tp = TW.sites_circuit(toy, "tp")
    values = _values(tp)
    win = _window(tp, values)
    got = win.close(salt_key=bytes(32))
    ms = win.window.members
    publics = {m.key: got.public_body(k) for k, m in enumerate(ms)}
    assert WI.check(ms, publics) == {} and WI.check_drawn(ms, publics, range(sum(m.n for m in ms))) == {}

    j = next(j for j, s in enumerate(toy[0].sites) if toy[0].fns[1][s.own[1][0][0]].id.startswith("Gemm_v2{K=32"))
    site, q = toy[0].sites[j], 5
    (tid, at, _), = tp.units[0, j, TW.COMBINE, 0]
    forged = {t: ({p: a.copy() for p, a in ins.items()}, outs) for t, (ins, outs) in values.items()}
    forged[tid][0]["parts"][at + q, 1] ^= 1
    with pytest.raises(ValueError, match=r"a parts#1 row is not the o0 row of the Gemm_v2\{K=32,N=1,.* unit that wrote it"):
        _window(tp, forged).close(salt_key=bytes(32))

    rank0 = TW.sites_circuit(toy, "rank0", ranks=[0])
    mine = _values(rank0)
    (tid0, at0, _), = rank0.units[0, j, TW.COMBINE, 0]
    mine[tid0][0]["parts"][at0 + q, 1] = forged[tid][0]["parts"][at + q, 1]
    alone = _window(rank0, mine)
    closed = alone.close(salt_key=bytes(32))
    ms0 = alone.window.members
    publics0 = {m.key: closed.public_body(k) for k, m in enumerate(ms0)}
    assert WI.check(ms0, publics0) == {}

    m, g, cell = _row(ms, publics, tid, at + q, "parts#1")
    m0, _, cell0 = _row(ms0, publics0, tid0, at0 + q, "parts#1")
    body = bytearray(publics[m.key])
    body[cell] = publics0[m0.key][cell0]
    assert bytes(body[cell]) != publics[m.key][cell]
    bad = {**publics, m.key: bytes(body)}
    caught, drawn = WI.check(ms, bad), WI.check_drawn(ms, bad, [g])
    assert list(caught) == list(drawn) == [m.key] and "parts#1 strings are not the o0 strings" in caught[m.key]
    others = [x for x in range(sum(m.n for m in ms)) if not m.base <= x < m.base + m.n]
    assert WI.check_drawn(ms, bad, others) == {}
    print(json.dumps({"site": [site.k, site.lo, site.hi], "coordinate": q,
                      "row": "rank 0's parts#1 (rank 1's partial at the coordinate, one bit flipped)",
                      "rank0_window": WI.check(ms0, publics0), "tp_window_check": caught, "tp_window_check_drawn": drawn,
                      "drawn_without_it": WI.check_drawn(ms, bad, others)}))


def test_a_collective_over_several_rows_is_a_coordinate_combine_a_word():
    """A route all-reduce over both of a rank's rows (`_Peers.whole`, three ranks), its own segment two producers' rows, is
    one coordinate combine a word of the two rows: every parts row of it and of the per-row collectives is wired, from the
    Gemm coordinate of the rank and row it names, and every wire runs forward."""
    b = TB.derive([rank_body(3, r, whole=True) for r in range(3)])
    sites = TW.sites_circuit([b], "w3")
    assert [len(s.own[0]) for s in b.sites] == [1, 1, 2] and sites.unwired == ()
    assert {(_fns(sites)[t].id, n) for (_i, j, s, _r), rs in sites.units.items() if s == TW.COMBINE and j == 2
            for t, _a, n in rs} == {("AllReduce_v2{WORLD=3,N=1}", ROWS * N)}
    assert peer_bits(sites) == _count(sites)["wired"] == sum(3 * 3 * (s.hi - s.lo) * 16 for s in b.sites) == 9 * 2 * ROWS * N * 16
    in_order(sites)


def test_a_producer_reading_a_combines_output_is_its_producer_units_reading_it():
    """A call that reads a combine's output and produces a later collective's partial (the LM head reading the last norm) is
    its producer units alone: per rank and row, the second all-reduce's Gemm coordinates, each with its x wired a word a row
    from the first all-reduce's combine units, and no unit of the whole call."""
    b = TB.derive([rank_body(3, r, chain=True) for r in range(3)])
    sites = TW.sites_circuit([b], "w3")
    fns = _fns(sites)
    xs = [w for w in sites.circuit.wires if w.port == "x"]
    assert sites.unwired == () and sites.consumers == {} and all(fns[w.reader].id.startswith("Gemm_v2{K=16,N=1,") for w in xs)
    assert {sites.circuit.cuts[w.reader]["x"] for w in xs} == {N} and _bits(sites, xs) == 3 * ROWS * N * N * 16
    assert peer_bits(sites) == sum(9 * (s.hi - s.lo) * 16 for s in b.sites) == 9 * 2 * ROWS * N * 16
    in_order(sites)


def test_a_combine_of_no_grain_is_refused_by_name():
    """A collective combined by a Definition that isn't an all-reduce or a gather (a top-k sum) has no combine units, and is
    refused by name."""
    b = TB.derive([rank_body(3, r, combine=bind(moe.MoeSum, TOPK=3, H=N)) for r in range(3)])
    with pytest.raises(ValueError, match=r"w3: request 0's collective 0 \[0, 16\): MoeSum_v1\{TOPK=3,H=16\} has no coordinate "
                                         r"combine units: MoeSum_v1: not one of tp_allreduce's all-reduces"):
        TW.sites_circuit([b], "w3")


def test_the_order_check_refuses_a_wire_running_backward(toy):
    """The order check is not vacuous: a wire from a combine unit back into its own producer's input runs backward in the
    window's order and is refused, naming it."""
    tp = TW.sites_circuit(toy, "tp")
    j = next(j for j, s in enumerate(toy[0].sites) if toy[0].fns[1][s.own[1][0][0]].id.startswith("Gemm_v2{K=32"))
    (pt, pu, _), = tp.units[0, j, TW.PRODUCER, 0]
    (ct, cu, _), = tp.units[0, j, TW.COMBINE, 0]
    back = I.Wire(pt, "x", ct, "o0", at=pu, radix=(1,), base=cu, step=(0,))
    in_order(tp)
    with pytest.raises(AssertionError, match=rf"wire {len(tp.circuit.wires)} .* not below it"):
        in_order(tp, (*tp.circuit.wires, back))


def test_each_ranks_sent_partial_is_its_own_segment_at_the_sites_slice(toy):
    """The binding pairs rank r's segment p with rank p's own partial at the site's key, so it relies on each rank sending
    its partial laid out as `_Peers` reads it: row i of a collective at leaves [i·n, (i+1)·n) of what the rank sends."""
    assert all(sent_at_own_slices(b) for b in toy)


def sent_at_own_slices(b: TB.RequestBinding) -> bool:
    """Each collective's sites, in slice order, are disjoint whole rows of one width from leaf 0 (rows a dead-free compile
    dropped leave gaps), each rank's own segment exactly its slice's leaves and its producers in slice order: rank p's
    partial at a site is the row of what it sends that every other rank reads there."""
    by: dict[int, list[TB.Site]] = {}
    for s in b.sites:
        by.setdefault(s.k, []).append(s)
    for ss in by.values():
        n = min(s.hi - s.lo for s in ss)
        if (any(s.lo % n or (s.hi - s.lo) % n for s in ss) or any(a.hi > c.lo for a, c in zip(ss, ss[1:]))
                or any(sum(hi - lo for _, lo, hi in o) != s.hi - s.lo for s in ss for o in s.own)
                or any([s.own[p][0][0] for s in ss] != sorted({s.own[p][0][0] for s in ss}) for p in range(b.world))):
            return False
    return True
