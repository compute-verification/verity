"""verity_vllm.pipeline.tp.wires: a tensor-parallel window's collective sites at a grain C-Flock proves. Each rank's
coordinate words are gathered into rows; one combine unit a chunk, which every rank reads, has its parts cut W ways, row p
wired from rank p's gathered row. Each read of a combine's output is wired from the units that compute it.

The toy at TP2 (`test_steps`' TINY, `_mixed`'s first request: 45 collectives, each per row) is the compiled request: its
GEMM and embedding partials are coordinate units gathered into 64-word rows, so its all-reduces are one combine a row, and
its logits' all-gather has none, each rank's token select reading every rank's gathered LM-head rows. Each unit's values are
random words, each wired row its writer's output row: the wiring check reads the committed strings alone, never a unit's
arithmetic."""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import Builder, bind
from verity.circuits.refs import Affine, Coll
from verity.circuits.types import Array
from verity.commitments.windows import interface as I
from verity.commitments.windows import wiring as WI
from verity.definitions.ops import gemm as GE
from verity.definitions.registry import moe
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry.prims import BF16

from tests.program.predict.test_steps import _mixed, _registered
from tests.program.test_tp_binding import ROWS, N, rank_body
from verity_vllm.pipeline.tp import grain as TG
from verity_vllm.pipeline.tp import wires as TW
from verity_vllm.program import tp_binding as TB
from verity_vllm.program.predict.common import _Peers
from verity_vllm.program.predict.tp_steps import compile_tp

RUN, LAW, CAP, W = "ab" * 64, {"law": "stratified", "k": 2}, 4, 2
#: the fused bodies' hidden row: four chunks of 64 (`grain.chunk`)
H = 256
FUSED = {"N": H, "SHOT": 1, "CLUSTER": 1, "EPS": 1e-5, "FMA": False}


@pytest.fixture(scope="module")
def toy() -> list[TB.RequestBinding]:
    """The toy's first request at TP2: its binding."""
    return list(compile_tp(_registered(tp={"world": W, "rank": 0}), *_mixed()).bindings[:1])


def fused_body(world: int, rank: int):
    """Rank ``rank``'s body of a ``world``-rank request whose all-reduces are the fused MNNVL all-reduce and RMSNorm, two a
    row (`_Peers.rows`): a Gemm partial of the row fused with the row as residual, then a Gemm partial of its normed row
    fused with its new residual, as a decoder's next layer reads it."""
    params = (("x", Array(ROWS * H, BF16)), ("w", Array(H, Array(H, BF16))), ("g", Array(H, BF16)),
              *((f"peers_{m}", Array(ROWS * H, BF16)) for m in range(2 * (world - 1))))
    fused, gemm = bind(TA.AllReduceRmsNormMnnvl, WORLD=world, **FUSED), bind(GE.Gemm, K=H, N=H, **GE.GEMM_AMPERE)
    B = Builder(params, Array(H, BF16))
    w, g = Coll(params[1][1], Affine("p", 1, 0, H * H)), Coll(params[2][1], Affine("p", 2, 0, H))
    p = _Peers([Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params[3:], start=3)], world, rank)
    row, operand = Array(H, BF16), fused.signature()[0][0][1]
    xs = [Coll(row, Affine("p", 0, i * H, H)) for i in range(ROWS)]
    first = [B.call(fused, Coll(operand, q), x, g) for q, x in zip(p.rows([B.call(gemm, x, w).refs for x in xs]), xs)]
    parts = p.rows([B.call(gemm, Coll(row, f[0].refs), w).refs for f in first])
    second = [B.call(fused, Coll(operand, q), Coll(row, f[1].refs), g) for q, f in zip(parts, first)]
    return B.finish(Coll(row, second[-1][0].refs))


def _fns(sites: TW.Sites) -> dict[str, Any]:
    return {descriptor_id(fn): fn for fn, _ in sites.circuit.templates}


def _count(sites: TW.Sites) -> dict[str, int]:
    return I.input_count(I.descriptor_record(sites.circuit, CAP), _fns(sites).__getitem__)


def _bits(sites: TW.Sites, wires=None) -> int:
    """The bits ``wires`` (every wire of ``sites``) fill, as `input_count` counts them."""
    cuts = sites.circuit.cuts
    return sum(w.units * 16 * dict(I.layout(_fns(sites)[w.reader]).rows)[w.port] // cuts.get(w.reader, {}).get(w.port, 1)
               for w in (sites.circuit.wires if wires is None else wires))


def _templates_at(sites: TW.Sites, stage: str) -> set[str]:
    return {tid for (_i, _j, s, _r), rs in sites.units.items() if s == stage for tid, _a, _n in rs}


def _ids(sites: TW.Sites, stage: str) -> dict[str, dict[str, int]]:
    """``stage``'s templates by Definition, each with its cuts."""
    fns = _fns(sites)
    return {fns[t].id: sites.circuit.cuts.get(t, {}) for t in _templates_at(sites, stage)}


def peer_bits(sites: TW.Sites) -> int:
    """The bits of the wires carrying the ranks' partial rows (each site's gathers, or its producers where it has none) into
    what reads every rank's: each combine unit's parts, or a gather's readers."""
    gathered = {(i, j) for (i, j, s, _r) in sites.units if s == TW.GATHER}
    partial = {tid for (i, j, s, _r), rs in sites.units.items()
               if s == TW.GATHER or (s == TW.PRODUCER and (i, j) not in gathered) for tid, _a, _n in rs}
    return _bits(sites, [w for w in sites.circuit.wires if w.writer in partial])


def parts_bits(b: TB.RequestBinding, ranks: int | None = None) -> int:
    """The bits of ``ranks`` ranks' (every rank's) partials that `peer_bits` wires: each once into an all-reduce's combine
    units, and a gather's into each of those ranks' readers."""
    n = b.world if ranks is None else ranks
    return sum(n * (n if b.fns[0][s.combine[0][0]].defn in TG.GATHERS else 1) * (s.hi - s.lo) * 16 for s in b.sites)


def proof_blocks(sites: TW.Sites) -> dict[str, str]:
    """The window's own units (gathers, combines, output gathers, norms, a token select's groups and select) whose proof
    block C-Flock's composer refuses (`interface.check_proof_block` under the window's cuts), by template, and why."""
    fns, out = _fns(sites), {}
    own = set().union(*(_templates_at(sites, s) for s in (TW.GATHER, TW.COMBINE, TW.OUTPUT, TW.NORM)))
    own |= {t for rs in sites.consumers.values() if len(rs) == 2 for t, _a, _n in rs}
    for t in sorted(own):
        try:
            I.check_proof_block(fns[t], sites.circuit.cuts.get(t))
        except ValueError as e:
            out[fns[t].id] = str(e)
    return out


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
    output row, wires applied in the window's order so a row read on is the row it was wired to."""
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


def _row(ms, publics: dict, tid: str, unit: int, row: str) -> tuple[Any, int, slice]:
    """The member holding ``tid``'s unit ``unit``, that unit's global index, and where its row ``row``'s string sits in the
    member's public body."""
    base = min(m.base for m in ms if m.template == tid)
    m = next(m for m in ms if m.template == tid and m.base <= base + unit < m.base + m.n)
    names = [q for q, _ in m.all_rows]
    at = ((base + unit - m.base) * len(names) + names.index(row)) * WI.BC_BYTES
    assert len(publics[m.key]) == m.n * len(names) * WI.BC_BYTES
    return m, base + unit, slice(at, at + WI.BC_BYTES)


def test_every_peer_row_of_a_tp_window_is_wired_from_its_ranks_gathered_row(toy):
    """The TP window holds both ranks' producer units and their gathers into 64-word rows, one combine unit a row that both
    ranks read, and the calls reading them. Every combine unit's parts are cut in two, each row its rank's gathered row;
    every read of a combine's output (the norms' x and residual) is wired uncut, and each rank's token select is the served
    window's four group units, each reading a gathered LM-head row of 64 words, and its select. None is left free, and each
    of the window's own units passes C-Flock's proof block check but the select, whose 16 groups' bests and indices are no
    row past 111 B at the toy's 256 logits (GLM-5.3's is, `test_tp_steps`). `input_count` counts every peer bit wired; rank 0's and
    rank 1's windows alone wire only their own rows, each peer row free. Every wire runs forward in the window's public
    order, and the verifier derives the same members from the record."""
    tp = TW.sites_circuit(toy, "tp")
    fns, refused = _fns(tp), proof_blocks(tp)
    assert tp.unwired == () and list(refused) == ["TokenSelectCombine_v2{V=256,C=16}"]
    assert "no row past 111 B" in refused["TokenSelectCombine_v2{V=256,C=16}"]
    assert _ids(tp, TW.GATHER) == {"AllGather_v1{WORLD=64,N=1}": {"parts": 64}}
    assert _ids(tp, TW.COMBINE) == {"AllReduce2_v1{N=64}": {"parts": W}} and not _ids(tp, TW.OUTPUT) | _ids(tp, TW.NORM)
    assert _ids(tp, TW.CONSUMER) == {"RMSNormTriton_v5{N=64,EPS=1e-05}": {}, "RMSNormFusedCuda_v2{N=64,EPS=1e-05}": {},
                                     "TokenSelectGroups_v1{C=64,G=16}": {},
                                     "TokenSelectCombine_v2{V=256,C=16}": {"best": 4, "idx": 4}}
    count, ranks = _count(tp), [TW.sites_circuit(toy, f"rank{r}", ranks=[r]) for r in range(W)]
    assert peer_bits(tp) == parts_bits(toy[0]) == 116736 and all(peer_bits(x) == parts_bits(toy[0], 1) for x in ranks)
    combines = _templates_at(tp, TW.COMBINE)
    assert _bits(tp, [w for w in tp.circuit.wires if w.reader in combines]) == sum(
        tp.circuit.units[t] * 16 * 64 * W for t in combines)
    assert all(_bits(x, [w for w in x.circuit.wires if w.reader in combines]) == sum(
        x.circuit.units[t] * 16 * 64 for t in combines) for x in ranks)
    assert count["wired"] == _bits(tp)
    in_order(tp)
    rec = I.descriptor_record(tp.circuit, CAP)
    assert I.window_of_record({**rec, "run": RUN, "context": ""}, LAW, fns.__getitem__).members == I.members(tp.circuit, CAP)


@pytest.mark.parametrize("before", [4, pytest.param(None, marks=pytest.mark.slow)])
def test_a_forged_peer_row_passes_the_per_rank_window_and_fails_the_tp_windows_wiring_check(toy, before):
    """The headline: a prover commits, as the parts row of rank 1 at an attention all-reduce's combine unit, rank 1's
    gathered row there with one bit of one coordinate flipped. In rank 0's window alone (today's per-rank form) the row is a
    free input: the window closes and its wiring check passes. In the TP window the honest prover refuses to commit it, and
    the same committed string in the TP window's public body is caught by the verifier's wiring check, and by `check_drawn`
    when the sample draws that combine unit, only then. The windows are of that site and the ``before`` sites before it,
    or (None) of the request's 45: three closes of the whole request's units, which only the slow tier pays for."""
    first = next(j for j, s in enumerate(toy[0].sites) if toy[0].fns[1][s.own[1][0][0]].id.startswith("Gemm_v2{K=32"))
    lo = 0 if before is None else first - before
    request = [toy[0] if before is None else replace(toy[0], sites=toy[0].sites[lo:first + 1])]
    j = first - lo
    tp = TW.sites_circuit(request, "tp")
    values = _values(tp)
    win = _window(tp, values)
    got = win.close(salt_key=bytes(32))
    ms = win.window.members
    publics = {m.key: got.public_body(k) for k, m in enumerate(ms)}
    assert WI.check(ms, publics) == {} and WI.check_drawn(ms, publics, range(sum(m.n for m in ms))) == {}

    site, q = request[0].sites[j], 5
    (tid, at, _), = tp.units[0, j, TW.COMBINE, TW.ALL]
    forged = {t: ({p: a.copy() for p, a in ins.items()}, outs) for t, (ins, outs) in values.items()}
    forged[tid][0]["parts"][at, 64 + q] ^= 1
    with pytest.raises(ValueError, match=r"AllReduce2_v1\{N=64\}/c\d+: a parts#1 row is not the o0 row of the "
                                         r"AllGather_v1\{WORLD=64,N=1\}/c\d+ unit that wrote it"):
        _window(tp, forged).close(salt_key=bytes(32))

    rank0 = TW.sites_circuit(request, "rank0", ranks=[0])
    mine = _values(rank0)
    (tid0, at0, _), = rank0.units[0, j, TW.COMBINE, TW.ALL]
    mine[tid0][0]["parts"][at0, 64:] = forged[tid][0]["parts"][at, 64:]
    alone = _window(rank0, mine)
    closed = alone.close(salt_key=bytes(32))
    ms0 = alone.window.members
    publics0 = {m.key: closed.public_body(k) for k, m in enumerate(ms0)}
    assert WI.check(ms0, publics0) == {}

    m, g, cell = _row(ms, publics, tid, at, "parts#1")
    m0, _, cell0 = _row(ms0, publics0, tid0, at0, "parts#1")
    body = bytearray(publics[m.key])
    body[cell] = publics0[m0.key][cell0]
    assert bytes(body[cell]) != publics[m.key][cell]
    bad = {**publics, m.key: bytes(body)}
    caught, drawn = WI.check(ms, bad), WI.check_drawn(ms, bad, [g])
    assert list(caught) == list(drawn) == [m.key] and "parts#1 strings are not the o0 strings" in caught[m.key]
    others = [x for x in range(sum(m.n for m in ms)) if not m.base <= x < m.base + m.n]
    assert WI.check_drawn(ms, bad, others) == {}
    print(json.dumps({"site": [site.k, site.lo, site.hi], "coordinate": q,
                      "row": "the combine unit's parts#1 (rank 1's gathered row, one bit of the coordinate flipped)",
                      "rank0_window": WI.check(ms0, publics0), "tp_window_check": caught, "tp_window_check_drawn": drawn,
                      "drawn_without_it": WI.check_drawn(ms, bad, others)}))


def test_the_fused_all_reduce_is_chunk_units_reading_the_residual_row_whole():
    """The fused MNNVL all-reduce and RMSNorm at three ranks and a 256-word row: each rank's Gemm coordinates gathered into
    four 64-word rows; four chunk units `AllReduceResidualMnnvlAt_v1`, each reading the three ranks' chunk rows and the
    residual row whole; one gather of the four chunks into the new residual row; one `RmsNormMnnvl_v1` reading that row
    whole. The next fused call's chunk units read the first's residual row whole and its Gemm coordinates read the normed
    row whole, so nothing is cut but the parts and the gathers, nothing is left free, every wire runs forward and each of
    the window's own units passes C-Flock's proof block check."""
    b = TB.derive([fused_body(3, r) for r in range(3)])
    sites = TW.sites_circuit([b], "f3")
    fns = _fns(sites)
    assert sites.unwired == () and proof_blocks(sites) == {}
    assert _ids(sites, TW.GATHER) == {"AllGather_v1{WORLD=64,N=1}": {"parts": 64}}
    assert _ids(sites, TW.COMBINE) == {f"AllReduceResidualMnnvlAt_v1{{WORLD=3,N=256,C=64,AT={c},SHOT=1}}": {"parts": 3}
                                       for c in range(4)}
    assert _ids(sites, TW.OUTPUT) == {"AllGather_v1{WORLD=4,N=64}": {"parts": 4}}
    assert _ids(sites, TW.NORM) == {"RmsNormMnnvl_v1{N=256,SHOT=1,CLUSTER=1,EPS=1e-05,FMA=False}": {}}
    res = [w for w in sites.circuit.wires if w.port == "res"]
    assert {fns[w.writer].id for w in res} == {"AllGather_v1{WORLD=4,N=64}"} and _bits(sites, res) == ROWS * 4 * H * 16
    xs = [w for w in sites.circuit.wires if w.port == "x"]
    assert {fns[w.writer].id.split("{")[0] for w in xs} == {"RmsNormMnnvl_v1"} and _bits(sites, xs) == 3 * ROWS * H * H * 16
    assert peer_bits(sites) == parts_bits(b) == 3 * 2 * ROWS * H * 16
    in_order(sites)


def test_a_collective_over_several_rows_is_one_combine_a_chunk_of_its_token_row():
    """A route all-reduce over both of a rank's rows (`_Peers.whole`, three ranks), its own segment two producers' rows,
    whose output no call reads by rows, has one token row of both rows' 32 words: each rank's coordinates gathered into one
    32-word row, across the two producers, and one combine unit reading the three. Every parts row of it and of the
    per-row collectives is wired, from the gathered row of the rank it names, and every wire runs forward."""
    b = TB.derive([rank_body(3, r, whole=True) for r in range(3)])
    sites = TW.sites_circuit([b], "w3")
    assert [len(s.own[0]) for s in b.sites] == [1, 1, 2] and sites.unwired == ()
    assert {(_fns(sites)[t].id, n) for (_i, j, s, _r), rs in sites.units.items() if s in (TW.GATHER, TW.COMBINE) and j == 2
            for t, _a, n in rs} == {("AllGather_v1{WORLD=32,N=1}", 1), ("AllReduce_v2{WORLD=3,N=32}", 1)}
    assert peer_bits(sites) == _bits(sites, [w for w in sites.circuit.wires if w.reader in _templates_at(sites, TW.COMBINE)])
    assert peer_bits(sites) == parts_bits(b) == 3 * 2 * ROWS * N * 16
    in_order(sites)


def test_a_producer_reading_a_combines_output_is_its_producer_units_reading_it():
    """A call that reads a combine's output and produces a later collective's partial (the LM head reading the last norm) is
    its producer units alone: per rank and row, the second all-reduce's Gemm coordinates, each with its x wired uncut from
    the first all-reduce's combine unit, and no unit of the whole call."""
    b = TB.derive([rank_body(3, r, chain=True) for r in range(3)])
    sites = TW.sites_circuit([b], "w3")
    fns = _fns(sites)
    xs = [w for w in sites.circuit.wires if w.port == "x"]
    assert sites.unwired == () and sites.consumers == {} and all(fns[w.reader].id.startswith("Gemm_v2{K=16,N=1,") for w in xs)
    assert all("x" not in sites.circuit.cuts.get(w.reader, {}) for w in xs) and _bits(sites, xs) == 3 * ROWS * N * N * 16
    assert peer_bits(sites) == parts_bits(b) == 3 * 2 * ROWS * N * 16
    in_order(sites)


def test_a_combine_of_no_grain_is_refused_by_name():
    """A collective combined by a Definition that isn't an all-reduce or a gather (a top-k sum) has no combine units, and is
    refused by name."""
    b = TB.derive([rank_body(3, r, combine=bind(moe.MoeSum, TOPK=3, H=N)) for r in range(3)])
    with pytest.raises(ValueError, match=r"w3: request 0's collective 0 \[0, 16\): MoeSum_v1\{TOPK=3,H=16\} has no coordinate "
                                         r"combine units: MoeSum_v1"):
        TW.sites_circuit([b], "w3")


def test_the_order_check_refuses_a_wire_running_backward(toy):
    """The order check is not vacuous: a wire from a combine unit back into its own producer's input runs backward in the
    window's order and is refused, naming it."""
    tp = TW.sites_circuit(toy, "tp")
    j = next(j for j, s in enumerate(toy[0].sites) if toy[0].fns[1][s.own[1][0][0]].id.startswith("Gemm_v2{K=32"))
    (pt, pu, _), = tp.units[0, j, TW.PRODUCER, 0]
    (ct, cu, _), = tp.units[0, j, TW.COMBINE, TW.ALL]
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


def _closing(sites: TW.Sites) -> tuple[tuple[I.Wire, ...], str, str]:
    """``sites``' wires and one more closing a cycle: the first site whose rank 0 producer reads a row as wide as its combine's
    output reads its ``x`` from that output, which sums the producer's own partial. -> the wires, the producer's template and
    the combine's."""
    fns = _fns(sites)
    for (i, j, stage, r), rs in sites.units.items():
        for (ptid, pu, _n) in (rs if stage == TW.PRODUCER and r == 0 else ()):
            for ctid, cu, _m in sites.units.get((i, j, TW.COMBINE, TW.ALL), ()):
                if dict(I.layout(fns[ptid]).rows).get("x") == dict(I.layout(fns[ctid]).outs)["o0"]:
                    forged = I.Wire(ptid, "x", ctid, "o0", at=pu, radix=(1,), base=cu, step=(0,))
                    return (*sites.circuit.wires, forged), ptid, ctid
    raise AssertionError("no producer reads a row as wide as its combine's output")


@pytest.mark.parametrize("world", [W, 8])
def test_a_wire_closing_a_cycle_is_refused_and_the_tp_windows_wires_have_none(toy, world):
    """Units that read one another through wires pass every unit's check on a fixed point the prover chose (a combine
    summing a partial its producer computed from that combine's output), so `interface.check_wires` refuses wires whose
    dataflow between units has a cycle, naming a wire on it (`Spot.Wires.check_acyclic`). The TP window's wires,
    at TP2 (the toy) and TP8 (a Gemm partial a row, `rank_body`), have none and need no order but the window's own. One
    wire more, rank 0's producer reading its ``x`` from its combine's output, is refused: under the window's order as out
    of it, with no order as a cycle of the two templates, under the honest wires' order as the first wire out of it, the
    forged one, and no order of single units holds it."""
    sites = TW.sites_circuit(toy, "tp") if world == W else \
        TW.sites_circuit([TB.derive([rank_body(world, r) for r in range(world)])], f"w{world}")
    c = sites.circuit
    lays = {descriptor_id(fn): (I.layout(fn), n) for fn, n in c.templates}
    assert c.order == sites.order and len({r for _i, _j, _s, r in sites.units if r != TW.ALL}) == world
    replace(c, order=None)
    wires, ptid, ctid = _closing(sites)
    with pytest.raises(ValueError, match=f"order: wire {len(wires) - 1} "):
        replace(c, wires=wires)
    with pytest.raises(ValueError, match=r"cycle: wire \d+ .* is on a cycle of templates") as e:
        replace(c, wires=wires, order=None)
    cyclic = re.match(r"cycle: wire (\d+)", str(e.value))
    assert cyclic is not None
    named = wires[int(cyclic[1])]
    assert {named.reader, named.writer, ptid, ctid} <= set(str(e.value).split("each reading the next: ")[1].split(";")[0]
                                                         .split(" <- "))
    honest = I.ordered(c.units, c.wires, {})
    I.check_wires(lays, c.wires, c.cuts, honest)
    with pytest.raises(ValueError, match=f"order: wire {len(wires) - 1} "):
        I.check_wires(lays, wires, c.cuts, honest)
    with pytest.raises(ValueError, match="cycle: no order of the ranges"):
        I.ordered(c.units, wires, {t: range(n) for t, n in c.units.items()})
