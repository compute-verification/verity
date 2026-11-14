"""A window as one replay Definition (`window_replay.replay`), on windows `window_builder` wires by hand from the token
select's chunk and combine templates, small enough for the gate-by-gate reference: the Definition's evaluation on
made-up values gives each unit the output its template gives on the rows the window's wires name, and the wires read
back from its body alone are the window's, and its id names the window's record less its run fields.

`_window` holds the four shapes a request's window has: reader units reading writer units by a unit step, a port cut
into several writer rows, a port fed by more than one writer, and templates that read each other, one of them itself.
`_whole` holds none of them, and round 2 as it stands derives its wire from the Definition (`second_stage.replay_wires`)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import reference as REF
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Concat, Explicit, Strided
from verity.circuits.types import axis_runs
from verity.commitments.windows import interface as IF
from verity.definitions.registry import dead_free as DF
from verity.protocols.proofs.spot.verifier import second_stage as SS
from verity.protocols.proofs.spot.verifier.one_stage import partition as P

from verity_vllm.pipeline import window_replay as WR
from verity_vllm.program.predict.window_builder import Grain, Row, _Builder

X, XS = bind(DF.TokenSelectChunkV2, C=64), bind(DF.TokenSelectChunkV2, C=32)
C1, C2, C3, C4 = (bind(DF.TokenSelectCombineV2, V=v, C=64) for v in (64, 128, 160, 256))


def _circuit(b: _Builder, external: tuple[str, ...], name: str = "window-replay-test",
             drop: tuple[str, str] | None = None) -> IF.Circuit:
    """``b``'s units and feeds as a circuit, ordered by its allocations as `window_circuit.lower` orders a window;
    without the wires of reader and writer ``drop``."""
    b.settle()
    wires = tuple(w for w in IF.coalesce(b.wires) if (w.reader, w.writer) != drop)
    starts = {tid: [at for at, _n, _slot, _step in runs] for tid, runs in b.spans.items()}
    return IF.Circuit(name, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()), external=external, wires=wires,
                      cuts=b.cuts, order=IF.ordered(b.units, wires, starts) if wires else None)


def _record(circuit: IF.Circuit) -> dict:
    """``circuit``'s record as a run writes it: the compiler's record (`descriptor_record`) with a run's fields."""
    return {**IF.descriptor_record(circuit, 1), "run": "a run root", "context": "the coins' context"}


def _and_record(circuit: IF.Circuit) -> tuple[IF.Circuit, dict]:
    return circuit, _record(circuit)


def _window(free: bool = False) -> IF.Circuit:
    """The window of the module docstring; ``free``: without the short chunk's wires, so a row of `C3`'s ports is free
    beside its wired ones."""
    b = _Builder(Grain(), provable=False)
    b.label = (0, 0)
    x, _ = b.unit(X, 6)
    b.unit(X, 4)
    xs, _ = b.unit(XS, 3)
    # by a unit step, a port cut four ways: unit u's rows are chunks 5u to 5u + 3, unit 1's in both of X's ranges
    c4, _ = b.unit(C4, 2)
    b.feed(c4, "best", 0, 2, [Row(x, j, "o0", 1) for j in range(4)], lead=((2, 5),))
    b.feed(c4, "idx", 0, 1, [Row(x, j, "o1", 2) for j in range(4)])
    # two digits and a row step that don't continue each other: unit (a, b) reads chunks 5a + b and 5a + b + 2
    c2, _ = b.unit(C2, 4)
    for port, out, words in (("best", "o0", 1), ("idx", "o1", 2)):
        b.feed(c2, port, 0, 4, [Row(x, 0, out, words), Row(x, 2, out, words)], lead=((2, 5), (2, 1)))
    # more than one writer: unit u's rows are chunks u and u + 2 of 64 and short chunk u
    c3, _ = b.unit(C3, 2)
    for port, out, words in (("best", "o0", 1), ("idx", "o1", 2)):
        b.feed(c3, port, 0, 2, [Row(x, 0, out, words), Row(x, 2, out, words), Row(xs, 0, out, words)], lead=((2, 1),))
    # templates that read each other, and one that reads itself: C1's unit 0, C4's unit 1, C1's units 1 and 2
    c1, _ = b.unit(C1, 1)
    b.feed(c4, "idx", 1, 1, [Row(c1, 0, "o0", 2), *(Row(x, j, "o1", 2) for j in range(5, 8))])
    b.unit(C1, 1)
    b.feed(c1, "idx", 1, 1, [Row(c4, 1, "o0", 2)])
    b.unit(C1, 1)
    b.feed(c1, "idx", 2, 1, [Row(c1, 1, "o0", 2)])
    # a fan-out: two units read one row
    b.unit(C1, 2)
    b.feed(c1, "best", 3, 2, [Row(xs, 2, "o0", 1)])
    return _circuit(b, (f"{x}/x", f"{c1}/o0", f"{c3}/o0"), drop=(c3, xs) if free else None)


def _whole(external: tuple[str, ...] | None = None) -> IF.Circuit:
    """One unit of `C2`, and one of `C1` reading its token as ``idx``: a one-instance writer read whole."""
    b = _Builder(Grain(), provable=False)
    b.label = (0, 0)
    c2, _ = b.unit(C2, 1)
    c1, _ = b.unit(C1, 1)
    b.feed(c1, "idx", 0, 1, [Row(c2, 0, "o0", 2)])
    return _circuit(b, (f"{c1}/o0",) if external is None else external, "window-replay-whole")


def _cells(circuit: IF.Circuit) -> dict[tuple[str, str], dict[tuple[int, int], tuple[str, str, int]]]:
    """The circuit's wires a row at a time: by reader port, ``(unit, row) -> (writer, output, unit)``."""
    out: dict[tuple[str, str], dict[tuple[int, int], tuple[str, str, int]]] = {}
    for w in circuit.wires:
        for row in range(w.cut[0], w.cut[0] + w.cut[1]):
            for u, src in zip(range(w.at, w.at + w.reader_units), w.writers(w.at, w.reader_units, row).tolist()):
                out.setdefault((w.reader, w.port), {})[u, row] = (w.writer, w.out, src)
    return out


def _sources(refs, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leaves ``pos`` of ``refs``: whether each is a node's, its node or parameter, and its leaf there."""
    if isinstance(refs, Concat):
        starts = np.cumsum([0, *(p.count for p in refs.parts)])
        part = np.searchsorted(starts, pos, side="right") - 1
        node, idx, leaf = np.zeros(len(pos), bool), np.zeros(len(pos), np.int64), np.zeros(len(pos), np.int64)
        for k in np.unique(part).tolist():
            at = part == k
            node[at], idx[at], leaf[at] = _sources(refs.parts[k], pos[at] - starts[k])
        return node, idx, leaf
    if isinstance(refs, Explicit):
        got = [refs[int(j)] for j in pos]
        return (np.array([s == "n" for s, _i, _l in got], bool), np.array([i for _s, i, _l in got], np.int64),
                np.array([leaf for _s, _i, leaf in got], np.int64))
    assert isinstance(refs, (Affine, Strided)), refs
    leaf = refs.base + pos if isinstance(refs, Affine) else \
        refs.base + (pos // refs.inner) * refs.outer_stride + (pos % refs.inner) * refs.inner_stride
    return np.full(len(pos), refs.space == "n"), np.full(len(pos), refs.idx, np.int64), leaf


def read_back(defn) -> tuple[dict, dict[str, dict[str, int]]]:
    """The wires a replay Definition's body states, read from its body alone, as `_cells` gives a circuit's, and each
    wired port's cut. Unit u of a template is the u-th member of its nodes in body order. A row of a port is as long as
    the writer's output field the port's first reference into a node names, and is wired where its first leaf is the first
    of such a field of one writer unit and its last leaf that field's last; under a batch's mapped axis past the first,
    that is asked of each stretch of the row the axis leaves whole. Any other read of a node is refused. It walks every
    unit: a test's reference, no verifier's."""
    nodes = defn.body.nodes
    at, seen = [], {}
    for node in nodes:
        tid = descriptor_id(node.fn)
        at.append(seen.get(tid, 0))
        seen[tid] = at[-1] + (node.n or 1)
    cells: dict = {}
    cuts: dict[str, dict[str, int]] = {}
    for k, node in enumerate(nodes):
        tid, n = descriptor_id(node.fn), node.n or 1
        for i, ((p, ty), a) in enumerate(zip(node.fn.signature()[0], node.args)):
            axis = node.axes[i] if node.form == "batch" else None
            _outer, _n, run = (1, n, ty.leaves) if axis is None else axis_runs(a.type, axis)
            port = np.arange(ty.leaves, dtype=np.int64)
            member = np.arange(n if axis is not None else 1, dtype=np.int64)[:, None]
            pos = ((port // run) * (n if axis is not None else 1) * run + member * run + port % run).reshape(-1)
            is_node, idx, leaf = _sources(a.refs, pos)
            if not is_node.any():
                continue
            first = int(np.flatnonzero(is_node)[0])
            fields = WR._fields(nodes[int(idx[first])].fn.signature()[1])
            L = next(t.leaves for f0, t in fields if f0 == leaf[first] % nodes[int(idx[first])].fn.signature()[1].leaves)
            assert ty.leaves % L == 0, (tid, p, L)
            cuts.setdefault(tid, {}).setdefault(p, ty.leaves // L)
            assert cuts[tid][p] == ty.leaves // L, (tid, p)
            is_node, idx, leaf = (x.reshape(-1, ty.leaves // L, L) for x in (is_node, idx, leaf))
            for m in range(is_node.shape[0]):
                for row in range(ty.leaves // L):
                    if not is_node[m, row].any():
                        continue
                    w = nodes[int(idx[m, row, 0])]
                    ret = w.fn.signature()[1]
                    unit, off = divmod(int(leaf[m, row, 0]), ret.leaves)
                    j = next(j for j, (f0, t) in enumerate(WR._fields(ret)) if f0 == off and t.leaves == L)
                    assert is_node[m, row].all() and (idx[m, row] == idx[m, row, 0]).all() \
                        and (leaf[m, row] == leaf[m, row, 0] + np.arange(L)).all(), (tid, p, m, row)
                    got = (descriptor_id(w.fn), f"o{j}", at[int(idx[m, row, 0])] + unit)
                    for u in range(at[k], at[k] + n) if axis is None else (at[k] + m,):
                        cells.setdefault((tid, p), {})[u, row] = got
    return cells, {t: {p: c for p, c in ps.items() if c > 1} for t, ps in cuts.items() if any(c > 1 for c in ps.values())}


@pytest.mark.parametrize("free", [False, True])
def test_the_wires_read_back_from_the_body_are_the_windows(free):
    circuit = _window(free)
    cells, cuts = read_back(WR.replay(circuit, _record(circuit)))
    assert cells == _cells(circuit)
    assert cuts == {t: dict(ps) for t, ps in circuit.cuts.items()}
    assert sum(len(c) for c in cells.values()) == 44 if free else 48


@pytest.mark.parametrize("loose", [False, True])
def test_each_template_port_with_a_free_row_is_a_parameter_and_the_external_outputs_are_the_return(loose):
    circuit = _window(loose)
    params, ret = WR.replay(circuit, _record(circuit)).signature()
    wired = _cells(circuit)
    free = {}
    for fn, n in circuit.templates:
        tid = descriptor_id(fn)
        for (p, ty), (_p, _words) in zip(fn.signature()[0], IF.layout(fn).rows):
            k = circuit.cuts.get(tid, {}).get(p, 1)
            if rows := n * k - len(wired.get((tid, p), {})):
                free[f"{tid}/{p}"] = (rows, ty.leaves // k)
    assert {name: (t.n, t.elem.leaves) for name, t in params} == free
    assert [name for name, _t in params] == [x for fn, _n in circuit.templates for x in
                                             (f"{descriptor_id(fn)}/{p}" for p, _t in fn.signature()[0]) if x in free]
    assert free.get(f"{descriptor_id(C3)}/best") == ((2, 1) if loose else None)
    assert repr(ret) == "(Array<5,Value<32>>, Array<2,Value<32>>)"
    empty = _whole(external=())
    assert WR.replay(empty, _record(empty)).signature()[1].leaves == 0


def _unit_outputs(circuit: IF.Circuit, given: dict[str, list[int]]) -> dict[tuple[str, int], tuple[int, ...]]:
    """Each unit's output, a unit at a time from the circuit's wires: a wired row its writer unit's output field, a free
    row the next of its port's values in ``given``, by unit then row."""
    wired, fns = _cells(circuit), {descriptor_id(fn): (fn, n) for fn, n in circuit.templates}
    done: dict[tuple[str, int], tuple[int, ...]] = {}
    free = {(tid, p): sorted((u, j) for u in range(n) for j in range(circuit.cuts.get(tid, {}).get(p, 1))
                             if (u, j) not in wired.get((tid, p), {}))
            for tid, (fn, n) in fns.items() for p, _t in fn.signature()[0]}

    def out(tid: str, u: int) -> tuple[int, ...]:
        if (tid, u) not in done:
            fn, args = fns[tid][0], []
            for p, ty in fn.signature()[0]:
                k = circuit.cuts.get(tid, {}).get(p, 1)
                L, leaves = ty.leaves // k, []
                for j in range(k):
                    if (src := wired.get((tid, p), {}).get((u, j))) is not None:
                        f0, t = WR._fields(fns[src[0]][0].signature()[1])[int(src[1][1:])]
                        leaves += out(src[0], src[2])[f0:f0 + t.leaves]
                    else:
                        at = free[tid, p].index((u, j)) * L
                        leaves += given[f"{tid}/{p}"][at:at + L]
                args.append(leaves if ty.leaves > 1 or hasattr(ty, "n") else leaves[0])
            done[tid, u] = tuple(REF.evaluate(fn, *args))
        return done[tid, u]

    for tid, (_fn, n) in fns.items():
        for u in range(n):
            out(tid, u)
    return done


@pytest.mark.parametrize("seed, free", [(0, False), (1, True)])
def test_the_reference_evaluation_gives_each_unit_its_templates_output_on_its_wired_rows(seed, free):
    circuit = _window(free)
    defn = WR.replay(circuit, _record(circuit))
    rng = np.random.default_rng(seed)
    given = {name: [int(v) for v in rng.integers(0, 1 << w, t.leaves)] for name, t in defn.signature()[0]
             for w in [t.homogeneous_width()]}
    want = _unit_outputs(circuit, given)
    prog, call = REF.standalone_call(defn)
    trace: dict[int, int] = {}
    got = REF.evaluate_call(prog.circuit, call, [v for name, _t in defn.signature()[0] for v in given[name]], trace)
    seen: dict[str, int] = {}
    units = 0
    for child in call.children():
        tid = descriptor_id(child.fn)
        u = seen[tid] = seen.get(tid, -1) + 1
        assert tuple(trace[g] for g in child.returns) == want[tid, u], (tid, u)
        units += 1
    assert units == sum(circuit.units.values()) == 26
    c1, c3 = descriptor_id(C1), descriptor_id(C3)
    assert got == tuple(v for tid in (c1, c3) for u in range(circuit.units[tid]) for v in want[tid, u])
    assert len({want[c1, u] for u in range(5)}) > 1


def test_a_window_of_one_wire_read_whole_is_in_the_form_round_two_derives():
    circuit = _whole()
    defn = WR.replay(circuit, _record(circuit))
    assert [(n.form, n.fn.id) for n in defn.body.nodes] == [("call", C2.id), ("call", C1.id)]
    c1, c2 = descriptor_id(C1), descriptor_id(C2)
    assert SS.replay_wires(P.population_program(defn, 3), descriptor_id(defn)) == (SS.Wire(c1, "idx", c2, "o0"),)
    assert read_back(defn)[0] == _cells(circuit) == {(c1, "idx"): {(0, 0): (c2, "o0", 0)}}


def test_a_window_is_named_by_its_record_less_its_run_fields():
    circuit = _window()
    record = _record(circuit)
    h = hashlib.sha512(IF.circuit_bytes(IF.circuit_of(record))).hexdigest()
    a = WR.replay(circuit, record)
    assert WR.window_digest(record) == h and a.id == f"WindowReplay_v1{{window={h}}}"
    assert WR.replay(circuit, {**record, "run": "another run", "context": "other coins"}).id == a.id
    assert WR.replay(circuit, {**record, "cap": 2}).id != a.id
    assert a.id != WR.replay(_whole(), _record(_whole())).id
    again = _window()
    assert P.program_digest(Program(a)) == P.program_digest(Program(WR.replay(again, _record(again))))


def test_refused_by_name():
    c1, c2 = descriptor_id(C1), descriptor_id(C2)
    # C2's token (one u32) as C1's ``best`` and its ``x0`` beside it would be two words read as one leaf of 32 bits
    x = descriptor_id(X)
    with pytest.raises(ValueError, match="names each leaf at its own width"):
        WR.replay(*_and_record(IF.Circuit("reinterpreted", ((X, 2), (C2, 1)), external=(f"{c2}/o0",),
                                          cuts={c2: {"idx": 1}}, wires=(IF.Wire(c2, "best", x, "o1", 0, (1,), 0, (0,)),))))
    # C1's unit 1 before its unit 0, C2 between them
    wires = (IF.Wire(c2, "idx", c1, "o0", 0, (1,), 1, (0,), cut=(0, 1)), IF.Wire(c1, "idx", c2, "o0", 0, (1,), 0, (0,)))
    with pytest.raises(ValueError, match="in unit order"):
        WR.replay(*_and_record(IF.Circuit("backwards", ((C1, 2), (C2, 1)), external=(f"{c1}/o0",),
                                          cuts={c2: {"idx": 2}}, wires=wires, order=((c1, 1, 1), (c2, 0, 1), (c1, 0, 1)))))
    # another circuit's record, a record naming other external ports, one listing its members
    circuit = _whole()
    record = _record(circuit)
    for other in (_record(_window()), {**record, "external": []}):
        with pytest.raises(ValueError, match="is not its record"):
            WR.replay(circuit, other)
    with pytest.raises(ValueError, match="states its members by a cap"):
        WR.replay(circuit, {k: v for k, v in record.items() if k != "cap"})
    # a return of 8,193 tokens of 32 bits is past the widest size class
    WR.replay(*_and_record(IF.Circuit("widest", ((C1, 8192),), external=(f"{c1}/o0",))))
    with pytest.raises(ValueError, match="262,144"):
        WR.replay(*_and_record(IF.Circuit("wider", ((C1, 8193),), external=(f"{c1}/o0",))))
