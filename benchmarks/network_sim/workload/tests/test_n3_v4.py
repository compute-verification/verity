"""N3Window_v4 (`n3_v4`): N3Window_v3 with its openings as runs. A range opening is its tree's on every run and refuses
a run with a gap, out of order or at another start; v4's reference accepts exactly where v3's does on the demo's, the
simulator's and the lab's windows, honest and tampered; a run witness that cheats (a skipped row, a repeat of another
row, a row fresh twice, another unit's row, a read in no run, a wrong start or unit, a slot past the tree) is refused;
and the circuit equals its reference."""
import copy
import dataclasses
import random

import numpy as np
import pytest

from test_n3_unit import (BLOCK, INGRESS, LAB_B, OTHER, SALTED_B, TAMPERS, data_end, flip, lab_setup, lab_windows, payloads,
                          sim_run)
from verity.core.protocols.certifier.grid import Reason
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates.frame_v3 import leaf_ref, node_ref
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from workload import n3_unit as U
from workload import n3_v4 as V

DOMAIN = bytes(range(64))


def tree(D: int, seed: int = 0) -> list[list[bytes]]:
    """A depth-D frame-v3-sha512 tree's levels over random values."""
    rng = random.Random(seed)
    levels = [[leaf_ref(DOMAIN, SCHEMA, rng.randbytes(64), x) for x in range(1 << D)]]
    for k in range(D):
        prev = levels[-1]
        levels.append([node_ref(DOMAIN, k, i, prev[2 * i], prev[2 * i + 1]) for i in range(len(prev) // 2)])
    return levels


def opened(levels, start: int, leaves, D: int) -> bytes:
    return V.range_levels(DOMAIN, start, leaves, *V.siblings(levels, start, len(leaves), D), D)[D]


def test_slots_are_the_most_nodes_a_run_spans():
    for S in range(1, 10):
        for k in range(5):
            spans = [((a + S - 1) >> k) - (a >> k) + 1 for a in range(64)]
            assert max(spans) == V.slots(S, k)


def test_a_range_opening_is_its_trees_on_every_run():
    D = 4
    levels = tree(D)
    for S in range(1, (1 << D) + 1):
        for start in range(0, (1 << D) - S + 1):
            out = V.range_levels(DOMAIN, start, levels[0][start:start + S], *V.siblings(levels, start, S, D), D)
            assert out == [levels[k][start >> k] for k in range(D + 1)]


def test_a_range_opening_refuses_a_run_with_a_gap_out_of_order_or_at_another_start():
    D = 4
    levels = tree(D)
    root = levels[D][0]
    for S in range(2, 9):
        for start in range(0, (1 << D) - S):
            run = levels[0][start:start + S]
            assert opened(levels, start, run, D) == root
            gap = levels[0][start:start + S - 1] + [levels[0][start + S]]
            swapped = run[:]
            swapped[0], swapped[1] = swapped[1], swapped[0]
            assert opened(levels, start, gap, D) != root
            assert opened(levels, start, swapped, D) != root
            assert opened(levels, start + 1, run, D) != root


def test_partition_puts_each_read_in_the_first_run_it_continues():
    def seg(kind, u, q):
        return U.Segment(kind, u, q)

    rs = [seg(1, 0, 0), seg(0, 0, 0), seg(1, 0, 0), seg(1, 1, 5), seg(1, 0, 1), seg(1, 0, 3), seg(1, 1, 6), seg(1, 0, 2)]
    assert V.partition(rs) == ([0, None, 0, 1, 0, 2, 1, 0], [True, False, False, True, True, True, True, True])


@pytest.mark.parametrize("R, N", [(0, 4), (True, 4), (1, 18)], ids=["no-runs", "runs-not-an-int", "past-the-layout"])
def test_a_size_class_v4_doesnt_cover_is_refused(R, N):
    with pytest.raises(ValueError):
        V.unit(U.Format(16, 13, 2, 2, (2, 0), 4), N, 1, False, U.Shape(4, 4, 8), R).signature()


# --- the demo ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo():
    return U.demo()


def v4(setup: U.Setup, inst: U.Instance, R: int | None = None, **kw) -> U.Instance:
    return V.convert(inst, setup.layout.tree, setup.units, V.runs_of(setup.layout, "p0") if R is None else R, **kw)


def v3_cases(demo) -> dict[str, U.Instance]:
    """test_n3_unit's demo cases, honest and with one witness or public input changed, as v3 instances."""
    inst = demo.window.instance
    a = inst.args

    def edit(**kw):
        x = copy.deepcopy(inst)
        x.args.update(kw)
        return x

    def row(i, g, at):
        rows = copy.deepcopy(a["rows"])
        rows[i][g] = flip(rows[i][g], at)
        return edit(rows=rows)

    fr = a["frames"]
    s = demo.setup
    overflow = dataclasses.replace(demo.window, reason=Reason.OVERFLOW.value, instance=None)
    rc = s.certifier.commit("demo", 0, [s.head("p0", 0, overflow), s.head("p1", 0, None)], s.grids["p0"],
                            s.grids["p1"], s.sync_set)
    return {
        "honest": inst,
        "frame-bit": edit(frames=[fr[0], flip(fr[1], 9), fr[2], fr[3]]),
        "suffix-bit": edit(frames=[fr[0], fr[1], flip(fr[2], 14), fr[3]]),
        "outbox-row": row(1, 0, 14),
        "row-tail": row(1, 0, 20),
        "zero-row": row(0, 1, 0),
        "descriptor-salt": edit(descriptor_salts=[flip(a["descriptor_salts"][0], 0)] + a["descriptor_salts"][1:]),
        "at": edit(at=a["at"] + 1),
        "record-salt": edit(record_salt=flip(a["record_salt"], 3)),
        "direction": edit(direction=1),
        "head-count": edit(head=flip(a["head"], len(a["head"]) - 1)),
        "incomplete": edit(record=rc.root, head=rc.head(0), record_salt=rc.opening.salts[U.N2.HEADS],
                           record_path=rc.path(U.N2.HEADS)),
    }


@pytest.fixture(scope="module")
def demo_cases(demo):
    """`v3_cases` as v3 and v4 instances."""
    return {k: (x, v4(demo.setup, x)) for k, x in v3_cases(demo).items()}


def test_the_demos_reads_are_two_runs(demo):
    inst = v4(demo.setup, demo.window.instance)
    a = inst.args
    assert inst.R == 2 and a["run_units"] == [0, 1] and a["run_starts"] == [0, 0]
    assert [r.index(1) if any(r) else None for r in a["runs"]] == [None, None, 0, None, 1, 0, 0, 0]
    assert a["fresh"] == [0, 0, 1, 0, 1, 1, 1, 0]


def test_v4_accepts_exactly_where_v3_does_on_the_demo(demo_cases):
    got = {k: (U.reference(x), V.reference(y)) for k, (x, y) in demo_cases.items()}
    assert got["honest"] == (True, True) and all(a == b for a, b in got.values())
    assert sum(a for a, _ in got.values()) == 1


def cheats(demo) -> dict[str, tuple[U.Instance, bool]]:
    """The demo window with R = 3 runs and each run witness a prover could give, honest where it opens what v3 opens.
    Its reads, t = 2i + g: 2 is u0's row 0, 4 u1's row 0, 5 u0's row 1, 6 and 7 u0's row 2; the rest are of kind 0."""
    v3 = demo.window.instance

    def runs(*of):
        return [[int(o == r) for r in range(3)] for o in of]

    honest_runs = runs(None, None, 0, None, 1, 0, 0, 0)
    honest_fresh = [0, 0, 1, 0, 1, 1, 1, 0]

    def case(r=honest_runs, fresh=honest_fresh, **kw):
        return v4(demo.setup, v3, 3, runs=r, fresh=fresh, **kw)

    def with_fresh(t, b):
        x = honest_fresh[:]
        x[t] = b
        return x

    return {
        "honest": (case(), True),
        "row-in-a-run-of-its-own": (case(runs(None, None, 2, None, 1, 0, 0, 0)), True),
        "repeat-in-a-run-of-its-own": (case(runs(None, None, 0, None, 1, 0, 0, 2), with_fresh(7, 1)), True),
        "kind-0-read-in-a-run": (case(runs(0, None, 0, None, 1, 0, 0, 0), with_fresh(0, 1)), True),
        "a-skipped-row": (case(runs(None, None, 0, None, 1, 2, 0, 0)), False),
        "a-repeat-of-another-row": (case(fresh=with_fresh(6, 0)), False),
        "a-row-fresh-twice": (case(fresh=with_fresh(7, 1)), False),
        "another-units-row": (case(runs(None, None, 0, None, 0, 0, 0, 0)), False),
        "a-read-in-no-run": (case(runs(None, None, None, None, 1, 0, 0, 0)), False),
        "a-wrong-start": (case(starts={0: 1}), False),
        "a-wrong-unit": (case(run_units={1: 0}), False),
        "a-slot-past-the-tree": (case(starts={1: 1}), False),
        "a-first-member-not-fresh": (case(fresh=with_fresh(2, 0)), False),
    }


def test_a_run_witness_that_cheats_is_refused(demo):
    got = {k: V.reference(x) for k, (x, _) in cheats(demo).items()}
    assert got == {k: want for k, (_, want) in cheats(demo).items()}


def test_a_slot_past_a_depth_0_tree_opens_its_root_and_only_the_fit_refuses_it(demo):
    """u1's tree is one row, so its run's level 0 is its root whatever the start: a fresh leaf at slot 1 would be
    unbound but for `N3RunFits` (start + count <= 2^depth)."""
    x, _ = cheats(demo)["a-slot-past-the-tree"]
    a = x.args
    assert demo.layout.fmt.DEPTHS[1] == 0 and a["run_units"][1] == 1 and a["run_starts"][1] == 1
    leaf = V._leaf(a["domains"][1], a["rows"][2][0], 16 * x.fmt.B, a["row_salts"][2][0], 0)
    assert leaf == a["roots"][1]


# --- the simulator and the lab ----------------------------------------------------------------------------------------

def agree(setup: U.Setup, windows) -> list[tuple[str, int, bool]]:
    """Each window with a unit: its port, window and v3's verdict, after checking v4's is the same."""
    out = []
    for w in windows:
        if w.instance is not None:
            a, b = U.reference(w.instance), V.reference(V.instance(setup, w))
            assert a == b, (w.port, w.window, a, b)
            out.append((w.port, w.window, a))
    return out


def test_v4_accepts_exactly_where_v3_does_on_the_simulators_windows():
    seen = {}
    for t in (None, *TAMPERS):
        (setup, windows), _ = sim_run(t)
        seen[t] = agree(setup, windows)
    assert all(ok for _, _, ok in seen[None]) and all(not all(ok for _, _, ok in seen[t]) for t in TAMPERS)


@pytest.fixture(scope="module")
def lab():
    return lab_setup(LAB_B)


@pytest.mark.parametrize("edit", [
    None,
    lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:], i),
    lambda e, i: (e, [flip(i[0], U.HEADER.size)] + i[1:]),
    lambda e, i: (e, i[:-1] + [flip(i[-1], U.HEADER.size + 4)]),
    lambda e, i: ([e[0][:-1] + b"\x01"] + e[1:], i),
    lambda e, i: (e[:1] + e, i),
], ids=["honest", "flipped-event", "flipped-request", "flipped-last-request", "fill", "extra-frame"])
def test_v4_accepts_exactly_where_v3_does_on_the_labs_windows(lab, edit):
    got = agree(*lab_windows(lab, edit))
    assert got and all(ok for _, _, ok in got) == (edit is None)


@pytest.fixture(scope="module")
def salted_lab():
    return lab_setup(SALTED_B)


def test_v4_accepts_exactly_where_v3_does_on_salted_lab_windows(salted_lab):
    salts = payloads(salted_lab)
    for s, ok in ((salts, True), ({}, False), (salts | {(BLOCK, 0): OTHER}, False)):
        got = agree(*lab_windows(salted_lab, salts=s))
        assert all(x for _, _, x in got) == ok


@pytest.mark.slow
def test_v4_accepts_exactly_where_v3_does_on_a_lab_whose_outboxes_are_two_rows_deep():
    """At the lab's MLP width every outbox is one row, so its runs open only slot 0 of a depth-0 tree. At width 8 over
    32 requests each unit's outbox is two rows, and later windows' runs start at row 1: the one test of whole windows
    whose runs open at a private start past 0 in a tree with levels. About 80 s, most of it committing the larger
    MLP."""
    lab = lab_setup(LAB_B, d=8, requests=32)
    for edit, ok in ((None, True), (lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:], i), False)):
        setup, windows = lab_windows(lab, edit)
        assert all(x >= 1 for x in setup.layout.fmt.DEPTHS)
        assert any(V.instance(setup, w).args["run_starts"][0] for w in windows if w.instance is not None)
        got = agree(setup, windows)
        assert got and all(x for _, _, x in got) == ok


# --- the circuit ------------------------------------------------------------------------------------------------------

def bits_of(d: bytes) -> list[int]:
    return U._digest_bits(d)


@pytest.mark.slow
@pytest.mark.parametrize("S", [1, 3, 6, 8])
def test_n3range_is_its_reference(S):
    """`N3Range{S, 3}` against `range_levels` at every start of a depth-3 tree, runs past its end included, with honest
    and random siblings: every level's first node, the garbage past the tree too. It is the one test of the range's
    wiring (the alignment, the right sibling's place, the nodes' indices), which the window tests reach only at the
    demo's starts. About 25 s for the first S, nearly all of it tracing the node's SHA-512 compression once; under a
    second for each after."""
    D, rng = 3, random.Random(S)
    levels = tree(D, S)
    cases = []
    for start in range(1 << D):
        leaves = [levels[0][(start + j) % (1 << D)] for j in range(S)]
        cases.append((start, leaves, *V.siblings(levels, start, S, D)))
        cases.append((start, leaves, [rng.randbytes(64) for _ in range(D)], [rng.randbytes(64) for _ in range(D)]))
    args = [[U._int_bits(c[0], D) for c in cases], [[b for x in c[1] for b in bits_of(x)] for c in cases],
            [bits_of(DOMAIN)] * len(cases), [[b for x in c[2] for b in bits_of(x)] for c in cases],
            [[b for x in c[3] for b in bits_of(x)] for c in cases]]
    out = np.asarray(evaluate_bits(bind(V.N3Range, S=S, D=D), *(np.asarray(a, dtype=np.uint8) for a in args)))
    want = [[b for x in V.range_levels(DOMAIN, *c, D) for b in bits_of(x)] for c in cases]
    assert out.reshape(len(cases), -1).tolist() == want
    honest = [c for c in cases[0::2] if c[0] + S <= 1 << D]
    assert all(V.range_levels(DOMAIN, *c, D)[D] == levels[D][0] for c in honest)


@pytest.mark.slow
def test_the_demo_window_is_its_reference(demo, demo_cases):
    """The whole v4 window at the demo's format: test_n3_unit's thirteen instances with R = 2, then the run witnesses
    above with R = 3. It is the one test of v4's wiring, which the reference tests can't catch: the descriptors' run,
    each read's runs and carry, the compaction of fresh leaves onto the slots, and each run's range, root and fit.
    About 70 s after the range's test, most of it tracing and evaluating the two windows' SHA-512 compressions."""
    insts = [y for _, y in demo_cases.values()]
    assert V.evaluate(*insts) == [int(V.reference(x)) for x in insts] == [int(U.reference(x)) for x, _ in
                                                                           demo_cases.values()]
    cases = cheats(demo)
    assert V.evaluate(*(x for x, _ in cases.values())) == [int(want) for _, want in cases.values()]


@pytest.mark.slow
def test_a_salted_window_is_its_reference(salted_lab):
    """The salted lab window (B = 240) as v4: honest, with a wrong salt payload, and with the salt frame's slot moved.
    It is the one test of v4 with the salt path, two segments and two runs (a request's frame reads the
    outside host's length and head, then the message). About 7 s after the demo window's test, whose
    traced parts it shares in the suite's one process."""
    salts = payloads(salted_lab)
    setup, windows = lab_windows(salted_lab, salts=salts)
    w = next(w for w in windows if w.port == INGRESS and w.window == 0)
    honest = V.instance(setup, w)
    wrong = copy.deepcopy(honest)
    wrong.args["salt"] = wrong.args["salt"][:8] + OTHER
    moved = copy.deepcopy(honest)
    moved.args["slot"] = 1
    cases = [honest, wrong, moved]
    assert honest.salted and honest.R == 2
    assert V.evaluate(*cases) == [int(V.reference(x)) for x in cases] == [1, 0, 0]
