"""N3Window_v5 (`n3_v5`): N3Window_v4 over outbox rows of B bytes, and with a template. An outbox's rows hold each byte
once and two of them are v4's row; v5's reference accepts exactly where v4's does on the demo's, the simulator's and
the lab's windows, honest and tampered, and on a segment across two rows tampered in either; a run witness that cheats
(v4's cheats, a read in two runs, a wrong head) is refused; a template accepts the canonical demo's frames and refuses a
frame off it, on rows of B and on rows narrower than its frames; and the circuit equals its reference."""
import copy
import random

import numpy as np
import pytest

from test_n3_unit import (BLOCK, INGRESS, LAB_B, OTHER, SALTED_B, TAMPERS, bits, data_end, flip, lab_setup, lab_windows,
                          payloads, sim_run)
from test_n3_v4 import v3_cases
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from workload import n3_unit as U
from workload import n3_v4 as V4
from workload import n3_v5 as V5


def test_an_outbox_is_its_bytes_in_rows_of_b_then_a_zero_row_up_to_a_power_of_two():
    data = bytes(range(1, 40))
    assert V5.outbox_rows(b"", 16) == [bytes(16)] * 2
    assert V5.outbox_rows(data[:16], 16) == [data[:16], bytes(16)]
    assert V5.outbox_rows(data[:17], 16) == [data[:16], data[16:17].ljust(16, b"\0"), bytes(16), bytes(16)]
    assert V5.outbox_rows(data, 16) == [data[:16], data[16:32], data[32:].ljust(16, b"\0"), bytes(16)]


@pytest.mark.parametrize("n", [0, 1, 15, 16, 17, 39, 64, 65])
def test_two_rows_of_v5s_outbox_are_v4s_row(n):
    data = bytes(range(1, n + 1))
    v4, v5 = U.outbox_rows(data, 16), V5.outbox_rows(data, 16)
    k = max(1, -(-n // 16))
    assert all(v5[q] + v5[q + 1] == v4[q] for q in range(k))
    assert b"".join(v5).rstrip(b"\0") == data and len(v5) & (len(v5) - 1) == 0


@pytest.mark.parametrize("T, S", [(1, 1), (2, 1), (8, 3), (8, 4), (9, 8), (35, 7), (35, 35)])
def test_a_compaction_stage_computes_only_the_positions_the_next_reads(T, S):
    stages, v4 = V5.compaction(T, S), V4.compaction(T, S)
    assert [w for _, w in stages] == [w for _, w in v4]
    for b, (ps, _) in enumerate(stages):
        want = range(min(S, T)) if b == len(stages) - 1 else sorted(
            {q for p in stages[b + 1][0] for q in (p, p + (2 << b)) if q < T})
        assert ps == list(want) and len(ps) <= v4[b][0]
    if S >= T:
        assert [len(ps) for ps, _ in stages] == [T] * len(stages)


@pytest.mark.parametrize("R, tpl", [(0, ()), (1, (0, 9, 14, "", "")), (1, (1, 9, 17, "", "")),
                                    (1, (1, 9, 14, "", "0a0a0a"))],
                         ids=["no-runs", "mode-0", "s-past-the-frame", "suffix-past-the-frame"])
def test_a_size_class_v5_doesnt_cover_is_refused(R, tpl):
    with pytest.raises(ValueError):
        V5.unit(U.Format(16, 13, 2, 2, (2, 1), 4), 4, 1, False, U.Shape(4, 4, 8), R, tpl).signature()


# --- the demo ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo():
    return U.demo()


def pair(setup: U.Setup, x: U.Instance, R: int | None = None) -> tuple[U.Instance, U.Instance]:
    """The v4 and v5 instances of a v3 one."""
    R = V4.runs_of(setup.layout, "p0") if R is None else R
    return V4.convert(x, setup.layout.tree, setup.units, R), V5.convert(x, V5.prepare(setup), R)


def two_row_cases(demo) -> dict[str, tuple[U.Setup, U.Instance]]:
    """Frame 1 reads u0's bytes 14 ... 18, across rows 0 and 1 of v5's outbox (one segment of v4's row 0): the frame's
    byte from each row flipped, and with it the row that byte comes from (the witness changed to match), each recorded
    by a certifier of its own, so the record opens and only the frame's content can refuse."""
    fr = demo.window.frames

    def case(at_frame, at_row=None):
        s = U.Setup(None, demo.units, demo.layout, demo.setup.links, demo.setup.grids, demo.setup.seed)  # type: ignore[arg-type]
        (w,) = s.attach([U.Window("p0", 0, [fr[0], flip(fr[1], at_frame), fr[2], fr[3]], (2, 2))])
        if at_row is not None:
            w.instance.args["rows"][1][0] = flip(w.instance.args["rows"][1][0], at_row)
        return s, w.instance

    return {"frame-from-row-0": case(9 + 1), "frame-from-row-1": case(9 + 3), "row-0-and-frame": case(9 + 1, 15),
            "row-1-and-frame": case(9 + 3, 17)}


def demo_cases(demo) -> dict[str, tuple[U.Setup, U.Instance]]:
    """test_n3_v4's demo cases but v4's zero-row, which has no v5 witness (a kind-0 read's window is zero whatever its
    row: `cheats`, a-kind-0-reads-row-is-anything), and the crossing segment's tampers."""
    return {k: (demo.setup, x) for k, x in v3_cases(demo).items() if k != "zero-row"} | two_row_cases(demo)


def test_the_demos_crossing_segment_reads_two_rows(demo):
    s = U.fields(demo.layout.fmt, demo.window.instance.args["descriptors"][1]).segs[0]
    assert (s.kind, s.u, s.q, s.o, s.n) == (1, 0, 0, 14, 5) and s.o + s.n > demo.layout.fmt.B


def test_v5_accepts_exactly_where_v4_does_on_the_demo(demo):
    got = {k: tuple(f(y) for f, y in zip((V4.reference, V5.reference), pair(s, x)))
           for k, (s, x) in demo_cases(demo).items()}
    assert got["honest"] == (True, True) and all(a == b for a, b in got.values()), got
    assert sum(a for a, _ in got.values()) == 1


def test_the_demos_v5_runs_are_v4s_with_a_head_a_row_before(demo):
    inst = V5.instance(demo.setup, demo.window)
    a, rows = inst.args, V5.prepare(demo.setup).units.trees
    assert inst.R == 2 and a["run_units"] == [0, 1] and a["run_starts"] == [0, 0]
    assert a["fresh"] == V4.instance(demo.setup, demo.window).args["fresh"]
    assert a["run_heads"] == [rows["u0"].reg.rows[0], rows["u1"].reg.rows[0]]
    assert [r for per in a["rows"] for r in per][2] == rows["u0"].reg.rows[1]


def cheats(demo) -> dict[str, tuple[U.Instance, bool]]:
    """The demo window with R = 3 runs and each run witness a prover could give, as test_n3_v4's, and v5's own: a read
    in two runs, a run whose head is another row, and a kind-0 read with a row. Reads t = 2i + g: 2 is u0's row 0, 4
    u1's row 0, 5 u0's row 1, 6 and 7 u0's row 2; the rest are of kind 0."""
    v3 = demo.window.instance
    prep = V5.prepare(demo.setup)

    def runs(*of):
        return [[int(o == r or (isinstance(o, tuple) and r in o)) for r in range(3)] for o in of]

    honest_runs = runs(None, None, 0, None, 1, 0, 0, 0)
    honest_fresh = [0, 0, 1, 0, 1, 1, 1, 0]

    def case(r=honest_runs, fresh=honest_fresh, **kw):
        return V5.convert(v3, prep, 3, runs=r, fresh=fresh, **kw)

    def with_fresh(t, b):
        x = honest_fresh[:]
        x[t] = b
        return x

    return {
        "honest": (case(), True),
        "row-in-a-run-of-its-own": (case(runs(None, None, 2, None, 1, 0, 0, 0)), True),
        "repeat-in-a-run-of-its-own": (case(runs(None, None, 0, None, 1, 0, 0, 2), with_fresh(7, 1)), True),
        "kind-0-read-in-a-run": (case(runs(0, None, 0, None, 1, 0, 0, 0), with_fresh(0, 1)), True),
        "a-kind-0-reads-row-is-anything": (case(rows={1: random.Random(1).randbytes(16)}), True),
        "a-skipped-row": (case(runs(None, None, 0, None, 1, 2, 0, 0)), False),
        "a-repeat-of-another-row": (case(fresh=with_fresh(6, 0)), False),
        "a-row-fresh-twice": (case(fresh=with_fresh(7, 1)), False),
        "another-units-row": (case(runs(None, None, 0, None, 0, 0, 0, 0)), False),
        "a-read-in-no-run": (case(runs(None, None, None, None, 1, 0, 0, 0)), False),
        "a-read-in-two-runs": (case(runs(None, None, (0, 2), None, 1, 0, 0, 0)), False),
        "a-wrong-start": (case(starts={0: 1}), False),
        "a-wrong-unit": (case(run_units={1: 0}), False),
        "a-slot-past-the-tree": (case(starts={1: 1}), False),
        "a-first-member-not-fresh": (case(fresh=with_fresh(2, 0)), False),
        "a-wrong-head": (case(heads={0: prep.units.trees["u0"].reg.rows[1]}), False),
    }


def test_a_run_witness_that_cheats_is_refused(demo):
    cases = cheats(demo)
    assert {k: V5.reference(x) for k, (x, _) in cases.items()} == {k: want for k, (_, want) in cases.items()}


def test_v4_takes_a_read_in_two_runs_and_v5_doesnt(demo):
    """v4 opens a read's one row in every run it names; v5 takes its window from its run's rows, so a kind-1 read is
    in exactly one run (`N3RunsOne`). The window's verdict doesn't change: a read's runs are the prover's."""
    two = [[0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
    x = V4.convert(demo.window.instance, demo.setup.layout.tree, demo.setup.units, 3, runs=two,
                   fresh=[0, 0, 1, 0, 1, 1, 1, 0])
    assert V4.reference(x) and not V5.reference(cheats(demo)["a-read-in-two-runs"][0])


# --- the simulator and the lab ----------------------------------------------------------------------------------------

def agree(setup: U.Setup, windows) -> list[tuple[str, int, bool]]:
    """Each window with a unit: its port, window and v4's verdict, after checking v5's is the same."""
    out = []
    for w in windows:
        if w.instance is not None:
            a, b = V4.reference(V4.instance(setup, w)), V5.reference(V5.instance(setup, w))
            assert a == b, (w.port, w.window, a, b)
            out.append((w.port, w.window, a))
    return out


def test_v5_accepts_exactly_where_v4_does_on_the_simulators_windows():
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
def test_v5_accepts_exactly_where_v4_does_on_the_labs_windows(lab, edit):
    got = agree(*lab_windows(lab, edit))
    assert got and all(ok for _, _, ok in got) == (edit is None)


@pytest.fixture(scope="module")
def salted_lab():
    return lab_setup(SALTED_B)


def test_v5_accepts_exactly_where_v4_does_on_salted_lab_windows(salted_lab):
    salts = payloads(salted_lab)
    for s, ok in ((salts, True), ({}, False), (salts | {(BLOCK, 0): OTHER}, False)):
        got = agree(*lab_windows(salted_lab, salts=s))
        assert all(x for _, _, x in got) == ok


@pytest.mark.slow
def test_v5_accepts_exactly_where_v4_does_on_a_lab_whose_outboxes_are_two_rows_deep():
    """At width 8 over 32 requests each unit's outbox is two of v4's rows or more and three of v5's, and later
    windows' runs start past row 0: the one test of whole windows whose v5 runs open at a private start past 0 with
    their heads. About 80 s, most of it committing the larger MLP."""
    lab = lab_setup(LAB_B, d=8, requests=32)
    for edit, ok in ((None, True), (lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:], i), False)):
        setup, windows = lab_windows(lab, edit)
        assert all(x >= 1 for x in V5.prepare(setup).layout.fmt.DEPTHS)
        assert any(V5.instance(setup, w).args["run_starts"][0] for w in windows if w.instance is not None)
        got = agree(setup, windows)
        assert got and all(x for _, _, x in got) == ok


# --- templates --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canonical():
    return V5.canonical_demo()


@pytest.fixture(scope="module")
def canonical_cases(canonical) -> list[tuple[str, U.Setup, U.Window, bool]]:
    """Each canonical window, honest and with its first frame's header, data or suffix byte flipped before the
    certifier records it."""
    out = []
    for w in canonical.windows:
        out.append((f"{w.port}-honest", canonical.setup, w, True))
        for name, at in (("header", 5), ("data", 9), ("suffix", 14 if w.port == "p0" else 13)):
            c = V5.canonical_demo(flipped=(w.port, 0, at))
            out.append((f"{w.port}-{name}", c.setup, next(x for x in c.windows if x.port == w.port), False))
    return out


def test_a_template_accepts_exactly_where_the_descriptors_placement_does_on_canonical_frames(canonical,
                                                                                            canonical_cases):
    for name, s, w, ok in canonical_cases:
        tpl = canonical.templates[w.port]
        got = (V4.reference(V4.instance(s, w)), V5.reference(V5.instance(s, w)),
               V5.reference(V5.instance(s, w, tpl=tpl)))
        assert got == (ok, ok, ok), name


def test_a_template_refuses_a_frame_off_it_that_its_descriptor_holds():
    c = V5.canonical_demo(off_template=True)
    p0, p1 = c.windows
    assert V4.reference(V4.instance(c.setup, p0)) and V5.reference(V5.instance(c.setup, p0))
    assert not V5.reference(V5.instance(c.setup, p0, tpl=c.templates["p0"]))
    assert V5.reference(V5.instance(c.setup, p1, tpl=c.templates["p1"]))


def place_cases(f: U.Format, tpl: tuple, seed: int, FB: int) -> list[tuple[int, bytes, bytes, int]]:
    """(r, stream, frame, want) for N3PlaceFixed: random streams at every r below 2^LB, each with its placement in a
    frame of FB bytes and with a byte of it flipped."""
    rng = random.Random(seed)
    one = U.Segment(1, 0, 0, 0, f.B, 0)
    out = []
    for r in range(1 << f.LB):
        stream = rng.randbytes(f.B)
        desc = U.encode(f, U.Descriptor(0, 0, r, 0, b"", b"", (one,)))
        frame = V5.assemble_fixed(f, tpl, desc, [stream + bytes(f.B)], 1, FB)
        out += [(r, stream, frame, 1), (r, stream, flip(frame, rng.randrange(FB), 1 << rng.randrange(8)), 0)]
    return out


NARROW = 11
"""Lever 4's rows in the canonical demo: 11 bytes under its frames of 16, p0's frame 2 reading u0's bytes 20 ... 22
across rows 1 and 2 (the format's B is more than a header's 9 bytes, so 10 is the narrowest)."""


def fmt_at(canonical, RB: int | None) -> U.Format:
    return V5.prepare(canonical.setup, RB).layout.fmt


@pytest.mark.parametrize("port", ["p0", "p1"])
@pytest.mark.parametrize("RB", [None, NARROW, 10])
def test_n3placefixed_is_its_reference(canonical, port, RB):
    f, tpl, FB = fmt_at(canonical, RB), canonical.templates[port], canonical.setup.layout.fmt.B
    cases = place_cases(f, tpl, ord(port[1]), FB)
    args = [[U._int_bits(r, f.LB) for r, *_ in cases], [bits(s, 8 * f.B) for _, s, _, _ in cases],
            [bits(x, 8 * FB) for _, _, x, _ in cases]]
    out = evaluate_bits(bind(V5.N3PlaceFixed, FMT=tuple(f), TPL=tpl, FB=FB),
                        *(np.asarray(a, dtype=np.uint8) for a in args))
    assert np.asarray(out).reshape(-1).tolist() == [w for *_, w in cases]


def test_n3placefixed_on_a_stream_of_b_is_assembles_placement(canonical):
    """With L = FB, `place_fixed` is `n3_unit.assemble`'s placement, so the reference at FB = B is v5's of rows of B."""
    f, rng = canonical.setup.layout.fmt, random.Random(7)
    one = U.Segment(1, 0, 0, 0, f.B, 0)
    for port, tpl in canonical.templates.items():
        mode, p, s, pre, suf = V5.template(tpl, f.B)
        wide = f._replace(P=max(f.P, p, len(pre), 1), Q=max(f.Q, len(suf), 1))
        for r in range(1 << f.LB):
            stream = rng.randbytes(f.B)
            desc = U.encode(wide, U.Descriptor(mode, p, r, s, pre, suf, (one,)))
            assert V5.place_fixed(tpl, f.B, r, stream) == U.assemble(wide, desc, [stream + bytes(f.B)], 1), (port, r)


@pytest.mark.parametrize("G", [1, 2])
@pytest.mark.parametrize("RB", [None, NARROW])
def test_n3framefixed_is_assemble_fixed(canonical, G, RB):
    """Random descriptors and rows under the RAW template: the frame the template puts on them, and that frame with a
    bit flipped, and a descriptor with a segment past G of kind 1."""
    f, tpl, rng = fmt_at(canonical, RB), canonical.templates["p0"], random.Random(G)
    FB = canonical.setup.layout.fmt.B
    cases = []
    for i in range(24):
        x = bytearray(rng.randbytes(-(-f.LW // 8)))
        if i % 3:
            for g in range(G, f.G):
                x[(f.HW + g * f.SW) // 8] &= ~(1 << (f.HW + g * f.SW) % 8)
        x = bytes(x)
        rows = [rng.randbytes(2 * f.B) for _ in range(G)]
        frame = V5.assemble_fixed(f, tpl, x, rows, G, FB)
        cases += [(x, rows, frame), (x, rows, flip(frame, rng.randrange(FB), 1 << rng.randrange(8)))]
    want = [int(not any(s.kind for s in U.fields(f, x).segs[G:]) and V5.assemble_fixed(f, tpl, x, rows, G, FB) == frame)
            for x, rows, frame in cases]
    assert 0 in want and 1 in want
    args = [[bits(x, f.LW) for x, _, _ in cases], [[b for r in rr for b in bits(r, 16 * f.B)] for _, rr, _ in cases],
            [bits(fr, 8 * FB) for _, _, fr in cases]]
    out = evaluate_bits(bind(V5.N3FrameFixed, FMT=tuple(f), G=G, TPL=tpl, FB=FB),
                        *(np.asarray(a, dtype=np.uint8) for a in args))
    assert np.asarray(out).reshape(-1).tolist() == want


# --- rows narrower than frames ----------------------------------------------------------------------------------------

def test_a_descriptor_on_narrower_rows_is_its_segments_bytes_from_those_rows(canonical):
    prep, units = V5.prepare(canonical.setup, NARROW), canonical.setup.units
    f, L = prep.layout.fmt, prep.layout
    assert (f.B, f.P, f.Q, f.G) == (NARROW, 1, 1, 2)
    assert [[(s.u, s.q, s.o, s.n, s.d) for s in d.segs] for d in L.descriptors["p0"]] == [
        [(0, 1, 3, 5, 0)], [(0, 3, 0, 3, 0)], [(1, 0, 0, 2, 0), (0, 1, 9, 3, 2)]]
    for u, name in enumerate(units.names):
        assert b"".join(prep.units.trees[name].reg.rows).rstrip(b"\0") == units.data[name].rstrip(b"\0")
    for p, ds in L.descriptors.items():
        for d, old in zip(ds, canonical.setup.layout.descriptors[p]):
            got = b"".join(b"".join(prep.units.trees[units.names[s.u]].reg.rows)[NARROW * s.q + s.o:][:s.n]
                           for s in d.segs)
            want = b"".join(units.data[units.names[s.u]][16 * s.q + s.o:][:s.n] for s in old.segs if s.kind)
            assert d.r == old.r and got == want


def test_rows_narrower_than_a_frames_data_or_with_no_template_are_refused(canonical):
    with pytest.raises(ValueError, match="doesn't fit rows of 4"):
        V5.resplit(U.Descriptor(r=5, segs=(U.Segment(1, 0, 0, 3, 5, 0),)), 16, 4)
    with pytest.raises(ValueError, match="take a template"):
        V5.reference(V5.instance(canonical.setup, canonical.windows[0], RB=NARROW))


@pytest.fixture(scope="module")
def narrow_cases(canonical_cases) -> list[tuple[str, U.Setup, U.Window, bool]]:
    """The canonical cases, and p0's frame 2 with its byte from each row of its crossing segment flipped (frame bytes
    11 and 13: u0's bytes 20 and 22, row 1's last byte and row 2's first on rows of 11)."""
    out = list(canonical_cases)
    for name, at in (("row-1-of-the-crossing", 11), ("row-2-of-the-crossing", 13)):
        c = V5.canonical_demo(flipped=("p0", 2, at))
        out.append((f"p0-{name}", c.setup, c.windows[0], False))
    return out


def test_narrow_rows_accept_exactly_where_rows_of_b_do_on_canonical_frames(canonical, narrow_cases):
    for name, s, w, ok in narrow_cases:
        tpl = canonical.templates[w.port]
        got = (V4.reference(V4.instance(s, w)), V5.reference(V5.instance(s, w, tpl=tpl)),
               *(V5.reference(V5.instance(s, w, tpl=tpl, RB=rb)) for rb in (NARROW, 10, 12)))
        assert got == (ok,) * 5, name


def test_narrow_rows_refuse_a_frame_off_the_template():
    c = V5.canonical_demo(off_template=True)
    p0, p1 = c.windows
    assert not V5.reference(V5.instance(c.setup, p0, tpl=c.templates["p0"], RB=NARROW))
    assert V5.reference(V5.instance(c.setup, p1, tpl=c.templates["p1"], RB=NARROW))


def narrow_cheats(canonical) -> dict[str, tuple[U.Instance, bool]]:
    """p0's window on rows of 11 and witnesses a prover could give: honest; read 5's row (row 2 of u0, which the
    crossing segment ends in) with the byte it reads flipped; and run 0's head (u0's row 1) another row."""
    w, tpl = canonical.windows[0], canonical.templates["p0"]
    prep = V5.prepare(canonical.setup, NARROW)
    honest = V5.instance(canonical.setup, w, tpl=tpl, RB=NARROW)
    u0 = prep.units.trees["u0"].reg.rows
    assert honest.args["run_units"][0] == 0 and honest.args["run_starts"][0] == 1

    def case(**kw):
        return V5.convert(w.instance, prep, honest.R, tpl, G=honest.G, **kw)

    return {"honest": (case(), True), "the-crossings-second-row": (case(rows={5: flip(u0[2], 0)}), False),
            "a-wrong-head": (case(heads={0: u0[2]}), False)}


def test_a_narrow_row_witness_that_cheats_is_refused(canonical):
    cases = narrow_cheats(canonical)
    assert {k: V5.reference(x) for k, (x, _) in cases.items()} == {k: want for k, (_, want) in cases.items()}


# --- the circuit ------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_demo_window_is_its_reference(demo):
    """The whole v5 window at the demo's format: the demo's cases and the crossing segment's tampers with R = 2, then
    the run witnesses above with R = 3. It is the one test of v5's wiring that the reference tests can't catch: each
    read's next position, run and carry, its run's two rows and its window, the head's slot before the compacted fresh
    leaves, and each run's range, root and fit. About 70 s, most of it tracing and evaluating the windows' SHA-512
    compressions."""
    cases = demo_cases(demo)
    insts = [pair(s, x)[1] for s, x in cases.values()]
    assert V5.evaluate(*insts) == [int(V5.reference(x)) for x in insts] == [int(k == "honest") for k in cases]
    got = cheats(demo)
    assert V5.evaluate(*(x for x, _ in got.values())) == [int(want) for _, want in got.values()]


@pytest.mark.slow
def test_a_salted_window_is_its_reference(salted_lab):
    """The salted lab window (B = 240) as v5: honest, with a wrong salt payload, and with the salt frame's slot moved:
    the one test of v5 with the salt path, two segments and two runs (a request's frame reads the outside
    host's length and head, then the message). About 10 s after the demo window's test."""
    setup, windows = lab_windows(salted_lab, salts=payloads(salted_lab))
    w = next(w for w in windows if w.port == INGRESS and w.window == 0)
    honest = V5.instance(setup, w)
    wrong = copy.deepcopy(honest)
    wrong.args["salt"] = wrong.args["salt"][:8] + OTHER
    moved = copy.deepcopy(honest)
    moved.args["slot"] = 1
    cases = [honest, wrong, moved]
    assert honest.salted and honest.R == 2
    assert V5.evaluate(*cases) == [int(V5.reference(x)) for x in cases] == [1, 0, 0]


@pytest.mark.slow
def test_the_canonical_windows_on_narrow_rows_are_their_reference(canonical, narrow_cases):
    """The canonical windows on rows of 11 under frames of 16, honest, tampered and with the witnesses that cheat: the
    one test of the window with frames wider than its rows (`N3FrameFixed` and the frames root at FB). About 40 s
    after the demo window's test."""
    for port in ("p0", "p1"):
        insts = [V5.instance(s, w, tpl=canonical.templates[port], RB=NARROW) for _, s, w, _ in narrow_cases
                 if w.port == port]
        want = [int(ok) for _, _, w, ok in narrow_cases if w.port == port]
        if port == "p0":
            got = narrow_cheats(canonical)
            insts += [x for x, _ in got.values()]
            want += [int(ok) for _, ok in got.values()]
        assert V5.evaluate(*insts) == [int(V5.reference(x)) for x in insts] == want


@pytest.mark.slow
def test_the_canonical_windows_are_their_reference(canonical, canonical_cases):
    """The canonical demo's two windows, with and without their templates, honest and tampered: the one test of the
    window with `N3FrameFixed`, RAW and BASE64. About 40 s after the demo window's test."""
    for port in ("p0", "p1"):
        ws = [(s, w, ok) for _, s, w, ok in canonical_cases if w.port == port]
        for tpl in ((), canonical.templates[port]):
            insts = [V5.instance(s, w, tpl=tpl) for s, w, _ in ws]
            assert V5.evaluate(*insts) == [int(V5.reference(x)) for x in insts] == [int(ok) for _, _, ok in ws]
