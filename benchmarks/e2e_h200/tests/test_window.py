"""The H200 window writer (`window`, CIR-115).

- At the frozen shape, FAMILIES' shared-row layout and its writers are circuits' artifacts.
- Every read a proof block's ports make lies in rows its producers wrote, and K and V are read at the steps that wrote them.
- A run served by continuous batching on the CPU at MINI: for proof blocks of every family drawn at every step, the ports
  are assembled from its committed rows (under 1 KiB rows, 64 B rows whose ports span rows, and today's ``families/v1``,
  whose slot roots are the run's own). They must be the inputs replay reads and the outputs it checks, and each row opens
  against its step's root.
"""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import run as RUN
from e2e_h200 import window as W
from e2e_h200.commit import Committer, _layout, _slots, fixture_salt, fixture_secrets, layout as families_layout
from e2e_h200.driver import Driver
from e2e_h200.shape import FROZEN, MINI, PV_K
from e2e_h200.slots import Request, SlotKV, SlotShape
from e2e_h200.store import GeneratorStore
from e2e_h200.units import Replayer
from e2e_h200.weights import Weights

#: art:8ce57586b2176a6364260a10963c9703b5b8919b26bd965a4f907667846323e9's writers, `window.writers_digest`
FROZEN_WRITERS = ("5f20869283c3b4c078193c42259d1f2aada1a82138e203a438216d4449c1fc6f"
                  "7a2f69952a2dddabbaab83e1d928551769c0b7b0bc26408b2f69686ef4cf63d4")
SLOTS, N, P, G, ADMIT = 4, 7, 2, 3, 1


def lockstep(steps: int, batch: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(slots=None, positions=np.full(batch, s)) for s in range(steps)]


@pytest.fixture(scope="module")
def frozen() -> W.Window:
    return W.Window(FROZEN, lockstep(FROZEN.context, 2))


def test_the_shared_layout_and_its_writers_are_circuits_at_the_frozen_shape(frozen) -> None:
    assert frozen.layout.row_count == W.FROZEN_ROWS and frozen.layout.digest.hex() == W.FROZEN_DIGEST
    ws = frozen.writers()
    assert len(ws) == 1203 and W.writers_digest(ws) == FROZEN_WRITERS


def _written(writers: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Each row's byte ranges its producers write (a quantizer's scale after its codes)."""
    out: dict[int, list[tuple[int, int]]] = {}
    for w in writers:
        for k in range(-(-w["units"] // w["G"])):
            n = min(w["G"], w["units"] - k * w["G"])
            out.setdefault(w["first"] + k, []).append((w["row_offset"], w["row_offset"] + n * w["window_bytes"]))
            if w["scale"]:
                out[w["first"] + k].append((w["row_bytes"] - 4, w["row_bytes"]))
    return out


def _inside(lo: int, hi: int, ranges: list[tuple[int, int]]) -> bool:
    for a, b in sorted(ranges):
        if a <= lo < b:
            lo = b
        if lo >= hi:
            return True
    return lo >= hi


def test_every_read_at_the_frozen_shape_is_in_rows_its_producers_wrote(frozen) -> None:
    ws = frozen.writers()
    written, by = _written(ws), {w["family"]: w for w in ws}
    step, slot = 7, 1
    for f in frozen.families():
        u = frozen.units[f]
        g = by[f]["G"]
        for i in sorted({0, g - 1, min(g, u.count - 1), u.count // 2, u.count - 1}):
            (codes, *scale) = frozen.outputs(step, slot, f, i)
            assert codes == (W.Segment(step, slot, by[f]["first"] + i // g, by[f]["row_offset"] + by[f]["window_bytes"] * (i % g),
                                       u.nbytes),), (f, i)
            for port in frozen.ports(f):
                for p in frozen.locate(step, slot, f, i, port):
                    if isinstance(p, W.Segment):
                        assert p.offset + p.nbytes <= frozen.row_bytes(p.row), (f, i, port, p)
                        assert _inside(p.offset, p.offset + p.nbytes, written[p.row]), (f, i, port, p)


def test_a_linears_x_is_its_quantizers_r_rows_and_attention_reads_k_and_v_at_their_steps(frozen) -> None:
    x = frozen.locate(7, 1, "l50.wdown", 11, "x7")
    rows = [frozen.locate(7, 1, "l50.wdown", 11, f"x{r}")[0].row for r in range(8)]
    first = frozen.outputs(7, 1, "l50.down_quant", 0)[0][0].row
    assert rows == list(range(first, first + 8)) and x == (W.Segment(7, 1, first + 7, 0, 8192),)
    assert frozen.locate(7, 1, "l50.wdown", 11, "sx") == (W.Segment(7, 1, first, 8192, 4),)
    assert frozen.locate(7, 1, "l50.wdown", 11, "wrow3") == (W.Registered("l50.wdown", 11, 3 * 8192, 8192),)
    k = frozen.locate(7, 1, "l3.softmax", 200, "k")
    kv = frozen.outputs(0, 0, "l3.wk", 0)[0][0].row
    assert k == tuple(W.Segment(j, 1, kv, 0, 128) for j in range(8)) + (W.Absent(128),) * (PV_K - 8)
    assert frozen.locate(7, 1, "l3.pv", 9, "v7") == (W.Segment(7, 1, kv + 1, 0, 128),)
    assert frozen.locate(7, 1, "l3.pv", 9, "v8") == (W.Absent(128),)
    with pytest.raises(ValueError, match="gather"):
        frozen.locate(7, 1, "l3.pv.gather", 0, "x")


def test_the_bundles_pieces_on_families_v1_at_the_frozen_shape_are_cmp_161s() -> None:
    """KVREADS's stub (CMP-161, 20:58Z) on the layout r20261026-195842-f3ca committed under: K and V are their gathers' rows,
    a linear's x its quantizer's chunk units, an output its own unit's row."""
    lay, bases = _layout(FROZEN)
    assert lay.digest.hex().startswith("1791efa8")
    win = W.Window(FROZEN, lockstep(FROZEN.context, 2), lay)
    read = lambda s, unit, rank, at, n, start=0: {"step": s, "slot": 1, "unit": unit, "rank": rank, "at": at,  # noqa: E731
                                                   "start": start, "nbytes": n}
    assert win.bundle(7, 1, "l3.softmax", 5, "k") == [read(t, bases["l3.wk.gather"], 0, 64 * t, 128) for t in range(8)]
    assert win.bundle(7, 1, "l3.softmax", 5, "n") == [{"offset": 1, "at": 0}]
    assert win.bundle(7, 1, "l3.softmax", 5, "q") == [read(7, bases["l3.wq.gather"] + 5, 0, 0, 128)]
    assert win.bundle(7, 1, "l3.pv", 5, "v7") == [read(7, bases["l3.wv.gather"], 0, 0, 128)]
    assert win.bundle(7, 1, "l3.pv", 5, "v8") == []
    assert win.bundle(7, 1, "l3.wdown", 9, "x5") == [read(7, bases["l3.down_quant"] + 5, 0, 0, 8192)]
    assert win.bundle(7, 1, "l3.wdown", 9, "sx") == [read(7, bases["l3.down_quant"], 1, 0, 4)]
    assert win.bundle(7, 1, "l3.wdown", 9, "wrow2") is None
    assert win.bundle(7, 1, "l3.wq", 5, "o0") == (bases["l3.wq"] + 5, 0, None)
    assert win.bundle(7, 1, "l3.attn_quant", 1, "o1") == (bases["l3.attn_quant"] + 1, 1, None)


def test_under_shared_rows_an_output_is_a_window_of_its_row(frozen) -> None:
    first = frozen.outputs(7, 1, "l3.wq", 0)[0][0].row
    assert frozen.bundle(7, 1, "l3.wq", 517, "o0") == [{"step": 7, "slot": 1, "unit": first + 1, "rank": 0, "at": 0,
                                                        "start": 10, "nbytes": 2}]


# -- the round trip at MINI

@pytest.fixture(scope="module")
def served():
    """A run served by continuous batching (slots reused, steps mixing positions) under the per-slot commit at MINI."""
    if not any(shutil.which(c) for c in ("g++", "c++", "clang++")):
        pytest.skip("the slots commit's host twin needs a C++ compiler")
    w = Weights(MINI)
    c = Committer("slots", *fixture_secrets("test"), MINI, "cpu", fixture_salt("test"))
    d = Driver(MINI, GeneratorStore(w), SLOTS, c, clock=lambda: 0.0)
    ps = RUN.prompts_for("serve", N, P)
    d.serve(SlotKV(SlotShape(SLOTS, MINI.layers, MINI.context, MINI.head_dim, prompt=P, generate=G), "cpu", admit=ADMIT),
            [Request(i, bytes(ps[i].astype(np.uint8))) for i in range(N)])
    win = W.Window(MINI, d.steps)
    assert any(r != b for s, st in enumerate(d.steps) for b in range(st.batch) for _, r in win.history(s, b)), \
        "a sequence's earlier keys are in other rows of their steps"
    return w, c, d


LAYOUTS = {"shared-1024": lambda: W.shared_layout(MINI), "shared-64": lambda: W.shared_layout(MINI, 64),
           "families-v1": lambda: families_layout(MINI)}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_drawn_blocks_of_every_family_assemble_from_committed_rows_to_what_replay_reads(served, layout) -> None:
    w, c, d = served
    S, _ = _slots()
    win = W.Window(MINI, d.steps, LAYOUTS[layout]())
    rows = {(s, b): win.slot_rows(st, b) for s, st in enumerate(d.steps) for b in range(st.batch)}
    if layout == "families-v1":
        assert all(S.commit_slot(c.context, win.layout, r.salt_key, b, rows[s, b]).root == r.slot_roots[b]
                   for s, r in enumerate(d.results) for b in range(d.steps[s].batch))
        assert len(win.locate(0, 0, "l0.softmax", 1, "q")) == 1, "a read opens the gather's row, not its parts'"
    tokens = [r.tokens_in for r in d.results]
    rp = Replayer(MINI, w, d.steps, tokens)
    run, weights = win.run_inputs(tokens), W.weights_of(w)
    spans = 0
    for s, st in enumerate(d.steps):
        units = rp.draw(s, 1, b"window", only=lambda f: not f.template.startswith("AllGather_v1"))
        assert {u.family for u in units} == set(win.families())
        for u in units:
            b = st.row_of(u.slot)
            got = win.inputs(s, b, u.family, u.index, lambda *k: rows[k], weights, run)
            ports = win.ports(u.family)
            if u.family.endswith(".pv"):
                assert got["n"] == (u.position + 1).to_bytes(2, "little")
                ports = ports[:-1]
            assert b"".join(got[p] for p in ports) == b"".join(np.ascontiguousarray(a).tobytes() for a in u.inputs), \
                (layout, s, u.family, u.index)
            outs = win.outputs(s, b, u.family, u.index)
            assert b"".join(win.assemble(o, lambda *k: rows[k], weights, run, s, b) for o in outs) == u.got.tobytes()
            spans += sum(len(win.locate(s, b, u.family, u.index, p)) > 1 for p in ports)
    assert layout != "shared-64" or spans


def test_a_drawn_attention_blocks_rows_open_against_their_steps_roots(served) -> None:
    w, c, d = served
    S, _ = _slots()
    win = W.Window(MINI, d.steps)
    s = len(d.steps) - 1
    st = d.steps[s]
    b = int(np.argmax(st.positions))
    assert st.positions[b] > 0, "the block reads keys of earlier steps"
    com, keys = {}, set()
    for p in win.locate(s, b, "l0.softmax", 1, "k") + win.locate(s, b, "l0.softmax", 1, "q"):
        if isinstance(p, W.Segment):
            keys.add((p.step, p.slot))
            if p.step not in com:
                salt = fixture_salt("test")(p.step)
                com[p.step] = (salt, S.commit(c.context, win.layout, p.step, salt, win.tensors(d.steps[p.step]),
                                              d.steps[p.step].batch))
            salt, sc = com[p.step]
            rows = win.slot_rows(d.steps[p.step], p.slot)
            unit, rank = win.unit_of(p.row)
            o = S.open_unit(c.context, win.layout, p.step, salt, sc.slot_roots, p.slot, unit, rows)
            base = win.layout.row_base[unit]
            mine = {r: (rows[base + r], S.row_salts(salt, p.slot, base + r, 1)) for r in range(len(win.layout.units[unit]))}
            assert S.verify_unit(sc.root, c.context, win.layout, p.step, d.steps[p.step].batch, o, mine)
            bad = dict(mine)
            bad[rank] = (bytes(len(mine[rank][0])), mine[rank][1])
            assert not S.verify_unit(sc.root, c.context, win.layout, p.step, d.steps[p.step].batch, o, bad)
    assert len({k for k, _ in keys}) == st.positions[b] + 1 and {r for _, r in keys} != {b}
