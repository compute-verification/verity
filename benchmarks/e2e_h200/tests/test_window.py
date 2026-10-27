"""The H200 window writer (`window`, CIR-115).

- At the frozen shape, FAMILIES' shared-row layout and its writers are circuits' artifacts.
- Every read a proof block's ports make lies in rows its producers wrote, and K and V are read at the steps that wrote them.
- A linear's weight reads are whole rows of the weights' registration (CMP-160).
- A run served by continuous batching on the CPU at MINI: for proof blocks of every family drawn at every step, the ports
  are assembled from its committed rows (under 1 KiB rows, 64 B rows whose ports span rows, and today's ``families/v1``,
  whose slot roots are the run's own). They must be the inputs replay reads and the outputs it checks, and each row opens
  against its step's root.
- All of it again on the map with the ``residual`` fusion (`plan.FUSIONS`), whose frozen layout drops the projections' rows,
  with ``norm`` as well, whose layout also drops each norm's rows, and with ``pv`` too, whose layout drops each layer's
  softmax rows (the round trip at MINI runs the unfused map, ``residual`` alone and all three).
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
from verity_torch_loop import weights as SW

#: art:8ce57586b2176a6364260a10963c9703b5b8919b26bd965a4f907667846323e9's writers, `window.writers_digest`
FROZEN_WRITERS = ("5f20869283c3b4c078193c42259d1f2aada1a82138e203a438216d4449c1fc6f"
                  "7a2f69952a2dddabbaab83e1d928551769c0b7b0bc26408b2f69686ef4cf63d4")
RESIDUAL = frozenset({"residual"})
#: the frozen shared layout under the ``residual`` fusion: rows, bytes a slot, `layout_digest` and its writers' digest
FUSED_ROWS, FUSED_BYTES = 18_529, 24_008_260
FUSED_DIGEST = ("6d4651773566e0caf2aad9c47748bf109da83a5dff3889ee2ee11721f51f41af"
                "4894fc249aa52f98dd974c1bc9d3eccd5ecadbfff1e70767b4bc4b9095211de8")
FUSED_WRITERS = ("59b05a362a01f35542a56228db5f37dc1172e956168562e4990ced295e0342b8"
                 "fbda60c5e1f52cfb19ca927699cf472697e5ac0c71ab9cbd09ed5ff6eeb317f9")
BOTH = RESIDUAL | {"norm"}
#: the frozen shared layout under the ``norm`` and ``residual`` fusions, as above
BOTH_ROWS, BOTH_BYTES = 15_169, 20_567_620
BOTH_DIGEST = ("f8928310f5007a766de4462f896d4f29e397e1fe18b2cd8f5334c866f0c08dda"
               "4c52246ea42289624f4e88c2d6999ead73da87f26d1a0a32986b163913b0490f")
BOTH_WRITERS = ("de53750b79bf349528ab6dd2e91e0d93ac0c52f23bcb007a4b5f64fd2e800920"
                "e3b766c593f9014540fd5deedb5f0556e9a5cca91e498fc63770e14996b0ff3d")
ALL = BOTH | {"pv"}
#: the frozen shared layout under the ``norm``, ``pv`` and ``residual`` fusions, as above
ALL_ROWS, ALL_BYTES = 14_337, 19_715_652
ALL_DIGEST = ("6c59619cd55b8a211d9211b6f68efa960760e6ad2931d0ce4661b6fda55c861a"
              "c8046308abbd1ad8353c131e52667ce9b99683f53f188bf78af841b17324dba3")
ALL_WRITERS = ("3641ecdd02908ae0873a3e02c4a50990b8568bb3f5080cc0319309faee4750e6"
               "ee18dc576d578424c7b13c0df53e3f5328c43fcd9c6b6872e619d848ec8c3e22")
SLOTS, N, P, G, ADMIT = 4, 7, 2, 3, 1


def lockstep(steps: int, batch: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(slots=None, positions=np.full(batch, s)) for s in range(steps)]


@pytest.fixture(scope="module")
def frozen() -> W.Window:
    return W.Window(FROZEN, lockstep(FROZEN.context, 2))


@pytest.fixture(scope="module")
def frozen_fused() -> W.Window:
    return W.Window(FROZEN, lockstep(FROZEN.context, 2), fused=RESIDUAL)


@pytest.fixture(scope="module")
def frozen_both() -> W.Window:
    return W.Window(FROZEN, lockstep(FROZEN.context, 2), fused=BOTH)


@pytest.fixture(scope="module")
def frozen_all() -> W.Window:
    return W.Window(FROZEN, lockstep(FROZEN.context, 2), fused=ALL)


def test_the_shared_layout_and_its_writers_are_circuits_at_the_frozen_shape(frozen) -> None:
    assert frozen.layout.row_count == W.FROZEN_ROWS and frozen.layout.digest.hex() == W.FROZEN_DIGEST
    ws = frozen.writers()
    assert len(ws) == 1203 and W.writers_digest(ws) == FROZEN_WRITERS


def test_the_residual_fusion_drops_the_projections_tables_and_the_residual_adds_at_the_frozen_shape(frozen, frozen_fused) -> None:
    """Each layer's ``wo`` and ``wdown`` tables go (32 KiB, 32 rows each) and with them the two ``Bf16AddRow_v2`` families;
    the projections' coordinates write the residuals' rows instead."""
    lay, L = frozen_fused.layout, FROZEN.layers
    assert (lay.row_count, lay.slot_bytes, lay.digest.hex()) == (FUSED_ROWS, FUSED_BYTES, FUSED_DIGEST)
    assert frozen.layout.row_count - lay.row_count == 64 * L and frozen.layout.slot_bytes - lay.slot_bytes == 65_536 * L
    ws = frozen_fused.writers()
    assert len(ws) == 1203 - 2 * L and W.writers_digest(ws) == FUSED_WRITERS
    assert {u.template for u in frozen.units.values()} - {u.template for u in frozen_fused.units.values()} == {"Bf16AddRow_v2"}
    for f, table, k in (("l50.wo", "l50.attn_residual", 16384), ("l50.wdown", "l50.mlp_residual", 65536)):
        u = frozen_fused.family(f)
        assert (u.table, u.count, u.key) == (table, FROZEN.width, f"ScaledMmS8ResidualCoordinateRows_v1{{K={k},R={k // 8192},"
                                                                 f"DOT=BlackwellS8ImmaDot32_v1}}")
        assert f in frozen.tables and f not in frozen_fused.tables
    out = lambda f, i: frozen_fused.outputs(7, 1, f, i)[0]  # noqa: E731
    assert frozen_fused.locate(7, 1, "l50.wo", 11, "r") == out("l49.wdown", 11)
    assert frozen_fused.locate(7, 1, "l50.wdown", 11, "r") == out("l50.wo", 11)
    assert frozen_fused.locate(7, 1, "l50.wdown", 11, "wrow3") == (W.Registered("l50.wdown", 11 * 8 + 3, 0, 8192),)


def test_the_norm_fusion_drops_each_norms_table_and_its_quantizer_reads_the_stream_at_the_frozen_shape(frozen_fused,
                                                                                                      frozen_both) -> None:
    """Each norm's output table goes (32 KiB, 32 rows) and with it its ``RmsNormOutSums_v1`` family; the amax and quantizer
    after it read the stream, the registered gain and the sums."""
    lay, norms = frozen_both.layout, 2 * FROZEN.layers + 1
    assert (lay.row_count, lay.slot_bytes, lay.digest.hex()) == (BOTH_ROWS, BOTH_BYTES, BOTH_DIGEST)
    assert frozen_fused.layout.row_count - lay.row_count == 32 * norms
    assert frozen_fused.layout.slot_bytes - lay.slot_bytes == 32_768 * norms
    ws = frozen_both.writers()
    assert len(ws) == len(frozen_fused.writers()) - norms and W.writers_digest(ws) == BOTH_WRITERS
    assert ({u.template for u in frozen_fused.units.values()} - {u.template for u in frozen_both.units.values()}
            == {"RmsNormOutSums_v1"})
    statics = f"S=128,G=2,N={FROZEN.width},EPS=1e-05}}"
    for f, count, key in (("l50.attn_amax", 1, f"RmsNormAbsMax_v1{{C={FROZEN.width},{statics}"),
                          ("l50.up_quant", 2, f"RmsNormQuantChunk_v1{{C=8192,{statics}"),
                          ("head_quant", 2, f"RmsNormQuantChunk_v1{{C=8192,{statics}")):
        u = frozen_both.family(f)
        assert (u.table, u.count, u.key) == (f, count, key)
    for t in ("l50.attn_norm", "l50.mlp_norm", "final_norm"):
        assert t in frozen_fused.tables and t not in frozen_both.tables
    at = lambda f, *a: frozen_both.locate(7, 1, f, *a)  # noqa: E731
    first = frozen_both.outputs(7, 1, "l49.wdown", 8192)[0][0].row
    assert at("l50.attn_quant", 1, "x") == tuple(W.Segment(7, 1, first + r, 0, 1024) for r in range(16))
    assert at("l50.attn_quant", 1, "w") == tuple(W.Registered("l50.attn_norm", 8192 + j, 0, 2) for j in range(8192))
    sums = frozen_both.outputs(7, 1, "l50.attn_sums", 0)[0][0].row
    assert at("l50.attn_quant", 1, "s") == at("l50.attn_amax", 0, "s") == (W.Segment(7, 1, sums, 0, 512),)
    assert at("l50.attn_quant", 1, "amax") == frozen_both.outputs(7, 1, "l50.attn_amax", 0)[0]
    assert len(at("l50.attn_amax", 0, "x")) == len(at("l50.attn_amax", 0, "w")) // 512 == 32


def test_the_pv_fusion_drops_each_layers_softmax_and_its_head_reads_q_k_and_v_at_the_frozen_shape(frozen_both,
                                                                                                   frozen_all) -> None:
    """Each layer's softmax table goes (16 KiB, 16 rows) and with it the ``QkSoftmaxRowPad_v1`` family; the PV head reads its
    q row, the key and value rows at their steps and n, and writes its 128 B of the pv row."""
    lay, L = frozen_all.layout, FROZEN.layers
    assert (lay.row_count, lay.slot_bytes, lay.digest.hex()) == (ALL_ROWS, ALL_BYTES, ALL_DIGEST)
    assert frozen_both.layout.row_count - lay.row_count == 16 * L
    assert frozen_both.layout.slot_bytes - lay.slot_bytes == 16_384 * L
    ws = frozen_all.writers()
    assert len(ws) == len(frozen_both.writers()) - L and W.writers_digest(ws) == ALL_WRITERS
    assert ({u.template for u in frozen_both.units.values()} - {u.template for u in frozen_all.units.values()}
            == {"QkSoftmaxRowPad_v1", "PvRow_v1"})
    u = frozen_all.family("l3.pv")
    assert (u.table, u.count, u.key) == ("l3.pv", FROZEN.heads, f"QkSoftmaxPvPad_v1{{T={PV_K},K={PV_K},D={FROZEN.head_dim},"
                                                                f"DOT=HopperBF16WgmmaDot16_v2}}")
    assert "l3.softmax" in frozen_both.tables and "l3.softmax" not in frozen_all.tables
    assert frozen_all.ports("l3.pv") == ("q", "k") + tuple(f"v{j}" for j in range(PV_K)) + ("n",)
    at = lambda port: frozen_all.locate(7, 1, "l3.pv", 9, port)  # noqa: E731
    d = FROZEN.head_dim
    q = frozen_all.outputs(7, 1, "l3.wq", 9 * d)[0][0]
    kv = frozen_all.outputs(0, 0, "l3.wk", 0)[0][0].row
    assert at("q") == (W.Segment(7, 1, q.row, q.offset, 2 * d),) and at("n") == (W.Run("n"),)
    assert at("k") == tuple(W.Segment(j, 1, kv, 0, 2 * d) for j in range(8)) + (W.Absent(2 * d),) * (PV_K - 8)
    assert at("v7") == (W.Segment(7, 1, kv + 1, 0, 2 * d),) and at("v8") == (W.Absent(2 * d),)
    ((out,),) = frozen_all.outputs(7, 1, "l3.pv", 9)
    assert (out.offset, out.nbytes) == (2 * d * 9 % 1024, 2 * d)


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


@pytest.mark.parametrize("which", ["frozen", "frozen_fused", "frozen_both", "frozen_all"])
def test_every_read_at_the_frozen_shape_is_in_rows_its_producers_wrote(which, request) -> None:
    frozen = request.getfixturevalue(which)
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
    assert frozen.locate(7, 1, "l50.wdown", 11, "wrow3") == (W.Registered("l50.wdown", 11 * 8 + 3, 0, 8192),)
    k = frozen.locate(7, 1, "l3.softmax", 200, "k")
    kv = frozen.outputs(0, 0, "l3.wk", 0)[0][0].row
    assert k == tuple(W.Segment(j, 1, kv, 0, 128) for j in range(8)) + (W.Absent(128),) * (PV_K - 8)
    assert frozen.locate(7, 1, "l3.pv", 9, "v7") == (W.Segment(7, 1, kv + 1, 0, 128),)
    assert frozen.locate(7, 1, "l3.pv", 9, "v8") == (W.Absent(128),)
    with pytest.raises(ValueError, match="gather"):
        frozen.locate(7, 1, "l3.pv.gather", 0, "x")


def test_a_linears_registered_reads_at_the_frozen_shape_are_whole_rows_of_cmp_160s_registration(frozen) -> None:
    """CMP-160 registered the frozen weights as 733 values of 25,284,864 rows (r20261026-204952-3496), a linear's ``wrow{r}``
    of coordinate n its row n R + r and ``sw`` its scale's row n."""
    ts = SW.tensors(FROZEN)
    assert (len(ts), sum(t.leaves for t in ts)) == (733, 25_284_864)
    rb = {t.name: 2 * t.words for t in ts}
    linears = [f for f in frozen.families() if frozen.units[f].template == "ScaledMmS8CoordinateRows_v1"]
    assert len(linears) == 6 * FROZEN.layers + 1
    for f in linears:
        u = frozen.units[f]
        R = sum(p.startswith("wrow") for p in frozen.ports(f))
        for n in sorted({0, u.count // 2, u.count - 1}):
            for r in range(R):
                assert frozen.locate(7, 1, f, n, f"wrow{r}") == (W.Registered(u.table, n * R + r, 0, rb[u.table]),)
            assert frozen.locate(7, 1, f, n, "sw") == (W.Registered(u.table + ".scale", n, 0, 4),)
    assert frozen.locate(7, 1, "embed", 9, "col") == (W.Registered("tok", 9, 0, 2 * FROZEN.vocab),)
    assert frozen.locate(7, 1, "embed", 9, "pos") == (W.Registered("pos", 9 * FROZEN.context + 7, 0, 2),)


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

@pytest.fixture(scope="module", params=[frozenset(), RESIDUAL, ALL], ids=["unfused", "fused-residual", "fused-norm-pv-residual"])
def served(request):
    """A run served by continuous batching (slots reused, steps mixing positions) under the per-slot commit at MINI, on the
    unfused map, on the ``residual`` fusion's and on all three fusions'."""
    if not any(shutil.which(c) for c in ("g++", "c++", "clang++")):
        pytest.skip("the slots commit's host twin needs a C++ compiler")
    fused = request.param
    w = Weights(MINI)
    c = Committer("slots", *fixture_secrets("test"), MINI, "cpu", fixture_salt("test"), fused=fused)
    d = Driver(MINI, GeneratorStore(w), SLOTS, c, clock=lambda: 0.0)
    ps = RUN.prompts_for("serve", N, P)
    d.serve(SlotKV(SlotShape(SLOTS, MINI.layers, MINI.context, MINI.head_dim, prompt=P, generate=G), "cpu", admit=ADMIT),
            [Request(i, bytes(ps[i].astype(np.uint8))) for i in range(N)])
    win = W.Window(MINI, d.steps, fused=fused)
    assert any(r != b for s, st in enumerate(d.steps) for b in range(st.batch) for _, r in win.history(s, b)), \
        "a sequence's earlier keys are in other rows of their steps"
    assert ("l0.wo" in d.steps[0].tables) == ("residual" not in fused)
    assert ("l0.attn_norm" in d.steps[0].tables) == ("norm" not in fused)
    assert ("l0.softmax" in d.steps[0].tables) == ("pv" not in fused)
    return w, c, d


LAYOUTS = {"shared-1024": lambda f: W.shared_layout(MINI, fused=f), "shared-64": lambda f: W.shared_layout(MINI, 64, f),
           "families-v1": lambda f: families_layout(MINI, f)}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_drawn_blocks_of_every_family_assemble_from_committed_rows_to_what_replay_reads(served, layout) -> None:
    w, c, d = served
    S, _ = _slots()
    win = W.Window(MINI, d.steps, LAYOUTS[layout](c.fused), c.fused)
    rows = {(s, b): win.slot_rows(st, b) for s, st in enumerate(d.steps) for b in range(st.batch)}
    if layout == "families-v1":
        assert all(S.commit_slot(c.context, win.layout, r.salt_key, b, rows[s, b]).root == r.slot_roots[b]
                   for s, r in enumerate(d.results) for b in range(d.steps[s].batch))
        head = "l0.pv" if "pv" in c.fused else "l0.softmax"
        assert len(win.locate(0, 0, head, 1, "q")) == 1, "a read opens the gather's row, not its parts'"
    tokens = [r.tokens_in for r in d.results]
    rp = Replayer(MINI, w, d.steps, tokens, fused=c.fused)
    run, weights = win.run_inputs(tokens), W.weights_of(w)
    spans = 0
    for s, st in enumerate(d.steps):
        units = rp.draw(s, 1, b"window", only=lambda f: not f.template.startswith("AllGather_v1"))
        assert {u.family for u in units} == set(win.families())
        for u in units:
            b = st.row_of(u.slot)
            got = win.inputs(s, b, u.family, u.index, lambda *k: rows[k], weights, run)
            ports = win.ports(u.family)
            if u.template.startswith("PvRow_v1"):
                assert got["n"] == (u.position + 1).to_bytes(2, "little")
                ports = ports[:-1]
            assert b"".join(got[p] for p in ports) == b"".join(np.ascontiguousarray(a).tobytes() for a in u.inputs), \
                (layout, s, u.family, u.index)
            outs = win.outputs(s, b, u.family, u.index)
            assert b"".join(win.assemble(o, lambda *k: rows[k], weights, run, s, b) for o in outs) == u.got.tobytes()
            assert u.ok, (layout, s, u.family, u.index)
            spans += sum(len(win.locate(s, b, u.family, u.index, p)) > 1 for p in ports)
    assert layout != "shared-64" or spans


def test_a_drawn_attention_blocks_rows_open_against_their_steps_roots(served) -> None:
    w, c, d = served
    S, _ = _slots()
    win = W.Window(MINI, d.steps, fused=c.fused)
    s = len(d.steps) - 1
    st = d.steps[s]
    b = int(np.argmax(st.positions))
    assert st.positions[b] > 0, "the block reads keys of earlier steps"
    com, keys = {}, set()
    head = "l0.pv" if "pv" in c.fused else "l0.softmax"
    for p in win.locate(s, b, head, 1, "k") + win.locate(s, b, head, 1, "q"):
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
