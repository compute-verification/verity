"""The served RMSNorms split at their scale (`pipeline.serving_templates`' ``PARTS``, PROVECUT's units): a vector-8 RMSNorm row,
plain or fused with its residual add (``RMSNormCudaVector8_v1``, ``RMSNormFusedCudaVector8_v1``/``_v3``), a DSA norm row
(``DsaRMSNorm_v1``) and the norm of the fused MNNVL all-reduce (``AllReduceRmsNormMnnvl_v1``) each hold its sum of squares once
(the ``part`` layout: the row -> its scale) and its apply at each halving G of N, N / G blocks (the ``cols`` layout: block j reads
columns [jG, (j + 1)G) of the row and its weights and the scale whole, and returns those columns of the row's outputs).  Serving
stores no scale, so it is an intermediate: the readers compute it by its twin (`part_rows.MIDS`), held to the row's served
outputs, and the window wires each apply block's scale row to its row's sum of squares' output row (`part_rows.wire`).  The sum
of squares then the apply blocks compute the row's outputs word for word.  The final norm, outside the decoder layers, is read
whole (``FINAL``): a partition naming the row's Definition and its parts reads a layer's norm split and the final norm whole."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.ops.boolean import mufu as BM
from verity.definitions.registry import boolean_dsa_norm as BD
from verity.definitions.registry import boolean_norms as BN, boolean_norms_split as BS
from verity.definitions.registry import boolean_sigma as SG
from verity.definitions.registry import cuda_rmsnorm as C
from verity.definitions.registry import cuda_rmsnorm_split as NS
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry import tp_allreduce as TA, tp_allreduce_split as TS
from verity_vllm.pipeline import part_rows as PR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST

N, G, EPS, BLOCK, WORLD = 512, 64, 1e-5, 32, 4
LAYER, FINAL = "model.layers.{L}.input_layernorm", "model.norm"
CHUNK_SUMSQ, CHUNK_APPLY = 1, 16


def _rsqrt():
    return bind(BM.RsqrtApprox)


#: by family: (the row's Definition, its argument groups (name, elements), its committed members, its sum of squares and apply
#: at G, word and Boolean, a layer's op_path, at layer {L})
FAMILIES = {
    "RMSNormCudaVector8_v1": (
        lambda: bind(C.RMSNormCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK), (("x", N), ("w", N)), ("out",),
        lambda b: bind(BS.RMSNormSumSqCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK, RSQRT=_rsqrt()) if b
        else bind(NS.RMSNormSumSqCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK),
        lambda b: bind(BS.RMSNormApplyCudaVector8 if b else NS.RMSNormApplyCudaVector8, G=G), LAYER),
    "RMSNormFusedCudaVector8_v3": (
        lambda: bind(C.RMSNormFusedCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK), (("x", N), ("res", N), ("w", N)), ("out", "residual"),
        lambda b: bind(BS.RMSNormFusedSumSqCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK, RSQRT=_rsqrt()) if b
        else bind(NS.RMSNormFusedSumSqCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK),
        lambda b: bind(BS.RMSNormFusedApplyCudaVector8 if b else NS.RMSNormFusedApplyCudaVector8, G=G),
        "model.layers.{L}.post_attention_layernorm"),
    "DsaRMSNorm_v1": (
        lambda: bind(DN.DsaRMSNorm, N=N, EPS=EPS), (("x", N), ("w", N)), ("out",),
        lambda b: bind(BD.DsaRMSNormSumSqV2 if b else DN.DsaRMSNormSumSq, N=N, EPS=EPS),
        lambda b: bind(BD.DsaRMSNormApplyV2 if b else DN.DsaRMSNormApply, G=G), "model.layers.{L}.self_attn.kv_a_layernorm"),
    "AllReduceRmsNormMnnvl_v1": (
        lambda: bind(TA.AllReduceRmsNormMnnvl, WORLD=WORLD, N=N, SHOT=1, CLUSTER=1, EPS=EPS, FMA=False),
        (("parts", WORLD * N), ("res", N), ("w", N)), ("normed", "residual"),
        lambda b: bind(SG.RmsNormMnnvlSumSq if b else TS.RmsNormMnnvlSumSq, N=N, SHOT=1, CLUSTER=1, EPS=EPS, FMA=False),
        lambda b: bind(SG.RmsNormMnnvlApply if b else TS.RmsNormMnnvlApply, G=G, SHOT=1),
        "model.layers.{L}.fused_allreduce_rms_norm_all_reduce"),
}


def bf16s(rng, n: int, scale: float) -> np.ndarray:
    v = ((rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    v[:3] = [0x8000, 0x0001, 0x0000]
    return v


def ev(fn, rows: list[np.ndarray]) -> np.ndarray:
    out = evaluate_batch(fn, *(np.asarray(a).astype(np.uint64) for a in rows))
    return np.concatenate([np.asarray(a).reshape(len(rows[0]), -1) for a in (out if isinstance(out, (tuple, list)) else [out])],
                          axis=1)


def words(data: dict[str, np.ndarray], ports, packed) -> list[np.ndarray]:
    """A packed member's u16 rows as its ports' elements, one row per unit."""
    return [np.ascontiguousarray(data[p], dtype="<u2").view(np.dtype(f"<u{b // 8}")).reshape(len(data[p]), -1).astype(np.uint64)
            for (p, _w), b in zip(ports, packed)]


@pytest.fixture
def served(monkeypatch):
    """Rows 7, 8, .. of one request of ``family``, one at each op_path of ``ops`` (None: row i's in layer i - 1): its arguments Gaussian bf16 rows, its committed
    members its Definition's outputs, behind a ProgramIndex stand-in; with ``bad``, row 7's first served output has a word
    flipped."""
    def make(family: str, ops=(None,), bad: bool = False):
        fn, groups, members, _s, _a, layer = FAMILIES[family]
        fn = fn()
        args, served_words, by_i, ds = {}, {}, {}, {}
        for i, op in enumerate(ops, start=7):
            rng = np.random.default_rng(i)
            vals = [bf16s(rng, n, 0.1 if nm == "w" else 1.0) for nm, n in groups]
            outs = ev(fn, [v[None] for v in vals]).astype(np.uint16).reshape(len(members), N)
            if bad and i == 7:
                outs[0, 5] ^= 1
            op = op or layer.format(L=i - 1)
            for k, a in enumerate(vals):
                args[10 * i + k] = a
            served_words[op] = dict(zip(members, outs))
            by_i[i] = {"i": i, "spec": descriptor_id(fn), "args": [[["x", 10 * i + k, 0, a.size]] for k, a in enumerate(vals)]}
            ds[i] = {"statics": {k: v for k, v in fn.bindings.items()}, "op_path": op,
                     "members": [(m, N, "bf16") for m in members], "row": 0, "step": 16}

        class Index:
            pnames = ["weights"]

            def __init__(self):
                self.by_i = by_i

            def describe(self, r):
                return ds[r["i"]]

        class Store:
            def words(self, rid, step, op_path, member, lo, hi):
                return served_words[op_path][member][lo:hi], None

        monkeypatch.setattr(SRW, "index_of", lambda inst, _pool, **kw: Index(), raising=False)
        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [args[group[0][1]][:group[0][3]]])
        return {"spec": descriptor_id(fn), "store": Store(), "args": args, "words": served_words}
    return make


def templates(family: str) -> tuple[str, str]:
    _fn, _g, _m, sumsq, apply, _op = FAMILIES[family]
    return descriptor_id(sumsq(True)), descriptor_id(apply(True))


def wired(tmpls: list[str], chunk_units: dict[str, int]):
    """The window's members as `serving_rows.population` cuts and wires them, without its population program."""
    rows, _defs, counts = SRW.member_rows({"r0": {}}, tmpls)
    return PR.wire(SRW.members(tmpls, rows, counts, chunk_units), tmpls, rows, SRW.row_units), rows


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_norm_row_holds_its_sum_of_squares_once_and_its_apply_in_column_blocks(family):
    fn, _g, _m, sumsq, apply, _op = FAMILIES[family]
    spec = descriptor_id(fn())
    got = dict(SRW.unit_templates(spec))
    assert got[descriptor_id(sumsq(False))] == 1 and got[descriptor_id(apply(False))] == N // G
    assert SRW.part(descriptor_id(sumsq(False)), spec) == ST.PARTS[family][-2]
    assert SRW.part(descriptor_id(apply(False)), spec) == ST.PARTS[family][-1]
    ts, ta = (SRW.template(t) for t in templates(family))
    assert (ts.layout, ts.word, ts.packed[-1]) == ("part", descriptor_id(sumsq(False)), 32)
    assert (ta.layout, ta.word) == ("cols", descriptor_id(apply(False)))


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_sum_of_squares_then_the_apply_blocks_compute_the_row_s_outputs(served, family):
    """Two rows: one sum of squares unit each, then N / G apply blocks each, every block's scale wired to its row's sum of
    squares' scale; the sum of squares' Definition on its unit's rows is the scale it holds, each block's on its rows its
    outputs, which joined in column order are the row's served outputs."""
    x = served(family, ops=(None, None))
    _fn, groups, members, sumsq, apply, _op = FAMILIES[family]
    TS, TA_ = templates(family)
    ms, rows = wired([TS, TA_], {TS: CHUNK_SUMSQ, TA_: CHUNK_APPLY})
    ts, ta = SRW.template(TS), SRW.template(TA_)
    sums, blocks = [m for m in ms if m.template == TS], [m for m in ms if m.template == TA_]
    nh = N // G
    assert [(m.base, m.n, m.wired) for m in sums] == [(0, 1, ()), (1, 1, ())]
    assert sum(m.n for m in blocks) == 2 * nh
    k = next(k for k, (src, _g) in enumerate(ST.PARTS[family][-1][1]) if src == "mid")
    for m in blocks:
        assert m.wired == ((f"in{k}", "o0", tuple((m.base + i - 2) // nh for i in range(m.n))),)
    got = SRW.read_all(ms, rows, x["store"], None)
    scales = []
    for m, (g, _e) in zip(ms, got):
        if m.template == TS:
            (scale,) = words(g["outs"], m.outs, ts.packed[len(m.rows):])
            assert ev(sumsq(False), words(g["rows"], m.rows, ts.packed[:len(m.rows)])).tolist() == scale.tolist()
            scales.append(g["outs"][m.outs[0][0]][0])
        else:
            for port, out, writers in m.wired:
                assert all(g["rows"][port][i].tobytes() == scales[u].tobytes() for i, u in enumerate(writers))
            outs = np.concatenate([g["outs"][p] for p, _w in m.outs], axis=1)
            assert ev(apply(False), words(g["rows"], m.rows, ta.packed[:len(m.rows)])).astype(np.uint16).tolist() == outs.tolist()
    outs = [np.concatenate([g["outs"][p] for m, (g, _e) in zip(ms, got) if m.template == TA_]) for p, _w in blocks[0].outs]
    for j, member in enumerate(ST.PARTS[family][-1][2]):
        joined = outs[j].reshape(2, nh * G)
        name = members[member]
        for i in (0, 1):
            assert joined[i].tolist() == x["words"][FAMILIES[family][5].format(L=6 + i)][name].tolist()


@pytest.mark.parametrize("family", ["RMSNormCudaVector8_v1", "AllReduceRmsNormMnnvl_v1"])
def test_a_layer_s_norm_is_read_split_and_the_final_norm_whole(served, family):
    """A partition naming the row's Definition (MNNVL's: its norm part, ``FINAL``) beside its sum of squares and apply: the
    layer's row is read by the split, the final norm's (outside every decoder layer: ``model.norm``, or MNNVL's
    ``model.fused_allreduce_rms_norm_*``) by the whole."""
    mnnvl = family == "AllReduceRmsNormMnnvl_v1"
    served(family, ops=(None, "model.fused_allreduce_rms_norm_all_reduce" if mnnvl else FINAL))
    whole = descriptor_id(bind(SG.RmsNormMnnvl, N=N, SHOT=1, CLUSTER=1, EPS=EPS, FMA=False) if mnnvl
                          else bind(BN.RMSNormCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK, RSQRT=_rsqrt()))
    TS, TA_ = templates(family)
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [whole, TS, TA_])
    assert [[r["i"] for _rid, _pi, r, _d in rs] for rs in rows] == [[8], [7], [7]]
    assert counts == [1, 1, N // G]


def test_a_served_output_that_is_not_the_twin_s_is_refused(served):
    family = "RMSNormFusedCudaVector8_v3"
    x = served(family, bad=True)
    TS, _ = templates(family)
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [TS])
    (m,) = SRW.members([TS], rows, counts)
    with pytest.raises(ValueError, match=r"row 7 \(.*\): the twin of its intermediates \['scale'\] gives its output out other "
                                         r"words than the served ones"):
        SRW.read(m, rows[0], x["store"], None)


def test_a_partition_without_the_sum_of_squares_is_refused(served):
    family = "DsaRMSNorm_v1"
    served(family)
    TS, TA_ = templates(family)
    with pytest.raises(ValueError, match=r"row 7 \(.*\) reads its r, which its part DsaRMSNormSumSq_v1 returns: name that part's "
                                         r"template in the partition"):
        wired([TA_], {TA_: CHUNK_APPLY})
    with pytest.raises(ValueError, match=rf"{TS}: its units are joined by a wire of row 7's r, and are no chunks"):
        wired([TS, TA_], {TA_: CHUNK_APPLY})
