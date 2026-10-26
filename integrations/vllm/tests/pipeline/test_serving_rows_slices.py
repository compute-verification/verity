"""The batched rows split into slices (`pipeline.serving_rows`' ``slice`` layout, ``SLICES``): a row of extents a_0, a_1, .. along
its family's axes is a_0 a_1 .. units of its Definition at every axis 1, unit (i_0, i_1, ..) reading element i_k along axis k of
each parameter that carries it and the others whole, and returning the row's output element (i_0, i_1, ..): the router GEMM, the
LM head and cuBLAS's split-K (at a bf16 and an f32 output) into W's rows, torch.bmm into its members' coordinates, DSA's logits into keys, the vocabulary
shard's embedding into columns (every unit reads the token) and the all-gather into its ranks' shards.  The units cover the
row's outputs with nothing left over (one index entry per row, the units' outputs the row's output in order), each reads its
own elements of the row's operands (an independent gather by index), and the Definition at every axis 1 on each unit's rows is
the row's Definition's element; in a window the units are chunks of the stager's bit rows, read once."""
from __future__ import annotations

import math

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock.class_statement import UnitStager
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import slice_rows as SL
from verity.ml.definitions.registry import b1_tp2 as T2
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import dsa as DSA
from verity.ml.definitions.registry import ll_bf16 as LL

H = HopperBF16WgmmaDot16
#: each family at small statics; the token of the embedding shard inside it (2 <= 4 < 2 + 5)
ROWS = {"LlBf16DotprodGemm_v1": (LL.LlBf16DotprodGemmWord, {"N": 3, "K": 64, "BS": 32, "VEC": 2, "FMA": 1}),
        "Gemm_v2": (G.Gemm, {"K": 32, "N": 3, "DOT": H}),
        "CublasSplitKGemm_v1": (CB.CublasSplitKGemmWord, {"K": 128, "N": 3, "SPLIT_K": 2, "DOT": H}),
        "CublasSplitKGemmF32Out_v1": (CB.CublasSplitKGemmF32OutWord, {"K": 128, "N": 3, "SPLIT_K": 2, "DOT": H}),
        "BatchedGemm_v1": (CB.BatchedGemmWord, {"B": 2, "K": 32, "N": 3, "DOT": H}),
        "DsaMqaLogits_v3": (DSA.DsaMqaLogitsV3, {"N": 3, "H": 8, "D": 32}),
        "EmbeddingShard_v1": (T2.EmbeddingShard, {"VS": 5, "H": 4, "START": 2}),
        "AllGather_v1": (T2.AllGather, {"WORLD": 4, "N": 3})}
OP = "model.layers.3.mlp.gate"
CAP = 4


def test_every_family_of_the_table_is_tested():
    assert set(ROWS) == set(SRW.SLICES)


def defn(family: str, unit: bool = False):
    d, st = ROWS[family]
    return bind(d, **{**st, **(dict.fromkeys(SRW.SLICES[family][0], 1) if unit else {})})


def extents(family: str) -> list[int]:
    return [int(ROWS[family][1][a]) for a in SRW.SLICES[family][0]]


def operands(family: str, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for name, ty in defn(family).signature()[0]:
        n, w = len(ty.leaf_widths()), ty.leaf_widths()[0]
        if family == "EmbeddingShard_v1" and name == "tok":
            out.append(np.array([4], dtype=np.int32))
        elif w == 16:
            out.append((rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))
        elif w == 8:
            c = rng.integers(0, 256, n, dtype=np.uint8)
            out.append(np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8))
        else:
            out.append(rng.uniform(0.5, 2, n).astype(np.float32))
    return out


def gather(o: np.ndarray, dims: list[int], carried: tuple, ix: tuple) -> np.ndarray:
    """Unit ``ix``'s elements of operand ``o`` by indexing: i_k along each dimension that carries axis k, all along the others."""
    at = {j: ix[k] for k, j in enumerate(carried) if j is not None}
    return o.reshape(dims or [1])[tuple(at.get(dim, slice(None)) for dim in range(max(len(dims), 1)))]


def leaves(a: np.ndarray, bits: int) -> np.ndarray:
    """A port's u16 words, (units, words), as its elements, (units, elements)."""
    return np.ascontiguousarray(a, dtype="<u2").view(f"<u{bits // 8}").astype(np.uint64)


@pytest.fixture
def row(monkeypatch):
    def make(family: str, seed: int = 1):
        ops = operands(family, seed)
        y = np.asarray(evaluate_batch(defn(family), *(o.view(f"<u{o.itemsize}").astype(np.uint64).reshape(1, -1) for o in ops))).reshape(-1)
        y = y.astype(np.uint16 if defn(family).signature()[1].leaf_widths()[0] == 16 else np.uint32)
        r = {"i": 9, "spec": descriptor_id(defn(family)), "args": [[["x", k, 0, o.size]] for k, o in enumerate(ops)]}
        d = {"statics": ROWS[family][1], "op_path": OP, "members": [("out", y.size, str(y.dtype))], "row": 0, "step": 16}

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return y[lo:hi], None

        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g, **_k: [ops[g[0][1]]])
        return ("r0", None, r, d), ops, y, Store()
    return make


@pytest.mark.parametrize("family", sorted(ROWS))
def test_a_row_holds_its_definition_at_every_axis_one(family):
    spec, unit, n = descriptor_id(defn(family)), descriptor_id(defn(family, unit=True)), math.prod(extents(family))
    assert SRW.unit_templates(spec) == ((spec, 1), (unit, n))
    t = SRW.template(unit)
    assert (t.layout, t.word) == ("slice", unit)
    assert SRW.template(spec).layout == "row"


@pytest.mark.parametrize("family", sorted(ROWS))
def test_the_slices_cover_the_row_s_output_read_their_own_elements_and_compute_them(row, family):
    x, ops, y, store = row(family)
    ext, (_axes, carry) = extents(family), SRW.SLICES[family]
    n, T = math.prod(ext), descriptor_id(defn(family, unit=True))
    tp = SRW.template(T)
    bits = tp.packed or (16,) * (len(ops) + 1)
    (m,) = SRW.members([T], [[x]], [n])
    assert m.n == n and [p for p, _w in m.rows] == [f"in{g}" for g in range(len(ops))] and m.outs[0][0] == "out.out"
    data, entries = SRW.read(m, [x], store, None)
    assert entries == [[0, n, "r0", 16, 3, OP, "out", 0, 9]]
    out = leaves(data["outs"]["out.out"], bits[-1])
    assert out.reshape(-1).tolist() == y.astype(np.uint64).tolist()
    params = defn(family).signature()[0]
    for u, ix in enumerate(np.ndindex(*ext)):
        for g, ((_nm, ty), c, o) in enumerate(zip(params, carry, ops, strict=True)):
            want = gather(o, SL.dims(ty), c, ix).reshape(-1).view(f"<u{o.itemsize}").astype(np.uint64)
            assert leaves(data["rows"][f"in{g}"][u], bits[g]).tolist() == want.tolist()
    ref = evaluate_batch(defn(family, unit=True), *(leaves(data["rows"][f"in{g}"], b) for g, b in enumerate(bits[:-1])))
    assert np.asarray(ref).reshape(n, -1).tolist() == out.tolist()


@pytest.mark.parametrize("family", ["LlBf16DotprodGemm_v1", "CublasSplitKGemmF32Out_v1", "BatchedGemm_v1", "EmbeddingShard_v1",
                                    "DsaMqaLogits_v3"])
def test_a_window_s_slices_are_chunks_of_the_stager_s_bit_rows_read_once(row, family):
    x, ops, y, store = row(family)
    n, T = math.prod(extents(family)), descriptor_id(defn(family, unit=True))
    tp = SRW.template(T)
    in_rows, out_rows = CR.unit_rows(tp)
    assert [(p, s) for p, s, _b, _ps in in_rows] == [(f"in{g}", V2) for g in range(len(ops))] and out_rows == [(V2, (0,))]
    ms = SRW.members([T], [[x]], [n], {T: CAP})
    assert [(m.base, m.n) for m in ms] == [(lo, min(CAP, n - lo)) for lo in range(0, n, CAP)]
    (whole,) = SRW.members([T], [[x]], [n])
    data, _entries = SRW.read(whole, [x], store, None)
    got = SRW.read_all(ms, [[x]], store, None)
    for g in range(len(ops)):
        assert np.array_equal(np.concatenate([c[0]["rows"][f"in{g}"] for c in got]), data["rows"][f"in{g}"])
    assert np.array_equal(np.concatenate([c[0]["outs"]["o0"] for c in got]), data["outs"]["out.out"])
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = tp.fn, "slice-test", {}, in_rows
    for c, _e in got:
        st.reference_check(c["rows"], c["outs"])
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check(got[0][0]["rows"], {"o0": got[0][0]["outs"]["o0"] ^ np.uint16(1)})


class TokenIndex:
    """The ProgramIndex calls the real `_slices` makes for a vocabulary shard's row: argument 0 is the sampler's committed token of
    dtype ``dt``, argument 1 the shard's registered weight."""
    pnames = ["weights"]

    def __init__(self, dt: str = "i64"):
        self.dt = dt

    def member_at(self, a, lo, hi):
        return ({"members": [["sampled_token_ids", 1, self.dt]], "step": 15, "op_path": "runner.sampler", "row": 0},
                "sampled_token_ids", lo, hi, self.dt)

    def weight_slice(self, lo, hi):
        return "model.embed_tokens.weight", lo, hi, (lo, hi)


def token_row(tok: int, dt: str = "i64", unit: bool = True):
    """EmbeddingShard_v1{VS=5,H=4,START=2}'s row reading the sampler's token ``tok`` as its committed u64 word, its table a
    registered weight, through the real `_slices`: as its four slice units, or (``unit`` false) as the whole row."""
    family = "EmbeddingShard_v1"
    _tok, table = operands(family, 3)
    y = np.asarray(evaluate_batch(defn(family), np.array([[4]], np.uint64), table.astype(np.uint64).reshape(1, -1))).reshape(-1)
    r = {"i": 9, "spec": descriptor_id(defn(family)), "args": [[["n", 0, 0, 1]], [["p", 0, 0, table.size]]]}
    d = {"statics": ROWS[family][1], "op_path": "model.embed_tokens", "members": [("out", y.size, "bf16")], "row": 0, "step": 16}

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return (np.array([tok], np.uint64) if member == "sampled_token_ids" else y.astype(np.uint16))[lo:hi], None

    x = ("r0", TokenIndex(dt), r, d)
    (m,) = SRW.members([descriptor_id(defn(family, unit=True))], [[x]], [4]) if unit else SRW.members([r["spec"]], [[x]])
    return m, x, Store(), (lambda _f, lo, hi: table[lo:hi]), y


def test_the_samplers_i64_token_is_read_by_value_into_the_vocabulary_shards_32_bit_tok():
    """Token ids are 32-bit words: the shard's ``tok`` is the sampler's committed i64's low word, so every unit reads the token's
    value, low half first, and computes the row's output column; the whole row, a packed row template, reads it the same way."""
    m, x, store, weights, y = token_row(4)
    data, _entries = SRW.read(m, [x], store, weights)
    assert data["rows"]["in0"].tolist() == [[4, 0]] * 4
    assert data["outs"]["out.out"].reshape(-1).tolist() == y.tolist()
    T = descriptor_id(defn("EmbeddingShard_v1", unit=True))
    bits = SRW.template(T).packed
    ref = evaluate_batch(defn("EmbeddingShard_v1", unit=True), *(leaves(data["rows"][f"in{g}"], b) for g, b in enumerate(bits[:-1])))
    assert np.asarray(ref).reshape(-1).tolist() == y.tolist()
    m, x, store, weights, y = token_row(4, unit=False)
    assert SRW.template(m.template).layout == "row" and SRW.template(m.template).packed[0] == 32
    data, _entries = SRW.read(m, [x], store, weights)
    assert data["rows"]["in0"].tolist() == [[4, 0]] and data["outs"]["out.out"].reshape(-1).tolist() == y.tolist()


@pytest.mark.parametrize("unit", [True, False])
@pytest.mark.parametrize(("tok", "shown"), [(2**32 + 4, 2**32 + 4), (2**64 - 1, -1)])
def test_a_token_whose_high_word_is_not_zero_is_refused(tok, shown, unit):
    m, x, store, weights, _y = token_row(tok, unit=unit)
    with pytest.raises(ValueError, match=rf"member sampled_token_ids of runner.sampler: element 0 of a uint64 member is {shown}, not in "
                                         rf"a 32-bit port's \[0, 2\^32\)"):
        SRW.read(m, [x], store, weights)


def test_an_unsigned_member_wider_than_the_tok_port_is_not_read_by_value():
    m, x, store, weights, _y = token_row(4, dt="u64")
    with pytest.raises(ValueError, match=r"argument 0: uint64 elements, the template's port is 32-bit elements"):
        SRW.read(m, [x], store, weights)


def test_a_row_whose_operand_is_not_its_dimensions_is_refused(row, monkeypatch):
    x, ops, _y, store = row("Gemm_v2")
    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g, **_k: [ops[g[0][1]][:-1]])
    (m,) = SRW.members([descriptor_id(defn("Gemm_v2", unit=True))], [[x]], [3])
    with pytest.raises(ValueError, match=r"row 9 \(model.layers.3.mlp.gate\) argument 0: 31 elements, not its dimensions \[32\]"):
        SRW.read(m, [x], store, None)
