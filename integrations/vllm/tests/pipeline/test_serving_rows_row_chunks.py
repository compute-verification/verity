"""A row template in chunks (`pipeline.serving_rows.chunked`): a row Definition the partition file's ``chunk_cap`` names is staged
by C-Flock's `UnitStager` on its own ports, one v2 bit row per parameter (``in<g>``) and one per return field (``o<j>``), each
chunk a member of at most its cap of the template's rows, read once; with no cap it stays the word-row member it was.  And a scope
of several op_path prefixes takes the rows under any of them."""
from __future__ import annotations

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.registry import b1 as B1
from verity.ml.definitions.registry import dead_free as DF
from verity.ml.definitions.registry import deepgemm as DG
from verity.ml.definitions.registry import fa2_check_inf as FA2
from verity.ml.definitions.registry import fa3_check_inf as FA3
from verity.ml.definitions.registry import hopper as HP

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import mufu as MU
from verity.ml.circuits.ops import prims as OP
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

#: one plain row template, and one packed with two return fields (a whole ``PARTS`` part, DeepGEMM's linear's quantisation, here
#: on rows of its own family); each row's committed members and their element dtypes
ROWS = {"ResidualAdd_v1": (B1.ResidualAdd, {"N": 8}, [("out", "bf16")]),
        "Fp8GroupQuantUe8m0_v1": (DG.Fp8GroupQuantUe8m0, {"K": 256, "G": 128}, [("q", "fp8"), ("s", "f32")])}
OPS = ["model.layers.3.mlp.experts", "moe/L3", "model.layers.4.mlp.experts", "moe/L4", "model.layers.3.mlp.experts"]
CAP = 2


def defn(family: str):
    d, st, _m = ROWS[family]
    return bind(d, **st)


def bf16s(rng, n: int) -> np.ndarray:
    return ((rng.standard_normal(n) * 2).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


@pytest.fixture
def rows(monkeypatch):
    """``len(OPS)`` rows of ``family``, row j at ``OPS[j]``, its operands registered by (row, argument) and its outputs by the
    Definition committed as each member's row j, behind a ProgramIndex stand-in that holds them as rows 10 ..."""
    def make(family: str):
        fn, rng = defn(family), np.random.default_rng(3)
        params, _ret = fn.signature()
        fields = CR.fields(SRW.template(descriptor_id(fn)))
        ops = [[bf16s(rng, len(ty.leaf_widths())) for _nm, ty in params] for _ in OPS]
        outs: dict[str, list[np.ndarray]] = {nm: [] for nm, _dt in ROWS[family][2]}
        for o in ops:
            y = np.asarray(evaluate_batch(fn, *(a.astype(np.uint64).reshape(1, -1) for a in o))).reshape(-1)
            at = 0
            for (nm, _dt), f in zip(ROWS[family][2], fields, strict=True):
                n, w = len(f.leaf_widths()), f.leaf_widths()[0]
                outs[nm].append(y[at:at + n].astype(f"<u{w // 8}"))
                at += n
        spec = descriptor_id(fn)
        by_i = {10 + j: {"i": 10 + j, "spec": spec, "args": [[["n", 8 * j + g, 0, a.size]] for g, a in enumerate(o)]}
                for j, o in enumerate(ops)}
        ds = {10 + j: {"statics": ROWS[family][1], "op_path": op, "row": j, "step": 16,
                       "members": [(nm, outs[nm][j].size, dt) for nm, dt in ROWS[family][2]]} for j, op in enumerate(OPS)}

        class Index:
            def __init__(self, *_a, **_k):
                self.by_i, self.pnames = by_i, []

            def describe(self, r):
                return ds[r["i"]]

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return np.concatenate(outs[member])[lo:hi], None

        monkeypatch.setattr(SRW, "index_of", lambda inst, _pool, **kw: Index(inst, **kw))
        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g, **_k: [ops[g[0][1] // 8][g[0][1] % 8]])
        return spec, Store(), ops, outs
    return make


@pytest.mark.parametrize("family", sorted(ROWS))
def test_a_row_template_the_caps_name_is_chunks_of_the_stager_s_bit_rows_on_its_own_ports(rows, family):
    T, store, _ops, _outs = rows(family)
    tp = SRW.template(T)
    params, _ret = tp.fn.signature()
    fields = CR.fields(tp)
    assert tp.layout == ("part" if family in SRW.PART_OF else "row") and len(fields) == len(ROWS[family][2])
    assert not SRW.chunked(T, None) and not SRW.chunked(T, {}) and SRW.chunked(T, {T: CAP})
    in_rows, out_rows = CR.unit_rows(tp)
    assert in_rows == [(f"in{g}", V2, sum(ty.leaf_widths()), [nm]) for g, (nm, ty) in enumerate(params)]
    assert out_rows == [(V2, (j,)) for j in range(len(fields))]
    (rs,), _defs, counts = SRW.member_rows({"r0": {}}, [T])
    assert counts == [len(OPS)] and [r["i"] for _rid, _pi, r, _d in rs] == [10 + j for j in range(len(OPS))]
    (whole,) = SRW.members([T], [rs], counts)
    assert not whole.name and not whole.schemas and [p for p, _w in whole.outs] == [f"out.{nm}" for nm, _dt in ROWS[family][2]]
    ms = SRW.members([T], [rs], counts, {T: CAP})
    assert [(m.name, m.base, m.n) for m in ms] == [(f"{T}/c{j}", lo, min(CAP, len(OPS) - lo))
                                                   for j, lo in enumerate(range(0, len(OPS), CAP))]
    for m in ms:
        assert m.rows == whole.rows and [w for _p, w in m.outs] == [w for _p, w in whole.outs]
        assert [p for p, _w in m.outs] == [f"o{j}" for j in range(len(fields))]
        assert dict(m.schemas) == dict.fromkeys([p for p, _w in m.rows + m.outs], V2)
    data, entries = SRW.read(whole, rs, store, None)
    got = SRW.read_all(ms, [rs], store, None)
    for g in range(len(params)):
        assert np.array_equal(np.concatenate([c[0]["rows"][f"in{g}"] for c in got]), data["rows"][f"in{g}"])
    for j, (p, _w) in enumerate(whole.outs):
        assert np.array_equal(np.concatenate([c[0]["outs"][f"o{j}"] for c in got]), data["outs"][p])
    assert [e for _c, es in got for e in es] == entries
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = tp.fn, "row-chunk-test", {}, in_rows
    for c, _e in got:
        st.reference_check(c["rows"], c["outs"])
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check(got[0][0]["rows"], {**got[0][0]["outs"], "o0": got[0][0]["outs"]["o0"] ^ np.uint16(1)})


def test_a_whole_attention_row_reads_its_tables_in_lookup_slots_and_an_rmsnorm_row_inline():
    """`chunk_rows.reads` of a whole row: lookup slots for Llama-3.2-1B's decode attention row under v5 and v18 (inline, its
    ``--zk`` prover ran out of memory at 600 GiB; in slots it was proved, `chunk_rows.reads`), inline for the RMSNorm rows,
    which read the rsq table too and were proved inline at their served shapes."""
    att = {"T": 1, "NH": 32, "KVH": 8, "D": 64, "BN": 128, "DOT": OP.HopperBF16WgmmaDot16, "INV": MU.Fa2InvSum, "MASKED_FROM": 0}
    norms = (bind(DF.RMSNormTritonV5, N=2048, EPS=1e-5), bind(B1.RMSNormFusedCuda, N=2048, EPS=1e-5))
    for fn, want in [(bind(d, **att), "slots") for d in (FA2.AttentionV5, DF.AttentionV18)] + [(fn, "inline") for fn in norms]:
        tp = SRW.template(descriptor_id(fn))
        assert (tp.layout, CR.reads(tp)) == ("row", want), tp.id


def test_an_attention_heads_per_key_block_calls_read_their_tables_in_lookup_slots():
    """`chunk_rows.reads` of the Calls the window circuit cuts a decode head into (`window_circuit.CUTS`), FA2's
    `AttentionHead_v19` and FA3's `AttentionHead_v20` at Llama-3.2-1B's statics on each target: its first visited block, the
    full block below it and its epilogue each read in lookup slots, as the whole row they cut is."""
    for head, inv, BN in ((DF.AttentionHeadV19, MU.Fa2InvSum, 128), (FA3.AttentionHeadV20, HP.Fa3InvSum, 192)):
        nodes = bind(head, T=2 * BN, D=64, BN=BN, DOT=OP.HopperBF16WgmmaDot16, INV=inv, MASKED_FROM=1).body.nodes
        assert len({n.fn.base_id for n in nodes}) == 3
        for n in nodes:
            tp = SRW.template(descriptor_id(n.fn))
            assert (tp.layout, CR.reads(tp)) == ("row", "slots"), tp.id


def test_a_scope_of_several_prefixes_takes_the_rows_under_any_of_them(rows):
    T, _store, _ops, _outs = rows("ResidualAdd_v1")
    def under(scope):
        return [r["i"] for _rid, _pi, r, _d in SRW.member_rows({"r0": {}}, [T], scope)[0][0]]

    assert under("moe/L3") == [11]
    assert under(["model.layers.3.", "moe/L3"]) == [10, 11, 14]
    assert under(("moe/L4", "model.layers.4.")) == [12, 13]
    assert under([]) == under(None) == [10, 11, 12, 13, 14]
    with pytest.raises(LookupError, match=r"under moe/L5 or model\.layers\.5\.; its rows exist outside the scope, e\.g\. op_path "
                                          r"model\.layers\.3\.mlp\.experts at step 16"):
        under(["moe/L5", "model.layers.5."])
