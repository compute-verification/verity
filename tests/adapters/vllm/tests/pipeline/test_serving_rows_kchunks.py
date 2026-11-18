"""A w13 coordinate cut along K (`pipeline.serving_rows`' ``kchunk`` layout, `kchunk_rows`): a ``MoeExpertFp8Gemm_v1{K}`` row holds
the first, middle and last ``MoeExpertFp8CoordinateChunk`` templates at every chunk width, and a window of the three:

* reads each unit's operands off the row and its ``acc`` off the twin's chain, so the chunk Definition on each chunk's rows, parsed
  as Flock's stager parses them, is its committed output, and the last chunk's is the row's served word;
* wires every ``acc`` row to the output row of the chunk before it (chunk c - 1 of the same row and coordinate), commits it under
  that row's salt, so the verifier's check on the public bodies holds, and refuses a wire to another unit, an ``acc`` row that is
  not its writer's output, and a partition that leaves out a writer's template.
"""
from __future__ import annotations


import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.ops.prims import HopperE4m3MmaV2Dot32
from verity.commitments.gates import embedding as EMB
from verity.commitments.periods import interface as IF
from verity.commitments.periods import rows as CS
from verity_flock.class_statement import UnitStager

from verity_vllm.definitions.registry import boolean_fp8_moe as BFM
from verity_vllm.definitions.registry import fp8_moe as FM
from verity_vllm.definitions.registry import fp8_moe_experts as X
from verity_vllm.pipeline import chain_rows as CH
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import kchunk_rows as KC
from verity_vllm.pipeline import serving_rows as SRW

E, K, N, G, KB, KC_ = 3, 128, 5, 32, 32, 32
NB, KG, C = -(-N // G), K // G, K // KC_
GEMM = bind(X.MoeExpertFp8Gemm, E=E, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=HopperE4m3MmaV2Dot32)


#: `chain_rows.pieces`' refusal of a port reading several templates' trees
NAME = "a wired port reads one template's tree"

def chunk(first: bool, last: bool, k: int = KC_, boolean: bool = True) -> str:
    st = {"E": E, "K": k, "G": G, "KB": KB, "SWAP": False, "FIRST": first, "LAST": last}
    if boolean:
        return descriptor_id(bind(BFM.MoeExpertFp8CoordinateChunkV2, **st, DOT=BG.HopperE4m3MmaV2Dot32))
    return descriptor_id(bind(X.MoeExpertFp8CoordinateChunk, **st, DOT=HopperE4m3MmaV2Dot32))


FIRST, MID, LAST = chunk(True, False), chunk(False, False), chunk(False, True)
TMPLS = [FIRST, MID, LAST]
CAPS = {FIRST: 4, MID: 7, LAST: 3}


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "kchunk-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def _bits(a: np.ndarray) -> np.ndarray:
    return a.view(f"<u{a.dtype.itemsize}") if a.dtype.itemsize > 1 else a


@pytest.fixture
def pairs(monkeypatch):
    """Two pairs of one layer (experts 2 and 0, one weight stack), each a ``MoeExpertFp8Gemm_v1{K=128}`` row, and the store of their
    served outputs."""
    rng = np.random.default_rng(11)

    def codes(n):
        c = rng.integers(0, 256, n, dtype=np.uint8)
        return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)

    def scales(n):
        return ((2.0 ** rng.integers(-8, 2, n)) * rng.uniform(1, 2, n)).astype(np.float32)

    w, ws = codes(E * N * K), scales(E * NB * KG)
    ops = {}
    for p, e in enumerate((2, 0)):
        ops.update({(p, 0): codes(K), (p, 1): scales(KG), (p, 2): np.array([e], np.int32)})
    ops[(9, 3)], ops[(9, 4)] = w, ws
    ys = [np.asarray(evaluate_batch(GEMM, *[_bits(ops[(p, g)]).astype(np.uint64).reshape(1, -1) for g in range(3)],
                                    _bits(w).astype(np.uint64).reshape(1, -1), _bits(ws).astype(np.uint64).reshape(1, -1))
                     ).reshape(-1).astype(np.uint16) for p in range(2)]
    rows = []
    for p in range(2):
        r = {"i": 7 + p, "spec": descriptor_id(GEMM),
             "args": [[["x", 10 * p + g, 0, ops[(p, g)].size]] for g in range(3)] + [[["x", 93, 0, w.size]], [["x", 94, 0, ws.size]]]}
        d = {"statics": {"E": E, "K": K, "N": N, "G": G, "KB": KB}, "op_path": "moe/L3", "members": [("gate_up", N, "bf16")],
             "row": p, "step": 16}
        rows.append(("r0", None, r, d))

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return np.concatenate(ys)[lo:hi], None

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g: [ops[divmod(g[0][1], 10)]])
    return rows, ops, np.concatenate(ys), Store()


def window(rows, tmpls=TMPLS):
    ms = KC.wire(SRW.members(tmpls, [rows] * len(tmpls), None, CAPS), tmpls, [rows] * len(tmpls))
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Binding(*IF.listed_program(defs, ()), {"law": "subset", "k": 2}, "ab" * 32, tuple(CH.shared_tables(ms)))


def test_a_w13_row_holds_its_first_middle_and_last_chunks_at_each_width():
    spec = descriptor_id(GEMM)
    assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(bind(X.MoeExpertFp8Gemm, E=E, K=K, N=1, G=G, KB=KB, SWAP=False,
                                                                       DOT=HopperE4m3MmaV2Dot32)), N),
                                        (chunk(True, False, 64, False), N), (chunk(False, True, 64, False), N),
                                        (chunk(True, False, 32, False), N), (chunk(False, False, 32, False), N * (C - 2)),
                                        (chunk(False, True, 32, False), N),
                                        (descriptor_id(bind(FM.MoeExpertFp8BlockGemmV3, K=K, N=1, G=G, KB=KB, SWAP=False,
                                                            DOT=HopperE4m3MmaV2Dot32)), N),
                                        (descriptor_id(EMB.read(E, K // 2)), N),
                                        (descriptor_id(bind(BFM.MoeExpertFp8ReadGemmV1, READ=EMB.read(E, K // 2), K=K, N=1, G=G, KB=KB,
                                                            SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32)), N),
                                        (descriptor_id(EMB.read(E, 2 * K // G)), NB))
    assert [(SRW.template(t).layout, SRW.template(t).word) for t in TMPLS] == [
        ("kchunk", chunk(f, la, boolean=False)) for f, la in ((True, False), (False, False), (False, True))]
    with pytest.raises(ValueError, match="no chunk cap"):
        SRW.members(TMPLS, [[("r0", None, {"i": 0}, {"statics": {"K": K, "N": N}})]] * 3, None, {})


def test_a_partition_without_a_writer_s_template_is_refused(pairs):
    rows, *_ = pairs
    for tmpls, missing in (([MID, LAST], "first"), ([FIRST, LAST], "mid")):
        with pytest.raises(ValueError, match=rf"\['{missing}'\] ones, which the partition does not name"):
            window(rows, tmpls)
    assert all(m.wired == () for m in window(rows, [FIRST]).members)


def test_a_chunk_reading_the_first_and_a_middle_chunk_is_refused_by_name(pairs):
    """A middle or last chunk's acc is the FIRST chunk's output at chunk 1 and a middle chunk's after it: one port reading
    two templates' trees, which `chain_rows.pieces` refuses by name."""
    rows, *_ = pairs
    with pytest.raises(ValueError, match=NAME):
        window(rows)
