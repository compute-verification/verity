"""CIR-242 scratch: the Flock block of each unit a sparse-MLA key block is served by, under each form that gives the
block's keys one writer, at GLM-5.3's statics (`boolean_sparse_mla.SPARSE_MLA_GLM53`), read off the Definitions
(`verity_flock.circuit.unit_block`) and held to the composer's limits (`interface.check_proof_block`).

Two variant bindings, unregistered composites over the served Boolean units, built only to be measured:

* `ScoresCols{NVIS, FIRST}`: `SparseMlaHeadScores_v2` with its ``kv`` laid out in column blocks of G words,
  ``Array((DV + DR) / G, Array(NVIS, Array(G)))``: block c < DV / G the keys' kv_c columns [c G, (c + 1) G), the last
  their k_pe, so cut ``(DV + DR) / G`` ways each row is one column block of the key block, the row a values unit reads;
* `ValuesWhole{NVIS, FIRST, C}`: `SparseMlaHeadValues_v2{G}` on column block C of whole key rows, its ``kv`` the
  block's keys [kv_c | k_pe] as the scores unit reads them;
* `GatherCols{C}`: column block C of 64 whole key rows, no gate, cut a key a row (each a wire from its key's writer).

``python cir242_key_rows.py [NAME ...]`` prints one JSON line a unit (all of them with no NAME)."""
from __future__ import annotations

import json
import sys
import time

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.commitments.gates import embedding as EG
from verity.commitments.windows import interface as IF
from verity.definitions.registry import boolean_sigma as SG
from verity.definitions.registry import boolean_sparse_mla as BM
from verity.definitions.registry.boolean_attention import BF16, F32
from verity_flock import circuit as FC
from verity_vllm.pipeline import chunk_rows as CR

GLM = BM.SPARSE_MLA_GLM53
S = {k: GLM[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
DR, DV, BN, G = GLM["DR"], GLM["DV"], GLM["BN"], 64
KW, NC = DR + DV, (DR + DV) // G


def _scores_cols_type(Sx):
    state = () if Sx.FIRST else (("m_prev", F32), ("l_prev", Array(4, F32)))
    return ((("q_nope", Array(DV, BF16)), ("q_rope", Array(DR, BF16)), ("kv", Array(NC, Array(Sx.NVIS, Array(G, BF16)))))
            + state, bind(BM.SparseMlaHeadScoresV2, NVIS=Sx.NVIS, FIRST=Sx.FIRST, **S).signature()[1])


def _scores_cols(B, Sx, q_nope, q_rope, kv, *state):
    rows = Coll(Array(Sx.NVIS, Array(KW, BF16)), concat([kv[c][k].refs for k in range(Sx.NVIS) for c in range(NC)]))
    return B.call(bind(BM.SparseMlaHeadScoresV2, NVIS=Sx.NVIS, FIRST=Sx.FIRST, **S), q_nope, q_rope, rows, *state)


ScoresCols = CompositeDefinition("Cir242ScoresCols", 0, ("NVIS", "FIRST"), _scores_cols_type, _scores_cols, register=False)


def _values_whole_type(Sx):
    state = () if Sx.FIRST else (("rescale", F32), ("o_prev", Array(G, F32)))
    return ((("pb", Array(Sx.NVIS, BF16)), ("kv", Array(Sx.NVIS, Array(KW, BF16)))) + state, Array(G, F32))


def _values_whole(B, Sx, pb, kv, *state):
    cols = Coll(Array(Sx.NVIS, Array(G, BF16)), concat([kv[k][Sx.C * G:(Sx.C + 1) * G].refs for k in range(Sx.NVIS)]))
    return B.call(bind(BM.SparseMlaHeadValuesV2, NVIS=Sx.NVIS, FIRST=Sx.FIRST, BN=BN, DOT=S["DOT"], G=G), pb, cols, *state)


ValuesWhole = CompositeDefinition("Cir242ValuesWhole", 0, ("NVIS", "FIRST", "C"), _values_whole_type, _values_whole,
                                  register=False)


def _gather_cols(B, Sx, keys):
    return Coll(Array(64 * G, BF16), concat([keys[k][Sx.C * G:(Sx.C + 1) * G].refs for k in range(64)]))


GatherCols = CompositeDefinition("Cir242GatherCols", 0, ("C",),
                                 lambda Sx: ((("keys", Array(64, Array(KW, BF16))),), Array(64 * G, BF16)), _gather_cols,
                                 register=False)


def scores(nvis, first):
    return bind(BM.SparseMlaHeadScoresV2, NVIS=nvis, FIRST=first, **S)


def values(nvis, first):
    return bind(BM.SparseMlaHeadValuesV2, NVIS=nvis, FIRST=first, BN=BN, DOT=S["DOT"], G=G)


#: name -> (Definition, cuts)
UNITS = {
    "block64L": (lambda: bind(BM.SparseMlaHeadBlockV3, NVIS=64, FIRST=False, **S), None),
    "scores64L": (lambda: scores(64, False), None),
    "scores64F": (lambda: scores(64, True), None),
    "scores16F": (lambda: scores(16, True), None),
    "values64L": (lambda: values(64, False), None),
    "values64F": (lambda: values(64, True), None),
    "values16F": (lambda: values(16, True), None),
    "scoresCols64L": (lambda: bind(ScoresCols, NVIS=64, FIRST=False), {"kv": NC}),
    "scoresCols64F": (lambda: bind(ScoresCols, NVIS=64, FIRST=True), {"kv": NC}),
    "scoresCols16F": (lambda: bind(ScoresCols, NVIS=16, FIRST=True), {"kv": NC}),
    "scoresKeys64L": (lambda: scores(64, False), {"kv": 64}),
    "scoresKeys64F": (lambda: scores(64, True), {"kv": 64}),
    "scoresKeys16F": (lambda: scores(16, True), {"kv": 16}),
    "valuesKeys64L": (lambda: values(64, False), {"kv": 64}),
    "valuesKeys64F": (lambda: values(64, True), {"kv": 64}),
    "valuesWhole64L": (lambda: bind(ValuesWhole, NVIS=64, FIRST=False, C=0), None),
    "valuesWhole64F": (lambda: bind(ValuesWhole, NVIS=64, FIRST=True, C=0), None),
    "valuesWholeKeys64L": (lambda: bind(ValuesWhole, NVIS=64, FIRST=False, C=0), {"kv": 64}),
    "writer": (lambda: bind(SG.AllGather, WORLD=64, N=G), None),
    "gather": (lambda: bind(SG.AllGather, WORLD=64, N=G), {"parts": 64}),
    "gatherCols": (lambda: bind(GatherCols, C=0), {"keys": 64}),
    "read": (lambda: EG.read(1 << 17, KW), None),
}


def measure(name: str) -> dict:
    make, cuts = UNITS[name]
    fn = make()
    t = time.perf_counter()
    b = FC.unit_block(fn, cuts)
    lay = IF.layout(fn)
    ins = [(p, 2 * w) for p, w, _ in lay.cut(cuts)]
    outs = [(p, 2 * w) for p, w in lay.outs]
    try:
        k = IF.check_proof_block(fn, cuts)
        cpb = {"ok": True, "rows_k_log": k}
    except ValueError as e:
        cpb = {"ok": False, "why": str(e)}
    return {"name": name, "unit": fn.id, "cuts": cuts, **{x: b[x] for x in ("ands", "port_bits", "rows", "bits", "k_log",
                                                                              "unit_log", "sha512x3", "hm96")},
            "past_block": FC.past_block(b), "check_proof_block": cpb, "slot_tables": list(CR.slot_tables(fn)),
            "in_rows": ins, "out_rows": outs, "seconds": round(time.perf_counter() - t, 1)}


if __name__ == "__main__":
    for n in sys.argv[1:] or list(UNITS):
        print(json.dumps(measure(n)), flush=True)
