"""CIR-265: stage one of the long row's two mask units with `DsaCountSum_v1`'s builder committing the sum it returns, and keep
the files.  No profiling: the wrapper, the stager, one instance.

    python scratch/cir265/stage_keep.py positions|whole DIR [N [K]]

    positions   DsaMaskPositions_v2{N,AT=0,K}       (N = 40960, K = 2048)
    whole       DsaTopKMaskWhole_v2{N,TOPK=K}

DIR/circuit.txt   the unit's circuit as `chunk_rows.chunk_cap` stages a unit: `UnitStager` on the unit's rows
                  (`interface.unit_rows`), partition and program bound to 128 zeros, reads inline
DIR/round/        one instance on zero rows as a round stages it (`UnitStager.stage`: inst.bin, pub.bin), its outputs held to
                  the Definition's reference (`UnitStager.reference_check`)
DIR/staged.json   and the last line printed: the unit's id, its pin and class, the circuit's bytes, ``k_log``, ``unit_log``,
                  the ANDs, rows and nonzeros laid out, and the block's ranges as the circuit's META has them (net, slot_log,
                  pos0, count) with ``block_bits``, where the last range ends, beside 2^``k_max``

The pin is the circuit file's SHA-512, so it holds whenever the same tree stages the unit this way."""
import json
import sys
from pathlib import Path

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.windows import interface as IF
from verity.definitions.registry import boolean_dsa_topk_mask as BM, dsa_topk_mask_cut as X, dsa_topk_reads as R
from verity_flock.class_statement import UnitStager


def committing(build):
    def run(C, *a):
        return [C.commit(list(w)) for w in build(C, *a)]
    return run


kind, out = sys.argv[1], Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else X.WHOLE_MAX
K = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
fn = {"positions": lambda: bind(R.DsaMaskPositionsV2, N=N, AT=0, K=K), "whole": lambda: bind(X.DsaTopKMaskWholeV2, N=N, TOPK=K)}[kind]()

BM.DsaCountSum.build = committing(BM.DsaCountSum.build)

in_rows, out_rows = IF.unit_rows(fn)
g = {nm: k for k, (nm, _ty) in enumerate(fn.signature()[0])}
in_rows = [(f"in{g[p]}", *rest) for p, *rest in in_rows]
out.mkdir(parents=True, exist_ok=True)
st = UnitStager(fn, "unit/serving-rows-cap", in_rows, out_rows, circuit=out / "circuit.txt",
                bind={"partition": "0" * 128, "program": "0" * 128}, reads="inline")
c = st.low.layout.counts
rec = {"id": descriptor_id(fn), "pin": st.pin, "class_sha512": st.class_sha512, "circuit_bytes": (out / "circuit.txt").stat().st_size,
       "k_log": st.meta["k_log"], "unit_log": st.meta["unit_log"], "ands": int(c["ands"]), "rows": int(c["rows"]),
       "nonzeros": int(c["nnz_a"]) + int(c["nnz_b"]),
       "ranges": [[r["net"], r["slot_log"], r["pos0"], r["count"]] for r in st.meta["ranges"]],
       "block_bits": max((r["pos0"] + r["count"]) << r["slot_log"] for r in st.meta["ranges"]), "k_max": IF.K_MAX}
rows = {p: np.zeros((1, -(-bits // 16)), dtype=np.uint16) for p, _sch, bits, _ps in in_rows}
got = st.stage(rows, out / "round", bytes(32), indices=[0])
rec["instance"] = {"ands": int(got["ands"]), "unit_nnz": int(got["unit_nnz"]),
                   "files": {f.name: f.stat().st_size for f in sorted((out / "round").iterdir())}}
(out / "staged.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
