"""red-team-standard-hash: SP1 relation-committed/v1's tree check accepts roots the prover chose (the R2 pattern).

A statement names the frozen fp8-ada set's range [0, 1) (its dataset, tier and manifest, so its bindings and id are the
ones `check` re-derives), but its roots are those of an all-zero VU with y = 0.  The honest guest accepts that VU (the
zero dot product is +0; `committed.rs` test `decide_parse_and_check_round_trip_and_reject`) and commits exactly the public
values built here, so a real SP1 proof of them exists under the pinned vk.  `verity_sp1.committed.check` (the Python side
of the host's `committed::check`) accepts; only roots recomputed from the instance set (as 06-core-roots does for
B-Ligero) tell the statement from the frozen set's.

    python -m backends.direct.ligero.redteam.rtsh_sp1_roots --committed PATH/TO/verity_sp1/committed.py --out OUT.json
"""
import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", required=True, help="sp1-committed's backends/sp1/python/verity_sp1/committed.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("sp1_committed", args.committed)
    C = importlib.util.module_from_spec(spec)
    sys.modules["sp1_committed"] = C
    spec.loader.exec_module(C)

    import numpy as np

    from .. import relchain
    from ..relations import relation

    rel = relation("fp8-ada")
    fmt = C.FORMATS["fp8-ada"]
    inst = {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "manifest_sha256": relchain.instances_digest(rel, 1),
            "lo": 0, "hi": 1}
    v0 = relchain.instances(rel, 1)[0]
    x_true = np.asarray(v0[0]).reshape(-1).astype(np.uint8).tobytes()
    w_true = np.asarray(v0[1]).reshape(-1).astype(np.uint8).tobytes()
    y_true = int(rel.y_public(int(v0[3])))
    assert len(x_true) == len(w_true) == fmt.row_bytes, (len(x_true), fmt.row_bytes)

    def public_of(x: bytes, w: bytes, y: int) -> tuple[dict, bytes]:
        stmt = C.statement(fmt, inst, x, w, struct.pack("<I", y))
        fv3 = C.configuration(fmt, inst)
        dx, dw = C.digests_of_rows(fmt, fv3, x, w)
        head = C.TAG + struct.pack("<I", fmt.id) + bytes.fromhex(stmt["id"]) + struct.pack("<II", C.K, 1) + bytes([C.VERDICT_ACCEPT])
        return stmt, head + struct.pack("<I", y) + dx + dw

    zero = bytes(fmt.row_bytes)
    forged_stmt, forged_pub = public_of(zero, zero, 0)
    honest_stmt, honest_pub = public_of(x_true, w_true, y_true)
    roots = lambda s: {t["name"]: t["root"] for t in s["trees"]}  # noqa: E731
    res = {
        "instances": inst,
        "frozen_vu0": {"y": y_true, "x_row_nonzero_bytes": int(np.count_nonzero(np.frombuffer(x_true, np.uint8)))},
        "forged": {"rows": "all-zero x row and W column, y = 0", "roots": roots(forged_stmt),
                   "bindings_equal_frozen": [t["binding"] for t in forged_stmt["trees"]] == [t["binding"] for t in honest_stmt["trees"]],
                   "id_equal_frozen": forged_stmt["id"] == honest_stmt["id"],
                   "check": C.check(forged_pub, forged_stmt)},
        "honest": {"roots": roots(honest_stmt), "check": C.check(honest_pub, honest_stmt)},
        "cross": {"forged_public_vs_frozen_statement": C.check(forged_pub, honest_stmt)},
    }
    res["forged_accepted"] = res["forged"]["check"] is None
    res["roots_differ_from_frozen"] = {n: roots(forged_stmt)[n] != roots(honest_stmt)[n] for n in "aby"}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    ok = res["forged_accepted"] and res["forged"]["bindings_equal_frozen"] and res["forged"]["id_equal_frozen"] and \
        res["roots_differ_from_frozen"]["a"] and res["honest"]["check"] is None and res["cross"]["forged_public_vs_frozen_statement"]
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
