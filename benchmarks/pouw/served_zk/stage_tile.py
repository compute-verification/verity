"""stage_tile.py COMMIT_DIR DRAW OUT: the drawn tile of a served call as one C-Flock --zk unit, on the committer's rows.

The unit is `Pc8TileHidden{K=4096,TM=1,TN=1,DEV=sm120}`: A row i (fp8-p, role X, row-seg) and B row j (fp8-p, role W, v2)
of the call `served_commit.py commit` committed, under the same salts, so the proof's committed rows have the b ‖ c the
served commitment's leaves hold; the tile digest is its output row.  `class_statement.stage_unit` writes circuit.txt,
inst.bin and pub.bin under OUT; this writes pin.txt, stage.json and opening.json (each row's frame-v3 path to the served
commitment's root, the prover's).  As `hidden_zk/stage.py`'s hiding tile, the PoUW word primitives lower through
`hash_scratch`'s pieces, checked against their primitives first, so a result is labelled 'hash scratch'; `HZ_STAGE`
(``fast``, the default, or ``library``) as there."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hidden_zk"))

from verity.primitives.commitments.rowleaf import ROLE_W, ROLE_X  # noqa: E402
from verity.protocols.accounting.work.pouw.circuit import leaves as L  # noqa: E402
from verity.protocols.accounting.work.pouw.circuit import pc8  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402

import served_commit as SC  # noqa: E402

TILE = (1, 1)
DEV = "sm120"


def main() -> None:
    t0 = time.time()
    src, draw, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    stage = os.environ.get("HZ_STAGE", "fast")
    out.mkdir(parents=True, exist_ok=True)
    if stage == "fast":
        import fast_stage
        fast_stage.install(out)
    import hash_scratch as H
    bad = H.check()
    assert not any(bad.values()), f"hash_scratch's pieces disagree with their primitives: {bad}"
    IL.PIECES.update(H.PIECES)
    c = json.loads((src / "commitment.json").read_text())
    priv = SC.load(src / "prover.npz")
    k = c["binding"]["shape"][1]
    if len(draw["units"]) != 1:
        raise SystemExit(f"the draw has {len(draw['units'])} units; this stages one tile")
    i, j = SC.tile_of(*c["statement"]["tiles"], draw["units"][0])
    d = pc8.templates(k, TILE, DEV, hidden=True)["tile"]
    bits = 16 * L.ROW_WORDS["fp8-p"](k)
    in_rows = [("a0", L.ROW_SCHEMA["fp8-p", ROLE_X], bits, ["ac[0]", "pa[0]"]),
               ("b0", L.ROW_SCHEMA["fp8-p", ROLE_W], bits, ["bc[0]", "pb[0]"])]
    sr = L.statement_rows(pc8.PORTS["tile"], k, {p: 1 for p in ("ac", "pa", "bc", "pb")})
    outs_ = sr["row_schemas"][2:]
    runs = sr.get("out_row_ports") or [1] * len(outs_)
    out_rows, at = [], 0
    for (s, _b), r in zip(outs_, runs):
        out_rows.append((s, tuple(range(at, at + r))))
        at += r
    rows = {"a0": priv["a"]["words"][i][None, :], "b0": priv["b"]["words"][j][None, :]}
    salts = {"a0": [priv["a"]["salts"][i]], "b0": [priv["b"]["salts"][j]]}
    got = CS.stage_unit(d, "unit/pouw-pc8-hidden-served", in_rows, out_rows, rows, out, os.urandom(32), salts=salts)
    text = fast_stage.composed() if stage == "fast" else (out / "circuit.txt").read_text()
    pin = FC.digest(text)
    (out / "pin.txt").write_text(pin)
    body = (out / "pub.bin").read_bytes().split(b"\n", 1)[1]
    assert body[:128] == priv["a"]["bcs"][i] and body[128:256] == priv["b"]["bcs"][j], \
        "the staged rows' b || c are not the committed rows'"
    (out / "opening.json").write_text(json.dumps({"a": SC.opening(c, "a", priv["a"]["bcs"], i),
                                                  "b": SC.opening(c, "b", priv["b"]["bcs"], j)}))
    rec = {"definition": d.id, "k": k, "tile": list(TILE), "draw": draw, "binding": "hash scratch", "staging": stage,
           "ands": got["ands"], "k_log": got["meta"]["k_log"], "row_schemas": got["meta"]["row_schemas"],
           "circuit_bytes": fast_stage.size(text) if stage == "fast" else len(text), "circuit_sha512": pin,
           "class_sha512": FC.class_digest(text), "tile_digest": L.digest_bytes(
               [int(x) for x in np.asarray(got["outs"]["o0"][0], dtype="<u2").view("<u4")]).hex(),
           "seconds": round(time.time() - t0, 1),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    if stage == "fast":
        del text
        fast_stage.release()
    (out / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
