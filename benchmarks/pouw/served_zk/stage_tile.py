"""stage_tile.py COMMIT_DIR PARAMS UNIT OUT: unit UNIT of a served window as one C-Flock --zk unit, on the committer's rows.

The unit is `Pc8TileSide{K=4096,TM=1,TN=1,DEV=sm120}`: A row i (fp8-pi, role X, row-seg) and B row j (fp8-pi, role W, v2)
of the window's entry the unit falls in (`pouw.window.tile_at` of ``advice.json``: a call's kept rows, or the filler's
zero rows), each a side row with its index field (`pouw.window.side_row`), under the salts `served_commit.py commit`
committed them with, so the proof's committed rows have the b ‖ c the entry's leaves hold; the tile digest is its output
row.  `class_statement.stage_unit` writes circuit.txt, inst.bin and pub.bin under OUT; this writes pin.txt, stage.json
and opening.json (the prover's: the entry, its salt and path to the window root, and each row's frame-v3 path to the
entry's side root).  As `hidden_zk/stage.py`'s hiding tile, the PoUW word primitives lower through `hash_scratch`'s
pieces, checked against their primitives first, so a result is labelled 'hash scratch'; `HZ_STAGE` (``fast``, the
default, or ``library``) as there.  A window committed before side rows carried their index is refused."""
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

from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X  # noqa: E402
from verity.core.protocols.pouw.circuit import leaves as L  # noqa: E402
from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.pouw.circuit import pc8  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402

import served_commit as SC  # noqa: E402

TILE = (1, 1)
DEV = "sm120"


def main() -> None:
    t0 = time.time()
    src, params, unit, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), int(sys.argv[3]), Path(sys.argv[4])
    stage = os.environ.get("HZ_STAGE", "fast")
    out.mkdir(parents=True, exist_ok=True)
    if stage == "fast":
        import fast_stage
        fast_stage.install(out)
    import hash_scratch as H
    bad = H.check()
    assert not any(bad.values()), f"hash_scratch's pieces disagree with their primitives: {bad}"
    IL.PIECES.update(H.PIECES)
    advice = json.loads((src / "advice.json").read_text())
    priv = SC.load(src / "prover.npz")
    k = params["k"]
    e, i, j = Wd.tile_at(SC.entries_of(advice), unit)
    side = {"a": priv[e]["a"], "b": priv[e]["b"]}
    bits = 16 * L.ROW_WORDS["fp8-pi"](k)
    if any(16 * side[s]["words"].shape[1] != bits for s in side):
        raise SystemExit(f"{src}: its side rows are {side['a']['words'].shape[1]} words, not Pc8TileSide's fp8-pi rows of "
                         f"{bits // 16}: a window committed before side rows carried their index (pouw.window.side_row)")
    d = SC.template(k)
    in_rows = [("a0", L.ROW_SCHEMA["fp8-pi", ROLE_X], bits, ["ac[0]", "pa[0]", "ia[0]"]),
               ("b0", L.ROW_SCHEMA["fp8-pi", ROLE_W], bits, ["bc[0]", "pb[0]", "jb[0]"])]
    sr = L.statement_rows(pc8.SIDE_PORTS, k, {p: 1 for p in ("ac", "pa", "ia", "bc", "pb", "jb")})
    outs_ = sr["row_schemas"][2:]
    runs = sr.get("out_row_ports") or [1] * len(outs_)
    out_rows, at = [], 0
    for (s, _b), r in zip(outs_, runs):
        out_rows.append((s, tuple(range(at, at + r))))
        at += r
    rows = {"a0": side["a"]["words"][i][None, :], "b0": side["b"]["words"][j][None, :]}
    salts = {"a0": [side["a"]["salts"][i]], "b0": [side["b"]["salts"][j]]}
    got = CS.stage_unit(d, "unit/pouw-pc8-hidden-served", in_rows, out_rows, rows, out, os.urandom(32), salts=salts)
    text = fast_stage.composed() if stage == "fast" else (out / "circuit.txt").read_text()
    pin = FC.digest(text)
    (out / "pin.txt").write_text(pin)
    body = (out / "pub.bin").read_bytes().split(b"\n", 1)[1]
    assert body[:128] == side["a"]["bcs"][i] and body[128:256] == side["b"]["bcs"][j], \
        "the staged rows' b || c are not the committed rows'"
    (out / "opening.json").write_text(json.dumps(SC.opening(advice, priv, params, unit)))
    rec = {"definition": d.id, "k": k, "tile": list(TILE), "unit": unit, "entry": e, "kind": advice["entries"][e]["kind"],
           "binding": "hash scratch", "staging": stage,
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
