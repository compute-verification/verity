"""stage_tile.py COMMIT_DIR PARAMS UNIT OUT: unit UNIT of a served window as one Flock --zk unit, on the committer's rows.

The unit is `Pc8TileSide{K=4096,TM=1,TN=1,DEV=sm120}`: A row i (fp8-pi, role X, row-seg) and B row j (fp8-pi, role W, v2)
of the window's entry the unit falls in (`pouw.window.tile_at` of ``advice.json``: a call's kept rows, or the filler's
zero rows), each a side row with its index field (`pouw.window.side_row`), under the salts `served_commit.py commit`
committed them with, so the proof's committed rows have the b ‖ c the entry's leaves hold; the tile digest is its output
row.  The stager (`Stager.stage`: `class_statement.stage_unit`) writes circuit.txt, inst.bin and pub.bin under OUT; this
writes pin.txt, stage.json and opening.json (the prover's: the entry, its salt and path to the window root, and each row's
frame-v3 path to the entry's side root).  As `hidden_zk/stage.py`'s hiding tile, the PoUW word primitives lower through
`hash_scratch`'s pieces, checked against their primitives first, so a result is labelled 'hash scratch'; ``staging``
(``fast``, the default, or ``library``) as there.  A window committed before side rows carried their index is refused.

Flock's staging is `verity_flock`'s and `hidden_zk`'s, which verity does not import: `Stager.stage` raises here, and
`benchmarks/pouw/hidden_zk/served_stage.py`, which runs this with its own `Stager`, supplies it.

  python benchmarks/pouw/hidden_zk/served_stage.py COMMIT_DIR PARAMS UNIT OUT"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments.rowleaf import ROLE_W, ROLE_X
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import leaves as L
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pc8
from verity.protocols.resource_accounting.compute.pouw.prover.kernels.served_zk import served_commit as SC
from verity.protocols.resource_accounting.compute.pouw.verifier import window as Wd

TILE = (1, 1)
DEV = "sm120"


class Stager:
    """One Flock ``--zk`` unit's staging: `stage` is `verity_flock.class_statement.stage_unit` under `staging`'s
    lowering (`hidden_zk`'s ``fast``, or the library's), through `hash_scratch`'s pieces once they agree with their
    primitives."""

    def stage(self, d: Any, name: str, in_rows: list, out_rows: list, rows: dict, out: Path, seed: bytes, salts: dict,
              staging: str) -> dict:
        """`stage_unit`'s result, with the staged circuit's `circuit_sha512`, `class_sha512` and `circuit_bytes`."""
        raise NotImplementedError("Stager.stage needs verity_flock and hidden_zk's staging, which verity does not import "
                                  "(compute-accounting/pouw-package-holds-kernels): benchmarks/pouw/hidden_zk/"
                                  "served_stage.py's Stager supplies it")


def main(argv: list[str] | None = None, *, stager: Stager | None = None, staging: str = "fast") -> None:
    t0 = time.time()
    argv = sys.argv[1:] if argv is None else argv
    src, params, unit, out = Path(argv[0]), json.loads(Path(argv[1]).read_text()), int(argv[2]), Path(argv[3])
    stager = Stager() if stager is None else stager
    out.mkdir(parents=True, exist_ok=True)
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
    got = stager.stage(d, "unit/pouw-pc8-hidden-served", in_rows, out_rows, rows, out, os.urandom(32), salts, staging)
    pin = got["circuit_sha512"]
    (out / "pin.txt").write_text(pin)
    body = (out / "pub.bin").read_bytes().split(b"\n", 1)[1]
    assert body[:128] == side["a"]["bcs"][i] and body[128:256] == side["b"]["bcs"][j], \
        "the staged rows' b || c are not the committed rows'"
    (out / "opening.json").write_text(json.dumps(SC.opening(advice, priv, params, unit)))
    rec = {"definition": d.id, "k": k, "tile": list(TILE), "unit": unit, "entry": e, "kind": advice["entries"][e]["kind"],
           "binding": "hash scratch", "staging": staging,
           "ands": got["ands"], "k_log": got["meta"]["k_log"], "row_schemas": got["meta"]["row_schemas"],
           "circuit_bytes": got["circuit_bytes"], "circuit_sha512": pin,
           "class_sha512": got["class_sha512"], "tile_digest": L.digest_bytes(
               [int(x) for x in np.asarray(got["outs"]["o0"][0], dtype="<u2").view("<u4")]).hex(),
           "seconds": round(time.time() - t0, 1),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    (out / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
