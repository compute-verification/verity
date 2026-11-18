"""The toy's C* (`store-toy/`) as an encoded store, with its registration: what `bandwidth.py session --ref-decode`
serves at fixture scale.

    bandwidth_toy.py OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str((Path(__file__).resolve().parents[3] / "verity/protocols/resource_accounting/memory/poub/prover/p2_stage/bandwidth.py").parent))

import bandwidth as B
from p2_store import store as ST
from p2_v1 import serve as SV

from verity.protocols.resource_accounting.memory.poub.verifier import p2
from verity.protocols.resource_accounting.memory.pous.verifier import verifier as V


def toy(out: Path) -> p2.Registration:
    """The toy's C* (`store-toy/`) as an encoded store under `out/store` (`setup.json`, `store.json`, `seg/`, `rows/`)
    and `out/layout.json`, with the verifier's registration of it (PoUS's setup decoding C* to the committed W) and its
    tensors' plan in `out/registration.json`: what `session --ref-decode` serves at fixture scale."""
    s, doc, c_star = SV.toy()
    reg = SV.registration(s, doc, c_star, SV.NODE)
    params, root = s.params, out / "store"
    bps = params.blocks_per_segment
    (root / "seg").mkdir(parents=True, exist_ok=True)
    (root / "rows").mkdir(exist_ok=True)
    (out / "layout.json").write_text(json.dumps(doc["layout"]))
    lsha = hashlib.sha512((out / "layout.json").read_bytes()).hexdigest()
    salt = bytes.fromhex(doc["salt"])
    (root / "setup.json").write_text(json.dumps({"commitment": doc["commitment"], "layout_sha512": lsha,
                                                 "manifest": doc["layout"]["params"], "source": doc["source"],
                                                 "salt": doc["salt"]}))
    for k in range(params.segments):
        data = c_star[B.BLOCK * bps * k:B.BLOCK * bps * (k + 1)]
        rows = b"".join(V.row_digest(B.BLOCK, data[B.BLOCK * v:B.BLOCK * (v + 1)]) for v in range(bps))
        (root / "seg" / f"{k:05d}.c").write_bytes(data)
        (root / "rows" / f"{k:05d}.rows").write_bytes(rows)
        meta = {"segment": k, "sha512": hashlib.sha512(data).hexdigest(), "rows_sha512": hashlib.sha512(rows).hexdigest()}
        if bps & (bps - 1) == 0:
            meta["root"] = ST.segment_root(params, salt, k, rows).hex()
        (root / "seg" / f"{k:05d}.json").write_text(json.dumps(meta))
    leaves = [V.leaf_hash(params, salt, j, c_star[B.BLOCK * j:B.BLOCK * (j + 1)]) for j in range(params.blocks)]
    (root / "store.json").write_text(json.dumps({"commitment": doc["commitment"], "salt": doc["salt"],
                                                 "source": doc["source"], "root": V.MerkleTree(leaves).root.hex()}))
    B.registration_file(reg, out / "registration.json", B.tensor_plan(doc["layout"], reg.blocks))
    return reg


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    reg = toy(a.out)
    print(json.dumps({"registration": reg.digest.hex(), "units": reg.blocks, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
