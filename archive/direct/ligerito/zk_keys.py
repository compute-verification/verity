"""The LGSC0004 (``--zk``) verifying keys a Rust verifier must pin: one per (relation, column count C) -- the committed
next / mask / product rows sit below the ``ceil(g_cells / C)`` Libra rows -- and one per relation once C >= g_cells.

    python -m archive.direct.ligerito.zk_keys [--out DIR] [--vus 12:256 4096:16384 ...] [--relation R ...]

Per (relation, VUs, l): the layout's C, rows, the system id, the ZK key's sha256 (``Key.digest``) and the non-ZK key's;
``--out`` also writes each ZK key (``<relation>_zk_C<C>.lgvk``) and ``zk_keys.json``."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .layout import layout_for
from .prove import Prover

RELATIONS = ["fp8-ada", "bf16-hopper", "fp8-hopper", "bf16-ampere", "fp4-nvf4"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", nargs="*", default=RELATIONS)
    ap.add_argument("--vus", nargs="*", default=["12:0", "4096:16384", "16384:16384"],
                    help="VUS:L (L 0 = run.py's default: 256 up to 64 VUs, else 16384)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows, provers = [], {}
    for rel in args.relation:
        for spec in args.vus:
            vus, l = (int(x) for x in spec.split(":"))
            l = l or (256 if vus <= 64 else 16384)
            pv = provers.get((rel, l)) or provers.setdefault((rel, l), Prover(rel, args.device, l))
            per = pv.l // pv.steps
            n_vus = [min(per, vus - i) for i in range(0, vus, per)]
            lay = layout_for(pv.sys, pv.l, n_vus, pv.steps, zk=True)
            _, key, kb, ksha = pv.key_for(lay)
            row = {"relation": rel, "vus": vus, "l": pv.l, "C": lay.C, "m": lay.m, "K": lay.K,
                   "system_id": pv.sys_id.hex() if isinstance(pv.sys_id, bytes) else pv.sys_id,
                   "zk_key_sha256": ksha.hex() if isinstance(ksha, bytes) else ksha,
                   "zk_key_file_sha256": hashlib.sha256(kb).hexdigest(), "zk_key_bytes": len(kb),
                   "nonzk_key_sha256": pv.key_sha256.hex() if isinstance(pv.key_sha256, bytes) else pv.key_sha256}
            if out:
                name = f"{rel}_zk_C{lay.C}.lgvk"
                (out / name).write_bytes(kb)
                row["file"] = name
            rows.append(row)
            print(json.dumps(row), flush=True)
    if out:
        (out / "zk_keys.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
