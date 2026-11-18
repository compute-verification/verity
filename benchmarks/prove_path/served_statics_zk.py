"""The cost of the served rows held to their Programs' statics (`served_statics.ServedStaticsTie_v1`).

    python benchmarks/prove_path/served_statics_zk.py cost [--out DIR]

A unit's ANDs (`served_statics.size`) and Flock block (``zk_block``) at the compile slice's class, the sealed run 9c10's
and the served class, at every G dividing P, and the largest G whose unit fits Flock's 2^K_MAX. The unit hashes its two
rows, the block's statics and its served rows, and no more.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from verity_vllm.compiler import served_statics as SS

CLASSES = {
    "slice": dict(R=2, P=7, IW=6, L=4096, T=256),
    "9c10": dict(R=8, P=256, IW=16, L=512, T=2048),
    "served": dict(R=256, P=256, IW=16, L=131072, T=8192),
}


def log(msg: str) -> None:
    print(f"STZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def zk_block(rows: list[tuple[str, int]], ands: int) -> dict:
    """A unit's Flock block as `circuit.compose` places it for one VU, on these rows."""
    from verity_flock import circuit as FC

    ports = [FC.Port(r, -(-b // 16), b) for r, b in rows]
    n_sha, _ = FC._row_slots(ports)
    ul = FC._slot_log(ands + sum(b for _r, b in rows))
    _r, k = FC._ranges([("unit", FC.SHA_LOG, n_sha, {"packed": True}), ("unit", FC.HM_LOG, FC._pow2(len(ports)), {}),
                        ("unit", ul, 1, {}), ("mask", FC.MASK_LOG, 1, {})])
    return {"k_log": k, "fits": k <= FC.K_MAX, "sha_slots": n_sha, "unit_log": ul,
            "row_bytes": {r: b // 8 for r, b in rows}}


def cost(a) -> int:
    out = {}
    for name, c in CLASSES.items():
        t = time.time()
        step = SS.step_size(**c)["ands"]
        by = {}
        for G in [g for g in range(c["P"], 0, -1) if c["P"] % g == 0 and (g & (g - 1) == 0 or g == c["P"])]:
            st = {**SS.statics(SimpleNamespace(**c), G=G)}
            unit = SS.size(**st)["ands"]
            by[G] = {"units": c["P"] // G, "unit_ands": unit, "window_ands": c["P"] // G * unit,
                     "zk_block": zk_block(SS.rows(SimpleNamespace(**st)), unit)}
        fit = [G for G, r in by.items() if r["zk_block"]["fits"]]
        out[name] = {"statics": c, "step_ands": step, "by_G": by, "largest_G_fitting": max(fit) if fit else None,
                     "seconds": round(time.time() - t, 1)}
        log(f"{name}: {json.dumps(out[name])}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "cost.json").write_text(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="served_statics_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cost", help="a unit's ANDs and Flock block at the slice's, 9c10's and the served class, by G")
    c.add_argument("--out")
    a = ap.parse_args(argv)
    return {"cost": cost}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
