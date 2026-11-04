"""Measure the reference circuits' structural properties and set them beside benchmarks/rho's shape-derived ones.

    uv run --extra torch-cpu verity bench circuit_properties [--models NAME,...] [--T T] [--out DIR]

For each configuration in `references.configs()`, one b1 layer at T positions on bits (`references.layer`) is measured
(`measure.measure`) under `references.GEMM_PRODUCTS`, and so is the Boolean GEMM `Gemm_v3{K=32,N=4}`. Beside each layer's
a_out, a_in and |𝒥| per position: rho's `fanout`, `legs` and active layer products, as recorded for the model in
benchmarks/rho/data/models.toml when it has a row, and as `derived_fanout` / `derived_legs` derive them from the
config's own matrices (q, k, v, o, gate, up, down) otherwise and always. Each circuit's products are also checked
whole-in-block (`measure.whole`) under two partitions: one block per root activation, and the gates cut in half. Writes
DIR/properties.json (DIR defaults to $OUT) and prints one line per circuit.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rho"))

from circuit_properties import references as R
from circuit_properties.measure import Blocks, describe, measure, whole


def rho_shapes(name: str, C: dict) -> dict:
    from rho import instances as I
    from rho.params import Registry
    H, NH, KVH, D, IM, NL = C["H"], C["NH"], C["KVH"], C["D"], C["I"], C["NL"]
    own = I.Model(name, 0, 0, H, NL, C["V"], 0, 0, matrices={
        "attention": tuple(I.Matrix(n, a, b, NL, NL) for n, a, b in
                           (("q", H, NH * D), ("k", H, KVH * D), ("v", H, KVH * D), ("o", NH * D, H))),
        "dense": tuple(I.Matrix(n, a, b, NL, NL) for n, a, b in (("gate", H, IM), ("up", H, IM), ("down", IM, H)))})
    out = {"config": {"fanout": I.derived_fanout(own), "legs": I.derived_legs(own),
                      "layer_products": own.active(I.LAYERED) / NL}}
    reg = Registry.load()
    if name in reg.names("model"):
        m = I.model(reg, name)
        out["models.toml"] = {"fanout": m.fanout, "legs": m.legs, "derived_fanout": I.derived_fanout(m),
                              "derived_legs": I.derived_legs(m), "layer_products": m.active(I.LAYERED) / m.layers}
    return out


def row(name: str, program, source: str, T: int, rho: dict | None) -> dict:
    t = time.time()
    m = measure(program, R.GEMM_PRODUCTS)
    out = {"circuit": name, "source": source, "positions": T, "program_fn": program.fn.id[:120],
           "seconds": round(time.time() - t, 2), **m.as_dict(), "a_out_at": describe(program, m.a_out_gate)}
    out["products_per_position"] = m.products // T
    out["whole"] = {}
    for part, blocks in (("by_root_activation", Blocks.by_root(program)),
                         ("halves", Blocks([(0, 0), (program.gates // 2, 1)]))):
        w = whole(program, R.GEMM_PRODUCTS, blocks)
        out["whole"][part] = {"per_block": {str(b): c for b, c in w.per_block.items()}, "split": w.split,
                              "split_at": [describe(program, g) for g, _ in w.split_examples[:2]]}
    if rho is not None:
        out["rho"] = rho
        cmp = {}
        for src, r in rho.items():
            cmp[src] = {"a_out": m.a_out == r.get("fanout"), "a_in": m.a_in == r.get("legs"),
                        "products": m.products // T == r["layer_products"]}
            if "derived_fanout" in r:
                cmp[src].update(derived_a_out=m.a_out == r["derived_fanout"], derived_a_in=m.a_in == r["derived_legs"])
        out["agrees"] = cmp
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    names = list(R.configs())
    ap.add_argument("--models", default=",".join(names))
    ap.add_argument("--T", type=int, default=1)
    ap.add_argument("--out", default=os.environ.get("OUT"))
    a = ap.parse_args()
    rows = [row("Gemm_v3{K=32,N=4}", R.boolean_gemm(32, 4), "verity.ml.circuits.ops.boolean.gemm", 1, None)]
    print(json.dumps({k: rows[0][k] for k in ("circuit", "gates", "products", "s_x", "a_out", "a_in")}), flush=True)
    configs = R.configs()
    for name in a.models.split(","):
        C, source = configs[name]
        t = time.time()
        program = R.layer(C, a.T)
        built = round(time.time() - t, 2)
        r = row(name, program, source, a.T, rho_shapes(name, C))
        r["build_seconds"] = built
        r["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("circuit", "gates", "products_per_position", "s_x", "a_out", "a_in", "agrees",
                                            "seconds", "build_seconds", "maxrss_gb")}
                         | {"split": {p: v["split"] for p, v in r["whole"].items()}}), flush=True)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "properties.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
