"""Run A9: GLM-5.3 TP8 B1's routed-expert GEMMs (91.6% of the step's computing gates) in the static pair form, on the Match
capture's values. It stages values for `a8.py --load-values` (the one-stage audit, M0 under ``--zk``, the Lean verifier) and
records what decides whether each unit can go there.

    python benchmarks/one_stage/a9.py --values DIR --roles derive-report.json --out OUT --save-values SAVED

``DIR`` is a8_slice.py's value slice with ``ckpt/`` holding ``a8_slice.ckpt_experts``' share of layer 3's routed experts (the
rank's gate/up rows and down columns, its block scales on the same grid), which `ShareWeights` composes at world 1: at TP8 each
cut is two whole 128-blocks, so that is the world-8 composition of the whole tensors.

For each of the w13 launch (`MoeExpertFp8Gemm_v1`) and the w2 launch (`MoeExpertFp8GemmW_v1`), at decode (step 1, M = 1: the
fold binds ``DOT=HopperE4m3MmaV2Dot32``, ``SWAP=False``) and prefill (step 0, M = 256: ``HopperE4m3QgmmaDot32``, ``SWAP=True``),
on layer 3's first routed pair:

- ``twin``: every output word of the pair, by `fp8_moe_rows`' numpy twin on the Call's operands, against the capture's output
  snapshot (`a8.call_twin`);
- ``configs``: the same with each of the four (DOT, SWAP) launch configs, so whether one pinned config reproduces both launches;
- ``units``: the coordinate unit (`MoeExpertFp8Coordinate(W)_v1`, the word Definition through the IR reference, and the Boolean
  static pair form `MoeExpertFp8Coordinate(W)_v2`, the expert's row picked by `boolean.gather`'s multiplexer tree inside the unit,
  bit-sliced) on ``--coords``' coordinates, each against the capture's word;
- ``m0``: the unit's rows (`a8_kinds.rows_of`) and the block M0 needs for them alone (`circuit.fits`' ranges without the unit's own),
  against ``K_MAX``.

The units that agree everywhere and fit M0 are saved (`a8.save_values`' layout, one kind per launch) with ``kinds.json``, a
table naming them, for ``a8.py --load-values SAVED --kinds SAVED/kinds.json``. ``OUT/a9.json`` holds every record.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a8  # noqa: E402
import a8_kinds as K  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.evaluation import evaluate_batch  # noqa: E402
from verity.ml.circuits.ops.boolean import gemm as BG  # noqa: E402
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402

UP, DOWN = "MoeExpertFp8Gemm_v1", "MoeExpertFp8GemmW_v1"
STEP_NAME = {1: "m1", 0: "m256"}
#: each word DOT step's Boolean version, which the `_v2` forms lift (`elementwise.on_bits(..., lifted=["DOT"])`)
BOOL_DOT = {HopperE4m3MmaV2Dot32.id: BG.HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32.id: BG.HopperE4m3QgmmaDot32}
CONFIGS = [("mmav2", False), ("hopper", True), ("mmav2", True), ("hopper", False)]


class ShareWeights(a8.Weights):
    """`a8.Weights` whose routed-expert stacks are composed at world 1 from the rank's share in ``ckpt/`` (a8_slice.ckpt_experts),
    the rest as `a8.Weights` composes them."""

    def _at_world_1(self, fn, arg):
        w, self.world = self.world, 1
        try:
            return fn(arg)
        finally:
            self.world = w

    def _raw(self, field: str):
        return self._at_world_1(super()._raw, field) if ".mlp.experts." in field else super()._raw(field)

    def field(self, name: str) -> np.ndarray:
        if ".mlp.experts." not in (self.served.get(name) or "") or name in self._cache:
            return super().field(name)
        w = self._at_world_1(super().field, name)
        self.notes[name]["shard"] = f"rank {self.rank}/{self.world}: the share a8_slice.ckpt_experts cut, composed at world 1"
        return w


def pick(res, kind: str, step: int, layer: int):
    """The kind's first Call of the step in the layer (its first routed pair)."""
    cands = [m for m in res.instances if K.family(m["spec"]) == kind and m["step"] == step
             and (lm := a8.LAYER.search(m["name"])) and int(lm[1]) == layer]
    return min(cands, key=lambda m: (m["event"], m["node"])) if cands else None


def units_of(spec):
    """The Call's coordinate unit: the word `_v1` and the Boolean `_v2` at the Call's statics."""
    from verity_vllm.program.registry import boolean_fp8_moe as B, fp8_moe_experts as X
    s = dict(spec.bindings)
    w = K.family(spec.id) == DOWN
    st = {k: s[k] for k in ("E", "K", "G", "KB", "SWAP")}
    v1 = bind(X.MoeExpertFp8CoordinateW if w else X.MoeExpertFp8Coordinate, **st, DOT=s["DOT"])
    v2 = bind(B.MoeExpertFp8CoordinateWV2 if w else B.MoeExpertFp8CoordinateV2, **st, DOT=BOOL_DOT[s["DOT"].id])
    return v1, v2, s


def capture_words(store, m: dict, late: dict | None) -> np.ndarray:
    """The Call's output words as the capture holds them: its one output view's snapshot at the Call (or the later one
    `a8.call_twin` held it to)."""
    view = m["output_views"][-1]
    ev = (late or {}).get(len(m["output_views"]) - 1)
    s = next((x for x in store.by_alloc.get(view.alloc, []) if x.event == ev and store.covers(x, view)), None) if ev is not None \
        else store.find(view, at=m["event"])
    return np.asarray(a8.view_words(store, s, view, m, "out", 0), dtype=np.uint64)


def rows_eval(defn, rows_spec, rows: dict) -> np.ndarray:
    """``defn`` on each instance's rows (`class_statement.UnitStager.reference_check`'s reading of rows into leaves), its return
    as one word: its leaves concatenated least significant first, so a word Definition's one bf16 leaf and a Boolean one's
    16 bits read alike."""
    params, ret = defn.signature()
    types, leaves = dict(params), {}
    for r, _s, _b, ps in rows_spec:
        bits, at = CS._row_bits(rows[r]), 0
        for q in ps:
            ws = CS._leaf_bits(types[q])
            leaves[q] = CS._leaf_words(bits, ws, at + np.cumsum([0] + ws[:-1]))
            at += sum(ws)
    args = [leaves[nm] for nm, _t in params]
    out = np.asarray(evaluate_batch(defn, *args)).reshape(len(args[0]), ret.leaves).astype(np.uint64)
    at = np.cumsum([0] + CS._leaf_bits(ret)[:-1]).astype(np.uint64)
    return (out << at).sum(axis=1)


def m0_rows(rows_spec, out_bits: int = 16) -> dict:
    """The input rows' and one output row's SHA-512 compressions, the ``sha512x3`` slots they take, and the block M0 needs for
    them alone at one unit per block (`circuit.compose`'s ``fits``, its unit, stage and lookup ranges left out): a lower bound
    on the unit's ``k_log``."""
    ports = [FC._port(r, -(-b // 16), b) for r, _s, b, _ps in rows_spec] + [FC._port("o0", -(-out_bits // 16), out_bits)]
    n_sha, _ = FC._row_slots(ports)
    items = [("unit", FC.SHA_LOG, n_sha, {"net": FC.SHA_NET, "packed": True}), ("unit", FC.HM_LOG, FC._pow2(len(ports)), {"net": FC.HM_NET}),
             ("mask", FC.MASK_LOG, 1, {"net": "mask"})]
    k_log = FC._ranges(items)[1]
    return {"rows": [[r, b] for r, _s, b, _ps in rows_spec], "compressions": sum(p.comps for p in ports), "sha_slots": n_sha,
            "k_log_rows": k_log, "k_max": FC.K_MAX, "fits": k_log <= FC.K_MAX}


def launch(res, store, vals, kind: str, step: int, a) -> dict:
    """One launch's records (module docstring) and, when they all hold, its units' rows and outputs."""
    from verity_vllm.program.kernels import fp8_moe_rows as FR
    from verity_vllm.program.registry.fp8 import weight_blocks
    m = pick(res, kind, step, a.layer)
    if m is None:
        return {"staged": False, "why": f"no {kind} Call at step {step} in layer {a.layer}"}
    spec = res.body.nodes[m["node"]].fn
    v1, v2, s = units_of(spec)
    E, Kd, N, G, KB = s["E"], s["K"], s["N"], s["G"], s["KB"]
    NB, KG = weight_blocks(N, G), Kd // G
    rec = {"name": m["name"], "spec": spec.id, "step": m["step"], "event": m["event"], "node": m["node"], "v1": v1.id, "v2": v2.id}
    t = time.time()
    args = a8.operands(res, store, vals, m)
    routed = kind == DOWN
    xq, sx = args[0].astype(np.uint8), args[1].astype(np.uint32)
    e = int(args[2][0])
    e = e - (1 << 32) if e & 0x80000000 else e
    g = int(args[3][0]) if routed else None
    w = args[-2].astype(np.uint8).reshape(E, N, Kd)
    ws = args[-1].astype(np.uint32).reshape(E, NB, KG)
    del args
    rec.update(expert=e, routed_weight=g, operands_s=round(time.time() - t, 1))
    if not 0 <= e < E:
        return {**rec, "staged": False, "why": f"routed id {e} outside [0, {E})"}

    t = time.time()
    dot = FR.TWIN_DOT_OF[s["DOT"].id]
    out, nonfin = FR.moe_expert_fp8_block_row(xq, sx, w[e], ws[e], G=G, KB=KB, swap=s["SWAP"], g=g, dot=dot)
    twin = a8.call_twin(store, m, [int(x) for x in out], a8.write_events(res))
    rec["twin"] = {**twin, "nonfinite": int(nonfin.sum()), "seconds": round(time.time() - t, 1)}
    cap = capture_words(store, m, twin.get("late"))
    rec["configs"] = {}
    for d, sw_ in CONFIGS:
        o2, _ = FR.moe_expert_fp8_block_row(xq, sx, w[e], ws[e], G=G, KB=KB, swap=sw_, g=g, dot=d)
        rec["configs"][f"{d}/swap={sw_}"] = {"agree": int((o2.astype(np.uint64) == cap).sum()), "of": int(N),
                                              "bound": (d, sw_) == (dot, s["SWAP"])}

    coords = [n for n in (int(c) for c in (a.coords_up if kind == UP else a.coords_down).split(",")) if 0 <= n < N]
    leaves_in = [[*xq.tolist(), *sx.tolist(), e & 0xFFFFFFFF, *([g] if routed else []), *w[:, n, :].reshape(-1).tolist(),
                  *ws[e, n // G].tolist()] for n in coords]
    rows_spec = K.rows_of(v1)
    rows, _ = a8.unit_rows(v1, rows_spec, leaves_in, [[0] for _ in coords])
    rec["m0"] = m0_rows(rows_spec)
    t = time.time()
    try:
        r1 = rows_eval(v1, rows_spec, rows)
    except Exception as ex:  # noqa: BLE001
        r1 = None
        rec["v1_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    t1 = round(time.time() - t, 1)
    t = time.time()
    try:
        r2 = rows_eval(v2, K.rows_of(v2), rows)
    except Exception as ex:  # noqa: BLE001
        r2 = None
        rec["v2_error"] = f"{type(ex).__name__}: {str(ex)[:300]}" + traceback.format_exc()[-600:]
    t2 = round(time.time() - t, 1)
    want = cap[coords]
    rec["units"] = {"coords": coords, "capture": [f"{int(x):04x}" for x in want],
                    "v1": None if r1 is None else [f"{int(x):04x}" for x in r1], "v2": None if r2 is None else [f"{int(x):04x}" for x in r2],
                    "v1_is_capture": None if r1 is None else bool((r1 == want).all()),
                    "v2_is_capture": None if r2 is None else bool((r2 == want).all()),
                    "v2_is_v1": None if r1 is None or r2 is None else bool((r1 == r2).all()), "seconds": {"v1": t1, "v2": t2}}
    u = rec["units"]
    ok = twin.get("agree") is True and u["v2_is_capture"] is True and u["v1_is_capture"] is not False
    rec["staged"] = bool(ok and rec["m0"]["fits"])
    if not rec["staged"]:
        rec["why"] = ("rows over M0's block" if ok else "a check disagrees")
        return rec
    n = min(a.per_kind, len(coords))
    outs = {"o0": np.stack([a8.pack([int(x)], [16], pad16=True) for x in want[:n]])}
    rec["_kv"] = {"unit": v2, "rows_spec": K.rows_of(v2), "rows": {r: v[:n] for r, v in rows.items()}, "outs": outs, "n": n}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True, help="a8_slice.py's tar and the expert share's, extracted")
    ap.add_argument("--roles", default=None, help="the Build's derive-report.json")
    ap.add_argument("--profile", default="derived_GLM53_tp8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-values", default=None, help="the staged units, for a8.py --load-values")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--steps", default="1,0")
    ap.add_argument("--kinds", default=f"{UP},{DOWN}")
    ap.add_argument("--coords-up", default="0,1,127,128,255,256,383,511", help="w13 output columns checked (gate 0..255, up 256..511)")
    ap.add_argument("--coords-down", default="0,1,127,128,3071,3072,4095,6143", help="w2 output columns checked")
    ap.add_argument("--per-kind", type=int, default=4, help="units saved per launch")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from verity_vllm.program.registry.catalog import load as load_registry
    doc: dict = {"row": a8.ROW, "rank": 0, "layer": a.layer, "registry_errors": load_registry()[1]}
    res, store, weights, vals, fold_s = a8.fold_values(Path(a.values), a.profile, Path(a.roles) if a.roles else None,
                                                     weights_cls=ShareWeights)
    print(f"A9 fold: {len(res.instances)} Calls, {len(store.entries)} snapshots, {fold_s} s", flush=True)
    doc["fold_s"] = fold_s
    staged, rows_of_kind = {}, {}
    for kind in a.kinds.split(","):
        for step in (int(x) for x in a.steps.split(",")):
            name = f"{kind}-{STEP_NAME.get(step, f's{step}')}"
            t = time.time()
            try:
                rec = launch(res, store, vals, kind, step, a)
            except Exception as ex:  # noqa: BLE001
                rec = {"staged": False, "why": f"{type(ex).__name__}: {str(ex)[:400]}", "trace": traceback.format_exc()[-1500:]}
            rec["seconds"] = round(time.time() - t, 1)
            kv = rec.pop("_kv", None)
            staged[name] = {**{x: y for x, y in rec.items()}, **({"unit": kv["unit"].id, "n": kv["n"]} if kv else {})}
            if kv:
                rows_of_kind[name] = kv
            print(f"A9 {name}: " + json.dumps(staged[name], default=str)[:3000], flush=True)
    doc["launches"] = staged
    doc["weights"] = {k: v for k, v in weights.notes.items() if ".experts." in str(v.get("served"))}
    (out / "a9.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    if a.save_values and rows_of_kind:
        d = Path(a.save_values)
        a8.save_values(d, {f: s for f, s in staged.items() if f in rows_of_kind}, rows_of_kind, weights.notes,
                       SimpleNamespace(values=a.values, steps=a.steps, layers=str(a.layer), per_kind=a.per_kind))
        (d / "kinds.json").write_text(json.dumps({"kinds": {f: {"gates": 1, "outcome": "lowers", "unit": kv["unit"].id}
                                                            for f, kv in rows_of_kind.items()}}, indent=1))
        print(f"A9 saved {sorted(rows_of_kind)} to {d}", flush=True)
    return 0 if rows_of_kind else 1


if __name__ == "__main__":
    raise SystemExit(main())
