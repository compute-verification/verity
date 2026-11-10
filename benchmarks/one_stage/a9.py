"""Run A9: GLM-5.3 TP8 B1's routed-expert GEMMs (91.6% of the step's computing gates) in the static pair form, on the Match
capture's values. It stages values for `a11.py --load-values` (the one-stage audit, M0 under ``--zk``, the Lean verifier) and
records what decides whether each unit can go there.

    python benchmarks/one_stage/a9.py --values DIR --roles derive-report.json --out OUT --save-values SAVED

``DIR`` is a11_slice.py's value slice with ``ckpt/`` holding ``a11_slice.ckpt_experts``' share of layer 3's routed experts (the
rank's gate/up rows and down columns, its block scales on the same grid), which `ShareWeights` composes at world 1: at TP8 each
cut is two whole 128-blocks, so that is the world-8 composition of the whole tensors.

For each of the w13 launch (`MoeExpertFp8Gemm_v1`) and the w2 launch (`MoeExpertFp8GemmW_v1`), at decode (step 1, M = 1: the
fold binds ``DOT=HopperE4m3MmaV2Dot32``, ``SWAP=False``) and prefill (step 0, M = 256: ``HopperE4m3QgmmaDot32``, ``SWAP=True``),
on layer 3's first routed pair:

- ``twin``: every output word of the pair, by `fp8_moe_rows`' numpy twin on the Call's operands, against the capture's output
  snapshot (`a11.call_twin`);
- ``configs``: the same with each of the four (DOT, SWAP) launch configs, so whether one pinned config reproduces both launches;
- ``units``: the coordinate unit (`MoeExpertFp8Coordinate(W)_v1`, the word Definition through the IR reference, and the Boolean
  static pair form `MoeExpertFp8Coordinate(W)_v2`, the expert's row picked by `boolean.gather`'s multiplexer tree inside the unit,
  bit-sliced) on ``--coords``' coordinates, each against the capture's word;
- ``m0``: the unit's rows (`a11_kinds.rows_of`) and the block M0 needs for them alone (`circuit.fits`' ranges without the unit's own),
  against ``K_MAX``.

A w13 coordinate unit reads its whole [E][K] slab column, so its rows alone are over M0's block. Its cut (``<launch>-gather``,
``<launch>-block``) is where the static pair form's own body cuts it: a `GatherE4m3x{E}_v2` unit per weight of the routed
expert's row (an e4m3 out) and the `MoeExpertFp8BlockCoordinate_v4` unit on that row (a bf16 out), each held to the expert's
weights and the capture's words as the whole unit is.

The units that agree everywhere and fit M0 are saved (`a11.save_values`' layout, one kind per launch or cut unit) with
``kinds.json``, a table naming them, for ``a11.py --load-values SAVED --kinds SAVED/kinds.json``. ``OUT/a9.json`` holds every
record.
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

import a11  # noqa: E402
import a11_kinds as K  # noqa: E402
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
#: a launch's Call, which each unit of its cut records as its own (a11.py's audit names the Call a member stands for)
CALL = ("name", "spec", "step", "event", "node", "expert", "routed_weight", "twin")


class ShareWeights(a11.Weights):
    """`a11.Weights` whose routed-expert stacks are composed at world 1 from the rank's share in ``ckpt/`` (a11_slice.ckpt_experts),
    the rest as `a11.Weights` composes them."""

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
        self.notes[name]["shard"] = f"rank {self.rank}/{self.world}: the share a11_slice.ckpt_experts cut, composed at world 1"
        return w


def pick(res, kind: str, step: int, layer: int):
    """The kind's first Call of the step in the layer (its first routed pair)."""
    cands = [m for m in res.instances if K.family(m["spec"]) == kind and m["step"] == step
             and (lm := a11.LAYER.search(m["name"])) and int(lm[1]) == layer]
    return min(cands, key=lambda m: (m["event"], m["node"])) if cands else None


def units_of(spec):
    """The Call's coordinate unit: the word `_v1` and the Boolean `_v2` at the Call's statics."""
    from verity.ml.definitions.registry import boolean_fp8_moe as B, fp8_moe_experts as X
    s = dict(spec.bindings)
    w = K.family(spec.id) == DOWN
    st = {k: s[k] for k in ("E", "K", "G", "KB", "SWAP")}
    v1 = bind(X.MoeExpertFp8CoordinateW if w else X.MoeExpertFp8Coordinate, **st, DOT=s["DOT"])
    v2 = bind(B.MoeExpertFp8CoordinateWV2 if w else B.MoeExpertFp8CoordinateV2, **st, DOT=BOOL_DOT[s["DOT"].id])
    return v1, v2, s


def capture_words(store, m: dict, late: dict | None) -> np.ndarray:
    """The Call's output words as the capture holds them: its one output view's snapshot at the Call (or the later one
    `a11.call_twin` held it to)."""
    view = m["output_views"][-1]
    ev = (late or {}).get(len(m["output_views"]) - 1)
    s = next((x for x in store.by_alloc.get(view.alloc, []) if x.event == ev and store.covers(x, view)), None) if ev is not None \
        else store.find(view, at=m["event"])
    return np.asarray(store.extract(s, view), dtype=np.uint64)


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


def check_unit(v1, v2, leaves_in: list[list[int]], want: np.ndarray, n_save: int) -> dict:
    """A unit on each instance's leaves (``v1``'s, words): the word ``v1`` through the IR reference and the Boolean ``v2``
    bit-sliced, on the same rows, each against ``want`` (the capture's words), and its rows against M0's block (`m0_rows`).
    ``_kv`` (`a11.save_values`' entry, ``n_save`` instances) when ``v2`` is the capture, ``v1`` doesn't disagree, and the rows
    fit."""
    specs = {"v1": K.rows_of(v1), "v2": K.rows_of(v2)}
    out_bits = sum(CS._leaf_bits(v2.signature()[1]))
    rows, _ = a11.unit_rows(v1, specs["v1"], leaves_in, [[0] for _ in leaves_in])
    rows_spec = specs["v2"]
    rec = {"v1": v1.id, "v2": v2.id, "m0": m0_rows(rows_spec, out_bits)}
    got, secs = {}, {}
    for tag, d in (("v1", v1), ("v2", v2)):
        t = time.time()
        try:
            got[tag] = rows_eval(d, specs[tag], rows)
        except Exception as ex:  # noqa: BLE001
            got[tag] = None
            rec[f"{tag}_error"] = f"{type(ex).__name__}: {str(ex)[:300]}" + traceback.format_exc()[-600:]
        secs[tag] = round(time.time() - t, 1)
    hexw = -(-out_bits // 4)
    is_cap = {tag: None if r is None else bool((r == want).all()) for tag, r in got.items()}
    rec["units"] = {"capture": [f"{int(x):0{hexw}x}" for x in want],
                    **{tag: None if r is None else [f"{int(x):0{hexw}x}" for x in r] for tag, r in got.items()},
                    "v1_is_capture": is_cap["v1"], "v2_is_capture": is_cap["v2"],
                    "v2_is_v1": None if any(r is None for r in got.values()) else bool((got["v1"] == got["v2"]).all()),
                    "seconds": secs}
    rec["ok"] = is_cap["v2"] is True and is_cap["v1"] is not False
    if rec["ok"] and rec["m0"]["fits"]:
        n = min(n_save, len(leaves_in))
        outs = {"o0": np.stack([a11.pack([int(x)], [out_bits], pad16=True) for x in want[:n]])}
        rec["_kv"] = {"unit": v2, "rows_spec": rows_spec, "rows": {r: v[:n] for r, v in rows.items()}, "outs": outs, "n": n}
    return rec


def cut(s: dict, xq, sx, e: int, w, ws, cap: np.ndarray, coords: list[int], n_save: int) -> dict:
    """The w13 coordinate cut where the static pair form's own body cuts it (`boolean_fp8_moe._pair_coordinate`): one
    `GatherE4m3x{E}_v2` unit per weight (the multiplexer tree over the expert id's bits, an e4m3 out) and one
    `MoeExpertFp8BlockCoordinate_v4` unit on the row they select (a bf16 out), each as `check_unit` records it."""
    from verity.ml.definitions.registry import boolean_fp8_moe as BFM, fp8_moe as FM, fp8_moe_experts as X
    E, Kd, G = s["E"], s["K"], s["G"]
    picks = [(n, k) for n in (coords[0], coords[-1]) for k in (0, 1, Kd // 2, Kd - 1)]
    gather = check_unit(X.gather_e4m3(E), BFM.gather_e4m3(E).bind(), [[e & 0xFFFFFFFF, *w[:, n, k].tolist()] for n, k in picks],
                        np.asarray([int(w[e, n, k]) for n, k in picks], dtype=np.uint64), n_save)
    st = {k: s[k] for k in ("K", "G", "KB", "SWAP")}
    block = check_unit(bind(FM.MoeExpertFp8BlockCoordinateV3, **st, DOT=s["DOT"]),
                       bind(BFM.MoeExpertFp8BlockCoordinateV4, **st, DOT=BOOL_DOT[s["DOT"].id]),
                       [[*xq.tolist(), *sx.tolist(), *w[e, n, :].tolist(), *ws[e, n // G].tolist()] for n in coords], cap[coords], n_save)
    return {"-gather": {**gather, "picks": picks, "per_coordinate": Kd}, "-block": {**block, "coords": coords, "per_coordinate": 1}}


def launch(res, store, vals, kind: str, step: int, a) -> dict:
    """One launch's records (module docstring) and, when they all hold, its units' rows and outputs; for a w13 launch whose
    coordinate unit is over M0's block, its cut's units (`cut`) under ``_cut``."""
    from verity.ml.definitions.numerics import fp8_moe_rows as FR
    from verity.ml.definitions.registry.fp8 import weight_blocks
    m = pick(res, kind, step, a.layer)
    if m is None:
        return {"staged": False, "why": f"no {kind} Call at step {step} in layer {a.layer}"}
    spec = res.body.nodes[m["node"]].fn
    v1, v2, s = units_of(spec)
    E, Kd, N, G, KB = s["E"], s["K"], s["N"], s["G"], s["KB"]
    NB, KG = weight_blocks(N, G), Kd // G
    rec = {"name": m["name"], "spec": spec.id, "step": m["step"], "event": m["event"], "node": m["node"], "v1": v1.id, "v2": v2.id}
    t = time.time()
    args = a11.operands(res, store, vals, m)
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
    twin = a11.call_twin(store, m, [int(x) for x in out], a11.write_events(res))
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
    u = check_unit(v1, v2, leaves_in, cap[coords], a.per_kind)
    kv = u.pop("_kv", None)
    rec.update(m0=u.pop("m0"), units={"coords": coords, **u.pop("units")}, **u)
    ok = twin.get("agree") is True and rec.pop("ok")
    rec["staged"] = bool(ok and kv)
    if rec["staged"]:
        rec["_kv"] = kv
    else:
        rec["why"] = "rows over M0's block" if ok else "a check disagrees"
    if kind == UP and ok and not rec["m0"]["fits"]:
        rec["_cut"] = cut(s, xq, sx, e, w, ws, cap, coords, a.per_kind)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True, help="a11_slice.py's tar and the expert share's, extracted")
    ap.add_argument("--roles", default=None, help="the Build's derive-report.json")
    ap.add_argument("--profile", default="derived_GLM53_tp8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-values", default=None, help="the staged units, for a11.py --load-values")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--steps", default="1,0")
    ap.add_argument("--kinds", default=f"{UP},{DOWN}")
    ap.add_argument("--coords-up", default="0,1,127,128,255,256,383,511", help="w13 output columns checked (gate 0..255, up 256..511)")
    ap.add_argument("--coords-down", default="0,1,127,128,3071,3072,4095,6143", help="w2 output columns checked")
    ap.add_argument("--per-kind", type=int, default=4, help="units saved per launch")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from verity.ml.definitions.registry.catalog import load as load_registry
    doc: dict = {"row": a11.ROW, "rank": 0, "layer": a.layer, "registry_errors": load_registry()[1]}
    res, store, weights, vals, fold_s = a11.fold_values(Path(a.values), a.profile, Path(a.roles) if a.roles else None,
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
            call = {k: rec[k] for k in CALL if k in rec}
            subs = {name: rec, **{name + sfx: {**c, "of": name, **call} for sfx, c in rec.pop("_cut", {}).items()}}
            for nm, r in subs.items():
                kv = r.pop("_kv", None)
                if nm != name:
                    r["staged"] = bool(kv)
                staged[nm] = {**r, **({"unit": kv["unit"].id, "n": kv["n"]} if kv else {})}
                if kv:
                    rows_of_kind[nm] = kv
                print(f"A9 {nm}: " + json.dumps(staged[nm], default=str)[:3000], flush=True)
    doc["launches"] = staged
    doc["weights"] = {k: v for k, v in weights.notes.items() if ".experts." in str(v.get("served"))}
    (out / "a9.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    if a.save_values and rows_of_kind:
        d = Path(a.save_values)
        a11.save_values(d, {f: s for f, s in staged.items() if f in rows_of_kind}, rows_of_kind, weights.notes,
                       SimpleNamespace(values=a.values, steps=a.steps, layers=str(a.layer), per_kind=a.per_kind))
        (d / "kinds.json").write_text(json.dumps({"kinds": {f: {"gates": 1, "outcome": "lowers", "unit": kv["unit"].id}
                                                            for f, kv in rows_of_kind.items()}}, indent=1))
        print(f"A9 saved {sorted(rows_of_kind)} to {d}", flush=True)
    return 0 if rows_of_kind else 1


if __name__ == "__main__":
    raise SystemExit(main())
