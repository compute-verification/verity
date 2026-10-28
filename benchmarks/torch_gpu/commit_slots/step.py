"""One streamed step of the commit call at a layout on a CUDA device, timed by phase, its roots checked.

    python benchmarks/torch_gpu/commit_slots/step.py count LAYOUT.json --out OUT.count.json          # host twin, any CPU
    python benchmarks/torch_gpu/commit_slots/step.py time LAYOUT.json [...] --out DIR                # on the GPU (gpu.sh)

``count`` commits steps of one and two slots of random outputs with the engine's host twin and reads its counters: the
compressions and ChaCha20 chunks a slot adds, what a step costs once, the compressions a row-slot (alone, and at a step of
``--slots``), and how far they are from ``compressions``' model (what a slot adds includes the step tree's growth from one
slot to two, a few compressions).

``time`` loads each file, plans it, fills each distinct group's tensors with random bytes on the device, and commits steps
of ``--slots`` slots on the plan's side stream (``slots_gpu.StepCommitter``):

- one step with each phase marked (the row digests, the leaves, the multi-row units, the fold, the slot trees, the step
  tree), with each phase's compressions counted from the layout as the engine hashes them, and its G compressions/s;
- ``--reps`` unmarked steps at each ``--temps`` pass size, ``--leaf-minb`` leaf kernel and ``--graphs`` mode (the first of
  each is the marked step's and the profile's), with the peak memory and how long the host took to enqueue the step (an
  NVTX range, ``enqueue``, for ``launches.py``); on graphs, after a step that captures them, and each step's roots held to
  the marked step's, then one more pair under another salt key;
- the leaf kernel alone over the widest group's first pass, at 1 and 2 CTAs an SM;
- one step under torch's profiler (on graphs if they were timed): each kernel's launches and time, and the side stream's
  time with no kernel running;
- ``--samples`` random slots plus the first and the last, rebuilt by the CPU reference (``slots.commit_slot``) and checked
  against the device's slot roots, and the step root against the reference's step tree over the device's slot roots.

JSON goes to DIR/step.json (with several layouts, DIR/<stem>/step.json), with a line on stdout as each part ends, and the
host twin's counts from a ``<stem>.count.json`` beside the layout. Each layout ends with a verdict line: its best step
against ``--target-s`` at 4,096 slots, with the ratio, and whether its roots equal the reference's; the run ends with one
over them all, which is also ``$RESEARCH_RUN_DIR/result.json``'s validation when research runs it. Exits 1 if a root differs.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from verity_config.settings import SHARED

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slots as S  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402

CTX = identity_digest_sha512("verity/commit-slots/step/context/v1", {"step": 1})
KEY = bytes(range(96, 128))
STEP = 7
#: commit_gpu's SHA-512 compressions/s on an H200, in G (sha512_rows/README.md), against which each phase's rate is read
PEAK_G = 4.79
#: the slots of the step ``--target-s`` is the target for
TARGET_SLOTS = 4096


# --- layouts -----------------------------------------------------------------------------------------------------------

def uniform(row_bytes: int, units: int) -> S.Layout:
    """``units`` one-row units of ``row_bytes`` each, the next bytes of one tensor: what a row of that size costs alone."""
    rows = tuple((S.Row((S.Part(0, i * row_bytes, row_bytes),)),) for i in range(units))
    return S.Layout(f"commit-slots/step/uniform-{units}x{row_bytes}b/v1", (S.Group((units * row_bytes,), rows),))


def describe(layout: S.Layout) -> dict[str, Any]:
    """Counts a slot: units, rows, bytes, and rows by size."""
    sizes = collections.Counter(r.nbytes for u in layout.units for r in u)
    shapes = collections.Counter(len(u) for u in layout.units)
    distinct = {id(g): g for g in layout.groups}
    return {"layout": layout.name, "layout_digest": S.layout_digest(layout).hex(), "units": layout.unit_count,
            "rows": layout.row_count, "bytes_a_slot": layout.slot_bytes, "groups": len(layout.groups),
            "distinct_groups": [{"units": len(g.units), "rows": sum(len(u) for u in g.units), "tensor_bytes": sum(g.sources),
                                 "times": sum(x is g for x in layout.groups)} for g in distinct.values()],
            "rows_by_bytes": {str(k): v for k, v in sorted(sizes.items())},
            "units_by_rows": {str(k): v for k, v in sorted(shapes.items())}}


def compressions(layout: S.Layout) -> dict[str, int]:
    """A slot's SHA-512 compressions by phase, as the engine hashes them (tests/test_cost.py):

    - a row's digest, one a chunk after its midstated P(l);
    - a row's leaf, five (c two, h two, its frame leaf one), and a one-row unit's slot-tree leaf one more, in the same thread;
    - a unit of m > 1 rows, its root: two a node of its tree padded to a power of two, a node midstate a level, a pad leaf a
      padded rank; then its slot-tree leaf, one;
    - the fold, two a slot-tree node the frontier completes; the close, two a level, none when the unit count is a power of
      two (no pads)."""
    out = dict.fromkeys(("rows", "leaves", "units", "fold", "slot_trees"), 0)
    per: dict[int, tuple[int, int, int]] = {}
    for g in layout.groups:
        if id(g) not in per:
            rows = leaves = units = 0
            for u in g.units:
                rows += sum(-(-(r.nbytes + 17) // 128) for r in u)
                m = len(u)
                if m == 1:
                    leaves += 6
                else:
                    d = (m - 1).bit_length()
                    leaves += 5 * m
                    units += 2 * ((1 << d) - 1) + d + ((1 << d) - m) + 1
            per[id(g)] = (rows, leaves, units)
        r, lf, un = per[id(g)]
        out["rows"] += r
        out["leaves"] += lf
        out["units"] += un
    n = layout.unit_count
    out["fold"] = 2 * (n - bin(n).count("1"))
    out["slot_trees"] = 0 if n & (n - 1) == 0 else 2 * (n - 1).bit_length()
    return out


def count(a: argparse.Namespace) -> int:
    """The layout's compressions and ChaCha20 chunks as the engine's host twin hashes them (its counters), over steps of one
    and two slots: what a slot adds, and what a step costs once (each unit's frame table an add)."""
    import slots_gpu as G

    t = time.perf_counter()
    layout = S.load(a.layout) if a.layout else uniform(a.uniform, a.units)
    plan = G.Plan(CTX, layout, "host")
    got = {}
    for slots in (1, 2):
        outputs = S.random_outputs(layout, slots, np.random.default_rng(slots))
        plan.engine.counts(reset=True)
        G.commit(plan, STEP, KEY, outputs, slots)
        got[slots] = plan.engine.counts(reset=True)
    a_slot = {k: got[2][k] - got[1][k] for k in got[1]}
    once = {k: got[1][k] - a_slot[k] for k in got[1]}
    rows, n = layout.row_count, a.slots
    model = compressions(layout)
    doc = {"layout": layout.name, "layout_digest": S.layout_digest(layout).hex(), "rows": rows, "units": layout.unit_count,
           "host_twin": {str(k): v for k, v in got.items()}, "a_slot": a_slot, "a_step_once": once,
           "compressions_a_row_slot": round(a_slot["compressions"] / rows, 3),
           "chacha_a_row_slot": round(a_slot["chacha"] / rows, 3),
           f"compressions_a_row_slot_at_{n}": round((once["compressions"] + n * a_slot["compressions"]) / (n * rows), 3),
           "model_a_slot": model, "host_twin_minus_model": a_slot["compressions"] - sum(model.values()),
           "s": round(time.perf_counter() - t, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    return 0


# --- timing ------------------------------------------------------------------------------------------------------------

def gib(x: float) -> float:
    return round(x / 2**30, 3)


def run(a: argparse.Namespace) -> int:
    import gc

    import torch

    docs = {}
    for path in a.layout:
        out = a.out if len(a.layout) == 1 else a.out / path.stem
        doc = docs[path.stem] = time_layout(a, path, out)
        if (c := path.with_suffix(".count.json")).exists():
            doc["count"] = json.loads(c.read_text())
            (out / "step.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"part": "verdict", "layout": path.stem, "verdict": verdict(path.stem, doc, a.target_s)}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    ok = all(d["ok"] for d in docs.values())
    line = ("ok: " if ok else "FAILED: ") + "; ".join(verdict(k, d, a.target_s) for k, d in docs.items())
    print(json.dumps({"part": "verdict", "verdict": line}), flush=True)
    if SHARED.RESEARCH_RUN_DIR:
        result(a, docs, ok, line)
    return 0 if ok else 1


def best_s(doc: dict[str, Any], graphs: bool | None = None) -> float:
    return min(s["best"]["gpu_s"] for s in doc["steps"] if graphs is None or s["graphs"] == graphs)


def best_enqueue_s(doc: dict[str, Any], graphs: bool) -> float:
    return min(s["best"]["enqueue_s"] for s in doc["steps"] if s["graphs"] == graphs)


def verdict(name: str, doc: dict[str, Any], target_s: float) -> str:
    s, slots, ref = best_s(doc), doc["slots"], doc["reference"]
    against = f", {s / target_s:.2f}x the <={target_s} s target" if slots == TARGET_SLOTS else ""
    modes = {st["graphs"] for st in doc["steps"]}
    host = "; ".join(f"{'on graphs' if g else 'eager'} {best_s(doc, g):.3f} s with the host enqueueing {best_enqueue_s(doc, g):.3f} s"
                     for g in sorted(modes))
    roots = "equal" if doc["ok"] else "DIFFER from"
    also = ", and every step on graphs equals the eager step's roots under two salt keys" if True in modes else ""
    return (f"{name} ({doc['layout']['rows']:,} rows a slot) at {slots:,} slots: step {s:.3f} s{against} ({host}); roots "
            f"{roots} the CPU reference on {len(ref['sampled_slots'])} sampled slots and the step tree{also}")


def result(a: argparse.Namespace, docs: dict[str, dict[str, Any]], ok: bool, line: str) -> None:
    """``$RESEARCH_RUN_DIR/result.json`` (research/result/v0.1): each layout's best step and, from its ``.count.json``, the host
    twin's compressions a row-slot."""
    run_dir = Path(SHARED.need("RESEARCH_RUN_DIR"))
    ms = []
    for k, d in docs.items():
        ms.append({"name": f"{k}.step_gpu_s", "value": best_s(d), "unit": "s"})
        for g in sorted({st["graphs"] for st in d["steps"]}):
            ms.append({"name": f"{k}.{'graphs' if g else 'eager'}.enqueue_s", "value": best_enqueue_s(d, g), "unit": "s"})
        ms.append({"name": f"{k}.rows_a_slot", "value": d["layout"]["rows"], "unit": "rows"})
        if "count" in d:
            ms.append({"name": f"{k}.compressions_a_row_slot", "value": d["count"]["compressions_a_row_slot"],
                       "unit": "compressions"})
    out = os.path.relpath(a.out, run_dir) if a.out.resolve().is_relative_to(run_dir.resolve()) else None
    doc = {"schema": "research/result/v0.1", "run_id": SHARED.RESEARCH_RUN_ID or "",
           "workload_fingerprint": {"tool": "commit_slots/step.py", "slots": a.slots,
                                    "layouts": {k: d["layout"]["layout_digest"] for k, d in docs.items()}},
           "validation": {"status": "passed" if ok else "failed", "checker": "step.py (slots.commit_slot, slots.step_tree)",
                          "detail": line, "evidence": [out] if out else []},
           "measurements": ms, "measurement_files": [], "artifacts": []}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def time_layout(a: argparse.Namespace, path: Path, out: Path) -> dict[str, Any]:
    import torch
    import slots_gpu as G

    out.mkdir(parents=True, exist_ok=True)
    slots = a.slots
    doc: dict[str, Any] = {"slots": slots, "device": torch.cuda.get_device_name(0), "peak_g_compressions_per_s": PEAK_G}

    def save(part: str) -> None:
        (out / "step.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"part": part, **{k: doc[k] for k in doc if k in (part, "ok")}}), flush=True)

    t = time.perf_counter()
    layout = S.load(path)
    doc["layout"] = {"load_s": round(time.perf_counter() - t, 2), **describe(layout)}
    comp = {k: slots * v for k, v in compressions(layout).items()}
    doc["layout"]["compressions_a_slot"] = {k: v // slots for k, v in comp.items()}
    doc["layout"]["compressions_a_row_slot"] = round(sum(comp.values()) / (slots * layout.row_count), 3)
    save("layout")

    m0 = torch.cuda.memory_allocated()
    t = time.perf_counter()
    plan = G.Plan(CTX, layout, temp_bytes=a.temps[0], leaf_minb=a.leaf_minb[0])
    torch.cuda.synchronize()
    distinct = {id(g): g for g in layout.groups}
    gen = torch.Generator(device="cuda")
    gen.manual_seed(5)
    bufs = {k: [torch.empty((slots, n), dtype=torch.uint8, device="cuda").random_(generator=gen) for n in g.sources]
            for k, g in distinct.items()}
    torch.cuda.synchronize()
    doc["plan"] = {"s": round(time.perf_counter() - t, 2), "plan_gib": gib(torch.cuda.memory_allocated() - m0), "depth": plan.depth,
                   "inputs_gib": gib(sum(x.numel() for b in bufs.values() for x in b)),
                   "passes_a_group": {str(plan.temp_bytes): sorted({-(-slots // plan.per_pass(grp.shape, slots))
                                                                     for grp in plan.groups})}}
    save("plan")

    def step(marks: list | None = None, graphs: bool = False, key: bytes = KEY) -> tuple[S.StepCommit, dict[str, Any]]:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(plan.stream)
        if marks is not None:
            marks.append(("start", start))
        torch.cuda.nvtx.range_push("enqueue")
        sc = G.StepCommitter(plan, STEP, slots, key, marks=marks, graphs=graphs)
        for g in layout.groups:
            sc.add(bufs[id(g)])
        enq = time.perf_counter() - t0
        torch.cuda.nvtx.range_pop()
        pend = sc.finish()
        end.record(plan.stream)
        res = pend.result()
        end.synchronize()
        return res, {"graphs": graphs, "temp_bytes": plan.temp_bytes, "leaf_minb": plan.leaf_minb,
                     "gpu_s": round(start.elapsed_time(end) / 1e3, 4),
                     "wall_s": round(time.perf_counter() - t0, 4), "enqueue_s": round(enq, 4),
                     "peak_gib": gib(torch.cuda.max_memory_allocated() - base),
                     "peak_total_gib": gib(torch.cuda.max_memory_allocated()),
                     "frontier_mib": round(sc.frontier.numel() * 8 / 2**20, 2), "reserved_gib": gib(torch.cuda.max_memory_reserved())}

    step()   # the first step builds and loads nothing new, but warms the allocator and the clocks
    marks: list = []
    res, marked = step(marks)
    split: dict[str, float] = {}
    phases: dict[str, int] = {}
    prev = None
    for name, ev in marks:
        if prev is not None:
            split[name] = split.get(name, 0.0) + prev.elapsed_time(ev) / 1e3
            phases[name] = phases.get(name, 0) + 1
        prev = ev
    marked["split_s"] = {k: round(v, 4) for k, v in split.items()}
    marked["phases"] = phases
    marked["g_compressions_per_s"] = {k: round(comp[k] / split[k] / 1e9, 3) for k in comp if split.get(k)}
    marked["g_compressions_per_s"]["step"] = round(sum(comp.values()) / marked["gpu_s"] / 1e9, 3)
    marked["of_peak"] = {k: round(v / PEAK_G, 3) for k, v in marked["g_compressions_per_s"].items()}
    marked["ns_a_row_slot"] = {k: round(v / (slots * layout.row_count) * 1e9, 3) for k, v in split.items()}
    doc["marked"] = marked
    save("marked")

    def leaf_probe() -> dict[str, Any]:
        grp = max(plan.groups, key=lambda q: q.shape.rows)
        g = layout.groups[plan.groups.index(grp)]
        per = plan.per_pass(grp.shape, slots)
        eng, out = plan.engine, {"rows": grp.shape.rows, "slots": per}
        with plan.side():
            x = eng.row_digests(plan.slots_of(plan.rows(grp.shape, bufs[id(g)]), 0, per))
            frames = eng.frame_table([plan.slot_did], grp.first_unit, grp.shape.units)
            run_, rest = plan.empty(per, grp.shape.units), plan.empty(grp.shape.rest, per)
            for minb in (1, 2, 1, 2):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(plan.stream)
                eng.leaves(grp.leaf_table, x, G.E.SaltKey(KEY), rest, minb, frames, grp.shape.dest, run_)
                e1.record(plan.stream)
                e1.synchronize()
                s = e0.elapsed_time(e1) / 1e3
                out[f"minb{minb}_s"] = round(s, 5)
                out[f"minb{minb}_ns_a_row_slot"] = round(s / (grp.shape.rows * per) * 1e9, 3)
        del x, run_, rest, frames
        return out

    doc["leaf_probe"] = leaf_probe()
    save("leaf_probe")

    doc["steps"], same = [], []
    for temp in a.temps:
        for minb in a.leaf_minb:
            for graphs in map(bool, a.graphs):
                plan.temp_bytes, plan.leaf_minb = temp, minb
                t = time.perf_counter()
                if graphs:
                    same.append(step(graphs=True)[0] == res)   # captures this pass size's and leaf kernel's graphs
                capture_s = round(time.perf_counter() - t, 3)
                runs = [step(graphs=graphs) for _ in range(a.reps)]
                same += [r == res for r, _ in runs if graphs]
                rs = [i for _, i in runs]
                best = min(rs, key=lambda r: r["gpu_s"])
                doc["steps"].append({"graphs": graphs, "temp_bytes": temp, "leaf_minb": minb,
                                     "passes_a_group": sorted({-(-slots // plan.per_pass(grp.shape, slots)) for grp in plan.groups}),
                                     "gpu_s": [r["gpu_s"] for r in rs], "enqueue_s": [r["enqueue_s"] for r in rs],
                                     **({"first_step_with_capture_s": capture_s} if graphs else {}), "best": best})
                save("steps")
    plan.temp_bytes, plan.leaf_minb = a.temps[0], a.leaf_minb[0]
    if True in map(bool, a.graphs):
        other = bytes(range(32))
        doc["graphs_check"] = {"steps_equal_eager": sum(same), "of": len(same),
                               "other_key_equal_eager": step(graphs=True, key=other)[0] == step(key=other)[0]}
        save("graphs_check")

    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _, r = step(graphs=bool(max(a.graphs)))
        kernels = []
        for e in prof.key_averages():
            us = getattr(e, "device_time_total", None)
            us = e.cuda_time_total if us is None else us
            if us > 0:
                kernels.append({"kernel": e.key[:120], "launches": e.count, "s": round(us / 1e6, 4)})
        kernels.sort(key=lambda k: -k["s"])
        busy = sum(k["s"] for k in kernels)
        doc["profile"] = {"graphs": r["graphs"], "gpu_s": r["gpu_s"], "enqueue_s": r["enqueue_s"], "kernels_s": round(busy, 4),
                          "idle_s": round(r["gpu_s"] - busy, 4), "launches": sum(k["launches"] for k in kernels),
                          "pageable_uploads": sum(k["launches"] for k in kernels if "Pageable -> Device" in k["kernel"]),
                          "kernels": kernels[:25]}
    except Exception as e:  # noqa: BLE001  (a profiler the node lacks costs only this part)
        doc["profile"] = {"error": repr(e)}
    save("profile")

    rng = np.random.default_rng(3)
    sampled = sorted({0, slots - 1, *rng.choice(slots, size=min(a.samples, slots), replace=False).tolist()})
    index = torch.tensor(sampled, device="cuda")
    kept = {k: [t.index_select(0, index).cpu().numpy() for t in b] for k, b in bufs.items()}
    outputs = [kept[id(g)] for g in layout.groups]
    t = time.perf_counter()
    differ = [s for i, s in enumerate(sampled)
              if S.commit_slot(CTX, layout, KEY, s, S.slot_rows(layout, outputs, i)).root != res.slot_roots[s]]
    step_root = S.step_tree(CTX, layout, STEP, res.slot_roots).root == res.root
    doc["reference"] = {"sampled_slots": sampled, "differ": differ, "step_root_from_slot_roots": step_root,
                        "s": round(time.perf_counter() - t, 1)}
    doc["host_max_rss_gib"] = gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    gc_ = doc.get("graphs_check")
    doc["ok"] = not differ and step_root and (gc_ is None or (gc_["steps_equal_eager"] == gc_["of"] and gc_["other_key_equal_eager"]))
    save("reference")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    co = sub.add_parser("count", help="count a layout's compressions with the engine's host twin")
    co.add_argument("layout", type=Path, nargs="?", help="a layout in slots._layout_json's form")
    co.add_argument("--uniform", type=int, metavar="BYTES", help="instead, --units one-row units of BYTES each")
    co.add_argument("--units", type=int, default=4096)
    co.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    co.add_argument("--slots", type=int, default=4096, help="the step whose compressions a row-slot are also given")
    tm = sub.add_parser("time", help="time and check a step at each layout on the GPU")
    tm.add_argument("layout", type=Path, nargs="+", help="layouts in slots._layout_json's form")
    tm.add_argument("--out", type=Path, required=True, help="DIR, or with several layouts the parent of DIR/<stem>")
    tm.add_argument("--slots", type=int, default=4096)
    tm.add_argument("--reps", type=int, default=3)
    tm.add_argument("--temps", type=int, nargs="+", default=[1 << 30, 4 << 30], help="pass sizes (Plan.temp_bytes)")
    tm.add_argument("--leaf-minb", type=int, nargs="+", choices=(0, 1, 2), default=[0],
                    help="the leaf kernel's CTAs an SM (Plan.leaf_minb; 0, the build's default)")
    tm.add_argument("--graphs", type=int, nargs="+", choices=(0, 1), default=[0],
                    help="1: steps whose adds are CUDA graph launches (StepCommitter(graphs=True)), their roots held to the eager "
                         "step's")
    tm.add_argument("--samples", type=int, default=4, help="random slots the reference rebuilds, beside the first and last")
    tm.add_argument("--target-s", type=float, default=0.95,
                    help="a step's target at 4,096 slots, at most 25%% of the served step (default: a quarter of run "
                         "r20261026-030611-8b64's, rounded up)")
    a = ap.parse_args(argv)
    if a.cmd == "count" and (a.layout is None) == (a.uniform is None):
        ap.error("count takes a layout or --uniform BYTES")
    return {"count": count, "time": run}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
