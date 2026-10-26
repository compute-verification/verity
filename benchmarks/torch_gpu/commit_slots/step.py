"""One streamed step of the commit call at a layout on a CUDA device, timed by phase, its roots checked (HASH-25).

    uv run python benchmarks/torch_gpu/commit_slots/step.py export --shared OUT.json      # the workspace environment
    python benchmarks/torch_gpu/commit_slots/step.py time LAYOUT.json --out DIR           # on the GPU (gpu.sh's step)

``export --shared`` writes FAMILIES' layout at the frozen shape (``benchmarks/e2e_h200``'s map, ``commit.layout``) with
CIR-113's shared rows, by circuits' row model: in each table a gather reads, only the gather's units stay, so the producers'
rows go and each gather's commit unit keeps its place. A slot then commits 102,448 rows in 101,718 units. The file is in
``slots._layout_json``'s form, the form circuits' export from its row model takes. ``export`` needs the workspace
environment, since the map imports ``verity``.

``time`` loads such a file, plans it, fills each distinct group's tensors with random bytes on the device, and commits steps
of ``--slots`` slots on the plan's side stream (``slots_gpu.StepCommitter``):

- one step with each phase marked (the row digests, the leaves, the multi-row units, the fold, the slot trees, the step
  tree), with each phase's compressions counted from the layout as the engine hashes them, and its G compressions/s;
- ``--reps`` unmarked steps at each ``--temps`` pass size and ``--leaf-minb`` leaf kernel (the first of each is the marked
  step's and the profile's), with the peak memory and how long the host took to enqueue the step;
- the leaf kernel alone over the widest group's first pass, at 1 and 2 blocks an SM;
- one step under torch's profiler: each kernel's launches and time, and the side stream's time with no kernel running;
- ``--samples`` random slots plus the first and the last, rebuilt by the CPU reference (``slots.commit_slot``) and checked
  against the device's slot roots, and the step root against the reference's step tree over the device's slot roots.

JSON goes to DIR/step.json, with a line on stdout as each part ends. Exits 1 if a root differs.
"""
from __future__ import annotations

import argparse
import collections
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import slots as S  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402

CTX = identity_digest_sha512("verity/commit-slots/step/context/v1", {"step": 1})
KEY = bytes(range(96, 128))
STEP = 7
#: commit_gpu's SHA-512 compressions/s on an H200, in G (sha512_rows/README.md), against which each phase's rate is read
PEAK_G = 4.79


# --- layouts -----------------------------------------------------------------------------------------------------------

def shared_layout() -> S.Layout:
    """FAMILIES' layout at the frozen shape with CIR-113's shared rows: in every table a gather (``AllGather_v1``) reads, the
    gather's units alone (circuits' row model, ``shrow_rows.py`` in art:d4fce7df)."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "e2e_h200"))
    from e2e_h200.commit import layout as families
    from e2e_h200.plan import Map
    from e2e_h200.shape import FROZEN

    fam, groups, done = families(FROZEN), [], {}
    for g, entries in zip(fam.groups, Map(FROZEN).groups(), strict=True):
        if id(g) not in done:
            keep, k = [], 0
            for _table, units in entries:
                gathered = any(u.template == "AllGather_v1" for u in units)
                for u in units:
                    if not gathered or u.template == "AllGather_v1":
                        keep += range(k, k + u.count)
                    k += u.count
            if k != len(g.units):
                raise SystemExit(f"the map's group has {k} units and the layout's {len(g.units)}")
            done[id(g)] = S.Group(g.sources, tuple(g.units[i] for i in keep))
        groups.append(done[id(g)])
    return S.Layout(fam.name.replace("/families/", "/families-shared-rows/"), tuple(groups))


def load(path: Path) -> S.Layout:
    """A layout in ``slots._layout_json``'s form. Groups with the same tensors and units become one object, as a layout's
    repeated groups (its layers) are, so the plan lays each out once."""
    doc = json.loads(path.read_text())
    seen: dict[str, S.Group] = {}
    groups = []
    for g in doc["groups"]:
        k = json.dumps(g, separators=(",", ":"))
        if k not in seen:
            seen[k] = S.Group(tuple(g["sources"]),
                              tuple(tuple(S.Row(tuple(S.Part(*p) for p in r)) for r in u) for u in g["units"]))
        groups.append(seen[k])
    return S.Layout(doc["name"], tuple(groups))


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

    - a row's digest, one a block after its midstated P(l);
    - a row's leaf, five (c two, h two, its frame leaf one), and a one-row unit's slot-tree leaf one more, in the same thread;
    - a unit of m > 1 rows, its root: two a node of its tree padded to a power of two, a node midstate a level, a pad leaf a
      padded rank; then its slot-tree leaf, one;
    - the fold, two a slot-tree node the frontier completes; the close, two a level."""
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
    out["slot_trees"] = 2 * max(1, (n - 1).bit_length())
    return out


def export(a: argparse.Namespace) -> int:
    t = time.perf_counter()
    layout = shared_layout()
    a.path.write_text(json.dumps(S._layout_json(layout), separators=(",", ":")) + "\n")
    doc = {"path": str(a.path), "s": round(time.perf_counter() - t, 1), **describe(layout),
           "compressions_a_slot": compressions(layout)}
    print(json.dumps(doc))
    return 0


# --- timing ------------------------------------------------------------------------------------------------------------

def gib(x: float) -> float:
    return round(x / 2**30, 3)


def run(a: argparse.Namespace) -> int:
    import torch
    import slots_gpu as G

    a.out.mkdir(parents=True, exist_ok=True)
    slots = a.slots
    doc: dict[str, Any] = {"slots": slots, "device": torch.cuda.get_device_name(0), "peak_g_compressions_per_s": PEAK_G}

    def save(part: str) -> None:
        (a.out / "step.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"part": part, **{k: doc[k] for k in doc if k in (part, "ok")}}), flush=True)

    t = time.perf_counter()
    layout = load(a.layout)
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

    def step(marks: list | None = None) -> tuple[S.StepCommit, dict[str, Any]]:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(plan.stream)
        if marks is not None:
            marks.append(("start", start))
        sc = G.StepCommitter(plan, STEP, slots, KEY, marks=marks)
        for g in layout.groups:
            sc.add(bufs[id(g)])
        enq = time.perf_counter() - t0
        pend = sc.finish()
        end.record(plan.stream)
        res = pend.result()
        end.synchronize()
        return res, {"temp_bytes": plan.temp_bytes, "leaf_minb": plan.leaf_minb, "gpu_s": round(start.elapsed_time(end) / 1e3, 4),
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

    doc["steps"] = []
    for temp in a.temps:
        for minb in a.leaf_minb:
            plan.temp_bytes, plan.leaf_minb = temp, minb
            rs = [step()[1] for _ in range(a.reps)]
            best = min(rs, key=lambda r: r["gpu_s"])
            doc["steps"].append({"temp_bytes": temp, "leaf_minb": minb,
                                 "passes_a_group": sorted({-(-slots // plan.per_pass(grp.shape, slots)) for grp in plan.groups}),
                                 "gpu_s": [r["gpu_s"] for r in rs], "best": best})
            save("steps")
    plan.temp_bytes, plan.leaf_minb = a.temps[0], a.leaf_minb[0]

    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _, r = step()
        kernels = []
        for e in prof.key_averages():
            us = getattr(e, "device_time_total", None)
            us = e.cuda_time_total if us is None else us
            if us > 0:
                kernels.append({"kernel": e.key[:120], "launches": e.count, "s": round(us / 1e6, 4)})
        kernels.sort(key=lambda k: -k["s"])
        busy = sum(k["s"] for k in kernels)
        doc["profile"] = {"gpu_s": r["gpu_s"], "enqueue_s": r["enqueue_s"], "kernels_s": round(busy, 4),
                          "idle_s": round(r["gpu_s"] - busy, 4), "launches": sum(k["launches"] for k in kernels),
                          "pageable_uploads": sum(k["launches"] for k in kernels if "Pageable -> Device" in k["kernel"]),
                          "kernels": kernels[:25]}
    except Exception as e:   # a profiler the node lacks costs only this part
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
    doc["ok"] = not differ and step_root
    save("reference")
    return 0 if doc["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a layout in slots._layout_json's form")
    ex.add_argument("--shared", action="store_true", required=True, help="FAMILIES' layout with CIR-113's shared rows")
    ex.add_argument("path", type=Path)
    tm = sub.add_parser("time", help="time and check a step at a layout on the GPU")
    tm.add_argument("layout", type=Path, help="a layout in slots._layout_json's form")
    tm.add_argument("--out", type=Path, required=True)
    tm.add_argument("--slots", type=int, default=4096)
    tm.add_argument("--reps", type=int, default=3)
    tm.add_argument("--temps", type=int, nargs="+", default=[1 << 30, 4 << 30], help="pass sizes (Plan.temp_bytes)")
    tm.add_argument("--leaf-minb", type=int, nargs="+", choices=(0, 1, 2), default=[0],
                    help="the leaf kernel's blocks an SM (Plan.leaf_minb; 0, the build's default)")
    tm.add_argument("--samples", type=int, default=4, help="random slots the reference rebuilds, beside the first and last")
    a = ap.parse_args(argv)
    return export(a) if a.cmd == "export" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
