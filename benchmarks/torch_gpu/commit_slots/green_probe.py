"""On one CUDA device: do the commit on a green context of N SMs and other work on the other SMs run at once? A step of
the commit (`slots_gpu`, the shared-row layout from `overlap.stand_in`'s tables, so no model) on `Plan(sms=N)`, and
``--gemms`` int8 GEMMs (`torch._int_mm`, M = K = N = ``--gemm-n``) on the partition's other SMs or on every SM (the
primary context), each alone and then both at once, each stream timed by its own events and the pair by wall clock. JSON
to ``--out``/green_probe.json, a line a case to stdout.

Together is uncoupled (no dependency between the streams) or coupled as the serving loop couples the forward and the
commit (`e2e_h200.commit`): a group's GEMMs stand for its forward, then the group is added (the side stream waits for
the GEMMs' stream), and the GEMMs' stream waits until the side stream has read the group `LAG` before. Two streams on
disjoint SMs that run at once take max(commit, GEMMs) together; streams that take turns take their sum. ``overlap`` is
(commit + GEMMs - together) / min(commit, GEMMs): 1 at once, 0 in turns. The partitions stay open to the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap as O

G, CM, S = O.G, O.CM, O.S


def commit_step(plan: Any, tabs: dict, groups: list, batch: int, salt: bytes, between: Any = None) -> bytes:
    """A step's commit from ``tabs``; with ``between`` (a function), it runs before each group's add on the current stream,
    which waits for the side stream's reads `LAG` groups back."""
    sc = G.StepCommitter(plan, 0, batch, salt)
    reads = []
    for g in groups:
        tensors = []
        for t in g:
            codes, word = tabs[t.op]
            tensors += [codes, word] if t.quant else [codes]
        if between is not None:
            between()
        reads.append(sc.add(tensors))
        if between is not None and len(reads) > CM.LAG:
            torch.cuda.current_stream().wait_event(reads.pop(0))
    return sc.finish().result().root


def timed(work: dict[str, tuple[Any, Any]]) -> dict[str, float]:
    """Runs each ``name: (stream, fn)`` issued in order, every stream's span by its events and all of them by wall clock."""
    torch.cuda.synchronize()
    for s, _ in work.values():
        s.synchronize()
    marks, t0 = {}, time.perf_counter()
    for name, (s, fn) in work.items():
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(s)
        with torch.cuda.stream(s):
            fn()
        b.record(s)
        marks[name] = (a, b)
    for _, b in marks.values():
        b.synchronize()
    out = {"wall_s": round(time.perf_counter() - t0, 4)}
    out |= {f"{k}_s": round(a.elapsed_time(b) / 1000, 4) for k, (a, b) in marks.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shared", type=Path, required=True, help="circuits' shared-row Layout JSON")
    ap.add_argument("--sms", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--gemms", type=int, default=0, help="GEMMs a case (0: as many as take the commit's time on every SM)")
    ap.add_argument("--run-id", default="e2e-h200-overlap")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    shape = O.SHAPES["h200"]
    layout = S.load(a.shared)
    ctx, _ = CM.fixture_secrets(a.run_id)
    salt = CM.fixture_salt(a.run_id)(0)
    tabs = O.stand_in(shape, a.batch, dev, 20261026)
    groups = CM.tables(shape)
    x = torch.randint(-128, 128, (a.gemm_n, a.gemm_n), dtype=torch.int8, device=dev)
    y = torch.randint(-128, 128, (a.gemm_n, a.gemm_n), dtype=torch.int8, device=dev)
    n_gemms = [a.gemms]
    per_group = [1]

    def gemms() -> None:
        for _ in range(n_gemms[0]):
            torch._int_mm(x, y)

    def group_gemms() -> None:
        for _ in range(per_group[0]):
            torch._int_mm(x, y)

    def overlap(r: dict, c: float, g: float) -> dict:
        r["sum_alone_s"], r["max_alone_s"] = round(c + g, 4), round(max(c, g), 4)
        r["overlap"] = round((c + g - r["wall_s"]) / min(c, g), 3)
        return r

    doc: dict[str, Any] = {"device": torch.cuda.get_device_name(dev), "batch": a.batch, "gemm_n": a.gemm_n,
                           "groups": len(groups), "lag": CM.LAG, "layout": O.describe(layout), "cases": []}
    full = G.Plan(ctx, layout, device=dev, leaf_minb=2)
    root = commit_step(full, tabs, groups, a.batch, salt)
    primary = torch.cuda.Stream(dev)
    base = timed({"commit": (full.stream, lambda: commit_step(full, tabs, groups, a.batch, salt))})
    if not n_gemms[0]:
        n_gemms[0] = 8
        one = timed({"gemm": (primary, gemms)})["gemm_s"] / 8
        n_gemms[0] = max(1, round(base["commit_s"] / one))
    per_group[0] = max(1, round(n_gemms[0] / len(groups)))
    n_gemms[0] = per_group[0] * len(groups)
    doc["gemms"], doc["gemms_a_group"] = n_gemms[0], per_group[0]
    doc["cases"].append({"case": "commit on every SM", **base})
    doc["cases"].append({"case": "GEMMs on every SM", **timed({"gemm": (primary, gemms)})})
    every = (base["commit_s"], doc["cases"][1]["gemm_s"])
    full_roots: list[bytes] = []
    r = timed({"both": (primary, lambda: full_roots.append(
        commit_step(full, tabs, groups, a.batch, salt, between=group_gemms)))})
    doc["cases"].append(overlap({"case": "coupled: commit and GEMMs on every SM", **r}, *every))
    roots_equal = {"every": all(x == root for x in full_roots)}
    for c in doc["cases"]:
        print(json.dumps(c), flush=True)
    for n in a.sms:
        plan = G.Plan(ctx, layout, device=dev, leaf_minb=2, sms=n)
        part = plan.partition
        rest = part.rest_stream()
        roots: list[bytes] = []

        def commit(plan: Any = plan, roots: list = roots) -> None:
            roots.append(commit_step(plan, tabs, groups, a.batch, salt))

        def coupled(plan: Any = plan, roots: list = roots) -> None:
            roots.append(commit_step(plan, tabs, groups, a.batch, salt, between=group_gemms))
        on, other = f"commit on {part.sms}", f"GEMMs on the other {part.rest}"
        cases = [
            (f"{on} SMs", {"commit": (plan.stream, commit)}, None),
            (other, {"gemm": (rest, gemms)}, None),
            (f"together: {on}, {other}", {"gemm": (rest, gemms), "commit": (plan.stream, commit)}, "rest"),
            (f"together: {on}, GEMMs on every SM", {"gemm": (primary, gemms), "commit": (plan.stream, commit)}, "every"),
            (f"coupled: {on}, {other}", {"both": (rest, coupled)}, "rest"),
            (f"coupled: {on}, GEMMs on every SM", {"both": (primary, coupled)}, "every"),
        ]
        alone: dict[str, float] = {}
        for name, work, gemms_on in cases:
            r = {"case": name, "sms": part.sms, "rest": part.rest, **timed(work)}
            if gemms_on is None:
                alone[next(iter(work))] = r[f"{next(iter(work))}_s"]
            else:
                overlap(r, alone["commit"], alone["gemm"] if gemms_on == "rest" else every[1])
            doc["cases"].append(r)
            print(json.dumps(r), flush=True)
        roots_equal[str(n)] = all(x == root for x in roots)
    doc["roots_equal"] = roots_equal
    (a.out / "green_probe.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"roots_equal": doc["roots_equal"]}), flush=True)
    return 0 if all(doc["roots_equal"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
