"""The served Commit's hashing alone on one GPU: one step of T tokens' rows-sha512 windows through the collector's own extension
(`native_ext.ext()`: native_collect.cpp's StepHasher over native_rows.cu), in the calls and order `NativeCollectCommitter.
_commit_step_windowed` makes for a plan-hit step: per window its leaves (`rows_leaves`), held and folded into the windows' top nodes
(`rows_tree_window` over many windows at once; a step that keeps levels folds at ROWS_FOLD_BYTES, its levels copied to pinned host on
a copy stream), the pad windows up to the tree's width, then the tree over the windows' top nodes (`rows_tree`).  FORM=window is the
worker before the held folds: each window's subtree on its own, its levels copied on the hashing stream and one event sync a window.
Every window is ready at once, so this is the hashing's own rate, not what serving lets it reach.  Each step's root is checked against
the first step's, and a small step's levels and root against `step_rows.levels`, in both forms.  Measurement only; JSON to
$OUT/hash_prof.<LABEL>.json.

    python benchmarks/served_commit/hash_prof.py    (TOKENS: T values, default 8192 16384 32768; BYTES_PER_TOKEN, default 1.73e6;
                                                     WINDOW_MB, default 256; REPS, default 3; LEVELS: 1 keeps levels (retain host or
                                                     sample), 0 keeps none (retain none); FORM: fold (default) or window; FOLD_MB:
                                                     a step that keeps levels folds at this many MiB of leaves instead; CHECK=0 skips
                                                     the small step's check; LABEL; the kernels' own knobs VERITY_ROWS_FORM and
                                                     VERITY_ROWS_PHASES pass through)
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from verity.ml.commitments.merkle import merkle_depth
from verity_vllm.commit import step_rows as R
from verity_vllm.commit.committer.native_collect import ROWS_FOLD_BYTES, ROWS_HOLD_BYTES
from verity_vllm.commit.committer.native_ext import ext

C = R.ROW_BYTES
PD, CTX, GEO = b"p" * 32, b"c" * 32, b"g" * 32
ROW_COMPRESSIONS = (C + 17 + 127) // 128 + 3 + 2 + 2 + 2   # the row after P's midstate, salt, c, h, frame leaf


def nvtx(name: str):
    return torch.cuda.nvtx.range(name)


class Step:
    """One step of n_total rows over W-byte windows, its calls as the windowed worker makes them."""

    def __init__(self, X: Any, H: Any, n_total: int, W: int, keep_levels: bool, ks: bytes, tn: torch.Tensor, form: str = "fold",
                 fold_bytes: int = 0) -> None:
        self.X, self.H, self.W, self.keep, self.tn, self.form = X, H, W, keep_levels, tn, form
        self.n_total, self.j = n_total, (W // C).bit_length() - 1
        self.width = 1 << merkle_depth(n_total)
        self.sizes = [self.width >> L for L in range(min(self.j, self.width.bit_length() - 1) + 1)]
        self.n_levels = len(self.sizes)
        self.bound = (fold_bytes or min(ROWS_HOLD_BYTES // 2, ROWS_FOLD_BYTES)) if keep_levels else ROWS_HOLD_BYTES
        dom = R.domain(PD, CTX, GEO, n_total)
        self.dom = dom
        self.rs = X.RowStep(dom.domain_id, ks, R.row_prefix(8 * C), R.INST.salt_prefix, R.INST.leaf_prefix(R.KEY), R.SCHEMA, C)
        self.pinned = torch.empty(sum(self.sizes) * 64, dtype=torch.uint8).pin_memory() if keep_levels else None
        self.views: list[torch.Tensor] = []
        if keep_levels:
            off = 0
            for nL in self.sizes:
                self.views.append(self.pinned[off:off + nL * 64].view(nL, 64))
                off += nL * 64
        self.upper = torch.empty((2 * max(1, self.width >> (self.n_levels - 1)) - 1) * 64, dtype=torch.uint8).pin_memory() \
            if keep_levels else torch.empty(0, dtype=torch.uint8)
        self.copy = torch.cuda.Stream() if keep_levels and form == "fold" else None
        self.folds = 0

    def compressions(self) -> int:
        return self.n_total * ROW_COMPRESSIONS + 2 * (self.width - self.n_total) + 2 * (self.width - 1)

    def run(self, slots: list[torch.Tensor], scratch: torch.Tensor, side: torch.cuda.Stream, ev: dict[str, list] | None) -> tuple[bytes, dict]:
        H, j, sp, views, copy = self.H, self.j, side.cuda_stream, self.views, self.copy
        tops: list[torch.Tensor] = []
        held: list[torch.Tensor] = []
        copied: list[torch.cuda.Event] = []
        S = self.n_total * C
        n_data = -(-self.n_total // (1 << j))
        t_host = {"leaves_call": 0.0, "tree_call": 0.0}
        self.folds = 0

        def mark(name: str) -> None:
            if ev is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record(side)
                ev.setdefault(name, []).append(e)

        def fold() -> None:
            first = sum(int(t.shape[0]) for t in tops) << (self.n_levels - 1)
            if copied:
                copied.pop().synchronize()
            lv = torch.cat(held) if len(held) > 1 else held[0]
            held.clear()
            n = int(lv.shape[0])
            dsts = [views[L][first >> L:(first >> L) + (n >> L)] for L in range(self.n_levels)] if copy is not None else []
            mark("fold0")
            tops.append(H.rows_tree_window(self.rs, lv, self.n_levels, first, sp, dsts, copy.cuda_stream if copy is not None else 0))
            mark("fold1")
            if copy is not None:
                e = torch.cuda.Event()
                e.record(copy)
                copied.append(e)
            self.folds += 1

        def absorb(k: int, lv: torch.Tensor) -> None:
            n_k = int(lv.shape[0])
            t0 = time.perf_counter()
            if self.keep and self.form == "window":
                dsts = [views[L][k << (j - L):(k << (j - L)) + (n_k >> L)] for L in range(self.n_levels)]
                mark("tree0")
                tops.append(H.rows_tree_window(self.rs, lv, self.n_levels, k << j, sp, dsts))
                mark("tree1")
            else:
                if held and sum(int(t.numel()) for t in held) + int(lv.numel()) > self.bound:
                    fold()
                held.append(lv)
            t_host["tree_call"] += time.perf_counter() - t0

        with torch.inference_mode(), torch.cuda.stream(side):
            for k in range(n_data):
                nbytes = min(self.W, S - k * self.W)
                first = k << j
                with nvtx(f"window {k}"):
                    t0 = time.perf_counter()
                    mark("leaf0")
                    lv = H.rows_leaves(self.rs, slots[k % len(slots)], nbytes, first, min(1 << j, self.width - first), self.n_total,
                                       self.tn, sp)[0]
                    mark("leaf1")
                    t_host["leaves_call"] += time.perf_counter() - t0
                    absorb(k, lv)
                del lv
            for kk in range(n_data, max(1, self.width >> j)):
                with nvtx(f"pad window {kk}"):
                    first = kk << j
                    mark("pad0")
                    lv = H.rows_leaves(self.rs, scratch, 0, first, min(1 << j, self.width - first), self.n_total, self.tn, sp)[0]
                    mark("pad1")
                    absorb(kk, lv)
                del lv
            if held:
                fold()
            if copy is not None:
                side.wait_stream(copy)
            t_cat = torch.cat(tops) if len(tops) > 1 else tops[0]
            mark("root0")
            with nvtx("root"):
                _upper, root, _tree_ms = H.rows_tree(self.rs, t_cat, self.n_levels - 1, sp, self.keep, self.upper)
            mark("root1")
        return bytes(root), t_host


def classes(ev: dict[str, list]) -> dict[str, float]:
    """GPU ms by class from paired events (each pair brackets one class's launches on the side stream)."""
    out: dict[str, float] = {}
    for a, b, name in (("leaf0", "leaf1", "leaves"), ("tree0", "tree1", "window_tree"), ("pad0", "pad1", "pad_leaves"),
                       ("fold0", "fold1", "fold"), ("root0", "root1", "root_tree")):
        out[name] = round(sum(x.elapsed_time(y) for x, y in zip(ev.get(a, []), ev.get(b, []))), 3)
    return out


def check_small(X: Any, H: Any, side: torch.cuda.Stream, tn: torch.Tensor) -> dict[str, bool]:
    """A step of 300 rows over 64-row windows (5 data windows, the last partial, and pad windows to 512), in each form, the folded
    one two windows a fold: its levels below the windows' tops and its root are step_rows'."""
    n, W = 300, 64 * C
    ks = os.urandom(R.KEY_BYTES)
    data = os.urandom(n * C)
    want = R.levels(R.domain(PD, CTX, GEO, n), [data[i * C:(i + 1) * C] for i in range(n)], ks, 8 * C)
    slot = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    slots = [slot[k * W:(k + 1) * W] for k in range(-(-n * C // W))]
    out = {}
    for form in ("window", "fold"):
        st = Step(X, H, n, W, True, ks, tn, form, fold_bytes=2 * 64 * 64)
        root, _ = st.run(slots, torch.zeros(W, dtype=torch.uint8, device="cuda"), side, None)
        levels_ok = all(v.numpy().tobytes() == b"".join(want[L]) for L, v in enumerate(st.views))
        out[form] = root == want[-1][0] and levels_ok and (form == "window" or st.folds > 1)
    return out


def main() -> None:
    out_dir = Path(os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    label = os.environ.get("LABEL", "served")
    tokens = [int(x) for x in os.environ.get("TOKENS", "8192 16384 32768").split()]
    bpt = float(os.environ.get("BYTES_PER_TOKEN", "1.73e6"))
    W = int(os.environ.get("WINDOW_MB", "256")) << 20
    reps = int(os.environ.get("REPS", "3"))
    keep = os.environ.get("LEVELS", "1") == "1"
    form = os.environ.get("FORM", "fold")
    fold_bytes = int(os.environ.get("FOLD_MB", "0")) << 20
    X = ext()
    H = X.StepHasher()
    side = torch.cuda.Stream()
    tn = torch.frombuffer(bytearray(R.t_nibbles()), dtype=torch.uint8).cuda()
    prop = torch.cuda.get_device_properties(0)
    res: dict[str, Any] = {"label": label, "device": prop.name, "sms": prop.multi_processor_count, "window_bytes": W, "keep_levels": keep,
                           "form": form, "fold_bytes": fold_bytes or None, "bytes_per_token": bpt, "row_compressions": ROW_COMPRESSIONS,
                           "env": {k: os.environ.get(k) for k in ("VERITY_ROWS_FORM", "VERITY_ROWS_PHASES")}, "steps": {}}
    checks = check_small(X, H, side, tn) if os.environ.get("CHECK", "1") == "1" else None
    res["matches_step_rows"] = all(checks.values()) if checks else None
    res["matches_step_rows_by_form"] = checks
    print(label, "matches_step_rows", checks, flush=True)
    if res["matches_step_rows"] is False:
        (out_dir / f"hash_prof.{label}.json").write_text(json.dumps(res, indent=1))
        raise SystemExit(1)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261028)
    slots = [torch.randint(0, 256, (W,), dtype=torch.uint8, device="cuda", generator=gen) for _ in range(8)]
    scratch = torch.zeros(W, dtype=torch.uint8, device="cuda")
    ks = bytes(range(64))
    for T in tokens:
        n_total = int(T * bpt) // C
        st = Step(X, H, n_total, W, keep, ks, tn, form, fold_bytes)
        roots, walls, cls, hosts = [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        for r in range(reps + 1):
            ev: dict[str, list] = {}
            torch.cuda.synchronize()
            with nvtx(f"{label} T {T} rep {r}"):
                t0 = time.perf_counter()
                root, t_host = st.run(slots, scratch, side, ev)
                side.synchronize()
                wall = time.perf_counter() - t0
            roots.append(root.hex())
            if r:
                walls.append(wall * 1e3)
                cls.append(classes(ev))
                hosts.append({k: round(v * 1e3, 3) for k, v in t_host.items()})
        comps = st.compressions()
        wall_ms = statistics.median(walls)
        gpu = {k: round(statistics.median(c[k] for c in cls), 3) for k in cls[0]}
        entry = {"T": T, "rows": n_total, "bytes": n_total * C, "width": st.width, "windows": -(-n_total * C // W),
                 "pad_windows": max(1, st.width >> st.j) - -(-n_total // (1 << st.j)), "compressions": comps, "folds": st.folds,
                 "fold_bound_bytes": st.bound if form == "fold" or not keep else None,
                 "peak_device_bytes": torch.cuda.max_memory_allocated() - base,
                 "wall_ms": round(wall_ms, 3), "walls_ms": [round(w, 3) for w in walls], "gpu_ms": gpu,
                 "gpu_classes_sum_ms": round(sum(gpu.values()), 3), "host_call_ms": hosts[len(hosts) // 2],
                 "gb_per_s": round(n_total * C / wall_ms * 1e-6, 2), "gcomp_per_s": round(comps / wall_ms * 1e-6, 4),
                 "leaves_gcomp_per_s": round(n_total * ROW_COMPRESSIONS / gpu["leaves"] * 1e-6, 4) if gpu["leaves"] else None,
                 "roots_equal": len(set(roots)) == 1, "root": roots[0]}
        res["steps"][str(T)] = entry
        print(label, json.dumps(entry), flush=True)
        del st
        torch.cuda.empty_cache()
    (out_dir / f"hash_prof.{label}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
