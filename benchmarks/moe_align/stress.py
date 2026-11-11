"""`verity_vllm.engine.moe_align`'s checks and timing on one GPU, at DeepSeek-V4-Pro's routed experts (E = 384, top-k 6)
with the moe block `fused_marlin_moe` picks for bf16 activations (`moe_block`).

``stress.py check --out DIR --tag T [--reps R]`` runs each case (rows, a routing, BM, the ids' dtype, an expert map) R
times through vLLM's op and R times through `Align`, into buffers poisoned before every call, every other call beside a
GEMM on a second stream, and writes ``DIR/check.T.json``: per case each call's output digest, whether every `Align`
output is `moe_align.reference`'s bit for bit, and whether every vLLM output has the reference's ``expert_ids`` and
``num_tokens_post_pad`` and, put in canonical order (`moe_align.canonical`), its ``sorted_token_ids``: the same pairs
for each expert. ``stress.py compare --out DIR`` reads every ``check.*.json`` into ``DIR/summary.json``: per case the
distinct outputs of each op across processes and repetitions. ``stress.py time --out DIR`` writes ``DIR/time.json``:
per row count, the median device time a call of each op by CUDA events and the host's time to issue it, for the op
alone and through a wrapper (vLLM's, which allocates the outputs, and `moe_align.wrapper`), and what `Align` adds to a
decode step of `LAYERS` calls. ``stress.py plugin --out DIR`` loads the integration's vLLM general plugins as vLLM does
(``load_general_plugins``) and runs cases through the function `fused_marlin_moe` calls, eagerly and replayed from CUDA
graphs, writing ``DIR/plugin.json``; in a venv where the integration is on ``PYTHONPATH`` but not installed, put the
directory ``stress.py site --out DIR`` writes on ``PYTHONPATH`` too: a dist-info declaring the integration's entry points.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import tomllib
import types
from pathlib import Path

import torch

from verity.definitions.numerics.marlin_moe_rows import MOE_BLOCKS, moe_block

HERE = Path(__file__).resolve().parent
#: the integration's distribution, whose entry points `site` declares
PYPROJECT = HERE.parents[1] / "adapters" / "vllm" / "pyproject.toml"
MODULE = "verity_vllm.engine.moe_align"
E, TOPK = 384, 6
#: DeepSeek-V4-Pro's layers (config.json ``num_hidden_layers``), each with routed experts: the op's calls a decode step
LAYERS = 61
DEV = torch.device("cuda")
POISON = -0x5A5A5A5B
ROUTINGS = ("uniform", "skew", "same6")
DECODE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 255, 256,
          257, 384, 460, 461, 511, 512, 513, 640, 768, 921, 922, 1000, 1023, 1024)
PREFILL = (1025, 1365, 1366, 1536, 2048, 2731, 3072, 4096, 6144, 8192, 12288, 16383, 16384)
EXTRA = (1, 16, 64, 256, 1024, 4096, 16384)
TIME_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
DECODE_STEP_ROWS = (1, 16, 64, 256)
MA: types.ModuleType


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def routing(kind: str, rows: int, seed: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """``[rows, TOPK]`` expert ids: ``uniform`` (each row's TOPK distinct experts uniformly), ``skew`` (distinct, weights
    1/rank over a shuffled ranking), ``same6`` (every row the same TOPK experts in its own slot order), ``invalid``
    (uniform with 10% of the ids no expert's: -1 or at least E; under int64 also ids whose low 32 bits are an expert and
    ids whose low 32 bits are negative)."""
    g = _gen(seed)
    if kind == "uniform":
        ids = torch.rand(rows, E, generator=g, device=DEV).topk(TOPK, dim=1).indices
    elif kind == "skew":
        w = 1.0 / torch.arange(1, E + 1, device=DEV, dtype=torch.float32)
        w = w[torch.randperm(E, generator=g, device=DEV)]
        ids = torch.multinomial(w.expand(rows, E).contiguous(), TOPK, replacement=False, generator=g)
    elif kind == "same6":
        six = torch.randperm(E, generator=g, device=DEV)[:TOPK]
        ids = six[torch.rand(rows, TOPK, generator=g, device=DEV).argsort(dim=1)]
    elif kind == "invalid":
        ids = routing("uniform", rows, seed, torch.int64)
        u = torch.rand(rows, TOPK, generator=g, device=DEV)
        ids = torch.where(u < 0.05, -1, ids)
        ids = torch.where((u >= 0.05) & (u < 0.1), E + ids % 7, ids)
        if dtype == torch.int64:
            ids = torch.where((u >= 0.1) & (u < 0.15), ids + (1 << 32), ids)
            ids = torch.where((u >= 0.15) & (u < 0.2), ids + (1 << 31), ids)
    else:
        raise ValueError(kind)
    return ids.to(dtype).contiguous()


def ep_map() -> torch.Tensor:
    """Expert parallelism over four ranks, as rank 1 sees it: global experts 96..191 are its local 0..95, the rest -1."""
    m = torch.full((E,), -1, dtype=torch.int32, device=DEV)
    m[96:192] = torch.arange(96, dtype=torch.int32, device=DEV)
    return m


class Case:
    def __init__(self, name: str, kind: str, rows: int, block: int, seed: int, dtype: torch.dtype = torch.int32,
                 mapped: bool = False) -> None:
        self.name, self.kind, self.rows, self.block, self.seed = name, kind, rows, block, seed
        self.dtype, self.mapped = dtype, mapped

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        return routing(self.kind, self.rows, self.seed, self.dtype), ep_map() if self.mapped else None


def cases() -> list[Case]:
    out = []
    for ki, kind in enumerate(ROUTINGS):
        for rows in DECODE + PREFILL:
            out.append(Case(f"{kind}-{rows}", kind, rows, moe_block(rows, TOPK, E), 1000 * ki + rows))
    for rows in EXTRA:
        for bm in MOE_BLOCKS:
            if bm != moe_block(rows, TOPK, E):
                out.append(Case(f"uniform-{rows}-bm{bm}", "uniform", rows, bm, 5000 + rows))
        bm = moe_block(rows, TOPK, E)
        out.append(Case(f"invalid-{rows}", "invalid", rows, bm, 6000 + rows))
        out.append(Case(f"invalid-{rows}-int64", "invalid", rows, bm, 7000 + rows, torch.int64))
        out.append(Case(f"uniform-{rows}-int64", "uniform", rows, bm, 8000 + rows, torch.int64))
        out.append(Case(f"uniform-{rows}-ep", "uniform", rows, bm, 9000 + rows, mapped=True))
        out.append(Case(f"same6-{rows}-ep", "same6", rows, bm, 9500 + rows, mapped=True))
    return out


def pack(ref: tuple[list[int], list[int], int]) -> torch.Tensor:
    s, b, n = ref
    return torch.tensor(s + b + [n], dtype=torch.int32)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


class Busy:
    """A GEMM on a second stream, overlapping the next op on the current one."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
        self.a = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)

    def __enter__(self):
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.a @ self.a
        return self

    def __exit__(self, *exc):
        torch.cuda.current_stream().wait_stream(self.stream)


def same_pairs(got: torch.Tensor, ref: tuple[list[int], list[int], int], L: int, block: int, pairs: int) -> bool:
    """Whether ``got`` (vLLM's three outputs, concatenated) has the reference's ``expert_ids`` and total, and each
    expert's padded range holds the reference's pairs."""
    s, b, n = got[:L].tolist(), got[L:-1].tolist(), int(got[-1])
    return b == ref[1] and n == ref[2] and MA.canonical(s, b, n, block, pairs) == ref[0]


def multi_pair_experts(ref: tuple[list[int], list[int], int], block: int, pairs: int) -> int:
    """The experts holding more than one pair: where the order within an expert is a choice."""
    s, b, n = ref
    count: dict[int, int] = {}
    for j in range(n // block):
        count[b[j]] = count.get(b[j], 0) + sum(1 for p in s[j * block:(j + 1) * block] if p < pairs)
    return sum(v > 1 for v in count.values())


def check(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = torch.ops._moe_C.moe_align_block_size
    align = MA.Align()
    busy = Busy()
    record = {"tag": args.tag, "pid": os.getpid(), "host": platform.node(), "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "reps": args.reps, "cases": []}
    for case in cases():
        ids, emap = case.inputs()
        P, BM = ids.numel(), case.block
        L = MA.sorted_len(P, E, BM)
        NB = MA.cdiv(L, BM)
        ref = MA.reference(ids.flatten().tolist(), E, BM, None if emap is None else emap.tolist())
        ref_d = digest(pack(ref))
        s, b, n = (torch.empty(k, dtype=torch.int32, device=DEV) for k in (L, NB, 1))
        digests: dict[str, list[str]] = {"vllm": [], "canonical": []}
        distinct: dict[str, torch.Tensor] = {}
        for rep in range(args.reps):
            for which, op in (("vllm", real), ("canonical", align)):
                for t in (s, b, n):
                    t.fill_(POISON)
                if rep % 2:
                    with busy:
                        op(ids, E, BM, s, b, n, emap)
                else:
                    op(ids, E, BM, s, b, n, emap)
                got = torch.cat([s, b, n]).cpu()
                d = digest(got)
                digests[which].append(d)
                if which == "vllm":
                    distinct.setdefault(d, got)
        row = {"case": case.name, "routing": case.kind, "rows": case.rows, "pairs": P, "block": BM,
               "dtype": str(case.dtype).removeprefix("torch."), "expert_map": case.mapped, "reference": ref_d,
               "multi_pair_experts": multi_pair_experts(ref, BM, P),
               "canonical_is_reference": all(d == ref_d for d in digests["canonical"]),
               "vllm_same_pairs": all(same_pairs(g, ref, L, BM, P) for g in distinct.values()),
               "vllm_in_pair_order": sum(d == ref_d for d in digests["vllm"]),
               "vllm": digests["vllm"], "canonical": digests["canonical"]}
        record["cases"].append(row)
        print(row["case"], "pairs", P, "bm", BM, "canonical_distinct", len(set(row["canonical"])), "ref",
              row["canonical_is_reference"], "vllm_distinct", len(distinct), "vllm_same_pairs", row["vllm_same_pairs"],
              "vllm_in_pair_order", row["vllm_in_pair_order"], flush=True)
        del ids, emap, s, b, n, distinct
    record["align"] = align.record()
    (out / f"check.{args.tag}.json").write_text(json.dumps(record, indent=1) + "\n")


def compare(args) -> None:
    d = Path(args.out)
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("check.*.json"))]
    by: dict[str, dict] = {}
    for run in runs:
        for row in run["cases"]:
            s = by.setdefault(row["case"], {k: row[k] for k in ("case", "routing", "rows", "pairs", "block", "dtype",
                                                                 "expert_map", "multi_pair_experts")}
                              | {"vllm": [], "canonical": [], "references": set(), "canonical_is_reference": True,
                                 "vllm_same_pairs": True, "vllm_in_pair_order": 0})
            s["vllm"] += row["vllm"]
            s["canonical"] += row["canonical"]
            s["references"].add(row["reference"])
            s["canonical_is_reference"] &= row["canonical_is_reference"]
            s["vllm_same_pairs"] &= row["vllm_same_pairs"]
            s["vllm_in_pair_order"] += row["vllm_in_pair_order"]
    table = []
    for s in by.values():
        s["outputs"] = len(s["canonical"])
        s["vllm_distinct"] = len(set(s.pop("vllm")))
        s["canonical_distinct"] = len(set(s.pop("canonical")))
        s["one_reference"] = len(s.pop("references")) == 1
        table.append(s)
    verdict = {
        "processes": len(runs), "cases": len(table),
        "outputs_per_op": sum(s["outputs"] for s in table),
        "canonical_identical": all(s["canonical_distinct"] == 1 and s["one_reference"] for s in table),
        "canonical_is_reference": all(s["canonical_is_reference"] for s in table),
        "vllm_same_pairs_expert_ids_and_total": all(s["vllm_same_pairs"] for s in table),
        "vllm_varies_in": sum(s["vllm_distinct"] > 1 for s in table),
        "vllm_outputs_not_in_pair_order": sum(s["outputs"] - s["vllm_in_pair_order"] for s in table),
        "vllm_max_distinct": max((s["vllm_distinct"] for s in table), default=0),
    }
    verdict["ok"] = (verdict["canonical_identical"] and verdict["canonical_is_reference"]
                     and verdict["vllm_same_pairs_expert_ids_and_total"] and len(runs) > 0)
    (d / "summary.json").write_text(json.dumps({"verdict": verdict, "cases": table}, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    for s in table:
        print(f"{s['case']:24s} pairs={s['pairs']:6d} bm={s['block']:2d} multi={s['multi_pair_experts']:3d}"
              f" vllm_distinct={s['vllm_distinct']:3d}/{s['outputs']} in_order={s['vllm_in_pair_order']:3d}"
              f" canonical_distinct={s['canonical_distinct']} ref={s['canonical_is_reference']}"
              f" same_pairs={s['vllm_same_pairs']}")
    if not verdict["ok"]:
        raise SystemExit(1)


def device_us(fn, iters: int, trials: int = 7) -> float:
    """The device's time a call, the median of ``trials`` batches of ``iters`` calls queued behind a sleep (so the
    host's launch time stays off the device's timeline)."""
    for _ in range(3):
        fn()
    t = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(200_000_000)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(t)[len(t) // 2]


def host_us(fn, iters: int) -> float:
    """The host's time to issue a call, queued behind a sleep so no call waits on the device."""
    torch.cuda.synchronize()
    torch.cuda._sleep(200_000_000)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e6 / iters


def variant(spec: str):
    """``SMALL_MAX:CHUNK:TILE:LONG_TILE:COLS[:slow]`` (``slow``: Triton's own launch on every call), or ``default``."""
    if spec == "default":
        return MA.Align()
    small, chunk, tile, long_tile, cols, *slow = spec.split(":")
    return MA.Align(small_max=int(small), chunk=int(chunk), tile=int(tile), long_tile=int(long_tile), cols=int(cols),
                    fast_launch=not slow)


def timed(fn, iters: int) -> dict:
    try:
        return {"device_us": device_us(fn, iters), "host_us": host_us(fn, iters)}
    except Exception as e:  # noqa: BLE001 - a variant that fails to compile or launch is recorded, not fatal
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def time_(args) -> None:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size as wrapper

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = getattr(torch.ops._moe_C, MA.OP)
    arms = {spec: variant(spec) for spec in dict.fromkeys(["default", *args.variants.split(",")])}
    rows_out = []
    for ki, kind in enumerate(("uniform", "same6")):
        for rows in TIME_ROWS:
            ids = routing(kind, rows, 11000 + 100 * ki + rows)
            BM = moe_block(rows, TOPK, E)
            P = ids.numel()
            L = MA.sorted_len(P, E, BM)
            s, b, n = (torch.empty(k, dtype=torch.int32, device=DEV) for k in (L, MA.cdiv(L, BM), 1))
            iters = 200 if P <= 8192 else 50
            row = {"routing": kind, "rows": rows, "pairs": P, "block": BM, "iters": iters, "op": {}, "wrapper": {}}
            row["op"]["vllm"] = timed(lambda: real(ids, E, BM, s, b, n, None), iters)
            for spec, al in arms.items():
                row["op"][spec] = timed(lambda al=al: al(ids, E, BM, s, b, n, None), iters)
            row["op"]["vllm_again"] = timed(lambda: real(ids, E, BM, s, b, n, None), iters)
            ours = MA.wrapper(arms["default"])
            row["wrapper"]["vllm"] = timed(lambda: wrapper(ids, BM, E, None, ignore_invalid_experts=True), iters)
            row["wrapper"]["default"] = timed(lambda: ours(ids, BM, E, None, ignore_invalid_experts=True), iters)
            ok = {k: v for k, v in row["op"].items() if k not in ("vllm", "vllm_again") and "device_us" in v}
            row["best"] = min(ok, key=lambda k: ok[k]["device_us"]) if ok else None
            v = min(row["op"]["vllm"]["device_us"], row["op"]["vllm_again"]["device_us"])
            row["vllm_device_us"] = v
            row["added_device_us"] = row["op"]["default"]["device_us"] - v if "device_us" in row["op"]["default"] else None
            rows_out.append(row)
            print(json.dumps(row), flush=True)
    step = {}
    for r in rows_out:
        if r["routing"] == "uniform" and r["rows"] in DECODE_STEP_ROWS and r["added_device_us"] is not None:
            op, w = r["op"], r["wrapper"]
            step[r["rows"]] = {
                "block": r["block"], "layers": LAYERS,
                "vllm_device_us_call": r["vllm_device_us"], "align_device_us_call": op["default"]["device_us"],
                "added_device_us_step": LAYERS * r["added_device_us"],
                "added_host_us_step": LAYERS * (op["default"]["host_us"] - op["vllm"]["host_us"]),
                "added_wrapper_host_us_step": LAYERS * (w["default"]["host_us"] - w["vllm"]["host_us"])
                if "host_us" in w.get("default", {}) and "host_us" in w["vllm"] else None,
            }
    rec = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
           "triton": importlib.import_module("triton").__version__, "variants": list(arms), "rows": rows_out,
           "decode_step": step, "align": {spec: al.record() for spec, al in arms.items()}}
    (out / "time.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(step, indent=1))


def plugin(args) -> None:
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    align = MA.installed()
    call = marlin_moe.moe_align_block_size
    rows_out, graphs = [], []
    for ki, kind in enumerate(("uniform", "same6", "skew")):
        for rows in EXTRA + (2731,):
            ids = routing(kind, rows, 12000 + 100 * ki + rows)
            BM = moe_block(rows, TOPK, E)
            ref = pack(MA.reference(ids.flatten().tolist(), E, BM))
            got = [torch.cat(call(ids, BM, E, None, ignore_invalid_experts=True)).cpu() for _ in range(args.reps)]
            rows_out.append({"routing": kind, "rows": rows, "block": BM, "ref": all(torch.equal(g, ref) for g in got)})
            print(json.dumps(rows_out[-1]), flush=True)
    for rows in (64, 2048):
        BM = moe_block(rows, TOPK, E)
        static = routing("uniform", rows, 13000 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                call(static, BM, E, None, ignore_invalid_experts=True)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = call(static, BM, E, None, ignore_invalid_experts=True)
        ok = []
        for k, kind in enumerate(ROUTINGS):
            new = routing(kind, rows, 13500 + 10 * rows + k)
            static.copy_(new)
            g.replay()
            torch.cuda.synchronize()
            ok.append(torch.equal(torch.cat(outs).cpu(), pack(MA.reference(new.flatten().tolist(), E, BM))))
        graphs.append({"rows": rows, "block": BM, "replays_ref": ok})
        print(json.dumps(graphs[-1]), flush=True)
    rec = {"module": MA.__file__, "loaded_by_plugin": args.loaded_by_plugin, "wrapped": align is not None,
           "marlin_calls": f"{call.__module__}.{call.__qualname__}", "align": align.record() if align else None,
           "rows": rows_out, "graphs": graphs}
    rec["ok"] = (align is not None and args.loaded_by_plugin and call.__module__ == MA.__name__
                 and rec["align"]["calls"]["one"] > 0 and rec["align"]["calls"]["chunked"] > 0
                 and all(r["ref"] for r in rows_out) and all(all(g["replays_ref"]) for g in graphs))
    print(json.dumps({k: rec[k] for k in ("ok", "module", "loaded_by_plugin", "wrapped", "marlin_calls", "align")}), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "plugin.json").write_text(json.dumps(rec, indent=1) + "\n")
    if not rec["ok"]:
        raise SystemExit(1)


def site(args) -> None:
    """A site directory whose dist-info declares the integration's entry points (its pyproject's), so a venv with the
    integration on ``PYTHONPATH`` but not installed loads its vLLM plugins."""
    eps = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]
    info = Path(args.out) / "verity_vllm_plugins-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-vllm-plugins\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("".join(f"[{group}]\n" + "".join(f"{k} = {v}\n" for k, v in names.items())
                                                   for group, names in eps.items()))
    print(info)


def main() -> None:
    global MA
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("check", "compare", "time", "plugin", "site"))
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="0")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--variants", default="default,2048:512:32:64:32:slow,1024:512:32:64:32,4096:512:32:64:32,"
                                         "2048:1024:32:64:32,2048:256:32:64:32,2048:512:64:64:32,"
                                         "2048:512:32:128:32,2048:512:32:64:16",
                   help="stress.py time: `Align`'s parameters (`variant`), by comma")
    args = p.parse_args()
    if args.mode == "site":
        return site(args)
    if args.mode == "plugin":
        # the module must be imported by vLLM's plugin loader, not by this script
        from vllm.plugins import load_general_plugins

        before = MODULE in sys.modules
        load_general_plugins()
        args.loaded_by_plugin = not before and MODULE in sys.modules
    else:
        importlib.import_module("vllm._custom_ops")
    from verity_vllm.engine import moe_align as MA
    {"check": check, "compare": compare, "time": time_, "plugin": plugin}[args.mode](args)


if __name__ == "__main__":
    main()
