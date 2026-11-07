"""The DSA top-k stress and timing on one GPU, at GLM-5.3's indexer shapes (k = 2048).

``stress.py check --out DIR --tag T [--reps R]`` runs each case (an op, its shape, a distribution of logits) R times
unpatched (vLLM's op) and R times canonical (the op under `dsa_topk.install`), every other repetition beside a GEMM on a
second stream, and writes ``DIR/check.T.json``: each repetition's output digest; whether the canonical output equals
`dsa_topk.reference_rows`; whether each unpatched output selects the reference's set up to keys tied at the k-th (the
keys above it all, and the rest from the tied keys, each once); whether the rows of at most k keys are byte for byte the
unpatched ones; and the pass's counters. ``stress.py compare DIR`` reads every ``check.*.json`` and writes
``DIR/summary.json``: per case the distinct digests unpatched and canonical across all processes and repetitions.
``stress.py time --out DIR`` writes ``DIR/time.json``: the median time of the op, the pass alone and both, by CUDA
events, at decode and prefill shapes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsa_topk

K = 2048
WS = 1024 * 1024  # vLLM's RADIX_TOPK_WORKSPACE_SIZE
DEV = torch.device("cuda")


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def logits(dist: str, rows: int, cols: int, seed: int) -> torch.Tensor:
    """The case's logits: ``randn`` (no ties), ``bf16`` (randn rounded to bf16: ties in the tail), ``int`` (32 values:
    a tie at the k-th of thousands of keys), ``relu`` (relu(randn - 2), the indexer's ReLU: under k keys above zero, so
    the k-th is a zero tied with most of the row), ``signed`` (relu's, with half its values negated, so its zeros are
    +0 and -0, and +-inf mixed in), ``edge`` (randn with NaN, +-0 and +-inf mixed in)."""
    g = _gen(seed)
    x = torch.randn(rows, cols, generator=g, device=DEV)
    if dist == "bf16":
        x = x.to(torch.bfloat16).float()
    elif dist == "int":
        x = torch.randint(0, 32, (rows, cols), generator=g, device=DEV).float()
    elif dist == "relu":
        x = torch.relu(x - 2.0)
    elif dist == "signed":
        u = torch.rand(rows, cols, generator=g, device=DEV)
        x = torch.relu(x - 2.0)
        x = torch.where(u < 0.5, -x, x)
        x = torch.where((u >= 0.9) & (u < 0.901), float("inf"), x)
        x = torch.where((u >= 0.901) & (u < 0.902), float("-inf"), x)
    elif dist == "edge":
        u = torch.rand(rows, cols, generator=g, device=DEV)
        x = torch.where(u < 0.01, float("nan"), x)
        x = torch.where((u >= 0.01) & (u < 0.03), 0.0, x)
        x = torch.where((u >= 0.03) & (u < 0.05), -0.0, x)
        x = torch.where((u >= 0.05) & (u < 0.051), float("inf"), x)
        x = torch.where((u >= 0.051) & (u < 0.052), float("-inf"), x)
    elif dist != "randn":
        raise ValueError(dist)
    return x


def lens(rows: int, lo: int, hi: int, seed: int) -> torch.Tensor:
    return torch.randint(lo, hi + 1, (rows,), generator=_gen(seed), device=DEV, dtype=torch.int32)


class Case:
    """One call of one op: ``call(ns, out)`` runs it from namespace ``ns``; ``lo``/``n`` are each row's keys."""

    def __init__(self, name: str, op: str, x: torch.Tensor, lo: torch.Tensor, n: torch.Tensor, call) -> None:
        self.name, self.op, self.x, self.lo, self.n, self.call = name, op, x, lo, n, call

    def run(self, ns) -> torch.Tensor:
        out = torch.full((self.x.shape[0], K), -7, dtype=torch.int32, device=DEV)
        self.call(ns, out)
        return out


def prefill(name: str, x: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor) -> Case:
    def call(ns, out):
        ns.top_k_per_row_prefill(x, ks, ke, out, x.shape[0], x.stride(0), x.stride(1), K)
    return Case(name, "top_k_per_row_prefill", x, ks, ke - ks, call)


def causal(rows: int, prefix: int) -> tuple[torch.Tensor, torch.Tensor]:
    ke = prefix + 1 + torch.arange(rows, device=DEV, dtype=torch.int32)
    return torch.zeros_like(ke), ke


def length_op(name: str, op: str, x: torch.Tensor, ln: torch.Tensor) -> Case:
    ws = torch.zeros(WS, dtype=torch.uint8, device=DEV)
    msl = int(ln.max()) if op == "cooperative_topk" else x.shape[1]

    def call(ns, out):
        getattr(ns, op)(x, ln, out, ws, K, msl)
    n = ln.clamp(0, min(x.shape[1], msl) if op == "persistent_topk" else x.shape[1])
    return Case(name, op, x, torch.zeros_like(ln), n, call)


def decode(name: str, x: torch.Tensor, seq_lens: torch.Tensor, next_n: int) -> Case:
    rows = x.shape[0]

    def call(ns, out):
        ns.top_k_per_row_decode(x, next_n, seq_lens, out, rows, x.stride(0), x.stride(1), K)
    if seq_lens.dim() == 2:
        n = seq_lens.reshape(-1).clamp(min=0)
    else:
        r = torch.arange(rows, device=DEV)
        n = (seq_lens[r // next_n] - next_n + r % next_n + 1).clamp(min=0)
    return Case(name, "top_k_per_row_decode", x, torch.zeros_like(n), n.clamp(max=x.shape[1]), call)


def cases(dist: str, seed: int) -> list[Case]:
    """GLM-5.3's: k = 2048 of up to about 33K keys a row; prefill calls of up to about 8K rows."""
    out = []
    x = logits(dist, 8192, 32768, seed)
    out.append(prefill("prefill-8192x32768-causal", x, *causal(8192, 32768 - 8192)))
    x = logits(dist, 4096, 4096, seed + 1)
    out.append(prefill("prefill-4096-causal-from-0", x, *causal(4096, 0)))
    # two sequences packed in one call: the second's keys start at column 3000
    x = logits(dist, 8000, 8000, seed + 2)
    ks, ke = causal(8000, 0)
    ks[3000:] = 3000
    ke[3000:] = 3000 + torch.arange(1, 5001, device=DEV, dtype=torch.int32)
    out.append(prefill("prefill-packed-3000+5000", x, ks, ke))
    for rows in (1, 16, 64):
        x = logits(dist, rows, 65536, seed + 10 + rows)
        out.append(length_op(f"cooperative-{rows}", "cooperative_topk", x, lens(rows, 1000, 33000, seed + rows)))
    for rows in (128, 512):
        x = logits(dist, rows, 65536, seed + 20 + rows)
        out.append(length_op(f"persistent-{rows}", "persistent_topk", x, lens(rows, 1000, 33000, seed + rows)))
    x = logits(dist, 32, 65536, seed + 30)
    out.append(decode("decode-32-2d", x, lens(32, 1000, 33000, seed + 31).reshape(32, 1), 1))
    x = logits(dist, 32, 65536, seed + 40)
    out.append(decode("decode-16x2-1d", x, lens(16, 1000, 33000, seed + 41), 2))
    return out


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


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


def up_to_ties(case: Case, got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Whether each long row of ``got`` selects the reference's keys above the k-th all, and the rest from the keys
    tied at the k-th, each index once; and whether the short rows equal ``ref``."""
    long = case.n > K
    short_ok = bool((got[~long] == ref[~long]).all())
    if not bool(long.any()):
        return {"long_rows": 0, "set_ok": True, "short_ok": short_ok}
    x, lo, n = case.x[long], case.lo[long].long(), case.n[long].long()
    g, r = got[long].long(), ref[long].long()
    in_range = bool(((g >= 0) & (g < n[:, None])).all())
    gs = g.clamp(min=0).sort(dim=1).values
    unique = bool((gs[:, 1:] != gs[:, :-1]).all())
    key = dsa_topk.keys(x)
    kg = key.gather(1, (lo[:, None] + g.clamp(min=0)).clamp(max=x.shape[1] - 1))
    kr = key.gather(1, lo[:, None] + r)
    tau = kr.min(dim=1, keepdim=True).values
    rel = torch.arange(x.shape[1], device=DEV)[None, :] - lo[:, None]
    tied = ((key == tau) & (rel >= 0) & (rel < n[:, None])).sum(1)
    above = (kg > tau).sum(1) == (kr > tau).sum(1)
    at_least = (kg >= tau).all(1)
    rows_in = ((g >= 0) & (g < n[:, None])).all(1)
    rows_unique = (gs[:, 1:] != gs[:, :-1]).all(1)
    bad = ~(above & at_least & rows_in & rows_unique)
    return {"long_rows": int(long.sum()), "set_ok": in_range and unique and bool(above.all()) and bool(at_least.all()),
            "short_ok": short_ok, "rows_order_differs": int((g != gs).any(1).sum()),
            "rows_bad": {"out_of_range": int((~rows_in).sum()), "repeated": int((~rows_unique).sum()),
                         "missing_above": int((~above).sum()), "below_kth": int((~at_least).sum())},
            "bad_rows_n": n[bad][:8].tolist(), "rows_tie_straddles_kth": int(((kr == tau).sum(1) < tied).sum())}


def check(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = SimpleNamespace(**{op: getattr(torch.ops._C, op) for op in dsa_topk.OPS})
    canon_ns = SimpleNamespace(**vars(real))
    canon = dsa_topk.install(canon_ns)
    busy = Busy()
    record = {"tag": args.tag, "pid": os.getpid(), "host": platform.node(), "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "reps": args.reps, "cases": []}
    for di, dist in enumerate(args.dists.split(",")):
        for case in cases(dist, 1000 * di):
            ref = dsa_topk.reference_rows(case.x, case.lo, case.n, K)
            row = {"case": case.name, "op": case.op, "dist": dist, "rows": case.x.shape[0], "cols": case.x.shape[1],
                   "long_rows": int((case.n > K).sum()), "unpatched": [], "canonical": [], "canonical_ref": [],
                   "unpatched_vs_ref": []}
            before = canon.stats(case.x.device).clone()
            for rep in range(args.reps):
                for which, ns in (("unpatched", real), ("canonical", canon_ns)):
                    if rep % 2:
                        with busy:
                            got = case.run(ns)
                    else:
                        got = case.run(ns)
                    torch.cuda.synchronize()
                    row[which].append(digest(got))
                    if which == "canonical":
                        row["canonical_ref"].append(bool(torch.equal(got, ref)))
                    else:
                        row["unpatched_vs_ref"].append(up_to_ties(case, got, ref))
            row["counters"] = dict(zip(dsa_topk.STATS, (canon.stats(case.x.device) - before).tolist()))
            # a garbage fast selection: the pass must bisect every long row and still write the reference
            junk = torch.randint(-3, case.x.shape[1] + 3, (case.x.shape[0], K), generator=_gen(di), device=DEV,
                                 dtype=torch.int32)
            mode = {"top_k_per_row_prefill": 0, "top_k_per_row_decode": 1}.get(case.op, 1)
            a, b = (case.lo, case.lo + case.n) if mode == 0 else (case.n, None)
            canon.run("garbage", case.x, junk, K, mode, a.to(torch.int32), None if b is None else b.to(torch.int32),
                      rows=case.x.shape[0], ncols=case.x.shape[1], ls0=case.x.stride(0), ls1=case.x.stride(1))
            long = case.n > K
            row["garbage_hint_ref"] = bool(torch.equal(junk[long], ref[long]))
            record["cases"].append(row)
            print(json.dumps({k: row[k] for k in ("case", "dist", "long_rows", "counters")}),
                  "unpatched", len(set(row["unpatched"])), "canonical", len(set(row["canonical"])),
                  "ref", all(row["canonical_ref"]), "garbage", row["garbage_hint_ref"],
                  "ties", all(u["set_ok"] for u in row["unpatched_vs_ref"]),
                  [u.get("rows_bad") for u in row["unpatched_vs_ref"] if not u["set_ok"]][:2], flush=True)
            del case
        torch.cuda.empty_cache()
    record["canon"] = canon.record()
    (out / f"check.{args.tag}.json").write_text(json.dumps(record, indent=1) + "\n")


def compare(args) -> None:
    d = Path(args.out)
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("check.*.json"))]
    by: dict[tuple[str, str], dict] = {}
    for run in runs:
        for row in run["cases"]:
            s = by.setdefault((row["case"], row["dist"]), {"case": row["case"], "dist": row["dist"], "op": row["op"],
                                                           "long_rows": row["long_rows"], "unpatched": [],
                                                           "canonical": [], "canonical_ref": True, "set_ok": True,
                                                           "short_ok": True, "counters": row["counters"],
                                                           "unpatched_rows_bad": {}, "unpatched_outputs_bad": 0})
            for u in row["unpatched_vs_ref"]:
                s["unpatched_outputs_bad"] += not u["set_ok"]
                for k, v in u.get("rows_bad", {}).items():
                    s["unpatched_rows_bad"][k] = s["unpatched_rows_bad"].get(k, 0) + v
            s["unpatched"] += row["unpatched"]
            s["canonical"] += row["canonical"]
            s["canonical_ref"] &= all(row["canonical_ref"]) and row["garbage_hint_ref"]
            s["set_ok"] &= all(u["set_ok"] for u in row["unpatched_vs_ref"])
            s["short_ok"] &= all(u["short_ok"] for u in row["unpatched_vs_ref"])
    table = []
    for s in by.values():
        s["unpatched_distinct"] = len(set(s.pop("unpatched")))
        s["canonical_distinct"] = len(set(s.pop("canonical")))
        table.append(s)
    total = sum(len(r["cases"][0]["unpatched"]) for r in runs) if runs else 0
    verdict = {
        "processes": len(runs), "outputs_per_case": total, "cases": len(table),
        "canonical_identical": all(s["canonical_distinct"] == 1 for s in table),
        "canonical_is_reference": all(s["canonical_ref"] for s in table),
        "unpatched_set_up_to_ties": {d: all(s["set_ok"] for s in table if s["dist"] == d)
                                     for d in dict.fromkeys(s["dist"] for s in table)},
        "short_rows_unchanged": all(s["short_ok"] for s in table),
        "unpatched_varies_in": sum(s["unpatched_distinct"] > 1 for s in table),
    }
    (d / "summary.json").write_text(json.dumps({"verdict": verdict, "cases": table}, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    for s in table:
        print(f"{s['case']:28s} {s['dist']:6s} long={s['long_rows']:5d} unpatched_distinct={s['unpatched_distinct']:2d}"
              f" canonical_distinct={s['canonical_distinct']} ref={s['canonical_ref']} ties={s['set_ok']}"
              f" short={s['short_ok']} {s['counters']} bad_outputs={s['unpatched_outputs_bad']}"
              f" {s['unpatched_rows_bad']}")


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
    import time

    torch.cuda.synchronize()
    torch.cuda._sleep(200_000_000)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e6 / iters


def time_cases() -> list[Case]:
    out = []
    for rows, prefix in ((8192, 32768 - 8192), (8192, 0), (2048, 32768 - 2048)):
        x = logits("randn", rows, prefix + rows, 7)
        out.append(prefill(f"prefill-{rows}x{prefix + rows}", x, *causal(rows, prefix)))
    for op, rows_list in (("cooperative_topk", (1, 4, 16, 64)), ("persistent_topk", (128, 512))):
        for rows in rows_list:
            for n in (8192, 32768):
                x = logits("randn", rows, 65536, 8)
                out.append(length_op(f"{op}-{rows}x{n}", op, x, torch.full((rows,), n, dtype=torch.int32,
                                                                            device=DEV)))
            x = logits("randn", rows, 65536, 9)
            out.append(length_op(f"{op}-{rows}x1500-short", op, x, torch.full((rows,), 1500, dtype=torch.int32,
                                                                               device=DEV)))
    return out


def variant(spec: str) -> dsa_topk.Canon:
    """``BLOCK:WARPS:SEG_MIN_KEYS[:FILL[:STAGES]]`` (``SEG_MIN_KEYS`` ``off``: one program a row), or ``default``."""
    if spec == "default":
        return dsa_topk.Canon()
    block, warps, seg, *rest = spec.split(":")
    return dsa_topk.Canon(seg_min_keys=1 << 40 if seg == "off" else int(seg), block=int(block), num_warps=int(warps),
                          fill=int(rest[0]) if rest else dsa_topk.FILL,
                          stages=int(rest[1]) if len(rest) > 1 else dsa_topk.STAGES)


def time_(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = SimpleNamespace(**{op: getattr(torch.ops._C, op) for op in dsa_topk.OPS})
    arms = {}
    for spec in args.variants.split(","):
        ns = SimpleNamespace(**vars(real))
        arms[spec] = (ns, dsa_topk.install(ns, variant(spec)))
    rows = []
    for case in time_cases():
        o = torch.empty((case.x.shape[0], K), dtype=torch.int32, device=DEV)
        iters = max(2, min(args.iters, int(2e5 // max(case.x.numel() // 1024, 1))))
        t_real = device_us(lambda: case.call(real, o), iters)
        row = {"case": case.name, "op": case.op, "rows": case.x.shape[0], "cols": case.x.shape[1],
               "long_rows": int((case.n > K).sum()), "iters": iters,
               "host_unpatched_us": host_us(lambda: case.call(real, o), iters), "canonical": {}}
        for spec, (ns, canon) in arms.items():
            t = device_us(lambda: case.call(ns, o), iters)
            row["canonical"][spec] = {"device_us": t, "host_us": host_us(lambda: case.call(ns, o), iters),
                                      "segments": canon.segments(DEV, case.x.shape[0], int(case.n.max()))}
        t_real2 = device_us(lambda: case.call(real, o), iters)
        row["unpatched_us_runs"] = [t_real, t_real2]
        row["unpatched_us"] = min(t_real, t_real2)
        best = min(row["canonical"], key=lambda s: row["canonical"][s]["device_us"])
        row["best"] = best
        row["added_us"] = row["canonical"][best]["device_us"] - row["unpatched_us"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    rec = {"device": torch.cuda.get_device_name(), "iters": args.iters, "variants": list(arms), "rows": rows,
           "canon": {spec: canon.record() for spec, (_, canon) in arms.items()}}
    (out / "time.json").write_text(json.dumps(rec, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("check", "compare", "time"))
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="0")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--dists", default="randn,bf16,int,relu,signed,edge")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--variants", default="default", help="stress.py time: the pass's parameters (`variant`), by comma")
    args = p.parse_args()
    if args.mode != "compare":
        import vllm._custom_ops  # noqa: F401
    {"check": check, "compare": compare, "time": time_}[args.mode](args)


if __name__ == "__main__":
    main()
