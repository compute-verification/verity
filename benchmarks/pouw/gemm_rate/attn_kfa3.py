"""A not-byte-equal prefill attention kernel against FA3 at the served 1B shape, each arm gated before it is timed.

    python attn_kfa3.py --stamp-cubin stamp.cubin --fa3-so NS=PATH [--fa3-so ...] [--prompt 8x1024 ...]
                        [--arms served,contig] [--window-ms 2000 --lead-ms 500 --rounds 3] [--table table.md] > kfa3.jsonl

The steps, caches and calls are attn_knobs.py's (32 query heads, 8 KV heads, head dim 64, causal, a --prompt BxL being B
sequences of L tokens; BxLs is that step with each sequence's first key spiked, `spike`, which an unchecked lazy max
overflows on). The arms are FA3's `served` and `contig` (the wheel) and, per --fa3-so, a rebuilt kernel's `so:NS`
and `so:NS:contig` (fa3_tile.sh). Its output need not equal FA3's bytes, so each arm must pass, before it is timed:

- finite, and determinism: equal bytes on a rerun;
- batch invariance: equal bytes with another sequence ahead of the step's in the batch;
- within ULP_BOUND bf16 ulp of FA3 (`served`) at the output's scale: max |out - FA3| at most ULP_BOUND ulp of max |FA3|;
- no farther from an fp32 reference (`reference`, TF32 off) than FA3 is, plus one ulp at the output's scale: max abs at
  most FA3's plus one ulp of max |reference|.

An arm that fails one is reported and never timed. Per-element ulps are reported, not gated: where an output is near zero
its ulp is tiny, so any kernel whose bytes differ from FA3's is many ulp from it there, and FA3 itself is ~30,000 from
the reference (a sign across zero). Each gate row carries
the max abs and max ulp against FA3 and the reference, and the shares of output elements that differ from FA3 and that
are more than ULP_BOUND ulp from it. Timing is attn_knobs.py's: rounds interleave the passing arms in graphed windows,
and a window's useful causal MACs a SM clock are against bf16's 2,048.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attn_knobs as ak  # noqa: E402
import attn_rate as ar  # noqa: E402
import torch_rate as tr  # noqa: E402

PROMPTS = ("8x1024", "16x1024", "32x1024")
ULP_BOUND = 2


def reference(s: ak.Step) -> torch.Tensor:
    """s's causal attention in fp32 with TF32 off, [T, Hq, D]: each query head against its KV head (h // (Hq / Hkv))."""
    torch.backends.cuda.matmul.allow_tf32 = False
    g = s.hq // s.k.shape[1]
    out = torch.empty(s.t, s.hq, s.d, dtype=torch.float32, device="cuda")
    o = 0
    for n in s.lens:
        q = s.q[o:o + n].float()
        k = s.k[o:o + n].float().repeat_interleave(g, dim=1)
        v = s.v[o:o + n].float().repeat_interleave(g, dim=1)
        sc = torch.einsum("qhd,khd->hqk", q, k) * s.scale
        sc.masked_fill_(torch.ones(n, n, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        out[o:o + n] = torch.einsum("hqk,khd->qhd", sc.softmax(-1), v)
        o += n
    return out


def spike(s: ak.Step, a: float = 16.0, b: float = 64.0) -> None:
    """Each sequence's first key, which FA3's causal loop reaches last, made to outscore the rest by ~170 in log2 units.

    Every query gains `a` on dimension 0 and that key is `b` there and 0 elsewhere, so it scores ~a * b * scale = 128
    (185 in log2 units) against ~a * 3 * scale = 6 for the largest other key: past exp2's range (128 in log2 units) from
    any other block's max. Its caches are rebuilt from the changed rows.
    """
    o = 0
    for n in s.lens:
        s.k[o] = 0
        s.k[o, :, 0] = b
        s.q[o:o + n, :, 0] += a
        o += n
    s.caches = {"served": ak.paged(s.k, s.v, s.lens, ak.BLOCK, True), "kattn": ak.paged(s.k, s.v, s.lens, ak.BLOCK, False),
                "page128": ak.paged(s.k, s.v, s.lens, 128, True)}


def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """How many bf16 ulps apart each pair of elements is, on ak.ulps's sign-magnitude integer line."""
    def line(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (line(a) - line(b)).abs()


def ulp_at(x: float) -> float:
    """One bf16 ulp at magnitude x (8 significand bits)."""
    return math.ldexp(1.0, math.frexp(x)[1] - 8)


def against_ref(o: torch.Tensor, ref: torch.Tensor) -> dict:
    e = (o.float() - ref).abs()
    return {"ulps": int(dist(o, ref.to(torch.bfloat16)).max()), "abs": float(e.max()), "mean_abs": float(e.mean())}


def gate(arms: ak.Arms, a: str, s: ak.Step, bat: ak.Step, fa3, ref: torch.Tensor, fa3_ref: dict) -> dict:
    c = ak.checks(arms, a, s, bat, fa3)
    o = ak.snap(arms.call(a, s, ak.poisoned(s))())[0]
    d = dist(o, fa3[0])
    fa3_abs = float((o.float() - fa3[0].float()).abs().max())
    fa3_scale = ulp_at(float(fa3[0].float().abs().max()))
    r = against_ref(o, ref)
    ref_abs_bound = fa3_ref["abs"] + ulp_at(float(ref.abs().max()))
    passed = (c["finite"] and c["repeat"]["eq"] and c["batch"]["eq"] and fa3_abs <= ULP_BOUND * fa3_scale
              and r["abs"] <= ref_abs_bound)
    return {**c, "vs_fa3_ulps": int(d.max()), "vs_fa3_abs": fa3_abs, "vs_fa3_scale_ulps": fa3_abs / fa3_scale,
            "vs_fa3_frac_ne": float((d > 0).float().mean()), "vs_fa3_frac_gt": float((d > ULP_BOUND).float().mean()),
            "vs_ref_ulps": r["ulps"], "vs_ref_abs": r["abs"], "vs_ref_mean_abs": r["mean_abs"],
            "ref_abs_bound": ref_abs_bound, "pass": passed}


def table(rows: list[dict]) -> str:
    by: dict = {}
    for r in rows:
        if r.get("kind") == "window" and r.get("sms"):
            by.setdefault((r["prompt"], r["arm"]), []).append(r)
    gates = {(r["prompt"], r["arm"]): r for r in rows if r.get("kind") == "gate"}
    med = {k: {f: statistics.median(x[f] for x in rs) for f in ("of_peak", "theta_1830", "sm_mhz", "us_call")}
           for k, rs in by.items()}
    refs = {r["prompt"]: r for r in rows if r.get("kind") == "reference"}
    out = ["| prompt | arm | of bf16 a clock | × FA3 | θ @1830 | SM MHz | µs a layer | vs FA3: max ulp, abs (ulp at scale), "
           f"share ≠, share > {ULP_BOUND} ulp | vs fp32: max abs, mean abs (FA3's) |", "|---|---|---|---|---|---|---|---|---|"]
    for (prompt, arm), m in sorted(med.items()):
        g = gates.get((prompt, arm), {})
        f = refs.get(prompt, {})
        fa3 = med.get((prompt, "served"), {}).get("of_peak")
        x = f"{m['of_peak'] / fa3:.2f}" if fa3 else "—"
        out.append(f"| {prompt} | {arm} | {m['of_peak']:.3f} | {x} | {m['theta_1830']:.3f} | {m['sm_mhz']:.0f} | "
                   f"{m['us_call']:.1f} | {g.get('vs_fa3_ulps', '—')}, {g.get('vs_fa3_abs', 0):.3g} "
                   f"({g.get('vs_fa3_scale_ulps', 0):.2f}), {g.get('vs_fa3_frac_ne', 0):.3%}, {g.get('vs_fa3_frac_gt', 0):.4%} | "
                   f"{g.get('vs_ref_abs', 0):.3g} ({f.get('fa3_vs_ref_abs', 0):.3g}), {g.get('vs_ref_mean_abs', 0):.3g} "
                   f"({f.get('fa3_vs_ref_mean_abs', 0):.3g}) |")
    refused = [f"{p} {a}" for (p, a), g in sorted(gates.items()) if not g.get("pass")]
    return "\n".join(out) + "\n" + (f"\nRefused (not timed): {', '.join(refused)}\n" if refused else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stamp-cubin", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ar.CONFIG)
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--arms", default="served,contig")
    ap.add_argument("--fa3-so", action="append", default=[], help="NS=PATH: a rebuilt kernel, arms so:NS and so:NS:contig")
    ap.add_argument("--window-ms", type=float, default=2000)
    ap.add_argument("--lead-ms", type=float, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--table", type=Path)
    args = ap.parse_args()
    sos = dict(x.split("=", 1) for x in args.fa3_so)
    names = args.arms.split(",") + [a for ns in sos for a in (f"so:{ns}", f"so:{ns}:contig")]
    m = ar.model(args.config)
    nv = tr.Nvml()
    st = tr.Stamps(args.stamp_cubin)
    st.ck("cuFuncSetAttribute", st.f, ar.CARVEOUT, 100)
    arms = ak.Arms(sos)
    rows: list[dict] = []

    def emit(d: dict) -> None:
        rows.append(d)
        print(json.dumps(d), flush=True)

    emit({"kind": "device", "uuid": nv.uuid, "name": torch.cuda.get_device_name(0), "sms": st.nsm, "torch": torch.__version__,
          "cuda": torch.version.cuda, "arms": names, "sos": sos, "ulp_bound": ULP_BOUND, **nv.sample()})
    for pi, prompt in enumerate(args.prompt or PROMPTS):
        lens = ar.parse_prompt(prompt.removesuffix("s"))
        arms.sched_cache.clear()
        s = ak.Step(m, lens, args.seed + pi)
        if prompt.endswith("s"):
            spike(s)
        bat = ak.batched(m, s, args.seed + 100 + pi)
        fa3 = ak.snap(arms.call("served", s, ak.poisoned(s))())
        ref = reference(s)
        fa3_ref = against_ref(fa3[0], ref)
        emit({"kind": "reference", "prompt": prompt, "fa3_vs_ref_ulps": fa3_ref["ulps"], "fa3_vs_ref_abs": fa3_ref["abs"],
              "fa3_vs_ref_mean_abs": fa3_ref["mean_abs"], "ref_max_abs": float(ref.abs().max())})
        guard = ak.Guard(s, bat)

        def intact(after: str, guard: ak.Guard = guard) -> None:
            if bad := guard.changed():
                emit({"kind": "corrupt", "prompt": prompt, "after": after, "changed": bad})
                raise SystemExit(f"{after} wrote into {bad}")

        macs = ar.attn_macs(m, lens)
        ready = []
        for a in names:
            g = gate(arms, a, s, bat, fa3, ref, fa3_ref)
            emit({"kind": "gate", "prompt": prompt, "arm": a, **g})
            intact(f"{a} gate")
            if not g["pass"]:
                continue
            f = arms.call(a, s)
            it = ar.prepare(emit, a, "bf16", macs, f, True, {"prompt": prompt, "t": s.t, "form": "whole"})
            it["call"] = f  # its out buffer, which the graph writes: torch.cuda.graph's empty_cache would free it otherwise
            for _ in range(3):
                it["fn"]()
            intact(f"{a} prepare")
            ready.append(it)
        del ref
        for r in range(args.rounds):
            for it in ready:
                emit({"kind": "start", "prompt": prompt, "arm": it["arm"], "round": r})
                emit(ar.window(args, nv, st, it, r))
                intact(f"{it['arm']} window")
        del ready, s, bat, guard
        torch.cuda.empty_cache()
    if args.table:
        args.table.write_text(table(rows))


if __name__ == "__main__":
    main()
