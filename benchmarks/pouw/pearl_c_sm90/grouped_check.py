"""The fused-MoE grouped call on one H100 or H200 against the same rows called unit by unit, at EP8's expert shapes (w13
(4096, 6144) and w2 (6144, 2048)), with decode's and prefill's routed counts (`COUNTS`), under each schedule.

  python benchmarks/pouw/pearl_c_sm90/grouped_check.py --out DIR --passes SCRATCH [--cubin C] [--reps 10] [--schedule S ...]
  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py verify --out SCRATCH/NxK      # on CPUs, after, a shape each

A shape is one install (`glm53_timing.device_run`) of `EXPERTS` experts, ``pearl-c-h100-v1-h3``.  At each count pattern P,
retained passes from outputs poisoned 0xA5, in SCRATCH/NxK: ``dense-P``, each unit's rows (`grouped.units`: an expert's rows in
blocks of at most 256) through `PearlCDeviceRun.gemm` in unit order under the serial schedule (the first pattern's before the
experts are grouped), then one `pouw_pearl_c_grouped.gemm_grouped` call under each schedule of ``--schedule`` (default serial,
deferred and deferred-a): ``grouped-P`` under serial, ``grouped-S-P`` under S, run eagerly; and with ``--graphs`` (the default)
each again with the run's graphs, ``grouped[-S]-graph-P``, after `WARM_GRAPHS` calls in a pass of its own, so the retained call
replays its layout's lane graphs (it must have).  Each grouped pass must agree with the dense one
byte for byte: y, and the manifests less their run name (the transcript and records roots, every A root, record and file
digest), with every call's A rows file read and compared.  Then the arms are timed interleaved, rep by rep (the order rotated),
the pearl arms in fresh unretained passes under their schedule, each sample closed only after the side lanes
(`glm53_timing.timed`), beside the stock grouped arms that build (`stock_arms`), each checked against x·Wᵀ.  SCRATCH/NxK gets
``capture.json`` and ``coins.json``, for glm53_passes.py verify; DIR gets ``grouped.json`` and each pass's manifest.  Exits 0
only when every grouped pass agrees.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm53  # noqa: E402

EXPERTS = 8
SHAPES = ((4096, 6144), (6144, 2048))
#: routed rows a local expert, by pattern: a decode step's few and a prefill chunk's (blocks of 256 and a partial one, an
#: expert of 2,048).  No expert is idle: a pass calls every weight its epoch commits (`audit.Epoch`), and verify reads these
COUNTS = {"decode": (3, 1, 17, 1, 40, 8, 2, 5), "prefill": (300, 256, 1000, 64, 17, 512, 2048, 128)}
SCHEDULES = ("serial", "deferred", "deferred-a")
#: calls a graphed arm makes before its retained pass: a layout's graphs are captured at its second call on a ring set, and
#: the ring has at most two
WARM_GRAPHS = 4


def _same(a: Path, b: Path) -> list[str]:
    """Where two retained passes differ: their manifests less the run name and the schedule the pass ran under (its lanes,
    which commit nothing), and every call's A rows file."""
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (a, b))
    out = [k for k in sorted(set(ma) | set(mb)) if k not in ("run", "schedule") and ma.get(k) != mb.get(k)]
    for x, y in zip(ma["matmuls"], mb["matmuls"]):
        if (a / x["rows"]["path"]).read_bytes() != (b / y["rows"]["path"]).read_bytes():
            out.append(f"matmul {x['index']} rows")
    return out


def stock_arms(Ws: list, dense: list, counts, x) -> tuple[dict, dict]:
    """The stock arms for the experts' weights `Ws` (BF16, n × k each) at `counts` (x's rows gathered by expert) that build:
    name -> f() -> y (x's rows by n, in x's order), and name -> the error of each that didn't.  Each arm quantizes x per
    128-wide token group in its call, as `glm53_timing.stock_arms`' do:

    * ``deepgemm-contiguous``: DeepGEMM's m-grouped contiguous FP8 GEMM, each expert's rows at its block of the alignment
      (padding rows -1 in the expert index), x permuted in and y out in the call;
    * ``deepgemm-masked``: DeepGEMM's m-grouped masked FP8 GEMM (expert e's first counts[e] of M_max rows; decode's EP form);
    * ``deepgemm-loop``, ``cutlass-loop``, ``triton-loop``: each expert's dense arm (`dense`, `glm53_timing.stock_arms`' of
      each weight), an expert at a time;
    * ``NAME-graph``: each of those that builds captured as one CUDA graph (`glm53_timing.capture_stock`, as vLLM's decode
      graphs run the EP call), held to the same bound with its output poisoned before a replay (`glm53_timing.replay_error`)."""
    import glm53_timing as T
    import torch
    arms, errors = {}, {}
    dev, (n, k), E = x.device, Ws[0].shape, len(Ws)
    src = [sum(counts[:e]) for e in range(E)]
    ref = torch.cat([x[src[e]:src[e] + c].float() @ Ws[e].float().T for e, c in enumerate(counts)])
    try:
        from vllm.model_executor.layers.quantization.utils import fp8_utils as FU
        from vllm.utils import deep_gemm as DG
    except Exception as e:  # noqa: BLE001 - recorded
        FU = DG = None
        errors["deepgemm"] = f"vllm's fp8_utils or deep_gemm: {type(e).__name__}: {e}"

    def quant_b():
        qs = [T._quant_weight(W, ue8m0=True) for W in Ws]
        return torch.stack([q for q, _ in qs]), torch.stack([s for _, s in qs])

    def quant_a(a):
        kw = {"column_major_scales": True, "use_ue8m0": True, "tma_aligned_scales": True}
        try:
            return FU.per_token_group_quant_fp8(a, 128, **kw)
        except TypeError:
            kw.pop("tma_aligned_scales")
            return FU.per_token_group_quant_fp8(a, 128, **kw)

    def contiguous():
        align = getattr(DG, "get_mk_alignment_for_contiguous_layout", None)
        align = align()[0] if align else 128
        bq, bs = quant_b()
        rows, dst, ids = 0, [], []
        for e, c in enumerate(counts):
            dst += range(rows, rows + c)
            ids += [e] * c + [-1] * (-(-c // align) * align - c)
            rows += -(-c // align) * align
        dst, ids = torch.tensor(dst, device=dev), torch.tensor(ids, dtype=torch.int32, device=dev)
        xp = torch.zeros(rows, k, dtype=torch.bfloat16, device=dev)

        def f():
            xp.index_copy_(0, dst, x)
            out = torch.empty(rows, n, dtype=torch.bfloat16, device=dev)
            DG.m_grouped_fp8_gemm_nt_contiguous(quant_a(xp), (bq, bs), out, ids)
            return out.index_select(0, dst)
        return f

    def masked():
        mm = getattr(DG, "fp8_m_grouped_gemm_nt_masked", None) or DG.m_grouped_fp8_gemm_nt_masked
        M = -(-max(counts) // 64) * 64
        bq, bs = quant_b()
        dst = torch.tensor([e * M + r for e, c in enumerate(counts) for r in range(c)], device=dev)
        masked_m = torch.tensor(counts, dtype=torch.int32, device=dev)
        expected = max(1, -(-sum(counts) // E))
        xp = torch.zeros(E * M, k, dtype=torch.bfloat16, device=dev)

        def f():
            xp.index_copy_(0, dst, x)
            aq, a_s = FU.per_token_group_quant_fp8(xp, 128, column_major_scales=False)
            out = torch.empty(E, M, n, dtype=torch.bfloat16, device=dev)
            mm((aq.view(E, M, k), a_s.view(E, M, k // 128)), (bq, bs), out, masked_m, expected)
            return out.view(E * M, n).index_select(0, dst)
        return f

    def loop(name):
        def make():
            bad = [b[1][name] for b in dense if name not in b[0]]
            if bad:
                raise RuntimeError(bad[0])
            fs = [b[0][name] for b in dense]

            def f():
                return torch.cat([fs[e](x[src[e]:src[e] + c]) for e, c in enumerate(counts) if c])
            return f
        return make

    makers = {"deepgemm-loop": loop("deepgemm"), "cutlass-loop": loop("cutlass"), "triton-loop": loop("triton")}
    if DG is not None:
        makers = {"deepgemm-contiguous": contiguous, "deepgemm-masked": masked, **makers}
    for name, make in makers.items():
        try:
            f = make()
            y = f()
            torch.cuda.synchronize()
            err = float(((y.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
            if not err < T.ERR:
                raise ValueError(f"rel RMS {err:.4f} against x·Wᵀ")
            arms[name] = f
        except Exception as e:  # noqa: BLE001 - recorded, the arm left out
            errors[name] = f"{type(e).__name__}: {e}"[:400]
    for name, f in list(arms.items()):
        g, captured, err = T.capture_stock(f)
        err = T.replay_error(g, ref) if captured else err
        if captured and err is None:
            arms[f"{name}-graph"] = g
        else:
            errors[f"{name}-graph"] = err
    return arms, errors


def check_shape(a, n: int, k: int) -> dict:
    import torch
    import glm53_timing as T
    from verity_vllm.protocols.pouw.pearl_c import grouped as PG

    run, weights, kernel, gen = T.device_run([(n, k)] * EXPERTS, a.scheme, a.cubin, graphs=a.graphs)
    R, G = run.kernel.run, PG.module(run)
    d = Path(a.passes) / f"{n}x{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "coins.json").write_text(json.dumps(run.coins.to_json()) + "\n")
    ws = tuple(sorted(weights))
    out = {"n": n, "k": k, "experts": EXPERTS, "cubin_sha256": kernel.cubin_sha256, "schedules": list(a.schedule),
           "has_pd_tree_g": bool(G.has_tree(kernel.dev)), "graphs": a.graphs}
    passes, dense_stock = {}, [T.stock_arms(weights[w]) for w in ws]

    def join():
        run.join_sides(torch.cuda.current_stream())

    def begin(sc: str, graphed: bool, **kw):
        run.end_pass(schedule=sc, **kw)
        run.graphs = graphed
    for pat, counts in COUNTS.items():
        x = torch.randn(sum(counts), k, device="cuda", generator=gen).to(torch.bfloat16)
        us = G.units(counts, n, R.PD_MAX_ROWS)

        def dense():
            return torch.cat([run.gemm(ws[u.pos], x[u.src:u.src + u.m], weights[ws[u.pos]], None) for u in us])

        def grouped():
            return PG.gemm_grouped(run, ws, x, counts)

        pearls = {"dense": ("serial", False, dense),
                  **{("grouped" if sc == "serial" else f"grouped-{sc}") + ("-graph" if gr else ""): (sc, gr, grouped)
                     for gr in ((False, True) if a.graphs else (False,)) for sc in a.schedule}}
        ys, ends, replayed, at = {}, {}, {}, None
        for name, (sc, gr, f) in pearls.items():
            if gr:                                       # its layout's graphs made on every ring set, so the pass replays them
                begin(sc, gr)
                for _ in range(WARM_GRAPHS):
                    f()
                done = PG.group(run, ws).replays
            run.repeat_at = at        # every arm's retained pass numbered from the first's, so their manifests compare
            begin(sc, gr, retain=str(d / f"{name}-{pat}"), run=f"grouped-check {n}x{k} {pat}")
            at = run.cur.base
            ys[name] = f()
            if gr:
                replayed[name] = PG.group(run, ws).replays > done
            ends[name] = {kk: v for kk, v in run.end_pass(schedule="serial").items() if kk != "manifest"}
            passes[f"{name}-{pat}"] = ends[name]
        run.graphs = False
        grouped_arms = [nm for nm in pearls if nm != "dense"]
        differ = {nm: _same(d / f"dense-{pat}", d / f"{nm}-{pat}") for nm in grouped_arms}
        same_y = {nm: torch.equal(ys["dense"], ys[nm]) for nm in grouped_arms}
        stock, errors = stock_arms([weights[w] for w in ws], dense_stock, counts, x)
        arms = {**{nm: (f, (lambda sc=sc, gr=gr: begin(sc, gr)), run.end_pass, join) for nm, (sc, gr, f) in pearls.items()},
                **{nm: (f, None, None, None) for nm, f in stock.items()}}
        for f, before, after, j in arms.values():          # warm-up: buffers, autotuning
            T.timed(f, before, after, join=j)
        samples, names = {nm: [] for nm in arms}, list(arms)
        for rep in range(a.reps):
            for nm in names[rep % len(names):] + names[:rep % len(names)]:
                f, before, after, j = arms[nm]
                samples[nm].append(T.timed(f, before, after, join=j))
        run.end_pass(schedule="serial")
        med = {nm: statistics.median(v) for nm, v in samples.items()}
        eager = [nm for nm in stock if not nm.endswith("-graph")]
        best = min(eager, key=med.get) if eager else None
        gbest = min((nm for nm in stock if nm.endswith("-graph")), key=med.get, default=None)
        out[pat] = {"counts": list(counts), "units": len(us), "tiles": sum(u.tiles for u in us), "y_equal": same_y,
                    "manifests_differ": differ, "replayed": replayed,
                    "transcript_roots": {nm: ends[nm]["transcript_root"] for nm in pearls},
                    "ms": med, "runs": samples, "stock_errors": errors, "fastest_stock": best,
                    "fastest_stock_graphed": gbest, "speedup": {nm: med["dense"] / med[nm] for nm in grouped_arms},
                    "slowdowns": {nm: med[nm] / med[best] for nm in pearls} if best else None,
                    "slowdowns_graphed": {nm: med[nm] / med[gbest] for nm in pearls} if gbest else None}
        print(f"{n}x{k} {pat}: {len(us)} units; " + "; ".join(
            f"{nm} y {'equal' if same_y[nm] else 'DIFFERS'}, manifests "
            f"{'equal' if not differ[nm] else 'differ in ' + ', '.join(differ[nm])}"
            + ("" if nm not in replayed else ", replayed" if replayed[nm] else ", NOT REPLAYED") for nm in grouped_arms),
            flush=True)
        print(f"{n}x{k} {pat}: " + ", ".join(f"{nm} {v:.3f} ms" for nm, v in med.items())
              + "".join(f"; slowdown over {b}: " + ", ".join(f"{nm} {med[nm] / med[b]:.2f}x" for nm in pearls)
                        for b in (best, gbest) if b)
              + ("" if best else "; no stock arm") + (f"; stock errors: {errors}" if errors else ""), flush=True)
    (d / "capture.json").write_text(json.dumps({"scheme": a.scheme, "shape": [n, k], "passes": passes}, indent=1) + "\n")
    out["ok"] = all(all(out[p]["y_equal"].values()) and not any(out[p]["manifests_differ"].values())
                    and all(out[p]["replayed"].values()) for p in COUNTS)
    run.finish()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--passes", required=True, help="where the retained passes go (each about EXPERTS weights' rows)")
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--schedule", nargs="+", default=list(SCHEDULES),
                   help="the grouped call's schedules, each its own arm (verity.core.protocols.pouw.serving.SCHEDULES)")
    p.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True,
                   help="each schedule's grouped arm a second time with the run's graphs (its layout's lane graphs replayed)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"shapes": [check_shape(a, n, k) for n, k in SHAPES]}
    for n, k in SHAPES:
        for m in sorted((Path(a.passes) / f"{n}x{k}").glob("*/manifest.json")):
            dst = out / f"{n}x{k}" / m.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(m, dst / "manifest.json")
    res["ok"] = all(s["ok"] for s in res["shapes"])
    (out / "grouped.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"grouped check {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
