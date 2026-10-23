"""The fused-MoE grouped call on one H100 or H200 against the same rows called unit by unit, at EP8's expert shapes (w13
(4096, 6144) and w2 (6144, 2048)), with decode's and prefill's routed counts (`COUNTS`).

  python benchmarks/pouw/pearl_c_sm90/grouped_check.py --out DIR --passes SCRATCH [--cubin C] [--reps 10]
  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py verify --out SCRATCH/NxK      # on CPUs, after, a shape each

A shape is one install (`glm53_timing.device_run`) of `EXPERTS` experts, ``pearl-c-h100-v1-h3``, the serial schedule.  At each
count pattern, two retained passes from outputs poisoned 0xA5, in SCRATCH/NxK: ``dense-P``, each unit's rows (`grouped.units`:
an expert's rows in blocks of at most 256) through `PearlCDeviceRun.gemm` in unit order (the first pattern's before the
experts are grouped), then ``grouped-P``, one `pouw_pearl_c_grouped.gemm_grouped` call.  The pair must agree byte for byte: y,
and the manifests less their run name (the transcript and records roots, every A root, record and file digest), with every
call's A rows file read and compared.  Then the two are timed interleaved, rep by rep, in fresh unretained passes
(`glm53_timing.timed`).  SCRATCH/NxK gets ``capture.json`` and ``coins.json``, for glm53_passes.py verify; DIR gets
``grouped.json`` and each pass's manifest.  Exits 0 only when every pair agrees.
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


def _same(a: Path, b: Path) -> list[str]:
    """Where two retained passes differ: their manifests less the run name, and every call's A rows file."""
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (a, b))
    out = [k for k in sorted(set(ma) | set(mb)) if k != "run" and ma.get(k) != mb.get(k)]
    for x, y in zip(ma["matmuls"], mb["matmuls"]):
        if (a / x["rows"]["path"]).read_bytes() != (b / y["rows"]["path"]).read_bytes():
            out.append(f"matmul {x['index']} rows")
    return out


def check_shape(a, n: int, k: int) -> dict:
    import torch
    import glm53_timing as T
    from verity_vllm.protocol_options import pouw_pearl_c_grouped as PG

    run, weights, kernel, gen = T.device_run([(n, k)] * EXPERTS, a.scheme, a.cubin)
    R, G = run.kernel.run, PG.module(run)
    d = Path(a.passes) / f"{n}x{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "coins.json").write_text(json.dumps(run.coins.to_json()) + "\n")
    ws = tuple(sorted(weights))
    out, passes = {"n": n, "k": k, "experts": EXPERTS, "cubin_sha256": kernel.cubin_sha256}, {}
    for pat, counts in COUNTS.items():
        x = torch.randn(sum(counts), k, device="cuda", generator=gen).to(torch.bfloat16)
        us = G.units(counts, n, R.PD_MAX_ROWS)

        def dense():
            return torch.cat([run.gemm(ws[u.pos], x[u.src:u.src + u.m], weights[ws[u.pos]], None) for u in us])

        def grouped():
            return PG.gemm_grouped(run, ws, x, counts)

        ys, ends = {}, {}
        for name, f in (("dense", dense), ("grouped", grouped)):
            run.end_pass(retain=str(d / f"{name}-{pat}"), run=f"grouped-check {n}x{k} {pat}")
            ys[name] = f()
            ends[name] = {kk: v for kk, v in run.end_pass().items() if kk != "manifest"}
            passes[f"{name}-{pat}"] = ends[name]
        differ = _same(d / f"dense-{pat}", d / f"grouped-{pat}")
        same_y = torch.equal(ys["dense"], ys["grouped"])
        arms = {"dense": dense, "grouped": grouped}
        for f in arms.values():
            T.timed(f, run.end_pass, run.end_pass)
        samples = {nm: [] for nm in arms}
        for rep in range(a.reps):
            for nm in list(arms)[::1 if rep % 2 == 0 else -1]:
                samples[nm].append(T.timed(arms[nm], run.end_pass, run.end_pass))
        med = {nm: statistics.median(v) for nm, v in samples.items()}
        out[pat] = {"counts": list(counts), "units": len(us), "tiles": sum(u.tiles for u in us), "y_equal": same_y,
                    "manifests_differ": differ, "transcript_roots": [ends[x]["transcript_root"] for x in ("dense", "grouped")],
                    "ms": med, "runs": samples, "speedup": med["dense"] / med["grouped"]}
        print(f"{n}x{k} {pat}: {len(us)} units, y {'equal' if same_y else 'DIFFERS'}, manifests "
              f"{'equal' if not differ else 'differ in ' + ', '.join(differ)}; dense {med['dense']:.3f} ms, grouped "
              f"{med['grouped']:.3f} ms ({out[pat]['speedup']:.2f}x)", flush=True)
    (d / "capture.json").write_text(json.dumps({"scheme": a.scheme, "shape": [n, k], "passes": passes}, indent=1) + "\n")
    out["ok"] = all(out[p]["y_equal"] and not out[p]["manifests_differ"] for p in COUNTS)
    run.finish()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--passes", required=True, help="where the retained passes go (each about EXPERTS weights' rows)")
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin")
    p.add_argument("--reps", type=int, default=10)
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
