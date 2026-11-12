"""GLM-5.3's MTP draft layer's eh_proj on an H200: the cuBLAS route vLLM d9105ea80 takes at K 12288, N 6144 for every token count a
draft step launches it at, and every word of the launches against the Definition each route binds.

At the pin `DeepSeekMultiTokenPredictorLayer` (models/deepseek_v32/nvidia/mtp.py) makes eh_proj a plain ``nn.Linear(2 * 6144,
6144, bias=False)`` (line 68), replicated on every rank, its weight bf16; its forward tries `run_glm52_plan` first (line 132),
whose plan exists only on compute capability 10.3 (glm52_low_latency_gemm.py:89), then ``self.eh_proj(eh_input)`` (line 134):
``F.linear`` on the contiguous [M, 12288] output of `fused_eh_norm`, which is cuBLAS at vLLM's served configuration, as
`dispatch_unquantized_gemm()` is on cc 9.0.  The first draft step launches it at the target step's token count, the later ones at
the request count, so M runs from 1 to the largest launch served: M_MAX, the route table's `m_max`.

One GPU process writes `<out>/glm53_mtp_eh_proj_capture.json`, the property record `glm53_mtp_eh_proj_capture`
(`properties.record`), the scans' records under `<out>/scan/`, and the compared rows' inputs and outputs under `<out>/words/`.  It
is `glm53_bf16_gemm_capture_gpu`'s served-default capture at one more shape (`MODEL` in its `MODELS`), with its own launches:

  gate     `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs; nothing else runs when it fails (exit 3).
  switch   vLLM's served cuBLAS configuration, read back before the first GEMM and after the last (`BG.switch`, `configuration`).
  scan     one process per range of SCAN_CHUNKS, which tile 1..M_MAX (`--scan-run`: `BG.scan_run` with `SCAN_M` the range): every
           M launched once under one profiler session, regimes where the kernels and the kept words stay the same, each
           regime's split count fitted out of its words; then `BG.route_table` joins them into this shape's route runs
           [m0, m1, split] (split None: one serial chain, `Gemm_v2`), refusing a gap, an overlap or conditions that differ.
  launch   F.linear twice at every M of LAUNCH_M and at the first M of each route class (and M_MAX), at the split count the
           routes give at that M; the profiler's reduce kernel present exactly when the routes say split; the crafted split-K
           probe on each split launch (`BG.launch`).
  words    every word of rows ROWS_ALL and below of each launch, and of the WINDOW rows at each end and the middle of larger ones,
           against `bf16_gemm_check.linear_rows` (`Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}`) or
           `cublas_splitk_rows` at the split (`CublasSplitKGemm_v1`, `cublas_bf16.CublasSplitKGemmWord`); the reference
           evaluator on rows 0, 3 and M - 1, three coordinates each; the probe's words against every split-K hypothesis, only
           the contiguous/f32/forward one fitting.
  bind     Gemm_v2 and CublasSplitKGemm_v1 at each split the routes hold bind with this shape's signature.

GO = all of it, no error.

    cd adapters/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_mtp_eh_proj_capture_gpu.py --out DIR --scratch /tmp
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import json
import multiprocessing
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_bf16_gemm_capture_gpu as BG
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-mtp-eh-proj-capture/v1"
RECORD = "glm53_mtp_eh_proj_capture.json"
MODEL, PHASE, NAME = "glm53-mtp", "linear", "mtp.eh_proj"
K, N = 2 * 6144, 6144
BG.MODELS.setdefault(MODEL, BG.Model(6144, 256, (), (), ((NAME, K, N),), (), ()))
#: the route table's m_max: the largest launch served (`bf16_gemm_check.CUBLAS_ROUTES`)
M_MAX = 8192
SCAN_CHUNKS = ((1, 1024), (1025, 2048), (2049, 4096), (4097, 8192))
LAUNCH_M = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
ROWS_ALL, WINDOW, CHUNK = 1024, 128, 64
TOTALS = ("rows", *BG.TOTALS)


# -------------------------------------------------------------------------------------------------------------------------- scan
def scan_chunk(a: argparse.Namespace, outdir: Path, lo: int, hi: int) -> dict[str, Any]:
    """One `--scan-run` process over M in [lo, hi]: its scan.json, `ok` False when the process or its record failed."""
    sub = outdir / "scan" / f"{lo}-{hi}"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "scan.json").unlink(missing_ok=True)
    argv = [sys.executable, str(Path(__file__).resolve()), "--scan-run", "--scan-m", f"{lo}:{hi}", "--out", str(sub)]
    argv += ["--scratch", a.scratch] if a.scratch else []
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    (sub / "scan.log").write_text(proc.stdout + proc.stderr)
    try:
        doc = json.loads((sub / "scan.json").read_text())
    except Exception as e:  # noqa: BLE001
        doc = {"ok": False, "cases": {}, "errors": {"scan": BG._error(e)}}
    doc["returncode"] = proc.returncode
    return doc


def scan_run(lo: int, hi: int, out: str, scratch: str | None) -> int:
    BG.SCAN_M = range(lo, hi + 1)
    return BG.scan_run(argparse.Namespace(only=PHASE, models=MODEL, out=out, scratch=scratch, served_default=True))


def split_at(routes: list[list[int | None]], M: int) -> int | None:
    (run,) = [r for r in routes if r[0] <= M <= r[1]]
    return run[2]


def launch_ms(routes: list[list[int | None]]) -> list[int]:
    """LAUNCH_M, the first M of each route class (a split count, or the serial chain), the first M past ROWS_ALL of each, and
    M_MAX."""
    first: dict[int | None, int] = {}
    past: dict[int | None, int] = {}
    for m0, m1, split in routes:
        first.setdefault(split, m0)
        if m1 > ROWS_ALL:
            past.setdefault(split, max(m0, ROWS_ALL + 1))
    return sorted(set(LAUNCH_M) | set(first.values()) | set(past.values()) | {M_MAX})


# ------------------------------------------------------------------------------------------------------------------------ launch
def row_spans(M: int) -> list[tuple[int, int]]:
    """Every row up to ROWS_ALL; past it WINDOW rows at the start, the middle and the end."""
    if M <= ROWS_ALL:
        spans = [(0, M)]
    else:
        mid = M // 2 - WINDOW // 2
        spans = [(0, WINDOW), (mid, mid + WINDOW), (M - WINDOW, M)]
    return [(r, min(r1, r + CHUNK)) for r0, r1 in spans for r in range(r0, r1, CHUNK)]


def jobs_of(job: dict[str, Any]) -> list[dict[str, Any]]:
    M = job["M"]
    ref = sorted({r for r in (0, 3, M - 1) if r < M})
    return [{**job, "rows": [r0, r1], "reference_rows": [r for r in ref if r0 <= r < r1]} for r0, r1 in row_spans(M)]


def keep_words(outdir: Path, job: dict[str, Any]) -> dict[str, Any]:
    """The compared rows' x and output, and their indices, under <out>/words/M<M>/; W once."""
    a = BG._load(job["paths"])
    rows = np.concatenate([np.arange(r0, r1) for r0, r1 in row_spans(job["M"])])
    d = outdir / "words"
    if not (d / "w.npy").exists():
        BG._save(d, w=np.asarray(a["w"]))
    BG._save(d / f"M{job['M']}", rows=rows, x=np.asarray(a["x"])[rows], out=np.asarray(a["out"])[rows])
    if job.get("probe_paths"):
        p = BG._load(job["probe_paths"])
        BG._save(d / f"M{job['M']}" / "splitk_probe", **{k: np.asarray(v) for k, v in p.items()})
    return {"M": job["M"], "rows": int(rows.size)}


def bind_check(splits: list[int]) -> dict[str, Any]:
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.defs import bind
    from verity.circuits.evaluation.batch import type_shape
    from verity.definitions.ops import gemm as CG
    from verity.definitions.ops.boolean import gemm as BGO
    from verity.definitions.registry import cublas_bf16 as CB

    hop = word_of(BGO.HopperBF16WgmmaDot16)
    targets = [bind(CG.Gemm, K=K, N=N, DOT=hop)] + [bind(CB.CublasSplitKGemmWord, K=K, N=N, SPLIT_K=s, DOT=hop) for s in splits]
    out = {}
    for t in targets:
        params, ret = t.signature()
        out[t.id] = [list(type_shape(ty)) for _, ty in params] + [list(type_shape(ret))] == [[K], [N, K], [N]]
    return {"bindings": out, "ok": bool(out) and all(out.values())}


# ---------------------------------------------------------------------------------------------------------------------------- run
def _case_ok(c: dict[str, Any]) -> bool:
    t = c["totals"]
    return bool(c["launch_ok"] and c["route_ok"] and t["words"] > 0 and t["mismatch"] == 0 and t["reference_coords"] > 0
                and t["twin_mismatch"] == 0 and c["hypothesis_ok"])


def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and not doc["errors"] and all(doc.get(k, {}).get("ok") is True for k in ("switch", "switch_after"))
                and isinstance(doc["switch"].get("cublas_version"), int) and doc["selection"].get("ok") is True
                and doc["scan"].get("ok") is True and doc["routes"] and doc["bind"].get("ok") is True
                and sorted(int(k[1:]) for k in doc["cases"]) == doc["expected"] and all(_case_ok(c) for c in doc["cases"].values()))


def run(a: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=a.scratch)
    work = Path(tmp.name)
    env, cleared = BG.environment(True, work / "triton_cache")
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "shape": {"K": K, "N": N},
                           "environment": G._environment(), "gate": C.gate(a.min_driver, a.want_cc, a.want_sms), "errors": {},
                           "launches": [], "selection": {}, "routes": [], "expected": [], "cases": {}}
    print(f"[glm53-mtp-eh-proj] gate {json.dumps(doc['gate'])} cleared {sorted(cleared)}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-MTP-EH-PROJ GATE-FAIL", flush=True)
        raise SystemExit(3)
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    from verity.definitions.registry import bf16_gemm_check as BC

    try:
        doc["switch"] = BG.switch(True)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["switch"] = BG._error(e)
        doc["switch"] = {"ok": False}
    if a.scans:
        scans = [json.loads((Path(a.scans) / f"{lo}-{hi}" / "scan.json").read_text()) for lo, hi in SCAN_CHUNKS]
    else:
        scans = [scan_chunk(a, outdir, lo, hi) for lo, hi in SCAN_CHUNKS]
    doc["scan"] = {"chunks": [{"m": [lo, hi], "ok": s.get("ok"), "returncode": s.get("returncode"), "errors": s.get("errors"),
                               "classes": {k: c.get("classes") for k, c in s.get("cases", {}).items()}}
                              for (lo, hi), s in zip(SCAN_CHUNKS, scans)]}
    try:
        table = BG.route_table(scans, "this run")
        doc["routes"] = table["routes"][BG.route_key(MODEL, PHASE, NAME)]
        doc["route_conditions"] = table["conditions"]
        doc["scan"]["ok"] = table["m_max"] == M_MAX
    except Exception as e:  # noqa: BLE001
        doc["errors"]["routes"] = BG._error(e)
        doc["scan"]["ok"] = False
    print(f"[glm53-mtp-eh-proj] scan ok={doc['scan']['ok']} routes ({len(doc['routes'])} runs) {json.dumps(doc['routes'])[:3000]}",
          flush=True)
    jobs: list[dict[str, Any]] = []
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        try:
            BG.distributed(work)
            gemm = dispatch_unquantized_gemm()
            doc["selection"] = {"linear_gemm": getattr(gemm, "__name__", None)}
            doc["selection"]["ok"] = doc["selection"]["linear_gemm"] == "default_unquantized_gemm"
        except Exception as e:  # noqa: BLE001
            doc["errors"]["selection"] = BG._error(e)
        weights = BG.Weights(work, 4000)
        doc["expected"] = launch_ms(doc["routes"]) if doc["routes"] else []
        for i, M in enumerate(doc["expected"]):
            try:
                split = split_at(doc["routes"], M)
                rec, job = BG.launch(work, weights, {}, MODEL, PHASE, NAME, M, 7000 + i, True, split)
                reduce = bool(rec["profile"].get("reduction_kernels"))
                rec["route_split"] = split
                rec["route_ok"] = reduce == (split is not None) and (split is None or rec["profile"].get("split_k") == split)
                doc["launches"].append(rec)
                jobs += jobs_of(job)
                doc.setdefault("kept", []).append(keep_words(outdir, job))
                print(f"[glm53-mtp-eh-proj] M={M:5d} split={split} reduce={reduce} route_ok={rec['route_ok']} "
                      f"det={rec['deterministic']} operands={rec['operands_ok']} profile={rec['profile'].get('ok')} "
                      f"kernels={[k['name'][:60] for k in rec['profile'].get('launches', [])]} us={rec['us']:.1f}", flush=True)
            except Exception as e:  # noqa: BLE001
                doc["errors"][f"M{M}"] = BG._error(e)
                print(f"[glm53-mtp-eh-proj] M={M} ERROR {doc['errors'][f'M{M}'][:400]}", flush=True)
        try:
            doc["switch_after"] = BG.configuration(True)
        except Exception as e:  # noqa: BLE001
            doc["errors"]["switch_after"] = BG._error(e)
            doc["switch_after"] = {"ok": False}
        del weights
        torch.cuda.empty_cache()
    splits = sorted({r[2] for r in doc["routes"] if r[2] is not None})
    try:
        doc["bind"] = bind_check(splits)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = BG._error(e)
        doc["bind"] = {"ok": False}
    print(f"[glm53-mtp-eh-proj] bind {json.dumps(doc['bind'])}; comparing {len(jobs)} jobs on {a.workers} workers", flush=True)
    for x in doc["launches"]:
        doc["cases"][f"M{x['M']}"] = {"split": x["route_split"], "launch_ok": BG._launch_ok(x), "route_ok": x["route_ok"],
                                      "totals": {k: 0 for k in TOTALS}, "first_mismatch": None, "hypothesis": None,
                                      "us": x.get("us"), "kernels": [k["name"] for k in x["profile"].get("launches", [])]}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(BG.compare, jobs):
            if "error" in res:
                doc["errors"][f"M{res['M']}/compare"] = res["error"]
                continue
            c = doc["cases"][f"M{res['M']}"]
            c["totals"]["rows"] += res["rows"][1] - res["rows"][0]
            for k in BG.TOTALS:
                c["totals"][k] += res[k]
            c["first_mismatch"] = c["first_mismatch"] or res["first_mismatch"]
            if res.get("hypothesis_mismatch"):
                fits = sorted(k for k, v in res["hypothesis_mismatch"].items() if v == 0)
                c["hypothesis"] = {"mismatch": res["hypothesis_mismatch"], "fits": fits, "ok": fits == [BC.CUBLAS_SPLITK_HYPOTHESIS]}
    for key, c in doc["cases"].items():
        c["hypothesis_ok"] = c["split"] is None or (c["hypothesis"] or {}).get("ok") is True
        c["ok"] = _case_ok(c)
        t = c["totals"]
        print(f"[glm53-mtp-eh-proj] case {key:6s} {'GO' if c['ok'] else 'NO-GO':5s} split={c['split']} rows={t['rows']} "
              f"words={t['words']} !={t['mismatch']} ref={t['reference_coords']} twin!={t['twin_mismatch']} "
              f"hypothesis={json.dumps(c['hypothesis'])} first={json.dumps(c['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(c["totals"][k] for c in doc["cases"].values()) for k in TOTALS}
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"GLM53-MTP-EH-PROJ {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} routes={json.dumps(doc['routes'])[:2000]} "
          f"totals={json.dumps(doc['totals'])} no-go={[k for k, c in doc['cases'].items() if not c['ok']]} "
          f"errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's MTP eh_proj (F.linear, K 12288, N 6144) on this Hopper GPU: its cuBLAS "
                                             "routes and words")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}, scan/ and words/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--scan-run", action="store_true", help="internal: one scan process over --scan-m, writing <out>/scan.json")
    ap.add_argument("--scan-m", default=f"1:{M_MAX}", help="LO:HI, with --scan-run")
    ap.add_argument("--scans", default=None, help="an earlier run's <out>/scan directory: its scans' routes, no scan run here")
    a = ap.parse_args(argv)
    if a.scan_run:
        lo, hi = (int(v) for v in a.scan_m.split(":"))
        return scan_run(lo, hi, a.out, a.scratch)
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
