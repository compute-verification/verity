#!/usr/bin/env python3
r"""On one GPU: the fallback commit call's roots (`commit_fb.commit_step` on CUDA tensors, so `row_tables_device.step_roots`)
equal the CPU references on random unit outputs at shape H's row widths.

    bash benchmarks/torch_gpu/commit_fb/run.sh check        # under gpu-lease, on the H200 (run.sh says how)

Cases (random values, seeded; row ids shuffled requests, or the default slots):

  layer+head   one layer's 16 tables and the head's 4 at B = 37, shuffled rows
  two-layers   two layers' 32 tables at B = 5, shuffled rows
  one-row      one layer at B = 1 (a one-leaf table)
  wide         one layer at B = 1,024, slots (688 MB of rows)

Each case's device root and table roots are held to `verity_vllm.commit.row_tables.step_roots` (vLLM's Python reference,
which the kernel was written against) and to `row_tables_device.host_step` (the kernel's C++ twin); the small cases also to
`commit_fb.reference` (`verity.ml.commitments.row_table.commit_step`, independent of vLLM's code), and their openings
(`commit_fb.open_rows`) verify against the device's root with `row_table.verify_step`.  Writes ``roots.json`` into ``--out``
(default ``$OUT``) and ``result.json`` into the run dir; exit 1 on any inequality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import commit_fb as F  # noqa: E402
from verity.ml.commitments import row_table as RT  # noqa: E402
from verity_vllm.commit import row_tables as RTB  # noqa: E402
from verity_vllm.commit import row_tables_device as RTD  # noqa: E402

RESULT_SCHEMA = "research/result/v0.1"
CASES = {"layer+head": (37, 1, True, True, True), "two-layers": (5, 2, False, True, True), "one-row": (1, 1, False, False, True),
         "wide": (1024, 1, False, False, False)}


def unit_outputs(rng: np.random.Generator, batch: int, layers: int, head: bool) -> dict:
    """Host unit outputs at shape H's widths: quantizers as (codes, scales), bf16 rows as u16, the argmax as i32."""
    spec = F.shape_h()
    parts = [(f"model.layers.{l}.", spec["layer"]) for l in range(layers)] + ([("model.head.", spec["head"])] if head else [])
    out = {}
    for prefix, part in parts:
        for name, (kind, n) in part.items():
            if kind == "s8":
                out[prefix + name] = (rng.integers(-128, 128, (batch, n), dtype=np.int8), rng.standard_normal(batch).astype(np.float32))
            elif kind == "bf16":
                out[prefix + name] = rng.integers(0, 1 << 16, (batch, n), dtype=np.uint16)
            else:
                out[prefix + name] = rng.integers(0, 256, (batch,), dtype=np.int32)
    return out


def to_device(outs: dict) -> dict:
    """The same values as CUDA tensors: bf16 tables as torch.bfloat16 (the wrapper commits their bytes)."""
    dev = {}
    for op, v in outs.items():
        if isinstance(v, tuple):
            dev[op] = (torch.from_numpy(v[0]).cuda(), torch.from_numpy(v[1]).cuda())
        elif v.dtype == np.uint16:
            dev[op] = torch.from_numpy(v.view(np.int16)).cuda().view(torch.bfloat16)
        else:
            dev[op] = torch.from_numpy(v).cuda()
    return dev


def host_outputs(outs: dict, rows: list) -> dict:
    """``row_tables.step_roots``' input: each table's (rows, codes, scales), a plain tensor's scale +0.0f."""
    res = {}
    for op, v in outs.items():
        if isinstance(v, tuple):
            res[op] = (rows, v[0], v[1])
        else:
            res[op] = (rows, np.ascontiguousarray(v).reshape(v.shape[0], -1).view(np.int8), np.zeros(v.shape[0], np.float32))
    return res


def run_case(name: str, batch: int, layers: int, head: bool, shuffled: bool, small: bool, salt: F.SaltSource) -> dict:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little"))
    outs = unit_outputs(rng, batch, layers, head)
    rows = [(f"req-{k}", int(t)) for k, t in zip(rng.permutation(batch), rng.integers(0, 31, batch), strict=True)] \
        if shuffled else F.slot_rows(batch)
    t0 = time.perf_counter()
    got = F.commit_step(to_device(outs), salt, rows=rows)
    t1 = time.perf_counter()
    host = host_outputs(outs, rows)
    vllm = RTB.step_roots(salt.context, salt.segment, host, salt.key, workers=min(16, os.cpu_count() or 1))
    twin = RTD.host_step(salt.context, salt.segment, host, salt.key)
    t2 = time.perf_counter()
    eq = {"vllm_reference": got.root == vllm.root and got.tables == vllm.tables,
          "cpp_twin": got.root == twin.root and got.tables == twin.tables}
    if small:
        ref = F.reference(outs, salt, rows=rows)
        eq["row_table_reference"] = got.root == ref.root and [r for _o, r in got.tables] == [t.root for t in ref.tables]
        opened = []
        for op in [got.tables[0][0], got.tables[len(got.tables) // 2][0], got.tables[-1][0]]:
            pick = sorted({0, batch // 2, batch - 1})
            o, w = F.open_rows(got, op, outs[op], salt, pick)
            opened.append(RT.verify_step(got.root, salt.context, salt.segment, len(got.tables), batch, o, w))
        eq["openings_verify"] = all(opened)
    nbytes = sum((v[0].nbytes + v[1].nbytes) if isinstance(v, tuple) else (v.nbytes + 4 * batch) for v in outs.values())
    return {"case": name, "batch": batch, "layers": layers, "head": head, "rows": "shuffled" if shuffled else "slots",
            "tables": len(got.tables), "bytes": nbytes, "root": got.root.hex(), "equal": eq, "ok": all(eq.values()),
            "device_s": round(t1 - t0, 3), "cpu_s": round(t2 - t1, 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.environ.get("OUT"))
    ap.add_argument("--cases", default=",".join(CASES))
    ns = ap.parse_args(argv)
    started = time.perf_counter()
    salt = F.SaltSource(hashlib.sha256(b"commit-fb/row-key").digest(), F.segment_of(7, 3, 53),
                        hashlib.sha512(b"commit-fb/context").digest())
    results = []
    for name in ns.cases.split(","):
        r = run_case(name, *CASES[name], salt)
        print(json.dumps(r), flush=True)
        results.append(r)
    ok = all(r["ok"] for r in results)
    summary = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "torch": torch.__version__, "cuda": torch.version.cuda, "segment": salt.segment, "cases": results, "ok": ok}
    print(json.dumps({"ok": ok, "cases": {r["case"]: r["equal"] for r in results}}), flush=True)
    if ns.out:
        d = Path(ns.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "roots.json").write_text(json.dumps(summary, indent=1) + "\n")
        result = {
            "schema": RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "commit_fb/check_roots", "cases": CASES, "gpu": summary["gpu"],
                                     "implementation": "commit_fb.commit_step -> verity_vllm.commit.row_tables_device.step_roots"},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": "device roots equal row_tables.step_roots, host_step and (small cases) "
                                      "row_table.commit_step; openings verify against the device root",
                           "detail": json.dumps({r["case"]: r["equal"] for r in results}), "evidence": ["roots.json"]},
            "measurements": [{"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["out/roots.json"], "artifacts": []}
        write_result(result, d)
    return 0 if ok else 1


def write_result(result: dict, out: Path) -> None:
    """``result.json`` in the run dir (``$RESEARCH_RUN_DIR``), else beside the outputs."""
    d = Path(os.environ.get("RESEARCH_RUN_DIR") or out)
    tmp = d / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, d / "result.json")


if __name__ == "__main__":
    raise SystemExit(main())
