"""The write gate (21:50Z steering) for the step's weight GEMM: under Nsight Compute, one launch of each linear's fused
kernel at the configs the step picked (`step.json`), sketched and untagged, and the reader's commit kernels, and per
kernel the L2 write sectors against what it must write. A sketched launch writes its receipts (`tag_bytes` a vector)
and adds its partial sums into y (L2 reductions, nchunks · M · N · 4 bytes); an untagged one only the adds; anything
past that (a spill, a stack, a stray store) shows as write sectors over the receipts' bytes and as local stores. A
launch passes with no local stores and its L2 writes and reductions within 1% (and a 4 KiB floor) of its outputs plus
partials; a commit with none and its writes within 1% of its digests.

    python write_gate.py --wt-lib wt.so [--reader-lib reader.so] --step-json OUT/step.json --out OUT
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench as BW  # noqa: E402
import sketching as SKT  # noqa: E402
import wide as W  # noqa: E402

METRICS = ("lts__t_sectors_op_write.sum", "lts__t_sectors_srcunit_tex_op_write.sum", "lts__t_sectors_op_red.sum",
           "lts__t_sectors_op_atom.sum", "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", "dram__bytes_write.sum")
KERNELS = "regex:fused_kernel|leaves_kernel|level_kernel|pads_kernel"


def launches(a) -> list[dict]:
    """The child: each (linear, k, variant) launched once, in order; returns what each must write."""
    import torch

    step = json.loads(Path(a.step_json).read_text())
    lib = BW.Lib(a.wt_lib)
    t, m = step["args"]["sketch_t"], step["gemm_m"]
    _, shapes = BW.MODELS[step["model_name"]]
    shapes = {**shapes, "lm_head": (128256, 4096 if step["model_name"] == "8b" else 8192)}
    st = BW.stream(torch)
    BW.set_sketch_tag(lib)
    bits = BW.sketch_bits(lib, BW.window_seed(lib), 16, SKT.vector_bytes(lib.sb, t))
    signs = torch.from_numpy(SKT.sign_fragments(bits, SKT.steps(lib.sb), t).view(np.int32)).cuda()
    frags = BW.sign_frags(lib, torch, signs, t, st)
    out = []
    for name in a.linears.split(","):
        n, kd = shapes[name]
        store = torch.empty(n * kd, dtype=torch.uint8, device="cuda").random_(0, 256)
        x = torch.randn(m, kd, device="cuda").to(torch.float8_e4m3fn)
        y = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        for k in step["ks"]:
            key = W.key(lib.field.name, BW.SEED, k)
            BW.set_key(lib, torch, key)
            for var in ("sketch", "untagged"):
                cfg = tuple(step["picks"][name]["k"][str(k)][var])
                launched = ctypes.c_int(0)
                lib.call("wt_fused_sketch", k, None, None, None, m, n, kd, 0, 0, cfg[0], cfg[1], 0, 0, None, None, 0,
                         cfg[3], None, None, t, BW.sketch_warps(cfg), ctypes.byref(launched), st)
                gr = launched.value // (kd // 512)
                steps = BW.tsteps(lib, torch, key, cfg[0], gr, kd // 512)
                nvec = lib.so.wt_sketch_vectors(n, kd, gr, t) if cfg[3] else 0
                tags = torch.empty(max(nvec, 1) * lib.tag_bytes, dtype=torch.uint8, device="cuda")
                sg = (frags if BW.sketch_warps(cfg) else signs) if cfg[3] else None
                lib.call("wt_fused_sketch", k, BW.ptr(store), BW.ptr(x.view(torch.uint8)), BW.ptr(y), m, n, kd, 0, 0,
                         cfg[0], cfg[1], gr, 0, BW.ptr(steps), BW.ptr(tags) if cfg[3] else None, lib.jbase, cfg[3],
                         BW.ptr(sg) if sg is not None else None, None, t, BW.sketch_warps(cfg), None, st)
                torch.cuda.synchronize()
                out.append({"kernel": "fused", "linear": name, "k": k, "variant": var, "cfg": list(cfg), "groups": gr,
                            "receipt_bytes": nvec * lib.tag_bytes, "red_bytes": (kd // 512) * m * n * 4})
        del store
    if a.reader_lib and lib.receipt:
        import reader as RD

        rd = RD.Reader(a.reader_lib, torch)
        n = 1 << 16
        plan = BW.SR.rb.Plan.whole(n)
        rd.begin(BW.RECEIPT_SEED, plan, BW.STORE_KEY)
        ys = torch.empty((n, 8), dtype=torch.int64, device="cuda").random_()
        rd.levels(rd.commit(ys, 0, rd.leaves), 0, pad=True)
        torch.cuda.synchronize()
        out.append({"kernel": "commit", "blocks": n, "leaves": rd.leaves, "leaf_bytes": 64 * rd.leaves,
                    "node_bytes": 64 * (rd.leaves - 1)})
    return out


def summarize(stdout: str, rc: int) -> dict:
    """Per kernel the counters (a metric the device doesn't expose, such as sm_120's DRAM writes, reads None), and
    each launch's L2 writes against what it must write."""
    want = next((json.loads(line[len("WRITE-GATE "):]) for line in stdout.splitlines()
                 if line.startswith("WRITE-GATE ")), None)
    rows = list(csv.DictReader(io.StringIO("\n".join(line for line in stdout.splitlines() if line.startswith('"')))))
    per: dict[str, dict] = {}
    for r in rows:
        v = r["Metric Value"].replace(",", "")
        per.setdefault(r["ID"], {"name": r["Kernel Name"][:80]})[r["Metric Name"]] = (
            None if v == "n/a" else float(v))
    kernels = list(per.values())

    def bytes_of(kv, metric):
        return 32 * (kv.get(metric) or 0)

    res = {"rc": rc, "metrics": METRICS, "launches": want, "kernels": kernels}
    fused = [kv for kv in kernels if "fused_kernel" in kv["name"]]
    if want is None or len(fused) != sum(w["kernel"] == "fused" for w in want):
        res["ok"] = False
        return res
    rows_out = []
    for w, kv in zip([w for w in want if w["kernel"] == "fused"], fused):
        wb, lb = bytes_of(kv, "lts__t_sectors_op_write.sum"), bytes_of(kv, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")
        rb, must = bytes_of(kv, "lts__t_sectors_op_red.sum"), w["receipt_bytes"] + w["red_bytes"]
        rows_out.append({**w, "l2_write_bytes": wb, "l2_write_sm_bytes": bytes_of(kv, METRICS[1]), "l2_red_bytes": rb,
                         "local_store_bytes": lb, "dram_write_bytes": kv.get("dram__bytes_write.sum"),
                         "write_over_receipts": wb / w["receipt_bytes"] if w["receipt_bytes"] else None,
                         "red_over_partials": rb / w["red_bytes"], "over_outputs_and_partials": (wb + rb) / must,
                         "ok": lb == 0 and wb + rb <= 1.01 * must + 4096})
    commit = [w for w in want if w["kernel"] == "commit"]
    if commit:
        ck = [kv for kv in kernels if any(s in kv["name"] for s in ("leaves_kernel", "level_kernel", "pads_kernel"))]
        must = commit[0]["leaf_bytes"] + commit[0]["node_bytes"]
        wb = sum(bytes_of(kv, "lts__t_sectors_op_write.sum") for kv in ck)
        lb = sum(bytes_of(kv, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum") for kv in ck)
        rows_out.append({**commit[0], "kernels": len(ck), "l2_write_bytes": wb, "local_store_bytes": lb,
                         "write_over_digests": wb / must, "ok": lb == 0 and wb <= 1.01 * must + 4096 * len(ck)})
    res["rows"] = rows_out
    res["ok"] = all(r["ok"] for r in rows_out)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt-lib", default="")
    ap.add_argument("--reader-lib", default="")
    ap.add_argument("--step-json", default="")
    ap.add_argument("--linears", default="qkv,down")
    ap.add_argument("--out", default="")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--parse", default="", help="summarize a captured write-gate.ncu.txt instead of profiling")
    a = ap.parse_args()
    if a.child:
        print("WRITE-GATE " + json.dumps(launches(a)), flush=True)
        return 0
    out = Path(a.out)
    if a.parse:
        res = summarize(Path(a.parse).read_text().split("\n--- stderr ---\n")[0], 0)
    else:
        ncu = shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
        cmd = [ncu, "--csv", "--print-units", "base", "--target-processes", "all", "--kernel-name", KERNELS,
               "--metrics", ",".join(METRICS),
               sys.executable, __file__, "--child", "--wt-lib", a.wt_lib, "--step-json", a.step_json, "--linears",
               a.linears] + (["--reader-lib", a.reader_lib] if a.reader_lib else [])
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        (out / "write-gate.ncu.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        res = summarize(p.stdout, p.returncode)
    (out / "write-gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("rows", res), indent=1)[:8000])
    return 0 if res.get("ok") else 7


if __name__ == "__main__":
    sys.exit(main())
