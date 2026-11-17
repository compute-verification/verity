"""The write gate's reading of an Nsight Compute profile and its rule, which the stack's gate
(`benchmarks/vllm_encoded/stack_write_gate.py`) runs on its own launches: per kernel the L2 write sectors from the SMs
against what it must write (all sources' are recorded beside them: the profiler's own writes come from other units, 15
sectors a launch and up to 2 MB at a KV launch). A sketched launch writes its receipts (`tag_bytes` a vector)
and adds its partial sums into y (L2 reductions, nchunks · M · N · 4 bytes); an untagged one only the adds; anything
past that (a spill, a stack, a stray store) shows as write sectors over the receipts' bytes and as local stores. A
launch passes with no local stores and its L2 writes and reductions within 1% (and a 4 KiB floor) of its outputs plus
partials; a commit (the pass's tree constants from its seed on the device, `chain_kernel`, then the leaves, pads and
tree) with none and its writes within 1% of its constants and digests.

The same for the KV side: attention, whose outputs are o and its split partials, and the append of a token, whose
outputs are its words. A partial record's m and l are each a group's GROUP floats from its own warp, half a sector at
GROUP = 4 (8B), so attention is held to the sectors its records span (`part_sector_bytes`) and its ratio to the bytes
is recorded beside it. The receipt form (kv_attn form 13) also stores each token-layer's head shares
(64 bytes a slot and KV head, whole sectors), and the KV's receipts kernel after it (`kv_rc_receipts`) its 64-byte
receipts, whole sectors a lane pair, held to 1% (and a 4 KiB floor) of them.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess

METRICS = ("lts__t_sectors_op_write.sum", "lts__t_sectors_srcunit_tex_op_write.sum", "lts__t_sectors_op_red.sum",
           "lts__t_sectors_op_atom.sum", "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", "dram__bytes_write.sum")
WRITE, ALL = METRICS[1], METRICS[0]   # the SMs' L2 writes, and every source's
RED, LOCAL = METRICS[2], METRICS[4]   # L2 reductions, and local stores
MARK = "WRITE-GATE "                  # starts the child's line of its launches


def profile(kernels: str, child: list[str]) -> subprocess.CompletedProcess:
    """`child`, a command line that prints its launches after MARK, under Nsight Compute: METRICS of each kernel whose
    name matches `kernels`, as CSV on its stdout."""
    ncu = shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    return subprocess.run([ncu, "--csv", "--print-units", "base", "--target-processes", "all", "--kernel-name", kernels,
                           "--metrics", ",".join(METRICS), *child], capture_output=True, text=True, check=False)


def profiled(stdout: str, name_chars: int = 80) -> tuple[list[dict] | None, list[dict]]:
    """A profile's launches (None if the child printed none) and each kernel's counters in launch order, its name cut to
    `name_chars` (a metric the device doesn't expose, such as sm_120's DRAM writes, reads None)."""
    want = next((json.loads(line[len(MARK):]) for line in stdout.splitlines() if line.startswith(MARK)), None)
    rows = csv.DictReader(io.StringIO("\n".join(line for line in stdout.splitlines() if line.startswith('"'))))
    per: dict[str, dict] = {}
    for r in rows:
        v = r["Metric Value"].replace(",", "")
        per.setdefault(r["ID"], {"name": r["Kernel Name"][:name_chars]})[r["Metric Name"]] = (
            None if v == "n/a" else float(v))
    return want, list(per.values())


def sector_bytes(kv: dict, metric: str) -> float:
    """A kernel's sectors of `metric` as bytes (0 where the device doesn't expose it)."""
    return 32 * (kv.get(metric) or 0)


def passes(written: float, local: float, must: int, kernels: int = 1) -> bool:
    """The gate's rule: no local stores, and L2 writes within 1% of what the kernels must write and 4 KiB a kernel."""
    return local == 0 and written <= 1.01 * must + 4096 * kernels


def summarize(stdout: str, rc: int) -> dict:
    """Per kernel the counters (`profiled`), and each launch's L2 writes against what it must write."""
    want, kernels = profiled(stdout)
    res = {"rc": rc, "metrics": METRICS, "launches": want, "kernels": kernels}
    fused = [kv for kv in kernels if "fused_kernel" in kv["name"]]
    if want is None or len(fused) != sum(w["kernel"] == "fused" for w in want):
        res["ok"] = False
        return res
    rows_out = []
    for w, kv in zip([w for w in want if w["kernel"] == "fused"], fused):
        wb, lb = sector_bytes(kv, WRITE), sector_bytes(kv, LOCAL)
        rb, must = sector_bytes(kv, RED), w["receipt_bytes"] + w["red_bytes"]
        rows_out.append({**w, "l2_write_bytes": wb, "l2_write_all_bytes": sector_bytes(kv, ALL), "l2_red_bytes": rb,
                         "local_store_bytes": lb, "dram_write_bytes": kv.get("dram__bytes_write.sum"),
                         "write_over_receipts": wb / w["receipt_bytes"] if w["receipt_bytes"] else None,
                         "red_over_partials": rb / w["red_bytes"], "over_outputs_and_partials": (wb + rb) / must,
                         "ok": passes(wb + rb, lb, must)})
    commit = [w for w in want if w["kernel"] == "commit"]
    if commit:
        ck = [kv for kv in kernels
              if any(s in kv["name"] for s in ("chain_kernel", "leaves_kernel", "level_kernel", "pads_kernel"))]
        must = commit[0]["leaf_bytes"] + commit[0]["node_bytes"] + commit[0].get("const_bytes", 0)
        wb = sum(sector_bytes(kv, WRITE) for kv in ck)
        lb = sum(sector_bytes(kv, LOCAL) for kv in ck)
        rows_out.append({**commit[0], "kernels": len(ck), "l2_write_bytes": wb,
                         "l2_write_all_bytes": sum(sector_bytes(kv, ALL) for kv in ck), "local_store_bytes": lb,
                         "write_over_digests": wb / must, "ok": passes(wb, lb, must, len(ck))})
    attn = [w for w in want if w["kernel"] == "attention"]
    groups: list[list[dict]] = []
    for kv in kernels:
        if "attn_" in kv["name"]:
            if "attn_combine" in kv["name"] and groups:
                groups[-1].append(kv)
            else:
                groups.append([kv])
    if len(groups) != len(attn):
        res["ok"] = False
        res["rows"] = rows_out
        return res
    for w, g in zip(attn, groups):
        wb = sum(sector_bytes(kv, WRITE) + sector_bytes(kv, RED) for kv in g)
        lb = sum(sector_bytes(kv, LOCAL) for kv in g)
        must = w["out_bytes"] + w["part_sector_bytes"] + w.get("share_bytes", 0)
        rows_out.append({**w, "kernels": [kv["name"][:40] for kv in g], "l2_write_bytes": wb,
                         "l2_write_all_bytes": sum(sector_bytes(kv, ALL) for kv in g), "local_store_bytes": lb,
                         "write_over_outputs_and_partials": wb / (w["out_bytes"] + w["part_bytes"]),
                         "write_over_sectors": wb / must, "ok": passes(wb, lb, must, len(g))})
    rcw = [w for w in want if w["kernel"] == "kv_receipts"]
    rck = [kv for kv in kernels if "rc_receipt_kernel" in kv["name"]]
    if len(rck) != len(rcw):
        res["ok"] = False
        res["rows"] = rows_out
        return res
    for w, kv in zip(rcw, rck):
        wb, lb = sector_bytes(kv, WRITE), sector_bytes(kv, LOCAL)
        rows_out.append({**w, "l2_write_bytes": wb, "l2_write_all_bytes": sector_bytes(kv, ALL), "local_store_bytes": lb,
                         "write_over_receipts": wb / w["receipt_bytes"],
                         "ok": passes(wb, lb, w["receipt_bytes"])})
    app = [w for w in want if w["kernel"] == "append"]
    ak = [kv for kv in kernels if "append_kernel" in kv["name"]]
    if len(ak) != len(app):
        res["ok"] = False
        res["rows"] = rows_out
        return res
    for w, kv in zip(app, ak):
        wb, lb = sector_bytes(kv, WRITE), sector_bytes(kv, LOCAL)
        rows_out.append({**w, "l2_write_bytes": wb, "l2_write_all_bytes": sector_bytes(kv, ALL), "local_store_bytes": lb,
                         "write_over_words": wb / w["word_bytes"],
                         "ok": passes(wb, lb, w["word_bytes"])})
    res["rows"] = rows_out
    res["ok"] = all(r["ok"] for r in rows_out)
    return res
