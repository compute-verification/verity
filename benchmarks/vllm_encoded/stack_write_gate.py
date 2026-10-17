"""The stack's write gate (codec_tensor/write_gate.py's rule on the stack's own launches): under Nsight Compute, one
launch of each kernel a decode step of the encoded stack runs, from the stack's own code (`stack.py`, `paged.py`) on
synthetic data at Llama-3.1-8B's shapes, and per kernel the SMs' L2 write sectors (plus reductions) against what it must
write. A launch passes with no local stores and its writes within 1% (and a 4 KiB floor a kernel) of its outputs:

- each linear shape at m = 1 and 32, at the config the serve arm picked (`--stack-json`, its `selection`): the fused
  kernel's chunk partial sums (stored, nchunks · m · n · 4 bytes) and receipts (64 bytes a vector), then pg_reduce's
  bf16 output;
- the step's commit over 65,536 receipts (reader.cu's leaves, pads and levels): its digests;
- paged.cu's attention at batch 32, context 512 (its output rows and the sectors its partial records span), the
  deferred append of a token for each of 32 sequences at all 32 layers in one launch (its words), the gather of 64
  blocks (their words), and the lm_head's chunk decode (wt_decode, its plain rows).

With --ptxas FILE... it also reads the builds' ptxas reports: every instantiation of those kernels the stack launches
(the fused kernel at k with stored partials and two sketch warps) has a 0-byte stack frame and no spills ("STACK 0 /
LOCAL 0").

    python stack_write_gate.py --wt-lib wt.so --reader-lib reader.so --paged-lib paged.so --k 3 \\
        --stack-json OUT/arm-stack.json --ptxas OUT/wt.ptxas OUT/paged.ptxas OUT/reader.ptxas --out OUT
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "codec_tensor"))
sys.path.insert(0, str(HERE))

import write_gate as WG

KERNELS = "regex:fused_kernel|leaves_kernel|level_kernel|pads_kernel|pg_|decode_kernel"
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
MS = (1, 32)
LOCAL = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
RED = "lts__t_sectors_op_red.sum"


def launches(a) -> list[dict]:
    import paged as P
    import stack as S
    import torch

    st = S.Stack(a, encoded=True)
    sel = {}
    if a.stack_json:
        for r in json.loads(Path(a.stack_json).read_text())["install"]["selection"]:
            sel[(r["n"], r["k"], r["m"])] = tuple(r["best"])
    gen = torch.Generator(device="cuda").manual_seed(3)
    out: list[dict] = []
    cache: dict = {}
    for name, (n, kd) in SHAPES.items():
        w = (torch.randn((n, kd), generator=gen, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        lin = S.StackLinear(st, name, w, torch.ones(1, device="cuda"))
        del w
        for m in MS:
            cfg = sel.get((n, kd, m)) or lin.candidates(m)[0]
            plan = lin.planned(m, cfg, cache)
            x8 = S.rand_fp8(m, kd, gen)
            part = torch.empty((lin.nch, m, n), dtype=torch.float32, device="cuda")
            tags = lin.own_tags(plan)
            torch.cuda.synchronize()
            lin.launch(x8, part, plan, 0, tags)
            torch.cuda.synchronize()
            out.append({"kernel": "fused", "linear": name, "m": m, "cfg": list(cfg), "groups": plan[1],
                        "receipt_bytes": plan[3] * st.wt.tag_bytes, "partial_bytes": lin.nch * m * n * 4})
            lin.reduce(part, torch.ones(1, device="cuda"))
            torch.cuda.synchronize()
            out.append({"kernel": "reduce", "linear": name, "m": m, "out_bytes": m * n * 2})
        st.linears.pop()
        del lin
    n = 1 << 16
    st.nvec = {1: n}
    st.buffers()
    st.reader.begin(S.BW.RECEIPT_SEED, S.SR.rb.Plan.whole(n), S.BW.STORE_KEY)
    st.tags.random_()
    so, stream = st.reader.so, S.stream()
    nl = -(-n // S.SR.rb.PER_LEAF)
    width = 1 << max(0, (nl - 1).bit_length())
    so.rb_leaves(st.tags.data_ptr(), 0, nl, st.dig.data_ptr(), None, stream)
    if width > nl:
        so.rb_pads(nl, width - nl, st.dig.data_ptr() + 64 * nl, stream)
    if width > 1:
        so.rb_tree(st.dig.data_ptr(), width, 0, st.up.data_ptr(), stream)
    torch.cuda.synchronize()
    out.append({"kernel": "commit", "receipts": n, "leaves": nl, "digest_bytes": 64 * width + 64 * (width - 1)})
    lib = st.plib
    sh = lib.shape
    G, D, H, bs = sh["kv_heads"], sh["head_dim"], sh["heads"], 16
    b, ctx, layers = 32, 512, 32
    nb = b * ctx // bs
    kvc = torch.empty((nb + 64, G, bs, 2 * D), dtype=torch.bfloat16, device="cuda")
    kvc.view(torch.int16).random_()
    lay = P.layout(kvc, G, sh["wb"])
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(b, ctx // bs)
    lens = torch.full((b,), ctx, dtype=torch.int32, device="cuda")
    q = torch.randn((b, H * D), device="cuda").to(torch.bfloat16)
    rows = torch.randn((b, G * D), device="cuda").to(torch.bfloat16)
    o = torch.empty_like(q)
    sp = P.splits_for(b, st.sms, G)
    part = torch.empty(P.part_floats(lib, b, sp), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    P.attn(lib, torch, kvc, lay, 0, layers, 0, 1, st.k, q, o, part, lens, bt, b, sp, ck=rows, cv=rows)
    torch.cuda.synchronize()
    records = part.numel() // sh["part"]
    out.append({"kernel": "attention", "batch": b, "ctx": ctx, "splits": sp, "out_bytes": o.numel() * 2,
                "part_bytes": part.numel() * 4, "part_sector_bytes": records * (4 * H * D + 2 * G * 32)})
    slots = torch.arange(b, dtype=torch.int64, device="cuda") * ctx + ctx - 1
    tab = P.table(torch, [(rows, rows, slots, b, kvc, 0, layers, l) for l in range(layers)]).cuda()
    torch.cuda.synchronize()
    P.append_multi(lib, torch, tab, layers, b, lay, 1, st.k)
    torch.cuda.synchronize()
    out.append({"kernel": "append", "rows": b, "layers": layers, "word_bytes": layers * b * G * 4 * D})
    blocks = torch.arange(64, dtype=torch.int32, device="cuda") * 7
    scratch = torch.empty((64, G, bs, 2 * D), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    P.gather(lib, torch, kvc, lay, 0, layers, 0, 1, st.k, blocks, scratch)
    torch.cuda.synchronize()
    out.append({"kernel": "gather", "blocks": 64, "word_bytes": scratch.numel() * 2})
    rows_lm, h = 4096, 4096
    store = torch.empty(rows_lm * h, dtype=torch.int16, device="cuda").random_()
    plain = torch.empty(rows_lm * h, dtype=torch.int16, device="cuda")
    torch.cuda.synchronize()
    st.wt.call("wt_decode", store.data_ptr(), plain.data_ptr(), rows_lm * h * 2 // st.wt.s, 0, st.k,
               st.xr.data_ptr(), 8 * st.sms, S.stream())
    torch.cuda.synchronize()
    out.append({"kernel": "decode", "rows": rows_lm, "out_bytes": rows_lm * h * 2})
    return out


def summarize(stdout: str, rc: int) -> dict:
    want = next((json.loads(line[len("WRITE-GATE "):]) for line in stdout.splitlines()
                 if line.startswith("WRITE-GATE ")), None)
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO("\n".join(line for line in stdout.splitlines() if line.startswith('"')))))
    per: dict[str, dict] = {}
    for r in rows:
        v = r["Metric Value"].replace(",", "")
        per.setdefault(r["ID"], {"name": r["Kernel Name"][:90]})[r["Metric Name"]] = None if v == "n/a" else float(v)
    kernels = list(per.values())
    res = {"rc": rc, "metrics": WG.METRICS, "launches": want, "kernels": kernels, "rows": []}
    if want is None:
        res["ok"] = False
        return res

    def b_(kv, metric):
        return 32 * (kv.get(metric) or 0)

    def check(w, ks, must, extra=None):
        wb = sum(b_(kv, WG.WRITE) + b_(kv, RED) for kv in ks)
        lb = sum(b_(kv, LOCAL) for kv in ks)
        row = {**w, "kernels": [kv["name"][:48] for kv in ks], "l2_write_bytes": wb,
               "l2_write_all_bytes": sum(b_(kv, WG.ALL) for kv in ks), "local_store_bytes": lb, "must_bytes": must,
               "over_must": wb / must if must else None, **(extra or {}),
               "ok": bool(ks) and lb == 0 and wb <= 1.01 * must + 4096 * len(ks)}
        res["rows"].append(row)

    queue = list(kernels)

    def take(pat: str, n: int = 1) -> list[dict]:
        got = []
        while queue and len(got) < n and re.search(pat, queue[0]["name"]):
            got.append(queue.pop(0))
        return got

    for w in want:
        kind = w["kernel"]
        if kind == "fused":
            check(w, take("fused_kernel"), w["receipt_bytes"] + w["partial_bytes"])
        elif kind == "reduce":
            check(w, take("pg_reduce"), w["out_bytes"])
        elif kind == "commit":
            ks = []
            while queue and re.search("leaves_kernel|pads_kernel|level_kernel", queue[0]["name"]):
                ks.append(queue.pop(0))
            check(w, ks, w["digest_bytes"])
        elif kind == "attention":
            check(w, take("pg_attn_kernel") + take("pg_combine"), w["out_bytes"] + w["part_sector_bytes"])
        elif kind == "append":
            check(w, take("pg_append_multi"), w["word_bytes"])
        elif kind == "gather":
            check(w, take("pg_gather"), w["word_bytes"])
        elif kind == "decode":
            check(w, take("decode_kernel"), w["out_bytes"])
    res["unmatched_kernels"] = [kv["name"] for kv in queue]
    res["ok"] = bool(res["rows"]) and all(r["ok"] for r in res["rows"]) and not queue
    return res


def ptxas(files: list[str], k: int) -> dict:
    """Each launched kernel's stack frame and spills, from ptxas -v (names demangled)."""
    text = "\n".join(Path(f).read_text() for f in files)
    names = sorted(set(re.findall(r"Compiling entry function '([^']+)'|Function properties for (\S+)", text)))
    mangled = sorted({a or b for a, b in names})
    filt = shutil.which("c++filt")
    dem = dict(zip(mangled, subprocess.run([filt], input="\n".join(mangled), capture_output=True, text=True,
                                           check=False).stdout.splitlines())) if filt else {m: m for m in mangled}
    props = {}
    for m, body in re.findall(r"Function properties for (\S+)\n\s*(\d+ bytes stack frame, \d+ bytes spill stores, "
                              r"\d+ bytes spill loads)", text):
        f, s_, l_ = (int(x) for x in re.findall(r"(\d+) bytes", body))
        props[m] = {"stack_frame": f, "spill_stores": s_, "spill_loads": l_}
    used = re.compile(rf"wt::fused_kernel<{k}, \d, true, 16, 2|wt::decode_kernel<{k},|kv::pg_|rb::(leaves|level|pads)")
    rows = [{"kernel": dem.get(m, m)[:120], **p} for m, p in props.items() if used.search(dem.get(m, m))]
    return {"kernels": rows, "ok": bool(rows) and all(r["stack_frame"] == 0 and r["spill_stores"] == 0 and
                                                      r["spill_loads"] == 0 for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt-lib", required=True)
    ap.add_argument("--reader-lib", required=True)
    ap.add_argument("--paged-lib", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--stack-json", default="")
    ap.add_argument("--ptxas", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        print("WRITE-GATE " + json.dumps(launches(a)), flush=True)
        return 0
    out = Path(a.out)
    ncu = shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    cmd = [ncu, "--csv", "--print-units", "base", "--target-processes", "all", "--kernel-name", KERNELS, "--metrics",
           ",".join(WG.METRICS), sys.executable, __file__, "--child", "--wt-lib", a.wt_lib, "--reader-lib",
           a.reader_lib, "--paged-lib", a.paged_lib, "--k", str(a.k)] + (
               ["--stack-json", a.stack_json] if a.stack_json else [])
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out / "stack-write-gate.ncu.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    res = summarize(p.stdout, p.returncode)
    if a.ptxas:
        res["ptxas"] = ptxas(a.ptxas, a.k)
        res["ok"] = res["ok"] and res["ptxas"]["ok"]
    (out / "stack-write-gate.json").write_text(json.dumps(res, indent=1))
    for r in res.get("rows", []):
        print(json.dumps({k: r[k] for k in ("kernel", "l2_write_bytes", "must_bytes", "over_must", "local_store_bytes",
                                            "ok") if k in r} | {k: r[k] for k in ("linear", "m") if k in r}))
    if "ptxas" in res:
        bad = [r for r in res["ptxas"]["kernels"] if r["stack_frame"] or r["spill_stores"] or r["spill_loads"]]
        print(f"ptxas: {len(res['ptxas']['kernels'])} kernels, {len(bad)} with a stack frame or spills: {bad[:6]}")
    print("unmatched:", res.get("unmatched_kernels"), "rc", p.returncode)
    print("stack write gate", "passed" if res.get("ok") else "FAILED")
    return 0 if res.get("ok") else 7


if __name__ == "__main__":
    sys.exit(main())
