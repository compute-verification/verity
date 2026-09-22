"""Diagnose the mismatching rows of an attn_check report: which stage is off?

    PYTHONPATH=. python -m verity_vllm_numerics.attn_diagnose FIXTURE_DIR CHECK.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import fa2_model as M


def f32(u: int) -> np.float32:
    return np.array([u], np.uint32).view(np.float32)[0]


def u32(f) -> int:
    return int(np.array([f], np.float32).view(np.uint32)[0])


def bf16(word32: int) -> int:
    lsb = (word32 >> 16) & 1
    return (word32 + 0x7FFF + lsb) >> 16


def main() -> None:
    fx, rep_path = sys.argv[1], sys.argv[2]
    idx = json.load(open(f"{fx}/index.json"))
    consts = idx["consts"]
    H, KVH = consts["num_heads"], consts["num_kv_heads"]
    scale = consts["softmax_scale"]
    tabs = M.MufuTables.load(f"{fx}/mufu")
    rep = json.load(open(rep_path))
    bad = rep["first_mismatches"]
    calls = {c["seq"]: c for c in idx["calls"]}
    print(len(bad), [(b["seq"], b["phase"], b["layer"], b["Lk"], b["row"], b["head"], b["d"]) for b in bad])
    for b in bad:
        c = calls[b["seq"]]
        q = np.load(f"{fx}/{c['q_file']}")
        gpu = np.load(f"{fx}/{c['out_file']}")
        k = np.load(f"{fx}/{c['k_file']}")
        v = np.load(f"{fx}/{c['v_file']}")
        rows = np.array(c["rows"])
        ri = int(np.where(rows == b["row"])[0][0])
        h, d = b["head"], b["d"]
        col_limit, nb = M.kernel_geometry(c["Lq"], c["Lk"], [b["row"]])
        res = M.attention_rows(q[ri, h][None, :], k[: c["Lk"]], v[: c["Lk"]], np.array([h // (H // KVH)]), col_limit, nb, scale, tabs, debug=True)
        dbg = res.dbg
        m = dbg["m"][0].view(np.float32)
        l = dbg["l"][0].view(np.float32).astype(np.float32)
        opre = dbg["o_pre"][0]
        nbk = int(nb[0])
        print(f"--- seq {b['seq']} {b['phase']} L{c['layer']} row {b['row']} head {h} d {d} Lk {c['Lk']} n_blocks {nbk} col_limit {int(col_limit[0])}")
        ms = m[:nbk][::-1]
        print("  running max after each processed block:", [float(x) for x in ms[:8]], "... #changes:", int((np.diff(ms) != 0).sum()))
        print("  lane sums", l.tolist())
        cands = {
            "(l0+l2)+(l1+l3)": np.float32(np.float32(l[0] + l[2]) + np.float32(l[1] + l[3])),
            "(l0+l1)+(l2+l3)": np.float32(np.float32(l[0] + l[1]) + np.float32(l[2] + l[3])),
            "(l0+l3)+(l1+l2)": np.float32(np.float32(l[0] + l[3]) + np.float32(l[1] + l[2])),
            "((l0+l1)+l2)+l3": np.float32(np.float32(np.float32(l[0] + l[1]) + l[2]) + l[3]),
        }
        o = f32(int(opre[d]))
        gw, mw = int(gpu[ri, h, d]), int(res.out[0, d])
        print(f"  gpu {gw:04x} model {mw:04x} model o*inv f32 {int(res.out_f32[0, d]):08x}  o_pre {int(opre[d]):08x}")
        for name, s in cands.items():
            inv = M.mufu_rcp(np.array([s], np.float32), tabs)[0]
            vals = (opre.view(np.float32) * inv).astype(np.float32)
            words = np.array([bf16(u32(x)) for x in vals])
            w = int(words[d])
            print(f"   {name:18s} sum={u32(s):08x} inv={u32(inv):08x} -> {w:04x} {'OK' if w == gw else 'x '} row {int((words == gpu[ri, h]).sum())}/128")
        inv = M.mufu_rcp(np.array([cands['(l0+l2)+(l1+l3)']], np.float32), tabs)[0]
        for name, (oo, ii) in {"o+1ulp": (f32(int(opre[d]) + 1), inv), "o-1ulp": (f32(int(opre[d]) - 1), inv),
                               "inv+1ulp": (o, f32(u32(inv) + 1)), "inv-1ulp": (o, f32(u32(inv) - 1))}.items():
            w = bf16(u32(np.float32(oo * ii)))
            print(f"   perturb {name:9s} -> {w:04x} {'OK' if w == gw else ''}")
        # which P columns are large (the attention pattern) and where the max lives
        P = dbg["p"][0].view(np.float32)
        top = np.argsort(P)[::-1][:5]
        print("  top P cols:", [(int(t), float(P[t])) for t in top], " sum P (f64):", float(P.astype(np.float64).sum()))


if __name__ == "__main__":
    main()
