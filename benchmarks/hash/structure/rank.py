"""The structure levers ranked by what each buys a Qwen3-8B B32 prefill step, from run.sh / probe.sh outputs.

    python3 rank.py OUT [OUT ...]

Each OUT is a run's output directory; runs are grouped by GPU, and a later OUT overrides an earlier one's rows. The cost
of hashing a prefill step at a configuration (unit bytes B, leaf bytes L, salts in or off the step, tree arity a, the
data phase separate or fused) is priced from this lane's measurements at step scale:

- leaves: B over the separate data phase's leaf rate at L (gate_up, M 4096, `gs`'s second part);
- salts: B / L leaves at the salt phase's per-leaf cost at 1M leaves (`chacha` run: PRF y, or ChaCha20 y);
- tree: the a-ary tree's time at 1M leaves, scaled by B / L / 2^20;
- fusion: the GEMM outputs' share of the leaves (54% of the per-Call bytes) scaled by what the fused arm adds over the
  GEMM alone, over the separate leaves' time (`fuse diag`, gate_up, 2 KiB), or by 1 - the pipe probe's overlap for the
  ceiling row.

Step time = stock serving + hashing on the critical path (circuits' records show no overlap at B32: the GPU is busy).
Stock serving and the step's bytes are H200 records: 57 ms and 13.49 GB a prefill step (r20261018-031955-d2b3, via
note:hash/20261018T0623Z-draft-baseline); RTX PRO 6000's stock step is estimated as 57 ms times the ratio of cuBLASLt's
s8 GEMM time over the four prefill GEMMs on the two GPUs, and marked "est".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SERVE_H200_MS = 57.0
STEP_BYTES = 13.49e9                     # per Call, a prefill step (d2b3 layouts)
SERVED_HASH_H200_MS = 174.0 + 52.9        # today's rows-sha512 at 256 B rows: leaves + tree a prefill step (d2b3)
UNIT = {"per Call": 1.0, "per block": 36_864 / 145_180, "per layer": 20_480 / 145_180}   # B / token / layer ratios
# fusion reaches only what a GEMM produces: qkv, o_proj, gate_up and down's outputs, of the 145,180 B (circuits' layout_bytes)
GEMM_SHARE = (12_288 + 8_192 + 49_152 + 8_192) / 145_180
PREFILL = {(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)}


def lines(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def collect(outs: list[Path]) -> dict[str, dict]:
    g: dict[str, dict] = {}
    for o in outs:
        run = o.parent.name if o.name == "out" else o.name
        for r in lines(o / "bench-fuse.jsonl"):
            d = g.setdefault(short(r["gpu"]), {"leaf": {}, "lt": {}, "run": {}})
            a = r["arms"]
            if r["m"] == 4096 and (r["n"], r["k"]) in PREFILL and r["r"] == 64:
                d["lt"][(r["n"], r["k"])] = a["lt"]["ms"]
            if r["m"] == 4096 and (r["n"], r["k"]) == (24576, 4096):
                d["leaf"][r["leaf_bytes"]] = {"rate": r["operand_bytes"] / a["gs"]["part2_ms"], "run": run}
                if r["r"] == 64:
                    f = "f4" if a["f4"]["ms"] <= a["f8"]["ms"] else "f8"
                    nodes = a["gs"]["ms"] - a["g"]["ms"] - a["gs"]["part2_ms"]
                    d["fuse"] = {"rho": (a[f]["ms"] - a["g"]["ms"] - nodes) / a["gs"]["part2_ms"], "arm": f, "run": run,
                                 "ratio": a["gs"]["part2_ms"] / a["lt"]["ms"]}
        for r in lines(o / "tree.jsonl"):
            t = r["tree"]
            if t["leaves"] == 1 << 20:
                g.setdefault(short(t["gpu"]), {"leaf": {}, "lt": {}, "run": {}})["tree"] = {
                    a: t[f"a{a}"]["ms"] for a in (2, 4, 8, 16)} | {"run": run}
        for r in lines(o / "chacha.jsonl"):
            c = r["chacha"]
            if c["leaves"] == 1 << 20:
                g.setdefault(short(c["gpu"]), {"leaf": {}, "lt": {}, "run": {}})["salt"] = {
                    "prf": c["salt_phase_prf_ms"] / c["leaves"], "chacha": c["salt_phase_given_ms"] / c["leaves"], "run": run}
        for r in lines(o / "diag.jsonl"):
            x = r["diag"]
            if (x["m"], x["n"], x["k"], x["r"]) == (4096, 24576, 4096, 64) and "p4" in x:
                g.setdefault(short(x["gpu"]), {"leaf": {}, "lt": {}, "run": {}})["diag"] = x | {"run": run}
        for r in lines(o / "probe.jsonl"):
            p = r["probe"]
            d = g.setdefault(short(p["gpu"]), {"leaf": {}, "lt": {}, "run": {}})
            d.setdefault("probe", []).append(p | {"run": run})
    return g


def short(name: str) -> str:
    return "H200" if "H200" in name else "RTX PRO 6000" if "RTX PRO 6000" in name else name


def hash_ms(d: dict, unit: str = "per Call", leaf: int = 2048, salts: str = "prf", arity: int = 2, rho: float = 1.0) -> float:
    b = STEP_BYTES * UNIT[unit]
    n = b / leaf
    s = 0.0 if salts == "off" else n * d["salt"][salts]
    return b / d["leaf"][leaf]["rate"] * (1 - GEMM_SHARE * (1 - rho)) + s + d["tree"][arity] * n / (1 << 20)


def ceiling(d: dict) -> tuple[float, dict] | None:
    ps = d.get("probe", [])
    if not ps or "fuse" not in d:
        return None
    want = min((1.0, 0.25), key=lambda x: abs(x - d["fuse"]["ratio"]))
    top = max(p["tops_alone"] for p in ps)   # only configurations near the GEMM's full tensor rate
    best = max((p for p in ps if p["ratio"] == want and p["tops_alone"] >= 0.9 * top), key=lambda p: p["overlap"])
    return 1 - best["overlap"], best


def table(g: dict[str, dict]) -> str:
    gpus = [x for x in ("H200", "RTX PRO 6000") if x in g and {"leaf", "tree", "salt"} <= g[x].keys()]
    serve = {}
    for x in gpus:
        serve[x] = SERVE_H200_MS
        if x != "H200" and "H200" in g and len(g[x]["lt"]) == 4 and len(g["H200"]["lt"]) == 4:
            serve[x] = SERVE_H200_MS * sum(g[x]["lt"].values()) / sum(g["H200"]["lt"].values())
    rows = []
    for x in gpus:
        d = g[x]
        r0 = hash_ms(d)
        runs = ", ".join(dict.fromkeys((d["leaf"][2048]["run"], d["tree"]["run"], d["salt"]["run"])))
        lev: list[tuple[str, float, float, str]] = []
        if x == "H200":
            lev.append(("leaf: today's 256 B rows (served record) -> 2 KiB tile leaves", SERVED_HASH_H200_MS, r0,
                        f"r20261018-031955-d2b3; {runs}"))
        if 512 in d["leaf"]:
            lev.append(("leaf: 512 B -> 2 KiB tile leaves", hash_ms(d, leaf=512), r0, runs))
        if 4096 in d["leaf"]:
            lev.append(("leaf: 2 KiB -> 4 KiB tile leaves", r0, hash_ms(d, leaf=4096), runs))
        lev.append(("salts off the step (salt phase ahead)", r0, hash_ms(d, salts="off"), runs))
        lev.append(("salts: ChaCha20 y instead of the SHA-512 PRF, in step", r0, hash_ms(d, salts="chacha"), runs))
        for a in (4, 8):
            lev.append((f"tree: binary -> {a}-ary", r0, hash_ms(d, arity=a), runs))
        for u in ("per block", "per layer"):
            lev.append((f"bytes: per Call -> {u}", r0, hash_ms(d, unit=u), runs))
        if "diag" in d:
            dg = d["diag"]
            for arms, what in ((("p4", "p8"), "a persistent hash kernel beside the GEMM"), (("f4", "f8"), "hash warps in the GEMM's CTA")):
                a = min(arms, key=lambda k: dg[k])
                lev.append((f"fusion: {what} (measured, {a})", r0, hash_ms(d, rho=(dg[a] - dg["g"]) / dg["sep"]), dg["run"]))
        elif "fuse" in d:
            f = d["fuse"]
            lev.append((f"fusion: leaves in the GEMM's CTA (measured, {f['arm']})", r0, hash_ms(d, rho=f["rho"]), f["run"]))
        c = ceiling(d)
        if c:
            k, p = c
            tag = f"{p['mma_warps']}+{p['sha_warps']} warps, {'wgmma' if p['wgmma'] else 'mma.sync'}, overlap {p['overlap']:.2f}"
            lev.append((f"fusion: pipe ceiling (probe {tag})", r0, hash_ms(d, rho=k), p["run"]))
        lev.append(("combined: per layer, salts off, 8-ary", r0, hash_ms(d, unit="per layer", salts="off", arity=8), runs))
        for name, wo, wi, rid in lev:
            rows.append((x, name, wo, wi, rid))
    out = [f"Prefill step, Qwen3-8B B32, {STEP_BYTES / 1e9:.2f} GB per Call. Reference: 2 KiB tile leaves, SHA-512 PRF salts in "
           "the step, binary frame-v3 tree, separate pass. Stock serving a step: "
           + ", ".join(f"{x} {serve[x]:.1f} ms{' (est)' if x != 'H200' else ''}" for x in gpus) + ".", "",
           "| rank | lever | GPU | hash ms without | hash ms with | step ms without | step ms with | saves, % of serve | runs |",
           "|---|---|---|---|---|---|---|---|---|"]
    rows.sort(key=lambda r: -(r[2] - r[3]) / serve[r[0]])
    for i, (x, name, wo, wi, rid) in enumerate(rows, 1):
        s = serve[x]
        out.append(f"| {i} | {name} | {x} | {wo:.1f} | {wi:.1f} | {s + wo:.1f} | {s + wi:.1f} | {100 * (wo - wi) / s:+.1f}% | {rid} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(collect([Path(a) for a in sys.argv[1:]])))
