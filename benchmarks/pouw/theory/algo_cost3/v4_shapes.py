"""DeepSeek-V4-Pro's static matmuls from its pinned checkpoint, against nofold's V4 shapes (RT-PROTO-2's P24).

`census` reads the safetensors header of every shard of deepseek-ai/DeepSeek-V4-Pro at the revision that
integrations/vllm/manifests/checkpoints.json pins (HTTP range requests, about 16 MB in all; `--fetch` gets what DIR
lacks), checks config.json against the pinned sha256, and sums the MACs a token of every 2-D weight it reads: the
routed experts at num_experts_per_tok of n_routed_experts (their I8 weights pack two e2m1 values a byte), mHC's F32
`hc_*_fn`, no MTP layer (the serve doesn't build it), and the embedding as a lookup. That total must equal `SERVED`'s,
vLLM's served instances (wq_a | wkv as fused_wqa_wkv, each compressor's wkv | wgate as one GEMM).

`coverage` prices V4's coverage at four readings with D6's world rows (a `d6.py --pickle` build at D6's own nofold,
stated 49e9): D6's (49e9 at the listed shapes' mix), every real matmul credited at its own shape, the same without
mHC's F32 GEMMs, and nofold's listed shapes only (wkv at the checkpoint's 512 columns).

`picks` gives v3.1's pick (`d6.rule_pick` at the build's headline τ) and D5's prices (`d5_v2.analyse`: served c, job
c, c at X = 1) at every served instance nofold's SHAPES lack, beside two it lists.

    python3 v4_shapes.py census DIR [--fetch] [OUT.json]
    PUBLIC_BEST_DIR=<D5>/benchmarks/pouw/theory python3 v4_shapes.py coverage <D6 accel_score dir> D6PKL [OUT.json]
    PUBLIC_BEST_DIR=<D5>/benchmarks/pouw/theory python3 v4_shapes.py picks <D6 accel_score dir> D6PKL [OUT.json]
"""
import hashlib
import json
import pickle
import re
import struct
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PIN = next(c for c in json.loads((REPO / "integrations/vllm/manifests/checkpoints.json").read_text())["checkpoints"]
           if c["repo"] == "deepseek-ai/DeepSeek-V4-Pro")
BASE = f"https://huggingface.co/{PIN['repo']}/resolve/{PIN['revision']}/"
H, L, V, CSA, HCA = 7168, 61, 129280, 30, 31
SERVED = {   # (K, N, reps) a token, as vLLM serves them
    "fused_wqa_wkv": (H, 1536 + 512, L),
    "wq_b": (1536, 128 * 512, L),
    "wo_a": (8 * 512, 16 * 1024, L),
    "wo_b": (16 * 1024, H, L),
    "experts w1|w3 (6 routed + 1 shared)": (H, 7 * 2 * 3072, L),
    "experts w2 (6 routed + 1 shared)": (3072, 7 * H, L),
    "head": (H, V, 1),
    "compressor wkv|wgate, CSA (ratio 4)": (H, 2 * 1024, CSA),
    "compressor wkv|wgate, HCA (ratio 128)": (H, 2 * 512, HCA),
    "indexer compressor wkv|wgate": (H, 2 * 256, CSA),
    "indexer wq_b": (1536, 64 * 128, CSA),
    "router gate": (H, 384, L),
    "indexer weights_proj": (H, 64, CSA),
    "mHC pre (hc_attn_fn, hc_ffn_fn), F32": (4 * H, 24, 2 * L),
    "mHC head (hc_head_fn), F32": (4 * H, 4, 1),
}
NEW = [k for k in SERVED if k.startswith(("compressor", "indexer", "router", "mHC"))]
F32 = [k for k in SERVED if "F32" in k]
PICK_SHAPES = {   # (K, N): what it serves, (layers)
    (7168, 2048): "fused_wqa_wkv (61), the CSA compressor's wkv|wgate (30)",
    (7168, 1024): "the HCA compressor's wkv|wgate (31)",
    (7168, 512): "wkv alone, as the checkpoint has it (61), the indexer compressor's wkv|wgate (30)",
    (1536, 8192): "the indexer's wq_b (30)",
    (7168, 384): "the router gate (61)",
    (7168, 64): "the indexer's weights_proj (30)",
    (28672, 24): "mHC's pre GEMMs, F32 in the checkpoint (122)",
    (7168, 576): "nofold's listed wkv (reference)",
    (7168, 1536): "nofold's listed wq_a (reference)",
}


def macs(keys) -> float:
    return float(sum(SERVED[k][0] * SERVED[k][1] * SERVED[k][2] for k in keys))


def get(name: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "config.json").exists():
        with urllib.request.urlopen(BASE + "config.json", timeout=120) as r:
            (d / "config.json").write_bytes(r.read())
    if not (d / "index.json").exists():
        with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=120) as r:
            (d / "index.json").write_bytes(r.read())

    def one(name):
        p = d / "headers" / f"{name}.json"
        if not p.exists():
            n = struct.unpack("<Q", get(name, 0, 7)[:8])[0]
            h = get(name, 8, 8 + n - 1)
            assert len(h) == n, (name, len(h), n)
            p.write_bytes(h)

    (d / "headers").mkdir(exist_ok=True)
    shards = sorted(set(json.loads((d / "index.json").read_text())["weight_map"].values()))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, shards))


def census(d: Path) -> dict:
    cfg_bytes = (d / "config.json").read_bytes()
    sha = hashlib.sha256(cfg_bytes).hexdigest()
    assert sha == PIN["config_sha256"], f"config.json is {sha}, the manifest pins {PIN['config_sha256']}"
    cfg = json.loads(cfg_bytes)
    shards = sorted(set(json.loads((d / "index.json").read_text())["weight_map"].values()))
    T = {}
    for s in shards:
        T.update({k: v for k, v in json.loads((d / "headers" / f"{s}.json").read_text()).items() if k != "__metadata__"})
    act = cfg["num_experts_per_tok"] / cfg["n_routed_experts"]
    kinds = defaultdict(lambda: {"macs": 0.0, "shapes": set(), "dtypes": set()})
    params = 0
    for k, v in T.items():
        n = 1
        for x in v["shape"]:
            n *= x
        packed = v["dtype"] in ("I8", "U8") and ".experts." in k
        if not k.endswith(".scale"):
            params += 2 * n if packed and k.endswith(".weight") else n
        if k.startswith("mtp.") or len(v["shape"]) != 2 or k == "embed.weight":
            continue
        if not (k.endswith(".weight") and "norm" not in k) and not re.search(r"hc_\w+_fn$", k):
            continue
        o, i = v["shape"]
        i *= 2 if packed else 1
        kind = re.sub(r"experts\.\d+\.", "experts.", re.sub(r"^layers\.\d+\.", "", k))
        kinds[kind]["macs"] += o * i * (act if ".experts." in kind else 1)
        kinds[kind]["shapes"].add((i, o))
        kinds[kind]["dtypes"].add(v["dtype"])
    total = sum(x["macs"] for x in kinds.values())
    emb = T["embed.weight"]["shape"]
    out = {"revision": PIN["revision"], "config_sha256": sha, "tensors": len(T), "params": params, "total": total,
               "served": macs(SERVED), "new": macs(NEW), "f32": macs(F32), "embedding": emb[0] * emb[1],
               "kinds": {k: {"macs": x["macs"], "shapes": sorted(x["shapes"]), "dtypes": sorted(x["dtypes"])}
                      for k, x in sorted(kinds.items(), key=lambda kv: -kv[1]["macs"])}}
    print(f"{PIN['repo']} at {PIN['revision'][:10]}: config.json sha256 {sha[:16]}… (the manifest's pin), "
          f"{len(T)} tensors, {params / 1e9:.2f}e9 parameters (MTP included, scales not)")
    print(f"\n{'weight, MACs a token (K, N)':44} {'MACs':>10}  shapes, dtypes")
    for k, x in out["kinds"].items():
        print(f"  {k:42} {x['macs'] / 1e6:9.2f}M  {x['shapes']} {x['dtypes']}")
    print(f"  {'all (no embedding, no MTP)':42} {total / 1e9:9.4f}e9; the embedding {emb} is a lookup "
          f"({out['embedding'] / 1e9:.4f}e9 parameters)")
    print("\nvLLM's served instances (K, N, layers):")
    for k, (K, N, r) in SERVED.items():
        print(f"  {k:42} ({K}, {N}, {r}): {K * N * r / 1e6:9.2f}M")
    print(f"  {'all':42} {out['served'] / 1e9:.4f}e9; not in nofold's SHAPES {out['new'] / 1e9:.4f}e9 "
          f"(F32 mHC {out['f32'] / 1e9:.4f}e9)")
    assert abs(out["served"] - total) < 1.0, (out["served"], total)
    return out


def d6_modules(d6dir: str):
    sys.path.insert(0, d6dir)
    import d6
    import nofold as nf
    return d6, nf


def coverage(d6dir: str, pkl: str) -> dict:
    d6, nf = d6_modules(d6dir)
    v4 = d6.model_of("DeepSeek-V4-Pro")
    assert v4["static"] == 49e9, "coverage reads D6's own nofold (stated 49e9)"
    listed = sum(K * N * r for K, N, r in nf.SHAPES[v4["name"]])
    listed512 = listed - H * 64 * L
    real = macs(SERVED)
    _, st49, qk, pv, ix = nf.world_macs(v4)
    tok, tot49 = nf.WORLD_TOKENS, st49 + qk + pv + ix
    rows = pickle.loads(Path(pkl).read_bytes())["rows"]
    out = {"real": real, "listed": listed, "listed_wkv512": listed512, "stated": v4["static"], "cards": {}}
    print(f"V4-Pro: real {real / 1e9:.4f}e9 MACs a token, nofold's listed {listed / 1e9:.4f}e9 ({listed512 / 1e9:.4f}e9 "
          f"with wkv at 512), stated {v4['static'] / 1e9:.2f}e9: stated − real {(v4['static'] - real) / 1e9:.4f}e9 = "
          f"{(v4['static'] - real) / real:.2%} of real; a request is {st49 / tot49:.2%} static, {qk / tot49:.2%} QKᵀ, "
          f"{pv / tot49:.2%} PV, {ix / tot49:.2%} the indexer's logits (at 49e9)")
    for card in d6.vc.CARDS:
        w = rows[d6.HEADLINE, card, "measured", v4["name"]]["world"]
        pv_unc = max(w["uncredited"] * tot49 - qk - ix, 0.0)

        def cov(total, credited, pv_unc=pv_unc):
            return (credited * tok + pv - pv_unc) / (total * tok + qk + pv + ix)

        c = {"D6 v1 (49e9 at the listed mix)": 1 - w["uncredited"],
             "every real matmul at its own shape": cov(real, real),
             "the same without mHC's F32 GEMMs": cov(real, real - macs(F32)),
             "nofold's listed shapes only (wkv 512)": cov(real, listed512),
             "PV uncredited, a share of PV": pv_unc / pv}
        out["cards"][card] = c
        print(f"  {card}: " + "; ".join(f"{k} {v:.2%}" for k, v in c.items()))
    return out


def picks(d6dir: str, pkl: str) -> dict:
    d6, nf = d6_modules(d6dir)
    tau = pickle.loads(Path(pkl).read_bytes())["rule"]["tau"]
    out = {"tau": tau, "rows": {}}
    print(f"v3.1's picks (d6.rule_pick at τ {tau}) and D5's prices at V4-Pro's served instances nofold lacks:")
    for card in d6.vc.CARDS:
        for (K, N), what in PICK_SHAPES.items():
            p = d6.rule_pick(card, K, N, tau[card])
            a = d6.d5v2.analyse(nf, card, K, N, p["t"], p["Lp"], sens=())
            r = {"what": what, "t": p["t"], "Lp": p["Lp"], "leak": float(p["leak"]), "gamma": float(p["gamma"]),
                     "feasible": p["feasible"], "served_c": a["served"]["c"], "job_c": a["job"]["c"],
                     "c_x1": a["fixed"][1.0]["c"], "grid": [(x["Lp"], float(x["leak"]), float(x["gamma"]))
                                                     for x in d6.grid(card, K, N)]}
            out["rows"][f"{card} | {K}x{N}"] = r
            print(f"  {card:13} {K:6}x{N:<6} t {r['t']} L′ {r['Lp']:4} leak {r['leak']:.4f} γ {r['gamma']:.3f} "
                  f"served c {r['served_c']:.4f} job c {r['job_c']:.4f} c at X = 1 {r['c_x1']:.4f}  ({what})")
    return out


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a[0] == "census":
        if "--fetch" in sys.argv:
            fetch(Path(a[1]))
        res, rest = census(Path(a[1])), a[2:]
    else:
        res, rest = {"coverage": coverage, "picks": picks}[a[0]](a[1], a[2]), a[3:]
    if rest:
        Path(rest[0]).write_text(json.dumps(res, indent=1, default=lambda x: sorted(x) if isinstance(x, set) else float(x)))


if __name__ == "__main__":
    main()
