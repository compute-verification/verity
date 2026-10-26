"""HYB's wrap densities on a real model: p_W on the served INT8 weights and p_X on the served activations.

Streams a pinned BF16 Llama checkpoint from the Hugging Face hub shard by shard (never more than the shards a layer needs
on disk), runs a float32 numpy forward over the first ``--tokens`` tokens of WikiText-2's test split, and for every linear:

* weights by verity-int8-channel127/v1 (`int8_checkpoint`): s = max|W[n,:]| / 127 per row, B = clamp(rint(W / s), ±127);
* activations by `benchmarks/ppl/steps.py`'s `act_quant` dither, per token: s = max|x| / 126, A = floor(x / s) on ±126;
* W* = B + c + U and X* = A + b + V, c and b fair coin bits, U and V uniform on ±u. The wrap density is exact from the
  histogram of B (or A): P[wrap | v] = E_c #{U : v + c + U outside [-128, 127]} / (2u + 1), no sampling.

Densities are aggregated by MACs (a linear's n·k), which is what the correction's adds per MAC are: p_W + p_X.

    uv run --no-project --with numpy --with tokenizers --with pyarrow python wrap_real.py --repo R --revision REV --out F
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

US = tuple(range(64, 128, 4)) + (127,)
WIKI = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")


def url(repo: str, rev: str, name: str, dataset: bool = False) -> str:
    return f"https://huggingface.co/{'datasets/' if dataset else ''}{repo}/resolve/{rev}/{name}"


def fetch(u: str, dst: str) -> str:
    hdr = {"User-Agent": "verity-accel-hybrid"}
    if os.environ.get("HF_TOKEN"):
        hdr["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=hdr)) as r, open(dst + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 24)
            os.replace(dst + ".part", dst)
            return dst
        except OSError as e:
            if getattr(e, "code", None) == 404:
                return ""
            print(f"fetch {u}: {e!r} (attempt {attempt + 1})", file=sys.stderr, flush=True)
    raise SystemExit(f"could not fetch {u}")


class Shard:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.meta = {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str) -> np.ndarray:
        m = self.meta[name]
        a, b = m["data_offsets"]
        raw = self.mm[self.base + a:self.base + b]
        if m["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
        if m["dtype"] == "F32":
            return np.array(raw.view(np.float32)).reshape(m["shape"])
        raise SystemExit(f"{name}: dtype {m['dtype']}")


def wrap_table(u: int) -> np.ndarray:
    """P[wrap | v] for v = -128..127, with a fair coin bit and noise uniform on ±u."""
    v = np.arange(-128, 128)
    out = np.zeros(256)
    for e in (0, 1):
        hi = np.clip(u - (127 - v - e), 0, 2 * u + 1)
        lo = np.clip(u - 128 - v - e, 0, 2 * u + 1)
        out += 0.5 * (hi + lo) / (2 * u + 1)
    return out


TABLES = {u: wrap_table(u) for u in US}


def hist(q: np.ndarray) -> np.ndarray:
    return np.bincount((q.astype(np.int64).ravel() + 128), minlength=256).astype(np.float64)


def quant_w(w: np.ndarray) -> np.ndarray:
    s = (np.maximum(np.abs(w).max(1), np.float32(1e-30)) / np.float32(127)).astype(np.float32)
    return np.clip(np.rint(w / s[:, None]), -127, 127)


def quant_x(x: np.ndarray) -> np.ndarray:
    s = (np.maximum(np.abs(x).max(1), np.float32(1e-30)) / np.float32(126)).astype(np.float32)
    return np.floor(np.clip(x / s[:, None], -126, 126))


def rope_freqs(cfg: dict, hd: int) -> np.ndarray:
    inv = 1.0 / (cfg.get("rope_theta", 10000.0) ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type", rs.get("type")) == "llama3":
        f, lf, hf, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
        wl = 2 * np.pi / inv
        scaled = np.where(wl > old / lf, inv / f, inv)
        sm = (old / wl - lf) / (hf - lf)
        med = (wl >= old / hf) & (wl <= old / lf)
        inv = np.where(med, (1 - sm) * scaled / f + sm * scaled, scaled)
    return inv


def rmsnorm(h, w, eps):
    return (h / np.sqrt((h * h).mean(-1, keepdims=True) + eps)) * w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--layers", type=int, default=0, help="0: all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    scratch = tempfile.mkdtemp(prefix="hyb-wrap-")
    try:
        run(a, scratch, t0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run(a, scratch: str, t0: float) -> None:
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    cfg = json.load(open(fetch(url(a.repo, a.revision, "config.json"), f"{scratch}/config.json")))
    ip = fetch(url(a.repo, a.revision, "model.safetensors.index.json"), f"{scratch}/index.json")
    if ip:
        idx = json.load(open(ip))
    else:
        one = Shard(fetch(url(a.repo, a.revision, "model.safetensors"), f"{scratch}/model.safetensors"))
        idx = {"weight_map": dict.fromkeys(one.meta, "model.safetensors")}
        one.mm._mmap.close()
    tok = Tokenizer.from_file(fetch(url(a.repo, a.revision, "tokenizer.json"), f"{scratch}/tokenizer.json"))
    wiki = pq.read_table(fetch(url(WIKI[0], "main", WIKI[1], dataset=True), f"{scratch}/wiki.parquet"))
    text = "\n\n".join(wiki.column("text").to_pylist())
    ids = np.array(tok.encode(text[:200000]).ids[:a.tokens])
    T = len(ids)
    wmap = idx["weight_map"]
    files = sorted(set(wmap.values()))
    pending = {f: {n for n, g in wmap.items() if g == f} for f in files}
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {f: pool.submit(fetch, url(a.repo, a.revision, f), f"{scratch}/{f}") for f in files}
    shards: dict[str, Shard] = {}

    def tensor(name: str) -> np.ndarray:
        f = wmap[name]
        if f not in shards:
            shards[f] = Shard(futs[f].result())
        out = np.array(shards[f].get(name), dtype=np.float32)
        pending[f].discard(name)
        if not pending[f]:
            shards.pop(f).mm._mmap.close()
            os.remove(f"{scratch}/{f}")
        return out

    H, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim", H // nh)
    eps = cfg.get("rms_norm_eps", 1e-5)
    nl = a.layers or cfg["num_hidden_layers"]
    inv = rope_freqs(cfg, hd)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(np.concatenate([ang, ang], 1)).astype(np.float32), np.sin(np.concatenate([ang, ang], 1)).astype(np.float32)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    acc = {}  # kind -> dict(hw: hist, hx: hist, macs per token)

    def record(kind: str, x: np.ndarray, ws: list[np.ndarray]) -> None:
        d = acc.setdefault(kind, {"hw": np.zeros(256), "hx": np.zeros(256), "nk": 0.0, "xk": 0.0})
        n = sum(w.shape[0] for w in ws)
        k = ws[0].shape[1]
        hx = hist(quant_x(x))
        d["hx"] += hx / hx.sum() * n * k       # activation entries weighted by the linear's MACs per token
        for w in ws:
            d["hw"] += hist(quant_w(w))         # every weight entry is n·k / (n·k) of one MAC per token
        d["nk"] += n * k

    def rot(z):
        z1, z2 = z[..., :hd // 2], z[..., hd // 2:]
        return z * cos + np.concatenate([-z2, z1], -1) * sin

    h = tensor("model.embed_tokens.weight")[ids]
    for li in range(nl):
        p = f"model.layers.{li}."
        x = rmsnorm(h, tensor(p + "input_layernorm.weight"), eps)
        wq, wk, wv = (tensor(p + f"self_attn.{s}_proj.weight") for s in "qkv")
        record("qkv", x, [wq, wk, wv])
        q = rot((x @ wq.T).reshape(T, nh, hd).transpose(1, 0, 2))
        k = rot((x @ wk.T).reshape(T, nkv, hd).transpose(1, 0, 2))
        v = (x @ wv.T).reshape(T, nkv, hd).transpose(1, 0, 2)
        rep = nh // nkv
        k, v = np.repeat(k, rep, 0), np.repeat(v, rep, 0)
        sc = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(hd)) + mask
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        att = ((sc / sc.sum(-1, keepdims=True)) @ v).transpose(1, 0, 2).reshape(T, nh * hd)
        wo = tensor(p + "self_attn.o_proj.weight")
        record("o", att, [wo])
        h = h + att @ wo.T
        x2 = rmsnorm(h, tensor(p + "post_attention_layernorm.weight"), eps)
        wg, wu = tensor(p + "mlp.gate_proj.weight"), tensor(p + "mlp.up_proj.weight")
        record("gate_up", x2, [wg, wu])
        g, u = x2 @ wg.T, x2 @ wu.T
        act = g / (1 + np.exp(-g)) * u
        wd = tensor(p + "mlp.down_proj.weight")
        record("down", act, [wd])
        h = h + act @ wd.T
        print(f"layer {li} done at {time.monotonic() - t0:.0f}s", flush=True)
    pool.shutdown(wait=False, cancel_futures=True)

    def dens(hw, u):
        return float((hw * TABLES[u]).sum() / hw.sum())

    vals = np.arange(-128, 128)
    res = {"repo": a.repo, "revision": a.revision, "tokens": T, "layers": nl, "kinds": {}, "total": {}}
    HW = sum(d["hw"] for d in acc.values())
    HX = sum(d["hx"] for d in acc.values())
    for kind, d in list(acc.items()) + [("total", {"hw": HW, "hx": HX, "nk": sum(d["nk"] for d in acc.values())})]:
        r = {"macs_per_token": d["nk"], "E|B|": float((d["hw"] * np.abs(vals)).sum() / d["hw"].sum()),
             "E|A|": float((d["hx"] * np.abs(vals)).sum() / d["hx"].sum()),
             "p_W": {u: dens(d["hw"], u) for u in US}, "p_X": {u: dens(d["hx"], u) for u in US},
             "hist_W": d["hw"].tolist(), "hist_X": d["hx"].tolist()}
        (res["total"] if kind == "total" else res["kinds"]).update(r if kind == "total" else {kind: r})
    res["seconds"] = time.monotonic() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    t = res["total"]
    print(json.dumps({"E|B|": t["E|B|"], "E|A|": t["E|A|"], "p_W": t["p_W"], "p_X": t["p_X"]}), flush=True)


if __name__ == "__main__":
    main()
