#!/usr/bin/env python3
"""D-OUT's T3: WikiText-2 perplexity of Llama-3.2-1B served int8 with 1-bit noise on every linear's int8 input.

Modes, on the same windows (so the rises are paired):
  bf16   the BF16 checkpoint: BF16 linears (torch's bf16 matmul, fp32 accumulation, bf16 out).
  int8   the int8 pin's linears as `int8_exact` serves them: per-token Int8RowQuant of the bf16 input, s8 weights with per-row
         scales (recipe `verity-int8-channel127/v1`, read from an int8 pin or made from the base by its rule), the product
         dequantized as bf16(RN32(RN32(Y s_a) s_w)).
  noise  int8 with D-OUT's noise: each linear's int8 input X becomes X - r, r a fresh uniform bit an element and a call,
         and each output gets ½·Σ_k W_nk back, so the served product is (X - r + ½)·Wᵀ. q, k and v are one call (one draw);
         gate and up draw their own.
  sink   noise on every row but the window's first (BOS at position 0, the attention sink, whose massive activations
         make one code of noise large against its other entries), which is served as in int8 and earns no credit.
  sink_nodown   sink with the down projection exact too.
In every mode the norms, RoPE (llama3 scaling), attention (fp32 softmax) and the tied LM head (fp32) are the same, and
every stage's output is rounded to bf16. The int8 products accumulate in fp32 rather than int32: exact while partial
sums stay under 2^24, and within a few fp32 ulps beyond.

The test split is joined with "\\n\\n" and tokenized without special tokens; window w is BOS followed by tokens
[w(ctx-1), (w+1)(ctx-1)), and every token after BOS is scored. --windows N takes N windows evenly spaced over the split.

  uv run --no-project --with torch --index https://download.pytorch.org/whl/cpu --with numpy --with tokenizers \\
    --with pyarrow python benchmarks/pouw/theory/hash_alt_ppl.py --base BF16.safetensors --tokenizer tokenizer.json \\
    --config config.json --text test.parquet --ctx 2048 8192 --windows 16 4 --seeds 5 --json OUT.json [--int8 PIN]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_alt_weights import tensors  # noqa: E402

CALLS = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
         "gate": ("mlp.gate_proj",), "up": ("mlp.up_proj",), "down": ("mlp.down_proj",)}
FLOOR = float(np.float32(1e-30))


def read(header: dict, raw: np.memmap, base: int, name: str) -> torch.Tensor:
    t = header[name]
    a, b = t["data_offsets"]
    dt = {"BF16": np.int16, "I8": np.int8, "F32": np.float32}[t["dtype"]]
    x = torch.from_numpy(np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(t["shape"]).copy())
    return x.view(torch.bfloat16) if t["dtype"] == "BF16" else x


def recipe_v1(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(s8 codes, per-row scales): s = RN32(max(absmax, 1e-30) / 127), q = clamp(rint(w / s), -127, 127)."""
    w = w.float()
    s = torch.clamp_min(w.abs().amax(dim=1, keepdim=True), FLOOR) / 127.0
    return torch.clamp(torch.round(w / s), -127, 127).to(torch.int8), s


def row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Int8RowQuant per token: (codes as fp32, scale)."""
    x = x.float()
    s = torch.clamp_min(x.abs().amax(dim=-1, keepdim=True), FLOOR) / 127.0
    return torch.clamp(torch.round(x / s), -127, 127), s


NOISY = {"noise", "sink", "sink_nodown"}


class Linear:
    """One served call: BF16 weights (transposed), and the s8 codes (as fp32, transposed), scales and ½·Σ_k W_nk."""

    def __init__(self, call: str, bf16: torch.Tensor, q: torch.Tensor, s: torch.Tensor):
        self.call = call
        self.bf16_t = bf16.t().contiguous()
        self.q_t = q.float().t().contiguous()
        self.s = s.float().reshape(1, -1)
        self.half_sum = (0.5 * q.double().sum(dim=1)).float().reshape(1, -1)

    def __call__(self, x: torch.Tensor, mode: str, gen: torch.Generator | None) -> torch.Tensor:
        if mode == "bf16":
            return x @ self.bf16_t
        X, s = row_quant(x)
        if mode in NOISY and not (mode == "sink_nodown" and self.call == "down"):
            r = torch.randint(0, 2, X.shape, generator=gen, dtype=torch.float32)
            half = self.half_sum.expand(X.shape[0], -1).clone()
            if mode != "noise":
                r[0], half[0] = 0, 0
            Y = (X - r) @ self.q_t + half
        else:
            Y = X @ self.q_t
        return ((Y * s) * self.s).bfloat16()


def llama3_inv_freq(cfg: dict) -> torch.Tensor:
    D, theta, r = cfg["head_dim"], cfg["rope_theta"], cfg["rope_scaling"]
    inv = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    lo_wl = r["original_max_position_embeddings"] / r["low_freq_factor"]
    hi_wl = r["original_max_position_embeddings"] / r["high_freq_factor"]
    wl = 2 * math.pi / inv
    smooth = (r["original_max_position_embeddings"] / wl - r["low_freq_factor"]) / (r["high_freq_factor"] - r["low_freq_factor"])
    scaled = torch.where(wl > lo_wl, inv / r["factor"], inv)
    mid = (wl <= lo_wl) & (wl >= hi_wl)
    return torch.where(mid, (1 - smooth) * inv / r["factor"] + smooth * inv, scaled).float()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF's rotate_half RoPE on [T, heads, D] in fp32."""
    h = x.shape[-1] // 2
    c, s = torch.cat([cos, cos], -1)[:, None], torch.cat([sin, sin], -1)[:, None]
    return x * c + torch.cat([-x[..., h:], x[..., :h]], -1) * s


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()).bfloat16()


class Model:
    def __init__(self, base: Path, config: Path, int8: Path | None):
        self.cfg = json.loads(config.read_text())
        hb, hp = tensors(base), tensors(int8) if int8 else None

        def call(i: int, c: str) -> Linear:
            names = CALLS[c]
            ws = [read(*hb, f"model.layers.{i}.{n}.weight") for n in names]
            qs = [(read(*hp, f"model.layers.{i}.{n}.weight"), read(*hp, f"model.layers.{i}.{n}.weight_scale")) if hp
                  else recipe_v1(w) for n, w in zip(names, ws)]
            return Linear(c, torch.cat(ws), torch.cat([q for q, _ in qs]), torch.cat([s for _, s in qs]))

        self.embed = read(*hb, "model.embed_tokens.weight")
        self.embed_t = self.embed.float().t().contiguous()
        self.norm = read(*hb, "model.norm.weight")
        self.layers = [{"in": read(*hb, f"model.layers.{i}.input_layernorm.weight"),
                        "post": read(*hb, f"model.layers.{i}.post_attention_layernorm.weight"),
                        **{c: call(i, c) for c in CALLS}} for i in range(self.cfg["num_hidden_layers"])]
        self.inv_freq = llama3_inv_freq(self.cfg)

    def nll(self, ids: torch.Tensor, mode: str, gen: torch.Generator | None) -> float:
        """The summed NLL of ids[1:] given their prefixes."""
        c = self.cfg
        T, H, KV, D, eps = ids.numel(), c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
        ang = torch.arange(T, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        cos, sin = torch.cos(ang), torch.sin(ang)
        x = self.embed[ids]
        for lay in self.layers:
            qkv = lay["qkv"](rmsnorm(x, lay["in"], eps), mode, gen).float()
            q = rope(qkv[:, :H * D].view(T, H, D), cos, sin).bfloat16().float()
            k = rope(qkv[:, H * D:(H + KV) * D].view(T, KV, D), cos, sin).bfloat16().float()
            v = qkv[:, (H + KV) * D:].view(T, KV, D)
            a = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True,
                enable_gqa=True)[0].transpose(0, 1).reshape(T, H * D).bfloat16()
            x = x + lay["o"](a, mode, gen)
            h = rmsnorm(x, lay["post"], eps)
            act = (torch.nn.functional.silu(lay["gate"](h, mode, gen).float()) * lay["up"](h, mode, gen).float()).bfloat16()
            x = x + lay["down"](act, mode, gen)
        h = rmsnorm(x, self.norm, eps).float()
        total = 0.0
        for a in range(0, T - 1, 1024):
            logits = h[a:min(a + 1024, T - 1)] @ self.embed_t
            tgt = ids[a + 1:a + 1 + logits.shape[0]]
            total += float((torch.logsumexp(logits, -1) - logits.gather(1, tgt[:, None])[:, 0]).double().sum())
        return total


def token_ids(tokenizer: Path, text: Path) -> list[int]:
    import pyarrow.parquet as pq
    import tokenizers
    s = "\n\n".join(pq.read_table(text).column("text").to_pylist())
    return tokenizers.Tokenizer.from_file(str(tokenizer)).encode(s, add_special_tokens=False).ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="the BF16 checkpoint (one safetensors file)")
    ap.add_argument("--int8", type=Path, help="the int8 pin (default: its recipe applied to --base)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--text", type=Path, required=True, help="WikiText-2 raw test parquet")
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 4], help="windows per ctx")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["bf16", "int8", "noise", "sink", "sink_nodown"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    ids = token_ids(args.tokenizer, args.text)
    model = Model(args.base, args.config, args.int8)
    bos = model.cfg["bos_token_id"]
    runs = [(m, None) for m in args.modes if m not in NOISY] + [(m, s) for m in args.modes if m in NOISY for s in range(args.seeds)]
    out = {"tokens": len(ids), "threads": torch.get_num_threads(), "ctx": {}}
    for ctx, n in zip(args.ctx, args.windows):
        count = len(ids) // (ctx - 1)
        picks = sorted({int(round(v)) for v in np.linspace(0, count - 1, n)})
        rows = []
        for w in picks:
            win = torch.tensor([bos] + ids[w * (ctx - 1):(w + 1) * (ctx - 1)])
            row = {"window": w, "scored": ctx - 1}
            for mode, seed in runs:
                t0 = time.perf_counter()
                gen = torch.Generator().manual_seed((seed * 65_537 + ctx) * 1_000_003 + w) if mode in NOISY else None
                key = mode if seed is None else f"{mode}:{seed}"
                row[key] = model.nll(win, mode, gen)
                print(f"ctx {ctx} window {w} {key}: ppl {math.exp(row[key] / row['scored']):.4f} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
            rows.append(row)
        out["ctx"][str(ctx)] = summary(rows, [m if s is None else f"{m}:{s}" for m, s in runs], count)
        if args.json:
            args.json.write_text(json.dumps(out, indent=1))
    print(json.dumps({c: v["summary"] for c, v in out["ctx"].items()}, indent=1))
    return 0


def summary(rows: list[dict], keys: list[str], count: int) -> dict:
    """Per key: perplexity over the windows. Per noisy key against int8 and bf16 (and int8 against bf16): the rise, with
    the standard error of the mean per-token log-ratio over windows; per noisy mode, the seeds' mean and range."""
    scored = sum(r["scored"] for r in rows)
    ppl = {k: math.exp(sum(r[k] for r in rows) / scored) for k in keys}
    s: dict = {"windows": len(rows), "of": count, "scored": scored, "ppl": ppl}

    def rise(a: str, b: str) -> dict:
        d = np.array([(r[a] - r[b]) / r["scored"] for r in rows])
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        return {"rise": ppl[a] / ppl[b] - 1, "log_se": se}

    noisy = [k for k in keys if ":" in k]
    for ref in ("int8", "bf16"):
        if ref in ppl:
            s[f"vs_{ref}"] = {k: rise(k, ref) for k in noisy + (["int8"] if ref == "bf16" and "int8" in ppl else [])}
    if "int8" in ppl:
        for mode in dict.fromkeys(k.split(":")[0] for k in noisy):
            r = [s["vs_int8"][k]["rise"] for k in noisy if k.split(":")[0] == mode]
            s.setdefault("mode_vs_int8", {})[mode] = {"mean": float(np.mean(r)), "min": min(r), "max": max(r), "seeds": len(r)}
    return {"summary": s, "rows": rows}


if __name__ == "__main__":
    sys.exit(main())
