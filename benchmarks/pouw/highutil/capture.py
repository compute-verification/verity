"""HIGHUTIL's activations: the s8 input of every linear of a W8A8 forward of the pinned checkpoint over real text.

The checkpoint is verity-int8/Llama-3.1-8B-INT8 (int8 weights, one float32 scale an output row). The text is the start of
wikitext-2-raw-v1's test split, tokenized by the checkpoint's tokenizer into `--seqs` sequences of `--len` tokens (BOS
first), M = seqs x len = 16,384 rows. Each linear is int8_exact's (integrations/vllm/verity_vllm/engine/int8_exact.py):
x quantized one scale a token (s_a = absmax / 127, round half to even, |X| <= 127), Y = X W^T exact in int32
(`torch._int_mm`), y = (Y s_a) s_w in bf16. The fused calls the period runs are the model's own: q, k and v share one
quantized input (QKV, 6144 rows of W), gate and up another (UP, 28672). Norms, RoPE (llama3 scaling), causal attention
(grouped-query) and SiLU run in bf16 with float32 norms.

    capture.py CFG --checkpoint DIR [--seqs 16] [--len 1024] [--text ARROW|PARQUET]

writes A-NAME.bin for each call in shapes.txt (CFG's blocks x M x K int8, block t's rows at offset t M K; a K-slice of a
linear, `down.0` and so on, holds the linear's input columns k0..k0 + K) into CFG and
capture.json: the text's source and the SHA-256 of its token ids, each A file's SHA-256, and the forward's perplexity on
the same tokens (next-token, all positions after the first), the check that the forward is the model's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WIKITEXT = Path("/workspace/hf/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/"
                "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")


class Checkpoint:
    def __init__(self, d: Path, dev: str) -> None:
        from safetensors import safe_open
        self.d, self.dev = d, dev
        self.cfg = json.loads((d / "config.json").read_text())
        self.idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        self._open = {}
        self._safe_open = safe_open

    def get(self, name: str) -> torch.Tensor:
        f = self.idx[name]
        if f not in self._open:
            self._open[f] = self._safe_open(str(self.d / f), framework="pt", device="cpu")
        return self._open[f].get_tensor(name).to(self.dev)


def tokens(ck: Checkpoint, seqs: int, length: int, path: Path = WIKITEXT) -> torch.Tensor:
    import pyarrow as pa
    from tokenizers import Tokenizer
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
    else:
        with pa.memory_map(str(path)) as src:
            table = pa.ipc.open_stream(src).read_all()
    text = "".join(table.column("text").to_pylist())
    ids = Tokenizer.from_file(str(ck.d / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    need = seqs * (length - 1)
    if len(ids) < need:
        raise SystemExit(f"wikitext-2's test split has {len(ids)} tokens, fewer than {need}")
    body = torch.tensor(ids[:need], dtype=torch.long).view(seqs, length - 1)
    return torch.cat([torch.full((seqs, 1), ck.cfg["bos_token_id"], dtype=torch.long), body], dim=1)


def row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    s = torch.clamp(xf.abs().amax(dim=-1), min=1e-30) / 127
    return torch.clamp(torch.round(xf / s[:, None]), -127, 127).to(torch.int8), s


def linear(xq: torch.Tensor, sa: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return ((torch._int_mm(xq, w.t()).float() * sa[:, None]) * sw[None, :]).to(torch.bfloat16)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def rope_tables(cfg: dict, length: int, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type") == "llama3":
        old, lo, hi, fac = rs["original_max_position_embeddings"], rs["low_freq_factor"], rs["high_freq_factor"], rs["factor"]
        wl = 2 * math.pi / inv
        smooth = ((old / wl) - lo) / (hi - lo)
        inv = torch.where(wl > old / lo, inv / fac, torch.where(wl < old / hi, inv, (1 - smooth) * inv / fac + smooth * inv))
    f = torch.outer(torch.arange(length, dtype=torch.float64), inv)
    e = torch.cat([f, f], dim=-1)
    return e.cos().float().to(dev), e.sin().float().to(dev)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    xf = x.float()
    rot = torch.cat([-xf[..., h:], xf[..., :h]], dim=-1)
    return (xf * cos + rot * sin).to(torch.bfloat16)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cfg")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--text", default=str(WIKITEXT), help="wikitext-2-raw-v1's test split: the datasets cache's arrow file, or "
                    "the hub's parquet at the same revision (wikitext-2-raw-v1/test-00000-of-00001.parquet)")
    a = ap.parse_args(argv)
    cfg_dir, dev = Path(a.cfg), "cuda"
    layers = [ln.split() for ln in (cfg_dir / "shapes.txt").read_text().splitlines() if ln.strip()]
    linears = list(dict.fromkeys(x[5] for x in layers))
    if linears != ["qkv", "o", "up", "down"]:
        raise SystemExit(f"{cfg_dir}/shapes.txt: linears {linears}, not the checkpoint's qkv o up down")
    slices = {lin: [(x[0], int(x[6]), int(x[3])) for x in layers if x[5] == lin] for lin in linears}
    m, nb = int(layers[0][1]), int(layers[0][4])
    if m != a.seqs * a.len:
        raise SystemExit(f"shapes.txt's M {m} isn't --seqs x --len = {a.seqs * a.len}")
    ck = Checkpoint(Path(a.checkpoint), dev)
    c = ck.cfg
    S, L, hd = a.seqs, a.len, c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    nh, nkv, eps = c["num_attention_heads"], c["num_key_value_heads"], c["rms_norm_eps"]
    ids = tokens(ck, S, L, Path(a.text))
    out = {x[0]: np.memmap(cfg_dir / f"A-{x[0]}.bin", mode="w+", dtype=np.int8, shape=(nb, m, int(x[3]))) for x in layers}
    cos, sin = rope_tables(c, L, dev)
    t0 = time.monotonic()
    with torch.no_grad():
        x = ck.get("model.embed_tokens.weight")[ids.to(dev)].view(m, -1)
        for t in range(c["num_hidden_layers"]):
            p = f"model.layers.{t}."
            w = {n: ck.get(p + n + ".weight") for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
            s = {n: ck.get(p + n + ".weight_scale").float().reshape(-1) for n in w}
            def cat(*ns, w=w, s=s):
                return torch.cat([w[n] for n in ns]), torch.cat([s[n] for n in ns])
            keep = t < nb

            def call(name: str, h: torch.Tensor, ws: tuple) -> torch.Tensor:
                xq, sa = row_quant(h)
                if keep:
                    x8 = xq.cpu().numpy()
                    for part, k0, k in slices[name]:
                        out[part][t] = x8[:, k0:k0 + k]
                return linear(xq, sa, *ws)

            qkv = call("qkv", rms(x, ck.get(p + "input_layernorm.weight"), eps),
                       cat("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
            q, k, v = qkv.split([nh * hd, nkv * hd, nkv * hd], dim=-1)
            q = rope(q.reshape(S, L, nh, hd).transpose(1, 2), cos, sin)
            k = rope(k.reshape(S, L, nkv, hd).transpose(1, 2), cos, sin)
            v = v.reshape(S, L, nkv, hd).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + call("o", att.transpose(1, 2).reshape(m, nh * hd), cat("self_attn.o_proj"))
            gu = call("up", rms(x, ck.get(p + "post_attention_layernorm.weight"), eps), cat("mlp.gate_proj", "mlp.up_proj"))
            g, u = gu.split(c["intermediate_size"], dim=-1)
            x = x + call("down", (F.silu(g.float()) * u.float()).to(torch.bfloat16), cat("mlp.down_proj"))
            del w, s, qkv, q, k, v, att, gu, g, u
        h = rms(x, ck.get("model.norm.weight"), eps).view(S, L, -1)
        head = ck.get("lm_head.weight")
        nll, count = 0.0, 0
        for r in range(S):
            logits = (h[r, :-1] @ head.t()).float()
            nll += F.cross_entropy(logits, ids[r, 1:].to(dev), reduction="sum").item()
            count += L - 1
    digests = {}
    for n, arr in out.items():
        arr.flush()
        with open(cfg_dir / f"A-{n}.bin", "rb") as f:
            digests[n] = hashlib.file_digest(f, "sha256").hexdigest()
    rec = {"checkpoint": str(a.checkpoint), "text": {"source": "Salesforce/wikitext wikitext-2-raw-v1 test", "file": a.text,
                                                     "seqs": S, "len": L, "bos": c["bos_token_id"],
                                                     "token_ids_sha256": hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()},
           "blocks": nb, "m": m, "a_sha256": digests, "ppl": math.exp(nll / count), "tokens_scored": count,
           "forward_s": time.monotonic() - t0,
           "linear": "int8_exact: x one scale a token (absmax/127, round half even), X W^T int32 exact, (Y s_a) s_w to bf16"}
    (cfg_dir / "capture.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
