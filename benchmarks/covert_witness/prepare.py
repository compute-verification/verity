"""The fixed side of the covert-witness speedrun: the registered starts, the decoders, the training-FLOP meter, the witness
checker, the classes' caps and `evaluate_bpb`. A candidate's `train.py` imports it and never edits it; `evaluate.py` runs it.

A candidate starts from a registered start θ (`load_start`), trains within its class's caps and the fixed budget, and submits
(`submit`):

- E, an encoding of its update in one of the fixed formats (`encode_full`, `encode_lora`, `encode_lora2`,
  `encode_seedlog`, `encode_sparse`); `decode(θ, E)` rebuilds the model, 8·|E| bits is the update-size witness, and
  `rebuild_macs` is what the rebuild costs;
- the model it claims, which the checker requires to equal decode(θ, E) bit for bit, with θ's tensors, shapes and dtypes (so
  its size, evaluation FLOPs and weight bytes per byte are θ's);
- its step log: one record per pass over data (`{"op": "train"|"forward", "tokens": T, "ctx": C}`) and any matmul FLOPs it
  spends outside passes (`{"op": "extra", "flops": F}`), from which `train_flops` counts analytic training FLOPs;
- `trainable`: the tensors it trains in full and the matrices it trains through rank-r adapters, which the meter prices;
- `changed`: the tensors that differ from θ, which must cover what does, and from which the checker computes the deep share φ
  and the lowest updated layer.

Positions (`positions`): the embedding lookup is layer 0, block i is layer i + 1, the final norm and the head are layer L + 1;
inside a block the stages are norm, q/k/v, o, norm, gate/up, down. Node u is upstream of v when u's (layer, stage) is earlier.
φ is the share of the forward's weight-matmul FLOPs (the tied head counted as a matmul, the lookup and the norm gains as
0-FLOP nodes) whose weight differs from θ and that has an updated node, a lookup or a gain included, upstream.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

#: the fixed side's version, in every witness record (README.md)
VERSION = "2.2"
ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
LF_CORPUS = Path(os.environ.get("CW_LF_CORPUS", "/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1"))
STARTS = ROOT / "starts"
HELD_OUT = ROOT / "eval"
DEVICE = os.environ.get("CW_DEVICE", "cuda")
EVAL_CTX = 1024
#: about 10 GPU-minutes of m's dense AdamW at its measured 1.49e14 FLOP/s (learning_frontier, RTX PRO 6000)
TRAIN_FLOP_BUDGET = 9e16
#: the budget is checked on 2x the MACs the harness counts (te_core), which include the elementwise and optimizer work
#: `train_flops` leaves out: 1.06-1.09x of `train_flops` on the reference island
HARNESS_FLOP_SLACK = 1.1
#: the candidate's steps' wall seconds as the harness times them (its own loading, encoding and evaluation excluded), on one
#: GPU; the harness runs eager and counted, about 2.4x a compiled run's time
GPU_SECONDS_CAP = 3000.0
CLASSES = {
    "reference": {"max_e_bytes": None, "formats": None, "top_blocks": None},
    "e1mb": {"max_e_bytes": 1_000_000, "formats": None, "top_blocks": None},
    "e100kb": {"max_e_bytes": 100_000, "formats": None, "top_blocks": None},
    "shallow": {"max_e_bytes": None, "formats": None, "top_blocks": 2},
    "seed": {"max_e_bytes": 100_000, "formats": ["seedlog"], "top_blocks": None},
}
MAGIC = {b"CWFULL1\n": "full", b"CWLORA1\n": "lora", b"CWLORA2\n": "lora2", b"CWSEED1\n": "seedlog",
         b"CWSPAR1\n": "sparse"}
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
EOT = {"gpt2": 50256, "smollm2": 0}


# ---- starts ------------------------------------------------------------------------------------------------------------

def digest(theta: dict[str, torch.Tensor]) -> str:
    """sha256 over the tensors in name order: each name, dtype, shape and its little-endian bytes."""
    h = hashlib.sha256()
    for name in sorted(theta):
        t = theta[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def register_start(name: str, theta: dict[str, torch.Tensor], config: dict, tokenizer: str, provenance: dict) -> dict:
    from safetensors.torch import save_file

    d = STARTS / name
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in theta.items()}, str(d / "theta.safetensors.tmp"))
    (d / "theta.safetensors.tmp").rename(d / "theta.safetensors")
    rec = {"schema": "covert-witness/start/v1", "name": name, "digest": digest(theta), "config": config,
           "tokenizer": tokenizer, "parameters": int(sum(v.numel() for v in theta.values())),
           "tensors": {k: [str(v.dtype), list(v.shape)] for k, v in sorted(theta.items())}, "provenance": provenance}
    (d / "start.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def start_record(name: str) -> dict:
    return json.loads((STARTS / name / "start.json").read_text())


def load_theta(name: str, check: bool = True) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    theta = load_file(str(STARTS / name / "theta.safetensors"))
    if check and digest(theta) != start_record(name)["digest"]:
        raise SystemExit(f"prepare: start {name}'s weights do not match its registered digest")
    return theta


class Net(torch.nn.Module):
    """A start's architecture: `logits(idx)`, and `named_parameters()` named as θ's tensors."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        if config["family"] == "lf":
            from model import LM, Size

            self.inner = LM(Size(config["name"], config["d_model"], config["n_layer"], config["n_head"], config["d_ff"]))
            self.prefix = "inner."
        else:
            from transformers import LlamaConfig, LlamaForCausalLM

            cfg = LlamaConfig(**config["hf"])
            cfg._attn_implementation = "sdpa"
            self.inner = LlamaForCausalLM(cfg)
            self.prefix = "inner."

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        if self.config["family"] == "lf":
            return self.inner(idx)
        return self.inner(input_ids=idx, use_cache=False).logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.logits(idx)

    def params(self) -> dict[str, torch.nn.Parameter]:
        return {n[len(self.prefix):]: p for n, p in self.named_parameters()}

    @torch.no_grad()
    def load(self, theta: dict[str, torch.Tensor]) -> "Net":
        ps = self.params()
        if set(ps) != set(theta):
            raise SystemExit(f"prepare: tensors differ from the architecture's: {sorted(set(ps) ^ set(theta))[:6]}")
        for n, p in ps.items():
            if p.shape != theta[n].shape:
                raise SystemExit(f"prepare: {n} has shape {tuple(theta[n].shape)}, the architecture {tuple(p.shape)}")
            p.data = theta[n].to(p.device).clone()
        return self


def load_start(name: str, device: str = DEVICE) -> tuple[Net, dict, dict[str, torch.Tensor]]:
    """(the start's model on `device`, its record, θ on the CPU)."""
    rec = start_record(name)
    theta = load_theta(name)
    with torch.device(device):
        net = Net(rec["config"])
    net.load(theta)
    return net, rec, theta


def register_smollm2(name: str = "smollm2") -> dict:
    """SmolLM2-135M from the hub at its current revision (recorded), its weights as stored, the tied head dropped."""
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import load_file

    repo = "HuggingFaceTB/SmolLM2-135M"
    sha = HfApi().model_info(repo).sha
    d = Path(snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors"]))
    hf = json.loads((d / "config.json").read_text())
    config = {"family": "hf-llama", "name": name, "d_model": hf["hidden_size"], "n_layer": hf["num_hidden_layers"],
              "n_head": hf["num_attention_heads"], "d_ff": hf["intermediate_size"], "vocab": hf["vocab_size"],
              "ctx": EVAL_CTX, "tied": bool(hf.get("tie_word_embeddings", True)), "hf": hf}
    raw = load_file(str(d / "model.safetensors"))
    with torch.device("cpu"):
        net = Net(config)
    names = set(net.params())
    theta = {k: v for k, v in raw.items() if k in names}
    if set(theta) != names:
        raise SystemExit(f"prepare: SmolLM2's file lacks {sorted(names - set(theta))[:4]}")
    return register_start(name, theta, config, "smollm2", {"repo": repo, "revision": sha,
                                                           "dropped": sorted(set(raw) - names)})


# ---- data --------------------------------------------------------------------------------------------------------------

class TrainData:
    """The start's tokenizer's training stream cut into ctx + 1 token windows at stride ctx, permuted by `seed`.

    gpt2: learning_frontier's shards 000-002 (the lf starts read the first 8 tokens per parameter of its seed-0 order);
    smollm2: shard 000's documents (`data.py`)."""

    def __init__(self, tokenizer: str, seed: int = 1, ctx: int = EVAL_CTX):
        path = LF_CORPUS / "train.bin" if tokenizer == "gpt2" else ROOT / "train" / "smollm2.bin"
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = ctx
        self.order = np.random.default_rng(seed).permutation((len(self.tokens) - 1) // ctx)
        self.at = 0

    def next(self, n: int, device: str = DEVICE) -> torch.Tensor:
        if self.at + n > len(self.order):
            self.at = 0
        idx = np.sort(self.order[self.at: self.at + n])
        self.at += n
        c = self.ctx
        rows = np.stack([self.tokens[i * c: i * c + c + 1] for i in idx]).astype(np.int64)
        return torch.from_numpy(rows).to(device, non_blocking=True)


# ---- positions, φ and the training-FLOP meter ----------------------------------------------------------------------

LF_STAGE = {"attn_norm": 0, "wq": 1, "wk": 1, "wv": 1, "wo": 2, "mlp_norm": 3, "w_gate": 4, "w_up": 4, "w_down": 5}
HF_STAGE = {"input_layernorm": 0, "self_attn.q_proj": 1, "self_attn.k_proj": 1, "self_attn.v_proj": 1, "self_attn.o_proj": 2,
            "post_attention_layernorm": 3, "mlp.gate_proj": 4, "mlp.up_proj": 4, "mlp.down_proj": 5}


def positions(rec: dict) -> list[dict]:
    """Every node of the forward: {param, layer, stage, flops (forward weight-matmul FLOPs per token), d_in, d_out}."""
    L = rec["config"]["n_layer"]
    lf = rec["config"]["family"] == "lf"
    embed = "embed" if lf else "model.embed_tokens.weight"
    final = "norm" if lf else "model.norm.weight"
    nodes = []
    for name, (_, shape) in rec["tensors"].items():
        if name == embed:
            nodes.append({"param": name, "layer": 0, "stage": 0, "flops": 0, "d_in": shape[0], "d_out": shape[1]})
            nodes.append({"param": name, "layer": L + 1, "stage": 1, "flops": 2 * shape[0] * shape[1], "d_in": shape[1],
                          "d_out": shape[0]})
            continue
        if name == final:
            nodes.append({"param": name, "layer": L + 1, "stage": 0, "flops": 0, "d_in": shape[0], "d_out": shape[0]})
            continue
        m = re.fullmatch(r"blocks\.(\d+)\.(\w+?)(?:\.weight)?" if lf else r"model\.layers\.(\d+)\.(.+)\.weight", name)
        if not m:
            raise SystemExit(f"prepare: {name} has no position")
        stage = (LF_STAGE if lf else HF_STAGE)[m.group(2)]
        two_d = len(shape) == 2
        nodes.append({"param": name, "layer": int(m.group(1)) + 1, "stage": stage,
                      "flops": 2 * shape[0] * shape[1] if two_d else 0,
                      "d_in": shape[1] if two_d else shape[0], "d_out": shape[0]})
    return sorted(nodes, key=lambda n: (n["layer"], n["stage"], n["param"]))


def deep_share(rec: dict, changed: set[str]) -> dict:
    nodes = positions(rec)
    total = sum(n["flops"] for n in nodes)
    updated = [(n["layer"], n["stage"]) for n in nodes if n["param"] in changed]
    first = min(updated) if updated else None
    deep = sum(n["flops"] for n in nodes if n["param"] in changed and first is not None and first < (n["layer"], n["stage"]))
    L = rec["config"]["n_layer"]
    return {"phi": deep / total, "deep_flops_per_token": deep, "weight_matmul_flops_per_token": total,
            "changed_matmul_flops_per_token": sum(n["flops"] for n in nodes if n["param"] in changed),
            "lowest_updated_layer": first[0] if first else None,
            "update_depth": (L + 2 - first[0]) if first else 0, "layers": L + 2,
            "layer_rule": "0 the embedding lookup, i + 1 block i, L + 1 the final norm and the head; update_depth counts the "
                          "layers from the head down to the lowest updated one"}


def attention_flops(rec: dict, ctx: int) -> int:
    """Forward FLOPs per token of the score and value matmuls, causal: 2 · 2 · d_attn · (ctx + 1) / 2 per layer."""
    c = rec["config"]
    d_attn = c["d_model"] if c["family"] == "lf" else c["hf"]["num_attention_heads"] * c["hf"].get(
        "head_dim", c["d_model"] // c["hf"]["num_attention_heads"])
    return 2 * d_attn * (ctx + 1)


def train_flops(rec: dict, trainable: dict, steps: list[dict]) -> dict:
    """Analytic training FLOPs of a step log. A `train` pass: the forward (base and adapters) plus, at and above the lowest
    trained node, every weight matmul's input gradient, the full-trained matrices' weight gradients, the adapters' backward
    (4 · r · (d_in + d_out) per token at each matmul, 4 · r · d_out at the lookup) and the attention's (twice its forward);
    a `forward` pass: the forward; `extra`: as logged. A tied embedding's adapter is used at the lookup and the head, so it
    is priced at both, and its lowest trained node is the lookup (layer 0), as `deep_share` places it."""
    nodes = positions(rec)
    by_param = {}
    for n in nodes:
        by_param.setdefault(n["param"], []).append(n)
    full = set(trainable.get("full", []))
    adapters = {k: int(v) for k, v in trainable.get("adapters", {}).items()}
    unknown = (full | set(adapters)) - set(by_param)
    if unknown:
        raise ValueError(f"trainable names no tensor of the start: {sorted(unknown)[:4]}")
    gains = sorted(p for p in adapters if not any(x["flops"] for x in by_param[p]))
    if gains:
        raise ValueError(f"an adapter needs a matrix; these have no matmul: {gains[:4]}")
    L = rec["config"]["n_layer"]
    trained = [(n["layer"], n["stage"]) for n in nodes if n["param"] in full or n["param"] in adapters]
    first = min(trained) if trained else None

    def adapter_entries(p):
        return adapters[p] * sum(x["d_in"] + x["d_out"] if x["flops"] else x["d_out"] for x in by_param[p])

    def per_token(op: str, ctx: int) -> int:
        attn = attention_flops(rec, ctx)
        fwd = sum(n["flops"] for n in nodes) + sum(2 * adapter_entries(p) for p in adapters) + L * attn
        if op == "forward" or first is None:
            return fwd
        above = [n for n in nodes if (n["layer"], n["stage"]) >= first]
        bwd = sum(n["flops"] for n in above)
        bwd += sum(n["flops"] for n in above if n["param"] in full)
        bwd += sum(4 * adapter_entries(p) for p in adapters)
        bwd += sum(2 * attn for layer in range(1, L + 1) if (layer, 1) >= first)
        return fwd + bwd

    total, tokens, cache = 0, {"train": 0, "forward": 0}, {}
    for s in steps:
        op = s["op"]
        if op == "extra":
            total += int(s["flops"])
            continue
        if op not in ("train", "forward"):
            raise ValueError(f"step log: unknown op {op!r}")
        key = (op, int(s.get("ctx", EVAL_CTX)))
        if key not in cache:
            cache[key] = per_token(*key)
        total += cache[key] * int(s["tokens"])
        tokens[op] += int(s["tokens"])
    return {"flops": total, "tokens": tokens, "per_token": {f"{k[0]}@{k[1]}": v for k, v in cache.items()},
            "lowest_trained": list(first) if first else None}


def eval_cost(rec: dict, tokens: int, nbytes: int) -> dict:
    """The evaluation's analytic forward FLOPs and weight bytes read per byte of held-out text, at EVAL_CTX: every
    parameter's bytes read once per window (a lookup table counted in full), the attention at the full context."""
    nodes = positions(rec)
    fwd = sum(n["flops"] for n in nodes) + rec["config"]["n_layer"] * attention_flops(rec, EVAL_CTX)
    weight_bytes = sum(math.prod(s) * DTYPES_BYTES[d] for d, s in rec["tensors"].values())
    windows = math.ceil((tokens - 1) / EVAL_CTX)
    return {"flops_per_byte": fwd * (tokens - 1) / nbytes, "weight_bytes_per_byte": weight_bytes * windows / nbytes,
            "parameters": rec["parameters"], "weight_bytes": weight_bytes}


DTYPES_BYTES = {"torch.float32": 4, "torch.bfloat16": 2, "torch.float16": 2}


# ---- encodings ---------------------------------------------------------------------------------------------------------

def _pack(magic: bytes, header: dict, payload: bytes) -> bytes:
    h = zlib.compress(json.dumps(header, separators=(",", ":")).encode(), 9)
    return magic + struct.pack("<I", len(h)) + h + payload


def _unpack(blob: bytes) -> tuple[str, dict, memoryview]:
    fmt = MAGIC.get(blob[:8])
    if fmt is None:
        raise ValueError("E: unknown format")
    (n,) = struct.unpack("<I", blob[8:12])
    return fmt, json.loads(zlib.decompress(blob[12:12 + n])), memoryview(blob)[12 + n:]


def encode_full(tensors: dict[str, torch.Tensor]) -> bytes:
    """E as replacement tensors (any subset of θ's, θ's dtypes)."""
    from safetensors.torch import save

    return b"CWFULL1\n" + save({k: v.detach().cpu().contiguous() for k, v in sorted(tensors.items())})


def encode_lora(factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]], dtype: str = "bf16",
                full: dict[str, torch.Tensor] | None = None) -> bytes:
    """E as rank-r factor pairs: W' = W + scale · B @ A for each weight, A (r x d_in) and B (d_out x r) stored in `dtype`,
    and optionally small tensors in full (norm gains), stored in `dtype` too."""
    dt = DTYPES[dtype]
    header = {"dtype": dtype, "factors": [], "full": []}
    parts = []
    for name in sorted(factors):
        a, b, scale = factors[name]
        header["factors"].append([name, int(a.shape[0]), float(scale)])
        parts += [a.detach().to(dt).cpu().contiguous(), b.detach().to(dt).cpu().contiguous()]
    for name in sorted(full or {}):
        header["full"].append(name)
        parts.append(full[name].detach().to(dt).cpu().contiguous())
    return _pack(b"CWLORA1\n", header, b"".join(p.view(torch.uint8).numpy().tobytes() for p in parts))


def quantize_coefficient(c: float, coeff: str = "f16") -> float:
    return float(torch.tensor(c, dtype=DTYPES[coeff]).float())


def encode_seedlog(names: list[str], seed: int, coefficients: list[float], coeff: str = "f16") -> bytes:
    """E as a seed log: step i adds −c_i · z_i to each named tensor's fp32 master, z_i ~ N(0, 1) drawn in `names` order
    from a CUDA generator seeded `seed + i` (`seedlog_step`); the masters are cast to θ's dtypes at the end."""
    header = {"names": list(names), "seed": int(seed), "steps": len(coefficients), "coeff": coeff}
    c = torch.tensor(coefficients, dtype=DTYPES[coeff])
    return _pack(b"CWSEED1\n", header, c.view(torch.uint8).numpy().tobytes())


@torch.no_grad()
def seedlog_step(masters: list[torch.Tensor], seed: int, c: float, gen: torch.Generator | None = None) -> None:
    gen = gen or torch.Generator(device=masters[0].device)
    gen.manual_seed(seed)
    for w in masters:
        w.add_(torch.randn(w.shape, generator=gen, device=w.device, dtype=torch.float32), alpha=-c)


@torch.no_grad()
def lora_merge(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float,
               magnitude: torch.Tensor | None = None) -> torch.Tensor:
    """θ's weight plus scale · B @ A, in fp32 on the weight's device without TF32, cast to the weight's dtype. With a
    magnitude m (DoRA, CWLORA2), each output row i of that sum is first rescaled in fp32 to norm m_i; a row of norm 0 is
    refused."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        v = w.float() + scale * (b.float() @ a.float())
        if magnitude is not None:
            n = torch.linalg.vector_norm(v, dim=1)
            if not bool((n > 0).all()):
                raise ValueError("E (lora2): a row of norm 0 under its magnitude")
            v = v * (magnitude.float() / n)[:, None]
        return v.to(w.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


LORA2_BITS = (4, 8)


def _quantize_rows(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row of x as integers in ±(2^(bits−1) − 1), rounded to nearest, times its f16 scale max|row| / (2^(bits−1) − 1)."""
    qmax = 2 ** (bits - 1) - 1
    x = x.detach().float()
    s = (x.abs().amax(dim=1) / qmax).to(torch.float16)
    sf = s.float()[:, None]
    q = torch.where(sf > 0, torch.round(x / torch.where(sf > 0, sf, 1.0)), 0.0)
    return q.clamp(-qmax, qmax).to(torch.int8), s


def _dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * s.float()[:, None]


def lora2_factors(a: torch.Tensor, b: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """The fp32 factors CWLORA2 stores for A (r x d_in) and B (d_out x r) at `bits`: A's rows and B's columns, the rank
    components, quantized each with its own f16 scale. `decode` rebuilds exactly these (integer times f16 is exact in fp32),
    so a program can train through them."""
    qa, sa = _quantize_rows(a, bits)
    qb, sb = _quantize_rows(b.t(), bits)
    return _dequantize(qa, sa), _dequantize(qb, sb).t().contiguous()


def _pack_ints(q: torch.Tensor, bits: int) -> bytes:
    v = q.reshape(-1).cpu().numpy().astype(np.int8)
    if bits == 8:
        return v.tobytes()
    n = v.astype(np.uint8) & 0xF
    if n.size % 2:
        n = np.append(n, np.uint8(0))
    return (n[0::2] | (n[1::2] << 4)).astype(np.uint8).tobytes()


def _unpack_ints(b: bytes, count: int, bits: int, name: str) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(b, dtype=np.int8).copy()
    u = np.frombuffer(b, dtype=np.uint8)
    n = np.empty(2 * u.size, dtype=np.uint8)
    n[0::2], n[1::2] = u & 0xF, u >> 4
    if count % 2 and n[-1]:
        raise ValueError(f"E (lora2): {name}: a padding nibble is not zero")
    n = n[:count].astype(np.int8)
    return np.where(n >= 8, n - 16, n).astype(np.int8)


def _tensor_bytes(t: torch.Tensor, dt: torch.dtype) -> bytes:
    return t.detach().to(dt).cpu().contiguous().view(torch.uint8).numpy().tobytes()


def encode_lora2(factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]], bits: int | dict[str, int] = 4,
                 dtype: str = "bf16", magnitudes: dict[str, torch.Tensor] | None = None,
                 full: dict[str, torch.Tensor] | None = None) -> bytes:
    """E as quantized rank-r factor pairs: W' = W + scale · B @ A with A and B as `lora2_factors` gives them at `bits` (4 or
    8, one for every pair or a dict by name). With a magnitude m for a weight (DoRA, d_out entries), each output row i of
    W' is rescaled to norm m_i (`lora_merge`). Magnitudes and the small tensors carried in full (norm gains) are stored in
    `dtype`.

    Header `{"dtype", "factors": [[name, r, scale, bits, has magnitude], ...], "full": [name, ...]}`, both lists in
    strictly increasing name order; payload, per pair: A's r·d_in integers, A's r f16 scales, Bᵀ's r·d_out integers, Bᵀ's
    r f16 scales, the magnitude if any; then each full tensor. 4-bit integers are two's-complement nibbles, low nibble
    first, a factor's odd last byte padded with a zero nibble."""
    dt = DTYPES[dtype]
    magnitudes, full = magnitudes or {}, full or {}
    if set(magnitudes) - set(factors) or set(full) & set(factors):
        raise ValueError("encode_lora2: a magnitude needs its weight's factors, and a tensor carried in full has none")
    header = {"dtype": dtype, "factors": [], "full": sorted(full)}
    parts = []
    for name in sorted(factors):
        a, b, scale = factors[name]
        nb = bits[name] if isinstance(bits, dict) else bits
        if nb not in LORA2_BITS:
            raise ValueError(f"encode_lora2: bits must be one of {LORA2_BITS}, not {nb!r}")
        header["factors"].append([name, int(a.shape[0]), float(scale), nb, name in magnitudes])
        for x in (a, b.t()):
            q, s = _quantize_rows(x, nb)
            parts += [_pack_ints(q, nb), _tensor_bytes(s, torch.float16)]
        if name in magnitudes:
            parts.append(_tensor_bytes(magnitudes[name], dt))
    parts += [_tensor_bytes(full[name], dt) for name in header["full"]]
    return _pack(b"CWLORA2\n", header, b"".join(parts))


def _decode_lora2(theta: dict[str, torch.Tensor], header: dict, payload: memoryview, out: dict, device: str) -> dict:
    try:
        dt, factors, full = DTYPES[header["dtype"]], list(header["factors"]), list(header["full"])
    except (KeyError, TypeError) as e:
        raise ValueError("E (lora2): malformed header") from e
    if not all(isinstance(f, list) and len(f) == 5 and isinstance(f[0], str) and type(f[1]) is int and f[1] >= 1
               and type(f[2]) in (int, float) and math.isfinite(f[2]) and f[3] in LORA2_BITS and type(f[4]) is bool
               for f in factors):
        raise ValueError("E (lora2): malformed factor entries")
    names = [f[0] for f in factors]
    for what, ns in (("factors", names), ("full", full)):
        if not all(isinstance(n, str) for n in ns) or any(x >= y for x, y in zip(ns, ns[1:])):
            raise ValueError(f"E (lora2): {what} not in strictly increasing name order")
    if set(names) & set(full):
        raise ValueError("E (lora2): a tensor both in factors and in full")
    for n in names + full:
        if n not in theta:
            raise ValueError(f"E (lora2): {n} is not one of θ's tensors")
    at = 0

    def take(nbytes: int) -> bytes:
        nonlocal at
        if at + nbytes > len(payload):
            raise ValueError("E (lora2): payload too short")
        at += nbytes
        return bytes(payload[at - nbytes:at])

    def floats(count: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.frombuffer(bytearray(take(count * torch.tensor([], dtype=dtype).element_size())), dtype=dtype)

    def rows(r: int, d: int, bits: int, name: str) -> torch.Tensor:
        q = _unpack_ints(take((r * d * bits + 7) // 8), r * d, bits, name)
        return _dequantize(torch.from_numpy(q).view(r, d), floats(r, torch.float16))

    for name, r, scale, bits, dora in factors:
        w = theta[name]
        if w.ndim != 2:
            raise ValueError(f"E (lora2): {name} is not a matrix")
        a, b = rows(r, w.shape[1], bits, name), rows(r, w.shape[0], bits, name).t().contiguous()
        m = floats(w.shape[0], dt).to(device) if dora else None
        out[name] = lora_merge(w.to(device), a.to(device), b.to(device), float(scale), m).cpu()
    for name in full:
        out[name] = floats(theta[name].numel(), dt).view(theta[name].shape).to(theta[name].dtype)
    if at != len(payload):
        raise ValueError("E (lora2): trailing bytes")
    return out


_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _varints(v: np.ndarray) -> bytes:
    """Unsigned LEB128 of each value."""
    v = v.astype(np.uint64)
    k = np.ones(len(v), dtype=np.int64)
    for s in range(1, 10):
        k += v >= np.uint64(1 << (7 * s))
    owner = np.repeat(np.arange(len(v)), k)
    j = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
    low = (v[owner] >> (7 * j).astype(np.uint64)) & np.uint64(0x7F)
    return (low | ((j < k[owner] - 1).astype(np.uint64) << np.uint64(7))).astype(np.uint8).tobytes()


def _unvarints(b: bytes, count: int) -> np.ndarray:
    """`count` canonical unsigned LEB128 values that fill `b` exactly."""
    a = np.frombuffer(b, dtype=np.uint8)
    end = (a & 0x80) == 0
    if int(end.sum()) != count or (len(a) and not end[-1]):
        raise ValueError("E (sparse): the gaps don't hold their count of varints")
    if not count:
        return np.zeros(0, dtype=np.uint64)
    starts = np.flatnonzero(np.concatenate([[True], end[:-1]]))
    lengths = np.diff(np.append(starts, len(a)))
    if lengths.max() > 9 or np.any((lengths > 1) & (a[starts + lengths - 1] == 0)):
        raise ValueError("E (sparse): an overlong varint")
    j = (np.arange(len(a)) - np.repeat(starts, lengths)).astype(np.uint64)
    return np.add.reduceat((a & 0x7F).astype(np.uint64) << (np.uint64(7) * j), starts)


def _sparse_index(form: str, b: bytes, nnz: int, n: int) -> np.ndarray:
    """The sorted flat indices an index field holds: "gaps" (i_0, i_1 − i_0 − 1, …) or a "bitmap" of n bits, LSB first."""
    if form == "gaps":
        gaps = _unvarints(b, nnz)
        if nnz and int(gaps.max()) >= n:
            raise ValueError("E (sparse): an index past the tensor")
        idx = np.cumsum(gaps.astype(np.int64) + 1) - 1
    elif form == "bitmap":
        if len(b) != (n + 7) // 8:
            raise ValueError("E (sparse): a bitmap of the wrong length")
        bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")
        if bits[n:].any():
            raise ValueError("E (sparse): bitmap bits past the tensor")
        idx = np.flatnonzero(bits[:n])
    else:
        raise ValueError(f"E (sparse): unknown index form {form!r}")
    if len(idx) != nnz or (nnz and idx[-1] >= n):
        raise ValueError("E (sparse): the index doesn't hold its count of entries in the tensor")
    return idx


def encode_sparse(theta: dict[str, torch.Tensor], deltas: dict[str, tuple[torch.Tensor, torch.Tensor]],
                  dtype: str = "bf16") -> bytes:
    """E as sparse deltas: for each tensor, flat indices into θ's tensor and the values (stored in `dtype`) that decode adds
    to θ's entries in fp32, casting the sums to θ's dtype; every other entry stays θ's. Each index is stored as varint gaps
    or as a bitmap, whichever is smaller."""
    dt = DTYPES[dtype]
    header = {"dtype": dtype, "tensors": []}
    parts = []
    for name in sorted(deltas):
        idx, val = deltas[name]
        w = theta[name]
        idx = torch.as_tensor(idx).reshape(-1).to(torch.int64).cpu()
        val = torch.as_tensor(val).reshape(-1).detach().to(dt).cpu()
        if not w.is_floating_point() or idx.numel() != val.numel():
            raise ValueError(f"E (sparse): {name} needs a float tensor and one value per index")
        if not idx.numel():
            continue
        order = torch.argsort(idx)
        idx, val = idx[order].numpy(), val[order].contiguous()
        if idx[0] < 0 or idx[-1] >= w.numel() or np.any(np.diff(idx) == 0):
            raise ValueError(f"E (sparse): {name}'s indices must be distinct and inside the tensor")
        form, index = "gaps", _varints(np.diff(idx, prepend=-1) - 1)
        if (w.numel() + 7) // 8 < len(index):
            bits = np.zeros(w.numel(), dtype=bool)
            bits[idx] = True
            form, index = "bitmap", np.packbits(bits, bitorder="little").tobytes()
        header["tensors"].append([name, form, int(len(idx)), len(index)])
        parts += [index, val.view(torch.uint8).numpy().tobytes()]
    return _pack(b"CWSPAR1\n", header, b"".join(parts))


def sparse_deltas(theta: dict[str, torch.Tensor], model: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The entries where `model` differs from θ, with their deltas in fp32: `encode_sparse(θ, …, "f32")` of these decodes
    to `model` wherever an fp32 delta is exact (the update is small next to the weight)."""
    out = {}
    for k, w in theta.items():
        m = model[k].detach().cpu().to(w.dtype).reshape(-1)
        idx = torch.nonzero(m.view(_INT[w.element_size()]) != w.reshape(-1).view(_INT[w.element_size()])).reshape(-1)
        if idx.numel():
            out[k] = (idx, m[idx].float() - w.reshape(-1)[idx].float())
    return out


def _decode_sparse(theta: dict[str, torch.Tensor], header: dict, payload: memoryview, out: dict) -> dict:
    dt = DTYPES[header["dtype"]]
    size = torch.tensor([], dtype=dt).element_size()
    entries = header["tensors"]
    if not isinstance(entries, list) or not all(isinstance(t, list) and len(t) == 4 and isinstance(t[0], str)
                                                and isinstance(t[1], str) for t in entries):
        raise ValueError("E (sparse): malformed tensor entries")
    names = [t[0] for t in entries]
    if names != sorted(set(names)):
        raise ValueError("E (sparse): tensors not in increasing name order")
    at = 0
    for name, form, nnz, nb in entries:
        w = theta.get(name)
        if w is None or not w.is_floating_point():
            raise ValueError(f"E (sparse): {name} is not one of θ's float tensors")
        if type(nnz) is not int or type(nb) is not int or not 0 < nnz <= w.numel() or nb < 0:
            raise ValueError(f"E (sparse): {name}'s counts are malformed")
        if at + nb + nnz * size > len(payload):
            raise ValueError("E (sparse): payload too short")
        idx = torch.from_numpy(_sparse_index(form, bytes(payload[at:at + nb]), nnz, w.numel()))
        at += nb
        val = torch.frombuffer(bytearray(payload[at:at + nnz * size]), dtype=dt).float()
        at += nnz * size
        flat = w.detach().cpu().reshape(-1).clone()
        flat[idx] = (flat[idx].float() + val).to(w.dtype)
        out[name] = flat.view(w.shape)
    if at != len(payload):
        raise ValueError("E (sparse): trailing bytes")
    return out


@torch.no_grad()
def decode(theta: dict[str, torch.Tensor], blob: bytes, device: str = DEVICE) -> dict[str, torch.Tensor]:
    """The model E encodes from θ, every tensor in θ's dtype on the CPU."""
    out = dict(theta)
    if blob[:8] == b"CWFULL1\n":
        from safetensors.torch import load

        for k, v in load(bytes(blob[8:])).items():
            if k not in theta or v.shape != theta[k].shape or v.dtype != theta[k].dtype:
                raise ValueError(f"E (full): {k} is not one of θ's tensors with its shape and dtype")
            out[k] = v
        return out
    fmt, header, payload = _unpack(blob)
    if fmt == "sparse":
        return _decode_sparse(theta, header, payload, out)
    if fmt == "lora2":
        return _decode_lora2(theta, header, payload, out, device)
    if fmt == "lora":
        dt = DTYPES[header["dtype"]]
        size = torch.tensor([], dtype=dt).element_size()
        at = 0

        def take(shape):
            nonlocal at
            n = math.prod(shape) * size
            if at + n > len(payload):
                raise ValueError("E (lora): payload too short")
            t = torch.frombuffer(bytearray(payload[at:at + n]), dtype=dt).view(*shape)
            at += n
            return t

        for name, r, scale in header["factors"]:
            w = theta[name]
            if w.ndim != 2:
                raise ValueError(f"E (lora): {name} is not a matrix")
            a, b = take((r, w.shape[1])), take((w.shape[0], r))
            out[name] = lora_merge(w.to(device), a.to(device), b.to(device), scale).cpu()
        for name in header["full"]:
            out[name] = take(tuple(theta[name].shape)).to(theta[name].dtype)
        if at != len(payload):
            raise ValueError("E (lora): trailing bytes")
        return out
    names, seed, n = header["names"], header["seed"], header["steps"]
    coeffs = torch.frombuffer(bytearray(payload), dtype=DTYPES[header["coeff"]]).float().tolist()
    if len(coeffs) != n:
        raise ValueError("E (seedlog): coefficient count differs from its header")
    masters = [theta[k].to(device).float().clone() for k in names]
    gen = torch.Generator(device=device)
    for i, c in enumerate(coeffs):
        seedlog_step(masters, seed + i, c, gen)
    for k, m in zip(names, masters):
        out[k] = m.to(theta[k].dtype).cpu()
    return out


def format_of(blob: bytes) -> str:
    return MAGIC.get(blob[:8], "unknown")


def rebuild_macs(theta: dict[str, torch.Tensor], blob: bytes) -> int:
    """The MACs `decode(θ, E)` spends, by te_core.codecs' rule (every matmul MAC, one per element of each elementwise
    arithmetic pass), for the scorer to compare with c when it credits E as a counterexample. θ may be meta tensors.

    - full: 0 (replacement tensors);
    - lora: (r + 2)·numel per factor pair (B @ A, the scale, the add); 0 for the tensors it carries in full;
    - lora2: lora's, plus r·(d_in + d_out) to dequantize the factors and, with a magnitude, 2·numel + 2·d_out (the row
      norms, their ratios to m, the rescale);
    - seedlog: steps · Σ numel, one fused multiply-add per element per step. Its first t coefficients rebuild the weights
      after step t, at t · Σ numel;
    - sparse: nnz, one add per listed entry.

    Every format but the seed log rebuilds only the weights at the step it encodes."""
    if blob[:8] == b"CWFULL1\n":
        return 0
    fmt, header, _ = _unpack(blob)
    if fmt == "lora":
        return sum((int(r) + 2) * theta[name].numel() for name, r, _ in header["factors"])
    if fmt == "lora2":
        return sum(r * sum(theta[name].shape) + (r + 2) * theta[name].numel()
                   + (2 * (theta[name].numel() + theta[name].shape[0]) if dora else 0)
                   for name, r, _, _, dora in header["factors"])
    if fmt == "seedlog":
        return int(header["steps"]) * sum(theta[k].numel() for k in header["names"])
    return sum(int(t[2]) for t in header["tensors"])


# ---- submission --------------------------------------------------------------------------------------------------------

def submit(out: Path, start: str, klass: str, blob: bytes, model: dict[str, torch.Tensor], trainable: dict,
           steps: list[dict], changed: list[str], notes: dict | None = None, snapshots: list[tuple[bytes, int]] = ()) -> None:
    """Write a submission: E.bin, model.safetensors, steps.jsonl, meta.json, and snapshots (E at a step-log prefix) for the
    FLOPs-to-target curve."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    (out / "E.bin").write_bytes(blob)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.items()}, str(out / "model.safetensors"))
    (out / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    for i, (b, n) in enumerate(snapshots):
        d = out / "snap" / f"{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "E.bin").write_bytes(b)
        (d / "meta.json").write_text(json.dumps({"steps_prefix": n}) + "\n")
    meta = {"schema": "covert-witness/submission/v1", "start": start, "theta_digest": start_record(start)["digest"],
            "class": klass, "trainable": trainable, "changed": sorted(changed), "notes": notes or {}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


# ---- evaluation --------------------------------------------------------------------------------------------------------

def held_out(split: str, tokenizer: str) -> tuple[np.ndarray, dict]:
    man = json.loads((HELD_OUT / split / "manifest.json").read_text())
    toks = np.fromfile(HELD_OUT / split / f"{tokenizer}.bin", dtype=np.uint16)
    if hashlib.sha256(toks.tobytes()).hexdigest() != man["tokenizers"][tokenizer]["sha256"]:
        raise SystemExit(f"prepare: the {split} set's {tokenizer} tokens do not match their manifest")
    return toks, man


@torch.no_grad()
def evaluate_bpb(net: Net, split: str, tokenizer: str, micro: int = 32, device: str = DEVICE) -> dict:
    """Bits per byte of the held-out set: windows of EVAL_CTX + 1 tokens at stride EVAL_CTX over the set's stream (the last
    one shorter), every token after the first a target, the summed negative log-likelihood over the set's UTF-8 bytes."""
    toks, man = held_out(split, tokenizer)
    t = torch.from_numpy(toks.astype(np.int64))
    n = len(t)
    starts = list(range(0, n - 1, EVAL_CTX))
    full = [s for s in starts if s + EVAL_CTX + 1 <= n]
    total = torch.zeros((), dtype=torch.float64, device=device)
    net.eval()
    for i in range(0, len(full), micro):
        rows = torch.stack([t[s: s + EVAL_CTX + 1] for s in full[i: i + micro]]).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logits = net.logits(rows[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), reduction="sum").double()
    for s in starts[len(full):]:
        rows = t[s:].unsqueeze(0).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logits = net.logits(rows[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), reduction="sum").double()
    nats = float(total)
    return {"split": split, "bpb": nats / man["bytes"] / math.log(2), "nats_per_byte": nats / man["bytes"],
            "targets": n - 1, "bytes": man["bytes"], "documents": man["documents"], "tokenizer": tokenizer}


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def read_steps(path: Path) -> list[dict]:
    return [json.loads(x) for x in io.StringIO(path.read_text()) if x.strip()]
