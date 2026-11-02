"""The fixed side of the covert-witness speedrun: the registered starts, the decoders, the training-FLOP meter, the witness
checker, the classes' caps and `evaluate_bpb`. A candidate's `train.py` imports it and never edits it; `evaluate.py` runs it.

A candidate starts from a registered start θ (`load_start`), trains within its class's caps and the fixed budget, and submits
(`submit`):

- E, an encoding of its update in one of the fixed formats (`encode_full`, `encode_lora`, `encode_seedlog`); `decode(θ, E)`
  rebuilds the model, and 8·|E| bits is the update-size witness;
- the model it claims, which the checker requires to equal decode(θ, E) bit for bit, with θ's tensors, shapes and dtypes (so
  its size, evaluation FLOPs and weight bytes per byte are θ's);
- its step log: one record per pass over data (`{"op": "train"|"forward", "tokens": T, "ctx": C}`) and any matmul FLOPs it
  spends outside passes (`{"op": "extra", "flops": F}`), from which `train_flops` counts analytic training FLOPs;
- `trainable`: the tensors it trains in full and the weights it trains through rank-r adapters, which the meter prices;
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

ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
LF_CORPUS = Path(os.environ.get("CW_LF_CORPUS", "/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1"))
STARTS = ROOT / "starts"
HELD_OUT = ROOT / "eval"
EVAL_CTX = 1024
#: about 10 GPU-minutes of m's dense AdamW at its measured 1.49e14 FLOP/s (learning_frontier, RTX PRO 6000)
TRAIN_FLOP_BUDGET = 9e16
#: the candidate's train.py process, compilation included, on one GPU
GPU_SECONDS_CAP = 1200.0
CLASSES = {
    "reference": {"max_e_bytes": None, "formats": None, "top_blocks": None},
    "e1mb": {"max_e_bytes": 1_000_000, "formats": None, "top_blocks": None},
    "e100kb": {"max_e_bytes": 100_000, "formats": None, "top_blocks": None},
    "shallow": {"max_e_bytes": None, "formats": None, "top_blocks": 2},
    "seed": {"max_e_bytes": 100_000, "formats": ["seedlog"], "top_blocks": None},
}
MAGIC = {b"CWFULL1\n": "full", b"CWLORA1\n": "lora", b"CWSEED1\n": "seedlog"}
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


def load_start(name: str, device: str = "cuda") -> tuple[Net, dict, dict[str, torch.Tensor]]:
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

    def next(self, n: int, device: str = "cuda") -> torch.Tensor:
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
    (4 · r · (d_in + d_out) per token) and the attention's (twice its forward); a `forward` pass: the forward; `extra`: as
    logged."""
    nodes = positions(rec)
    by_param = {}
    for n in nodes:
        by_param.setdefault(n["param"], []).append(n)
    full = set(trainable.get("full", []))
    adapters = {k: int(v) for k, v in trainable.get("adapters", {}).items()}
    unknown = (full | set(adapters)) - set(by_param)
    if unknown:
        raise ValueError(f"trainable names no tensor of the start: {sorted(unknown)[:4]}")
    L = rec["config"]["n_layer"]
    trained = [(n["layer"], n["stage"]) for n in nodes if n["param"] in full or (n["param"] in adapters and n["flops"])]
    first = min(trained) if trained else None

    def adapter_entries(p):
        n = [x for x in by_param[p] if x["flops"]][0]
        return adapters[p] * (n["d_in"] + n["d_out"])

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
def lora_merge(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """θ's weight plus scale · B @ A, in fp32 on the weight's device without TF32, cast to the weight's dtype."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@torch.no_grad()
def decode(theta: dict[str, torch.Tensor], blob: bytes, device: str = "cuda") -> dict[str, torch.Tensor]:
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
def evaluate_bpb(net: Net, split: str, tokenizer: str, micro: int = 32, device: str = "cuda") -> dict:
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net.logits(rows[:, :-1])
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), reduction="sum").double()
    for s in starts[len(full):]:
        rows = t[s:].unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
