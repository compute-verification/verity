"""The fixed side of the covert-witness speedrun: the registered starts, the decoders, the training-FLOP meter, the witness
checker, `evaluate_bpb` and the `Scorer`. A candidate's `train.py` imports it and never edits it;
`evaluate.py` and the harness run it.

A search score (`Scorer.search`) is taken after the weights are hashed: a slice of the search pool drawn from a fresh
coin and the digest (`draw_slice`), scored paired with the start (the start's whole-pool bpb plus the candidate's minus
the start's on the slice). `Scorer.seal` writes the promotion set's score under the digest, which only `sealed.py`
reads. In final mode (`CW_FINAL=1`) every score is the final set's, and the sealed start (`register_pythia`) loads.

A candidate starts from a registered start θ (`load_start`), trains within the fixed budget, and submits (`submit`). Nothing caps
its update, its size or its shape: the training formalism v1 scores what it did (te_core.formalism). It submits:

- E, an encoding of its update in one of the fixed formats (`encode_full`, `encode_lora`, `encode_lora2`,
  `encode_seedlog`, `encode_sparse`, `encode_quant`); `decode(θ, E)` rebuilds the model, 8·|E| bits is the update-size
  witness, and `rebuild_macs` is what the rebuild costs;
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
import lzma
import math
import os
import re
import struct
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import data_v3
from sealed import DATA, SEALED, value_path
from starts_store import STARTS

#: the fixed side's version, in every witness record (README.md)
VERSION = "3.0"
ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
LF_CORPUS = Path(os.environ.get("CW_LF_CORPUS", "/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1"))
#: the held-out sets (`data_v3.py`, data version `sealed.DATA`): a search evaluation scores a fresh slice of the pool; the
#: promotion and final sets are sealed (`sealed.py`)
SEARCH_POOL, PROMOTION, FINAL = "search-pool", "promotion", "final"
HELD_OUT = ROOT / "heldout"
SET_DIRS = {SEARCH_POOL: HELD_OUT / DATA / SEARCH_POOL, PROMOTION: SEALED / "heldout" / DATA / PROMOTION,
            FINAL: SEALED / "heldout" / DATA / FINAL}
DATA_JSON = HELD_OUT / DATA / "data.json"
#: each held-out set's tokens (sha256 of `<set>/<tokenizer>.bin`) as data_v3.py built them: a set on a node that isn't
#: these is refused, so every score of this version is on the same text
SET_SHA256: dict[str, dict[str, str]] = {
    SEARCH_POOL: {"bpe32k": "1424e777b2d6f319f9662fbe0c22bdbc962074849f10779708ca76941154c5f3",
                  "gpt2": "16714461f0fdb7e352782040d097e4885a9fb195fa17e40d8264f8a1791e5295"},
    PROMOTION: {"bpe32k": "030ef6fdcccf193b5a46ec58af136cc0359fabfca6ab239c21991b43df69c969",
                "gpt2": "7bd18c31b799112d96b93abb7c440a7806bdffbba1dc4e8562b80ab33c3ed055"},
    FINAL: {"bpe32k": "b753d476cfeebd6bf7b71ede9604158d10e0de834798d9646150b8fd4a8d38e4",
            "gpt2": "c1af95122572788a397c408976501fb5180df04c859e91000831c03d946c6570",
            "pythia": "ffc8b77142d72143dd5f0225ce0d497b4e52aed2fca6143bd8410f1f669c8464"},
}
#: the search's starts as `starts_store.json` records them (digest, files, shape)
STORE = json.loads((Path(__file__).resolve().parent / "starts_store.json").read_text())["starts"]
#: each start's θ digest (`digest`): the search's starts', and the sealed start's
START_DIGESTS = {name: s["digest"] for name, s in STORE.items()} | {
    "pythia160m": "e2b4a7f33b5c9578631b91ca13f59fb7a3648be4aa35e1f892b26939e8ab1681"}
#: a search slice's GPT-2 tokens, whole windows of the pool's stream (`draw_slice`), and the promotion set's that a sealed
#: value scores (its first windows): with the start's scores kept, a checkpoint costs about one 2^21-token evaluation
SLICE_TOKENS = 2 ** 20
PROMOTION_TOKENS = 2 ** 20
#: what a candidate's train.py may not read (the harness's audit hook): the held-out sets (v3's raw downloads are built
#: under them), the sealed side, v2.0's sets, learning_frontier's validation stream and hub cache (the search pool's
#: tokens, and the held-out shards' raw text), and v3's ladder validation stream
UNREADABLE = (HELD_OUT, SEALED, ROOT / "eval", LF_CORPUS / "val.bin", LF_CORPUS / "hf-cache", data_v3.TRAIN / "val.bin")
#: final scoring (the scorer's, after the search): the harness scores the final set, and the sealed start loads
FINAL_MODE = os.environ.get("CW_FINAL") == "1"
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
#: the run's tier (`budget`): search, or long (`launch.py --long`, only for a run the board assigns as long), whose budget
#: is LONG_C0 times the start's own training compute and LONG_GPU_SECONDS of its steps
TIER = os.environ.get("CW_TIER", "search")
LONG_C0 = 100
LONG_GPU_SECONDS = 30 * 3600.0
MAGIC = {b"CWFULL1\n": "full", b"CWLORA1\n": "lora", b"CWLORA2\n": "lora2", b"CWSEED1\n": "seedlog",
         b"CWSPAR1\n": "sparse", b"CWQUAN1\n": "quant", b"CWMODL1\n": "model"}
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
EOT = {"gpt2": 50256, "pythia": 0, data_v3.TOKENIZER: data_v3.EOT[data_v3.TOKENIZER]}
PYTHIA_REPO = "EleutherAI/pythia-160m"
START_NAME = re.compile(r"^[a-z0-9_-]+$")


# ---- starts ------------------------------------------------------------------------------------------------------------

def digest(theta: dict[str, torch.Tensor]) -> str:
    """sha256 over the tensors in name order: each name, dtype, shape and its little-endian bytes."""
    h = hashlib.sha256()
    for name in sorted(theta):
        t = theta[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def register_start(name: str, theta: dict[str, torch.Tensor], config: dict, tokenizer: str, provenance: dict,
                   root: Path = STARTS) -> dict:
    from safetensors.torch import save_file

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in theta.items()}, str(d / "theta.safetensors.tmp"))
    (d / "theta.safetensors.tmp").rename(d / "theta.safetensors")
    rec = {"schema": "covert-witness/start/v1", "name": name, "digest": digest(theta), "config": config,
           "tokenizer": tokenizer, "parameters": int(sum(v.numel() for v in theta.values())),
           "tensors": {k: [str(v.dtype), list(v.shape)] for k, v in sorted(theta.items())}, "provenance": provenance}
    (d / "start.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def start_dir(name: str) -> Path:
    """A registered start's directory: STARTS's, or in final mode the sealed start's."""
    if not START_NAME.match(name):
        raise SystemExit(f"prepare: {name!r} is not a start name")
    for d in (STARTS, *([SEALED / "starts"] if FINAL_MODE else [])):
        if (d / name / "start.json").is_file():
            return d / name
    raise SystemExit(f"prepare: no start {name} in {STARTS} (install them with `python3 starts_store.py install --keep`)")


def start_record(name: str) -> dict:
    rec = json.loads((start_dir(name) / "start.json").read_text())
    if rec["digest"] != START_DIGESTS.get(name):
        raise SystemExit(f"prepare: start {name}'s registered digest is not prepare.START_DIGESTS's")
    return rec


def c0_flops(name: str) -> float:
    """A search start's own training compute in FLOPs, 2x its MACs: its shape's counted `c0_macs` (`starts_store.json`),
    else three forward passes' MACs per training token (te_core.policy.c0's stub)."""
    if name not in STORE:
        raise SystemExit(f"prepare: {name} is not a search start (starts_store.json), so it has no C0")
    shape = STORE[name]["shape"]
    return 2.0 * float(shape.get("c0_macs") or 3 * shape["forward_macs_per_token"] * shape["tokens"])


def budget(name: str, tier: str | None = None) -> dict:
    """A training job's budget on start `name` at `tier` (default TIER): its FLOPs, 2x the MACs the harness counts (within
    HARNESS_FLOP_SLACK), and its steps' wall seconds. search: TRAIN_FLOP_BUDGET and GPU_SECONDS_CAP; long: LONG_C0 x
    `c0_flops` and LONG_GPU_SECONDS."""
    tier = tier or TIER
    if tier == "search":
        return {"tier": tier, "train_flops": TRAIN_FLOP_BUDGET, "gpu_seconds": GPU_SECONDS_CAP}
    if tier == "long":
        c0 = c0_flops(name)
        return {"tier": tier, "train_flops": LONG_C0 * c0, "gpu_seconds": LONG_GPU_SECONDS, "c0_flops": c0,
                "c0_multiple": LONG_C0}
    raise SystemExit(f"prepare: no tier {tier!r} (search or long)")


def load_theta(name: str, check: bool = True) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    theta = load_file(str(start_dir(name) / "theta.safetensors"))
    if check and digest(theta) != start_record(name)["digest"]:
        raise SystemExit(f"prepare: start {name}'s weights do not match its registered digest")
    return theta


class Net(torch.nn.Module):
    """A start's architecture: `logits(idx)`, and `named_parameters()` named as θ's tensors."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        if config["family"] == "lf":
            from model import LM, VOCAB, Size

            self.inner = LM(Size(config["name"], config["d_model"], config["n_layer"], config["n_head"], config["d_ff"]),
                            vocab=config.get("vocab", VOCAB))
        elif config["family"] == "hf-neox":
            from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

            cfg = GPTNeoXConfig(**config["hf"])
            cfg._attn_implementation = "sdpa"
            self.inner = GPTNeoXForCausalLM(cfg)
        else:
            raise SystemExit(f"prepare: no architecture for family {config['family']!r}")
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


def register_pythia(name: str = "pythia160m") -> dict:
    """The sealed start: Pythia-160M from the hub at its current revision (recorded), its weights cast to fp32 (exact),
    the checkpoint's buffers dropped, into SEALED/starts, which loads only in final mode. It trained on the Pile, so the
    held-out text is a domain shift for it, and its architecture (GPT-NeoX: biases, a parallel residual, an untied head)
    is not the search's."""
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(PYTHIA_REPO).sha
    d = Path(snapshot_download(PYTHIA_REPO, revision=sha, allow_patterns=["config.json", "*.safetensors", "*.bin"]))
    hf = json.loads((d / "config.json").read_text())
    config = {"family": "hf-neox", "name": name, "d_model": hf["hidden_size"], "n_layer": hf["num_hidden_layers"],
              "n_head": hf["num_attention_heads"], "d_ff": hf["intermediate_size"], "vocab": hf["vocab_size"],
              "ctx": EVAL_CTX, "tied": bool(hf.get("tie_word_embeddings", False)), "hf": hf}
    if (d / "model.safetensors").is_file():
        from safetensors.torch import load_file

        raw, source = load_file(str(d / "model.safetensors")), "model.safetensors"
    else:
        raw, source = torch.load(d / "pytorch_model.bin", map_location="cpu", weights_only=True), "pytorch_model.bin"
    with torch.device("cpu"):
        net = Net(config)
    names = set(net.params())
    for old, new in (("embed_out.weight", "lm_head.weight"), ("lm_head.weight", "embed_out.weight")):
        if new in names and new not in raw and old in raw:
            raw[new] = raw.pop(old)
    if not names <= set(raw):
        raise SystemExit(f"prepare: Pythia's checkpoint lacks {sorted(names - set(raw))[:4]}")
    theta = {k: raw[k].float() for k in names}
    return register_start(name, theta, config, "pythia", {
        "repo": PYTHIA_REPO, "revision": sha, "file": source, "stored_dtypes": sorted({str(raw[k].dtype) for k in names}),
        "cast": "float32 (exact)", "dropped": sorted(set(raw) - names)}, root=SEALED / "starts")


# ---- data --------------------------------------------------------------------------------------------------------------

class TrainData:
    """The start's tokenizer's training stream cut into ctx + 1 token windows at stride ctx, permuted by `seed`.

    gpt2: learning_frontier's shards 000-002 (the lf starts read the first 8 tokens per parameter of its seed-0 order);
    pythia: shard 000's documents (`data.py`); bpe32k: data v3's stream (`data_v3.py`), its parts' windows (each inside
    one part) as one sequence: the first `parts` parts, else all of manifest.json's, else (while it is being installed)
    those whose records are in place from part 000 on. The order depends on how many, so a run that must reproduce
    another's names `parts`."""

    def __init__(self, tokenizer: str, seed: int = 1, ctx: int = EVAL_CTX, parts: int | None = None):
        if tokenizer == data_v3.TOKENIZER:
            d = data_v3.TRAIN
            if (d / "manifest.json").is_file():
                index = [p["index"] for p in json.loads((d / "manifest.json").read_text())["parts"]]
            else:
                index = []
                while (d / "parts" / f"{len(index):03d}.json").is_file():
                    index.append(len(index))
            if parts is not None:
                if parts > len(index):
                    raise SystemExit(f"prepare: data v3 has {len(index)} training parts here, not {parts}")
                index = index[:parts]
            if not index:
                raise SystemExit(f"prepare: no data v3 training parts in {d / 'parts'}")
            self.parts = [np.memmap(d / "parts" / f"{i:03d}.bin", dtype=np.uint16, mode="r") for i in index]
        else:
            path = LF_CORPUS / "train.bin" if tokenizer == "gpt2" else ROOT / "train" / f"{tokenizer}.bin"
            self.parts = [np.memmap(path, dtype=np.uint16, mode="r")]
        self.ctx = ctx
        self.starts = np.cumsum([0] + [(len(p) - 1) // ctx for p in self.parts])
        self.order = np.random.default_rng(seed).permutation(int(self.starts[-1]))
        self.at = 0

    def next(self, n: int, device: str = DEVICE) -> torch.Tensor:
        if self.at + n > len(self.order):
            self.at = 0
        idx = np.sort(self.order[self.at: self.at + n])
        self.at += n
        c = self.ctx
        part = np.searchsorted(self.starts, idx, side="right") - 1
        rows = np.stack([self.parts[p][(i - self.starts[p]) * c: (i - self.starts[p]) * c + c + 1]
                         for p, i in zip(part, idx)]).astype(np.int64)
        return torch.from_numpy(rows).to(device, non_blocking=True)


# ---- positions, φ and the training-FLOP meter ----------------------------------------------------------------------

#: per family: the lookup, the head (the lookup's name when tied), the final norm's tensors, a block tensor's name (block,
#: module) and the modules' stages. GPT-NeoX's parallel residual feeds its MLP from the block's input, not the attention's
#: output; its MLP is placed after the attention as in the sequential order, so its φ is an upper bound. Its head is
#: `embed_out` in transformers 4 and `lm_head` from 5 (the start is registered under the installed one's name).
FAMILIES = {
    "lf": {"embed": "embed", "head": "embed", "final": r"norm", "block": r"blocks\.(\d+)\.(\w+?)(?:\.weight)?",
           "stages": {"attn_norm": 0, "wq": 1, "wk": 1, "wv": 1, "wo": 2, "mlp_norm": 3, "w_gate": 4, "w_up": 4,
                      "w_down": 5}},
    "hf-neox": {"embed": r"gpt_neox\.embed_in\.weight", "head": r"(?:embed_out|lm_head)\.weight",
                "final": r"gpt_neox\.final_layer_norm\.(?:weight|bias)",
                "block": r"gpt_neox\.layers\.(\d+)\.(.+)\.(?:weight|bias)",
                "stages": {"input_layernorm": 0, "attention.query_key_value": 1, "attention.dense": 2,
                           "post_attention_layernorm": 3, "mlp.dense_h_to_4h": 4, "mlp.dense_4h_to_h": 5}},
}


def positions(rec: dict) -> list[dict]:
    """Every node of the forward: {param, layer, stage, flops (forward weight-matmul FLOPs per token), d_in, d_out}."""
    L = rec["config"]["n_layer"]
    fam = FAMILIES[rec["config"]["family"]]
    nodes = []
    for name, (_, shape) in rec["tensors"].items():
        is_embed, is_head = re.fullmatch(fam["embed"], name), re.fullmatch(fam["head"], name)
        if is_embed or is_head:
            if is_embed:
                nodes.append({"param": name, "layer": 0, "stage": 0, "flops": 0, "d_in": shape[0], "d_out": shape[1]})
            if is_head:
                nodes.append({"param": name, "layer": L + 1, "stage": 1, "flops": 2 * shape[0] * shape[1],
                              "d_in": shape[1], "d_out": shape[0]})
            continue
        if re.fullmatch(fam["final"], name):
            nodes.append({"param": name, "layer": L + 1, "stage": 0, "flops": 0, "d_in": shape[0], "d_out": shape[0]})
            continue
        m = re.fullmatch(fam["block"], name)
        if not m or m.group(2) not in fam["stages"]:
            raise SystemExit(f"prepare: {name} has no position")
        stage = fam["stages"][m.group(2)]
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


def encode_model(config: dict, tensors: dict[str, torch.Tensor]) -> bytes:
    """E as a model of its own architecture: `config` (θ's family, vocabulary and context, any shape: grown, pruned,
    distilled) and the tensors θ doesn't give it. Each of the architecture's tensors is E's if E holds it, else θ's of the
    same name, shape and dtype, so every new parameter is in 8·|E|."""
    from safetensors.torch import save

    return _pack(b"CWMODL1\n", {"config": config},
                 save({k: v.detach().cpu().contiguous() for k, v in sorted(tensors.items())}))


def architecture(config: dict) -> dict[str, tuple[int, ...]]:
    """The tensors (name: shape) of a model of `config`, as `Net` names them."""
    with torch.device("meta"):
        net = Net(config)
    return {k: tuple(v.shape) for k, v in net.params().items()}


def model_config(rec: dict, blob: bytes) -> dict:
    """The config of the model E encodes: θ's, or a model format's own, which must keep θ's family, vocabulary, context
    and tying (the tokenizer and the scoring are the start's)."""
    if blob[:8] != b"CWMODL1\n":
        return rec["config"]
    config = _unpack(blob)[1].get("config")
    keep = ("family", "vocab", "ctx", "tied")
    if not isinstance(config, dict) or config.get("family") != "lf" or any(
            config.get(k) != rec["config"].get(k) for k in keep):
        raise ValueError(f"E (model): its config must be an lf model with θ's {', '.join(keep)}")
    return config


def model_record(rec: dict, config: dict, state: dict[str, torch.Tensor]) -> dict:
    """A start record's fields `positions`, `deep_share`, `train_flops` and `eval_cost` read, for the model of `state`."""
    if config == rec["config"]:
        return rec
    return {"config": config, "parameters": int(sum(v.numel() for v in state.values())),
            "tensors": {k: [str(v.dtype), list(v.shape)] for k, v in sorted(state.items())}}


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


def declare_step(base: bytes = b"", suffixes: tuple[bytes, ...] = ()) -> None:
    """Declare the weights the training step in progress reads, as encodings from θ the harness credits as a
    counterexample (te_core/cw.py): a weight read at one value is decode(θ, base)'s (θ's for an empty base), one read at
    several is decode(decode(θ, base), suffixes[i])'s at its i-th, in the order the step reads them. The rebuild must cost
    at most c MACs (4096 x the start's forward MACs per token). Encode the base before the suffixes: the first E encoded in
    each 5% of the run is scored. A no-op outside the harness."""


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


class QArray(NamedTuple):
    """`bits`-bit codes (1 to 8) into an f16 codebook of 2**bits entries, times an f16 scale per group of `group`
    consecutive entries in row-major order: entry i is codebook[codes[i]] · scales[i // group], exact in fp32."""

    codebook: torch.Tensor
    scales: torch.Tensor
    codes: torch.Tensor
    bits: int
    group: int
    shape: tuple[int, ...]


QUANT_XZ = {"format": lzma.FORMAT_XZ, "check": lzma.CHECK_NONE, "preset": 1}
QUANT_KEYS = {"delta": {"n", "k", "b", "g"}, "lora": {"n", "k", "r", "s", "A", "B"}, "full": {"n", "k"}}


def _pack_codes(codes: np.ndarray, bits: int) -> bytes:
    if bits == 8:
        return codes.astype(np.uint8).tobytes()
    flat = ((codes.astype(np.uint8)[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(flat, bitorder="little").tobytes()


def _unpack_codes(b: bytes, count: int, bits: int, name: str) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(b, dtype=np.uint8).copy()
    flat = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")
    if flat[count * bits:].any():
        raise ValueError(f"E (quant): {name}: a padding bit is not zero")
    return (flat[:count * bits].reshape(count, bits) << np.arange(bits, dtype=np.uint8)).sum(1, dtype=np.uint8)


def _quant_array(q, name: str) -> tuple[dict, list[bytes]]:
    bits, group, shape = int(q.bits), int(q.group), tuple(int(s) for s in q.shape)
    n = math.prod(shape)
    cb = torch.as_tensor(q.codebook).detach().reshape(-1).cpu()
    sc = torch.as_tensor(q.scales).detach().reshape(-1).cpu()
    codes = torch.as_tensor(q.codes).detach().reshape(-1).cpu().long()
    if not 1 <= bits <= 8 or group < 1 or n % group or cb.numel() != 2 ** bits or sc.numel() != n // group \
            or codes.numel() != n or (n and not 0 <= int(codes.min()) <= int(codes.max()) < 2 ** bits):
        raise ValueError(f"E (quant): {name}'s codes, codebook and scales don't fit {bits} bits in groups of {group} "
                         f"for {shape}")
    cb, sc = cb.to(torch.float16), sc.to(torch.float16)
    if not (torch.isfinite(cb).all() and torch.isfinite(sc).all()):
        raise ValueError(f"E (quant): {name}'s codebook and scales must be finite in f16")
    return {"b": bits, "g": group}, [cb.view(torch.uint8).numpy().tobytes(), sc.view(torch.uint8).numpy().tobytes(),
                                     _pack_codes(codes.to(torch.uint8).numpy(), bits)]


def encode_quant(entries: dict[str, dict], theta: dict[str, torch.Tensor]) -> bytes:
    """E as low-bit codes (CWQUAN1, the quant island's qformat.py): b"CWQUAN1\\n" + xz(4-byte header length, a JSON header,
    each array's bytes in order). `entries` maps each changed tensor of θ to one of

    - {"kind": "delta", "q": QArray}: W' = (W.float() + Q).to(W.dtype), Q the QArray's values in W's shape, so the rebuild
      is the same on every device;
    - {"kind": "lora", "a": QArray, "b": QArray, "scale": s}: W' = lora_merge(W, Q_A, Q_B, s), Q_A r x d_in, Q_B d_out x r;
    - {"kind": "full", "t": tensor}: the tensor itself, in θ's dtype.

    A QArray is `QArray` or anything with its fields (qformat's). Codes are packed `bits` wide, least significant bit
    first, and 8·|E| counts every byte after the entropy coder (xz preset 1, no check)."""
    header, parts = [], []
    for name in sorted(entries):
        e, w = entries[name], theta[name]
        if e["kind"] == "delta":
            if tuple(e["q"].shape) != tuple(w.shape) or not w.is_floating_point():
                raise ValueError(f"E (quant): {name}: a delta of shape {tuple(e['q'].shape)} for a {tuple(w.shape)} tensor")
            spec, arrays = _quant_array(e["q"], name)
            header.append({"n": name, "k": "delta", **spec})
            parts += arrays
        elif e["kind"] == "lora":
            a, b, r = e["a"], e["b"], int(e["a"].shape[0])
            if w.ndim != 2 or tuple(a.shape) != (r, w.shape[1]) or tuple(b.shape) != (w.shape[0], r) or not 1 <= r <= min(w.shape):
                raise ValueError(f"E (quant): {name}: factors of shapes {tuple(a.shape)}, {tuple(b.shape)} for {tuple(w.shape)}")
            (sa, pa), (sb, pb) = _quant_array(a, name), _quant_array(b, name)
            header.append({"n": name, "k": "lora", "r": r, "s": float(e["scale"]), "A": sa, "B": sb})
            parts += pa + pb
        elif e["kind"] == "full":
            header.append({"n": name, "k": "full"})
            parts.append(e["t"].detach().to(w.dtype).cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            raise ValueError(f"E (quant): {name}: unknown kind {e['kind']!r}")
    h = json.dumps({"t": header}, separators=(",", ":")).encode()
    return b"CWQUAN1\n" + lzma.compress(struct.pack("<I", len(h)) + h + b"".join(parts), **QUANT_XZ)


def _quant_header(theta: dict[str, torch.Tensor], blob: bytes) -> tuple[list[dict], memoryview]:
    """CWQUAN1's checked entries and payload. The decompressed size is bounded by θ's (a header and, per tensor, its
    codebooks, scales and codes at 8 bits in groups of 1, or its bytes in full)."""
    cap = (1 << 20) + sum(8 * w.numel() + 2048 for w in theta.values())
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    try:
        inner = dec.decompress(bytes(blob[8:]), max_length=cap)
    except lzma.LZMAError as e:
        raise ValueError(f"E (quant): not an xz stream ({e})") from e
    if not dec.eof or dec.unused_data:
        raise ValueError("E (quant): not one xz stream holding at most θ's size")
    if len(inner) < 4 or 4 + struct.unpack("<I", inner[:4])[0] > len(inner):
        raise ValueError("E (quant): payload too short")
    n = struct.unpack("<I", inner[:4])[0]
    try:
        header = json.loads(inner[4:4 + n])
    except ValueError as e:
        raise ValueError("E (quant): malformed header") from e
    entries = header.get("t") if isinstance(header, dict) and set(header) == {"t"} else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) and e.get("k") in QUANT_KEYS and set(e) == QUANT_KEYS[e["k"]]
                                                and isinstance(e["n"], str) for e in entries):
        raise ValueError("E (quant): malformed entries")
    names = [e["n"] for e in entries]
    if any(x >= y for x, y in zip(names, names[1:])):
        raise ValueError("E (quant): entries not in strictly increasing name order")
    for e in entries:
        w = theta.get(e["n"])
        if w is None:
            raise ValueError(f"E (quant): {e['n']} is not one of θ's tensors")
        specs = [({"b": e["b"], "g": e["g"]}, tuple(w.shape))] if e["k"] == "delta" else []
        if e["k"] == "lora":
            r, s = e["r"], e["s"]
            if w.ndim != 2 or type(r) is not int or not 1 <= r <= min(w.shape) or type(s) not in (int, float) \
                    or not math.isfinite(s):
                raise ValueError(f"E (quant): {e['n']}'s rank or scale is malformed for a {tuple(w.shape)} tensor")
            specs = [(e["A"], (r, w.shape[1])), (e["B"], (w.shape[0], r))]
        if e["k"] != "full" and not w.is_floating_point():
            raise ValueError(f"E (quant): {e['n']} is not one of θ's float tensors")
        for spec, shape in specs:
            if not isinstance(spec, dict) or set(spec) != {"b", "g"} or type(spec["b"]) is not int \
                    or type(spec["g"]) is not int or not 1 <= spec["b"] <= 8 or spec["g"] < 1 or math.prod(shape) % spec["g"]:
                raise ValueError(f"E (quant): {e['n']}'s bits or group is malformed for {shape}")
    return entries, memoryview(inner)[4 + n:]


def _decode_quant(theta: dict[str, torch.Tensor], blob: bytes, out: dict, device: str) -> dict:
    entries, payload = _quant_header(theta, blob)
    at = 0

    def take(nbytes: int) -> bytes:
        nonlocal at
        if at + nbytes > len(payload):
            raise ValueError("E (quant): payload too short")
        at += nbytes
        return bytes(payload[at - nbytes:at])

    def values(spec: dict, shape: tuple[int, ...], name: str) -> torch.Tensor:
        bits, group, n = spec["b"], spec["g"], math.prod(shape)
        cb = torch.frombuffer(bytearray(take(2 * 2 ** bits)), dtype=torch.float16)
        sc = torch.frombuffer(bytearray(take(2 * (n // group))), dtype=torch.float16)
        if not (torch.isfinite(cb).all() and torch.isfinite(sc).all()):
            raise ValueError(f"E (quant): {name}'s codebook and scales must be finite")
        codes = torch.from_numpy(_unpack_codes(take((n * bits + 7) // 8), n, bits, name)).to(device).long()
        v = cb.float().to(device)[codes].view(-1, group) * sc.float().to(device)[:, None]
        return v.view(shape)

    for e in entries:
        name = e["n"]
        w = theta[name]
        if e["k"] == "delta":
            out[name] = (w.to(device).float() + values(e, tuple(w.shape), name)).to(w.dtype).cpu()
        elif e["k"] == "lora":
            r = e["r"]
            a, b = values(e["A"], (r, w.shape[1]), name), values(e["B"], (w.shape[0], r), name)
            out[name] = lora_merge(w.to(device), a, b, float(e["s"])).cpu()
        else:
            out[name] = torch.frombuffer(bytearray(take(w.numel() * w.element_size())), dtype=w.dtype).view(w.shape).clone()
    if at != len(payload):
        raise ValueError("E (quant): trailing bytes")
    return out


@torch.no_grad()
def decode(theta: dict[str, torch.Tensor], blob: bytes, device: str = DEVICE) -> dict[str, torch.Tensor]:
    """The model E encodes from θ, on the CPU: θ's tensors in θ's dtypes, or a model format's architecture's tensors."""
    out = dict(theta)
    if blob[:8] == b"CWFULL1\n":
        from safetensors.torch import load

        for k, v in load(bytes(blob[8:])).items():
            if k not in theta or v.shape != theta[k].shape or v.dtype != theta[k].dtype:
                raise ValueError(f"E (full): {k} is not one of θ's tensors with its shape and dtype")
            out[k] = v
        return out
    if blob[:8] == b"CWQUAN1\n":
        return _decode_quant(theta, blob, out, device)
    fmt, header, payload = _unpack(blob)
    if fmt == "model":
        from safetensors.torch import load

        shapes = architecture(header["config"])
        given = load(bytes(payload))
        if set(given) - set(shapes):
            raise ValueError(f"E (model): {sorted(set(given) - set(shapes))[:4]} are not its architecture's tensors")
        model = {}
        for k, shape in shapes.items():
            v = given.get(k, theta.get(k))
            if v is None or tuple(v.shape) != shape or v.dtype not in DTYPES.values():
                raise ValueError(f"E (model): {k} is neither E's nor θ's with its shape {shape}")
            model[k] = v
        return model
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
    coeffs = torch.frombuffer(bytearray(payload), dtype=DTYPES[header["coeff"]]).float().tolist() if len(payload) else []
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

    - full and model: 0 (replacement tensors);
    - lora: (r + 2)·numel per factor pair (B @ A, the scale, the add); 0 for the tensors it carries in full;
    - lora2: lora's, plus r·(d_in + d_out) to dequantize the factors and, with a magnitude, 2·numel + 2·d_out (the row
      norms, their ratios to m, the rescale);
    - seedlog: steps · Σ numel, one fused multiply-add per element per step. Its first t coefficients rebuild the weights
      after step t, at t · Σ numel;
    - sparse: nnz, one add per listed entry;
    - quant: 2·numel per delta (the scale, the add); a lora entry's r·(d_in + d_out) (the factors' scales) plus lora's
      (r + 2)·numel; 0 for a tensor in full.

    Every format but the seed log rebuilds only the weights at the step it encodes."""
    if blob[:8] == b"CWFULL1\n":
        return 0
    if blob[:8] == b"CWQUAN1\n":
        return sum(2 * theta[e["n"]].numel() if e["k"] == "delta" else
                   e["r"] * sum(theta[e["n"]].shape) + (e["r"] + 2) * theta[e["n"]].numel() if e["k"] == "lora" else 0
                   for e in _quant_header(theta, blob)[0])
    fmt, header, _ = _unpack(blob)
    if fmt == "model":
        return 0
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

def submit(out: Path, start: str, blob: bytes, model: dict[str, torch.Tensor], trainable: dict, steps: list[dict],
           changed: list[str], notes: dict | None = None, snapshots: list[tuple[bytes, int]] = ()) -> None:
    """Write a submission: E.bin, steps.jsonl, meta.json, and snapshots (E at a step-log prefix) for the FLOPs-to-target
    curve. `model` isn't written: the evaluation job rebuilds the model from θ and E with the fixed decoder."""
    if isinstance(blob, str):
        raise SystemExit(f"prepare: harness v3 has no classes: call submit(out, start, blob, model, trainable, steps, "
                         f"changed, ...) without the class ({blob!r})")
    out.mkdir(parents=True, exist_ok=True)
    (out / "E.bin").write_bytes(blob)
    (out / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    for i, (b, n) in enumerate(snapshots):
        d = out / "snap" / f"{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "E.bin").write_bytes(b)
        (d / "meta.json").write_text(json.dumps({"steps_prefix": n}) + "\n")
    meta = {"schema": "covert-witness/submission/v1", "start": start, "theta_digest": start_record(start)["digest"],
            "trainable": trainable, "changed": sorted(changed), "notes": notes or {}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


# ---- evaluation --------------------------------------------------------------------------------------------------------

def manifest(split: str) -> dict:
    return json.loads((SET_DIRS[split] / "manifest.json").read_text())


def held_out(split: str, tokenizer: str) -> tuple[np.ndarray, dict]:
    man = manifest(split)
    toks = np.fromfile(SET_DIRS[split] / f"{tokenizer}.bin", dtype=np.uint16)
    got = hashlib.sha256(toks.tobytes()).hexdigest()
    if got != man["tokenizers"][tokenizer]["sha256"]:
        raise SystemExit(f"prepare: the {split} set's {tokenizer} tokens do not match their manifest")
    if got != SET_SHA256.get(split, {}).get(tokenizer):
        raise SystemExit(f"prepare: the {split} set's {tokenizer} tokens are not prepare.SET_SHA256's (data {DATA})")
    return toks, man


def windows(n: int) -> tuple[list[int], list[int]]:
    """The evaluation windows of a stream of n tokens, by their first token: EVAL_CTX + 1 tokens at stride EVAL_CTX, every
    token after a window's first a target (so each token after the stream's first is a target once); the full windows,
    and the shorter last one."""
    starts = list(range(0, n - 1, EVAL_CTX))
    full = [s for s in starts if s + EVAL_CTX + 1 <= n]
    return full, starts[len(full):]


def token_bytes(tokenizer: str) -> np.ndarray:
    """Each token id's UTF-8 bytes, end-of-text 0, for GPT-2 and bpe32k (`data_v3.token_bytes`): both BPEs are byte-level,
    so a document's tokens' bytes are its text's (`data.py` encodes with `encode_ordinary`, `data_v3.py` with special
    tokens encoded as text)."""
    if tokenizer == data_v3.TOKENIZER:
        return data_v3.token_bytes()
    if tokenizer != "gpt2":
        raise SystemExit(f"prepare: token bytes are GPT-2's and bpe32k's, not {tokenizer}'s")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out = np.array([len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)], dtype=np.int64)
    out[EOT["gpt2"]] = 0
    return out


def draw_slice(weights_digest: str, coin: str | None = None, tokenizer: str = "gpt2") -> dict:
    """A fresh slice of the search pool for weights already hashed to `weights_digest`: a live coin (16 bytes from the
    OS, drawn now), and SLICE_TOKENS / EVAL_CTX of the pool's full windows (`windows`, by index, in the tokenizer's
    stream) drawn without replacement by a generator seeded with sha256(coin ‖ digest). The recorded coin (hex), the
    digest and the tokenizer reproduce it."""
    full, _ = windows(manifest(SEARCH_POOL)["tokenizers"][tokenizer]["tokens"])
    raw = os.urandom(16) if coin is None else bytes.fromhex(coin)
    seed = int.from_bytes(hashlib.sha256(raw + weights_digest.encode()).digest(), "little")
    pick = np.random.default_rng(seed).choice(len(full), size=min(SLICE_TOKENS // EVAL_CTX, len(full)), replace=False)
    return {"coin": raw.hex(), "weights_digest": weights_digest, "windows": sorted(int(i) for i in pick)}


@torch.no_grad()
def window_nats(net: Net, toks: np.ndarray, starts: list[int], micro: int = 32, device: str = DEVICE) -> np.ndarray:
    """Each window's summed negative log-likelihood of its targets, in nats (the windows of `toks` that begin at `starts`;
    one running past the stream's end is shorter)."""
    t = torch.from_numpy(toks.astype(np.int64))
    out = torch.zeros(len(starts), dtype=torch.float64, device=device)
    full = [i for i, s in enumerate(starts) if s + EVAL_CTX + 1 <= len(t)]
    net.eval()
    for j in range(0, len(full), micro):
        idx = full[j: j + micro]
        rows = torch.stack([t[starts[i]: starts[i] + EVAL_CTX + 1] for i in idx]).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logits = net.logits(rows[:, :-1])
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), reduction="none")
        out[torch.tensor(idx, device=device)] = nll.view(len(idx), -1).double().sum(1)
    for i in sorted(set(range(len(starts))) - set(full)):
        rows = t[starts[i]:].unsqueeze(0).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logits = net.logits(rows[:, :-1])
        out[i] = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1),
                                 reduction="none").double().sum()
    return out.cpu().numpy()


def evaluate_bpb(net: Net, split: str, tokenizer: str, micro: int = 32, device: str = DEVICE) -> dict:
    """Bits per byte of a held-out set: every window's negative log-likelihood (`windows`) over the documents' UTF-8
    bytes."""
    toks, man = held_out(split, tokenizer)
    full, tail = windows(len(toks))
    nats = float(window_nats(net, toks, full + tail, micro, device).sum())
    return {"split": split, "bpb": nats / man["bytes"] / math.log(2), "nats_per_byte": nats / man["bytes"],
            "targets": len(toks) - 1, "bytes": man["bytes"], "documents": man["documents"], "tokenizer": tokenizer}


class Scorer:
    """The fixed scores of models from one start, the start θ their pairing base, each model in θ's architecture or its
    own (`config`, a model format's: `model_config`). Both the harness (at every checkpoint) and `evaluate.py` (the
    submission) score through it.

    - `start_bpb`: the start's bpb on the whole search pool (in final mode, the final set), each of the pool's windows'
      negative log-likelihood under the start kept.
    - `search(state, digest, config)`: the model's search score on a fresh slice drawn after its weights were hashed to
      `digest` (`draw_slice`), paired: start_bpb plus the model's bpb on the slice's windows less the start's on the same
      windows, which cancels most of the slice's own difficulty. In final mode, the model's bpb on the final set.
    - `seal(key, state, search, config)`: the model's and the start's bpb on the promotion set's first PROMOTION_TOKENS, with
      `search`'s scores, written once to the sealed values (`sealed.py`) under `key`, the model's output digest; nothing
      returns them. A no-op in final mode."""

    def __init__(self, rec: dict, theta: dict[str, torch.Tensor], device: str = DEVICE):
        self.rec, self.tok, self.device = rec, rec["tokenizer"], device
        with torch.device(device):
            self.base, self.net = Net(rec["config"]), Net(rec["config"])
        self.nets = {json.dumps(rec["config"], sort_keys=True): self.net}
        self.base.load(theta)
        self.start_promotion_bpb: float | None = None
        self.promotion: tuple[np.ndarray, list[int], int] | None = None
        if FINAL_MODE:
            self.split = FINAL
            self.start_bpb = evaluate_bpb(self.base, FINAL, self.tok, device=device)["bpb"]
            return
        self.split = SEARCH_POOL
        self.pool, man = held_out(SEARCH_POOL, self.tok)
        self.token_bytes = token_bytes(self.tok)
        at = np.concatenate([[0], np.cumsum(self.token_bytes[self.pool])])
        if int(at[-1] - at[1]) != man["bytes"]:
            raise SystemExit("prepare: the search pool's tokens' bytes are not its documents' bytes")
        self.full, tail = windows(len(self.pool))
        nats = window_nats(self.base, self.pool, self.full + tail, device=device)
        self.start_nats = nats[: len(self.full)]
        self.window_bytes = np.array([at[s + EVAL_CTX + 1] - at[s + 1] for s in self.full], dtype=np.int64)
        self.start_bpb = float(nats.sum()) / man["bytes"] / math.log(2)

    def model(self, state: dict[str, torch.Tensor], config: dict | None = None) -> Net:
        """A model of `config` (θ's by default) holding `state`: one Net per architecture, reused."""
        config = config or self.rec["config"]
        key = json.dumps(config, sort_keys=True)
        if key not in self.nets:
            with torch.device(self.device):
                self.nets[key] = Net(config)
        return self.nets[key].load(state)

    def search(self, state: dict[str, torch.Tensor], weights_digest: str, config: dict | None = None) -> dict:
        net = self.model(state, config)
        if FINAL_MODE:
            return evaluate_bpb(net, FINAL, self.tok, device=self.device) | {"start_bpb": self.start_bpb}
        sl = draw_slice(weights_digest, tokenizer=self.tok)
        pick = np.asarray(sl["windows"])
        nbytes = int(self.window_bytes[pick].sum())
        a = float(window_nats(net, self.pool, [self.full[i] for i in pick], device=self.device).sum())
        b = float(self.start_nats[pick].sum())
        a, b = a / nbytes / math.log(2), b / nbytes / math.log(2)
        return {"split": "search", "bpb": self.start_bpb + a - b, "slice_bpb": a, "start_slice_bpb": b,
                "start_bpb": self.start_bpb, "tokenizer": self.tok,
                "slice": {"coin": sl["coin"], "weights_digest": weights_digest, "windows": len(pick), "bytes": nbytes,
                          "targets": len(pick) * EVAL_CTX},
                "rule": "start_bpb (the whole search pool) + slice_bpb - start_slice_bpb on the same windows; the slice "
                        "drawn after the weights were hashed (prepare.draw_slice(weights_digest, coin) reproduces it)"}

    def _promotion_bpb(self, net: Net) -> float:
        if self.promotion is None:
            toks, _ = held_out(PROMOTION, self.tok)
            starts = windows(len(toks))[0][: PROMOTION_TOKENS // EVAL_CTX]
            tb = self.token_bytes[toks]
            self.promotion = (toks, starts, int(sum(tb[s + 1: s + EVAL_CTX + 1].sum() for s in starts)))
        toks, starts, nbytes = self.promotion
        return float(window_nats(net, toks, starts, device=self.device).sum()) / nbytes / math.log(2)

    def seal(self, key: str, state: dict[str, torch.Tensor], search: dict, config: dict | None = None) -> bool:
        path = value_path(key)
        if FINAL_MODE or path.exists():
            return False
        bpb = self._promotion_bpb(self.model(state, config))
        if self.start_promotion_bpb is None:
            self.start_promotion_bpb = self._promotion_bpb(self.base)
        rec = {"schema": "covert-witness/sealed-value/v1", "key": key, "start": self.rec["name"],
               "start_digest": self.rec["digest"], "search_bpb": search["bpb"], "start_search_bpb": search["start_bpb"],
               "promotion_bpb": bpb, "start_promotion_bpb": self.start_promotion_bpb, "salt": os.urandom(16).hex(),
               "run": os.environ.get("RESEARCH_RUN_ID", ""), "version": VERSION}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{key}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True) + "\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
        return True


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def read_steps(path: Path) -> list[dict]:
    return [json.loads(x) for x in io.StringIO(path.read_text()) if x.strip()]
