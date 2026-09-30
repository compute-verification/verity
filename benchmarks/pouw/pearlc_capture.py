#!/usr/bin/env python3
"""Capture once, replay on the CPU: the first input X of every linear layer in a Llama model's selected decoder layers, from a
bf16 forward on WikiText-2 test streamed one decoder layer at a time from the model's safetensors (Llama-3.1-70B does not fit
one GPU, and a full CPU forward would hold all of it), for `pearlc_activations.py --replay`.

X is saved per layer and role (the q/k/v input, the o_proj input, the gate/up input, the down_proj input) as float32 .npy,
which holds the bf16 values exactly; W stays in the model's safetensors, which the replay reads. The manifest records the
model, its revision and snapshot directory, the tokens, the library versions and each file's sha256. The streamed forward runs
the model's own decoder layers, rotary embedding and causal mask; the self-test, which every capture runs first, checks it bit
for bit against the full model's forward (`from_pretrained` in bf16, as `pearlc_activations.py` loads a model) on a tiny
random Llama with Llama 3.1's rope scaling.

It is restartable: the hidden state is checkpointed after every decoder layer, past --budget-min it exits 99 (the fill
runner's "more chunks remain"), and it exits 0 once the manifest is written.

    python benchmarks/pouw/pearlc_capture.py --model unsloth/Meta-Llama-3.1-70B \
        --revision 1b7306651142d0cc65d993076a250a6a82cf046c --every 8 --budget-min 20 --out llama70b-w0
    python benchmarks/pouw/pearlc_capture.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.masking_utils import create_causal_mask
from transformers.models.llama import modeling_llama as L

ROLES = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
         "gateup": ("mlp.gate_proj", "mlp.up_proj"), "down": ("mlp.down_proj",)}
MORE = 99


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


class Stream:
    """A Llama model's decoder layers, one at a time, from its safetensors (one file or an index of shards)."""

    def __init__(self, path: Path, cfg):
        self.path, self.cfg = path, cfg
        idx = path / "model.safetensors.index.json"
        if idx.exists():
            self.wmap = json.loads(idx.read_text())["weight_map"]
        else:
            with safe_open(path / "model.safetensors", framework="pt") as f:
                self.wmap = {k: "model.safetensors" for k in f.keys()}

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(self.path / self.wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    def start(self, ids: torch.Tensor, dtype: torch.dtype | None = None):
        """The embedded tokens (batch × tokens, optionally cast), their positions, rotary embedding and causal mask, as
        LlamaModel.forward builds them."""
        h = self.tensor("model.embed_tokens.weight")[ids]
        h = h if dtype is None else h.to(dtype)
        pos = torch.arange(h.shape[1])[None].expand(h.shape[0], -1)
        rope = L.LlamaRotaryEmbedding(config=self.cfg)(h, pos)
        mask = create_causal_mask(config=self.cfg, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos)
        return h, pos, rope, mask

    def load(self, i: int):
        """Decoder layer i with the checkpoint's own tensors (its dtype), built without initializing weights."""
        with torch.device("meta"):
            layer = L.LlamaDecoderLayer(self.cfg, i)
        pre = f"model.layers.{i}."
        layer.load_state_dict({k[len(pre):]: self.tensor(k) for k in self.wmap if k.startswith(pre)}, assign=True, strict=True)
        return layer.eval()

    def layer(self, i: int, h, pos, rope, mask, capture: bool):
        """Layer i's output and, if `capture`, each role's input (the role's linears must see one tensor)."""
        layer = self.load(i)
        seen: dict[str, torch.Tensor] = {}
        hooks = [layer.get_submodule(m).register_forward_pre_hook(lambda mod, inp, m=m: seen.setdefault(m, inp[0].detach()))
                 for mods in ROLES.values() for m in mods] if capture else []
        with torch.no_grad():
            out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=rope)
        for hk in hooks:
            hk.remove()
        cap = {}
        for role, mods in ROLES.items() if capture else ():
            if not all(torch.equal(seen[m], seen[mods[0]]) for m in mods):
                raise RuntimeError(f"layer {i}: the {role} linears saw different inputs")
            cap[role] = seen[mods[0]]
        return (out[0] if isinstance(out, tuple) else out), cap


def selftest(seed: int = 0) -> dict:
    """The streamed forward against the full model's, bit for bit, on a tiny random Llama: every captured input of every
    layer, and the last layer's output (the final norm's input)."""
    torch.manual_seed(seed)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=352, num_hidden_layers=3, num_attention_heads=8,
                        num_key_value_heads=2, max_position_embeddings=1024, rope_theta=500000.0, tie_word_embeddings=False,
                        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                      "original_max_position_embeddings": 64})
    with tempfile.TemporaryDirectory() as d:
        L.LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
        full = L.LlamaForCausalLM.from_pretrained(d, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 192))
        ref: dict = {}
        for i, layer in enumerate(full.model.layers):
            for mods in ROLES.values():
                for m in mods:
                    layer.get_submodule(m).register_forward_pre_hook(
                        lambda mod, inp, key=(i, m): ref.setdefault(key, inp[0].detach()))
        full.model.norm.register_forward_pre_hook(lambda mod, inp: ref.setdefault("last", inp[0].detach()))
        with torch.no_grad():
            full(ids, use_cache=False)
        scfg = transformers.AutoConfig.from_pretrained(d)
        scfg._attn_implementation = "sdpa"
        S = Stream(Path(d), scfg)
        h, pos, rope, mask = S.start(ids)
        bad = []
        for i in range(cfg.num_hidden_layers):
            h, cap = S.layer(i, h, pos, rope, mask, True)
            bad += [f"{i}.{m}" for role, mods in ROLES.items() for m in mods if not torch.equal(cap[role], ref[(i, m)])]
        if not torch.equal(h, ref["last"]):
            bad.append("last")
    return {"passed": not bad, "mismatched": bad, "compared": 7 * cfg.num_hidden_layers + 1, "tokens": ids.shape[1],
            "config": "Llama, hidden 128, intermediate 352, 3 layers, 8 heads (2 kv), llama3 rope scaling, bf16, sdpa"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-70B")
    ap.add_argument("--revision", default="1b7306651142d0cc65d993076a250a6a82cf046c")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--every", type=int, default=8, help="capture decoder layers 0, every, 2·every, ...")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--budget-min", type=float, default=0.0, help="exit 99 past this many minutes while layers remain (0: no limit)")
    ap.add_argument("--selftest", action="store_true", help="only the streamed-versus-full check on a tiny random Llama")
    ap.add_argument("--out", default="pearlc_capture")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    st = selftest()
    print(json.dumps({"selftest": st}), flush=True)
    if args.selftest or not st["passed"]:
        return 0 if st["passed"] else 1

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(snapshot_download(args.model, revision=args.revision, allow_patterns=["*.json", "tokenizer*"]))
    cfg = transformers.AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    want = list(range(0, cfg.num_hidden_layers, args.every))
    ids_file = out / "ids.npy"
    if not ids_file.exists():
        tok = transformers.AutoTokenizer.from_pretrained(path)
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        ids = tok(text, return_tensors="np").input_ids[0][args.window * args.tokens:(args.window + 1) * args.tokens]
        np.save(ids_file, ids.astype(np.int64))
        (out / "tokens.json").write_text(json.dumps({"bos_token_id": tok.bos_token_id, "first": ids[:8].tolist()}) + "\n")
    ids = torch.from_numpy(np.load(ids_file))[None]
    S = Stream(path, cfg)
    state = out / "state.pt"
    h, pos, rope, mask = S.start(ids)
    nxt, spent = 0, 0.0
    if state.exists():
        ck = torch.load(state)
        h, nxt, spent = ck["h"], ck["next"], ck["seconds"]
    for i in range(nxt, max(want) + 1):
        h, cap = S.layer(i, h, pos, rope, mask, i in want)
        for role, x in cap.items():
            f = out / f"L{i:03d}-{role}.npy"
            np.save(f.with_suffix(".tmp.npy"), x.reshape(-1, x.shape[-1]).to(torch.float32).numpy())
            f.with_suffix(".tmp.npy").replace(f)
        torch.save({"h": h, "next": i + 1, "seconds": spent + time.time() - t0}, out / "state.tmp")
        (out / "state.tmp").replace(state)
        print(json.dumps({"layer": i, "captured": sorted(cap), "seconds": round(time.time() - t0, 1)}), flush=True)
        if args.budget_min and time.time() - t0 > 60 * args.budget_min and i < max(want):
            return MORE
    inputs = {}
    for i in want:
        for role, mods in ROLES.items():
            f = out / f"L{i:03d}-{role}.npy"
            x = np.load(f, mmap_mode="r")
            ent = {"file": f.name, "sha256": sha256(f), "shape": list(x.shape), "dtype": str(x.dtype)}
            for m in mods:
                inputs[f"model.layers.{i}.{m}"] = ent
    man = {"tool": "pouw_pearlc_capture", "model": args.model, "revision": args.revision, "snapshot": str(path),
           "dataset": "wikitext-2-raw-v1:test", "window": args.window, "tokens": int(ids.shape[1]),
           "tokenizer": json.loads((out / "tokens.json").read_text()), "layers": want, "num_hidden_layers": cfg.num_hidden_layers,
           "forward": "bf16 weights and activations on the CPU, sdpa attention, streamed one decoder layer at a time",
           "x_dtype": "float32, holding the bf16 inputs exactly", "torch": torch.__version__, "transformers": transformers.__version__,
           "threads": args.threads, "selftest": st, "seconds": round(spent + time.time() - t0, 1), "inputs": inputs}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "inputs"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
