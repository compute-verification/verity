"""Pins and checkpoints for the store binding, whose layout and decision are PoUB's own
(`verity/core/protocols/remote_bandwidth/store.py`: the pin's form and checks, the arena, the store, its expected payload
root and `Binding`): a checkpoint's local pin, a small synthetic BF16 checkpoint for tests, and the plain SHA-256 a GiB
of an engine's arena holding the checkpoint's weights (`node.py --model-dir`'s check of each engine against the layout).

    python store_layout.py pin DIR --name ROLE --out PIN.json          the local pin of a checkpoint in DIR
    python store_layout.py synthetic DIR [--layers 2 --hidden 48 ...]  a small BF16 checkpoint and its pin, for tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import store as rstore  # noqa: E402

GIB = 1 << 30


def _sha256(a: rstore.Arena, lo: int, n: int) -> str:
    return hashlib.sha256(a.read(lo, n)).hexdigest()


def arena_sha256(a: rstore.Arena, nbytes: int, workers: int = 1) -> list[str]:
    """The SHA-256 of each GiB of an engine's arena of `nbytes` holding `a` (`engine.Arena.hashes`' plain ones)."""
    spans = [(lo, min(GIB, nbytes - lo)) for lo in range(0, nbytes, GIB)]
    if workers > 1 and len(spans) > 1:
        with rstore.pool(min(workers, len(spans))) as ex:
            return list(ex.map(_sha256, *zip(*[(a, lo, n) for lo, n in spans])))
    return [_sha256(a, lo, n) for lo, n in spans]


def synthetic(directory: Path, *, name: str = "synthetic-bf16", layers: int = 2, hidden: int = 48, heads: int = 4,
              kv_heads: int = 2, intermediate: int = 80, vocab: int = 128, tied: bool = False, seed: int = 0,
              architecture: str = "Qwen2ForCausalLM", shards: int = 2) -> dict:
    """A small BF16 checkpoint in `directory` (config.json and `shards` safetensors files, with the norms and q/k/v
    biases a Qwen2 checkpoint has), and its pin, written beside it as pin.json: the pin."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    hd = hidden // heads

    def bf16(*shape: int) -> tuple[tuple[int, ...], bytes]:
        n = math.prod(shape)
        f32 = struct.pack(f"<{n}f", *(rng.gauss(0.0, 0.02) for _ in range(n)))
        out = bytearray(2 * n)
        out[0::2], out[1::2] = f32[2::4], f32[3::4]
        return shape, bytes(out)

    ts: dict[str, tuple[tuple[int, ...], bytes]] = {"model.embed_tokens.weight": bf16(vocab, hidden),
                                                    "model.norm.weight": bf16(hidden)}
    for i in range(layers):
        p = f"model.layers.{i}."
        ts.update({p + "self_attn.q_proj.weight": bf16(heads * hd, hidden), p + "self_attn.q_proj.bias": bf16(heads * hd),
                   p + "self_attn.k_proj.weight": bf16(kv_heads * hd, hidden),
                   p + "self_attn.k_proj.bias": bf16(kv_heads * hd),
                   p + "self_attn.v_proj.weight": bf16(kv_heads * hd, hidden),
                   p + "self_attn.v_proj.bias": bf16(kv_heads * hd), p + "self_attn.o_proj.weight": bf16(hidden, hidden),
                   p + "mlp.gate_proj.weight": bf16(intermediate, hidden), p + "mlp.up_proj.weight": bf16(intermediate, hidden),
                   p + "mlp.down_proj.weight": bf16(hidden, intermediate), p + "input_layernorm.weight": bf16(hidden),
                   p + "post_attention_layernorm.weight": bf16(hidden)})
    if not tied:
        ts["lm_head.weight"] = bf16(vocab, hidden)
    names = sorted(ts)
    for s in range(shards):
        mine = names[s::shards]
        header, at = {"__metadata__": {"format": "pt"}}, 0
        for t in mine:
            shape, raw = ts[t]
            header[t] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [at, at + len(raw)]}
            at += len(raw)
        h = json.dumps(header).encode()
        h += b" " * (-len(h) % 8)
        with open(directory / f"model-{s + 1:05d}-of-{shards:05d}.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h)
            for t in mine:
                f.write(ts[t][1])
    (directory / "config.json").write_text(json.dumps({
        "architectures": [architecture], "hidden_size": hidden, "intermediate_size": intermediate,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "num_hidden_layers": layers,
        "vocab_size": vocab, "tie_word_embeddings": tied, "torch_dtype": "bfloat16"}, indent=1))
    return write_pin(directory, name, directory / "pin.json")


def write_pin(directory: Path, name: str, out: Path) -> dict:
    """The pin of every file in `directory` but `out` itself, written to `out`."""
    files = {p.name: rstore.file_sha512(str(p)) for p in sorted(Path(directory).iterdir())
             if p.is_file() and p.resolve() != Path(out).resolve()}
    pin = {"name": name, "checkpoint": files, "checkpoint_sha512": rstore.checkpoint_sha512(files)}
    Path(out).write_text(json.dumps(pin, indent=1) + "\n")
    return pin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("pin", help="the local pin of the checkpoint in DIR, every file in it")
    p.add_argument("dir", type=Path)
    p.add_argument("--name", required=True, help="the model's role, the name --model takes")
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("synthetic", help="a small BF16 checkpoint and its pin (DIR/pin.json)")
    s.add_argument("dir", type=Path)
    s.add_argument("--name", default="synthetic-bf16")
    for flag, default in (("layers", 2), ("hidden", 48), ("heads", 4), ("kv-heads", 2), ("intermediate", 80),
                          ("vocab", 128), ("seed", 0), ("shards", 2)):
        s.add_argument(f"--{flag}", type=int, default=default)
    s.add_argument("--tied", action="store_true")
    a = ap.parse_args(argv)
    if a.verb == "pin":
        pin = write_pin(a.dir, a.name, a.out)
    else:
        pin = synthetic(a.dir, name=a.name, layers=a.layers, hidden=a.hidden, heads=a.heads, kv_heads=a.kv_heads,
                        intermediate=a.intermediate, vocab=a.vocab, seed=a.seed, shards=a.shards, tied=a.tied)
    print(json.dumps({"name": pin["name"], "files": len(pin["checkpoint"]),
                      "checkpoint_sha512": pin["checkpoint_sha512"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
