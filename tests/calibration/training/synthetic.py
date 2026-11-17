"""A tiny data and start, built in a test's directory in the layout a node holds them in: a byte-level tokenizer of 259
tokens, a stream of three parts (six windows at the start's context), a search pool of six full windows and a final
set of two, an lf start of one block (or more), and a registry naming both (`tiny_registry`); and one job of a config
of that start, run as a node runs it (`run_job`) or counted in the test's process (`job_counted`)."""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from training_slowdown import config, run, starts
from training_slowdown.data import Data, Stream, bytes_to_unicode
from training_slowdown.model import LM
from training_slowdown.starts import Start

EOT = 0
MERGES = ("ab", "abc")
VOCAB = 1 + 256 + len(MERGES)
#: a block's parameters: its four attention matrices, its three of the MLP and its two gains
BLOCK = 16 * 16 * 4 + 3 * 16 * 32 + 2 * 16
SHAPE = {"family": "lf", "d_model": 16, "n_layer": 1, "n_head": 2, "d_ff": 32, "vocab": VOCAB, "ctx": 1024, "tied": True,
         "parameters": BLOCK + (VOCAB + 1) * 16, "tokens": 0}
#: each token's bytes: end-of-text none, a byte one, a merge its letters
TOKEN_BYTES = np.array([0] + [1] * 256 + [len(m) for m in MERGES], dtype=np.int64)
PARTS = (2100, 1100, 3100)
SETS = {"search": 6 * 1024 + 300, "final": 2 * 1024 + 50}
#: How far apart two scores of one model on the same windows may be, relative. Under bf16 autocast on a CPU a forward
#: pass is not bit-reproducible: now and then a few logits come out one bf16 step away. The tolerance is above that
#: spread and below the nearest scores the evaluator's tests tell apart, θ's and a perturbed state's on a slice of the
#: search pool: `test_a_search_score_is_the_states_bpb_on_the_slice_paired_with_thetas` fails if it grows past them.
SCORE_REL = 3e-4


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tokens(rng: np.random.Generator, n: int, alphabet: int | None = None) -> np.ndarray:
    """`n` token ids, a document starting at every 97th: of the whole vocabulary, or of its first `alphabet` bytes,
    which a few steps learn to predict."""
    t = rng.integers(1, 1 + alphabet if alphabet else VOCAB, n).astype(np.uint16)
    t[::97] = EOT
    return t


def tiny_data(root: Path, alphabet: int | None = None) -> Data:
    rng = np.random.default_rng(0)
    train = root / "train"
    (train / "parts").mkdir(parents=True)
    enc = bytes_to_unicode()
    vocab = {"<|endoftext|>": EOT} | {enc[b]: 1 + b for b in range(256)}
    vocab |= {"".join(enc[c] for c in m.encode()): 257 + i for i, m in enumerate(MERGES)}
    tok = json.dumps({"model": {"vocab": vocab}}).encode()
    (train / "tokenizer.json").write_bytes(tok)
    selection = b'{"sources": {}}\n'
    man = {"tokenizer": "tiny", "vocab": VOCAB, "selection": {"file": "selection.json", "sha256": sha256(selection)},
           "parts": []}
    for i, n in enumerate(PARTS):
        part = tokens(rng, n, alphabet).tobytes()
        (train / "parts" / f"{i:03d}.bin").write_bytes(part)
        man["parts"].append({"index": i, "tokens": n, "sha256": sha256(part)})
    raw = json.dumps(man).encode()
    (train / "manifest.json").write_bytes(raw)
    sets, pins = {}, {}
    for split, n in SETS.items():
        d = root / split
        d.mkdir()
        t = tokens(rng, n, alphabet)
        (d / "tiny.bin").write_bytes(t.tobytes())
        pins[split] = sha256(t.tobytes())
        (d / "manifest.json").write_text(json.dumps({"tokenizers": {"tiny": {"sha256": pins[split], "tokens": n}},
                                                     "bytes": int(TOKEN_BYTES[t].sum()), "documents": int((t == EOT).sum())}))
        sets[split] = d
    return Data("tiny", "tiny", VOCAB, EOT, train, sets, sha256(raw), sha256(tok), sha256(selection), pins)


def write_safetensors(path: Path, state: dict[str, torch.Tensor]) -> None:
    names = {torch.float32: "F32", torch.bfloat16: "BF16"}
    header, blobs, at = {}, [], 0
    for k in sorted(state):
        t = state[k].detach().contiguous().cpu()
        b = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[k] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def shape_of(blocks: int) -> dict:
    """The tiny start's shape with `blocks` blocks."""
    return SHAPE | {"n_layer": blocks, "parameters": SHAPE["parameters"] + (blocks - 1) * BLOCK}


def tiny_start(root: Path, name: str = "tiny", read: list[dict] | None = None,
               blocks: int = 1) -> tuple[Start, dict[str, torch.Tensor]]:
    """An installed start under `root` and its registry entry, of `blocks` blocks. Its start.json names `read` as the
    parts of the stream its own training read (None: it says nothing of what it read)."""
    shape = shape_of(blocks)
    torch.manual_seed(0)
    theta = {k: v.detach().clone() for k, v in LM.of(shape).state_dict().items()}
    d = root / name
    d.mkdir(parents=True)
    write_safetensors(d / "theta.safetensors", theta)
    digest = "d" * 64
    rec = {"name": name, "digest": digest, "provenance": {"c0": {"macs": 12_345}, "search_pool": {"bpb": 7.5}} | (
        {} if read is None else {"data": {"parts": read}})}
    (d / "start.json").write_text(json.dumps(rec))
    s = Start(name, "tiny", "art:" + "0" * 64, f"starts/{name}", sha256((d / "theta.safetensors").read_bytes()),
              sha256((d / "start.json").read_bytes()), digest, shape, 12_345.0, 7.5)
    return s, theta


def _toml(v) -> str:
    if isinstance(v, dict):
        return "{ " + ", ".join(f'"{k}" = {_toml(x)}' for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return json.dumps(str(v) if isinstance(v, Path) else v)


def tiny_registry(root: Path, read: tuple[int, ...] | list[dict] | None = None, alphabet: int | None = None,
                  blocks: int = 1) -> Path:
    """`root/starts.toml`, naming the tiny data and start built under `root`, the start of `blocks` blocks. `read` is
    what the start names as read by its own training: parts of the tiny stream by index, each as the stream's
    manifest lists it, or the entries as given; `alphabet` is `tokens`'."""
    data = tiny_data(root / "data", alphabet)
    parts = data.manifest()["parts"]
    s, _ = tiny_start(root / "starts", read=read and [parts[q] if isinstance(q, int) else q for q in read],
                      blocks=blocks)
    d = {"tokenizer": data.tokenizer, "vocab": data.vocab, "eot": data.eot, "train": data.train,
         "manifest_sha256": data.manifest_sha256, "tokenizer_sha256": data.tokenizer_sha256,
         "selection_sha256": data.selection_sha256}
    d |= {split: data.sets[split] for split in data.sets} | {f"{k}_sha256": v for k, v in data.set_sha256.items()}
    entry = {k: v for k, v in s.entry().items() if k != "name"}
    lines = [f"installed = {_toml(root / 'starts')}", "", "[data.tiny]", *(f"{k} = {_toml(v)}" for k, v in d.items()),
             "", f"[starts.{s.name}]", *(f"{k} = {_toml(v)}" for k, v in entry.items())]
    (root / "starts.toml").write_text("\n".join(lines) + "\n")
    return root / "starts.toml"


#: the tiny data's alphabet where a config is graded: training on it reaches the gain in a few steps
ALPHABET = 8
#: the frontier the grade's fit reads where a config is graded, L(C) = E + A·C^(−γ): the tiny start's point, and
#: `FRONTIER` more starts on its data at ten, a hundred and a thousand times its compute. Its E and γ put the gain's
#: target a few thousandths of a bit per byte under the tiny start's own loss on the data of `ALPHABET`
E, GAMMA, FRONTIER = 7.0, 0.08, 3


def graded_registry(root: Path) -> Path:
    """`tiny_registry` on data of `ALPHABET`, with the starts of the made frontier beside the tiny start (registry
    entries alone: no job reads them, and the grade's fit reads each one's C0 and loss)."""
    path = tiny_registry(root, alphabet=ALPHABET)
    tiny = starts.registry(path)["starts"]["tiny"]
    lines = [line for k in range(1, FRONTIER + 1) for line in (
        "", f"[starts.tiny-{k}]", *(f"{n} = {_toml(v)}" for n, v in (tiny | {
            "c0_macs": tiny["c0_macs"] * 10.0**k, "l0_bpb": E + (tiny["l0_bpb"] - E) * 10.0 ** (-GAMMA * k)}).items()))]
    path.write_text(path.read_text() + "\n".join(lines) + "\n")
    return path


def write_config(path: Path, **over) -> Path:
    """An attack config of the tiny start: `over` replaces a top-level key (None drops it)."""
    cfg = {"start": "tiny", "route": "subset", "target": {"gain": 1.1}, "budget": {"steps": 3, "batch": 2},
           "seeds": {"search": [0], "final": []}, "policy": "console", "dials": {"trainable": "blocks.0.w_*", "lr": 1e-2}}
    cfg |= over
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k} = {_toml(v)}\n" for k, v in cfg.items() if v is not None and k != "dials")
                    + ("[dials]\n" + "".join(f"{k} = {_toml(v)}\n" for k, v in cfg["dials"].items())
                       if cfg.get("dials") is not None else ""))
    return path


def run_job(root: Path, *more: str, split: str = "search", read: tuple[int, ...] | None = None, **over):
    """One job of the tiny start's config (`over` its keys) under `root`, through the job's own entry: its record, its
    config, the registry and its output directory. `read` is `tiny_registry`'s."""
    reg = tiny_registry(root / "reg", read)
    path = write_config(root / "a.toml", **over)
    cfg = config.load(path, starts.registry(reg))
    out = root / "out"
    rc = run.main(["--config", path, "--seed", 0, "--split", split, "--out", out, "--starts", reg, "--device", "cpu",
                   *more])
    rec = json.loads((out / "run.json").read_text())
    assert rc == 0, rec["invalid"]
    return rec, cfg, reg, out


LORA = {"targets": "blocks.0.wq.weight,blocks.0.w_up.weight", "rank": 2, "alpha": 4.0, "lr": 1e-2}
#: the tiny start's one block grown to three, the two copies of it with their output projections zero, trained whole
GROW = {"depth": 3, "lr": 1e-2, "adam_lr": 2e-3, "embed_lr": 3e-3, "weight_decay": 0.05, "warmup": 1,
        "micro_batches": 2}
#: the whole model, one block, LoRA's adapters, one block's average, one block under Muon in two micro-batches (its
#: Newton-Schulz products cast to bf16), one block under Shampoo (its factors' products and roots in f32), and the
#: model grown by copies of its block
JOBS = {"whole": {"dials": {"trainable": "*", "lr": 1e-2}},
        "block": {"dials": {"trainable": "blocks.0.*,norm", "lr": 1e-2}},
        "lora": {"route": "lora", "dials": LORA},
        "ema": {"route": "ema", "dials": {"trainable": "blocks.0.*,norm", "lr": 1e-2, "ema_decay": 0.5}},
        "muon": {"dials": {"trainable": "blocks.0.*,norm", "opt": "muon", "lr": 1e-2, "micro_batches": 2}},
        "shampoo": {"dials": {"trainable": "blocks.0.*,norm", "opt": "shampoo", "lr": 1e-2}},
        "grow": {"route": "grow", "dials": GROW}}
#: Muon on the matrices as the library's configs run it: decayed, beside AdamW in torch's fused kernel, in two
#: micro-batches
FULLMUON = {"opt": "muon", "lr": 1e-2, "weight_decay": 0.05, "fused": True, "micro_batches": 2}
#: QAT of the top block's matrices, the block below it frozen, as the library's QAT configs train their top blocks
QAT = {"targets": "blocks.1.w*.weight", "keep": 0.35, "lr": 1e-2}
#: each kind of step (`kind_of`) a config of the library (`attacks/`) takes that none of the jobs above takes, on the
#: tiny start, deeper (`blocks`) where blocks are frozen below or between trained ones
KINDS = {"adamw-decay": {"dials": {"trainable": "*", "lr": 1e-2, "weight_decay": 0.1}},
         "accumulate": {"dials": {"trainable": "*", "lr": 1e-2, "weight_decay": 0.1, "micro_batches": 2}},
         "adafactor": {"dials": {"trainable": "*", "opt": "adafactor", "lr": 1e-2}},
         "lion": {"dials": {"trainable": "*", "opt": "lion", "lr": 1e-2, "weight_decay": 1.0}},
         "sgd": {"dials": {"trainable": "*", "opt": "sgd", "lr": 1e-2, "weight_decay": 1e-4}},
         "last-blocks": {"blocks": 2, "dials": {"trainable": "blocks.1.*,norm", "lr": 1e-2, "weight_decay": 0.1}},
         "muon-whole": {"dials": {"trainable": "*", "opt": "muon", "lr": 1e-2, "weight_decay": 0.05}},
         "muon-whole-fused": {"dials": {"trainable": "*"} | FULLMUON | {"micro_batches": 1}},
         "fullmuon": {"dials": {"trainable": "*"} | FULLMUON},
         "muon-decay": {"dials": {"trainable": "blocks.0.*,norm", "opt": "muon", "lr": 1e-2, "weight_decay": 0.05}},
         "muon-fused": {"dials": {"trainable": "blocks.0.*,norm"} | FULLMUON | {"micro_batches": 1}},
         "fullmuon-fe": {"dials": {"trainable": "blocks.0.*,norm"} | FULLMUON},
         "depthshare": {"dials": {"trainable": "blocks.*,norm"} | FULLMUON | {"weight_decay": 0.0}},
         "depthshare-top": {"blocks": 2, "dials": {"trainable": "blocks.1.*,norm"} | FULLMUON
                            | {"weight_decay": 0.0}},
         "strideshare": {"blocks": 4, "dials": {"trainable": "blocks.3.*,blocks.1.*,norm"} | FULLMUON
                         | {"weight_decay": 0.0}},
         "shampoo-decay": {"dials": {"trainable": "blocks.0.*,norm", "opt": "shampoo", "lr": 1e-2, "weight_decay": 0.05}},
         "shampoo-whole": {"dials": {"trainable": "*", "opt": "shampoo", "lr": 1e-2, "weight_decay": 0.05}},
         "ema-whole": {"route": "ema", "dials": {"trainable": "*", "lr": 1e-2, "weight_decay": 0.1, "ema_decay": 0.99}},
         "ema-fullmuon": {"route": "ema", "dials": {"trainable": "*"} | FULLMUON | {"ema_horizon": 0.1}},
         "ema-fullmuon-fe": {"route": "ema", "dials": {"trainable": "blocks.0.*,norm"} | FULLMUON | {"ema_horizon": 0.1}},
         "qat": {"route": "qat", "blocks": 2, "dials": QAT},
         "qat-grad4": {"route": "qat", "blocks": 2, "dials": QAT | {"grad_code": "rtn4"}},
         "qat-grad4-lowstate": {"route": "qat", "blocks": 2, "dials": QAT | {"grad_code": "rtn4", "low_state": True}}}
JOBS |= KINDS
#: the dials whose branches `kind_of` reads, from what a route builds or from the dial; grow's `embed_lr` is one, its
#: embedding frozen at 0 and trained above
KIND_DIALS = {"trainable", "embed", "embed_lr", "targets", "opt", "fused", "weight_decay", "low_state",
              "micro_batches", "clip", "grad_code", "zero"}
#: the dials that take no branch of a step: rates (a rate adds a group to an optimizer outside torch's fused kernel,
#: which `kind_of` reads), schedules, shares and sizes, and `store` (E, encoded outside the count); `ema_horizon`
#: branches the step only at its first step, which no config of the library measures
OTHER_DIALS = {"lr", "gain_lr", "adam_lr", "warmup", "decay", "schedule", "cooldown", "ema_decay",
               "ema_horizon", "keep", "group", "rank", "alpha", "depth", "store"}


def _part(name: str, p: torch.Tensor) -> str:
    """What a tensor of a route's model is, by its name: the embedding, the final gain, an adapter (LoRA's), a latent
    (QAT's), a block's gain or a block's matrix."""
    if name in ("embed", "norm"):
        return {"embed": "embedding", "norm": "final gain"}[name]
    if name.endswith((".a", ".b", ".v")):
        return "latent" if name.endswith(".v") else "adapter"
    return "gain" if p.ndim == 1 else "matrix"


def kind_of(route: str, given: dict, blocks: int) -> tuple:
    """The kind of step a config of the route `route` and the dials `given` takes on a start of `blocks` blocks, read
    from what the route builds (at the tiny start's widths): each optimizer's class, whether it decays, steps in
    torch's fused kernel, keeps its state at low width or normalizes a matrix as a stack, its groups and what it
    trains; whether the step runs micro-batches and clips; which blocks are frozen below, between or above trained
    ones; whether QAT codes its gradients; and whether a grown model's repeated copies start zeroed. Two configs of
    one kind run the same code, the same branches taken, on tensors of other shapes and at other rates: what a dial
    of `OTHER_DIALS` sets is no part of it. An optimizer's number of parameter groups is part of the kind: outside
    torch's fused kernel it keeps one group a rate, so a config that adds a rate is a new kind until a job takes it.
    A QAT config's 8-bit gradient codes are its 4-bit ones: the two differ only in shifts and ors, which no count
    holds."""
    mod = config.route(route)
    dials = config.dials(mod, given, route) | getattr(mod, "FIXED", {})
    torch.manual_seed(0)
    model, optimizers = mod.build(shape_of(blocks), dials)
    names = {id(p): n for n, p in model.named_parameters()}
    trained = sorted({int(n.split(".")[1]) for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("blocks.")})
    frozen = tuple(where for where, holds in (
        ("below", trained and trained[0] > 0), ("between", trained and len(trained) < trained[-1] - trained[0] + 1),
        ("above", trained and trained[-1] < len(model.blocks) - 1), ("every", not trained)) if holds)
    steps = tuple((type(o).__name__, bool(o.defaults.get("fused")), any(g.get("weight_decay") for g in o.param_groups),
                   bool(o.defaults.get("low_state")), bool(o.defaults.get("stack")), len(o.param_groups),
                   tuple(sorted({_part(names[id(p)], p) for g in o.param_groups for p in g["params"]})))
                  for o in optimizers)
    return (("route", route), ("optimizers (class, fused, decayed, low state, stack, groups, trains)", steps),
            ("micro-batches", dials["micro_batches"] > 1), ("clipped", dials["clip"] > 0),
            ("frozen blocks", frozen), ("coded gradients", dials.get("grad_code", "fp32") != "fp32"),
            ("zeroed copies", bool(dials.get("zero"))))


def job_counted(root: Path, steps: int, math: bool = False, blocks: int = 1, **over):
    """A job's steps through the harness's own loop (`run.train` under `run.Harness`), in this process, under the
    attention kernel torch picks (a fused one, a job's on a GPU) or under the math kernel (a job's on a CPU,
    `run.cpu_attention`), on the tiny start of `blocks` blocks: the counter's record, and what the job's trace is
    taken from (its config, route, model and optimizers)."""
    reg = tiny_registry(root / "reg", blocks=blocks)
    registry = starts.registry(reg)
    cfg = config.load(write_config(root / "a.toml", budget={"steps": steps, "batch": 4}, **over), registry)
    te = run.Harness(starts.load(cfg.start, reg.parent / "starts"), root / "out", start=cfg.start.entry(),
                     clock=lambda: 0.0)
    route = config.route(cfg.route)
    torch.manual_seed(0)
    with sdpa_kernel([SDPBackend.MATH]) if math else nullcontext(), te.counter:
        model, optimizers = route.build(cfg.start.shape, cfg.dials)
        stream = Stream(Data.of(cfg.start.data, registry), 0, ctx=int(cfg.start.shape["ctx"]))
        run.train(te, cfg, route, stream, model, optimizers)
    return te.finish({}), (cfg, route, model, optimizers)
