"""The covert-training search's fixed half: data, the registered starts, and the evaluator. Island agents never edit this file.

Layout under ROOT (`--root`, default `$COVERT_ROOT` or /workspace/research/scratch/covert-search on vy-nebius-2):

    train/<tok>/train.bin            training tokens (uint16), the only data the block executor (meter.py) reads
    heldout/<tok>/search.bin         SEARCH: fineweb-edu shard 013's first 2,097,153 GPT-2 tokens (the first curves' held-out
                                     text), the score every candidate gets
    heldout/<tok>/final.bin          FINAL: shard 012's first 2,097,153 GPT-2 tokens, which no candidate is scored on; the
                                     integrator reads it once, on kept entries
    heldout/<tok>/*.bytes.npy        per target token, its UTF-8 byte length (end-of-text: 0)
    starts/<name>/start.json         a registered start: architecture, tokenizer, theta's file and sha256, c, the eval cap
    starts/<name>/theta.pt           its weights (fp32)

`<tok>` is `gpt2` (our s and m models) or `smollm2` (SmolLM2-135M). SmolLM2's streams are the same documents, decoded from the
GPT-2 streams and tokenized again, so bits per byte is over the same text.

Subcommands (each on the node, once, by the integrator):

    prepare.py data --lf-data DIR            held-out and training streams from the learning-frontier corpus (prep.py's)
    prepare.py register --name s50 --lf-theta FILE --size s     a dense start written by learning_frontier/train.py --save-theta
    prepare.py register --name smol --hf HuggingFaceTB/SmolLM2-135M
    prepare.py evaluate --start s50 --model FILE [--split search]   one entry's bits per byte, as JSON (meter.py calls this)
    prepare.py show                          every registered start

The evaluator (`evaluate_bpb`) loads an entry onto its start's architecture (strict: the same parameters, the same shapes),
counts the forward FLOPs of one 1024-token window with torch's FlopCounterMode and refuses an entry over its start's (the
eval-compute cap per byte: the same tokenizer and windows, so per window is per byte), then scores every full window of the
split at context 1024: bits per byte = summed target cross-entropy / ln 2 / the targets' UTF-8 bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LF = HERE.parent / "learning_frontier"
sys.path.insert(0, str(LF))

ROOT = Path(os.environ.get("COVERT_ROOT", "/workspace/research/scratch/covert-search"))
CTX = 1024
C_SEQS = 64                      # c is the FLOPs of one forward and backward pass over C_SEQS sequences of CTX tokens
TOKEN_BITS = 16                  # a data token crosses a block boundary as uint16
HELD_OUT_TOKENS = 2048 * 1024 + 1
FINAL_SHARD = "012"
SMOL_REPO = "HuggingFaceTB/SmolLM2-135M"
SMOL_TRAIN_TOKENS = 160_000_000  # 2.4x a 1000-block budget at 64 sequences per block
EOT = {"gpt2": 50256, "smollm2": 0}

# Size classes: k, the bits of fresh input one block may import beyond theta (MB = 10^6 bytes).
CLASSES = {"ref": None, "10MB": 8 * 10_000_000, "1MB": 8 * 1_000_000, "100kB": 8 * 100_000}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- data

def gpt2_token_bytes():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    table = np.zeros(50304, dtype=np.int64)
    for t in range(enc.n_vocab):
        if t != EOT["gpt2"]:
            table[t] = len(enc.decode_single_token_bytes(t))
    return enc, table


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level alphabet (byte -> printable character), which SmolLM2's tokenizer shares."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def smol_revision(root: Path = ROOT) -> str:
    """The SmolLM2 revision `data` resolved and recorded; every later step reads that one."""
    f = root / "data.json"
    if f.exists():
        return json.loads(f.read_text())["smollm2_revision"]
    from huggingface_hub import HfApi
    return HfApi().model_info(SMOL_REPO).sha


def smol_tokenizer(revision: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SMOL_REPO, revision=revision)
    inv = {v: k for k, v in bytes_to_unicode().items()}
    table = np.zeros(len(tok), dtype=np.int64)
    special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
    for t in range(len(tok)):
        if t not in special:
            piece = tok.convert_ids_to_tokens(t)
            table[t] = sum(1 if ch in inv else len(ch.encode()) for ch in piece)
    if tok.convert_tokens_to_ids("<|endoftext|>") != EOT["smollm2"]:
        raise SystemExit("prepare: SmolLM2's end-of-text is not token 0")
    return tok, table


def gpt2_documents(tokens: np.ndarray, enc) -> list[str]:
    """The documents a GPT-2 stream spells (each starts at an end-of-text); a cut last document is kept as cut."""
    cuts = np.flatnonzero(tokens == EOT["gpt2"]).tolist() + [len(tokens)]
    return [enc.decode(tokens[a + 1:b].tolist()) for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def smol_stream(docs: list[str], tok, limit: int) -> np.ndarray:
    pieces, have = [], 0
    for i in range(0, len(docs), 4096):
        for ids in tok(docs[i:i + 4096], add_special_tokens=False)["input_ids"]:
            pieces.append(np.asarray([EOT["smollm2"], *ids], dtype=np.uint16))
            have += len(ids) + 1
        if have >= limit:
            break
    return np.concatenate(pieces)[:limit]


def write_heldout(dest: Path, name: str, tokens: np.ndarray, table: np.ndarray) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(dest / f"{name}.bin")
    nbytes = table[tokens[1:].astype(np.int64)].astype(np.uint8)
    np.save(dest / f"{name}.bytes.npy", nbytes)
    return {"file": f"{name}.bin", "tokens": int(len(tokens)), "sha256": sha256_file(dest / f"{name}.bin"),
            "target_bytes": int(nbytes.astype(np.int64).sum())}


def cmd_data(a: argparse.Namespace) -> int:
    from prep import REPO, REVISION, SHARD, fetch, documents, tokenize  # learning_frontier/prep.py
    from huggingface_hub import HfApi
    root = a.root
    lf = json.loads((a.lf_data / "manifest.json").read_text())
    enc, g_table = gpt2_token_bytes()
    manifest = {"schema": "covert-search/data/v1", "lf_corpus": {"dir": str(a.lf_data), "train_sha256": lf["train"]["sha256"],
                                                                 "val_sha256": lf["val"]["sha256"]}, "tokenizers": {}}
    (root / "train/gpt2").mkdir(parents=True, exist_ok=True)
    link = root / "train/gpt2/train.bin"
    if not link.exists():
        link.symlink_to(a.lf_data / "train.bin")
    search = np.fromfile(a.lf_data / "val.bin", dtype=np.uint16)
    if sha256_file(a.lf_data / "val.bin") != lf["val"]["sha256"] or len(search) != HELD_OUT_TOKENS:
        raise SystemExit("prepare: the learning-frontier val.bin does not match its manifest")
    files = HfApi().list_repo_tree(REPO, path_in_repo="sample/10BT", repo_type="dataset", revision=REVISION)
    expected = {f.path: f.lfs.sha256 for f in files if getattr(f, "lfs", None)}
    shard = fetch(FINAL_SHARD, a.root / "hf-cache", expected)
    pieces, have = [], 0
    for texts in documents(shard["path"]):
        for arr in tokenize(enc, texts, 16):
            pieces.append(arr)
            have += len(arr)
        if have >= HELD_OUT_TOKENS:
            break
    final = np.concatenate(pieces)[:HELD_OUT_TOKENS]
    g = {"search": write_heldout(root / "heldout/gpt2", "search", search, g_table) | {"source": "shard 013, the first curves' val.bin"},
         "final": write_heldout(root / "heldout/gpt2", "final", final, g_table) | {"source": f"shard {FINAL_SHARD}"},
         "train": {"file": "train/gpt2/train.bin", "sha256": lf["train"]["sha256"], "tokens": lf["train"]["tokens"]}}
    if g["search"]["target_bytes"] != lf["val"]["val_bytes"]:
        raise SystemExit(f"prepare: search bytes {g['search']['target_bytes']} != the first curves' {lf['val']['val_bytes']}")
    manifest["tokenizers"]["gpt2"] = g
    manifest["smollm2_revision"] = rev = smol_revision(root)
    tok, s_table = smol_tokenizer(rev)
    s = {}
    for name, toks in (("search", search), ("final", final)):
        stream = smol_stream(gpt2_documents(toks, enc), tok, 10**12)
        s[name] = write_heldout(root / "heldout/smollm2", name, stream, s_table)
    train = np.memmap(a.lf_data / "train.bin", dtype=np.uint16, mode="r")
    stream = smol_stream(gpt2_documents(np.asarray(train[: int(SMOL_TRAIN_TOKENS * 1.25)]), enc), tok, SMOL_TRAIN_TOKENS)
    (root / "train/smollm2").mkdir(parents=True, exist_ok=True)
    stream.tofile(root / "train/smollm2/train.bin")
    s["train"] = {"file": "train/smollm2/train.bin", "sha256": sha256_file(root / "train/smollm2/train.bin"),
                  "tokens": int(len(stream)), "source": "the first documents of the GPT-2 training stream, decoded and "
                                                        f"tokenized by {SMOL_REPO}@{rev}"}
    manifest["tokenizers"]["smollm2"] = s
    (root / "data.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


def data_manifest(root: Path = ROOT) -> dict:
    return json.loads((root / "data.json").read_text())


class TrainStream:
    """The training sequences in the meter's fixed order: windows of CTX + 1 tokens at stride CTX, permuted once by
    default_rng(0) (the first curves' order), read from `offset` on (a start's own run read the ones before)."""

    def __init__(self, tokenizer: str, offset: int, root: Path = ROOT):
        info = data_manifest(root)["tokenizers"][tokenizer]["train"]
        self.tokens = np.memmap(root / info["file"], dtype=np.uint16, mode="r")
        self.n_seq = (len(self.tokens) - 1) // CTX
        self.order = np.random.default_rng(0).permutation(self.n_seq)
        self.offset = self.next = offset
        self.sha256 = info["sha256"]

    def take(self, n: int) -> np.ndarray:
        if self.next + n > self.n_seq:
            raise SystemExit(f"meter: the training stream has {self.n_seq} sequences")
        idx = self.order[self.next:self.next + n]
        self.next += n
        return np.stack([self.tokens[i * CTX: i * CTX + CTX + 1] for i in idx]).astype(np.int64)


# ---------------------------------------------------------------- starts

class HFLM:
    """A Hugging Face causal LM as a module whose forward(idx) gives logits, like learning_frontier's LM."""

    @staticmethod
    def build(hf_dir: Path):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, idx):
                return self.m(input_ids=idx, use_cache=False).logits

        cfg = AutoConfig.from_pretrained(hf_dir)
        cfg._attn_implementation = "sdpa"
        return Wrap(AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32))


def start_info(name: str, root: Path = ROOT) -> dict:
    return json.loads((root / "starts" / name / "start.json").read_text())


def build_model(info: dict, root: Path = ROOT, device: str = "cuda"):
    """The start's architecture with theta loaded and checked against its registered sha256."""
    import torch
    from model import LM, SIZES, theta_digest
    if info["arch"] == "lf":
        model = LM(SIZES[info["size"]], vocab=info["vocab"])
    else:
        model = HFLM.build(root / "starts" / info["name"] / "hf")
    state = torch.load(root / "starts" / info["name"] / "theta.pt", map_location="cpu", weights_only=True)
    if theta_digest(state) != info["theta_sha256"]:
        raise SystemExit(f"prepare: {info['name']}'s theta.pt does not match its registered sha256")
    model.load_state_dict(state, strict=True)
    return model.to(device)


def count_flops(fn) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


def measure(model, vocab: int, device: str) -> dict:
    """c (forward and backward over C_SEQS x CTX, every parameter trained) and the forward FLOPs of one window."""
    import torch
    import torch.nn.functional as F
    for p in model.parameters():
        p.requires_grad_(True)
    dev = device.split(":")[0]

    def fwd_bwd():
        for _ in range(C_SEQS // 8):
            x = torch.zeros(8, CTX, dtype=torch.long, device=device)
            with torch.autocast(dev, dtype=torch.bfloat16):
                logits = model(x)
            F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x.view(-1)).backward()
        model.zero_grad(set_to_none=True)

    def fwd():
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            model(torch.zeros(1, CTX, dtype=torch.long, device=device))

    c = count_flops(fwd_bwd)
    f = count_flops(fwd)
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    if c < 0.9 * 6 * n * C_SEQS * CTX:
        raise SystemExit(f"prepare: FlopCounterMode counted c = {c:.3e}, under 0.9 x 6 N T = {0.9 * 6 * n * C_SEQS * CTX:.3e}; "
                         "it is missing ops (an attention kernel it has no formula for?)")
    return {"c_flops": c, "fwd_flops_per_window": f, "params": n, "vocab": vocab}


def cmd_register(a: argparse.Namespace) -> int:
    import torch
    from model import LM, SIZES, VOCAB, load_theta, theta_digest
    d = a.root / "starts" / a.name
    d.mkdir(parents=True, exist_ok=True)
    if a.lf_theta:
        state, meta = load_theta(a.lf_theta)
        model = LM(SIZES[a.size])
        model.load_state_dict(state, strict=True)
        info = {"arch": "lf", "size": a.size, "tokenizer": "gpt2", "vocab": VOCAB,
                "provenance": meta | {"file": str(a.lf_theta)},
                "train_offset_sequences": int(meta["sequences_read"])}
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM
        rev = smol_revision(a.root)
        src = Path(snapshot_download(a.hf, revision=rev, cache_dir=str(a.root / "hf-cache")))
        hf = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
        (d / "hf").mkdir(exist_ok=True)
        hf.config.save_pretrained(d / "hf")
        model = HFLM.build(d / "hf")
        model.m.load_state_dict(hf.state_dict(), strict=True)
        info = {"arch": "hf", "tokenizer": "smollm2", "vocab": hf.config.vocab_size,
                "provenance": {"repo": a.hf, "revision": rev}, "train_offset_sequences": 0}
    state = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save(state, d / "theta.pt")
    info |= {"schema": "covert-search/start/v1", "name": a.name, "theta_file": "theta.pt", "theta_sha256": theta_digest(state)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    info |= measure(model, info["vocab"], dev)
    (d / "start.json").write_text(json.dumps(info, indent=1) + "\n")
    info["search_bpb"] = evaluate_bpb(model, info, "search", a.root, dev)
    (d / "start.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


# ---------------------------------------------------------------- evaluator

def evaluate_bpb(model, info: dict, split: str, root: Path = ROOT, device: str = "cuda", windows: int = 0,
                 micro: int = 16) -> dict:
    """Bits per byte of `model` on a held-out split at context CTX, refusing a model whose forward costs more per window
    than its start's."""
    import torch
    import torch.nn.functional as F
    dev = device.split(":")[0]
    model.eval()
    with torch.no_grad():
        def fwd():
            with torch.autocast(dev, dtype=torch.bfloat16):
                model(torch.zeros(1, CTX, dtype=torch.long, device=device))
        f = count_flops(fwd)
        if f > info["fwd_flops_per_window"]:
            raise SystemExit(f"evaluate: the entry's forward is {f} FLOPs per window, over its start's cap "
                             f"{info['fwd_flops_per_window']}")
        d = root / "heldout" / info["tokenizer"]
        tokens = torch.from_numpy(np.fromfile(d / f"{split}.bin", dtype=np.uint16).astype(np.int64))
        nbytes = np.load(d / f"{split}.bytes.npy")
        n = (len(tokens) - 1) // CTX
        n = min(n, windows) if windows else n
        total = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(0, n, micro):
            j = min(n, i + micro)
            rows = torch.stack([tokens[w * CTX: w * CTX + CTX + 1] for w in range(i, j)]).to(device)
            with torch.autocast(dev, dtype=torch.bfloat16):
                logits = model(rows[:, :-1])
            total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1),
                                     reduction="sum").double()
        target_bytes = int(nbytes[: n * CTX].astype(np.int64).sum())
        nats = float(total)
    return {"split": split, "bpb": nats / math.log(2) / target_bytes, "nats_per_byte": nats / target_bytes,
            "windows": n, "targets": n * CTX, "target_bytes": target_bytes, "fwd_flops_per_window": f}


def apply_entry(model, entry: dict, start: str) -> None:
    """Write an entry's parameter overrides (keyed by `named_parameters`, so a tied matrix is one key) onto the start."""
    import torch
    params = dict(model.named_parameters())
    unknown = sorted(set(entry) - set(params))
    bad = [k for k in entry if k in params and entry[k].shape != params[k].shape]
    if unknown or bad:
        raise SystemExit(f"evaluate: the entry is not {start}'s architecture (unknown {unknown[:5]}, misshapen {bad[:5]})")
    with torch.no_grad():
        for k, v in entry.items():
            params[k].copy_(v)


def cmd_evaluate(a: argparse.Namespace) -> int:
    import torch
    info = start_info(a.start, a.root)
    model = build_model(info, a.root, a.device)
    if a.model:
        apply_entry(model, torch.load(a.model, map_location=a.device, weights_only=True), a.start)
    out = evaluate_bpb(model, info, a.split, a.root, a.device, a.windows)
    print(json.dumps(out | {"start": a.start}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--lf-data", type=Path, required=True)
    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--lf-theta", type=Path)
    r.add_argument("--size")
    r.add_argument("--hf")
    e = sub.add_parser("evaluate")
    e.add_argument("--start", required=True)
    e.add_argument("--model", type=Path, help="the entry: a torch file of parameter overrides on the start (none: the start)")
    e.add_argument("--split", choices=("search", "final"), default="search")
    e.add_argument("--windows", type=int, default=0, help="score only the first N windows (a smoke test)")
    e.add_argument("--device", default="cuda")
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "data":
        return cmd_data(a)
    if a.cmd == "register":
        if bool(a.lf_theta) == bool(a.hf) or (a.lf_theta and not a.size):
            ap.error("register takes --lf-theta FILE --size S, or --hf REPO")
        return cmd_register(a)
    if a.cmd == "evaluate":
        return cmd_evaluate(a)
    for p in sorted((a.root / "starts").glob("*/start.json")):
        i = json.loads(p.read_text())
        print(f"{i['name']:8s} {i['arch']:3s} {i['tokenizer']:8s} c={i['c_flops']:.4e} fwd/window={i['fwd_flops_per_window']:.4e} "
              f"search_bpb={i.get('search_bpb', {}).get('bpb')} theta={i['theta_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
