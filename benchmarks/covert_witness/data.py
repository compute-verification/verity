"""Build the held-out sets and the SmolLM2 training stream (CPU, network). Fixed: candidates never run it or read its output.

Two held-out sets of `HuggingFaceFW/fineweb-edu` (`sample-10BT`, learning_frontier's pinned revision), from shards no training
stream reads (training reads 000-002):

- `search`: shard 013's documents in row order, whole documents until they hold at least 2^21 GPT-2 tokens; the only set a
  candidate is scored on;
- `final`: shard 012's, by the same rule; never shown to the search.

Each set is tokenized by every start's tokenizer (`gpt2`: tiktoken; `smollm2`: SmolLM2-135M's), each document prefixed by the
tokenizer's end-of-text token, into `eval/<set>/<tokenizer>.bin` (uint16). `eval/<set>/manifest.json` records the documents'
UTF-8 bytes, which every tokenizer's bits per byte divides by, so scores are comparable across tokenizers. The SmolLM2 training
stream is shard 000's documents, in row order, until `SMOL_TRAIN_TOKENS`, in `train/smollm2.bin`.

    python data.py [--threads 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/{}_00000.parquet"
SETS = {"search": "013", "final": "012"}
EVAL_GPT2_TOKENS = 2 ** 21
SMOL_REPO = "HuggingFaceTB/SmolLM2-135M"
SMOL_TRAIN_SHARD = "000"
SMOL_TRAIN_TOKENS = 400_000_000
GPT2_EOT = 50256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, cache: Path, expected: dict[str, str]) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO, SHARD.format(name), repo_type="dataset", revision=REVISION, cache_dir=str(cache)))
    if sha256(path) != expected[SHARD.format(name)]:
        raise SystemExit(f"data: {path} does not match the hub's LFS id")
    return path


def documents(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
        yield batch.column(0).to_pylist()


class Tokenizers:
    def __init__(self, threads: int):
        import tiktoken
        from transformers import AutoTokenizer

        self.threads = threads
        self.gpt2 = tiktoken.get_encoding("gpt2")
        self.smol = AutoTokenizer.from_pretrained(SMOL_REPO)
        self.smol_eot = self.smol.convert_tokens_to_ids("<|endoftext|>")

    def encode(self, which: str, texts: list[str]) -> list[np.ndarray]:
        if which == "gpt2":
            ids, eot = self.gpt2.encode_ordinary_batch(texts, num_threads=self.threads), GPT2_EOT
        else:
            ids, eot = self.smol(texts, add_special_tokens=False)["input_ids"], self.smol_eot
        return [np.asarray([eot, *x], dtype=np.uint16) for x in ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--cache", type=Path, default=Path("/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1/hf-cache"))
    ap.add_argument("--manifest-out", type=Path, default=None)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from huggingface_hub import HfApi

    files = HfApi().list_repo_tree(REPO, path_in_repo="sample/10BT", repo_type="dataset", revision=REVISION)
    expected = {f.path: f.lfs.sha256 for f in files if getattr(f, "lfs", None)}
    tok = Tokenizers(a.threads)
    out = {}
    for split, shard in SETS.items():
        path = fetch(shard, a.cache, expected)
        texts, have = [], 0
        for batch in documents(path):
            for t in batch:
                texts.append(t)
                have += len(tok.gpt2.encode_ordinary(t)) + 1
                if have >= EVAL_GPT2_TOKENS:
                    break
            if have >= EVAL_GPT2_TOKENS:
                break
        d = ROOT / "eval" / split
        d.mkdir(parents=True, exist_ok=True)
        rec = {"shard": SHARD.format(shard), "shard_sha256": expected[SHARD.format(shard)], "documents": len(texts),
               "bytes": sum(len(t.encode("utf-8")) for t in texts), "tokenizers": {}}
        for which in ("gpt2", "smollm2"):
            arr = np.concatenate(tok.encode(which, texts))
            arr.tofile(d / f"{which}.bin")
            rec["tokenizers"][which] = {"tokens": int(len(arr)), "sha256": sha256(d / f"{which}.bin")}
        (d / "manifest.json").write_text(json.dumps(rec, indent=1) + "\n")
        out[split] = rec
        print(json.dumps({split: rec}), flush=True)
    path = fetch(SMOL_TRAIN_SHARD, a.cache, expected)
    d = ROOT / "train"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / "smollm2.bin.part").open("wb") as f:
        for batch in documents(path):
            for arr in tok.encode("smollm2", batch):
                f.write(arr.tobytes())
                n += len(arr)
            if n >= SMOL_TRAIN_TOKENS:
                break
    (d / "smollm2.bin.part").rename(d / "smollm2.bin")
    out["smollm2_train"] = {"shard": SHARD.format(SMOL_TRAIN_SHARD), "tokens": n, "sha256": sha256(d / "smollm2.bin")}
    (d / "manifest.json").write_text(json.dumps(out["smollm2_train"], indent=1) + "\n")
    out |= {"dataset": {"repo": REPO, "revision": REVISION}, "smollm2_tokenizer": SMOL_REPO}
    (ROOT / "eval" / "data.json").write_text(json.dumps(out, indent=1) + "\n")
    if a.manifest_out:
        a.manifest_out.mkdir(parents=True, exist_ok=True)
        (a.manifest_out / "data.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["smollm2_train"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
