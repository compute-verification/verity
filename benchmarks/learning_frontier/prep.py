"""Build the learning-frontier corpus: fixed fineweb-edu shards, GPT-2 tokens, a fixed held-out split.

Downloads the named parquet shards of `HuggingFaceFW/fineweb-edu` (config `sample-10BT`) at a pinned revision, checks each
file's sha256 against the hub's LFS id, and tokenizes every document with tiktoken's `gpt2` encoding, each document prefixed
by the end-of-text token (50256). The training stream is the training shards' documents in file order, concatenated
(`train.bin`, uint16); the held-out stream is the held-out shard's documents in file order, cut at `VAL_TOKENS + 1` tokens
(`val.bin`). `manifest.json` records the dataset, revision, files and digests, the tokenizer, the counts, and `val_bytes`:
the UTF-8 bytes of the text the held-out targets (tokens 1..VAL_TOKENS) spell, the end-of-text token counting zero bytes,
which `train.py` divides the summed held-out loss by to report nats per byte.

    python prep.py --dest DIR [--train-shards 000,001,002] [--val-shard 013] [--threads 16]

Run it once per corpus version; `train.py --data DIR` checks the digests before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/{}_00000.parquet"
ENCODING = "gpt2"
EOT = 50256
VAL_TOKENS = 2048 * 1024
BATCH_DOCS = 8192


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, cache: Path, expected: dict[str, str]) -> dict:
    path = Path(hf_hub_download(REPO, SHARD.format(name), repo_type="dataset", revision=REVISION, cache_dir=str(cache)))
    digest = sha256(path)
    if digest != expected[SHARD.format(name)]:
        raise SystemExit(f"prep: {path} has sha256 {digest}, the hub's LFS id is {expected[SHARD.format(name)]}")
    return {"file": SHARD.format(name), "sha256": digest, "bytes": path.stat().st_size, "path": path}


def documents(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
        yield batch.column(0).to_pylist()


def tokenize(enc: tiktoken.Encoding, texts: list[str], threads: int) -> list[np.ndarray]:
    out = []
    for ids in enc.encode_ordinary_batch(texts, num_threads=threads):
        a = np.empty(len(ids) + 1, dtype=np.uint16)
        a[0] = EOT
        a[1:] = ids
        out.append(a)
    return out


def token_bytes(enc: tiktoken.Encoding) -> np.ndarray:
    table = np.zeros(enc.n_vocab, dtype=np.int64)
    for t in range(enc.n_vocab):
        if t != EOT:
            table[t] = len(enc.decode_single_token_bytes(t))
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--train-shards", default="000,001,002")
    ap.add_argument("--val-shard", default="013")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--manifest-out", type=Path, default=None, help="also copy manifest.json here (a run's $OUT)")
    a = ap.parse_args()
    train_names = a.train_shards.split(",")
    if a.val_shard in train_names:
        raise SystemExit("prep: the held-out shard must not be a training shard")
    a.dest.mkdir(parents=True, exist_ok=True)
    cache = a.dest / "hf-cache"
    files = HfApi().list_repo_tree(REPO, path_in_repo="sample/10BT", repo_type="dataset", revision=REVISION)
    expected = {f.path: f.lfs.sha256 for f in files if getattr(f, "lfs", None)}
    enc = tiktoken.get_encoding(ENCODING)
    if enc.eot_token != EOT or enc.n_vocab != 50257:
        raise SystemExit(f"prep: {ENCODING} has eot {enc.eot_token} and {enc.n_vocab} tokens, expected {EOT} and 50257")
    ranks = hashlib.sha256(json.dumps(sorted((k.hex(), v) for k, v in enc._mergeable_ranks.items())).encode()).hexdigest()

    shards, n_docs, n_tokens, text_bytes = [], 0, 0, 0
    with (a.dest / "train.bin.part").open("wb") as out:
        for name in train_names:
            s = fetch(name, cache, expected)
            docs = toks = nbytes = 0
            for texts in documents(s["path"]):
                for arr in tokenize(enc, texts, a.threads):
                    out.write(arr.tobytes())
                    toks += len(arr)
                docs += len(texts)
                nbytes += sum(len(t.encode("utf-8")) for t in texts)
            shards.append({k: v for k, v in s.items() if k != "path"} | {"documents": docs, "tokens": toks, "text_bytes": nbytes})
            n_docs, n_tokens, text_bytes = n_docs + docs, n_tokens + toks, text_bytes + nbytes
            print(f"prep: {s['file']}: {docs} documents, {toks} tokens", file=sys.stderr, flush=True)
    (a.dest / "train.bin.part").rename(a.dest / "train.bin")

    v = fetch(a.val_shard, cache, expected)
    pieces, have, docs = [], 0, 0
    for texts in documents(v["path"]):
        for arr in tokenize(enc, texts, a.threads):
            pieces.append(arr)
            have += len(arr)
            docs += 1
            if have >= VAL_TOKENS + 1:
                break
        if have >= VAL_TOKENS + 1:
            break
    val = np.concatenate(pieces)[: VAL_TOKENS + 1]
    val.tofile(a.dest / "val.bin")
    val_bytes = int(token_bytes(enc)[val[1:].astype(np.int64)].sum())

    manifest = {
        "schema": "learning-frontier/corpus/v1",
        "dataset": {"repo": REPO, "config": "sample-10BT", "revision": REVISION},
        "tokenizer": {"library": "tiktoken", "version": tiktoken.__version__, "encoding": ENCODING, "n_vocab": enc.n_vocab,
                      "eot": EOT, "mergeable_ranks_sha256": ranks},
        "document_rule": "each document's text, tokenized with encode_ordinary and prefixed by the end-of-text token, "
                         "in the shard's row order; shards in the order listed",
        "train": {"shards": shards, "documents": n_docs, "tokens": n_tokens, "text_bytes": text_bytes, "file": "train.bin",
                  "dtype": "uint16", "sha256": sha256(a.dest / "train.bin")},
        "val": {"shard": {k: val_ for k, val_ in v.items() if k != "path"}, "documents_touched": docs, "tokens": int(len(val)),
                "targets": VAL_TOKENS, "val_bytes": val_bytes, "file": "val.bin", "dtype": "uint16",
                "sha256": sha256(a.dest / "val.bin"),
                "rule": f"the held-out shard's first documents, cut at {VAL_TOKENS + 1} tokens; targets are tokens 1.."
                        f"{VAL_TOKENS}; val_bytes sums their UTF-8 byte lengths, end-of-text counting 0"},
    }
    (a.dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if a.manifest_out:
        a.manifest_out.mkdir(parents=True, exist_ok=True)
        (a.manifest_out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"train_tokens": n_tokens, "val_tokens": int(len(val)), "val_bytes": val_bytes}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
