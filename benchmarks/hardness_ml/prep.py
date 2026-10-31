"""A text corpus tokenized with a 4096-entry byte-level BPE, as uint16 token files.

``--source tinystories``: TinyStories V2 (GPT-4 split), its own train and valid files. ``--source fineweb-edu``: the
first parquet shard of FineWeb-Edu's sample-10BT; its first ``--val-bytes`` of text are the validation split and the
next ``--max-train-bytes`` the training split.

Writes into --data: tokenizer.json, train.bin, val.bin, meta.json. Idempotent: an existing meta.json with the same
settings is kept.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
FW_REPO = "HuggingFaceFW/fineweb-edu"
FW_FILE = "sample/10BT/000_00000.parquet"
EOT = "<|endoftext|>"


def stories(path: Path, max_bytes: int | None):
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_bytes is not None:
        text = text[:max_bytes]
    return [s.strip() for s in text.split(EOT) if s.strip()]


def parquet_docs(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=10_000):
        yield from (t.strip() for t in batch.column(0).to_pylist() if t and t.strip())


def split_docs(docs, budgets):
    """Consecutive splits of ``docs``, each up to its byte budget (None: the rest)."""
    out, it = [], iter(docs)
    for budget in budgets:
        part, n = [], 0
        for d in it:
            part.append(d)
            n += len(d.encode("utf-8"))
            if budget is not None and n >= budget:
                break
        out.append((part, n))
    return out


def train_tokenizer(docs, vocab):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer=trainer)
    return tok


def write_tokens(tok, docs, path: Path, eot: int):
    """Encodes ``docs`` in chunks, each document followed by EOT, into ``path``; (tokens, sha256)."""
    h, n = hashlib.sha256(), 0
    with open(path, "wb") as f:
        for i in range(0, len(docs), 100_000):
            encs = tok.encode_batch(docs[i:i + 100_000])
            arr = np.fromiter((t for e in encs for t in (*e.ids, eot)), dtype=np.uint16)
            f.write(arr.tobytes())
            h.update(arr.tobytes())
            n += arr.size
    return n, h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", choices=("tinystories", "fineweb-edu"), default="tinystories")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokenizer-bytes", type=int, default=200_000_000)
    ap.add_argument("--max-train-bytes", type=int, default=None)
    ap.add_argument("--val-bytes", type=int, default=20_000_000, help="fineweb-edu only")
    ap.add_argument("--out", default=os.environ.get("OUT", "."))
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    if a.source == "tinystories":
        settings = {"repo": REPO, "files": FILES, "vocab": a.vocab, "tokenizer_bytes": a.tokenizer_bytes,
                    "max_train_bytes": a.max_train_bytes}
    else:
        settings = {"repo": FW_REPO, "files": {"all": FW_FILE}, "vocab": a.vocab, "tokenizer_bytes": a.tokenizer_bytes,
                    "max_train_bytes": a.max_train_bytes, "val_bytes": a.val_bytes}
    meta_path = data / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("settings") == settings:
        print("prep: up to date", meta_path)
        Path(a.out, "prep.json").write_text(meta_path.read_text())
        return

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    if a.source == "tinystories":
        raw = {k: Path(hf_hub_download(REPO, f, repo_type="dataset", cache_dir=str(data / "hf")))
               for k, f in FILES.items()}
        print(f"prep: downloaded in {time.time() - t0:.0f}s", {k: v.stat().st_size for k, v in raw.items()})
        tok = train_tokenizer(stories(raw["train"], a.tokenizer_bytes), a.vocab)
        splits = {"val": (stories(raw["val"], None), raw["val"].stat().st_size),
                  "train": (stories(raw["train"], a.max_train_bytes), None)}
    else:
        path = Path(hf_hub_download(FW_REPO, FW_FILE, repo_type="dataset", cache_dir=str(data / "hf")))
        print(f"prep: downloaded in {time.time() - t0:.0f}s", path.stat().st_size)
        (val, val_n), (train, _) = split_docs(parquet_docs(path), [a.val_bytes, a.max_train_bytes])
        tok = train_tokenizer(split_docs(train, [a.tokenizer_bytes])[0][0], a.vocab)
        splits = {"val": (val, val_n), "train": (train, None)}
    tok.save(str(data / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    print(f"prep: tokenizer trained in {time.time() - t0:.0f}s, eot={eot}")

    meta = {"settings": settings, "eot": eot, "vocab": tok.get_vocab_size(), "files": {}}
    for split, (docs, nbytes) in splits.items():
        n, h = write_tokens(tok, docs, data / f"{split}.bin", eot)
        meta["files"][split] = {"docs": len(docs), "tokens": n, "sha256": h,
                                "bytes_per_token": nbytes / max(n, 1) if nbytes else None}
        print(f"prep: {split} {len(docs)} docs {n} tokens ({time.time() - t0:.0f}s)")
    meta_path.write_text(json.dumps(meta, indent=1))
    Path(a.out, "prep.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
