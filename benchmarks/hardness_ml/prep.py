"""TinyStories V2 (GPT-4 split) tokenized with a 4096-entry byte-level BPE, as uint16 token files.

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
EOT = "<|endoftext|>"


def stories(path: Path, max_bytes: int | None):
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_bytes is not None:
        text = text[:max_bytes]
    return [s.strip() for s in text.split(EOT) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokenizer-bytes", type=int, default=200_000_000)
    ap.add_argument("--max-train-bytes", type=int, default=None)
    ap.add_argument("--out", default=os.environ.get("OUT", "."))
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    settings = {"repo": REPO, "files": FILES, "vocab": a.vocab, "tokenizer_bytes": a.tokenizer_bytes,
                "max_train_bytes": a.max_train_bytes}
    meta_path = data / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("settings") == settings:
        print("prep: up to date", meta_path)
        Path(a.out, "prep.json").write_text(meta_path.read_text())
        return

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    t0 = time.time()
    raw = {k: Path(hf_hub_download(REPO, f, repo_type="dataset", cache_dir=str(data / "hf"))) for k, f in FILES.items()}
    print(f"prep: downloaded in {time.time() - t0:.0f}s", {k: v.stat().st_size for k, v in raw.items()})

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(stories(raw["train"], a.tokenizer_bytes), trainer=trainer)
    tok.save(str(data / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    print(f"prep: tokenizer trained in {time.time() - t0:.0f}s, eot={eot}")

    meta = {"settings": settings, "eot": eot, "vocab": tok.get_vocab_size(), "files": {}}
    for split, path in raw.items():
        docs = stories(path, a.max_train_bytes if split == "train" else None)
        ids = []
        for i in range(0, len(docs), 100_000):
            for enc in tok.encode_batch(docs[i:i + 100_000]):
                ids.extend(enc.ids)
                ids.append(eot)
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(data / f"{split}.bin")
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        meta["files"][split] = {"docs": len(docs), "tokens": int(arr.size), "sha256": h,
                                "bytes_per_token": path.stat().st_size / max(arr.size, 1) if split == "val" else None}
        print(f"prep: {split} {len(docs)} docs {arr.size} tokens ({time.time() - t0:.0f}s)")
    meta_path.write_text(json.dumps(meta, indent=1))
    Path(a.out, "prep.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
