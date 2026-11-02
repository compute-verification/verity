"""Build the held-out sets and the Pythia training stream (CPU, network). Fixed: candidates never run it or read its output.

Three held-out sets of `HuggingFaceFW/fineweb-edu` (`sample-10BT`, learning_frontier's pinned revision), each a shard's
documents in row order, whole documents until they hold the set's GPT-2 tokens, from shards no training stream reads
(training reads 000-002):

- `search-pool` (shard 013, 2^23 tokens; its first 2^21 are v2.0's search set), in `CW_ROOT/heldout/`: each search
  evaluation scores a fresh slice of it (`prepare.draw_slice`);
- `promotion` (shard 011, 2^21), sealed: it decides promotions, one bit each (`sealed.py`);
- `final` (shard 010, 2^21), sealed: scored only in final mode (`CW_FINAL=1`), by the search's starts and the sealed start.

The sealed sets are in `CW_SEALED/heldout/` (`sealed.SEALED`). Each set is tokenized by the tokenizers that score it
(`gpt2`: tiktoken; `pythia`: Pythia-160M's, the final set only), each document prefixed by the tokenizer's end-of-text
token, into `<set>/<tokenizer>.bin` (uint16). `<set>/manifest.json` records the documents' UTF-8 bytes, in total and per
document (`doc_bytes`), and each tokenizer's tokens per document (`doc_tokens`), so a slice's bits per byte divide by its
own documents' bytes and scores compare across tokenizers. The Pythia training stream is shard 000's documents in row
order until `PYTHIA_TRAIN_TOKENS`, in `CW_ROOT/train/pythia.bin`. v2.0's sets (`CW_ROOT/eval/`) are left as they were.

    python data.py [--threads 32] [--manifest-out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import prepare

REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD = "sample/10BT/{}_00000.parquet"
PYTHIA_TRAIN_SHARD = "000"
PYTHIA_TRAIN_TOKENS = 200_000_000


@dataclass(frozen=True)
class Set:
    shard: str
    gpt2_tokens: int
    tokenizers: tuple[str, ...]


SETS = {
    prepare.SEARCH_POOL: Set("013", 2 ** 23, ("gpt2",)),
    prepare.PROMOTION: Set("011", 2 ** 21, ("gpt2",)),
    prepare.FINAL: Set("010", 2 ** 21, ("gpt2", "pythia")),
}


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
    def __init__(self, threads: int, pythia_revision: str):
        import tiktoken
        from transformers import AutoTokenizer

        self.threads = threads
        self.gpt2 = tiktoken.get_encoding("gpt2")
        self.pythia = AutoTokenizer.from_pretrained(prepare.PYTHIA_REPO, revision=pythia_revision)
        if self.pythia.convert_tokens_to_ids("<|endoftext|>") != prepare.EOT["pythia"]:
            raise SystemExit("data: Pythia's end-of-text id is not prepare.EOT['pythia']")

    def encode(self, which: str, texts: list[str]) -> list[np.ndarray]:
        if which == "gpt2":
            ids = self.gpt2.encode_ordinary_batch(texts, num_threads=self.threads)
        else:
            ids = self.pythia(texts, add_special_tokens=False)["input_ids"]
        return [np.asarray([prepare.EOT[which], *x], dtype=np.uint16) for x in ids]


def build_set(name: str, spec: Set, tok: Tokenizers, path: Path, shard_sha256: str) -> dict:
    texts, gpt2, have = [], [], 0
    for batch in documents(path):
        for t, arr in zip(batch, tok.encode("gpt2", batch)):
            texts.append(t)
            gpt2.append(arr)
            have += len(arr)
            if have >= spec.gpt2_tokens:
                break
        if have >= spec.gpt2_tokens:
            break
    d = prepare.SET_DIRS[name]
    d.mkdir(parents=True, exist_ok=True)
    doc_bytes = [len(t.encode("utf-8")) for t in texts]
    rec = {"schema": "covert-witness/heldout/v2", "set": name, "shard": SHARD.format(spec.shard), "shard_sha256": shard_sha256,
           "documents": len(texts), "bytes": sum(doc_bytes), "doc_bytes": doc_bytes, "tokenizers": {}}
    for which in spec.tokenizers:
        arrs = gpt2 if which == "gpt2" else tok.encode(which, texts)
        np.concatenate(arrs).tofile(d / f"{which}.bin")
        rec["tokenizers"][which] = {"tokens": int(sum(len(a) for a in arrs)), "sha256": sha256(d / f"{which}.bin"),
                                    "doc_tokens": [len(a) for a in arrs]}
    (d / "manifest.json").write_text(json.dumps(rec) + "\n")
    return rec


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
    pythia_revision = HfApi().model_info(prepare.PYTHIA_REPO).sha
    tok = Tokenizers(a.threads, pythia_revision)
    out = {"dataset": {"repo": REPO, "revision": REVISION}, "pythia_tokenizer": {"repo": prepare.PYTHIA_REPO,
                                                                                "revision": pythia_revision}, "sets": {}}
    for name, spec in SETS.items():
        rec = build_set(name, spec, tok, fetch(spec.shard, a.cache, expected), expected[SHARD.format(spec.shard)])
        out["sets"][name] = {k: v for k, v in rec.items() if k != "doc_bytes"} | {
            "dir": str(prepare.SET_DIRS[name]), "tokenizers": {t: {k: v for k, v in r.items() if k != "doc_tokens"}
                                                              for t, r in rec["tokenizers"].items()}}
        print(json.dumps({name: out["sets"][name]}), flush=True)
    path = fetch(PYTHIA_TRAIN_SHARD, a.cache, expected)
    d = prepare.ROOT / "train"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / "pythia.bin.part").open("wb") as f:
        for batch in documents(path):
            for arr in tok.encode("pythia", batch):
                f.write(arr.tobytes())
                n += len(arr)
            if n >= PYTHIA_TRAIN_TOKENS:
                break
    (d / "pythia.bin.part").rename(d / "pythia.bin")
    out["pythia_train"] = {"shard": SHARD.format(PYTHIA_TRAIN_SHARD), "tokens": n, "sha256": sha256(d / "pythia.bin")}
    (d / "pythia.json").write_text(json.dumps(out["pythia_train"], indent=1) + "\n")
    prepare.HELD_OUT.mkdir(parents=True, exist_ok=True)
    (prepare.HELD_OUT / "data.json").write_text(json.dumps(out, indent=1) + "\n")
    if a.manifest_out:
        a.manifest_out.mkdir(parents=True, exist_ok=True)
        (a.manifest_out / "data.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pythia_train"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
