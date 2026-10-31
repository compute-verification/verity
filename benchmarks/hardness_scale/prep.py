"""FineWeb-Edu sample-10BT tokenized with SmolLM2's 49,152-entry tokenizer (the Tier 2 teacher's), as uint16 shards.

Each parquet file becomes a train shard (train_NNN.bin) and a val shard (val_NNN.bin). A document goes to val when crc32 of
its id is 0 mod 1000, so val is 0.1% of documents from every crawl dump (the sample is ordered by dump). Documents are
separated by <|endoftext|>. A shard is written under a temporary name and renamed when complete, with NNN.json beside it,
so --files can split the files over parallel runs; whichever run finds every file done writes meta.json with
``complete``. Parquet files are downloaded to --scratch and deleted once tokenized. Idempotent.
"""

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

REPO = "HuggingFaceFW/fineweb-edu"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
TOKENIZER = "HuggingFaceTB/SmolLM2-1.7B"
EOT = "<|endoftext|>"


def tokenizer_path(data: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dst = data / "tokenizer.json"
    if dst.exists():
        return dst
    for cache in (os.environ.get("HSC_HF_HUB", "/workspace/hf/hub"), None):
        try:
            src = hf_hub_download(TOKENIZER, "tokenizer.json", cache_dir=cache, local_files_only=cache is not None)
            break
        except Exception as e:  # noqa: BLE001 — fall back to a download into the default cache
            print(f"prep: tokenizer not in {cache}: {type(e).__name__}", flush=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(Path(src).read_bytes())
    tmp.rename(dst)
    return dst


def one_file(i: int, data: Path, scratch: Path, tok, eot: int, batch_docs: int):
    done = data / f"{i:03d}.json"
    if done.exists():
        print(f"prep: file {i} already done", flush=True)
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    path = Path(hf_hub_download(REPO, FILES[i], repo_type="dataset", cache_dir=str(scratch)))
    t1 = time.time()
    outs = {k: open(data / f"{k}_{i:03d}.bin.tmp", "wb") for k in ("train", "val")}
    n = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    val_bytes = 0
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_docs, columns=["text", "id"]):
        texts, ids = rb.column("text").to_pylist(), rb.column("id").to_pylist()
        split = ["val" if zlib.crc32(x.encode()) % 1000 == 0 else "train" for x in ids]
        enc = tok.encode_batch(texts, add_special_tokens=False)
        bufs = {"train": [], "val": []}
        for e, k, t in zip(enc, split, texts):
            bufs[k].append(np.asarray(e.ids, dtype=np.uint16))
            bufs[k].append(np.asarray([eot], dtype=np.uint16))
            docs[k] += 1
            if k == "val":
                val_bytes += len(t.encode())
        for k, b in bufs.items():
            if b:
                a = np.concatenate(b)
                a.tofile(outs[k])
                n[k] += a.size
    for k, f in outs.items():
        f.close()
        os.rename(data / f"{k}_{i:03d}.bin.tmp", data / f"{k}_{i:03d}.bin")
    rec = {"file": FILES[i], "train_tokens": n["train"], "val_tokens": n["val"], "train_docs": docs["train"],
           "val_docs": docs["val"], "val_bytes": val_bytes, "download_s": t1 - t0, "tokenize_s": time.time() - t1}
    tmp = done.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(done)
    path.unlink()
    print(f"prep: file {i}: {rec}", flush=True)


def write_meta(data: Path, out: Path | None):
    recs = []
    for i in range(len(FILES)):
        p = data / f"{i:03d}.json"
        if not p.exists():
            print(f"prep: file {i} not done yet; meta.json waits for it", flush=True)
            return
        recs.append(json.loads(p.read_text()))
    meta = {
        "complete": True,
        "source": f"{REPO} sample-10BT",
        "tokenizer": TOKENIZER,
        "vocab": 49152,
        "eot": json.loads((data / "eot.json").read_text())["eot"],
        "train": [{"file": f"train_{i:03d}.bin", "tokens": r["train_tokens"]} for i, r in enumerate(recs)],
        "val": [{"file": f"val_{i:03d}.bin", "tokens": r["val_tokens"]} for i, r in enumerate(recs)],
        "train_tokens": sum(r["train_tokens"] for r in recs),
        "val_tokens": sum(r["val_tokens"] for r in recs),
        "val_bytes_per_token": sum(r["val_bytes"] for r in recs) / max(1, sum(r["val_tokens"] for r in recs)),
        "files": recs,
    }
    tmp = data / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.rename(data / "meta.json")
    if out:
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"prep: complete, {meta['train_tokens']} train and {meta['val_tokens']} val tokens", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("HSC_DATA"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--files", default="0-13", help="a range, e.g. 0-3")
    ap.add_argument("--batch-docs", type=int, default=4096)
    ap.add_argument("--out", default=os.environ.get("OUT"))
    a = ap.parse_args()
    from tokenizers import Tokenizer

    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(tokenizer_path(data)))
    eot = tok.token_to_id(EOT)
    if tok.get_vocab_size() != 49152 or eot is None:
        raise SystemExit(f"unexpected tokenizer: vocab {tok.get_vocab_size()}, eot {eot}")
    tmp = data / f"eot.json.{os.getpid()}"
    tmp.write_text(json.dumps({"eot": eot}))
    tmp.rename(data / "eot.json")
    lo, hi = (int(x) for x in a.files.split("-"))
    for i in range(lo, hi + 1):
        one_file(i, data, Path(a.scratch), tok, eot, a.batch_docs)
    write_meta(data, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
