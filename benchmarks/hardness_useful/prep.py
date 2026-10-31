"""FineWeb-Edu sample-10BT, one parquet file, tokenized with the model's tokenizer into uint32 train and val shards.

A document goes to val when crc32 of its id is 0 mod 500 (0.2% of documents), else to train. Each document is
BOS + its tokens + EOS. The shards are written under temporary names and renamed when complete, then meta.json. The
parquet file is downloaded into --scratch. Idempotent: a complete meta.json for the same file and tokenizer is kept.
"""

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

REPO = "HuggingFaceFW/fineweb-edu"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("HU_DATA", "data")))
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--file", default="sample/10BT/000_00000.parquet")
    ap.add_argument("--batch-docs", type=int, default=2048)
    args = ap.parse_args()
    meta_path = args.data / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("file") == args.file and meta.get("model") == args.model:
            print(f"prep: {meta_path} complete: {meta}", flush=True)
            return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    args.data.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    t0 = time.time()
    path = Path(hf_hub_download(REPO, args.file, repo_type="dataset", cache_dir=str(args.scratch)))
    t1 = time.time()
    outs = {k: open(args.data / f"{k}.bin.tmp", "wb") for k in ("train", "val")}
    n = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=args.batch_docs, columns=["text", "id"]):
        texts, ids = rb.column("text").to_pylist(), rb.column("id").to_pylist()
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        bufs = {"train": [], "val": []}
        for e, i in zip(enc, ids):
            k = "val" if zlib.crc32(i.encode()) % 500 == 0 else "train"
            bufs[k].append(np.asarray([bos] + e + [eos], dtype=np.uint32))
            docs[k] += 1
        for k, b in bufs.items():
            if b:
                a = np.concatenate(b)
                a.tofile(outs[k])
                n[k] += a.size
    for k, f in outs.items():
        f.close()
        os.rename(args.data / f"{k}.bin.tmp", args.data / f"{k}.bin")
    meta = {"file": args.file, "model": args.model, "train_tokens": n["train"], "val_tokens": n["val"],
            "train_docs": docs["train"], "val_docs": docs["val"], "bos": bos, "eos": eos, "dtype": "uint32",
            "download_s": t1 - t0, "tokenize_s": time.time() - t1}
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.rename(meta_path)
    print(f"prep: {meta}", flush=True)


if __name__ == "__main__":
    main()
