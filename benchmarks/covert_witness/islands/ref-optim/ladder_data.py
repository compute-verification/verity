"""The starts ladder's training stream: FineWeb-Edu `sample-100BT` at data.py's revision, GPT-2 tokens, each document
prefixed by end-of-text (learning_frontier's format), with every document of the held-out shards left out.

The held-out shards are `sample-10BT`'s 010, 011 and 013, the sources of the final, promotion and search-pool sets
(data.py). Only their documents' ids and text sha256 are read, to drop any training document that shares one; no
token of them is written. The training files are `sample-100BT`'s first `--files` in name order; its last file
(never a training file) gives `val.bin`, the first VAL_TOKENS of its kept documents, which the trainer's pilots
validate on so that hyperparameters are never chosen on a covert_witness set.

    python ladder_data.py --dest DIR --files 57 [--workers 4] [--threads 2]

DIR/parts/NNN.bin is training file NNN's tokens (uint16) and DIR/parts/NNN.json its record, written after the .bin
(the trainer waits for it). DIR/manifest.json, written last, lists every part. A rerun keeps each part whose record
matches its .bin and redoes the rest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from pathlib import Path

import numpy as np

REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TRAIN_DIR = "sample/100BT"
HELD_OUT_SHARDS = tuple(f"sample/10BT/{s}_00000.parquet" for s in ("010", "011", "013"))
EOT = 50256
VAL_TOKENS = 2 ** 22
BATCH_DOCS = 8192


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def listing() -> dict[str, str]:
    """Every parquet file under sample-100BT and the held-out shards, with the hub's LFS sha256."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = {}
    for d in (TRAIN_DIR, "sample/10BT"):
        for f in api.list_repo_tree(REPO, path_in_repo=d, repo_type="dataset", revision=REVISION):
            if getattr(f, "lfs", None) and f.path.endswith(".parquet"):
                out[f.path] = f.lfs.sha256
    return out


def fetch(name: str, cache: Path, want: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, cache_dir=str(cache)))
    got = sha256(path)
    if got != want:
        raise SystemExit(f"ladder_data: {name} has sha256 {got}, the hub's LFS id is {want}")
    return path


def drop_download(path: Path) -> None:
    """Remove a fetched parquet (this job's own cache): the symlink and the blob it points to."""
    blob = path.resolve()
    path.unlink(missing_ok=True)
    blob.unlink(missing_ok=True)


def batches(path: Path, columns=("id", "text")):
    import pyarrow.parquet as pq

    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH_DOCS, columns=list(columns)):
        yield [b.column(i).to_pylist() for i in range(len(columns))]


def text_sha(t: str) -> bytes:
    return hashlib.sha256(t.encode("utf-8")).digest()


def held_out_keys(cache: Path, files: dict[str, str]) -> tuple[set[str], set[bytes], dict]:
    ids, texts, rec = set(), set(), {}
    for name in HELD_OUT_SHARDS:
        path = fetch(name, cache, files[name])
        n = 0
        for doc_ids, docs in batches(path):
            ids.update(doc_ids)
            texts.update(text_sha(t) for t in docs)
            n += len(docs)
        rec[name] = {"sha256": files[name], "documents": n}
        drop_download(path)
    return ids, texts, rec


def encode(enc, docs: list[str], threads: int) -> np.ndarray:
    ids = enc.encode_ordinary_batch(docs, num_threads=threads)
    n = sum(len(x) for x in ids) + len(ids)
    return np.fromiter(chain.from_iterable(chain((EOT,), x) for x in ids), dtype=np.uint16, count=n)


def do_part(i: int, name: str, want: str, dest: Path, keys: Path, threads: int, limit: int | None = None) -> dict:
    """Tokenize one source file into parts/NNN.bin (or val.bin with `limit`), dropping held-out documents."""
    import pickle

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    with keys.open("rb") as f:
        bad_ids, bad_texts = pickle.load(f)
    path = fetch(name, dest / "hf-cache", want)
    out = dest / ("val.bin" if limit else f"parts/{i:03d}.bin")
    tmp = out.with_suffix(".bin.part")
    docs = kept = tokens = nbytes = 0
    dropped = []
    with tmp.open("wb") as fo:
        for doc_ids, texts in batches(path):
            keep = []
            for j, (d, t) in enumerate(zip(doc_ids, texts)):
                if d in bad_ids or text_sha(t) in bad_texts:
                    dropped.append({"row": docs + j, "id": d})
                else:
                    keep.append(t)
            docs += len(texts)
            arr = encode(enc, keep, threads)
            if limit:
                arr = arr[: limit - tokens]
            fo.write(arr.tobytes())
            kept += len(keep)
            tokens += len(arr)
            nbytes += sum(len(t.encode("utf-8")) for t in keep)
            if limit and tokens >= limit:
                break
    tmp.rename(out)
    drop_download(path)
    rec = {"index": i, "file": name, "file_sha256": want, "rows_read": docs, "documents": kept, "dropped": dropped,
           "tokens": tokens, "sha256": sha256(out)} | ({} if limit else {"text_bytes": nbytes})
    if not limit:
        (dest / f"parts/{i:03d}.json").write_text(json.dumps(rec) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--files", type=int, required=True, help="sample-100BT's first N files (name order) are the stream")
    ap.add_argument("--workers", type=int, default=4, help="files tokenized at once (one process each)")
    ap.add_argument("--threads", type=int, default=2, help="tiktoken threads per worker")
    ap.add_argument("--manifest-out", type=Path, default=None, help="also copy manifest.json here (a run's $OUT)")
    a = ap.parse_args()
    (a.dest / "parts").mkdir(parents=True, exist_ok=True)
    files = listing()
    train = sorted(f for f in files if f.startswith(TRAIN_DIR + "/"))
    if a.files >= len(train):
        raise SystemExit(f"ladder_data: --files {a.files}, but sample-100BT has {len(train)} files and its last is val.bin's")
    names, val_name = train[: a.files], train[-1]
    keys = a.dest / "held-out-keys.pkl"
    held = a.dest / "held-out.json"
    if not (keys.is_file() and held.is_file()):
        import pickle

        ids, texts, rec = held_out_keys(a.dest / "hf-cache", files)
        with (a.dest / "held-out-keys.pkl.tmp").open("wb") as f:
            pickle.dump((ids, texts), f)
        (a.dest / "held-out-keys.pkl.tmp").rename(keys)
        held.write_text(json.dumps({"shards": rec, "ids": len(ids), "text_sha256": len(texts)}) + "\n")
    print(json.dumps({"held_out": json.loads(held.read_text())}), flush=True)

    def done(i: int) -> dict | None:
        j, b = a.dest / f"parts/{i:03d}.json", a.dest / f"parts/{i:03d}.bin"
        if not (j.is_file() and b.is_file()):
            return None
        rec = json.loads(j.read_text())
        return rec if rec.get("file") == names[i] and b.stat().st_size == 2 * rec["tokens"] else None

    recs = {i: r for i in range(len(names)) if (r := done(i))}
    todo = [i for i in range(len(names)) if i not in recs]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        val = None if (a.dest / "val.json").is_file() else ex.submit(
            do_part, -1, val_name, files[val_name], a.dest, keys, a.threads, VAL_TOKENS)
        futs = [ex.submit(do_part, i, names[i], files[names[i]], a.dest, keys, a.threads) for i in todo]
        if val is not None:
            futs.append(val)
        for f in as_completed(futs):
            r = f.result()
            if f is val:
                (a.dest / "val.json").write_text(json.dumps(r) + "\n")
            else:
                recs[r["index"]] = r
            print(json.dumps({k: v for k, v in r.items() if k != "dropped"} | {"dropped": len(r["dropped"])}), flush=True)
    v = json.loads((a.dest / "val.json").read_text())
    man = {"schema": "te-ladder/stream/v1", "dataset": {"repo": REPO, "revision": REVISION, "dir": TRAIN_DIR},
           "tokenizer": "gpt2", "format": "uint16, each document prefixed by end-of-text 50256", "eot": EOT,
           "held_out": json.loads(held.read_text()),
           "rule": "a document whose id or text sha256 is a document's of sample-10BT 010, 011 or 013 is left out",
           "parts": [{k: v for k, v in recs[i].items() if k != "dropped"} | {"dropped": len(recs[i]["dropped"])}
                     for i in range(len(names))],
           "tokens": sum(recs[i]["tokens"] for i in range(len(names))),
           "val": {k: x for k, x in v.items() if k != "dropped"} | {"dropped": len(v["dropped"])}}
    (a.dest / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    if a.manifest_out:
        a.manifest_out.mkdir(parents=True, exist_ok=True)
        (a.manifest_out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({"tokens": man["tokens"], "parts": len(names), "val_tokens": v["tokens"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
