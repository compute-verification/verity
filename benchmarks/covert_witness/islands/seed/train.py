"""AUDIT candidate (measurement auditor, not a search entry): the held-out sets' copies outside `prepare.HELD_OUT`, and
the held-out documents' overlap with the training shards. Nothing is trained; the submission is θ (an empty seed log).

- `LF_CORPUS/val.bin` (learning_frontier's held-out stream, shard 013) against the search set's tokens: is it their
  prefix, read with np.fromfile, which the guard allows there;
- the fineweb-edu parquet shards in `LF_CORPUS/hf-cache` (data.py's default cache): which are there, readable, and equal
  to the shards the held-out manifests name (sha256);
- overlap: each held-out set's documents (the first `documents` rows of its shard, data.py's rule) against training
  shards 000-002, by exact text, by whitespace- and case-normalized text, and by URL; counts and byte shares only.

Held-out files are read with torch.from_file only to compare; nothing read is printed beyond counts, lengths and
equalities.

    python guard.py islands/seed/train.py --start s50 --out SUB
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

import prepare

KLASS = "seed"
TRAIN_SHARDS = ("000", "001", "002")


def from_file(path: Path, dtype=torch.uint8) -> np.ndarray:
    size = os.path.getsize(path)
    item = torch.tensor([], dtype=dtype).element_size()
    return torch.from_file(str(path), shared=False, size=size // item, dtype=dtype).numpy()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def shards() -> dict[str, Path]:
    out = {}
    for root, _, files in os.walk(prepare.LF_CORPUS / "hf-cache"):
        for f in files:
            m = re.fullmatch(r"(\d{3})_00000\.parquet", f)
            if m and "10BT" in root:
                out[m.group(1)] = Path(root) / f
    return out


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def key(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=12).digest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    a, _ = ap.parse_known_args()
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    audit: dict = {"lf_corpus": str(prepare.LF_CORPUS), "lf_corpus_files": sorted(os.listdir(prepare.LF_CORPUS))}
    man = {s: json.loads(bytes(from_file(prepare.HELD_OUT / s / "manifest.json").tolist())) for s in ("search", "final")}
    try:
        lf_man = json.loads((prepare.LF_CORPUS / "manifest.json").read_text())
        audit["lf_manifest"] = {"train_files": [f.get("file") for f in lf_man.get("train", {}).get("files", [])]
                                if isinstance(lf_man.get("train"), dict) else None,
                                "val_shard": lf_man.get("val", {}).get("shard", {}).get("file")}
    except Exception as e:  # noqa: BLE001 - the probe's result is the record
        audit["lf_manifest"] = f"unreadable: {type(e).__name__}: {e}"
    val = np.fromfile(prepare.LF_CORPUS / "val.bin", dtype=np.uint16)
    search = from_file(prepare.HELD_OUT / "search" / "gpt2.bin", torch.int16).view(np.uint16)
    n = min(len(val), len(search))
    audit["lf_val_vs_search"] = {"lf_val_tokens": int(len(val)), "search_tokens": int(len(search)),
                                 "equal_prefix_tokens": int((val[:n] == search[:n]).sum()),
                                 "lf_val_is_search_prefix": bool(n == len(val) and (val == search[:n]).all())}
    print(json.dumps({"lf_val_vs_search": audit["lf_val_vs_search"]}), flush=True)
    found = shards()
    audit["hf_cache_shards"] = sorted(found)
    for split in ("search", "final"):
        shard = man[split]["shard"].split("/")[-1][:3]
        p = found.get(shard)
        audit[f"{split}_source_shard"] = {"shard": shard, "in_cache": p is not None,
                                          "readable": bool(p and os.access(p, os.R_OK)),
                                          "sha256_matches_manifest": bool(p and sha256_file(p) == man[split]["shard_sha256"])}
    print(json.dumps({k: audit[k] for k in ("hf_cache_shards", "search_source_shard", "final_source_shard")}), flush=True)

    held = {}
    for split in ("search", "final"):
        p = found.get(man[split]["shard"].split("/")[-1][:3])
        if p is None:
            continue
        rows, need = [], man[split]["documents"]
        for batch in pq.ParquetFile(p).iter_batches(batch_size=4096, columns=["text", "url"]):
            d = batch.to_pydict()
            rows += list(zip(d["text"], d["url"]))
            if len(rows) >= need:
                break
        rows = rows[:need]
        held[split] = {"exact": {key(t): len(t.encode()) for t, _ in rows}, "norm": {key(norm(t)) for t, _ in rows},
                       "url": {u for _, u in rows}, "rows": rows,
                       "bytes_match_manifest": sum(len(t.encode()) for t, _ in rows) == man[split]["bytes"]}
    hits = {s: {"exact": set(), "norm": set(), "url": set()} for s in held}
    scanned = {}
    for shard in TRAIN_SHARDS:
        p = found.get(shard)
        if p is None:
            scanned[shard] = "not in cache"
            continue
        docs = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8192, columns=["text", "url"]):
            d = batch.to_pydict()
            docs += len(d["text"])
            for t, u in zip(d["text"], d["url"]):
                ke, kn = key(t), key(norm(t))
                for s, h in held.items():
                    if ke in h["exact"]:
                        hits[s]["exact"].add(ke)
                    if kn in h["norm"]:
                        hits[s]["norm"].add(kn)
                    if u in h["url"]:
                        hits[s]["url"].add(u)
        scanned[shard] = docs
    audit["train_shards_scanned"] = scanned
    overlap = {}
    for s, h in held.items():
        rows = h["rows"]
        total = sum(len(t.encode()) for t, _ in rows)
        by = {"exact": [len(t.encode()) for t, _ in rows if key(t) in hits[s]["exact"]],
              "norm": [len(t.encode()) for t, _ in rows if key(norm(t)) in hits[s]["norm"]],
              "url": [len(t.encode()) for t, u in rows if u in hits[s]["url"]]}
        overlap[s] = {"documents": len(rows), "bytes": total, "bytes_match_manifest": h["bytes_match_manifest"]} | {
            f"{k}_documents": len(v) for k, v in by.items()} | {f"{k}_byte_share": sum(v) / total for k, v in by.items()}
    audit["overlap_with_train_shards"] = overlap
    audit["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({"overlap_with_train_shards": overlap, "seconds": audit["seconds"]}), flush=True)

    net, rec, theta = prepare.load_start(a.start)
    final = "norm" if rec["config"]["family"] == "lf" else "model.norm.weight"
    blob = prepare.encode_seedlog([final], 0, [0.0], "f32")
    model = prepare.decode(theta, blob)
    changed = [k for k in theta if not prepare.same_bits(model[k], theta[k])]
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, {"full": [final]}, [], changed=changed,
                   notes={"audit": True, "method": "held-out reach and overlap probe; the submission is θ",
                          "probes": audit})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
