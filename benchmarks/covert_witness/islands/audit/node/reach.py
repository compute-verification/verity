"""AUDIT node probe (measurement auditor, not a search entry; CPU, read-only): what a candidate's train.py can reach on
vy-nebius-2 beside the held-out directory the harness guards, and whether the search set overlaps the training stream.

Writes $OUT/reach.json:

- the learning_frontier corpus's files (LF_CORPUS, which the guard doesn't cover) and its manifest; whether its val.bin
  holds the search set's tokens (shared prefix, and how many of the search set's 1024-token windows occur in val.bin);
- the hub parquet shards in LF_CORPUS/hf-cache, the sha256 of 012 and 013 against the held-out manifests' shard_sha256;
- exact-duplicate documents: the search set's documents (shard 013's first `documents` rows) hashed against shards
  000-002's;
- os.access(R_OK / W_OK) on the held-out files, the starts' θ and records, the training stream, and the other candidates'
  scratch directories (counted, never opened), as the run's user.

    python benchmarks/covert_witness/islands/audit/node/reach.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
LF = Path(os.environ.get("CW_LF_CORPUS", "/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1"))
OUT = Path(os.environ.get("OUT", "out"))
W = 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def access(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "read": os.access(path, os.R_OK),
            "write": os.access(path, os.W_OK), "dir_write": os.access(path.parent, os.W_OK),
            "owner_uid": path.stat().st_uid if path.exists() else None,
            "mode": oct(path.stat().st_mode & 0o777) if path.exists() else None}


def listing(d: Path, depth: int = 2) -> list[dict]:
    out = []
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        if len(p.relative_to(d).parts) > depth + 4:
            continue
        if p.is_file():
            out.append({"path": str(p.relative_to(d)), "bytes": p.stat().st_size})
    return out[:200]


def windows_in(search: np.ndarray, other: np.ndarray) -> dict:
    """How many of the set's W-token windows at stride W occur anywhere in `other`, and their shared prefix."""
    n = (len(search) - W) // W + 1
    mine = {search[i * W: i * W + W].tobytes() for i in range(max(0, n))}
    prefix = 0
    m = min(len(search), len(other))
    if m:
        neq = np.nonzero(search[:m] != other[:m])[0]
        prefix = int(neq[0]) if len(neq) else m
    found = 0
    if mine and len(other) >= W:
        b = other.tobytes()
        for k in mine:
            if b.find(k) >= 0:
                found += 1
    return {"search_windows": len(mine), "found_in_other": found, "shared_prefix_tokens": prefix,
            "search_tokens": int(len(search)), "other_tokens": int(len(other))}


def doc_hashes(path: Path, limit: int | None = None) -> list[str]:
    import pyarrow.parquet as pq

    out = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
        for t in batch.column(0).to_pylist():
            out.append(hashlib.sha256(t.encode()).hexdigest())
            if limit is not None and len(out) >= limit:
                return out
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"schema": "te-audit/node-reach/v1", "root": str(ROOT), "lf": str(LF), "uid": os.getuid(),
                 "host": os.uname().nodename}
    man = {s: json.loads((ROOT / "eval" / s / "manifest.json").read_text()) for s in ("search", "final")}
    rep["held_out_manifests"] = {s: {k: v for k, v in m.items() if k != "tokenizers"} for s, m in man.items()}
    rep["lf_files"] = listing(LF)
    lfman = LF / "manifest.json"
    rep["lf_manifest"] = json.loads(lfman.read_text()) if lfman.exists() else None
    search = np.fromfile(ROOT / "eval" / "search" / "gpt2.bin", dtype=np.uint16)
    final = np.fromfile(ROOT / "eval" / "final" / "gpt2.bin", dtype=np.uint16)
    for name in ("val.bin",):
        p = LF / name
        if p.exists():
            other = np.fromfile(p, dtype=np.uint16)
            rep[f"lf_{name}_vs_search"] = windows_in(search, other)
            rep[f"lf_{name}_vs_final"] = windows_in(final, other)
    shards = {}
    for p in sorted(LF.rglob("*.parquet")):
        shards[p.name] = {"path": str(p.relative_to(LF)), "bytes": p.stat().st_size, "read": os.access(p, os.R_OK)}
    for s, m in man.items():
        name = Path(m["shard"]).name
        if name in shards:
            p = LF / shards[name]["path"]
            shards[name]["sha256_matches_" + s] = sha256(p) == m["shard_sha256"]
    rep["lf_parquet"] = shards
    if "013_00000.parquet" in shards and all(f"00{i}_00000.parquet" in shards for i in range(3)):
        n = man["search"]["documents"]
        mine = doc_hashes(LF / shards["013_00000.parquet"]["path"], n)
        train = set()
        for i in range(3):
            train.update(doc_hashes(LF / shards[f"00{i}_00000.parquet"]["path"]))
        rep["search_docs_in_train_shards"] = {"search_documents": len(mine), "distinct": len(set(mine)),
                                              "exact_duplicates_in_000_002": sum(h in train for h in mine),
                                              "train_documents": len(train)}
    paths = [ROOT / "eval" / "search" / "gpt2.bin", ROOT / "eval" / "search" / "manifest.json",
             ROOT / "eval" / "final" / "gpt2.bin", LF / "train.bin", LF / "val.bin"]
    for d in sorted((ROOT / "starts").glob("*"))[:8]:
        paths += [d / "theta.safetensors", d / "start.json"]
    rep["access"] = [access(p) for p in paths]
    cands = ROOT / "candidates"
    live = sorted(cands.glob("*")) if cands.exists() else []
    rep["candidates_scratch"] = {"dir": str(cands), "entries": len(live),
                                 "readable_models": sum(os.access(d / "sub" / "model.safetensors", os.R_OK) for d in live),
                                 "readable_E": sum(os.access(d / "sub" / "E.bin", os.R_OK) for d in live),
                                 "dir_write": os.access(cands, os.W_OK) if cands.exists() else None}
    runs = Path("/workspace/research/runs")
    rep["runs_dir"] = {"dir": str(runs), "exists": runs.exists(),
                       "witness_E_files": len(list(runs.glob("*/out/E.bin"))) if runs.exists() else None}
    rep["seconds"] = round(time.time() - t0, 1)
    (OUT / "reach.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in rep if k.startswith(("lf_val", "search_docs", "candidates", "runs_dir"))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
