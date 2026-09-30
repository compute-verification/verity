"""Fill the Nebius server's weight cache from weights.tsv: `$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>/`, the layout
the vLLM integration reads offline (HF_HOME=/workspace/hf, HF_HUB_OFFLINE=1).

    python3 prefetch.py --list [GROUP...]              the rows, their sizes, and which are cached (stdlib only)
    uv run --no-project --with 'huggingface_hub[hf_transfer]>=0.34' python prefetch.py [GROUP... | all]

Groups: bf16-sweep (16 models, ~170 GB), fp8-port (~5 GB), frontier (DeepSeek-V3 and Kimi K2 in FP8 and NVFP4, ~2.74 TB).  Default:
bf16-sweep fp8-port.  Each row downloads at its pinned revision (resumable: rerun after an interruption), is skipped when there
is not room for it plus 5%, and leaves `$HF_HOME/prefetch/<org>--<name>@<revision>.json` (bytes, files, when) once complete.
HF_TOKEN is optional (every repo listed is public).  At the disk's ~2 GiB/s and a typical 0.5-1 GB/s from the Hub, the frontier
group takes about 1-1.5 h.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

TSV = Path(__file__).resolve().parent / "weights.tsv"
DEFAULT_GROUPS = ("bf16-sweep", "fp8-port")
IGNORE = ["original/*", "*.pth", "*.gguf", "*.msgpack", "*.h5", "onnx/*", "flax_model*", "tf_model*"]


def rows(path: Path = TSV) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        group, repo, revision, fmt, gb, role = line.split("\t")
        out.append({"group": group, "repo": repo, "revision": revision, "format": fmt, "gb": float(gb), "role": role})
    return out


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or "/workspace/hf")


def record_path(r: dict) -> Path:
    return hf_home() / "prefetch" / f"{r['repo'].replace('/', '--')}@{r['revision']}.json"


def select(all_rows: list[dict], groups: list[str]) -> list[dict]:
    groups = groups or list(DEFAULT_GROUPS)
    known = {r["group"] for r in all_rows}
    if "all" in groups:
        return all_rows
    if bad := sorted(set(groups) - known):
        raise SystemExit(f"prefetch: unknown group(s) {bad}; known: {sorted(known)} or all")
    return [r for r in all_rows if r["group"] in groups]


def listing(chosen: list[dict]) -> None:
    for r in chosen:
        print(f"{'cached ' if record_path(r).is_file() else 'missing'} {r['group']:<10} {r['gb']:>7.1f} GB  {r['repo']}@{r['revision'][:12]}")
    print(f"{sum(r['gb'] for r in chosen):.1f} GB in {len(chosen)} checkpoints")


def fetch(r: dict) -> str:
    from huggingface_hub import snapshot_download

    rec = record_path(r)
    if rec.is_file():
        return "cached"
    free_gb = shutil.disk_usage(hf_home()).free / 1e9
    if free_gb < r["gb"] * 1.05:
        return f"skipped: {free_gb:.0f} GB free, needs {r['gb'] * 1.05:.0f}"
    t0 = time.time()
    snap = Path(snapshot_download(r["repo"], revision=r["revision"], cache_dir=hf_home() / "hub", ignore_patterns=IGNORE, max_workers=16))
    files = [p for p in snap.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    rec.parent.mkdir(parents=True, exist_ok=True)
    tmp = rec.with_suffix(".tmp")
    tmp.write_text(json.dumps({"repo": r["repo"], "revision": r["revision"], "format": r["format"], "role": r["role"], "snapshot": str(snap),
                               "files": len(files), "bytes": size, "seconds": round(time.time() - t0),
                               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    os.replace(tmp, rec)
    return f"{size / 1e9:.1f} GB in {time.time() - t0:.0f} s"


def main(argv: list[str]) -> int:
    lst = "--list" in argv
    chosen = select(rows(), [a for a in argv if a != "--list"])
    if lst:
        listing(chosen)
        return 0
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    failed = 0
    for r in chosen:
        try:
            what = fetch(r)
        except Exception as e:  # noqa: BLE001  (one bad repo must not stop the rest; rerun resumes it)
            what, failed = f"FAILED {type(e).__name__}: {str(e)[:200]}", failed + 1
        print(f"{r['group']:<10} {r['repo']}@{r['revision'][:12]}: {what}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
