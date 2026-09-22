"""Register a checkpoint role for the generalization sweep (laptop-side, urllib only, no HF token needed for public repos).

    uv run python -m verity_capture.bench.add_checkpoint --case TINYLLAMA --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 [--revision REV]

Appends an entry to `manifests/checkpoints.json` (frozen entries untouched; an `appended` note is added) with the pinned
revision, the sha256 of every `*.safetensors` file (from the Hub's LFS metadata) and of `config.json` (downloaded), the pod-side
`local_path` (`/workspace/hf/hub/models--<org>--<name>/snapshots/<revision>`, where `pod_up.sh`'s seed step puts it) and
`role = <case>`.  `verity_vllm_adapter.profile.checkpoint_for` resolves a case through `CASES` or, failing that, through this
`role`, so a lane needs no edit of `profile.py` to run a new model.  Also writes `data/hf_configs/<case>.config.json` (the
checkpoint's config.json) for `analytic.config_of`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests" / "checkpoints.json"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "verity-capture/add_checkpoint"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def resolve(repo: str, revision: str | None) -> dict:
    info = json.loads(_get(f"https://huggingface.co/api/models/{repo}/revision/{revision or 'main'}?blobs=true"))
    sha = info["sha"]
    files = {}
    st_sha = {}
    for s in info.get("siblings", []):
        name = s["rfilename"]
        if "/" in name:
            continue
        entry = {"size": s.get("size"), "digest_source": "hub-lfs" if s.get("lfs") else "hub"}
        if s.get("lfs"):
            entry["sha256"] = s["lfs"]["sha256"]
            if name.endswith(".safetensors"):
                st_sha[name] = s["lfs"]["sha256"]
        files[name] = entry
    cfg = _get(f"https://huggingface.co/{repo}/resolve/{sha}/config.json")
    files.setdefault("config.json", {})["sha256"] = hashlib.sha256(cfg).hexdigest()
    files["config.json"]["digest_source"] = "downloaded"
    return {"sha": sha, "files": files, "safetensors_sha256": st_sha, "config": json.loads(cfg), "gated": bool(info.get("gated"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", required=True, help="checkpoint role, e.g. TINYLLAMA (letters, digits, dashes)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None, help="commit sha to pin (default: main's current sha)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--by", default="gen", help="lane name for the appended note")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    r = resolve(a.repo, a.revision)
    if not r["safetensors_sha256"]:
        print(f"{a.repo}@{r['sha']}: no *.safetensors files at the top level", file=sys.stderr)
        return 2
    local = f"/workspace/hf/hub/models--{a.repo.replace('/', '--')}/snapshots/{r['sha']}"
    entry = {"role": a.case, "repo": a.repo, "revision": r["sha"], "gated": r["gated"], "downloaded": True, "local_path": local,
             "config_sha256": r["files"]["config.json"]["sha256"], "safetensors_sha256": r["safetensors_sha256"], "files": r["files"],
             "resolved_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "resolved_by": a.by,
             "note": "generalization sweep: seeded on the pod by verity_capture/bench/pod_up.sh (POD_CASES) / pod_env_extra.sh"}
    doc = json.loads(Path(a.manifest).read_text())
    existing = [c for c in doc["checkpoints"] if c.get("role") == a.case]
    if existing:
        print(f"role {a.case} already registered: {existing[0]['repo']}@{existing[0]['revision'][:10]}", file=sys.stderr)
        return 1
    if any(c["repo"] == a.repo and c.get("revision") == r["sha"] for c in doc["checkpoints"]):
        print(f"note: {a.repo}@{r['sha'][:10]} already present under another role; adding role {a.case} as a new entry", file=sys.stderr)
    doc["checkpoints"].append(entry)
    doc.setdefault("appended", []).insert(0, {"at": entry["resolved_at"], "by": a.by, "repos": [a.repo],
                                              "note": f"new entry only (role {a.case}); frozen entries unchanged (generalization sweep)"})
    cfg_out = REPO / "data" / "hf_configs" / f"{a.case}.config.json"
    if a.dry_run:
        print(json.dumps(entry, indent=1))
        return 0
    Path(a.manifest).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    cfg_out.write_text(json.dumps(r["config"], indent=1, sort_keys=True) + "\n")
    print(f"{a.case}: {a.repo}@{r['sha'][:10]} -> {a.manifest}; config -> {cfg_out}; "
          f"safetensors {sum(int(r['files'][f].get('size') or 0) for f in r['safetensors_sha256']) / 2**30:.2f} GiB in {len(r['safetensors_sha256'])} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
