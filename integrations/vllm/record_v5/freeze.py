"""[A9 contract 2026-09-10] Source freeze.

``freeze.json`` (``veritor/a9/source-freeze/v1``) is written once at the start of a run: HEAD commit, ``sha256(git diff HEAD)`` (the dirty
patch, saved verbatim as ``dirty.patch``), the untracked files under the source roots, and a sha256 per source file under the roots
the runner imports (this repo's ``e2e``, ``verity_vllm_adapter``, ``verity_vllm_sampler``, ``scripts``,
``src/veritor``, ``record_v5``, ``record_6k`` (imported read-only) and every dependency export's ``vllm-poc``).  Every later phase calls
:func:`check` FIRST: it recomputes the manifest and REFUSES (``FreezeError``, exit 3) if any file changed / appeared / disappeared, and
records the outcome as ``freeze-check-<phase>.json`` (``ok`` or the list of changed paths) -- so the record shows the check was made.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from record_v5 import POC, ROOT

SCHEMA = "veritor/a9/source-freeze/v1"
SOURCE_ROOTS = ("e2e", "verity_vllm_adapter", "verity_vllm_sampler", "scripts", "src/veritor", "record_v5", "record_6k")
SUFFIXES = (".py", ".sh", ".json", ".toml", ".rs", ".txt")
SKIP_DIRS = ("__pycache__", ".pytest_cache", "target", "results", "bundles", "fixtures", "node_modules")


class FreezeError(RuntimeError):
    code = "E_SOURCE_CHANGED"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            if fn.endswith(SUFFIXES):
                out.append(Path(dp) / fn)
    return out


def manifest(extra_roots: list[Path] | None = None) -> dict[str, str]:
    roots = [ROOT / r for r in SOURCE_ROOTS] + [Path(p) / "vllm-poc" for p in (extra_roots or [])]
    m: dict[str, str] = {}
    for r in roots:
        for f in _files(r):
            try:
                key = str(f.relative_to(ROOT))
            except ValueError:
                key = str(f)
            m[key] = _sha(f)
    return m


def _dep_roots() -> list[Path]:
    return [Path(v) for k, v in os.environ.items() if k.startswith("VERITOR_DEP_") and v]


def freeze(record_dir: Path) -> dict[str, Any]:
    """Write ``freeze.json`` + ``dirty.patch``; return the document."""

    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    export = ROOT / "EXPORT.json"  # a pod checkout is a `git archive` export: the laptop writes commit + dirty-patch digest into it (record_v5/ship.sh)
    if (ROOT / ".git").exists():
        head = _git("rev-parse", "HEAD").strip()
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
        patch = _git("diff", "HEAD", "--", *[r for r in SOURCE_ROOTS if (ROOT / r).exists()])
        untracked = [l for l in _git("ls-files", "--others", "--exclude-standard", "--", *[r for r in SOURCE_ROOTS if (ROOT / r).exists()]).splitlines() if l]
        (record_dir / "dirty.patch").write_text(patch)
        rev = {"branch": branch, "commit": head, "dirty_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "dirty_patch_bytes": len(patch), "untracked": untracked, "revision_source": "git (worktree)"}
    elif export.exists():
        e = json.loads(export.read_text())
        rev = {"branch": e.get("branch"), "commit": e["commit"], "dirty_patch_sha256": e["dirty_patch_sha256"], "dirty_patch_bytes": e.get("dirty_patch_bytes"), "untracked": e.get("untracked", []),
               "revision_source": f"EXPORT.json written at export time ({e.get('exported_utc')}) from {e.get('from_worktree')}; the file manifest below is computed here"}
    else:
        raise FreezeError("E_NO_REVISION: neither a git worktree nor EXPORT.json (record_v5/ship.sh) -- the source revision cannot be recorded; refusing to start")
    deps = _dep_roots()
    doc = {
        "schema": SCHEMA, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "worktree": str(ROOT), **rev,
        "dependency_exports": [str(d) for d in deps], "source_roots": list(SOURCE_ROOTS), "manifest": manifest(deps),
    }
    (record_dir / "freeze.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def check(record_dir: Path, phase: str) -> dict[str, Any]:
    """Recompute; refuse if anything under the source roots changed since ``freeze.json``; record the check either way."""

    record_dir = Path(record_dir)
    fz = json.loads((record_dir / "freeze.json").read_text())
    now = manifest([Path(d) for d in fz.get("dependency_exports", [])])
    old = fz["manifest"]
    changed = sorted(k for k in set(old) | set(now) if old.get(k) != now.get(k))
    if (ROOT / ".git").exists():
        head = _git("rev-parse", "HEAD").strip()
        patch_sha = hashlib.sha256(_git("diff", "HEAD", "--", *[r for r in SOURCE_ROOTS if (ROOT / r).exists()]).encode()).hexdigest()
    else:
        # a git-archive export (record_v5/ship.sh): the revision is EXPORT.json's; the file manifest above is the live check
        e = json.loads((ROOT / "EXPORT.json").read_text())
        head, patch_sha = e["commit"], e["dirty_patch_sha256"]
    res = {
        "schema": "veritor/a9/source-freeze-check/v1", "phase": phase, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_same": head == fz["commit"], "dirty_patch_same": patch_sha == fz["dirty_patch_sha256"], "changed_files": changed,
        "status": "ok" if not changed and head == fz["commit"] and patch_sha == fz["dirty_patch_sha256"] else "REFUSED",
    }
    (record_dir / f"freeze-check-{phase}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if res["status"] != "ok":
        raise FreezeError(f"{FreezeError.code}: sources changed since freeze (commit_same={res['commit_same']}, dirty_patch_same={res['dirty_patch_same']}, "
                          f"{len(changed)} files: {changed[:8]}{'...' if len(changed) > 8 else ''}) -- phase {phase} refused")
    return res


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="source freeze: `freeze DIR` writes freeze.json; `check DIR PHASE` refuses on change (exit 3)")
    ap.add_argument("cmd", choices=("freeze", "check"))
    ap.add_argument("record_dir")
    ap.add_argument("phase", nargs="?", default="manual")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "freeze":
            d = freeze(Path(a.record_dir))
            print(f"[freeze] {d['commit'][:12]} dirty_patch sha256 {d['dirty_patch_sha256'][:16]} ({d['dirty_patch_bytes']} B) files={len(d['manifest'])}")
        else:
            r = check(Path(a.record_dir), a.phase)
            print(f"[freeze] check {a.phase}: {r['status']}")
        return 0
    except FreezeError as e:
        print(f"[freeze] {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
