"""verity_vllm/check/protected.py — PROTECTED.  The frozen list of protected paths and the mechanical diff check the sealed
verifier runs on a lane's candidate commit.

    uv run python -m verity_vllm.check.protected --base <core-commit> --head <lane-commit> --out provenance.json

"What counts as evidence" is not lane-writable: a lane's mergeable diff may touch only its capability surface (config,
weight-name maps, more-specific patterns, specializations, quarantined definitions, tests, workloads, notes).  This module
turns that rule from prose into a check: `git diff --name-status base...head` is classified against `PROTECTED` and the
counts land on the card's top section (`PROTECTED FILES MODIFIED`, `EXISTING PRIMITIVES MODIFIED`, `NEW ROOT CLASSES`,
`NEW ALLOWLIST ENTRIES`, `OLD GOLDEN DIGEST CHANGES`).  A lane may hack anything locally while diagnosing; only the
candidate commit is judged.

Also reports the informational rows: `new_patterns` (pattern names in head's profiles that the base profiles lack) and
`new_definitions` (definitions added under `verity_vllm/program/registry/quarantine/`).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: Frozen.  Paths a lane may not touch at all (any diff status).  Globs are repo-relative; fnmatch `*` crosses `/`.
PROTECTED: tuple[str, ...] = (
    # what counts as evidence
    "verity_vllm/check/gates.py", "verity_vllm/harness/card.py", "verity_vllm/input_provenance/root_policy.py", "verity_vllm/check/golden.py",
    "verity_vllm/check/golden/*", "verity_vllm/check/holdout.py", "verity_vllm/check/protected.py", "verity_vllm/check/kernel_allowlist.py",
    "verity_vllm/check/replay.py", "verity_vllm/check/operand_provenance.py", "verity_vllm/check/noninterference.py", "verity_vllm/check/census.py", "verity_vllm/check/fold_compare.py",
    "verity_vllm/input_provenance/analytic.py", "verity_vllm/check/difftest.py", "verity_vllm/check/quarantine_lint.py",
    "verity_vllm/harness/run_config.py", "verity_vllm/ops/verify_lane.sh",
    # the derivation itself (observer, resolver core, existing matchers, memory, fold, program identity, engine adapter)
    "verity_vllm/observe/observer.py", "verity_vllm/observe/events.py", "verity_vllm/observe/log.py", "verity_vllm/observe/memory.py",
    "verity_vllm/observe/fold.py", "verity_vllm/observe/resolver.py", "verity_vllm/observe/patterns.py", "verity_vllm/observe/vllm_adapter.py",
    # the two base profiles (a lane derives its own profile module from `build_profile`; it does not edit these)
    "verity_vllm/observe/profiles/vllm_d9105ea80_sm89_eager.py", "verity_vllm/observe/profiles/vllm_d9105ea80_sm89_eager_qwen15.py",
    # existing registered semantics and the IR
    "verity_vllm/*",
    # frozen decisions / accepted results / the design notes the integrator owns
    "out/capture/decisions.md", "data/logs/*", "verity_vllm/observe/DESIGN.md", "out/gen/RULES.md",
)
#: Frozen.  Existing files a lane may not modify or delete, in trees where ADDING files is the normal lane move.
PROTECTED_EXISTING: tuple[str, ...] = (
    "tests/observe/*", "verity_vllm/observe/profiles/*", "verity_vllm/ops/*", "verity_vllm/observe/*.py",
    "verity_vllm/observe/*", "manifests/*",
)
#: Frozen.  Lane-writable exceptions inside otherwise protected trees.
WRITABLE_EXCEPTIONS: tuple[str, ...] = (
    "verity_vllm/program/registry/quarantine/*",
)
#: Existing registered primitive semantics (a change here is `EXISTING PRIMITIVES MODIFIED`).
PRIMITIVE_PATHS: tuple[str, ...] = ("verity_vllm/program/registry/*",)
#: The checkpoint manifest: appending entries is lane work; changing an existing entry is not.
MANIFEST = "manifests/checkpoints.json"


def _match(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, p.split(":")[0]) for p in patterns)


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def classify(changed: list[tuple[str, str]]) -> dict[str, Any]:
    """`changed`: [(status, path)] from `git diff --name-status`.  Returns the provenance rows."""
    prot, prims, allow, golden, roots = [], [], [], [], []
    for st, path in changed:
        if _match(path, WRITABLE_EXCEPTIONS):
            continue
        if path == MANIFEST:
            continue                                        # judged by `manifest_frozen_entries_changed`
        if _match(path, PROTECTED):
            prot.append({"status": st, "path": path, "rule": "protected"})
        elif st != "A" and _match(path, PROTECTED_EXISTING):
            prot.append({"status": st, "path": path, "rule": "existing file in a lane-additive tree"})
        if _match(path, PRIMITIVE_PATHS) and st != "A":
            prims.append(path)
        if path == "verity_vllm/check/kernel_allowlist.py":
            allow.append(path)
        if path.startswith("verity_vllm/check/golden/"):
            golden.append(path)
        if path == "verity_vllm/input_provenance/root_policy.py":
            roots.append(path)
    return {"protected_files_modified": prot, "existing_primitives_modified": prims, "new_allowlist_entries": allow,
            "golden_digest_changes": golden, "new_root_classes": roots}


def diff(base: str, head: str, cwd: Path = REPO) -> list[tuple[str, str]]:
    out = _git("diff", "--name-status", f"{base}...{head}", cwd=cwd)
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0][0], parts[-1]))
    return rows


def new_patterns(head_checkout: Path, base_profiles: tuple[str, ...] = ("vllm_d9105ea80_sm89_eager", "vllm_d9105ea80_sm89_eager_qwen15")) -> dict[str, Any]:
    """Pattern names registered by head's profile modules that the base profiles do not have (runs `python` in the head checkout)."""
    code = r'''
import importlib, json, pkgutil, sys
import verity_vllm.observe.profiles as P
base = set(sys.argv[1].split(","))
names = {}
for m in pkgutil.iter_modules(P.__path__):
    if m.name.startswith("vllm_") or m.name.startswith("gen_"):
        try:
            mod = importlib.import_module(f"verity_vllm.observe.profiles.{m.name}")
            names[m.name] = sorted({p.name for p in mod.PROFILE.patterns})
        except Exception as e:
            names[m.name] = f"import failed: {type(e).__name__}: {e}"
basenames = set()
for b in base:
    v = names.get(b)
    if isinstance(v, list):
        basenames |= set(v)
new = sorted({n for k, v in names.items() if isinstance(v, list) and k not in base for n in v} - basenames)
print(json.dumps({"profiles": names, "new_patterns": new}))
'''
    py = head_checkout / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    r = subprocess.run([str(py), "-c", code, ",".join(base_profiles)], cwd=head_checkout, capture_output=True, text=True)
    if r.returncode != 0:
        return {"profiles": {}, "new_patterns": [], "error": r.stderr[-800:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def new_definitions(changed: list[tuple[str, str]], head: str, cwd: Path = REPO) -> list[str]:
    """Definition names added under the quarantine registry (best effort: `name="..."` / `def <name>` in added lines)."""
    import re
    out = []
    for st, path in changed:
        if path.startswith("verity_vllm/program/registry/quarantine/") and path.endswith(".py"):
            try:
                src = _git("show", f"{head}:{path}", cwd=cwd)
            except subprocess.CalledProcessError:
                continue
            for m in re.finditer(r"^\s*(?:@(?:composite|primitive)\b.*\n\s*def\s+(\w+)|(\w+)\s*=\s*(?:Primitive|Composite)Definition\()", src, re.M):
                out.append(f"{path}:{m.group(1) or m.group(2)}")
    return sorted(set(out))


def manifest_frozen_entries_changed(base: str, head: str, cwd: Path = REPO) -> list[str]:
    """Entries of the checkpoint manifest present at `base` whose content differs at `head` (appending is fine)."""
    try:
        b = json.loads(_git("show", f"{base}:{MANIFEST}", cwd=cwd))
        h = json.loads(_git("show", f"{head}:{MANIFEST}", cwd=cwd))
    except subprocess.CalledProcessError:
        return []
    # Append-only: the base entries must be a position-wise prefix of the head entries.  (Keyed comparison on
    # (repo, revision) mis-fired when the manifest legitimately lists the same checkpoint twice — D83/P7.)
    bl, hl = b.get("checkpoints", []), h.get("checkpoints", [])
    out = []
    for i, c in enumerate(bl):
        tag = f"{c.get('repo')}@{str(c.get('revision'))[:8]} (#{i})"
        if i >= len(hl):
            out.append(f"{tag}: removed")
        elif hl[i] != c:
            out.append(f"{tag}: changed")
    for k in ("schema", "frozen_utc", "note"):
        if k in b and b.get(k) != h.get(k):
            out.append(f"manifest field {k} changed")
    return out


def provenance(base: str, head: str, *, cwd: Path = REPO, head_checkout: Path | None = None) -> dict[str, Any]:
    changed = diff(base, head, cwd)
    rows = classify(changed)
    for m in manifest_frozen_entries_changed(base, head, cwd):
        rows["protected_files_modified"].append({"status": "M", "path": MANIFEST, "rule": f"frozen manifest entry: {m}"})
    rows.update(core_commit=_git("rev-parse", base, cwd=cwd).strip(), lane_commit=_git("rev-parse", head, cwd=cwd).strip(),
                n_changed=len(changed), changed=[{"status": s, "path": p} for s, p in changed],
                new_definitions=new_definitions(changed, head, cwd))
    if head_checkout is not None:
        np_ = new_patterns(head_checkout)
        rows["new_patterns"] = np_.get("new_patterns", [])
        rows["profiles"] = np_.get("profiles", {})
        if "error" in np_:
            rows["new_patterns_error"] = np_["error"]
    else:
        rows["new_patterns"] = None
    rows["sealed"] = not rows["protected_files_modified"]
    rows["schema"] = "verity-gen/provenance/v1"
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="core commit / branch (lane/capture)")
    ap.add_argument("--head", required=True, help="lane candidate commit / branch")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--head-checkout", default=None, help="a checkout of --head (to import its profiles for the new-pattern count)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = provenance(a.base, a.head, cwd=Path(a.repo), head_checkout=Path(a.head_checkout) if a.head_checkout else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[protected] {'SEALED' if rep['sealed'] else 'PROTECTED PATHS TOUCHED'}: {len(rep['protected_files_modified'])} protected, "
          f"{len(rep['existing_primitives_modified'])} primitives, {len(rep['new_allowlist_entries'])} allowlist, "
          f"{len(rep['golden_digest_changes'])} golden, {len(rep['new_root_classes'])} root-policy; {rep['n_changed']} files changed; "
          f"new patterns {rep['new_patterns']}; new definitions {rep['new_definitions']}", file=sys.stderr)
    for p in rep["protected_files_modified"]:
        print(f"   protected: {p['status']} {p['path']}", file=sys.stderr)
    return 0 if rep["sealed"] else 1


if __name__ == "__main__":
    sys.exit(main())
