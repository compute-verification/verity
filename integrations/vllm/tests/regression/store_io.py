"""The ONE module that talks to the `research` data API (`research data ...`, `research.store`; main's `tools/research/src/research/store/README.md`).

Every other file of this harness reaches the evidence store through these three functions, so a change of the store's CLI or Python
API touches this file only:

    fetch(artifact_id) -> Path                 materialise an artifact's payload (tree dir or file) and return its local path
    resolve_snapshot(name) -> list[artifact_id]  the member artifact ids of the dataset-snapshot/v1 artifact named `name`
    put_fixture(path, meta) -> artifact_id      store a dir (tree payload) or a file as a `fixture/v1` artifact, return its `art:` id

Transport: the CLI (`python -m research.cli data ...`) in a subprocess -- the store package is stdlib-only, so `sys.executable` with
`<repo>/tools/research/src` on PYTHONPATH is enough; the CLI's printed contract (`put` / `snapshot` print the art id, `fetch` prints the
path, `select --ids` prints ids, `show --json` dumps the manifest document) is what is parsed here.  Credentials for a remote put come
from the environment (`set -a; source ~/.config/verity/r2.env; set +a` on the control machine; a pod uses a minted temporary credential,
README §7.1) -- nothing here reads or copies a secret.

`reachable()` answers "can this machine use the store at all" (package importable + a local store root): the tests skip on False.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ART_RE = re.compile(r"\bart:[0-9a-f]{64}\b")
FIXTURE_KIND = "fixture/v1"
SNAPSHOT_KIND = "dataset-snapshot/v1"

_HERE = Path(__file__).resolve()
#: integrations/vllm/tests/regression/store_io.py -> the repository root
REPO_ROOT = _HERE.parents[4]


class StoreUnavailable(RuntimeError):
    """The store cannot be reached from here (package missing, no store root, CLI failure)."""


def _research_src() -> Path | None:
    """`tools/research/src` of this checkout (the branch may predate the merge that brings `research.store`), else `$RESEARCH_SRC`."""
    for cand in (os.environ.get("RESEARCH_SRC"), REPO_ROOT / "tools" / "research" / "src"):
        if cand and (Path(cand) / "research" / "store").is_dir():
            return Path(cand)
    return None


def _env() -> dict[str, str]:
    env = dict(os.environ)
    src = _research_src()
    if src is not None:
        env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """`python -m research.cli data <args>`; a `research` executable on PATH is used when the package is not in this tree."""
    if _research_src() is not None:
        argv = [sys.executable, "-m", "research.cli", "data", *args]
    elif shutil.which("research"):
        argv = ["research", "data", *args]
    else:
        raise StoreUnavailable("neither tools/research/src/research/store (this checkout) nor a `research` executable is available")
    p = subprocess.run(argv, env=_env(), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise StoreUnavailable(f"{' '.join(argv[-len(args) - 1:])} failed rc={p.returncode}: {p.stderr.strip()[-2000:]}")
    return p


def reachable() -> bool:
    """True iff the store CLI answers here (package present and a local store root usable).  Never raises."""
    try:
        p = _cli("select", "--kind", SNAPSHOT_KIND, "--ids", check=False)
    except StoreUnavailable:
        return False
    return p.returncode == 0


def fetch(artifact_id: str, to: Path | None = None) -> Path:
    """Materialise `artifact_id`'s payload (tree -> a directory, file -> the file) and return its path (the store's cache unless `to`)."""
    if not ART_RE.fullmatch(artifact_id):
        raise ValueError(f"not an artifact id: {artifact_id!r}")
    args = ["fetch", artifact_id] + (["--to", str(to)] if to else [])
    p = _cli(*args)
    path = p.stdout.strip().splitlines()[-1].strip() if p.stdout.strip() else ""
    if not path or not Path(path).exists():
        raise StoreUnavailable(f"fetch {artifact_id}: no path printed (stdout={p.stdout!r}, stderr={p.stderr.strip()[-500:]!r})")
    return Path(path)


def show(artifact_id: str) -> dict:
    """The artifact manifest document (`research data show ART --json`)."""
    p = _cli("show", artifact_id, "--json")
    doc = json.loads(p.stdout)
    return doc.get("manifest", doc) if isinstance(doc, dict) else doc


def resolve_snapshot(name: str) -> list[str]:
    """The member artifact ids of the `dataset-snapshot/v1` named `name` (`refs.members`).  Several snapshots with one name (frozen from
    different member sets) are a mistake this harness names rather than merges: the LAST id by the CLI's order is used and the others
    reported on stderr.  Empty list when no snapshot has that name."""
    p = _cli("select", "--kind", SNAPSHOT_KIND, "--where", f"meta.name={name}", "--ids")
    ids = ART_RE.findall(p.stdout)
    if not ids:
        return []
    if len(ids) > 1:
        print(f"[store_io] {len(ids)} snapshots named {name!r}: using {ids[-1]} (others: {ids[:-1]})", file=sys.stderr)
    doc = show(ids[-1])
    members = (doc.get("refs") or {}).get("members") or []
    return list(members) if isinstance(members, list) else [members]


def put_fixture(path: Path, meta: dict, *, preserve: bool = False, refs: dict[str, str] | None = None) -> str:
    """Store `path` (a directory -> tree payload; a file -> file payload) as a `fixture/v1` artifact with inline `meta`; returns `art:...`.
    `preserve=True` asks the CLI to push to the remote and verify (`--preserve`; the id is still returned when the push fails -- the
    caller decides whether local-only is acceptable).  Idempotent: identical bytes + meta give the same id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    args = ["put", "--kind", FIXTURE_KIND, "--meta", json.dumps(meta, sort_keys=True), "--tree" if path.is_dir() else "--file", str(path)]
    for k, v in (refs or {}).items():
        args += ["--ref", f"{k}={v}"]
    if preserve:
        args.append("--preserve")
    p = _cli(*args, check=False)
    ids = ART_RE.findall(p.stdout)
    if not ids:
        raise StoreUnavailable(f"put {path}: no art id printed (rc={p.returncode}; stderr={p.stderr.strip()[-1500:]!r})")
    if p.returncode != 0:
        print(f"[store_io] put {path.name}: {ids[-1]} stored locally; NOT preserved: {p.stderr.strip()[-400:]}", file=sys.stderr)
    return ids[-1]


def snapshot(name: str, members: list[str], note: str | None = None) -> str:
    """Freeze `members` as a `dataset-snapshot/v1` named `name`; returns its `art:` id (content-addressed: the same member set = same id)."""
    args = ["snapshot", "--name", name, "--members", ",".join(sorted(set(members)))]
    if note:
        args += ["--note", note]
    p = _cli(*args)
    ids = ART_RE.findall(p.stdout)
    if not ids:
        raise StoreUnavailable(f"snapshot {name}: no art id printed ({p.stdout!r})")
    return ids[-1]


def push(artifact_ids: list[str]) -> subprocess.CompletedProcess:
    """`research data push ART...` (preserve on the remote; needs credentials in the environment).  Returns the completed process."""
    return _cli("push", *artifact_ids, check=False)
