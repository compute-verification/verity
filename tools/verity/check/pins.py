"""`pins`: every tracked file that records the sha256 of Lean sources, with the command that regenerates it.

    python tools/verity/check/pins.py            each pin, whether it is stale, and its command
    python tools/verity/check/pins.py --stale    exit 1 listing each stale pin with its command
    python tools/verity/check/pins.py --regen    run each stale pin's command at the root, in order; exit 1 while one is stale
    python tools/verity/check/pins.py --merged OURS THEIRS [--conflicted PATH...]   regenerate what a merge of THEIRS into
                                                 OURS, under way at the root, wrote by merging text (`merged`)

A pin is written only by its command, never by hand: #1499's revert came from a hand-written one. A pin is stale when a file it
records has another digest in the tree (or is gone), and its command, run at the root of a checkout, rewrites it from the tree.
`tests/test_pins.py` fails while a pin is stale, naming its command, and while a tracked file records the digest of a Lean source
(`*.lean`, a Lake manifest, lakefile or toolchain) without being listed here. `tools/move/restack.py` runs `--regen` on every
restacked branch. Not listed, since another check owns them: a `lean-audit.json`'s records (the Lean check, `check.py --update`)
and `tools/verity/lean/lean-deps.json`, a bundle per manifest that a cold run exports (preflight names its command).

`--merged` is the merge's half (`research merge --train --prepare` runs it on every merge of a train): Lake manifests and pins
are generated, so their merged text is never kept.  Drive's merge onto `next` has no checkout to run commands in, so when its
only conflicts are in pins it rebuilds each from the merged tree itself, reading `PINS` from this file's text
(`research.queue.repinned`). Each Lake package (a directory with a tracked `lakefile.toml` and
`lake-manifest.json`) whose lakefile, manifest or toolchain, or one of a package it requires by path, both sides changed gets
`lake update`, packages a package requires first (0fcc: two branches each added a dependency, and the merged manifest held
neither); a package whose manifest moves to a new dependency key gets the pin of the bundle one side had for it, when the new
manifest fetches the same packages at the same revisions (otherwise it is named: a cold run exports its bundle). Then every
stale pin is regenerated. A conflict in a manifest, a pin or `lean-deps.json` is resolved by taking OURS and regenerating;
one in any other file exits 3. It prints {"regenerated": [paths], "notes": [...]} and exits 1 when a command failed or a pin
is still stale.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEAN_DEPS = "tools/verity/lean/lean-deps.json"
LAKE_FILES = ("lakefile.toml", "lake-manifest.json", "lean-toolchain")
#: what `lake update` runs: Mathlib's post-update hook would fetch its whole build cache, which a manifest doesn't need
LAKE_UPDATE_ENV = {"MATHLIB_NO_CACHE_ON_UPDATE": "1"}
LEAN_SOURCE = re.compile(r"(\.lean|(^|/)lake-manifest\.json|(^|/)lakefile\.toml|(^|/)lean-toolchain)$")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Pin:
    path: str
    #: the JSON key whose {path: digest} it records, or None for a `sha256sum` listing
    key: str | None
    #: what the recorded paths are relative to, from the root
    under: str
    #: run at the root; None: the `sha256sum ... > <path's name>` command in `under`'s README.md, run in `under`
    command: str | None
    #: the Lake package the command builds in, which `tools/verity/lean/setup.sh` readies first (a fresh worktree has neither
    #: its dependencies nor Mathlib's built cache, and `lake build` would compile Mathlib)
    builds: str | None = None

    def recorded(self, root: Path) -> dict[str, str]:
        """{root-relative path: recorded digest}."""
        text = (root / self.path).read_text()
        if self.key is None:
            pairs = [ln.split("  ", 1)[::-1] for ln in text.splitlines() if ln.strip()]
        else:
            pairs = json.loads(text)[self.key].items()
        return {str(Path(self.under, p)) if self.under else p: h for p, h in pairs}

    def stale(self, root: Path) -> list[str]:
        """The recorded files whose digest in the tree differs, or that are gone. One git ignores (a fetched fixture) is judged
        only where it has been fetched."""
        rec = self.recorded(root)
        missing = [p for p in rec if not (root / p).is_file()]
        return sorted({p for p, h in rec.items() if p not in missing and sha256(root / p) != h} |
                      set(missing) - ignored(root, missing))

    def shell(self, root: Path) -> str:
        if self.command is not None:
            return self.command
        name = Path(self.path).name
        readme = re.sub(r"\\\n\s*", "", (root / self.under / "README.md").read_text())
        m = re.search(rf"^(sha256sum [^\n]*) > {re.escape(name)}$", readme, re.M)
        if not m:
            raise SystemExit(f"{self.under}/README.md: no `sha256sum ... > {name}` command")
        return f"cd {self.under} && {m.group(1)} > {name}"


NCP = "verity/core/protocols/pouw/tests/ncp_lean.py"
PINS = (
    Pin("verity/core/protocols/pous/lean/TRUSTED.sha256", None, "verity/core/protocols/pous/lean", None),
    Pin("verity/ml/flock/tests/fixtures/randomness_lean.json", "inputs", "", "bash verity/ml/flock/tests/lean/generate.sh",
        "Security/Proofs"),
    Pin("verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json", "inputs", "",
        "bash verity/core/protocols/sampled_proofs/tests/lean/generate.sh", "Security/Proofs"),
    Pin("verity/ml/flock/tests/fixtures/inner_claims.json", "inputs", "",
        "bash verity/ml/flock/tests/lean/generate_inner_claims.sh", "verity/core/service/flock"),
    Pin("verity/core/protocols/pouw/tests/vectors/ncp_lean.json", "sources_sha256", "Security",
        f"uv run python {NCP} write Security && (cd Security && lake build Definitions.Pouw.NCP.RouteU && "
        f"lake env lean NcpCheck.lean > ncp_lean.out) && uv run python {NCP} pin Security/ncp_lean.out Security; "
        f"s=$?; rm -f Security/NcpCheck.lean Security/ncp_lean.out; exit $s", "Security"),
)
SETUP = "tools/verity/lean/setup.sh"
#: tracked files that record Lean sources' digests and are someone else's to keep: the Lean check's locks, and the moves' history
EXEMPT = re.compile(r"(^|/)lean-audit\.json$|^tools/move/|^archive/")


def ignored(root: Path, paths: list[str]) -> set[str]:
    if not paths:
        return set()
    p = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=root, input="\0".join(paths).encode(),
                       capture_output=True)
    return {s for s in p.stdout.decode().split("\0") if s}


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def unlisted(root: Path) -> dict[str, list[str]]:
    """{tracked file: the Lean sources whose digest it records} for each such file that isn't a pin here or exempt."""
    files = tracked(root)
    sources = {sha256(root / p): p for p in files if LEAN_SOURCE.search(p) and (root / p).is_file()}
    sources.pop(hashlib.sha256(b"").hexdigest(), None)
    listed = {p.path for p in PINS}
    found = {}
    for p in files:
        if p in listed or EXEMPT.search(p) or not (root / p).is_file():
            continue
        hits = {sources[h.decode()] for h in re.findall(rb"\b[0-9a-f]{64}\b", (root / p).read_bytes()) if h.decode() in sources}
        if hits - {p}:
            found[p] = sorted(hits - {p})
    return found


def stale(root: Path) -> list[tuple[Pin, list[str]]]:
    return [(pin, s) for pin in PINS if (s := pin.stale(root))]


def run_pin(pin: Pin, root: Path, ready: set[str], **kw) -> subprocess.CompletedProcess:
    """``pin``'s command at ``root``, after `setup.sh` on the package it builds in (once per package in ``ready``)."""
    if pin.builds and pin.builds not in ready and (root / SETUP).is_file():
        ready.add(pin.builds)
        r = subprocess.run(["bash", SETUP, pin.builds], cwd=root, **kw)
        if r.returncode:
            return r
    return subprocess.run(["bash", "-c", pin.shell(root)], cwd=root, **kw)


def regen(root: Path) -> list[tuple[Pin, list[str]]]:
    """Run each stale pin's command; what is still stale after."""
    ready: set[str] = set()
    for pin, _ in stale(root):
        print(f"pins: regenerating {pin.path}: {pin.shell(root)}", flush=True)
        r = run_pin(pin, root, ready)
        if r.returncode:
            print(f"pins: {pin.path}: its command exited {r.returncode}", file=sys.stderr)
    return stale(root)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def lake_packages(root: Path) -> list[str]:
    """Every Lake package with a tracked manifest, by its root-relative directory."""
    files = set(tracked(root))
    return sorted(str(Path(p).parent) for p in files if Path(p).name == "lakefile.toml" and
                  str(Path(p).parent / "lake-manifest.json") in files)


def path_requires(root: Path, pkg: str) -> list[str]:
    """The root-relative directories ``pkg``'s lakefile requires by path."""
    doc = tomllib.loads((root / pkg / "lakefile.toml").read_text())
    return [os.path.relpath(os.path.normpath(root / pkg / r["path"]), root) for r in doc.get("require", []) if "path" in r]


def lake_order(root: Path) -> dict[str, set[str]]:
    """{package: the packages it requires by path, transitively, itself included}."""
    seen: dict[str, set[str]] = {}

    def close(p: str) -> set[str]:
        if p not in seen:
            seen[p] = {p}
            if (root / p / "lakefile.toml").is_file():
                for q in path_requires(root, p):
                    seen[p] |= close(q)
        return seen[p]

    return {p: close(p) for p in lake_packages(root)}


def touched(root: Path, base: str, side: str) -> set[str]:
    """The directories whose lakefile, manifest or toolchain ``side`` changed since ``base``."""
    return {str(Path(p).parent) for p in git(root, "diff", "--name-only", base, side).split() if Path(p).name in LAKE_FILES}


def lake_affected(root: Path, ours: str, theirs: str) -> list[str]:
    """The packages whose closure both sides' Lake changes reach, each after the packages it requires."""
    base = git(root, "merge-base", ours, theirs).strip()
    a, b = touched(root, base, ours), touched(root, base, theirs)
    order = lake_order(root)
    hit = {p for p, cl in order.items() if cl & a and cl & b}
    return sorted(hit, key=lambda p: (len(order[p]), p))


def _lean_audit():
    spec = importlib.util.spec_from_file_location("lean_audit_for_pins", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deps_key(manifest: str, toolchain: bytes) -> str | None:
    """`lean_audit.deps_key` of a package with this manifest text and toolchain."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "lake-manifest.json").write_text(manifest)
        Path(d, "lean-toolchain").write_bytes(toolchain)
        return _lean_audit().deps_key(Path(d))


def fetched(manifest: str) -> set[tuple]:
    return {(p.get("name"), p.get("url"), p.get("rev")) for p in json.loads(manifest).get("packages", []) if p.get("type") != "path"}


def repoint(root: Path, pkg: str, sides: list[str], notes: list[str]) -> bool:
    """Pin ``pkg``'s new manifest to the bundle a side had pinned for it, when the two fetch the same packages; whether
    `lean-deps.json` changed."""
    pin_file = root / LEAN_DEPS
    if not pin_file.is_file():
        return False
    doc = json.loads(pin_file.read_text())
    manifest = (root / pkg / "lake-manifest.json").read_text()
    toolchain = (root / pkg / "lean-toolchain").read_bytes() if (root / pkg / "lean-toolchain").is_file() else b""
    new = deps_key(manifest, toolchain)
    if new is None or new in doc["deps"]:
        return False
    for side in sides:
        try:
            old_text = git(root, "show", f"{side}:{pkg}/lake-manifest.json")
        except subprocess.CalledProcessError:
            continue
        old = deps_key(old_text, toolchain)
        if old in doc["deps"] and doc["deps"][old].get("package") == pkg and fetched(old_text) == fetched(manifest):
            doc["deps"][new] = doc["deps"].pop(old)
            doc["deps"] = dict(sorted(doc["deps"].items()))
            pin_file.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            notes.append(f"{LEAN_DEPS}: {pkg}'s bundle {old[:12]} pinned as {new[:12]} (the same packages at the same revisions)")
            return True
    notes.append(f"{pkg}: no bundle is pinned for its new manifest ({new[:12]}): the audit fetches its dependencies cold; pin "
                 "the bundle that run exports (`lean_audit.py --pin RUN`)")
    return False


def write_deps(root: Path, deps: dict) -> None:
    doc = json.loads((root / LEAN_DEPS).read_text())
    (root / LEAN_DEPS).write_text(json.dumps({**doc, "deps": dict(sorted(deps.items()))}, indent=1, sort_keys=True) + "\n")


def current_keys(root: Path) -> set[str | None]:
    out = set()
    for pkg in lake_packages(root):
        tc = root / pkg / "lean-toolchain"
        out.add(deps_key((root / pkg / "lake-manifest.json").read_text(), tc.read_bytes() if tc.is_file() else b""))
    return out


def generated(path: str) -> bool:
    """A file `merged` regenerates, so a conflict in it takes OURS."""
    return Path(path).name == "lake-manifest.json" or path == LEAN_DEPS or path in {p.path for p in PINS}


def merged(root: Path, ours: str, theirs: str, conflicted: list[str]) -> tuple[int, dict]:
    """`--merged`: (exit status, {"regenerated", "notes"}); see the module docstring."""
    notes: list[str] = []
    if other := [p for p in conflicted if not generated(p)]:
        return 3, {"regenerated": [], "notes": [f"not generated, so not resolved here: {', '.join(other)}"]}
    for p in conflicted:
        git(root, "checkout", "--ours", "--", p)
    union = None
    if LEAN_DEPS in conflicted:
        union = {k: v for side in (ours, theirs) for k, v in json.loads(git(root, "show", f"{side}:{LEAN_DEPS}"))["deps"].items()}
        write_deps(root, union)
    watched = [*(f"{q}/lake-manifest.json" for q in lake_packages(root)), LEAN_DEPS, *(p.path for p in PINS)]
    before = {p: (root / p).read_bytes() for p in watched if (root / p).is_file()}
    rc = 0
    for pkg in lake_affected(root, ours, theirs):
        r = subprocess.run(["lake", "update"], cwd=root / pkg, env={**os.environ, **LAKE_UPDATE_ENV}, capture_output=True, text=True)
        if r.returncode:
            notes.append(f"{pkg}: `lake update` exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
            rc = 1
            continue
        notes.append(f"{pkg}: lake-manifest.json from `lake update`")
        repoint(root, pkg, [ours, theirs], notes)
    if union is not None:
        keys = current_keys(root)
        write_deps(root, {k: v for k, v in json.loads((root / LEAN_DEPS).read_text())["deps"].items() if k in keys})
    ready: set[str] = set()
    for pin, _ in stale(root):
        r = run_pin(pin, root, ready, capture_output=True, text=True)
        notes.append(f"{pin.path}: `{pin.shell(root)}`" + (f" exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}"
                                                          if r.returncode else ""))
    if still := stale(root):
        notes.append(report(still, root))
        rc = 1
    changed = {p for p, b in before.items() if not (root / p).is_file() or (root / p).read_bytes() != b}
    return rc, {"regenerated": sorted(changed | set(conflicted)), "notes": notes}


def report(rows: list[tuple[Pin, list[str]]], root: Path) -> str:
    return "\n".join(f"{pin.path} is stale ({', '.join(s[:3])}{', ...' if len(s) > 3 else ''}): run `{pin.shell(root)}`"
                     for pin, s in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stale", action="store_true")
    g.add_argument("--regen", action="store_true")
    g.add_argument("--merged", nargs=2, metavar=("OURS", "THEIRS"))
    ap.add_argument("--conflicted", nargs="*", default=[], metavar="PATH", help="--merged: the merge's conflicted paths")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    if a.merged:
        rc, doc = merged(a.root, *a.merged, a.conflicted)
        print(json.dumps(doc))
        return rc
    if a.regen or a.stale:
        rows = regen(a.root) if a.regen else stale(a.root)
        if rows:
            print(report(rows, a.root), file=sys.stderr)
        return 1 if rows else 0
    for pin in PINS:
        s = pin.stale(a.root)
        print(f"{'stale' if s else 'ok   '}  {pin.path}  `{pin.shell(a.root)}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
