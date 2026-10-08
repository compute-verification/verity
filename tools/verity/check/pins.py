"""`pins`: every tracked file that records the sha256 of Lean sources, with the command that regenerates it.

    python tools/verity/check/pins.py            each pin, whether it is stale, and its command
    python tools/verity/check/pins.py --stale    exit 1 listing each stale pin with its command
    python tools/verity/check/pins.py --regen    run each stale pin's command at the root, in order; exit 1 while one is stale

A pin is written only by its command, never by hand: #1499's revert came from a hand-written one. A pin is stale when a file it
records has another digest in the tree (or is gone), and its command, run at the root of a checkout, rewrites it from the tree.
`tests/test_pins.py` fails while a pin is stale, naming its command, and while a tracked file records the digest of a Lean source
(`*.lean`, a Lake manifest, lakefile or toolchain) without being listed here. `tools/move/restack.py` runs `--regen` on every
restacked branch. Not listed, since another check owns them: a `lean-audit.json`'s records (the Lean check, `audit.py --update`)
and `tools/verity/lean/lean-deps.json`, a bundle per manifest that a cold run exports (preflight names its command).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
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
    Pin("verity/ml/flock/tests/fixtures/randomness_lean.json", "inputs", "", "bash verity/ml/flock/tests/lean/generate.sh"),
    Pin("verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json", "inputs", "",
        "bash verity/core/protocols/sampled_proofs/tests/lean/generate.sh"),
    Pin("verity/ml/flock/tests/fixtures/inner_claims.json", "inputs", "",
        "bash verity/ml/flock/tests/lean/generate_inner_claims.sh"),
    Pin("verity/core/protocols/pouw/tests/vectors/ncp_lean.json", "sources_sha256", "Security",
        f"uv run python {NCP} write Security && (cd Security && lake build Definitions.Pouw.NCP.RouteU && "
        f"lake env lean NcpCheck.lean > ncp_lean.out) && uv run python {NCP} pin Security/ncp_lean.out Security; "
        f"s=$?; rm -f Security/NcpCheck.lean Security/ncp_lean.out; exit $s"),
)
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


def regen(root: Path) -> list[tuple[Pin, list[str]]]:
    """Run each stale pin's command; what is still stale after."""
    for pin, _ in stale(root):
        print(f"pins: regenerating {pin.path}: {pin.shell(root)}", flush=True)
        r = subprocess.run(["bash", "-c", pin.shell(root)], cwd=root)
        if r.returncode:
            print(f"pins: {pin.path}: its command exited {r.returncode}", file=sys.stderr)
    return stale(root)


def report(rows: list[tuple[Pin, list[str]]], root: Path) -> str:
    return "\n".join(f"{pin.path} is stale ({', '.join(s[:3])}{', ...' if len(s) > 3 else ''}): run `{pin.shell(root)}`"
                     for pin, s in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stale", action="store_true")
    g.add_argument("--regen", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
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
