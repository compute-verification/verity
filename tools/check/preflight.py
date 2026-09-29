"""`check`'s preflight (`Tool.preflight` in `tool.py`): whether this machine is prepared for a check of this tree. `research run
--on` runs it on the machine, in the shipped tree, before the run is claimed, and refuses the run when it exits non-zero: an
unprepared pod is an error, never a verdict on the commit.

    python3 tools/check/preflight.py        by the interpreter `research` runs on, since nothing else may be installed yet

One line per fact on stdout, one per failure on stderr. It checks:
- that `pod_setup.sh` ran: uv at the version it pins, elan with the toolchain the Lake packages pin, and cargo, on the PATH a
  run sees;
- that each Lean package's dependencies are warm on this machine, or restorable from the store: `lean-deps.json` pins a bundle
  for its manifest and toolchain and the run carries its presigned URL (`check.py --lean-deps-files`), so the audit never
  clones them from GitHub;
- that the store answers: every bundle still to restore, or else one pinned object, is read with the run's custody key
  (RESEARCH_PREFLIGHT_CUSTODY), else through its sent URL;
- that the CPU has `avx512f` when the run carries the pinned upstream build (so lean-agreement runs) and its RUSTFLAGS need it;
- that the tree's and the cache's filesystems have the space a check takes, plus each dependency tree still to restore.
The run's `--send` files are in RESEARCH_PREFLIGHT_INPUTS.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GB = 1 << 30
#: what a check with warm Lean dependencies takes on the tree's filesystem (the workspace's .venv, the verifier's build) and on
#: the cache's (the audit's scratch builds, uv's cache, the Flock checkout's target/): together the 25 GB the merge trains'
#: launcher required free on check pods
TREE_GB, CACHE_GB = 10, 15
#: one Lean dependency tree unpacked from its bundle (8.0 to 10 GB for the three packages), besides the bundle itself
DEPS_TREE_GB = 10
PROBE_TIMEOUT_S = 30
#: a pinned build needs AVX-512 when it was built for the build host's CPU or for x86-64-v4, or names an AVX-512 feature
AVX512 = re.compile(r"target-cpu=(native|x86-64-v4)\b|avx512")
POD_SETUP = "research run --on POD --project verity --source . --cwd source -- bash tools/check/pod_setup.sh"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LA = _module(ROOT / "tools" / "check" / "lean_audit.py", "check_preflight_lean_audit")
CHECK = LA.CHECK


def _out(argv: list[str]) -> str | None:
    """The stdout of ``argv``, or None when it does not run or fails."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def pinned_uv(root: Path) -> str | None:
    m = re.search(r"^UV_VERSION=\$\{UV_VERSION:-([^}]+)\}", (root / "tools" / "check" / "pod_setup.sh").read_text(), re.M)
    return m.group(1) if m else None


def setup(root: Path, lean_packages: list[str]) -> tuple[list[str], list[str]]:
    """What `pod_setup.sh` installs, as a run finds it on its PATH: (facts, failures)."""
    facts, fails = [], []
    want = pinned_uv(root)
    if not shutil.which("uv"):
        fails.append("uv is not on the PATH")
    elif (got := (_out(["uv", "--version"]) or "").split()[1:2]) != [want]:
        fails.append(f"uv is {got[0] if got else 'broken'}, not the {want} pod_setup.sh pins")
    else:
        facts.append(f"uv {want}")
    cargo = _out(["cargo", "--version"]) if shutil.which("cargo") else None
    if cargo:
        facts.append(cargo.strip())
    else:
        fails.append("cargo is not on the PATH, or does not run")
    if lean_packages:
        pinned = sorted({(root / p / "lean-toolchain").read_text().strip() for p in lean_packages if (root / p / "lean-toolchain").is_file()})
        listed = _out(["elan", "toolchain", "list"]) if shutil.which("elan") else None
        if listed is None:
            fails.append("elan is not on the PATH, or does not run")
        for tc in pinned if listed is not None else []:
            ver = re.escape(tc.split(":", 1)[-1])
            if re.search(rf"(^|[:\s]){ver}(\s|$)", listed, re.M):
                facts.append(f"elan has {tc}")
            else:
                fails.append(f"elan has no {tc}, which the Lake packages pin")
        if not shutil.which("lake"):
            fails.append("lake is not on the PATH")
    return facts, fails


def pins(root: Path) -> dict[str, dict]:
    p = root / LA.PIN
    return json.loads(p.read_text())["deps"] if p.is_file() else {}


def lean_deps(root: Path, packages: list[str], cache: Path, urls: dict[str, str]) -> tuple[list[str], list[str], list[dict]]:
    """(facts, failures, the bundles the audit will restore from the store): each package's dependencies, warm here under
    ``cache``/lean-deps or restorable through a sent URL, the pinned bundle's."""
    facts, fails, restore, seen = [], [], [], set()
    pinned = pins(root)
    for p in packages:
        k = LA.deps_key(root / p)
        if k is None or k in seen:
            continue
        seen.add(k)
        pin = pinned.get(k)
        if (cache / "lean-deps" / k).is_dir():
            facts.append(f"{p}: Lean dependencies warm here ({k[:12]})")
        elif pin is None:
            fails.append(f"{p}: {LA.PIN} pins no bundle for its manifest and toolchain ({k[:12]}) and none is warm here, so its audit "
                         f"would clone them from GitHub: pin one (`lean_audit.py --pin RUN` after a cold `--export` run)")
        elif k not in urls:
            fails.append(f"{p}: its Lean dependencies ({k[:12]}) are not warm here and the run carries no URL for their bundle: send "
                         f"`$(uv run python tools/check/check.py --lean-deps-files)`")
        else:
            restore.append({"name": f"{p}'s Lean bundle {k[:12]}", "key": k, "sha256": pin["sha256"], "bytes": pin["bytes"]})
            facts.append(f"{p}: Lean dependencies restorable from the store ({k[:12]}, {pin['bytes'] >> 20} MiB)")
    return facts, fails, restore


def probe(root: Path, packages: list[str]) -> list[dict]:
    """One pinned object to read when nothing is to be restored: a bundle this tree's packages pin, else the upstream build."""
    keys = {LA.deps_key(root / p) for p in packages}
    for k, v in sorted(pins(root).items()):
        if k in keys:
            return [{"name": f"{v['package']}'s Lean bundle {k[:12]}", "key": k, "sha256": v["sha256"], "bytes": v["bytes"]}]
    up = root / "backends" / "flock" / "verifier" / "upstream.json"
    if up.is_file():
        return [{"name": "the pinned upstream build", "sha256": json.loads(up.read_text())["upstream"]["sha256"]}]
    return []


def store(targets: list[dict], urls: dict[str, str], custody: Path | None) -> tuple[list[str], list[str]]:
    """Read each of ``targets`` from the store's remote: with the run's custody key (a HEAD, its size checked against the pin),
    else through its sent URL (one byte). A URL is never printed: it grants a read until it expires."""
    if not targets:
        return [], ["store: this tree pins nothing to read, so nothing shows the store answers here"]
    facts, fails = [], []
    if custody is not None and (custody / "cred.json").is_file():
        from research.store.custody import custody_env
        from research.store.local import object_key
        from research.store.remote import load_remote

        try:
            with custody_env(custody):
                remote = load_remote()
                if remote is None:
                    return [], [f"store: the run's custody dir {custody} names no remote"]
                for t in targets:
                    h = remote.head(object_key(t["sha256"]))
                    if h is None or (t.get("bytes") and h["bytes"] != t["bytes"]):
                        fails.append(f"store: {t['name']} is not on the remote as pinned")
                    else:
                        facts.append(f"store: {t['name']} read with the run's custody key")
        except Exception as e:  # noqa: BLE001
            fails.append(f"store: a read with the run's custody key failed ({type(e).__name__}: {e})")
        return facts, fails
    sent = [t for t in targets if t.get("key") in urls]
    if not sent:
        return [], ["store: the run carries neither a custody key nor a URL for what it must read, so nothing shows the store answers "
                    "here: launch with custody (the default with --on) or send `$(uv run python tools/check/check.py --lean-deps-files)`"]
    for t in sent:
        try:
            with urllib.request.urlopen(urllib.request.Request(urls[t["key"]], headers={"Range": "bytes=0-0"}), timeout=PROBE_TIMEOUT_S) as r:
                r.read(1)
            facts.append(f"store: {t['name']} read through its URL")
        except (OSError, ValueError) as e:              # urllib's errors are OSErrors
            fails.append(f"store: {t['name']} could not be read through its URL ({getattr(e, 'code', None) or type(e).__name__})")
    return facts, fails


def cpu_flags(cpuinfo: Path) -> set[str]:
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("flags"):
            return set(line.partition(":")[2].split())
    return set()


def _sent(inputs: Path | None, sha256: str) -> bool:
    """Whether a file among the run's `--send` files is the one with this sha256 (the smallest hashed first)."""
    if inputs is None or not inputs.is_dir():
        return False
    return any(CHECK._sha256(p) == sha256 for p in sorted((q for q in inputs.iterdir() if q.is_file()), key=lambda q: q.stat().st_size))


def cpu(root: Path, inputs: Path | None, cpuinfo: Path) -> tuple[list[str], list[str]]:
    """avx512f, when the run carries the pinned upstream build (lean-agreement runs) and that build needs it."""
    pin = root / "backends" / "flock" / "verifier" / "upstream.json"
    if not pin.is_file() or not (root / "backends" / "flock" / "verifier" / "lean" / "lakefile.toml").exists():
        return ["cpu: no lean-agreement on this tree"], []
    up = json.loads(pin.read_text())["upstream"]
    flags = up.get("rustflags") or "the build host's own (target-cpu=native)"
    if not AVX512.search(flags):
        return [f"cpu: the pinned upstream build ({flags}) needs no AVX-512"], []
    if not _sent(inputs, up["sha256"]):
        return [f"cpu: the run carries no upstream build, so lean-agreement skips and its AVX-512 ({flags}) is not needed"], []
    if "avx512f" in cpu_flags(cpuinfo):
        return [f"cpu: avx512f, which the pinned upstream build ({flags}) needs"], []
    return [], [f"cpu: no avx512f, which the pinned upstream build the run carries needs ({flags}): its replays would die of SIGILL; "
                f"use a pod with AVX-512"]


def _existing(p: Path) -> Path:
    """``p`` or its nearest existing ancestor: where it will be created."""
    while not p.exists():
        p = p.parent
    return p


def disk(tree: Path, cache: Path, restore: list[dict], usage=shutil.disk_usage) -> tuple[list[str], list[str]]:
    """Free space on the tree's and the cache's filesystems (one sum when they are one) against what a check takes there."""
    need: dict[int, list] = {}
    for path, gb in ((tree, TREE_GB), (cache, CACHE_GB + sum(DEPS_TREE_GB + r["bytes"] / GB for r in restore))):
        at = _existing(path)
        need.setdefault(at.stat().st_dev, [at, 0.0])[1] += gb
    facts, fails = [], []
    restoring = f" with {len(restore)} Lean dependency tree(s) to restore" if restore else ""
    for at, gb in need.values():
        free = usage(at).free / GB
        line = f"disk: {free:.0f} GB free on {at}'s filesystem, where a check needs {gb:.0f} GB{restoring}"
        if free >= gb:
            facts.append(line)
        else:
            fails.append(f"{line}: free some (idle trees under <root>/src) or use a larger pod")
    return facts, fails


def files(root: Path) -> list[str]:
    """The tree's files as the audit lists them, or (no git here) its `lean-audit.json` files."""
    got = CHECK.tracked_files(root)
    if got is not None:
        return got
    out = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".lake", ".venv", ".git")]
        out += [os.path.relpath(os.path.join(base, n), root) for n in names if n == "lean-audit.json"]
    return sorted(out)


def run(root: Path, inputs: Path | None, custody: Path | None, cache: Path, cpuinfo: Path = Path("/proc/cpuinfo"),
        usage=shutil.disk_usage) -> tuple[list[str], list[str]]:
    lean = (root / "backends" / "flock" / "verifier" / "lean" / "lakefile.toml").exists()
    packages = LA.packages(files(root)) if lean else []
    sent = inputs / CHECK.LEAN_DEPS_URLS if inputs is not None else None
    urls = json.loads(sent.read_text()) if sent is not None and sent.is_file() else {}
    facts, fails = setup(root, packages)
    if fails:
        fails.append(f"prepare the pod: `{POD_SETUP}`")
    f, x, restore = lean_deps(root, packages, cache, urls)
    facts, fails = facts + f, fails + x
    for f, x in (store(restore or probe(root, packages), urls, custody), cpu(root, inputs, cpuinfo), disk(root, cache, restore, usage)):
        facts, fails = facts + f, fails + x
    return facts, fails


def _env_path(name: str) -> Path | None:
    return Path(os.environ[name]) if os.environ.get(name) else None


def main() -> int:
    facts, fails = run(ROOT, _env_path("RESEARCH_PREFLIGHT_INPUTS"), _env_path("RESEARCH_PREFLIGHT_CUSTODY"), CHECK.cache_dir())
    for line in facts:
        print(line, flush=True)
    for line in fails:
        print(line, file=sys.stderr, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
