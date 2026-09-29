"""`tools/check/preflight.py`: what `research run --on` checks on a machine before a check is claimed there, on a fake tree, a fake
PATH of the tools `pod_setup.sh` installs, a directory standing in for the store's remote, and a fake disk."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GIT = shutil.which("git")
spec = importlib.util.spec_from_file_location("check_preflight", ROOT / "tools" / "check" / "preflight.py")
PF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PF)

BUNDLE = b"a pinned bundle"
UPSTREAM = b"the upstream build"
TOOLCHAIN = "leanprover/lean4:v4.99.0"
AVX = "flags\t\t: fpu sse2 avx2 avx512f\n"
NO_AVX = "flags\t\t: fpu sse2 avx2\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tree(tmp_path: Path, *, rustflags: str = "-C target-cpu=x86-64-v3", pinned: bool = True, at: Path | None = None) -> Path:
    t = at or tmp_path / "tree"
    (t / "tools" / "check").mkdir(parents=True)
    (t / "tools" / "check" / "pod_setup.sh").write_text("set -e\nUV_VERSION=${UV_VERSION:-0.12.20}\n")
    for pkg, deps in (("backends/flock/verifier/lean", []), ("pkg/a", [{"type": "git", "name": "mathlib", "rev": "r1"}])):
        d = t / pkg
        d.mkdir(parents=True)
        (d / "lakefile.toml").write_text("")
        (d / "lean-toolchain").write_text(TOOLCHAIN + "\n")
        (d / "lean-audit.json").write_text("{}")
        (d / "lake-manifest.json").write_text(json.dumps({"packages": deps}))
    (t / "backends" / "flock" / "verifier" / "upstream.json").write_text(json.dumps({"upstream": {"sha256": _sha(UPSTREAM), "rustflags": rustflags}}))
    deps = {PF.LA.deps_key(t / "pkg" / "a"): {"package": "pkg/a", "sha256": _sha(BUNDLE), "bytes": len(BUNDLE), "run": "r1"}} if pinned else {}
    (t / "tools" / "check" / "lean-deps.json").write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": deps}))
    for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "tree"]):         # not this machine's git config (a signing helper)
        subprocess.run([GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=t, check=True, capture_output=True,
                       env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
    return t


def _head(tree: Path) -> str:
    return subprocess.run([GIT, "rev-parse", "HEAD"], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()


def _key(tree: Path) -> str:
    return PF.LA.deps_key(tree / "pkg" / "a")


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    """The preflight may remove the caches under HOME: each test's are its own, never this machine's."""
    home = tmp_path / "home"
    home.mkdir()
    for k, v in {"HOME": home, "XDG_CACHE_HOME": home / ".cache", "UV_CACHE_DIR": home / ".cache" / "uv",
                 "TORCH_EXTENSIONS_DIR": home / ".cache" / "torch_extensions", "FLOCK_CHECKOUT": home / "flock"}.items():
        monkeypatch.setenv(k, str(v))


@pytest.fixture
def path(tmp_path, monkeypatch):
    """A PATH holding git and what `pod_setup.sh` installs; `path(uv=None)` leaves a tool out."""
    b = tmp_path / "bin"
    b.mkdir()

    def make(uv: str | None = "0.12.20", toolchain: str | None = TOOLCHAIN, cargo: bool = True, lake: bool = True) -> None:
        for p in b.iterdir():
            p.unlink()
        (b / "git").symlink_to(GIT)
        for name, out in (("uv", uv and f"uv {uv} (abc 2026-09-01)"), ("elan", toolchain and f"{toolchain} (default)"),
                          ("cargo", cargo and "cargo 1.90.0"), ("lake", lake and "Lake version 5.0.0")):
            if out:
                (b / name).write_text(f"#!/bin/sh\necho '{out}'\n")
                (b / name).chmod(0o755)
        monkeypatch.setenv("PATH", str(b))
    make()
    return make


def _custody(tmp_path: Path, *objects: bytes) -> Path:
    remote = tmp_path / "remote"
    for data in objects:
        (remote / "objects" / "sha256").mkdir(parents=True, exist_ok=True)
        (remote / "objects" / "sha256" / _sha(data)).write_bytes(data)
    c = tmp_path / "custody"
    c.mkdir()
    (c / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{remote}"\n')
    (c / "cred.json").write_text(json.dumps({"AWS_ACCESS_KEY_ID": "dummy", "AWS_SECRET_ACCESS_KEY": "dummy", "AWS_SESSION_TOKEN": "dummy"}))
    return c


def _inputs(tmp_path: Path, urls: dict[str, str] | None = None, *files: bytes) -> Path:
    d = tmp_path / "inputs"
    d.mkdir()
    if urls is not None:
        (d / PF.CHECK.LEAN_DEPS_URLS).write_text(json.dumps(urls))
    for i, data in enumerate(files):
        (d / f"sent-{i}.tar.gz").write_bytes(data)
    return d


def _usage(free_gb: float):
    return lambda p: types.SimpleNamespace(free=int(free_gb * PF.GB))


def _run(tmp_path, tree, *, inputs=None, custody=None, warm=True, cpu=NO_AVX, free_gb=100.0, sha=None, runs=None, usage=None, now=None):
    cache = tmp_path / "cache"
    if warm:
        (cache / "lean-deps" / _key(tree)).mkdir(parents=True, exist_ok=True)
    (tmp_path / "cpuinfo").write_text(cpu)
    return PF.run(tree, inputs, custody, cache, tmp_path / "cpuinfo", usage or _usage(free_gb), sha=sha, runs=runs, now=now)


def test_a_prepared_pod_passes(tmp_path, path):
    tree = _tree(tmp_path)
    facts, fails = _run(tmp_path, tree, inputs=_inputs(tmp_path), custody=_custody(tmp_path, BUNDLE), sha=_head(tree))
    assert fails == []
    assert f"tree: a clean checkout of {_head(tree)[:12]}" in facts
    assert "uv 0.12.20" in facts and "cargo 1.90.0" in facts and f"elan has {TOOLCHAIN}" in facts
    assert f"pkg/a: Lean dependencies warm here ({_key(tree)[:12]})" in facts
    assert f"store: pkg/a's Lean bundle {_key(tree)[:12]} read with the run's custody key" in facts
    assert any(f.startswith("disk: 100 GB free") and "needs 45 GB" in f for f in facts), "tree and cache share a filesystem: one sum"


@pytest.mark.parametrize("tools, why", [
    ({"uv": None}, "uv is not on the PATH"),
    ({"uv": "0.9.0"}, "uv is 0.9.0, not the 0.12.20 pod_setup.sh pins"),
    ({"toolchain": "leanprover/lean4:v4.9"}, f"elan has no {TOOLCHAIN}, which the Lake packages pin"),
    ({"toolchain": None}, "elan is not on the PATH, or does not run"),
    ({"cargo": False}, "cargo is not on the PATH, or does not run"),
    ({"lake": False}, "lake is not on the PATH"),
])
def test_a_pod_setup_sh_did_not_prepare_is_refused_with_the_remedy(tmp_path, path, tools, why):
    path(**tools)
    _, fails = _run(tmp_path, _tree(tmp_path), inputs=_inputs(tmp_path), custody=_custody(tmp_path, BUNDLE))
    assert fails == [why, f"prepare the pod: `{PF.POD_SETUP}`"]


def test_a_tree_that_is_not_a_clean_checkout_of_the_commit_is_refused(tmp_path, path):
    """`suites.py` keys from git's view of the tree, and a changed tree records its attempt dirty, which no merge accepts."""
    tree = _tree(tmp_path)
    ok = dict(inputs=_inputs(tmp_path), custody=_custody(tmp_path, BUNDLE))
    _, fails = _run(tmp_path, tree, **ok, sha="0" * 40)
    assert fails == [f"tree: {tree} is a checkout of {_head(tree)[:12]}, not of the commit the run names (000000000000)"]
    (tree / "stray.txt").write_text("left by an earlier run\n")
    _, fails = _run(tmp_path, tree, **ok, sha=_head(tree))
    assert len(fails) == 1 and fails[0].startswith(f"tree: git sees 1 change(s) in {tree} (?? stray.txt)")
    shutil.rmtree(tree / ".git")
    _, fails = _run(tmp_path, tree, **ok)
    assert fails[0].startswith(f"tree: {tree} is not a git checkout (the archive stream, or a launcher that predates checkouts")


def test_the_uv_pin_is_read_from_pod_setup_sh():
    assert PF.pinned_uv(ROOT) and PF.pinned_uv(ROOT)[0].isdigit()


def test_lean_dependencies_come_from_the_store_or_nowhere(tmp_path, path):
    tree = _tree(tmp_path)
    k = _key(tree)
    _, fails = _run(tmp_path, tree, inputs=_inputs(tmp_path), custody=_custody(tmp_path, BUNDLE), warm=False)
    assert fails == [f"pkg/a: its Lean dependencies ({k[:12]}) are not warm here and the run carries no URL for their bundle: send "
                     f"`$(uv run python tools/check/check.py --lean-deps-files)`"]


def test_an_unpinned_manifest_would_clone_from_github_and_is_refused(tmp_path, path):
    tree = _tree(tmp_path, pinned=False)
    _, fails = _run(tmp_path, tree, inputs=_inputs(tmp_path), custody=_custody(tmp_path, UPSTREAM), warm=False)
    assert len(fails) == 1 and "pins no bundle for its manifest and toolchain" in fails[0] and "clone them from GitHub" in fails[0]


def test_a_bundle_to_restore_is_read_through_its_url_and_counts_on_disk(tmp_path, path):
    tree = _tree(tmp_path)
    k = _key(tree)
    (tmp_path / "bundle").write_bytes(BUNDLE)
    facts, fails = _run(tmp_path, tree, inputs=_inputs(tmp_path, {k: (tmp_path / "bundle").as_uri()}), warm=False)
    assert fails == [] and f"pkg/a: Lean dependencies restorable from the store ({k[:12]}, 0 MiB)" in facts
    assert f"store: pkg/a's Lean bundle {k[:12]} read through its URL" in facts
    assert any("needs 58 GB (1 Lean dependency tree(s) to restore)" in f for f in facts), "45 GB and 1.25 x a 10 GB tree"
    _, fails = _run(tmp_path / "gone", _tree(tmp_path / "gone"), warm=False,
                    inputs=_inputs(tmp_path / "gone", {k: (tmp_path / "missing").as_uri()}))
    assert fails == [f"store: pkg/a's Lean bundle {k[:12]} could not be read through its URL (URLError)"]


def test_the_store_read_uses_the_custody_key_and_fails_without_a_credential(tmp_path, path):
    tree = _tree(tmp_path)
    _, fails = _run(tmp_path, tree, inputs=_inputs(tmp_path), custody=_custody(tmp_path, b"something else"))
    assert fails == [f"store: pkg/a's Lean bundle {_key(tree)[:12]} is not on the remote as pinned"]
    _, fails = _run(tmp_path / "none", _tree(tmp_path / "none"), inputs=_inputs(tmp_path / "none"))
    assert len(fails) == 1 and fails[0].startswith("store: the run carries neither a custody key nor a URL")


@pytest.mark.parametrize("rustflags, sent, cpu, fail", [
    ("-C target-cpu=x86-64-v3", True, NO_AVX, False),
    ("-C target-cpu=native", False, NO_AVX, False),
    ("-C target-cpu=native", True, AVX, False),
    ("-C target-cpu=native", True, NO_AVX, True),
    ("-C target-cpu=x86-64-v4", True, NO_AVX, True),
    ("", True, NO_AVX, True),
])
def test_avx512f_is_required_only_when_the_run_carries_a_build_that_needs_it(tmp_path, path, rustflags, sent, cpu, fail):
    tree = _tree(tmp_path, rustflags=rustflags)
    inputs = _inputs(tmp_path, None, *([b"other input", UPSTREAM] if sent else [b"other input"]))
    _, fails = _run(tmp_path, tree, inputs=inputs, custody=_custody(tmp_path, BUNDLE), cpu=cpu)
    flags = rustflags or "the build host's own (target-cpu=native)"
    want = f"cpu: no avx512f, which the pinned upstream build the run carries needs ({flags}): its replays would die of SIGILL; use a pod with AVX-512"
    assert fails == ([want] if fail else [])


def test_too_little_disk_is_refused(tmp_path, path):
    """TO3's check ran out of space on a pod that started with 36 GB free."""
    _, fails = _run(tmp_path, _tree(tmp_path), inputs=_inputs(tmp_path), custody=_custody(tmp_path, BUNDLE), free_gb=36)
    assert len(fails) == 1 and fails[0].startswith("disk: 36 GB free") and "needs 45 GB" in fails[0] and "larger pod" in fails[0]


def _recorded(runs: Path, run_id: str, tool: str, free_gb: list[float]) -> None:
    """A finished run of ``tool`` whose sampler read ``free_gb`` on its run directory's filesystem, sample by sample."""
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"store": {"tool": {"name": tool}}}))
    rows = [{"kind": "cgroup", "scope": "workload"}] + [{"kind": "storage", "mounts": [{"path": str(d), "free_bytes": int(g * PF.GB)}]}
                                                         for g in free_gb]
    (d / "resources.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def test_the_floor_rises_to_what_the_recent_checks_here_took(tmp_path, path):
    tree, runs = _tree(tmp_path), tmp_path / "runs"
    ok = dict(inputs=_inputs(tmp_path), custody=_custody(tmp_path, BUNDLE))
    _recorded(runs, "r20260929-090000-aaaa", "check", [90, 70, 50, 60])         # took 40 GB at its lowest
    _recorded(runs, "r20260929-100000-bbbb", "check", [80, 75])                 # 5 GB: its verdicts mostly came from a pack
    _recorded(runs, "r20260929-110000-cccc", "lean_audit", [90, 5])             # not a check
    _, fails = _run(tmp_path, tree, **ok, free_gb=48, runs=runs)
    assert len(fails) == 1 and fails[0].startswith(f"disk: 48 GB free on {tree}'s filesystem, where a check needs 50 GB "
                                                   f"(1.25 x the 40 GB check r20260929-090000-aaaa took here)"), fails
    _, fails = _run(tmp_path, tree, **ok, free_gb=52, runs=runs)
    assert fails == []
    for i in range(PF.RECENT):                                                  # five newer checks, each small: the 40 GB one ages out
        _recorded(runs, f"r20260929-12000{i}-dddd", "check", [80, 70])
    _, fails = _run(tmp_path, tree, **ok, free_gb=46, runs=runs)
    assert fails == [], "12.5 GB is under the 45 GB floor, which a recorded check only ever raises"


NOW = 2_000_000_000.0                       # the clock the eviction tests hand the preflight
OLD, JUST = NOW - 3 * 3600, NOW - 60


def _du(*dirs: Path) -> int:
    return sum(p.stat().st_size for d in dirs if d.exists() for p in d.rglob("*") if p.is_file() and not p.is_symlink())


def _junk(d: Path, size: int, mtime: float = OLD, marker: str | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "blob").write_bytes(b"x" * size)
    for p in ([d / marker] if marker else []):
        p.write_text("{}\n")
        os.utime(p, (mtime, mtime))
    os.utime(d, (mtime, mtime))
    return d


@pytest.fixture
def pod(tmp_path, monkeypatch):
    """A machine as `research run --on` leaves one, measured in bytes (1 GB is 1 byte): this commit's shipped tree with its warm
    dependencies (50), a dead audit's scratch (300), an unpinned dependency tree (150), an idle tree (200), a tree shipped a
    minute ago (200) and uv's cache (400); its disk has 50 free."""
    monkeypatch.setattr(PF, "GB", 1)
    monkeypatch.setattr(PF, "TREE_GB", 0)
    root, cache, home = tmp_path / "root", tmp_path / "cache", tmp_path / "home"
    tree = _tree(tmp_path, at=root / "src" / ("a" * 40))
    (tree / ".git" / "info" / "exclude").write_text("/READY.json\n")
    (tree / "READY.json").write_text("{}\n")
    (root / "runs").mkdir()
    k = _key(tree)
    j = {"scratch": _junk(cache / "lean-audit-scratch-dead", 300), "unpinned": _junk(cache / "lean-deps" / ("f" * 64), 150),
         "needed": _junk(cache / "lean-deps" / k, 50), "idle": _junk(root / "src" / ("b" * 40), 200, marker="READY.json"),
         "shipped": _junk(root / "src" / ("c" * 40), 200, JUST, marker="READY.json"), "uv": _junk(home / ".cache" / "uv", 400)}
    others = lambda: [d for d in (root / "src").iterdir() if d != tree]     # noqa: E731
    cap = _du(cache, home, *others()) + 50

    def disk(floor: int):
        monkeypatch.setattr(PF, "CACHE_GB", floor)
        usage = lambda p: types.SimpleNamespace(free=cap - _du(cache, home, *others()))     # noqa: E731
        return PF.disk(tree, cache, [], usage, root / "runs", {k}, NOW)
    return types.SimpleNamespace(tree=tree, root=root, junk=j, disk=disk)


def _left(pod) -> set[str]:
    return {name for name, d in pod.junk.items() if d.exists()}


def test_room_is_made_in_order_and_only_down_to_the_floor(pod):
    facts, fails = pod.disk(100)
    assert fails == [] and _left(pod) == {"unpinned", "needed", "idle", "shipped", "uv"}, "a dead audit's scratch was enough"
    assert any(f.startswith(f"disk: made room on {pod.tree}'s filesystem, 50 -> 350 GB free: a dead Lean audit's scratch") for f in facts)
    facts, fails = pod.disk(700)
    assert fails == [] and _left(pod) == {"needed", "shipped", "uv"}, "then the unpinned dependencies, then the idle tree"


def test_a_pod_is_refused_only_once_nothing_it_can_spare_is_left(pod):
    _, fails = pod.disk(1250)
    assert _left(pod) == {"needed", "shipped"}, "uv's cache goes too, and this commit's dependencies and a fresh ship stay"
    assert len(fails) == 1 and fails[0].endswith(f"'s filesystem, where a check needs 1250 GB, after removing what this machine can "
                                                 f"spare: use a larger pod"), fails


def test_nothing_a_live_run_or_process_holds_goes(pod):
    run = pod.root / "runs" / "r20260929-230000-live"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    (run / "launch.json").write_text(json.dumps({"runner_pid": os.getpid(), "cwd": str(pod.junk["idle"])}))
    sitting = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], cwd=pod.junk["scratch"], stdin=subprocess.PIPE)
    try:
        _, fails = pod.disk(1250)
    finally:
        sitting.stdin.close()
        sitting.wait()
    assert _left(pod) == {"scratch", "needed", "idle", "shipped", "uv"}, "shared caches wait while another run is live"
    assert len(fails) == 1


def test_the_check_tool_declares_this_preflight():
    spec = importlib.util.spec_from_file_location("check_tool_decl", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.CHECK.preflight == ["tools/check/preflight.py"] and (ROOT / tool.CHECK.preflight[0]).is_file()
