"""What live work names, and what a source tree holds besides its commit: the checks the resource steward's node-sweep.sh made
before deleting (its REFS_PY, EXTRA_PY and JOBS_PY), so `research retention gc` deletes nothing live work may still read.

`Live.names(patterns)` -> {pattern: [ref, ...]} for each pattern something live names:
  - a request under <research>/requests/r2* not finished or refused whose runner is alive or not yet launched: its request files,
    launch.json and job.json (`request:<id>`; a refused request never runs, and a dead runner's request holds nothing);
  - a Kueue pod or workload not finished, where k3s is installed: its spec (`pods:<name>`, `workloads:<name>`);
  - a fill job that may still run, in every fill dir but FILL_ENDED (queue/, running/, the held-*/ moved back by hand): its
    script (`fill:queue/<name>`);
  - every installed systemd unit, running or not, in UNIT_DIRS (system and users', drop-ins and links): its file
    (`unit:<path>`);
  - every process: its cwd, root, open files, maps, argv, environment and the scripts its argv names (`proc:<pid>(<comm>)`),
    this one's included.
That is this machine's. What another node runs from a tree here (node 2's custody loop runs node 1's research over ssh) is seen
only by asking that node (`peers`: SshPeer, which runs `research retention names` there); a `research keep` record also pins it.
A pattern starting with `/` names that path or anything beneath it: it matches where no name character follows (`pytest-1` is not
`pytest-12`, while a quote, a colon or `=` after it counts). Any other pattern, a commit sha, names any text that carries it.
LivenessError when it can't tell: kubectl fails, or a process's files can't be read (only root reads every process), so a caller
never takes "nothing names it" from a scan that couldn't look. A process gone mid-scan names nothing.

`outside_commit(tree, bare)`: the files a source tree holds besides its commit's files, build caches and registered fixtures
(fixtures/artifacts.json), which may be a run's output. With .git: git status (untracked, ignored and changed files); without, the
bare repo's commit (`ls-tree`) and files changed after READY.json. git runs as the tree's owner, so root never runs a hook or
filter its config names, and reads neither the system's nor a global config (no fsmonitor daemon). Unknown when it can't tell:
git exits nonzero or says anything on stderr, or a path in the tree (.git and caches aside) is unreadable to the user git runs
as (`unreadable`), since git lists nothing for a directory it can't open.

`JobTrees`: node 1's /workspace/jobs/src copies (job_tree.sh) and which of them a live pod still reads.
"""
from __future__ import annotations

import glob
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

TERMINAL = frozenset({"done", "failed", "cancelled", "exited", "timeout", "timed_out", "refused"})
NAME_TAIL = r"(?![\w.-])"
CACHE = re.compile(r"(^|/)(\.venv|\.lake|__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.hypothesis|node_modules|target|[^/]*\.egg-info)"
                   r"(/|$)|\.pyc$|^READY\.json$|^integrations/vllm/verity_vllm/program/kernels/cpp/build(/|$)")
GONE = (FileNotFoundError, ProcessLookupError)
#: systemd's unit directories, system and user, persistent and transient (globs)
UNIT_DIRS = ("/etc/systemd/system", "/run/systemd/system", "/run/systemd/transient", "/etc/systemd/user", "/root/.config/systemd/user",
             "/home/*/.config/systemd/user", "/run/user/*/systemd/transient")
#: the fill dirs whose jobs never run again; every other one (queue, running, held-*) holds jobs that may
FILL_ENDED = frozenset({"done", "failed", "logs", "withdrawn"})
SCRIPT_MAX = 1 << 20

Kube = Callable[[str], list[dict]]


class LivenessError(RuntimeError):
    """What live work names can't be established; the message says what failed."""


class Unknown(RuntimeError):
    """Whether a tree holds files outside its commit can't be told; the message says why."""


def k3s_kubectl(kind: str) -> list[dict]:
    try:
        out = subprocess.run(["k3s", "kubectl", "get", kind, "-A", "-o", "json"], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        raise LivenessError(f"kubectl get {kind}: {e}") from None
    if out.returncode:
        raise LivenessError(f"kubectl get {kind} exited {out.returncode}: {out.stderr.strip()[:200]}")
    try:
        return json.loads(out.stdout).get("items", [])
    except ValueError as e:
        raise LivenessError(f"kubectl get {kind}: not JSON ({e})") from None


def default_kube() -> Kube | None:
    """`k3s kubectl` where k3s is installed (node 1), else None: a node without Kubernetes has no pods to name anything."""
    return k3s_kubectl if shutil.which("k3s") else None


def finished(obj: dict) -> bool:
    st = obj.get("status") or {}
    return st.get("phase") in ("Succeeded", "Failed") or any(c.get("type") == "Finished" and c.get("status") == "True"
                                                              for c in st.get("conditions") or [])


def matcher(patterns: Iterable[str]) -> Callable[[str], list[str]]:
    pats = list(dict.fromkeys(patterns))
    rx = {p: re.compile(re.escape(p) + NAME_TAIL) for p in pats if p.startswith("/")}
    return lambda text: [p for p in pats if (rx[p].search(text) if p in rx else p in text)]


def _read(path: str | Path) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def _head(path: str | Path) -> str:
    with open(path, "rb") as f:
        return f.read(SCRIPT_MAX).decode(errors="replace")


def _unreadable_units(e: OSError) -> None:
    if not isinstance(e, GONE):
        raise LivenessError(f"can't read {e.filename}: {e.strerror or e} (gc reads every unit file: run it as root)")


def _scripts(args: list[bytes], cwd: str) -> list[str]:
    """The text of each regular file of at most SCRIPT_MAX bytes that a process's arguments name: the script an interpreter
    runs, whose defaults (push.sh's SRC_N1) name what its environment doesn't."""
    out = []
    for a in args:
        if not a or b"\n" in a:
            continue
        p = os.path.join(cwd, os.fsdecode(a))
        try:
            st = os.stat(p)
            if stat.S_ISREG(st.st_mode) and st.st_size <= SCRIPT_MAX:
                out.append(_head(p))
        except PermissionError as e:
            raise LivenessError(f"can't read {p}, which a process's arguments name: {e.strerror}") from None
        except OSError:
            continue
    return out


def _load(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


@dataclass
class Live:
    """The live work on this machine (module docstring), and on each of `peers` (SshPeer). `units`: the systemd unit directories
    (globs). `pids` limits the process scan (a test's own children)."""
    research: Path = Path("/workspace/research")
    fill: Path = Path("/workspace/pouw/fill")
    proc: Path = Path("/proc")
    kube: Kube | None = field(default_factory=default_kube)
    pids: Sequence[int] | None = None
    units: Sequence[str | Path] = UNIT_DIRS
    peers: Sequence[Callable[[list[str]], dict[str, list[str]]]] = ()

    def names(self, patterns: Iterable[str]) -> dict[str, list[str]]:
        pats = list(dict.fromkeys(patterns))
        hits = matcher(pats)
        out: dict[str, list[str]] = {}
        for ref, text in self._texts():
            for p in hits(text):
                out.setdefault(p, []).append(ref)
        for peer in self.peers:
            for p, refs in peer(pats).items():
                out.setdefault(p, []).extend(refs)
        return out

    def _texts(self) -> Iterable[tuple[str, str]]:
        yield from self._requests()
        if self.kube is not None:
            for kind in ("pods", "workloads.kueue.x-k8s.io"):
                for o in self.kube(kind):
                    if not finished(o):
                        yield f"{kind.split('.')[0]}:{(o.get('metadata') or {}).get('name')}", json.dumps(o.get("spec") or {})
        for d in sorted(self.fill.iterdir()) if self.fill.is_dir() else []:
            if d.is_dir() and not d.is_symlink() and d.name not in FILL_ENDED:
                for f in sorted(d.iterdir()):
                    if f.is_file():
                        yield f"fill:{d.name}/{f.name}", _read(f)
        yield from self._units()
        yield from self._processes()

    def _units(self) -> Iterable[tuple[str, str]]:
        """Every unit file, drop-in and link in the unit directories: an installed unit names its tree while it is stopped."""
        for base in sorted({b for d in self.units for b in glob.glob(str(d))}):
            for dp, dns, fns in os.walk(base, onerror=_unreadable_units):
                dns.sort()
                for name in sorted(fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]):
                    p = os.path.join(dp, name)
                    try:
                        text = os.readlink(p) if os.path.islink(p) else _head(p)
                    except GONE:
                        continue
                    except OSError as e:
                        raise LivenessError(f"can't read {p}: {e.strerror or e} (gc reads every unit file: run it as root)") from None
                    yield f"unit:{p}", text

    def _requests(self) -> Iterable[tuple[str, str]]:
        req = self.research / "requests"
        for q in sorted(req.glob("r2*")) if req.is_dir() else []:
            run = self.research / "runs" / q.name
            if _load(run / "status.json").get("state", "") in TERMINAL or (run / "refused.json").exists():
                continue
            if (run / "launch.json").exists():
                pid = str(_load(run / "launch.json").get("runner_pid") or "")
                if not pid or not (self.proc / pid).exists():
                    continue
            files = [f for f in sorted(q.iterdir()) if f.is_file()] if q.is_dir() else []
            yield f"request:{q.name}", "".join(_read(f) for f in [*files, run / "launch.json", run / "job.json"])

    def _processes(self) -> Iterable[tuple[str, str]]:
        pids = [str(p) for p in self.pids] if self.pids is not None else [p for p in os.listdir(self.proc) if p.isdigit()]
        for pid in pids:
            d = self.proc / pid
            try:
                parts = self._process(d)
            except GONE:
                continue
            except OSError as e:
                raise LivenessError(f"can't read {d}: {e.strerror or e} (gc reads every process: run it as root)") from None
            yield f"proc:{pid}({_read(d / 'comm').strip()})", "\n".join(parts)

    @staticmethod
    def _process(d: Path) -> list[str]:
        parts, cwd, argv = [], "/", [b""]
        for link in ("cwd", "root"):
            try:
                parts.append(os.readlink(d / link))
            except GONE:
                pass
            else:
                cwd = parts[-1] if link == "cwd" else cwd
        for fd in os.listdir(d / "fd"):
            try:
                parts.append(os.readlink(d / "fd" / fd))
            except GONE:
                pass
        for f in ("maps", "cmdline", "environ"):
            with open(d / f, "rb") as fh:
                raw = fh.read()
            argv = raw.split(b"\0") if f == "cmdline" else argv
            parts.append(raw.decode(errors="replace").replace("\0", " "))
        return parts + _scripts(argv[1:], cwd)


@dataclass(frozen=True)
class SshPeer:
    """Another node's live work: `remote` (`research retention names --json -` there, as root) reads the patterns on stdin and
    answers {"names": {pattern: [ref, ...]}}, each ref then prefixed `<node>:`. As root, ssh runs as `user`, whose ssh config
    names `host`. LivenessError when the node can't be asked or can't tell (its refusal in a timed window included)."""
    node: str
    host: str
    remote: tuple[str, ...]
    user: str = "research"

    def __call__(self, patterns: list[str]) -> dict[str, list[str]]:
        pats = [p for p in dict.fromkeys(patterns) if "\n" not in p]
        if not pats:
            return {}
        argv = [*(["runuser", "-u", self.user, "--"] if os.geteuid() == 0 else []), "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                self.host, *self.remote]
        try:
            cp = subprocess.run(argv, input="".join(p + "\n" for p in pats), capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.SubprocessError) as e:
            raise LivenessError(f"can't ask {self.node} what it names: {e}") from None
        try:
            doc = json.loads(cp.stdout)
        except ValueError:
            doc = None
        if cp.returncode not in (0, 1) or not isinstance(doc, dict) or not isinstance(doc.get("names"), dict):
            why = (doc or {}).get("refused") if isinstance(doc, dict) else None
            why = why or ((cp.stderr or "").strip().splitlines() or [""])[-1][:200] or f"exit {cp.returncode}, no answer"
            raise LivenessError(f"can't ask {self.node} what it names ({self.host}: {why})")
        return {p: [f"{self.node}:{r}" for r in refs] for p, refs in doc["names"].items() if p in pats and isinstance(refs, list)}


# ---- what a source tree holds besides its commit ----------------------------------------------------------------------------------------

def _git(args: list[str], owner_of: Path) -> subprocess.CompletedProcess:
    st = os.stat(owner_of)
    try:
        home = pwd.getpwuid(st.st_uid).pw_dir
    except KeyError:
        home = "/nonexistent"
    as_owner = {"user": st.st_uid, "group": st.st_gid, "extra_groups": []} if os.geteuid() == 0 and st.st_uid != 0 else {}
    env = {"PATH": os.environ.get("PATH") or "/usr/bin:/bin", "HOME": home, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    try:
        return subprocess.run(["git", *args], capture_output=True, cwd="/", env=env, timeout=1800, **as_owner)
    except (OSError, subprocess.SubprocessError) as e:
        raise Unknown(f"git {args[0]}: {e}") from None


def _git_ok(args: list[str], owner_of: Path, what: str) -> bytes:
    """git's stdout; Unknown when it exits nonzero or says anything on stderr (`could not open directory`, run as the owner,
    while it lists nothing for that directory and exits 0)."""
    cp = _git(args, owner_of)
    err = cp.stderr.decode(errors="replace").strip()
    if cp.returncode or err:
        raise Unknown(f"{what} exited {cp.returncode}" + (f" and said {err.splitlines()[0][:200]!r}" if err else ""))
    return cp.stdout


def unreadable(tree: Path) -> str | None:
    """The first path in `tree` (.git and build caches aside) that git, run as `_git` runs it (the tree's owner and group when
    this is root), can't read, else None; OSError when the walk fails."""
    top = os.stat(tree)
    as_owner = os.geteuid() == 0 and top.st_uid != 0

    def ok(path: str, need: int) -> bool:
        if not as_owner:
            return os.access(path, need)
        st = os.lstat(path)
        shift = 6 if st.st_uid == top.st_uid else 3 if st.st_gid == top.st_gid else 0
        return (st.st_mode >> shift) & need == need

    def fail(e: OSError) -> None:
        raise e

    if not ok(str(tree), os.R_OK | os.X_OK):
        return "."
    for dp, dns, fns in os.walk(tree, onerror=fail):
        r, sub = os.path.relpath(dp, tree), []
        for name, need in sorted([(d, os.R_OK | os.X_OK) for d in dns] + [(f, os.R_OK) for f in fns]):
            p, rel = os.path.join(dp, name), os.path.normpath(os.path.join(r, name))
            if rel == ".git" or os.path.islink(p) or CACHE.search(rel + ("/" if name in dns else "")):
                continue
            if not ok(p, need):
                return rel + ("/" if name in dns else "")
            if name in dns:
                sub.append(name)
        dns[:] = sub
    return None


def outside_commit(tree: Path | str, bare: Path | str) -> list[str]:
    """What `tree` holds besides its commit's files (module docstring): [] when nothing, each extra as a path, a changed tracked file
    as `<git status code> <path>` (with .git) or `changed <path>` (without); Unknown when it can't tell."""
    tree = Path(tree)
    try:
        reg = {v for k, v in _load(tree / "fixtures/artifacts.json").items() if k.startswith("art:") and isinstance(v, str)}
    except AttributeError:
        reg = set()

    def keep(rel: str) -> bool:
        return not CACHE.search(rel) and not any(rel == r or rel.startswith(r.rstrip("/") + "/") for r in reg)

    def files(rel: str) -> Iterable[str]:
        p = tree / rel
        if not p.is_dir() or p.is_symlink():
            yield os.path.normpath(rel)
            return
        for dp, dns, fns in os.walk(p):
            r = os.path.relpath(dp, tree)
            dns[:] = sorted(d for d in dns if not CACHE.search(os.path.normpath(os.path.join(r, d)) + "/"))
            yield from (os.path.normpath(os.path.join(r, f)) for f in sorted(fns))

    try:
        if bad := unreadable(tree):
            uid = os.stat(tree).st_uid if os.geteuid() == 0 else os.geteuid()
            raise Unknown(f"{bad} can't be read by uid {uid}, as whom git runs, so git can't tell what it holds")
    except OSError as e:
        raise Unknown(f"can't walk it: {e.strerror or e}") from None
    extra: list[str] = []
    if (tree / ".git").is_dir():
        out = _git_ok(["-C", str(tree), "status", "--porcelain", "--ignored", "-z", "--untracked-files=normal"], tree, "git status")
        entries = iter(out.split(b"\0"))
        for e in entries:
            if not e:
                continue
            code, rel = e[:2].decode(), os.path.normpath(e[3:].decode(errors="replace"))
            if code[0] in "RC":
                next(entries, None)
            if code in ("??", "!!"):
                if not CACHE.search(rel + "/"):
                    extra += [f for f in files(rel) if keep(f)]
            else:
                extra.append(f"{code.strip()} {rel}")
        return extra
    try:
        ls = _git_ok(["--git-dir", str(bare), "ls-tree", "-r", "--name-only", "-z", tree.name], tree, "git ls-tree")
    except Unknown as e:
        raise Unknown(f"no .git, and the bare repo can't list the commit ({e})") from None
    tracked = {x.decode(errors="replace") for x in ls.split(b"\0") if x}
    try:
        shipped = os.stat(tree / "READY.json").st_mtime + 60
    except OSError:
        raise Unknown("no .git and no READY.json") from None
    for f in files("."):
        if f in tracked:
            if os.lstat(tree / f).st_mtime > shipped:
                extra.append(f"changed {f}")
        elif keep(f):
            extra.append(f)
    return extra


# ---- node 1's job trees -----------------------------------------------------------------------------------------------------------------

CONTENT = re.compile(r"[0-9a-f]{16}")


def live_pods(kube: Kube) -> set[str]:
    return {(o.get("metadata") or {}).get("name") for o in kube("pods") if (o.get("status") or {}).get("phase") not in ("Succeeded", "Failed")}


def _mtime(p: Path) -> float:
    """A copy's age stamp: `cp -a` gives it its source's mtime and a fresh ctime."""
    try:
        st = os.stat(p)
    except OSError:
        return 0.0
    return max(st.st_mtime, st.st_ctime)


@dataclass
class JobTrees:
    """/workspace/jobs/src (job_tree.sh): a per-pod copy (`<hostname>`, `pod-<hostname>`, made at pod start, read by nothing once its
    pod ends) and a content copy (`<id16>`, shared by every job of that content, aged by the newest by-pod/<pod> file naming it,
    since a reused copy keeps its mtime). A content copy without `.copied` is job_tree.sh's to remake, so it is never a candidate.
    `pods`: the live pods (`live_pods`); the ages alone need none."""
    root: Path
    pods: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.newest: dict[str, float] = {}
        self.by_pod: dict[str, list[tuple[str, float]]] = {}
        bp = self.root / "by-pod"
        for f in sorted(bp.iterdir()) if bp.is_dir() else []:
            try:
                path, mt = f.read_text().strip(), f.stat().st_mtime
            except OSError:
                continue
            if os.path.dirname(path) != str(self.root):
                continue
            e = os.path.basename(path)
            self.newest[e] = max(self.newest.get(e, 0.0), mt)
            self.by_pod.setdefault(e, []).append((f.name, mt))

    def age_stamp(self, e: str) -> float:
        p = self.root / e
        t = _mtime(p)
        return max(t, _mtime(p / ".copied"), self.newest.get(e, 0.0)) if CONTENT.fullmatch(e) else t

    def held(self, e: str, since: float | None = None) -> str:
        """Why a live pod still reads copy `e` ('' when none does); `since`: a by-pod file written after it also holds a content copy."""
        if CONTENT.fullmatch(e):
            if live := [pod for pod, _ in self.by_pod.get(e, []) if pod in self.pods]:
                return f"by-pod of live pod {live[0]}"
            if since is not None and (new := [pod for pod, mt in self.by_pod.get(e, []) if mt >= since]):
                return f"by-pod written since the rename: {new[0]}"
            return ""
        host = e[4:] if e.startswith("pod-") else e
        return f"live pod {host}" if host in self.pods or e in self.pods else ""

    def copies(self) -> Iterable[tuple[str, str | None]]:
        """(name, why it is never a candidate, else None) of every copy."""
        for e in sorted(os.listdir(self.root)):
            p = self.root / e
            if e == "by-pod" or e.startswith(".") or ".partial." in e or p.is_symlink() or not p.is_dir():
                continue
            if CONTENT.fullmatch(e) and not (p / ".copied").exists():
                yield e, "no .copied, so job_tree.sh may be remaking it"
            else:
                yield e, None


__all__ = ["CACHE", "JobTrees", "Live", "LivenessError", "TERMINAL", "Unknown", "default_kube", "finished", "k3s_kubectl", "live_pods",
           "matcher", "outside_commit", "unreadable"]
