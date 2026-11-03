"""answers: the verifier of record's answers (`proofs.verify`) on a window's statements, each run once and shared by
every step that judges them, the drawn ones J at a time.

Credit's window check and the period's (`credit.window`, `credit.period_credit`, through `credit._verified` or
`credit._lean_verified`) each run `proofs.verify` on every statement they ask for: the same runs on the same files.
`Answers` is that ``run``. It answers a run whose inputs it has answered before by its key, from its memo, and otherwise
runs the verifier and keeps the answer; each step still judges every answer itself, in its own order, and refuses where
it refuses. `fill` runs a window's drawn statements (`window_runs`: the runs `credit._verified` makes, in its order)
into the memo first, `jobs` at a time, one ``LEAN`` line a verdict as each ends, so a run cut short says how far it got.

A run's key is the SHA-512 of all of its inputs (`key`): the verifier's executable, the circuit file, the public file,
every file of each session directory (its name and SHA-512), the paths the verifier is handed, the committed circuit
digest, the program's bytes, and the partition object and the ``--registered`` file as canonical JSON. Those are the
statement's proof files and what credit builds from the window's entries, root, rounds, salt and draw, so a changed
input is another key. The memo is the verifier's own: in memory, or a directory in the verifier's scratch (0700, each
answer a 0600 file), never the serve dir or anywhere the prover writes. ``drawn.py fill`` (benchmarks/pouw/h1_prove) is
`fill`'s command line, and ``--lean-answers DIR`` hands a memo directory to credit's window check
(``pouw_h1_gpu --check-challenger``) and to the period's (``pouw_piece.py period``).
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import functools
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

#: the peak resident memory of one lane statement's ``flock-verify verify --zk`` at K = 2048, which `jobs` sizes J by (GiB)
RUN_GB = 47.5
#: the available memory `jobs` leaves beside the runs: the Python verifier holding the window's statements and asks (a
#: few GiB in the lanes form) and slack for the kernel (GiB)
HEADROOM_GB = 16.0
#: the settings that override J and the headroom
JOBS_ENV, HEADROOM_ENV = "H1_LEAN_JOBS", "H1_LEAN_HEADROOM_GB"
#: the errors credit reads as a verifier that can't run (`credit._verified`'s), kept in the memo as they were raised
ERRORS: dict[str, type[Exception]] = {"LeanError": PF.LeanError, "OSError": OSError, "ValueError": ValueError,
                                      "KeyError": KeyError, "TypeError": TypeError}
SAY = functools.partial(print, flush=True)


def memory_gb() -> float:
    """The memory free for the runs now (GiB): the host's ``MemAvailable``, or less where this process's cgroup v2 or an
    ancestor of it has a ``memory.max``, that limit less its ``memory.current``."""
    with open("/proc/meminfo") as f:
        free = next(int(x.split()[1]) * 1024 for x in f if x.startswith("MemAvailable:"))
    try:
        rel = next(x.split("::", 1)[1].strip() for x in Path("/proc/self/cgroup").read_text().splitlines()
                   if x.startswith("0::"))
    except (OSError, StopIteration):
        rel = None
    if rel is not None:
        d = Path("/sys/fs/cgroup") / rel.lstrip("/")
        for p in (d, *d.parents):
            try:
                cap, used = (p / "memory.max").read_text().strip(), (p / "memory.current").read_text().strip()
            except OSError:
                cap, used = "max", "0"
            if cap.isdigit() and used.isdigit():
                free = min(free, max(0, int(cap) - int(used)))
            if p == Path("/sys/fs/cgroup"):
                break
    return free / 2**30


def jobs(setting: int | str | None = None, memory: float | None = None, headroom: float | None = None
         ) -> tuple[int, str]:
    """J, the runs `fill` keeps going at once, and how it was sized: `setting` (else ``H1_LEAN_JOBS``) when given, else
    floor((memory − headroom) / `RUN_GB`), at least 1, with `memory` `memory_gb`'s (what is available as the fill starts)
    and `headroom` ``H1_LEAN_HEADROOM_GB`` or `HEADROOM_GB`."""
    setting = setting if setting is not None else os.environ.get(JOBS_ENV) or None
    if setting is not None:
        return max(1, int(setting)), f"set (--jobs or {JOBS_ENV})"
    memory = memory_gb() if memory is None else memory
    headroom = float(os.environ.get(HEADROOM_ENV) or HEADROOM_GB) if headroom is None else headroom
    return (max(1, math.floor((memory - headroom) / RUN_GB)),
            f"floor(({memory:.1f} GiB available - {headroom:g} GiB headroom) / {RUN_GB:g} GiB a run)")


def canonical(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


#: the size from which `file_digest` hashes a file once per process (a circuit file); a smaller one it hashes each time
CACHED_BYTES = 1 << 30
_DIGESTS: dict[tuple, str] = {}
_DIGESTS_LOCK = threading.Lock()


def file_digest(path: Path) -> str:
    """`proofs.file_sha512` of the file at `path`: once per process for a file of at least `CACHED_BYTES` whose device,
    inode, size, mtime and ctime are unchanged (a lane's circuit file is GBs, and every lane statement names it), and
    each time for a smaller one (session files and public files, which a rewrite in place within one clock tick would
    leave with the same stat)."""
    st = os.stat(path)
    if st.st_size < CACHED_BYTES:
        return PF.file_sha512(Path(path))
    k = (str(Path(path).resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    with _DIGESTS_LOCK:
        got = _DIGESTS.get(k)
    if got is None:
        got = PF.file_sha512(Path(path))
        with _DIGESTS_LOCK:
            _DIGESTS[k] = got
    return got


def key(proof: PF.Proof, *, exe: Path, circuit_sha512: str, program: bytes, partition: Mapping[str, Any],
        registered: Mapping[str, Any] | None = None) -> str:
    """The SHA-512 of every input of `proofs.verify`'s run on `proof` (the module docstring); OSError when a file it names
    is missing, which `Answers` leaves to the verifier's own answer."""
    sessions = []
    for s in proof.sessions:
        files = sorted(p for p in Path(s).rglob("*") if p.is_file())
        if not files:
            raise FileNotFoundError(f"session directory {s} holds no file")
        sessions.append([[str(p.relative_to(s)), file_digest(p)] for p in files])
    return hashlib.sha512(canonical({
        "exe": file_digest(exe), "circuit": file_digest(proof.circuit), "public": file_digest(proof.public),
        "sessions": sessions, "paths": [str(proof.circuit), str(proof.public), [str(s) for s in proof.sessions]],
        "circuit_sha512": circuit_sha512, "program": hashlib.sha512(program).hexdigest(), "partition": partition,
        "registered": registered})).hexdigest()


def _dump(answer: tuple[list[PF.Finding] | None, tuple[str, str] | None]) -> dict[str, Any]:
    got, why = answer
    return {"findings": None if got is None else [dataclasses.asdict(f) for f in got],
            "why": None if why is None else list(why)}


@functools.cache
def _kept_error(name: str) -> type[Exception]:
    """`ERRORS[name]` whose message is the kept one exactly (a KeyError's ``str`` would quote it again)."""
    return type(name, (ERRORS[name],), {"__str__": lambda self: str(self.args[0])})


def _loaded(entry: Mapping[str, Any]) -> tuple[list[PF.Finding] | None, tuple[str, str] | None]:
    if "error" in entry:
        name, message = entry["error"]
        raise _kept_error(name)(message)
    a = entry["answer"]
    got = None if a["findings"] is None else [PF.Finding(**{**f, "units": tuple(f["units"])}) for f in a["findings"]]
    return got, None if a["why"] is None else (a["why"][0], a["why"][1])


def verdict(entry: Mapping[str, Any]) -> str:
    """One kept answer in a line: its sessions accepted, the first rejected one and why, the refusal, or the error."""
    if "error" in entry:
        return f"error {entry['error'][0]}: {entry['error'][1]}"[:300]
    a = entry["answer"]
    if a["why"] is not None:
        return f"refused {a['why'][0]}: {a['why'][1]}"[:300]
    bad = [f for f in a["findings"] if not f["accepted"]]
    if bad:
        return f"rejected session {bad[0]['session']}: {bad[0]['why']}"[:300]
    return f"accepted {len(a['findings'])} of {len(a['findings'])} sessions"


class Answers:
    """A ``run`` for credit (`credit.window`'s, `credit.period_credit`'s, `credit._verified`'s): `verify`'s answer on a
    run, from the memo when a run of the same key was answered before (in this process, or by any step whose memo is
    `directory`), else `verify`'s own, kept. `by` names the step in each answer it keeps and in its ``LEAN-ANSWER``
    lines, one a run: ``ran``, or ``by key from`` the step that ran it. `ran` and `taken` count them (`taken` by step),
    and `unkeyed` the runs left to `verify` unkept because a file they name is missing; `summary` says all three."""

    def __init__(self, directory: Path | None = None, *, by: str, verify: Callable[..., Any] | None = None,
                 exe: Callable[[], Path] | None = None, say: Callable[[str], Any] | None = SAY,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.directory, self.by, self.say, self.clock = directory, by, say, clock
        self.verify = verify or (lambda proof, **kw: PF.verify(proof, **kw))
        self.exe = exe or (lambda: PF.executable())
        self.memo: dict[str, dict[str, Any]] = {}
        self.ran, self.unkeyed, self.taken = 0, 0, {}
        if directory is not None:
            old = os.umask(0o077)
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            finally:
                os.umask(old)

    def key(self, proof: PF.Proof, **kw: Any) -> str:
        return key(proof, exe=self.exe(), **kw)

    def lookup(self, k: str) -> dict[str, Any] | None:
        got = self.memo.get(k)
        if got is None and self.directory is not None:
            try:
                got = json.loads((Path(self.directory) / f"{k}.json").read_text())
            except FileNotFoundError:
                return None
            self.memo[k] = got
        return got

    def keep(self, k: str, entry: dict[str, Any]) -> None:
        self.memo[k] = entry
        if self.directory is not None:
            tmp = Path(self.directory) / f".{k}.{os.getpid()}.{threading.get_ident()}"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(entry))
            os.replace(tmp, Path(self.directory) / f"{k}.json")

    def run(self, k: str, proof: PF.Proof, **kw: Any) -> dict[str, Any]:
        """`verify` on the run of key `k`, its answer (or the error credit reads as one that can't run) kept."""
        t = self.clock()
        try:
            entry: dict[str, Any] = {"answer": _dump(self.verify(proof, **kw))}
        except tuple(ERRORS.values()) as e:
            entry = {"error": [next(n for n, c in ERRORS.items() if isinstance(e, c)), str(e)]}
        entry.update(by=self.by, seconds=round(self.clock() - t, 1))
        self.keep(k, entry)
        return entry

    def __call__(self, proof: PF.Proof, **kw: Any) -> tuple[list[PF.Finding] | None, tuple[str, str] | None]:
        try:
            k = self.key(proof, **kw)
        except OSError:
            self.unkeyed += 1
            return self.verify(proof, **kw)
        entry = self.lookup(k)
        if entry is None:
            entry, how = self.run(k, proof, **kw), "ran"
            self.ran += 1
        else:
            how = f"by key from {entry['by']}"
            self.taken[entry["by"]] = self.taken.get(entry["by"], 0) + 1
        if self.say is not None:
            self.say(f"LEAN-ANSWER {k[:16]} {how}: {verdict(entry)}")
        return _loaded(entry)

    def summary(self) -> str:
        taken = sum(self.taken.values())
        froms = ", ".join(f"{n} from {by}" for by, n in sorted(self.taken.items()))
        return (f"{taken} of {taken + self.ran + self.unkeyed} runs by key{f' ({froms})' if froms else ''}, "
                f"{self.ran} ran" + (f", {self.unkeyed} unkeyed" if self.unkeyed else ""))


def window_runs(statements: Any, files: Any, raw: bytes, layout: dict, entries: list, window: str, units: list[int],
                steps: int, salt: str, draw: Any = None) -> tuple[list[tuple[str, PF.Proof, dict[str, Any]]],
                                                               tuple[str, str] | None]:
    """The runs `credit._verified` makes over a window's drawn statements, in its order (`credit._asks`): each its label,
    its proof files and `proofs.verify`'s keywords; and the refusal `_asks` stops at, or None."""
    asks, refused = CR._asks(statements, files, raw, layout, entries, window, units, steps, salt, draw)
    lanes = statements if isinstance(statements, dict) and sorted(statements) == list(CR.LANES_KEYS) else None
    listed = lanes["statements"] if lanes is not None else statements
    out = []
    for a in asks:
        s = listed[a["statement"]]
        if "leaf" in a:
            name = " ".join(str(x) for x in s["statement"] if x is not None)
            label = f"statement {a['statement']}: unit {a['unit']}'s {name}"
            program = bytes.fromhex(lanes["leaf_programs"][a["leaf"]])
        else:
            label, program = f"statement {a['statement']}: call {s['call']}", raw
        out.append((label, PF.Proof(Path(s["circuit"]), Path(s["public"]), tuple(map(Path, s["sessions"]))),
                    {"circuit_sha512": a["circuit_sha512"], "program": program, "partition": a["partition"],
                     "registered": a["registered"]}))
    return out, refused


def fill(answers: Answers, runs: Sequence[tuple[str, PF.Proof, dict[str, Any]]], j: int | None = None,
         say: Callable[[str], Any] = SAY) -> dict[str, int]:
    """Each of `runs` (`window_runs`') answered into `answers`' memo, J at a time (`jobs` of the setting `j`), a run the
    memo holds not run again: one ``LEAN`` line a verdict as each ends (how many have ended, the run's place among
    `runs`, its label, the verdict and its seconds). The counts of runs ``ran``, ``held`` (already in the memo) and
    ``unkeyed`` (a file missing: credit's own run meets it)."""
    answers.exe()
    j, sized = jobs(j)
    todo, held, unkeyed = [], 0, 0
    for i, (label, proof, kw) in enumerate(runs):
        try:
            k = answers.key(proof, **kw)
        except OSError:
            unkeyed += 1
            continue
        if answers.lookup(k) is not None:
            held += 1
        else:
            todo.append((i, label, k, proof, kw))
    say(f"LEAN {len(runs)} runs: {len(todo)} to run, {j} at a time, J {sized} ({held} already answered, "
        f"{unkeyed} unkeyed)")
    with cf.ThreadPoolExecutor(max_workers=j) as ex:
        futures = {ex.submit(answers.run, k, proof, **kw): (i, label) for i, label, k, proof, kw in todo}
        for n, f in enumerate(cf.as_completed(futures), 1):
            i, label = futures[f]
            try:
                entry = f.result()
            except Exception as e:  # noqa: BLE001 - not an answer credit reads: kept out, so credit's own run meets it
                say(f"LEAN {n}/{len(todo)} run {i + 1} of {len(runs)} ({label}): not kept, {type(e).__name__}: {e}"[:400])
                continue
            say(f"LEAN {n}/{len(todo)} run {i + 1} of {len(runs)} ({label}): {verdict(entry)} in {entry['seconds']} s")
    return {"ran": len(todo), "held": held, "unkeyed": unkeyed}

