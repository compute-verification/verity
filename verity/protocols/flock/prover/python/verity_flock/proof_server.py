"""SPOT's proof server: long-lived `flock-circuit` processes that prove sessions of statements of any circuits as they arrive.

* **Provers.** ``provers`` a device (default one): `prove --zk [--gpu] --stream -`, their GPU their
  `CUDA_VISIBLE_DEVICES`, or CPU provers when the server has no devices. Each pinned circuit (a block type) goes to one
  device at its first session, the caller's ``route`` or else the device with the fewest types so far, and each of its
  sessions to that device's prover with the fewest unfinished sessions, so each of a device's provers loads each of its
  types. A prover loads each circuit and builds each population's statement once, and its witness pipeline, statement
  cache and ``FC_VERIFY_AHEAD`` (``verify_ahead``, default 2) span every statement it proves. Provers that share the node
  (more than one in all, ``devices`` × ``provers``) split its threads evenly: each one's ``RAYON_NUM_THREADS`` is its share
  of the server's (the server's own ``RAYON_NUM_THREADS``, else rayon's default: the CPUs it may run on, within its cgroup's
  CPU quota), so each sizes its work, its host witness's lane groups a pass among it, for the cores it has rather than for
  every core.
* **Verifiers.** One a type and prover: `serve --zk --queue -`, its circuit loaded once (a templated statement's with its
  unit, `Statement.tmpl_unit`, which the prover's line names too), each session's public file and
  sample taken from the line `submit` writes for it, its record filed under the session's ``out`` (else the verifier's own
  directory). A verifier serves its queue in the order its connections arrive, so it takes one prover's sessions, and
  `submit` writes a session's two lines together.
* **The verdict of record.** A session's ``verdict`` (Lean `flock-verify verify --zk` on its record, say) runs on a pool
  of ``verdicts`` threads once both sides have recorded the session, off the provers' path.
* **The challenger** (``challenger=True``, FLOCK_CHALLENGER=1's): each verifier takes every session's coin tree from a
  challenger of its own (`verity_flock.challengers`, `serve --queue --tree-from`), whose session ends with the verifier's
  queue. Its audit holds every session the verifier served to the challenger's release log at once (no tree unused, none
  shared), so it runs at `close`, into ``challengers`` by verifier name, and a session counts only if its verifier's passed.

`submit` returns a `Session` at once: `Session.result` waits for both sides' records, `Session.verdict` for the verdict of
record. A process that exits ends its unfinished sessions as failures (`ProofServerError`) carrying its log's tail;
nothing is retried. A prover's exit also retires its types' verifiers once they have filed the sessions it finished
(their queues hold sessions it will never connect for), and the next session on its device or types starts fresh
processes. `close` (or leaving the ``with`` block) ends every queue and stream, waits for the processes and the verdicts,
and returns each process's exit status by name (a restarted process's name gets ``.2``, ``.3``, ..).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity_flock import challengers as CH

__all__ = ["ProofServer", "ProofServerError", "Session", "Statement"]


@dataclass(frozen=True)
class Statement:
    """A staged statement: its pinned circuit (``pin``, the circuit file's SHA-512 in hex, which names its type), the
    circuit's tables, the prover's instance file and the verifier's public file, for a circuit that declares links
    the verifier's links file, and for a templated statement (``verity/flock-circuit/tmpl``) its unit's rows at their
    positions (``tmpl_unit``, `flock-tmpl-unit`'s), which both sides load its circuit with."""

    pin: str
    circuit: Path
    tables: Path
    instances: Path
    public: Path
    links: Path | None = None
    tmpl_unit: Path | None = None


class ProofServerError(RuntimeError):
    """A session the server could not finish: a process it ran on exited, or the server closed, before its records."""


class Session:
    """One submitted session: ``label`` (unique on its server), its type (``pin``), ``device`` (None: a CPU prover) and
    ``prover`` (its prover's name); once done, ``live`` (the prover's ``LIVE`` record), ``served`` (the verifier's
    ``SESSION`` line), ``dir`` (its record's directory) and ``seconds`` from submission to both records; on a server with
    challengers, once it has closed, ``challenger`` (its verifier's challenger's exit and audit, `challengers.end`'s)."""

    def __init__(self, label: str, pin: str, device: int | None, verdict: Callable[[Session], Any] | None,
                 prover: str = ""):
        self.label, self.pin, self.device, self.prover = label, pin, device, prover
        self.live: dict | None = None
        self.served: dict | None = None
        self.dir: Path | None = None
        self.seconds: float | None = None
        self.challenger: dict | None = None
        self._home: Path | None = None
        self._t0 = time.monotonic()
        self._fn = verdict
        self._done, self._judged = threading.Event(), threading.Event()
        self._error: ProofServerError | None = None
        self._verdict: Any = None
        self._verdict_error: BaseException | None = None

    @property
    def accepted(self) -> bool:
        """Both sides accepted the session."""
        return bool((self.live or {}).get("accepted") is True and (self.served or {}).get("accepted") is True)

    def result(self, timeout: float | None = None) -> Session:
        """This session once both sides have recorded it; raises `ProofServerError` if it can't finish."""
        if not self._done.wait(timeout):
            raise TimeoutError(f"session {self.label}: no records yet")
        if self._error is not None:
            raise self._error
        return self

    def verdict(self, timeout: float | None = None) -> Any:
        """The verdict of record (None when the session was submitted without one), once it has run; re-raises its error."""
        self.result(timeout)
        if self._fn is None:
            return None
        if not self._judged.wait(timeout):
            raise TimeoutError(f"session {self.label}: no verdict of record yet")
        if self._verdict_error is not None:
            raise self._verdict_error
        return self._verdict

    def _fail(self, why: str) -> None:
        if not self._done.is_set():
            self._error = ProofServerError(f"session {self.label}: {why}")
            self._done.set()
            self._judged.set()


#: Where `_quota` reads the process's cgroup (``proc/self/cgroup``) and the cgroup filesystem (``sys/fs/cgroup``).
_ROOT = Path("/")


def _threads(env: Mapping[str, str], cpus: frozenset[int] | None = None) -> int:
    """The threads a server's provers share, as rayon sizes its pool: ``RAYON_NUM_THREADS`` in the server's environment
    when it is a positive count, else rayon's default, Rust's `available_parallelism`: the CPUs the provers may run on
    (``cpus``, else this process's affinity), at most its cgroups' CPU quota (`_quota`)."""
    v = env.get("RAYON_NUM_THREADS", "")
    if v.isdigit() and int(v) > 0:
        return int(v)
    n, q = len(os.sched_getaffinity(0) if cpus is None else cpus), _quota(_ROOT)
    return n if q is None else min(n, q)


#: The command that runs a process on a set of CPUs (`taskset -c LIST CMD...`, util-linux's).
TASKSET = "taskset"


def _cpulist(text: str) -> frozenset[int]:
    """A CPU list as taskset and `/sys/devices/system/cpu` write it: ``44-47``, ``44,45,46,47`` or ``0-3,8``."""
    cpus: set[int] = set()
    for part in text.split(","):
        lo, sep, hi = part.strip().partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()) or (sep and int(hi) < int(lo)):
            raise ValueError(f"not a CPU list: {text!r} (write it as taskset does: 44-47, or 44,45,46,47)")
        cpus.update(range(int(lo), int(hi if sep else lo) + 1))
    return frozenset(cpus)


def _cpus_text(cpus: frozenset[int]) -> str:
    """``cpus`` as a CPU list of ascending runs (``0-9,12``)."""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _placement(env: Mapping[str, str]) -> dict[str, frozenset[int]]:
    """``FC_VERIFIER_CPUS`` in the server's environment, a CPU list within its affinity: the verifiers' CPUs, and the
    provers' the server's others (by process kind); unset or empty, none (every process on every CPU of the server's)."""
    text = env.get("FC_VERIFIER_CPUS", "").strip()
    if not text:
        return {}
    mine, own = _cpulist(text), frozenset(os.sched_getaffinity(0))
    if not mine <= own:
        raise ValueError(f"FC_VERIFIER_CPUS={text}: CPUs {_cpus_text(mine - own)} are outside the server's affinity "
                         f"({_cpus_text(own)}); name CPUs within it, or start the server on more")
    if not own - mine:
        raise ValueError(f"FC_VERIFIER_CPUS={text}: every CPU of the server's ({_cpus_text(own)}), so none is left for the "
                         "provers")
    return {"verifier": mine, "prover": own - mine}


def _quota(root: Path) -> int | None:
    """The whole CPUs this process's cgroups allow, as Rust's `available_parallelism` reads them: the least quota over
    period, rounded down, of its own cgroup (``/proc/self/cgroup``; a v1 line with the cpu controller over the v2 line)
    and each one above it, from ``cpu.max`` (v2) or ``cpu.cfs_quota_us`` and ``cpu.cfs_period_us`` (v1, at the standard
    mounts ``cpu`` and ``cpu,cpuacct`` only, not Rust's search of other mounts); at least one, and None where no cgroup
    sets one. A cgroup without the file sets none; a file that is there but doesn't parse raises, where Rust skips it."""
    cgroup = root / "proc/self/cgroup"
    if not cgroup.is_file():
        return None
    own = None
    for f in (line.split(":", 2) for line in cgroup.read_text().splitlines()):
        if len(f) == 3 and (f[1] == "" and own is None or "cpu" in f[1].split(",")):
            own = (f[1] == "", f[2].lstrip("/"))
    if own is None:
        return None
    v2, path = own
    cg = root / "sys/fs/cgroup"
    if v2:
        mounts = [cg] if (cg / path / "cgroup.controllers").exists() else []
    else:
        mounts = [cg / "cpu", cg / "cpu,cpuacct"]
    mount = next((m for m in mounts if (m / path).is_dir()), None)
    if mount is None:
        return None
    quotas = []
    for d in [mount / path, *(mount / path).parents]:
        if d != mount and mount not in d.parents:
            break
        if v2 and (d / "cpu.max").is_file():
            q, p = (d / "cpu.max").read_text().split()[:2]
            if q != "max":
                quotas.append(int(q) // int(p))
        elif not v2 and (d / "cpu.cfs_quota_us").is_file():
            q, p = (int((d / f"cpu.cfs_{k}_us").read_text()) for k in ("quota", "period"))
            if q >= 0:
                quotas.append(q // p)
    return max(1, min(quotas)) if quotas else None


class _Process:
    """One long-lived `flock-circuit`, a prover (``key``: its device and index there) or a verifier (``key``: its type's
    pin and its prover's key): lines
    written to its stdin, its stdout read by a thread of its own and copied to its log beside its stderr."""

    def __init__(self, name: str, kind: str, key: Any, cmd: list[str], log: Path, env: dict[str, str],
                 on_line: Callable[[str, dict], None], on_exit: Callable[[_Process], None]):
        self.name, self.kind, self.key, self.log = name, kind, key, log
        self.addr: str | None = None
        self.started = threading.Event()
        self.sessions: set[str] = set()
        self.challenger: tuple[Path, subprocess.Popen] | None = None
        self.home: Path | None = None
        self._out = open(log, "w", buffering=1)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._out, text=True,
                                  bufsize=1, env=env, start_new_session=True)
        self._reader = threading.Thread(target=self._read, args=(on_line, on_exit), name=f"read-{name}", daemon=True)
        self._reader.start()

    def _read(self, on_line, on_exit) -> None:
        for line in self.p.stdout:
            self._out.write(line)
            line = line.rstrip("\n")
            if line.startswith("SERVING "):
                self.addr = line.split(" ", 1)[1].strip()
                self.started.set()
                continue
            tag, sep, body = line.partition("\t")
            if sep and tag in ("LIVE", "SESSION"):
                on_line(tag, json.loads(body))
        self.p.wait()
        self.started.set()
        on_exit(self)

    def send(self, obj: dict) -> None:
        self.p.stdin.write(json.dumps(obj, sort_keys=True) + "\n")
        self.p.stdin.flush()

    def end(self) -> None:
        if self.p.stdin and not self.p.stdin.closed:
            try:
                self.p.stdin.close()
            except BrokenPipeError as e:
                self._out.write(f"proof-server: {self.name} had closed its stdin before end() ({e}); nothing more sent\n")

    def join(self) -> int:
        self._reader.join()
        self._out.close()
        return self.p.returncode

    def tail(self, n: int = 1200) -> str:
        try:
            return self.log.read_text(errors="replace")[-n:]
        except OSError:
            return ""


class ProofServer:
    """SPOT's proof server on ``binary`` (a `flock-circuit` with `prove --stream` and `serve --queue`), its processes'
    logs and the verifiers' own records under ``out``. ``devices``: the GPUs to prove on (None: the CPU). ``provers``: the
    provers a device (or on the CPU). ``route``: a type's device by pin, for types the caller places itself. ``env``: added
    to every process's environment (a None value removes the variable). ``zk``: the masked protocol (`--zk`), SPOT's; off,
    the plain one. ``challenger``: each verifier on a challenger of its own (`--zk`, or ``FC_COINS=tree`` in ``env``).

    ``FC_VERIFIER_CPUS`` in the server's environment (a CPU list within its affinity, as taskset takes it) runs every
    verifier on those CPUs and every prover on the server's other CPUs (`taskset -c`), so a verifier's check of each
    session it serves takes no CPU from the provers, whose threads split their own CPUs; unset, every process runs on all
    the server's CPUs. A verifier's challenger keeps the server's."""

    def __init__(self, binary: str | os.PathLike, out: str | os.PathLike, *, devices: Sequence[int] | None = None,
                 provers: int = 1, verify_ahead: int = 2, verdicts: int = 4, route: Mapping[str, int] | None = None,
                 env: Mapping[str, str | None] | None = None, zk: bool = True, challenger: bool = False):
        if provers < 1:
            raise ValueError(f"provers must be at least 1, not {provers}")
        self.binary, self.out = str(binary), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.devices: list[int | None] = list(devices) if devices else [None]
        self.provers, self.verify_ahead, self.zk, self.challenger = provers, verify_ahead, zk, challenger
        self.challengers: dict[str, dict] = {}
        self._env = {k: v for k, v in {**os.environ, **(env or {})}.items() if v is not None}
        self._cpus = _placement(self._env)
        self._taskset = shutil.which(TASKSET) if self._cpus else None
        if self._cpus and self._taskset is None:
            raise ValueError(f"FC_VERIFIER_CPUS places processes with {TASKSET} (util-linux), which is not on PATH")
        n = len(self.devices) * provers
        share = _threads(self._env, self._cpus.get("prover")) // n
        self._share = {} if n == 1 else {"RAYON_NUM_THREADS": str(max(1, share))}
        self._routes: dict[str, int | None] = dict(route or {})
        self._lock, self._send, self._typing = threading.Lock(), threading.Lock(), threading.Lock()
        self._provers: dict[tuple[int | None, int], _Process] = {}
        self._verifiers: dict[tuple[str, tuple[int | None, int]], _Process] = {}
        self._started: list[_Process] = []
        self._retiring: list[threading.Thread] = []
        self._sessions: dict[str, Session] = {}
        self._pool = ThreadPoolExecutor(max(1, verdicts), thread_name_prefix="verdict")
        self._closed = False
        self.exits: dict[str, int] = {}

    def __enter__(self) -> ProofServer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def routes(self) -> dict[str, int | None]:
        """Each type's device so far, by pin."""
        return dict(self._routes)

    @property
    def verifier_cpus(self) -> str | None:
        """The verifiers' CPUs as ``FC_VERIFIER_CPUS`` placed them (a CPU list), or none when unplaced."""
        cpus = self._cpus.get("verifier")
        return _cpus_text(cpus) if cpus else None

    @property
    def pids(self) -> dict[str, int]:
        """Each process started so far, its pid by name (``prover-...``, ``verifier-...``)."""
        with self._lock:
            return {p.name: p.p.pid for p in self._started}

    def submit(self, st: Statement, draw: dict, label: str, *, out: str | os.PathLike | None = None,
               self_check: str | None = None, verdict: Callable[[Session], Any] | None = None) -> Session:
        """Proves ``st`` on the verifier's sample ``draw`` (a sample object over its population's units, served as
        `serve --draw-file` serves one) as session ``label``, its record filed under ``out`` (else the verifier's own
        directory). ``self_check``: ``"skip"`` for a negative control's prover. ``verdict``: the session's verdict of
        record, run on the session once it ends."""
        with self._lock:
            key = self._pick(self._route(st.pin))
        v = self._ensure_verifier(st, key)
        with self._lock:
            if self._closed:
                raise ProofServerError("the proof server is closed")
            if label in self._sessions:
                raise ValueError(f"session {label!r} was already submitted")
            p = self._prover(key)
            s = Session(label, st.pin, key[0], verdict, p.name)
            s._home = v.home
            self._sessions[label] = s
            v.sessions.add(label)
            p.sessions.add(label)
        # a new type's verifier loads its circuit before it serves; the other types' sessions go on meanwhile
        v.started.wait()
        if v.addr is None:
            s._fail(f"{v.name} exited before serving (rc {v.p.poll()}): {v.tail()}")
            return s
        q = {"instances": str(st.public), "draw": draw, "label": label}
        if st.links is not None:
            q["links"] = str(st.links)
        if out is not None:
            Path(out).mkdir(parents=True, exist_ok=True)
            q["out"] = str(out)
        line = {"circuit": str(st.circuit), "tables": str(st.tables), "instances": str(st.instances), "pin": st.pin,
                "verifier": v.addr, "label": label}
        if st.tmpl_unit is not None:
            line["tmpl_unit"] = str(st.tmpl_unit)
        if self_check is not None:
            line["self_check"] = self_check
        with self._send:
            try:
                v.send(q)
                p.send(line)
            except (BrokenPipeError, OSError, ValueError) as e:
                s._fail(f"its processes are gone ({e}): verifier {v.tail()!r}, prover {p.tail()!r}")
        return s

    def close(self) -> dict[str, int]:
        """Ends every queue and stream, waits for every process and verdict of record, fails any session left without its
        records, and returns each process's exit status by name."""
        with self._lock:
            self._closed = True
            provers, verifiers = list(self._provers.values()), list(self._verifiers.values())
        for p in [*provers, *verifiers]:
            p.end()
        for p in provers:
            p.join()
        # every prover has ended, so a verifier still holding a session no prover will connect for is retired too
        for v in verifiers:
            self._retire(v)
        for t in list(self._retiring):
            t.join()
        for p in self._started:
            self.exits[p.name] = p.join()
        self._pool.shutdown(wait=True)
        for s in self._sessions.values():
            s._fail("the proof server closed before its records")
        for v in self._started:
            if v.challenger is not None:
                mine = [self._sessions[x] for x in sorted(v.sessions)]
                got = self.challengers[v.name] = CH.end(*v.challenger, [str(s.dir) for s in mine if s.dir is not None])
                for s in mine:
                    s.challenger = got
        return dict(self.exits)

    # --- processes ---------------------------------------------------------------------------------------------------

    def _route(self, pin: str) -> int | None:
        if pin not in self._routes:
            load = {d: 0 for d in self.devices}
            for d in self._routes.values():
                if d in load:
                    load[d] += 1
            self._routes[pin] = min(self.devices, key=lambda d: load[d])
        return self._routes[pin]

    def _pick(self, dev: int | None) -> tuple[int | None, int]:
        """Device ``dev``'s prover with the fewest unfinished sessions (the first such), by its key."""
        def load(i: int) -> int:
            p = self._provers.get((dev, i))
            return 0 if p is None else sum(1 for x in p.sessions if not self._sessions[x]._done.is_set())
        return dev, min(range(self.provers), key=load)

    def _name(self, base: str, kind: str, key: Any) -> str:
        n = sum(1 for p in self._started if p.kind == kind and p.key == key)
        return base if n == 0 else f"{base}.{n + 1}"

    def _start(self, name: str, kind: str, key: Any, cmd: list[str], env: dict[str, str]) -> _Process:
        if cpus := self._cpus.get(kind):
            cmd = [self._taskset, "-c", _cpus_text(cpus), *cmd]
        p = _Process(name, kind, key, cmd, self.out / f"{name}.log", env, self._line, self._exited)
        self._started.append(p)
        return p

    def _ensure_verifier(self, st: Statement, key: tuple[int | None, int]) -> _Process:
        """``st``'s type's verifier for the prover ``key``, started (with its challenger) if it has none. One start at a
        time, outside the server's lock: a challenger is started only once it listens."""
        with self._typing:
            with self._lock:
                if self._closed:
                    raise ProofServerError("the proof server is closed")
                v = self._verifiers.get((st.pin, key))
                if v is not None:
                    return v
                base = f"verifier-{st.pin[:16]}" + (f"-{key[1]}" if self.provers > 1 else "")
                name = self._name(base, "verifier", (st.pin, key))
            ch = None
            if self.challenger:
                d = self.out / f"{name}-challenger"
                try:
                    p, flags = CH.start(d)
                except RuntimeError as e:
                    raise ProofServerError(f"{name}'s challenger: {e}") from e
                ch = (d, p)
            cmd = [self.binary, "serve", *(["--zk"] if self.zk else []), "--queue", "-", "--listen", "127.0.0.1:0",
                   "--out", str(self.out / base), "--circuit", str(st.circuit),
                   *([] if st.tmpl_unit is None else ["--tmpl-unit", str(st.tmpl_unit)]), "--tables", str(st.tables), "--pin", st.pin,
                   "--operator", "proof-server", "--concurrency", str(self.verify_ahead + 1), *(flags if ch else [])]
            with self._lock:
                v = self._verifiers[(st.pin, key)] = self._start(name, "verifier", (st.pin, key), cmd, self._env)
                v.challenger, v.home = ch, self.out / base
            return v

    def _prover(self, key: tuple[int | None, int]) -> _Process:
        p = self._provers.get(key)
        if p is None:
            dev = key[0]
            env = {**self._env, "FC_VERIFY_AHEAD": str(self.verify_ahead), **self._share}
            if dev is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(dev)
            cmd = [self.binary, "prove", *(["--zk"] if self.zk else []), *([] if dev is None else ["--gpu"]), "--stream", "-"]
            base = ("prover-cpu" if dev is None else f"prover-gpu{dev}") + (f"-{key[1]}" if self.provers > 1 else "")
            p = self._provers[key] = self._start(self._name(base, "prover", key), "prover", key, cmd, env)
        return p

    def _line(self, tag: str, rec: dict) -> None:
        with self._lock:
            s = self._sessions.get(rec.get("label"))
            if s is None or s._done.is_set():
                return
            if tag == "LIVE":
                s.live = rec
            else:
                s.served = rec
                s.dir = Path(rec.get("out") or s._home) / rec["session"]
            if s.live is None or s.served is None:
                return
            s.seconds = round(time.monotonic() - s._t0, 3)
            s._done.set()
            if s._fn is None:
                s._judged.set()
            else:
                self._pool.submit(self._judge, s)

    @staticmethod
    def _judge(s: Session) -> None:
        try:
            s._verdict = s._fn(s)
        except BaseException as e:  # noqa: BLE001 -- the caller's verdict fails its session's verdict(), not the server
            s._verdict_error = e
        finally:
            s._judged.set()

    def _exited(self, p: _Process) -> None:
        stale = []
        with self._lock:
            left = [s for x in p.sessions if not (s := self._sessions[x])._done.is_set()]
            mine = self._provers if p.kind == "prover" else self._verifiers
            if mine.get(p.key) is p:
                del mine[p.key]
            if p.kind == "prover":
                for k, v in list(self._verifiers.items()):
                    if k[1] == p.key and any(self._sessions[x].served is None for x in v.sessions):
                        del self._verifiers[k]
                        stale.append(v)
        for s in left:
            s._fail(f"{p.name} exited (rc {p.p.returncode}) before its records: {p.tail()}")
        for v in stale:
            t = threading.Thread(target=self._retire, args=(v,), name=f"retire-{v.name}", daemon=True)
            self._retiring.append(t)
            t.start()

    def _retire(self, v: _Process) -> None:
        """Ends verifier ``v`` once it has filed the sessions its prover finished: the rest of its queue never comes."""
        with self._lock:
            mine = [self._sessions[x] for x in v.sessions]
        for s in mine:
            if s.live is not None:
                s._done.wait()
        v.end()
        if v.p.poll() is None and any(s.served is None for s in mine):
            v.p.terminate()
