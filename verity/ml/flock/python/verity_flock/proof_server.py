"""SPOT's proof server: long-lived `flock-circuit` processes that prove sessions of statements of any circuits as they arrive.

* **Provers.** One a device: `prove --zk [--gpu] --stream -`, its GPU its `CUDA_VISIBLE_DEVICES`, or one CPU prover when
  the server has no devices. Each pinned circuit (a block type) goes to one device at its first session, the caller's
  ``route`` or else the device with the fewest types so far. A prover loads each circuit and builds each population's
  statement once, and its witness pipeline, statement cache and ``FC_VERIFY_AHEAD`` (``verify_ahead``, default 2) span
  every statement it proves.
* **Verifiers.** One a type: `serve --zk --queue -`, its circuit loaded once, each session's public file and sample
  taken from the line `submit` writes for it, its record filed under the session's ``out`` (else the verifier's own
  directory).
* **The verdict of record.** A session's ``verdict`` (Lean `flock-verify verify --zk` on its record, say) runs on a pool
  of ``verdicts`` threads once both sides have recorded the session, off the provers' path.

`submit` returns a `Session` at once: `Session.result` waits for both sides' records, `Session.verdict` for the verdict of
record. A process that exits ends its unfinished sessions as failures (`ProofServerError`) carrying its log's tail;
nothing is retried. `close` (or leaving the ``with`` block) ends every queue and stream, waits for the processes and the
verdicts, and returns each process's exit status.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = ["ProofServer", "ProofServerError", "Session", "Statement"]


@dataclass(frozen=True)
class Statement:
    """A staged statement: its pinned circuit (``pin``, the circuit file's SHA-512 in hex, which names its type), the
    circuit's tables, the prover's instance file and the verifier's public file, and for a circuit that declares links
    the verifier's links file."""

    pin: str
    circuit: Path
    tables: Path
    instances: Path
    public: Path
    links: Path | None = None


class ProofServerError(RuntimeError):
    """A session the server could not finish: a process it ran on exited, or the server closed, before its records."""


class Session:
    """One submitted session: ``label`` (unique on its server), its type (``pin``) and ``device`` (None: the CPU prover);
    once done, ``live`` (the prover's ``LIVE`` record), ``served`` (the verifier's ``SESSION`` line), ``dir`` (its record's
    directory) and ``seconds`` from submission to both records."""

    def __init__(self, label: str, pin: str, device: int | None, verdict: Callable[[Session], Any] | None):
        self.label, self.pin, self.device = label, pin, device
        self.live: dict | None = None
        self.served: dict | None = None
        self.dir: Path | None = None
        self.seconds: float | None = None
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


class _Process:
    """One long-lived `flock-circuit`: lines written to its stdin, its stdout read by a thread of its own and copied to
    its log beside its stderr."""

    def __init__(self, name: str, cmd: list[str], log: Path, env: dict[str, str], on_line: Callable[[str, dict], None],
                 on_exit: Callable[[_Process], None]):
        self.name, self.log = name, log
        self.addr: str | None = None
        self.started = threading.Event()
        self.sessions: set[str] = set()
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
            except BrokenPipeError:
                pass

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
    logs and the verifiers' own records under ``out``. ``devices``: the GPUs to prove on (None: one CPU prover).
    ``route``: a type's device by pin, for types the caller places itself. ``env``: added to every process's environment
    (a None value removes the variable). ``zk``: the masked protocol (`--zk`), SPOT's; off, the plain one."""

    def __init__(self, binary: str | os.PathLike, out: str | os.PathLike, *, devices: Sequence[int] | None = None,
                 verify_ahead: int = 2, verdicts: int = 4, route: Mapping[str, int] | None = None,
                 env: Mapping[str, str | None] | None = None, zk: bool = True):
        self.binary, self.out = str(binary), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.devices: list[int | None] = list(devices) if devices else [None]
        self.verify_ahead, self.zk = verify_ahead, zk
        self._env = {k: v for k, v in {**os.environ, **(env or {})}.items() if v is not None}
        self._routes: dict[str, int | None] = dict(route or {})
        self._lock, self._send = threading.Lock(), threading.Lock()
        self._provers: dict[int | None, _Process] = {}
        self._verifiers: dict[str, _Process] = {}
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

    def submit(self, st: Statement, draw: dict, label: str, *, out: str | os.PathLike | None = None,
               self_check: str | None = None, verdict: Callable[[Session], Any] | None = None) -> Session:
        """Proves ``st`` on the verifier's sample ``draw`` (a sample object over its population's units, served as
        `serve --draw-file` serves one) as session ``label``, its record filed under ``out`` (else the verifier's own
        directory). ``self_check``: ``"skip"`` for a negative control's prover. ``verdict``: the session's verdict of
        record, run on the session once it ends."""
        with self._lock:
            if self._closed:
                raise ProofServerError("the proof server is closed")
            if label in self._sessions:
                raise ValueError(f"session {label!r} was already submitted")
            dev = self._route(st.pin)
            v = self._verifier(st)
            p = self._prover(dev)
            s = Session(label, st.pin, dev, verdict)
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
        if self_check is not None:
            line["self_check"] = self_check
        # a verifier serves its queue in the order its prover connects, so each session's two lines go out together
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
            provers, verifiers = list(self._provers.values()), dict(self._verifiers)
        for p in [*provers, *verifiers.values()]:
            p.end()
        for p in provers:
            self.exits[p.name] = p.join()
        # every prover has ended: a verifier still holding a session no prover will connect for would wait for it forever,
        # so once it has filed the sessions the provers finished it is stopped
        with self._lock:
            unfiled = [s for s in self._sessions.values() if s.served is None]
        for pin, v in verifiers.items():
            mine = [s for s in unfiled if s.pin == pin]
            for s in mine:
                if s.live is not None:
                    s._done.wait()
            if any(s.served is None for s in mine) and v.p.poll() is None:
                v.p.terminate()
            self.exits[v.name] = v.join()
        self._pool.shutdown(wait=True)
        for s in self._sessions.values():
            s._fail("the proof server closed before its records")
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

    def _verifier(self, st: Statement) -> _Process:
        v = self._verifiers.get(st.pin)
        if v is None:
            name = f"verifier-{st.pin[:16]}"
            cmd = [self.binary, "serve", *(["--zk"] if self.zk else []), "--queue", "-", "--listen", "127.0.0.1:0",
                   "--out", str(self.out / name), "--circuit", str(st.circuit), "--tables", str(st.tables), "--pin", st.pin,
                   "--operator", "proof-server", "--concurrency", str(self.verify_ahead + 1)]
            v = self._verifiers[st.pin] = _Process(name, cmd, self.out / f"{name}.log", self._env, self._line, self._exited)
        return v

    def _prover(self, dev: int | None) -> _Process:
        p = self._provers.get(dev)
        if p is None:
            name = "prover-cpu" if dev is None else f"prover-gpu{dev}"
            env = {**self._env, "FC_VERIFY_AHEAD": str(self.verify_ahead)}
            if dev is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(dev)
            cmd = [self.binary, "prove", *(["--zk"] if self.zk else []), *([] if dev is None else ["--gpu"]), "--stream", "-"]
            p = self._provers[dev] = _Process(name, cmd, self.out / f"{name}.log", env, self._line, self._exited)
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
                v = self._verifiers[s.pin]
                s.dir = Path(rec.get("out") or self.out / v.name) / rec["session"]
            if s.live is None or s.served is None:
                return
            s.seconds = round(time.monotonic() - s._t0, 3)
            s._done.set()
            self._verifiers[s.pin].sessions.discard(s.label)
            self._provers[s.device].sessions.discard(s.label)
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
        with self._lock:
            left = [self._sessions[x] for x in p.sessions]
            p.sessions.clear()
        for s in left:
            s._fail(f"{p.name} exited (rc {p.p.returncode}) before its records: {p.tail()}")
