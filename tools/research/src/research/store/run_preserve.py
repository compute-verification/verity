"""A workload preserves one file of its out/ while it runs, through its runner."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

from ..telemetry.run import OUT_DIR
from ..telemetry.scope import claim
from . import kinds

DIR = Path("control") / "preserve"
REQUESTS, ANSWERS = DIR / "requests", DIR / "answers"
CLOSED, FULL = ANSWERS / "closed", ANSWERS / "full"
REQUEST_SCHEMA, ANSWER_SCHEMA = "research/run-preserve-request/v1", "research/run-preserve-answer/v1"
KIND = "run-file/v1"
PRESERVED, REFUSED, NOT_PRESERVED = "PRESERVED", "REFUSED", "NOT_PRESERVED"
MAX_BYTES = 1 << 20
MAX_REQUESTS = 64
REQUEST_MAX_BYTES = 64 << 10
PUT_DEADLINE_S = 60.0
TIMEOUT_S = 150.0
POLL_S = 0.25
LOG = "preserve.log"
STORE = "preserve-store"
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\.json")
MODULE = "research.store.run_preserve"

AT_END = "it stays in out/, which the run's custody records when the run ends"
NO_OFFER = ("this run's runner takes no request: it holds no custody key (a run without --on MACHINE, or one launched with "
            "--no-custody-r2; launch it with `research run --on MACHINE`, where custody is the default), or its workload has ended "
            "(ask before the workload exits)")
ENDED = ("the workload has ended, and its runner takes no request after that: ask before the workload exits; the run's custody "
         "records out/ now")
TOO_MANY = (f"this run has made {MAX_REQUESTS} requests, the most a run may: keep such records in one file and preserve it at the points "
            f"that matter; everything else in out/ the run's custody records when the run ends")
USE = "ask with `research run-preserve PATH`"


class Refusal(Exception):
    """A request the runner won't carry out; its text is the answer's reason, and says what to do instead."""


class NotPreserved(Exception):
    def __init__(self, what: str, errors: str):
        super().__init__(f"{what}: {errors}")
        self.what = what


def _say(msg: str) -> None:
    print(f"research run-preserve: {msg}", file=sys.stderr)


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _wait(run_dir: Path, rid: str, timeout: float) -> dict[str, Any]:
    answer, end = run_dir / ANSWERS / f"{rid}.json", time.monotonic() + timeout
    while True:
        marker = next((m for m in (CLOSED, FULL) if (run_dir / m).exists()), None)
        if (ans := _read(answer)) is not None:
            return ans
        if marker is not None:
            return _read(run_dir / marker) or {}
        if time.monotonic() >= end:
            return {"verdict": NOT_PRESERVED,
                    "reason": f"no answer from this run's runner within {timeout:g} s (it takes one request at a time and gives each put "
                              f"{PUT_DEADLINE_S:g} s), so nothing says the file is preserved; {AT_END}. Its answer, if one comes, will "
                              f"be {answer}"}
        time.sleep(POLL_S)


def main(argv: list[str], environ: Mapping[str, str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="research run-preserve", allow_abbrev=False,
        description="From a workload `research run --on MACHINE` started: have the run's runner put one file of $OUT in the "
                    "evidence store now, verified on its remote, instead of when the run ends. Prints the art id (exit 0), or "
                    "says why not (exit 1).")
    ap.add_argument("path", metavar="PATH", help=f"a regular file under $OUT (the run's out/), at most {MAX_BYTES} bytes")
    ap.add_argument("--kind", default=KIND, help=f"the artifact's kind (default {KIND})")
    ap.add_argument("--meta", default=None, help="the artifact's meta, a JSON object or @file; the runner sets its run_id and path")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, metavar="S",
                    help=f"seconds to wait for the runner's answer (default {TIMEOUT_S:g}); at most {MAX_REQUESTS} requests a run")
    ns = ap.parse_args(argv)
    environ = os.environ if environ is None else environ
    try:
        from .cli import _meta

        meta = _meta(ns.meta)
    except (OSError, ValueError) as e:
        ap.error(f"--meta: {e}")
    if not kinds.is_kind(ns.kind):
        ap.error(f"--kind {ns.kind!r} is not a kind (<name>/v<int>, as {KIND})")
    if not ns.timeout > 0:
        ap.error("--timeout takes a positive number of seconds")
    doc = {"schema": REQUEST_SCHEMA, "path": os.path.join(os.getcwd(), ns.path), "kind": ns.kind, "meta": meta}
    if len(json.dumps(doc)) > REQUEST_MAX_BYTES:
        ap.error(f"the request is over {REQUEST_MAX_BYTES} bytes: put less in --meta")
    if not (run_dir := environ.get("RESEARCH_RUN_DIR", "")):
        _say("RESEARCH_RUN_DIR is not set: this is for a workload that `research run` started")
        return 1
    d = Path(run_dir)
    if not (d / REQUESTS).is_dir():
        _say(f"NOT_PRESERVED: {NO_OFFER}. {ns.path} stays where it is")
        return 1
    if not kinds.is_known(ns.kind):
        _say(f"unknown kind {ns.kind!r} (store/kinds.py lists the known ones); asking anyway")
    rid = f"{os.getpid()}-{secrets.token_hex(4)}"
    ans = next((_read(d / m) for m in (CLOSED, FULL) if (d / m).exists()), None)
    if ans is None:
        try:
            claim(d / REQUESTS / f"{rid}.json", doc, mode=0o644)
        except OSError as e:
            _say(f"NOT_PRESERVED: can't write the request under {d / REQUESTS} ({e.strerror}); {ns.path} stays where it is")
            return 1
        ans = _wait(d, rid, ns.timeout)
    if ans.get("verdict") == PRESERVED:
        print(ans["art"])
        _say(f"PRESERVED {ans['path']}: {ans['bytes']} bytes, sha256 {ans['sha256']}, verified on the store's remote")
        return 0
    _say(f"{ans.get('verdict') or NOT_PRESERVED}: {ans.get('reason') or 'the answer gives no reason'}")
    return 1


class Preserver:
    def __init__(self, run_dir: Path | str, custody_dir: Path | str, *, shared: bool = False,
                 popen: Callable[..., Any] = subprocess.Popen):
        self.run_dir, self.custody_dir, self.shared, self.popen = Path(run_dir), Path(custody_dir), shared, popen
        self.env = dict(os.environ)
        self.seen: set[str] = set()
        self.child: tuple[str, Any, float] | None = None
        self.full = self.closed = False
        for d in (ANSWERS, REQUESTS):
            (self.run_dir / d).mkdir(parents=True, exist_ok=True)
        if shared:
            for d, mode in ((DIR.parent, 0o755), (DIR, 0o755), (ANSWERS, 0o755), (REQUESTS, 0o1777)):
                os.chmod(self.run_dir / d, mode)

    def _pending(self) -> list[str]:
        d = self.run_dir / REQUESTS
        if not d.is_dir():
            return []
        with os.scandir(d) as entries:
            return sorted(e.name for e in entries if not e.name.startswith(".") and e.name not in self.seen)

    def _answer(self, name: str, verdict: str, reason: str) -> None:
        path = self.run_dir / ANSWERS / name
        claim(path, {"schema": ANSWER_SCHEMA, "request": name.removesuffix(".json"), "verdict": verdict, "reason": reason}, mode=0o644)
        ans = _read(path) or {}
        said = [ans.get("verdict"), ans.get("path"), ans.get("art") or str(ans.get("reason"))[:200]]
        print(f"research: run-preserve {name.removesuffix('.json')}: {' '.join(str(x) for x in said if x)}", file=sys.stderr)

    def _start(self, name: str) -> None:
        try:
            log = os.open(self.custody_dir / LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                proc = self.popen([sys.executable, "-m", MODULE, str(self.run_dir), str(self.custody_dir), name],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=self.env)
            finally:
                os.close(log)
        except OSError as e:
            self._answer(name, NOT_PRESERVED, f"the runner could not start its put ({e.strerror or type(e).__name__}); {AT_END}")
            return
        self.child = (name, proc, time.monotonic() + PUT_DEADLINE_S)

    def _reap(self, wait: bool = False) -> bool:
        if self.child is None:
            return True
        name, proc, end = self.child
        if wait and proc.poll() is None:
            try:
                proc.wait(timeout=max(0.0, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                print(f"research: run-preserve {name.removesuffix('.json')}: still putting at its deadline", file=sys.stderr)
        if (rc := proc.poll()) is None:
            if not wait and time.monotonic() < end:
                return False
            proc.kill()
            proc.wait()
            why = (f"the runner's put did not finish within {PUT_DEADLINE_S:g} s and was stopped (the store's remote is slow or out "
                   f"of reach from this machine): ask again, or stop if the workload must not go on without the record; {AT_END}")
        else:
            why = (f"the runner's put ended with status {rc} and no answer; {AT_END}. Its output is in the runner's {LOG}, beside "
                   "the run's custody key")
        self.child = None
        self._answer(name, NOT_PRESERVED, why)
        return True

    def poll(self) -> None:
        if self.closed or not self._reap() or self.full:
            return
        for name in self._pending()[:1]:
            if len(self.seen) >= MAX_REQUESTS:
                self.full = True
                claim(self.run_dir / FULL, {"schema": ANSWER_SCHEMA, "verdict": REFUSED, "reason": TOO_MANY}, mode=0o644)
                print(f"research: run-preserve: {MAX_REQUESTS} requests taken, the most a run may make; no more are read", file=sys.stderr)
                return
            self.seen.add(name)
            if NAME.fullmatch(name):
                self._start(name)
            else:
                print(f"research: run-preserve: {name!r} under {REQUESTS} is not a request's name, and counts as one; {USE}", file=sys.stderr)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._reap(wait=True)
            if self.seen or not self._retract():
                claim(self.run_dir / CLOSED, {"schema": ANSWER_SCHEMA, "verdict": REFUSED, "reason": ENDED}, mode=0o644)
                for name in self._pending()[: max(0, MAX_REQUESTS - len(self.seen))]:
                    self.seen.add(name)
                    if NAME.fullmatch(name):
                        self._answer(name, REFUSED, ENDED)
                if self.shared:
                    os.chmod(self.run_dir / REQUESTS, 0o755)
        except OSError as e:
            print(f"research: run-preserve: closing {self.run_dir / DIR}: {e}", file=sys.stderr)
        shutil.rmtree(self.custody_dir / STORE, ignore_errors=True)

    def _retract(self) -> bool:
        try:
            os.rmdir(self.run_dir / REQUESTS)
        except OSError:
            return False
        for d in (ANSWERS, DIR):
            os.rmdir(self.run_dir / d)
        return True


def _request(path: Path) -> dict[str, Any]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        raise Refusal(f"the runner can't read the request ({e.strerror}); {USE}") from None
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > REQUEST_MAX_BYTES:
            raise Refusal(f"the request is not a regular file of at most {REQUEST_MAX_BYTES} bytes; {USE}")
        raw = f.read(REQUEST_MAX_BYTES + 1)
    try:
        doc = json.loads(raw)
    except ValueError:
        raise Refusal(f"the request is not JSON; {USE}") from None
    if not (isinstance(doc, dict) and doc.get("schema") == REQUEST_SCHEMA and isinstance(doc.get("path"), str)
            and isinstance(doc.get("meta"), dict) and kinds.is_kind(doc.get("kind"))):
        raise Refusal(f"the request is not a {REQUEST_SCHEMA} (schema, path, kind, meta); {USE}")
    return doc


def _open(run_dir: Path, path: str) -> tuple[int, str]:
    out = os.path.join(os.path.realpath(run_dir), OUT_DIR)
    real = os.path.realpath(path) if os.path.isabs(path) and "\0" not in path else ""
    if not real.startswith(out + os.sep):
        raise Refusal(f"{path!r} is not inside this run's out/ ({out})" + (f": it resolves to {real}" if real not in ("", path) else "")
                      + ". The runner preserves only what the run's own record will hold: write the file under $OUT and name it there")
    parts = Path(real).relative_to(out).parts
    rel = "/".join((OUT_DIR, *parts))
    fd = -1
    try:
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            nxt = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | (os.O_NONBLOCK if last else os.O_DIRECTORY), dir_fd=fd)
            os.close(fd)
            fd = nxt
        st = os.fstat(fd)
    except OSError as e:
        if fd >= 0:
            os.close(fd)
        fix = {errno.ENOENT: "write the file and close it, then ask",
               errno.EACCES: "make the file readable and its directories searchable for the runner's user (chmod a+r FILE, a+rx DIR)",
               errno.ELOOP: "it became a link after its path was checked; ask again once the path is settled",
               errno.ENOTDIR: "a directory on its path is not one, or became a link after the path was checked; name a regular "
                              "file under $OUT once the path is settled"}
        raise Refusal(f"the runner can't open {rel} ({e.strerror}): {fix.get(e.errno, 'name a regular file under $OUT')}") from None
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise Refusal(f"{rel} is not a regular file: name one file. A directory goes as one file (tar it), or is left to the run's "
                      "custody, which records all of out/ when the run ends")
    if st.st_size > MAX_BYTES:
        os.close(fd)
        raise Refusal(_too_big(rel, f"{st.st_size} bytes"))
    return fd, rel


def _too_big(rel: str, size: str) -> str:
    return (f"{rel} is {size}, over the {MAX_BYTES} bytes ({MAX_BYTES >> 20} MiB) a file preserved while the run goes on may have: "
            f"preserve the small record that can't wait (a digest, a coin, a verdict) and leave the rest in out/, which the run's "
            f"custody records when the run ends")


def _copy(src: IO[bytes], into: Path, rel: str) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(into, "wb") as dst:
        while chunk := src.read(1 << 16):
            n += len(chunk)
            if n > MAX_BYTES:
                raise Refusal(_too_big(rel, "still growing"))
            h.update(chunk)
            dst.write(chunk)
    return h.hexdigest(), n


def _warn(msg: str) -> None:
    print(f"research run-preserve: {msg}", file=sys.stderr)


def _withdrawn(st: Any, run_id: str, rel: str) -> str | None:
    from . import withdraw

    for path, art, w in withdraw.refused(st, run_id, [rel, Path(rel).name], _warn):
        if path == rel or (st.get_manifest(art).meta or {}).get("path") == rel:
            return (f"{rel} was withdrawn from this run's {art} on purpose ({w['reason']!r}, by {w['by']}"
                    + (f", ref {w['ref']}" if w["ref"] else "") + "), and nothing puts it back in the shared store: delete the file, "
                    "or write what may be shared under another name and preserve that")
    return None


def _preserve(run_dir: Path, custody_dir: Path, req: dict[str, Any]) -> dict[str, Any]:
    from . import custody as CU
    from . import preserved
    from .local import LocalStore

    fd, rel = _open(run_dir, req["path"])
    run_id, root = CU.run_id_of(run_dir), custody_dir / STORE
    tmp = root / "scratch" / f".preserve-{run_id}-{secrets.token_hex(4)}"
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        tmp.mkdir(parents=True)
    except OSError:
        os.close(fd)
        raise
    try:
        staged = tmp / Path(rel).name
        with os.fdopen(fd, "rb") as src:
            sha, n = _copy(src, staged, rel)
        with CU.custody_env(custody_dir):
            st = LocalStore(root)
            if why := _withdrawn(st, run_id, rel):
                raise Refusal(why)
            art = st.put_artifact(req["kind"], {**req["meta"], "run_id": run_id, "path": rel}, file=staged)
            rep = st.push(art, preserve=True)
            if not rep.get("preserved"):
                raise NotPreserved("the runner's put did not reach the store's remote", f"push {art}: {rep.get('errors')}")
            chk = preserved.check(st, [art], mode="head")
            if not chk["preserved"]:
                raise NotPreserved("the store's remote does not hold it after the runner's put",
                                   f"check {art}: {chk.get('not_preserved') or chk.get('reason')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"verdict": PRESERVED, "art": art, "bytes": n, "sha256": sha, "path": rel, "kind": req["kind"], "run_id": run_id}


def _failed(custody_dir: Path, e: Exception) -> str:
    """The kind of failure, never its text, which can name the remote."""
    from . import custody as CU

    expires = CU._key_expiry(custody_dir)
    if CU._expired(expires):
        return (f"the run's custody key expired at {expires} (the run outlived --custody-ttl), so its runner can put nothing more: "
                f"launch the next run with a longer --custody-ttl. The file stays in out/; `research data custody RUN --publish` "
                f"with a fresh key records the run once it has ended")
    what = e.what if isinstance(e, NotPreserved) else f"the runner could not put it on the store's remote ({type(e).__name__})"
    return (f"{what}: ask again, or stop if the workload must not go on without the record; {AT_END}. The error is in the runner's "
            f"{LOG}, beside the run's custody key")


def serve(run_dir: Path | str, custody_dir: Path | str, name: str) -> dict[str, Any]:
    d = Path(run_dir)
    try:
        ans = _preserve(d, Path(custody_dir), _request(d / REQUESTS / name))
    except Refusal as e:
        ans = {"verdict": REFUSED, "reason": str(e)}
    except Exception as e:  # noqa: BLE001  whatever the store raised: the workload gets an answer, the log the error
        traceback.print_exc()
        ans = {"verdict": NOT_PRESERVED, "reason": _failed(Path(custody_dir), e)}
    claim(d / ANSWERS / name, {"schema": ANSWER_SCHEMA, "request": name.removesuffix(".json"), **ans}, mode=0o644)
    print(f"research run-preserve: {name}: {json.dumps(ans, sort_keys=True)}", file=sys.stderr)
    return ans


if __name__ == "__main__":
    serve(sys.argv[1], sys.argv[2], sys.argv[3])
