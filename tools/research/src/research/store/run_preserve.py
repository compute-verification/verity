"""A workload preserves one file of its out/ while it runs: `research run-preserve PATH`, which its runner answers.

A machine holds no store write key, and the custody key `research run --on` mints is the runner's alone (store.custody), so
a workload puts nothing in the store itself, and the run's custody uploads out/ only once the workload has exited.  For a
record that must not have its only copy on the machine until then, the workload asks and the runner puts; no key or remote
address reaches the workload's environment or a file it can read:

    workload   `research run-preserve PATH [--kind K] [--meta JSON] [--timeout S]` claims a request under
               <run dir>/control/preserve/requests/, waits for the answer of the same name under answers/, prints the art id and
               exits 0 when it says PRESERVED; anything else it says on stderr and exits 1, within --timeout
    runner     `Preserver.poll`, once on each pass of the loop that waits for the workload (telemetry.run): it takes one request
               at a time and starts a child of its own for it (`serve`), which checks the path, copies the file, reads the
               custody key (`custody.custody_env`), puts the copy as an artifact carrying the run id, pushes it, verifies it
               as custody verifies a run (`preserved.check`, head mode) and claims the answer: PRESERVED with the art id, bytes
               and sha256, else REFUSED or NOT_PRESERVED with the reason and what to do about it

A child and not the loop itself, because a remote that stalls would hold the loop that ends the workload on its timeout or
a cancel: the runner stops a child at PUT_DEADLINE_S and answers for it.  The key is read only in that child, once the path
is accepted.  Whatever the child writes that can name the remote stays in the custody dir, beside the key: its output (a
store error) in preserve.log, and the store it puts from (whose index records where each artifact went) in preserve-store/,
which the runner deletes once the workload has ended.  Its one file in the run dir is the answer.

The file stays in out/, so the run's record at the end holds it too, by the same blob.  The answers are the runner's own
files in that record (custody.RUN_FILE_GLOBS); a request is the workload's, like any other file it writes outside out/;
and a run that made no request keeps nothing of this (`Preserver.close`).  Under --job-user the job writes into requests/
and nowhere else here, so an answer is always the runner's, and the runner reads a request and the file it names as files
it doesn't trust: no link followed, a regular file, a bounded read.
"""
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

DIR = Path("control") / "preserve"                   # run-dir relative, the runner's: made only by a runner that holds a custody key
REQUESTS, ANSWERS = DIR / "requests", DIR / "answers"
#: Claimed by the runner when it takes no more requests, each holding the refusal a request without an answer gets: the
#: workload has ended, or the run has made MAX_REQUESTS.  Beside the answers: all that the runner writes here is in one place.
CLOSED, FULL = ANSWERS / "closed", ANSWERS / "full"
REQUEST_SCHEMA, ANSWER_SCHEMA = "research/run-preserve-request/v1", "research/run-preserve-answer/v1"
KIND = "run-file/v1"
PRESERVED, REFUSED, NOT_PRESERVED = "PRESERVED", "REFUSED", "NOT_PRESERVED"
#: The largest file a request may name.  A record that can't wait for the run's end (a coin, a digest, a verdict) is a few
#: KiB; this is hundreds of them, still one conditional PUT (no multipart) that takes about a second beside the workload it
#: must not disturb, and anything bigger is a result, which the run's custody uploads from out/ when the run ends.
MAX_BYTES = 1 << 20
#: The most requests a run may make: what a workload that asks in a loop can make its runner upload while it runs (this
#: many files of MAX_BYTES) and add to its record (two small files a request).
MAX_REQUESTS = 64
REQUEST_MAX_BYTES = 64 << 10                         # a request file, its --meta included
#: How long the runner gives one put before it stops it and answers NOT_PRESERVED: a healthy remote takes about a second.
PUT_DEADLINE_S = 60.0
#: How long the command waits for its answer by default: a put ahead of it and its own, each to its deadline.
TIMEOUT_S = 150.0
POLL_S = 0.25                                        # the command's wait between looks, the runner loop's own period
LOG = "preserve.log"                                 # in the custody dir: the children's output, which may name the remote
STORE = "preserve-store"                             # in the custody dir: the store the children put from, the run's alone
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
    """The remote isn't known to hold the artifact.  `what` is for the answer; the text, with the store's errors, is for the
    runner's log only."""

    def __init__(self, what: str, errors: str):
        super().__init__(f"{what}: {errors}")
        self.what = what


# ---- the workload's side ------------------------------------------------------------------------------------------------------------------

def _say(msg: str) -> None:
    print(f"research run-preserve: {msg}", file=sys.stderr)


def _read(path: Path) -> dict[str, Any] | None:
    """A claimed file's document once it is there (`claim` creates it whole)."""
    return json.loads(path.read_text()) if path.exists() else None


def _wait(run_dir: Path, rid: str, timeout: float) -> dict[str, Any]:
    """The answer to request `rid`: the runner's, else the refusal of the marker that says it takes no more, else, once `timeout`
    has passed, NOT_PRESERVED."""
    answer, end = run_dir / ANSWERS / f"{rid}.json", time.monotonic() + timeout
    while True:
        # the marker first: the runner answers what it took before it claims one, so a request with neither has no answer coming
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
        from .cli import _meta  # `research data put`'s reading of --meta

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
    # taken no more: say so without leaving a request nothing will answer
    ans = next((_read(d / m) for m in (CLOSED, FULL) if (d / m).exists()), None)
    if ans is None:
        try:
            claim(d / REQUESTS / f"{rid}.json", doc, mode=0o644)         # the runner may be another user (--job-user)
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


# ---- the runner's side: the loop ---------------------------------------------------------------------------------------------------------

class Preserver:
    """The runner's side for one run dir.  Making it is the offer (the directories a workload's command looks for), so only a
    runner that holds a custody key makes one; `poll` on each pass of the runner's loop; `close` once the workload has ended
    and before anything takes the run's record.  `shared`: the workload is another user (--job-user), who may write requests
    and read answers."""

    def __init__(self, run_dir: Path | str, custody_dir: Path | str, *, shared: bool = False,
                 popen: Callable[..., Any] = subprocess.Popen):
        self.run_dir, self.custody_dir, self.shared, self.popen = Path(run_dir), Path(custody_dir), shared, popen
        self.env = dict(os.environ)                  # the runner's own, as it is before the workload's --env is set for the spawn
        self.seen: set[str] = set()
        self.child: tuple[str, Any, float] | None = None       # the request in flight, its process and when it is stopped
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
        """Answer for a request no child answered (`claim` creates once: a child's own answer stands), and say so in the log."""
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
        """Whether no put is in flight now.  One that has exited is answered for if it left no answer, and so is one stopped at its
        deadline; with `wait`, after waiting for it until then."""
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
        """One pass of the runner's loop: answer for the put in flight once it is over, else take the next request."""
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
        """The workload has ended: the put in flight gets the rest of its deadline, each request not taken is refused (up to the
        run's MAX_REQUESTS answers), none is read after, and the store the puts went through is deleted.  A run that made no
        request keeps nothing of the offer."""
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
        except OSError as e:            # the run's custody comes next, whatever became of this directory
            print(f"research: run-preserve: closing {self.run_dir / DIR}: {e}", file=sys.stderr)
        shutil.rmtree(self.custody_dir / STORE, ignore_errors=True)

    def _retract(self) -> bool:
        """Remove the offer's directories; False when a request is there after all."""
        try:
            os.rmdir(self.run_dir / REQUESTS)
        except OSError:
            return False
        for d in (ANSWERS, DIR):
            os.rmdir(self.run_dir / d)
        return True


# ---- the runner's side: one request, in a child -------------------------------------------------------------------------------------------

def _request(path: Path) -> dict[str, Any]:
    """The request at `path`, read as a file the runner doesn't trust (under --job-user the job wrote it)."""
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
    """(an open descriptor of the regular file `path` names inside the run's out/, its path relative to the run dir).  Links and
    `..` are resolved first, and what they resolve to must be under out/; that path is then opened one component at a time
    following no link, so one swapped for a link after the check is refused, not read."""
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
    """Copy `src` to `into`, at most MAX_BYTES (a file that grew past it since it was opened is refused); (sha256, bytes)."""
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
    """The refusal when a `withdrawn` label of one of the run's artifacts covers `rel` (`withdraw.refused`, as `research data put
    --tree` asks): a tree of the run names the file by its run-dir path, a file this command preserved earlier by its meta."""
    from . import withdraw

    for path, art, w in withdraw.refused(st, run_id, [rel, Path(rel).name], _warn):
        if path == rel or (st.get_manifest(art).meta or {}).get("path") == rel:
            return (f"{rel} was withdrawn from this run's {art} on purpose ({w['reason']!r}, by {w['by']}"
                    + (f", ref {w['ref']}" if w["ref"] else "") + "), and nothing puts it back in the shared store: delete the file, "
                    "or write what may be shared under another name and preserve that")
    return None


def _preserve(run_dir: Path, custody_dir: Path, req: dict[str, Any]) -> dict[str, Any]:
    """The PRESERVED answer for `req`, once the remote was checked to hold its artifact; else Refusal, or whatever the store
    raised."""
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
        staged = tmp / Path(rel).name                # the runner's own copy: what is hashed is what is put
        with os.fdopen(fd, "rb") as src:
            sha, n = _copy(src, staged, rel)
        with CU.custody_env(custody_dir):            # the key, in this process's environment alone, for the store calls below
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
    """The answer's reason for a put that failed: the kind of failure, never its text, which can name the remote."""
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
    """Decide the request `name` of `run_dir` and claim its answer, which is returned.  PRESERVED only once the remote was
    checked to hold the artifact; a failure's text goes to stderr (the runner's LOG), its kind into the answer."""
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
