"""The asker of a run in a pool's line: while the run waits there, whatever it waits on, it asks `auto` again, and when a
place has stood on another machine's pool for STAND_MIN minutes it launches the run's late twin there.

A run may move unless its launch says this machine and no other: a pool run `--on auto` placed, and a run sent by name to
the pool `auto` reads (`research run --on MACHINE --pool research`) that asks no GPUs and no `--stay`. Such a run carries
a pair (slot.py's `--twin`, the run's own id), and `research run` leaves one asker of
that launch behind (`leave`): a detached process on the launching VM that holds what the launch shipped in its memory alone.
A launch that finds MAX_ASKERS of them alive under its runs root leaves none and says so: its run waits where it is.
At minute FIRST_ASK_MIN of its life, then every ASK_EVERY_MIN minutes, it looks at the run (`see`). While the run waits,
it asks (`ask`: `auto.read` and `auto.elsewhere`) by the files of the commit the launch
shipped (`files`: the few `auto.read` reads, no checkout), never by what the launching checkout holds by then. The answer
is a pool that has a place now for a job of this run's CPUs and memory, as a waiter with its ask finds the line there
(`auto._pool`), not a count of free slots. A place stands when the ask STAND_MIN minutes later, which it then makes, finds
that pool still has one. There it launches the same request (`launch`) from a checkout of that commit
(`tree`, made only for a launch) as `--twin PAIR@HOME` (`slot_only.late_twin`). slot.py's claim then lets one of the two
run: the twin runs only if it claims the pair on the first's machine before the first takes a slot, and the first leaves
its line once it finds the claim. A twin waits in its line no longer than ASK_EVERY_MIN minutes, or what is left of the
first's wait: it was sent where it would start now, and one that hasn't started by then leaves, as one that can't reach
the first's machine does at once. A machine whose twin left unrun is asked for no other twin of the run. One twin is live
at a time. The asker ends when the run has started on either machine, has ended or been cancelled (it then cancels a live
twin), when its twin was cancelled, or when the first's wait for a slot is over.

Its record (`record`) is <runs root>/DIR/<run id>.json on the launching VM, written before every sleep and before a twin is
launched, and copied beside the run's request on its machine (<root>/requests/<run id>/FILE), where another VM's
`research status` reads it; a twin is launched only once both name it. It says who asks (the launching agent, since
agent VMs share one host name; pid, since), when it looked last and looks next, each ask and its answer, the twin (its run
and machine, why the first waited and for how many minutes, and when the place it moved to was seen and seen again) and how
the asker ended. It is no file of the run dir, which
the run's custody seals. `alive` reads it, and `told` is what `research status RUN` says from it; a copy that can't be
read is said to be unread, never taken for no asker.

A run whose launching VM is gone has no asker: it waits where it is, `research status` says so, and the `stranded` monitor
is what is left to see it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "research/asker/v1"
#: a run that waits asks again at this minute of its wait and every so many after
ASK_EVERY_MIN = 10
#: and first at this minute: a run told it starts now, which found the memory gone when it reached the line (launches that
#: asked `auto` in the same seconds were each told of the same place), asks before it has waited ASK_EVERY_MIN
FIRST_ASK_MIN = 1
#: a place an ask finds on another machine's pool stands once the ask this many minutes later finds it there still: one seen
#: once may be one that the launches of the same minute each read and one of them takes
STAND_MIN = 2
#: an asker whose next look is this many seconds overdue is not alive (`alive`)
LATE_S = 300
#: the most askers alive under one runs root (`live`): each is a process of the launching VM, and those of launches of one
#: minute ask every machine, and check a twin's tree out, in the same minute. Soft: the count reads the askers' records
#: and takes no lock, so two launches of the same second may each count one short of it and both leave one
MAX_ASKERS = 8
#: the askers' records and logs, under the runs root: <run id>.json, <run id>.log
DIR = ".askers"
#: the copy of an asker's record beside its run's request on the run's machine
FILE = "asker.json"
#: the states of a twin that may still be, or is, the run (`twin`)
LIVE = ("launching", "launched", "unknown", "started")
#: a UTC time of a record (`_epoch`)
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z")


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(text: Any) -> float | None:
    """A UTC time as `_utc` and slot.py's records write it, in seconds; None for anything else."""
    if not (isinstance(text, str) and STAMP.fullmatch(text)):
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _sha256(path: str) -> str | None:
    """The digest of the file at ``path``; None where there is no such file."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def started(s: dict) -> bool:
    """Whether the run of summary ``s`` (`runs.summarize`) took a slot: it runs, or ran."""
    return (s.get("slot") or {}).get("state") == "held"


def _cancelled(s: dict) -> bool:
    return str(s.get("class") or "").startswith("CANCELLED")


# ---- leaving one behind -------------------------------------------------------------------------------------------------------

def live(root: Path, now: float) -> tuple[list[str], int]:
    """The runs under ``root`` whose askers hold a place on this VM at ``now``, by the records it keeps of them: those alive
    (`alive`, each a process of this VM) and those whose record can't be read, which may be; and how many are the second."""
    from . import remote

    held, unread = [], 0
    for p in sorted((root / DIR).glob("*.json")):
        rec = remote._read_json(p)
        if rec is None or alive({**rec, "here": True}, now):
            held.append(p.stem)
            unread += rec is None
    return held, unread


def leave(run_dir: Path, asks: dict, source: Path, what: dict, sends: list[Path], *, now: Callable[[], float] = time.time) -> str:
    """Leave an asker of the launch of the run in ``run_dir`` behind (`start`) and say so in a line. ``asks`` is what the
    launch asked: of `auto` (cpus, mem_gb, lean), its answer (home, the machine, and pool), the commit it shipped from the
    checkout ``source`` and the first's wait for a slot (wait_s); ``what`` is the request as `remote.launch` took it and
    ``sends`` its --send files. A launch whose asker can't be started is still launched: the line says the run won't move
    by itself. So is one that finds MAX_ASKERS alive under its runs root already (`live`): it leaves none, and the line
    names their runs. Such a run still carries its pair, which is harmless: only a run's asker launches its twin, so no
    twin of it is ever launched and nothing but the run itself claims the pair."""
    try:
        held, unread = live(run_dir.parent, now())
        if len(held) >= MAX_ASKERS:
            return (f"{len(held)} askers are alive on this VM already, and it holds at most {MAX_ASKERS} (their runs: "
                    f"{', '.join(held[:3])}" + (f" and {len(held) - 3} more" if len(held) > 3 else "")
                    + (f"; {unread} of them by a record that could not be read, counted as alive" if unread else "")
                    + f"), so this run will not move by itself: it waits in {asks['home']}'s line; `research status RUN` shows "
                      "when one of them has started or ended, and its asker ends at its next look after that")
        git = subprocess.run(["git", "-C", str(source), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, check=True).stdout.strip()
        m = {"run": run_dir.name, "root": str(run_dir.parent), "git": git, **asks, "sends": {str(p): _sha256(str(p)) for p in sends},
             "launch": {**what, "job_namespaces": list(what.get("job_namespaces") or ())}}
        pid = start(run_dir.parent, m)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as e:
        return (f"no asker of this launch could be left here ({type(e).__name__}: {e}), so the run will not move by itself: "
                f"it waits in {asks['home']}'s line")
    return (f"an asker of this launch stays on this VM (pid {pid}): at minute {FIRST_ASK_MIN} and each "
            f"{ASK_EVERY_MIN} min the run waits in {asks['home']}'s line it asks `auto` again and, where a place has stood "
            f"{STAND_MIN} min on another machine's pool, launches its late twin there. It lives only while this VM is up: "
            f"`research status {run_dir.name}` says whether one is alive")


def start(root: Path, m: dict) -> int:
    """Start the asker of launch ``m`` (`run`) as a process of its own session, with no terminal, its output in
    <runs root>/DIR/<run id>.log and ``m`` handed over its stdin, so nothing of the request is on disk or in its arguments.
    Its pid."""
    (root / DIR).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")]))
    with open(root / DIR / f"{m['run']}.log", "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "research.asker"], stdin=subprocess.PIPE, stdout=log, stderr=log,
                             start_new_session=True, cwd="/", env=env)
    p.stdin.write(json.dumps(m).encode())
    p.stdin.close()
    return p.pid


# ---- what it does at a look ---------------------------------------------------------------------------------------------------

def see(run_dir: Path) -> tuple[dict, bool]:
    """The run in ``run_dir`` as its machine has it now (`remote.fetch`, then `runs.summarize`), and whether the machine
    answered; why it didn't goes to the asker's log."""
    from . import remote, runs

    try:
        reached = not remote.fetch(run_dir, timeout=120).get("last_fetch_error")
    except (remote.Refused, subprocess.SubprocessError, OSError) as e:
        print(f"asker: {run_dir.name} could not be read from its machine ({type(e).__name__}: {e}); it is read as it was "
              "last seen", file=sys.stderr, flush=True)
        reached = False
    return runs.summarize(run_dir), reached


@contextlib.contextmanager
def files(git: str, commit: str, paths: Iterable[Path]) -> Iterator[Path]:
    """A temporary directory holding ``paths`` of ``commit`` alone, from the repository ``git`` (its common git dir), removed
    on the way out: what an ask reads of the tree the launch shipped (`auto.READS`), without a checkout of the whole tree
    at every ask. OSError when they can't be read."""
    import io
    import tarfile

    top = Path(tempfile.mkdtemp(prefix="research-asker-"))
    try:
        r = subprocess.run(["git", f"--git-dir={git}", "archive", "--format=tar", commit, "--", *map(str, paths)], cwd=top,
                           capture_output=True)
        if r.returncode:
            raise OSError(f"no files of {commit[:12]} from {git}: {r.stderr.decode(errors='replace').strip()[-300:]}")
        with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
            tar.extractall(top, filter="data")
        yield top
    finally:
        shutil.rmtree(top, ignore_errors=True)


@contextlib.contextmanager
def tree(git: str, commit: str) -> Iterator[Path]:
    """A checkout of ``commit`` from the repository ``git`` (its common git dir), as a detached worktree in a temporary
    directory, removed on the way out: what the launch shipped, whatever the launching checkout holds by now, for a twin's
    launch to ship. OSError when the commit can't be checked out."""
    top = Path(tempfile.mkdtemp(prefix="research-asker-"))
    at = top / commit[:12]
    try:
        r = subprocess.run(["git", f"--git-dir={git}", "worktree", "add", "--quiet", "--detach", str(at), commit],
                           capture_output=True, text=True)
        if r.returncode:
            raise OSError(f"no checkout of {commit[:12]} from {git}: {r.stderr.strip()[-300:]}")
        yield at
    finally:
        subprocess.run(["git", f"--git-dir={git}", "worktree", "remove", "--force", str(at)], capture_output=True)
        shutil.rmtree(top, ignore_errors=True)


def ask(source: Path, m: dict, tried: dict[str, str], seen: str | None = None) -> tuple[tuple[str, str] | None, str]:
    """Ask `auto` again for launch ``m``, by its tree's files in ``source`` (`files`): the pool of another machine that would
    start the run now (`auto.elsewhere`), or None, and why; the pool of machine ``seen``, the one the last ask found, while
    it still would. A machine in ``tried`` reads as what it says there."""
    from . import auto

    docs = {**auto.read(source, m["cpus"], peers=True, **({"lean": True} if m["lean"] else {})), **tried}
    if seen and (still := auto.elsewhere({seen: docs.get(seen, "not read")}, m["home"], m["cpus"], m["mem_gb"], lean_pool=m["lean"]))[0]:
        return still
    return auto.elsewhere(docs, m["home"], m["cpus"], m["mem_gb"], lean_pool=m["lean"])


def launch(source: Path, m: dict, twin: dict, wait_s: float) -> dict:
    """Launch ``twin`` of launch ``m``'s run from its tree ``source``: the same request on the twin's machine, its command as
    the run's late twin's (`slot_only.late_twin`), which waits in its line no longer than ``wait_s``. `remote.launch`'s record; its
    `auto` says where it moved from, which run it is the twin of, why that one waited and for how many minutes, and when the
    place it moved to was seen and seen again."""
    from . import remote
    from .cli import lean_records_urls
    from .pods.nebius import slot_only

    what = dict(m["launch"])
    what.update(command=slot_only.late_twin(what["command"], m["home"], wait_s), job_namespaces=tuple(what["job_namespaces"]))
    urls = lean_records_urls(source, what["command"])
    try:
        return remote.launch(run_id=twin["run"], local_dir=Path(m["root"]) / twin["run"], machine_name=twin["machine"], source=source,
                             sends=[*map(Path, m["sends"]), *([urls] if urls else [])], queue=None,
                             auto={"machine": twin["machine"], "pool": twin["pool"], "said": [twin["said"]],
                                   "moved": {k: twin[k] for k in ("from", "of", "why", "after_min", "stood") if k in twin}}, **what)
    finally:
        if urls:
            shutil.rmtree(urls.parent, ignore_errors=True)


def cancel(run_dir: Path, reason: str) -> str | None:
    """`remote.cancel` of the run in ``run_dir``, by the asker; why it wasn't cancelled, or None."""
    from . import remote

    try:
        remote.cancel(run_dir, reason=reason, by="research asker")
    except remote.Refused as e:
        return str(e)
    return None


def _beside(run_dir: Path) -> tuple[list[str], str] | None:
    """(ssh, path) of the asker's record beside the run's request on its machine, from the launcher's record of the run."""
    from . import remote

    rr = remote._read_json(run_dir / "remote.json") or {}
    if not (rr.get("host") and rr.get("root")):
        return None
    return remote.ssh_argv(rr["host"], rr["port"], user=rr.get("user") or "root"), f"{rr['root']}/requests/{run_dir.name}/{FILE}"


def tell(root: Path, rec: dict) -> str | None:
    """Write the asker's record ``rec`` here and copy it beside the run's request on its machine. Why the copy wasn't
    written, which the asker's log says too (the machine then keeps its last copy), or None."""
    from . import remote

    (root / DIR).mkdir(parents=True, exist_ok=True)
    remote.write_json_atomic(root / DIR / f"{rec['run']}.json", rec)
    if not (there := _beside(root / rec["run"])):
        return None
    q = shlex.quote(there[1])
    try:
        r = remote.ssh_run(there[0], ["sh", "-c", f"cat > {q}.tmp && mv {q}.tmp {q}"], stdin=json.dumps(rec).encode(), timeout=60)
        lost = f"exit {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}" if r.returncode else None
    except (OSError, subprocess.SubprocessError) as e:
        lost = f"{type(e).__name__}: {e}"
    if lost:
        print(f"asker: {rec['run']}'s record was not copied beside its request on its machine ({lost})", file=sys.stderr, flush=True)
    return lost


# ---- the asker ----------------------------------------------------------------------------------------------------------------

def run(root: Path, m: dict, *, now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> str:
    """The asker of launch ``m`` (module doc), until it ends; how it ended, which its record keeps."""
    from . import auto, remote, runs

    began = now()
    over = began + m["wait_s"]
    every, early = ASK_EVERY_MIN * 60, began + FIRST_ASK_MIN * 60
    rec: dict[str, Any] = {"schema": SCHEMA, "run": m["run"], "home": m["home"], "pool": m["pool"], "host": os.uname().nodename,
                           "by": remote.sender(), "pid": os.getpid(), "since": _utc(began), "every_min": ASK_EVERY_MIN, "asks": [],
                           "twin": None}
    twin: dict | None = None                                   # the live twin
    tried: dict[str, str] = {}                                 # machines a twin of the run left unrun: asked for no other
    again = False                                              # a twin was just launched: look at once, in case of a cancel
    place: dict | None = None                                  # the place the last ask found, until an ask finds it stood

    def beat(pause: float) -> str | None:
        rec.update(seen=_utc(now()), next=_utc(now() + pause))
        return tell(root, rec)

    def due(t: float) -> float:
        """Its next look after ``t``: minute FIRST_ASK_MIN of its life, then each ASK_EVERY_MIN, STAND_MIN after an ask that
        found a place, and never past the wait."""
        at = early if t < early else began + (int((t - began) // every) + 1) * every
        return min(at, place["t"] + STAND_MIN * 60 if place else at, over)

    while True:
        pause = 0 if again else max(due(now()) - now(), 0)
        again = False
        beat(pause)
        sleep(pause)
        first, reached = see(root / m["run"])
        other = see(root / twin["run"])[0] if twin else None
        slot = first.get("slot") or {}
        if other and started(other):
            twin["state"], why = "started", f"it moved: its twin {twin['run']} took a slot on {twin['machine']}"
            break
        if started(first):
            why = f"it started on {m['home']}" + (f", and its twin {twin['run']} leaves {twin['machine']}'s line" if twin else "")
            if twin:                                           # the first holds the pair's claim: its twin can't run
                twin.update(state="left", left=f"{m['run']} took a slot on {m['home']} first")
            break
        if _cancelled(first):
            left = cancel(root / twin["run"], f"its first, {m['run']} on {m['home']}, was cancelled") if twin else None
            why = "it was cancelled" + (f", and its twin {twin['run']} on {twin['machine']} "
                                        + (f"could not be ({left})" if left else "with it") if twin else "")
            if twin and not left:
                twin["state"] = "cancelled"
            break
        if other and _cancelled(other):
            twin["state"], why = "cancelled", f"its twin {twin['run']} was cancelled, so it asks no more; the run keeps its place on {m['home']}"
            break
        if first["state"] in runs.ENDED:
            why = f"it ended on {m['home']} without starting ({first['state']}: {slot.get('why') or first.get('why') or 'no reason recorded'})" \
                + (f"; its twin {twin['run']} on {twin['machine']} is what is left of it" if twin else "")
            break
        if other and other["state"] in runs.ENDED:
            said = (other.get("slot") or {}).get("why") or other.get("why") or other["state"]
            twin.update(state="left", left=said)
            tried[twin["machine"]] = f"a twin of {m['run']} left its line unrun ({said})"
            twin = None
        if now() >= over:
            why = "its wait for a slot is over"
            break
        if twin or not reached or first["state"] != "waiting" or slot.get("state") != "waiting":
            place = None                                       # a place stands only by the next ask's finding it
            continue
        asked: dict[str, Any] = {"at": _utc(now()), "after_min": round((now() - (_epoch(slot.get("since")) or began)) / 60)}
        rec["asks"].append(asked)
        try:
            if changed := [p for p, h in m["sends"].items() if _sha256(p) != h]:
                raise OSError(f"--send {', '.join(changed)} changed since the launch, and a twin ships the same request")
            with files(m["git"], m["commit"], auto.READS) as source:
                target, asked["said"] = ask(source, m, tried, place and place["machine"])
            if target is None:
                place = None
                continue
            if not place or (place["machine"], place["pool"]) != tuple(target):
                place = {"machine": target[0], "pool": target[1], "t": now(), "at": asked["at"]}
                asked["stands"] = f"not yet: {target[0]}'s {target[1]} pool is asked again in {STAND_MIN} min"
                continue
            if now() - place["t"] < STAND_MIN * 60:
                continue
            twin = {"run": runs.new_run_id(now()), "machine": target[0], "pool": target[1], "from": m["home"], "of": m["run"],
                    "why": slot.get("why") or "no reason recorded yet", "after_min": asked["after_min"], "at": asked["at"],
                    "stood": [place["at"], asked["at"]], "state": "launching"}
            place = None
            twin["said"] = (f"moved from {m['home']}'s {m['pool']} pool, where {m['run']} waited {twin['after_min']} min "
                            f"({twin['why']}), to {twin['machine']}'s, where a place stood from {twin['stood'][0]} to "
                            f"{twin['stood'][1]} ({asked['said']})")
            rec["twin"] = twin
            if lost := beat(every):                            # before the launch: no twin is ever launched unrecorded
                rec["twin"] = twin = None
                raise OSError(f"its record could not be copied beside its request on {m['home']} ({lost}), and a twin "
                              "launched then is one no other VM could follow or cancel")
            with tree(m["git"], m["commit"]) as source:
                sent = launch(source, m, twin, max(min(over - now(), every), 1))
            twin["state"] = "launched" if sent.get("phase") == "launched" else "refused"
        except Exception as e:  # noqa: BLE001  (an ask or a launch that failed is recorded, and the run keeps its place)
            asked["error"] = f"{type(e).__name__}: {e}"
            place = None
            if twin:
                phase = (remote._read_json(root / twin["run"] / "remote.json") or {}).get("phase")
                twin.update(state="unknown" if phase == "unknown" else "not launched", error=asked["error"])
        if twin and twin["state"] in ("refused", "not launched"):
            tried[twin["machine"]] = f"it took no twin of {m['run']} ({twin.get('error') or 'its launch was refused'})"
            twin = None
        again = twin is not None
    rec.update(seen=_utc(now()), next=None, ended={"at": _utc(now()), "why": why})
    tell(root, rec)
    return why


# ---- reading its record -------------------------------------------------------------------------------------------------------

def record(run_dir: Path) -> dict | None:
    """The asker's record of the run in ``run_dir``: this VM's (marked "here": its asker is, or was, a process of this VM),
    else, for a run that carries its own id as its pair (its slot record, as fetched), the copy beside its request on its
    machine, one ssh; None when there is none. A copy that can't be read is {"unread": why}: whether the run has an asker,
    or a twin, is then not known."""
    from . import remote, runs

    if doc := remote._read_json(run_dir.parent / DIR / f"{run_dir.name}.json"):
        return {**doc, "here": True}
    if (remote._read_json(run_dir / runs.SLOT_FILE) or {}).get("twin") != run_dir.name or not (at := _beside(run_dir)):
        return None
    q = shlex.quote(at[1])
    try:
        r = remote.ssh_run(at[0], ["sh", "-c", f"test ! -e {q} || cat {q}"], timeout=30)
        if r.returncode:
            return {"unread": f"exit {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}"}
        doc = json.loads(r.stdout) if r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {"unread": f"{type(e).__name__}: {e}"}
    return doc if doc is None or isinstance(doc, dict) else {"unread": "it holds no record"}


def alive(rec: dict | None, now: float) -> bool:
    """Whether the asker of record ``rec`` (`record`) is alive at ``now``: it hasn't ended, its next look isn't LATE_S
    overdue, and, where the record is this VM's own, its process is there. Another VM's asker is known by its record's
    times alone: agent VMs share one host name, so a record's host tells no VM from another."""
    from . import remote

    if not rec or rec.get("ended") or (due := _epoch(rec.get("next"))) is None or now > due + LATE_S:
        return False
    return not rec.get("here") or remote._pid_alive(rec.get("pid"))


def twin(run_dir: Path) -> dict | None:
    """The late twin that may be, or is, the run in ``run_dir`` (its asker's record: run, machine, state, why), or None: the
    run has no twin, or its twin left unrun. A record that can't be read names none, and stderr says so."""
    rec = record(run_dir) or {}
    if "unread" in rec:
        print(f"research: {run_dir.name}: whether it has a late twin is not known: its asker's record could not be read "
              f"({rec['unread']})", file=sys.stderr)
    tw = rec.get("twin")
    return tw if isinstance(tw, dict) and tw.get("state") in LIVE else None


def told(s: dict, now: float) -> list[str]:
    """What `research status RUN` says of a pool run that may move (module doc), from its summary ``s`` (`runs.summarize`) and its
    asker's record: its late twin, and, while it waits, whether an asker is alive for it (where, since when) or that none
    is, so it will not move by itself; of a record that can't be read, that neither is known. Of a late twin: which run it
    is the twin of. Nothing of any other run."""
    from . import auto
    from .store import ids

    slot, rr = s.get("slot") or {}, s.get("remote_rec") or {}
    pair, _, home = str(slot.get("twin") or "").partition("@")
    if moved := (rr.get("auto") or {}).get("moved"):
        return [f"the late twin of {moved['of']}, which waited {moved['after_min']} min on {moved['from']} ({moved['why']})"]
    if home and ids.is_run_id(pair):
        return [f"the late twin of {pair}, which waited on {home}"]
    rec = record(Path(s["dir"]))
    if rec is None and pair != s["id"]:
        return []
    out, where = [], rr.get("machine")
    if rec and "unread" in rec:
        return [f"whether it has an asker, or a late twin, is not known: its asker's record beside its request on {where} could "
                f"not be read ({rec['unread']})"]
    vm = f"the VM that launched it ({rec.get('by') or rec['host']}, pid {rec['pid']})" if rec else None
    if tw := (rec or {}).get("twin"):
        head = {"started": f"moved to {tw['machine']} as {tw['run']}", "launched": f"its late twin {tw['run']} waits on {tw['machine']} too, "
                "and the one that takes a slot first runs:", "launching": f"its asker was launching its late twin {tw['run']} on "
                f"{tw['machine']} at {tw['at']}:"}.get(tw.get("state"), f"its late twin {tw['run']} on {tw['machine']} did not run "
                                                         f"({tw.get('state')}: {tw.get('left') or tw.get('error') or 'no reason recorded'}):")
        stood = f", where a place stood from {tw['stood'][0]} to {tw['stood'][1]}" if tw.get("stood") else ""
        out.append(f"{head} after {tw['after_min']} min waiting on {tw['from']} ({tw['why']}){stood}")
    if s["state"] != "waiting":
        return out
    if alive(rec, now):
        return [*out, f"an asker is alive on {vm} since {rec['since']}; it looked last at {rec['seen']} and looks next at "
                      f"{rec['next']}: while the run waits it asks `auto` again, and moves the run to a pool where a place has "
                      f"stood {STAND_MIN} min"]
    gone = "" if rec is None else f" (the one on {vm} ended at {rec['ended']['at']}: {rec['ended']['why']})" if rec.get("ended") \
        else f" (the one on {vm} looked last at {rec.get('seen')} and is gone)"
    return [*out, f"waits{' on memory' if auto.MEMORY_BLOCK in str(slot.get('why')) else ''} on {where}; no asker is alive{gone}, so it "
                  "will not move by itself"]


def main() -> int:
    m = json.loads(sys.stdin.read())
    print(f"asker of {m['run']}: {run(Path(m['root']), m)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
