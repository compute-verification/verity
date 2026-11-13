"""The asker of a run `--on auto` sent to a pool: while the run waits there behind reserved memory it asks `auto` again, and
when another machine's pool would start the run now it launches the run's late twin there.

A pool run `--on auto` placed carries a pair (slot.py's `--twin`, the run's own id), and `research run` leaves one asker of
that launch behind (`leave`): a detached process on the launching VM that holds what the launch shipped in its memory alone.
At minute FIRST_ASK_MIN of its life, then every ASK_EVERY_MIN minutes, it looks at the run (`see`). While the run waits
and its last reason is memory, it asks (`ask`: `auto.read` and `auto.elsewhere`) from a checkout of the commit the launch
shipped (`tree`), never from what the launching checkout holds by then. The answer is a pool that has a place now for a
job of this run's CPUs and memory, as a waiter with its ask finds the line there (`auto._pool`), not a count of free
slots. There it launches the same request (`launch`) as `--twin PAIR@HOME` (`slot_only.late_twin`). slot.py's claim then
lets one of the two run: the twin runs only if it claims the pair on the first's machine before the first takes a slot,
and the first leaves its line once it finds the claim. A twin waits in its line no longer than ASK_EVERY_MIN minutes, or
what is left of the first's wait: it was sent where it would start now, and one that hasn't started by then leaves, as
one that can't reach the first's machine does at once. A machine whose twin left unrun is asked for no other twin of the
run. One twin is live at a time. The asker ends when the run has started on either machine, has ended or been cancelled
(it then cancels a live twin), when its twin was cancelled, or when the first's wait for a slot is over.

Its record (`record`) is <runs root>/DIR/<run id>.json on the launching VM, written before every sleep and before a twin is
launched, so a twin is never launched unrecorded, and copied beside the run's request on its machine
(<root>/requests/<run id>/FILE), where another VM's `research status` reads it: who asks (the launching agent, since
agent VMs share one host name; pid, since), when it looked last and looks next, each ask and its answer, the twin (its run
and machine, why the first waited and for how many minutes) and how the asker ended. It is no file of the run dir, which
the run's custody seals. `alive` reads it, and `told` is what `research status RUN` says from it.

A run whose launching VM is gone has no asker: it waits where it is, `research status` says so, and the `stranded` monitor
is what is left to see it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "research/asker/v1"
#: a run that waits on memory asks again at this minute of its wait and every so many after
ASK_EVERY_MIN = 10
#: and first at this minute: a run told it starts now, which found the memory gone when it reached the line (launches that
#: asked `auto` in the same seconds were each told of the same place), asks before it has waited ASK_EVERY_MIN
FIRST_ASK_MIN = 1
#: an asker whose next look is this many seconds overdue is not alive (`alive`)
LATE_S = 300
#: the askers' records and logs, under the runs root: <run id>.json, <run id>.log
DIR = ".askers"
#: the copy of an asker's record beside its run's request on the run's machine
FILE = "asker.json"
#: the states of a twin that may still be, or is, the run (`twin`)
LIVE = ("launching", "launched", "unknown", "started")


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(text: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def started(s: dict) -> bool:
    """Whether the run of summary ``s`` (`runs.summarize`) took a slot: it runs, or ran."""
    return (s.get("slot") or {}).get("state") == "held"


def _cancelled(s: dict) -> bool:
    return str(s.get("class") or "").startswith("CANCELLED")


# ---- leaving one behind -------------------------------------------------------------------------------------------------------

def leave(run_dir: Path, asks: dict, source: Path, what: dict, sends: list[Path]) -> str:
    """Leave an asker of the launch of the run in ``run_dir`` behind (`start`) and say so in a line. ``asks`` is what the
    launch asked: of `auto` (cpus, mem_gb, lean), its answer (home, the machine, and pool), the commit it shipped from the
    checkout ``source`` and the first's wait for a slot (wait_s); ``what`` is the request as `remote.launch` took it and
    ``sends`` its --send files. A launch whose asker can't be started is still launched: the line says the run won't move
    by itself."""
    try:
        git = subprocess.run(["git", "-C", str(source), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, check=True).stdout.strip()
        m = {"run": run_dir.name, "root": str(run_dir.parent), "git": git, **asks, "sends": {str(p): _sha256(str(p)) for p in sends},
             "launch": {**what, "job_namespaces": list(what.get("job_namespaces") or ())}}
        pid = start(run_dir.parent, m)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as e:
        return (f"no asker of this launch could be left here ({type(e).__name__}: {e}), so the run will not move by itself: "
                f"it waits in {asks['home']}'s line")
    return (f"an asker of this launch stays on this VM (pid {pid}): at minute {FIRST_ASK_MIN} and each "
            f"{ASK_EVERY_MIN} min the run waits on memory in {asks['home']}'s line it asks `auto` again and, where another "
            f"machine's pool would start the run now, launches its late twin there. It lives only while this VM is up: "
            f"`research status {run_dir.name}` says whether one is alive")


def start(root: Path, m: dict) -> int:
    """Start the asker of launch ``m`` (`run`) as a process of its own session, with no terminal, its output in
    <runs root>/DIR/<run id>.log and ``m`` handed over its stdin, so nothing of the request is on disk or in its arguments.
    Its pid."""
    (root / DIR).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                                     os.environ.get("PYTHONPATH")]))}
    with open(root / DIR / f"{m['run']}.log", "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "research.asker"], stdin=subprocess.PIPE, stdout=log, stderr=log,
                             start_new_session=True, cwd="/", env=env)
    p.stdin.write(json.dumps(m).encode())
    p.stdin.close()
    return p.pid


# ---- what it does at a look ---------------------------------------------------------------------------------------------------

def see(run_dir: Path) -> tuple[dict, bool]:
    """The run in ``run_dir`` as its machine has it now (`remote.fetch`, then `runs.summarize`), and whether the machine
    answered."""
    from . import remote, runs

    try:
        reached = not remote.fetch(run_dir, timeout=120).get("last_fetch_error")
    except (remote.Refused, subprocess.SubprocessError, OSError):
        reached = False
    return runs.summarize(run_dir), reached


@contextlib.contextmanager
def tree(git: str, commit: str) -> Iterator[Path]:
    """A checkout of ``commit`` from the repository ``git`` (its common git dir), as a detached worktree in a temporary
    directory, removed on the way out: what the launch shipped, whatever the launching checkout holds by now. OSError when
    the commit can't be checked out."""
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


def ask(source: Path, m: dict, tried: dict[str, str]) -> tuple[tuple[str, str] | None, str]:
    """Ask `auto` again for launch ``m``, from its tree ``source``: the pool of another machine that would start the run now
    (`auto.elsewhere`), or None, and why. A machine in ``tried`` reads as what it says there."""
    from . import auto

    docs = auto.read(source, m["cpus"], peers=True, **({"lean": True} if m["lean"] else {}))
    return auto.elsewhere({**docs, **tried}, m["home"], m["cpus"], m["mem_gb"], lean_pool=m["lean"])


def launch(source: Path, m: dict, twin: dict, wait_s: float) -> dict:
    """Launch ``twin`` of launch ``m``'s run from its tree ``source``: the same request on the twin's machine, its command as
    the run's late twin's (`slot_only.late_twin`), which waits in its line no longer than ``wait_s``. `remote.launch`'s record; its
    `auto` says where it moved from, which run it is the twin of, why that one waited and for how many minutes."""
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
                                   "moved": {k: twin[k] for k in ("from", "of", "why", "after_min")}}, **what)
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


def tell(root: Path, rec: dict) -> None:
    """Write the asker's record ``rec``: here, and its copy beside the run's request on its machine (as it can: a machine
    that doesn't answer keeps its last copy)."""
    from . import remote

    (root / DIR).mkdir(parents=True, exist_ok=True)
    remote.write_json_atomic(root / DIR / f"{rec['run']}.json", rec)
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        if there := _beside(root / rec["run"]):
            q = shlex.quote(there[1])
            remote.ssh_run(there[0], ["sh", "-c", f"cat > {q}.tmp && mv {q}.tmp {q}"], stdin=json.dumps(rec).encode(), timeout=60)


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

    def beat(pause: float) -> None:
        rec.update(seen=_utc(now()), next=_utc(now() + pause))
        tell(root, rec)

    def due(t: float) -> float:
        """Its next look after ``t``: minute FIRST_ASK_MIN of its life, then each ASK_EVERY_MIN, and never past the wait."""
        return min(early if t < early else began + (int((t - began) // every) + 1) * every, over)

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
        if twin or not reached or first["state"] != "waiting" or auto.MEMORY_BLOCK not in str(slot.get("why")):
            continue
        asked: dict[str, Any] = {"at": _utc(now()), "after_min": round((now() - (_epoch(slot.get("since")) or began)) / 60)}
        rec["asks"].append(asked)
        try:
            with tree(m["git"], m["commit"]) as source:
                if changed := [p for p, h in m["sends"].items() if _sha256(p) != h]:
                    raise OSError(f"--send {', '.join(changed)} changed since the launch, and a twin ships the same request")
                target, asked["said"] = ask(source, m, tried)
                if target is None:
                    continue
                twin = {"run": runs.new_run_id(now()), "machine": target[0], "pool": target[1], "from": m["home"], "of": m["run"],
                        "why": slot["why"], "after_min": asked["after_min"], "at": asked["at"], "state": "launching"}
                twin["said"] = (f"moved from {m['home']}'s {m['pool']} pool, where {m['run']} waited {twin['after_min']} min "
                                f"({twin['why']}), to {twin['machine']}'s, which would start it now ({asked['said']})")
                rec["twin"] = twin
                beat(every)                                    # before the launch: no twin is ever launched unrecorded
                sent = launch(source, m, twin, max(min(over - now(), every), 1))
                twin["state"] = "launched" if sent.get("phase") == "launched" else "refused"
        except Exception as e:  # noqa: BLE001  (an ask or a launch that failed is recorded, and the run keeps its place)
            asked["error"] = f"{type(e).__name__}: {e}"
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
    machine, one ssh; None when there is none or it can't be read."""
    from . import remote, runs

    if doc := remote._read_json(run_dir.parent / DIR / f"{run_dir.name}.json"):
        return {**doc, "here": True}
    try:
        if (remote._read_json(run_dir / runs.SLOT_FILE) or {}).get("twin") != run_dir.name or not (at := _beside(run_dir)):
            return None
        r = remote.ssh_run(at[0], ["cat", at[1]], timeout=30)
        doc = json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


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
    run has no twin, or its twin left unrun."""
    tw = (record(run_dir) or {}).get("twin")
    return tw if isinstance(tw, dict) and tw.get("state") in LIVE else None


def told(s: dict, now: float) -> list[str]:
    """What `research status RUN` says of a pool run `--on auto` placed, from its summary ``s`` (`runs.summarize`) and its
    asker's record: its late twin, and, while it waits, whether an asker is alive for it (where, since when) or that none
    is, so it will not move by itself. Of a late twin: which run it is the twin of. Nothing of any other run."""
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
    vm = f"the VM that launched it ({rec.get('by') or rec['host']}, pid {rec['pid']})" if rec else None
    if tw := (rec or {}).get("twin"):
        head = {"started": f"moved to {tw['machine']} as {tw['run']}", "launched": f"its late twin {tw['run']} waits on {tw['machine']} too, "
                "and the one that takes a slot first runs:", "launching": f"its asker was launching its late twin {tw['run']} on "
                f"{tw['machine']} at {tw['at']}:"}.get(tw.get("state"), f"its late twin {tw['run']} on {tw['machine']} did not run "
                                                         f"({tw.get('state')}: {tw.get('left') or tw.get('error') or 'no reason recorded'}):")
        out.append(f"{head} after {tw['after_min']} min waiting on {tw['from']} ({tw['why']})")
    if s["state"] != "waiting":
        return out
    if alive(rec, now):
        return [*out, f"an asker is alive on {vm} since {rec['since']}; it looked last at {rec['seen']} and looks next at "
                      f"{rec['next']}: while the run waits on memory it asks `auto` again, and moves the run to a pool that would "
                      "start it now"]
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
