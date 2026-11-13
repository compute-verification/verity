"""`served_ab_row.sh` on the CPU: the hold for the card, the deadline, and what is left of the run after either.

The card is a stub ``nvidia-smi`` and a stub ``gpu-lease`` that answer by which look at the card it is, and the time is a
command the test names (``ROW_NOW``), so each test moves on an event and none on the clock. The hold runs a stub
``served_ab.sh`` in a tree of its own. The deadline runs the tree's ``served_ab.sh`` for one round with a stub
interpreter whose endpoint ignores SIGTERM and has a process that left its session, with its load still out.
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROW = HERE.parent / "served_ab_row.sh"
START_BY, DEADLINE = "1970-01-01T00:30:00Z", "1970-01-01T01:00:00Z"   # 1800 and 3600 on the row's clock
#: the row's last line over a free card once a lifetime was ended by SIGKILL
ENDED = "end: card 1 4 MiB, 0 compute process(es), lease free; 0 process(es) of the run left; served_ab.sh exit 137"

#: the card at each look, a word a look and the last word from then on: $MIB its memory, $APPS the compute process there
#: is (0 none, 1 one on the card, 2 one on another card), $LEASES its lease, with another card's the reverse of it
PICK = 'pick() { local n=$1; shift; [ "$n" -lt $# ] || n=$(($# - 1)); shift "$n"; echo "$1"; }'
SMI = f"""#!/usr/bin/env bash
{PICK}
case "$*" in
  *--query-gpu=uuid*) echo GPU-stub ;;
  *--query-gpu=memory.used*) n=$(cat "$LOOKS" 2> /dev/null || echo 0); echo $((n + 1)) > "$LOOKS"; pick "$n" ${{MIB:-4}} ;;
  *--query-compute-apps*) case $(pick $(($(cat "$LOOKS") - 1)) ${{APPS:-0}}) in 1) echo GPU-stub ;; 2) echo GPU-other ;; esac ;;
  -q) echo "every card" ;;
esac
"""
LEASE = f"""#!/usr/bin/env bash
{PICK}
state=$(pick $(($(cat "$LOOKS") - 1)) ${{LEASES:-free}})
[ -z "${{LEASES:-}}" ] || {{ [ "$state" = free ] && echo "0 held 9m who=another" || echo "0 free"; }}
[ "$state" = free ] && echo "1 free" || echo "1 held 3m who=another"
"""
SERVED_AB = """#!/usr/bin/env bash
[ "$1" != imports ] || { printf '%s\\n' ${IMPORTS:-$PWD/verity/__init__.py $PWD/integrations/vllm/serve_pearl.py}; exit; }
{ echo "args $*"; echo "arms $ARMS"; echo "not after $NOT_AFTER"; echo "tagged ${SERVED_AB_ROW:+yes}"; } > "$RECORD"
mkdir -p "$SERVE_ROOT/runs/$RESEARCH_RUN_ID/rep-1"
exit 7
"""
#: an endpoint that does not stop when asked, one process of it gone to a session of its own, and a load that stays out
STUB_PY = """#!/usr/bin/env bash
case "$*" in
  "-c import verity,"*) echo "$PWD/verity_vllm/__init__.py" ;;
  "-c "*) echo 0123456789abcdef ;;
  *" serve-openai "*)
    trap '' TERM
    setsid bash -c 'echo $$ >> "$PIDS"; echo there > "$READY"; read < "$HOLD"' &
    read < "$READY"; echo $$ >> "$PIDS"
    echo "SERVE-OPENAI listening on 127.0.0.1"; read < "$HOLD" ;;
  *serve_load.py*) echo $$ >> "$PIDS"; echo up > "$UP"; read < "$HOLD" ;;
esac
"""
#: 1000 until a load is out (3600), or until the row has said $LATER_AT ($LATER). A row that goes on asking with its load
#: out is no longer waited for: the load and the endpoint are let go.
CLOCK = """#!/usr/bin/env bash
t=1000
if [ -s "$UP" ]; then
  t=3600; echo >> "$UP.asked"
  [ "$(wc -l < "$UP.asked")" -lt 400 ] || : > "$HOLD"
fi
grep -qs "$LATER_AT" "$OUT/row.txt" && t=$LATER
echo "$t"
"""


def _env(tmp_path: Path, **env: str) -> dict[str, str]:
    """The row's environment over a stub card: ``bin`` holds the stubs, and the clock reads 1000 until told otherwise."""
    (tmp_path / "bin").mkdir(parents=True)
    for name, text in (("nvidia-smi", SMI), ("gpu-lease", LEASE), ("clock", CLOCK)):
        (tmp_path / "bin" / name).write_text(text)
        (tmp_path / "bin" / name).chmod(0o755)
    return {"PATH": f"{tmp_path}/bin:/usr/bin:/bin", "HOME": str(tmp_path), "OUT": str(tmp_path / "out"),
            "SERVE_ROOT": str(tmp_path / "root"), "RESEARCH_RUN_ID": "t", "LOOKS": str(tmp_path / "looks"),
            "UP": str(tmp_path / "up"), "START_BY": START_BY, "DEADLINE": DEADLINE, "ROW_NOW": "echo 1000",
            "ROW_POLL": "0.05", "ROW_HOLD_POLL": "0.05", "LATER_AT": "no such line", "LATER": "0"} | env


def _tree(tmp_path: Path) -> Path:
    """A tree whose ``served_ab.sh`` records what it was handed, leaves a serve dir and exits 7."""
    at = tmp_path / "tree" / "benchmarks" / "one_stage"
    at.mkdir(parents=True)
    (at / "served_ab.sh").write_text(SERVED_AB)
    return tmp_path / "tree"


def _row(env: dict[str, str], cwd: Path, *args: str) -> tuple[int, str]:
    """The row run to its end in `cwd`: its exit and what it printed, kept in a file so that a process it left behind,
    which holds a pipe open, does not hold the test."""
    log = Path(env["OUT"]).parent / "row.log"
    with log.open("w") as out:
        rc = subprocess.run(["bash", str(ROW), *args], cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    return rc, log.read_text()


def _said(tmp_path: Path) -> list[str]:
    """The row's lines, each without its time."""
    return [x.split(" ", 1)[1] for x in (tmp_path / "out" / "row.txt").read_text().splitlines()]


def _up(pid: int) -> bool:
    """A process that is there and still has its memory: one that has given it up is ending, or its exit status waits to
    be read."""
    try:
        return Path(f"/proc/{pid}/statm").read_text().split()[0] != "0"
    except (FileNotFoundError, ProcessLookupError):
        return False


@pytest.fixture
def lifetime(tmp_path):
    """The environment of one round of the tree's ``served_ab.sh`` under the row, its interpreter `STUB_PY`: each of its
    processes writes its pid to ``pids``, and the load says ``up`` once the endpoint listens and it is out. A process
    of it that is still there after the test is let go."""
    yield lambda **env: _lifetime(tmp_path, **env)
    hold = tmp_path / "hold"
    if hold.exists():                              # read-write opens a FIFO at once, whether or not a process waits on it
        os.close(os.open(hold, os.O_RDWR | os.O_NONBLOCK))


def _lifetime(tmp_path: Path, **env: str) -> dict[str, str]:
    env = _env(tmp_path, **env)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    for name in ("python", "python3"):
        (venv / name).write_text(STUB_PY)
        (venv / name).chmod(0o755)
    for fifo in ("hold", "ready"):
        os.mkfifo(tmp_path / fifo)
    return env | {"VENV": str(tmp_path / "venv"), "STAGE": str(tmp_path / "stage"), "HF_HOME": str(tmp_path),
                  "TAP_CUDA": str(tmp_path), "CUBIN_CUDA": str(tmp_path), "SERVED_AB_PINNED": "1", "ARMS": "deferred",
                  "LOADS": "1", "PIDS": str(tmp_path / "pids"), "HOLD": str(tmp_path / "hold"),
                  "READY": str(tmp_path / "ready")}


def _left(tmp_path: Path) -> list[int]:
    """The lifetime's processes that have not ended, by the pids they wrote themselves."""
    pids = [int(x) for x in (tmp_path / "pids").read_text().split()]
    assert len(pids) == 3                          # the endpoint, its process in another session, the load
    return [p for p in pids if _up(p)]


def test_a_row_holds_for_the_run_before_and_the_card_then_runs_served_ab_and_ends_with_the_cards_line(tmp_path):
    """A process of the run before, then a card with memory in use, then one with a compute process on it, then one whose
    lease is held, each alone holds the row; on a free card, with another card in use, `served_ab.sh` runs with the
    row's arguments and environment and a tag of the row's, and its exit is the row's. The serve dir it left is
    removed, and the card's line and both ``nvidia-smi -q`` captures are written."""
    fifo, after = tmp_path / "before", f"RESEARCH_RUN_DIR={tmp_path}/the-run-before"
    os.mkfifo(fifo)
    before = subprocess.Popen(["bash", "-c", 'read < "$0"', str(fifo)], env=dict([after.split("=")], PATH="/usr/bin:/bin"))
    env = _env(tmp_path, AFTER=after, MIB="81000 4", APPS="0 1 0 2 0", LEASES="free free held free", ARMS="stock deferred",
               NOT_AFTER="1970-01-01T00:50Z", RECORD=str(tmp_path / "record"))
    p = subprocess.Popen(["bash", str(ROW), "round", "rep-1"], cwd=_tree(tmp_path), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    first = p.stdout.readline()                    # the row says what holds it, or has ended
    fifo.open("w").close()                         # the run before's last process goes
    assert before.wait() == 1                      # its read ended with nothing read
    out, _ = p.communicate()
    assert p.returncode == 7, first + out
    assert _said(tmp_path) == [
        f"holding: a process still carrying {after}",
        "holding: card 1 81000 MiB, 0 compute process(es), lease free",
        "holding: card 1 4 MiB, 1 compute process(es), lease free",
        "holding: card 1 4 MiB, 0 compute process(es), lease held",
        f"start: card 1 4 MiB, 0 compute process(es), lease free; deadline {DEADLINE}, then 15 s to SIGKILL",
        "end: card 1 4 MiB, 0 compute process(es), lease free; 0 process(es) of the run left; served_ab.sh exit 7"]
    assert (tmp_path / "record").read_text().splitlines() == ["args round rep-1", "arms stock deferred",
                                                              "not after 1970-01-01T00:50Z", "tagged yes"]
    assert not (tmp_path / "root" / "runs" / "t").exists()
    assert [(tmp_path / "out" / f"nvidia-smi-q-{w}.txt").read_text() for w in ("start", "end")] == ["every card\n"] * 2


def test_a_row_starts_nothing_once_its_start_bound_has_come(tmp_path):
    """A row still holding at START_BY starts nothing, though the card is free one look later; one that arrives at
    START_BY starts nothing on a free card either, nor does one whose interpreter imports the endpoint from outside
    its tree. Without START_BY or DEADLINE it is refused."""
    busy = "card 1 81000 MiB, 0 compute process(es), lease free"
    env = _env(tmp_path / "a", MIB="81000 4", ROW_NOW=f"{tmp_path}/a/bin/clock", LATER_AT="holding:",
               LATER="1800", RECORD=str(tmp_path / "record"))
    rc, out = _row(env, _tree(tmp_path))
    assert rc == 20, out
    assert _said(tmp_path / "a") == [f"holding: {busy}", f"not started: START_BY {START_BY} has come with {busy}"]
    env = _env(tmp_path / "b", ROW_NOW="echo 1800", RECORD=str(tmp_path / "record"))
    rc, out = _row(env, tmp_path / "tree")
    assert rc == 20 and _said(tmp_path / "b") == [f"not started: START_BY {START_BY} has come"]
    assert not (tmp_path / "record").exists() and not (tmp_path / "b" / "looks").exists()
    elsewhere = "/site-packages/verity_vllm/x.py"
    env |= {"ROW_NOW": "echo 1000", "IMPORTS": f"{tmp_path}/tree/verity/__init__.py {elsewhere}"}
    rc, out = _row(env, tmp_path / "tree")
    assert rc == 8 and _said(tmp_path / "b")[-1] == f"not started: the venv imports {elsewhere}, not this tree's"
    assert not (tmp_path / "record").exists()
    for missing in ("START_BY", "DEADLINE"):                 # before anything else is read
        rc, out = _row(env | {missing: ""}, tmp_path / "tree")
        assert rc == 2 and f"{missing} is required" in out


def test_the_deadline_ends_a_lifetime_that_is_up_and_leaves_no_process_of_the_run(tmp_path, lifetime):
    """With an endpoint listening and its load out, the deadline sends SIGTERM to the run's processes, and GRACE seconds
    later SIGKILL to those still there: the endpoint, which ignores SIGTERM, its process in another session, and
    `served_ab.sh` waiting for the endpoint. None is left, the serve dir is gone, and the row exits 21."""
    rc, out = _row(lifetime(ROW_NOW=f"{tmp_path}/bin/clock", LATER_AT="SIGTERM to", LATER="3615"), HERE.parents[2],
                   "round", "rep-1", "0")
    assert rc == 21, out
    said = _said(tmp_path)
    assert said[1] == f"DEADLINE {DEADLINE} has come with the run still up"
    assert said[2].startswith("SIGTERM to the run's ") and said[3].startswith("SIGKILL to the run's ")
    assert said[4:] == [ENDED]
    assert _left(tmp_path) == []
    assert not (tmp_path / "root" / "runs" / "t").exists()


def test_a_row_told_to_stop_kills_its_run_at_once(tmp_path, lifetime):
    """SIGTERM to the row with a lifetime up, as `research run`'s timeout sends it: the run's processes get SIGKILL with
    no grace, none is left, the card's line is written and the row exits 143."""
    os.mkfifo(tmp_path / "up")
    with (tmp_path / "row.log").open("w") as out:
        p = subprocess.Popen(["bash", str(ROW), "round", "rep-1", "0"], cwd=HERE.parents[2], env=lifetime(), stdout=out,
                             stderr=subprocess.STDOUT)
    up, ended = os.open(tmp_path / "up", os.O_RDONLY | os.O_NONBLOCK), os.pidfd_open(p.pid)
    ready, _, _ = select.select([up, ended], [], [])   # the load's word that it is out, or the row's end without it
    os.close(up)
    os.close(ended)
    if ready == [up]:
        p.send_signal(signal.SIGTERM)
    assert (p.wait(), ready) == (143, [up]), (tmp_path / "row.log").read_text()
    said = _said(tmp_path)
    assert said[1] == "told to stop" and said[2].startswith("SIGKILL to the run's ")
    assert said[3:] == [ENDED]
    assert _left(tmp_path) == []
