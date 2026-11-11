"""`warm_twice.py`'s guards and its cache's removal, on a stand-in for `warm`: a shell command that composes one small "circuit"
into the cache it is handed and, on its second pass, finds it kept. The free space the job reads comes from a file
(``--free-from``), and a stand-in that must still be running when something happens blocks on a FIFO nobody writes, so no
test waits on a clock. Under a free-space minimum the job starts nothing; two passes time the compose and then the kept
circuit; a free-space reading under the floor in mid-pass, and a SIGTERM in mid-pass, each end the pass's processes; and on
each way out the cache's parent is empty again and the last line of output says so."""
from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

JOB = Path(__file__).resolve().parents[1] / "warm_twice.py"

#: one template composed the way a5's cache holds it (`<key>/circuit.txt` and `made.json`, made aside and renamed), then kept
COMPOSE = r'''c="$1"
if [ -f "$c/k/made.json" ]; then kept=true; else
  kept=false; mkdir "$c/k.tmp$$" && head -c 4096 /dev/zero > "$c/k.tmp$$/circuit.txt" && echo "{}" > "$c/k.tmp$$/made.json" \
    && mv "$c/k.tmp$$" "$c/k"
fi
echo "{\"template\": \"T_v1\", \"seconds\": 0.5, \"kept\": $kept, \"k_log\": 3, \"pin\": \"p\"}"
echo "WARM {\"templates\": 1, \"kept\": 0, \"verifier_window_s\": 0.1, \"seconds\": 0.6}"
'''
#: a pass in the middle of a compose, with a worker of its own: it says it has started (`$SAY` into `$STARTED`, a FIFO a test
#: reads, or a file replaced whole) and then blocks for good, both processes on `$HOLD`
MID_PASS = r'''c="$1"
mkdir "$c/k.tmp$$" && head -c 4096 /dev/zero > "$c/k.tmp$$/circuit.txt"
(read -r _ < "$HOLD") &
if [ -p "$STARTED" ]; then echo "$SAY" > "$STARTED"; else echo "$SAY" > "$STARTED.new" && mv "$STARTED.new" "$STARTED"; fi
read -r _ < "$HOLD"
'''


def job(tmp_path: Path, stand_in: str, *more: str, **env: str) -> subprocess.Popen:
    for d in ("tree", "parent", "out"):
        (tmp_path / d).mkdir()
    return subprocess.Popen(
        [sys.executable, str(JOB), "--window", str(tmp_path / "no-window"), "--cache-parent", str(tmp_path / "parent"),
         "--stand-in", stand_in, "--watch-every", "0.05", "--guardian", "detached", *more],
        cwd=tmp_path / "tree", env={**os.environ, "OUT": str(tmp_path / "out"), **env}, stdout=subprocess.PIPE, text=True)


def ended(tmp_path: Path, p: subprocess.Popen) -> tuple[int, list[str], dict]:
    out, _ = p.communicate()
    assert list((tmp_path / "parent").iterdir()) == [], "the job's cache is still under its parent"
    return p.returncode, out.splitlines(), json.loads((tmp_path / "out/result.json").read_text())


def nobody_reads(fifo: Path) -> bool:
    """Whether no process has ``fifo`` open for reading, or waits to: the pass's processes, blocked on it, are gone."""
    try:
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    except OSError as e:
        return e.errno == errno.ENXIO
    return False


def test_refuses_to_start_under_the_minimum_and_makes_no_cache(tmp_path: Path) -> None:
    (tmp_path / "free").write_text(f"{547 * 10**9 - 1}\n")
    rc, lines, r = ended(tmp_path, job(tmp_path, COMPOSE, "--free-from", str(tmp_path / "free")))
    assert rc == 70 and lines[0].startswith("REFUSED: ") and lines[-1].startswith("CACHE REMOVED (none was made)")
    assert r["exit"] == 70 and r["passes"] == [] and not r["cache"].get("dir")


def test_two_passes_time_the_compose_and_then_find_it_kept(tmp_path: Path) -> None:
    (tmp_path / "free").write_text(f"{547 * 10**9}\n")
    rc, lines, r = ended(tmp_path, job(tmp_path, COMPOSE, "--free-from", str(tmp_path / "free")))
    assert rc == 0 and lines[-1].startswith("CACHE REMOVED (in ") and "PASS 2: 1 of 1 template(s) were in the cache" in lines
    first, second = r["passes"]
    assert [first["rows"][0]["kept"], second["rows"][0]["kept"]] == [False, True] and r["pass2_all_kept"] is True
    assert first["rows"][0] == {"template": "T_v1", "seconds": 0.5, "kept": False, "k_log": 3}
    assert first["cache"] == second["cache"] == r["cache"]["at_removal"] and first["cache"]["circuits"] == 1
    assert first["cache"]["bytes"] == 4096 + 3 and first["warm"]["verifier_window_s"] == 0.1
    assert first["progress"][-1][:3] == [first["wall_s"], 1, first["cache"]["disk_bytes"]]
    assert set(r["free_bytes"]) == {"at_start", "before_pass1", "after_pass1", "after_pass2", "after_removal"}
    assert r["cache"]["removed"] is True and all(p["exit"] == 0 and "wall_s" in p for p in r["passes"])
    assert Path(r["cache"]["dir"]).parent.parent == tmp_path / "parent"


def test_a_low_free_space_reading_in_mid_pass_ends_the_pass_and_removes_the_cache(tmp_path: Path) -> None:
    free, hold = tmp_path / "free", tmp_path / "hold"
    free.write_text(f"{600 * 10**9}\n")
    os.mkfifo(hold)
    # the stand-in itself writes the low reading once its compose has begun: the job reads it on its next look
    p = job(tmp_path, MID_PASS, "--free-from", str(free), HOLD=str(hold), STARTED=str(free), SAY=str(15 * 10**9 - 1))
    rc, lines, r = ended(tmp_path, p)
    assert rc == 71 and r["exit"] == 71 and lines[-1].startswith("CACHE REMOVED (in ")
    assert any(ln.startswith("LOW DISK in pass 1: ") for ln in lines)
    assert r["passes"][0]["exit"] == "ended by this job" and r["passes"][0]["cache"]["bytes"] == 4096
    progress = r["passes"][0]["progress"]  # at the pass's end: no circuit composed yet, the bytes of the one begun
    assert progress[-1][1:3] == [0, r["passes"][0]["cache"]["disk_bytes"]] and all(row[1] == 0 for row in progress)
    assert nobody_reads(hold)


def test_a_sigterm_in_mid_pass_ends_the_pass_and_removes_the_cache(tmp_path: Path) -> None:
    hold, started = tmp_path / "hold", tmp_path / "started"
    os.mkfifo(hold)
    os.mkfifo(started)
    p = job(tmp_path, MID_PASS, "--min-free-gb", "0", "--floor-gb", "0", HOLD=str(hold), STARTED=str(started), SAY="go")
    assert started.read_text() == "go\n"  # returns once the pass has made its cache entry and its worker
    assert len(list((tmp_path / "parent").glob("warm-twice.*/cache/k.tmp*/circuit.txt"))) == 1
    p.send_signal(signal.SIGTERM)
    rc, lines, r = ended(tmp_path, p)
    assert rc == 77 and r["why"] == "ended by SIGTERM" and lines[-1].startswith("CACHE REMOVED (in ")
    assert r["passes"][0]["exit"] == "ended by this job" and len(r["passes"]) == 1
    assert nobody_reads(hold)
