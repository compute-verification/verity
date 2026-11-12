"""`warm_twice.py`'s guards and its cache's removal, on a stand-in for `warm`: a shell command that composes one small "circuit"
into the cache it is handed and, on its second pass, finds it kept. The free space the job reads comes from a file
(``--free-from``), and a stand-in that must still be running when something happens blocks on a FIFO nobody writes, so no
test waits on a clock. Under a free-space minimum the job starts nothing; two passes time the compose and then the kept
circuit; a free-space reading under the floor in mid-pass, and a SIGTERM in mid-pass, each end the pass's processes; and on
each way out the cache's parent is empty again and the last line of output says so. With ``--then-window`` the second pass
gets the second window and the cache the first filled, and ``--take`` runs the take on the last window after it. The take
itself (``warm_twice.py take``) runs on a toy tree whose records_audit.py hands it toy windows and whose cache a5's own warm
filled: a window of the entries' partition and program takes each as a link, another's takes each as the copy bound to its
own, a template the cache doesn't hold is taken as none and composed by nobody, and a copy the disk has no room for is not
started."""
from __future__ import annotations

import errno
import inspect
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


#: a warm that keeps two templates in the cache, `one` and its window's (named by the window's directory), and a take that
#: says how it took the window's
BY_WINDOW = r'''c="$1"; w="$(basename "$2")"
if [ "$3" = take ]; then
  echo "{\"template\": \"T_$w\", \"seconds\": 0.1, \"kept\": true, \"k_log\": 3, \"taken\": \"copy\", \"bytes\": 4096, \"bound_here\": false, \"meta_is_the_lines\": true, \"bound_to_window\": true}"
  echo "TAKE {\"templates\": 1, \"copy\": 1}"; exit 0
fi
for t in one "$w"; do
  if [ -f "$c/$t/made.json" ]; then kept=true; else
    kept=false; mkdir "$c/$t.tmp$$" && head -c 4096 /dev/zero > "$c/$t.tmp$$/circuit.txt" && echo "{}" > "$c/$t.tmp$$/made.json" \
      && mv "$c/$t.tmp$$" "$c/$t"
  fi
  echo "{\"template\": \"T_$t\", \"seconds\": 0.5, \"kept\": $kept, \"k_log\": 3, \"pin\": \"p\"}"
done
'''


def test_a_second_window_is_warmed_over_the_first_ones_cache_and_then_taken(tmp_path: Path) -> None:
    (tmp_path / "free").write_text(f"{547 * 10**9}\n")
    p = job(tmp_path, BY_WINDOW, "--free-from", str(tmp_path / "free"), "--then-window", str(tmp_path / "two"), "--take")
    rc, lines, r = ended(tmp_path, p)
    assert rc == 0 and lines[-1].startswith("CACHE REMOVED (in ") and "both passes ran and the take" in lines
    first, second, took = r["passes"]
    assert [(x["template"], x["kept"]) for x in first["rows"]] == [("T_one", False), ("T_no-window", False)]
    assert [(x["template"], x["kept"]) for x in second["rows"]] == [("T_one", True), ("T_two", False)]
    assert r["second_window"] is True and r["pass2_all_kept"] is False and second["cache"]["circuits"] == 3
    assert "PASS 2: 1 of 2 template(s) were in the cache" in lines
    assert "TAKE: 0 link, 1 copy, 0 no room, True as_the_audit_reads" in lines
    assert took["rows"] == [{"template": "T_two", "seconds": 0.1, "kept": True, "k_log": 3, "taken": "copy", "bytes": 4096,
                             "bound_here": False, "meta_is_the_lines": True, "bound_to_window": True}]
    assert took["take"] == {"templates": 1, "copy": 1} and took["cache"] == second["cache"]
    assert r["take"] == {"link": 0, "copy": 1, "no room": 0, "as_the_audit_reads": True} and "after_take" in r["free_bytes"]
    # before pass 1 the second window was read as the take reads it, with the cache still empty
    assert r["second_window_read"]["pass"] == 0 and r["second_window_read"]["cache"]["circuits"] == 0
    assert r["second_window_read"]["command"].split()[-2:] == [str(tmp_path / "two"), "take"]


def test_a_second_window_the_take_cannot_read_is_refused_before_anything_is_composed(tmp_path: Path) -> None:
    stand_in = 'if [ "$3" = take ]; then exit 9; fi\n' + COMPOSE
    rc, lines, r = ended(tmp_path, job(tmp_path, stand_in, "--min-free-gb", "0", "--then-window", str(tmp_path / "two"), "--take"))
    assert rc == 64 and r["passes"] == [] and lines[-1].startswith("CACHE REMOVED (in ")
    assert any(ln.startswith("REFUSED: the second window isn't read here, and nothing was composed: PASS 0 FAILED") for ln in lines)
    assert r["cache"]["at_removal"]["circuits"] == 0 and r["second_window_read"]["exit"] == 9


def test_a_tryout_ended_early_still_reads_the_second_window_and_takes(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "hold")
    # a warm that keeps one circuit, if the cache has none, and then composes on for good: only the job's stop ends it
    stand_in = BY_WINDOW.replace('for t in one "$w"; do', 'for t in one; do') + 'read -r _ < "$HOLD"\n'
    p = job(tmp_path, stand_in, "--min-free-gb", "0", "--floor-gb", "0", "--then-window", str(tmp_path / "two"), "--take",
            "--stop-at-templates", "1", HOLD=str(tmp_path / "hold"))
    rc, _lines, r = ended(tmp_path, p)
    assert rc == 0 and [x["state"] for x in r["passes"]] == ["stopped early", "stopped early", "done"]
    assert r["second_window_read"]["exit"] == 0 and r["passes"][2]["take"] == {"templates": 1, "copy": 1} and r["take"]["copy"] == 1
    assert nobody_reads(tmp_path / "hold")


def test_a_second_window_without_a_second_pass_is_refused(tmp_path: Path) -> None:
    p = job(tmp_path, COMPOSE, "--min-free-gb", "0", "--then-window", str(tmp_path / "two"), "--passes", "1")
    rc, lines, r = ended(tmp_path, p)
    assert rc == 64 and lines[0].startswith("REFUSED: --then-window") and r["passes"] == []


def test_a_take_that_fails_is_its_own_exit_and_the_cache_is_removed(tmp_path: Path) -> None:
    p = job(tmp_path, 'if [ "$3" = take ]; then exit 9; fi\n' + COMPOSE, "--min-free-gb", "0", "--floor-gb", "0", "--take")
    rc, lines, r = ended(tmp_path, p)
    assert rc == 75 and "PASS 3 FAILED: exit 9 (pass3.err)" in lines and lines[-1].startswith("CACHE REMOVED (in ")
    assert [x["exit"] for x in r["passes"]] == [0, 0, 9] and "take" not in r


def _toy_circuit(width: int, partition: str, program: str, rule: str) -> bytes:
    """A toy template's circuit bound to a partition and a program: its META holds the binding as `circuit.compose` writes
    it, once each."""
    meta = json.dumps({"k_log": width, "partition": {"digest": partition, "rule": rule}, "program_sha512": program},
                      sort_keys=True, separators=(",", ":"))
    return f"toy circuit {width}\nMETA {meta}\n".encode() + b"g\n" * width


#: a tree's records_audit.py as the take reads it, and its `warm` as the job starts it, over the toy window its directory's
#: toy.json states (its partition, program and templates, each by the width of its Definition and statement), through a5's
#: own cache; each compose a composer makes is a line of $COMPOSED
TOY_RECORDS_AUDIT = '''import hashlib, importlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, os.environ["ONE_STAGE"])
import a5
from verity_flock import circuit as FC

circuits = a5.Program.__module__.rsplit(".", 1)[0]  # the circuits package, by the name a5 imports it under
BIT = importlib.import_module(circuits + ".boolean").BIT
definition = importlib.import_module(circuits + ".boolean.trace").definition
bind = importlib.import_module(circuits + ".defs").bind
Array = importlib.import_module(circuits + ".types").Array
D = SimpleNamespace(parse_law=lambda law: law)


''' + inspect.getsource(_toy_circuit) + '''

def toy(width):
    defn = bind(definition(f"WarmTwiceToy{width}", 1, lambda C, a: [C.AND(a[0], a[1])], (("a", Array(width, BIT)),),
                           Array(1, BIT), register=False))

    def compose(out, partition, program):
        with open(os.environ["COMPOSED"], "a") as f:
            f.write(f"{width}\\n")
        body = _toy_circuit(width, partition, program, FC.PARTITION_V1)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(body)
        return {"pin": hashlib.sha512(body).hexdigest(), "class_sha512": hashlib.sha512(b"toy %d" % width).hexdigest(),
                "k_log": width}
    compose.inputs = ["unit/window", {"width": width}]
    return compose, defn


def verifier_window(window_dir, law, template_sets):
    w = json.loads((Path(window_dir) / "toy.json").read_text())
    mine = SimpleNamespace(members=[SimpleNamespace(template=t) for t in w["templates"]], partition_digest=w["partition"],
                           partition={"program": w["program"]})
    return mine, {}, {t: toy(width) for t, width in w["templates"].items()}


if __name__ == "__main__":  # warm --window W --law L --compose-cache C --jobs N
    arg = lambda name: sys.argv[sys.argv.index(name) + 1]
    mine, _doc, sets = verifier_window(arg("--window"), None, None)
    for g in a5.warm(SimpleNamespace(compose_cache=arg("--compose-cache")), sets, mine, int(arg("--jobs"))):
        print(json.dumps(g, sort_keys=True))
'''
B1 = {"partition": "a1" * 64, "program": "b2" * 64}
B2 = {"partition": "c3" * 64, "program": "d4" * 64}


def toy_window(tmp_path: Path, name: str, binding: dict, templates: dict) -> Path:
    (tmp_path / name).mkdir()
    (tmp_path / name / "toy.json").write_text(json.dumps({**binding, "templates": templates}))
    return tmp_path / name


def toy_tree(tmp_path: Path) -> dict:
    """A toy tree, and the cache its warm filled from window ``one``, two templates under B1 -> the environment its programs
    run in."""
    (tmp_path / "tree/benchmarks/prove_path").mkdir(parents=True)
    (tmp_path / "tree/benchmarks/prove_path/records_audit.py").write_text(TOY_RECORDS_AUDIT)
    env = {**os.environ, "ONE_STAGE": str(JOB.parents[1] / "one_stage"), "COMPOSED": str(tmp_path / "composed")}
    one = toy_window(tmp_path, "one", B1, {"Toy{N=1}": 16, "Toy{N=2}": 32})
    subprocess.run([sys.executable, "benchmarks/prove_path/records_audit.py", "warm", "--window", str(one), "--law", "subset:1",
                    "--compose-cache", str(tmp_path / "cache"), "--jobs", "2"], cwd=tmp_path / "tree", env=env, check=True,
                   stdout=subprocess.DEVNULL)
    assert sorted((tmp_path / "composed").read_text().split()) == ["16", "32"]
    return env


def files(root: Path) -> list:
    return sorted((str(f.relative_to(root)), f.read_bytes()) for f in root.rglob("*") if f.is_file())


def took(tmp_path: Path, env: dict, window: Path, *more: str) -> tuple[dict, dict]:
    """``warm_twice.py take`` of ``window`` from the toy tree's cache, which it leaves as it was, composing nothing and
    leaving nothing in its scratch directory -> each template's line, and the ``TAKE`` line."""
    before = files(tmp_path / "cache")
    out = subprocess.run([sys.executable, str(JOB), "take", "--window", str(window), "--law", "subset:1", "--compose-cache",
                          str(tmp_path / "cache"), "--scratch", str(tmp_path / "taken"), *more], cwd=tmp_path / "tree", env=env,
                         check=True, stdout=subprocess.PIPE, text=True).stdout.splitlines()
    assert list((tmp_path / "taken").iterdir()) == [] and files(tmp_path / "cache") == before
    assert sorted((tmp_path / "composed").read_text().split()) == ["16", "32"], "the take composed"
    assert out[-1].startswith("TAKE ")
    return {r["template"]: r for r in map(json.loads, out[:-1])}, json.loads(out[-1][5:])


def test_the_take_links_under_the_entries_binding_and_copies_bound_to_another_windows(tmp_path: Path) -> None:
    env = toy_tree(tmp_path)
    counts = lambda total: [total[how] for how in ("link", "copy", "none", "no room")]  # noqa: E731
    rows, total = took(tmp_path, env, tmp_path / "one")
    assert {t: (r["kept"], r["taken"], r["bound_here"]) for t, r in rows.items()} == {"Toy{N=1}": (True, "link", True),
                                                                                      "Toy{N=2}": (True, "link", True)}
    assert all(r["meta_is_the_lines"] and r["bound_to_window"] and r["bytes"] > 0 for r in rows.values())
    assert counts(total) == [2, 0, 0, 0]
    # another window's partition and program, with one template the cache holds and one it doesn't
    other = toy_window(tmp_path, "two", B2, {"Toy{N=2}": 32, "Toy{N=3}": 64})
    rows, total = took(tmp_path, env, other, "--floor-gb", "0")
    assert (rows["Toy{N=2}"]["kept"], rows["Toy{N=2}"]["taken"], rows["Toy{N=2}"]["bound_here"]) == (True, "copy", False)
    assert rows["Toy{N=2}"]["meta_is_the_lines"] and rows["Toy{N=2}"]["bound_to_window"]
    assert rows["Toy{N=3}"] == {"template": "Toy{N=3}", "kept": False, "taken": "none"} and counts(total) == [0, 1, 1, 0]
    # a floor the file system doesn't have free: the copy isn't started, and a link, which takes no room, still is
    rows, total = took(tmp_path, env, other, "--floor-gb", "1e12")
    assert rows["Toy{N=2}"]["taken"] == "no room" and "seconds" not in rows["Toy{N=2}"] and counts(total) == [0, 0, 1, 1]
    rows, total = took(tmp_path, env, tmp_path / "one", "--floor-gb", "1e12")
    assert [r["taken"] for r in rows.values()] == ["link", "link"]
