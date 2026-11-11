"""window.py's `prove` end to end on a stub prove_time.sh, a rehearsal of a served card's window off the GPU node: the stub
plays each statement a group is handed (its start, sessions, exit status and done files, and the group's summary), so the
test holds what window.py does around the prover. It balances the statements over the groups by the model or by an earlier
prove's seconds (WIN_SECONDS, where a statement its prover didn't finish with exit 0 goes first), runs each group on its own
cores with settings.py's defaults, links a WIN_ALSO statement, keeps each ended statement's KEEP files and WIN_KEEP_ALSO's
beside its prover's exit status (never its circuit or witness), writes seconds.tsv and prove.tsv, and removes FLOCK_WORK and
not what its links point to. The clock is fixed and `sleep` waits on the groups' exits."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import load

NOW, START, DONE = 1000.0, 100.0, 160.5

STUB = r'''
import json, os, signal, sys
from pathlib import Path
a = dict(x.split("=", 1) for x in sys.argv[1:])
out, fw = Path(os.environ["RESEARCH_RUN_DIR"]) / "out", Path(os.environ["FLOCK_WORK"])
plan = json.loads(Path(os.environ["STUB_PLAN"]).read_text())
out.mkdir(parents=True, exist_ok=True)
(out / "args.json").write_text(json.dumps({"args": a, "cpus": sorted(os.sched_getaffinity(0)),
                                           "who": os.environ.get("GPU_LEASE_WHO")}))
summary = {}
for spec in a["PT_UNITS"].split():
    u, _, ns = spec.rpartition("@")
    for n in ns.split(","):
        nm, rc = plan[f"{u}@{n}"]
        o, d = out / a["PT_DEVICE"] / nm, fw / "data" / f"{nm}-k0"
        o.mkdir(parents=True)
        (o / "start").write_text(f"{os.environ['STUB_START']}\n")
        for s in ("s0", "s1"):
            (d / "sessions" / s).mkdir(parents=True)
            for f in ("session.json", "circuit.rep0.bin", "verifier.json", "witness.bin"):
                (d / "sessions" / s / f).write_text(f)
        (d / "sessions" / "index.jsonl").write_text("{}\n")
        if rc is None:
            os.kill(os.getpid(), signal.SIGKILL)
        (o / "exit").write_text(f"{rc}\n")
        (o / "done").write_text(f"{os.environ['STUB_DONE']}\n")
        summary[nm] = {"n": n, "k_log": 20, "prove_total_s": 60.5, "rc": rc}
(out / "summary.json").write_text(json.dumps({"statements": summary}))
'''

#: UNITS_FILE's statements: (unit, its sizes, the circuit's k_log, staged under FLOCK_WORK or WIN_ALSO or not at all), and
#: each statement's prover exit status in the stub (None: its group is killed while it runs)
UNITS = [("census:A_v1{X=1}", ("full/4", "full"), 26, "fw"), ("gq:K=2048", ("full/4",), 23, "fw"),
         ("census:B_v1{Y=2}", ("full",), 20, "fw"), ("census:C_v1{}", ("full",), 20, None),
         ("census:D_v1{Z=3}", ("full",), 24, "also")]
RC = {"census:A_v1{X=1}": 0, "gq:K=2048": 137, "census:B_v1{Y=2}": None, "census:D_v1{Z=3}": 0}
STAGED = {"stage.json", "pub.bin", "pin.txt", "partition.json", "tables/t.bin", "inst.bin", "circuit.txt", "extra.txt",
          "other.txt"}
KEPT = {"stage.json", "pub.bin", "pin.txt", "partition.json", "tables/t.bin", "extra.txt", "exit", "sessions/index.jsonl",
        *(f"sessions/{s}/{f}" for s in ("s0", "s1") for f in ("session.json", "circuit.rep0.bin", "verifier.json"))}


def files(d: Path) -> set[str]:
    return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}


def rehearse(w, mp: pytest.MonkeyPatch, root: Path, seconds: str = "") -> tuple[int, Path, dict]:
    """One `window.py prove` under `root`: FLOCK_WORK and a WIN_ALSO staged per UNITS, the stub as prove_time.sh, two
    groups; returns its exit, its out and each statement's name."""
    fw, also, out = root / "fw", root / "also", root / "run" / "out"
    names = {f"{u}@{n}": w.name(u, n) for u, ns, _k, _w in UNITS for n in ns}
    for u, ns, k_log, where in UNITS:
        for n in ns if where else ():
            d = (fw if where == "fw" else also) / "data" / f"{names[f'{u}@{n}']}-k0"
            for f in STAGED:
                (d / f).parent.mkdir(parents=True, exist_ok=True)
                (d / f).write_text(f)
            (d / "stage.json").write_text(json.dumps({"circuit_bytes": 1000, "k_log": k_log}))
            (d / ".done").write_text("")
    (fw / "flock-gpu-link/flock/target/release").mkdir(parents=True)
    (fw / "flock-gpu-link/flock/target/release/flock-circuit").write_text("")
    (root / "stub.py").write_text(STUB)
    (root / "stub.sh").write_text('exec "$STUB_PY" "$STUB" "$@"\n')
    (root / "plan.json").write_text(json.dumps({un: (nm, RC.get(un.rpartition("@")[0])) for un, nm in names.items()}))
    (root / "units.tsv").write_text("# the plan\n" + "".join(f"{u}@{','.join(ns)}\t1\t1\t1\t1\n" for u, ns, *_x in UNITS))
    out.mkdir(parents=True)

    started = []

    def start(*a, **k):
        p = real_start(*a, **k)
        started.append(p)
        return p

    real_start = w.start
    clock = SimpleNamespace(time=lambda: NOW, sleep=lambda _s: [p.wait() for p in started],
                            strftime=time.strftime, gmtime=lambda *_a: time.gmtime(NOW), strptime=time.strptime)
    mp.setattr(w, "start", start)
    mp.setattr(w, "time", clock)
    mp.setattr(w, "OUT", out)
    mp.setattr(w, "RAM", f"{root}/")
    mp.setattr(w, "PT", root / "stub.sh")
    mp.setattr(w, "key", lambda: "k0")
    mp.setattr(w.tempfile, "tempdir", str(root))
    env = {"FLOCK_WORK": str(fw), "WIN_ALSO": str(also), "WIN_CPUS": str(len(os.sched_getaffinity(0))),
           "WIN_DEVICE": "cpu", "WIN_GPUS": "2", "WIN_KEEP_ALSO": "extra.txt", "WIN_SECONDS": seconds,
           "STUB_PY": sys.executable, "STUB": str(root / "stub.py"), "STUB_PLAN": str(root / "plan.json"),
           "STUB_START": str(START), "STUB_DONE": str(DONE)}
    for k, v in env.items():
        mp.setenv(k, v)
    return w.prove(w.read_units(root / "units.tsv")), out, names


@pytest.fixture
def w(monkeypatch):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("two groups need two cores in the affinity")
    w = load(monkeypatch, "window.py")
    for k in load(monkeypatch, "settings.py").DECLARED:
        if k != "PATH":
            monkeypatch.delenv(k, raising=False)
    return w


def test_prove(w, monkeypatch, tmp_path):
    rc, out, nm = rehearse(w, monkeypatch, tmp_path)
    assert rc == 0
    a, a4, gq, b, d = (nm[x] for x in ("census:A_v1{X=1}@full", "census:A_v1{X=1}@full/4", "gq:K=2048@full/4",
                                       "census:B_v1{Y=2}@full", "census:D_v1{Z=3}@full"))
    g0, g1 = (json.loads((out / f"gpu{g}/out/args.json").read_text()) for g in (0, 1))
    assert g0["args"]["PT_UNITS"] == "census:A_v1{X=1}@full/4,full"
    assert g1["args"]["PT_UNITS"] == "census:D_v1{Z=3}@full gq:K=2048@full/4 census:B_v1{Y=2}@full"
    cpus = sorted(os.sched_getaffinity(0))
    per = len(cpus) // 2
    assert (g0["cpus"], g1["cpus"]) == (cpus[:per], cpus[per:2 * per])
    assert {k: g0["args"][k] for k in ("STEPS", "PT_DEVICE", "WARM", "RUNS", "VCPUS", "LEASE_MIN")} == \
        {"STEPS": "prove", "PT_DEVICE": "cpu", "WARM": "1", "RUNS": "1", "VCPUS": "2", "LEASE_MIN": "30"}
    assert g0["who"] == g1["who"] == "servedcard"

    rows = [ln.split("\t") for ln in (out / "seconds.tsv").read_text().splitlines()]
    assert rows == [["name", "group", "seconds", "done", "exit"],
                    [a, "gpu0", "60.5", "1", "0"], [a4, "gpu0", "60.5", "1", "0"],
                    [b, "gpu1", f"{NOW - START:.1f}", "0", ""], [d, "gpu1", "60.5", "1", "0"],
                    [gq, "gpu1", "60.5", "0", "137"]]
    assert sorted(p.name for p in (out / "kept").iterdir()) == sorted([a, a4, d, gq])
    for k in (a, a4, d, gq):
        assert files(out / "kept" / k) == KEPT, k
    assert (out / "kept" / gq / "exit").read_text() == "137\n" and (out / "kept" / a / "exit").read_text() == "0\n"
    prove = [ln.split("\t")[:5] for ln in (out / "prove.tsv").read_text().splitlines()]
    assert prove == [["group", "statement", "n", "k_log", "prove_total_s"], ["gpu0", a4, "full/4", "20", "60.5"],
                     ["gpu0", a, "full", "20", "60.5"]]

    log = (out / "window.log").read_text()
    assert "not staged, skipped: census:C_v1{}" in log and "linked 1 statements staged under" in log
    assert "3 of 5 started statements finished with exit 0, 1 ended otherwise" in log
    assert not (tmp_path / "fw").exists()
    assert files(tmp_path / "also" / "data" / f"{d}-k0") == STAGED | {".done"}, "removing FLOCK_WORK reached WIN_ALSO's files"


def test_retry_balances_by_seconds(w, monkeypatch, tmp_path):
    """A second prove reading the first's seconds.tsv: the statements whose prover was killed (exit 137) or whose group
    ended under them are estimated at WIN_UNMEASURED_S and start first in their groups."""
    rehearse(w, monkeypatch, tmp_path / "first")
    rc, out, _nm = rehearse(w, monkeypatch, tmp_path / "retry", str(tmp_path / "first/run/out/seconds.tsv"))
    assert rc == 0
    g0, g1 = (json.loads((out / f"gpu{g}/out/args.json").read_text())["args"]["PT_UNITS"] for g in (0, 1))
    assert (g0, g1) == ("gq:K=2048@full/4 census:A_v1{X=1}@full/4,full", "census:B_v1{Y=2}@full census:D_v1{Z=3}@full")
    assert "estimates: 2 of 4 types measured in" in (out / "window.log").read_text()


def test_read_seconds(w, tmp_path):
    """seconds.tsv with its exit column, and a prove's from before it (four columns), read alike."""
    (tmp_path / "s5.tsv").write_text("name\tgroup\tseconds\tdone\texit\na\tgpu0\t60.5\t1\t0\nb\tgpu1\t7.0\t0\t137\n"
                                     "c\tgpu1\t9.0\t0\t\n")
    (tmp_path / "s4.tsv").write_text("name\tgroup\tseconds\tdone\na\tgpu0\t60.5\t1\nb\tgpu1\t7.0\t0\n")
    assert w.read_seconds(str(tmp_path / "s5.tsv")) == {"a": (60.5, True), "b": (7.0, False), "c": (9.0, False)}
    assert w.read_seconds(str(tmp_path / "s4.tsv")) == {"a": (60.5, True), "b": (7.0, False)}
    assert w.read_seconds("") == {}
