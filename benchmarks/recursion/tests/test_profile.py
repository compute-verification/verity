"""The recursion profile's reader (`rec_profile`) and CLI (`run.py`) on records written in 85-rec-reprice.sh's formats: the
timed sessions' LIVE lines, `/usr/bin/time -v` files, serve's SESSION lines, the replay's UPSTREAM line and Lean's VERDICT.
No GPU, node or store: the job's own records are made here."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import rec_profile as RP
import run as RUN


def timev(wall: str, rss_kb: int = 2 * 2**20, user: float = 1.5, sys_: float = 0.5) -> str:
    return (f"\tUser time (seconds): {user}\n\tSystem time (seconds): {sys_}\n"
            f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {wall}\n\tMaximum resident set size (kbytes): {rss_kb}\n")


def live(t: float, warm: bool, prove: float, e2e: float, accepted=True, proof=1000, witness=0.1) -> str:
    """One stamped LIVE line as 85's `stamp` writes flock-circuit prove's record: time, LIVE, the JSON."""
    rec = {"warm": warm, "accepted": accepted, "prove_s": [prove / 2, prove / 2], "prove_total_s": prove, "e2e_s": e2e,
           "witness_s": witness, "tp_s": prove + 0.05, "verify_ahead": 4,
           "proof_bytes": [proof, proof], "wire_sent": 10, "wire_recv": 20, "k_log": 23, "dense_m": 27, "instances": 2,
           "why": None if accepted else "rec/inner-transcript/v1: nothing verified here",
           "buckets": [{"t.zerocheck": prove * 0.1, "t.ligerito": prove * 0.3, "t.arithmetic": prove * 0.9, "h.setup": 5.0},
                       {"t.zerocheck": prove * 0.1, "t.ligerito": prove * 0.3, "t.arithmetic": prove * 0.9}]}
    return f"{t}\tLIVE\t" + json.dumps(rec) + "\n"


def prover_dir(d: Path, proves=(2.0, 1.0, 3.0), accepted=True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "prove.tsv").write_text(live(100.0, True, 9.0, 9.5) + "".join(
        live(110.0 + 2 * i, False, p, p + 0.5, accepted=accepted) for i, p in enumerate(proves)))
    (d / "time-prove.txt").write_text(timev("0:20.50", 3 * 2**20))
    (d / "gpu-mib.txt").write_text("100\n4000\n2500\n")
    (d / "serve.log").write_text("".join("SESSION\t" + json.dumps({"accepted": True, "verify_s": 0.2}) + "\n" for _ in range(4)))
    (d / "sizes.txt").write_text("123\tinst.bin\n")


def job(tmp: Path) -> Path:
    """A job's out/ with every step: build once, inner and oprove (two modes) once, CPU steps twice."""
    o = tmp / "out"
    (o / "params.txt").parent.mkdir(parents=True)
    (o / "params.txt").write_text("commit=abc123 K=4096 N=2 TAG=t WARM=1 RUNS=3 REPEAT=2\n")

    def rep(step: str, i: int) -> Path:
        r = o / "steps" / step / f"r{i}"
        (r / "out").mkdir(parents=True)
        (r / "rc").write_text("0\n")
        (r / "time-step.txt").write_text(timev(f"1:{10 * i:02d}.00"))
        return r / "out"

    b = rep("build", 1)
    (b / "stage").mkdir()
    (b / "stage" / "time-stage.txt").write_text(timev("2:00.00"))
    (b / "binary.sha256").write_text("ff  flock-circuit\n")
    i = rep("inner", 1)
    for sub in ("proxy", "m0", "zk"):
        prover_dir(i / "inner" / sub)
    p = rep("oprove", 1)
    prover_dir(p / "oprove" / "L0-firewall", (5.0, 6.0, 7.0))
    prover_dir(p / "oprove" / "L0-dev", (1.0, 1.0, 1.0))
    prover_dir(p / "oprove" / "alg-p0-dev", (2.0, 2.0, 2.0), accepted=False)
    for n in (1, 2):
        c = rep("icheck", n)
        (c / "icheck").mkdir()
        (c / "icheck" / "time-replay-proxy.txt").write_text(timev(f"0:0{n}.00"))
        (c / "icheck" / "replay-proxy.txt").write_text("UPSTREAM\t" + json.dumps({"verdict": {"accepted": True}}) + "\n")
        (c / "icheck" / "time-vstar.txt").write_text(timev("0:03.00"))
        v = rep("vstage", n)
        (v / "vstage").mkdir()
        for name in ("points", "fold", "L0", "alg"):
            (v / "vstage" / f"time-{name}.txt").write_text(timev(f"0:1{n}.00"))
        lr = rep("lean", n)
        (lr / "lean").mkdir()
        (lr / "lean" / "time-L0.txt").write_text(timev(f"1:0{n}.00", 9 * 2**20))
        (lr / "lean" / "L0.log").write_text('VERDICT {"accepted": true, "why": "ok"}\n')
        (lr / "lean" / "L0.err").write_text("inputs: 1000 ms\nsetup: 4000 ms\n/x/L0: 64000 ms\n")
        (lr / "lean" / "time-alg-p0.txt").write_text(timev("0:30.00"))
        (lr / "lean" / "alg-p0.log").write_text('VERDICT {"accepted": false, "why": "refused: the batched constraint fails"}\n')
    return o


def row(doc: dict, stage: str) -> dict:
    return next(r for r in doc["rows"] if r["stage"] == stage)


def test_wall_seconds_reads_both_forms() -> None:
    assert RP.wall_seconds("1:02:03") == 3723.0
    assert RP.wall_seconds("0:12.50") == 12.5


def test_stat_is_median_and_range() -> None:
    s = RP.Stat.of([3.0, None, 1.0, 2.0])
    assert (s.median, s.min, s.max, s.values) == (2.0, 1.0, 3.0, (3.0, 1.0, 2.0))
    assert RP.Stat.of([None]) is None


def test_prover_row_counts_timed_sessions_only(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    r = row(doc, "inner M0")
    assert r["kind"] == "gpu" and r["wall_s"]["values"] == [2.0, 1.0, 3.0]          # the warm 9.0 s session left out
    assert r["wall_s"]["median"] == 2.0 and r["e2e_s"]["median"] == 2.5
    assert r["peak_rss_gb"] == 3.0 and r["gpu_peak_mib"] == 4000.0
    assert r["proof_bytes"] == 2000 and r["ok"] is True
    d = r["dominant"]                                     # the median session: prove 2.0, e2e 2.5, witness 0.1
    assert d["phase"] == "t.ligerito" and d["s"] == 1.2 and d["share"] == 0.48 and d["of"] == "e2e_s"
    assert d["phases"]["wait.verifier"] == 0.4 and "t.arithmetic" not in d["phases"] and "h.setup" not in d["phases"]
    assert r["extra"]["serve_verify_s"]["values"] == [0.2, 0.2, 0.2]
    assert r["extra"]["session_s"] == pytest.approx(14.0 / 3)   # last warm verdict at 100, last timed at 114
    assert r["extra"]["witness_s"]["median"] == 0.1


def test_a_session_waiting_on_its_verifier_is_dominated_by_the_wait(tmp_path: Path) -> None:
    d = tmp_path / "p"
    d.mkdir()
    (d / "prove.tsv").write_text(live(1.0, True, 0.3, 6.0) + live(5.0, False, 0.25, 4.0))
    r = RP.prover_row(d, "inner M0", "inner", "x").to_json()
    assert r["dominant"]["phase"] == "wait.verifier" and r["dominant"]["s"] == pytest.approx(3.65)


def test_a_proxy_session_with_no_verdict_is_neither_accepted_nor_refused(tmp_path: Path) -> None:
    d = tmp_path / "p"
    d.mkdir()
    (d / "prove.tsv").write_text(live(1.0, True, 1.0, 2.0, accepted=None) + live(3.0, False, 1.0, 2.0, accepted=None))
    r = RP.prover_row(d, "inner M0", "inner", "x").to_json()
    assert r["ok"] is None and "nothing verified here" in r["note"]


def test_outer_rows_by_mode_and_the_summary(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    fw = next(r for r in doc["rows"] if r["stage"] == "outer L0" and r["extra"]["mode"] == "firewall")
    assert fw["wall_s"]["median"] == 6.0 and fw["extra"]["statement"] == "L0"
    alg = next(r for r in doc["rows"] if r["stage"] == "outer alg-p0")
    assert alg["ok"] is False and alg["extra"]["mode"] == "dev"
    s = doc["summary"]
    assert s["outer"]["firewall"] == {"prove_s": 6.0, "e2e_s": 6.5, "proof_bytes": 2000, "statements": 1}
    assert s["outer"]["dev"]["prove_s"] == 3.0 and s["outer"]["dev"]["statements"] == 2
    assert s["slowest"]["stage"] == "lean L0"                                             # 61.5 s, the largest median
    assert s["pipeline"]["outer prove (firewall) e2e_s"] == 6.5
    assert s["pipeline"]["V* staging_s (vstage + vcap steps)"] == 75.0     # the step's wall, not its parallel stages' sum
    assert s["pipeline"]["Lean step_s"] == 75.0
    assert "outer alg-p0 (dev)" in s["not_ok"] and "lean alg-p0" in s["not_ok"]


def test_cpu_steps_over_their_repeats(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    rep = row(doc, "inner replay")
    assert rep["kind"] == "cpu" and rep["wall_s"]["values"] == [1.0, 2.0] and rep["ok"] is True
    assert row(doc, "stage L0")["wall_s"]["values"] == [11.0, 12.0]
    assert row(doc, "V* points")["wall_s"]["median"] == 11.5
    lean = row(doc, "lean L0")
    assert lean["ok"] is True and lean["peak_rss_gb"] == 9.0
    assert lean["extra"]["setup_s"]["median"] == 4.0 and lean["extra"]["session_s"]["median"] == 60.0
    assert lean["dominant"]["phase"] == "lean.checks" and lean["dominant"]["share"] == round(60 / 61.5, 3)
    refused = row(doc, "lean alg-p0")
    assert refused["ok"] is False and refused["note"].startswith("refused: the batched constraint fails")
    assert row(doc, "lean step")["wall_s"]["values"] == [70.0, 80.0]


def test_a_repeat_still_running_is_no_refusal(tmp_path: Path) -> None:
    o = job(tmp_path)
    r3 = o / "steps" / "lean" / "r3" / "out" / "lean"
    r3.mkdir(parents=True)
    (r3 / "time-L0.txt").write_text("")
    (r3 / "L0.log").write_text("reading inputs\n")
    lean = row(RP.read(o), "lean L0")
    assert lean["ok"] is True and lean["wall_s"]["values"] == [61.0, 62.0]


def test_build_is_once_and_steps_report_exit_codes(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    b = row(doc, "build")
    assert b["kind"] == "setup" and b["wall_s"]["values"] == [70.0] and b["extra"]["stage"]["wall_s"] == 120.0
    assert doc["steps"]["icheck"] == ["0", "0"] and doc["params"]["K"] == "4096"


def test_a_step_that_never_ran_is_not_a_row_and_a_failed_prover_says_so(tmp_path: Path) -> None:
    o = job(tmp_path)
    (o / "steps" / "oprove" / "r1" / "out" / "oprove" / "L1-dev").mkdir()
    (o / "steps" / "oprove" / "r1" / "out" / "oprove" / "L1-dev" / "prove.err").write_text("FIREWALL-REFUSED: the setup\n")
    doc = RP.read(o)
    assert not [r for r in doc["rows"] if r["step"] == "vcap"]
    l1 = row(doc, "outer L1")
    assert l1["ok"] is False and "FIREWALL-REFUSED" in l1["note"]


def test_a_preempted_statement_says_how_often(tmp_path: Path) -> None:
    o = job(tmp_path)
    out = o / "steps" / "oprove" / "r1" / "out"
    (out / "oprove" / "L1-firewall").mkdir()
    (out / "oprove" / "L1-firewall" / "prove.err").write_text("gpu-lease: GPU 1 (GPU-x) held 2m00s\n")
    (out / "preempted.txt").write_text("".join(f"loopback L1-firewall: lease preempted, try {t}\n" for t in (1, 2, 3)))
    l1 = row(RP.read(o), "outer L1")
    assert l1["ok"] is False and l1["extra"]["preempted"] == 3
    assert l1["note"] == "lease preempted 3x; no session finished in the lease's 2m00s"


def test_markdown_has_one_line_per_row(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    md = RP.markdown(doc)
    table = [ln for ln in md.splitlines() if ln.startswith("| ") and not ln.startswith("| stage | step |")]
    assert len(table) == len(doc["rows"])
    assert "| inner M0 | inner |" in md and "2 [1–3] (n=3)" in md and "t.ligerito 1.2 s (48% of e2e_s)" in md
    assert "| 4.67 |" in md
    assert "Slowest repeated stage: lean L0" in md


def test_result_command_writes_both_files(tmp_path: Path) -> None:
    o = job(tmp_path)
    assert RUN.main(["profile", "result", "--dir", str(o), "--out", str(tmp_path / "p.json"), "--md", str(tmp_path / "t.md")]) == 0
    doc = json.loads((tmp_path / "p.json").read_text())
    assert doc["schema"] == RP.SCHEMA and (tmp_path / "t.md").read_text().startswith("Recursion profile: K = 4096")


def test_job_command_is_a_gpu_host_job_on_node_1(capsys) -> None:
    assert RUN.main(["profile", "run", "--dry-run", "--N", "16", "--steps", "lean,build"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[:6] == ["research", "run", "--queue", "--on", "vy-nebius-1", "--gpu-host"]
    assert "--gpus" not in cmd                              # 85 refuses a job holding a lease: each prove leases its own
    assert cmd[cmd.index("--") + 1:][:2] == ["bash", "benchmarks/recursion/job.sh"]
    assert "N=16" in cmd and "STEPS=build,lean" in cmd     # the pipeline's order, whatever order --steps gives


def test_env_passes_through_and_must_be_key_value(capsys) -> None:
    assert RUN.main(["profile", "run", "--dry-run", "--env", "LEVELS=L0", "--env", "GPU_PREEMPTIBLE="]) == 0
    tail = capsys.readouterr().out.split(" -- ")[1].split()
    assert tail[-2:] == ["LEVELS=L0", "GPU_PREEMPTIBLE="]
    with pytest.raises(SystemExit, match="not KEY=VALUE"):
        RUN.main(["profile", "run", "--dry-run", "--env", "levels"])


def test_unknown_step_is_refused() -> None:
    with pytest.raises(SystemExit, match="ostage"):
        RUN.main(["profile", "plan", "--steps", "build,ostage"])


def test_merge_takes_a_later_runs_row_for_the_same_step_stage_and_mode(tmp_path: Path, capsys) -> None:
    a = RP.read(job(tmp_path / "a")) | {"ref": "rA"}
    o = job(tmp_path / "b")
    prover_dir(o / "steps" / "oprove" / "r1" / "out" / "oprove" / "L0-dev", (4.0, 4.0, 4.0))
    b = RP.read(o) | {"ref": "rB"}
    b["rows"] = [r for r in b["rows"] if r["step"] == "oprove"]
    m = RP.merge([a, b])
    l0 = [r for r in m["rows"] if r["stage"] == "outer L0"]
    assert {(r["extra"]["mode"], r["extra"]["run"], r["wall_s"]["median"]) for r in l0} == {("firewall", "rB", 6.0), ("dev", "rB", 4.0)}
    assert next(r for r in m["rows"] if r["stage"] == "build")["extra"]["run"] == "rA"
    assert [r["step"] for r in m["rows"]].index("oprove") > [r["step"] for r in m["rows"]].index("vstage")
    assert m["summary"]["outer"]["dev"]["prove_s"] == 6.0          # rB's L0 4.0 and alg-p0's 2.0
    for name, d in (("a.json", a), ("b.json", b)):
        (tmp_path / name).write_text(json.dumps(d))
    assert RUN.main(["profile", "table", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--merge"]) == 0
    md = capsys.readouterr().out
    assert " run |" in md and "| rB |" in md and "| rA |" in md and "rA: commit abc123, 3 timed sessions" in md


def test_a_run_id_is_read_from_its_records_not_its_profile_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(RUN.subprocess, "run", lambda *a, **k: None)
    out = tmp_path / ".research" / "runs" / "r20261024-000000-abcd"
    job(out)
    (out / "out" / "profile.json").write_text(json.dumps({"schema": RP.SCHEMA, "rows": [], "stale": True}))
    doc = RUN.load("r20261024-000000-abcd")
    assert "stale" not in doc and doc["run"] == "r20261024-000000-abcd" and row(doc, "inner M0")["ok"] is True


def test_table_refuses_another_schema(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps({"schema": "proof-throughput/v1"}))
    with pytest.raises(SystemExit, match="not recursion-profile/v1"):
        RUN.main(["profile", "table", str(tmp_path / "x.json")])
