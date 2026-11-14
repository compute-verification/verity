"""The recursion bench's reader (`rec_profile`), table (`rec_table`) and CLI (`run.py`) on records written in
85-rec-reprice.sh's formats: the timed runs' LIVE lines, `/usr/bin/time -v` files, serve's SESSION lines, the replay's
UPSTREAM line and Lean's VERDICT. No GPU, node or store: the job's own records are made here, by `job`, which
`test_run.py` (the CLI's launches and commands) and `test_headline.py` (an accepted recursive run) use too."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import rec_logs as RL
import rec_profile as RP
import rec_rows as RR
import rec_table as RT
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
    assert RL.wall_seconds("1:02:03") == 3723.0
    assert RL.wall_seconds("0:12.50") == 12.5


def test_stat_is_median_and_range() -> None:
    s = RR.Stat.of([3.0, None, 1.0, 2.0])
    assert s is not None
    assert (s.median, s.min, s.max, s.values) == (2.0, 1.0, 3.0, (3.0, 1.0, 2.0))
    assert RR.Stat.of([None]) is None


def test_prover_row_counts_timed_runs_only(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    r = row(doc, "inner M0")
    assert r["kind"] == "gpu" and r["wall_s"]["values"] == [2.0, 1.0, 3.0]          # the warm 9.0 s run left out
    assert r["wall_s"]["median"] == 2.0 and r["e2e_s"]["median"] == 2.5
    assert r["peak_rss_gb"] == 3.0 and r["gpu_peak_mib"] == 4000.0
    assert r["proof_bytes"] == 2000 and r["ok"] is True
    d = r["dominant"]                                     # the median run: prove 2.0, e2e 2.5, witness 0.1
    assert d["phase"] == "t.ligerito" and d["s"] == 1.2 and d["share"] == 0.48 and d["of"] == "e2e_s"
    assert d["phases"]["wait.verifier"] == 0.4 and "t.arithmetic" not in d["phases"] and "h.setup" not in d["phases"]
    assert r["extra"]["serve_verify_s"]["values"] == [0.2, 0.2, 0.2]
    assert r["extra"]["session_s"] == pytest.approx(14.0 / 3)   # last warm verdict at 100, last timed at 114
    assert r["extra"]["witness_s"]["median"] == 0.1


def test_a_run_waiting_on_its_verifier_is_dominated_by_the_wait(tmp_path: Path) -> None:
    d = tmp_path / "p"
    d.mkdir()
    (d / "prove.tsv").write_text(live(1.0, True, 0.3, 6.0) + live(5.0, False, 0.25, 4.0))
    r = RP.prover_row(d, "inner M0", "inner", "x").to_json()
    assert r["dominant"]["phase"] == "wait.verifier" and r["dominant"]["s"] == pytest.approx(3.65)


def test_a_proxy_run_with_no_verdict_is_neither_accepted_nor_refused(tmp_path: Path) -> None:
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
    assert s["pipeline"]["outer prove (dev) e2e_s"] == 4.0    # dev proved both statements, ZK server one
    assert s["pipeline"]["V* staging_s (vstage + vcap steps)"] == 75.0     # the step's wall, not its parallel stagings' sum
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
    assert l1["note"] == "lease preempted 3x; no run finished in the lease's 2m00s"


def test_the_lean_firewall_is_read_from_the_runs_process_samples(tmp_path: Path) -> None:
    o = job(tmp_path)

    def sample(t: str, cpu: float) -> str:
        procs = [{"pid": 7, "comm": "flock-firewall", "cpu_s": cpu, "rss_hwm_bytes": 8 * 2**30, "threads": 203},
                 {"pid": 8, "comm": "flock-firewall", "cpu_s": 0.1, "rss_hwm_bytes": 0, "threads": 4},
                 {"pid": 9, "comm": "flock-circuit", "cpu_s": 16.0}]
        return json.dumps({"kind": "process", "t_utc": t, "processes": procs}, separators=(",", ":")) + "\n"
    (tmp_path / "resources.jsonl").write_text(sample("2026-10-24T07:17:46.6Z", 2.0) + sample("2026-10-24T07:26:27.6Z", 1177.0))
    doc = RP.read(o)
    assert doc["summary"]["flock_firewall"] == [{"pid": 7, "first": "2026-10-24T07:17:46.6Z", "last": "2026-10-24T07:26:27.6Z",
                                                 "cpu_s": 1177.0, "rss_hwm_gb": 8.0, "threads": 203}]
    assert "flock-firewall" in next(r for r in doc["rows"] if r["stage"] == "outer L0" and r["extra"]["mode"] == "firewall")["note"]
    assert "1177 CPU s, peak RSS 8.0 GB, 203 threads, seen 07:17:46–07:26:27Z" in RT.markdown(doc)


def test_markdown_has_one_line_per_row(tmp_path: Path) -> None:
    doc = RP.read(job(tmp_path))
    md = RT.markdown(doc)
    table = [ln for ln in md.split("\nOuter")[0].splitlines() if ln.startswith("| ") and not ln.startswith("| stage | step |")]
    assert len(table) == len(doc["rows"])
    assert "| inner M0 | inner |" in md and "2 [1–3] (n=3)" in md and "t.ligerito 1.2 s (48% of e2e_s)" in md
    assert "| 4.67 |" in md
    assert "Slowest row (a run's e2e, else a tool's wall): lean L0" in md


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
    assert RUN.main(["table", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--merge"]) == 0
    md = capsys.readouterr().out
    assert " run |" in md and "| rB |" in md and "| rA |" in md and "rA: commit abc123, the inner's 3 timed runs" in md


def test_a_cancelled_run_is_read_from_its_records(tmp_path: Path, monkeypatch, capsys) -> None:
    """`research cancel` ends the scope before job.sh's trap writes a result: `table RUN` reads the records it left."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(RUN.subprocess, "run", lambda *a, **k: pytest.fail("load ran a command"))
    o = job(tmp_path / ".research" / "runs" / "r20261024-000000-cafe")
    for step in ("icheck", "vstage", "lean"):
        shutil.rmtree(o / "steps" / step)
    stopped = o / "steps" / "oprove" / "r1" / "out" / "oprove" / "L1-firewall"
    stopped.mkdir()
    (stopped / "prove.err").write_text("ZKRANK\t{}\n")
    (o / "steps" / "oprove" / "r1" / "rc").unlink()
    (o / "result.err").write_text("")
    assert not (o / "recursion_e2e.json").exists()
    assert RUN.main(["table", "r20261024-000000-cafe"]) == 0
    md = capsys.readouterr().out
    assert "## r20261024-000000-cafe" in md and "Verdict: **not a whole recursive run**" in md
    assert "| outer L1 | oprove |" in md and "no run finished before the step stopped" in md
    assert "oprove ?" in md


def negative(o: Path, refusals: dict[str, dict[str, bool | None]]) -> None:
    """The negative step's lean records: each honest statement, then each coin's forged copy of it (85's
    ``forged-inner-COIN-STATEMENT``); a refusal exits 1 with a refusing VERDICT, None leaves no VERDICT."""
    r = o / "steps" / "negative" / "r1"
    d = r / "out" / "lean"
    d.mkdir(parents=True)
    (r / "rc").write_text("0\n")
    (r / "time-step.txt").write_text(timev("9:00.00"))
    for coin, by in refusals.items():
        for stmt, refused in by.items():
            (d / f"time-{stmt}.txt").write_text(timev("1:00.00") + "\tExit status: 0\n")
            (d / f"{stmt}.log").write_text('VERDICT {"accepted": true}\n')
            n = f"forged-inner-{coin}-{stmt}"
            (d / f"time-{n}.txt").write_text(timev("1:05.00") + f"\tExit status: {1 if refused else 0}\n")
            if refused is not None:
                (d / f"{n}.log").write_text('VERDICT {"accepted": %s, "why": "%s"}\n'
                                            % (("false", "rec: v disagrees") if refused else ("true", "")))


def test_a_flipped_coin_is_refused_when_one_statement_refuses_it_and_every_one_has_a_verdict(tmp_path: Path) -> None:
    o = job(tmp_path)
    negative(o, {"challenge": {"L0": False, "alg-p0": True}, "query": {"L0": True, "alg-p0": None}})
    doc = RP.read(o)
    ch, q = row(doc, "negative inner-challenge"), row(doc, "negative inner-query")
    assert ch["ok"] is True and ch["extra"]["refused_by"] == ["alg-p0"] and ch["extra"]["accepted_by"] == ["L0"]
    assert ch["note"] == "refused by alg-p0; accepted by L0 (reads no flipped coin)" and ch["extra"]["why"] == ["rec: v disagrees"]
    assert q["ok"] is False and q["extra"]["no_verdict"] == ["alg-p0"]   # a statement that never judged it: not refused
    assert doc["headline"]["negative_refused"] == {"challenge": True, "query": False} and doc["headline"]["ok"] is False


def test_vstage_rows_carry_what_rec_vstage_staged(tmp_path: Path) -> None:
    o = job(tmp_path)
    summ = {"statements": {"alg": {"stage": {"algebra": [{"part": 0, "template": "InnerRepCheck", "build_s": 3.5,
                                                            "circuit_bytes": 99},
                                                           {"part": 1, "template": "InnerRepCheck", "build_s": 4.0,
                                                            "circuit_bytes": 98}]}},
                           "L0": {"stage": {"levels": {"L0": {"template": "RecOpen", "k_log": 20}}}}}}
    (o / "steps" / "vstage" / "r1" / "out" / "summary-vstage.json").write_text(json.dumps(summ))
    assert RP.staged_units(o / "steps" / "vstage" / "r1" / "out" / "summary-vstage.json")["alg-p0"]["circuit_bytes"] == 99
    doc = RP.read(o)
    assert row(doc, "stage L0")["extra"]["units"]["template"] == "RecOpen"
    assert sorted(row(doc, "stage alg")["extra"]["units"]) == ["alg-p0", "alg-p1"]


def test_each_statement_has_its_own_lease_from_its_firewalled_release(capsys) -> None:
    assert RUN.stmt_lease_min("L0") == 94 and RUN.stmt_lease_min("L3") == 12 and RUN.stmt_lease_min("cap-L3") == 2
    assert RUN.lease_min(1) == 94 and RUN.lease_min(2) == 187 and RUN.lease_min(1, firewall=False) == 10
    assert RUN.lease_min(1, levels="L3") == 12 and RUN.lease_min(1, levels="alg") == 22      # alg: alg-p0 and alg-p1
    assert RUN.statements("alg,cap") == ["alg-p0", "alg-p1", "cap"] and len(RUN.statements()) == 10
    assert RUN.main(["plan"]) == 0
    assert "leases (min): L0 94, L1 23, L2 21, L3 12, alg-p0 21, alg-p1 22, cap 2," in capsys.readouterr().out
    assert RUN.main(["run", "--dry-run"]) == 0
    cmd = capsys.readouterr().out.split()
    assert "MODES=firewall" in cmd and "GPU_MAX_MIN=94" in cmd and "OPROVE_RUNS=1" in cmd
    every = int(cmd[cmd.index("--timeout") + 1].removesuffix("m"))
    assert every > 7870 * 1.5 / 60                                         # every statement's release, one at a time
    assert cmd[cmd.index("--mem-gb") + 1] == "128"                          # L0's firewalled release peaked at 99 GB
    assert RUN.main(["run", "--dry-run", "--env", "LEVELS=L3"]) == 0
    cmd = capsys.readouterr().out.split()
    assert "GPU_MAX_MIN=12" in cmd and int(cmd[cmd.index("--timeout") + 1].removesuffix("m")) < every - 60
    assert RUN.main(["run", "--dry-run", "--lease-min", "30", "--modes", "dev"]) == 0
    cmd = capsys.readouterr().out.split()
    assert "GPU_MAX_MIN=30" in cmd and cmd[cmd.index("--mem-gb") + 1] == "64"
    assert RUN.main(["run", "--dry-run", "--mem-gb", "200"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[cmd.index("--mem-gb") + 1] == "200"


def test_firewalled_statements_name_the_cpu_prover(tmp_path: Path, capsys) -> None:
    o = job(tmp_path)
    (o / "params.txt").write_text("commit=abc K=4096 N=2 WARM=1 RUNS=3 REPEAT=2\n"
                                  "OPROVE_WARM=0 OPROVE_RUNS=1 OPROVE_GPU=0 OPROVE_JOBS=1\n")
    doc = RP.read(o)
    r = next(r for r in doc["rows"] if r["stage"] == "outer L0" and r["extra"]["mode"] == "firewall")
    assert r["what"].endswith(", CPU prover") and r["extra"]["prover"] == "cpu"
    assert "V* statements 1 at a time" in RT.markdown(doc)
    assert RUN.main(["run", "--dry-run", "--oprove-gpu", "1"]) == 0
    assert "OPROVE_GPU=1" in capsys.readouterr().out.split()


def test_forged_statements_are_ok_when_refused_and_forge_after_reads_each_case(tmp_path: Path) -> None:
    o = job(tmp_path)
    d = o / "steps" / "oprove" / "r1" / "out" / "oprove" / "forged-coef-L0-firewall"
    d.mkdir(parents=True)
    stop = "FIREWALL-REFUSED: circuit: the constraints lincheck fails"
    (d / "prove.tsv").write_text(f"1792871908.2\tLIVE\t{json.dumps({'accepted': False, 'prover_stopped': stop})}\n")
    d = d.parent / "forged-dirs-L0-firewall"
    d.mkdir()
    (d / "prove.err").write_text("FIREWALL-REFUSED R7: the witness does not open the registered row; no proof\n")
    for n in (1, 2):
        lr = o / "steps" / "lean" / f"r{n}" / "out" / "lean"
        (lr / "time-forged-top-L0.txt").write_text(timev("0:04.00"))
        (lr / "forged-top-L0.log").write_text('VERDICT {"accepted": false, "why": "an opening climbs to another node"}\n')
        # forged-sum leaves alg-p1 as the honest one, so it stands refused; both of forged-comb's parts accepted
        for part, accepted in (("alg-p1", True), ("forged-sum-alg-p0", False), ("forged-sum-alg-p1", True),
                               ("forged-comb-alg-p0", True), ("forged-comb-alg-p1", True)):
            (lr / f"time-{part}.txt").write_text(timev("0:30.00"))
            (lr / f"{part}.log").write_text(f'VERDICT {{"accepted": {json.dumps(accepted)}, "why": "w"}}\n')
    fa = o / "steps" / "forge-after" / "r1"
    (fa / "out").mkdir(parents=True)
    (fa / "rc").write_text("1\n")
    (fa / "time-step.txt").write_text(timev("3:00.00"))
    verify = {"run/coef-honest": {"rc": 0, "accepted": True, "why": "accepted"},
              "run/coef-value": {"rc": 1, "accepted": False, "why": "setup: rec-coins: the statement's coef row is not the verifier's"},
              "run/message-honest": {"rc": 0, "accepted": True, "why": "accepted"},
              "run/message-bc": {"rc": 0, "accepted": True, "why": "accepted"}}
    (fa / "out" / "forge-after.json").write_text(json.dumps({"verify": verify, "untouched": True}))
    doc = RP.read(o)
    coef = row(doc, "outer forged-coef-L0")
    assert coef["ok"] is True and coef["note"].endswith(stop)        # the release the ZK server stopped, from its LIVE record
    outer = row(doc, "outer forged-dirs-L0")
    assert outer["ok"] is True and outer["note"].startswith("no run finished") and "FIREWALL-REFUSED" in outer["note"]
    assert doc["summary"]["outer"]["firewall"]["statements"] == 1      # a forged statement is no part of the run's cost
    assert row(doc, "lean forged-top-L0")["ok"] is True
    assert row(doc, "lean forged-sum-alg-p0")["ok"] is True
    honest_part = row(doc, "lean forged-sum-alg-p1")
    assert honest_part["ok"] is True and honest_part["note"].startswith("accepted: a part this forgery leaves as the honest one")
    assert [row(doc, f"lean forged-comb-alg-p{i}")["note"].startswith("ACCEPTED: no part") for i in (0, 1)] == [True, True]
    assert row(doc, "forge-after run/coef-honest")["ok"] is True and row(doc, "forge-after run/coef-value")["ok"] is True
    bad = row(doc, "forge-after run/message-bc")
    assert bad["ok"] is False and bad["note"].startswith("ACCEPTED")
    assert doc["headline"]["forged_refused"] == {"coef": True, "dirs": True, "comb": False, "sum": True, "top": True,
                                                 "message": False}
    assert "forge-after run/message-bc" in doc["summary"]["not_ok"] and doc["steps"]["forge-after"] == ["1"]
    md = RT.markdown(doc)
    line = next(ln for ln in md.splitlines() if ln.startswith("Forgeries"))
    assert "forge-after run/message-bc" in md and line.endswith(
        "each refused: " + ", ".join(f"{k} {'yes' if v else '**no**'}" for k, v in doc["headline"]["forged_refused"].items()) + ".")
    assert "comb **no**" in line and "sum yes" in line and "message **no**" in line
