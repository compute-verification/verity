"""The recursion bench's reader (`rec_profile`), table (`rec_table`) and CLI (`run.py`) on records written in
85-rec-reprice.sh's formats: the timed runs' LIVE lines, `/usr/bin/time -v` files, serve's SESSION lines, the replay's
UPSTREAM line and Lean's VERDICT. No GPU, node or store: the job's own records are made here, by `job`, which
`test_run.py` (the CLI's launches and commands) and `test_headline.py` (an accepted recursive run) use too."""
from __future__ import annotations

import json
from pathlib import Path

import rec_profile as RP
import rec_table as RT

#: oprove's mode through the ZK server, as 85 names its directories
ZK = RP.ZK_SERVER_MODE


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
    prover_dir(p / "oprove" / f"L0-{ZK}", (5.0, 6.0, 7.0))
    prover_dir(p / "oprove" / "L0-dev", (1.0, 1.0, 1.0))
    prover_dir(p / "oprove" / "alg-p0-dev", (2.0, 2.0, 2.0), accepted=False)
    for n in (1, 2):
        c = rep("icheck", n)
        (c / "icheck").mkdir()
        (c / "icheck" / "time-replay-proxy.txt").write_text(timev(f"0:0{n}.00"))
        (c / "icheck" / "replay-proxy.txt").write_text("UPSTREAM\t" + json.dumps({"verdict": {"accepted": True}}) + "\n")
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


def test_a_proxy_run_with_no_verdict_is_neither_accepted_nor_refused(tmp_path: Path) -> None:
    d = tmp_path / "p"
    d.mkdir()
    (d / "prove.tsv").write_text(live(1.0, True, 1.0, 2.0, accepted=None) + live(3.0, False, 1.0, 2.0, accepted=None))
    r = RP.prover_row(d, "inner M0", "inner", "x").to_json()
    assert r["ok"] is None and "nothing verified here" in r["note"]


def test_a_repeat_still_running_is_no_refusal(tmp_path: Path) -> None:
    o = job(tmp_path)
    r3 = o / "steps" / "lean" / "r3" / "out" / "lean"
    r3.mkdir(parents=True)
    (r3 / "time-L0.txt").write_text("")
    (r3 / "L0.log").write_text("reading inputs\n")
    lean = row(RP.read(o), "lean L0")
    assert lean["ok"] is True and lean["wall_s"]["values"] == [61.0, 62.0]


def test_a_step_that_never_ran_is_not_a_row_and_a_failed_prover_says_so(tmp_path: Path) -> None:
    o = job(tmp_path)
    (o / "steps" / "oprove" / "r1" / "out" / "oprove" / "L1-dev").mkdir()
    (o / "steps" / "oprove" / "r1" / "out" / "oprove" / "L1-dev" / "prove.err").write_text("ZK-SERVER-REFUSED: the setup\n")
    doc = RP.read(o)
    assert not [r for r in doc["rows"] if r["step"] == "vcap"]
    l1 = row(doc, "outer L1")
    assert l1["ok"] is False and "ZK-SERVER-REFUSED" in l1["note"]


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


def test_forged_statements_are_ok_when_refused_and_forge_after_reads_each_case(tmp_path: Path) -> None:
    o = job(tmp_path)
    d = o / "steps" / "oprove" / "r1" / "out" / "oprove" / f"forged-coef-L0-{ZK}"
    d.mkdir(parents=True)
    stop = "ZK-SERVER-REFUSED: circuit: the constraints lincheck fails"
    (d / "prove.tsv").write_text(f"1792871908.2\tLIVE\t{json.dumps({'accepted': False, 'prover_stopped': stop})}\n")
    d = d.parent / f"forged-dirs-L0-{ZK}"
    d.mkdir()
    (d / "prove.err").write_text("ZK-SERVER-REFUSED R7: the witness does not open the registered row; no proof\n")
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
    assert outer["ok"] is True and outer["note"].startswith("no run finished") and "ZK-SERVER-REFUSED R7" in outer["note"]
    assert doc["summary"]["outer"][ZK]["statements"] == 1      # a forged statement is no part of the run's cost
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
