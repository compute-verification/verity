"""`zk_cost.py overhead` divides by `verity.hardware.cards.rate`, the one peak calculator: on the H200 an e4m3 block's peak is
132 SMs x 4096 MACs at the 1830 MHz tensor clock, the data sheet's 1,979 dense FP8 TFLOPS, and never the 1980 MHz SM clock's
1.0705e15. Toy records of one served type, staged at two sizes and proved warm by the dev arm, give a known marginal second
a block, so its h is fixed by the peak alone. A served type's size of 2 proves its probe's staging rather than staging the
same 2 instances again. Passes prove each type's sizes in ABBA order, a session's CPU seconds are its own rather than its
processes', and `overhead` fits each part of a session, a CPU node's with no card. A served type's session is verified at
its statement's partition and Program, and each Lean verdict is recorded and reported as what happened: accepted or
refused, and a refusal, the verifier's want of a Program among them, is never counted as an acceptance."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from verity.hardware import cards

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zk_cost as Z

K, FIXED_S, MARGINAL_S = 6144, 2.0, 1e-3
PUBLIC_INPUTS = "refused: public input ports (--public-inputs): a statement form the soundness proof does not cover"
NO_PROGRAM = "refused: a session without the verifier's program (--program)"


def _records(out: Path, sizes: tuple[int, ...] = (1024, 2048)) -> None:
    rec = out / "records"
    rec.mkdir(parents=True)
    stage = [{"unit": f"fp8-coord:K={K}", "tag": f"fp8-coord-k{K}", "c": c, "ands": 1, "k_log": 24, "full": max(sizes),
              "definition": f"Fp8Coordinate_v1{{K={K}}}", "macs": K, "format": "e4m3", "stage_s": 1.0} for c in sizes]
    lives = lambda c: [{"warm": w, "e2e_s": FIXED_S + MARGINAL_S * c, "prove_total_s": 1.0, "witness_s": None,  # noqa: E731
                        "dense_m": 1} for w in (True, False)]
    prove = [{"unit": f"fp8-coord:K={K}", "tag": f"fp8-coord-k{K}", "arm": "dev", "mode": "warm", "c": c, "k": c,
              "accepted": True, "lives": lives(c), "peak_rss_kb": 1} for c in sizes]
    (rec / "stage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stage))
    (rec / "prove.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prove))
    (rec / "verify.jsonl").write_text("")
    (rec / "gpu.csv").write_text("name,uuid,driver,memory,compute_cap\nNVIDIA H200,GPU-0,595.91.07,143771 MiB,9.0\n")


def test_the_h200_e4m3_peak_is_the_data_sheets_at_the_tensor_clock():
    card = cards.load("h200-sxm-141gb")
    peak = cards.rate(card, fmt="e4m3")
    assert peak == 132 * 4096 * 1830 * 10**6
    assert float(peak) == pytest.approx(9.8945e14, rel=1e-4)
    assert cards.rate(card, fmt="e4m3", mhz=1980) == 132 * 4096 * 1980 * 10**6


def test_overhead_divides_a_blocks_seconds_by_the_cards_rate(tmp_path):
    _records(tmp_path)
    assert Z.overhead(argparse.Namespace(out=str(tmp_path), card="h200-sxm-141gb", run="r-test")) == 0
    (t,) = json.loads((tmp_path / "overhead.json").read_text())["types"].values()
    peak = float(cards.rate(cards.load("h200-sxm-141gb"), fmt="e4m3"))
    assert t["peak_mac_per_s"] == peak
    dev = t["arms"]["dev"]
    assert dev["marginal_s"] == pytest.approx(MARGINAL_S) and dev["fixed_s"] == pytest.approx(FIXED_S)
    assert dev["overhead_marginal"] == pytest.approx(MARGINAL_S / (K / 9.8945e14), rel=1e-4)
    assert dev["overhead_marginal"] == pytest.approx(MARGINAL_S * peak / K, rel=1e-9)


@pytest.mark.parametrize("verdict, outcome, rc", [
    ({"accepted": True, "why": ""}, "accepted", 0),
    ({"accepted": False, "why": NO_PROGRAM}, "refused", 1),
    ({"accepted": False, "why": PUBLIC_INPUTS}, "refused", 1)])
def test_a_served_sessions_record_keeps_leans_verdict_and_counts_no_refusal_as_an_acceptance(
        tmp_path, monkeypatch, capsys, verdict, outcome, rc):
    (tmp_path / "records").mkdir()
    (tmp_path / "points").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps([{"unit": "gq:K=384", "tag": "gq-k384", "c": 2, "k": 2,
                                                     "stage_dir": str(tmp_path), "pub": "pub.bin", "partition": "partition.json",
                                                     "program": "program.json"}]))
    (tmp_path / "records" / "prove.jsonl").write_text(json.dumps(
        {"unit": "gq:K=384", "tag": "gq-k384", "c": 2, "k": 2, "arm": "dev", "mode": "warm", "rep": 0, "accepted": True,
         "sessions": [str(tmp_path / "s0")], "dir": str(tmp_path / "points")}) + "\n")
    cmds = []
    monkeypatch.setattr(Z, "timed", lambda cmd, fo, fe, **kw: (cmds.append(cmd), (
        0, {"wall_s": 1.0, "user_s": 0.5, "sys_s": 0.1, "maxrss_kb": 1}, [(0.5, "VERDICT " + json.dumps(verdict))]))[1])
    # a served type's session is verified at its statement's partition and Program, and only Lean's accepting it holds
    assert Z.verify(argparse.Namespace(out=str(tmp_path), lean="lean")) == rc
    (cmd,) = cmds
    assert cmd[cmd.index("--partition") + 1] == "partition.json" and cmd[cmd.index("--program") + 1] == "program.json"
    (rec,) = Z.read_jsonl(tmp_path / "records" / "verify.jsonl")
    assert (rec["accepted"], rec["why"]) == (verdict["accepted"], verdict["why"])
    assert "zk_verify_passed" not in rec and Z.lean_outcome(rec) == outcome
    out = capsys.readouterr().out
    assert f"gq:K=384 c=2 k=2 dev: {outcome}" in out
    assert f"Lean accepted {1 - rc} of 1 sessions and refused {rc}" in out


def test_overhead_reports_each_lean_verdict_as_what_happened(tmp_path):
    _records(tmp_path)
    tag = f"fp8-coord-k{K}"
    base = {"unit": f"fp8-coord:K={K}", "tag": tag, "arm": "dev", "c": 2048, "k": 2048, "user_s": 1.0, "sys_s": 0.1}
    verdicts = [{"accepted": True, "why": None},
                {"accepted": False, "why": NO_PROGRAM, "zk_verify_passed": True},   # kept while it counted as honest
                {"accepted": False, "why": PUBLIC_INPUTS}]
    (tmp_path / "records" / "verify.jsonl").write_text("".join(json.dumps(base | v) + "\n" for v in verdicts))
    assert Z.overhead(argparse.Namespace(out=str(tmp_path), card="h200-sxm-141gb", run="r-test")) == 0
    (t,) = json.loads((tmp_path / "overhead.json").read_text())["types"].values()
    assert [v["accepted"] for v in t["arms"]["dev"]["lean"]] == [True, False, False]
    assert not any("zk_verify_passed" in v for v in t["arms"]["dev"]["lean"])
    row = next(x for x in (tmp_path / "overhead.md").read_text().splitlines() if x.startswith(f"| {tag} | Fp8"))
    assert row.endswith("| accepted,refused,refused |")


def test_overhead_refuses_a_card_that_isnt_the_runs_gpu(tmp_path):
    _records(tmp_path)
    with pytest.raises(SystemExit, match="ZK_CARD"):
        Z.overhead(argparse.Namespace(out=str(tmp_path), card="rtx-pro-6000-server", run="r-test"))


def test_a_size_of_two_proves_the_probe_staging(tmp_path, monkeypatch):
    staged = []

    def stage_served(spec, n, d):
        staged.append((n, d))
        d.mkdir(parents=True)
        (d / "pin.txt").write_text(f"pin-{n}")
        return {"n": n, "k_log": 22, "vus_per_block": 1, "ands": 1, "definition": "Fp8GroupQuantUe8m0_v1{K=384,G=128}",
                "statics": {"K": 384, "G": 128}, "seconds": {"total": 1.0}, "stage_cpu_s": 1.0, "maxrss_gb": 1.0,
                "circuit_sha512": "ab" * 64, "twin": None}

    monkeypatch.setattr(Z, "stage_served", stage_served)
    (tmp_path / "records").mkdir()
    plan = Z.stage_units(argparse.Namespace(units="gq:K=384", sizes="2,64", draws="c:dev"), tmp_path)
    assert staged == [(2, tmp_path / "gq-k384-probe"), (64, tmp_path / "gq-k384-c64" / "stage")]
    assert not (tmp_path / "gq-k384-probe").exists()
    assert [(p["c"], p["k"], p["pin"]) for p in plan] == [(2, 2, "pin-2"), (64, 64, "pin-64")]
    assert plan[0]["stage_dir"] == str(tmp_path / "gq-k384-c2" / "stage")


def test_passes_prove_each_types_sizes_in_abba_order():
    plan = [{"tag": t, "c": c} for t in ("a", "b") for c in (2, 64)]
    assert [(p, pt["tag"], pt["c"]) for p, pt in Z.passes(plan, 2)] == [
        (0, "a", 2), (0, "a", 64), (0, "b", 2), (0, "b", 64), (1, "a", 64), (1, "a", 2), (1, "b", 64), (1, "b", 2)]
    assert [pt for _, pt in Z.passes(plan, 1)] == plan


def test_a_sessions_cpu_seconds_are_its_own_not_its_processes():
    # the ZK server's statement build is in its tree's seconds before the prover starts, the prover's in its first session's
    marks = [{"t": 0.0, "load1": 1.0, "cpu": [(300.0, 20.0, 0), (0.0, 0.0, 0)]},
             {"t": 400.0, "load1": 2.0, "cpu": [(310.0, 21.0, 0), (350.0, 30.0, 0)]},
             {"t": 500.0, "load1": 3.0, "cpu": [(312.0, 23.0, 1), (400.0, 31.5, 0)]}]
    assert Z.session_cpu(marks, ("serve", "prover")) == [
        {"load1": 2.0, "gone": 0, "serve_user_s": 10.0, "serve_sys_s": 1.0, "prover_user_s": 350.0, "prover_sys_s": 30.0},
        {"load1": 3.0, "gone": 1, "serve_user_s": 2.0, "serve_sys_s": 2.0, "prover_user_s": 50.0, "prover_sys_s": 1.5}]


def test_proc_cpu_reads_a_live_process_and_counts_a_gone_one():
    user, sys_, gone = Z.proc_cpu([os.getpid()])
    assert user > 0 and sys_ >= 0 and gone == 0
    assert Z.proc_cpu([(1 << 22) + 12345]) == (0.0, 0.0, 1)   # above Linux's largest pid_max


def test_overhead_fits_each_part_of_a_cpu_nodes_sessions(tmp_path):
    rec = tmp_path / "records"
    rec.mkdir()
    stage = [{"unit": "gq:K=6144", "tag": "gq-k6144", "c": c, "ands": 1, "k_log": 25, "full": 1024, "definition": "Gq",
              "macs": 0, "format": None, "stage_s": 1.0} for c in (2, 64)]
    # the proving grows 0.1 s a block and the wait inside e2e 0.01 s, the witness and the rest don't; the sessions scatter by
    # ±0.5 s; wait_s's final verdict is outside e2e and no part of it
    lives = lambda c: [{"warm": w, "witness_s": 10.0, "prove_total_s": 1.0 + 0.1 * c, "wait_e2e_s": 40.0 + 0.01 * c,  # noqa: E731
                        "wait_s": 70.0 + 0.5 * c, "e2e_s": 101.0 + 0.11 * c + d, "dense_m": 1,
                        "cpu_s": {"load1": 1.0, "prover_user_s": 50.0 + c, "prover_sys_s": 5.0, "serve_user_s": 30.0,
                                  "serve_sys_s": 3.0}} for w, d in ((True, 0.0), (False, -0.5), (False, 0.5))]
    prove = [{"unit": "gq:K=6144", "tag": "gq-k6144", "arm": "dev", "mode": "warm", "c": c, "k": c, "accepted": True,
              "lives": lives(c), "peak_rss_kb": 1} for c in (2, 64)]
    (rec / "stage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stage))
    (rec / "prove.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prove))
    assert Z.overhead(argparse.Namespace(out=str(tmp_path), card="none", run="r-test")) == 0
    dev = json.loads((tmp_path / "overhead.json").read_text())["types"]["gq-k6144"]["arms"]["dev"]
    # four sessions off the line by 0.5 s each, two degrees of freedom, sizes 31 blocks either side of their mean
    assert dev["marginal_s"] == pytest.approx(0.11) and dev["marginal_se_s"] == pytest.approx(0.5 ** 0.5 / 62, rel=1e-6)
    for n, want in {"witness": [10.0, 0.0], "proving": [1.0, 0.1], "wait": [40.0, 0.01], "rest": [50.0, 0.0]}.items():
        assert [dev["parts"][n]["fixed_s"], dev["parts"][n]["marginal_s"]] == pytest.approx(want, abs=1e-9), n
    assert dev["cpu_s"]["64"]["prover_user_s"] == 114.0 and "overhead_marginal" not in dev
    # the toy records from before wait_e2e_s was kept have no parts
    _records(tmp_path / "old")
    Z.overhead(argparse.Namespace(out=str(tmp_path / "old"), card="h200-sxm-141gb", run="r-test"))
    (old,) = json.loads((tmp_path / "old" / "overhead.json").read_text())["types"].values()
    assert old["arms"]["dev"]["parts"] == {}
