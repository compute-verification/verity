"""`verity bench prove layers`: each layer's row read from the files its command writes, built here small in each format
(recursion_e2e's ``recursion_e2e.json``, prove_path's arm ``audit.log`` and ``summary.json``, a5_llama1b_h200's
``summary.txt`` with a5.sh's out), and the table, its JSON line and the kit's measurements written from them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from verity_bench import proof_layers as PL
from verity_bench.cli import main

INNER = ("N x GemmCoordinate_v2{K,DOT=HopperBF16WgmmaDot16_v1} under Q_template_instance (rec_reprice.py), "
         "verity/flock-circuit+plain-leaves, M0 (no --zk), on the challenger's coin tree (FC_COINS=tree), through rec_live's proxy")


def recursion(root: Path, composed: bool | None = True) -> Path:
    out = root / "out"
    out.mkdir(parents=True)
    headline = {"complete": True, "ok": composed is True, "parts_ok": True, "composed": composed,
                "not_composed": None if composed else "not composed: no rec-compose record",
                "pipeline": {"inner M0 e2e_s": 2.0, "V* staging_s (vstage + vcap steps)": 200.0,
                             "outer prove (firewall) e2e_s": 1000.0, "Lean step_s": 90.0},
                "outer": {"firewall": {"e2e_s": 1000.0, "proof_bytes": 18000000, "prove_s": 900.0, "statements": 10}},
                "outer_mode": "outer prove (firewall) e2e_s"}
    if composed is None:
        del headline["composed"], headline["parts_ok"], headline["not_composed"]
        headline["ok"] = True
    doc = {"schema": "recursion-e2e/v1", "headline": headline, "params": {"K": "4096", "N": "2", "tree_from": "test-os"},
           "inputs": {"inner": INNER}, "steps": {"build": ["0"], "lean": ["0"]}}
    (out / "recursion_e2e.json").write_text(json.dumps(doc))
    return root


def prove_path_arm(out: Path, name: str, *, proof: str = "zk", forged: bool = False, lean: bool = True) -> Path:
    arm = out / name
    arm.mkdir(parents=True)
    a5 = {"audit": {"accepted": lean, "verdicts": {"lean-draw/m0": True, "lean-verify/m0": lean, "m0-verifier/m0": True}}}
    rank = {"accepted": lean, "ok": lean, "population": 8, "draw_counts": {"SiluMul_v2{I=8192}": 2},
            "per_unit": {"SiluMul_v2{I=8192}": {"units": 2, "proof_bytes": 1700000, "prove_s_per_unit": 3.0,
                                                "lean_verify_s_per_unit": 1.0}}}
    line = {"forged": forged, "ok": lean, "proof": proof, "coins": "challenger", "ranks": {"0": rank}}
    (arm / "audit.log").write_text("A5 {\"lean_workers\": 1}\n" + "A5 " + json.dumps(a5) + "\n"
                                   + "PROVE-PATH " + json.dumps(line) + "\n")
    (arm / "summary.json").write_text(json.dumps({"coins": {"source": "challenger", "challenger": "local"}}))
    return arm


def a5_out(root: Path, *, stopped: bool = False) -> Path:
    out = root / "out"
    (out / "a5" / "a5").mkdir(parents=True)
    (out / "machine.txt").write_text("ubuntu\nnode\nabc\nrow dir /w/r/llama32-1b__int8__h200__tp1__b4\n")
    a5 = {"audit": {"accepted": True, "verdicts": {"challenger/m0": True, "lean-draw/m0": True, "lean-verify/m0": True}},
          "draw_counts": {"RoPEHead_v1{D=64}": 4}, "source": "served"}
    lines = ["STOPPED at commit (no window)"] if stopped else [
        "REGISTERED kind=served commitment=rows-sha512 run_root_bytes=64 run=ab...", "A5 " + json.dumps(a5),
        "replay rc=0 a5 rc=0"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "verdicts.txt").write_text("build PASS 2026-11-15T10:00:00Z\n")
    if not stopped:
        (out / "a5" / "a5.log").write_text("A5 " + json.dumps(a5) + "\n")
        timing = {"RoPEHead_v1{D=64}": {"lean_verify_s": 8.0, "timing": {"prove_total_s": 20.0, "proof_bytes": [500, 500]}}}
        (out / "a5" / "a5" / "summary.json").write_text(json.dumps({"timing": timing, "population": 112, "arm": "dev",
                                                                    "draw_counts": a5["draw_counts"]}))
    return root


def test_an_accepted_recursive_run_gives_its_path_without_leans_step_and_its_releases_bytes(tmp_path):
    row = PL.read("recursion", str(recursion(tmp_path)))
    assert row.accepted and row.why is None
    assert row.prover_s == 1202.0 and row.prover_unit == "s a run" and row.lean_s == 90.0
    assert row.proof_bytes == 18000000 and "10 releases" in row.bytes_what
    assert row.workload == ("K = 4096, N = 2: N x GemmCoordinate_v2{K,DOT=HopperBF16WgmmaDot16_v1} under "
                            "Q_template_instance (rec_reprice.py)")
    assert row.coins == ("coins: on the challenger's coin tree (FC_COINS=tree); coin tree from test-os; "
                         "parameters: not stated")


@pytest.mark.parametrize("composed", [False, None])
def test_a_recursive_run_lean_did_not_compose_prints_no_number(tmp_path, composed):
    """A headline with its parts passed but no composition (or one from before ``composed``) is no accepted recursive
    run, as recursion_e2e's own verdict says."""

    row = PL.read("recursion", str(recursion(tmp_path, composed)))
    assert not row.accepted and row.prover_s is None and row.lean_s is None and row.proof_bytes is None
    assert "not an accept" in row.why and "composed" in row.lean


def test_a_prove_path_run_takes_its_honest_arm_and_its_blocks_means(tmp_path):
    out = tmp_path / "out"
    prove_path_arm(out, "llama")
    prove_path_arm(out, "llama-forged", forged=True, lean=False)
    row = PL.read("zk", str(tmp_path))
    assert row.accepted and row.format == PL.PROVE_PATH and row.proof == "zk"
    assert (row.prover_s, row.lean_s, row.proof_bytes) == (3.0, 1.0, 850000.0) and row.prover_unit == "s a block"
    assert row.workload == "arm llama: SiluMul_v2{I=8192} x 2 drawn of 8 units"
    assert row.coins == "coins: challenger, local; proof: zk; parameters: not stated"


def test_the_flock_row_is_the_plain_proofs(tmp_path):
    zk = PL.read("flock", str(prove_path_arm(tmp_path / "a", "llama")))
    assert zk.prover_s is None and "plain" in zk.why
    plain = PL.read("flock", str(prove_path_arm(tmp_path / "b", "llama", proof="plain")))
    assert plain.accepted and plain.prover_s == 3.0


def test_a_prove_path_run_whose_lean_verify_refused_prints_no_number(tmp_path):
    row = PL.read("zk", str(prove_path_arm(tmp_path, "llama", lean=False)))
    assert not row.accepted and row.prover_s is None and "lean-verify/m0 False" in row.lean


def test_an_a5_run_reads_its_audit_its_timing_and_states_no_more_of_its_coins_than_it_says(tmp_path):
    row = PL.read("sampling", str(a5_out(tmp_path)))
    assert row.accepted and row.format == PL.A5
    assert (row.prover_s, row.lean_s, row.proof_bytes) == (5.0, 2.0, 250.0)
    assert row.workload == ("row llama32-1b__int8__h200__tp1__b4: RoPEHead_v1{D=64} x 4 drawn of 112 units "
                            "(kind=served commitment=rows-sha512)")
    assert row.coins == ("coins: a challenger per drawn member (challenger/m0), local or deployed not stated; proof: zk, "
                         "arm dev; parameters: not stated")
    assert "replay rc=0 a5 rc=0" in row.notes and "serving stage build PASS 2026-11-15T10:00:00Z" in row.notes


def test_a_stopped_a5_run_says_where_it_stopped(tmp_path):
    row = PL.read("sampling", str(a5_out(tmp_path, stopped=True)))
    assert not row.accepted and row.prover_s is None and row.why == "STOPPED at commit (no window)"


def test_a_row_given_another_commands_output_or_nothing_readable_says_so(tmp_path):
    assert "takes prove_path" in PL.read("flock", str(recursion(tmp_path / "r"))).why
    assert PL.read("zk", str(tmp_path / "missing")).status == "unreadable"
    (tmp_path / "empty").mkdir()
    assert "none of the outputs" in PL.read("zk", str(tmp_path / "empty")).why


def test_a_run_id_is_read_from_the_record_its_fetcher_lands(tmp_path):
    root = recursion(tmp_path)
    row = PL.read("recursion", "r20261115-183449-c6f6", fetcher=lambda run: root)
    assert row.accepted and row.given == "r20261115-183449-c6f6"


def test_the_command_prints_every_layer_and_writes_a_measurement_a_row(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", "local")
    monkeypatch.setenv("OUT", str(tmp_path / "report"))
    assert main(["prove-layers", "--recursion", str(recursion(tmp_path / "r")),
                 "--sampling", str(a5_out(tmp_path / "s"))]) == 0
    text, line = capsys.readouterr().out.rstrip("\n").rsplit("\n", 1)
    for title in PL.LAYERS.values():
        assert f"| {title} |" in text
    assert "| Flock | not run |" in text and "- Flock: no number: not run" in text
    doc = json.loads(line)
    assert doc["schema"] == PL.SCHEMA and [r["layer"] for r in doc["rows"]] == list(PL.LAYERS)
    assert json.loads((tmp_path / "report" / "proof-layers.json").read_text()) == doc
    result = json.loads((tmp_path / "report" / "result.json").read_text())
    got = {m["member"]: m for m in result["measurements"]}
    assert set(got) == set(PL.LAYERS) and all(m["name"] == PL.MEASUREMENT and m["mark"] == "measured" for m in got.values())
    assert got["recursion"]["value"] == 1202.0 and got["sampling"]["value"] == 5.0
    assert got["flock"]["value"] is None and got["flock"]["why"].startswith("not run")
