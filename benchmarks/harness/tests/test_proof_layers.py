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


def a5_out(root: Path, *, stopped: bool = False, **summary: object) -> Path:
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
                                                                    "draw_counts": a5["draw_counts"], **summary}))
    return root


def test_an_accepted_recursive_run_gives_its_path_without_leans_step_and_its_releases_bytes(tmp_path):
    row = PL.read("recursion", str(recursion(tmp_path)))
    assert row.accepted and row.why is None
    assert row.prover_s == 1202.0 and row.prover_unit == "s a run" and row.lean_s == 90.0
    assert row.proof_bytes == 18000000 and "10 releases" in row.bytes_what
    assert row.workload == ("K = 4096, N = 2: N x GemmCoordinate_v2{K,DOT=HopperBF16WgmmaDot16_v1} under "
                            "Q_template_instance (rec_reprice.py)")
    assert row.ran_with == "ran with: not stated"
    assert row.detail == "inputs.inner: on the challenger's coin tree (FC_COINS=tree); params.tree_from test-os"


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
    assert row.ran_with == ("ran with: coins local challenger; parameters not stated "
                            "(PROVE-PATH coins challenger; summary.json coins.challenger local)")
    assert row.detail == "PROVE-PATH coins challenger; summary.json coins.challenger local"


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
    assert row.ran_with == "ran with: not stated"
    assert row.detail == "audit verdicts challenger/m0, a challenger per drawn member"
    assert "replay rc=0 a5 rc=0" in row.notes and "serving stage build PASS 2026-11-15T10:00:00Z" in row.notes
    assert "arm dev" in row.notes


def test_an_a5_row_is_checked_against_the_proof_of_the_layer_it_is_given_as(tmp_path):
    """A plain a5 out given as ``--zk`` prints no seconds in the zero-knowledge row; its zk out does, with the coins its
    summary names."""

    plain = PL.read("zk", str(a5_out(tmp_path / "plain", plain=True)))
    assert plain.proof == "plain" and plain.prover_s is None and "says its proof is plain" in plain.why
    coins = {"source": "challenger", "challenger": "local"}
    zk = PL.read("zk", str(a5_out(tmp_path / "zk", coins=coins)))
    assert zk.accepted and zk.prover_s == 5.0
    assert zk.ran_with == ('ran with: coins local challenger; parameters not stated '
                           '(summary.json coins {"challenger": "local", "source": "challenger"})')


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


#: `benchmarks/prove/run.py`'s ``LAYER_METRICS``
IDS = {"benchmarks/proof-layer-prover-seconds": "prover_s", "benchmarks/proof-layer-lean-seconds": "lean_s",
       "benchmarks/proof-layer-proof-bytes": "proof_bytes"}


def test_the_command_prints_every_layer_and_writes_each_id_flock_zero_knowledge_then_sampling(tmp_path, monkeypatch,
                                                                                              capsys):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", "local")
    monkeypatch.setenv("OUT", str(tmp_path / "report"))
    assert main(["prove-layers", "--recursion", str(recursion(tmp_path / "r")),
                 "--zk", str(prove_path_arm(tmp_path / "z", "llama")),
                 "--sampling", str(a5_out(tmp_path / "s"))], metrics=IDS) == 0
    text, line = capsys.readouterr().out.rstrip("\n").rsplit("\n", 1)
    for title in PL.LAYERS.values():
        assert f"| {title} |" in text
    assert "| Flock | not run |" in text and "- Flock: no number: not run" in text
    doc = json.loads(line)
    assert doc["schema"] == PL.SCHEMA and [r["layer"] for r in doc["rows"]] == list(PL.LAYERS)
    assert json.loads((tmp_path / "report" / "proof-layers.json").read_text()) == doc
    written = json.loads((tmp_path / "report" / "result.json").read_text())["measurements"]
    assert [(m["name"], m.get("member")) for m in written] == [
        (name, member) for name in IDS for member in ("flock", "zero knowledge", None)]
    values = {(m["name"], m.get("member")): m for m in written}
    assert [values[name, "zero knowledge"]["value"] for name in IDS] == [3.0, 1.0, 850000.0]
    assert [values[name, None]["value"] for name in IDS] == [5.0, 2.0, 250.0]
    assert all(values[name, "flock"]["value"] is None and values[name, "flock"]["why"].startswith("not run") for name in IDS)
    assert {m["unit"] for m in written} == {"s a proof block", "bytes a proof block"} and all(m["mark"] == "measured"
                                                                                           for m in written)
    assert doc["rows"][3]["ran_with"] == "ran with: not stated" and doc["rows"][3]["prover_s"] == 1202.0
    assert "| ran with |" in text and "| ran with: not stated |" in text


def test_a_layer_without_an_accept_or_a_per_block_figure_writes_no_number_under_any_id(tmp_path):
    """A refused Lean verify and an a5 member whose timing is keyed by its chunk's name (no ``draw_counts`` entry) each
    write null with why, never a figure; recursion's row is no member of any id."""

    refused = PL.read("zk", str(prove_path_arm(tmp_path / "z", "llama", lean=False)))
    chunk = PL.read("sampling", str(a5_out(tmp_path / "s", draw_counts={"OtherTemplate_v1": 4})))
    assert chunk.accepted and chunk.prover_s is None
    assert "no per-block figure: member RoPEHead_v1{D=64}'s timing has no drawn-block count (draw_counts)" in chunk.notes
    rows = [PL.read("flock", None), refused, chunk, PL.read("recursion", str(recursion(tmp_path / "r")))]
    written = PL.measured(rows, IDS)
    assert len(written) == 9 and all(m["value"] is None and m["why"] for m in written)
    assert {m["member"] for m in written} == {"flock", "zero knowledge", None}
    with pytest.raises(ValueError, match="not one of"):
        PL.measured(rows, {"benchmarks/x": "seconds"})


@pytest.mark.parametrize("out", ["", None])
def test_an_empty_out_or_run_dir_is_no_place_to_write(tmp_path, monkeypatch, capsys, out):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_RUN_DIR", "")
    if out is None:
        monkeypatch.delenv("OUT", raising=False)
    else:
        monkeypatch.setenv("OUT", out)
    assert PL.report(PL.REPORT, "text", {"rows": []}, PL.measured([PL.read(x, None) for x in PL.LAYERS], IDS)) is None
    assert capsys.readouterr().out == 'text\n{"rows": []}\n' and not any(tmp_path.iterdir())
