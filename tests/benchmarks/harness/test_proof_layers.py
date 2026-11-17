"""`verity bench prove layers`: the benchmark result for each layer, read from the files its command writes, built here
small in each format (recursion_e2e's ``recursion_e2e.json``, prove_path's arm ``audit.log`` and ``summary.json``,
a5_llama1b_h200's ``summary.txt`` with a5.sh's out), and the table, its JSON line and the kit's measurements written
from them."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from verity_bench import kit
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


def prove_path_arm(out: Path, name: str, *, proof: str = "zk", forged: bool = False, lean: bool = True, ranks: int = 1,
                   a5_lines: int | None = None) -> Path:
    arm = out / name
    arm.mkdir(parents=True)
    a5 = {"audit": {"accepted": lean, "verdicts": {"lean-draw/m0": True, "lean-verify/m0": lean, "m0-verifier/m0": True}}}
    rank = {"accepted": lean, "ok": lean, "population": 8, "draw_counts": {"SiluMul_v2{I=8192}": 2},
            "per_unit": {"SiluMul_v2{I=8192}": {"units": 2, "proof_bytes": 1700000, "prove_s_per_unit": 3.0,
                                                "lean_verify_s_per_unit": 1.0}}}
    line = {"forged": forged, "ok": lean, "proof": proof, "coins": "challenger", "delta": 0.5, "ranks": {str(r): rank for r in range(ranks)}}
    (arm / "audit.log").write_text("A5 {\"lean_workers\": 1}\n" + ("A5 " + json.dumps(a5) + "\n") * (a5_lines or ranks)
                                   + "PROVE-PATH " + json.dumps(line) + "\n")
    (arm / "summary.json").write_text(json.dumps({"coins": {"source": "challenger", "challenger": "local"}}))
    return arm


def a5_out(root: Path, *, stopped: bool = False, steps: str | None = None, prover_log: str | None = None,
           **summary: object) -> Path:
    out = root / "out"
    (out / "a5" / "a5").mkdir(parents=True)
    if steps is not None:
        (out / "steps.txt").write_text(steps)
    if prover_log is not None:
        (out / "a5" / "window" / "proof-server").mkdir(parents=True)
        (out / "a5" / "window" / "proof-server" / "prover-cpu.log").write_text(prover_log)
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
    result = PL.read("recursion", str(recursion(tmp_path)))
    assert result.accepted and result.why is None
    assert result.prover_s == 1202.0 and result.prover_unit == "s a run" and result.lean_s == 90.0
    assert result.proof_bytes == 18000000 and "10 releases" in result.bytes_what
    assert result.workload == ("K = 4096, N = 2: N x GemmCoordinate_v2{K,DOT=HopperBF16WgmmaDot16_v1} under "
                            "Q_template_instance (rec_reprice.py)")
    assert result.ran_with == ("ran with: coins in-process; parameters not stated (inputs.inner: on the challenger's coin "
                               "tree (FC_COINS=tree); params.tree_from test-os)")
    assert result.detail == "inputs.inner: on the challenger's coin tree (FC_COINS=tree); params.tree_from test-os"


def test_a_stored_recursive_run_states_its_coins_and_proof_but_not_parameters_its_record_lacks(tmp_path):
    root = recursion(tmp_path)
    path = root / "out" / "recursion_e2e.json"
    doc = json.loads(path.read_text())
    doc["params"]["modes"] = "zk_server"
    doc["inputs"]["outer"] = "each V* statement with --zk on loopback, every proof block sampled, one at a time, in each of MODES"
    path.write_text(json.dumps(doc))
    result = PL.read("recursion", str(root))
    assert result.proof == "zk"
    assert result.ran_with == ("ran with: coins in-process; parameters not stated (inputs.inner: on the challenger's coin "
                               "tree (FC_COINS=tree); params.tree_from test-os; inputs.outer: every proof block sampled)")


@pytest.mark.parametrize(("tree_from", "coins"), [("local", "local challenger"), ("10.0.0.2:7000", "not stated")])
def test_a_recursive_runs_record_of_development_parameters_is_stated_with_its_coins(tmp_path, tree_from, coins):
    root = recursion(tmp_path)
    path = root / "out" / "recursion_e2e.json"
    doc = json.loads(path.read_text())
    doc["params"] |= {"tree_from": tree_from, "parameters": "development"}
    path.write_text(json.dumps(doc))
    assert PL.read("recursion", str(root)).ran_with == (
        f"ran with: coins {coins}; parameters development (inputs.inner: on the challenger's coin tree (FC_COINS=tree); "
        f"params.tree_from {tree_from}; params.parameters development)")


@pytest.mark.parametrize("composed", [False, None])
def test_a_recursive_run_lean_did_not_compose_prints_no_number(tmp_path, composed):
    """A headline with its parts passed but no composition (or one from before ``composed``) is no accepted recursive
    run, as recursion_e2e's own verdict says."""

    result = PL.read("recursion", str(recursion(tmp_path, composed)))
    assert not result.accepted and result.prover_s is None and result.lean_s is None and result.proof_bytes is None
    assert "not an accept" in result.why and "composed" in result.lean


def test_a_prove_path_run_takes_its_honest_arm_and_its_blocks_means(tmp_path):
    out = tmp_path / "out"
    prove_path_arm(out, "llama")
    prove_path_arm(out, "llama-forged", forged=True, lean=False)
    result = PL.read("zk", str(tmp_path))
    assert result.accepted and result.format == PL.PROVE_PATH and result.proof == "zk"
    assert (result.prover_s, result.lean_s, result.proof_bytes) == (3.0, 1.0, 850000.0) and result.prover_unit == "s a proof block"
    assert result.workload == "arm llama: SiluMul_v2{I=8192} x 2 sampled of 8 proof blocks"
    assert result.ran_with == ("ran with: coins local challenger; parameters not stated "
                            "(PROVE-PATH coins challenger; summary.json coins.challenger local)")
    assert result.detail == "PROVE-PATH coins challenger; summary.json coins.challenger local"


def test_the_flock_result_is_the_plain_proofs(tmp_path):
    zk = PL.read("flock", str(prove_path_arm(tmp_path / "a", "llama")))
    assert zk.prover_s is None and "plain" in zk.why
    plain = PL.read("flock", str(prove_path_arm(tmp_path / "b", "llama", proof="plain")))
    assert plain.accepted and plain.prover_s == 3.0


def test_a_prove_path_run_whose_lean_verify_refused_prints_no_number(tmp_path):
    result = PL.read("zk", str(prove_path_arm(tmp_path, "llama", lean=False)))
    assert not result.accepted and result.prover_s is None and "lean-verify/m0 False" in result.lean


def test_every_rank_needs_its_own_lean_verify_verdict(tmp_path):
    """Two ranks with one audited A5 line between them are not every rank verified by Lean, though that one line is."""

    assert PL.read("zk", str(prove_path_arm(tmp_path / "a", "llama", ranks=2))).accepted
    result = PL.read("zk", str(prove_path_arm(tmp_path / "b", "llama", ranks=2, a5_lines=1)))
    assert not result.accepted and result.prover_s is None and "not every rank has a Lean verify verdict" in result.lean
    assert "1 A5 result line(s) for 2 rank(s), 0 without a Lean verify verdict: not one verdict a rank" in result.notes


def test_the_printed_result_states_its_proof(tmp_path):
    """Each printed benchmark result says its proof; a plain a5 out given as ``--sampling`` (Flock with zero knowledge at the sampling
    plan) says plain, prints no number and says why."""

    results = [PL.read("sampling", str(a5_out(tmp_path / "p", plain=True))), PL.read("zk", str(prove_path_arm(tmp_path, "z"))),
            PL.read("flock", None)]
    text = PL.render(results)
    line = table_lines(text)
    assert line["with sampling"].startswith("| with sampling | values not stated: workload llama32-1b__int8__h200__tp1__b4: ")
    assert "| plain | not stated | not stated | — |" in line["with sampling"]
    assert "| zk | not stated | ε not stated at δ 0.5 | 3 s a sampled proof block (Flock's prove_total_s" in line["Flock with zero knowledge"]
    assert line["Flock"].startswith("| Flock | not run | — | — | — | — |")
    assert ("- with sampling: no number: --sampling is the zk proof's benchmark result, and this output says its proof is plain"
            in text.splitlines())


def table_lines(text: str) -> dict[str, str]:
    return {x.split(" | ")[0][2:]: x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| layer ")}


def test_a_plain_sampling_result_writes_null_with_why_under_every_id(tmp_path):
    results = [PL.read("flock", None), PL.read("zk", None), PL.read("sampling", str(a5_out(tmp_path, plain=True)))]
    own = [m for m in PL.measured(results, IDS) if m["member"] == "sampling"]
    assert [m["name"] for m in own] == list(IDS)
    assert all(m["value"] is None and "says its proof is plain" in m["why"] and m["extra"]["proof"] == "plain" for m in own)


def test_an_a5_run_reads_its_audit_its_timing_and_states_no_more_of_its_coins_than_it_says(tmp_path):
    result = PL.read("sampling", str(a5_out(tmp_path)))
    assert result.accepted and result.format == PL.A5
    assert (result.prover_s, result.lean_s, result.proof_bytes) == (5.0, 2.0, 125.0)
    assert result.workload == ("values not stated: workload llama32-1b__int8__h200__tp1__b4: RoPEHead_v1{D=64} x 4 sampled of 112 proof blocks "
                            "(kind=served commitment=rows-sha512)")
    assert result.ran_with == "ran with: not stated"
    assert result.detail == "A5 line's verdicts challenger/m0, a challenger per sampled member"
    assert "replay rc=0 a5 rc=0" in result.notes and "serving stage build PASS 2026-11-15T10:00:00Z" in result.notes
    assert "arm dev" in result.notes


def test_an_a5_result_is_checked_against_the_proof_of_the_layer_it_is_given_as(tmp_path):
    """A plain a5 out given as ``--zk`` prints no seconds in the zero-knowledge benchmark result; its zk out does, with the coins its
    summary names."""

    plain = PL.read("zk", str(a5_out(tmp_path / "plain", plain=True)))
    assert plain.proof == "plain" and plain.prover_s is None and "says its proof is plain" in plain.why
    coins = {"source": "challenger", "challenger": "local"}
    zk = PL.read("zk", str(a5_out(tmp_path / "zk", coins=coins)))
    assert zk.accepted and zk.prover_s == 5.0
    assert zk.ran_with == "ran with: coins local challenger; parameters not stated (summary.json coins.challenger local)"


def test_a_stopped_a5_run_says_where_it_stopped(tmp_path):
    result = PL.read("sampling", str(a5_out(tmp_path, stopped=True)))
    assert not result.accepted and result.prover_s is None and result.why == "STOPPED at commit (no window)"


def test_a_result_given_another_commands_output_or_nothing_readable_says_so(tmp_path):
    assert "takes prove_path" in PL.read("flock", str(recursion(tmp_path / "r"))).why
    assert PL.read("zk", str(tmp_path / "missing")).status == "unreadable"
    (tmp_path / "empty").mkdir()
    assert "none of the outputs" in PL.read("zk", str(tmp_path / "empty")).why


def test_a_run_id_is_read_from_the_record_its_fetcher_lands(tmp_path):
    root = recursion(tmp_path)
    result = PL.read("recursion", "r20261115-183449-c6f6", fetcher=lambda run: root)
    assert result.accepted and result.given == "r20261115-183449-c6f6"


#: `benchmarks/prove/run.py`'s ``LAYER_METRICS``
IDS = {"benchmarks/proof-layer-prover-seconds": "prover_s", "benchmarks/proof-layer-lean-seconds": "lean_s",
       "benchmarks/proof-layer-proof-bytes": "proof_bytes", "benchmarks/proof-layer-build-seconds": "build_s"}
STEPS = ("10:00:00Z build: --stages build --config-run 1 --replay-k 460\n10:01:29Z build rc=0\n"
         "10:01:29Z build cache: no (no Program cache; the JIT caches under /w/a5-llama1b/cache held 0 files before it)\n")


def test_the_command_prints_every_layer_and_writes_each_id_flock_zero_knowledge_then_sampling(tmp_path, monkeypatch,
                                                                                              capsys):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", "local")
    monkeypatch.setenv("OUT", str(tmp_path / "report"))
    assert main(["prove-layers", "--recursion", str(recursion(tmp_path / "r")),
                 "--zk", str(prove_path_arm(tmp_path / "z", "llama")),
                 "--sampling", str(a5_out(tmp_path / "s", steps=STEPS))], metrics=IDS) == 0
    text, line = capsys.readouterr().out.rstrip("\n").rsplit("\n", 1)
    for title in PL.LAYERS.values():
        assert f"| {title} |" in text
    assert "| Flock | not run |" in text and "- Flock: no number: not run" in text
    doc = json.loads(line)
    assert doc["schema"] == PL.SCHEMA and [r["layer"] for r in doc["results"]] == list(PL.LAYERS)
    assert json.loads((tmp_path / "report" / "proof-layers.json").read_text()) == doc
    written = json.loads((tmp_path / "report" / "result.json").read_text())["measurements"]
    assert [(m["name"], m.get("member")) for m in written] == [
        (name, member) for name in IDS for member in ("flock", "flock-zero-knowledge", "sampling")]
    values = {(m["name"], m.get("member")): m for m in written}
    assert [values[name, "flock-zero-knowledge"]["value"] for name in IDS] == [3.0, 1.0, 850000.0, None]
    assert [values[name, "sampling"]["value"] for name in IDS] == [5.0, 2.0, 125.0, 89.0]
    assert all(values[name, "flock"]["value"] is None and values[name, "flock"]["why"].startswith("not run") for name in IDS)
    assert {m["unit"] for m in written} == {"s a proof block", "bytes a proof block", "s a build"} and all(m["mark"] == "measured"
                                                                                           for m in written)
    assert doc["results"][3]["ran_with"].startswith("ran with: coins in-process; parameters not stated (")
    assert doc["results"][3]["prover_s"] == 1202.0
    head = text.splitlines()
    assert head[1:6] == [PL.NOT_AN_AUDIT, "- Flock: ran with: not stated",
                         ("- Flock with zero knowledge: ran with: coins local challenger; parameters not stated "
                          "(PROVE-PATH coins challenger; summary.json coins.challenger local)"),
                         "- with sampling: ran with: not stated",
                         "- with recursion: " + doc["results"][3]["ran_with"]]
    assert doc["audit"] is False and "audit" not in text.replace(PL.NOT_AN_AUDIT, "")
    line = table_lines(text)
    assert "| 1202 s a run, wall (one recursive run's path" in line["with recursion"]
    assert "| 90 s a run, wall (the lean step whole" in line["with recursion"]
    assert "| 18,000,000 bytes a run (V*'s 10 releases" in line["with recursion"]
    assert "| 850,000 bytes a sampled proof block (every rep's proof files" in line["Flock with zero knowledge"]



def test_an_a5_runs_build_step_gives_its_wall_and_its_cache_and_its_prover_log_the_batch(tmp_path):
    """The build step's wall is its start line to its ``rc=0`` line, across midnight too, and the point carries its ``build
    cache:`` words and the proof blocks the prover's statement holds."""

    log = "statement rope-head/neox-bf16/pair instances=880\nprove ok\nstatement rope-head/neox-bf16/pair instances=880\n"
    result = PL.read("sampling", str(a5_out(tmp_path / "a", steps=STEPS, prover_log=log)))
    assert (result.build_s, result.prover_batch) == (89.0, 880)
    assert result.build_cache.startswith("no (no Program cache;") and "holds 880 proof blocks, the prover's batch" in result.prover_what
    none = [PL.read("flock", None), PL.read("zk", None)]
    (point,) = [m for m in PL.measured([*none, result], {"benchmarks/proof-layer-build-seconds": "build_s"}) if m["value"] is not None]
    assert point["unit"] == "s a build" and point["extra"]["build_cache"] == result.build_cache
    assert point["extra"]["prover_batch"] == 880
    assert "| 89 s a build, wall (steps.txt's build step" in table_lines(PL.render([result]))["with sampling"]
    late = PL.read("sampling", str(a5_out(tmp_path / "b", steps="23:59:30Z build: x\n00:00:40Z build rc=0\n")))
    assert (late.build_s, late.build_cache) == (70.0, PL.NOT_STATED)


@pytest.mark.parametrize("steps", ["10:00:00Z build: x\n10:01:00Z build rc=10\n", "10:00:00Z build: x\n", ""])
def test_a_build_step_with_no_rc_0_gives_no_build_seconds(tmp_path, steps):
    result = PL.read("sampling", str(a5_out(tmp_path, steps=steps,
                                            prover_log="statement a instances=4\nstatement b instances=8\n")))
    assert result.build_s is None and result.prover_batch is None
    (point,) = PL.measured([PL.read("flock", None), PL.read("zk", None), result],
                           {"benchmarks/proof-layer-build-seconds": "build_s"})[2:]
    assert point["value"] is None and "no build step's wall" in point["why"]

def test_a_layer_without_an_accept_or_a_per_block_figure_writes_no_number_under_any_id(tmp_path):
    """A refused Lean verify and an a5 member whose timing is keyed by its chunk's name (no count of its sampled proof
    blocks) each write null with why, never a figure; recursion's benchmark result is no member of any id."""

    refused = PL.read("zk", str(prove_path_arm(tmp_path / "z", "llama", lean=False)))
    chunk = PL.read("sampling", str(a5_out(tmp_path / "s", draw_counts={"OtherTemplate_v1": 4})))
    assert chunk.accepted and chunk.prover_s is None
    assert any(n.startswith("no figure a proof block: member RoPEHead_v1{D=64}'s timing has no count of sampled proof blocks")
               for n in chunk.notes)
    results = [PL.read("flock", None), refused, chunk, PL.read("recursion", str(recursion(tmp_path / "r")))]
    written = PL.measured(results, IDS)
    assert len(written) == 12 and all(m["value"] is None and m["why"] for m in written)
    assert {m["member"] for m in written} == {"flock", "flock-zero-knowledge", "sampling"}
    with pytest.raises(ValueError, match="not one of"):
        PL.measured(results, {"benchmarks/x": "seconds"})


@pytest.mark.parametrize("out", ["", None])
def test_an_empty_out_or_run_dir_is_no_place_to_write(tmp_path, monkeypatch, capsys, out):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_RUN_DIR", "")
    if out is None:
        monkeypatch.delenv("OUT", raising=False)
    else:
        monkeypatch.setenv("OUT", out)
    kit.report(PL.REPORT, "text", {"results": []}, PL.measured([PL.read(x, None) for x in PL.LAYERS], IDS))
    assert capsys.readouterr().out == 'text\n{"results": []}\n' and not any(tmp_path.iterdir())


def test_with_no_number_in_any_result_the_command_says_so_first_and_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("OUT", raising=False)
    assert main(["prove-layers", "--zk", str(tmp_path / "missing")]) == 1
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("prove layers: no number: flock: not run: no output given (--flock DIR|RUN); zk: unreadable: ")
    assert main(["prove-layers", "--recursion", str(recursion(tmp_path / "r"))]) == 0
    assert capsys.readouterr().out.startswith("The proof benchmark by layer")


def test_the_help_names_the_command_as_the_readme_does(capsys):
    with pytest.raises(SystemExit):
        main(["prove-layers", "--help"])
    assert capsys.readouterr().out.startswith("usage: verity bench prove layers ")


def test_a_run_id_the_store_cant_give_says_to_give_its_out_directory(monkeypatch):
    monkeypatch.setitem(sys.modules, "research.store", None)
    result = PL.read("recursion", "r20261115-183449-c6f6")
    assert result.status == "unreadable" and "isn't installed here" in result.why and PL.FIX in result.why


def test_the_device_is_the_outputs_own_words(tmp_path):
    cpu = {"RoPEHead_v1{D=64}": {"gpus": 0, "device": None, "lean_verify_s": 8.0,
                                 "timing": {"prove_total_s": 20.0, "proof_bytes": [500, 500]}}}
    assert PL.read("sampling", str(a5_out(tmp_path / "c", timing=cpu))).device == "CPU (timing gpus 0)"
    assert PL.read("sampling", str(a5_out(tmp_path / "n"))).device == "not stated"
    assert PL.read("recursion", str(recursion(tmp_path / "r"))).device == "not stated"


def test_prove_without_the_store_refuses_in_one_line(monkeypatch, capsys):
    """An outside machine has no store client, or no remote it can read: `prove` names what it needs and exits 2."""

    monkeypatch.setitem(sys.modules, "research.store.local", None)
    assert main(["prove", "--system", "flock", "--workload", "glm53"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("verity bench prove: needs the evidence store's client")

    class NoRemote:
        remote, remote_error = None, "no remote is configured"

    local = types.ModuleType("research.store.local")
    local.LocalStore = NoRemote  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "research.store.local", local)
    assert main(["prove", "--system", "flock", "--workload", "glm53"]) == 2
    assert capsys.readouterr().err.strip().endswith("the evidence store has no remote here (no remote is configured)")


SHA = "85b64d2c451637c4ac5b6bd0e2f6a934baea69ac"
STMTS = (("s8-coord-k2048-c0", "ScaledMmS8Coordinate_v1{K=2048}", 2048), ("s8-coord-k8192-c0", "ScaledMmS8Coordinate_v1{K=8192}", 512))


def layers_sh(root: Path, layer: str, *, k: int = 2560, zk_flag: bool | None = None, negative: bool = True) -> Path:
    """One layer of a `benchmarks/one_stage/layers.sh all` run under `root` (its flock/, zk/ or sampled/), in the shapes
    its stand-in run wrote."""

    d = root / layer / "out"
    (d / "a6").mkdir(parents=True)
    (d / "layers.txt").write_text("ran with: coins local challenger; parameters development (a development challenger, "
                                  f"local to each session; δ=2^-20; k={k})\na measurement, not an audit: stand-in values, never served\n")
    (d / "job.log").write_text(f"21:36:07Z hidden-zk: source {SHA}, 96 CPUs\n{'7b' * 32}  flock-circuit\n")
    (root / layer / "gpu-lease-usage.jsonl").write_text(json.dumps(
        {"cmd": "flock-circuit prove --zk --gpu --verifier 127.0.0.1:7561", "gpus": [{"index": 4}]}) + "\n")
    for name, _, _ in STMTS:
        (d / "a6" / f"session-{name}").mkdir()
        (d / "a6" / f"session-{name}" / "lean-verify.log").write_text(
            "$ verity-verify-proof verify --statement verity/flock-circuit"
            + (" --zk" if (layer != "flock" if zk_flag is None else zk_flag) else "") + " --coins tree\n")
    if layer == "flock":
        summary = {"accepted": True, "plain": True, "windows": [
            {"name": name, "units": n, "live": {"accepted": True, "prove_total_s": 2.0 + i, "proof_bytes": [900 + i, 900 + i]},
             "lean_verify": {"accepted": True, "verify_s": 5.0 + i, "unit_draw": {"strata": [{"template": t}]}}}
            for i, (name, t, n) in enumerate(STMTS)]}
    else:
        counts = {"s8-coord-k2048-c0": 2048, "s8-coord-k8192-c0": 512} if k == 2560 else {"s8-coord-k2048-c0": 26, "s8-coord-k8192-c0": 6}
        summary = {"audit": {"accepted": True, "complete": True, "verdicts": {f"lean-verify/{n}": True for n, _, _ in STMTS},
                             "wrong_units": {"bound": 0 if k == 2560 else 895, "of": 2560, "delta": 2.0 ** -20}},
                   "checks_before_draw": {"registration": True}, "negatives": {"draw-outside-the-partition": negative},
                   "draw_counts": counts, "k": k, "population": 2560,
                   "members": [{"name": n, "template": t} for n, t, _ in STMTS],
                   "timing": {n: {"lean_verify_s": 6.0 + i, "timing": {"prove_total_s": 4.0 + i, "proof_bytes": [1100, 1100]}}
                              for i, (n, _, _) in enumerate(STMTS)}}
    (d / "a6" / "summary.json").write_text(json.dumps(summary))
    return root


def layers_run(root: Path) -> Path:
    layers_sh(root, "flock"), layers_sh(root, "zk"), layers_sh(root, "sampled", k=32)
    return root


def test_a_layers_sh_run_gives_its_three_layers_each_a_statements_figures_as_stand_in_values(tmp_path):
    """layers.sh's output is recognised by its own files (layers.txt beside a6/summary.json), each layer from its own
    directory of the run, and the benchmark result says what the output says: stand-in values, figures a statement, its ε at δ."""

    run = layers_run(tmp_path / "run")
    results = {x: PL.read(x, str(run)) for x in ("flock", "zk", "sampling")}
    assert {r.format for r in results.values()} == {PL.LAYERS_SH} and all(r.accepted for r in results.values())
    assert all(r.workload.startswith("stand-in values, never served: 2 statements of 2,560 proof blocks") for r in results.values())
    assert (results["zk"].commit, results["zk"].commit_from) == (SHA, "job.log's source") and "32 of them sampled" in results["sampling"].workload
    assert [r.proof for r in results.values()] == ["plain", "zk", "zk"]
    assert results["sampling"].ran_with.startswith("ran with: coins local challenger; parameters development (a development")
    assert results["sampling"].eps_delta == "at most 895 of 2,560 incorrect proof blocks except with probability 2^-20"
    assert results["flock"].eps_delta == PL.NOT_STATED and results["zk"].device.startswith("GPU 4 (gpu-lease's")
    line = table_lines(PL.render(list(results.values()) + [PL.read("recursion", None)]))
    assert "| 2 s a statement of 2,048 proof blocks; 3 s a statement of 512 proof blocks (Flock's prove_total_s) |" in line["Flock"]
    assert "| 6 s a statement of 26 proof blocks; 7 s a statement of 6 proof blocks (Lean's verify) |" in line["with sampling"]
    assert "900 bytes a statement of 2,048 proof blocks; 901 bytes a statement of 512 proof blocks (one rep's proof)" in line["Flock"]


def test_a_stand_in_result_writes_no_figure_a_proof_block_under_the_metric_ids(tmp_path):
    run = str(layers_run(tmp_path / "run"))
    results = [PL.read(x, run if x != "recursion" else None) for x in PL.LAYERS]
    written = PL.measured(results, IDS)
    assert len(written) == 12 and all(m["value"] is None for m in written)
    assert all(m["why"] == ("its output gives no build step's wall (steps.txt)" if IDS[m["name"]] == "build_s" else
                            "stand-in values, never served: its figures are a statement's, not a proof block's") for m in written)


def test_a_layers_sh_result_with_a_negative_case_accepted_or_no_zk_flag_prints_no_number(tmp_path):
    neg = PL.read("zk", str(layers_sh(tmp_path / "a", "zk", negative=False)))
    assert not neg.accepted and neg.statements == [] and "negative draw-outside-the-partition not refused" in neg.lean
    plain = PL.read("zk", str(layers_sh(tmp_path / "b", "zk", zk_flag=False)))
    assert plain.proof is None and plain.statements == [] and "states no proof" in plain.why
    assert PL.read("recursion", str(layers_run(tmp_path / "c"))).status == "unreadable"


def test_a_prove_path_result_names_the_commit_its_machine_txt_states(tmp_path):
    out = tmp_path / "out"
    prove_path_arm(out, "z")
    (out / "machine.txt").write_text(f"11\nMem: 755\n{'c352729e2ee' + '0' * 29}\n{'5e' * 20}\n")
    result = PL.read("zk", str(out))
    assert result.commit == "c352729e2ee" + "0" * 29 and result.workload.startswith("arm z: ")
    commits = next(x for x in PL.render([result]).splitlines() if x.startswith("- commits: "))
    assert commits.startswith("- commits: Flock with zero knowledge c352729e2ee (machine.txt's git HEAD); a benchmark result reruns from main")
    (out / "machine.txt").write_text("11\n")
    assert PL.read("zk", str(out)).commit is None
    assert "- commits: Flock with zero knowledge not stated; " in PL.render([PL.read("zk", str(out))])


def test_a_served_a5_result_states_its_values_its_bound_its_coins_and_its_commit_from_its_files(tmp_path):
    """A served a5 run's shapes: summary.json's source served beside REGISTERED kind=served, the audit's
    bound on incorrect proof blocks, summary.txt's COINS line and machine.txt's git HEAD (a5_llama1b_h200.sh's third line)."""

    out = a5_out(tmp_path, source="served", audit={"wrong_units": {"bound": 2, "of": 880, "delta": 0.5}}) / "out"
    (out / "machine.txt").write_text(f"uid=0(root)\n13722cef5f43\n{'45400e83995d' + '0' * 28}\nrow dir /w/r/llama32-1b__int8\n")
    with (out / "summary.txt").open("a") as f:
        f.write("COINS draw: one local challenger session (a development stand-in for the auditor's), law subset k=4 of 112 "
                "units, delta 0.5; sessions: Flock --zk behind the Lean ZK server\n")
    result = PL.read("sampling", str(tmp_path))
    assert result.workload.startswith("served values: workload llama32-1b__int8: RoPEHead_v1{D=64} x 4 sampled of 112 proof blocks (kind=served")
    assert result.eps_delta == "at most 2 of 880 incorrect proof blocks except with probability 0.5"
    assert result.ran_with == ("ran with: coins local challenger; parameters not stated (summary.txt COINS draw: one local "
                            "challenger session (a development stand-in for the auditor's), law subset k=4 of 112 units, delta 0.5; "
                            "sessions: Flock --zk behind the Lean ZK server)")
    assert result.commit == "45400e83995d" + "0" * 28 and result.proof_bytes == 125.0


def test_an_a5_plain_run_is_the_flock_result_and_its_epsilon_table_gives_its_first_and_last_pairs(tmp_path):
    """The one-stage script's plain run as ``--flock``, and a sampled run's ``epsilon`` table."""

    table = [{"delta": d, "wrong_units": w} for d, w in ((0.5, 2), (0.1, 6), (1e-9, 57))]
    flock = PL.read("flock", str(a5_out(tmp_path / "p", plain={"coins": "tree"}, epsilon={"population": 880, "table": table})))
    assert flock.accepted and flock.format == PL.A5 and flock.proof == "plain" and flock.prover_s == 5.0
    assert flock.eps_delta == ("at most 2 of 880 incorrect proof blocks except with probability 0.5; at most 57 except with "
                               "probability 1e-9 (summary.json epsilon, the first and last of its 3 pairs)")


def served_arm(root: Path, layer: str, *, records: bool = False) -> Path:
    """prove_path's served-arm run (with PROVE-PATH, or under PP_RECORDS=1 with RECORDS): its layer's
    ``out/`` with ran-with.txt, machine.txt, the served arm and a forged arm whose commit failed."""

    out = root / layer / "out"
    arm = out / "served"
    (arm / ("window" if records else "rank0")).mkdir(parents=True)
    (out / "ran-with.txt").write_text(
        "ran with: coins local challenger; parameters development (a local development challenger, under a small test key)\n"
        f"served, compiled and verified at {SHA} (branch b), not main: window 0 of vllm's run r1, layer 0's SiluMul_v2 units\n")
    (out / "machine.txt").write_text(f"96\n{SHA}\n")
    audit = {"accepted": True, "verdicts": {"lean-draw/m0": True, "lean-verify/m0": True},
             "wrong_units": {"bound": 0, "delta": 0.5, "of": 3}}
    counts = {"SiluMul_v2{I=8192}": 3}
    last = ("RECORDS " + json.dumps({"accepted": True, "ok": True, "negatives_refused": True, "draw_counts": counts,
                                     "population": 3}) if records else
            "PROVE-PATH " + json.dumps({"forged": False, "ok": True, "proof": "plain", "coins": "challenger", "delta": 0.5,
                                        "ranks": {"0": {"accepted": True, "population": 3, "draw_counts": counts,
                                                        "per_unit": {"SiluMul_v2{I=8192}": {"units": 3, "proof_bytes": 1248644}}}}}))
    (arm / "audit.log").write_text("A5 " + json.dumps({"audit": audit, "source": "stand-in"}) + "\n" + last + "\n")
    (arm / "summary.json").write_text(json.dumps({"source": "stand-in"} if records else {"plain": {"coins": "tree"}}))
    timing = {"SiluMul_v2{I=8192}": {"gpus": 0, "lean_verify_s": 3.0,
                                      "timing": {"prove_total_s": 6.0, "proof_bytes": [624321, 624321]}}}
    (arm / ("window" if records else "rank0") / "summary.json").write_text(
        json.dumps({"source": "stand-in", "draw_counts": counts, "timing": timing}))
    (out / "served-forged").mkdir()
    (out / "served-forged" / "commit.log").write_text(
        "Traceback\nLookupError: no rows of SiluMul_v2{I=8192} at steps [24, 25]; its rows exist outside the scope\n")
    return root


@pytest.mark.parametrize("layer", ["flock", "zk"])
def test_a_served_arm_run_reads_its_ran_with_its_one_rep_and_says_it_has_no_forged_run(tmp_path, layer):
    result = PL.read(layer, str(served_arm(tmp_path, layer, records=layer == "zk")))
    assert result.accepted and result.proof == ("plain" if layer == "flock" else "zk")
    assert (result.prover_s, result.lean_s, result.proof_bytes) == (2.0, 1.0, 208107.0)
    assert result.bytes_what.startswith("one rep's proof over its sampled proof blocks, of 2 reps in summary.json's timing")
    assert result.ran_with == ("ran with: coins local challenger; parameters development (ran-with.txt: a local development "
                            "challenger, under a small test key)")
    assert result.workload == ("the output disagrees on its values: ran-with.txt says served (window 0 of vllm's run r1, "
                            "layer 0's SiluMul_v2 units), the audit's source field says stand-in: arm served: "
                            "SiluMul_v2{I=8192} x 3 sampled of 3 proof blocks; no forged run")
    assert "no forged run: served-forged/commit.log: LookupError: no rows of SiluMul_v2{I=8192} at steps [24, 25]" in result.notes
    assert result.eps_delta == "at most 0 of 3 incorrect proof blocks except with probability 0.5"
    assert (result.commit, result.device) == (SHA, "CPU (timing gpus 0)")


def test_a_records_arm_whose_negatives_were_not_refused_prints_no_number(tmp_path):
    arm = served_arm(tmp_path, "zk", records=True) / "zk" / "out" / "served"
    log = (arm / "audit.log").read_text().replace('"negatives_refused": true', '"negatives_refused": false')
    (arm / "audit.log").write_text(log)
    result = PL.read("zk", str(tmp_path))
    assert not result.accepted and result.prover_s is None and "RECORDS negatives_refused False" in result.lean


def test_a_layers_sh_result_and_an_a5_result_whose_output_states_neither_stand_in_nor_served_say_values_not_stated(tmp_path):
    d = layers_sh(tmp_path / "l", "zk") / "zk" / "out"
    (d / "layers.txt").write_text((d / "layers.txt").read_text().splitlines()[0] + "\n")
    assert PL.read("zk", str(tmp_path / "l")).workload.startswith("values not stated: 2 statements of 2,560 proof blocks")
    a5 = PL.read("sampling", str(a5_out(tmp_path / "a")))
    assert a5.workload.startswith("values not stated: workload llama32-1b__int8__h200__tp1__b4: ")
    assert PL.read("sampling", str(a5_out(tmp_path / "s", source="served"))).workload.startswith("served values: ")


def test_a_layers_sh_results_commit_is_the_sha_after_source_never_one_printed_before_it(tmp_path):
    d = layers_sh(tmp_path, "zk") / "zk" / "out"
    (d / "job.log").write_text(f"21:30:00Z fetch: main at {'1' * 40}\n" + (d / "job.log").read_text())
    assert PL.read("zk", str(tmp_path)).commit == SHA
    (d / "job.log").write_text(f"21:30:00Z fetch: main at {'1' * 40}\n")
    assert PL.read("zk", str(tmp_path)).commit is None
