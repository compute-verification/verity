"""The lead's kernel off the device (benchmarks/pouw/np_kernel): cpu_check.cc's ragged calls, whose padding rows carry their
tags, lane-coin A' and coins (and a control whose padding rows have none is rejected), the fold's window, exact at
`serving.LEAD_FOLD_BOUND` and at no other bound, G's rejection draw, ssn.cu's check holding served outputs within the same
bound, gate.py's GateRecord from a run's out directory, and the recorded one (gate.json) against this tree."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from verity.core.protocols.pouw.serving import LEAD_FOLD_BOUND, POUW_V2, SERIAL, variant_refusals

HERE = Path(__file__).resolve().parents[1] / "np_kernel"
spec = importlib.util.spec_from_file_location("np_kernel_gate", HERE / "gate.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


@pytest.fixture(scope="module")
def cpu_check(tmp_path_factory) -> Path:
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for cpu_check.cc")
    exe = tmp_path_factory.mktemp("np-kernel") / "cpu_check"
    subprocess.run([gxx, "-O2", "-std=c++17", "-ffp-contract=off", "-o", str(exe), str(HERE / "cpu_check.cc")], check=True)
    return exe


def run(exe: Path, *args: str) -> tuple[int, list[dict]]:
    p = subprocess.run([str(exe), *args], capture_output=True, text=True)
    return p.returncode, [json.loads(s) for s in p.stdout.splitlines()]


def test_a_ragged_calls_padding_rows_carry_their_tags_lane_coin_a_and_coins(cpu_check):
    rc, lines = run(cpu_check, "--bound", str(LEAD_FOLD_BOUND), "ragged")
    calls = [x for x in lines if x.get("level") == "ragged"]
    honest = [x for x in calls if not x["control"]]
    assert rc == 0 and lines[-1]["pass"] and all(x["pass"] for x in calls)
    assert {x["layout"] for x in calls} == {"blc4a_t64_e7_s1_rot_fold"}
    decode = next(x for x in honest if x["call_m"] == 1)
    assert (decode["committed_m"], decode["computed_m"], decode["padding_rows"]) == (16, 64, 15)
    assert all(x["padding_rows_wrong"] == x["padding_rows_tagless"] == x["word_mismatches"] == x["served_mismatches"] == 0
               and x["served_outputs"] > 0 for x in honest)
    controls = [x for x in calls if x["control"]]
    assert controls and all(x["padding_rows_wrong"] == x["padding_rows"] > 0 and x["word_mismatches"] > 0 for x in controls)


def test_the_fold_is_exact_within_the_leads_bound_and_misreads_past_it(cpu_check):
    rc, lines = run(cpu_check, "--bound", str(LEAD_FOLD_BOUND), "window")
    w = lines[0]
    assert rc == 0 and w["pass"] and w["read_back_not_p"] == w["served_differ_from_stock"] == 0
    assert w["read_back_at_bound_plus_1"] == LEAD_FOLD_BOUND + 2
    assert w["read_back_at_minus_bound_minus_1"] == -LEAD_FOLD_BOUND - 0.5


@pytest.mark.parametrize("bound", [LEAD_FOLD_BOUND // 2, LEAD_FOLD_BOUND * 2])
def test_no_other_bound_is_the_folds(cpu_check, bound):
    """Half the bound reads back past itself (so it isn't the fold's edge); twice the bound misreads within itself."""
    rc, lines = run(cpu_check, "--bound", str(bound), "window")
    assert rc == 1 and not lines[0]["pass"] and not lines[-1]["pass"]


@pytest.mark.slow
def test_gs_draw_keeps_each_g_from_equally_many_words(cpu_check):
    """All 2^32 words through g_word_of (about a minute): a bias of a few words in one G, which no sample shows."""
    rc, lines = run(cpu_check, "draw")
    assert rc == 0 and lines[0]["pass"] and lines[0]["kept_per_g_min"] == lines[0]["kept_per_g_max"]


def test_ssn_checks_served_outputs_within_the_leads_bound():
    src = (HERE / "ssn.cu").read_text()
    m = re.search(r"^constexpr long long FOLD_BOUND = 1ll << (\d+);$", src, re.M)
    assert m and 1 << int(m.group(1)) == LEAD_FOLD_BOUND
    assert "pv < -FOLD_BOUND || pv > FOLD_BOUND" in src and '\\"fold_bound\\": %lld' in src


SERVED = ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192))


def line(n, k, **kw):
    return {"check": G.LEAD_LAYOUT, "control": None, "rejected": False, "poison": "0xA5", "fold_bound": LEAD_FOLD_BOUND,
            "call_m": 1, "call_n": n, "K": k, "served_outputs": 64 * n, **kw}


def out_dir(tmp_path: Path, lines: list[dict], sass_ok: bool = True) -> tuple[Path, Path]:
    root, out = tmp_path / "tree", tmp_path / "out"
    (root / "k").mkdir(parents=True)
    out.mkdir()
    (root / "k" / "ssn.cu").write_text("// the kernel\n")
    (root / "k" / "layout.h").write_text("// the layout\n")
    (out / "sources.sha256").write_text("".join(f"{hashlib.sha256((root / p).read_bytes()).hexdigest()}  {p}\n"
                                                for p in ("k/layout.h", "k/ssn.cu")))
    (out / "sass_gate.json").write_text(json.dumps({"ok": sass_ok, "binaries": [
        {"sha256": "ab" * 32, "ok": sass_ok, "cubins": [{"toolkit": "13.0.1-sm_90a", "functions": 12}]}]}))
    (out / "check.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return root, out


HONEST = [line(n, k) for n, k in SERVED]
CONTROLS = [line(3072, 2048, control="no-write", rejected=True), line(13, 100, control="no-write", rejected=True)]


def refusals(tmp_path, lines, **kw):
    root, out = out_dir(tmp_path, lines, **kw)
    v, g = G.record(out, "r-test", root)
    return v, g, variant_refusals(v, POUW_V2, SERIAL.name, SERVED, g)


def test_a_run_whose_lead_lines_pass_admits_the_lead(tmp_path, capsys):
    extreme = line(256, 8192, served_outputs=0)
    v, g, got = refusals(tmp_path, HONEST + [extreme] + CONTROLS + [{"check": "ssn_1l", "rejected": False}])
    assert got == [] and g.passed and g.variant == v.id and v.binary_sha256 == "ab" * 32
    assert g.fold_bound == LEAD_FOLD_BOUND and g.toolkit == "13.0.1-sm_90a" and set(g.shapes) == set(SERVED)
    root, out = tmp_path / "tree", tmp_path / "out"
    assert G.main([str(out), "--run", "r-test"], root) == 0 and capsys.readouterr().out.endswith("admitted\n")


@pytest.mark.parametrize(("lines", "sass_ok", "want"), [
    (HONEST[:3] + [line(2048, 8192, rejected=True)] + CONTROLS, True, ["failed exact", "didn't cover [(2048, 8192)]"]),
    (HONEST, True, ["failed controls_rejected"]),
    (HONEST + [{**CONTROLS[0], "rejected": False}], True, ["failed controls_rejected"]),
    ([{**x, "poison": None} for x in HONEST] + CONTROLS, True, ["failed poisoned_accepted"]),
    (HONEST + CONTROLS, False, ["failed sass_ok"]),
    ([{**x, "fold_bound": LEAD_FOLD_BOUND // 2} for x in HONEST] + CONTROLS, True,
     ["held outputs exact within |p| <= 2097152, its fold serves |p| <= 4194304"]),
    ([{**x, "served_outputs": 0} for x in HONEST[:3]] + HONEST[3:] + CONTROLS, True,
     ["didn't cover [(2048, 2048), (3072, 2048), (16384, 2048)]"]),
])
def test_a_run_that_missed_a_check_doesnt_admit_the_lead(tmp_path, lines, sass_ok, want):
    _, _, got = refusals(tmp_path, lines, sass_ok=sass_ok)
    assert got and all(any(w in r for r in got) for w in want), got


def test_a_run_built_from_other_sources_isnt_gated(tmp_path):
    root, out = out_dir(tmp_path, HONEST + CONTROLS)
    (root / "k" / "layout.h").write_text("// another layout\n")
    with pytest.raises(G.GateError, match="k/layout.h differ"):
        G.record(out, "r-test", root)
    assert G.main([str(out), "--run", "r-test"], root) == 2


def test_write_records_an_admitting_gate_which_admits_the_lead_only_while_its_sources_are_the_trees(tmp_path, capsys):
    root, out = out_dir(tmp_path, HONEST + CONTROLS)
    rec = tmp_path / "gate.json"
    assert G.main([str(out), "--run", "r-test", "--write"], root, rec) == 0
    v, g = G.recorded(root, rec)
    assert g.run == "r-test" and variant_refusals(v, POUW_V2, SERIAL.name, SERVED, g) == []
    assert json.loads(rec.read_text())["sources"] == {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                                      for p in ("k/layout.h", "k/ssn.cu")}
    assert G.main(["--recorded"], root, rec) == 0
    (root / "k" / "ssn.cu").write_text("// another kernel\n")
    capsys.readouterr()
    assert G.main(["--recorded"], root, rec) == 1 and f"the gate is {v.id}'s" in capsys.readouterr().out


def test_write_records_no_refused_gate(tmp_path):
    root, out = out_dir(tmp_path, HONEST)
    rec = tmp_path / "gate.json"
    assert G.main([str(out), "--run", "r-test", "--write"], root, rec) == 1 and not rec.exists()


def test_the_recorded_gate_admits_the_lead_this_tree_names():
    """gate.json against this tree (`gate.py --recorded`): the variant its listed sources name here is the one it gated, so
    a change to any of them (some are other suites' files, declared in this suite's inputs) fails here until the gate is
    rerun."""
    rec = json.loads(G.RECORD.read_text())
    v, g = G.recorded()
    assert rec["variant"] == v.id == g.variant, (
        f"gate.json gated {rec['variant']}, but this tree's sources name {v.id}: rerun run.sh (CHECK=1) on an H200, then "
        "gate.py OUT --run RUN --write")
    assert variant_refusals(v, POUW_V2, SERIAL.name, SERVED, g) == [], (
        "gate.json no longer admits the lead: rerun run.sh (CHECK=1) on an H200, then gate.py OUT --run RUN --write")


def test_the_variant_is_named_by_its_sources_under_the_leads_config(tmp_path):
    v, _, _ = refusals(tmp_path, HONEST + CONTROLS)
    root = tmp_path / "tree"
    G.record(tmp_path / "out", "r-test", root)
    import ledger

    assert v.source_sha256 == ledger.source_sha256([Path("k/layout.h"), Path("k/ssn.cu")], root,
                                                   {"BLC": "2", "ROT": "1", "FOLD": "1"})
    assert v.source_sha256 != ledger.source_sha256([Path("k/layout.h"), Path("k/ssn.cu")], root)
