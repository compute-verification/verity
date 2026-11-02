"""te_core.rescore scores the ledger rows whose formalism block was measured on a traced step, under both readings of
Slowdown's frozen-tail case, beside an analysis's blocks for the same checkpoints; with --quantities analysis it scores
the analysis's blocks, joined by run and MACs, and leaves out the valid rows the analysis has no block for; with
--stubs every valid row is scored on te_core.formalism's stubs."""
import json

import pytest

pytest.importorskip("matplotlib")

from te_core import formalism, plots, policy, rescore
from test_plots import SHAPE, rows

GATES_PER_MAC = 7000.0
ANALYSIS = f'''
TAU = 16
MACS = {[4e14, 1e15, 6e14]!r}
E = {[9e7, 9e7, 8e4]!r}

def gate_factors(cache):
    cache.write_text("{{}}")
    return {{}}

def load(scorer, ledger):
    return {{}}, {{f"s{{i}}": {{"run": f"run{{i}}"}} for i in range(3)}}, {{}}, {{}}

def build(subs, launch, starts, counted, g):
    records = {{f"s{{i}}": {{"delta_grad": 1e9, "a": 512 * 16, "t": 0.5 if i == 0 else 0.0,
                            "family": "full" if i < 2 else "adapter"}} for i in range(3)}}
    cps = [{{"id": f"s{{i}}", "macs": MACS[i], "gates": {GATES_PER_MAC} * MACS[i], "tokens": 1e6, "phi": 0.9,
             "delta": 8 * E[i]}} for i in range(3)]
    return records, cps
'''


TRACE = {"tau_ids": 1.125, "tau_other_bits": 0.0, "a_bits": 512 * 16.0, "u1_bits": 4e8, "tail_share": 0.3,
         "u": [[0.0, 0.0], [0.0, 1e8], [0.5, 4e8], [1.0, 4e8]], "frozen_tail_share": 0.1, "update_cut_bits": 4e8}


def ledger_rows():
    """r0 to r3 at their MACs; r0 and r1 measured on a traced step (φ 0.95, the step's tokens, Δ_grad 2e9)."""
    out = [r | {"macs": m} for r, m in zip(rows(), (4e14, 1e15, 6e14, 2e15))]
    for r in out[:2]:
        f = formalism.of_trace(TRACE)
        r["te_core"] = "4.0"
        r["formalism"] = f | {"parts": [{"gates": GATES_PER_MAC * r["macs"], "tokens": 1e6, "steps": 10,
                                         "dep_gates": 0.95 * GATES_PER_MAC * r["macs"], "delta_grad_bits": 2e9}]}
    return out


def run(tmp_path, monkeypatch, *extra):
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    fake = tmp_path / "rescore_v1.py"
    fake.write_text(ANALYSIS)
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(ledger_rows),
                                                                      "root": tmp_path})())
    out = tmp_path / "out"
    assert rescore.main(["--out", str(out), "--pin", "pin", "--manifest", str(man), "--analysis", str(fake),
                         "--scorer", str(tmp_path), *(extra or ("--quantities", "analysis"))]) == 0
    return out, fake


def test_the_analysis_blocks_score_the_rows_they_cover_and_the_rest_are_left_out(tmp_path, monkeypatch):
    out, fake = run(tmp_path, monkeypatch)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == str(fake) and fr["not_analysed"] == ["r3"]
    assert fr["readings"] == list(formalism.READINGS)
    assert {t["reading"] for v in fr["starts"].values() for t in v["targets"]} == set(formalism.READINGS)
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1", "r2"] and {k["source"] for k in kept} == {"analysis"}
    assert [k["family"] for k in kept] == ["full", "full", "adapter"]
    f0, f1 = kept[0]["formalism"], kept[1]["formalism"]
    assert f0["u"] == [list(p) for p in formalism.even(8 * 9e7)] and f0["tail_share"] == 0.5
    assert f0["labels"]["tail_share"] == rescore.TAIL and f1["tail_share"] == 0.0 and "tail_share" not in f1["labels"]
    for n in plots.NAMES:
        assert (out / f"{n}.png").exists() and (out / f"{n}.json").exists()
    assert "| literal-x0 |" in (out / "frontier.md").read_text() and "| tail-charged |" in (out / "frontier.md").read_text()
    assert (out / "analysis-gates.json").exists()


def test_the_readings_differ_only_on_a_record_with_a_frozen_tail(tmp_path, monkeypatch):
    """r0's block has a frozen tail (t = 0.5): tail-charged's ρ is above literal-x0's at the reference policy; r1 and r2
    have none, so the readings agree there."""
    out, _ = run(tmp_path, monkeypatch)
    cells = {"reference": policy.cells(SHAPE)["reference"]}
    rho = {}
    for line in (out / "formalism.jsonl").read_text().splitlines():
        k = json.loads(line)
        by, _ = formalism.rhos(formalism.measure({"formalism": k["formalism"]}, SHAPE), cells)
        rho[k["id"]] = {name: v["reference"] for name, v in by.items()}
    assert rho["r0"][formalism.TAIL_CHARGED] > rho["r0"][formalism.LITERAL] > 1
    assert rho["r1"][formalism.TAIL_CHARGED] == rho["r1"][formalism.LITERAL]


def test_one_reading_scores_only_under_it(tmp_path, monkeypatch):
    out, _ = run(tmp_path, monkeypatch, "--quantities", "analysis", "--reading", formalism.LITERAL)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["readings"] == [formalism.LITERAL]
    assert {t["reading"] for v in fr["starts"].values() for t in v["targets"]} == {formalism.LITERAL}
    p2 = plots.frame(json.loads((out / "p2-slowdown-vs-gain.json").read_text()))
    assert {r["reading"] for r in p2} == {formalism.LITERAL}


def test_an_attached_block_is_what_the_formalism_measures(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    fake = rescore.load_analysis(tmp_path / "rescore_v1.py")
    _, launch, _, _ = fake.load(None, None)
    records, cps = fake.build(None, launch, None, None, None)
    blocks = {(launch[c["id"]]["run"], c["macs"]): (rescore.block(c, records[c["id"]], fake.TAU),
                                                    records[c["id"]]["family"]) for c in cps}
    got, source = rescore.attach(ledger_rows(), blocks)
    assert source == {"r0": "analysis", "r1": "analysis", "r2": "analysis", "r3": "stubs"}
    assert got[0]["family"] == "full" and "family" not in got[3] and "trace" not in got[0]["formalism"]
    m = formalism.measure(got[0], SHAPE)
    assert m.gates == pytest.approx(GATES_PER_MAC * 4e14) and m.tail_share == 0.5
    assert m.delta_bits == m.u1_bits == 8 * 9e7 and m.tau_bits == 16
    assert m.labels["u1_bits"].startswith("upper bound") and m.labels["a_bits"].startswith("upper bound")


def test_stubs_score_every_valid_row(tmp_path, monkeypatch):
    out, _ = run(tmp_path, monkeypatch, "--stubs")
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == "stubs: --stubs" and fr["not_analysed"] == [] == fr["not_measured"]
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1", "r2", "r3"] and {k["source"] for k in kept} == {"stubs"}
    assert not (out / "analysis-gates.json").exists()


@pytest.mark.parametrize("tail", ["literal", "frozen"])
def test_measured_rows_are_scored_beside_the_analysis(tmp_path, monkeypatch, tail):
    """The default scores r0 and r1, the rows measured on a traced step, and leaves out r2 and r3; τ is the traced
    positions at ⌈log2 V⌉ bits; compare.json puts each beside the analysis's block for its checkpoint. --tail frozen
    reads t as the trace's contiguous frozen tail."""
    out, _ = run(tmp_path, monkeypatch, "--tail", tail)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == f"measured (trace; t {tail})" and fr["not_measured"] == ["r2", "r3"]
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1"] and {k["source"] for k in kept} == {"measured"}
    t = 0.3 if tail == "literal" else 0.1
    assert {k["formalism"]["tail_share"] for k in kept} == {t}
    c = json.loads((out / "compare.json").read_text())
    (r0, r1) = c["rows"]
    assert r0["measured"]["tau_bits"] == pytest.approx(1.125 * 16) and r0["analysis"]["tau_bits"] == 16
    assert r0["ratio"]["phi"] == pytest.approx(0.95 / 0.9) and r0["ratio"]["a_bits"] == 1
    assert r0["ratio"]["u1_bits"] == pytest.approx(4e8 / (8 * 9e7)) and r0["measured"]["tail_share"] == t
    assert r0["update_cut_over_delta"] == pytest.approx(4e8 / (8 * 9e7))
    assert set(r0["rho"]) == set(formalism.READINGS) and set(r0["rho_analysis"]) == set(formalism.READINGS)
    assert {(e["start"], e["reading"]) for e in c["S"]} == {("s1", r) for r in formalism.READINGS}
    assert "| s1 |" in (out / "compare.md").read_text()
