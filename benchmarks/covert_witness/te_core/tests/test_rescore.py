"""te_core.rescore scores ledger rows with an analysis's formalism blocks, joined by run and MACs, and leaves out the
valid rows the analysis has no block for; with --stubs every valid row is scored on te_core.formalism's stubs."""
import json

import pytest

pytest.importorskip("matplotlib")

from te_core import formalism, rescore  # noqa: E402
from test_plots import SHAPE, rows  # noqa: E402

GATES_PER_MAC = 7000.0
ANALYSIS = f'''
TAU = 16
MACS = {[4e14, 1e15, 6e14]!r}
E = {[9e7, 9e7, 8e4]!r}

def gate_factors(cache):
    cache.write_text("{{}}")
    return {{}}

def load(scorer, ledger):
    return {{}}, {{f"s{{i}}": {{"run": f"run{{i}}"}} for i in range(3)}}, {{}}, None, {{}}

def build(subs, launch, starts, counted, g):
    records = {{f"s{{i}}": {{"delta_grad": 1e9, "a": 512 * 16, "t": 0.5 if i == 0 else 0.0}} for i in range(3)}}
    cps = [{{"id": f"s{{i}}", "macs": MACS[i], "gates": {GATES_PER_MAC} * MACS[i], "tokens": 1e6, "phi": 0.9,
             "delta": 8 * E[i]}} for i in range(3)]
    return records, cps
'''


def ledger_rows():
    return [r | {"macs": m} for r, m in zip(rows(), (4e14, 1e15, 6e14, 2e15))]


def run(tmp_path, monkeypatch, *extra):
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    fake = tmp_path / "rescore_v1.py"
    fake.write_text(ANALYSIS)
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(ledger_rows),
                                                                      "root": tmp_path})())
    out = tmp_path / "out"
    assert rescore.main(["--out", str(out), "--pin", "pin", "--manifest", str(man), "--analysis", str(fake),
                         "--scorer", str(tmp_path), *extra]) == 0
    return out, fake


def test_the_analysis_blocks_score_the_rows_they_cover_and_the_rest_are_left_out(tmp_path, monkeypatch):
    out, fake = run(tmp_path, monkeypatch)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == str(fake) and fr["not_analysed"] == ["r3"]
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1", "r2"] and {k["source"] for k in kept} == {"analysis"}
    assert kept[0]["formalism"]["u"] == [list(p) for p in formalism.frozen_tail(8 * 9e7, 0.5)]
    assert kept[0]["formalism"]["labels"]["u"] == rescore.U_TAIL and kept[1]["formalism"]["labels"]["u"] == rescore.U_EVEN
    for n in ("P1", "P2", "P3", "P4", "A2"):
        assert (out / f"{n}.png").exists() and (out / f"{n}.json").exists()
    assert (out / "analysis-gates.json").exists()


def test_an_attached_block_is_what_the_formalism_measures(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    fake = rescore.load_analysis(tmp_path / "rescore_v1.py")
    _, launch, _, _, _ = fake.load(None, None)
    records, cps = fake.build(None, launch, None, None, None)
    blocks = {(launch[c["id"]]["run"], c["macs"]): rescore.block(c, records[c["id"]], fake.TAU) for c in cps}
    got, source = rescore.attach(ledger_rows(), blocks)
    assert source == {"r0": "analysis", "r1": "analysis", "r2": "analysis", "r3": "stubs"}
    m = formalism.measure(got[0], SHAPE)
    assert m.gates == pytest.approx(GATES_PER_MAC * 4e14)
    assert m.delta_bits == m.u1_bits == 8 * 9e7 and m.tau_bits == 16
    assert m.labels["u1_bits"].startswith("upper bound") and m.labels["a_bits"].startswith("upper bound")


def test_stubs_score_every_valid_row(tmp_path, monkeypatch):
    out, _ = run(tmp_path, monkeypatch, "--stubs")
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == "stubs: --stubs" and fr["not_analysed"] == []
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1", "r2", "r3"] and {k["source"] for k in kept} == {"stubs"}
    assert not (out / "analysis-gates.json").exists()
