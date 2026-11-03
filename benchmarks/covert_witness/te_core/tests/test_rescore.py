"""te_core.rescore scores the ledger rows whose formalism block was measured on a traced step, by the formalism v2 (a
frozen tail always charged), beside an analysis's blocks for the same checkpoints, and cuts the console's panels; with
--quantities analysis it scores the analysis's blocks, joined by run and MACs, and leaves out the valid rows the
analysis has no block for; with --stubs every valid row is scored on te_core.formalism's stubs."""
import json

import pytest

pytest.importorskip("matplotlib")

from te_core import console, formalism, plots, policy, rescore
from test_plots import SHAPE, rows

GATES_PER_MAC = 7000.0
ANALYSIS = f'''
TAU = 16
MACS = {[4e14, 1e15, 6e14]!r}
E = {[9e7, 9e7, 8e4]!r}
T = [0.5, 0.0, 0.0]
LATENTS = [3e9, None, None]

def gate_factors(cache):
    cache.write_text("{{}}")
    return {{}}

def load(scorer, ledger):
    return {{}}, {{f"s{{i}}": {{"run": f"run{{i}}"}} for i in range(3)}}, {{}}, {{}}

def profiles(u1, t):
    if 0 < t < 1:
        return [[0.0, 0.0], [t, 0.0], [1.0, u1]], [[0.0, 0.0], [0.0, u1], [1.0, u1]]
    return [[0.0, 0.0], [1.0, u1]], [[0.0, 0.0], [1.0, u1]]

def build(subs, launch, starts, counted, g):
    records = {{f"s{{i}}": {{"delta_grad": 1e9, "a": 512 * 16, "t": T[i], "delta_t": LATENTS[i],
                            "family": "full" if i < 2 else "adapter"}} for i in range(3)}}
    cps = []
    for i in range(3):
        G = {GATES_PER_MAC} * MACS[i]
        u, u_in = profiles(8 * E[i], T[i])
        cps.append({{"id": f"s{{i}}", "macs": MACS[i], "gates": G, "token_gates": 0.9 * G, "affected_gates": 0.8 * G,
                     "sigma": 0.9, "tokens": 1e6, "delta": 8 * E[i], "u": u, "u_in": u_in}})
    return records, cps
'''


TRACE = {"tau_ids": 1.125, "tau_other_bits": 0.0, "a_bits": 512 * 16.0, "u1_bits": 4e8, "tail_share": 0.3,
         "u_in": [[0.0, 0.0], [0.0, 1e8], [0.5, 4e8], [1.0, 4e8]], "frozen_tail_share": 0.1, "update_cut_bits": 4e8,
         "affected_share": 0.85, "data_dependent": [], "carried_scalars": []}
CLIP = {"kind": "a comparison of a read-back float", "where": "train.py:399", "steps": [0, 4], "count": 5}


def ledger_rows():
    """r0 to r3 at their MACs; r0 and r1 measured on a traced step (σ 0.92, affected share 0.85, the step's tokens,
    Δ_grad 2e9)."""
    out = [r | {"macs": m} for r, m in zip(rows(), (4e14, 1e15, 6e14, 2e15))]
    for r in out[:2]:
        f = formalism.of_trace(TRACE)
        r["te_core"] = "4.0"
        g = GATES_PER_MAC * r["macs"]
        r["formalism"] = f | {"parts": [{"gates": g, "tokens": 1e6, "steps": 10, "token_gates": 0.92 * g,
                                         "affected_gates": 0.85 * g, "delta_grad_bits": 2e9}]}
    return out


def run(tmp_path, monkeypatch, *extra, index=ledger_rows):
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    fake = tmp_path / "rescore_v1.py"
    fake.write_text(ANALYSIS)
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(index),
                                                                      "root": tmp_path})())
    out = tmp_path / "out"
    assert rescore.main(["--out", str(out), "--pin", "pin", "--manifest", str(man), "--analysis", str(fake),
                         "--scorer", str(tmp_path), *(extra or ("--quantities", "analysis"))]) == 0
    return out, fake


def test_the_analysis_blocks_score_the_rows_they_cover_and_the_rest_are_left_out(tmp_path, monkeypatch):
    out, fake = run(tmp_path, monkeypatch)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == str(fake) and fr["not_analysed"] == ["r3"] and "readings" not in fr
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1", "r2"] and {k["source"] for k in kept} == {"analysis"}
    assert [k["family"] for k in kept] == ["full", "full", "adapter"]
    assert {fr["te_core_version"], *(k["te_core_version"] for k in kept)} == {rescore.VERSION}
    f0, f1 = kept[0]["formalism"], kept[1]["formalism"]
    u, u_in = formalism.tail(8 * 9e7, 0.5)
    assert (f0["u"], f0["u_in"], f0["tail_share"]) == ([list(p) for p in u], [list(p) for p in u_in], 0.5)
    assert f0["labels"]["u"] == rescore.TAIL["u"] and f0["labels"]["u_in"] == rescore.TAIL["u_in"]
    assert f1["tail_share"] == 0.0 and f1["labels"]["u"] == rescore.EVEN["u"]
    for n in plots.NAMES:
        assert (out / f"{n}.png").exists() and (out / f"{n}.json").exists()
        c = json.loads((out / "console" / f"{n}.json").read_text())
        assert len(console.body(c)) <= console.BODY_MAX and len(c["note"]) <= console.NOTE_MAX
    md = (out / "frontier.md").read_text()
    assert "x0 bound" in md and "s* known" in md and "literal-x0" not in md and "tail-charged" not in md
    assert (out / "analysis-gates.json").exists() and not (out / "console" / "put.json").exists()


def test_a_frozen_tail_is_charged_through_a(tmp_path, monkeypatch):
    """r0's block has a frozen tail (t = 0.5): u_in jumps, so x0 = 0 and ρ = σ s a / room(u) at the reference policy."""
    out, _ = run(tmp_path, monkeypatch)
    c, k = policy.cells(SHAPE)["reference"]
    k0 = json.loads((out / "formalism.jsonl").read_text().splitlines()[0])
    m = formalism.measure({"formalism": k0["formalism"]}, SHAPE)
    r = formalism.at(m, c, k)
    assert r["x0"] == 0 and r["s_star"] == pytest.approx(formalism.room(m.u, k) / (512 * 16))
    assert r["rho"] == pytest.approx(0.9 * (c * 1e6 / (0.9 * GATES_PER_MAC * 4e14)) * 512 * 16 / formalism.room(m.u, k))
    assert 0 < r["s_star_known"] <= r["s_star"]


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
    assert m.gates == pytest.approx(GATES_PER_MAC * 4e14) and m.tail_share == 0.5 and m.sigma == pytest.approx(0.9)
    assert m.delta_bits == m.u1_bits == 8 * 9e7 and m.tau_bits == 16
    assert m.labels["u1_bits"].startswith("upper bound") and m.labels["a_bits"].startswith("upper bound")
    assert m.labels["u"].startswith("assumption (A1)") and m.labels["u_in"].startswith("assumption (A1')")


def test_the_cut_panels_are_written_and_never_put(tmp_path, monkeypatch):
    monkeypatch.setattr(console, "put", lambda p: pytest.fail(f"PUT {p['id']}"))
    out, _ = run(tmp_path, monkeypatch)
    assert sorted(p.stem for p in (out / "console").glob("*.json")) == sorted(plots.NAMES)
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, "--publish")


def test_stubs_score_every_valid_row(tmp_path, monkeypatch):
    out, _ = run(tmp_path, monkeypatch, "--stubs")
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == "stubs: --stubs" and fr["not_analysed"] == [] == fr["not_measured"]
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1", "r2", "r3"] and {k["source"] for k in kept} == {"stubs"}
    assert not (out / "analysis-gates.json").exists()


def test_charge_context_charges_each_run_its_facts_place(tmp_path, monkeypatch):
    """--charge-context: r0's run changes the embedding, so its K/V are affected from layer 0 and, with a = d_model·16
    at the reference policy's k, ρ is infinite there; r1's run has no changed tensors in the facts, so it keeps v2's ρ,
    labelled. The default scores the same rows uncharged; the option without --facts is refused."""
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"run0": {"changed": ["embed"]}}))
    base, _ = run(tmp_path, monkeypatch, "--facts", str(facts))
    plain = [json.loads(line) for line in (base / "formalism.jsonl").read_text().splitlines()]
    assert all("context" not in k["formalism"] for k in plain)
    out, _ = run(tmp_path, monkeypatch, "--facts", str(facts), "--charge-context")
    assert json.loads((out / "frontier.json").read_text())["quantities"] == "measured (trace; t literal); context charged"
    k0, k1 = (json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines())
    assert k0["formalism"]["context"] == formalism.context(["embed"], SHAPE)
    assert "context" not in k1["formalism"] and k1["formalism"]["labels"]["context"] == rescore.UNCHARGED
    c, k = policy.cells(SHAPE)["reference"]
    m0, m1 = formalism.measure(k0, SHAPE), formalism.measure(k1, SHAPE)
    assert m0.labels["context"] == formalism.CONTEXT and formalism.at(m0, c, k)["rho"] == float("inf")
    assert formalism.at(m1, c, k)["rho"] == formalism.at(formalism.measure(plain[1], SHAPE), c, k)["rho"]
    with pytest.raises(SystemExit, match="--facts"):
        run(tmp_path, monkeypatch, "--charge-context")


@pytest.mark.parametrize("quantities", ["measured", "analysis"])
def test_a_straight_through_runs_latents_are_its_delta_t(tmp_path, monkeypatch, quantities):
    """r0's run trains its code through its rounding with 3e9 bits of fp32 latents (the analysis reads them from its
    record): its block carries them as Δ_t, measured or analysed, labelled, and the update application reads them;
    r1's run has none, so its Δ_t is Δ (the coordinator's 00:47Z ruling)."""
    out, _ = run(tmp_path, monkeypatch, "--quantities", quantities)
    k0, k1 = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()][:2]
    assert k0["formalism"]["delta_t_bits"] == 3e9 and k0["formalism"]["labels"]["delta_t_bits"] == rescore.LATENTS
    assert k1["formalism"].get("delta_t_bits") is None
    # formalism.jsonl's lines carry no output; d 0 (E stores θ') leaves Fit to the update application alone
    m0, m1 = (formalism.measure(k | {"output": {"rebuild_macs": 0}}, SHAPE) for k in (k0, k1))
    assert m0.delta_t == 3e9 != m0.delta_bits and m1.delta_t == m1.delta_bits
    c, _ = policy.cells(SHAPE)["reference"]
    k = 3e9 + m0.delta_grad_bits
    assert formalism.at(m0, c, k)["parts"][0]["fit"]["update"]
    assert not formalism.at(m0, c, k * (1 - 1e-9))["parts"][0]["fit"]["update"]


@pytest.mark.parametrize("tail", ["literal", "frozen"])
def test_measured_rows_are_scored_beside_the_analysis(tmp_path, monkeypatch, tail):
    """The default scores r0 and r1, the rows measured on a traced step, and leaves out r2 and r3; τ is the traced
    positions at ⌈log2 V⌉ bits; u is 0 up to t, then u_in; compare.json puts each beside the analysis's block for its
    checkpoint. --tail frozen reads t as the trace's contiguous frozen tail and rebuilds u on it."""
    out, _ = run(tmp_path, monkeypatch, "--tail", tail)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["quantities"] == f"measured (trace; t {tail})" and fr["not_measured"] == ["r2", "r3"]
    kept = [json.loads(line) for line in (out / "formalism.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == ["r0", "r1"] and {k["source"] for k in kept} == {"measured"}
    t = 0.3 if tail == "literal" else 0.1
    u_in = tuple(map(tuple, TRACE["u_in"]))
    assert {k["formalism"]["tail_share"] for k in kept} == {t}
    assert all(k["formalism"]["u"] == [list(p) for p in formalism.with_tail(u_in, t)] for k in kept)
    c = json.loads((out / "compare.json").read_text())
    (r0, r1) = c["rows"]
    assert r0["measured"]["tau_bits"] == pytest.approx(1.125 * 16) and r0["analysis"]["tau_bits"] == 16
    assert r0["ratio"]["sigma"] == pytest.approx(0.92 / 0.9) and r0["ratio"]["affected_share"] == pytest.approx(0.85 / 0.8)
    assert r0["ratio"]["a_bits"] == 1 and r0["measured"]["tail_share"] == t
    assert r0["ratio"]["u1_bits"] == pytest.approx(4e8 / (8 * 9e7))
    assert r0["update_cut_over_delta"] == pytest.approx(4e8 / (8 * 9e7))
    assert {"rho", "fit", "x0", "s_star"} <= set(r0["cell"]) and r0["cell_analysis"]["x0"] == 0
    f = r0["factors"]
    assert f["sigma"] * f["reach"] * f["s"] * f["a"] * f["alpha"] / f["room"] == pytest.approx(
        r0["cell"]["slowdown"] / r0["cell_analysis"]["slowdown"])
    assert {(e["start"], e["drop"]) for e in c["S"]} == {("s1", d) for d in (0.005, 0.01, 0.02, 0.05)}
    routed = [e["routes"]["measured"] for e in c["S"] if e["measured"]["id"]]
    assert routed and all(w["id"] in {"r0", "r1"} and w["factors"] for w in routed)
    assert "| s1 |" in (out / "compare.md").read_text()


def test_a_refused_row_is_listed_with_its_reason_and_never_scored(tmp_path, monkeypatch):
    """seed-mezo's 04:15Z LOOPHOLE: r1's capture lists a clip on a read-back loss, so its measured quantities are
    refused: frontier.json lists it under `refused` with the reason, formalism.jsonl carries it, and no S, panel or
    comparison reads it."""
    def index():
        out = ledger_rows()
        out[1]["formalism"]["trace"] = out[1]["formalism"]["trace"] | {"data_dependent": [CLIP]}
        return out

    out, _ = run(tmp_path, monkeypatch, "--tail", "literal", index=index)
    fr = json.loads((out / "frontier.json").read_text())
    assert fr["refused"] == {"r1": formalism.refusal(index()[1])} and fr["refused"]["r1"].startswith("data-dependent")
    assert fr["skipped"]["r1"].startswith("refused: data-dependent: not measured") and fr["scored"] == 1
    kept = {k["id"]: k for k in map(json.loads, (out / "formalism.jsonl").read_text().splitlines())}
    assert kept["r1"]["refused"] == fr["refused"]["r1"] and kept["r0"]["refused"] is None
    routes = {e["of"]["id"] for s in fr["starts"].values() for t in s["targets"] for e in t["cells"].values() if e["of"]}
    assert routes <= {"r0"} and [r["id"] for r in json.loads((out / "compare.json").read_text())["rows"]] == ["r0"]
