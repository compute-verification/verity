"""finals_table.py scores each finals point's reruns on the sealed final set: per seed the target from the start's data
fit with l0 the seed's own final-set start_bpb, the first checkpoint reaching it with its |C| (te_core.formalism's, as
the search rows'); per point the seeds reached and C's spread; H_any the least mean C over every route and the
headline p_any = C/H_any (the honest point's H and p beside it), the search split's p where the search's H has no
finished finals seed; ρ the search run's from the scorer's stored table and S = p_any·ρ; and every point the table
doesn't show, or seed it lacks, says why."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from te_core import frontier, frontier_fit

_SPEC = importlib.util.spec_from_file_location("finals_table", Path(__file__).resolve().parents[1] / "finals_table.py")
FT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FT)

SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "tokens_per_param": 3.42, "forward_macs_per_token": 51_342_080}
DATA = frontier_fit.data_of_shape(SHAPE)
#: the start's data fit; its targets from l0 = 1.2 are 1.1915 at 1.1x, 1.1437 at 2x and 1.0462 at 11x
LAW = {"E": 0.9, "A": 1.0, "gamma": 0.3}
#: the start's final-set bpb, which every rerun's record carries; the search pool's (the scorer's start_loss) differs
L0, POOL_L0 = 1.2, 1.25
OTHER = "f" * 40
#: (point's search run, island, planned seeds, label, rerun, its checkpoints' final-set bpb by step, measured at)
POINTS = [("r-honest", "ref-optim", ["own", "101"], None, "f-a", {4: 1.18, 9: 1.12}, None),
          ("r-covert", "lowrank", ["own", {"seed": "101", "dropped": "late"}], FT.SINGLE, "f-b", {1: 1.19, 9: 1.16},
           OTHER)]
#: the scorer's search rows: (record, gain target, reached, rho_own, rho_default), C_gates 1e8 per step
SEARCH = [("r-honest@40", 1.1, "yes", 1.5, 2.0), ("r-honest@90", 2, "yes", 1.5, 2.0), ("r-honest@90", 11, "no", 1.5, 2.0),
          ("r-covert@10", 1.1, "yes", 3.0, 4.0), ("r-covert@90", 2, "no", 3.0, 4.0), ("r-covert@90", 11, "no", 3.0, 4.0)]
#: an honest finals point cheaper in search than r-honest at 1.1x (dearer than r-covert), its rerun f-c
CHEAP = ("r-cheap@20", 1.1, "yes", 1.0, 1.0)
#: a covert search point cheaper than r-covert at 1.1x, not a finals point
CHEAPEST = ("r-other@3", 1.1, "yes", 2.0, 2.0)
DIRECTION = {"r-honest": "ref-optim", "r-covert": "lowrank", "r-cheap": "ref-data", "r-other": "lowrank"}
#: the scorer's label for the start s1
START = "45M · 3.42 tok/p"
STILL_RUNNING = "no run record in the store (still running or waiting, or not uploaded yet)"


def record(bpb: dict[int, float], commit: str) -> dict:
    """A harness record of a toy final-mode run from start s1: ten equal steps of bf16 GEMMs, so a checkpoint's |C| is
    its steps' share of the run's, and a scored E at each checkpoint."""
    steps = [{"step": t, "tokens": 8, "macs": 9600, "by": {"gemm/bf16": 9600}, "struct": "s", "weights": {}, "shapes": {}}
             for t in range(10)]
    struct = {"updated": [[100, i * 100, 800, 8, 16] for i in range(4)], "log": [], "beta": 2.0, "cut_bits": 800}
    return {"setup": {"macs": 0, "tokens": 0, "by": {}}, "tail": {"macs": 0, "tokens": 0, "by": {}}, "steps": steps,
            "structures": {"s": struct}, "tier": 0, "seed": 1, "theta_sha256": "t", "c": 10,
            "totals": {"cpu_s": 1, "wall_s": 1, "gpu_s": 1}, "start": {"name": "s1", "digest": "d", "val_bpb": L0},
            "evaluation": {"measurement_commit": commit},
            "checkpoints": [{"step": s, "val_bpb": b, "e_bytes": 30, "source": "submit"} for s, b in bpb.items()]}


def fixture(tmp_path: Path, rho_own: bool = True, cheap: str | None = None, other: bool = False) -> list[str]:
    """FINALS.json, the finals job's launch file, the reruns' records, the scorer's tables and a starts manifest; and
    `build`'s arguments over them. The covert point's first launch was superseded by its step check. With `cheap`, the
    `CHEAP` point too, its rerun f-c "running" (launched, no record yet), "unlaunched", "dropped" in FINALS.json, or
    "short" (finished, never reaching 1.1x on the final set); with `other`, the search row `CHEAPEST`."""
    pin = frontier.pinned()[0]
    finals = {"points": [{"point": f"{run} point", "run": run, "start": "s1", "island": island,
                          "mechanism": f"{island}/m",
                          "reruns": [s if isinstance(s, dict) else {"seed": s} for s in seeds]}
                         | ({"label": label} if label else {}) for run, island, seeds, label, *_ in POINTS]}
    if cheap:
        finals["points"].append({"point": "r-cheap point", "run": "r-cheap", "start": "s1", "island": "ref-data",
                                 "mechanism": "ref-data/m",
                                 "reruns": [{"seed": "own"} | ({"dropped": "late"} if cheap == "dropped" else {})]})
    (tmp_path / "FINALS.json").write_text(json.dumps(finals))
    launches = [{"key": "r-honest seed own", "node": "n1", "run": "f-a", "rc": 0},
                {"key": "r-covert seed own", "node": "n1", "run": "f-b-old", "rc": 0},
                {"key": "r-covert seed own", "run": "f-b-old", "superseded": True, "steps": 3},
                {"key": "r-covert seed own", "node": "n1", "run": "f-b", "rc": 0}]
    launches += [{"key": "r-cheap seed own", "node": "n1", "run": "f-c", "rc": 0}] if cheap in ("running", "short") else []
    (tmp_path / "launches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in launches))
    recs = [(rerun, bpb, commit) for *_, rerun, bpb, commit in POINTS]
    for rerun, bpb, commit in recs + ([("f-c", {9: 1.195}, None)] if cheap == "short" else []):
        out = tmp_path / "records" / rerun / "out"
        (out / "run").mkdir(parents=True)
        (out / "run" / "run.json").write_text(json.dumps(record(bpb, commit or pin)))
        (out / "meta.json").write_text(json.dumps({"schema": "covert-witness/submission/v1", "start": "s1",
                                                   "theta_digest": "d"}))
        (out / "witness.json").write_text(json.dumps({"curve": [{"step": s, "train_flops": 1e9 * (s + 1), "bpb": b}
                                                                for s, b in bpb.items()]}))
    (tmp_path / "records" / "fetch.json").write_text(json.dumps({"f-c": STILL_RUNNING} if cheap == "running" else {}))
    search = SEARCH + ([CHEAP] if cheap else []) + ([CHEAPEST] if other else [])
    gates = {rec: 1e8 * int(rec.rpartition("@")[2]) for rec, *_ in search}
    h = {g: min((gates[rec] for rec, gg, hit, *_ in search if gg == g and hit == "yes"), default=None)
         for g in FT.GAINS}
    cols = ["start", "data", "record", "method_key", "gain_target", "reached", "C_gates", "H_gates", "p", "start_loss",
            "rho_default", "basis_default", "te_core"]
    cols += ["rho_own", "basis_own"] if rho_own else []
    rows = [[START, DATA, rec, f"{DIRECTION[rec.partition('@')[0]]}/m", g, hit, gates[rec], h[g],
             h[g] and gates[rec] / h[g], POOL_L0, default, "A", "4.13.1"] + ([own, "A"] if rho_own else [])
            for rec, g, hit, own, default in search]
    pol = tmp_path / "tables"
    pol.mkdir()
    (pol / "algorithm-policy.json").write_text(json.dumps({"columns": cols, "rows": rows, "source": "test",
                                                           "updated_at": "T"}))
    fcols = ["row", "data", "E_bpb", "A", "gamma", "points", "status"]
    (pol / "frontier-fit.json").write_text(json.dumps({"columns": fcols, "rows": [
        ["fit", DATA, LAW["E"], LAW["A"], LAW["gamma"], 30, "fit"], ["point", DATA, None, None, None, None, None]]}))
    (tmp_path / "starts.json").write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    return ["build", "--records", str(tmp_path / "records"), "--policy", str(pol), "--launches",
            str(tmp_path / "launches.jsonl"), "--finals", str(tmp_path / "FINALS.json"), "--manifest",
            str(tmp_path / "starts.json"), "--policy-art", "art:test", "--out", str(tmp_path / "out")]


def table(tmp_path: Path, rho_own: bool = True, cheap: str | None = None, other: bool = False) -> tuple[dict, str]:
    tmp_path.mkdir(exist_ok=True)
    assert FT.main(fixture(tmp_path, rho_own, cheap, other)) == 0
    return (json.loads((tmp_path / "out" / "finals-table.json").read_text()),
            (tmp_path / "out" / "finals-table.md").read_text())


def test_two_reruns_give_p_against_the_best_route_with_the_search_runs_rho(tmp_path):
    """At 1.1x the honest rerun reaches at step 4 (5 of 10 steps' |C|) and the covert one at step 1 (2 of 10): H_any
    is the covert point's, the headline p_any 1 and 2.5, H_honest the honest point's, p_honest 0.4 and 1; S =
    p_any·rho_own, the search run's at its first reaching checkpoint. Each H of the search split has a finished
    seed, so p reads the final set. The targets are the fit's from the records' final-set l0, not the search pool's."""
    t, md = table(tmp_path)
    by = {(tb["start"], tb["gain"]): tb for tb in t["tables"]}
    assert set(by) == {("s1", g) for g in FT.GAINS}
    one = by[("s1", 1.1)]
    assert one["H_any"]["run"] == "r-covert" and one["H_honest"]["run"] == "r-honest" and one["H_honest_why"] is None
    assert one["H_any_search"]["record"] == "r-covert@10" and one["H_honest_search"]["record"] == "r-honest@40"
    assert one["p_any_basis"] == one["p_honest_basis"] == FT.FINAL_SET and one["search_start"] == START
    rows = {r["run"]: r for r in one["rows"]}
    assert [r["run"] for r in one["rows"]] == ["r-covert", "r-honest"]  # by the headline p_any
    assert rows["r-covert"]["p_any"] == 1 and rows["r-honest"]["p_any"] == pytest.approx(2.5)
    assert rows["r-honest"]["p_honest"] == 1 and rows["r-covert"]["p_honest"] == pytest.approx(0.4)
    assert rows["r-honest"]["p_any_search"] == pytest.approx(4) and rows["r-covert"]["p_honest_search"] == 0.25
    assert rows["r-covert"]["S"] == pytest.approx(1 * 3.0) and rows["r-honest"]["S"] == pytest.approx(2.5 * 1.5)
    assert "p" not in rows["r-covert"] and "H" not in one  # no column keeps the old honest-only meaning
    assert rows["r-covert"]["rho_record"] == "r-covert@10" and rows["r-covert"]["S_reads"] == "rho_own"
    assert rows["r-covert"]["rho_label"].startswith(FT.RHO_LABEL + ", at its first checkpoint reaching it")
    assert rows["r-honest"]["flops"]["mean"] == 5e9 and rows["r-honest"]["C"]["n"] == 1
    assert rows["r-honest"]["bpb"]["mean"] == 1.18
    assert rows["r-honest"]["below_target"]["mean"] == pytest.approx(frontier_fit.target(LAW, L0, 1.1) - 1.18)
    assert one["H_honest"]["C"]["mean"] == pytest.approx(2.5 * one["H_any"]["C"]["mean"])
    pts = {p["run"]: p for p in t["points"]}
    seed = pts["r-honest"]["at"]["1.1"]["seeds"]["own"]
    assert seed["l0"] == L0 and seed["target"] == pytest.approx(frontier_fit.target(LAW, L0, 1.1))
    assert (seed["reached"], seed["step"]) == (True, 4)
    assert pts["r-covert"]["seeds"]["own"]["rerun"] == "f-b"  # the superseded launch is not the rerun
    assert t["inputs"]["policy"].startswith("art:test") and t["te_core"] == FT.VERSION
    assert "## s1 at 1.1x" in md and FT.RHO_LABEL in md and "  - H honest on the final set: r-honest point" in md
    assert "- p_any, p relative to the best route found, reads: final set" in md
    assert all(tb["H_caveat"] is None for tb in t["tables"]) and "CAVEAT" not in md


def test_where_the_search_splits_h_has_no_finished_finals_seed_p_is_the_search_splits(tmp_path):
    """At 1.1x the search's honest H is r-cheap (C 2e9, below r-honest's 4e9), whose rerun f-c hasn't finished: the
    honest p is the search split's, C_gates over r-cheap's, so labelled, never a final-set p against r-honest, which
    goes beside it; p_any still reads the final set, its search H r-covert having a finished seed. With a cheaper
    covert search point outside the finals, p_any too is the search split's, the scorer's stored p, and S reads it.
    An H that isn't launched or is dropped says so; one that finished without reaching leaves p on the final set,
    with a caveat."""
    t, md = table(tmp_path, cheap="running")
    by = {tb["gain"]: tb for tb in t["tables"]}
    one, rows = by[1.1], {r["run"]: r for r in by[1.1]["rows"]}
    running = "p on the search split: H's reruns f-c not finished"
    assert one["H_honest_search"] | {"seeds": None} == {
        "record": "r-cheap@20", "C_gates": 2e9, "point": "r-cheap point", "seeds": None, "finished": 0, "planned": 1}
    assert one["p_honest_basis"] == running and rows["r-covert"]["p_honest_basis"] == running
    assert rows["r-covert"]["p_honest"] == 0.5 and rows["r-covert"]["p_honest_final"] == pytest.approx(0.4)
    assert rows["r-honest"]["p_honest"] == 2 and rows["r-honest"]["p_honest_final"] == 1
    assert rows["r-honest"]["p_honest_range"] is None and rows["r-honest"]["p_honest_final_range"] == [1, 1]
    assert rows["r-cheap"]["p_honest"] == 1 and rows["r-cheap"]["C"] is None  # shown: its search run reached
    assert one["p_any_basis"] == FT.FINAL_SET and rows["r-covert"]["p_any"] == 1
    assert rows["r-covert"]["S"] == pytest.approx(1 * 3.0)
    assert rows["r-cheap"]["p_any"] is None and rows["r-cheap"]["p_any_search"] == 2
    assert by[2]["p_honest_basis"] == FT.FINAL_SET  # r-honest is the search's honest H there, and it finished
    assert f"- p_honest, p against the honest reference, reads: {running}" in md
    assert "0.5 (search split) (final set 0.4; search split 0.5)" in md

    t, md = table(tmp_path / "other", cheap="running", other=True)
    one = t["tables"][0]
    rows = {r["run"]: r for r in one["rows"]}
    assert one["p_any_basis"] == "p on the search split: H (r-other@3) is not a finals point"
    assert one["H_any_search"] == {"record": "r-other@3", "C_gates": 3e8, "point": None}
    assert rows["r-covert"]["p_any"] == pytest.approx(1e9 / 3e8) and rows["r-covert"]["p_any_final"] == 1
    assert rows["r-covert"]["S"] == pytest.approx(1e9 / 3e8 * 3.0) and rows["r-honest"]["p_any_final"] == 2.5
    assert [r["run"] for r in one["rows"]] == ["r-covert", "r-cheap", "r-honest"]  # by the search split's p_any
    assert rows["r-covert"]["p_honest_basis"] == running and "not a finals point" in md

    for mode, basis in (("unlaunched", "p on the search split: H's reruns not launched (seeds own planned)"),
                        ("dropped", "p on the search split: H (r-cheap point, r-cheap@20) not rerun in the finals")):
        t, _ = table(tmp_path / mode, cheap=mode)
        assert t["tables"][0]["p_honest_basis"] == basis
        assert {r["run"]: r for r in t["tables"][0]["rows"]}["r-covert"]["p_honest"] == 0.5

    t, md = table(tmp_path / "short", cheap="short")
    one = t["tables"][0]
    assert one["p_honest_basis"] == FT.FINAL_SET and one["H_honest"]["run"] == "r-honest"
    assert {r["run"]: r for r in one["rows"]}["r-covert"]["p_honest"] == pytest.approx(0.4)
    assert one["H_caveat"] == ("the search split's H honest at 1.1x, r-cheap point (r-cheap@20, C 2e+09 gates), has 1 "
                               "of 1 seeds finished and none reaching it on the final set")
    assert "- CAVEAT: the search split's H honest at 1.1x" in md and t["tables"][1]["H_caveat"] is None


def test_every_point_the_table_leaves_out_or_seed_it_lacks_says_why(tmp_path):
    """At 2x only the honest point reached (the covert seed and its search run didn't), and at 11x no point did, so
    no H, each counted with why; a planned seed never launched, a single-seed point and a record measured outside the
    pin's version are labelled, never silently absent."""
    t, md = table(tmp_path)
    by = {(tb["start"], tb["gain"]): tb for tb in t["tables"]}
    two = by[("s1", 2)]
    assert [r["run"] for r in two["rows"]] == ["r-honest"] and two["rows"][0]["p_any"] == 1
    assert two["not_shown"] == {"neither the search run nor any of its 1 finished seeds reached 2x": 1}
    eleven = by[("s1", 11)]
    assert eleven["rows"] == [] and eleven["H_honest"] is None and eleven["H_any"] is None
    assert eleven["H_any_search"] is None and eleven["p_any_basis"] == FT.FINAL_SET
    assert "  - H on the final set: no point has a finished seed reaching it" in md
    assert f"  - on the search split: none: no reached row at {START} in the scorer's table" in md
    assert sum(eleven["not_shown"].values()) == 2
    pts = {p["run"]: p for p in t["points"]}
    assert pts["r-covert"]["at"]["2"]["seeds"]["own"]["reached"] is False
    assert pts["r-covert"]["at"]["2"]["seeds"]["own"]["step"] == 9  # not reached: at its last checkpoint
    assert "1 of 2 seeds finished: seed 101 not launched" in pts["r-honest"]["label"]
    assert pts["r-covert"]["label"][0] == FT.SINGLE
    assert pts["r-covert"]["at"]["2"]["planned"] == 1 and "101" not in pts["r-covert"]["seeds"]  # dropped, unlaunched
    assert f"seed own measured at {OTHER[:12]}, outside the pin's version" in pts["r-covert"]["label"]
    assert pts["r-honest"]["seeds"]["101"] == {"rerun": None, "state": "not launched"}
    assert "Not shown: 1 point(s): neither the search run" in md
    assert "## s1 at 11x" in md and "No point shown." in md and "Not shown: 2 point(s)" in md


def test_a_table_without_rho_own_gives_s_on_rho_default_so_labelled(tmp_path):
    t, md = table(tmp_path, rho_own=False)
    one = next(tb for tb in t["tables"] if tb["gain"] == 1.1)
    r = next(r for r in one["rows"] if r["run"] == "r-covert")
    assert r["S_reads"] == "rho_default" and r["S"] == pytest.approx(1 * 4.0) and r["rho_own"] is None
    assert "has no rho_own, so S reads rho_default" in r["rho_label"]


def test_fetch_takes_each_reruns_record_from_the_store_and_says_why_it_has_none(tmp_path, monkeypatch):
    """A rerun whose record is already fetched is skipped; one with an art fetches its FILES; one the store has no run
    record for says so."""
    (tmp_path / "launches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"key": "a seed own", "node": "n", "run": "have", "rc": 0}, {"key": "a seed 101", "node": "n", "run": "art", "rc": 0},
        {"key": "b seed own", "node": "n", "run": "none", "rc": 0}]))
    (tmp_path / "rec" / "have" / "out" / "run").mkdir(parents=True)
    (tmp_path / "rec" / "have" / FT.FILES[0]).write_text("{}")
    calls = []

    def run(argv, **_):
        calls.append(argv[1:3] + argv[3:4])
        if argv[1:3] == ["data", "fetch"]:
            dest = Path(argv[argv.index("--to") + 1])
            (dest / "out" / "run").mkdir(parents=True)
            (dest / FT.FILES[0]).write_text("{}")
        return type("P", (), {"stdout": "{}", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(FT, "research", lambda: ["research"])
    monkeypatch.setattr(FT.subprocess, "run", run)
    why = FT.fetch(tmp_path / "launches.jsonl", tmp_path / "rec", {"art": "art:x"})
    assert why == {"none": "no run record in the store (still running or waiting, or not uploaded yet)"}
    assert calls == [["data", "fetch", "art:x"], ["data", "show", "none"]]
    assert (tmp_path / "rec" / "art" / FT.FILES[0]).exists()
