"""replot.py: a rotating run's ledger rows are rewritten from its record and every other file linked, panels are
compared cell by cell, and a rebuild with runs traced again moves only those runs' rows, in the leaves expected."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import replot
from te_core import gates, ledger
from te_core.ledger import Ledger, ingest
from te_core.rotations import Unread

T, D = 32768, 512


def row(phase, role, shape, n):
    return [phase, role, "mm", json.dumps(shape), n, shape[0][0] * shape[0][1] * shape[1][1] * n]


def record(args, island=("lowrank-restart", "delta_grad.py")):
    log = [row("fwd", "updated", [[T, D], [D, D]], 48), row("other", "param", [[2048, 64], [64, 64]], 54)]
    by = {}
    for r in log:
        by[f"{r[0]}/{r[1]}"] = by.get(f"{r[0]}/{r[1]}", 0) + r[5]
    return {"island": island[0], "program": island[1], "script_args": list(args), "te_core": "3.20",
            "evaluation": {"island": {"name": island[0], "program": island[1]}},
            "start": {"name": "s100", "digest": "d", "val_bpb": 1.2, "parameters": 45_029_888},
            "setup": {"step": -1, "tokens": 0, "macs": 0, "struct": "0", "by": {}},
            "tail": {"macs": 0, "tokens": 0, "by": {}, "struct": None},
            "steps": [{"step": t, "tokens": T, "macs": sum(by.values()), "by": dict(by), "struct": "s",
                       "weights": {"blocks.5.wq.weight": 1}, "shapes": {}} for t in range(10)],
            "structures": {"s": {"log": log, "updated": [[100, 0, 100 * T, T, 16]]}, "0": {"log": [], "updated": []}},
            "tier": 0, "seed": 1, "theta_sha256": "t", "c": 10, "totals": {"cpu_s": 1, "wall_s": 1, "gpu_s": 1},
            "checkpoints": [{"step": s, "val_bpb": 1.1 - 0.01 * s, "e_bytes": 30, "source": "encode"} for s in (4, 9)]}


def scorer_state(tmp_path, monkeypatch, runs):
    """K/state with each run's record, and a ledger of the rows ingest wrote before it named any rotation."""
    K, index = tmp_path / "K", []
    for run_id, rec in runs.items():
        d = K / "state" / "records" / f"{run_id}-art" / "run"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(rec))
        index.append({"run": run_id, "id": f"art:{run_id}-art", "valid": True})
    (K / "state" / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index))
    new = {}
    for r in runs:
        try:
            new[r] = ingest(K / "state" / "records" / f"{r}-art" / "run", Ledger(tmp_path / "new", "a"), direction="d",
                            mechanism="m", run_id=r)
        except Unread:
            new[r] = None
    monkeypatch.setattr(ledger, "labelled", lambda r: r)
    for r in runs:
        ingest(K / "state" / "records" / f"{r}-art" / "run", Ledger(tmp_path / "measured", "a"), direction="d",
               mechanism="m", run_id=r)
    monkeypatch.undo()
    (tmp_path / "measured" / "records" / "a" / "a-000000.art").write_text("art:x\n")
    return K, new


def test_a_rotating_runs_rows_are_rewritten_and_every_other_file_is_linked(tmp_path, monkeypatch):
    runs = {"r-rot": record(["--rotate", "--group", "64"]), "r-plain": record(["--group", "64"]),
            "r-odd": record(["--rotate"], ("shallow", "train.py")), "r-none": record(["--rotate", "--group", "32"])}
    K, new = scorer_state(tmp_path, monkeypatch, runs)
    measured, priced = tmp_path / "measured", tmp_path / "priced"
    before = {p.name: p.read_text() for p in (measured / "records" / "a").iterdir()}
    out = replot.reprice(measured, priced, K)
    assert {p.name: p.read_text() for p in (measured / "records" / "a").iterdir()} == before, "the source is untouched"
    assert sorted(out["runs"]) == ["r-rot"] and out["counts"] == {"rows rewritten": 2, "rows linked": 6, "other files": 1}
    assert sorted(out["rewritten"]) == sorted(f"records/a/{w['id']}.json" for w in new["r-rot"])
    assert list(out["unknown_program"]) == ["r-odd"] and list(out["unread"]) == ["r-none"]
    assert out["no_record"] == [] and out["no_arguments"] == []
    got = {json.loads(p.read_text())["id"]: p for p in (priced / "records" / "a").glob("*.json")}
    price = gates.per_mac("gemm/tf32") - gates.per_mac("gemm/bf16")
    for want in new["r-rot"]:
        p = got[want["id"]]
        now = json.loads(p.read_text())
        assert now["macs_log"] == json.loads(json.dumps(want["macs_log"]))
        assert now["formalism"]["parts"] == json.loads(json.dumps(want["formalism"]["parts"]))
        assert p.stat().st_ino != (measured / "records" / "a" / p.name).stat().st_ino
        cp = out["runs"]["r-rot"]["checkpoints"][str(want["checkpoint_step"])]
        steps = want["checkpoint_step"] + 1
        assert cp["rotation_macs"] == steps * 54 * 2048 * 64 * 64
        assert cp["C_after"] - cp["C_before"] == pytest.approx(cp["rotation_macs"] * price) == pytest.approx(cp["gates_added"])
    assert out["runs"]["r-rot"]["structures"] == {"s": {"other": 54, "steps": 10}}
    assert new["r-odd"] is None and new["r-none"] is None, "ingest refuses what it can't read"
    rewritten = {w["id"] for w in new["r-rot"]}
    assert len(got) == 8 and all(p.stat().st_ino == (measured / "records" / "a" / p.name).stat().st_ino
                                 for i, p in got.items() if i not in rewritten)
    art = priced / "records" / "a" / "a-000000.art"
    assert art.stat().st_ino == (measured / "records" / "a" / "a-000000.art").stat().st_ino


def test_a_run_whose_rows_say_no_arguments_is_read_from_its_record(tmp_path, monkeypatch):
    K, _ = scorer_state(tmp_path, monkeypatch, {"r-rot": record(["--rotate", "--group", "64"])})
    for p in (tmp_path / "measured" / "records" / "a").glob("*.json"):
        r = json.loads(p.read_text())
        del r["args"]
        p.chmod(0o644)
        p.write_text(json.dumps(r))
    out = replot.reprice(tmp_path / "measured", tmp_path / "priced", K)
    assert sorted(out["runs"]) == ["r-rot"] and out["counts"]["rows rewritten"] == 2
    (K / "state" / "index.jsonl").write_text("")
    out = replot.reprice(tmp_path / "measured", tmp_path / "priced-2", K)
    assert out["runs"] == {} and out["no_arguments"] == ["r-rot"] and out["counts"]["rows linked"] == 2


def no_torch(name="torch"):
    def lookups():
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    return lookups


def test_rows_ingested_with_their_rotations_named_are_kept_and_priced_as_the_rewritten_ones(tmp_path, monkeypatch):
    runs = {"r-rot": record(["--rotate", "--group", "64"]) | {"te_core": "3.5"}, "r-plain": record(["--group", "64"])}
    K, new = scorer_state(tmp_path, monkeypatch, runs)
    monkeypatch.setattr(ledger, "lookup_tables", no_torch())
    was = replot.reprice(tmp_path / "measured", tmp_path / "priced", K)
    out = replot.reprice(tmp_path / "new", tmp_path / "priced-new", K)
    assert was["counts"] == {"rows rewritten": 2, "rows linked": 2, "other files": 1} and was["named"] == []
    assert out["counts"] == {"rows named when ingested": 2, "rows linked": 2} and out["rewritten"] == []
    assert sorted(out["named"]) == sorted(f"records/a/{w['id']}.json" for w in new["r-rot"]) and out["unread"] == {}
    kept = {p.name: p for p in (tmp_path / "priced-new" / "records" / "a").glob("*.json")}
    assert all(p.stat().st_ino == (tmp_path / "new" / "records" / "a" / n).stat().st_ino for n, p in kept.items())
    assert replot.priced_rows(out) == replot.priced_rows(was) and sorted(replot.priced_rows(out)) == ["r-rot@4", "r-rot@9"]
    then, now = was["runs"]["r-rot"]["checkpoints"], out["runs"]["r-rot"]["checkpoints"]
    assert {k: (v["C_after"], v["gates_added"]) for k, v in now.items()} == {
        k: (pytest.approx(v["C_after"]), v["gates_added"]) for k, v in then.items()}
    assert all(v["C_before"] is None for v in now.values())


def test_rows_only_the_lookups_could_give_are_kept_where_the_interpreter_holds_no_torch(tmp_path, monkeypatch):
    K, _ = scorer_state(tmp_path, monkeypatch, {"r-rot": record(["--rotate", "--group", "64"]) | {"te_core": "3.5"}})
    for p in (tmp_path / "measured" / "records" / "a").glob("*.json"):
        r = json.loads(p.read_text())
        r["macs_log"]["ops"] = {k + " as another ingest wrote it": v for k, v in r["macs_log"]["ops"].items()}
        p.chmod(0o644)
        p.write_text(json.dumps(r))
    monkeypatch.setattr(ledger, "lookup_tables", no_torch())
    out = replot.reprice(tmp_path / "measured", tmp_path / "priced", K)
    assert out["runs"] == {} and "torch" in out["unread"]["r-rot"]
    assert out["counts"] == {"rows linked": 2, "other files": 1}
    assert replot.priced_rows(out) == {"r-rot": "unread"}
    monkeypatch.setattr(ledger, "lookup_tables", no_torch("prepare"))
    with pytest.raises(ModuleNotFoundError, match="prepare"):
        replot.reprice(tmp_path / "measured", tmp_path / "priced-2", K)


def panel(rows, **more):
    return {"id": "training/algorithm-policy", "updated_at": "t", "source": "s", "note": "n",
            "columns": ["record", "gain_target", "C_gates", "p", "rho_own", "te_core"], "rows": rows} | more


def test_panels_are_compared_cell_by_cell_whatever_the_route_made_of_a_number():
    a = panel([["r1@5", 1.1, 7.07423e+20, 2.0, 100.0, "4.13.3"], ["r2@7", 1.1, 3e20, 1.0, 50.0, "4.13.3"],
               ["r2@7", 2, 3e20, 0.5, 50.0, "4.13.3"], ["r3@1", 1.1, 1e20, 1.5, None, "4.13.3"]])
    b = panel([["r1@5; search split, 1 seed", 1.1, 707423000000000000000, 2, 100, "4.13.5"],
               ["r2@7", 1.1, 3.03e20, 1.01, 49.5, "4.13.5"], ["r2@7", 2, 3.03e20, 0.5, 49.5, "4.13.5"]], updated_at="u")
    out = replot.compare({"before": a, "after": b}, {"r2"})
    assert out["rows"] == 4 and out["only"] == [{"run": "r3", "g": 1.1, "in": ["before"]}]
    assert [(m["run"], m["g"], m["rotating"], sorted(m["cells"])) for m in out["moved"]] == [
        ("r1", 1.1, False, ["record"]), ("r2", 1.1, True, ["C_gates", "p", "rho_own"]),
        ("r2", 2.0, True, ["C_gates", "rho_own"])]
    cell = out["moved"][1]["cells"]["rho_own"]
    assert cell == {"before": 50.0, "after": 49.5, "change": pytest.approx(-0.01)}
    assert out["panels"]["after"]["te_core"] == ["4.13.5"] and out["moved_rotating"] == 2 and out["moved_other"] == 1
    assert replot.same_rows(a, a | {"rows": [[x if not isinstance(x, float) or x < 1e19 else int(x) for x in r]
                                             for r in a["rows"]], "note": "m"}) == {
        "columns": True, "rows": True, "row_counts": [4, 4], "differ": ["note"]}
    with pytest.raises(SystemExit, match="two rows"):
        replot.table(panel([["r1@5", 1.1, 1.0, 1.0, 1.0, "v"], ["r1@6", 1.1, 1.0, 1.0, 1.0, "v"]]))


def test_documents_are_compared_leaf_by_leaf_with_runs_as_keys():
    a = {"built_at": "t", "points": [{"run": "r1", "at": {"1.1": {"C": 1.0}}}, {"run": "r2", "at": {"1.1": {"C": 2.0}}}],
         "tables": [{"rows": [1, 2]}]}
    b = {"built_at": "u", "points": [{"run": "r2", "at": {"1.1": {"C": 2.5}}}, {"run": "r1", "at": {"1.1": {"C": 1}}}],
         "tables": [{"rows": [1, 3]}]}
    assert replot.deep(a, b, skip=("built_at",)) == [
        {"at": "/points/r2/at/1.1/C", "before": 2.0, "after": 2.5, "change": 0.25},
        {"at": "/tables/0/rows/1", "before": 2, "after": 3, "change": 0.5}]


def test_the_builders_script_is_read_from_the_refreshs_source_without_importing_it():
    script = replot.policy_source()
    assert script.rstrip().endswith("sys.exit(algorithm_policy.main(sys.argv[1:]))") and "TE_TRACE_FAILED" in script
    assert "v1cycle" not in sys.modules


def whole(rows):
    cols = ["record", "gain_target", "te_core", "C_gates", "p", "rho_own", "kinds", "rho_inputs"]
    return {"columns": cols, "rows": [[r.get(c) for c in cols] for r in rows]}


def a_row(run, reduction, te="1", rho=3.0, spaced=False):
    kinds = {"bound": {"fresh_products": rho}, "sensitivity": {"reduction": reduction, "update": reduction / 10}}
    return {"record": f"{run}@40", "gain_target": 1.1, "te_core": te, "C_gates": 100.0, "p": 2.0, "rho_own": rho,
            "kinds": json.dumps(kinds, indent=1 if spaced else None),
            "rho_inputs": json.dumps({"delta_grad_bits": reduction * 8, "fresh_per_c": {"65536": [1e-9, reduction]}})}


def rebuilt(root, builds, layers, publish="fixed"):
    """`node`'s OUT for `builds` (name: rows), the published tables being the control's."""
    name = replot.TABLES[0]
    for n, rows in builds.items():
        replot.dump(root / "search" / n / name, whole(rows))
        replot.dump(root / "search" / n / replot.TABLES[4], {"rows": [["other@3", "refused"]]})
        replot.dump(root / ("overlay" if n == publish else f"overlays/{n}") / "tables" / name, whole(rows))
        replot.dump(root / "rotations" / f"{n}.json", {
            "runs": {"rot": {"checkpoints": {"40": {"rotation_macs": 7, "C_before": None if n == "fixed" else 1.0}}}},
            "counts": {"rows named when ingested" if n == "fixed" else "rows rewritten": 1},
            "unread": {}, "no_record": ["gone"], "no_arguments": [], "unknown_program": {}})
    replot.dump(root / "published" / "search" / name, whole(builds["control"]))
    replot.dump(root / "published" / "overlay.json", whole(builds["control"]))
    same = {"rows": True, "columns": True}
    replot.dump(root / "diff" / "frontier-fit.json", {"tables": dict.fromkeys(builds, same), "points": dict.fromkeys(builds, True),
                                                      "refused": dict.fromkeys(builds, True),
                                                      "console_served_against_rebuilt": same})
    replot.dump(root / "diff" / "finals-table.json", {"differ": []})
    replot.dump(root / "diff" / "console.json", {
        "served_rows": 2, "rebuilt_rows": 2, "bytes": 1, "note": "", "served_rows_the_rebuild_drops": [],
        "rows_the_rebuild_adds": [], "rows": replot.compare({"served": whole(builds["control"]),
                                                             "rebuilt": whole(builds[publish])})})
    replot.dump(root / "inputs.json", {"code": "c", "main_head": "m"})
    replot.dump(root / "builds.json", {"publish": publish, "layers": {
        n: {"te_core": n, "commit": "c", "te_tree": "t", "runs": runs, "errors": {}} for n, runs in layers.items()}})


def test_a_traced_runs_row_moves_only_in_the_leaves_expected_and_no_other_row_moves(tmp_path):
    expect = ("kinds/sensitivity/*", "rho_inputs/delta_grad_bits", "rho_inputs/fresh_per_c/*")
    builds = {"control": [a_row("cut", 100.0), a_row("still", 7.0)],
              "old": [a_row("cut", 100.0, "2", spaced=True), a_row("still", 7.0, "2")],
              "fixed": [a_row("cut", 200.0, "2"), a_row("still", 7.0, "2")]}
    rebuilt(tmp_path / "a", builds, {"old": ["cut"], "fixed": ["cut"]})
    v = replot.verdict(tmp_path / "a", ("old",), expect)
    assert v["holds"], [k for k, ok in v["checks"].items() if not ok]
    assert v["checks"]["overlay, old: every cell is the control's"]
    [row] = v["lists"]["search, fixed: rows that move"]
    assert row["run"] == "cut" and row["leaves"]["kinds/sensitivity/reduction"] == {
        "before": 100.0, "after": 200.0, "change": 1.0}
    assert sorted(row["leaves"]) == ["kinds/sensitivity/reduction", "kinds/sensitivity/update",
                                     "rho_inputs/delta_grad_bits", "rho_inputs/fresh_per_c/65536/1"]
    n = v["summary"]["cells"]["search"]["control against fixed"]
    assert (n["rows"], n["cells"], n["moved"], n["unmoved"], n["rows_moved"]) == (2, 14, 2, 12, 1)
    assert v["summary"]["cells"]["search"]["control against old"]["moved"] == 0
    assert v["summary"]["console"]["columns_moved"] == ["kinds", "rho_inputs"]
    # without the leaves named, the same rebuild fails, naming them
    v = replot.verdict(tmp_path / "a", ("old",), expect[:1])
    assert not v["holds"] and not v["checks"]["search, fixed: a row moves only in the leaves expected"]
    [row] = v["lists"]["search, fixed: leaves that move beyond the expected"]
    assert sorted(row["leaves"]) == ["rho_inputs/delta_grad_bits", "rho_inputs/fresh_per_c/65536/1"]
    # a row of a run the layer didn't trace moves, the old rule's build is not the control, and a row is gone
    builds = {"control": builds["control"], "old": [a_row("cut", 101.0, "2"), a_row("still", 7.0, "2")],
              "fixed": [a_row("cut", 200.0, "2", rho=4.0), a_row("still", 7.0, "2", rho=2.0)]}
    rebuilt(tmp_path / "b", builds, {"old": ["cut"], "fixed": ["cut"]})
    v = replot.verdict(tmp_path / "b", ("old",), expect)
    assert not v["holds"] and not v["checks"]["search, old: every cell is the control's"]
    assert not v["checks"]["search, fixed: no row of a run the layer didn't trace moves, and no row comes or goes"]
    [row] = v["lists"]["search, fixed: rows of other runs that move"]
    assert row["run"] == "still" and sorted(row["leaves"]) == ["kinds/bound/fresh_products", "rho_own"]
    [row] = v["lists"]["search, fixed: leaves that move beyond the expected"][:1]
    assert row["run"] == "cut" and sorted(row["leaves"]) == ["kinds/bound/fresh_products", "rho_own"]
    assert v["checks"]["search: the control is the published table in every cell"]
    with pytest.raises(SystemExit, match="--same nope"):
        replot.verdict(tmp_path / "b", ("nope",), expect)
    # a build that prices a rotating row otherwise than the control, or leaves a run apart, is named with the row
    assert v["checks"]["every build prices the same rotating rows"]
    f = tmp_path / "b" / "rotations" / "fixed.json"
    rot = json.loads(f.read_text())
    rot["runs"]["rot"]["checkpoints"]["40"]["rotation_macs"] = 8
    replot.dump(f, rot | {"unread": {"lost": "why"}})
    v = replot.verdict(tmp_path / "b", ("old",), expect)
    assert not v["checks"]["every build prices the same rotating rows"]
    assert v["lists"]["rotating rows a build prices otherwise than the control"] == {
        "old": {}, "fixed": {"rot@40": {"control": 7, "fixed": 8}, "lost": {"control": None, "fixed": "unread"}}}
