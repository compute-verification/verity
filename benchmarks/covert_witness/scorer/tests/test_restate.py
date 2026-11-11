"""restate.py: which steps of a record state their own tokens, which keep the run's factor and which runs are left as
they are; the restated record differs only in its steps' tokens; and the overlay links everything it does not restate."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import restate

ROWS, D, V = 32768, 512, 50304


def head(n, rows=ROWS, width=D):
    return ["fwd", "updated", "mm", json.dumps([[rows, width], [width, V]]), n, rows * width * V * n]


def record(steps, structures):
    """A record of `steps` ([struct, tokens, weights] each) over `structures` ({struct: log})."""
    return {"start": {"vocab": V}, "te_core": "3.5",
            "steps": [{"step": i, "struct": s, "tokens": t, "weights": w} for i, (s, t, w) in enumerate(steps)],
            "structures": {k: {"log": log, "updated": []} for k, log in structures.items()}}


ONE = {"blocks.0.w": ["aa"], "derived/7": ["bb", "cc"]}
TWO = {"blocks.0.w": ["aa", "dd"]}
TRAIN = [("train", ROWS, ONE)] * 6


def after(rec, factor, same_name=False):
    p = restate.plan(rec, factor, same_name)
    restate.restate(rec, p["factor"], p["tokens"])
    return [s["tokens"] for s in rec["steps"]], p


def test_a_training_step_of_two_forwards_states_both():
    tokens, p = after(record(TRAIN, {"train": [head(2)]}), 2.0)
    assert tokens == [2 * ROWS] * 6
    assert p["tokens"] == {} and p["structures"]["train"]["how"] == "the factor: the most common structure"


def test_a_validation_pass_recorded_as_a_step_keeps_its_own_count():
    rec = record(TRAIN + [("val", ROWS, ONE)], {"train": [head(2)], "val": [head(1)]})
    tokens, p = after(rec, 2.0)
    assert tokens == [2 * ROWS] * 6 + [ROWS]
    assert p["tokens"] == {"val": {str(ROWS): ROWS}}
    assert p["structures"]["val"]["how"] == "its head forwards x rows"


def test_an_evaluation_of_more_forwards_than_a_training_step_states_them_all():
    rec = record(TRAIN + [("eval", ROWS, ONE)], {"train": [head(4)], "eval": [head(16)]})
    assert after(rec, 4.0)[0] == [4 * ROWS] * 6 + [16 * ROWS]


def test_a_step_that_validates_the_weights_it_wrote_keeps_the_factor():
    rec = record(TRAIN + [("snap", 2 * ROWS, TWO)], {"train": [head(2)], "snap": [head(6)]})
    tokens, p = after(rec, 2.0)
    assert tokens == [2 * ROWS] * 6 + [4 * ROWS]
    assert p["structures"]["snap"]["how"] == "the factor: a step reads two values of a weight"
    assert p["structures"]["snap"]["steps_reading_two_values"] == 1 and p["tokens"] == {}


def test_two_tensors_under_one_name_are_one_value():
    rec = record([("train", ROWS, TWO)] * 6 + [("eval", 8 * ROWS, TWO)], {"train": [head(2)], "eval": [head(8)]})
    assert "kept" in restate.plan(rec, 2.0)
    assert after(rec, 2.0, same_name=True)[0] == [2 * ROWS] * 6 + [8 * ROWS]


def test_a_structure_with_no_head_forward_keeps_the_factor():
    rec = record(TRAIN + [("other", ROWS, ONE)], {"train": [head(2)], "other": [head(2, width=64)[:2] + ["addmm"] + head(2)[3:]]})
    assert after(rec, 2.0)[0] == [2 * ROWS] * 7


def test_a_run_whose_head_count_is_not_the_traced_count_keeps_the_factor_throughout():
    # an adapter's own product onto the vocabulary counts among the head's
    rec = record(TRAIN + [("val", ROWS, ONE)], {"train": [head(2), head(2, width=2)], "val": [head(1)]})
    tokens, p = after(rec, 2.0)
    assert tokens == [2 * ROWS] * 7 and p["tokens"] == {}
    assert p["structures"]["val"]["how"].endswith("is not the traced count")


@pytest.mark.parametrize("factor, why", [(None, "no trace"), (1.0, "1 times"), (0.25, "0.25 times")])
def test_a_run_with_no_factor_above_one_is_kept(factor, why):
    assert why in restate.plan(record(TRAIN, {"train": [head(2)]}), factor)["kept"]


def test_a_run_whose_steps_read_two_values_is_kept():
    kept = restate.plan(record([("train", ROWS, TWO)] * 6, {"train": [head(16)]}), 8.0)["kept"]
    assert kept.startswith("6 of its most common structure's 6 steps read two values")


def test_a_count_that_is_not_whole_is_refused():
    with pytest.raises(ValueError, match="no whole count"):
        restate.plan(record([("train", 3, ONE)] * 2, {"train": [head(1, rows=4)]}), 4 / 3 + 0.1)


def test_only_the_tokens_differ():
    rec = record(TRAIN + [("val", ROWS, ONE)], {"train": [head(2)], "val": [head(1)]})
    before = json.loads(json.dumps(rec))
    after(rec, 2.0)
    again = json.loads(restate.encode(rec))
    for s in again["steps"] + before["steps"]:
        s.pop("tokens")
    assert again == before


def state(tmp_path, runs):
    """A scorer's state holding `runs` ({run: (record, factor or None)}), its scope and one trace cache."""
    st, scope = tmp_path / "state", []
    (st / "trace40" / "tree" / "records").mkdir(parents=True)
    (st / "trace40" / "other").mkdir()
    (st / "ledger.json").write_text("{}")
    for run, (rec, factor) in runs.items():
        raw = json.dumps(rec, indent=1).encode()
        art = hashlib.sha256(raw).hexdigest()
        (st / "records" / art / "run").mkdir(parents=True)
        (st / "records" / art / "run" / "run.json").write_bytes(raw)
        (st / "records" / art / "run" / "E.bin").write_bytes(b"e")
        (st / "records" / art / "witness.json").write_text("{}")
        scope.append({"run_id": run, "record": art})
        if factor is not None:
            with gzip.open(st / "trace40" / "tree" / "records" / f"{run}.jsonl.gz", "wt") as fh:
                fh.write(json.dumps({"formalism": {"trace": {"token_factor": factor}}}) + "\n")
    (tmp_path / "scope.json").write_text(json.dumps(scope))
    return st, {r["run_id"]: r["record"] for r in scope}


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["restate.py", *map(str, argv)])
    return restate.main()


def test_plan_and_overlay(tmp_path, monkeypatch):
    runs = {"r-two": (record(TRAIN + [("val", ROWS, ONE)], {"train": [head(2)], "val": [head(1)]}), 2.0),
            "r-one": (record(TRAIN, {"train": [head(1)]}), 1.0),
            "r-none": (record(TRAIN, {"train": [head(2)]}), None),
            "r-mezo": (record([("train", ROWS, TWO)] * 6, {"train": [head(16)]}), 8.0)}
    st, arts = state(tmp_path, runs)
    held = {p: p.read_bytes() for p in st.rglob("*") if p.is_file()}
    plan = tmp_path / "plan.json"
    assert run(monkeypatch, "plan", "--state", st, "--scope", tmp_path / "scope.json", "--layers", "missing,tree",
               "--out", plan, "--workers", 1) == 0
    doc = json.loads(plan.read_text())
    assert sorted(doc["restated"]) == ["r-two"] and sorted(doc["kept"]) == ["r-mezo", "r-none", "r-one"]
    assert doc["kept"]["r-none"] == "no trace" and doc["restated"]["r-two"]["trace_cache"] == "tree"

    over = tmp_path / "overlay"
    assert run(monkeypatch, "overlay", "--state", st, "--plan", plan, "--overlay", over, "--drop-cache", "other",
               "--out", tmp_path / "overlay.json") == 0
    assert (over / "ledger.json").is_symlink() and sorted(p.name for p in (over / "trace40").iterdir()) == ["tree"]
    for name, art in arts.items():
        d = over / "records" / art
        assert d.is_symlink() == (name != "r-two")
    d = over / "records" / arts["r-two"]
    assert (d / "witness.json").is_symlink() and (d / "run" / "E.bin").is_symlink()
    out = (d / "run" / "run.json").read_bytes()
    assert not (d / "run" / "run.json").is_symlink()
    assert hashlib.sha256(out).hexdigest() == doc["restated"]["r-two"]["after_sha256"]
    assert [s["tokens"] for s in json.loads(out)["steps"]] == [2 * ROWS] * 6 + [ROWS]
    assert {p: p.read_bytes() for p in st.rglob("*") if p.is_file()} == held

    (tmp_path / "only.txt").write_text("r-other\n")
    assert run(monkeypatch, "overlay", "--state", st, "--plan", plan, "--overlay", tmp_path / "none", "--only",
               tmp_path / "only.txt", "--out", tmp_path / "none.json") == 0
    assert (tmp_path / "none" / "records" / arts["r-two"]).is_symlink()

    assert run(monkeypatch, "one", "--record", st / "records" / arts["r-two"] / "run" / "run.json", "--plan", plan,
               "--run", "r-two", "--out", tmp_path / "one.json") == 0
    assert (tmp_path / "one.json").read_bytes() == out
    with pytest.raises(SystemExit, match="not the record the plan read"):
        run(monkeypatch, "one", "--record", st / "records" / arts["r-one"] / "run" / "run.json", "--plan", plan,
            "--run", "r-two", "--out", tmp_path / "bad.json")


def traced(d, tree, runs):
    (d / "records").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"te_commit": "c", "te_tree": tree, "runs": len(runs)}))
    (d / "traced.json").write_text(json.dumps({r: ({"error": v} if v.startswith("!") else {"rows": 1}) for r, v in runs.items()}))
    for r, v in runs.items():
        if not v.startswith("!"):
            with gzip.open(d / "records" / f"{r}.jsonl.gz", "wt") as fh:
                fh.write(v + "\n")


def test_layer(tmp_path, monkeypatch):
    traced(tmp_path / "base", "t", {"a": "base a", "b": "base b", "c": "base c"})
    traced(tmp_path / "o1", "t", {"a": "o1 a", "b": "!timed out", "d": "o1 d"})
    traced(tmp_path / "o2", "t", {"a": "o2 a", "e": "!refused"})
    assert run(monkeypatch, "layer", "--base", tmp_path / "base", "--over", tmp_path / "o1", "--over", tmp_path / "o2",
               "--dest", tmp_path / "layer", "--out", tmp_path / "layer.json") == 0
    doc = json.loads((tmp_path / "layer.json").read_text())
    assert doc["from_base"] == ["b", "c"] and doc["from_over"] == ["a", "d"] and sorted(doc["over_failed"]) == ["b", "e"]
    got = {p.name.split(".")[0]: gzip.open(p, "rt").read().strip() for p in (tmp_path / "layer" / "records").iterdir()}
    assert got == {"a": "o2 a", "b": "base b", "c": "base c", "d": "o1 d"}
    (tmp_path / "only.txt").write_text("d\n")
    assert run(monkeypatch, "layer", "--base", tmp_path / "base", "--over", tmp_path / "o1", "--only", tmp_path / "only.txt",
               "--dest", tmp_path / "l1", "--out", tmp_path / "l1.json") == 0
    doc = json.loads((tmp_path / "l1.json").read_text())
    assert doc["from_base"] == ["a", "b", "c"] and doc["from_over"] == ["d"] and doc["over_failed"] == {}
    traced(tmp_path / "o3", "u", {"a": "o3 a"})
    with pytest.raises(SystemExit, match="te_core tree"):
        run(monkeypatch, "layer", "--base", tmp_path / "base", "--over", tmp_path / "o3", "--dest", tmp_path / "l2",
            "--out", tmp_path / "l2.json")


def test_every_run_of_the_data_file_has_a_reason():
    names = restate.one_name()
    assert names and all(isinstance(v, str) and v for v in names.values())
