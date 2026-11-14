"""holdout.py (protected): a seed's workloads are the same workloads at every run, so a holdout can be made again from its
seed, and every request keeps to the generator's pools and off the reserved token ids; `assemble` passes the holdouts whose
six gates each pass; and what `main` writes, prints and exits with."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.check.gates import HOLDOUT_GATES
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.properties import holdout

CASE, MODEL, VOCAB, SEED = "B0", "m/x", 1000, 20260917


def _generate(out: Path, seed: int = SEED) -> list[dict[str, Any]]:
    paths = holdout.generate(out, case=CASE, model=MODEL, vocab=VOCAB, n=3, seed=seed, n_requests=2)
    assert [p.parent for p in paths] == [out] * 3
    return [{"file": p.name, **json.loads(p.read_text())} for p in paths]


def test_a_seeds_workloads_are_the_ones_it_gave_before(tmp_path: Path) -> None:
    docs = _generate(tmp_path)
    assert [d["file"] for d in docs] == ["holdout_B0_0_1363946732.json", "holdout_B0_1_1302387300.json",
                                         "holdout_B0_2_1637491940.json"]
    got = [[(len(r["prompt_token_ids"]), r["max_tokens"], r["prompt_token_ids"][:3]) for r in d["requests"]] for d in docs]
    assert got == [[(31, 5, [164, 540, 449]), (65, 24, [620, 459, 844])],
                   [(33, 9, [748, 408, 384]), (96, 5, [402, 491, 656])],
                   [(128, 24, [204, 452, 621]), (63, 9, [321, 804, 220])]]
    assert docs[0]["requests"][0]["request_id"] == "8dfc6cb1-6c2e-5dcc-b730-0cad0e76770d"


def test_the_workloads_are_a_function_of_the_seed(tmp_path: Path) -> None:
    first, again, other = _generate(tmp_path / "a"), _generate(tmp_path / "b"), _generate(tmp_path / "c", SEED + 1)
    assert [d["sha256"] for d in first] == [d["sha256"] for d in again]
    assert not {d["sha256"] for d in first} & {d["sha256"] for d in other}


def test_every_request_keeps_to_the_pools_and_off_the_reserved_ids(tmp_path: Path) -> None:
    for d in _generate(tmp_path):
        assert d["prompt_lengths"] == [len(r["prompt_token_ids"]) for r in d["requests"]]
        assert len({r["request_id"] for r in d["requests"]}) == 2
        for r in d["requests"]:
            assert len(r["prompt_token_ids"]) in holdout.LENGTH_POOL and r["max_tokens"] in holdout.DECODE_POOL
            assert all(100 <= t < VOCAB - 100 for t in r["prompt_token_ids"])


def _run(root: Path, name: str, gates: Any = None, workload: str | None = None) -> Path:
    """A holdout's run directory under `root`, with its `gates.json` when `gates` is given and a workload file named `workload`."""
    d = root / name
    d.mkdir(parents=True)
    if gates is not None:
        (d / "gates.json").write_text(json.dumps(gates))
    if workload:
        (d / workload).write_text("{}")
    return d


def test_assemble_passes_a_holdout_whose_six_gates_each_pass(tmp_path: Path) -> None:
    """`gates.json` in each of its three forms: the gates by id as the gates document has them, a list of them, and the
    statuses by id alone. A gate past G6 is not a holdout's, and a file beside the run directories is none of them."""
    passing = dict.fromkeys(HOLDOUT_GATES, "pass")
    a = _run(tmp_path, "a", {"gates": {**{g: {"status": "pass"} for g in HOLDOUT_GATES}, "G7": {"status": "fail"}}}, "workload.json")
    _run(tmp_path, "b", {"gates": [{"id": g, "status": "pass"} for g in (*HOLDOUT_GATES, "G8")]}, "holdout_B0_0_7.json")
    _run(tmp_path, "c", passing)
    (tmp_path / "holdouts.json").write_text("{}")
    rep = holdout.assemble(tmp_path)
    assert rep["holdouts"] == [
        {"name": "a", "run_dir": str(a), "workload": str(a / "workload.json"), "gates": passing, "status": "pass"},
        {"name": "b", "run_dir": str(tmp_path / "b"), "workload": None, "gates": passing, "status": "pass"},
        {"name": "c", "run_dir": str(tmp_path / "c"), "workload": None, "gates": passing, "status": "pass"}]
    assert (rep["schema"], rep["status"], rep["ok"], rep["n"]) == ("verity-gen/holdouts/v1", "pass", True, 3)
    assert set(rep) == {"schema", "status", "holdouts", "n", "ok", "digest"}


@pytest.mark.parametrize("gates, status", [
    (None, "missing gates.json"),
    ({}, "fail"),
    ({"G7": "pass"}, "fail"),
    ({**dict.fromkeys(HOLDOUT_GATES, "pass"), "G3": "fail"}, "fail"),
    ({g: "pass" for g in HOLDOUT_GATES[:-1]}, "fail")])
def test_assemble_fails_on_one_holdout_that_does_not_pass(tmp_path: Path, gates: Any, status: str) -> None:
    _run(tmp_path, "a", dict.fromkeys(HOLDOUT_GATES, "pass"))
    _run(tmp_path, "b", gates)
    rep = holdout.assemble(tmp_path)
    assert [(h["name"], h["status"]) for h in rep["holdouts"]] == [("a", "pass"), ("b", status)]
    assert rep["holdouts"][1]["gates"] == {k: v for k, v in (gates or {}).items() if k in HOLDOUT_GATES}
    assert (rep["status"], rep["ok"], rep["n"]) == ("fail", False, 2)


def test_assemble_of_no_holdout_fails(tmp_path: Path) -> None:
    rep = holdout.assemble(tmp_path)
    assert (rep["status"], rep["ok"], rep["n"], rep["holdouts"]) == ("fail", False, 0, [])


def test_main_assembles_into_the_file_named_and_exits_by_the_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(tmp_path / "holdouts", "a", dict.fromkeys(HOLDOUT_GATES, "pass"))
    out = tmp_path / "holdouts.json"
    assert holdout.main(cli.parse("holdout", ["--assemble", str(tmp_path / "holdouts"), "--out", str(out)], {})) == 0
    assert json.loads(out.read_text()) == holdout.assemble(tmp_path / "holdouts")
    assert out.read_text() == json.dumps(holdout.assemble(tmp_path / "holdouts"), indent=1) + "\n"
    _run(tmp_path / "holdouts", "b")
    assert holdout.main(cli.parse("holdout", ["--assemble", str(tmp_path / "holdouts"), "--out", str(out)], {})) == 1
    assert json.loads(out.read_text())["status"] == "fail"
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.split("\n") == [
        "[holdout] pass: 1 holdouts a=pass", "[holdout] fail: 2 holdouts a=pass, b=missing gates.json", ""]


def test_main_generates_a_seeds_workloads_and_prints_each(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--generate", "--case", CASE, "--model", MODEL, "--vocab", str(VOCAB), "--n", "2", "--n-requests", "2",
            "--seed", str(SEED), "--tokenizer-revision", "abc", "--out", str(tmp_path / "B0")]
    assert holdout.main(cli.parse("holdout", argv, {})) == 0
    docs = _generate(tmp_path / "again")
    names = ["holdout_B0_0_1363946732.json", "holdout_B0_1_1302387300.json"]
    assert sorted(p.name for p in (tmp_path / "B0").iterdir()) == names == [d["file"] for d in docs[:2]]
    made = [json.loads((tmp_path / "B0" / n).read_text()) for n in names]
    assert [d["requests"] for d in made] == [d["requests"] for d in docs[:2]]
    assert [d["engine"]["revision"] for d in made] == ["abc", "abc"] and [d["engine"]["revision"] for d in docs] == [None] * 3
    printed = capsys.readouterr()
    assert printed.out.split("\n") == [f"{tmp_path / 'B0' / names[0]}  lengths=[31, 65] max_tokens=[5, 24]",
                                       f"{tmp_path / 'B0' / names[1]}  lengths=[33, 96] max_tokens=[9, 5]", ""]
    assert printed.err == f"[holdout] seed {SEED}: 2 workloads\n"


@pytest.mark.parametrize("argv, why", [
    (["--generate", "--model", MODEL, "--vocab", "1000"], "--generate needs --case, --model, --vocab"),
    (["--generate", "--case", CASE, "--vocab", "1000"], "--generate needs --case, --model, --vocab"),
    (["--generate", "--case", CASE, "--model", MODEL], "--generate needs --case, --model, --vocab"),
    ([], "one of --generate / --assemble")])
def test_main_without_what_it_needs_is_a_usage_error(tmp_path: Path, argv: list[str], why: str) -> None:
    with pytest.raises(UsageError) as refused:
        holdout.main(cli.parse("holdout", [*argv, "--out", str(tmp_path / "out")], {}))
    assert str(refused.value) == why and not (tmp_path / "out").exists()
