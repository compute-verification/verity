"""prove_rows.py on a stand-in `window.py prove` run: a row is a block type's largest timed statement, in the fields
the break-even planner's reader requires and under the census type the statement was staged for; a statement whose
prover didn't exit 0, that has no time, that filed a session not accepted or whose stage.json the run lacks gives no
time and is listed with its reason; and the columns it reads are the ones window.py writes."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from conftest import HERE, load

ROOT = HERE.parent.parent
RUN = "r20260101-000000-0001"
HEAD_V1 = 'SparseMlaHeadBlock_v1{NVIS=64,DOT={"fn":"Dot_v1"}}'
HEAD_V2 = 'SparseMlaHeadBlock_v2{NVIS=64,DOT={"fn":"Dot_v2"}}'
QUANT = "Fp8GroupQuantUe8m0_v1{K=2048,G=128}"


def statement(out: Path, name: str, n: int, seconds: float | str, *, stage: dict | None = None, exit: str | None = "0",
              accepted: list | None = None, k_log: int = 24, **record) -> None:
    """One statement of a run's out/: its prove.tsv line, and its stage.json and exit status in its group's files."""
    tsv = out / "prove.tsv"
    if not tsv.exists():
        out.mkdir(parents=True, exist_ok=True)
        tsv.write_text("group\tstatement\tn\tk_log\tprove_total_s\trecord\n")
    rec = {"device": "gpu", "accepted": [True, True] if accepted is None else accepted, **record}
    with tsv.open("a") as f:
        f.write("\t".join(["gpu0", f"gpu/{name}", str(n), str(k_log), str(seconds), json.dumps(rec)]) + "\n")
    staged, session = out / "gpu0" / "out" / "stage" / name, out / "gpu0" / "out" / "gpu" / name
    staged.mkdir(parents=True)
    session.mkdir(parents=True)
    if stage is not None:
        (staged / "stage.json").write_text(json.dumps({"n": n, "k_log": k_log, **stage}))
    if exit is not None:
        (session / "exit").write_text(exit + "\n")


@pytest.fixture
def reader(monkeypatch):
    return load(monkeypatch, "prove_rows.py")


def test_a_row_is_a_block_types_largest_timed_statement_as_the_planner_reads_it(reader, tmp_path):
    """A type staged at a quarter and at a full statement gives its full statement's row and lists the quarter beside
    it; a Boolean version staged for a census type prices that type and names the Definition staged; a type staged by
    its Definition alone is its own; and every row holds the planner's fields."""
    out = tmp_path / "out"
    head = {"definition": HEAD_V2, "word": HEAD_V1, "census": HEAD_V2}
    statement(out, "head-nfull-4", 16, 2.1, stage=head, session_s=2.6)
    statement(out, "head-nfull", 64, 2.6, stage=head, session_s=3.3)
    statement(out, "gq-k2048-n128", 128, 4.0, stage={"definition": QUANT}, k_log=23)
    doc = reader.rows(out, RUN, cores=12)
    assert doc["rows"] == [
        {"block": QUANT, "statement_seconds": 4.0, "blocks": 128, "provers": 1, "cores": 12, "record": RUN,
         "statement": "gq-k2048-n128", "definition": QUANT, "k_log": 23},
        {"block": HEAD_V1, "statement_seconds": 2.6, "blocks": 64, "provers": 1, "cores": 12, "record": RUN,
         "statement": "head-nfull", "definition": HEAD_V2, "k_log": 24, "session_seconds": 3.3}]
    assert [(r["statement"], r["blocks"], r["statement_seconds"]) for r in doc["beside"]] == [("head-nfull-4", 16, 2.1)]
    assert doc["refused"] == []
    assert "cores" not in reader.rows(out, RUN)["rows"][0]
    planner = ast.parse((ROOT / "benchmarks" / "spot_figures" / "breakeven" / "reprice.py").read_text())
    fields = next(ast.literal_eval(n.value) for n in planner.body
                  if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "STEADY_FIELDS")
    assert fields == reader.FIELDS and all(set(fields) <= set(r) for r in doc["rows"])


@pytest.mark.parametrize("more, why", [
    ({"exit": "137"}, "its prover exited 137"),
    ({"exit": None}, "the run holds no exit status of its prover"),
    ({"stage": None}, "the run holds no stage.json of it, so it names no block type"),
    ({"seconds": ""}, "it has no time (prove_total_s: none)"),
    ({"seconds": "None"}, "it has no time (prove_total_s: None)"),
    ({"accepted": [True, False]}, "a session of it wasn't accepted, by the prover's record (`accepted`: [true, false])"),
    ({"accepted": []}, "a session of it wasn't accepted, by the prover's record (`accepted`: [])"),
    ({"verifier_accepted": [True, None]},
     "a session of it wasn't accepted, by the verifier's record (`verifier_accepted`: [true, null])"),
    ({"stage": {"definition": QUANT, "n": 64}}, "its summary's blocks (128) aren't its stage.json's (64)"),
])
def test_a_statement_gives_no_time_unless_it_finished_and_every_session_was_accepted(reader, tmp_path, more, why):
    """Each way a statement measures nothing: it is refused with its reason, its type takes the next statement that
    was timed, and a run with no timed statement exits 1."""
    out = tmp_path / "out"
    kw = {"seconds": 4.0, "stage": {"definition": QUANT}, **more}
    statement(out, "gq-n128", 128, kw.pop("seconds"), **kw)
    statement(out, "gq-n32", 32, 1.5, stage={"definition": QUANT})
    doc = reader.rows(out, RUN)
    assert doc["refused"] == [{"statement": "gq-n128", "why": why}]
    assert [(r["statement"], r["blocks"]) for r in doc["rows"]] == [("gq-n32", 32)] and doc["beside"] == []
    assert reader.main([str(out), "--record", RUN, "--leave", "gq-n32"]) == 1


def test_a_verifier_record_that_accepted_every_session_gives_the_time(reader, tmp_path):
    out = tmp_path / "out"
    statement(out, "gq-n128", 128, 4.0, stage={"definition": QUANT}, verifier_accepted=[True, True])
    assert [r["statement_seconds"] for r in reader.rows(out, RUN)["rows"]] == [4.0]


def test_two_timed_statements_of_one_type_and_size_are_refused_until_one_is_left_out(reader, tmp_path, capsys):
    """The word form and the Boolean version of one census type, each at a full statement, are two measurements of
    it: the reader chooses neither, and `--leave` names the one that isn't the type's row."""
    out = tmp_path / "out"
    statement(out, "head-v1-nfull", 64, 3.0, stage={"definition": HEAD_V1, "word": HEAD_V1})
    statement(out, "head-v2-nfull", 64, 2.6, stage={"definition": HEAD_V2, "word": HEAD_V1})
    with pytest.raises(SystemExit, match="head-v1-nfull, head-v2-nfull.*--leave all but one"):
        reader.rows(out, RUN)
    assert reader.main([str(out), "--record", RUN, "--leave", "head-v1-nfull"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [r["statement"] for r in doc["rows"]] == ["head-v2-nfull"]
    assert doc["refused"] == [{"statement": "head-v1-nfull", "why": "left out (--leave)"}]
    with pytest.raises(SystemExit, match="--leave names no statement.*absent"):
        reader.rows(out, RUN, leave=("absent",))


def test_the_columns_read_are_the_ones_window_py_writes(reader, tmp_path):
    """prove.tsv's header in window.py's source is the reader's, and a file with any other is refused."""
    assert "\\t".join(reader.HEAD) + "\\n" in (HERE / "window.py").read_text()
    out = tmp_path / "out"
    out.mkdir()
    (out / "prove.tsv").write_text("name\tgroup\tseconds\tdone\texit\n")
    with pytest.raises(SystemExit, match="doesn't start with window.py's columns"):
        reader.rows(out, RUN)
