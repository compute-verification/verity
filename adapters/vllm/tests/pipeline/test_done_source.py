"""A workload of record is done only at its public cap: its file declares ``sampling.ignore_eos`` and every
request a ``max_tokens``, and its row file, the row precheck and GP-01 refuse it by name otherwise, before any GPU time.  A
research workload keeps its EOS stops, and the generator writes each as before.  What makes a workload of record is the circuit
database's index (``config.RECORDS``)."""
from __future__ import annotations

import json
import os
import sys

import pytest

from tests.pipeline.test_global_program_regress import _derived_dir, _manifest
from tests.pipeline.test_workload import _generate
from verity_vllm import config as C
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import global_program as gp

#: A research workload whose request 0 stops by EOS before its cap.
EOS_ROW = "olmoe-1b-7b__bf16__l40s__tp1__b2__i32__o32__mixedstop__greedy__bi-eager"
SINGLE = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"


def _of_record(monkeypatch, tmp_path, *rows: str) -> None:
    index = tmp_path / "records.json"
    index.write_text(json.dumps({"schema": "verity-vllm/records-index/v1", "rows": {r: {} for r in rows}}))
    monkeypatch.setattr(C, "RECORDS", index)


def _row_file(row: str) -> dict:
    return json.loads((C.WORKLOADS / f"{row}.json").read_text())


def test_every_row_of_record_is_done_at_its_public_cap():
    rows = sorted(C.rows_of_record())
    assert rows
    for row in rows:
        cfg = C.RowConfig.load(row)
        assert C.done_source(cfg.workload) == "public-cap", row
        assert all(isinstance(r["max_tokens"], int) and r["max_tokens"] >= 1 for r in cfg.workload["requests"]), row


def test_the_generator_writes_a_row_of_record_at_its_public_cap_and_a_research_row_as_before(tmp_path):
    row = sorted(C.rows_of_record())[0]
    rec = _generate(tmp_path, row)
    assert rec["sampling"]["ignore_eos"] is True and "done only at its public cap" in rec["note"]
    assert C.done_source(C.RowConfig.of(rec).workload) == "public-cap"
    research = _generate(tmp_path, SINGLE)
    assert "ignore_eos" not in research["sampling"] and research["note"].endswith("EOS untouched.")


def _no_ignore_eos(d):
    d["sampling"].pop("ignore_eos")


def _ignore_eos_string(d):
    d["sampling"]["ignore_eos"] = "true"


def _stop_ids(d):
    d["sampling"]["stop_token_ids"] = [7]


def _uncapped(d):
    d["requests"][0].pop("max_tokens")


def _bool_cap(d):
    d["requests"][0]["max_tokens"] = True


@pytest.mark.parametrize("change, named", [(_no_ignore_eos, "its done source is 'eos-token' (sampling.ignore_eos is None)"),
                                           (_ignore_eos_string, "its done source is 'eos-token' (sampling.ignore_eos is 'true')"),
                                           (_stop_ids, "sampling.stop_token_ids [7] stops a request before its cap"),
                                           (_uncapped, "declare no public cap"), (_bool_cap, "declare no public cap")])
def test_a_row_of_record_with_another_done_source_is_refused_by_name(change, named):
    row = sorted(C.rows_of_record())[0]
    d = _row_file(row)
    C.RowConfig.of(d)
    change(d)
    with pytest.raises(ValueError, match=r"a row of record \(records.json\) is done only at its public cap") as e:
        C.RowConfig.of(d)
    assert str(e.value).startswith(f"{row}: ") and named in str(e.value)


def test_a_research_row_keeps_its_eos_stops_until_the_index_lists_it(monkeypatch, tmp_path):
    assert EOS_ROW not in C.rows_of_record()
    cfg = C.RowConfig.load(EOS_ROW)
    assert C.done_source(cfg.workload) == "eos-token" and C.done_source_refusal(cfg.workload) is None
    _of_record(monkeypatch, tmp_path, EOS_ROW)
    with pytest.raises(ValueError, match="its done source is 'eos-token'"):
        C.RowConfig.load(EOS_ROW)


def test_the_row_precheck_refuses_a_row_of_record_that_stops_at_eos_before_writing_anything(monkeypatch, tmp_path, capsys):
    from verity_vllm.pipeline import row
    _of_record(monkeypatch, tmp_path, SINGLE)
    sweep = tmp_path / "sweep"
    env = {k: v for k, v in os.environ.items() if k not in ("VERITY_TREE", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID")}
    o = cli.parse("row", ["run", SINGLE, "LLAMA32_1B", "unsloth/Llama-3.2-1B", "9535bd9b", "--sweep-dir", str(sweep), "--py", sys.executable,
                          "--native-collect-default", str(tmp_path / "nc_build")], env)
    assert row.run_row(o, "build,match,commit") == 3
    err = capsys.readouterr().err
    assert f"PRECHECK_FAIL_DONE_SOURCE: {SINGLE}: a row of record" in err and "its done source is 'eos-token'" in err
    assert not sweep.exists()


def _swept(path: str, row: str) -> str:
    m = json.loads(open(path).read())
    m["sweep"] = {"row_id": row}
    open(path, "w").write(json.dumps(m))
    return path


def test_gp01_refuses_a_row_of_record_that_stops_at_eos_and_composes_it_at_its_cap(monkeypatch, tmp_path):
    _of_record(monkeypatch, tmp_path, "toy-row")
    reqs = [{"LP": 3, "cap": 3}, {"LP": 5, "cap": 1, "arrive": 1}]
    derived = [_derived_dir(tmp_path, 3, 2), _derived_dir(tmp_path, 5, 0)]
    eos = _swept(_manifest(tmp_path, reqs, name="eos.json", eos_token_ids=[7]), "toy-row")
    res = gp.build(eos, derived, str(tmp_path / "out_eos"))
    assert not res["ok"] and res["reason"] == "done_source" and "toy-row: a row of record" in res["message"]
    assert "its done source is 'eos-token'" in res["message"]
    assert json.loads((tmp_path / "out_eos" / "workload_program.json").read_text()) == res
    capped = _swept(_manifest(tmp_path, reqs, name="capped.json", ignore_eos=True), "toy-row")
    art = gp.build(capped, derived, str(tmp_path / "out_cap"))
    assert art["ok"], art
    assert art["stop_rule"]["ignore_eos"] is True and art["stop_rule"]["eos_token_ids"] == []


def test_gp01_composes_a_research_row_with_its_eos_stop(monkeypatch, tmp_path):
    _of_record(monkeypatch, tmp_path, "toy-row")
    derived = [_derived_dir(tmp_path, 3, 2)]
    research = _swept(_manifest(tmp_path, [{"LP": 3, "cap": 3}], eos_token_ids=[7]), "toy-research-row")
    art = gp.build(research, derived, str(tmp_path / "out"))
    assert art["ok"], art
    assert "ignore_eos" not in art["stop_rule"] and art["stop_rule"]["eos_token_ids"] == [7]
