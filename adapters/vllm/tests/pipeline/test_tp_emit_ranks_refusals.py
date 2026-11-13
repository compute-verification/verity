"""`verity_vllm.pipeline.tp.emit_ranks` where a rank gets no directory: a template whose derivation kept no session, a rank
whose report is a refusal (the relabel's own, or the lowering's of it), a rank whose files cannot be finished, and a template
whose derive failed.  Each is derived for real by the row, so the rank's record says why and no directory of it is left: written,
half written, or left by an earlier relabel into the same root.  The relabel's report is a stand-in: no derivation runs."""
from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.pipeline.tp import emit_ranks as ER
from verity_vllm.program.frontend import relabel

#: the template is rank 1 of a world of three; its ranks' embedding rows are four each
WORLD, TEMPLATE = 3, 1
SESSION = object()
NO_SESSION = "RelabelRefused: the template's derivation kept no session"


def _shards(r: int) -> list[dict]:
    return [{"module": "embed", "START": 4 * r, "END": 4 * r + 4}]


PRE: dict[str, Any] = {"tp": {"world": WORLD, "rank": TEMPLATE, "embedding_shards": _shards(TEMPLATE)}, "model_pin": {"model": "toy", "rank": TEMPLATE},
                       "ok": True}


def _report(r: int, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(rank=r, ok=True, unsupported=[], **fields)


def _refusal(r: int, *unsupported: str) -> SimpleNamespace:
    return SimpleNamespace(rank=r, ok=False, unsupported=list(unsupported))


def _emit(tmp_path: Path, monkeypatch, *, session: Any = SESSION, reports: dict[int, Any] | None = None, post: Any = None,
          ranks: tuple[int, ...] = (0, 2)) -> tuple[ER.Emit, str, list, list]:
    """`emit` of `ranks` from the template's Build `rank1/build_step` into `staged`: (the ranks' record, the template's directory,
    what the relabel was asked for, what the writer was given).  `reports` gives a rank's report; any other rank's is ok."""
    asked: list = []
    written: list = []

    def rank_report(session, r, *, shards):
        asked.append((session, r, shards))
        return (reports or {}).get(r) or _report(r)

    def write(d, res, rep, rank):
        written.append({"dir": d, "rank": rank, "report": rep, "tp": res["tp"], "model_pin": res["model_pin"]})
        Path(d, "program.json").write_text("{}")

    monkeypatch.setattr(relabel, "rank_report", rank_report)
    e = ER.Emit(ranks, str(tmp_path / "staged"))
    out = str(tmp_path / f"rank{TEMPLATE}" / "build_step")
    ER.emit(e, out, copy.deepcopy(PRE), session, shards=_shards, context=contextlib.nullcontext, post=post, write=write)
    return e, out, asked, written


def _dirs(e: ER.Emit, out: str, r: int) -> tuple[str, str]:
    """(rank r's directory, the hidden one it is written in)."""
    return e.dir_of(out, r), ER.tmp_of(e.dir_of(out, r))


# emit

def test_a_template_that_kept_no_session_writes_no_rank_and_each_ranks_record_says_so(tmp_path, monkeypatch):
    e, out, asked, written = _emit(tmp_path, monkeypatch, session=None)
    assert [(rec["rank"], rec["dir"], rec["ok"], rec["error"]) for rec in e.report] == [(r, e.dir_of(out, r), False, NO_SESSION) for r in (0, 2)]
    assert all(rec["traceback_tail"].rstrip().endswith("the template's derivation kept no session") for rec in e.report)
    assert (asked, written, e.pending) == ([], [], [])
    assert not (tmp_path / "staged").exists(), "a rank refused before its relabel makes no directory, hidden or not"


def test_a_rank_whose_report_is_a_refusal_is_not_written_and_the_rank_after_it_is(tmp_path, monkeypatch):
    e, out, asked, written = _emit(tmp_path, monkeypatch, reports={0: _refusal(0, "aten::roll", "aten::flip")})
    refused, taken = e.report
    assert (refused["rank"], refused["ok"], refused["error"]) == (0, False, "RelabelRefused: rank 0's report is a refusal: ['aten::roll', 'aten::flip']")
    assert (taken["rank"], taken["ok"], taken["error"]) == (2, True, None) and "traceback_tail" not in taken
    assert not os.path.exists(_dirs(e, out, 0)[1]) and not os.path.exists(_dirs(e, out, 0)[0]), "the refused rank's hidden directory is gone"
    dst, tmp = _dirs(e, out, 2)
    assert [w["rank"] for w in written] == [2] and written[0]["dir"] == tmp and os.path.isfile(f"{tmp}/program.json") and not os.path.exists(dst)
    assert written[0]["tp"] == {"world": WORLD, "rank": 2, "embedding_shards": _shards(2)} and written[0]["model_pin"] == {"model": "toy", "rank": 2}
    assert [(r, t, d, res["tp"]["rank"]) for r, t, d, res in e.pending] == [(2, tmp, dst, 2)]
    assert asked == [(SESSION, 0, {(4, 8): (0, 4)}), (SESSION, 2, {(4, 8): (8, 12)})]


def test_a_report_the_lowering_refuses_is_not_written_and_one_it_passes_is_written_as_lowered(tmp_path, monkeypatch):
    lowered: list = []

    def post(rep):
        lowered.append(rep)
        return _refusal(rep.rank, "no Boolean lowering of aten::roll") if rep.rank == 2 else _report(rep.rank, lowered=True)

    reports = {0: _report(0), 2: _report(2)}
    e, out, _, written = _emit(tmp_path, monkeypatch, reports=reports, post=post)
    assert lowered == [reports[0], reports[2]], "the lowering is given the relabel's report of each rank"
    assert [(rec["rank"], rec["ok"], rec["error"]) for rec in e.report] == [
        (0, True, None), (2, False, "RelabelRefused: rank 2's report is a refusal: ['no Boolean lowering of aten::roll']")]
    assert [w["rank"] for w in written] == [0] and written[0]["report"].lowered is True
    assert not os.path.exists(_dirs(e, out, 2)[1])


def test_a_hidden_directory_an_earlier_run_left_is_made_anew_before_the_rank_is_written(tmp_path, monkeypatch):
    left = tmp_path / "staged" / "rank2" / ".build_step.relabel-tmp"
    left.mkdir(parents=True)
    (left / "half_written.bin").write_text("an earlier run's")
    e, out, _, _ = _emit(tmp_path, monkeypatch, ranks=(2,))
    assert _dirs(e, out, 2)[1] == str(left) and os.listdir(left) == ["program.json"]


def test_a_relabel_that_raises_a_long_message_keeps_its_first_four_thousand_characters(tmp_path, monkeypatch):
    def post(rep):
        raise RuntimeError("x" * 3990 + "y" * 100)

    e, out, _, written = _emit(tmp_path, monkeypatch, post=post, ranks=(2,))
    assert e.report[0]["error"] == ("RuntimeError: " + "x" * 3990)[:4000] and len(e.report[0]["error"]) == 4000
    assert written == [] and not os.path.exists(_dirs(e, out, 2)[1])


# finish

def _finish(tmp_path: Path, monkeypatch, *, inputs: bool = True, manifest: Any = None, **template: Any) -> tuple[ER.Emit, str, dict]:
    """`finish` after an `emit` that wrote ranks 0 and 2: (the ranks' record, the template's directory, `<out>.relabel.json`).
    `manifest(dir, res)` is the manifest's writer (default: one that writes the rank's number); `template` are fields of the
    template's finished record over the ones it had before its writer ran."""
    e, out, _, _ = _emit(tmp_path, monkeypatch)
    os.makedirs(out)
    if inputs:
        Path(out, "inputs.json").write_text('{"n_open_events": 1}')
    res = dict(copy.deepcopy(PRE), wall_secs_total=12.5, max_rss_mb=640.0) | template
    ER.finish(e, out, res, manifest or (lambda d, res_r: Path(d, "construction_manifest.json").write_text(str(res_r["tp"]["rank"]))))
    return e, out, json.loads(Path(f"{out}.relabel.json").read_text())


def test_a_finished_rank_is_renamed_into_place_with_the_templates_trace_and_telemetry(tmp_path, monkeypatch):
    e, out, doc = _finish(tmp_path, monkeypatch)
    assert doc["template"] == {"dir": out, "world": WORLD, "rank": TEMPLATE, "ok": True} and doc["root"] == str(tmp_path / "staged")
    assert [(rec["rank"], rec["ok"], rec["error"]) for rec in doc["ranks"]] == [(0, True, None), (2, True, None)]
    for r in (0, 2):
        dst, tmp = _dirs(e, out, r)
        assert sorted(os.listdir(dst)) == ["construction_manifest.json", "inputs.json", "program.json", "result.json"] and not os.path.exists(tmp)
        result = json.loads(Path(dst, "result.json").read_text())
        assert result["relabeled_from"] == {"rank": TEMPLATE, "dir": out} and (result["wall_secs_total"], result["max_rss_mb"]) == (12.5, 640.0)
        assert result["tp"]["rank"] == r and Path(dst, "inputs.json").read_text() == '{"n_open_events": 1}'
        assert Path(dst, "construction_manifest.json").read_text() == str(r)


@pytest.mark.parametrize("ok", [False, None, 1, "true"], ids=["false", "null", "one", "the text true"])
def test_the_report_says_the_template_is_ok_only_when_its_record_says_true(tmp_path, monkeypatch, ok):
    assert _finish(tmp_path, monkeypatch, ok=ok)[2]["template"]["ok"] is False


def test_a_rank_whose_manifest_cannot_be_written_has_no_directory_and_its_error_in_the_report(tmp_path, monkeypatch):
    def manifest(d, res_r):
        if res_r["tp"]["rank"] == 0:
            raise RuntimeError("manifest of rank 0: " + "x" * 5000)
        Path(d, "construction_manifest.json").write_text("{}")

    e, out, doc = _finish(tmp_path, monkeypatch, manifest=manifest)
    failed, finished = doc["ranks"]
    assert (failed["rank"], failed["ok"], failed["error"]) == (0, False, ("RuntimeError: manifest of rank 0: " + "x" * 5000)[:4000])
    assert (finished["rank"], finished["ok"], finished["error"]) == (2, True, None)
    assert not os.path.exists(_dirs(e, out, 0)[0]) and not os.path.exists(_dirs(e, out, 0)[1]), "neither the rank's directory nor its hidden one is left"
    assert os.path.isfile(f"{_dirs(e, out, 2)[0]}/result.json") and not os.path.exists(_dirs(e, out, 2)[1])


def test_a_template_without_its_trace_finishes_no_rank(tmp_path, monkeypatch):
    _, out, doc = _finish(tmp_path, monkeypatch, inputs=False)
    assert [(rec["rank"], rec["ok"]) for rec in doc["ranks"]] == [(0, False), (2, False)]
    assert all(rec["error"] == f"FileNotFoundError: [Errno 2] No such file or directory: '{out}/inputs.json'" for rec in doc["ranks"])
    assert [os.listdir(tmp_path / "staged" / f"rank{r}") for r in (0, 2)] == [[], []], "neither a directory nor a hidden one is left"


@pytest.mark.parametrize("earlier", ["program.json", "stale.bin"])
def test_a_rank_finished_over_an_earlier_directory_of_its_own_replaces_it_whole(tmp_path, monkeypatch, earlier):
    stale = tmp_path / "staged" / "rank2" / "build_step"
    stale.mkdir(parents=True)
    (stale / earlier).write_text("stale")
    doc = _finish(tmp_path, monkeypatch)[2]
    assert doc["ranks"][1]["ok"] is True and Path(stale, "program.json").read_text() == "{}"
    assert sorted(os.listdir(stale)) == ["construction_manifest.json", "inputs.json", "program.json", "result.json"]


# a second relabel into one root: a rank it does not write keeps no directory of the first

def _earlier(tmp_path: Path, *ranks: int) -> list[Path]:
    """Each of `ranks`' directories as an earlier relabel into `staged` finished it."""
    dirs = [tmp_path / "staged" / f"rank{r}" / "build_step" for r in ranks]
    for d in dirs:
        d.mkdir(parents=True)
        (d / "program.json").write_text("an earlier relabel's")
    return dirs


@pytest.mark.parametrize("session,reports", [(None, None), (SESSION, {0: _refusal(0, "aten::roll")})], ids=["no session kept", "a report that is a refusal"])
def test_a_rank_refused_in_emit_keeps_no_directory_of_an_earlier_relabel(tmp_path, monkeypatch, session, reports):
    earlier, = _earlier(tmp_path, 0)
    e = _emit(tmp_path, monkeypatch, session=session, reports=reports)[0]
    assert (e.report[0]["rank"], e.report[0]["ok"]) == (0, False) and not earlier.exists()


def test_a_rank_that_fails_in_finish_keeps_no_directory_of_an_earlier_relabel(tmp_path, monkeypatch):
    earlier = _earlier(tmp_path, 0, 2)
    doc = _finish(tmp_path, monkeypatch, inputs=False)[2]
    assert [(rec["rank"], rec["ok"]) for rec in doc["ranks"]] == [(0, False), (2, False)] and not any(d.exists() for d in earlier)


def test_a_template_whose_derive_failed_relabels_no_rank_and_no_rank_keeps_a_directory_of_an_earlier_relabel(tmp_path):
    earlier = _earlier(tmp_path, 0, 2)
    e, out = ER.Emit((0, 2), str(tmp_path / "staged")), str(tmp_path / f"rank{TEMPLATE}" / "build_step")
    os.makedirs(out)
    ER.finish(e, out, dict(copy.deepcopy(PRE), ok=False), lambda d, res_r: pytest.fail("no rank was written, so none has a manifest"))
    doc = json.loads(Path(f"{out}.relabel.json").read_text())
    assert doc["template"] == {"dir": out, "world": WORLD, "rank": TEMPLATE, "ok": False}
    assert doc["ranks"] == [{"rank": r, "dir": str(d), "ok": False, "error": "the template's derive failed: no relabel ran"} for r, d in zip((0, 2), earlier)]
    assert not any(d.exists() for d in earlier)
