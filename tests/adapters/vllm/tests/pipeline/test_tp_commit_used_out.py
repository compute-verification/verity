"""`tp-commit` (`verity_vllm.pipeline.tp.commit`) into the out dir of an earlier deferred Commit (`--replay-deferred`), and what
`tp-commit-replay` (`verity_vllm.pipeline.tp.commit_replay`) then decides there.  From where a Commit writes its `versions.json`
the out dir holds no verdict of an earlier one: not its `summary.json`, not its `replay_deferred.json`, not a rank's replay
result.  So the replay of a second deferred Commit waits for that Commit's own ranks, and after a Commit refused past that point
there is no deferred Commit to decide.  A Commit refused before that point (at its served switches) leaves the earlier record and
writes its own failing summary beside it: `tp-commit-replay` refuses a summary that names no deferred record, and writes nothing.
CPU-only, in the frames of `tests.pipeline.test_tp_commit_pair_replay` (a deferred pair over two stand-in ranks, whose sealed
bundles hold no store) and `tests.pipeline.test_tp_replay_deferred` (a rank's replay record): a rank's replay on CPU is its
result file, written as `tp-commit-replay` names it, and `tp-commit-replay --fold-only` decides the record."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.pipeline import test_tp_commit_options as TO
from tests.pipeline import test_tp_commit_pair as TP
from tests.pipeline import test_tp_commit_pair_replay as PR
from tests.pipeline import test_tp_replay_deferred as RD
from verity_vllm.engine import served as SV
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import commit_replay as TPR

driver, pair = TO.driver, TP.pair

NO_RANKS = "^fail-closed: committer constructed on 1/2 ranks$"
NO_SWITCHES = "^fail-closed: served switches: no rank answered$"


def _deferred_commit(pair: TP._Pair, root: Path) -> tuple[str, ...]:
    """A deferred Commit over the pair, both ranks pending; its flags, for a second Commit into the same out dir."""
    for r in (0, 1):
        pair.alter("tp2_commit_finalize", r, PR._deferred(pair, r))
    flags = (*PR._of_record(root), "--replay-deferred")
    assert pair.run(*flags) == 0
    return flags


def _replayed(pair: TP._Pair, record: dict[str, Any]) -> list[Path]:
    """Each rank's replay of ``record``'s bundles, as a replay that passed writes its result; the result files."""
    files = []
    for b in record["bundles"]:
        doc = {"schema": TPR.RESULT_SCHEMA, "pair": b["pair"], "rank": b["rank"], "bundle": b["path"], "manifest_sha256": b["manifest_sha256"],
               "where": "cpu", "host": "h", "sampled_replay": dict(RD._sr(rank=b["rank"]), unreplayable=[]),
               "boundary_linkage": RD._lk(True, b["rank"]), "wall_s": 2.5, "peak_rss_bytes": 1 << 30, "utc": "u"}
        files.append(TPR.result_path(pair.out, b["pair"], b["rank"]))
        files[-1].write_text(json.dumps(doc))
    return files


def _files(pair: TP._Pair) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(pair.out.iterdir()) if p.is_file()}


def _no_summary(pair: TP._Pair) -> str:
    return (f"{pair.out / 'summary.json'} names no deferred record, so it is not the summary of the Commit {pair.out / C2.DEFERRED} is of: "
            "run the Commit again")


def _fold(pair: TP._Pair, caplog, *, fold_only: bool = True) -> tuple[int, str]:
    """`tp-commit-replay --fold-only` on the out dir: its exit code and what it logs of the record (nothing where its fold decides).
    ``fold_only=False``: the command as a row runs it, which replays each rank that has no result first."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=TPR.__name__):
        code = TPR.main(TPR.Options(commit_out=str(pair.out), checkpoint_dir="/ck", fold_only=fold_only))
    return code, "\n".join(r.getMessage() for r in caplog.records if r.name == TPR.__name__)


def test_the_replay_of_a_second_deferred_commit_waits_for_its_own_ranks(pair, tmp_path, caplog):
    flags = _deferred_commit(pair, tmp_path)
    first = pair.read(C2.DEFERRED)
    earlier = _replayed(pair, first)
    assert _fold(pair, caplog) == (0, "") and pair.read("summary.json")["pass"] is True
    assert pair.run(*flags) == 0
    second = pair.read(C2.DEFERRED)
    assert "decided" not in second and second["pending_pairs"] == [0] and pair.read("summary.json")["pass"] is None
    assert not {b["manifest_sha256"] for b in second["bundles"]} & {b["manifest_sha256"] for b in first["bundles"]}
    assert not any(p.exists() for p in earlier)
    assert _fold(pair, caplog) == (4, "[tp-replay] PENDING: (pair, rank) [(0, 0), (0, 1)] have no replay on the record yet")
    _replayed(pair, second)
    assert _fold(pair, caplog) == (0, "") and pair.read(C2.DEFERRED)["decided"]["pass"] is True


@pytest.mark.parametrize("decided", [True, False], ids=["the earlier one decided", "the earlier one pending"])
def test_a_commit_refused_at_its_ranks_make_leaves_no_verdict_of_an_earlier_deferred_one(pair, tmp_path, caplog, decided):
    flags = _deferred_commit(pair, tmp_path)
    earlier = pair.read(C2.DEFERRED)
    if decided:
        _replayed(pair, earlier)
        assert _fold(pair, caplog) == (0, "") and pair.read("summary.json")["pass"] is True
    pair.lose("tp2_commit_make", 1)
    with pytest.raises(SystemExit, match=NO_RANKS):
        pair.run(*flags)
    left = sorted(p.name for p in pair.out.iterdir() if p.is_file())
    assert not {"summary.json", C2.DEFERRED, "runs.jsonl"} & set(left) and not [name for name in left if name.startswith("sampled_replay_p")]
    files = _replayed(pair, earlier)                    # the earlier Commit's ranks replay late, from its bundles, which are still sealed
    none = f"[tp-replay] REFUSED: {pair.out / C2.DEFERRED}: not a deferred TP Commit (tp-commit --replay-deferred) or decided in the Commit"
    assert _fold(pair, caplog) == (2, none)
    assert not (pair.out / "summary.json").exists() and all(p.exists() for p in files)


@pytest.mark.parametrize("decided", [False, True], ids=["the earlier one pending", "the earlier one decided"])
def test_the_summary_of_a_commit_refused_at_its_served_switches_never_gains_an_earlier_ones_pass(pair, tmp_path, caplog, decided):
    flags = _deferred_commit(pair, tmp_path)
    earlier = pair.read(C2.DEFERRED)
    if decided:
        _replayed(pair, earlier)
        assert _fold(pair, caplog) == (0, "") and pair.read(C2.DEFERRED)["decided"]["pass"] is True
    for r in (0, 1):
        pair.lose(SV.read_rank, r)
    with pytest.raises(SystemExit, match=NO_SWITCHES):
        pair.run(*flags)
    refusal = pair.read("summary.json")                 # the refusal comes before `versions.json`: the earlier record is as it was
    assert refusal["pass"] is False and "replay_deferred" not in refusal and pair.read(C2.DEFERRED)["bundles"] == earlier["bundles"]
    _replayed(pair, earlier)
    before = _files(pair)
    with pytest.raises(TPR.Refused) as e:
        TPR.fold(pair.out)
    assert str(e.value) == _no_summary(pair) and _fold(pair, caplog) == (2, f"[tp-replay] REFUSED: {_no_summary(pair)}")
    assert _files(pair) == before and pair.read("summary.json") == refusal


IN_PROCESS = {"pass": True, "replay_deferred": {"skipped": "world 4: deferral is proven at [2] only"}}
ANOTHER = {"pass": None, "replay_deferred": {"record": "replay_deferred_p0.json", "pending_pairs": [0], "bundles": []}}


@pytest.mark.parametrize("summary", [None, IN_PROCESS, ANOTHER], ids=[
    "no summary", "the summary of a Commit that replayed its ranks in process", "a summary that names a record by another name"])
def test_a_deferred_record_whose_summary_names_no_deferred_record_is_refused_before_a_rank_replays_and_nothing_is_written(
        pair, tmp_path, caplog, summary):
    _deferred_commit(pair, tmp_path)
    (pair.out / "summary.json").unlink()
    if summary is not None:
        (pair.out / "summary.json").write_text(json.dumps(summary))
    before = _files(pair)
    with pytest.raises(TPR.Refused):                    # no rank's replay is on the record: without the refusal the fold is pending
        TPR.fold(pair.out)
    assert _fold(pair, caplog, fold_only=False) == (2, f"[tp-replay] REFUSED: {_no_summary(pair)}") and _files(pair) == before
    _replayed(pair, pair.read(C2.DEFERRED))
    before = _files(pair)
    assert _fold(pair, caplog) == (2, f"[tp-replay] REFUSED: {_no_summary(pair)}") and _files(pair) == before
