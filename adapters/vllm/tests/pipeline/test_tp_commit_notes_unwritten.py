"""`verity_vllm.pipeline.tp.commit`: what `tp-commit` does when a note it adds to one of its own records cannot be written, when its
challenger session does not close, and when it is asked to defer its replay at a world whose deferral is not proven.  The notes
are beside the Commit's verdict: the families of a rank's replay that the cross-rank pass evaluated (`finalize_pair<N>.json`),
the value check, the Match record's linkage and the C2 of record (`coverage.json`), and the verdict of a negative that was not
applied (`runs.jsonl`).  The Commit says the note failed and decides as it would have.
CPU-only, in the frame of `tests.pipeline.test_tp_commit_pair`; a write fails as one to a full disk does."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.pipeline import test_tp_commit_options as TO
from tests.pipeline import test_tp_commit_pair as TP
from tests.pipeline import test_tp_commit_pair_match as PM
from tests.pipeline import test_tp_commit_pair_replay as PR
from tests.pipeline import test_tp_commit_pair_xrank as PX
from verity_vllm.pipeline import commit_coins as CC
from verity_vllm.pipeline.tp import commit as TPCM

driver, pair = TO.driver, TP.pair

FULL = "OSError(28, 'No space left on device')"


def _full_disk(monkeypatch, name: str, nth: int) -> list[str]:
    """The ``nth`` `Path.write_text` to a file named ``name`` fails as a write to a full disk does; every other write is made.
    Returns the texts written to that name, in order (the failed one too)."""
    real, texts = Path.write_text, list[str]()

    def write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == name:
            texts.append(data)
            if len(texts) == nth:
                raise OSError(28, "No space left on device")
        return real(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    return texts


def _printed(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


# ------------------------------------------------------------------------------------------------ a note that cannot be written

def test_the_families_the_cross_rank_pass_evaluated_that_cannot_be_noted_are_said_and_the_pair_keeps_its_verdict(pair, monkeypatch, capsys):
    PX._dumped(pair)
    texts = _full_disk(monkeypatch, "finalize_pair0.json", 2)
    assert pair.run(*PX.ON) == 0
    assert f"[tp2commit] unreplayable.evaluated_by annotate failed: {FULL}" in _printed(capsys)
    c = pair.rows()[1]["commit"]
    assert c["commit_pass"] is True and [c["per_rank"][r]["cross_rank_collectives"] for r in "01"] == [PX.PASSED, PX.PASSED]
    assert len(texts) == 2 and pair.read("finalize_pair0.json") == json.loads(texts[0]), "the ranks' finalize records as first written"
    assert pair.read("summary.json")["pass"] is True


def test_a_value_check_that_cannot_be_noted_in_the_coverage_is_said_and_the_commit_keeps_its_verdict(pair, monkeypatch, capsys):
    texts = _full_disk(monkeypatch, "coverage.json", 2)
    assert pair.run("--value-check") == 0
    assert f"[tp2commit] coverage.json value_correspondence annotate failed: {FULL}" in _printed(capsys)
    cov = pair.read("coverage.json")
    unwritten = json.loads(texts[1])
    assert (unwritten["value_correctness"], unwritten["value_correspondence"]["file"]) == (True, "value_check.json")
    # the C2 of record is then noted on the coverage as the pair wrote it
    assert len(texts) == 3 and cov == json.loads(texts[2]) and cov["value_correctness"] is None
    assert (cov["value_correspondence"]["method"], cov["value_correspondence"]["file"]) == ("no value check / Match record this attempt", None)
    assert pair.read("value_check.json")["verdict"] == "PASS" and pair.read("summary.json")["value_check"]["verdict"] == "PASS"
    assert pair.read("summary.json")["pass"] is True and cov["verdict"] == "PASS"


def test_a_match_records_linkage_that_cannot_be_noted_in_the_coverage_is_said_and_the_commit_keeps_its_verdict(pair, monkeypatch, tmp_path, capsys):
    PM._agreed(pair)
    texts = _full_disk(monkeypatch, "coverage.json", 2)
    assert pair.run("--match-dir", str(PM._match(tmp_path))) == 0
    assert f"[tp2commit] coverage.json linkage annotate failed: {FULL}" in _printed(capsys)
    assert json.loads(texts[1])["value_correspondence"]["method"].startswith("Match-record compare only (no --value-check this attempt)")
    cov = pair.read("coverage.json")
    assert len(texts) == 3 and cov == json.loads(texts[2]) and cov["value_correspondence"]["method"] == "no value check / Match record this attempt"
    assert pair.read("summary.json")["pass"] is True and cov["verdict"] == "PASS"


def test_the_c2_of_record_that_cannot_be_noted_in_the_coverage_is_said_and_the_commit_keeps_its_verdict(pair, monkeypatch, capsys):
    texts = _full_disk(monkeypatch, "coverage.json", 2)
    assert pair.run() == 0
    said = _printed(capsys)
    assert f"[tp2commit] coverage.json sampled_replay annotate failed: {FULL}" in said and said[-1].startswith("[tp2commit] PASS ")
    cov = pair.read("coverage.json")
    assert len(texts) == 2 and cov == json.loads(texts[0]) and "value_correspondence" not in cov, "the coverage as the pair wrote it"
    assert sorted(json.loads(texts[1])["value_correspondence"]["components"]) == ["boundary_linkage", "cross_rank_collectives", "fold_match_binding",
                                                                                   "query_population", "sampled_replay", "weights_pin"]
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"]) == (True, [True]) and "replay_deferred" not in summary


SWAP = ("--swap-requests", "1:q0:q1")
NOT_SWAPPED = {"swap_requests": {"rank": "1", "swap_requests_applied": [None], "applied": False}}
WHY_NOT = f"NEGATIVE NOT APPLIED: ['swap_requests'] -- {json.dumps(NOT_SWAPPED)}; the run is an error, not a verdict"


def test_rows_that_cannot_be_rewritten_for_a_negative_not_applied_are_said_and_the_run_is_still_an_error(pair, monkeypatch, capsys):
    texts = _full_disk(monkeypatch, "runs.jsonl", 1)
    assert pair.run(*SWAP) == 3
    said = _printed(capsys)
    assert said[-2:] == [f"[tp2commit] runs.jsonl rewrite failed: {FULL}", f"[tp2commit] {WHY_NOT}"]
    control, run = pair.rows()
    assert (control["arm"], run["commit"]["commit_pass"], run["commit"]["commit_fail_reason"]) == ("control", True, "PASS"), "the rows as the pair wrote them"
    assert [json.loads(line)["commit"]["commit_fail_reason"] for line in texts[0].splitlines()[1:]] == [WHY_NOT]
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["negative_not_applied"]) == (None, ["swap_requests"])


def test_a_blank_line_among_the_rows_is_no_row_and_is_left_out_of_the_rewrite(pair):
    pair.out.mkdir(parents=True)
    (pair.out / "runs.jsonl").write_text("\n  \n")
    assert pair.run(*SWAP) == 3
    text = (pair.out / "runs.jsonl").read_text()
    lines = text.splitlines()
    assert [json.loads(line)["arm"] for line in lines] == ["control", "instrumented"] and text == "".join(line + "\n" for line in lines)
    assert json.loads(lines[1])["commit"]["commit_fail_reason"] == WHY_NOT


# ------------------------------------------------------------------------------------------------ a challenger session that does not close

class _Challenger:
    """The challenger's end of the Commit's challenge phase once it is open: it opens each coin asked of it, and at the close it
    hangs up with ``why`` (None: it closes)."""

    def __init__(self, why: str | None) -> None:
        self.why, self.opened, self.closed = why, list[tuple[str, int, bytes]](), 0
        self.commitment = SimpleNamespace(to_json=lambda: {"scheme": CC.SCHEME_TAG})

    def open(self, family: str, index: int, root: bytes) -> SimpleNamespace:
        self.opened.append((family, index, root))
        coin = bytes([len(self.opened)]) * 64
        return SimpleNamespace(coin=coin, to_json=lambda: {"coin": coin.hex()})

    def close(self) -> None:
        self.closed += 1
        if self.why:
            raise CC.CoinError(self.why)


NOT_CLOSED = "[tp2commit] the challenger session did not close (its ledger has no end, and its coins credit nothing): "


@pytest.mark.parametrize("why", ["the challenger hung up before its end entry", None], ids=["it does not close", "it closes"])
def test_a_challenger_session_that_does_not_close_is_said_once_and_the_commit_keeps_its_verdict(pair, monkeypatch, capsys, why):
    challenger, runs = _Challenger(why), list[int]()

    def of_options(args: Any, n: int) -> CC.ChallengePhase:
        runs.append(n)
        phase = CC.ChallengePhase(CC.ChallengerGate("a gate no test calls"), {"session": "s"}, n)
        phase.client = challenger                                                   # the phase as it is once opened
        return phase

    monkeypatch.setattr(CC, "of_options", of_options)
    assert pair.run() == 0
    said = _printed(capsys)
    assert [line for line in said if line.startswith(NOT_CLOSED)] == ([NOT_CLOSED + why] if why else [])
    c = pair.rows()[1]["commit"]
    root = bytes.fromhex(c["tp_run_root"])
    # one phase for the Commit's one run of record, its three coins opened after the pair's TP run root, closed once
    assert runs == [1] and challenger.closed == 1 and challenger.opened == [(family, 0, root) for family in CC.FAMILIES]
    assert (c["coins"]["source"], c["coins"]["index"], c["coins"]["root"]) == ("challenger", 0, root.hex())
    assert CC.tp_keys(c["coins"]) == (bytes([1]) * 64, bytes([2]) * 64) and c["commit_pass"] is True
    assert pair.read("summary.json")["pass"] is True


# ------------------------------------------------------------------------------------------------ a deferral that is not taken

def test_a_commit_asked_to_defer_at_a_world_whose_deferral_is_not_proven_replays_in_process_and_says_so_in_its_summary(pair, monkeypatch, tmp_path, capsys):
    """The stand-in engine has two ranks, so the proven worlds are set to TP8 alone."""
    monkeypatch.setattr(TPCM.C2, "DEFER_WORLDS", (8,))
    PR._replayed(pair)
    assert pair.run(*PR._of_record(tmp_path), "--replay-deferred") == 0
    why = "world 2: deferral is proven at TP8 only"
    said = _printed(capsys)
    assert f"[tp2commit] --replay-deferred not taken ({why}): every rank's C2 replay runs in its process" in said
    assert said[-1].startswith("[tp2commit] PASS ")
    assert [kw for method, kw in pair.sent if method == "tp2_commit_finalize"] == [{}, {}], "no rank is told to defer"
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"], summary["replay_deferred"]) == (True, [True], {"skipped": why})
    assert not (pair.out / TPCM.C2.DEFERRED).exists() and not list(pair.out.glob("replay_bundle_*"))
    c = pair.rows()[1]["commit"]
    assert [c["per_rank"][r]["sampled_replay"]["grade"] for r in "01"] == ["complete", "complete"]
