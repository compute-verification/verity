"""The small rehearsal's negative controls (`two_stage_control`), up to the replay side's proof blocks of a sampled
span (`two_stage.spans_of`): what each control changes in the class's files, and where the honest replay side stops or
goes on. The stages and the sessions run on a node (``two_stage_small.sh`` with ``SMALL_SERVE_PY`` and
``SMALL_SECOND_PY``)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.definitions.registry import greedy_stages as GS
from verity_vllm.pipeline import boundary_rows as BR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
TS = importlib.import_module("two_stage")
SM = importlib.import_module("two_stage_small")
TC = importlib.import_module("two_stage_control")

V, UNITS, O, N = 256, 4, 2, 6
SAMPLE = [1, 4]
COMBINE = f"GreedyCombine_v2{{V={V},N={V // UNITS}}}"


def _served(tmp: Path, control: str, claimed: bool = False):
    """A control's class and capture, the class reopened from its files as the replay side opens it."""
    got = TC.serve(tmp, control, V, O, N, inline=True, units=UNITS, claimed=claimed)
    return got, BR.reopen(got["class"], BR.check_template(SM.template(V, UNITS)))


def _ids(span) -> list[int]:
    words = span.data["outs"]["o0"].astype(np.uint32)
    return (words[:, 0] | words[:, 1] << 16).tolist()


def test_a_changed_id_is_committed_and_an_honest_replay_does_not_reproduce_the_span(tmp_path):
    """Every committed id is another than its logits select, and the replay side lays out nothing: the capture's ids
    are not the served ones."""
    got, c = _served(tmp_path, "id")
    assert all(a != b for r in got["tokens"] for a, b in zip(got["tokens"][r], got["committed"][r], strict=True))
    assert [list(s.served) for s in c.spans] == [got["committed"][f"r{i}"] for i in range(N)]
    with pytest.raises(ValueError, match="didn't reproduce the span"):
        TS.spans_of(c, got["capture"], SAMPLE)


def test_a_capture_that_claims_the_changed_ids_stops_at_the_units_unless_the_prover_claims_them_too(tmp_path, monkeypatch):
    """With the capture's tokens the committed ids, the honest replay side still stops: its units select other ids.
    A prover that claims them (`two_stage_control.claim`) lays the spans out, each combine's units returning the
    committed ids over logits that select others and every chunk's output as it is."""
    got, c = _served(tmp_path, "id", claimed=True)
    with pytest.raises(ValueError, match="select other ids"):
        TS.spans_of(c, got["capture"], SAMPLE)
    honest = {u: [GS.twin(V, row, UNITS)[1] for row in got["rows"][f"r{u}"]] for u in SAMPLE}
    monkeypatch.setattr(GS, "twin", GS.twin)
    TC.claim()
    per, _launches = TS.spans_of(c, got["capture"], SAMPLE)
    for u, span in zip(SAMPLE, per, strict=True):
        assert len(span) == UNITS and span[-1].member.template == COMBINE
        assert _ids(span[-1]) == got["committed"][f"r{u}"] != got["tokens"][f"r{u}"]
        for piece, unit in zip(span[:-1], GS.plan(V, UNITS)[:-1], strict=True):
            (out, _w), = piece.member.outs
            assert piece.data["outs"][out].tolist() == [w[unit.key][0].tolist() for w in honest[u]]


def test_a_changed_max_tokens_is_committed_and_the_replay_side_goes_on_as_for_the_honest_class(tmp_path):
    """The class commits ``max_tokens`` one more than the ids served, finished by length, and nothing on the replay
    side reads it: the spans are the honest class's."""
    got, c = _served(tmp_path, "max-tokens")
    assert got["committed"] == got["tokens"]
    assert {s.sampling["max_tokens"] for s in c.spans} == {O + 1} and {len(s.served) for s in c.spans} == {O}
    assert {s.finish for s in c.spans} == {"length"}
    per, _launches = TS.spans_of(c, got["capture"], SAMPLE)
    assert [_ids(span[-1]) for span in per] == [got["tokens"][f"r{u}"] for u in SAMPLE]
