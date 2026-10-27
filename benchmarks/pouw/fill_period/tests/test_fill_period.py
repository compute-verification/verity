"""One filler period end to end on a local gate (`fill_period.run`): the auditor registers the loop's checkpoint under the
filler circuit and opens the fill period on it, the window's salt and draw are its session's live coins, and
`fill._period_outcomes` credits the fill only for the window's own tables, fixed before the draw and accepted unit by
unit."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fill_period as FP  # noqa: E402


@pytest.mark.slow
def test_a_filler_period_on_a_local_gate_is_credited_and_its_negatives_refused(tmp_path: Path, monkeypatch):
    """The filler window's coins come from its session (the stub logged their requests), its draw is the committed K's,
    every drawn unit is accepted, and the fill is the window's M·n·K a call at that K; a z/y table whose root changed
    after the draw coin and a drawn unit the verifier rejects each refuse the whole fill.

    Slow (about 90 s and 8 GB, most of it the J10-R descriptor's 60 MB, built once and parsed by each of the three
    verdicts): it is the one test that runs the gate's
    model-register, period-open, reserve, open and period-close around a filler window, which the difftest's cases
    (`pouw_fill_cases`) build by hand."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = [FP.WS.a5_coins().on_main([])]
    g = FP.WS.local(tmp_path / "gate", main)
    monkeypatch.setattr(g.A, "serve_argv", g.A.serve_argv)
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    FP.stub(g, tmp_path / "coins.log")
    doc = FP.run(g, main, tmp_path / "out", "demo")
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    assert doc["period"] == "demo" + F.SUFFIX and doc["period_open"][Cr.K_FIELD] == FP.DRAW_K
    assert doc["period_open"][Cr.MODEL_FIELD]["name"] == FP.ROLE
    assert len(doc["drawn"]) == FP.DRAW_K and doc["verdicts"] == ["accept"] * FP.DRAW_K
    assert doc.get("fill") == {"work": [2 * 128 * FP.TILE[0] * 128], "k": FP.DRAW_K, "delta": str(Cr.DELTA),
                               "eps": doc["fill"]["eps"]}, doc.get("refused")
    stale, rejected = doc["negatives"]["stale-entry"], doc["negatives"]["unit-rejected"]
    assert stale["refused"][0] == "fill" and ": draw: " in stale["refused"][1], stale
    assert rejected["refused"][0] == "fill" and ": failed: " in rejected["refused"][1], rejected
