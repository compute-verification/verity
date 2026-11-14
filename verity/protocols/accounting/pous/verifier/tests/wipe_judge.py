"""A stand-in `wipe-protocol` for the leg's tests: the period's Lean judge, answering each case as the Python answered
it (`wipe_period.run`'s `answer`), so a test of the leg's accept runs without a Lean build."""

from __future__ import annotations

import json
from pathlib import Path

from verity.protocols.accounting.pous.verifier import wipe_period as W


def judge(period: Path, monkeypatch) -> list[dict]:
    """`wipe-protocol` among ``period``'s tools, stood in through ``monkeypatch``; the list returned gets each answer it
    gave."""
    doc = json.loads((period / "period.json").read_text())
    (period / "period.json").write_text(json.dumps(doc | {"tools": {"wipe-protocol": "wipe-protocol"}}))
    wipes, given, run = [], [], W.run

    def ran(*args, **kw):
        wipes.append(run(*args, **kw))
        return wipes[-1]

    def serve(exe, cases):
        assert exe == "wipe-protocol" and list(cases) == [wipes[-1].case]
        given.append(wipes[-1].answer)
        return [wipes[-1].answer]

    monkeypatch.setattr(W, "run", ran)
    monkeypatch.setattr(W, "serve", serve)
    return given
