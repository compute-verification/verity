"""A stand-in `wipe-protocol` for the subprotocol's tests: the period's Lean judge, answering each case as the Python answered
it (`wipe_run.run`'s `answer`), so a test of the subprotocol's accept runs without a Lean build."""

from __future__ import annotations

import json
from pathlib import Path

from verity.protocols.resource_accounting.memory.pous.prover import wipe_run


def judge(period: Path, monkeypatch) -> list[dict]:
    """`wipe-protocol` among ``period``'s tools, stood in through ``monkeypatch``; the list returned gets each answer it
    gave."""
    doc = json.loads((period / "period.json").read_text())
    (period / "period.json").write_text(json.dumps(doc | {"tools": {"wipe-protocol": "wipe-protocol"}}))
    wipes, given, run = [], [], wipe_run.run

    def ran(*args, **kw):
        wipes.append(run(*args, **kw))
        return wipes[-1]

    def serve(exe, cases):
        assert exe == "wipe-protocol" and list(cases) == [wipes[-1].case]
        given.append(wipes[-1].answer)
        return [wipes[-1].answer]

    monkeypatch.setattr(wipe_run, "run", ran)
    monkeypatch.setattr(wipe_run, "serve", serve)
    return given
