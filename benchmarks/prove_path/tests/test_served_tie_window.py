"""`served_tie_window.run_cases` at the compile slice's class on a three-step window (two prefills, then two decode steps),
its port built as the Commit lays it out: the honest rows are accepted by the tie and by ``ids``, a start one on, a prefill
one short and another request's row by both, and a decode row's request id changed by ``ids`` alone, since the tie binds
start and n; the circuit's per-step verdicts are `served_tie.check`'s."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from verity.core.protocols.certifier import compile_slice as CS, served_tie as ST

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_tie_window as STW  # noqa: E402

FIXTURE = HERE.parents[2] / "verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json"
WINDOW = [((0, 1, 0, 0, 5), (1, 2, 0, 0, 3)), ((0, 1, 1, 5, 1), (1, 2, 1, 3, 1)), ((0, 1, 1, 6, 1), (1, 2, 1, 4, 1))]
SEQS = {1: list(range(101, 111)), 2: list(range(201, 211))}


def _steps(St, window, seqs):
    out = []
    for rows in window:
        ids = [seqs[req][start + t] for _s, req, _k, start, n in rows for t in range(n)]
        pos = [start + t for _s, _r, _k, start, n in rows for t in range(n)]
        pad = [0] * (St.T - len(ids))
        out.append(SimpleNamespace(N=len(ids), chunks=ST.port_chunks(St, ids, pos), ids=ids + pad, positions=pos + pad))
    return out


def test_real_window_cases_on_a_small_window():
    fx = json.loads(FIXTURE.read_text())
    S = SimpleNamespace(**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"],
                                     K=fx["K"], SEG=fx["SEG"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"]))
    st = ST.statics(S)
    run = STW.run_cases(S, st, list(WINDOW), _steps(SimpleNamespace(**st), WINDOW, SEQS), SEQS)
    assert run.verdicts == {"honest": "accepted", "start": "refused by tie and ids", "short": "refused by tie and ids",
                            "id": "refused by ids", "other": "refused by tie and ids"}
    assert run.passed and all(c["circuit_matches"] for c in run.cases.values())
    assert run.cases["id"]["circuit"] == ST.ACCEPT
    assert run.edits == {"prefill": [0, 0, [0, 1, 0, 0, 5]], "decode": [1, 0, [0, 1, 1, 5, 1]], "other_row": [1, 2, 1, 4, 1]}
