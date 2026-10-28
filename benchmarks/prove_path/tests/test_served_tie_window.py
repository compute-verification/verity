"""`served_tie_window.run_cases` at the compile slice's class on a three-step window (two prefills, then two decode steps),
its port built as the Commit lays it out: the honest rows are accepted by the tie and by ``ids``, a start one on, a prefill
one short and another request's row by both, and a decode row's request id changed by ``ids`` alone, since the tie binds
start and n; the circuit's per-step verdicts are `served_tie.check`'s."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from verity.core.protocols.certifier import compile_slice as CS
from verity.ml.protocols.certifier import served_tie as ST

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
                                     K=fx["K"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"]))
    st = ST.statics(S)
    run = STW.run_cases(S, st, list(WINDOW), _steps(SimpleNamespace(**st), WINDOW, SEQS), SEQS)
    assert run.verdicts == {"honest": "accepted", "start": "refused by tie and ids", "short": "refused by tie and ids",
                            "id": "refused by ids", "other": "refused by tie and ids"}
    assert run.passed and all(c["circuit_matches"] for c in run.cases.values())
    assert run.cases["id"]["circuit"] == ST.ACCEPT
    assert run.edits == {"prefill": [0, 0, [0, 1, 0, 0, 5]], "decode": [1, 0, [0, 1, 1, 5, 1]], "other_row": [1, 2, 1, 4, 1]}


def test_a_run_sealed_before_the_async_and_cache_statics_reads_them_as_0():
    """A seal's record from before AS, EW and the cache's statics, its engine async or not, reads them as 0, which
    `compile_slice.step_statics` needs; a record that has them keeps its own."""
    old = {"R": 8, "W": 1, "A": 1, "M": 8, "T": 2048, "L": 512, "B": 16, "H": 0, "CH": 1, "WM": 0, "FULL": 1, "PC": 0,
           "NB": 101626, "IW": 16, "Q": 1, "D": 1, "P": 256, "K": 1}
    S = STW.statics_of(SimpleNamespace(priv={"statics": old}))
    assert CS.step_statics(S).AS == 0 and S.EW == 0 and (S.CN, S.CTW, S.CU) == (0, 0, 0)
    assert all(hasattr(S, k) for k in CS.STATICS)
    S = STW.statics_of(SimpleNamespace(priv={"statics": {**old, "AS": 1, "EW": 3, "CN": 2}}))
    assert (S.AS, S.EW, S.CN) == (1, 3, 2)


def test_the_programs_statics_are_the_ports_positions():
    """`statics_held`: the Programs the rows describe are the positions each step's port committed, and a row moved one
    on, or a step's rows in another order, are not."""
    St = SimpleNamespace(**ST.statics(SimpleNamespace(R=2, P=7, IW=6, L=4096, T=256)))
    steps = _steps(St, WINDOW, SEQS)
    assert STW.statics_held(list(WINDOW), steps)
    moved = [WINDOW[0], ((0, 1, 1, 6, 1), WINDOW[1][1]), WINDOW[2]]
    assert not STW.statics_held(moved, steps)
    assert not STW.statics_held(list(WINDOW), _steps(St, [tuple(reversed(WINDOW[0])), *WINDOW[1:]], SEQS))
