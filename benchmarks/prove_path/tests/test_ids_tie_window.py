"""`ids_tie_window.run_cases` on a three-step window of four requests, two link-windows of two, its ports built as the
Commit lays them out: the honest rows are accepted by the served rows' tie and in both link-windows by the ids tie; a
relabel of two prefill slots across the link-windows (start and n kept) is accepted by the served rows' tie and refused by
the ids tie, by the map where the relabelled slot is short of its new request's prompt and by the reference where it
covers it; a prompt id the user sent other than the one computed, and a decode slot relabelled onto a prompt position,
are refused in the link-window holding the field."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from verity.core.protocols.certifier import compile_slice as CS, served_tie as ST

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ids_tie_window as ITW  # noqa: E402

FIXTURE = HERE.parents[2] / "verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json"
PROMPTS = [[11, 12, 13, 14], [21, 22, 23, 24, 25], [31, 32, 33], [41, 42, 43, 44, 45, 46]]
OUTS = [[100 * q + k for k in range(2)] for q in range(4)]
WINDOW = [tuple((q, q, 0, 0, len(p)) for q, p in enumerate(PROMPTS)),
          tuple((q, q, 1, len(p), 1) for q, p in enumerate(PROMPTS)),
          tuple((q, q, 1, len(p) + 1, 1) for q, p in enumerate(PROMPTS))]


def _steps(St):
    seqs = [p + o for p, o in zip(PROMPTS, OUTS)]
    out = []
    for rows in WINDOW:
        ids = [seqs[req][start + t] for _s, req, _k, start, n in rows for t in range(n)]
        pos = [start + t for _s, _r, _k, start, n in rows for t in range(n)]
        out.append(SimpleNamespace(N=len(ids), chunks=ST.port_chunks(St, ids, pos)))
    return out


def test_relabels_and_a_wrong_id_on_a_small_window():
    fx = json.loads(FIXTURE.read_text())
    S = SimpleNamespace(**{**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"],
                                        P=fx["P"], K=fx["K"], SEG=fx["SEG"]), "R": 4})
    St = SimpleNamespace(**ST.statics(S))
    run = ITW.run_cases(S, St, list(WINDOW), _steps(St), PROMPTS)
    c = run.cases
    assert run.windows == [0, 1] and run.passed
    assert {n: (r["served_tie"], r["ids_tie"]) for n, r in c.items()} == {
        "honest": ("accepted", "accepted"), "relabel": ("accepted", "refused in window 0, 1"),
        "wrong_id": ("accepted", "refused in window 0"), "decode_id": ("accepted", "refused in window 0")}
    assert c["relabel"]["windows"][0] == {"map": "refused: no step computes request 0's prompt position 3"}
    assert c["relabel"]["windows"][1]["reference"] == 0 and c["decode_id"]["windows"][0]["steps"] == [0, 1]
    assert run.edits["relabel"] == {"step": 0, "slots": [0, 2], "rows": [[0, 0, 0, 0, 4], [2, 2, 0, 0, 3]]}
