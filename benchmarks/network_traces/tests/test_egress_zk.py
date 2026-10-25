"""egress_zk's host side: the link-window it reads (each bucket's state renders the warden's row, its registration
read under the test key) and each member's host read (`egress_zk.refused`, as Lean's registered reads make it, and
the reference's verdicts) as the member's: honest accepted with every verdict 1, lying's changed unit's verdict 0 on
the warden's row, forged's row the changed state's and not the warden's, moved's at the other unit's position.
Nothing here lowers or proves (`egress_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_egress_zk", HERE / "egress_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ez = _load()


def test_each_members_host_read_is_the_members(tmp_path):
    at = ez.link(tmp_path)
    T, sh = ez.PARAMS.window, ez.SHAPE
    reads = {}
    for name in ez.MEMBERS:
        mr = ez.member_rows(name, at)
        reads[name] = mr
        assert len(mr.rows) == len(mr.words) == T and all(len(w) == sh.state_words for w in mr.words)
        m = {"predicted": ez.refused(at, mr, 3), "verdicts": mr.verdicts}
        assert ez.predicted(name, m), (name, m)
    honest, lying, forged, moved = (reads[k] for k in ("honest", "lying", "forged", "moved"))
    assert honest.verdicts == [1] * T and honest.rows == list(at.cells.rows)
    t = lying.changed["bucket"]
    assert lying.verdicts == [int(i != t) for i in range(T)] and lying.rows == honest.rows
    assert forged.states == lying.states and forged.verdicts == [1] * T and forged.rows[t] != honest.rows[t]
    a, b = moved.changed["units"]
    assert moved.rows[a] == honest.rows[b] and moved.rows[b] == honest.rows[a] != honest.rows[b]
    assert moved.states[a] == honest.states[b] and moved.verdicts == [1] * T
