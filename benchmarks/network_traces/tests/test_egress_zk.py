"""egress_zk's host side: the link-window it reads (the encoder's reference rows are the warden's, its registration
read under the test key) and each member's registered read by the host (`egress_zk.refused`, as Lean's registered
reads make it) as the member's: honest accepted, poisoned's row with one slot `POISON`, moved's at the other unit's
position. Nothing here lowers or proves (`egress_zk.prove` does, on a node)."""

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
    B, R = ez.PARAMS.cell_bytes, ez.PARAMS.grid.allocation
    words = (R * 5 * 32 + 16 * ez.EG.ready_words(ez.NR)) // 16
    assert ez.WHY.keys() == {k for k, ok in ez.MEMBERS.items() if not ok}
    for name, accepted in ez.MEMBERS.items():
        mr = ez.member_rows(name, at)
        why = ez.refused(at, mr, 3)
        assert (why is None) == accepted and ez.WHY.get(name, "") in (why or ""), (name, why)
        assert all(len(w) == words for w in mr.wit) and len(mr.rows) == ez.PARAMS.window
    poisoned = ez.member_rows("poisoned", at)
    t, s = poisoned.changed["bucket"], poisoned.changed["slot"]
    diff = [k for k in range(R) if poisoned.rows[t][k * B:(k + 1) * B] != at.cells.rows[t][k * B:(k + 1) * B]]
    assert diff == [s] and poisoned.rows[t][s * B:(s + 1) * B] == bytes([ez.EG.POISON]) * B
    moved = ez.member_rows("moved", at)
    a, b = moved.changed["units"]
    assert moved.rows[a] == at.cells.rows[b] and moved.rows[b] == at.cells.rows[a] != at.cells.rows[b]
