"""egress_zk's host side: the link-window it reads (each bucket's state renders the warden's row, its registration
read under the test key) and each member's host read (`egress_zk.refused`, as Lean's registered reads make it, of the
rows the units compute) as the member's: honest accepted with every o0 1 and every row the warden's, lying's changed
unit's row not the warden's, moved's at the other unit's position.
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
    honest, lying, moved = (reads[k] for k in ("honest", "lying", "moved"))
    assert honest.verdicts == [1] * T and honest.rows == list(at.cells.rows)
    t = lying.changed["bucket"]
    assert lying.verdicts == [1] * T and lying.rows[t] != honest.rows[t]
    assert [r for i, r in enumerate(lying.rows) if i != t] == [r for i, r in enumerate(honest.rows) if i != t]
    assert ez.refused(at, lying, 3) == f"registered reads: table row {t} does not open the registered root at position {t}"
    a, b = moved.changed["units"]
    assert moved.rows[a] == honest.rows[b] and moved.rows[b] == honest.rows[a] != honest.rows[b]
    assert moved.states[a] == honest.states[b] and moved.verdicts == [1] * T


def test_the_control_holds_every_piece_but_a_rows_packed_tail(tmp_path):
    """The instance file lays the ``state`` table's rows back to back. Every piece the scan cuts of them is there but
    each row's zero-padded last piece, which `packed` names; a nonzero tail, as the honest member's states 3 to 5 end
    in, is not there."""
    mr = ez.member_rows("honest", ez.link(tmp_path))
    states = {f"state {i}": b"".join(w.to_bytes(2, "little") for w in ws) for i, ws in enumerate(mr.words)}
    assert {len(s) % 16 for s in states.values()} == {ez.SHAPE.state_bits // 8 % 16} != {0}
    laid, pieces, tails = b"".join(states.values()), ez.WZ.LS.cut(states, 16), ez.packed(states, 16)
    assert tails and all(x in laid for k, x in pieces.items() if k not in tails)
    assert any(x not in laid for k, x in pieces.items() if k in tails)
    assert ez.packed({"a": bytes(32), "b": bytes(20)}, 16) == {"b @16"}
