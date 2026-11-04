"""egress_zk's host side: the execution phase it reads (each bucket's state renders the network certifier's row, its
registration read under the test key) and each member's host read (`egress_zk.refused`, as Lean's registered reads make it, of the
rows the units compute) as the member's: honest accepted with every o0 1 and every row the warden's, lying's changed
unit's row not the warden's, moved's at the other unit's position. Lean's statement setup (`flock-verify statement`,
`Registered.checkComplete` among its checks) holds each staged member to the verifier's registered reads of every
port. Nothing here proves (`egress_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))
from test_warden_zk import needs_lean, reads_of, statement  # noqa: E402


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


def test_the_control_holds_every_held_piece_of_the_states(tmp_path):
    """The instance file lays the ``state`` table's 204-byte rows back to back, so a state's zero-padded last chunk
    isn't there when its tail isn't zeros (the honest member's states 3 to 5): `prove` cuts with ``fields``, which holds
    every other piece, all of them there."""
    mr = ez.member_rows("honest", ez.link(tmp_path))
    states = {f"state {i}": b"".join(w.to_bytes(2, "little") for w in ws) for i, ws in enumerate(mr.words)}
    assert {len(s) for s in states.values()} == {ez.SHAPE.state_bits // 8} and ez.SHAPE.state_bits // 8 % 16
    laid = b"".join(states.values())
    a8 = SimpleNamespace(PIECE=16, needles=lambda p: {x: k for k, x in p.items()}, find=lambda d, t: {})
    kept, held, _ = ez.WZ.LS.public_pieces(a8, states, set(states), b"", fields=True)
    assert held and all(kept[k] in laid for k in held)
    assert any(kept[k] not in laid for k in set(kept) - held)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """Each member's statement, staged as `egress_zk.prove` stages it."""
    tmp = tmp_path_factory.mktemp("egress-zk")
    at, M = ez.link(tmp), ez.WZ._statement()
    sg = ez.stager(M, ez.PARAMS.window, tmp / "stager")
    return at, sg, {name: ez.stage(M, sg, at, name) for name in ez.MEMBERS}


@pytest.mark.slow
@needs_lean
def test_lean_holds_every_port_of_each_members_statement_to_the_verifiers_registered_reads(staged, tmp_path):
    """Lean's statement setup, `Registered.checkComplete` with it: honest passes under the verifier's file, which names
    every port (``o1`` at the warden's root, ``state`` at the member's committed states, ``o0`` at the verifier's own
    known words 1); lying fails at ``o1``'s rows and moved at its positions; honest's ``o0`` fails a verifier whose
    known words are 0, as a unit whose state renders no row would; and honest shaped as before every port was
    registered (only ``o1``) fails for the first port it leaves out. The host reads above only mirror Lean: this holds
    the staged files, and the known content Lean recomputes the root of, to the checks the verifier runs."""
    at, sg, ms = staged
    honest, own = ms["honest"], reads_of(ms["honest"])
    rc, line = statement(honest, tmp_path)
    assert rc == 0, line
    assert sorted(own) == ["o0", "o1", "state"]
    assert {p: f["root"] for p, f in json.loads(line)["registered"].items()} == {p: e["root"] for p, e in own.items()}
    t = ms["lying"]["changed"]["bucket"]
    rc, line = statement(ms["lying"], tmp_path)
    assert rc == 1 and line.startswith(f"setup: registered reads: port o1: table row {t} does not open"), line
    a, _ = ms["moved"]["changed"]["units"]
    rc, line = statement(ms["moved"], tmp_path)
    assert rc == 1 and line.startswith(f"setup: registered reads: port o1: unit {sg.base + a} reads position"), line
    zeros = ez.known_port(ez.known_o0(sg.program_sha512, at.head, [0] * sg.n), sg.base)
    rc, line = statement(honest, tmp_path, ports=own | {"o0": zeros})
    assert rc == 1 and line.startswith("setup: registered reads: port o0: table row 0 does not open"), line
    rc, line = statement(honest, tmp_path, ports={"o1": own["o1"]}, header=("o1",))
    assert rc == 1 and line.startswith("setup: registered reads (complete): the statement's port state is no port"), line
