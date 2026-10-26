"""n2_tie_zk's members at the pinned size class: each bucket's output word by IoTieRequest's reference, and whether the
prover's ``pub`` row is the verifier's own, give the member's verdicts (`EXPECT`): honest tied; carried, IoTieRequest
accepting bucket 4 on an added carried field count whose row isn't the verifier's; forged and split, refused. Each
witness row is whole u16 words, and the scan's blobs are the warden's held cells and the rows' salts, never an empty
slot. ``--jobs`` is 1 unless given and 2 is refused, naming the rule (`n2_zk.ONE_SESSION`). Nothing here lowers or
proves (`n2_tie_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_tie_zk", HERE / "n2_tie_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tz = _load()


def _tie():
    return tz.TS.IngressTie(tz.LINK, tz.PARAMS, tz.SESSION)


def test_each_members_references_give_its_verdicts():
    tie = _tie()
    assert tz.HC.pinned(tie.gates()[0]) and list(tz.EXPECT) == list(tz.MEMBERS)
    for name, (link, adv, edit, _) in tz.members().items():
        rd = tz.link_windows(tie, link, adv, edit)
        assert len(rd.windows) == 2 and all(len(x.words) == tz.T for x in rd.windows), name
        assert tuple(bool(x.expect) for x in rd.windows) == tz.EXPECT[name], name
        assert all(len(tie.row(a, r)) % 2 == 0 for x in rd.windows for a in x.args for r in tz.TS.ROWS)
    carried = tz.link_windows(tie, *tz.members()["carried"][:3])
    assert [x.words for x in carried.windows] == [[1] * tz.T] * 2
    assert [x.opens for x in carried.windows] == [[True] * tz.T, [False] + [True] * (tz.T - 1)]
    forged = tz.link_windows(tie, *tz.members()["forged"][:3])
    assert all(x.own is not None and 0 in x.words for x in forged.windows)


def test_the_scan_looks_for_held_cells_and_salts_alone():
    tie = _tie()
    link, adv, _, _ = tz.members()["honest"]
    rd = tz.link_windows(tie, link, adv)
    blobs, held = tz.blobs_of(rd)
    cells = [row[i:i + tz.B] for x in rd.windows for row in x.cells.rows for i in range(0, len(row), tz.B)]
    assert sorted(blobs[k] for k in held) == sorted(c for c in cells if c != bytes(tz.B))
    assert held and sum(1 for k in blobs if k.startswith("salt")) == len(rd.windows) * tz.T


def test_jobs_is_one_firewalled_session_at_a_time(monkeypatch):
    monkeypatch.setattr(tz, "prove", lambda a: a.jobs)
    argv = ["prove", "--out", "o", "--dry"]
    assert tz.main(argv) == 1 and tz.main([*argv, "--jobs", "1"]) == 1
    with pytest.raises(SystemExit, match="outer proofs go through the firewall one at a time"):
        tz.main([*argv, "--jobs", "2"])
