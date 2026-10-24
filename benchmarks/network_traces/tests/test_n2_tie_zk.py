"""n2_tie_zk's members at the pinned size class: each window's output word by IoTieRequest's reference, and whether the
prover's ``pub`` row is the verifier's own, give the member's verdicts (`EXPECT`): honest tied; carried, IoTieRequest
accepting window 1 on an added carried field count whose row isn't the verifier's; forged and split, IoTieRequest
refusing. Each witness row is whole u16 words, and the scan's blobs are the request frames and salts, never a public
cell. Nothing here lowers or proves (`n2_tie_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_tie_zk", HERE / "n2_tie_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tz = _load()


def test_each_members_references_give_its_verdicts():
    tie = tz.TS.IngressTie(tz.LINK, tz.SHAPE, tz.RUN)
    assert tz.HC.pinned(tie.gates()[0]) and list(tz.EXPECT) == list(tz.MEMBERS)
    for name, (cells, link, edit, _) in tz.members().items():
        rd = tz.link_windows(tie, cells, link, edit)
        assert tuple(bool(x.expect) for x in rd.windows) == tz.EXPECT[name], name
        assert all(len(tie.row(x.args, r)) % 2 == 0 for x in rd.windows for r in tz.TS.ROWS)
    carried = tz.link_windows(tie, *tz.members()["carried"][:3])
    assert [x.word for x in carried.windows] == [1, 1] and [x.opens for x in carried.windows] == [True, False]


def test_the_scan_looks_for_request_frames_and_salts_alone():
    tie = tz.TS.IngressTie(tz.LINK, tz.SHAPE, tz.RUN)
    cells, link, _, _ = tz.members()["honest"]
    rd = tz.link_windows(tie, cells, link)
    blobs, held = tz.blobs_of(rd)
    public = {tz.IT.pad_cell(tz.B)} | {tz.Q.own_frame(w, tz.B) for w in range(len(cells))}
    frames = [c for window in cells for c in window if c not in public]
    assert len(frames) == sum(tz.COUNTS) and sorted(blobs[k] for k in held) == sorted(frames)
    assert sum(1 for k in blobs if k.startswith("salt")) == len(cells) * tz.SHAPE.T
