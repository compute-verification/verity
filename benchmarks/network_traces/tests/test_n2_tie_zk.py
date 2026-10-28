"""n2_tie_zk's members at the pinned size class: each bucket's output word by IoTieRequest's reference, and whether the
prover's ``pub`` row is the verifier's own, give the member's verdicts (`EXPECT`): honest tied; carried, IoTieRequest
accepting bucket 4 on an added carried field count whose row isn't the verifier's; forged and split, refused. Each
witness row is whole u16 words, and the scan's blobs are the warden's held cells and the rows' salts, never an empty
slot. ``--jobs`` is 1 unless given and 2 is refused, naming the rule (`n2_zk.ONE_SESSION`). Nothing here lowers or
proves (`n2_tie_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_the_served_class_is_one_window_of_60_buckets_of_one_4096_byte_cell_honest_tied_and_forged_refused():
    sv = _load()
    sv.use("served")
    tie = sv.TS.IngressTie(sv.LINK, sv.PARAMS, sv.SESSION)
    assert tie.key == (4096, 1, 1) and sv.T == 60 and sv.HC.pinned(tie.gates()[0])
    assert list(sv.members()) == list(sv.EXPECT) == ["honest", "forged"] and f"pub-{sv.T}.bin" in sv.RECEIVED
    for name, (link, adv, edit, _) in sv.members().items():
        rd = sv.link_windows(tie, link, adv, edit)
        assert len(rd.windows) == 1 and len(rd.windows[0].words) == 60 and all(rd.windows[0].opens), name
        assert tuple(bool(x.expect) for x in rd.windows) == sv.EXPECT[name], name
    assert tz.T == 4 and tz.B == 512


def test_a_warden_runs_release_is_its_cells_and_its_forged_row_is_the_registered_reads_to_refuse():
    """`warden_link`'s reading of a warden's cells (`_released`) gives the release and prompt lengths the warden's own
    ingress served, and `forge`'s row, an id request of the same prompt length, keeps every output word 1 while the
    registered read refuses it, in its window alone."""
    tie = _tie()
    link = tz.served(tz.SPAN, tz.ARRIVALS, tz.RATES)
    cells = [tz.WR.Cells.of(tz.LINK, tz.Direction.INGRESS, rec, tz.PARAMS, tz.SESSION) for rec in link.records]
    releases, lens = tz._released(tz.PARAMS, cells)
    assert releases == list(link.releases) and lens == tz.SPAN
    real = SimpleNamespace(cells=cells, records=link.records, releases=releases)
    f = tz.forge(real)
    rd = tz.link_windows(tie, real, tz.advice(lens, tz.received(real)), f)
    assert f.window == 0 and rd.windows[0].rows[f.bucket] != cells[0].rows[f.bucket]
    assert [x.words for x in rd.windows] == [[1] * tz.T] * 2 and all(all(x.opens) for x in rd.windows)
    assert "does not open" in rd.windows[0].held and rd.windows[1].held is None
    assert [x.expect for x in rd.windows] == [0, 1]
    honest = tz.link_windows(tie, real, tz.advice(lens, tz.received(real)))
    assert [x.expect for x in honest.windows] == [1, 1] and all(x.held is None for x in honest.windows)


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


def _a8():
    """`a8`'s scan as `_member_scan` calls it: 16-byte pieces found raw or as hex, per file the verifier received, and
    the control holding the held ones."""
    def needles(p):
        return {f: k for k, x in p.items() for f in (x, x.hex().encode(), x.hex().upper().encode())}

    def find(data, t):
        return {k: 1 for nd, k in t.items() if nd in data}

    def received(vdir, control, blobs, held):
        files = {f.name: {"bytes": f.stat().st_size, "found": find(f.read_bytes(), needles(blobs))}
                 for f in sorted(vdir.iterdir())}
        ctl = find(control.read_bytes(), needles(blobs))
        return {"blobs": len(blobs), "files": files, "verifier_bytes": sum(x["bytes"] for x in files.values()),
                "verifier_found": sorted({k for x in files.values() for k in x["found"]}),
                "control": {"blobs_found": len(ctl), "holds_all": held <= set(ctl)}}

    return SimpleNamespace(PIECE=16, needles=needles, find=find, received=received)


def test_a_member_fails_on_what_the_scan_reads_as_leaked_not_on_a_find_across_the_formats_fields(tmp_path):
    """A held cell's piece the verifier received across two elements of a round message (the format's structure, as
    d951's cell headers lay in its proof streams) is found raw but read structured, and the member stays as expected;
    inside one element it is leaked, and the member fails."""
    tie = _tie()
    link, adv, _, _ = tz.members()["honest"]
    rd = tz.link_windows(tie, link, adv)
    for x in rd.windows:
        x.statement = SimpleNamespace(dir=tmp_path / f"w{x.window}")
        x.statement.dir.mkdir()
        (x.statement.dir / f"inst-{tz.T}.bin").write_bytes(b"".join(x.rows))
    sg = SimpleNamespace(circuit=tmp_path / "circuit.txt")
    sg.circuit.write_text("circuit")
    M = SimpleNamespace(a8=_a8())
    blobs, held = tz.blobs_of(rd)
    pieces, held_pieces, _ = tz.Z.public_pieces(M.a8, blobs, held, sg.circuit.read_bytes() + bytes(tz.B))
    key = next(k for k in sorted(held_pieces) if len(set(pieces[k])) > 2)
    piece = pieces[key]
    res = {"windows": [{"as_expected": True}] * len(rd.windows),
           "tie": {"accepted": list(tz.EXPECT["honest"]), "advice": tz.TS.digest(rd.advice), "link": tz.LINK,
                   "registrations": [{}] * len(rd.windows)}}

    def member(body: bytes, out):
        vdir = out / "verifier"
        vdir.mkdir(parents=True)
        msg = bytes([3, 2]) + bytes(6) + struct.pack("<Q", len(body) // 16) + body
        streams = [{"label": "flock-circuit/fast100x2/rep0", "rounds": [{"msg": msg.hex()}]}]
        (vdir / "s-session.json").write_text(json.dumps({"coin_tree": {"key": "00" * 16}, "streams": streams}))
        rec = tz._member_scan(M, sg, rd, out)
        return rec, tz._member_ok("honest", {**res, "received": rec}, rd)

    rec, ok = member(bytes(range(100, 115)) + piece + b"\x01", tmp_path / "across")
    assert rec["verifier_found"] == [key] and rec["structured"] == {key: "across"} and not rec["leaked"]
    assert rec["control"]["holds_all"] and ok
    rec, ok = member(bytes(range(100, 116)) + piece, tmp_path / "inside")
    assert rec["verifier_found"] == rec["leaked"] == [key] and not rec["structured"] and not ok
