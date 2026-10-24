"""REDTEAM phase 13 against #3647's port of the I/O tie's negatives to WARDEN's v2 rows API (`negatives.warden_frame`,
stacked on #3618/#3698): the frames negative now flips a byte inside a bucket row's data cell, so I check it is still a
real tamper the verifier must refuse, and not one it refuses for an unrelated reason. It is: the flip keeps a
structurally valid DATA frame (same header: kind, stream, length), so no active-layer or structural check would catch
it, and the verifier refuses it only through the row's content commitment under the warden's signed root. No finding.

The two "unrelated reason" failures I rule out: that the honest row doesn't open either (the negative would be
vacuous), and that the tampered row is refused for being malformed, in the wrong position, or under the wrong salt or
path rather than for its changed content. I show the tampered row is a well-formed row that opens under its own fresh
root at the same position, and that honest and tampered reads differ by exactly the one content byte, same position,
same salt, same path.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier.active import DATA, HEADER, decode_frame

HERE = Path(__file__).resolve().parents[1]


def _warden_run(tmp_path):
    """A warden-serve run driven tick by tick (the harness #3647's own test uses), and its negatives module."""
    import importlib.util
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_negatives_p13", HERE / "negatives.py")
    NG = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = NG
    spec.loader.exec_module(NG)
    WR = pytest.importorskip(NG.WARDEN_ROWS_MODULE)
    sys.path.insert(0, str(HERE / "tests"))
    import test_warden_serve as tws

    run, key = tmp_path / "run", tmp_path / "key"
    run.mkdir()
    tws._run(run, key, [tws.ws.http_request({"prompt": "x" * 10}, "h:1")], [tws._response([" word"] * 8)])
    return run, NG, WR


def _targeted(run, NG, WR):
    """Reconstruct the cell `warden_frame` targets: its egress record, bucket rows, committed cells, the row, and the
    data cell's slot — exactly `warden_frame`'s own reads, so the test and the negative target the same bytes."""
    got = NG.warden_frame(run)
    WS = importlib.import_module("warden_serve")
    link, direction = WS.CERTIFIED[WS.EGRESS]
    dep = json.loads((run / "deployment.json").read_text())
    params = WS.WardenParams(**dep["params"])
    session = bytes.fromhex(dep["session"])
    records = {r.window: r for r in WS.read_records(run, params)[WS.EGRESS]}
    regs = {r["window"]: (r, salts, wiring_salt) for d, salts, wiring_salt in WS.read_cells(run)
            if (r := d.get("registration")) and (r["link"], r["direction"]) == (link, direction.value)}
    w, t, j, B = got["window"], got["row"], got["slot"], got["cell_bytes"]
    _, salts, wiring_salt = regs[w]
    cells = WR.Cells.of(link, direction, records[w], params, session, salts, wiring_salt)
    rows = WR.window_rows(records[w], params)
    return got, WR, cells, rows, (link, direction, w, t, j, B, params, session)


def test_phase13_warden_frame_flips_a_payload_byte_of_a_valid_data_frame(tmp_path):
    """The negative flips the first payload byte of a bucket row's data cell. The cell is a DATA frame, and the flip
    leaves its header (kind, stream, length) untouched, so the tampered cell still decodes as a valid DATA frame with
    the same length and only one payload byte changed: a real content tamper, not a structural one an active-layer check
    would catch first."""
    run, NG, WR = _warden_run(tmp_path)
    got, WR, cells, rows, (link, direction, w, t, j, B, params, session) = _targeted(run, NG, WR)
    assert got["ok"] and got["honest_reads"] and not got["tampered_reads"]

    cell = rows[t][j * B:(j + 1) * B]
    assert cell[0] == DATA, "warden_frame targets a data cell"
    kind, stream, payload = decode_frame(cell)
    assert kind == DATA and len(payload) > 0, "a DATA cell carries payload the flip can change"

    off = HEADER.size  # the offset warden_frame flips by default (the first payload byte)
    assert 0 < off < B
    bad_cell = bytearray(cell)
    bad_cell[off] ^= 1
    bkind, bstream, bpayload = decode_frame(bytes(bad_cell))
    assert (bkind, bstream, len(bpayload)) == (kind, stream, len(payload)), \
        "the header (kind, stream, length) is untouched: a structurally valid DATA frame"
    assert bpayload != payload and bpayload[1:] == payload[1:] and bad_cell[0] == DATA, \
        "exactly one payload byte differs; the kind byte is unchanged"


def test_phase13_the_refusal_is_the_content_binding_not_salt_path_or_malformed(tmp_path):
    """The verifier refuses the tampered row only through the row's content commitment under the warden's signed root.
    Honest and tampered reads are the same position, salt and path and differ by one content byte; the tampered row is
    well formed and opens under its own fresh root at the same position, so it is not refused for being malformed, in
    the wrong position, or under the wrong salt or path."""
    run, NG, WR = _warden_run(tmp_path)
    got, WR, cells, rows, (link, direction, w, t, j, B, params, session) = _targeted(run, NG, WR)

    bad_row = bytearray(rows[t])
    bad_row[j * B + HEADER.size] ^= 1
    assert NG.cell_reads(WR, cells, t, rows[t]), "the honest row opens under the signed root (not vacuous)"
    assert not NG.cell_reads(WR, cells, t, bytes(bad_row)), "the one changed content byte is refused"
    assert NG.cell_reads(WR, cells, t, bytes(bytearray(bad_row[:j * B + HEADER.size]) +
                                             bytes([bad_row[j * B + HEADER.size] ^ 1]) +
                                             bad_row[j * B + HEADER.size + 1:])), \
        "flipping the byte back opens again: the refusal is that one content byte, same salt and path"

    fresh = WR.Cells.commit(link, direction, w, rows[:t] + [bytes(bad_row)] + rows[t + 1:], cells.wiring, params, session)
    assert fresh.root != cells.root
    assert NG.cell_reads(WR, fresh, t, bytes(bad_row)), \
        "the tampered row is well formed and opens under its own fresh root: refused only for differing from the signed one"

    assert not NG.cell_reads(WR, cells, (t + 1) % len(rows), rows[t]), "position binding is separate from content"
