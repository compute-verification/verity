"""Red-team phase 15, target 4 (#3783): the egress tie on the warden's own cells refuses the three the lead named — a
PAD that hides data, a cell credited to the wrong connection, and a row read at a position the statement doesn't fix.

All three are already pinned by this tree's tests (`test_each_way_the_cells_differ_from_the_map_and_the_port_is_refused`
has "a covert byte in a pad", "a cell on another request's stream" and "two connections' slots swapped";
`test_the_bucket_port_reads_every_row_whole` has `warden_rows.refusal` refuse a bucket read at a swapped position). These
add sharper, non-redundant variants and confirm the property holds:

- a covert output written into an unheld slot (dead space no connection holds), and a PAD whose header length field
  claims a payload — two covert channels the exact-cell check (`parse_cell`, `expected_cell`) closes;
- a data cell credited to another live connection's slot in the same bucket — the content tie holds each slot to its
  map's stream;
- a whole bucket row read at a swapped or duplicated position — the registered root (`reference_parts`' ``cells_root``,
  which `IoTieBucket` itself doesn't hash, so the registered reads fix it) pins the row order.

The `IoTieBucket` unit never hashes its row; the position is fixed outside the unit, by the registered reads
(`warden_rows.bucket_port`/`refusal`), and in this reference by ``cells_root``. Helpers are the io_tie test's.
"""
from __future__ import annotations

from dataclasses import replace

from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import HEADER, PAD
from verity.core.protocols.certifier.tests.test_network_certifier_io_tie import (BIG, busiest, busy, commit, with_cells)


def _inst():
    link = busy(1)
    return commit(BIG, link, busiest(link))


def _set(cells, i, cell):
    return [*cells[:i], cell, *cells[i + 1:]]


def test_phase15_a_covert_output_written_into_an_unheld_slot_is_refused():
    """A slot no connection holds is `UNHELD` = (0, EMPTY); its only cell is B zero bytes. A prover who writes a real
    output there — a full `DATA` event in dead space — has the warden register it (``cells_root`` holds), but the
    content tie refuses it: its record admits only the all-zero pad, not a data cell."""
    inst = _inst()
    unheld = [i for i in range(inst.shape.N) if inst.map[i] == IT.UNHELD]
    assert unheld
    covert = with_cells(inst, _set(inst.cells, unheld[0], IT.data_cell(12_345, 7, inst.shape.B)))
    assert IT.reference_parts(covert) == {"cells_root": True, "port": True, "content": False}


def test_phase15_a_pad_that_claims_a_payload_in_its_length_field_is_refused():
    """A covert channel in a `PAD`'s header: a pad on q is `HEADER(PAD, q, 0)` then zeros, so its length field is 0. A
    pad whose length field claims a payload (the body still zero) isn't `pad_cell`, so `parse_cell` gives it no reading
    and the content tie refuses it."""
    inst = _inst()
    pads = [i for i in range(inst.shape.N) if inst.map[i].e == IT.EMPTY and inst.map[i].q]
    assert pads
    q = inst.map[pads[0]].q
    forged = HEADER.pack(PAD, q, 5).ljust(inst.shape.B, b"\0")
    assert IT.parse_cell(forged) is None
    tampered = with_cells(inst, _set(inst.cells, pads[0], forged))
    assert IT.reference_parts(tampered) == {"cells_root": True, "port": True, "content": False}


def test_phase15_a_data_cell_credited_to_another_live_connection_is_refused():
    """A cell credited to the wrong connection: one connection's `DATA` cell, on its own stream, placed in a slot the
    map gives to another live connection of the same bucket. The content tie holds each slot to its map's stream, so
    the stream mismatch refuses it; the registered root still holds, since the warden committed what it laid out."""
    inst, R = _inst(), BIG.R
    data = [i for i in range(inst.shape.N) if inst.map[i].e not in (IT.EMPTY, IT.END_ENTRY)]
    i = data[0]
    bucket = i // R
    j = next(k for k in range(bucket * R, bucket * R + R)
             if inst.map[k].q not in (0, inst.map[i].q))
    credited = with_cells(inst, _set(inst.cells, j, inst.cells[i]))
    assert IT.parse_cell(inst.cells[i]).stream == inst.map[i].q != inst.map[j].q
    assert IT.reference_parts(credited) == {"cells_root": True, "port": True, "content": False}


def test_phase15_a_whole_bucket_row_read_at_the_wrong_position_is_refused():
    """A row read at a position the statement doesn't fix. `IoTieBucket` never hashes its row; the row's position is
    fixed outside the unit, by the registered reads here standing as ``cells_root`` (the registered root of the rows in
    order). Swapping two whole bucket rows, or duplicating one over another, while keeping the honest registered root,
    breaks ``cells_root`` — the registered order pins which row each unit reads."""
    inst, R = _inst(), BIG.R
    assert list(inst.cells[0:R]) != list(inst.cells[R:2 * R])
    swapped = list(inst.cells)
    swapped[0:R], swapped[R:2 * R] = inst.cells[R:2 * R], inst.cells[0:R]
    moved = replace(inst, cells=tuple(swapped))
    assert IT.reference_parts(moved)["cells_root"] is False
    dup = replace(inst, cells=tuple(inst.cells[R:2 * R]) + inst.cells[R:])
    assert IT.reference_parts(dup)["cells_root"] is False
