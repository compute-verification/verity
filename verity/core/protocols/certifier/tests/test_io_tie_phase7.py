"""REDTEAM phase 7, target 1: IO-TIE's v4 (#3566, cursor/io-tie-061a @50dcfb0cc8), the tie rebuilt after my phase-4
finding that cells were checked only against the prover's own table.

v4 drops the prover table. The verifier's map comes from advice v0 (`window_map`), and each data cell's id is held to the
Commit's port opened under the Commit's registered root (`cells_match` + the `port` opening in `reference_parts`). So the
two phase-4 forgeries — a wrong id with a matching table, a fabricated response with a matching table — and the two-sided
swap no longer pass: there is no table to match, and matching the id now means forging the Commit's root.

This file re-runs those three on v4, and checks the two ingress ties N2-ADVICE now rests on: first row = arrival order
(checked in-unit, by holding each data cell's stream to the map's q) and prompt_len = the certified ingress record (NOT
checked here — the named gap, `IoTieRequest`, owned by network-accounting and not yet implemented).

Red-team evidence, not a merge-gate suite. Remaps `verity` to this worktree and borrows the author's harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
try:
    from editables.redirector import RedirectingFinder as _F

    _F.map_module("verity", str(_ROOT / "verity" / "__init__.py"))
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verity.core.protocols.certifier import io_tie as IT  # noqa: E402
import test_network_certifier_io_tie as A  # noqa: E402

SH = A.BIG


def _honest():
    served = A.serve(SH, A.responses(5, SH, count=4), seed=11)
    return served, A.commit(SH, served)


def _data_cells(inst):
    return [(i, IT.parse_cell(c)) for i, c in enumerate(inst.cells) if IT.parse_cell(c) and IT.parse_cell(c).data]


def test_honest_window_is_accepted():
    _, inst = _honest()
    assert IT.reference(inst)


def test_a_wrong_id_is_refused_and_matching_it_forges_the_commit():
    """PHASE-4 CASE 1 on v4. CLAIM (my phase-4 finding): the reference ties cells to a prover table, so a wrong id passes
    with a matching table. v4: the id is held to the Commit's port (`port_id`), opened under the Commit's root; a wrong
    id is refused, and the only way to make it pass is to change the committed port — a different `port_root`, i.e. a
    forged Commit, not the certified value.
    """
    served, inst = _honest()
    i0, p0 = _data_cells(inst)[0]
    bad = list(inst.cells)
    bad[i0] = IT.data_cell(p0.id ^ 1, p0.stream, SH.B)  # flip the id, keep the stream
    assert not IT.reference(A.with_cells(inst, bad))

    # "match" it the phase-4 way: rewrite the committed port so port_id equals the forged id.
    m0 = inst.map[i0]
    sigma, row = divmod(m0.e, SH.P)
    actual_step = inst.steps[sigma]
    forged_ids = list(int.from_bytes(served.rows[actual_step][4 * r:4 * r + 4], "big") for r in range(SH.P))
    forged_ids[row] = p0.id ^ 1
    rows2 = list(served.rows)
    rows2[actual_step] = IT.port_row(forged_ids, SH.P)
    matched = A.commit(SH, served._replace(cells=bad, rows=rows2))
    assert IT.reference(matched)                       # self-consistent, as in phase 4
    assert matched.port_root != inst.port_root         # but it is a DIFFERENT Commit root: forging it, not matching it


def test_a_fabricated_response_is_refused():
    """PHASE-4 CASE 2 on v4. A data cell where the verifier's map (from advice v0) has a pad: no entry names it, so the
    content check refuses it. There is no table to add it to."""
    served, inst = _honest()
    pad = next(i for i, m in enumerate(inst.map) if m is None)
    bad = list(inst.cells)
    bad[pad] = IT.data_cell(123_456, 424_242, SH.B)
    assert not IT.reference(A.with_cells(inst, bad))


def test_a_two_sided_swap_of_ids_is_refused():
    """The two-sided tamper: swap two data cells' ids. The map is the verifier's (from advice v0), so the prover cannot
    swap the routing with the ids; each swapped id then disagrees with its map entry's port id."""
    served, inst = _honest()
    data = _data_cells(inst)
    (i0, p0), (i1, p1) = next((a, b) for a in data for b in data if a[1].id != b[1].id and a[0] != b[0])
    bad = list(inst.cells)
    bad[i0] = IT.data_cell(p1.id, p0.stream, SH.B)
    bad[i1] = IT.data_cell(p0.id, p1.stream, SH.B)
    assert not IT.reference(A.with_cells(inst, bad))


def test_first_row_arrival_tie_is_checked_in_unit():
    """TIE 1 (checked): the unit holds each data cell's stream to the map's q (advice v0's first-row number). A request
    whose cells are on a stream other than its arrival ordinal — a later arrival admitted first — is refused."""
    resp = A.responses(5, SH, count=4)
    q = sorted(resp)[0]
    served = A.serve(SH, resp, seed=11, arrival={q: 999_999})  # put q's cells on a stream the map doesn't name
    assert not IT.reference(A.commit(SH, served))


def test_prompt_len_tie_is_absent_from_the_egress_unit():
    """TIE 2 (the gap): N2-ADVICE now rests on prompt_len being the certified ingress record's. IO-TIE v4's egress unit
    does not — and cannot — constrain it: there is no `IoTieRequest`, and the reference, the Instance and `window_map`
    carry no prompt_len. Until `IoTieRequest` lands, the prompt_len tie that closes phase-6 target 1 is not provided by
    IO-TIE; it exists only in advice v1's compile-unit tie (`derived.tie_fields`).
    """
    assert not hasattr(IT, "IoTieRequest")
    assert "prompt_len" not in [f for f in IT.Instance.__dataclass_fields__]
    assert "prompt_len" not in (IT.window_map.__doc__ or "")
    # the whole egress reference takes only the Instance; no prompt input exists to bind.
    import inspect

    assert list(inspect.signature(IT.reference).parameters) == ["inst"]
