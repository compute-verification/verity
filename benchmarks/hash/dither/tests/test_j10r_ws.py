"""j10r_ws: a J10-R call's workspace as the loaded h1u.cu build sizes it, on stand-in builds whose exports follow
h1u.cu's formulas (kh_j10r_rows_bytes, kh_j10r_jw_bytes, kh_j10r_ws_bytes, kh_j10r_flags, kh_j10r_workspace_bytes)."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import j10r_ws as JWS

DITHER = Path(__file__).resolve().parents[1]

#: the frozen shape's up and down (kraqwxsc's served shapes on the H200) and Llama-3.1-8B's four linears
#: at served_h1's 64-row slot, (M, N, K)
SHAPES = [(m, n, k) for m in (4096, 16384) for n, k in ((65536, 16384), (16384, 65536))] + \
    [(64, n, k) for n, k in ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336))]


def _cb(k: int) -> int:
    return 64 * (k // 512 + 1)


def _rows(jx: bool):
    """h1u.cu's kh_j10r_rows_bytes: JX's X and A_d after the row tiles, or the row tiles alone (kr)."""
    return (lambda m, k: _cb(k) * m + m * k + 4 * (m // 32) * 3 * 8 * 32) if jx else (lambda m, k: _cb(k) * m)


def _build(jw: bool = False, jrc: bool = False, jx: bool = True, query: bool = True, jw_export: bool = True,
           flags: int | None = None, workspace=None) -> SimpleNamespace:
    """A stand-in h1u.cu build: H1U_JW's workspace the rows, then each column tile's 64 bytes, then W' [N, K]."""
    rows = _rows(jx)

    def jw_bytes(m, n, k):
        return rows(m, k) + _cb(k) * n + n * k if jw else rows(m, k)

    def ws_bytes(m, n, k):
        return max(rows(m, k), _cb(k) * (m + n))

    h = SimpleNamespace(kh_j10r_rows_bytes=lambda m, k: rows(m, k))
    if jw_export:
        h.kh_j10r_jw_bytes = jw_bytes
    if jrc:
        h.kh_units_multi_j10rc = lambda *a: 0
        h.kh_j10r_ws_bytes = ws_bytes
    if query:
        h.kh_j10r_flags = lambda: (int(jw) | 2 * int(jrc)) if flags is None else flags
        h.kh_j10r_workspace_bytes = workspace or (lambda m, n, k, rc: (ws_bytes(m, n, k) if jrc else 0) if rc
                                                  else jw_bytes(m, n, k))
    return h


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_an_h1u_jw_build_is_sized_with_its_column_blocks_and_w_prime(m, n, k):
    """kraqwxsc's form (H1U_JW, its flags saying so): the workspace is its rows, then 64 bytes a column tile and W'
    (N K bytes), past kh_j10r_rows_bytes by at least W' (1 GiB at up's 65536 x 16384), the size the build asks for."""
    h = _build(jw=True)
    rows = _rows(True)(m, k)
    assert JWS.flags(h) == JWS.JW
    assert JWS.workspace_bytes(h, m, n, k) == rows + _cb(k) * n + n * k
    assert JWS.workspace_bytes(h, m, n, k) - JWS.rows_bytes(h, m, k) >= n * k


@pytest.mark.parametrize("jx", [True, False])
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_any_other_build_is_sized_by_its_rows(m, n, k, jx):
    """kr, kra and kraq's forms (no H1U_JW, with or without JX): the workspace is kh_j10r_rows_bytes, whatever N."""
    h = _build(jx=jx)
    assert JWS.flags(h) == 0
    assert JWS.workspace_bytes(h, m, n, k) == _rows(jx)(m, k)


def test_a_jrc_build_is_sized_by_its_rows_and_columns_under_rc_and_by_its_rows_otherwise():
    """H1U_JRC: kh_units_multi_j10rc's workspace (rows, then column tiles) under rc, which kh_units_multi_j10r may run
    on too, and its rows for kh_units_multi_j10r alone."""
    h = _build(jrc=True)
    m, n, k = 64, 28672, 4096
    assert JWS.flags(h) == JWS.JRC
    assert JWS.workspace_bytes(h, m, n, k, rc=True) == _cb(k) * (m + n) > _rows(True)(m, k)
    assert JWS.workspace_bytes(h, m, n, k) == _rows(True)(m, k)


def test_builds_from_before_the_flags_query_are_sized_by_what_they_export():
    """A build without kh_j10r_flags: H1U_JW's (patched builds, and the fold before the query) by its
    kh_j10r_jw_bytes, one from before H1U_JW by its rows, and an H1U_JRC build's rc workspace by kh_j10r_ws_bytes."""
    m, n, k = 4096, 65536, 16384
    old_jw = _build(jw=True, query=False)
    assert JWS.flags(old_jw) == JWS.JW
    assert JWS.workspace_bytes(old_jw, m, n, k) == _rows(True)(m, k) + _cb(k) * n + n * k
    for h in (_build(query=False), _build(query=False, jw_export=False)):
        assert JWS.flags(h) == 0 and JWS.workspace_bytes(h, m, n, k) == _rows(True)(m, k)
    old_rc = _build(jrc=True, query=False, jw_export=False)
    assert JWS.flags(old_rc) == JWS.JRC
    assert JWS.workspace_bytes(old_rc, m, n, k, rc=True) == max(_rows(True)(m, k), _cb(k) * (m + n))


@pytest.mark.parametrize("h,rc,why", [
    (_build(), True, "without H1U_JRC"),
    (_build(jw=True), True, "without H1U_JRC"),
    (_build(jw=True, query=False), True, "without H1U_JRC"),
    (_build(flags=4), False, "no loader sizes"),
    (_build(flags=3), False, "both H1U_JW and H1U_JRC"),
    (_build(jw=True, workspace=lambda m, n, k, rc: _rows(True)(m, k)), False, "no more than its rows"),
    (_build(flags=1), False, "no more than its rows"),
    (_build(workspace=lambda m, n, k, rc: _rows(True)(m, k) + 64), False, "not its rows"),
    (_build(jw=True, workspace=lambda m, n, k, rc: _rows(True)(m, k) + n * k), False, "but its kh_j10r_jw_bytes"),
    (_build(workspace=lambda m, n, k, rc: 0), False, "short of its rows"),
    (_build(jw=True, workspace=lambda m, n, k, rc: 64), False, "short of its rows"),
], ids=["rc-plain", "rc-jw", "rc-old-jw", "unknown-flag", "jw-and-jrc", "jw-sized-by-rows", "jw-flag-rows-size",
        "plain-not-rows", "jw-disagrees", "zero", "short"])
def test_a_build_and_its_caller_that_disagree_are_refused_never_sized_short(h, rc, why):
    """Every disagreement raises WorkspaceError instead of a size: rc of a build without H1U_JRC, a flag bit no loader
    sizes, H1U_JW with H1U_JRC, an H1U_JW build's size no larger than its rows (its W' past the buffer), another build's
    not its rows, a size its own kh_j10r_jw_bytes contradicts, and a size short of the rows."""
    with pytest.raises(JWS.WorkspaceError, match=why):
        JWS.workspace_bytes(h, 4096, 65536, 16384, rc=rc)


def test_a_build_with_half_of_the_query_is_refused():
    """kh_j10r_flags without kh_j10r_workspace_bytes, or the other way round, is no h1u.cu build: refused."""
    for drop in ("kh_j10r_flags", "kh_j10r_workspace_bytes"):
        h = _build(jw=True)
        delattr(h, drop)
        with pytest.raises(JWS.WorkspaceError, match="without the other"):
            JWS.workspace_bytes(h, 64, 4096, 4096)


def test_no_loader_in_dither_sizes_a_workspace_itself():
    """Every dither loader that allocates J10-R's workspace (dither_so's tables, unitsbench's MultiJ10R) takes
    j10r_ws.workspace_bytes: none calls a build's kh_j10r_*_bytes, so none sizes an H1U_JW build by its rows. served_h1's
    producer is held to the same in the pouw suite."""
    call = re.compile(r"\bkh_j10r_(?:rows|jw|ws|workspace)_bytes\s*\(")
    hits = [f"{p.name}:{i}" for p in sorted(DITHER.glob("*.py")) if p.name != "j10r_ws.py"
            for i, line in enumerate(p.read_text().splitlines(), 1) if call.search(line)]
    assert hits == []
