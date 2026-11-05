"""A J10-R call's workspace (h1u.cu's `kh_units_multi_j10r` with its _sm and _smc forms, and `kh_units_multi_j10rc`
with ``rc``), sized by the loaded build: its `kh_j10r_workspace_bytes`, which dispatches on the build's own flags
(`kh_j10r_flags`: H1U_JW's column tiles and W' after the rows, H1U_JRC's column tiles). Every loader that allocates
the workspace (dither_so's tables, served_h1's `producer.Units`, unitsbench's `MultiJ10R`, j10r_sm90's `H1uCall`) takes
its size from `workspace_bytes`, so no caller sizes an H1U_JW build by its rows alone and lets its units write past the
buffer.

A build from before `kh_j10r_flags` is sized by its `kh_j10r_jw_bytes` (every H1U_JW build exports it) or, built before
H1U_JW, by its rows (`kh_j10r_rows_bytes`). `WorkspaceError` on any disagreement, never a smaller buffer: a flag this
module doesn't know, a size that contradicts the build's flags or its other sizes, ``rc`` on a build without H1U_JRC.

Stdlib only: served_h1's producer loads it beside dither's loaders.
"""
from __future__ import annotations

import ctypes as C
from typing import Any

#: `kh_j10r_flags`' bits
JW, JRC = 1, 2
KNOWN = JW | JRC
#: the shape a build from before `kh_j10r_flags` is asked whether it is H1U_JW at (its W' makes it exceed its rows)
_PROBE = (32, 32, 512)


class WorkspaceError(ValueError):
    """The loaded h1u.cu build and its caller disagree on a J10-R call's workspace."""


def _fn(h: Any, name: str, argtypes: list, restype: Any) -> Any:
    f = getattr(h, name, None)
    if f is not None:
        f.argtypes, f.restype = argtypes, restype
    return f


def _query(h: Any) -> tuple[Any, Any]:
    u32 = C.c_uint
    fl = _fn(h, "kh_j10r_flags", [], u32)
    ws = _fn(h, "kh_j10r_workspace_bytes", [u32, u32, u32, C.c_int], C.c_ulonglong)
    if (fl is None) != (ws is None):
        raise WorkspaceError("this h1u.cu build exports one of kh_j10r_flags and kh_j10r_workspace_bytes without the "
                             "other: rebuild it from one h1u.cu")
    return fl, ws


def _rows(h: Any) -> Any:
    f = _fn(h, "kh_j10r_rows_bytes", [C.c_uint, C.c_uint], C.c_ulonglong)
    if f is None:
        raise WorkspaceError("this h1u.cu build has no kh_j10r_rows_bytes: no J10-R units")
    return f


def rows_bytes(h: Any, m: int, k: int) -> int:
    """The build's row tiles alone at M rows and contraction K (`kh_j10r_rows_bytes`): what a caller sizing by the rows
    allocated, short of an H1U_JW build's workspace."""
    return int(_rows(h)(int(m), int(k)))


def flags(h: Any) -> int:
    """The build's J10-R flags (`JW`, `JRC`), from its `kh_j10r_flags`; for a build from before it, `JW` where its
    `kh_j10r_jw_bytes` exceeds its rows and `JRC` where it has `kh_units_multi_j10rc`. `WorkspaceError` on a bit this
    module doesn't know (a form whose workspace no caller here sizes)."""
    fl, _ = _query(h)
    if fl is not None:
        f = int(fl())
        if f & ~KNOWN:
            raise WorkspaceError(f"this h1u.cu build's kh_j10r_flags is {f:#x}, with bits {f & ~KNOWN:#x} past H1U_JW "
                                 "(1) and H1U_JRC (2) that no loader sizes: teach j10r_ws.py the form before loading "
                                 "it")
        if f & JW and f & JRC:
            raise WorkspaceError("this h1u.cu build's kh_j10r_flags has both H1U_JW and H1U_JRC, which h1u.cu "
                                 "refuses")
        return f
    jw = _fn(h, "kh_j10r_jw_bytes", [C.c_uint, C.c_uint, C.c_uint], C.c_ulonglong)
    m, n, k = _PROBE
    is_jw = jw is not None and int(jw(m, n, k)) > rows_bytes(h, m, k)
    return (JW if is_jw else 0) | (JRC if getattr(h, "kh_units_multi_j10rc", None) is not None else 0)


def workspace_bytes(h: Any, m: int, n: int, k: int, rc: bool = False) -> int:
    """The bytes the build's units read and write in one call's workspace at M rows, N columns and contraction K:
    `kh_units_multi_j10rc`'s with ``rc`` (which `kh_units_multi_j10r` may run on too), else `kh_units_multi_j10r`'s
    (its rows' blocks, and under H1U_JW its column tiles and W'). Monotone in M: a workspace of T rows serves any
    M <= T. `WorkspaceError` where the build and the request disagree: ``rc`` without H1U_JRC, a size of 0, short of
    the rows, or not its flags' (an H1U_JW build's no larger than its rows, any other's not its rows; its
    `kh_j10r_jw_bytes`, where it has one, not the same)."""
    f = flags(h)
    _, ws = _query(h)
    m, n, k = int(m), int(n), int(k)
    if rc and not f & JRC:
        raise WorkspaceError("a J10-RC workspace asked of an h1u.cu build without H1U_JRC (kh_units_multi_j10rc)")
    rows = rows_bytes(h, m, k)
    jw = _fn(h, "kh_j10r_jw_bytes", [C.c_uint, C.c_uint, C.c_uint], C.c_ulonglong)
    if ws is not None:
        got = int(ws(m, n, k, 1 if rc else 0))
    elif rc:
        old = _fn(h, "kh_j10r_ws_bytes", [C.c_uint, C.c_uint, C.c_uint], C.c_ulonglong)
        if old is None:
            raise WorkspaceError("this H1U_JRC build has no kh_j10r_ws_bytes")
        got = int(old(m, n, k))
    else:
        got = int(jw(m, n, k)) if jw is not None else rows
    what = f"{'J10-RC' if rc else 'J10-R'} workspace at M = {m}, N = {n}, K = {k}"
    if got <= 0 or got < rows:
        raise WorkspaceError(f"this h1u.cu build sizes a {what} at {got} bytes, short of its rows' {rows}")
    if not rc:
        if f & JW and got <= rows:
            raise WorkspaceError(f"this H1U_JW build sizes a {what} at {got} bytes, no more than its rows' {rows}: its "
                                 "W' would be written past the buffer")
        if not f & JW and got != rows:
            raise WorkspaceError(f"this h1u.cu build, without H1U_JW, sizes a {what} at {got} bytes, not its rows' "
                                 f"{rows}")
        if jw is not None and int(jw(m, n, k)) != got:
            raise WorkspaceError(f"this h1u.cu build sizes a {what} at {got} bytes but its kh_j10r_jw_bytes at "
                                 f"{int(jw(m, n, k))}")
    return got
