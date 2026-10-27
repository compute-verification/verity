"""A row of a served H1 window's registered tree, rebuilt from what the prover kept: the served trees a window
saves (`pouw_h1_handoff.suspend`) keep no rows (`pouw_h1_registered.Rows`), so an opening of a drawn proof block's row takes its
words from the prover's directory (`pouw_h1_prover`: each call's A codes, scales and z/y records) and the saved state (each
linear's weight), as `pouw_h1_served.Served` committed them, and holds them to the tree's ``b || c`` at the row.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import records as ZR
from verity_vllm.protocols.pouw.h1 import served as SV


def kept_row(directory: str | Path, state: Mapping[str, Any], calls: Sequence[Mapping[str, Any]], value: str, row: int
             ) -> list[int]:
    """Row `row`'s u16 words of the window's served value `value` (``<prefix>a/<K>@<s>``, ``<prefix>zy@<c>`` or
    ``<prefix>b/<K>``), rebuilt from the prover's `directory` and the saved window's `state` as `Served.commit` laid it
    out: step s's A table of K is its calls of K's rows over the Program's M slots each (`Served.a_words`), call c's z
    table its records over M slots (`pouw_h1_served.record_words`), ``b/<K>`` K's linears' weight rows in turn (the
    state's weights). An empty call kept nothing: zero codes of scale 0, zero records. `calls` are the record's manifest
    rows. ValueError on a value of no such form."""
    _kind, served, _layers = state["served"]
    name = value[len(served["prefix"]):] if value.startswith(served["prefix"]) else ""
    m, linears, order = int(served["m"]), served["linears"], served["calls"]
    ks = {lin: (k, n) for lin, k, n in linears}
    by = {int(c["index"]): c for c in calls}

    def kept(c: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        served_rows = int(by[c]["served"])
        if not served_rows:
            return np.zeros((0, k), np.int8), np.zeros(0, "<f4"), np.zeros((0, 0), np.uint8)
        with np.load(Path(directory) / PV.CALLS / f"{c}.npz") as x:
            return x["a"].reshape(served_rows, k), x["s_a"].reshape(-1), x["records"]

    if g := re.fullmatch(r"a/(\d+)@(\d+)", name):
        k, s, per = int(g[1]), int(g[2]), len(linears)
        mine = [d for d in range(s * per, min((s + 1) * per, len(order))) if ks[order[d][0]][0] == k]
        c, at = mine[row // m], row % m
        codes, scales, _records = kept(c, k)
        full, sc = np.zeros((1, k), np.int8), np.zeros(1, "<f4")
        if at < len(codes):
            full[0], sc[0] = codes[at], scales[at]
        return SV.row_words(full, sc)[0].tolist()
    if g := re.fullmatch(r"zy@(\d+)", name):
        c = int(g[1])
        k, n = ks[order[c][0]]
        _a, _s, records = kept(c, k)
        zv, yv = ZR.split(records, int(by[c]["rows"]), n) if records.size else (np.zeros((0, n), "<i4"),
                                                                                 np.zeros((0, n), np.int16))
        return SV.record_words(zv, yv, m, tuple(served["tile"]))[row].tolist()
    if g := re.fullmatch(r"b/(\d+)", name):
        k, lo = int(g[1]), 0
        for w, (_lin, kw, n) in enumerate(linears):
            if kw != k:
                continue
            if row < lo + n:
                codes, scales = state["weights"][w]
                j = row - lo
                return SV.row_words(codes[j:j + 1], np.asarray(scales, "<f4").reshape(-1)[j:j + 1])[0].tolist()
            lo += n
    raise ValueError(f"{value} row {row}: no kept value rebuilds it (a/<K>@<s>, zy@<c> or b/<K> of the window)")


__all__ = ["kept_row"]
