"""A served window's record: its class circuit's units recorded into a `Window` under its run and design, closed with the
strings its caller hands in, and written with its docket. ``window-commit`` (`benchmarks/prove_path`) and the serve's close
(`close_plan.ClosePlan`) both make it here."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.commitments import randomness as RND
from verity.commitments.periods import interface as IF
from verity.protocols.proof.spot.prover import docket as DK

from . import window_rows as WRS

__all__ = ["SALTS_DOMAIN", "commit", "key_of", "place_slots", "window_design"]

SALTS_DOMAIN = "verity/window-commit/salts/v1"


def key_of(run: str) -> RND.Key:
    """A window's salt key, drawn fresh for the one window under ``run``."""
    return RND.derive(os.urandom(64), SALTS_DOMAIN, {"window": run})


def window_design(rule: str, circuit: IF.Circuit) -> dict[str, Any]:
    """The sampling design a window of ``circuit`` is committed under: the one the rate ``rule``, which the window's audit
    states to its stream, names over the circuit's proof blocks (`service.params_sampling_design`), so the window registers
    the design the stream holds it to. A fixed count is refused by name (`service.SamplingDesign`): no stream takes it, and
    the audit would hold the window to another design."""
    from verity.protocols.proof.spot.verifier import service as S

    S.SamplingDesign(rule)
    program, query = circuit.program()
    return dict(S.params_sampling_design(rule, program, query, None)[0])


def place_slots(cls: Any, served: Sequence[str], ingress: Mapping[str, Mapping[str, Any]],
                scan: tuple[Sequence[str | None], Sequence[tuple[Sequence[int], Sequence[int]]]] | None = None,
                ) -> list[tuple[str | None, list[int], int]]:
    """Each slot of a window of class ``cls`` serving the requests of I1 ids ``served``, whose bodies ``ingress`` holds:
    its request (None for an empty slot), its row (occupancy, prompt length, ids) and its ``max_tokens``. Without a
    scan, slot k holds the k-th served request, its row its prompt's. With one, ``scan`` is the scan's starts
    and each slot's row and head (`window_rows.scan_slots`'s): slot k holds the request the scan starts
    there, its row and ``max_tokens`` the scan's. Refused where the scan starts one request in two slots, a served
    request in none, or a request that wasn't served, and where a slot's prompt or ``max_tokens`` isn't its served
    request's."""
    if scan is None:
        placed, heads = [served[k] if k < len(served) else None for k in range(cls.slots)], None
    else:
        placed, heads = list(scan[0]), scan[1]
        named = [s for s in placed if s is not None]
        if len(set(named)) != len(named):
            raise ValueError("the scan starts one request in two slots of the window")
        if unplaced := set(served) - set(named):
            raise ValueError(f"{len(unplaced)} of the window's {len(set(served))} served requests start in none of its "
                             "slots: the scan anchors them in another window")
        if extra := sorted(set(named) - set(served)):
            raise ValueError(f"the scan starts certified requests in the window that it didn't serve: {extra[:4]}")
    out = []
    for k, u in enumerate(placed):
        row, max_tokens = [0] * (cls.prompt + 2), 0
        if heads is not None:
            row, max_tokens = list(heads[k][0]), int(heads[k][1][1])
        if u is not None:
            body = ingress[u]
            ids = [int(x) for x in body["prompt"]]
            want = [1, len(ids), *ids, *[0] * (cls.prompt - len(ids))]
            if heads is None:
                row, max_tokens = want, int(body["max_tokens"])
            elif row[:len(ids) + 2] != want[:len(ids) + 2] or max_tokens != int(body["max_tokens"]):
                raise ValueError(f"slot {k}: the scan's request isn't the served one: its prompt or max_tokens differ")
        out.append((u, row, max_tokens))
    return out


def commit(low: Any, cls: Any, slots: Sequence[Any], values: Mapping[str, Any], run: str, design: Mapping[str, Any],
           cap: int, advice: Mapping[str, Any], out: str | Path, window: int, *,
           strings: Callable[[Any, Any], Any], workers: int, lap: Callable[[str], None],
           key: RND.Key | None = None) -> tuple[Any, Any]:
    """``low``'s window recorded and written to ``out``: its slots' and output rows (`window_rows.Rows`) recorded into a
    `Window` under ``run``, ``design`` and ``cap``, bound to ``advice`` (`interface.advice_binding`) with the registered
    ``values``, closed with ``strings(win, key)`` under ``key``, the one key drawn fresh for the window (`key_of`, here
    where none is given), and written with its docket.
    -> the rows and the window."""
    filled = WRS.Rows(low, slots, values)
    lap("rows")
    win = IF.Window(low.circuit, run, design, cap, advice=advice, registered=values, forms=WRS.FORMS)
    win.supply()
    filled.record(win)
    lap("record")
    key = key if key is not None else key_of(run)
    c = win.close(workers=workers, strings=strings(win, key))
    lap("close")
    prov = {"window_circuit": {"class": [cls.slots, cls.prompt, cls.outputs, cls.lanes, cls.every], "window": int(window)}}
    win.write(c, out, prov, registrar=DK)
    lap("write")
    return filled, win
