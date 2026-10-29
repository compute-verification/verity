"""dither_so's J10-R units form ($DITHER_SO_UNITS=j10r; the verifier's form: each unit's call is its call's
first position and its blocks `core_check.j10r_blocks` at the call's slot M, as credit derives them from the window's
layout, when M is "slot" on a slot T with every row tile computed, `forms.j10r_part`): the gate's references for it,
swapped into dither_a's check as j10served.py swaps banded noise's.

    every capture, the window's end   each receipt against h1r.cu's version 1 under J10-R (`RefH1R`: h1r_launch_j10r, its
                                      own seeds kernel and SHA-512, the call's stream at Layout(M, N, K))
    the gate step                     `j10r_units.call_receipts` (held to J10R's Definition, `h1tile.j10r_reference`, by
                                      its anchor) on every unit of `checked_calls` against the captured receipts, word for
                                      word; dither_a's three CPU units' receipts from it; and those three units by J10R's
                                      own `j10r_reference(stream=True)` on its tree ($DITHER_J10R_TREE, sbench.sh's
                                      J10R_SHA): receipt and z (= A B^T)

M, the calls' layout rows (`h1tile.j10r_blocks`' m), is `rows` of the step's live rows under $DITHER_SO_J10R_M.
"""
from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
import torch

import dither_a as DA
import dither_one as DO
from h1r import H1R
from verity_config.settings import of

settings = of(__file__)
J10R_SHA = "559bc0aa69"
TREE_ENV = "DITHER_J10R_TREE"
LABEL = f"J10-R units (J10R's Definition: h1tile.j10r_reference, PouwH1Tile_v5's receipt, at {J10R_SHA}; the verifier's form)"
M_MODES = {"slot": "each call's A slot, T rows (j10r_blocks' docstring: the call's slot)",
           "live": "the step's live rows rounded up to 32"}
# dither_so's $DITHER_UNITS_SKIP_DEAD: a row tile wholly past the live rows computed in full (unset), or skipped ("1")
DEAD = {False: "computed in full (PouwH1Tile_v5 / v6)",
        True: "skipped: the zero receipt (h1tile's canonical dead tile, $DITHER_UNITS_SKIP_DEAD=1)"}


def rows(mode: str, t: int, m: int) -> int:
    """A call's J10-R layout rows at m live rows of a T-row slot: "slot" T, "live" m rounded up to 32 (at least 32)."""
    if mode == "slot":
        return t
    if mode == "live":
        return min(t, 32 * max(1, -(-m // 32)))
    raise ValueError(f"J10-R's layout rows: one of {tuple(M_MODES)}, not {mode!r}")


def checked_calls(L: int) -> list[int]:
    """The gate step's calls held to j10r_units whole: the first and last layers' four (every linear's shape)."""
    return sorted({*range(min(4, L)), *range(max(0, L - 4), L)})


class RefH1R(H1R):
    """h1r.py's H1R whose version 1 is h1r_launch_j10r at the class's layout rows M; unless SKIP, every call's row tiles
    all live (a dead one's units computed in full)."""
    M = 0
    SKIP = True

    def plan(self, calls, seeds=None, out=None):
        if not self.SKIP:
            calls = [dataclasses.replace(c, live=torch.ones_like(c.live)) for c in calls]
        return super().plan(calls, seeds, out)

    def launch(self, plan, salt, nowrite=False, version=2, step_base=None) -> None:
        if version != 1 or not self.M:
            raise RuntimeError(f"RefH1R: version 1 at a set M only, not version {version} at M = {self.M}")
        self.launch_j10r(plan, salt, self.M, nowrite, step_base)


class Refs:
    """dither_a.check's references under J10-R for capture g of step s at layout rows M (`swapped`), and after it the gate
    step's whole calls against j10r_units (`definition`)."""

    def __init__(self, win, g: dict, M: int, skip: bool = True):
        self.win, self.g, self.M, self.s, self.skip = win, g, M, g["s"], skip
        self.cache: dict[tuple, np.ndarray] = {}
        self.itself0 = DO.h1v3_itself

    def where(self, pos: int) -> tuple[int, int]:
        win = self.win
        rel = pos - self.s * win.u_step
        w = max(i for i in range(win.L) if win.offs[i] <= rel)
        return w, rel - win.offs[w]

    def base(self, w: int) -> int:
        return self.s * self.win.u_step + self.win.offs[w]

    def receipts(self, w: int, salt: bytes, root_a: bytes, root_b: bytes) -> np.ndarray:
        """Call w's receipts (units, 2 sets): each live unit's by j10r_units, and a dead row tile's by j10r_units too, or
        under `skip` the zero receipt (the dead tile, as h1u and h1r then write it)."""
        import j10r_units as JU
        key = (w, salt, root_a, root_b)
        if key not in self.cache:
            ncb = self.win.nk[w][0] // 32
            units = (self.win.t // 32) * ncb
            live = [u for u in range(units) if not self.skip or 32 * (u // ncb) < self.g["m"]]
            b = self.win.wt[w]
            rec = JU.call_receipts(self.g["a"][w][:self.M].to(b.device), b, salt, self.base(w), root_a, root_b, only=live)
            out = np.zeros((units, rec.shape[1]), dtype=np.uint32)
            out[live] = rec
            self.cache[key] = out
        return self.cache[key]

    def unit(self, a32: np.ndarray, b32: np.ndarray, salt: bytes, pos: int, roots: tuple[bytes, bytes]) -> dict:
        """dither_one.h1v3_unit's place: the unit's receipt from j10r_units on its whole call, z today's A B^T."""
        w, u = self.where(pos)
        rec = self.receipts(w, salt, *roots)[u]
        return {"receipt": [int(x) for x in rec], "today": a32.astype(np.int64) @ b32.astype(np.int64).T,
                "j10r_call": self.base(w), "j10r_rows": self.M}

    def itself(self, units: list[dict]) -> dict:
        """dither_one.h1v3_itself's place: J10R's own j10r_reference(stream=True) on its tree, each unit with its call and
        its (ri, cj) at layout rows M, from which the Definition derives its blocks."""
        tree = settings.get(TREE_ENV)
        if not tree:
            return {"tree": None, "sha": J10R_SHA, "error": f"no J10R tree: ${TREE_ENV} (sbench.sh's J10R_SHA={J10R_SHA})"}
        ins = []
        for x in units:
            w, u = self.where(x["position"])
            ncb = self.win.nk[w][0] // 32
            ins.append({**x, "j10r": {"call": self.base(w), "m": self.M, "ri": u // ncb, "cj": u % ncb}})
        saved = {k: settings.raw(k) for k in ("DITHER_H1V3_TREE", "DITHER_H1V3_SHA")}
        settings.put("DITHER_H1V3_TREE", tree)
        settings.put("DITHER_H1V3_SHA", settings.DITHER_J10R_SHA or J10R_SHA)
        try:
            return self.itself0(ins)
        finally:
            for k, v in saved.items():
                if v is None:
                    settings.unset(k)
                else:
                    settings.put(k, v)

    @contextlib.contextmanager
    def swapped(self):
        RefH1R.M, RefH1R.SKIP = self.M, self.skip
        h1r0, unit0, itself0 = DA.H1R, DO.h1v3_unit, DO.h1v3_itself
        self.itself0 = itself0
        DA.H1R, DO.h1v3_unit, DO.h1v3_itself = RefH1R, self.unit, self.itself
        try:
            yield self
        finally:
            DA.H1R, DO.h1v3_unit, DO.h1v3_itself = h1r0, unit0, itself0

    def definition(self, salt: bytes) -> dict:
        """The gate step: every unit of `checked_calls` by j10r_units against the captured receipts, the roots the
        captured A roots (the CPU's whenever dither_a's a_roots_bad is empty) and B roots."""
        win, g = self.win, self.g
        bad, words = {}, 0
        for w in checked_calls(win.L):
            o_, units, nw = win.plan.offs[w]
            ref = self.receipts(w, salt, g["a_roots"][w], g["b_roots"][w])
            got = g["receipts"][o_:o_ + units * nw].view(units, nw).cpu().numpy().view(np.uint32)
            off = int((ref != got).sum())
            words += units * nw
            if off:
                bad[w] = {"words_off": off, "units_off": int((ref != got).any(axis=1).sum())}
        return {"reference": "j10r_units.call_receipts (anchored to h1tile.j10r_reference) on every live unit, a dead row "
                             f"block's units {DEAD[self.skip]}", "calls": checked_calls(win.L), "live_rows": g["m"],
                "words": words, "bad": bad, "ok": not bad and words > 0}


def check(win, lib_h1o: str, which: str, n_cpu_units: int, g: dict, r_round: dict, mode: str, skip: bool = True) -> dict:
    """dither_a.check of capture g under J10-R's references; on the gate step also `Refs.definition`."""
    M = rows(mode, win.t, g["m"])
    refs = Refs(win, g, M, skip)
    with refs.swapped():
        doc = DA.check(win, lib_h1o, which, n_cpu_units, g=g, r_round=r_round)
    j = {"label": LABEL, "rows": M, "rows_mode": mode, "receipts_reference": "h1r.cu's h1r_launch_j10r (version 1)",
         "dead_tiles": DEAD[skip]}
    if which == "gate" and doc.get("complete"):
        with torch.inference_mode():
            j["definition"] = refs.definition(g["salt"])
        hv = doc.get("h1v3", {})
        j["j10r_reference_units"] = {"tree": hv.get("tree"), "sha": hv.get("sha"), "error": hv.get("error"),
                                     "units": len(doc.get("vs_h1v3_cpu", []))}
        j["ok"] = j["definition"]["ok"]
        doc["ok"] = bool(doc["ok"] and j["ok"])
    doc["j10r"] = j
    return doc
