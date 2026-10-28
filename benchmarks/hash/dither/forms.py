"""Every dither arm's label, from one function (`form_of`): form=`FORM_A` (provisional) only for a dither-a arm in the
statement's order (-so<D>) with -zy, whose served epilogue commits the per-unit z/y records under round s's root before
coin s, and only on a form that is ruled in (J10-R's units on credit's layout among them, `j10r_part`); form=beside
(unadopted arm) for an arm in that order on a form nobody has ruled in (-ab/-abn's per-step A tables, -ov's rounds
beside the next forward, J10-R's units off credit's layout); form=beside for everything else: dither_a's own arms (outside
the order, or z the row tables, or no z), dither_one's (dither-1s) and dither_fused's (dither, dither-side).  An arm
the table doesn't know is beside.  This module imports nothing, so every arm's module reads the same table.

Each arm's stats, every check document and every gate verdict carry {"form", "form_basis"} from it: dither_one's
and dither_fused's `Window.stats`, `check` and `gate_verdict`; dither_a's and dither_so's through `dither_a.Side.label`."""
from __future__ import annotations

FORM_A = "A (provisional)"
BESIDE = "beside"
UNADOPTED = "beside (unadopted arm)"

# a dither-a arm's z commitment in the -so order: its form and why
Z_FORMS = {"zy": (FORM_A, ("the served epilogue commits the per-unit z/y records under round s's root before coin s; "
                           "the receipts need no z of their own")),
           "row": (BESIDE, ("receipts beside the served GEMM; z committed as the window's row tables, not the "
                            "per-unit records (ambiguous: the same z under the same round, in another leaf shape)")),
           "none": (BESIDE, "receipts beside the served GEMM; no z committed (y only, never credited)")}
OUT_OF_ORDER = ("receipts beside the served GEMM; outside the statement's order: the step's units read a "
                "coin drawn before its A and z/y went under a round")
# a -so arm's parts that no ruling has taken in, each making the arm unadopted
UNADOPTED_PARTS = {"ab": "-ab/-abn: every call's A table built once a step (h1agate's batch), a per-step A commitment not "
                         "ruled in",
                   "ov": "-ov/-ov1/-ovp: round s built beside forward s + 1 from ring copies, a round order not ruled in",
                   "j10r-off": ("DITHER_SO_UNITS=j10r off credit's layout ($DITHER_SO_J10R_M=live, "
                                "$DITHER_UNITS_SKIP_DEAD=1, or a T that isn't a slot): blocks or dead tiles credit doesn't "
                                "derive (its M is the window's slot, a power of two at least 32 rows, every row block "
                                "computed; served_h1's producer refuses skip_dead)")}
# a -so arm's parts a ruling has taken in, each named in its arm's basis
ADOPTED_PARTS = {"j10r": ("DITHER_SO_UNITS=j10r: the units under J10-R's call-scale noise in the verifier's form "
                          "(`credit.SHARED_J10R`: a J10-R unit's public row holds its call's first position and its "
                          "blocks, which credit derives from the window's layout at its slot M; PouwH1Tile_v5's receipt, "
                          "every row block computed)")}
# the arms of the other two families, which aren't `FORM_A`'s form at all
FAMILIES = {"dither-1s": ("dither_one: the served product is H1V3's noised mainloop and z the window's row tables, its "
                          "noise from a coin drawn before its own A and z went under a round (outside the statement's "
                          "order); not form A"),
            "dither-fused": ("dither_fused (dither, dither-side): z committed as the window's row tables (AROW's "
                             "format), not the per-unit z/y records; not form A")}


def j10r_part(rows_mode: str, skip_dead: bool, t: int) -> str:
    """A J10-R arm's part: "j10r" (ADOPTED_PARTS) when its units' layout rows are the window's T ("slot"), T is a slot
    (a power of two, at least 32) and every row tile is computed, else "j10r-off" (UNADOPTED_PARTS)."""
    return "j10r" if rows_mode == "slot" and not skip_dead and t >= 32 and t & (t - 1) == 0 else "j10r-off"


def form_of(family: str, z: str | None = None, in_order: bool = False, unadopted: tuple[str, ...] = (),
            adopted: tuple[str, ...] = ()) -> dict:
    """An arm's label, {"form", "form_basis"}: `family` one of FAMILIES or "dither-a"; a dither-a arm by its z commitment
    `z` (Z_FORMS), whether it runs in the -so order, the parts of it no ruling has taken in (UNADOPTED_PARTS) and those a
    ruling has (ADOPTED_PARTS, named after its z's basis)."""
    if family in FAMILIES:
        return {"form": BESIDE, "form_basis": FAMILIES[family]}
    if family != "dither-a" or z not in Z_FORMS:
        return {"form": BESIDE, "form_basis": "no dither-a side"}
    if not in_order:
        return {"form": BESIDE, "form_basis": OUT_OF_ORDER}
    if unadopted:
        return {"form": UNADOPTED, "form_basis": "; ".join(UNADOPTED_PARTS[u] for u in unadopted)}
    f, basis = Z_FORMS[z]
    return {"form": f, "form_basis": "; ".join((basis, *(ADOPTED_PARTS[a] for a in adopted)))}
