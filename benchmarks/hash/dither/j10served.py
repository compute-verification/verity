"""J10 banded noise served (measured only, nothing here is served or credited): sbench.py through hb_sbench.py,
plus each credited one-stream arm's twin on J10 banded noise at bands of G column tiles, paired with it in one run. Every
banded arm's gate holds it to credit's banded reference itself (`banded.banded_reference` under the tree
`j10b_ref.ca_tree` finds: send banded-1a4d1de024.tar.gz) and requires it; its check docs carry `j10b_ref.CA_LABEL`.

    dither-1s-hbk-bG     (H200, B200) dither-1s-hbk with h1u.cu's kh_one_peel_band at G in place of kh_one_peel, from the
                         same .so (HB_H1U_KSPLIT): h1j_noise hashes each band's row tiles once, into its leader's
                         workspace region, and every unit's columns. Gated as dither-1s-hbk (hb_peel.check, its tokens
                         equal to exact's on the gate and every rep), with the receipts reference h1r_launch_band at G
                         (j10rtx.BandH1R), the CPU units' rows from their band leader's seed_j (j10b_ref.rows_of: in
                         dither_one.h1v3_unit and hb_peel.peel_cpu), and credit's banded_reference on the same units in
                         place of H1V3's tree (per-unit noise, which a non-leader doesn't use; the reference at G = 1 is
                         it).
    dither-a-hbk*-bG     (H200, B200) a dither-a arm on the hbk unit (option A, labelled by `forms.form_of`: beside) with
                         kh_one_band at G in its side unit's kh_one place, from the same .so. Gated as its base
                         (dither_a.check: commitments, z = int_mm, y, the no-write control and the fault), with the
                         same three swaps in dither_a's check: h1r_launch_band at G, leader rows, credit's banded_reference.
    dither-1s-mix-cpeel-bG[s][p]  (RTX PRO 6000) dither-1s-mix-cpeel on j10rtx.BandWindow (h1o_band_launch; s: block2's
                         rows staged, p: the units as the band-rows pass's programmatic dependents), gated by
                         j10rtx.band_check and, on the gate step, j10rtx.noise_check of the band workspace.

    hb_served.sh run sm_90a --arms dither-1s-hbk,dither-1s-hbk-b64 --gate ...   (with HB_SCRIPT=j10served.py)
    sbench.sh --arms dither-1s-mix-cpeel,dither-1s-mix-cpeel-b64p --gate ...     (with SBENCH_SCRIPT=j10served.py)

sbench's workers import this file as their main module before unpickling `sbench.worker`, so the arms exist there too.
"""
from __future__ import annotations

import ctypes as C
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("HB_DITHER_DIR", str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_sbench as HS  # puts its arms on sbench.build
import j10b_ref as JB
import sbench

_build = sbench.build
HBK, MIX = "dither-1s-hbk", "dither-1s-mix-cpeel"


def band_of(arm: str, base: str) -> tuple[int, str] | None:
    """`BASE-bG[s][p]`: (G, its flags), else None."""
    if not arm.startswith(base + "-b"):
        return None
    tail = arm[len(base) + 2:]
    g = tail.rstrip("sp")
    return (int(g), tail[len(g):]) if g.isdigit() and int(g) >= 1 else None


@contextmanager
def leader_rows(seed_l: bytes):
    """h1tile.noise with its rows from the band leader's seed_j and its columns from the seed it's given."""
    from verity.ml.protocols.pouw.circuit import h1tile as HT
    noise0 = HT.noise
    HT.noise = lambda seed_j, lay: (JB.rows_of(seed_l, lay), JB.cols_of(seed_j, lay))
    try:
        yield
    finally:
        HT.noise = noise0


def layout(win, s: int, g: int):
    """For step s of the window's layout: lead(pos), the position of its unit's band leader, and cell(pos), the banded
    reference's (I, J, g, row_positions) for it."""
    offs = list(win.offs)

    def where(pos: int) -> tuple[int, int, int, int]:
        rel = pos - s * win.u_step
        w = max(i for i in range(win.L) if offs[i] <= rel)
        ncb = win.nk[w][0] // 32
        return (s * win.u_step + offs[w], ncb, *divmod(rel - offs[w], ncb))

    def lead(pos: int) -> int:
        base, ncb, ri, cj = where(pos)
        return base + JB.leader(ri * ncb + cj, ncb, g)

    def cell(pos: int) -> dict:
        base, ncb, ri, cj = where(pos)
        return {"I": ri, "J": cj, "g": g, "row_positions": [base + ri * ncb + j for j in range(ncb)]}

    return lead, cell


class HBBand:
    """`llm.dither_mod` of dither-1s-hbk-bG: hb_peel's check and verdict on banded noise."""
    CG = None

    def __init__(self, g: int):
        import dither_one as DO
        self.g, self.LIB_ENV = g, DO.LIB_ENV

    def check(self, win, lib: str, which: str) -> dict:
        import dither_one as DO
        import hb_peel
        from j10rtx import BandH1R

        from verity.ml.protocols.pouw.circuit import seeds as SD
        lead, cell = layout(win, win.caps.get(which, {}).get("s", 0), self.g)
        unit0, peel0, itself0, peeled0, h1r0 = DO.h1v3_unit, hb_peel.peel_cpu, DO.h1v3_itself, hb_peel.h1v3_peeled, hb_peel.H1R

        def unit(a32, b32, salt, pos, roots):
            with leader_rows(SD.seed(salt, lead(pos), *roots)):
                out = unit0(a32, b32, salt, pos, roots)
            out["band_leader_position"] = lead(pos)
            return out

        def peel(z, a32, b32, salt, pos, roots):
            with leader_rows(SD.seed(salt, lead(pos), *roots)):
                return peel0(z, a32, b32, salt, pos, roots)

        def ca(units: list[dict]) -> dict:
            return JB.ca_reference([{**x, **cell(x["position"])} for x in units])

        BandH1R.G = self.g
        hb_peel.H1R, DO.h1v3_unit, hb_peel.peel_cpu = BandH1R, unit, peel
        DO.h1v3_itself, hb_peel.h1v3_peeled = ca, ca
        try:
            doc = hb_peel.check(win, lib, which)
        finally:
            hb_peel.H1R, DO.h1v3_unit, hb_peel.peel_cpu = h1r0, unit0, peel0
            DO.h1v3_itself, hb_peel.h1v3_peeled = itself0, peeled0
        if which == "gate":
            cu, hp = doc.get("vs_cpu", []), doc.get("h1v3_peel", {})
            doc["ca_reference_ok"] = bool(hp.get("tree") and not hp.get("error") and cu and all(
                x.get("h1v3_peel_z_equal") and x.get("h1v3_peel_receipt_equal") for x in cu))
            doc["ok"] = doc["ok"] and doc["ca_reference_ok"]
            doc["cpu_units_nonleader"] = sum(
                JB.leader(x["unit"], win.nk[x["call"]][0] // 32, self.g) != x["unit"] for x in cu)
        doc["noise"], doc["band"] = JB.CA_LABEL, self.g
        return doc

    def gate_verdict(self, *caps) -> dict:
        import hb_peel
        return hb_peel.gate_verdict(*caps)


class RTXBand:
    """`llm.dither_mod` of dither-1s-mix-cpeel-bG[s][p]: j10rtx's banded check, and its band workspace on the gate step."""
    CG = None

    def __init__(self, win):
        import dither_one as DO
        self.win, self.LIB_ENV, self.rng = win, DO.LIB_ENV, random.Random(0x4A3110)

    def check(self, win, lib: str, which: str) -> dict:
        import j10rtx as JR
        doc = JR.band_check(win, lib, which)
        doc["noise"] = JB.CA_LABEL
        return doc

    def gate_verdict(self, gate: dict, *rest) -> dict:
        import dither_one as DO
        import j10rtx as JR
        v = DO.gate_verdict(gate, *rest)
        v["noise"] = JR.noise_check(self.win, self.rng)
        v["ok"] = v["ok"] and v["noise"]["ok"]
        return v


def da_band_of(arm: str) -> tuple[str, int] | None:
    """`BASE-bG` for a dither-a arm BASE on the hbk unit that commits z: (BASE, G), else None."""
    base, _, g = arm.rpartition("-b")
    unit, z, _ = sbench.DITHER_A.get(base, (None, None, None))
    ok = g.isdigit() and int(g) >= 1 and unit == "hbk" and z != "none"
    return (base, int(g)) if ok else None


class DABand:
    """`llm.dither_mod` of dither-a-hbk*-bG: dither_a's check and verdict on banded noise, as HBBand's for hb_peel."""
    CG = None

    def __init__(self, g: int):
        import dither_a as DA
        self.g, self.LIB_ENV = g, DA.LIB_ENV

    def check(self, win, lib: str, which: str) -> dict:
        import dither_a as DA
        import dither_one as DO
        from j10rtx import BandH1R

        from verity.ml.protocols.pouw.circuit import seeds as SD
        lead, cell = layout(win, win.caps.get(which, {}).get("s", 0), self.g)
        unit0, itself0, h1r0 = DO.h1v3_unit, DO.h1v3_itself, DA.H1R

        def unit(a32, b32, salt, pos, roots):
            with leader_rows(SD.seed(salt, lead(pos), *roots)):
                out = unit0(a32, b32, salt, pos, roots)
            out["band_leader_position"] = lead(pos)
            return out

        def ca(units: list[dict]) -> dict:
            return JB.ca_reference([{**x, **cell(x["position"])} for x in units])

        BandH1R.G = self.g
        DA.H1R, DO.h1v3_unit, DO.h1v3_itself = BandH1R, unit, ca
        try:
            doc = DA.check(win, lib, which)
        finally:
            DA.H1R, DO.h1v3_unit, DO.h1v3_itself = h1r0, unit0, itself0
        if which == "gate":
            hv, cu = doc.get("h1v3", {}), doc.get("vs_h1v3_cpu", [])
            doc["ca_reference_ok"] = bool(hv.get("tree") and not hv.get("error") and hv.get("peeled") and cu and all(
                x.get("h1v3_receipt_equal") and x.get("h1v3_z_equal") for x in cu))
            doc["ok"] = doc["ok"] and doc["ca_reference_ok"]
            doc["cpu_units_nonleader"] = sum(
                JB.leader(x["unit"], win.nk[x["call"]][0] // 32, self.g) != x["unit"] for x in cu)
        doc["noise"], doc["band"] = JB.CA_LABEL, self.g
        return doc

    def gate_verdict(self, *caps) -> dict:
        import dither_a as DA
        return DA.gate_verdict(*caps)


class BandH:
    """A dither_a Side's h1u library with kh_one_band at g (mode 0) in kh_one's place."""

    def __init__(self, h, g: int):
        if not hasattr(h, "kh_one_band"):
            raise RuntimeError("the hbk .so has no kh_one_band: rebuild h1u.cu from this tree")
        h.kh_one_band.argtypes = [*h.kh_one.argtypes[:-1], C.c_uint, C.c_int, C.c_void_p]
        fn, self._h = h.kh_one_band, h
        self.kh_one = lambda *a: fn(*a[:-1], g, 0, a[-1])

    def __getattr__(self, name: str):
        return getattr(self._h, name)


def band_one(base, g: int):
    """hb_one.One with kh_one_peel_band at g (mode 0: the noise launch, then the units) in kh_one_peel's place."""
    class BandOne(base):
        def __init__(self, path: str, win, peel: bool = False):
            super().__init__(path, win, peel)
            h = self.h
            if not peel or not hasattr(h, "kh_one_peel_band"):
                raise RuntimeError(f"{path} has no kh_one_peel_band: rebuild h1u.cu from this tree")
            h.kh_one_peel_band.argtypes = [*h.kh_one.argtypes[:-1], C.c_uint, C.c_int, C.c_void_p]
            fn = h.kh_one_peel_band
            self.fn = lambda *a: fn(*a[:-1], g, 0, a[-1])
    return BandOne


def build(arm: str, scratch, gpu_mem: float):
    hb, rtx, da = band_of(arm, HBK), band_of(arm, MIX), da_band_of(arm)
    if da:
        base, g = da
        if not JB.ca_tree():
            raise RuntimeError(f"{arm}: no banded reference tree: send banded-{JB.CA_SHA}.tar.gz")
        import dither_a as DA

        class BandSide(DA.Side):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.h = BandH(self.h, g)

        DA.Side = BandSide
        llm = _build(base, scratch, gpu_mem)
        llm.dither_mod = DABand(g)
        return llm
    if hb:
        g, flags = hb
        if flags:
            raise ValueError(f"{arm}: {HBK}'s banded twin takes no s or p")
        if not JB.ca_tree():
            raise RuntimeError(f"{arm}: no banded reference tree: send banded-{JB.CA_SHA}.tar.gz")
        import hb_one
        hb_one.One = band_one(hb_one.One, g)
        llm = _build(HBK, scratch, gpu_mem)
        llm.dither_mod = HBBand(g)
        return llm
    if rtx:
        g, flags = rtx
        if not JB.ca_tree():
            raise RuntimeError(f"{arm}: no banded reference tree: send banded-{JB.CA_SHA}.tar.gz")
        import dither_one as DO
        import j10rtx as JR

        class Served(JR.BandWindow):
            def __init__(self, t, key, h1c, h1o, gate, dev):
                super().__init__(t, key, h1c, h1o, gate, dev, g, "s" in flags, "p" in flags)

        DO.DitherState = Served
        llm = _build(MIX, scratch, gpu_mem)
        llm.dither_mod = RTXBand(llm.dither_win)
        return llm
    return _build(arm, scratch, gpu_mem)


def main() -> int:
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else []
    banded = [a for a in arms if band_of(sbench.base_arm(a), HBK) or band_of(sbench.base_arm(a), MIX)
              or da_band_of(sbench.base_arm(a))]
    for a in banded:
        b = sbench.base_arm(a)
        sbench.REF_OF[b] = "exact"
    HS.TOKENS_EVERY_REP = (*HS.TOKENS_EVERY_REP, *(sbench.base_arm(a) for a in banded
                                                   if band_of(sbench.base_arm(a), HBK) or da_band_of(sbench.base_arm(a))))
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "j10served.json").write_text(json.dumps({
        "schema": "kernels-j10/served/v1", "label": JB.CA_LABEL, "ca_tree": JB.ca_tree(), "ca_sha": JB.CA_SHA,
        "arms": arms, "banded_arms": banded}, indent=1) + "\n")
    print(json.dumps({"j10served": {"label": JB.CA_LABEL, "banded_arms": banded, "ca_tree": JB.ca_tree()}}), flush=True)
    return HS.main()


sbench.build = build

if __name__ == "__main__":
    sys.exit(main())
