"""lean_headline_dc: the H200's and B200's headline instances in Lean are the machine model's data, exactly."""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_lean_headline_dc", Path(__file__).parents[1] / "lean_headline_dc.py")
H = importlib.util.module_from_spec(spec)
sys.modules["pouw_lean_headline_dc"] = H
spec.loader.exec_module(H)


def test_the_committed_instances_are_the_exporters():
    """Security/Definitions/Accounting/HeadlineDc/Sm90.lean, Sm90Measured.lean, Sm100.lean and Sm100Measured.lean are
    what lean_headline_dc.py writes from the data, π* included, and every reading's bound is at most its target."""
    assert H.main(["--check"]) == 0


def test_each_certificate_is_one_and_prices_the_bound():
    """Each π* is a certificate (weights ≥ 0 summing to 1 exactly), the bound is the larger branch at its prices, χ* is
    0 on every reading (the form `headline_of_branches` proves), and a uniform writer heads 2 words (RT-HOPPER-7)."""
    for fam in H.FAMILIES:
        for v in (fam.sheet, fam.measured):
            r = H.reading(fam, v)
            lam = r.data.lam
            assert all(x > 0 for x in lam.values()) and sum(lam.values()) == 1
            assert r.data.bound == max(Fraction(r.exact["bound"]), Fraction(0)) and r.bound <= r.target
            assert r.chi == 0 and not r.data.nx
            assert dict(r.data.writers)["UNI"] == 2


def test_the_credit_is_what_the_checks_read():
    """RT-LOGIC-12: the design's fold is honest's, the one PouwWarpCheck recomputes (♭ folds by the cheapest mix of it
    and its checkpoint routes, which the check can't tell apart), and a filler is credited only at p = 1 while no check
    reads it."""
    for fam in H.FAMILIES:
        d = H.E.design(fam.design, fam.data)
        assert d.fold == H.E.honest(d.honest, fam.data).fold, fam.design
        assert all(f.p == 1 for f in d.filler), fam.design


def _price(x, lam: dict, n: dict) -> Fraction:
    return sum((w * x.c[k].get(r, Fraction(0)) * q for r, w in lam.items() for k, q in n.items()), Fraction(0))


def test_folds_meet_fold_opt_at_pi_star():
    """FOLD-OPT at each statement's π* (`FoldOptFoldedAt`, an open conjecture under Security/lean-audit.json's
    `shape.conjectures`), which a fold priced below N_w·φ* at π* falsifies: ♭'s fold meets it, and so does ♭'s fold
    with each IMAD (its one IMAD-pipe class) replaced by 32 ALU shift-adds, at over ten times N_w·φ*. That fold is why
    FOLD-OPT at every certificate (`FoldOptFolded`) is not asked (RT3-LOGIC-6): at 1/20 issue + 19/20 IMAD on the
    sheets and 1/20 I + 19/20 UP on the measured rows it prices below N_w·φ* on all four readings."""
    for fam in H.FAMILIES:
        s = H.reading(fam, fam.sheet).data
        pipe = dict(zip(s.classes, s.sheet["pipe"]))
        for x, rt3 in ((s, {"DS_issue": Fraction(1, 20), "DS_imad": Fraction(19, 20)}),
                       (H.reading(fam, fam.measured).data, {"I": Fraction(1, 20), "UP": Fraction(19, 20)})):
            assert [k for k in x.nf if pipe.get(k) == "imad"] == ["IMAD"]
            alu = {**x.nf, "IMAD": Fraction(0), "ALU": x.nf["ALU"] + 32 * x.nf["IMAD"]}
            at = {"pi*": x.lam, "rt3": rt3}
            nphi = {c: x.Nw * min(_price(x, lam, f) for f in x.folds) for c, lam in at.items()}
            assert _price(x, x.lam, x.nf) >= nphi["pi*"] and _price(x, x.lam, alu) > 10 * nphi["pi*"], fam.ns
            assert _price(x, rt3, alu) < nphi["rt3"], fam.ns


ROUTES = ["cp-seg4", "cp-seg4u", "cp-ats", "cp-ats2", "cp-ats8", "cp-red", "cp-red2", "cp-red8", "cp-red-generic",
          "cp-red2-generic", "cp-red8-generic", "cp-xt"]


def test_the_fold_menu_is_every_bit_exact_checkpoint_loop():
    """RT3-DC-1, -4: on both devices the fold menu is the design's fold and every bit-exact checkpoint loop rt3_dc times
    (r20261019-183610-89bd on the H200, r20261019-192247-8771 on the B200), with RT4-SHEET-7's L2 XORs through a generic
    address (ATOM.E, bit-exact on the H200 and the B200, r20261020-031119-3113 and r20261020-031119-013a), and π*
    prices each class each of them issues
    above 0, so no route folds for free at the certificate FOLD-OPT is asked at. Each reading's catch-all is priced at
    its issue slot: the sheet's at DS_issue alone, the measured reading's on the measured issue row (DC-CENSUS-13) and,
    once a run has timed its instructions beside the tensor, the joint-issue lines every timed scalar class carries
    (DC-90-ri: LD.E and IMAD.WIDE, r20261021-070616-d06c)."""
    for fam in H.FAMILIES:
        for v in (fam.sheet, fam.measured):
            x = H.reading(fam, v).data
            assert x.fold_names == ROUTES, (fam.ns, v)
            free = {k for f in x.folds for k, n in f.items() if n and _price(x, x.lam, {k: Fraction(1)}) <= 0}
            assert not free, (fam.ns, v, free)
        k = H.CATCH_ALL[fam.key]
        assert H.reading(fam, fam.sheet).data.c[k] == {"DS_issue": 1}, (fam.ns, k)
        k, c = H.CATCH_ALL[fam.measured], H.reading(fam, fam.measured).data.c
        joint = {r: v for r, v in c[H.SCALAR_REF].items() if H.JOINT_ISSUE.fullmatch(r)}
        assert c[k] in ({"issue": 1}, {"issue": 1, **joint}), (fam.ns, k)
        if fam.measured == "DC-90-ri":
            assert joint and all(c[x] == {"issue": 1, **joint} for x in (k, "ATOME32")), (fam.ns, c[k], c["ATOME32"])


def test_the_h200_measured_menu_is_honests_form():
    """On DC-90-ri the writer kinds are honest's IGMMA_S8_N256 and UNI: every other tensor form carries the scalar
    coefficient there, which leaves the full menu's bound far above 1%. Its ♭ is `setup`'s floor, which heads every
    word on the IGMMA, as the sheet's does, and folds by the generic CP_RED2 and CP_RED8 alone: ATOME32 and REDUX carry
    the joint-issue lines their runs measured (r20261021-070616-d06c, r20261021-074303-da7f), so the uniform and
    shared-memory checkpoints no longer undercut them. The reading is over 1% (RT-HB-HEAD-1)."""
    r, s = H.reading(H.SM90, "DC-90-ri"), H.reading(H.SM90, "DS-90")
    assert [k for k, _ in r.data.writers] == ["IGMMA_S8_N256", "UNI"] and r.menu == "honest"
    assert 0.01 < r.machine_core <= r.bound
    assert H.reading(H.SM90, "DC-90-ri", full=True).bound > 0.1
    assert r.data.words_mix == s.data.words_mix and "def flatWords" not in H.render_measured(H.SM90, r, s)
    assert re.fullmatch(r"the fold routes' cheapest mix, by share of the words: cp-red2-generic 0\.\d+, "
                        r"cp-red8-generic 0\.\d+", r.data.fold_mix), r.data.fold_mix
