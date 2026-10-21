"""lean_headline_dc: the H200's and B200's headline instances in Lean are the machine model's data, exactly."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_lean_headline_dc", Path(__file__).parents[1] / "lean_headline_dc.py")
H = importlib.util.module_from_spec(spec)
sys.modules["pouw_lean_headline_dc"] = H
spec.loader.exec_module(H)


def test_the_exporter_reads_its_own_trees_machine_model(tmp_path):
    """RT4-SHEET-26: the readings are this tree's machine-model data, and an interpreter that imported another
    checkout's verity_machine_model first gets a refusal, not that tree's numbers."""
    assert Path(H.verity_machine_model.__file__).resolve().parents[1] == H.MODEL
    (tmp_path / "verity_machine_model").mkdir()
    (tmp_path / "verity_machine_model" / "__init__.py").write_text("")
    code = (f"import sys, runpy; sys.path.insert(0, {str(tmp_path)!r}); import verity_machine_model; "
            f"runpy.run_path({str(Path(H.__file__))!r})")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "ImportError: lean_headline_dc reads" in r.stderr, r.stderr[-400:]


def test_the_committed_instances_are_the_exporters():
    """Security/Definitions/Accounting/HeadlineDc/Sm90.lean, Sm90Measured.lean, Sm90MeasuredI3.lean, Sm100.lean and
    Sm100Measured.lean are what lean_headline_dc.py writes from the data, π* included, and every reading's bound is at
    most its target."""
    assert H.main(["--check"]) == 0


def test_each_certificate_is_one_and_prices_the_bound():
    """Each π* is a certificate (weights ≥ 0 summing to 1 exactly), the bound is the larger branch at its prices, χ* is
    0 on every reading (the form `headline_of_branches` proves), and a uniform writer heads 2 words (RT-HOPPER-7). A6's
    vector classes hold both catch-alls, where an instruction no class names counts."""
    for fam in H.FAMILIES:
        for v in (fam.sheet, fam.measured):
            r = H.reading(fam, v)
            lam = r.data.lam
            assert all(x > 0 for x in lam.values()) and sum(lam.values()) == 1
            assert r.data.bound == max(Fraction(r.exact["bound"]), Fraction(0)) and r.bound <= r.target
            assert r.chi == 0 and not r.data.nx
            assert dict(r.data.writers)["UNI"] == 2
            assert H.CATCH_ALLS <= set(r.data.vec), (fam.ns, v, r.data.vec)


def test_the_i3_twin_is_dc_90_ri_at_three_quarters():
    """DC-90-ri's twin (`Sm90MeasuredI3.lean`) reads (I3) `ScalarRoute` at q = q_T = ¾, an assumed premise
    (`accounting-ts-scalar-route`), beside A6's proved ½ floor, not in its place: the measured file's machine, menu,
    routes and ♭, its own π*, under the sheet's 1% where the ½ floor is over it, both at ln(8/7) ≥ 2/15."""
    fam = H.SM90
    assert [f.ns for f in H.FAMILIES if f.twin] == ["Sm90"] and fam.twin == Fraction(3, 4)
    r, t = H.reading(fam, fam.measured), H.reading(fam, fam.measured, q=fam.twin)
    x, xm = t.data, r.data
    assert (x.classes, x.rows, x.vec, x.writers, x.folds, x.B) == (xm.classes, xm.rows, xm.vec, xm.writers, xm.folds, xm.B)
    assert (x.q, xm.q) == (fam.twin, H.Q) and x.ln_p == xm.ln_p == H.LH.LN_P_M
    assert all(v > 0 for v in x.lam.values()) and sum(x.lam.values()) == 1 and x.lam != xm.lam
    assert x.bound == max(Fraction(t.exact["bound"]), Fraction(0)) and t.chi == 0
    assert t.bound <= t.target == 0.01 < r.bound <= r.target


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
    """FOLD-OPT at each statement's π* (`FoldOptFoldedAt`, a conjecture under Security/lean-audit.json's
    `shape.conjectures`), which a fold priced below N_w·φ* at π* falsifies. It is open at all five π*, and on the H200
    a premise only about folds off the menu, which holds every counted fold known (RT4-SHEET-13, -14, -17, -22): at the
    (I3) twin's π* li-bulk's trimmed W = ∞ member is the menu's cheapest. Every class a fold issues prices above 0 at
    each π* (the stores carry the JI90 lines r20261021-161119-62ac measured). ♭'s fold meets it, and so does ♭'s fold
    with each IMAD (its one IMAD-pipe class) replaced by 32 ALU shift-adds, at over ten times N_w·φ*. On the B200 that
    fold is why FOLD-OPT at every certificate (`FoldOptFolded`) is not asked (RT3-LOGIC-6): at 1/20 issue + 19/20 IMAD
    on the sheet and 1/20 I + 19/20 UP on the measured rows it prices below N_w·φ*. On the H200 it falsifies FOLD-OPT at
    no certificate: N_w times a menu route with its multiplies as IMAD.WIDE prices at most it on every row."""
    for fam in H.FAMILIES:
        s = H.reading(fam, fam.sheet).data
        pipe = dict(zip(s.classes, s.sheet["pipe"]))
        measured = [H.reading(fam, fam.measured, q=q).data for q in (H.Q, *([fam.twin] if fam.twin else []))]
        for x, rt3 in ((s, {"DS_issue": Fraction(1, 20), "DS_imad": Fraction(19, 20)}),
                       *((m, {"I": Fraction(1, 20), "UP": Fraction(19, 20)}) for m in measured)):
            assert [k for k in x.nf if pipe.get(k) == "imad"] == ["IMAD"]
            alu = {**x.nf, "IMAD": Fraction(0), "ALU": x.nf["ALU"] + 32 * x.nf["IMAD"]}
            at = {"pi*": x.lam, "rt3": rt3}
            nphi = {c: x.Nw * min(_price(x, lam, f) for f in x.folds) for c, lam in at.items()}
            assert _price(x, x.lam, x.nf) >= nphi["pi*"] and _price(x, x.lam, alu) > 10 * nphi["pi*"], (fam.ns, x.q)
            if fam is H.SM90:
                rows = {r for c in x.c.values() for r in c}
                under = [n for n, f in zip(x.fold_names, x.folds[1:])
                         if all(x.Nw * _price(x, {r: 1}, f) <= _price(x, {r: 1}, alu) for r in rows)]
                assert under and all(n.endswith("-wideh") for n in under), (x.q, under)
            else:
                assert _price(x, rt3, alu) < nphi["rt3"], (fam.ns, x.q)
            free = {k for f in x.folds for k, n in f.items() if n and _price(x, x.lam, {k: Fraction(1)}) <= 0}
            assert not free, (fam.ns, x.q, free)
        if fam.twin:
            t = H.reading(fam, fam.measured, q=fam.twin).data
            nphi = t.Nw * min(_price(t, t.lam, f) for f in t.folds)
            assert [n for n, f in zip(t.fold_names, t.folds[1:]) if t.Nw * _price(t, t.lam, f) == nphi] == ["li-bulk-trim-inf"]



ROUTES = ["cp-seg4", "cp-seg4u", "cp-ats", "cp-ats2", "cp-ats8", "cp-red", "cp-red2", "cp-red8", "cp-red-generic",
          "cp-red2-generic", "cp-red8-generic", "cp-xt"]
#: the H200's counted folds after them: RT4-SHEET-13's and -14's moves of LI on li and each route, then li-bulk's three
COUNTED = [r + m for r in ("li", *ROUTES) for m in ("-rt3", "-wideh", "-rt3-wideh")] + [
    "li-bulk", "li-bulk-bt1024", "li-bulk-trim-inf"]


def test_the_fold_menu_is_every_bit_exact_checkpoint_loop():
    """RT3-DC-1, -4: on both devices the fold menu is the design's fold and every bit-exact checkpoint loop rt3_dc times
    (r20261019-183610-89bd on the H200, r20261019-192247-8771 on the B200), with RT4-SHEET-7's L2 XORs through a generic
    address (ATOM.E, bit-exact on the H200 and the B200, r20261020-031119-3113 and r20261020-031119-013a), and each of
    the four readings' π* at A6's ½ prices each class each of them issues above 0, so no route folds for free at the
    certificate FOLD-OPT is asked at. On the H200 the counted folds come after them, none built: RT4-SHEET-13's and
    -14's moves of LI on li and each route, and li-bulk at its measured cost with its counted BT1024 and trimmed W = ∞
    members (RT4-SHEET-17, -22; r20261021-161119-62ac). Each reading's catch-all is priced at
    its issue slot: the sheet's at DS_issue alone, the measured reading's on the measured issue row (DC-CENSUS-13) and,
    once a run has timed its instructions beside the tensor, the joint-issue lines every timed scalar class carries
    (DC-90-ri: LD.E and IMAD.WIDE, r20261021-070616-d06c)."""
    for fam in H.FAMILIES:
        for v in (fam.sheet, fam.measured):
            x = H.reading(fam, v).data
            assert x.fold_names == ROUTES + (COUNTED if fam is H.SM90 else []), (fam.ns, v)
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
    word on the IGMMA, as the sheet's does, and folds by the generic CP_RED2 and li-bulk's trimmed W = ∞ member alone:
    ATOME32 and REDUX carry the joint-issue lines their runs measured (r20261021-070616-d06c, r20261021-074303-da7f),
    so the uniform and shared-memory checkpoints no longer undercut them. The reading is over 1% (RT-HB-HEAD-1)."""
    r, s = H.reading(H.SM90, "DC-90-ri"), H.reading(H.SM90, "DS-90")
    assert [k for k, _ in r.data.writers] == ["IGMMA_S8_N256", "UNI"] and r.menu == "honest"
    assert 0.01 < r.machine_core <= r.bound
    assert H.reading(H.SM90, "DC-90-ri", full=True).bound > 0.1
    assert r.data.words_mix == s.data.words_mix and "def flatWords" not in H.render_measured(H.SM90, r, s)
    assert re.fullmatch(r"the fold routes' cheapest mix, by share of the words: cp-red2-generic 0\.\d+, "
                        r"li-bulk-trim-inf 0\.\d+", r.data.fold_mix), r.data.fold_mix
