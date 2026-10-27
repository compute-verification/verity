"""lean_headline: the families' headline instances in Lean are the machine model's data, exactly."""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

from verity_machine_model.model import census, model, vector

POUW = Path(__file__).resolve().parents[1]
PROGRAMS = Path(__file__).resolve().parent / "fixtures" / "census_programs.json"
FOLD_BULK = Path(__file__).resolve().parent / "fixtures" / "fold_bulk_loops.json"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("pouw_lean_headline", POUW / "lean_headline.py")
loads = _load("facet_break_loads", POUW / "facet_break" / "loads.py")

#: Bit-exact ways to compute fx1's word step (d' = w·m1, u = y ⊕ d' ⊕ k_x, y' = u·a + z, z' = rotl(u, r) + y'), lane
#: instructions a word by census class. rotl(u, r) is u·2^r mod 2^32 plus ⌊u·2^r/2^32⌋, whose bits don't overlap.
FOLDS = {
    "menu": {"IMAD": 2, "ALU": 2},                       # IMAD, LOP3, IMAD, LEA.HI
    "funnel rotate": {"IMAD": 2, "ALU": 3},              # z' = SHF.L.W(u, u, r), then IADD3 with y'
    "shift rotate": {"IMAD": 3, "ALU": 3},               # IMAD.SHL u·2^r, SHF u >> (32 − r), IADD3 of both and y'
    "IMAD rotate": {"IMAD": 4, "ALU": 1},                # IMAD u·2^r + y', IMAD.HI ⌊u·2^r/2^32⌋ on it
    "wide rotate": {"IMAD": 2, "IMADW": 1, "ALU": 2},    # IMAD.WIDE.U32 u·2^r, IADD3 of its halves and y'
    "wide multiply": {"IMAD": 1, "IMADW": 1, "ALU": 2},  # d' or y' as IMAD.WIDE's low word
    "wide multiplies": {"IMADW": 2, "ALU": 2},
}
UNIFORM_FOLD = {"UNI": 4}                                # the menu's uniform route, warp instructions a word


def lanes(fold: dict[str, int]) -> dict[str, Fraction]:
    return {k: Fraction(v, 32) for k, v in fold.items()}


def beaten(name: str) -> list[tuple[float, str, str, str]]:
    """Each (load, row, run, probe) with a census program (fixtures/census_programs.json) above 1 on a row of the named
    vector, heaviest first. The load is loads.py's over the classes of the program's x alone, so it is a lower bound."""
    m, v = model(), vector(name)
    c = census(m, v.epoch)
    out = []
    for run, probe, x in json.loads(PROGRAMS.read_text())["programs"]:
        out += [(u, r, run, probe) for r, u in loads.facet_loads(x, c, m.cycle_u).items() if r in v.rows and u > 1]
    return sorted(out, reverse=True)


def test_the_committed_instances_are_the_exporters():
    """Security/Definitions/Accounting/Headline/Sm120.lean and Sm120Measured.lean are what lean_headline.py writes from
    the data, π* included."""
    assert H.main(["--check"]) == 0


def test_the_instance_meets_the_headline():
    """The LP's facet is issue and ♭'s bound is its issue count; π* moves 197/3761 of it onto DS_smem, the least weight
    at which the bulk-reduction floors cost N_w φ* (`bulk_cert`), and N_w(w* + φ*) is 17·3564/3761 there. The bound
    Sm120SheetHeadline's proof reads off the identity is below 5.860%: its folded branch at p = 7/8, the receipt
    split's (ln 2 and ln(8/7) by their rational bounds, above the exact logs' value), is below 5.806%, and the unfolded
    one, fold skippers at the whole-unit luck ½, binds. The vector classes are CUDA's lane classes and the 64-bit
    ones."""
    i = H.instance()
    assert i.pistar == "DS_issue" and i.B == H.load(i, "DS_issue", {k: i.nw.get(k, 0) + i.nf.get(k, 0) + i.ns.get(k, 0)
                                                                   for k in {*i.nw, *i.nf, *i.ns}})
    assert i.cert == {"DS_issue": Fraction(3564, 3761), "DS_smem": Fraction(197, 3761)}
    w, phi = H.stars(i, i.cert, i.writers, i.folds)
    assert (i.Nw * w, i.Nw * phi) == (Fraction(3564, 3761), 16 * Fraction(3564, 3761))
    assert H.P == Fraction(7, 8) and H.LN_P == Fraction(1, 8)
    assert i.folded < Fraction(5806, 100000) < i.unfolded == i.bound
    assert H.exact_logs(i) <= float(i.bound) < float(H.EPS) and H.EPS == Fraction(5860, 100000)
    assert set(H.VEC_64) <= set(i.vec) and len(set(i.vec)) == len(i.vec)


def test_the_catch_all_takes_its_issue_slot_alone():
    """A2's catch-all: the census counts MOV in NOROW, and DS-120 prices NOROW at one issue slot and on
    no other row, so an instruction no other class names, counted there, is never free; `instance` refuses a class off
    one slot on DS_issue, so counting it there moves no bound."""
    i = H.instance()
    k = H.CATCH_ALL["sm_120"]
    assert loads.class_of("MOV") == k and k in i.vec
    rows = ["DS_issue", "DS_tensor", "DS_smem", *H.SM120.pipes]
    assert {r: H.coef(i, r, k) for r in rows} == {r: Fraction(r == "DS_issue") for r in rows}


def test_the_measured_instance_meets_its_reading():
    """On VF3s-16 π* is JI3t_0 alone and ♭'s bound is its load there, w* is honest's IMMA16832 at its 128 words, no
    writer kind is the forms' fallback line or UNI64, and the bound Sm120MeasuredHeadline's proof reads off the identity
    at ln(8/7) ≥ 2/15 is below 0.591%, its folded branch binding."""
    i = H.instance()
    m = H.measured(i)
    n = {k: i.nw.get(k, 0) + i.nf.get(k, 0) + m.ns.get(k, 0) for k in {*i.nw, *i.nf, *m.ns}}
    assert H.VECTOR == "VF3s-16" and H.LN_P_M == Fraction(2, 15)
    assert m.pistar == "JI3t_0" and m.B == H.mload(m, "JI3t_0", n) == max(H.mload(m, r, n) for r in m.rows)
    assert m.wstar == "IMMA16832" and dict(m.writers)["IMMA16832"] == 128 and not set(H.OFF_MENU) & set(dict(m.writers))
    assert m.unfolded < m.folded == m.bound
    assert H.exact_logs(m) < float(m.bound) < float(H.EPS_M) and H.EPS_M == Fraction(591, 100000)


def test_no_census_program_beats_the_measured_vector():
    """No program of the census runs VF3s-16's joint issue rows were fitted on loads one of the measured headline's
    vector's rows above 1: a measured program above 1 on a row ran faster than the row allows, and the headline's
    FacetCeilings would price it below what it ran. VF3r-16, model.toml's `current`, is refused: bit-exact programs
    load its π* JI3s_0 at up to 1.0383, and at 1.0107 in honest's IMMA16832 form."""
    assert beaten(H.VECTOR) == []
    old = beaten("VF3r-16")
    at = {(run, probe): u for u, r, run, probe in old if r == "JI3s_0"}
    assert max(at.values()) > 1.038
    assert at[("r20261017-192516-dd11", "jm_i16832_fmuli_q120_p0_a3u20_w8")] > 1.01


def test_bit_exact_folds_meet_fold_opt_at_pi_star():
    """FOLD-OPT at the headline's certificate (`FoldOptFoldedAt`, an open conjecture under Security/lean-audit.json's
    `shape.conjectures`), which a bit-exact fold priced below φ* at π* falsifies: each of `FOLDS` costs at least the
    menu's cheapest route at both headlines' π*, DS-120's issue and DS_smem mix and JI3t_0 on VF3s-16. The menu is not
    cheapest at every certificate, which is why FOLD-OPT at every certificate (`FoldOptFolded`) is not asked: the IMAD
    rotate costs 20 to the menu's 16 on DS_issue, and 14 to its 16 halfway between DS_issue and DS_lop."""
    i = H.instance()
    m = H.measured(i)
    assert i.folds == [lanes(FOLDS["menu"]), UNIFORM_FOLD]
    phi = min(H.price(i, i.cert, f) for f in i.folds)
    phi_m = min(H.mload(m, m.pistar, f) for f in i.folds)
    for name, fold in FOLDS.items():
        assert H.price(i, i.cert, lanes(fold)) >= phi, name
        assert H.mload(m, m.pistar, lanes(fold)) >= phi_m, name
    phi = min(H.load(i, "DS_issue", f) for f in i.folds)
    rotate = lanes(FOLDS["IMAD rotate"])
    assert H.load(i, "DS_issue", rotate) / phi == Fraction(20, 16)
    half = lambda f: (H.load(i, "DS_issue", f) + H.load(i, "DS_lop", f)) / 2  # noqa: E731
    assert half(rotate) / min(map(half, i.folds)) == Fraction(14, 16)


def bulk_fold(i, loop: dict) -> dict[str, Fraction]:
    """A fold_bulk kernel's counts a unit (fixtures/fold_bulk_loops.json): every warp's opcodes over its 8K units and
    thread 0's bodies over the block's, MMA forms and NOP dropped (the word route's), a bulk instruction at its bytes
    (`loads.bulk_counts`, A5b) and an opcode no class names in A2's catch-all."""
    W, units = loop["BT"] // 32, 8 * loop["K"]
    n: dict[str, Fraction] = {}
    for part, share in (("every_warp", Fraction(1, units)), ("body", Fraction(1, units * W))):
        for key, c in loop[part].items():
            op, _, nbytes = key.partition(" ")
            k = loads.class_of(op)
            if op == "NOP" or k in loads.MMA_FORMS:
                continue
            counts = loads.bulk_counts(op, int(nbytes)) if nbytes else {
                k if k in i.classes else H.CATCH_ALL["sm_120"]: Fraction(1)}
            for k, v in counts.items():
                n[k] = n.get(k, Fraction(0)) + c * share * v
    return n


def test_bulk_folds_meet_fold_opt_at_pi_star():
    """fold_bulk.cu's whole-block folds, fx1's fold with its XOR on the bulk engine (bit-exact on sm_120), each cost at
    least N_w φ* a unit at π*. At DS_issue alone, with each bulk reduction one uniform instruction, fbw4096_K4_BT256
    cost 15.7656 against the menu's 16, and its full-work run 0.982 of ♭. A5b counts its 1024 B of shared-memory source
    a unit on DS_smem, 32 SMSP-cycles beside STS.128's 4, and π*'s DS_smem weight prices it above N_w φ*, and so the
    full-work run with any of them above `1 − ε` of ♭. The bulk-reduction floors (`BULK_FLOORS`), each kernel's route at
    its widest forms, cost at least N_w φ*, the STS.128 one exactly: it sets π*'s weight."""
    i = H.instance()
    w, phi = H.stars(i, i.cert, i.writers, i.folds)
    loops = json.loads(FOLD_BULK.read_text())["loops"]
    assert len(loops) == 12
    words = {k: i.nw.get(k, 0) + i.ns.get(k, 0) for k in {*i.nw, *i.ns}}
    for name, loop in loops.items():
        fold = bulk_fold(i, loop)
        assert H.load(i, "DS_smem", fold) >= 32, name
        assert H.price(i, i.cert, fold) >= i.Nw * phi, name
        run = {k: words.get(k, 0) + fold.get(k, 0) for k in {*words, *fold}}
        assert H.price(i, i.cert, run) >= (1 - H.EPS) * i.B, name
    fbw = bulk_fold(i, loops["fbw4096_K4_BT256"])
    old = {**fbw, "UTMABIG": Fraction(0), "UNI": fbw["UNI"] + Fraction(12, 256)}
    assert H.load(i, "DS_issue", old) == Fraction(4036, 256) < 16 == i.Nw * min(H.load(i, "DS_issue", f)
                                                                                for f in i.folds)
    assert H.load(i, "DS_smem", fbw) == 36 and H.price(i, i.cert, fbw) > Fraction(1679, 100)
    floors = {name: H.price(i, i.cert, f) for name, f in H.BULK_FLOORS.items()}
    assert min(floors.values()) == floors["d′ by STS.128"] == i.Nw * phi
    assert all(H.load(i, "DS_issue", f) < 16 for f in H.BULK_FLOORS.values())
