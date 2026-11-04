"""cpw_floor.py, cpw_sass_count.py and a2u_sat_count.py: the DS-120 bound's break line and CP-W's checkpoint
floors, and the two SASS counters on small listings in cuobjdump's form."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1] / "theory"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, THEORY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cf = _load("cpw_floor")
cs = _load("cpw_sass_count")
a2 = _load("a2u_sat_count")

FOLD = ["IMMA.16832.S8.S8 R4, R8, R10, R4 ;", "IMAD R2, R4, 0x3, RZ ;", "LOP3.LUT R3, R2, R5, 0x7, 0x96, !PT ;",
        "IMAD R5, R3, 0x9, R6 ;", "LEA.HI R6, R3, R5, R3, 0x5 ;"]


def _listing(functions: dict[str, list[str]]) -> str:
    """A cuobjdump -sass listing: each function a prologue, its loop (a backward branch to the loop's first address), EXIT."""
    out = ["\tcode for sm_120a", "\t\tarch = sm_120a"]
    for name, body in functions.items():
        out.append(f"\t\tFunction : {name}")
        out.append("        /*0000*/                   LDC R1, c[0x0][0x37c] ;")
        addr = 0x10
        for ins in body:
            out.append(f"        /*{addr:04x}*/                   {ins}")
            addr += 0x10
        out.append(f"        /*{addr:04x}*/                   BRA.U UP0, 0x10 ;")
        out.append(f"        /*{addr + 0x10:04x}*/                   EXIT ;")
    return "\n".join(out) + "\n"


def test_bound_is_the_lean_headline_and_breaks_a_hair_under_the_floor():
    _, i, w, phi = cf.headline()
    c = cf.issue_price()
    assert 0 < c <= 1 and (w, phi) == (c / 128, c / 8) and cf.lean_floor() == 17 * c
    b = cf.bound()
    assert b["eps_folded"] < b["eps_unfolded"] <= b["eps_stated"] == float(i.fam.eps)
    assert b["eps_unfolded"] == float(1 - cf.lean_floor() / i.B * (1 - cf.Fraction(41, i.checks)))
    assert 0 < b["break_slack"] < 1e-3


def test_checkpoint_floors_clear_the_break_line():
    c = cf.checkpoint()
    assert (c["state_words_a_lane"], c["lower_bound_issue_merged"], c["lower_bound_egress"]) == (16, 8, 2)
    routes = c["routes_slots_a_set_warp"]
    assert routes["pair2 (6 LOP3, 2 REDG.E.XOR.64 into one pair)"] == c["lower_bound_issue_merged"]
    assert routes["st256 (2 STG.E.ENL2.256; TMA XOR tree amortized)"] == c["lower_bound_egress"]
    p = cf.issue_price()
    assert (c["floor_egress"], c["floor_issue_merged"]) == (float(17 * p + p / 4), float(18 * p))
    assert c["floor_issue_merged"] > c["floor_egress"] > float(cf.break_line())


def test_pricing_the_checkpoint_in_both_holds_the_bound_only_at_equal_routes():
    e = cf.checkpoint()["eps_priced"]
    unfolded = cf.bound()["eps_unfolded"]
    assert e["floor issue-merged, flat pair2"] <= unfolded and e["floor egress, flat egress"] <= unfolded
    assert e["floor egress, flat pair2"] > unfolded + 0.03
    assert cf.priced(cf.Fraction(0), cf.Fraction(0)) == cf.headline()[1].unfolded


def test_pair_route_is_the_warp_checkpoint():
    assert cf.pair_route_is_warp_checkpoint()


def test_pair_route_check_refuses_a_lane_short(monkeypatch):
    """The comparison can fail: with one lane's state left out of the XOR it does."""
    real = cf.SG.warp_checkpoint
    monkeypatch.setattr(cf.SG, "warp_checkpoint", lambda states: real(states[1:]))
    assert not cf.pair_route_is_warp_checkpoint()


def test_cpw_sass_count_loop_and_beyond_fold(tmp_path):
    p = tmp_path / "cpw.sass"
    p.write_text(_listing({"_Z3cpwILi0EEvPKjS1_PjS2_Pxij": FOLD,
                           "_Z3cpwILi6EEvPKjS1_PjS2_Pxij": FOLD + ["LOP3.LUT R7, R5, R6, R3, 0x96, !PT ;",
                                                                   "REDG.E.XOR.64.STRONG.GPU desc[UR6][R40.64], R6 ;"]}),
                 encoding="utf-8")
    r = cs.count(str(p))["sm_120a"]
    assert r["cpw_none"]["body_per_set"] == len(FOLD) + 1
    assert r["cpw_none"]["classes_per_set"] == {"IMAD": 2, "ALU": 2, "IMMA": 1, "BRA": 1}
    assert r["cpw_pair2"]["beyond_fold_per_set"] == 2
    assert r["cpw_pair2"]["opcodes"]["REDG.E.XOR.64.STRONG.GPU"] == 1


def test_a2u_sat_count_rates(tmp_path):
    p = tmp_path / "a2u.sass"
    stream = ["FFMA R3, R3, R2, 1 ;"] * 4
    p.write_text(_listing({"s_16_0": stream + ["UIADD3 UR4, UPT, UPT, UR4, 0x1, URZ ;"],
                           "u_16_64_2": stream + ["UIMAD UR4, UR4, -0x61c8864f, UR5 ;",
                                                  "ULOP3.LUT UR5, UR5, 0x1b6d1c8d, UR4, 0x96, !UPT ;"],
                           "t_16_2": stream + ["IMMA.16832.S8.S8 R4, R8, R10, R4 ;"]}), encoding="utf-8")
    counts = a2.count(str(p))["sm_120a"]
    assert (counts["u_16_64_2"]["body"], counts["u_16_64_2"]["uniform"], counts["u_16_64_2"]["vector"]) == (7, 2, 5)
    assert counts["t_16_2"]["imma"] == 1 and counts["s_16_0"]["ffma"] == 4
    dev = tmp_path / "device.txt"
    dev.write_text("device cc=12.0 sms=188\n"
                   "a2u s_16_0 w=8 rep=0 smsp_cyc_per_warp_loop=6.1000 err=no error\n"
                   "a2u s_16_0 w=8 rep=1 smsp_cyc_per_warp_loop=6.0200 err=no error\n"
                   "a2u u_16_64_2 w=1 rep=0 smsp_cyc_per_warp_loop=5.0000 err=no error\n", encoding="utf-8")
    arch, rows = a2.device(str(dev))
    assert arch == "sm_120a"
    r = a2.rates(counts, rows)
    assert r["s_16_0@w8"]["total_ipc"] == round(6 / 6.02, 4) and r["s_16_0@w8"]["a2u_holds"]
    assert r["u_16_64_2@w1"]["total_ipc"] == 1.4 and not r["u_16_64_2@w1"]["a2u_holds"]
