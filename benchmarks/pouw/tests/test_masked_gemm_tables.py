"""masked_gemm's five tables on made-up bench and gate files: an arm's time over its baseline's, the geometric means, the
cheapest route among the arms that passed, and the gate's verdict for an arm (pass, FAIL when one check failed at one gate
shape, unchecked when the gate has no check of it). A table that printed a cost beside a gate it did not read would pass
a wrong kernel's number into a report."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import etable
import lptable
import mtable
import pvtable
import pytest
import stable


def arms(ms: dict[str, float]) -> dict[str, dict]:
    return {name: {"ms": t, "mhz": 2000.0} for name, t in ms.items()}


def run(mod: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture, bench: list[dict], gate: list[dict]) -> list[str]:
    """The table's lines for these bench rows and gate lines, as its command writes them."""
    b, g = tmp_path / "bench.jsonl", tmp_path / "gate.jsonl"
    b.write_text("".join(json.dumps(r) + "\n" for r in bench))
    g.write_text("".join(json.dumps(r) + "\n" for r in gate))
    mod.main(str(b), str(g))
    return capsys.readouterr().out.splitlines()


def cells(lines: list[str], *start: str) -> list[str]:
    """The one row whose first cells are ``start``, as its cells."""
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines if ln.startswith("|")]
    found = [r for r in rows if tuple(r[: len(start)]) == start]
    assert len(found) == 1, (start, found)
    return found[0]


def test_mtable_ratios_digits_and_verdicts(tmp_path, capsys):
    T = mtable
    low = {"8": {"L": [2, 3], "Kc": [0, 0]}}
    ms = {"plain_i8": 2.0, "plain_e4": 1.0, "full_i8": 3.0, "cg_full_i8": 4.0, "cduo32_full_i8": 5.0, "low8_i8": 2.5,
          "lowtc8_i8": 2.2, "low8_e4": 1.5}
    bench = [{"shape": "o", "arms": arms(ms), "low": low},
             {"shape": "down", "arms": arms({k: 4 * v for k, v in ms.items()} | {"full_i8": 24.0}), "low": {"8": {"L": [4, 5]}}}]
    ok = {"digit_overflow_flag": True, "U_plain_i8": True, "U_plain_e4": True, "U_full_i8": True, "U_cg_full_i8": True,
          "U_cduo32_full_i8": True, "R_cduo32_full_i8": True, "R_cduo32_full_i8_nonzero": True, "full_i8_peel": True,
          "full_i8_grid": True, "U_low8_i8": True, "low8_i8_peel": True, "low8_i8_grid": True, "low8_i8_digits_fit": True,
          "U_low8_e4": True, "low8_e4_peel": True, "low8_e4_grid": True, "low8_e4_digits_fit": True, "ok": True}
    out = run(T, tmp_path, capsys, bench, [ok, ok | {"low8_e4_grid": False, "ok": False}])
    # per shape, then the geometric mean over the Llama linears: sqrt(1.5 * 3)
    assert cells(out, "full rank", "i8", "-", "-", "-", "CUDA cores", "cuBLASLt") == [
        "full rank", "i8", "-", "-", "-", "CUDA cores", "cuBLASLt", "1.500", "3.000", "2.121", "pass"]
    assert cells(out, "full rank", "i8", "-", "-", "duo 32")[-4:] == ["2.500", "2.500", "2.500", "pass"]
    # the digits at k = 4096 and 14336, in the arm's format
    assert cells(out, "low rank", "i8", "8", "2/4", "-", "CUDA cores")[-4:] == ["1.250", "1.250", "1.250", "pass"]
    assert cells(out, "low rank", "e4", "8", "3/5")[-4:] == ["1.500", "1.500", "1.500", "FAIL"]       # one check, one shape
    assert cells(out, "low rank", "i8", "8", "2/4", "-", "tensor cores")[-1] == "unchecked"          # the gate never saw it
    # the compact table takes the cheaper of cuBLASLt and the custom mainloop, and has no 4096^3 here
    assert cells(out, "full rank", "i8", "-", "2.00 (nan)")[4] == "2.50 (nan)"                   # cg_full, not full
    assert out[-1] == 'digits l, per r, [i8, e4] at k = 4096 and 14336: {"8": [[2, 3], [4, 5]]}'


def test_etable_groups_and_verdicts(tmp_path, capsys):
    T = etable
    ms = {"plain_i8": 2.0, "plain_e4": 1.0, "cg_i8": 2.4, "fus_cduo32_i8": 3.0, "sep_lt_i8": 2.6}
    row = {"gpu": "a made-up GPU", "sms": 188, "lt_algo": [1, 2], "m": 8, "n": 8, "k": 32, "batch": 1}
    bench = [row | {"shape": s, "arms": arms({k: f * v for k, v in ms.items()})} for s, f in (("qk2k", 1), ("qk8k", 3))]
    bench[1]["arms"]["cg_i8"]["ms"] = 9.6
    bench[1]["skipped"] = ["fov_cg_i8"]
    gate = [{"every_arm_checked": True, "AB_i8_range": True, "AB_e4_range": True, "U_plain_i8": True, "U_plain_e4": True,
             "U_cg_i8": True, "U_fus_cduo32_i8": True, "R_fus_cduo32_i8": True, "noise_keys": True,
             "noise_uniform_on_S": False, "AeBe_i8_range": True, "AeBe_i8_exact_range": True, "receipts": True, "ok": False}]
    out = run(T, tmp_path, capsys, bench, gate)
    assert out[2] == "GPU a made-up GPU, 188 SMs; cuBLASLt algorithms (plain int8, plain E4M3) [1, 2]."
    assert out[3] == ("Gate (eref.py, every arm bit-exact at four small shapes): fus_cduo32_i8 FAIL, sep_lt_i8 unchecked.")
    at = out.index("## attention QK^T (qk2k, qk8k), against cuBLASLt int8 (s8 -> s32), geomean")
    group = out[at:at + 12]
    # sqrt(1.2 * 1.6); an arm of the group that was not timed has no mean; an edited arm rests on the noise's checks
    assert cells(group, "no edit, custom mainloop", "i8") == [
        "no edit, custom mainloop", "i8", "1.39", "nan (untimed)", "nan (untimed)", "nan (untimed)"]
    assert cells(group, "edit, fused, serialized", "i8")[3] == "1.50 (FAIL)"
    assert cells(group, "edit, separate pass + cuBLASLt", "i8")[2] == "1.30"
    assert cells(out, "qk8k", "8", "8", "32", "1", "cg_i8") == [
        "qk8k", "8", "8", "32", "1", "cg_i8", "9.6000", "1.600", "3.200", "2000", "pass"]
    assert cells(out, "qk8k", "", "", "", "", "skipped: fov_cg_i8")


def test_stable_takes_the_cheapest_arm_that_passed(tmp_path, capsys):
    T = stable
    ms = {"plain_i8": 2.0, "plain_e4": 1.0, "gen_pm1": 0.5, "sform_pm1_i8": 1.0, "slt_pm1_i8": 3.0, "fus_pm1_i8": 2.5,
          "sduo_pm1_i8": 4.0}
    row = {"gpu": "a made-up GPU", "sms": 188, "lt_algo": [1, 2], "shape": "sq4096", "m": 8, "n": 8, "k": 32, "batch": 1}
    gate = {"every_arm_checked": True, "AB_i8_range": True, "sigma": True, "noise_pm1_in_S": True, "noise_pm1_uniform": True,
            "AeBe_pm1_i8_range": True, "AeBe_pm1_i8_exact_range": True, "U_plain_i8": True, "U_plain_e4": True,
            "G_gen_pm1": True, "U_slt_pm1_i8": True, "U_fus_pm1_i8": False, "U_sduo_pm1_i8": True, "R_sduo_pm1_i8": True}
    out = run(T, tmp_path, capsys, [row | {"arms": arms(ms)}], [gate])
    assert out[3].endswith("fus_pm1_i8 FAIL, sform_pm1_i8 unchecked.")
    at = out.index("## Against cuBLASLt int8 (s8 -> s32) (alphabets pm1, u17, u17w, u17p, u17wp in each cell)")
    # fus is faster and failed its gate, so slt is the cheapest; the other alphabets were not timed
    assert cells(out[at:at + 5], "sq4096") == [
        "sq4096", "2.0000", "0.250 / nan / nan / nan / nan", "0.500 / nan / nan / nan / nan",
        "1.50 (slt_pm1_i8) / – / – / – / –", "2.00 (sduo_pm1_i8) / – / – / – / –"]
    assert ("- 4096^3: gamma_gen 0.250, nan, nan, nan, nan; no receipts 1.500, nan, nan, nan, nan; "
            "duo 32 2.000, nan, nan, nan, nan") in out[at:at + 12]
    assert cells(out, "sq4096", "8", "8", "32", "1", "fus_pm1_i8")[6:] == ["2.5000", "1.250", "2.500", "2000", "FAIL"]


def test_pvtable_takes_the_noise_free_producer_off_a_producers_arm(tmp_path, capsys):
    T = pvtable
    ms = {"plain": 2.0, "prodN": 1.0, "prodmax": 0.5, "cduo": 3.0, "genP_pm1": 0.4, "genV_pm1": 0.6, "sepS_pm1": 3.0,
          "fusS_pm1": 2.6, "prodSlt_pm1": 3.4, "sepSd_pm1": 5.0}
    row = {"gpu": "a made-up GPU", "sms": 188, "H": 32, "lt_algo": [1], "s": 2048, "arms": arms(ms)}
    gate = {"every_arm_checked": True, "sigma": True, "inputs_range": True, "noise_pm1_in_S": True, "Ps_pm1_range": True,
            "Pt_range": True, "Pt_softmax_within_1": True, "mx_setup": True, "mx_prodmax": True, "U_plain": True,
            "U_prodN": True, "U_prodmax": True, "U_cduo": True, "G_genP_pm1": True, "G_genV_pm1": True, "U_sepS_pm1": True,
            "U_fusS_pm1": False, "U_prodSlt_pm1": True, "XA_prodS_pm1": True, "U_sepSd_pm1": True, "R_sepSd_pm1": True}
    out = run(T, tmp_path, capsys, [row], [gate])
    assert ("## s = 2048 (plain P V 2.0000 ms; noise-free producer prodN 1.0000 ms = 0.50, its row-max pass 0.25; "
            "duo 32 alone 1.50)") in out
    assert cells(out, "gamma_gen, shared (E + V's noise)") == ["gamma_gen, shared (E + V's noise)", "0.500", "–", "–", "–", "–"]
    assert cells(out, "producer, shared E + cuBLASLt")[1] == "1.200"                       # (3.4 - 1.0) / 2.0
    assert cells(out, "shared, fused (serialized)")[1] == "1.300 (FAIL)"
    assert cells(out, "**cheapest, no receipts**")[1] == "1.20 (prodSlt)"                  # fusS is cheaper and failed
    assert cells(out, "**cheapest, duo 32**")[1] == "2.50 (sepSd)"
    assert cells(out, "2048", "prodmax") == ["2048", "prodmax", "0.5000", "0.250", "2000", "pass"]


def test_lptable_routes_parts_and_the_share_over_the_reference(tmp_path, capsys):
    T = lptable
    ms = {"plain": 2.0, "genU": 0.2, "genF": 0.4, "lt_bsep": 3.0, "t_bfus": 2.8, "td_bsep": 4.0, "bonly": 2.4, "ltK": 2.5,
          "tK": 2.6, "cplain": 2.2}
    row = {"gpu": "a made-up GPU", "sms": 188, "shape": "o", "m": 8192, "n": 4096, "k": 4096, "batch": 1, "Kp": 4192, "mx": 2,
           "arms": arms(ms)}
    every = dict.fromkeys(T.GLOBAL, True)
    gate = every | {f"U_{a}": True for a in ms} | {"U_t_bfus": False, "R_td_bsep": True}
    out = run(T, tmp_path, capsys, [row, row | {"shape": "sq4096", "batch": 2}], [gate])
    assert out[3].endswith("t_bfus FAIL.")
    assert cells(out, "m × n × k (× problems)") == ["m × n × k (× problems)", "8192×4096×4096", "8192×4096×4096 ×2", "–"]
    assert cells(out, "gamma_gen (genU + genF)") == ["gamma_gen (genU + genF)", "0.300", "0.300", "0.300"]
    assert cells(out, "cuBLASLt on the extended operands (A' + ext pass), B' and S two passes")[1:] == ["1.500"] * 3
    assert cells(out, "tail peel, A' by a pass, B' and S one pass")[1:] == ["–"] * 3      # failed its gate: no cost shown
    assert cells(out, "**cheapest, no receipts**")[1:] == ["1.500"] * 3
    assert cells(out, "(cheapest, no receipts: route)")[1:] == ["lt_bsep", "lt_bsep", "–"]
    assert cells(out, "U's and the peel's share, no receipts (cheapest − reference)")[1:] == ["0.300", "0.300", "–"]
    assert cells(out, "the tail's increment (tK - cplain)")[1:] == ["0.200", "0.200", "–"]
    assert cells(out, "o", "t_bfus") == ["o", "t_bfus", "2.8000", "1.400", "2000", "FAIL"]
