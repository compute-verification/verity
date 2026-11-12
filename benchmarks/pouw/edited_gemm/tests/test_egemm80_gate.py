"""egemm80.py's SASS gate on made-up disassembly: a k-stage loop that holds each arm's MMAs, loads and fold passes, and
each thing the gate refuses is refused by its own reason."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("egemm80", HERE.parent / "egemm80.py")
E = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = E
_spec.loader.exec_module(E)

ARM_ID = {arm: i for i, arm in E.ARMS.items()}


def name_of(arm: str) -> str:
    return f"_Z7egemm80ILi{ARM_ID[arm]}EEv4Args"


def loop_ops(arm: str) -> list[str]:
    """The opcodes of a k-stage loop the gate passes: the arm's MMAs, 16 LDSM.x4, 4 LDGSTS and, on a committing arm, duo's
    three ops a word with its ties."""
    mma, _, n_mma, n_com, _ = E.SHAPE[arm]
    words = 4 * n_com
    return ([f"{mma}.S8.S8"] * n_mma + ["LDSM.16.M88.4"] * 16 + ["LDGSTS.E.BYPASS.128"] * 4 + ["LEA.HI"] * words
            + ["LOP3.LUT"] * (words + E.TIES[arm]) + ["IMAD"] * words)


def function(arm: str, ops: list[str]) -> str:
    """One function's disassembly: a prologue instruction, then ``ops`` as the body of a backward branch."""
    lines = [f"\t\tFunction : {name_of(arm)}", "        /*0000*/                   MOV R1, c[0x0][0x28] ;"]
    for i, op in enumerate(ops):
        lines.append(f"        /*{0x10 * (i + 1):04x}*/                   {op} R0, R1, R2 ;")
    lines.append(f"        /*{0x10 * (len(ops) + 1):04x}*/                   BRA 0x10 ;")
    return "\n".join(lines) + "\n"


def run_gate(tmp_path: Path, loops: dict[str, list[str]]) -> dict:
    sass, log = tmp_path / "egemm80.sass", tmp_path / "build.log"
    sass.write_text("".join(function(arm, ops) for arm, ops in loops.items()))
    log.write_text("".join(f"ptxas info    : Compiling entry function '{name_of(arm)}' for 'sm_80'\n"
                           f"ptxas info    : Used 40 registers, 8 bytes spill stores, 12 bytes spill loads\n" for arm in loops))
    return E.gate(sass, log)


def test_a_right_loop_passes(tmp_path):
    g = run_gate(tmp_path, {arm: loop_ops(arm) for arm in E.SHAPE})
    assert g["ok"] and set(g["arms"]) == set(E.SHAPE) and "missing" not in g
    for arm, v in g["arms"].items():
        mma, _, n_mma, n_com, _ = E.SHAPE[arm]
        assert v["refused"] == [] and v["function"] == name_of(arm) and v["spill_bytes"] == [8, 12]
        assert (v["loop_mma"], v["loop_ldsm4"], v["loop_ldgsts"], v["local"]) == (n_mma, 16, 4, 0)
        assert v["loop_lea_hi"] == v["loop_imad"] == 4 * n_com and v["loop_lop3"] == 4 * n_com + E.TIES[arm]
        assert v["loop_insns"] == len(loop_ops(arm)) + 1 and v["function_counts"]["MOV"] == 1 and "MOV" not in v["loop_counts"]


@pytest.mark.parametrize("change, why", [
    (lambda ops: ops[1:], "31 IMMA.16832 in the loop, not 32"),
    (lambda ops: [op for op in ops if not op.startswith("LDSM")] + ["LDSM.16.M88.4"] * 15, "15 LDSM.16.M88.4, not 16"),
    (lambda ops: [*ops, "LDGSTS.E.BYPASS.128"], "5 LDGSTS, not 4"),
    (lambda ops: [*ops, "STL"], "1 local-memory instructions in the k-stage loop"),
    (lambda ops: [op for op in ops if op != "IMAD"], "the fold: 136 LOP3, 0 IMAD, 128 LEA.HI for 128 words"),
])
def test_each_refusal_names_its_reason(tmp_path, change, why):
    loops = {arm: loop_ops(arm) for arm in E.SHAPE}
    loops["e32"] = change(loops["e32"])
    g = run_gate(tmp_path, loops)
    assert not g["ok"] and g["arms"]["e32"]["refused"] == [why]
    assert all(v["refused"] == [] for arm, v in g["arms"].items() if arm != "e32")


def test_a_missing_arm_fails_the_gate(tmp_path):
    g = run_gate(tmp_path, {arm: loop_ops(arm) for arm in E.SHAPE if arm != "e64"})
    assert not g["ok"] and g["missing"] == ["e64"] and all(v["refused"] == [] for v in g["arms"].values())
