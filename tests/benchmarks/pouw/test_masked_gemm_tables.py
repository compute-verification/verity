"""masked_gemm's table (mtable.py, which run.sh writes) on made-up bench and gate files: an arm's time over its
baseline's, the geometric means, the cheapest route among the arms that passed, and the gate's verdict for an arm (pass,
FAIL when one check failed at one gate shape, unchecked when the gate has no check of it). A table that printed a cost
beside a gate it did not read would pass a wrong kernel's number into a report."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import mtable
import pytest


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
