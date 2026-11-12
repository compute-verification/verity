"""dc/uniform.py's table writer on a stand-in probes directory: each gated binary's source gets its own probes' rows
and nothing else of it changes. The probes themselves need an H200 or a B200 (dc/uni.sh)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

DC = Path(__file__).resolve().parents[1] / "dc"


@pytest.fixture(scope="module")
def uniform():
    spec = importlib.util.spec_from_file_location("dc_uniform_under_test", DC / "uniform.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict(binary: str, probe: str, top: dict[str, int]) -> dict:
    return {"binary": binary, "probe": probe, "verdict": {"loops": [{"top": top}]}}


def test_write_table_gives_each_binary_its_own_rows(uniform, tmp_path, monkeypatch):
    """An alone chain counts its first opcode and every other probe the MMA, the branch is left out, and an IMAD-form
    binary's rows are P_I."""
    monkeypatch.setattr(uniform, "HERE", tmp_path)
    probes = tmp_path / "probes"
    probes.mkdir()
    for binary in ("sm90_uniform", "sm90_imad"):
        (probes / f"{binary}.cu").write_text(
            f'// {binary}\nstatic const OpProbe PROBES[] = {{\n    P_U(old, "X=1", "X"),\n}};\nint main() {{}}\n')
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"verdicts": [
        _verdict("sm90_uniform", "uz_64s", {"ULEA.HI": 32, "BRA": 1, "UIADD3": 1}),
        _verdict("sm90_imad", "ui_0", {"IMAD": 32, "IMMA.16832.S8.S8": 8}),
        _verdict("sm90_uniform", "uh_0", {"LOP3.LUT": 32, "IMMA.16832.S8.S8": 8, "BRA": 1})]}))

    assert uniform.write_table(gate) == [probes / "sm90_uniform.cu", probes / "sm90_imad.cu"]
    assert (probes / "sm90_uniform.cu").read_text() == (
        '// sm90_uniform\nstatic const OpProbe PROBES[] = {\n'
        '    P_U(uz_64s, "ULEA.HI=32,UIADD3=1", "ULEA.HI"),\n'
        '    P_U(uh_0, "LOP3.LUT=32,IMMA.16832.S8.S8=8", "IMMA.16832.S8.S8"),\n};\nint main() {}\n')
    assert (probes / "sm90_imad.cu").read_text() == (
        '// sm90_imad\nstatic const OpProbe PROBES[] = {\n'
        '    P_I(ui_0, "IMAD=32,IMMA.16832.S8.S8=8", "IMMA.16832.S8.S8"),\n};\nint main() {}\n')
