"""The interface `test_pearlplus_lean`'s cases drive, read from the Lean's source without building it: `pouw-pearlplus`'s
options (`PouwPearlPlus/Main.lean`'s `parse`) are the ones the cases' argv passes, each taking a value or not as it does
there; its `over_cap` rules and their default are `verify_run`'s; and the window record's value names it ties a run to
(`PouwPearlPlus/Tie.lean`), why it refuses a version-2 W that registers no weight, a version-1 W that registers one and
a W of another version, W's version, and the name it gives a version-1 W's untied weights, are `verify_run`'s and
`pearl_c.window`'s; and the rows of a class's fixed load and why passes of another load are refused are `verify_run`'s."""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LEAN = ROOT / "verity" / "protocols" / "pouw" / "verifier" / "lean" / "pearlplus" / "PouwPearlPlus"
CASES = Path(__file__).resolve().parent / "test_pearlplus_lean.py"


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


VR = _module("pearl_c_vllm_verify_run", ROOT / "benchmarks" / "pouw" / "pearl_c_vllm" / "verify_run.py")


def _block(text: str, head: str) -> str:
    """The Lean definition starting at line `head`, up to the next blank line."""
    start = text.index(head)
    return text[start:text.index("\n\n", start)]


def test_the_lean_parses_the_options_the_cases_pass():
    parse = _block((LEAN / "Main.lean").read_text(), "def parse ")
    lean = {f: bool(v) for f, v in re.findall(r'\| "(--[a-z-]+)" :: (v :: )?rest', parse)}
    cases = {f: sep == "," for f, sep in re.findall(r'"(--[a-z-]+)"([,\]])', CASES.read_text())}
    assert lean == cases == {"--coins": True, "--draws": True, "--source": True, "--record": True, "--over-cap": True,
                             "--test-instance": False, "--tie": True, "--class": True, "--of-record": False}


def test_the_over_cap_rules_and_default_are_verify_runs():
    check = (LEAN / "Check.lean").read_text()
    names = dict(re.findall(r'\| \.(\w+) => "([a-z]+)"', _block(check, "def OverCap.name ")))
    parsed = {n: c for n, c in re.findall(r'\| "([a-z]+)" => some \.(\w+)', _block(check, "def OverCap.ofName? "))}
    assert set(names.values()) == set(parsed) == set(VR.OVER_CAP)
    assert all(names[c] == n for n, c in parsed.items())
    default = re.search(r"overCap : OverCap := \.(\w+)", _block((LEAN / "Main.lean").read_text(), "structure Args ")).group(1)
    assert names[default] == inspect.signature(VR.verify_passes).parameters["over_cap"].default == VR.UNCREDITED


def test_the_tie_names_are_verify_runs():
    tie = (LEAN / "Tie.lean").read_text()
    prefixes = re.search(r"def TIE_PREFIXES : List String := \[(.*)\]", tie).group(1)
    assert tuple(re.findall(r'"([^"]+)"', prefixes)) == VR.TIE_PREFIXES
    assert sorted(set(re.findall(r's!"(pearl-[ay]/\{s\}/\{c\})"', tie))) == [VR.TIE_A, VR.TIE_Y]
    assert set(re.findall(r's!"(pearl-b/\{w\})"', tie)) == {VR.TIE_B}
    def lit(name: str) -> str:
        return re.search(rf'def {name} : String :=\s*"([^"]*)"', tie).group(1)
    assert (lit("UNTIED_WEIGHTS"), lit("WEIGHTS_UNTIED"), lit("V1_WEIGHTS")) == (VR.UNTIED_WEIGHTS, VR.WEIGHTS_UNTIED,
                                                                                 VR.V1_WEIGHTS)
    assert int(re.search(r"def W_VERSION : Nat := (\d+)", tie).group(1)) == VR.PWR.VERSION
    unknown = re.search(r'def UNKNOWN_VERSION : String :=\s*s!"([^"]*)" \+\+\s*"([^"]*)"', tie).groups()
    assert "".join(unknown).replace("{W_VERSION}", str(VR.PWR.VERSION)) == VR.PWR.UNKNOWN_VERSION


def test_the_fixed_loads_rows_and_refusal_are_verify_runs():
    tie = (LEAN / "Tie.lean").read_text()
    align = int(re.search(r"def ROW_ALIGN : Nat := (\d+)", tie).group(1))
    assert align == VR.PD.ROW_ALIGN
    assert "let rows := if s == 0 then c.slots * c.prompt else c.slots" in _block(tie, "def loadRows ")
    assert "(rows, (rows + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN)" in _block(tie, "def loadRows ")
    for cls in ((4, 32, 4), (2, 8, 2), (3, 7, 5)):
        slots, prompt, outputs = cls
        assert [VR.load_rows(cls, s) for s in (0, 1, outputs - 1)][:2] == [
            (slots * prompt, -(-slots * prompt // align) * align), (slots, -(-slots // align) * align)]
        text = "".join(re.findall(r's!"([^"]*)"', _block(tie, "def offLoad ")))
        for name, value in (("c.slots * c.prompt", slots * prompt), ("c.slots", slots), ("c.prompt", prompt),
                            ("c.outputs", outputs), ("ROW_ALIGN", align)):
            text = text.replace("{" + name + "}", str(value))
        assert text == VR.off_load(cls) and "{" not in text
