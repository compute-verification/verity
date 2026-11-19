"""dc's one table of what it times on each architecture (calibration/pouw/dc/archs.py): every entry is an architecture
file's name, analyze reads every entry's tensor instruction, and an architecture or label the table lacks is refused by
name."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import load

ROOT = Path(__file__).resolve().parents[3]
DC = ROOT / "calibration" / "pouw" / "dc"
archs = load("dc/archs.py")
an = load("dc/analyze.py")


def test_every_entry_is_an_architecture_file_and_analyze_reads_its_tensor_instruction():
    for name, entry in archs.ARCHS.items():
        assert (ROOT / "calibration" / "hardware" / "architectures" / f"{name}.toml").is_file(), name
        assert entry.tensor in an.TIMED, (name, entry.tensor)
    assert len(set(archs.FAMILIES)) == len(archs.ARCHS)


def test_an_architecture_the_table_lacks_is_refused_by_name():
    with pytest.raises(SystemExit, match="^the card's target is sm_120a, and the tensor-form probe has no instructions"):
        archs.of("sm_120a")
    with pytest.raises(SystemExit, match="no architecture of the tensor-form probe has the family label 'sm120'"):
        an.analyze("sm120", Path("unused"))
    done = subprocess.run([sys.executable, str(DC / "archs.py"), "sm_120a", "family"], capture_output=True, text=True)
    assert done.returncode == 2 and done.stderr.strip() == archs.refusal("sm_120a") and not done.stdout


def test_the_command_prints_a_field_a_line_and_the_capability_from_the_architecture_file():
    def field(arch, name):
        return subprocess.run([sys.executable, str(DC / "archs.py"), arch, name], capture_output=True, text=True,
                              check=True).stdout
    assert field("sm_90a", "family") == "sm90\n" and field("sm_100a", "s8_forms") == "mma\ntc\n"
    assert field("sm_100a", "tmem") == "1\n" and field("sm_90a", "tmem") == "\n"
    assert field("sm_90a", "capability") == "9.0\n" and field("sm_100a", "capability") == "10.0\n"
