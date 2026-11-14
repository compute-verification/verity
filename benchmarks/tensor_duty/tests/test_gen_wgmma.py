"""wgmma_e4m3.cuh, the e4m3 wgmma wrapper duty.cu includes, is a generated file: gen_wgmma.py writes it back byte for
byte."""
from __future__ import annotations

import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def test_the_wgmma_header_is_what_its_generator_writes(tmp_path: Path) -> None:
    """The generator writes beside itself, so a copy of it runs in ``tmp_path``."""
    shutil.copy(HERE / "gen_wgmma.py", tmp_path)
    runpy.run_path(str(tmp_path / "gen_wgmma.py"))
    assert (tmp_path / "wgmma_e4m3.cuh").read_bytes() == (HERE / "wgmma_e4m3.cuh").read_bytes()
