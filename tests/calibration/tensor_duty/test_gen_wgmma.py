"""wgmma_e4m3.cuh and wgmma_i8.cuh, the wgmma wrappers duty.cu includes, are generated files: gen_wgmma.py writes them
back byte for byte."""
from __future__ import annotations

import runpy
import shutil
from pathlib import Path

HERE = (Path(__file__).resolve().parents[3] / "calibration" / "tensor_duty")


def test_the_wgmma_headers_are_what_their_generator_writes(tmp_path: Path) -> None:
    """The generator writes beside itself, so a copy of it runs in ``tmp_path``."""
    shutil.copy(HERE / "gen_wgmma.py", tmp_path)
    runpy.run_path(str(tmp_path / "gen_wgmma.py"))
    for header in ("wgmma_e4m3.cuh", "wgmma_i8.cuh"):
        assert (tmp_path / header).read_bytes() == (HERE / header).read_bytes(), header
