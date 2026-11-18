"""The POUS-only harness's settings registry (benchmarks/pous/settings.py): it loads by path, and every read in the files
that run in `pod.sh`'s own venv (`STANDALONE`) is a declaration's, the registry's or the shared one's."""
from __future__ import annotations

from pathlib import Path

from verity_config.settings import of

POUS = Path(__file__).resolve().parents[3] / "benchmarks" / "pous"


def test_loads_by_path_and_holds_the_harness_to_its_declarations():
    reg = of(POUS / "bench.py")
    assert reg is of(POUS / "band_gpu" / "native.py") and Path(reg.path) == POUS / "settings.py"
    assert all((POUS / f).is_file() for f in reg.standalone) and len(reg.standalone) == 7
    assert reg.standalone_problems() == []
