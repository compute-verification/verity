"""The spot verifier suite's `stand_in` (its conftest), which the one-stage difftest `reference` loads imports."""
import pathlib
import runpy

runpy.run_path(str(pathlib.Path(__file__).resolve().parents[3] / "proof/spot/verifier/conftest.py"))
