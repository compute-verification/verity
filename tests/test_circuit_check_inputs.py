"""circuit-check's suite declares every sidecar it reads.

`circuit_check.sidecars.discover` finds every sidecar under the Definition packages, and the registries import each one's module,
so the suite reads them wherever they are; its read guard fails it on one outside `[tool.verity.tests] inputs`. The suite runs
only when a file under those inputs changes, so a PR that adds a Definition module and its sidecar elsewhere leaves it cached and
the failure lands on whichever later change reruns it. This suite keys on the whole tree and so runs on that PR.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from circuit_check import sidecars as SC

ROOT = Path(__file__).resolve().parents[1]
SUITE = "tools/verity/circuit_check"


def _covered(path: str, inputs: list[str]) -> bool:
    return any(p == "." or path == p or path.startswith(p.rstrip("/") + "/") for p in inputs)


def test_every_sidecar_its_pins_and_its_module_are_circuit_check_suite_inputs():
    tests = tomllib.loads((ROOT / SUITE / "pyproject.toml").read_text())["tool"]["verity"]["tests"]
    assert tests.get("inputs_only") is True, f"{SUITE}/pyproject.toml: this test reads its inputs as the suite's whole input set"
    inputs = [SUITE, *tests["inputs"]]
    found = SC.discover()
    assert len(found) > 50, found
    need = []
    for files in found.values():
        for p in files.values():
            stem = p.name[:-len(f"{SC.SUFFIX}{p.suffix}")]
            module = p.with_name(f"{stem}.py") if p.with_name(f"{stem}.py").is_file() else p.parent / stem / "__init__.py"
            need += [SC.rel(p), SC.rel(module)]
    missing = sorted({f for f in need if not _covered(f, inputs)})
    assert not missing, (f"{SUITE}/pyproject.toml's [tool.verity.tests] inputs leave out these files circuit-check reads (a sidecar, "
                         f"its pins or its module), so its read guard would fail the suite; add them:\n"
                         + "\n".join(f'          "{f}",' for f in missing))
