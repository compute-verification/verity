"""The package file holds nothing: importing the record, the registration log or the Site runs no card-loading code."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "__init__.py"


def test_the_package_file_is_a_docstring_alone() -> None:
    """Whatever ``verity/hardware/__init__.py`` holds runs on every import of a module under it, a verifier's included:
    it re-exports nothing, and the loader is ``verity.hardware.cards``."""
    body = ast.parse(PACKAGE.read_text(encoding="utf-8")).body
    assert len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0].value, "value", None), str), (
        "verity/hardware/__init__.py holds more than its docstring; put the code in the module that owns it "
        "(the cards' loader is verity/hardware/cards/__init__.py) and import it from there")


def test_the_record_the_registration_and_the_site_import_no_cards() -> None:
    """In an interpreter of its own, since this process's other tests import the cards: after the three are imported,
    the loader is not among the modules, so none of them, nor the package file, imports it."""
    done = subprocess.run(
        [sys.executable, "-c",
         "import sys, verity.hardware.record, verity.hardware.registration, verity.hardware.site; "
         "sys.exit('verity.hardware.cards' in sys.modules)"],
        capture_output=True, text=True)
    assert done.returncode == 0, (
        "importing verity.hardware.record, .registration and .site imported verity.hardware.cards (or failed): "
        f"import the cards only where a card is read\n{done.stderr}")
