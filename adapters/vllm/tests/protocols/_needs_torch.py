"""torch, or a skip of the module that imports this one where torch isn't installed: imported among a test module's
other imports, it stands in for a module-level `pytest.importorskip("torch")`, so the imports after it need no E402
exception."""
import pytest

torch = pytest.importorskip("torch")
