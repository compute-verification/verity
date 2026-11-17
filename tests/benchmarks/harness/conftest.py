"""The tests' shared stand-ins are modules beside them, whose asserts pytest rewrites as it rewrites a test's, and the
fixture the tests of `verity bench exfiltration` share."""

from __future__ import annotations

import pytest

pytest.register_assert_rewrite("kit_standins", "overhead_standins", "steady_standins")

from exfiltration_helpers import reader  # noqa: E402


@pytest.fixture
def listed(monkeypatch) -> dict[str, str]:
    """The reader with a format listed for each of its four verdicts, as its own tests list them: it lists none, so
    without this every verdict of a run's files reads as missing. With it the reader still holds no execution
    phase: it asks the network attester's rule without a schedule. The formats, by verdict."""

    sv = reader()
    formats = {kind: f"a format of the {kind}" for kind in sv.CC.VERDICTS}
    monkeypatch.setattr(sv, "FORMATS", {form: kind for kind, form in formats.items()})
    return formats


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result: under `research run` a test inherits `RESEARCH_RUN_DIR` and `OUT`,
    where a command a test starts writes the run's measurements whatever directory its caller names, so each test starts
    with both unset, and a test that writes as a run sets its own (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)

