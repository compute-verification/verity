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
    phase: it asks the network certifier's rule without a schedule. The formats, by verdict."""

    sv = reader()
    formats = {kind: f"a format of the {kind}" for kind in sv.CC.VERDICTS}
    monkeypatch.setattr(sv, "FORMATS", {form: kind for kind, form in formats.items()})
    return formats
