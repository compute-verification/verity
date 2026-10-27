"""`verity_all_protocols.settings`: an unset variable is its default, an empty string is a value (``CERTIFIER_TAMPER=`` relays
no tampered run) except for an int, and an undeclared name or an unparseable string is refused, naming it."""
from __future__ import annotations

import pytest

from verity_all_protocols import settings


def test_unset_is_the_default_and_empty_is_a_value():
    assert settings.get("CERTIFIER_TAMPER", {}) is None
    assert settings.get("CERTIFIER_TAMPER", {"CERTIFIER_TAMPER": ""}) == ""
    assert settings.get("WARDEN_TAMPER", {"WARDEN_TAMPER": "late"}) == "late"


def test_an_int_parses_and_empty_is_its_default():
    assert settings.get("CERTIFIER_HEIGHT", {}) == 10
    assert settings.get("CERTIFIER_HEIGHT", {"CERTIFIER_HEIGHT": ""}) == 10
    assert settings.get("SCAN_DRAW", {"SCAN_DRAW": "8"}) == 8
    with pytest.raises(ValueError, match="SCAN_DRAW='many'"):
        settings.get("SCAN_DRAW", {"SCAN_DRAW": "many"})


def test_an_undeclared_name_is_refused():
    with pytest.raises(AttributeError, match="NOT_A_SETTING"):
        settings.get("NOT_A_SETTING", {})
    with pytest.raises(AttributeError, match="NOT_A_SETTING"):
        settings.put("NOT_A_SETTING", "1")
