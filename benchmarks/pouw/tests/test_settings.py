"""`settings`, the PoUW benchmarks' registry: each setting declared whole, read with its default and type, the record a run
carries, and the refusal of a ``VERITY_POUW_*`` name nothing reads."""
from __future__ import annotations

import hashlib
import json

import pytest

import settings


@pytest.mark.parametrize("name", sorted(settings.SETTINGS))
def test_each_setting_is_declared_whole(name):
    s = settings.SETTINGS[name]
    assert s.name == name and s.role in settings.ROLES and s.doc.strip()
    assert s.default is None or isinstance(s.default, s.type)


def test_a_read_parses_and_an_unset_or_empty_variable_is_its_default():
    assert settings.get("RASTER", {}) == 8
    assert settings.get("RASTER", {"RASTER": ""}) == 8
    assert settings.get("RASTER", {"RASTER": "4"}) == 4
    assert settings.get("REPS", {}) is None
    with pytest.raises(ValueError, match="RASTER='four'"):
        settings.get("RASTER", {"RASTER": "four"})


def test_an_undeclared_name_is_refused():
    with pytest.raises(AttributeError, match="NOT_A_SETTING"):
        settings.get("NOT_A_SETTING", {})
    with pytest.raises(AttributeError):
        settings.NOT_A_SETTING  # noqa: B018
    with pytest.raises(AttributeError):
        settings.put("NOT_A_SETTING", "1")


def test_the_record_carries_the_value_settings_with_their_digest():
    rec = settings.record({"REPS": "5", "SHAPE": "8192", "RESEARCH_RUN_ID": "r1", "UNRELATED": "x"})
    assert rec["schema"] == settings.RECORD_SCHEMA and rec["values"] == {"REPS": "5", "SHAPE": "8192"}
    assert rec["digest"] == hashlib.sha512(json.dumps(rec["values"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_the_record_refuses_a_debug_setting():
    with pytest.raises(ValueError, match="STAMPS"):
        settings.record({"STAMPS": "1"})
    assert settings.record({"STAMPS": ""})["values"] == {}


def test_a_prefixed_variable_nothing_reads_is_refused():
    assert settings.unknown(["VERITY_POUW_H1_PIN", "VERITY_POUW_CHECK", "PATH"]) == []
    with pytest.raises(ValueError, match="VERITY_POUW_TYPO"):
        settings.validate_environ(["VERITY_POUW_TYPO"])
