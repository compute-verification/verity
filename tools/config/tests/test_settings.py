from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from verity_config import settings as ST
from verity_config.settings import DEBUG, SHARED, VALUE, WHERE, Registry, Setting


def registry(**kw) -> Registry:
    return Registry("dist/settings.py", ("PX_",),
                    Setting("PX_N", int, 3, VALUE, "a count"),
                    Setting("PX_ON", ST.one, False, VALUE, "on at 1"),
                    Setting("PX_FORM", ST.choice("a", "b"), "a", VALUE, "a form"),
                    Setting("PX_DIR", str, None, WHERE, "where outputs go"),
                    Setting("PX_TRACE", ST.present, False, DEBUG, "trace the launches"),
                    Setting("PX_KEEP", str, None, VALUE, "kept empty", empty=True),
                    Setting("PX_PASS", str, None, VALUE, "a credential by declaration", secret=True), **kw)


def test_a_setting_is_parsed_at_each_read_and_unset_or_empty_is_its_default(monkeypatch) -> None:
    r = registry()
    monkeypatch.delenv("PX_N", raising=False)
    assert r.PX_N == 3 and r.get("PX_N") == 3
    monkeypatch.setenv("PX_N", "")
    assert r.PX_N == 3 and not r.is_set("PX_N")
    monkeypatch.setenv("PX_N", "7")
    assert r.PX_N == 7 and r.is_set("PX_N") and r.raw("PX_N") == "7"
    monkeypatch.setenv("PX_N", "seven")
    with pytest.raises(ValueError, match=r"PX_N='seven'.*dist/settings.py"):
        _ = r.PX_N
    monkeypatch.setenv("PX_KEEP", "")
    assert r.PX_KEEP == "" and r.is_set("PX_KEEP")
    assert r.get("PX_ON", {"PX_ON": "1"}) is True and r.get("PX_ON", {"PX_ON": "true"}) is False
    assert r.get("PX_FORM", {"PX_FORM": "b"}) == "b"
    with pytest.raises(ValueError, match="one of a, b"):
        r.get("PX_FORM", {"PX_FORM": "c"})
    with pytest.raises(AttributeError, match="PX_NOPE is not a declared setting"):
        r.get("PX_NOPE")
    with pytest.raises(AttributeError):
        _ = r.PX_NOPE
    assert r.need("PX_N", {"PX_N": "0"}) == 0 and r.need("PX_KEEP", {"PX_KEEP": ""}) == ""
    with pytest.raises(KeyError, match=r"PX_DIR is not set: where outputs go \(dist/settings.py\)"):
        r.need("PX_DIR", {"PX_DIR": ""})


def test_put_setdefault_and_unset_write_only_declared_variables(monkeypatch) -> None:
    r = registry()
    monkeypatch.delenv("PX_DIR", raising=False)
    assert r.setdefault("PX_DIR", "/a") == "/a" and r.setdefault("PX_DIR", "/b") == "/a"
    r.put("PX_DIR", "/c")
    assert r.PX_DIR == "/c" and r.environ()["PX_DIR"] == "/c"
    r.unset("PX_DIR")
    assert r.PX_DIR is None
    with pytest.raises(AttributeError):
        r.put("PX_NOPE", "1")


def test_every_registry_reads_the_shared_settings_and_declares_each_once() -> None:
    r = registry()
    assert r.setting("RESEARCH_RUN_ID") is SHARED.setting("RESEARCH_RUN_ID")
    assert r.get("CUDA_VISIBLE_DEVICES", {"CUDA_VISIBLE_DEVICES": ""}) == ""
    with pytest.raises(ValueError, match="RESEARCH_RUN_ID is declared twice"):
        Registry("x", (), Setting("RESEARCH_RUN_ID", str, None, WHERE, "again"))
    with pytest.raises(ValueError, match="PX_A is declared twice"):
        Registry("x", (), Setting("PX_A", str, None, WHERE, "a"), Setting("PX_A", str, None, WHERE, "b"))
    with pytest.raises(ValueError, match="role 'loud'"):
        Registry("x", (), Setting("PX_A", str, None, "loud", "a"))
    with pytest.raises(ValueError, match="says what it does"):
        Registry("x", (), Setting("PX_A", str, None, WHERE, " "))
    assert all(s.role == WHERE for s in SHARED.settings.values())


def test_an_unknown_variable_with_an_owned_prefix_is_refused() -> None:
    r = registry(elsewhere=("PX_OTHER", "PX_FAM_"))
    assert r.unknown(["PX_N", "PX_TYPO", "PX_OTHER", "PX_FAM_1", "QX_TYPO", "PATH"]) == ["PX_TYPO"]
    r.validate_environ(["PX_N", "QX_TYPO"])
    with pytest.raises(ST.UnknownSetting, match=r"unknown settings PX_A, PX_B: nothing reads them.*dist/settings.py"):
        r.validate_environ(["PX_B", "PX_A"])


def test_the_record_holds_the_value_settings_set_with_their_digest() -> None:
    r = registry(families={"VLLM_": VALUE})
    env = {"PX_N": "7", "PX_DIR": "/d", "PX_TRACE": "1", "VLLM_X": "1", "PX_PASS": "hunter2", "HF_TOKEN": "t", "PATH": "/bin"}
    rec = r.record(env)
    assert rec["values"] == {"PX_N": "7", "PX_PASS": ST.REDACTED, "VLLM_X": "1"}
    blob = json.dumps(rec["values"], sort_keys=True, separators=(",", ":")).encode()
    assert rec["digest"] == hashlib.sha512(blob).hexdigest() and rec["schema"] == ST.RECORD_SCHEMA
    assert r.role_of("VLLM_Y") == VALUE and r.role_of("PX_DIR") == WHERE and r.role_of("PATH") is None
    assert r.redact({"HF_TOKEN": "t", "A": "b"}) == {"HF_TOKEN": ST.REDACTED, "A": "b"}


def test_a_run_of_record_refuses_a_debug_setting() -> None:
    r = registry()
    assert r.of_record({"PX_N": "7"}) == r.record({"PX_N": "7"})
    assert r.of_record({"PX_TRACE": ""})["values"] == {}
    with pytest.raises(ST.DebugOnRecord, match="PX_TRACE: debug setting, refused on a run of record"):
        r.of_record({"PX_TRACE": "0"})


def test_of_loads_the_distributions_registry_by_its_path(tmp_path, monkeypatch) -> None:
    for name in ("a", "b"):
        d = tmp_path / name
        (d / "sub").mkdir(parents=True)
        (d / "pyproject.toml").write_text("")
        (d / "settings.py").write_text(
            "from verity_config.settings import VALUE, Registry, Setting\n"
            f"REGISTRY = Registry(__file__, ('{name.upper()}_',), Setting('{name.upper()}_N', int, 1, VALUE, 'n'))\n")
    monkeypatch.setenv("A_N", "5")
    a, b = ST.of(tmp_path / "a" / "sub" / "x.py"), ST.of(tmp_path / "b" / "y.py")
    assert a.A_N == 5 and b.B_N == 1 and a is ST.of(tmp_path / "a" / "z.py")
    assert "settings" not in sys.modules or sys.modules["settings"] not in (a, b)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "pyproject.toml").write_text("")
    with pytest.raises(LookupError, match="has no settings registry"):
        ST.of(tmp_path / "c" / "x.py")
    d = tmp_path / "d"
    d.mkdir()
    (d / "pyproject.toml").write_text("")
    (d / "settings.py").write_text(
        "from verity_config.settings import VALUE, Registry, Setting\n"
        "REGISTRY = Registry(__file__, ('D_',), Setting('D_N', int, 1, VALUE, 'n'))\n")
    monkeypatch.setenv("D_TYPO", "1")
    with pytest.raises(ST.UnknownSetting, match="D_TYPO"):
        ST.of(d / "x.py")
    monkeypatch.delenv("D_TYPO")
    assert ST.of(d / "x.py").D_N == 1
    ship = tmp_path / "ship"
    ship.mkdir()
    assert not any((p / "pyproject.toml").is_file() for p in ship.parents)
    with pytest.raises(LookupError, match="no settings.py beside it"):
        ST.of(ship / "check.py")
    (ship / "settings.py").write_text((d / "settings.py").read_text())
    assert ST.of(ship / "check.py").D_N == 1 and Path(ST.of(ship / "bench.py").path) == ship / "settings.py"


def test_a_standalone_file_reads_only_declared_variables_at_their_declared_defaults(tmp_path) -> None:
    (tmp_path / "ship").mkdir()
    (tmp_path / "ship" / "run.py").write_text(
        "import os\n"
        "n = int(os.environ.get('PX_N', '3'))\n"
        "d = os.getenv('PX_DIR')\n"
        "on = 'PX_ON' in os.environ\n"
        "f = os.environ['PX_FORM']\n"
        "m = int(os.environ.get('PX_N', '4'))\n"
        "t = os.environ.get('PX_TYPO', '')\n"
        "k = int(os.environ.get('PX_N', 'three'))\n")
    r = Registry(str(tmp_path / "settings.py"), ("PX_",), Setting("PX_N", int, 3, VALUE, "a count"),
                 Setting("PX_ON", ST.one, False, VALUE, "on at 1"), Setting("PX_FORM", str, "a", VALUE, "a form"),
                 Setting("PX_DIR", str, None, WHERE, "where outputs go"), standalone=("ship/run.py",))
    assert [(line, name) for line, name, _ in ST.direct_reads((tmp_path / "ship" / "run.py").read_text())] == [
        (2, "PX_N"), (3, "PX_DIR"), (4, "PX_ON"), (5, "PX_FORM"), (6, "PX_N"), (7, "PX_TYPO"), (8, "PX_N")]
    assert r.standalone_problems() == [
        "ship/run.py:6: PX_N's default '4' here is not its declared 3",
        f"ship/run.py:7: PX_TYPO is not declared in {tmp_path / 'settings.py'}",
        "ship/run.py:8: PX_N's default 'three' here doesn't parse as declared: invalid literal for int() with base 10: 'three'"]
    assert ST.words(" a  b ") == ("a", "b") and ST.words("") == ()
