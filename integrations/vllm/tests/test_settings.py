"""`verity_vllm.settings`: every environment variable declared once (a type, a role, a doc), read at the read, the run's record of
the value settings, and the refusal of a ``VERITY_*`` variable nothing reads."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

from verity import settings as verity_settings
from verity_vllm import settings as S
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import research_result as RR

PKG = Path(__file__).resolve().parents[1] / "verity_vllm"
NAME = re.compile(r"^VERITY_[A-Z0-9_]*[A-Z0-9]$")
TOKEN = re.compile(r"\bVERITY_[A-Z0-9_]*[A-Z0-9]\b")
ACCESSORS = frozenset({"get", "raw", "is_set", "put", "setdefault", "unset", "setting"})


def test_every_setting_has_a_type_a_role_and_a_doc() -> None:
    for name, s in S.SETTINGS.items():
        assert s.name == name and callable(s.type) and s.role in S.ROLES and s.doc.strip(), s


def test_settings_imports_nothing_from_the_package() -> None:
    """Every layer reads it, so it closes no import cycle (as ``engine.env`` and ``engine.hooks``, P09's leaf modules)."""
    nodes = list(ast.walk(ast.parse((PKG / "settings.py").read_text(encoding="utf-8"))))
    mods = {a.name for n in nodes if isinstance(n, ast.Import) for a in n.names}
    mods |= {"." * n.level + (n.module or "") for n in nodes if isinstance(n, ast.ImportFrom)}
    assert not {m for m in mods if m.startswith((".", "verity_vllm"))}, sorted(mods)


def test_the_registry_is_built_on_veritys_machinery() -> None:
    assert type(S.REGISTRY) is verity_settings.Registry and S.SETTINGS is S.REGISTRY.settings


def test_flag_reads_both_spellings_and_refuses_the_rest() -> None:
    assert [S.flag(v) for v in ("1", "TRUE", " yes", "on", "0", "False", "no", "OFF")] == [True] * 4 + [False] * 4
    with pytest.raises(ValueError):
        S.flag("2")


def test_a_read_parses_the_environment_at_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_REPLAY_DUMP_MAX", raising=False)
    assert S.VERITY_REPLAY_DUMP_MAX == 1024
    monkeypatch.setenv("VERITY_REPLAY_DUMP_MAX", "4")
    assert S.VERITY_REPLAY_DUMP_MAX == 4 and S.get("VERITY_REPLAY_DUMP_MAX", {"VERITY_REPLAY_DUMP_MAX": "2"}) == 2
    monkeypatch.setenv("VERITY_REPLAY_DUMP_MAX", "")
    assert S.VERITY_REPLAY_DUMP_MAX == 1024 and S.is_set("VERITY_REPLAY_DUMP_MAX")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert S.CUDA_VISIBLE_DEVICES == ""
    monkeypatch.setenv("VERITY_REPLAY_DUMP_MAX", "four")
    with pytest.raises(ValueError, match="VERITY_REPLAY_DUMP_MAX='four'"):
        S.get("VERITY_REPLAY_DUMP_MAX")
    with pytest.raises(AttributeError):
        _ = S.VERITY_NOT_A_SETTING
    with pytest.raises(AttributeError, match="not a declared setting"):
        S.get("VERITY_NOT_A_SETTING")


def test_put_and_unset_write_only_declared_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_INT8_PARTS", "")
    S.put("VERITY_INT8_PARTS", "linears,attention")
    assert S.VERITY_INT8_PARTS == ("linears", "attention")
    S.unset("VERITY_INT8_PARTS")
    assert S.VERITY_INT8_PARTS == () and not S.is_set("VERITY_INT8_PARTS")
    with pytest.raises(AttributeError):
        S.put("VERITY_NOT_A_SETTING", "1")


def test_setdefault_keeps_what_the_environment_holds_and_environ_is_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_JOBS", raising=False)
    assert S.setdefault("MAX_JOBS", "32") == "32" == S.raw("MAX_JOBS")
    assert S.setdefault("MAX_JOBS", "8") == "32"
    monkeypatch.setenv("MAX_JOBS", "")
    assert S.setdefault("MAX_JOBS", "8") == "" and S.MAX_JOBS is None
    env = S.environ()
    env["MAX_JOBS"] = "4"
    assert S.raw("MAX_JOBS") == "" and env == {**os.environ, "MAX_JOBS": "4"}
    with pytest.raises(AttributeError):
        S.setdefault("VERITY_NOT_A_SETTING", "1")


def test_the_narrow_parsers_and_a_kept_empty_read_as_their_readers_did() -> None:
    assert [S.one(v) for v in ("1", " 1", "true", "")] == [True, False, False, False]
    assert S.get("VERITY_ADMIT_OVER_BOUND", {"VERITY_ADMIT_OVER_BOUND": ""}) is False
    assert [S.yes(v) for v in (" yes", "1", "true", "on", "TRUE", "0", "  ")] == [True, True, True, False, False, False, None]
    assert [S.not_zero(v) for v in ("1", " 0 ", "false", "")] == [True, False, True, True]
    assert [S.not_off(v) for v in ("1", " no", "false", "0", "off", "")] == [True, False, False, False, True, True]
    assert [S.number(v) for v in (" 16 ", "0", " ")] == [16.0, 0.0, None] and [S.int_or_zero(v) for v in ("5", " ")] == [5, 0]
    assert S.get("VERITY_MEMTRACE", {"VERITY_MEMTRACE": "0"}) is True and S.get("VERITY_MEMTRACE", {"VERITY_MEMTRACE": ""}) is False
    assert S.get("VERITY_HI_PRIO", {}) is True and S.get("VERITY_HI_PRIO", {"VERITY_HI_PRIO": ""}) is False
    assert S.get("VERITY_FA3_ZERO", {"VERITY_FA3_ZERO": ""}) is True and S.get("VERITY_FA2_TAP_CAP_MB", {"VERITY_FA2_TAP_CAP_MB": ""}) == 0
    assert S.get("VERITY_RETAIN", {"VERITY_RETAIN": ""}) == "" == S.get("VERITY_WEIGHTS_HASH", {"VERITY_WEIGHTS_HASH": ""})
    assert S.get("MATCH_COMPACT", {}) is True and S.get("MATCH_COMPACT", {"MATCH_COMPACT": ""}) is True
    assert S.get("VERITY_SPARSE_MLA_NUM_SPLITS", {"VERITY_SPARSE_MLA_NUM_SPLITS": ""}) == ""
    for v in (None, "", "1", " no ", "false", "0", "off", "x"):
        env = {} if v is None else {"VERITY_REPLAY_FORK_GC_FREEZE": v}
        assert S.get("VERITY_REPLAY_FORK_GC_FREEZE", env) is (env.get("VERITY_REPLAY_FORK_GC_FREEZE", "1").strip() not in ("0", "false", "no"))
    for v in (None, "", "1", " 1", "true", "0"):
        env = {} if v is None else {"STOCH_DECLARED_CHRONOLOGY_S": v}
        assert S.get("STOCH_DECLARED_CHRONOLOGY_S", env) is (env.get("STOCH_DECLARED_CHRONOLOGY_S") == "1")


def test_the_record_carries_the_value_settings_and_families_in_force() -> None:
    env = {"VERITY_WINDOW_MB": "256", "VERITY_FAULT": "", "VERITY_TREE": "/src", "VERITY_REPLAY_DUMP_DIR": "/d", "PATH": "/bin",
           "VLLM_BATCH_INVARIANT": "1", "NCCL_P2P_DISABLE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "RESEARCH_RUN_ID": "r",
           "VLLM_LOGGING_LEVEL": "DEBUG"}
    rec = S.record(env)
    assert rec["schema"] == "verity-vllm/settings/v1"
    assert rec["values"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_P2P_DISABLE": "1", "VERITY_FAULT": "",
                             "VERITY_WINDOW_MB": "256", "VLLM_BATCH_INVARIANT": "1"}
    assert rec["digest"] == ("70e39789e8aa69ee275d97481501353e60666901016d8d9848aeb12ee15e1b85"
                             "61a9fc35cadf35633c52daf378e95b17872eedbf699c896f9872136d0a1f5273")
    assert S.record({})["values"] == {}


def test_the_record_names_a_credential_and_never_holds_its_value() -> None:
    """``VLLM_API_KEY`` is declared ``secret``; an undeclared ``VLLM_*_TOKEN`` is one by its name."""
    key, token = "dummy-not-a-real-key", "dummy-not-a-real-token"
    rec = S.record({"VLLM_API_KEY": key, "VLLM_DUMMY_TOKEN": token, "HF_TOKEN": token, "VLLM_BATCH_INVARIANT": "1"})
    assert rec["values"] == {"VLLM_API_KEY": S.REDACTED, "VLLM_BATCH_INVARIANT": "1", "VLLM_DUMMY_TOKEN": S.REDACTED}
    assert key not in json.dumps(rec) and token not in json.dumps(rec)
    assert S.setting("VLLM_API_KEY").secret and S.is_secret("NCCL_X_SECRET") and not S.is_secret("VLLM_BATCH_INVARIANT")


def test_an_unknown_verity_variable_is_refused_by_name() -> None:
    assert S.unknown({"VERITY_WINDOW_MB": "1", "VERITY_SUITE_ROOT": "x", "VERITY_REGRESSION": "1", "PATH": "x"}) == []
    with pytest.raises(S.UnknownSetting, match="unknown setting VERITY_WINDOW_MBB: nothing in the tree reads it"):
        S.validate_environ({"VERITY_WINDOW_MBB": "1", "VERITY_WINDOW_MB": "1"})
    assert not S.ELSEWHERE & set(S.SETTINGS) and not S.NOT_SETTINGS & (set(S.SETTINGS) | S.ELSEWHERE)


def test_verity_vllm_refuses_an_unknown_setting_at_startup(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("VERITY_WINDOW_MBB", "256")
    monkeypatch.setattr(sys, "argv", ["verity-vllm", "target-family", "--help"])
    assert cli.main() == 2
    assert "unknown setting VERITY_WINDOW_MBB" in capsys.readouterr().err


def _library_names() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """The ``VERITY_*`` names the library spells out whole, by file: Python strings and C ``getenv``s (which only `SETTINGS` and
    `NOT_SETTINGS` may name), shell tokens (also `ELSEWHERE`: a script sets what another tool reads), and the names
    ``settings.NAME`` and ``settings.get("NAME")`` read (`SETTINGS` only)."""
    strings: dict[str, set[str]] = {}
    shell: dict[str, set[str]] = {}
    reads: dict[str, set[str]] = {}
    for p in sorted(PKG.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix not in (".py", ".sh", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h"):
            continue
        rel, src = str(p.relative_to(PKG.parent)), p.read_text(encoding="utf-8", errors="replace")
        if p.suffix == ".sh":
            for n in TOKEN.findall(src):
                shell.setdefault(n, set()).add(rel)
        elif p.suffix != ".py":
            for n in re.findall(r'getenv\("(VERITY_\w+)"\)', src):
                strings.setdefault(n, set()).add(rel)
        elif "VERITY_" in src or "settings" in src:
            nodes = list(ast.walk(ast.parse(src)))
            registry = any(isinstance(n, ast.ImportFrom) and n.module == "verity_vllm"
                           and any(a.name == "settings" and a.asname is None for a in n.names) for n in nodes)
            for node in nodes:
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and NAME.match(node.value):
                    strings.setdefault(node.value, set()).add(rel)
                elif not registry:
                    continue
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "settings" \
                        and node.attr.isupper():
                    reads.setdefault(node.attr, set()).add(rel)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ACCESSORS \
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "settings" and node.args \
                        and isinstance(node.args[0], ast.Constant):
                    reads.setdefault(node.args[0].value, set()).add(rel)
    return strings, shell, reads


def test_every_name_the_library_reads_is_declared() -> None:
    strings, shell, reads = _library_names()
    fix = "declare it in verity_vllm/settings.py (SETTINGS; NOT_SETTINGS for a C macro, ELSEWHERE when another tool reads it)"
    assert not (bad := {n: sorted(f) for n, f in strings.items() if n not in S.SETTINGS and n not in S.NOT_SETTINGS}), (fix, bad)
    assert not (bad := {n: sorted(f) for n, f in shell.items() if not S.known(n) and n not in S.NOT_SETTINGS}), (fix, bad)
    assert not (bad := {n: sorted(f) for n, f in reads.items() if n not in S.SETTINGS}), (fix, bad)
    assert {"VERITY_REPLAY_DUMP_MAX", "RESEARCH_RUN_ID", "CUDA_VISIBLE_DEVICES"} <= set(reads) and "VERITY_INT8_PARTS" in strings


def _digest(doc: object) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


@pytest.mark.parametrize(("value", "digest"), [
    (None, "2c2ad4071664fe66f69f7f30e82b8b66"),
    ("", "2c2ad4071664fe66f69f7f30e82b8b66"),
    ("r20261025-0000-x", "fa8a56a3f92a12bc97710e70d153dc0c"),
])
def test_research_result_records_the_run_id_as_before(monkeypatch: pytest.MonkeyPatch, value: str | None, digest: str) -> None:
    if value is None:
        monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("RESEARCH_RUN_ID", value)
    assert _digest(RR.build_result(Path("no-evidence-dir"), "row", "role", ["build"], None, None)) == digest
