"""`verity_vllm.settings`: every environment variable declared once (a type, a role, a doc), read at the read, the run's record of
the value settings, and the refusal of a ``VERITY_*`` variable nothing reads."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from verity_vllm import settings as S
from verity_vllm.check import kernel_identity as KI
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import research_result as RR

PKG = Path(__file__).resolve().parents[1] / "verity_vllm"
NAME = re.compile(r"^VERITY_[A-Z0-9_]*[A-Z0-9]$")
TOKEN = re.compile(r"\bVERITY_[A-Z0-9_]*[A-Z0-9]\b")
ACCESSORS = frozenset({"get", "raw", "is_set", "put", "unset", "setting"})


def test_every_setting_has_a_type_a_role_and_a_doc() -> None:
    for name, s in S.SETTINGS.items():
        assert s.name == name and callable(s.type) and s.role in S.ROLES and s.doc.strip(), s


def test_flag_reads_both_spellings_and_refuses_the_rest() -> None:
    assert [S.flag(v) for v in ("1", "TRUE", " yes", "on", "0", "False", "no", "OFF")] == [True] * 4 + [False] * 4
    with pytest.raises(ValueError):
        S.flag("2")


def test_a_read_parses_the_environment_at_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_WINDOW_SLOTS", raising=False)
    assert S.VERITY_WINDOW_SLOTS == 8
    monkeypatch.setenv("VERITY_WINDOW_SLOTS", "4")
    assert S.VERITY_WINDOW_SLOTS == 4 and S.get("VERITY_WINDOW_SLOTS", {"VERITY_WINDOW_SLOTS": "2"}) == 2
    monkeypatch.setenv("VERITY_WINDOW_SLOTS", "")
    assert S.VERITY_WINDOW_SLOTS == 8 and S.is_set("VERITY_WINDOW_SLOTS")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert S.CUDA_VISIBLE_DEVICES == ""
    monkeypatch.setenv("VERITY_WINDOW_SLOTS", "four")
    with pytest.raises(ValueError, match="VERITY_WINDOW_SLOTS='four'"):
        S.get("VERITY_WINDOW_SLOTS")
    with pytest.raises(AttributeError):
        S.VERITY_NOT_A_SETTING  # noqa: B018
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


def test_the_record_carries_the_value_settings_and_families_in_force() -> None:
    env = {"VERITY_WINDOW_MB": "256", "VERITY_FAULT": "", "VERITY_TREE": "/src", "VERITY_REPLAY_DUMP_DIR": "/d", "PATH": "/bin",
           "VLLM_BATCH_INVARIANT": "1", "NCCL_P2P_DISABLE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TRITON_CACHE_DIR": "/t"}
    rec = S.record(env)
    assert rec["schema"] == "verity-vllm/settings/v1"
    assert rec["values"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_P2P_DISABLE": "1", "VERITY_FAULT": "",
                             "VERITY_WINDOW_MB": "256", "VLLM_BATCH_INVARIANT": "1"}
    assert rec["digest"] == ("70e39789e8aa69ee275d97481501353e60666901016d8d9848aeb12ee15e1b85"
                             "61a9fc35cadf35633c52daf378e95b17872eedbf699c896f9872136d0a1f5273")
    assert S.record({})["values"] == {}


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
    assert {"VERITY_REPLAY_DUMP_MAX", "CXX", "TRITON_CACHE_DIR"} <= set(reads) and "VERITY_INT8_PARTS" in strings


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


@pytest.mark.parametrize(("value", "digest"), [
    (None, "927a4b3d3db42708a974740baadab9c4"),
    ("", "7ff16ab570d79909b8849b9689ad0c9e"),
    ("/nonexistent/triton", "09ff3b378460f2186fb5dfa06cf554b2"),
])
def test_kernel_identity_records_the_triton_cache_dir_as_before(monkeypatch: pytest.MonkeyPatch, value: str | None, digest: str) -> None:
    if value is None:
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("TRITON_CACHE_DIR", value)
    assert _digest(KI.collect(None, KI.triton_cache_dir())) == digest
