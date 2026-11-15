"""`verity_vllm.log`: vLLM's record format in UTC, continuation lines under the head, the `*_once` methods, stderr of the moment
and never stdout, and one handler that stays quiet when a handler above takes the record (so `caplog` sees every one)."""
from __future__ import annotations

import ast
import io
import logging
import re
from pathlib import Path

import pytest

from verity_vllm import log as L

PKG = Path(__file__).resolve().parents[1] / "verity_vllm"


@pytest.fixture(autouse=True)
def _defaults():
    yield
    L.configure()


def _record(msg: str, *args: object, exc_info=None) -> logging.LogRecord:
    r = logging.LogRecord("verity_vllm.pipeline.commit", logging.INFO, "/x/verity_vllm/pipeline/commit.py", 412, msg, args,
                          exc_info)
    r.created = 31 * 86400 + 3723.5
    return r


def test_a_record_reads_vllms_head_with_the_time_in_utc() -> None:
    assert L.Formatter().format(_record("matched %d of %d", 3, 4)) == "INFO 02-01 01:02:03 [commit.py:412] matched 3 of 4"


def test_each_further_line_repeats_the_head_a_tracebacks_too() -> None:
    head = "INFO 02-01 01:02:03 [commit.py:412] "
    assert L.Formatter().format(_record("one\ntwo")) == f"{head}one\n{head}two"
    try:
        raise KeyError("k")
    except KeyError as e:
        text = L.Formatter().format(_record("failed", exc_info=(type(e), e, e.__traceback__)))
    lines = text.splitlines()
    assert len(lines) > 2 and all(ln.startswith(head) for ln in lines) and lines[-1] == f"{head}KeyError: 'k'"


def test_caplog_sees_every_record_and_the_handler_stays_quiet_when_a_handler_above_takes_it(caplog) -> None:
    out = io.StringIO()
    L.configure(stream=out)
    caplog.set_level(logging.INFO, logger=L.PACKAGE)
    L.init_logger("verity_vllm.pipeline.commit").info("[tp2commit] rank %d done", 1)
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("verity_vllm.pipeline.commit", logging.INFO, "[tp2commit] rank 1 done")]
    assert out.getvalue() == ""


def _above(monkeypatch, level: int) -> None:
    """Every handler above the package's logger (pytest's, on the root) takes only records at ``level`` and over."""
    for h in logging.getLogger().handlers:
        monkeypatch.setattr(h, "level", level)


def test_with_no_handler_above_a_record_goes_to_stderr_as_it_is_then_and_never_to_stdout(capsys, monkeypatch) -> None:
    _above(monkeypatch, logging.CRITICAL + 1)
    logger = L.init_logger("verity_vllm.check.verdict")
    logger.info("checked %s", "row 3")
    logger.debug("not at INFO")
    out, err = capsys.readouterr()
    assert out == "" and re.fullmatch(r"INFO \d\d-\d\d \d\d:\d\d:\d\d \[test_log\.py:\d+\] checked row 3\n", err), err
    L.configure("DEBUG")
    logger.debug("now at DEBUG")
    assert capsys.readouterr().err.startswith("DEBUG ")


def test_a_handler_above_that_takes_only_warnings_leaves_info_to_this_one(caplog, monkeypatch) -> None:
    _above(monkeypatch, logging.WARNING)
    out = io.StringIO()
    L.configure(stream=out)
    logger = L.init_logger("verity_vllm.engine")
    logger.info("info")
    logger.warning("warning")
    assert out.getvalue().endswith("] info\n") and [r.getMessage() for r in caplog.records] == ["warning"]


def test_once_logs_a_message_with_its_arguments_once(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger=L.PACKAGE)
    logger = L.init_logger("verity_vllm.observe")
    for _ in range(3):
        logger.warning_once("pin %s moved", "a")
        logger.warning_once("pin %s moved", "b")
        logger.info_once("info once")
        logger.debug_once("debug once")
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("WARNING", "pin a moved"), ("WARNING", "pin b moved"), ("INFO", "info once"), ("DEBUG", "debug once")]
    assert {(r.filename, r.funcName) for r in caplog.records} == {("test_log.py", "test_once_logs_a_message_with_its_arguments_once")}


def test_a_logger_outside_the_package_is_refused() -> None:
    for name in ("__main__", "verity_vllmx", "verity.circuits"):
        with pytest.raises(ValueError, match="pass the `__name__` of a module of verity_vllm"):
            L.init_logger(name)
    assert L.init_logger("verity_vllm") is logging.getLogger("verity_vllm")


def test_the_package_logger_has_one_handler_at_info_however_often_it_is_imported() -> None:
    package = logging.getLogger(L.PACKAGE)
    assert L.handler() is L.handler() and [h.get_name() for h in package.handlers] == [L.PACKAGE]
    assert package.level == logging.INFO and package.propagate


def test_the_logger_imports_nothing_from_the_package() -> None:
    """Every layer may log, so it closes no import cycle (as ``settings`` and ``errors``)."""
    nodes = list(ast.walk(ast.parse((PKG / "log.py").read_text(encoding="utf-8"))))
    mods = {a.name for n in nodes if isinstance(n, ast.Import) for a in n.names}
    mods |= {"." * n.level + (n.module or "") for n in nodes if isinstance(n, ast.ImportFrom)}
    assert not {m for m in mods if m.startswith((".", "verity_vllm"))}, sorted(mods)
