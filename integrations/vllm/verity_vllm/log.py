"""The integration's logger, in vLLM's style (``vllm/logger.py``): a library module logs through ``logger =
init_logger(__name__)``, never ``print``.

A record reads ``LEVEL MM-DD HH:MM:SS [file.py:line] message``, vLLM's format, with the time in UTC so lines from several
machines compare; each further line of a message (a traceback's too) repeats that head, so every line says where it came
from.  ``debug_once``, ``info_once`` and ``warning_once`` log a message with its arguments once per process.

Records go to stderr, not to stdout as vLLM's do: a command's stdout carries what it prints for a reader (a JSON record a
caller redirects to a file), which a diagnostic must not interleave with.  The stream is ``sys.stderr`` as it is at each
record, so a redirect or a test's capture sees every one, and each record is flushed.

One handler, on the ``verity_vllm`` logger, which propagates: when a handler above it takes a record (an application that
configures the root logger, pytest's ``caplog``), that handler writes it and this one stays quiet, so no record is written
twice and ``caplog`` sees them all.  Importing this module adds the handler and sets the package's level to INFO, and
configures nothing else; a program's entry point sets the level and stream with ``configure``.  Inside a vLLM process,
vLLM's ``VLLM_CONFIGURE_LOGGING`` and ``VLLM_LOGGING_*`` govern only vLLM's own logger, never this one.  It imports nothing
from the package, so every layer may log.
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Hashable
from functools import cache
from types import MethodType
from typing import TextIO, cast

PACKAGE = "verity_vllm"
#: vLLM's head, which each further line of a record repeats, and its date format
HEAD = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] "
DATE_FORMAT = "%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


class Formatter(logging.Formatter):
    """vLLM's ``NewLineFormatter``: ``HEAD``, then the message, with the head again on each further line; the time in UTC."""

    converter = time.gmtime

    def __init__(self) -> None:
        super().__init__(HEAD + "%(message)s", DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        record.fileinfo = record.filename
        text = super().format(record)
        return text.replace("\n", "\n" + HEAD % record.__dict__)


def taken_above(levelno: int) -> bool:
    """Whether a record at ``levelno`` that the package's logger propagates reaches a handler that takes it."""
    logger: logging.Logger | None = logging.getLogger(PACKAGE)
    while logger is not None and logger.propagate:
        logger = logger.parent
        if logger is not None and any(levelno >= h.level for h in logger.handlers):
            return True
    return False


class Handler(logging.StreamHandler):
    """Writes a record to ``sys.stderr`` as it is then (or the stream ``configure`` set), unless a handler above the package's
    logger takes it."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        self._stream: TextIO | None = None
        self.set_name(PACKAGE)
        self.setFormatter(Formatter())

    @property
    def stream(self) -> TextIO:
        return self._stream or sys.stderr

    @stream.setter
    def stream(self, stream: TextIO | None) -> None:
        self._stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        if not taken_above(record.levelno):
            super().emit(record)


@cache
def _once(logger: logging.Logger, level: int, msg: str, *args: Hashable) -> None:
    # 3: past this function and the `*_once` method, the line that called it
    logger.log(level, msg, *args, stacklevel=3)


class VerityLogger(logging.Logger):
    """The type ``init_logger`` returns: a standard logger with these methods set on the instance, as vLLM does, so the
    logging module's logger class stays the standard one."""

    def debug_once(self, msg: str, *args: Hashable) -> None:
        """As ``debug``, but only the first call with this message and these arguments logs."""
        _once(self, logging.DEBUG, msg, *args)

    def info_once(self, msg: str, *args: Hashable) -> None:
        """As ``info``, but only the first call with this message and these arguments logs."""
        _once(self, logging.INFO, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        """As ``warning``, but only the first call with this message and these arguments logs."""
        _once(self, logging.WARNING, msg, *args)


ONCE = ("debug_once", "info_once", "warning_once")


def init_logger(name: str) -> VerityLogger:
    """The logger of the package's module ``name`` (pass ``__name__``), under the package's handler."""
    if name != PACKAGE and not name.startswith(PACKAGE + "."):
        raise ValueError(f"init_logger({name!r}): pass the `__name__` of a module of {PACKAGE}, whose logger is under the "
                         "package's handler")
    logger = logging.getLogger(name)
    for method in ONCE:
        setattr(logger, method, MethodType(getattr(VerityLogger, method), logger))
    return cast(VerityLogger, logger)


def handler() -> logging.Handler:
    """The package logger's handler, added once however often this module is imported."""
    logger = logging.getLogger(PACKAGE)
    mine = [h for h in logger.handlers if h.get_name() == PACKAGE]
    if mine:
        return mine[0]
    h = Handler()
    logger.addHandler(h)
    if logger.level == logging.NOTSET:
        logger.setLevel(DEFAULT_LEVEL)
    return h


def configure(level: int | str = DEFAULT_LEVEL, stream: TextIO | None = None) -> None:
    """Set the package's level (a number, or a name such as ``"DEBUG"``) and its handler's stream (None: ``sys.stderr`` at
    each record).  A program's entry point calls it; the library never does."""
    logging.getLogger(PACKAGE).setLevel(level)
    cast(Handler, handler()).stream = stream


handler()

__all__ = ["VerityLogger", "configure", "init_logger"]
