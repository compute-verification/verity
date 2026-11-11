"""Durable raw observation log: one JSON object per line, optionally gzip-compressed (`*.jsonl.gz`).

`fold(read_log(path), profile)` must work with no GPU and no vLLM process: this module imports no torch.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import IO, Iterable, Iterator

from .events import Event, RunHeader, StepEnd, StepStart, from_json, to_json


def _open(path: str | Path, mode: str) -> IO:
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode + "t", encoding="utf-8")
    return open(p, mode, encoding="utf-8")


class LogWriter:
    """Append-only writer.  `write` is cheap (one `json.dumps` per event); call `flush` at step boundaries."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: IO = _open(self.path, "w")
        self.count = 0

    def write(self, ev: Event) -> None:
        self._f.write(json.dumps(to_json(ev), separators=(",", ":"), ensure_ascii=True))
        self._f.write("\n")
        self.count += 1

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "LogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path) -> Iterator[Event]:
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield from_json(json.loads(line))


def read_log_lines(lines: Iterable[str]) -> Iterator[Event]:
    for line in lines:
        line = line.strip()
        if line:
            yield from_json(json.loads(line))


def dumps(events: Iterable[Event]) -> str:
    buf = io.StringIO()
    for ev in events:
        buf.write(json.dumps(to_json(ev), separators=(",", ":"), ensure_ascii=True))
        buf.write("\n")
    return buf.getvalue()


def header_of(events: Iterable[Event]) -> RunHeader:
    for ev in events:
        if isinstance(ev, RunHeader):
            return ev
        break
    raise ValueError("log does not start with a RunHeader")


def split_steps(events: Iterable[Event]) -> Iterator[tuple[list[Event], list[Event]]]:
    """Yield `(preamble, step_events)` groups: `preamble` are the events since the previous StepEnd that precede
    this StepStart (roots, allocations, notes); `step_events` run from StepStart through StepEnd inclusive.
    A trailing preamble (after the last step) is yielded with an empty step list."""
    pre: list[Event] = []
    cur: list[Event] | None = None
    for ev in events:
        if isinstance(ev, StepStart):
            cur = [ev]
        elif cur is not None:
            cur.append(ev)
            if isinstance(ev, StepEnd):
                yield pre, cur
                pre, cur = [], None
        else:
            pre.append(ev)
    if pre or cur:
        yield pre, cur or []
