"""serve-openai's day (``--day DAY --day-log LOG --day-program DIGEST``): before the day's first window, the server declares
the one circuit it serves that day in the deployment's registration log, and every window it ties names that declaration.

* **The declaration.** A ``verity/registration/declaration/v1`` entry (`verity.hardware.registration.Declaration`):
  the day, by its index among the monitoring period's days (the caller's, on the auditor's clock), the SHA-512 of the
  program the day's windows register (``--day-program``: under ``--h1-window-commit`` the audited window circuit's, R's
  ``program``), and its block Definitions by descriptor key (``--day-blocks``).
* **The log.** LOG is the deployment's one registration log as committed (`registration.committed`: ``{"composite",
  "entries"}``), kept across serve runs. `declare` appends the declaration to it only once core's own check
  (`registration.check`, under ``--day-params``' auditor, else an empty one) accepts the whole log with it, so a second
  declaration of the day is refused by core's name, ``duplicate-declaration``, before the model loads and before anything
  is written. A log that already declares the day the same program and blocks is the day's declaration, read again (a
  server restarted within its day); one that declares it another is the refusal. A new log takes ``--day-params``'
  composite. The read, the check and the write hold an exclusive lock on ``LOG.lock``, so two servers of one deployment
  can't both declare a day.
* **Where it is registered.** The serve dir's `FILE` (the entry, the digest windows name it by and the log's head after
  it), ``serve.json``'s ``day``, and, under ``--h1-window-commit``, the first line of the window log ``h1/log.jsonl``
  (`serve_tie`, kind ``declaration``), which the windows' W and R then follow.
* **Each window names it.** Each W (`pouw_h1_commit.registration`) carries ``window.day`` and ``window.declaration``, the
  declaration's digest with no ``previous`` (`named`), which R2 rebuilds from the day, program and blocks of the log's
  entry or of the challenger device's signed one, so both paths name it alike. Before an R is named the window's Commit
  record, its ``program`` must be the day's (`Day.refusal`): the server's own guard, the window's visible refusal. What counts is R2's (sampled proofs' ``R2-declared``, which the audit
  runs with the day and the log's declarations).
* **The registered compiler.** The log registers B once, as an information entry's item of kind ``builder``. The run
  knows its compiler where it binds its advice (`advice.write`), so that is where `register_compiler` registers the
  item, in the log the run's `FILE` names, before any advice is bound: under the log's lock, appended only once core's
  check accepts the log with it, and only to the log that holds the run's declaration. A log that already registers the
  same item is read again, unchanged; one that registers another compiler refuses this one by core's name,
  ``duplicate-information``, and the run binds no advice under a compiler its log doesn't register.

Without ``--day`` nothing here runs, and every byte the server writes is as before.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.hardware import site as S
from verity.hardware import registration as G
from verity_vllm.config import UsageError, option

#: the serve dir's record of the day's declaration
FILE = "declaration.json"


@dataclass(frozen=True, kw_only=True)
class DayOptions:
    """serve-openai's ``--day`` group (`declare`)."""

    day: int | None = option("--day", type=int, default=None, metavar="N",
                             help="declare day N's one circuit in --day-log before the first window, and name it in each "
                                  "window's registration (serve_day); a second declaration of day N is refused before the "
                                  "model loads (default: none, every byte as without it)")
    day_log: str | None = option("--day-log", default=None, metavar="LOG",
                                 help="with --day: the deployment's registration log (verity/registration/log/v1, committed "
                                      "JSON), appended to in place")
    day_program: str | None = option("--day-program", default=None, metavar="HEX",
                                     help="with --day: the SHA-512 of the program the day's windows register (under "
                                          "--h1-window-commit, the audited window circuit's)")
    day_blocks: str = option("--day-blocks", default="", metavar="KEYS",
                             help="with --day: the program's block Definitions by descriptor key, comma-separated")
    day_params: str | None = option("--day-params", default=None, metavar="FILE",
                                    help="with --day: the log's parameters, {\"auditor\", \"composite\"} "
                                         "(registration.params_of); needed for a new log, and for one with site entries")


@dataclass(frozen=True)
class Day:
    """The day's declaration as the log holds it: the entry, the digest windows name it by (`named`), the log's head, and
    whether this server wrote it (``now``) or a server before it that day (``earlier``)."""

    entry: G.Declaration
    digest: str
    head: str
    log: Path
    declared: str

    @property
    def day(self) -> int:
        return self.entry.day

    @property
    def program(self) -> str:
        return self.entry.program

    def naming(self) -> dict[str, Any]:
        """What a window's registration carries of it, in its ``window``."""
        return {"day": self.entry.day, "declaration": self.digest}

    def refusal(self, record: Any) -> str | None:
        """Why ``record`` can't be one of the day's windows (its ``program`` isn't the day's), or None."""
        got = record.get("program") if isinstance(record, Mapping) else None
        if got == self.program:
            return None
        return (f"the record's program {str(got)[:16]}… is not day {self.day}'s declared {self.program[:16]}…: R2 refuses "
                "it (R2-declared)")

    def describe(self) -> dict[str, Any]:
        return {"day": self.entry.day, "program": self.entry.program, "blocks": list(self.entry.blocks),
                "declaration": self.digest, "log": str(self.log), "head": self.head, "declared": self.declared,
                "file": FILE}

    def line(self) -> dict[str, Any]:
        """Its line in a window log (`serve_tie`)."""
        return {"window": None, "kind": "declaration", "file": f"../{FILE}", "digest": self.digest}


def refusal(opts: Any) -> str | None:
    """Why ``--day``'s options can't declare (before the model loads), or None."""
    d = opts.day
    if d.day is None:
        given = [f for f, v in (("--day-log", d.day_log), ("--day-program", d.day_program), ("--day-params", d.day_params),
                                ("--day-blocks", d.day_blocks)) if v]
        return f"{', '.join(given)} without --day" if given else None
    if d.day < 0:
        return f"--day {d.day}: a day is its index among the monitoring period's days, at least 0"
    if not d.day_log or not d.day_program:
        return "--day needs --day-log (the deployment's registration log) and --day-program (the day's program digest)"
    if not re.fullmatch("[0-9a-f]{128}", d.day_program):
        return f"--day-program {d.day_program[:20]}…: not a SHA-512 digest in lowercase hex"
    return None


def blocks_of(text: str) -> tuple[str, ...]:
    return tuple(b for b in (x.strip() for x in text.split(",")) if b)


def declare(opts: Any, out: Path) -> Day | None:
    """``--day``'s declaration (module docstring), its record written to ``out/FILE``; None without ``--day``.
    `UsageError` naming core's refusal when the log refuses it."""
    d = opts.day
    if d.day is None:
        return None
    params = None
    if d.day_params:
        try:
            params = G.params_of(json.loads(Path(d.day_params).read_text()))
        except (OSError, ValueError) as e:
            raise UsageError(f"--day-params {d.day_params}: {e}") from None
    try:
        day = append(Path(d.day_log), G.Declaration(d.day, d.day_program, blocks_of(d.day_blocks)), params)
    except G.Refused as e:
        raise UsageError(f"--day {d.day}: the registration log {d.day_log} refuses its declaration, {e}: day {d.day} is "
                         "declared already, to another circuit, and a day has one") from None
    except (OSError, ValueError) as e:
        raise UsageError(f"--day-log {d.day_log}: {e}") from None
    (out / FILE).write_text(json.dumps({**day.describe(), "entry": day.entry.to_json()}, indent=1, sort_keys=True) + "\n")
    return day


def append(path: Path, declaration: G.Declaration, params: tuple[str, S.Auditor] | None = None) -> Day:
    """``declaration`` appended to the log at ``path`` once core's check accepts the log with it, under the log's lock: the
    day's `Day`. A log that already declares the day the same circuit is read again, unchanged. `registration.Refused`
    when the check refuses it (``duplicate-declaration`` for another circuit that day), ValueError for a log that isn't
    one or isn't ``params``' composite."""
    with _held(path, params) as (composite, log, auditor):
        same = [e for e in log if isinstance(e, G.Declaration) and e.day == declaration.day
                and (e.program, e.blocks) == (declaration.program, declaration.blocks)]
        if same:
            return Day(same[0], named(same[0]), G.head(composite, log), path, "earlier")
        entry = _chained(path, composite, log, auditor, declaration)
        return Day(entry, named(entry), entry.digest(), path, "now")


def named(entry: G.Declaration) -> str:
    """The digest a window names a day's declaration by: `registration.Declaration`'s with no ``previous``, the day, program
    and blocks alone, which R2 rebuilds alike from the prover's log and from the challenger device's signed entry. The log's
    own chained entry (its ``previous`` set) is the log's, not what a window names."""
    return G.Declaration(entry.day, entry.program, entry.blocks).digest()


@dataclass(frozen=True)
class Item:
    """An information item as the log registers it: the item, the log's head, and whether this call registered it
    (``now``) or the log held it already (``earlier``)."""

    item: Mapping[str, Any]
    head: str
    log: Path
    registered: str

    def describe(self) -> dict[str, Any]:
        return {"item": dict(self.item), "log": str(self.log), "head": self.head, "registered": self.registered}


def register(path: Path, item: Mapping[str, Any], params: tuple[str, S.Auditor] | None = None,
             declaration: str | None = None) -> Item:
    """``item`` (an information entry's, as `registration.item_of` decodes it) registered in the log at ``path``, in an
    information entry of its own appended once core's check accepts the log with it, under the log's lock. A log that
    already registers the same item is read again, unchanged. Given ``declaration`` (the digest a window names a day's
    by, `named`), only a log that holds it takes the item. ValueError for a log that isn't there, isn't one, isn't
    ``params``' composite or doesn't hold ``declaration``, and, naming core's refusal (``duplicate-information``), for
    one that registers another item under this one's key (`registration.item_key`): the log registers each once."""
    item = G.item_of(dict(item))
    if not path.exists():
        raise ValueError("no log there")
    with _held(path, params) as (composite, log, auditor):
        if declaration is not None and declaration not in {named(e) for e in log if isinstance(e, G.Declaration)}:
            raise ValueError(f"the log holds no declaration {declaration[:16]}…")
        key = G.item_key(item)
        held = [dict(i) for e in log if isinstance(e, G.Information) for i in e.items if G.item_key(i) == key]
        if held == [item]:
            return Item(item, G.head(composite, log), path, "earlier")
        try:
            entry = _chained(path, composite, log, auditor, G.Information((item,)))
        except G.Refused as e:
            was = f": it registers {json.dumps(held[0], sort_keys=True)} already, and registers each item once" if held else ""
            raise ValueError(f"the log refuses the item, {e}{was}") from None
        return Item(item, entry.digest(), path, "now")


def register_compiler(run: Path, item: Mapping[str, Any], log: str | os.PathLike | None = None,
                      params: str | os.PathLike | None = None) -> Item | None:
    """B's item (kind ``builder``) registered in the log the serve run ``run`` declared its day in (module docstring):
    the log its `FILE` names, or ``log``, a copy of it, held to the declaration `FILE` names, under ``params`` (the
    log's parameters, as ``--day-params``). None for a run that declared no day and is given no ``log``: nothing is
    registered. ValueError, naming the log, where it isn't the run's or refuses the item (`register`)."""
    file = Path(run) / FILE
    if not file.exists():
        if log is None and params is None:
            return None
        raise ValueError(f"--day-log and --day-params go with a run that declared a day (serve-openai --day), and "
                         f"{run} holds no {FILE}")
    try:
        day = json.loads(file.read_text())
        path, declaration = Path(log if log is not None else day["log"]), str(day["declaration"])
        got = G.params_of(json.loads(Path(params).read_text())) if params is not None else None
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"the run's {FILE} and --day-params: {type(e).__name__}: {e}") from None
    try:
        return register(path, item, got, declaration)
    except (OSError, ValueError) as e:
        raise ValueError(f"the registration log {path} (the run's {FILE} names day {day.get('day')}'s declaration "
                         f"{declaration[:16]}… in it; --day-log gives a copy of it): {e}") from None


@contextmanager
def _held(path: Path, params: tuple[str, S.Auditor] | None) -> Iterator[tuple[str, list[G.Entry], S.Auditor]]:
    """The log at ``path`` under its exclusive lock, read and held to core's check as it stands: its composite, its
    entries and the auditor it is checked under (``params``', else an empty one)."""
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        composite, log = _read(path, params)
        auditor = params[1] if params is not None else S.Auditor()
        try:
            G.check(auditor, composite, log)
        except G.Refused as e:
            raise ValueError(f"the log as it stands is refused, {e} (its site entries need --day-params' auditor)") from None
        yield composite, log, auditor


def _chained(path: Path, composite: str, log: list[G.Entry], auditor: S.Auditor, entry: G.Entry) -> G.Entry:
    """``entry`` chained after ``log`` and the log written with it, once core's check accepts the log with it:
    `registration.Refused` otherwise, and nothing written."""
    entry = G.chain(composite, [*log, entry])[-1]
    G.check(auditor, composite, [*log, entry])
    _write(path, G.committed(composite, [*log, entry]))
    return entry


def _read(path: Path, params: tuple[str, S.Auditor] | None) -> tuple[str, list[G.Entry]]:
    if not path.exists():
        if params is None:
            raise ValueError("no log there yet: a new log takes its composite from --day-params")
        return params[0], []
    composite, log = G.log_of(json.loads(path.read_text()))
    if params is not None and params[0] != composite:
        raise ValueError(f"the log is {composite}'s, not --day-params' {params[0]}'s")
    return composite, log


def _write(path: Path, body: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)


__all__ = ["FILE", "Day", "DayOptions", "Item", "append", "declare", "named", "refusal", "register", "register_compiler"]
