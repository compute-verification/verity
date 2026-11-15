"""The core benchmarks' interface as code a command calls (`benchmarks/README.md`'s "The core benchmarks' interface"),
re-exported by `kit`:

- `record_dir(arg)`: what a command takes, a run's record as a directory, read with nothing of ours; a run id is fetched
  through research's store client where it is installed, and refused in one line naming the fix where it is not.
- `ran_with(coins, parameters, detail=None)`: the one line of the coins and parameters a run ran with, in the fixed
  words (`COINS`, `PARAMETERS`), and its object for the document.
- `report(name, text, doc, measured, ...)`: a command's last step: prints the text and the one line of JSON, writes them
  where there is a place, gives no number for a run the Lean verifier didn't accept, and says where it is not an audit.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from verity_bench import settings

#: who drew the verifier's coins: the auditor's deployed challenger, a challenger as its own process on this machine,
#: or the serve itself holding the challenger
COINS = ("deployed challenger", "local challenger", "in-process")
#: the parameters the run ran at: the deployed ones, or development ones
PARAMETERS = ("deployed", "development")
#: the line of a record that predates the field, the one record that may not say
NOT_STATED = "ran with: not stated"
NOT_AN_AUDIT = "a measurement, not an audit"
#: research's run ids, the one form `record_dir` reads through the store
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
_UNSAID = object()


class NoRecord(ValueError):
    """A `--from` that names no record this machine can read; its message is one line and names the fix."""


def record_dir(arg: str | os.PathLike, *, paths: list[str] | None = None) -> Path:
    """The directory of the run's record `arg` names: a directory as given, imported and fetched through nothing; a
    run id (`RUN_ID`) its record fetched through research's store client (only `paths` of it where given). Anything
    else, or a run id where research isn't installed or holds no record of it, is refused (`NoRecord`)."""
    path = Path(arg)
    if path.is_dir():
        return path
    if not RUN_ID.fullmatch(str(arg)):
        raise NoRecord(f"{arg}: no such directory, and not a run id: give --from the directory of a run's record")
    try:
        from research.store.local import LocalStore, StoreError
    except ImportError:
        raise NoRecord(f"{arg}: a run id is read through research, which isn't installed here: give --from the run's "
                       "record directory, or install research (uv sync --all-packages)") from None
    try:
        store = LocalStore()
        art = ((store.get_attempt(str(arg)) or {}).get("outputs") or {}).get("run_record")
        if not art:
            raise NoRecord(f"{arg}: the evidence store holds no record of it: give --from the run's record directory")
        return Path(store.fetch(art, paths=paths))
    except StoreError as e:
        raise NoRecord(f"{arg}: the evidence store can't give its record ({e}): give --from the run's record "
                       "directory") from None


def ran_with(coins: str, parameters: str, detail: str | None = None) -> tuple[str, dict[str, Any]]:
    """The run's coins and parameters as its one line, ``ran with: coins COINS; parameters PARAMETERS[ (DETAIL)]``, and
    as the document's ``ran_with``; `detail` is the command's own words (draws, δ, scheme, key). Words other than
    `COINS`' and `PARAMETERS`' are refused."""
    if coins not in COINS:
        raise ValueError(f"coins {coins!r}: one of {', '.join(map(repr, COINS))}")
    if parameters not in PARAMETERS:
        raise ValueError(f"parameters {parameters!r}: one of {', '.join(map(repr, PARAMETERS))}")
    line = f"ran with: coins {coins}; parameters {parameters}" + (f" ({detail})" if detail else "")
    return line, {"coins": coins, "parameters": parameters, "detail": detail}


def report(name: str, text: str, doc: Mapping[str, Any], measured: Iterable[Mapping[str, Any]] = (), *,
           document: bool = True, mark: str = "measured", out: str | os.PathLike | None = None,
           ran_with: tuple[str, Mapping[str, Any]] | None | object = _UNSAID, accepted: bool | None = None,
           why: str | None = None) -> int:
    """A command's last step, the one way it prints and writes its number: prints `text`, then `doc` as one line of
    JSON, and where it has a place to write (`out` where the caller names one, its `--out`; else under a research run its
    `out/`, `$RESEARCH_RUN_DIR/out`, and outside one `$OUT` where set) writes that line to `<name>.json` there and
    appends each of `measured`, the keywords of a `Writer.measurement`, at `mark` to the run's result (`kit.writer`). An
    empty variable names no place. With `document` False the place keeps the `<name>.json` it holds.

    `ran_with` (`ran_with()`'s pair, or None for a record that predates the field, printed as `NOT_STATED`) is printed
    after the text's first line and is the document's ``ran_with``; below it, `NOT_AN_AUDIT` where the coins aren't the
    deployed challenger's or the parameters aren't deployed, and the document's ``audit``. With `accepted` False (the
    Lean verifier didn't accept the run) the first line is ``NAME: no number: WHY``, the document's value is null beside
    ``no_number``, and each measurement is written with no value and `why`. Given none of these, it prints and writes as
    it did before them. Its exit code: 1 with no number, else 0."""
    lines = text.split("\n")
    doc = dict(doc)
    if accepted is False:
        if not why:
            raise ValueError("a run the Lean verifier didn't accept prints no number and says why: give why")
        lines = [f"{name}: no number: {why}", *(lines if text else [])]
        doc |= {"value": None, "no_number": why}
        measured = [{k: v for k, v in m.items() if k not in ("value", "interval", "verdict", "why")} | {"value": None,
                                                                                                         "why": why}
                    for m in measured]
    if ran_with is not _UNSAID:
        line, said = (NOT_STATED, None) if ran_with is None else ran_with
        audit = said is not None and (said["coins"], said["parameters"]) == (COINS[0], PARAMETERS[0])
        lines[1:1] = [line, *([] if audit else [NOT_AN_AUDIT])]
        doc |= {"ran_with": None if said is None else dict(said), "audit": audit}
    print("\n".join(lines))
    line = json.dumps(doc)
    print(line)
    if out is not None:
        place = out
    else:
        place = Path(settings.RESEARCH_RUN_DIR) / "out" if settings.is_set("RESEARCH_RUN_DIR") else settings.OUT
    if place:
        place = Path(place)
        if document:
            place.mkdir(parents=True, exist_ok=True)
            (place / f"{name}.json").write_text(line + "\n")
        if measured := list(measured):
            from verity_bench.kit import writer  # kit re-exports this module, so it is imported here and not above

            w = writer(place)
            for m in measured:
                w.measurement(**m, mark=mark)
    return 1 if accepted is False else 0
