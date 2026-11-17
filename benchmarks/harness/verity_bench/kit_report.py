"""The core benchmarks' interface as code a command calls (`benchmarks/README.md`'s "The core benchmarks' interface"),
re-exported by `kit`:

- `record_dir(arg)`: what a command takes, a run's record as a directory, read with nothing of ours; a run id is fetched
  through research's store client where it is installed, past its withdrawals, and refused in one line naming the fix
  where it is not. `record(arg)` gives the same directory and what was withdrawn of the rest of the record.
- `ran_with(coins, parameters, detail=None, verified=False)`: the one line of the coins and parameters a run ran with,
  in the fixed words (`COINS`, `PARAMETERS`), and its object for the document.
- `report(name, text, doc, measured, ...)`: a command's last step: prints the text and the one line of JSON, writes them
  where there is a place, gives no number for a run the Lean verifier didn't accept, says where it is not an audit and
  where its record is made up, and writes no measurement where it reprints another run's record (`from_run=`).
"""

from __future__ import annotations

import enum
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from verity_bench import settings

#: who drew the verifier's coins: the auditor's deployed challenger, a challenger as its own process on this machine,
#: or the serve itself holding the challenger
COINS = ("deployed challenger", "local challenger", "in-process")
#: the parameters the run ran at: the deployed ones, or development ones
PARAMETERS = ("deployed", "development")
#: the line of a record that predates the field, the one record that may not say
NOT_STATED = "ran with: not stated"
NOT_AN_AUDIT = "a measurement, not an audit"
MADE_UP = "a made-up record, not a run's"
#: the mark of a made-up record's measurements: figures no run measured (`kit.MARKS`)
MADE_UP_MARK = "modeled"
WITHDRAWN_LINE = "  part of the run's record was withdrawn, none of it a file read here: "
#: the line under the audit line where the deployed challenger's coins are only the record's word: the command didn't
#: check the challenger's signature against a pinned key itself
OWN_WORD = "coins by the record's own word, not checked"
#: how the store ends the reason of a withdrawal of only the files under some prefixes of a record
SCOPED = re.compile(r" \(only ([^()]+)\)$")
#: what `report` says, on stderr, where it writes no measurement of a reprint
REPRINT = "{name}: a reprint of run {run}'s record: its numbers are that run's measurements, and this writes none"
#: research's run ids, the one form `record_dir` reads through the store
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")


class _Unsaid(enum.Enum):
    UNSAID = enum.auto()


_UNSAID = _Unsaid.UNSAID


class NoRecord(ValueError):
    """A `--from` that names no record this machine can read; its message is one line and names the fix."""


class Record(NamedTuple):
    """A run's record as `record` gives it: its directory, and what was withdrawn of the rest of it, each as its label
    says it, which the command passes to `report` (`withdrawn=`)."""
    path: Path
    withdrawn: list[str]


def _under(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")


def record_dir(arg: str | os.PathLike, *, paths: list[str] | None = None) -> Path:
    """The directory of the run's record `arg` names, as `record` gives it, without what was withdrawn of the rest."""
    return record(arg, paths=paths).path


def record(arg: str | os.PathLike, *, paths: list[str] | None = None) -> Record:
    """The run's record `arg` names: a directory as given, imported and fetched through nothing; a run id (`RUN_ID`)
    its record fetched through research's store client (only `paths` of it where given, each a file's whole path in
    the run). Anything else, or a run id where research isn't installed or holds no record of it, is refused
    (`NoRecord`). A record withdrawn whole, or in a part that holds a file read (every file where `paths` is None), is
    refused before any of it is fetched, as `benchmarks/compute/run.py`'s `EvidenceStore.files` refuses it; a
    withdrawal of only other files of it is the `Record`'s `withdrawn`."""
    path = Path(arg)
    if path.is_dir():
        return Record(path, [])
    run = str(arg)
    if not RUN_ID.fullmatch(run):
        raise NoRecord(f"{arg}: no such directory, and not a run id: give --from the directory of a run's record")
    try:
        from research.store.labels_sync import read_through
        from research.store.local import LocalStore, StoreError
        from research.store.vocab import WITHDRAWN
    except ImportError:
        raise NoRecord(f"{arg}: a run id is read through research, which isn't installed here: give --from the run's "
                       "record directory, or install research (uv sync --all-packages)") from None
    fix = "give --from the run's record directory"
    try:
        store = LocalStore()
        try:
            attempt = store.get_attempt(run)
        except ValueError as e:
            raise StoreError(str(e)) from e
        outputs = (attempt or {}).get("outputs") or {}
        art = outputs.get("run_record") or outputs.get("run_files")
        if not art:
            raise NoRecord(f"{arg}: the evidence store holds no record of it: {fix}")
        read_through(store, [run, art], lambda m: print(f"record_dir: {m}", file=sys.stderr))
        names = [f["path"] for f in store.listing(art)]
        read = names if paths is None else paths
        gone = []
        for label in (lab for target in (art, run) for lab in store.labels(target, key=WITHDRAWN)):
            said = f"{label.value!r}, by {label.by}"
            scoped = SCOPED.search(str(label.value))
            prefixes = scoped.group(1).split(", ") if scoped else []
            # a reason can end like a list of prefixes without being one: each must hold a file of this record
            partial = bool(prefixes) and all(any(_under(n, p) for n in names) for p in prefixes)
            if not partial or any(_under(f, p) for f in read for p in prefixes):
                raise NoRecord(f"{arg} was withdrawn ({said}): nothing of it is fetched")
            gone.append(said)
        return Record(Path(store.fetch(art, paths=paths)), gone)
    except StoreError as e:
        raise NoRecord(f"{arg}: the evidence store can't give its record ({e}): {fix}") from None


def ran_with(coins: str, parameters: str, detail: str | None = None, *,
             verified: bool = False) -> tuple[str, dict[str, Any]]:
    """The run's coins and parameters as its one line, ``ran with: coins COINS; parameters PARAMETERS[ (DETAIL)]``, and
    as the document's ``ran_with``; `detail` is the command's own words (its sample, δ, scheme, key). Words other than
    `COINS`' and `PARAMETERS`' are refused. `verified` is True only where the command itself checked the challenger's
    signature on the coins against a pinned key; the deployed challenger's coins without it are the record's own word,
    which the object holds as ``verified: false`` and `report` states (`OWN_WORD`) and never takes for an audit."""
    if coins not in COINS:
        raise ValueError(f"coins {coins!r}: one of {', '.join(map(repr, COINS))}")
    if parameters not in PARAMETERS:
        raise ValueError(f"parameters {parameters!r}: one of {', '.join(map(repr, PARAMETERS))}")
    if verified is not True and verified is not False:
        raise ValueError(f"verified is True or False, not {verified!r}")
    line = f"ran with: coins {coins}; parameters {parameters}" + (f" ({detail})" if detail else "")
    said = {"coins": coins, "parameters": parameters, "detail": detail}
    return line, said | ({"verified": verified} if verified or coins == COINS[0] else {})


#: `ran_with`, under a name `report`'s parameter of that name doesn't hide
_pair = ran_with


def report(name: str, text: str, doc: Mapping[str, Any], measured: Iterable[Mapping[str, Any]] = (), *,
           document: bool = True, mark: str | None = None, out: str | os.PathLike | None = None,
           ran_with: tuple[str, Mapping[str, Any]] | None | _Unsaid = _UNSAID, accepted: bool | None = None,
           why: str | None = None, made_up: bool = False, withdrawn: Sequence[str] = (),
           from_run: str | None = None) -> int:
    """A command's last step, the one way it prints and writes its number: prints `text`, then `doc` as one line of
    JSON, and where it has a place to write (`out` where the caller names one, its `--out`; else under a research run its
    `out/`, `$RESEARCH_RUN_DIR/out`, and outside one `$OUT` where set) writes that line to `<name>.json` there and
    appends each of `measured`, the keywords of a `Writer.measurement`, at its own ``mark`` where it gives one (a chosen
    parameter's, which no run measured), else at `mark` (``"measured"`` where None), to the run's result (`kit.writer`). An empty variable names no place. With `document` False the place keeps the
    `<name>.json` it holds.

    `ran_with` (`ran_with()`'s pair, or None for a record that predates the field, printed as `NOT_STATED`) is printed
    after the text's first line and is the document's ``ran_with``; below it, `NOT_AN_AUDIT` unless the coins are the
    deployed challenger's, verified, and the parameters deployed, and the document's ``audit``; below that `OWN_WORD`
    where the deployed challenger's coins aren't verified. With `made_up` (a record
    made up for a test or a fixture, not a run's) `MADE_UP` is printed below those, the document holds ``made_up: true``
    and is no audit, and each measurement is written at `MADE_UP_MARK` with ``made_up: true``. With `accepted` False (the
    Lean verifier didn't accept the run) the first line is ``NAME: no number: WHY``, in place of the text's headline (its
    first line where that opens with ``NAME:``) and above the rest of the text, the document's value is null beside
    ``no_number``, and each measurement is written with no value and `why`. `withdrawn` (`record_dir`'s) is printed
    last and is the document's ``withdrawn``. `from_run` is the run whose record the command read; where it names a run
    other than this one (`RESEARCH_RUN_ID`) the command is a reprint, and `report` prints and writes the document as
    above but writes no measurement, saying so on stderr (`REPRINT`): a number reaches the Metrics tab once, from the run
    that measured it, with that run's commit. Given none of these, it prints and writes as it did before them. Its exit
    code: 1 with no number, else 0."""
    if from_run is not None and (not isinstance(from_run, str) or not from_run):
        raise ValueError(f"from_run names the run whose record was read, a run id, not {from_run!r}")
    lines = text.split("\n")
    doc = dict(doc)
    if made_up:
        if mark not in (None, MADE_UP_MARK):
            raise ValueError(f"a made-up record's measurements are marked {MADE_UP_MARK!r}, figures no run measured, not "
                             f"{mark!r}: drop mark=")
        mark = MADE_UP_MARK
        measured = [{**m, "mark": mark, "extra": {**(m.get("extra") or {}), "made_up": True}} for m in measured]
    if accepted is False:
        if not why:
            raise ValueError("a run the Lean verifier didn't accept prints no number and says why: give why")
        # the headline is the text's first line opening with the command's name; the no-number line takes its place
        lines = [f"{name}: no number: {why}", *(lines[lines[0].startswith(f"{name}:"):] if text else [])]
        doc |= {"value": None, "no_number": why}
        measured = [{k: v for k, v in m.items() if k not in ("value", "interval", "verdict", "why")} | {"value": None,
                                                                                                         "why": why}
                    for m in measured]
    under = []
    if ran_with is not _UNSAID:
        line, said = (NOT_STATED, None) if ran_with is None else ran_with
        if said is not None and _pair(said["coins"], said["parameters"], said.get("detail"),
                                      verified=said.get("verified", False)) != (line, said):
            raise ValueError(f"ran_with is ran_with()'s pair, not {ran_with!r}")
        audit = (not made_up and said is not None and said.get("verified") is True
                 and (said["coins"], said["parameters"]) == (COINS[0], PARAMETERS[0]))
        own_word = said is not None and said["coins"] == COINS[0] and said.get("verified") is not True
        under = [line, *([] if audit else [NOT_AN_AUDIT]), *([OWN_WORD] if own_word else [])]
        doc |= {"ran_with": None if said is None else dict(said), "audit": audit}
    if made_up:
        under.append(MADE_UP)
        doc["made_up"] = True
    lines[1:1] = under
    if withdrawn:
        lines.append(WITHDRAWN_LINE + "; ".join(withdrawn))
        doc["withdrawn"] = list(withdrawn)
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
        measured = list(measured)
        if measured and from_run is not None and from_run != settings.RESEARCH_RUN_ID:
            print(REPRINT.format(name=name, run=from_run), file=sys.stderr)
            measured = []
        if measured:
            # kit re-exports this module, so it is imported here and not above
            from verity_bench.kit import writer

            w = writer(place)
            for m in measured:
                w.measurement(**{"mark": mark or "measured", **m})
    return 1 if accepted is False else 0
