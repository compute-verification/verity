"""Adopt a measured census, such as an opcode probe's `census.csv` (`benchmarks/pouw/opcode_probe`), at the last epoch.

Each (kind, row) keeps the faster of its line at the epoch before and the measured one: a row's u is what every program
pays on it at least, so the lower measurement is the one every program satisfies. A faster measurement closes the old
line at the epoch before, adds its own from the new epoch and adds its runs to its row's; a slower one changes nothing.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from verity_machine_model.model import DATA, holds, model, rows, u_of

COLUMNS = ("kind", "row", "rate", "noise", "cycles", "u", "since", "until", "runs")
ROW_COLUMNS = ("row", "status", "runs", "description")


@dataclass(frozen=True)
class Adoption:
    kind: str
    row: str
    before: float | None    # u at the epoch before (None: a (kind, row) the census didn't have)
    measured: float
    adopted: bool


def _read(path: Path, columns: tuple[str, ...] = COLUMNS) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if tuple(r.fieldnames or ()) != columns:
            raise SystemExit(f"{path}: the columns are {','.join(columns)}, not {','.join(r.fieldnames or ())}")
        return list(r)


def _write(path: Path, columns: tuple[str, ...], lines: list[dict[str, str]]) -> None:
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    w.writerows(lines)
    path.write_text(s.getvalue())


def adopt(path: Path, epoch: str, data: Path = DATA) -> list[Adoption]:
    """Merge `path`'s lines into `data/census.csv` at `epoch`, which must be the last of model.toml's epochs."""
    m = model(data)
    if len(m.epochs) < 2 or m.epochs[-1] != epoch:
        raise SystemExit(f"append {epoch!r} to model.toml's epochs first (adopt writes the last epoch; it is "
                         f"{m.epochs[-1]!r})")
    prev, known = m.epochs[-2], rows(data)
    lines, new = _read(data / "census.csv"), _read(path)
    u = lambda d: u_of(m.cycle_u, d["rate"], d["noise"], d["cycles"], d["u"])
    out: list[Adoption] = []
    after: dict[int | None, list[dict[str, str]]] = {}
    seen = set()
    for d in new:
        key = (d["kind"], d["row"])
        if key in seen:
            raise SystemExit(f"{path}: two lines for {d['kind']} on {d['row']}")
        seen.add(key)
        if d["row"] not in known:
            raise SystemExit(f"{d['kind']} on {d['row']}: add the row to rows.csv first")
        if d["since"] or d["until"]:
            raise SystemExit(f"{d['kind']} on {d['row']}: a measured line has no epochs; adopt sets them")
        if not d["runs"]:
            raise SystemExit(f"{d['kind']} on {d['row']}: a measured line names its runs")
        if any((e["kind"], e["row"]) == key and e["since"] == epoch for e in lines):
            raise SystemExit(f"{d['kind']} on {d['row']} already has a line from {epoch}")
        held = [i for i, e in enumerate(lines) if (e["kind"], e["row"]) == key and holds(m, e["since"], e["until"], prev)]
        before = u(lines[held[0]]) if held else None
        if before is not None and before <= u(d):
            out.append(Adoption(d["kind"], d["row"], before, u(d), False))
            continue
        if held:
            if lines[held[0]]["until"]:
                raise SystemExit(f"{d['kind']} on {d['row']}: its line closes at {lines[held[0]]['until']} already")
            lines[held[0]]["until"] = prev
        after.setdefault(held[0] if held else None, []).append(dict(d, since=epoch, until=""))
        out.append(Adoption(d["kind"], d["row"], before, u(d), True))
    merged = []
    for i, e in enumerate(lines):
        merged += [e, *after.get(i, [])]
    merged += after.get(None, [])
    _write(data / "census.csv", COLUMNS, merged)
    runs: dict[str, list[str]] = {}
    for d in (d for ds in after.values() for d in ds):
        runs.setdefault(d["row"], []).extend(d["runs"].split())
    table = _read(data / "rows.csv", ROW_COLUMNS)
    for r in table:
        have = r["runs"].split()
        r["runs"] = " ".join(have + [x for x in dict.fromkeys(runs.get(r["row"], [])) if x not in have])
    _write(data / "rows.csv", ROW_COLUMNS, table)
    return out
