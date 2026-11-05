"""The price row a probe writes, in the machine model's row schema (`data/census/EPOCH.csv`,
read by `verity_machine_model.model`).

A census line is ``kind,row,rate,noise,cycles,u,since,until,runs``: the instruction class, the machine row it holds, its
measured rate (warp instructions/SM/clk, 4 SMSPs an SM) derated by `noise`, the epochs it holds and its runs. `u_of`
is machine_model's own reading of a line (exactly one of rate, cycles and u). The provenance a census line has no column
for goes in the sidecar `rows.jsonl`, a line a probe: the class, the rate, MACs a word, the run id and the kernel's SASS
hash, with the census line it produced. When machine_model lands on main, CENSUS_COLUMNS and u_of become its imports.

MACs a word is how many tensor MACs one word moved by the instruction costs at the row's price: ``4 cycle_u / (rate
MAC_U) / words``, with cycle_u = 300.08 (model.toml's R*: u in one SMSP-cycle) and MAC_U = 0.15922 u, VF3e's MAC price,
which the VF3f-VF3j solvers' ``macs()`` use (the vf3j note's SHFL at 0.499815 is 471 a word; a 16-B TMA box at 0.027522
is 68,480).
"""
from __future__ import annotations

import csv
import io
import re

CENSUS_COLUMNS = ("kind", "row", "rate", "noise", "cycles", "u", "since", "until", "runs")
CYCLE_U = 300.08
MAC_U = 0.15922
NOISE = "0.02"
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")


class RowError(ValueError):
    pass


def u_of(cycle_u: float, rate: str, noise: str, cycles: str, u: str) -> float:
    """machine_model's `model.u_of`: the u one warp instruction holds its row, from exactly one of rate, cycles and u."""
    given = [c for c in (rate, cycles, u) if c != ""]
    if len(given) != 1:
        raise RowError(f"a census line gives exactly one of rate, cycles and u, not {given}")
    if rate != "":
        r = float(rate)
        return cycle_u * 4 / (r * (1 + float(noise))) if noise != "" else cycle_u * 4 / r
    if noise != "":
        raise RowError("noise derates a rate only")
    return float(cycles) * cycle_u if cycles != "" else float(u)


def macs_a_word(rate: float, words: float, cycle_u: float = CYCLE_U, mac_u: float = MAC_U) -> float:
    return 4 * cycle_u / (rate * mac_u) / words


def census_line(kind: str, row: str, rate: float, run: str, since: str = "", noise: str = NOISE) -> dict[str, str]:
    """One measured census line; refused unless machine_model would read it (u_of) and it names its run."""
    if not kind or not row:
        raise RowError(f"a census line needs its kind and row (got {kind!r}, {row!r})")
    if not RUN_ID.fullmatch(run):
        raise RowError(f"runs {run!r}: a census line names the run that measured it (rYYYYMMDD-HHMMSS-xxxx; run the probe "
                       "under research run, which sets RESEARCH_RUN_ID)")
    if not rate > 0:
        raise RowError(f"{kind}: rate {rate}: a measured rate is positive")
    line = {"kind": kind, "row": row, "rate": f"{rate:.6g}", "noise": noise, "cycles": "", "u": "", "since": since,
            "until": "", "runs": run}
    u_of(CYCLE_U, line["rate"], line["noise"], line["cycles"], line["u"])
    return line


def census_csv(lines: list[dict[str, str]]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=CENSUS_COLUMNS, lineterminator="\n")
    w.writeheader()
    for line in lines:
        if tuple(line) != CENSUS_COLUMNS:
            raise RowError(f"a census line has the columns {CENSUS_COLUMNS}, not {tuple(line)}")
        w.writerow(line)
    return out.getvalue()
