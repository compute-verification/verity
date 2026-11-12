"""The rules for a steady-state row of the proof server: the ten checks a row of `benchmarks/one_stage/server_rate.py`
passes before its rate is a steady-state measurement, as a function of the run's own files. Standard library only, as
`overhead` is: a node and the store side can both read it.

**A run's files** (`Files`: a file's text by its path in the run's files, None where the run holds none; `files_in` gives
a directory's):

| file | written by | read for |
| --- | --- | --- |
| `out/rate/rate.json` (`RATES`) | `server_rate.py` | the rows, one a staged type and prover count |
| `out/rate/TYPE-kK/sessions.jsonl` (`sessions_file`) | `server_rate.py` | each session's index and filing time |
| `gpu-lease-usage.jsonl` (`LEASES`) | `gpu-lease`, when a lease ends | the run's leases and the cards each held |
| `out/cotenants-kK-start.txt`, `-end.txt` (`card_file`) | what launches the row | what else was on the node then |

**The checks** (`CHECKS`), by name, and what each reads:

- `accepted`: every session the row ran was accepted, none was refused and each was filed by its verifier. The row's
  `accepted` is `provers` x `warm` + `sessions`, and its `refused_n` and `unfiled` are 0.
- `exits`: every process of the row's server exited 0 (the row's `exits`).
- `not_void`: the row has a rate: no `void`, `error` or `skipped`, and a `statements_per_s` above 0.
- `flat`: the provers' statement caches had stopped filling (the row's `flat`, which the script decides).
- `no_throttling`: the kernel throttled the row's CPUs no times while it ran (the row's `throttled.nr_throttled`).
- `one_lease_own_card`: the run took one lease, of one card, that card is the row's `device`, and the row counts one
  GPU (`LEASES`, the row's `gpus`).
- `card_clear_start_end`: at the row's start and at its end, no other run's lease held the row's card and no compute
  process was on it (the two `card_file`s). The record is one a prover count, so it clears a row only where the run's
  `RATES` holds no other row at that count.
- `level`: the window the row rates, cut into three equal counts of sessions by their filing times, takes at most
  `LEVEL` times as long a statement in its slowest third as in its fastest (`sessions_file`). The window is rebuilt as
  the script cuts it, and must be the row's own: its `window_sessions` and `window_s`. A row whose sessions file in
  bursts declares the `period` they repeat over (`PERIOD`), and its window is cut at whole bursts instead
  (`burst_parts`): a third that cuts a burst reads the burst's phase and not a drift. A window that isn't whole bursts
  is cut twice, from its start and back from its end, so every session is in a part, and it is level when both cuts
  are.
- `every_prover_warmed`: each of the row's provers finished a warm session before any timed one was submitted, and none
  filed nothing in the window (the row's `warm_per_prover` and `idle_provers`).
- `one_numa_node`: the CPUs the row's processes could run on are all on one NUMA node (the row's `cores.numa_nodes`:
  the nodes that hold any CPU of the script's affinity, which every process of the row runs within). So a row whose
  verifiers are placed on another node than its provers fails. It reads the CPUs alone: the row doesn't say which node
  its card is on.

**Each check answers** `PASS`, `FAIL` or `CANNOT_TELL` (`Check`), the last where the run's files don't show it: a file
the run doesn't hold, a field the row lacks, a record that can't be read. Only a pass counts: a row passed
(`RowVerdict`) when all ten did, failed when any failed, and otherwise can't be told. `verdict` judges one row against
a run's files, and `verdicts` every row of a run's `RATES`. A number is a finite one: NaN and the infinities, which
JSON's readers let through, are no seconds and no rate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PASS, FAIL, CANNOT_TELL = "pass", "fail", "cannot tell"
#: the window's thirds: the slowest's seconds a statement over the fastest's, at most
LEVEL = 1.25
#: the row's field for the sessions a prover's filings repeat over, where the row's sessions file in bursts: a burst is
#: that many sessions a prover. A row without it, or with None, declares none
PERIOD = "period"
#: `sessions.jsonl` rounds each filing to a millisecond and the row rounds its window to ten: the window rebuilt from the
#: one is the other's when their seconds agree within this
WINDOW_SLACK_S = 0.011
RATES = "out/rate/rate.json"
LEASES = "gpu-lease-usage.jsonl"
#: a run's file's text by its path in the run's files, or None where the run holds no such file
Files = Callable[[str], str | None]


def sessions_file(row: Mapping[str, Any]) -> str:
    """Where a run keeps a row's sessions: one JSON record a line, with the session's index `i` (the warm ones first) and
    `filed`, the seconds at which its verifier filed it (null for one it never filed)."""

    return f"{RATES.rsplit('/', 1)[0]}/{row.get('type')}-k{row.get('provers')}/sessions.jsonl"


def card_file(row: Mapping[str, Any], when: str) -> str:
    """Where a run keeps what else was on its node at a row's `start`, before the row's provers start, or its `end`,
    after they exit: a line `epoch SECONDS`; a line `lease N ...` for each card N another run's lease held; and a line
    `app GPU N pid P MEMORY` for each compute process on a card (N is `?` for a card it couldn't name). Where the node
    gave no answer on one of the two lists, a line names the tool that gave none (`gpu-lease status gave no answer`,
    `nvidia-smi gave no answer`). The name holds the row's prover count alone, so a run holds one such record for the
    rows at one count."""

    return f"out/cotenants-k{row.get('provers')}-{when}.txt"


def files_in(directory: Path) -> Files:
    """A directory's files as a run's: a run record's tree from the store, or a run's own directory on its node."""

    def read(path: str) -> str | None:
        file = directory / path
        return file.read_text(errors="replace") if file.is_file() else None

    return read


@dataclass(frozen=True)
class Check:
    """One check's answer on a row: `PASS`; `FAIL`, with what the files show against it; or `CANNOT_TELL`, with what
    would show it and doesn't."""

    name: str
    status: str
    why: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == PASS


@dataclass(frozen=True)
class RowVerdict:
    """A row's ten answers, in `CHECKS`' order."""

    type: Any
    provers: Any
    checks: tuple[Check, ...]

    @property
    def status(self) -> str:
        """`FAIL` when any check failed, else `CANNOT_TELL` when any couldn't be told, else `PASS`."""

        got = {check.status for check in self.checks}
        return FAIL if FAIL in got else CANNOT_TELL if CANNOT_TELL in got else PASS

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def why(self) -> str | None:
        """Each check that isn't a pass, with its answer and its reason; None when the row passed."""

        return "; ".join(f"`{c.name}` ({c.status}): {c.why}" for c in self.checks if not c.ok) or None

    def record(self) -> dict:
        """The verdict as JSON, with the rule's constant."""

        return {"type": self.type, "provers": self.provers, "status": self.status, "level": LEVEL,
                "checks": {c.name: {"status": c.status, "why": c.why} for c in self.checks}}


_ABSENT = object()


def _field(row: Mapping[str, Any], name: str) -> Any:
    """A row's field by its dotted name, or `_ABSENT`."""

    at: Any = row
    for key in name.split("."):
        if not isinstance(at, Mapping) or key not in at:
            return _ABSENT
        at = at[key]
    return at


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _seconds(x: Any) -> bool:
    """A finite number a float holds: `json` reads `NaN` and `Infinity`, which compare false with everything and so pass
    a bound, and an integer of any length."""

    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:
        return False


def _unread(name: str, row: Mapping[str, Any], counts: Sequence[str] = (), others: Sequence[str] = ()) -> Check | None:
    """`CANNOT_TELL` for the fields a check reads and the row lacks, or for those of `counts` that aren't counts; None
    when the row holds them all."""

    if lacks := [f for f in (*counts, *others) if _field(row, f) is _ABSENT]:
        return Check(name, CANNOT_TELL, "its row lacks " + ", ".join(f"`{f}`" for f in lacks))
    if bad := [f for f in counts if not _count(_field(row, f))]:
        return Check(name, CANNOT_TELL, ", ".join(f"`{f}` ({_field(row, f)!r})" for f in bad) + " of its row "
                     + ("isn't a count" if len(bad) == 1 else "aren't counts"))
    return None


def _records(text: str, path: str) -> list[Mapping[str, Any]] | str:
    """A JSON-lines file's records, or why it isn't one."""

    found = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            return f"line {n} of `{path}` isn't JSON ({e.msg})"
        if not isinstance(record, Mapping):
            return f"line {n} of `{path}` isn't a record"
        found.append(record)
    return found


def _rows(files: Files) -> list[Mapping[str, Any]] | str:
    """The rows of a run's `RATES`, or why it gives none."""

    text = files(RATES)
    if text is None:
        return f"the run holds no `{RATES}`"
    try:
        found = json.loads(text)
    except json.JSONDecodeError as e:
        return f"the run's `{RATES}` isn't JSON ({e.msg})"
    if not isinstance(found, list) or not all(isinstance(row, Mapping) for row in found):
        return f"the run's `{RATES}` isn't a list of rows"
    return found


def rows(files: Files) -> list[Mapping[str, Any]]:
    """The rows of a run's `RATES`. A run that holds no `RATES`, or one that isn't a list of rows, has no row to judge:
    ValueError."""

    found = _rows(files)
    if isinstance(found, str):
        raise ValueError(f"{found}: it has no row to judge")
    return found


# ---- the checks a row alone shows -------------------------------------------------------------------------------------------


def accepted(row: Mapping[str, Any]) -> Check:
    name = "accepted"
    if unread := _unread(name, row, ("provers", "warm", "sessions", "accepted", "refused_n", "unfiled")):
        return unread
    ran = row["provers"] * row["warm"] + row["sessions"]
    if row["accepted"] != ran:
        return Check(name, FAIL, f"{row['accepted']} of its {ran} sessions accepted")
    if row["refused_n"]:
        return Check(name, FAIL, f"{row['refused_n']} sessions refused")
    if row["unfiled"]:
        return Check(name, FAIL, f"{row['unfiled']} sessions never filed by their verifier")
    return Check(name, PASS)


def exits(row: Mapping[str, Any]) -> Check:
    name = "exits"
    if unread := _unread(name, row, others=("exits",)):
        return unread
    codes = row["exits"]
    if isinstance(codes, (list, tuple)):
        codes = dict(enumerate(codes))
    if not isinstance(codes, Mapping) or not codes:
        return Check(name, CANNOT_TELL, f"its row's `exits` ({codes!r}) names no process")
    if bad := [f"{process} rc {rc}" for process, rc in codes.items() if not (_count(rc) and rc == 0)]:
        return Check(name, FAIL, "exited nonzero: " + ", ".join(bad))
    return Check(name, PASS)


def not_void(row: Mapping[str, Any]) -> Check:
    name = "not_void"
    for key, what in (("error", "failed"), ("skipped", "skipped")):
        if key in row:
            return Check(name, FAIL, f"{what}: {row[key]}")
    if void := row.get("void"):
        return Check(name, FAIL, "void: " + ("; ".join(map(str, void)) if isinstance(void, list) else str(void)))
    if unread := _unread(name, row, others=("statements_per_s",)):
        return unread
    rate = row["statements_per_s"]
    if not (_seconds(rate) and rate > 0):
        return Check(name, FAIL, f"no rate: its `statements_per_s` is {rate!r}")
    return Check(name, PASS)


def flat(row: Mapping[str, Any]) -> Check:
    name = "flat"
    if unread := _unread(name, row, others=("flat",)):
        return unread
    if row["flat"] is not True:
        return Check(name, FAIL, "its statement cache's hit rate wasn't flat")
    return Check(name, PASS)


def no_throttling(row: Mapping[str, Any]) -> Check:
    name = "no_throttling"
    if unread := _unread(name, row, ("throttled.nr_throttled",)):
        return unread
    throttled = row["throttled"]
    if n := throttled["nr_throttled"]:
        usec = throttled.get("throttled_usec")
        return Check(name, FAIL, f"its CPUs were throttled {n} times while it ran"
                     + (f", for {usec / 1e6:.3g} s" if _seconds(usec) else ""))
    return Check(name, PASS)


def every_prover_warmed(row: Mapping[str, Any]) -> Check:
    name = "every_prover_warmed"
    if unread := _unread(name, row, ("provers", "idle_provers"), ("warm_per_prover",)):
        return unread
    provers, warm = row["provers"], row["warm_per_prover"]
    if not isinstance(warm, Mapping) or not all(_count(n) for n in warm.values()):
        return Check(name, CANNOT_TELL, f"its row's `warm_per_prover` ({warm!r}) isn't each prover's warm sessions")
    warmed = sum(1 for n in warm.values() if n >= 1)
    if warmed != provers:
        return Check(name, FAIL, f"{warmed} of its {provers} provers finished a warm session before the timed ones")
    if row["idle_provers"]:
        return Check(name, FAIL, f"{row['idle_provers']} of its {provers} provers filed no session in the window")
    return Check(name, PASS)


def one_numa_node(row: Mapping[str, Any]) -> Check:
    name = "one_numa_node"
    if unread := _unread(name, row, others=("cores.numa_nodes",)):
        return unread
    nodes = row["cores"]["numa_nodes"]
    if not isinstance(nodes, (list, tuple)) or not all(_count(node) for node in nodes):
        return Check(name, CANNOT_TELL, f"`cores.numa_nodes` ({nodes!r}) of its row isn't a list of NUMA nodes")
    if not nodes:
        return Check(name, CANNOT_TELL, "`cores.numa_nodes` of its row names no NUMA node: its machine listed none")
    if len(held := sorted(set(nodes))) > 1:
        cpus = row["cores"].get("list")
        return Check(name, FAIL, f"its CPUs{f' ({cpus})' if isinstance(cpus, str) else ''} are on {len(held)} NUMA nodes "
                     f"({', '.join(map(str, held))})")
    return Check(name, PASS)


# ---- the checks that read the run's other files -----------------------------------------------------------------------------


def one_lease_own_card(row: Mapping[str, Any], files: Files) -> Check:
    name = "one_lease_own_card"
    if unread := _unread(name, row, ("gpus",), ("device",)):
        return unread
    text = files(LEASES)
    if text is None:
        return Check(name, CANNOT_TELL, f"the run holds no `{LEASES}`, which gpu-lease writes when a lease ends")
    leases = _records(text, LEASES)
    if isinstance(leases, str):
        return Check(name, CANNOT_TELL, leases)
    if not leases:
        return Check(name, CANNOT_TELL, f"the run's `{LEASES}` holds no lease")
    if len(leases) > 1:
        return Check(name, FAIL, f"the run took {len(leases)} leases")
    gpus = leases[0].get("gpus")
    if not isinstance(gpus, list) or not all(isinstance(gpu, Mapping) and "index" in gpu for gpu in gpus):
        return Check(name, CANNOT_TELL, f"the lease in `{LEASES}` doesn't say which cards it held")
    cards, device = [str(gpu["index"]) for gpu in gpus], str(row["device"])
    if cards != [device]:
        return Check(name, FAIL, f"its lease held GPU {', '.join(cards) or 'none'}; its provers ran on GPU {device}")
    if row["gpus"] != 1:
        return Check(name, FAIL, f"its row counts {row['gpus']} GPUs on a lease of one")
    return Check(name, PASS)


def card_clear_start_end(row: Mapping[str, Any], files: Files) -> Check:
    name = "card_clear_start_end"
    if unread := _unread(name, row, others=("provers", "device")):
        return unread
    others = _rows(files)
    if isinstance(others, str):
        return Check(name, CANNOT_TELL, f"{others}: it would say whether the record of its card is this row's alone")
    if (at := sum(1 for other in others if other.get("provers") == row["provers"])) > 1:
        return Check(name, CANNOT_TELL, f"the run holds {at} rows at {row['provers']} provers and one record of its card "
                     f"for them (`{card_file(row, 'start')}`)")
    device, on_card, untold = str(row["device"]), [], []
    for when in ("start", "end"):
        path = card_file(row, when)
        text = files(path)
        if text is None:
            untold.append(f"the run holds no record of its card at the row's {when} (`{path}`)")
            continue
        lines = [line.split() for line in text.splitlines()]
        if not any(line[:1] == ["epoch"] for line in lines):
            untold.append(f"`{path}` isn't a record of the node: it has no `epoch` line")
        for line in lines:
            if line[:1] == ["lease"] and line[1:2] == [device]:
                on_card.append(f"at its {when}, another run's lease ({' '.join(line[1:])[:160]})")
            elif line[:2] == ["app", "GPU"] and line[2:3] == [device]:
                on_card.append(f"at its {when}, a compute process ({' '.join(line[3:])[:160]})")
            elif line[:3] == ["app", "GPU", "?"]:
                untold.append(f"`{path}` names a compute process on a card it couldn't name ({' '.join(line[3:])[:160]})")
            elif "gave no answer" in (said := " ".join(line[1:] if line[:1] in (["lease"], ["app"]) else line)):
                what = ("other runs' leases" if "gpu-lease" in said else "compute processes" if "nvidia-smi" in said
                        else "what else was on the node")
                untold.append(f"`{path}` has no list of {what}: {said[:160]}")
    if on_card:
        return Check(name, FAIL, f"GPU {device} wasn't its own: " + "; ".join(on_card))
    if untold:
        return Check(name, CANNOT_TELL, "; ".join(untold))
    return Check(name, PASS)


def window(filed: Sequence[float], provers: int) -> list[float]:
    """The filing times of the window `server_rate.py` rates, from the timed sessions' in order: from the K-th's to the
    last but K's (all of them, where that leaves fewer than two sessions)."""

    first, last = provers - 1, len(filed) - 1 - provers
    if last - first < 2:
        first, last = 0, len(filed) - 1
    return list(filed[first:last + 1])


def thirds(times: Sequence[float]) -> list[float]:
    """Seconds a statement in each third of a window, cut into three equal counts of sessions (fewer than three figures
    for a window of under three sessions)."""

    n = len(times) - 1
    cuts = [0, n // 3, 2 * n // 3, n]
    return [(times[b] - times[a]) / (b - a) for a, b in zip(cuts, cuts[1:]) if b > a]


def burst_parts(times: Sequence[float], burst: int, *, back: bool = False) -> list[float]:
    """Seconds a statement in each part of a window whose sessions file in bursts of `burst`: its whole bursts, from the
    window's start, in parts of a third of them each, rounded down to whole bursts, with a last shorter part for the
    bursts left over. So a window of three, six or nine bursts gives its thirds, and one of four gives each burst (fewer
    than three figures for a window of under three bursts). The sessions after the last whole burst are in no part.
    `back`: the same cut counted back from the window's end, in the window's order, which leaves out the sessions
    before its first whole burst instead."""

    n = len(times) - 1
    whole = n // burst
    part = max(whole // 3, 1) * burst
    cuts = [*range(0, whole * burst, part), whole * burst]
    if back:
        cuts = [n - cut for cut in reversed(cuts)]
    return [(times[b] - times[a]) / (b - a) for a, b in zip(cuts, cuts[1:])]


def level(row: Mapping[str, Any], files: Files) -> Check:
    name = "level"
    if unread := _unread(name, row, ("provers", "warm", "sessions", "window_sessions"), ("type", "window_s")):
        return unread
    period = row.get(PERIOD)
    if period is not None and not (_count(period) and period >= 1):
        return Check(name, CANNOT_TELL, f"`{PERIOD}` ({period!r}) of its row isn't a count of sessions a prover, at least 1")
    burst = None if period is None else period * max(row["provers"], 1)
    if not _seconds(row["window_s"]):
        return Check(name, CANNOT_TELL, f"`window_s` ({row['window_s']!r}) of its row isn't a window's seconds")
    path = sessions_file(row)
    text = files(path)
    if text is None:
        return Check(name, CANNOT_TELL, f"the run holds no `{path}`")
    sessions = _records(text, path)
    if isinstance(sessions, str):
        return Check(name, CANNOT_TELL, sessions)
    if not all(_count(s.get("i")) and (s.get("filed") is None or _seconds(s.get("filed"))) for s in sessions):
        return Check(name, CANNOT_TELL, f"`{path}` isn't each session's index `i` and its `filed` time")
    warm = row["provers"] * row["warm"]
    if len(sessions) != warm + row["sessions"]:
        return Check(name, CANNOT_TELL, f"`{path}` holds {len(sessions)} sessions; its row ran {warm + row['sessions']}")
    times = window(sorted(s["filed"] for s in sessions if s["i"] >= warm and s["filed"] is not None), row["provers"])
    span = times[-1] - times[0] if times else 0.0
    if len(times) - 1 != row["window_sessions"] or abs(span - row["window_s"]) > WINDOW_SLACK_S:
        return Check(name, CANNOT_TELL, f"the window `{path}` gives ({max(len(times) - 1, 0)} sessions over {span:.2f} s) "
                     f"isn't its row's ({row['window_sessions']} over {row['window_s']} s)")
    if burst is None:
        cuts = [(thirds(times), "thirds")]
    else:
        parts = f"parts of whole bursts of {burst} sessions"
        cuts = [(burst_parts(times, burst), parts)]
        if (len(times) - 1) % burst:
            cuts.append((burst_parts(times, burst, back=True), f"{parts} counted back from its end"))
    for took, parts in cuts:
        if len(took) < 3 or min(took) <= 0:
            return Check(name, CANNOT_TELL, f"its window of {len(times) - 1} sessions can't be cut into three {parts} that "
                         "each took time")
    for took, parts in cuts:
        if max(took) > LEVEL * min(took):
            return Check(name, FAIL, "its window isn't level: a statement took " + ", ".join(f"{t:.4g}" for t in took)
                         + f" s in its {parts}, the slowest over {LEVEL} times the fastest")
    return Check(name, PASS)


# ---- a row's verdict, and a run's -------------------------------------------------------------------------------------------

CHECKS = ("accepted", "exits", "not_void", "flat", "no_throttling", "one_lease_own_card", "card_clear_start_end", "level",
          "every_prover_warmed", "one_numa_node")


def verdict(row: Mapping[str, Any], files: Files) -> RowVerdict:
    """A row's verdict from a run's files: the ten checks, in `CHECKS`' order. `row` is the run's own (`verdicts`), or
    the same row as completed or corrected after the run, judged against the files of the run that measured it."""

    checks = (accepted(row), exits(row), not_void(row), flat(row), no_throttling(row), one_lease_own_card(row, files),
              card_clear_start_end(row, files), level(row, files), every_prover_warmed(row), one_numa_node(row))
    return RowVerdict(row.get("type"), row.get("provers"), checks)


def verdicts(files: Files) -> list[RowVerdict]:
    """The verdict of each row of a run's `RATES` (`rows`), in the file's order."""

    return [verdict(row, files) for row in rows(files)]
