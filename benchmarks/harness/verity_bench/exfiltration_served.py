"""The share of compute a workload measures, as `verity bench exfiltration` (`verity_bench.exfiltration`) reads it
from the run's record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity_bench.exfiltration_common import (
    AT_MOST,
    NO_RUN,
    PROVE_SOURCE,
    PROVE_SOURCE_GAP,
    PROVE_SOURCES,
    SERVED_PARTS,
    SERVING_GAP,
)
from verity_bench.exfiltration_text import _pct


def _gpu_s(entries: Any, field: str, run: str) -> tuple[float | None, list[str]]:
    """The GPU-seconds a record's field holds, a list of what held GPUs for it (one entry a session or a step, each
    `{"seconds": s, "gpus": g}`) or one such entry, and the keys an entry lacks. None and the gap where it holds none.
    An entry on 0 GPUs is a step that started no process on a card: it is read, and adds no GPU-seconds."""

    if not entries:
        return None, [field]
    lacks, total = [], 0.0
    for e in [entries] if isinstance(entries, Mapping) else entries:
        seconds, gpus = e.get("seconds"), e.get("gpus")
        lacks += [f"{field}.{k}" for k, v in (("seconds", seconds), ("gpus", gpus)) if v is None]
        if seconds is not None and gpus is not None:
            if not (seconds >= 0 and isinstance(gpus, int) and not isinstance(gpus, bool) and gpus >= 0):
                raise ValueError(f"{run}'s record holds {field} at {seconds!r} seconds on {gpus!r} GPUs: an entry is "
                                 "seconds of 0 or more on a whole count of GPUs, 0 or more")
            total += seconds * gpus
    return (None if lacks else total), sorted(set(lacks))


def _on_cpus(entries: Any) -> bool:
    """Whether every entry of a record's field is on 0 GPUs: the part ran on CPUs alone, and its 0 GPU-seconds are not
    GPU time it was given free."""

    its = [entries] if isinstance(entries, Mapping) else entries or []
    return bool(its) and all(isinstance(e, Mapping) and e.get("gpus") == 0 for e in its)


def served_share(record: Mapping[str, Any] | None) -> dict:
    """The audit's share of compute on a served two-stage run, from its ``record`` (the reader of a workload's
    record gives it): its GPU-seconds over the window's serving GPU-seconds. Each part of the audit (`SERVED_PARTS`)
    is read from the record and marked measured, and where the record lacks it, its gap is named and the share is not
    claimed. None: no record, and the share is `NO_RUN`.

    The record is one object: ``run``, the workload's id; ``prove``, ``staging``, ``replay`` and ``interior``, each
    a list of what held GPUs for that part, one entry a session or a step, each ``{"seconds": s, "gpus": g}`` (the
    seconds it held them and how many it held); ``serving``, one such entry, the window's load: the seconds it was
    served over and the GPUs its engine held; and for the serving charge, ``stock``, one such entry for stock vLLM
    serving the same load at the same launch settings save Verity's changes, or else ``serving_cost``, `verity bench
    serve_tax`'s cost of the served arm over stock on its own loads, as a fraction (stock's output tokens a second over
    the arm's, less one), with its ``serving_cost_source`` (the run and load it is of). A field absent or None is one
    the record lacks; so is an entry's key. The serving charge is the served GPU-seconds less stock's, measured; or
    from the serving cost c, the served GPU-seconds times c / (1 + c), modeled, since its loads are not the window's.
    A part whose entries are all on 0 GPUs is read as 0 GPU-seconds and held `on_cpus`, which the print says beside
    its 0 (`ON_CPUS`).

    ``prove_seconds_source`` says which seconds ``prove``'s are (`PROVE_SOURCES`): ``busy``, the proof server's busy
    seconds a card, measured; ``session``, a session's seconds with its wait included, so the proving and the share
    that sums it are marked measured, at most. Absent, ``prove`` lacks it and the share is not claimed; any other
    value is refused. On both, a part's GPU-seconds are the seconds held times the GPUs held."""

    if record is None:
        return {"run": None, "value": None, "mark": None, "why": NO_RUN, "serving_gpu_s": None, "parts": {},
                "missing": []}
    if not isinstance(record, Mapping):
        raise ValueError(f"a workload's record is one JSON object, not {type(record).__name__}")
    run = str(record.get("run") or "the workload")
    said = record.get(PROVE_SOURCE)
    if said is not None and said not in PROVE_SOURCES:
        raise ValueError(f"{run}'s record holds `{PROVE_SOURCE}` {said!r}: one of "
                         + ", ".join(f"{k} ({why})" for k, (_, why) in PROVE_SOURCES.items()))
    serving, missing = _gpu_s(record.get("serving"), "serving", run)
    gaps = [{"field": "serving", "lacks": missing, "why": SERVING_GAP}] if missing else []
    parts: dict[str, dict] = {}
    for name, (what, why) in SERVED_PARTS.items():
        if name == "serving_charge":
            stock, lacks = _gpu_s(record.get("stock"), "stock", run)
            cost = record.get("serving_cost")
            if record.get("stock") is not None or cost is None:
                gpu_s, mark = (None if stock is None or serving is None else serving - stock), "measured"
                source = "the served GPU-seconds less those of stock vLLM serving the same load (`stock`)"
            else:
                if not (isinstance(cost, (int, float)) and cost > -1):
                    raise ValueError(f"{run}'s record holds a serving cost of {cost!r}: a fraction over -1")
                lacks, mark = [], "modeled"
                gpu_s = None if serving is None else serving * cost / (1 + cost)
                source = (f"the serving cost of the served arm over stock, {_pct(cost)} (`serving_cost`, "
                          f"{record.get('serving_cost_source') or 'its run and load unnamed'}), on its own loads, not "
                          "the window's: the served GPU-seconds times c / (1 + c)")
        else:
            (gpu_s, lacks), mark, source = _gpu_s(record.get(name), name, run), "measured", f"`{name}`"
        if name == "prove":
            if said is None:
                gap = [why] if lacks else []
                lacks, why = [*lacks, PROVE_SOURCE], "; ".join([*gap, PROVE_SOURCE_GAP])
            else:
                mark, held = PROVE_SOURCES[said]
                source += f", {held} (`{PROVE_SOURCE}`: `{said}`)"
        if lacks:
            gaps.append({"field": name, "lacks": lacks, "why": why})
        parts[name] = {"what": what, "gpu_s": gpu_s, "mark": "missing" if gpu_s is None or lacks else mark,
                       "source": source,
                       "share": None if gpu_s is None or not serving else gpu_s / serving,
                       "on_cpus": name != "serving_charge" and gpu_s is not None and _on_cpus(record.get(name))}
    if serving is not None and not serving > 0:
        raise ValueError(f"{run}'s record holds the window's serving at {serving!r} GPU-seconds: the share is of more "
                         "than none")
    claimed = not gaps
    return {"run": run, "value": sum(p["share"] for p in parts.values()) if claimed else None,
            "mark": (("modeled" if any(p["mark"] == "modeled" for p in parts.values()) else "measured")
                     + (AT_MOST if any(p["mark"].endswith(AT_MOST) for p in parts.values()) else "")
                     if claimed else None),
            "why": None if claimed else "not measured: " + "; ".join(
                f"{run}'s record lacks {', '.join(f'`{k}`' for k in g['lacks'])} ({g['why']})" for g in gaps),
            "serving_gpu_s": serving, "parts": parts, "missing": gaps}
