"""`stress.py units`: each pass's device time a call at decode or prefill shapes, in the lines `priced.units` reads,
and the run's result. Needs no torch: `stress.py` hands it a timer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Callable

KIND = "dsa-topk-units/v1"
#: `dsa_topk.MODES`' passes by the names `priced.units` reads
PASSES = {"canonical-bisect": "present", "canonical": "cut"}
#: by shape set: the rows and the longest row's keys timed, the values' distributions, and the one the forced
#: failures are planted on
SHAPES = {
    "decode": ((6, 12, 24), (20000, 40000, 70000, 100000, 131072),
               ("randn", "bf16", "indexer", "indexer_pos", "shift8", "shift32")),
    "prefill": ((2048, 4096, 8192), (16384, 32768), ("randn", "bf16", "shift8", "shift32")),
}
FORCED_ON, FORCED = "shift32", ("missed", "overflow")
PRICED = Path(__file__).resolve().parents[1] / "prod_repro" / "priced.py"

#: (values, rows, keys, forced or None) -> a line's timed fields: unforced, ``fast_us`` and for each pass ``ok``,
#: ``us``, ``added_us`` and ``per_call``; forced, ``copy_us`` and for each pass ``ok``, ``one_call`` and ``us``
Timer = Callable[[str, int, int, "str | None"], dict]


def lines(shapes: str, timer: Timer) -> list[dict]:
    rows, keys, values = SHAPES[shapes]
    out = []
    for v in values:
        for r in rows:
            for n in keys:
                for how in (None, *FORCED) if v == FORCED_ON else (None,):
                    head = {"dist": v} if how is None else {"forced": how}
                    out.append({**head, "rows": r, "keys": n, **timer(v, r, n, how)})
                    print(json.dumps(out[-1]), flush=True)
    return out


def problems(path: Path, shapes: str) -> list[str]:
    """What keeps `path` from pricing every shape of `shapes`, by `priced.units` itself; a pass whose output is not
    `reference_rows`', and a forced line whose pass reached no fallback, are each one."""
    spec = importlib.util.spec_from_file_location("priced", PRICED)
    priced = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(priced)
    try:
        table = priced.units(path)
    except SystemExit as ex:
        return [str(ex)]
    rows, keys, _ = SHAPES[shapes]
    want = {(r, n) for r in rows for n in keys}
    out = [f"{p}: no times" for p in PASSES.values() if p not in table]
    out += [f"{p} {part}: no time at {sorted(want - set(t[part]))}"
            for p, t in table.items() for part in ("base", "failed") if want - set(t[part])]
    for r in map(json.loads, path.read_text().splitlines()):
        what = f"{r.get('forced') or r.get('dist')} at {r['rows']} x {r['keys']}"
        for p in PASSES.values():
            if not r[p]["ok"]:
                out.append(f"{p} {what}: not reference_rows'")
            if "forced" in r and not (r[p]["one_call"].get("rows_bisected") or r[p]["one_call"].get("rows_refined")):
                out.append(f"{p} {what}: no row reached the fallback")
    return out


def write(out: Path, result: Path, shapes: str, timed: list[dict], device: str) -> bool:
    """Write `out`/units.jsonl and `result`, validation passed only where `problems` finds none."""
    out.mkdir(parents=True, exist_ok=True)
    result = result.resolve()
    f = out / "units.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in timed))
    bad = problems(f, shapes)
    rows, keys, values = SHAPES[shapes]
    rel = os.path.relpath(f, result.parent)
    detail = (f"{len(timed)} lines at {shapes} shapes, rows {rows} over keys {keys}, both passes "
              f"({', '.join(f'{m} as {p}' for m, p in PASSES.items())}) on random values ({', '.join(values)}) and "
              f"on forced failures ({', '.join(FORCED)}) planted on {FORCED_ON}: "
              + ("; ".join(bad) if bad else "priced.units reads every shape"))
    result.write_text(json.dumps({
        "schema": "research/result/v0.1", "kind": KIND, "run_id": result.parent.name,
        "workload_fingerprint": {"benchmark": "dsa_topk", "mode": "units", "shapes": shapes, "device": device},
        "measurements": [], "measurement_files": [rel], "artifacts": [rel],
        "validation": {"status": "failed" if bad else "passed", "checker": "stress.py units (priced.units)",
                       "detail": detail, "evidence": [rel]}}, indent=1) + "\n")
    print(detail, flush=True)
    return not bad
