"""The description of each template GLM-5.3's served partitions name, measured as `program.json` holds it, against `SIZE`:
the record (`description_sizes.json`, one entry a template) that `tests/tools/test_description_sizes.py` holds them to.

What the served partitions name (`served`): every template a partition file names (`PARTITIONS`, GLM-5.3's at TP 8, each
step's), and the units of the DSA indexer's long row as its cut names them (`dsa_topk_reads.cut`), at the longest class the
mask's cut leaves whole (`dsa_topk_mask_cut.WHOLE_MAX`) and at the history table's capacity, which no partition file
reaches yet.  A template's description is ``canonical_json(encode_program(population_program(template, 1)))``: every
Definition it reaches, a traced body gate by gate, so its size is what a verifier that reads the description pays.

    python tools/description_sizes.py list     each template and what names it, one JSON line each (nothing is built)
    python tools/description_sizes.py one T    measure template T; its entry as JSON on stdout
    python tools/description_sizes.py update [-j N] [--run RUN] [--remeasure T ...]

``update`` measures, one template a process, every named template the record has no entry for or records at most `SIZE`,
keeps each recorded past `SIZE` as it stands unless ``--remeasure`` names it, drops what nothing names, and writes the
record, each measured entry citing ``--run``.  Run it as a node job with ``--mem-gb`` stated: a template past `SIZE` is
built only under ``--remeasure`` or when no entry has it yet, and a traced one can take far more memory than a laptop has."""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).with_suffix(".json")
#: the most bytes a served template's description holds: past it, a verifier that reads the description reads a traced
#: body, O(unrolled circuit) rather than O(circuit description)
SIZE = 4 * 2**20
#: the served partitions, from the repository's root
PARTITIONS = "benchmarks/one_stage/glm53_*_tp8_step*.json"
#: the long row's statics that are GLM-5.3's: the indexer's top-k and the latent row a selected position reads
TOPK, LATENT = 2048, 576
#: a Definition id longer than this is cut to it in an entry, which names it only for a reader
ID_CHARS = 160


@cache
def long_row_units() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """The long row's units by descriptor id: the bound Definition, and each class and count its cut names it at."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind
    from verity.commitments.gates.embedding import read
    from verity.commitments.history import CAPACITY
    from verity.definitions.registry import dsa_topk_mask_cut as X
    from verity.definitions.registry import dsa_topk_reads as R

    out: dict[str, tuple[Any, tuple[str, ...]]] = {}
    for n in (X.WHOLE_MAX, CAPACITY):
        row = bind(R.DsaTopKReadRows, N=n, TOPK=TOPK, BLOCK=R.BLOCK, READ=read(CAPACITY, LATENT))
        for unit, count in R.cut(row):
            t = descriptor_id(unit)
            out[t] = (unit, out.get(t, (unit, ()))[1] + (f"long-row:N={n} x{count}",))
    return out


def served() -> dict[str, list[str]]:
    """Each template the served partitions name, with what names it (``partition:FILE``, ``long-row:N=n xcount``)."""
    out: dict[str, list[str]] = {}
    for f in sorted(ROOT.glob(PARTITIONS)):
        for t in json.loads(f.read_text())["templates"]:
            out.setdefault(t, []).append(f"partition:{f.name}")
    for t, (_unit, where) in long_row_units().items():
        out.setdefault(t, []).extend(where)
    return out


def definition(t: str) -> Any:
    """Template ``t``'s Definition: a long-row unit as its cut binds it, else the one a partition names through its word
    link (`serving_templates.template`)."""
    unit = long_row_units().get(t)
    if unit is not None:
        return unit[0]
    from verity_vllm.pipeline.serving_templates import template

    return template(t).fn


def description(fn: Any) -> dict:
    from verity.circuits.codec import encode_program
    from verity.circuits.population import population_program

    return encode_program(population_program(fn, 1))


def size(desc: dict) -> int:
    from verity.circuits.codec import canonical_json

    return len(canonical_json(desc))


def largest(desc: dict) -> list:
    """The largest Definition in ``desc``: its id (cut to `ID_CHARS`) and its bytes."""
    from verity.circuits.codec import canonical_json

    fid, n = max(((f, len(canonical_json(d))) for f, d in desc["definitions"].items()), key=lambda r: r[1])
    return [fid if len(fid) <= ID_CHARS else fid[:ID_CHARS - 1] + "…", n]


def measure(t: str) -> dict:
    """Template ``t``'s entry: its bytes and largest Definition, the seconds its build and encoding took past what a test
    process has already loaded (the registry, the long row's cut), and this process's peak GB."""
    from verity.circuits.codec import descriptor_id
    from verity.definitions import boolean as BB

    BB.load()
    long_row_units()
    t0 = time.monotonic()
    fn = definition(t)
    if descriptor_id(fn) != t:
        raise ValueError(f"{t} resolves to {descriptor_id(fn)}")
    desc = description(fn)
    n = size(desc)
    seconds = round(time.monotonic() - t0, 1)
    return {"bytes": n, "largest": largest(desc), "seconds": seconds,
            "gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}


def load() -> dict[str, dict]:
    return json.loads(RECORD.read_text())["templates"] if RECORD.exists() else {}


def write(rec: dict[str, dict]) -> None:
    rows = [f"{json.dumps(t)}: {json.dumps(e, sort_keys=True, ensure_ascii=False)}" for t, e in sorted(rec.items())]
    RECORD.write_text('{"templates": {\n' + ",\n".join(rows) + "\n}}\n")


def update(jobs: int, run: str | None, remeasure: set[str]) -> int:
    named = served()
    rec = {t: e for t, e in load().items() if t in named}
    todo = [t for t in named if t not in rec or rec[t]["bytes"] <= SIZE or t in remeasure]

    def one(t: str) -> tuple[str, dict | None, str]:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", t], capture_output=True, text=True)
        return t, json.loads(r.stdout.splitlines()[-1]) if r.returncode == 0 else None, r.stderr[-2000:]

    failed = []
    with ThreadPoolExecutor(jobs) as pool:
        for t, e, err in pool.map(one, todo):
            if e is None:
                failed.append(t)
                print(f"FAILED {t}\n{err}", file=sys.stderr, flush=True)
                continue
            rec[t] = {**e, **({"run": run} if run else {})}
            print(f"{e['bytes']:>14,} {e['seconds']:>7}s {e['gb']:>6}GB {t}", flush=True)
    write(rec)
    print(f"{len(todo) - len(failed)} measured, {len(rec) - len(todo) + len(failed)} kept, {len(failed)} failed; "
          f"{sum(e['bytes'] > SIZE for e in rec.values())} of {len(rec)} past {SIZE:,} bytes", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("one")
    p.add_argument("template")
    p = sub.add_parser("update")
    p.add_argument("-j", "--jobs", type=int, default=4)
    p.add_argument("--run", help="the run this measures in, cited in each entry it writes")
    p.add_argument("--remeasure", nargs="*", default=[], help="templates past SIZE to build again")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for t, where in served().items():
            print(json.dumps({"template": t, "named_by": where}))
        return 0
    if a.cmd == "one":
        print(json.dumps(measure(a.template), ensure_ascii=False))
        return 0
    return update(a.jobs, a.run, set(a.remeasure))


if __name__ == "__main__":
    sys.exit(main())
