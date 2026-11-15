"""prove_rows.py OUT --record RUN [--cores N] [--leave NAME ...]: a `window.py prove` run's timed statements as rows by
block type, in the format the break-even planner reads a measured row in (`benchmarks/spot_figures/breakeven/reprice.py`'s
`steady`: the file `headline.py --steady` takes, which `verity bench prove --steady-rows` writes from the proof server's
runs). It prints `{"rows": [...], "beside": [...], "refused": [...]}`.

OUT is the run's `out/`: `prove.tsv` (a statement a line: its group, its name DEV/NAME, its blocks, its k_log and the
median of its timed sessions' prover seconds, `prove_time.py`'s `prove_total_s`, with the rest of its summary as
`record`) and each group's own files as `prove_time.sh` leaves them, `GROUP/out/stage/NAME/stage.json` (what was
staged) and `GROUP/out/DEV/NAME/exit` (the prover's exit status).

A row is one block type's largest timed statement. Its `block` is the census type the statement was staged for
(`stage.json`'s `word`; else its `census`; else the Definition staged, `definition`), its `statement_seconds` the
statement's `prove_total_s`, its `blocks` the statement's (`stage.json`'s `n`), its `provers` 1 (`prove_time.sh` times one
prover on a GPU of its own) and its `record` RUN, the run that holds the measurement. Beside those it keeps the
statement's name, the Definition staged, its k_log and, where the summary holds them, the wall seconds a timed session
took (`session_seconds`, the live verifier's turn included); `--cores N` states the CPU cores a prover held, as `cores`.

A statement gives no time unless its prover exited 0, it has a time, and every session it filed was accepted (the
summary's `accepted`, and its `verifier_accepted` where the summary holds one). One whose `stage.json` the run doesn't
hold names no block type. Each such statement is under `refused` with its reason, as is one `--leave` names, and a type's other timed
statements are under `beside`, so no measurement is dropped unsaid. Two timed statements of one block type that hold as
many blocks are refused together: `--leave` one. Exits 1 when no statement gives a row.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

#: prove.tsv's columns, as `window.py prove` writes them
HEAD = ("group", "statement", "n", "k_log", "prove_total_s", "record")
#: what the planner's reader requires of a row (`reprice.STEADY_FIELDS`)
FIELDS = ("block", "statement_seconds", "blocks", "provers", "record")
#: `prove_time.sh` times one prover on the device
PROVERS = 1
#: a time as prove.tsv holds one, a float's `str`; a statement with no time holds `None` or nothing
TIME = re.compile(r"[0-9]+(?:\.[0-9]*)?(?:e[-+]?[0-9]+)?")


def _count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _accepted(verdicts) -> bool:
    return isinstance(verdicts, list) and bool(verdicts) and all(v is True for v in verdicts)


def statements(out: Path) -> list[dict]:
    """prove.tsv's statements, each with its `stage.json` and its prover's exit status where its group's files hold
    them."""
    lines = (out / "prove.tsv").read_text().splitlines()
    if not lines or tuple(lines[0].split("\t")) != HEAD:
        raise SystemExit(f"prove_rows.py: {out / 'prove.tsv'} doesn't start with window.py's columns, {', '.join(HEAD)}")
    found = []
    for line in filter(str.strip, lines[1:]):
        cells = line.split("\t", len(HEAD) - 1)
        if len(cells) != len(HEAD):
            raise SystemExit(f"prove_rows.py: a line of {out / 'prove.tsv'} holds {len(cells)} columns, not {len(HEAD)}: {line[:120]}")
        s = dict(zip(HEAD, cells, strict=True))
        s["name"] = s["statement"].rsplit("/", 1)[-1]
        s["record"] = json.loads(s["record"]) if s["record"] else {}
        group = out / s["group"] / "out"
        staged, status = group / "stage" / s["name"] / "stage.json", group / s["statement"] / "exit"
        s["stage"] = json.loads(staged.read_text()) if staged.exists() else None
        s["exit"] = status.read_text().strip() if status.exists() else None
        found.append(s)
    return found


def seconds(s: dict) -> float | None:
    """A statement's `prove_total_s` where it is a time, else None."""
    t = float(s["prove_total_s"]) if TIME.fullmatch(s["prove_total_s"]) else 0.0
    return t if math.isfinite(t) and t > 0 else None


def why_not(s: dict) -> str | None:
    """Why a statement gives no row, or None when it gives one."""
    stage, rec = s["stage"], s["record"]
    if stage is None:
        return "the run holds no stage.json of it, so it names no block type"
    if not _count(stage.get("n")) or str(stage["n"]) != s["n"]:
        return f"its summary's blocks ({s['n']}) aren't its stage.json's ({stage.get('n')})"
    if s["exit"] != "0":
        return "the run holds no exit status of its prover" if s["exit"] is None else f"its prover exited {s['exit']}"
    if seconds(s) is None:
        return f"it has no time (prove_total_s: {s['prove_total_s'] or 'none'})"
    for who, key, required in (("the prover", "accepted", True), ("the verifier", "verifier_accepted", False)):
        if (required or key in rec) and not _accepted(rec.get(key)):
            return f"a session of it wasn't accepted, by {who}'s record (`{key}`: {json.dumps(rec.get(key))})"
    return None


def block(stage: dict) -> str:
    """The block type a staged statement measures: the census type it was staged for, else the Definition staged."""
    return next(stage[k] for k in ("word", "census", "definition") if isinstance(stage.get(k), str))


def row(s: dict, record: str, cores: int | None) -> dict:
    stage, wall = s["stage"], s["record"].get("session_s")
    return {"block": block(stage), "statement_seconds": seconds(s), "blocks": stage["n"], "provers": PROVERS,
            **({} if cores is None else {"cores": cores}), "record": record, "statement": s["name"],
            "definition": stage.get("definition"), "k_log": int(s["k_log"]) if s["k_log"].isdigit() else stage.get("k_log"),
            **({"session_seconds": wall} if isinstance(wall, (int, float)) else {})}


def rows(out: Path, record: str, cores: int | None = None, leave: tuple[str, ...] = ()) -> dict:
    """The run's rows by block type, its timed statements beside them and the statements that gave no time."""
    read = statements(out)
    if absent := sorted(set(leave) - {s["name"] for s in read}):
        raise SystemExit(f"prove_rows.py: --leave names no statement of {out / 'prove.tsv'}: {', '.join(absent)}")
    refused, timed = [], {}
    for s in read:
        why = "left out (--leave)" if s["name"] in leave else why_not(s)
        if why:
            refused.append({"statement": s["name"], "why": why})
        else:
            timed.setdefault(block(s["stage"]), []).append(row(s, record, cores))
    best, beside = [], []
    for name, found in sorted(timed.items()):
        found.sort(key=lambda r: (-r["blocks"], r["statement"]))
        if len(found) > 1 and found[0]["blocks"] == found[1]["blocks"]:
            both = ", ".join(r["statement"] for r in found if r["blocks"] == found[0]["blocks"])
            raise SystemExit(f"prove_rows.py: `{name}` has timed statements of {found[0]['blocks']} blocks each ({both}): "
                             "which is the type's row isn't this reader's to choose, so --leave all but one")
        best.append(found[0])
        beside += found[1:]
    return {"rows": best, "beside": beside, "refused": refused}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the run's out/, holding prove.tsv and each group's files")
    ap.add_argument("--record", required=True, help="the run or art that holds the measurement, each row's `record`")
    ap.add_argument("--cores", type=int, help="the CPU cores a prover held, each row's `cores`")
    ap.add_argument("--leave", action="append", default=[], metavar="NAME", help="a statement to list under `refused`")
    a = ap.parse_args(argv)
    doc = rows(a.out, a.record, a.cores, tuple(a.leave))
    print(json.dumps(doc, indent=1))
    print(f"prove_rows.py: {len(doc['rows'])} block types timed, {len(doc['beside'])} other timed statements, "
          f"{len(doc['refused'])} statements that gave no time", file=sys.stderr)
    return 0 if doc["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
