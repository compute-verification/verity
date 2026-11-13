"""A stop class's verdict records: what a verifier's command wrote of one class (its ``summary.json``) as one JSON
object, a line of a run's ``verdicts.jsonl``, saying which verdict it is and which class, serve, registration and rate
it is of.

    python benchmarks/one_stage/verdict_record.py PIECE --stage1 F --summary F --serve RUN --out DIR \\
      [--run RUN] [--commit SHA] [--control id|max_tokens]

A stop class's response lengths are held by a verdict on each of its pieces, each accepted and complete, and a reader
of those verdicts has to be able to refuse a set that is not one class's: verdicts of different classes, serves or
registrations, or one that ran at another rate than its registration registers. So a record holds the fields
`benchmarks/harness/verity_bench/exfiltration.py` reads (`VERDICT_FIELDS`: ``piece``, ``control``, ``run``,
``verifier``, ``verdict``, ``proved_by``, ``open``) and four that say what the verdict is of:

* ``registration``: the digest of the class's boundary record (`registration.digest`), which the record of every later
  stream of the class names as its ``previous``. It is the same on every verdict of a class.
* ``serve``: the id, in the evidence store, of the served run the class's files came from. It is the caller's word
  (``--serve``): no file of a class names a run of the store. What a class's registration itself binds of its serve
  is its window's ``run`` (`boundary_rows.run_root`: the serve's nonce and the window's index).
* ``class``: the class's place in its serve, ``w<window>/p<P>-o<O>``, ending ``-nostop`` for a class filed without a
  stop block under a stop rule: the directory `boundary_windows` writes the class to, read here from the boundary
  record's ``window`` (`class_of`).
* ``rate``: the rate the verdict's population is registered at, never the verifier's own argument (`rate`): K of the N
  proof blocks its registration record registers, which the verifier holds to its own sampling design and count (R1,
  R2), written NUM/DEN in lowest terms. A population whose every block is proved reads ``1/1``.

**The pieces** (`PIECES`), each a verifier's command and the registration of the population it sampled:

* ``stop_record``: the stop audit (`stop_audit.py`), whose population's registration is the stop record
  (``stage1.json``'s ``stop.record``). ``accept`` when its audit record accepted and is complete, ``not complete`` when
  it accepted with entries that did not run, else ``refuse``.
* ``second_stage``: the compute audit's second stage (`two_stage.py second`), whose population's registration is the
  class's boundary record. ``refuse`` unless the second stage held (`two_stage.held`). Then ``accept`` or ``not
  complete`` as its outcome accepted with or without ``complete`` (an outcome that carries no ``complete`` is not
  complete), and ``withheld`` when its outcome did not accept: a second stage that held reaches that only where the
  verifier's own two-stage form refused it (`two_stage.FORM_REFUSED`).

``proved_by`` is null on both: no Lean guarantee covers either command, and ``open`` says what stands between its
verdict and proved (`Piece.open`).

**Refused** (`record`, `append`): a summary that does not name this class's registration, so a command that refused
before it registered gets no record, since nothing ties what it wrote to the class; a boundary or stop record that is
not the one whose digest ``stage1.json`` names; and a second record on one piece and control in one file."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.protocols.spot.verifier.one_stage import registration as R

HERE = Path(__file__).resolve().parent
#: a run's file of verdict records, one JSON object a line
FILE = "verdicts.jsonl"
#: the negative controls: a run on the same path with one thing changed (null on the honest run)
CONTROLS = ("id", "max_tokens")
ACCEPT, NOT_COMPLETE, REFUSE, WITHHELD = "accept", "not complete", "refuse", "withheld"
#: the environment variable `research run` gives a job its run's id in
RUN_ENV = "RESEARCH_RUN_ID"


def rate(registered: Mapping[str, Any]) -> str:
    """The rate a registration record registers its population at: K of N as NUM/DEN in lowest terms for a subset
    design, so K = N reads ``1/1``; any other design as itself (``work:K``), which is no rate and never ``1/1``.
    ValueError where the record registers no design, or a subset that is not 1 <= K <= N."""
    law, n = registered.get("law"), registered.get("population")
    if not isinstance(law, Mapping) or type(law.get("law")) is not str or type(law.get("k")) is not int:
        raise ValueError(f"the registration record registers no sampling design: its `law` is {law!r}")
    kind, k = law["law"], law["k"]
    if kind != "subset":
        return f"{kind}:{k}"
    if type(n) is not int or not 1 <= k <= n:
        raise ValueError(f"the registration record's subset design is {k} of {n!r} proof blocks, not 1 <= K <= N")
    g = math.gcd(k, n)
    return f"{k // g}/{n // g}"


def class_of(boundary: Mapping[str, Any]) -> str:
    """A class's place in its serve from its boundary record's ``window`` (the module docstring's ``class``)."""
    w = boundary.get("window")
    if not isinstance(w, Mapping) or any(type(w.get(k)) is not int for k in ("row", "prompt", "outputs")):
        raise ValueError("the boundary record's `window` names no class: its `row`, `prompt` and `outputs`")
    unheld = "stop" in w and w.get("stopped") is not True
    return f"w{w['row']}/p{w['prompt']}-o{w['outputs']}{'-nostop' if unheld else ''}"


def _named(held: object, what: str) -> tuple[Mapping[str, Any], str]:
    """The record ``held`` (``stage1.json``, or its ``stop``) holds and its digest, the one it names beside it."""
    rec = held.get("record") if isinstance(held, Mapping) else None
    if not isinstance(rec, Mapping):
        raise ValueError(f"stage1.json holds no {what}")
    if R.digest(rec) != held.get("registration"):
        raise ValueError(f"stage1.json's {what} is not the record whose digest it names as its registration")
    return rec, held["registration"]


def _stop_record(stage1: Mapping[str, Any], summary: Mapping[str, Any], boundary: Mapping[str, Any],
                 digest: str) -> tuple[str, Mapping[str, Any]]:
    rec, stop = _named(stage1.get("stop"), "stop record")
    if rec.get("previous") != digest:
        raise ValueError("the stop record's `previous` is not the class's boundary record")
    if summary.get("registration") != {"boundary": digest, "stop": stop}:
        raise ValueError("the stop audit's summary does not name this class's boundary record and stop record as its "
                         f"`registration`, but {summary.get('registration')!r}: nothing ties it to the class")
    word = REFUSE if summary.get("accepted") is not True else ACCEPT if summary.get("complete") is True else NOT_COMPLETE
    return word, rec


def _second_stage(stage1: Mapping[str, Any], summary: Mapping[str, Any], boundary: Mapping[str, Any],
                  digest: str) -> tuple[str, Mapping[str, Any]]:
    first = summary.get("first_stage")
    if not isinstance(first, Mapping) or first.get("registration") != digest:
        raise ValueError("the second stage's summary does not name this class's boundary record as its first stage's "
                         f"`registration`, but {first!r}: nothing ties it to the class")
    outcome = summary.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    if summary.get("held") is not True:
        word = REFUSE
    elif outcome.get("accepted") is True:
        word = ACCEPT if outcome.get("complete") is True else NOT_COMPLETE
    else:
        word = WITHHELD
    return word, boundary


@dataclass(frozen=True)
class Piece:
    """A verdict on a stop class. `module`: the verifier's command, as a record names it. `open`: what stands between
    its verdict and proved, in words. `read`: (``stage1.json``, the command's ``summary.json``, the class's boundary
    record, its digest) -> the verdict's word and the registration record of the population it sampled."""

    module: str
    open: str
    read: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str], tuple[str, Mapping[str, Any]]]


PIECES = {
    "stop_record": Piece("benchmarks/one_stage/stop_audit.py",
                         "no Lean guarantee covers the stop audit's command, which is Python around Lean's verdicts",
                         _stop_record),
    "second_stage": Piece("benchmarks/one_stage/two_stage.py", "the two-stage form's soundness is not proved",
                          _second_stage),
}


def record(piece: str, stage1: Mapping[str, Any], summary: Mapping[str, Any], *, run: str, serve: str, commit: str,
           control: str | None = None) -> dict[str, Any]:
    """The verdict record of ``piece`` on the class of ``stage1`` (its ``stage1.json``) from the verifier's command's
    ``summary`` (the module docstring). ``run``: the run that produced the verdict. ``serve``: the served run the
    class came from. ``commit``: the commit the verifier ran at. ``control``: the negative control the run is, or
    None on the honest run. ValueError for what the module docstring's "Refused" lists and an argument that names
    nothing."""
    if piece not in PIECES:
        raise ValueError(f"a verdict's piece is one of {', '.join(PIECES)}, not {piece!r}")
    if control is not None and control not in CONTROLS:
        raise ValueError(f"a control is one of {', '.join(CONTROLS)}, or none on the honest run, not {control!r}")
    for name, value in (("run", run), ("serve", serve), ("commit", commit)):
        if type(value) is not str or not value:
            raise ValueError(f"a verdict's record names its {name}")
    p = PIECES[piece]
    boundary, digest = _named(stage1, "boundary record")
    word, registered = p.read(stage1, summary, boundary, digest)
    return {"piece": piece, "control": control, "run": run, "verifier": {"module": p.module, "commit": commit},
            "verdict": word, "proved_by": None, "open": p.open,
            "registration": digest, "serve": serve, "class": class_of(boundary), "rate": rate(registered)}


def append(path: Path, rec: Mapping[str, Any]) -> None:
    """``rec`` as one more line of the verdicts file ``path``. ValueError where the file already holds a record on its
    piece and control: a run holds one verdict on each."""
    lines = path.read_text().splitlines() if path.exists() else []
    for line in filter(str.strip, lines):
        got = json.loads(line)
        if (got.get("piece"), got.get("control")) == (rec["piece"], rec["control"]):
            raise ValueError(f"{path} already holds a verdict on {rec['piece']}"
                             + ("" if rec["control"] is None else f" with the control {rec['control']}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(dict(rec), sort_keys=True) + "\n")


def _head() -> str:
    got = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return got.stdout.strip() if got.returncode == 0 else ""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("piece", choices=list(PIECES))
    ap.add_argument("--stage1", required=True, help="the class's stage1.json (two_stage.py first)")
    ap.add_argument("--summary", required=True, help="the verifier's command's summary.json on that class")
    ap.add_argument("--serve", required=True, help="the id of the served run the class came from")
    ap.add_argument("--run", default=os.environ.get(RUN_ENV), help=f"the id of the run that produced the verdict (${RUN_ENV})")
    ap.add_argument("--commit", help="the commit the verifier ran at (this tree's HEAD)")
    ap.add_argument("--control", choices=CONTROLS, help="the negative control this run is; none on the honest run")
    ap.add_argument("--out", required=True, help=f"the run's directory of kept files: the record is one more line of its {FILE}")
    a = ap.parse_args(argv)
    run, commit = a.run or "", a.commit or _head()
    if lacks := [flag for flag, got in ((f"--run (or ${RUN_ENV}, which `research run` sets)", run),
                                        ("--commit (this tree is no git checkout)", commit)) if not got]:
        ap.error(f"a verdict's record names the run that produced it and the commit its verifier ran at: {', '.join(lacks)}")
    try:
        rec = record(a.piece, json.loads(Path(a.stage1).read_text()), json.loads(Path(a.summary).read_text()),
                     run=run, serve=a.serve, commit=commit, control=a.control)
        append(Path(a.out) / FILE, rec)
    except (ValueError, KeyError, OSError) as e:
        print(f"verdict_record: no record written: {e}", file=sys.stderr)
        return 1
    print("VERDICT " + json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
