"""A stop class's verdict records: what a verifier's command wrote of one class (its ``summary.json``) as one JSON
object, a line of a run's ``verdicts.jsonl``, saying which verdict it is and which class, serve, registration and rate
it is of.

    python benchmarks/one_stage/verdict_record.py PIECE --first F --summary F --serve RUN --run RUN --out DIR \\
      [--stream F] [--commit SHA] [--control id|max_tokens]

``--first`` is the class's first-stage file, the one `two_stage.py first` writes for it: the class's boundary record
and its digest and, for a stop class, its stop record and that one's (its ``stop``). ``--stream`` is the registration
record of the verdict's own stream where the first-stage file does not hold it: the second stage's interior
registration, which its command writes as ``interior/registration.json`` under its ``--out``.

A stop class's response lengths are held by a verdict on each of its pieces, each accepted and complete, and a reader
of those verdicts has to be able to refuse a set that is not one class's: verdicts of different classes, serves or
registrations, or one that ran at another rate than its registration registers. So a record holds the fields
`benchmarks/harness/verity_bench/exfiltration.py` reads (`VERDICT_FIELDS`: ``piece``, ``control``, ``run``,
``verifier``, ``verdict``, ``proved_by``, ``open``), four that say what the verdict is of, and two that hold the
verdict's own stream to that class:

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
* ``stream_registration``: the digest of the registration record of the verdict's own stream, the one its verifier
  sampled and proved on: the stop record for the stop audit, the interior registration for the second stage. Null
  only on a second stage that refused before it registered an interior.
* ``stream_names_boundary``: true where that record is the one with that digest and its ``previous`` is the class's
  boundary record, so the stream was registered after this class's boundary record and no other's; null beside a
  null ``stream_registration``. A stream registered after another record gets no verdict record: a verdict that
  named the right ``registration`` over a stream registered against another would be of no one class.

**The pieces** (`PIECES`), each a verifier's command, its stream's registration and the record that registers its rate:

* ``stop_record``: the stop audit (`stop_audit.py`), whose stream's registration is the stop record (the first-stage
  file's ``stop.record``), which registers the rate too. ``accept`` when its audit record accepted and is complete,
  ``not complete`` when it accepted with entries that did not run, else ``refuse``.
* ``second_stage``: the compute audit's second stage (`two_stage.py second`), whose stream's registration is the
  interior registration (``--stream``), held to the digest its summary names. Its rate is the class's boundary
  record's, the replay sample's: the interior registration registers every proof block of the sampled spans.
  ``refuse`` unless the second stage held (`two_stage.held`). Then ``accept`` or ``not complete`` as its outcome
  accepted with or without ``complete`` (an outcome that carries no ``complete`` is not complete), and ``withheld``
  when its outcome did not accept: a second stage that held reaches that only where the verifier's own two-stage
  form refused it (`two_stage.FORM_REFUSED`).

``proved_by`` is null on both: no Lean guarantee covers either command, and ``open`` says what stands between its
verdict and proved (`Piece.open`).

**Refused** (`record`, `append`): a summary that does not name this class's registration, so a command that refused
before it registered gets no record, since nothing ties what it wrote to the class; a boundary or stop record that is
not the one whose digest the first-stage file names; a stream's registration record that is not the one its summary
names, or whose ``previous`` is not the class's boundary record; a second stage that held and names no interior
registration; and a second record on one piece and control in one file."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.protocols.proofs.spot.verifier.one_stage import registration as R

HERE = Path(__file__).resolve().parent
#: a run's file of verdict records, one JSON object a line
FILE = "verdicts.jsonl"
#: the negative controls: a run on the same path with one thing changed (null on the honest run)
CONTROLS = ("id", "max_tokens")
ACCEPT, NOT_COMPLETE, REFUSE, WITHHELD = "accept", "not complete", "refuse", "withheld"
#: a registration record's key for its sampling design, and that design's key for its kind
DESIGN = "law"


def rate(registered: Mapping[str, Any]) -> str:
    """The rate a registration record registers its population at: K of N as NUM/DEN in lowest terms for a subset
    design, so K = N reads ``1/1``; any other design as itself (``work:K``), which is no rate and never ``1/1``.
    ValueError where the record registers no design, or a subset that is not 1 <= K <= N."""
    design, n = registered.get(DESIGN), registered.get("population")
    if not isinstance(design, Mapping) or type(design.get(DESIGN)) is not str or type(design.get("k")) is not int:
        raise ValueError(f"the registration record registers no sampling design, but {design!r}")
    kind, k = design[DESIGN], design["k"]
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
    """The record ``held`` (the first-stage file, or its ``stop``) holds and its digest, the one it names beside it."""
    rec = held.get("record") if isinstance(held, Mapping) else None
    if not isinstance(rec, Mapping):
        raise ValueError(f"the first-stage file holds no {what}")
    if R.digest(rec) != held.get("registration"):
        raise ValueError(f"the first-stage file's {what} is not the record whose digest it names as its registration")
    return rec, held["registration"]


def _stop_record(first: Mapping[str, Any], summary: Mapping[str, Any], boundary: Mapping[str, Any], digest: str,
                 stream: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any], str | None]:
    rec, stop = _named(first.get("stop"), "stop record")
    if stream is not None and R.digest(stream) != stop:
        raise ValueError("the stop audit's stream is registered by the first-stage file's stop record, not the record handed")
    if rec.get("previous") != digest:
        raise ValueError("the stop record's `previous` is not the class's boundary record")
    if summary.get("registration") != {"boundary": digest, "stop": stop}:
        raise ValueError("the stop audit's summary does not name this class's boundary record and stop record as its "
                         f"`registration`, but {summary.get('registration')!r}: nothing ties it to the class")
    word = REFUSE if summary.get("accepted") is not True else ACCEPT if summary.get("complete") is True else NOT_COMPLETE
    return word, rec, stop


def _second_stage(first: Mapping[str, Any], summary: Mapping[str, Any], boundary: Mapping[str, Any], digest: str,
                  stream: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any], str | None]:
    named = summary.get("first_stage")
    if not isinstance(named, Mapping) or named.get("registration") != digest:
        raise ValueError("the second stage's summary does not name this class's boundary record as its first stage's "
                         f"`registration`, but {named!r}: nothing ties it to the class")
    outcome = summary.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    if summary.get("held") is not True:
        word = REFUSE
    elif outcome.get("accepted") is True:
        word = ACCEPT if outcome.get("complete") is True else NOT_COMPLETE
    else:
        word = WITHHELD
    second = summary.get("second_stage")
    own = second.get("registration") if isinstance(second, Mapping) else None
    if own is None:
        if word != REFUSE or stream is not None:
            raise ValueError("the second stage's summary names no interior registration (`second_stage.registration`)"
                             + (", and a second stage that held registered one" if stream is None else
                                " for the record handed"))
        return word, boundary, None
    if not isinstance(stream, Mapping) or R.digest(stream) != own:
        raise ValueError("the second stage's summary names its interior registration, and the record handed "
                         "(`--stream`: its `interior/registration.json`) is "
                         + ("none" if stream is None else "not the record with that digest"))
    if stream.get("previous") != digest:
        raise ValueError("the interior registration's `previous` is not the class's boundary record")
    return word, boundary, own


@dataclass(frozen=True)
class Piece:
    """A verdict on a stop class. `module`: the verifier's command, as a record names it. `open`: what stands between
    its verdict and proved, in words. `read`: (the first-stage file, the command's ``summary.json``, the class's
    boundary record, its digest, the stream's registration record where one is handed) -> the verdict's word, the
    registration record that registers its rate, and the digest of its own stream's registration record, held to
    name the boundary record as its ``previous`` (None where the command registered none)."""

    module: str
    open: str
    read: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str, Mapping[str, Any] | None],
                   tuple[str, Mapping[str, Any], str | None]]


PIECES = {
    "stop_record": Piece("benchmarks/one_stage/stop_audit.py",
                         "no Lean guarantee covers the stop audit's command, which is Python around Lean's verdicts",
                         _stop_record),
    "second_stage": Piece("benchmarks/one_stage/two_stage.py", "the two-stage form's soundness is not proved",
                          _second_stage),
}


def record(piece: str, first: Mapping[str, Any], summary: Mapping[str, Any], *, run: str, serve: str, commit: str,
           control: str | None = None, stream: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The verdict record of ``piece`` on the class of ``first`` (its first-stage file) from the verifier's command's
    ``summary`` (the module docstring). ``run``: the run that produced the verdict. ``serve``: the served run the
    class came from. ``commit``: the commit the verifier ran at. ``control``: the negative control the run is, or
    None on the honest run. ``stream``: the registration record of the verdict's own stream, where ``first`` does not
    hold it (the module docstring's ``--stream``). ValueError for what the module docstring's "Refused" lists and an
    argument that names nothing."""
    if piece not in PIECES:
        raise ValueError(f"a verdict's piece is one of {', '.join(PIECES)}, not {piece!r}")
    if control is not None and control not in CONTROLS:
        raise ValueError(f"a control is one of {', '.join(CONTROLS)}, or none on the honest run, not {control!r}")
    for name, value in (("run", run), ("serve", serve), ("commit", commit)):
        if type(value) is not str or not value:
            raise ValueError(f"a verdict's record names its {name}")
    p = PIECES[piece]
    boundary, digest = _named(first, "boundary record")
    word, registered, own = p.read(first, summary, boundary, digest, stream)
    return {"piece": piece, "control": control, "run": run, "verifier": {"module": p.module, "commit": commit},
            "verdict": word, "proved_by": None, "open": p.open,
            "registration": digest, "serve": serve, "class": class_of(boundary), "rate": rate(registered),
            "stream_registration": own, "stream_names_boundary": None if own is None else True}


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
    ap.add_argument("--first", required=True, help="the class's first-stage file, which two_stage.py first writes")
    ap.add_argument("--summary", required=True, help="the verifier's command's summary.json on that class")
    ap.add_argument("--stream", help="the registration record of the verdict's own stream where the first-stage file "
                                     "does not hold it: the second stage's interior/registration.json")
    ap.add_argument("--serve", required=True, help="the id of the served run the class came from")
    ap.add_argument("--run", required=True, help="the id of the run that produced the verdict")
    ap.add_argument("--commit", help="the commit the verifier ran at (this tree's HEAD)")
    ap.add_argument("--control", choices=CONTROLS, help="the negative control this run is; none on the honest run")
    ap.add_argument("--out", required=True, help=f"the run's directory of kept files: the record is one more line of its {FILE}")
    a = ap.parse_args(argv)
    commit = a.commit or _head()
    if not commit:
        ap.error("a verdict's record names the commit its verifier ran at: --commit (this tree is no git checkout)")
    try:
        rec = record(a.piece, json.loads(Path(a.first).read_text()), json.loads(Path(a.summary).read_text()),
                     run=a.run, serve=a.serve, commit=commit, control=a.control,
                     stream=json.loads(Path(a.stream).read_text()) if a.stream else None)
        append(Path(a.out) / FILE, rec)
    except (ValueError, KeyError, OSError) as e:
        print(f"verdict_record: no record written: {e}", file=sys.stderr)
        return 1
    print("VERDICT " + json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
