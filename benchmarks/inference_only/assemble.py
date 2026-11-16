"""A tied serve's `out/registered/` and the inference-only files beside it, assembled after the serve from what the
serve wrote, for `report.py --from`: the step between a serve under the Pearl+ tie and the record check.

    python benchmarks/inference_only/assemble.py --from DIR [--first FILE ...] [--registration R] [--pins WREG]
        [--day YYYY-MM-DD] [--circuit FILE] [--phi FILE] [--window FILE]

``DIR`` is the serve's run directory or its `out/`, holding what the tied serve writes (`SERVED`): the class circuit
record (`record/circuit-record.json`), the serve's φ as stated before it served (`record/serve-phi.json`), Pearl+'s
window record W (`record/window-record.json`) and the inference-only policy. ``--circuit``, ``--phi`` and ``--window``
name the first three where a serve keeps them elsewhere: with ``--registration``, the circuit record is the one R
commits, the `circuit.json` window-commit writes beside it. It writes (`record.REGISTERED`, `OUT_FILES`):

* **The Program** (`registered/program.json`), which the tied serve writes none of: rebuilt from φ, the policy's class
  and its Pearl+ scheme by the call the serve's circuit record was written by (`program_of`), and refused unless its
  digest is the circuit record's ``program`` and, with ``--registration``, R's.
* **The circuit record** (`registered/circuit.json`): the serve's, byte for byte; one already there must be the same.
* **The registration log** (`registered/log.jsonl`, one record a line): the ``--first`` records (the deployment's
  registration of the weights, first), W, and with ``--registration`` the window's registration R chained right after
  W, the chain held from its first record (each record's ``previous`` the digest of the one before, the first's none)
  and R naming the Program's digest and the circuit record's SHA-512. Without R no record of the log names the
  Program, and the record check refuses the record (`NO_REGISTRATION`).
* **The rate rule's limits** (`inference-only-limits.json`), where the serve wrote none: α = k/c and Γ = c of the
  policy's own c and k (`limits`).
* **The count** (`block-counts-CLASS.json`): the class circuit record's inference blocks judged against the policy's
  c, k and n by the integration's count (`count`), at the verifier's pins ``--pins`` (none: every read is free) and R's
  registered roots; its exact counts go to the run directory's `private/`, outside `out/`.

A refusal of the serve's files exits 2 with its reason, before anything is written."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import policy as P
import record as R

#: what a tied serve writes under its `out/` that this reads
SERVED = {"circuit": "record/circuit-record.json", "phi": "record/serve-phi.json", "window": "record/window-record.json",
          "policy": R.OUT_FILES["policy"]}
LIMITS = "inference-only-limits.json"
LIMITS_FORMAT = "verity-bench/inference-only-limits/v1"
NO_REGISTRATION = ("no registration chained after W names the Program (--registration R, the window's registration "
                   "that `prove_path.py window-commit --pearl-passes` writes), so the record check refuses the record")


class Refused(Exception):
    """The serve's files can't be assembled as they are; the message says why."""


def digest(rec: Mapping[str, Any]) -> str:
    """A registration record's digest, the one each record's ``previous`` names (`docket.digest`)."""
    from verity.protocols.proofs.spot.verifier.one_stage.docket import digest as of

    return of(rec)


def program_digest(text: str) -> str:
    """The digest of a Program's v1 descriptor ``text``, the one a circuit record and a registration name it by."""
    from verity.circuits.codec import program_digest as of

    return of(json.loads(text))


def program_of(phi: Mapping[str, Any], cls: tuple[int, int, int], scheme: str) -> str:
    """The class Program a serve of φ ``phi`` at class ``cls`` cut at Pearl+'s linears under ``scheme`` registers, as
    its v1 descriptor's canonical JSON: the call the tied serve's circuit record is written by
    (`prove_path.py class-circuit`: `window_circuit.class_circuit` on `registered_of_serve`, with every option the
    serve leaves at its default)."""
    sys.path.append(str(P.ROOT / "benchmarks" / "prove_path"))
    import prove_path as PP
    from verity.circuits.codec import canonical_json, encode_program
    from verity_vllm.program.predict.window_circuit import PearlTables, class_circuit

    registered = PP.registered_of_serve(phi, None, None, PP.CAP_BITS)
    circuit = class_circuit(registered, PP.window_class_of(",".join(map(str, cls))), tables=PearlTables(scheme))
    return canonical_json(encode_program(circuit.program()[0])).decode()


def limits(policy_bytes: bytes) -> dict[str, Any]:
    """The rate rule's limits fixed from the policy alone: α = k/c and Γ = c, in the unit the policy counts c in."""
    p = json.loads(policy_bytes)
    a = Fraction(p["k"], p["c"])
    return {"format": LIMITS_FORMAT, "alpha": [a.numerator, a.denominator], "gamma": int(p["c"]),
            "unit": "Boolean gates by the policy's gate table, the unit of its c",
            "count": {x: p.get(x) for x in ("c", "k", "n")},
            "from": "the policy's c and k (alpha = k/c, gamma = c)",
            "policy_sha512": hashlib.sha512(policy_bytes).hexdigest(), "written_by": "the developer"}


def count(circuit: Path, roots: Path | None, pins: Path | None, policy: Path, day: str, out: Path,
          private: Path) -> None:
    """The class circuit record's inference blocks judged against the policy by the integration's count
    (`block_counts`), the public judgment to ``out`` and the exact counts to ``private``."""
    from verity_vllm.program.predict import block_counts as BC

    # the count prints its judgment, which goes to ``out``; only its one line on stderr is kept
    with contextlib.redirect_stdout(io.StringIO()):
        code = BC.main(BC.Options(day=day, declaration=policy, private=private,
                                  window=[[str(circuit), str(roots or "-")]], pins=pins, out=out))
    if code != 0:
        raise Refused(f"the count wrote no judgment of {circuit}")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: can't be read as JSON ({type(e).__name__}: {e})") from None


def chained(log: Sequence[Mapping[str, Any]]) -> None:
    """Refused unless each record's ``previous`` is the digest of the record before it, the first's none."""
    before = None
    for k, rec in enumerate(log):
        if rec.get("previous") != before:
            raise Refused(f"the log's record {k} follows {str(rec.get('previous'))[:16]}..., not "
                          f"{'nothing' if before is None else before[:16] + '...'}: the chain the check holds is broken")
        before = digest(rec)


def assemble(served: Path, first: Sequence[Path] = (), registration: Path | None = None, pins: Path | None = None,
             day: str | None = None, named: Mapping[str, Path] | None = None,
             build: Callable[[Mapping[str, Any], tuple[int, int, int], str], str] = program_of,
             counted: Callable[..., None] = count) -> list[str]:
    """The tied serve's files under ``served`` assembled (module docstring), each of `SERVED` at its place there
    unless ``named`` names it apart; what was written, a line each."""
    out = served / "out" if (served / "out").is_dir() and not (served / SERVED["policy"]).exists() else served
    files = {what: Path((named or {}).get(what) or out / rel) for what, rel in SERVED.items()}
    for what, path in files.items():
        if not path.is_file():
            raise Refused(f"the serve wrote no {what} ({path}): this assembles a serve under the Pearl+ tie")
    policy_bytes = files["policy"].read_bytes()
    policy, body = json.loads(policy_bytes), files["circuit"].read_bytes()
    try:
        cls = tuple(int(policy["class"][x]) for x in ("slots", "prompt", "outputs"))
        scheme = policy["pearl_tables"]
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"{SERVED['policy']}: no {e}") from None
    if not isinstance(scheme, str) or not scheme:
        raise Refused(f"{SERVED['policy']}: pearl_tables names no Pearl+ scheme, so the class isn't cut at its linears")
    want = json.loads(body).get("program")
    w = _json(files["window"])
    log = [*(_json(p) for p in first), w]
    if registration is not None:
        reg = _json(registration)
        if reg.get("program") != want:
            raise Refused(f"R names Program {str(reg.get('program'))[:16]}..., not the circuit record's {str(want)[:16]}...")
        if (reg.get("window") or {}).get("circuit") != hashlib.sha512(body).hexdigest():
            raise Refused("R's window.circuit is not the SHA-512 of the serve's circuit record")
        log.append(reg)
    chained(log)
    text = build(_json(files["phi"]), cls, scheme)
    if (got := program_digest(text)) != want:
        raise Refused(f"the Program rebuilt from the serve's φ, class {','.join(map(str, cls))} and {scheme} has digest "
                      f"{got[:16]}..., not the {str(want)[:16]}... the circuit record names")
    reg_dir = out / "registered"
    held = out / R.REGISTERED["circuit"]
    if held.exists() and held.read_bytes() != body:
        raise Refused(f"{held} is not the serve's circuit record ({files['circuit']}) byte for byte")
    reg_dir.mkdir(exist_ok=True)
    said = []
    (out / R.REGISTERED["program"]).write_text(text)
    said.append(f"registered/program.json: the Program rebuilt, digest {got[:16]}..., the circuit record's"
                + (" and R's" if registration is not None else ""))
    if not held.exists():
        shutil.copyfile(files["circuit"], held)
    said.append(f"registered/circuit.json: the serve's circuit record, SHA-512 {hashlib.sha512(body).hexdigest()[:16]}...")
    (out / R.REGISTERED["log"]).write_text("".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n"
                                                  for r in log))
    said.append(f"registered/log.jsonl: {len(first)} first record(s), W"
                + (", R chained right after W" if registration is not None else f"; {NO_REGISTRATION}"))
    if not (out / LIMITS).exists():
        doc = limits(policy_bytes)
        (out / LIMITS).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        said.append(f"{LIMITS}: alpha = {doc['alpha'][0]}/{doc['alpha'][1]} and gamma = {doc['gamma']}, {doc['from']}")
    tag = "-".join(map(str, cls))
    run_dir = out.parent if out.name == "out" else out
    private = run_dir / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    counted(held, registration, pins, files["policy"], day or str(policy.get("written_utc", ""))[:10],
            out / f"block-counts-{tag}.json", private / f"block-counts-{tag}.json")
    said.append(f"block-counts-{tag}.json: the class circuit record's blocks judged against the policy's c, k and n"
                + ("" if pins is not None else ", every read free (no --pins)"))
    return said


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="served", type=Path, required=True, metavar="DIR",
                    help="the tied serve's run directory, or its out/")
    ap.add_argument("--first", type=Path, nargs="+", default=[], metavar="FILE",
                    help="the registration records W follows, the first first: the deployment's registration of the weights")
    ap.add_argument("--registration", type=Path, metavar="R", help="the window's registration, chained right after W")
    ap.add_argument("--pins", type=Path, metavar="WREG", help="the verifier's pins: the committer's weights registration")
    ap.add_argument("--day", help="the UTC day served, YYYY-MM-DD (default the policy's written_utc day)")
    for what in ("circuit", "phi", "window"):
        ap.add_argument(f"--{what}", type=Path, metavar="FILE",
                        help=f"the serve's {what} where it isn't at DIR's {SERVED[what]}")
    a = ap.parse_args(argv)
    try:
        said = assemble(a.served, a.first, a.registration, a.pins, a.day,
                        {x: getattr(a, x) for x in ("circuit", "phi", "window") if getattr(a, x) is not None})
    except (Refused, R.Refused) as e:
        print(f"assemble: refused: {e}", file=sys.stderr)
        return 2
    print("\n".join(said))
    return 0


if __name__ == "__main__":
    sys.exit(main())
