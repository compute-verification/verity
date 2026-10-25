"""PoUW's leg of all-protocols: one period of the torch loop through PoUW's Core.Protocol, `verity/pouw/v1`.

``leg(period_dir, node, out)`` reads the period's records as `verity_torch_loop.serve` writes them (``registration.json``,
``circuit.json``, ``index.json``) and answers the period with PoUW's window check (`core_check.answer`, the Python
reference the difftest holds to the Lean `pouw-check`). With ``VERITY_POUW_CHECK`` naming a built ``pouw-check``, the
Lean answers too and must agree with it byte for byte. It returns the canonical answer {commitment, profile, protocol,
public, reason, verdict}, and writes into the directory ``out``:

* ``pouw.case.json``: the case, ``{"commitment": hex, "evidence": hex, "params": JSON}``, what ``pouw-check serve`` and a
  composition's component case read;
* ``pouw.answer.json``: the answer's canonical line;
* ``pouw.detail.json``: what the loop served (units, K, MACs, max_tokens), labelled not certified, and which verifiers
  answered.

The parameters are the verifier's for crediting this period's work: the loop's template with its work per unit (K MACs),
the registration's query and run, ε = 1/1000 at δ = 2^-40, and the fixed length the records' labels give (each request's
decode steps). The evidence is one-stage's answer for the registration, ``<period_dir>/answers/sampled-proofs.json``
when the sampled-proofs leg wrote one, else none. PoUW's verifier of record pins its tile templates
(`core_check.TILES`: FP8 Pc8 tiles and the salted H1 tile at K = 64), so it refuses the loop's
``ScaledMmS8Coordinate_v1{K=128}`` at its parameters, before it reads the registration or the evidence: no W is
certified. A per-period credit of the loop's s8 GEMMs is H1's (`credit.period_credit`), once the loop serves them as H1
tiles under a period's live coins.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import window as Wd

EPS = "1/1000"
DELTA = str(CR.DELTA)
ANSWERS = "answers/sampled-proofs.json"
LEAN_ENV = "VERITY_POUW_CHECK"


def served(period_dir: Path) -> dict:
    """What the period's loop served, from its records: the template, its units, K, MACs, and each request's max_tokens
    (its decode steps, requests in order of their labels)."""
    circuit = json.loads((period_dir / "circuit.json").read_text())
    [(template, units)] = circuit["templates"]
    k = int(re.search(r"\{K=([0-9]+)", template).group(1))
    steps: dict[str, set[int]] = defaultdict(set)
    for member in json.loads((period_dir / "index.json").read_text())["members"]:
        for _, _, label in member["labels"]:
            steps[label["request"]].add(label["step"])
    order = sorted(steps, key=lambda r: (len(r), r))
    return {"template": template, "units": units, "k": k, "macs": units * k,
            "max_tokens": [len(steps[r]) for r in order]}


def params(period_dir: Path, work: dict) -> dict:
    """The verifier's parameters for crediting the period's work under `verity/pouw/v1`."""
    rec = json.loads((period_dir / "registration.json").read_text())
    check = {Wd.COINS: "live", "entries": 2, "scheme": CC.SCHEME, "template": work["template"], "tile_work": work["k"]}
    return {"check": check, "context": rec["window"]["run"], "delta": DELTA, "eps": EPS, "grid_bits": Wd.GRID_BITS,
            "length": {"max_tokens": work["max_tokens"], "stop": Wd.STOP_FIXED}, "query": rec["query"],
            "range": [1, rec["population"]]}


def case(period_dir: Path) -> tuple[dict, dict]:
    """The period's case for `verity/pouw/v1` and what the loop served."""
    work = served(period_dir)
    rec = json.loads((period_dir / "registration.json").read_text())
    answers = period_dir / ANSWERS
    evidence = CC.line(json.loads(answers.read_text())).encode() if answers.exists() else b""
    return ({"commitment": CC.line(rec).encode().hex(), "evidence": evidence.hex(), "params": params(period_dir, work)},
            work)


def lean_line(exe: str, c: dict) -> str:
    """`pouw-check check PARAMS COMMITMENT EVIDENCE`'s outcome line on the case."""
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (("params", CC.line(c["params"]).encode()), ("commitment", bytes.fromhex(c["commitment"])),
                           ("evidence", bytes.fromhex(c["evidence"]))):
            (Path(d) / name).write_bytes(data)
            files.append(str(Path(d) / name))
        p = subprocess.run([exe, "check", *files], capture_output=True, text=True)
    if p.returncode not in (0, 1, 2):  # accept, reject, undecided (`Core.Protocol.main`)
        raise RuntimeError(f"pouw-check exited {p.returncode}: {p.stderr.strip()}")
    return p.stdout.strip()


def leg(period_dir: Path, node: str, out: Path) -> dict:
    """The period's PoUW answer (module docstring); ``node`` is where the loop ran, recorded in the detail."""
    period_dir, out = Path(period_dir), Path(out)
    c, work = case(period_dir)
    answer = CC.case(c)
    line = CC.line(answer)
    lean = None
    if exe := os.environ.get(LEAN_ENV):
        lean = lean_line(exe, c)
        if lean != line:
            raise RuntimeError(f"pouw-check and core_check disagree on the period:\n  lean   {lean}\n  python {line}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pouw.case.json").write_text(CC.line(c) + "\n")
    (out / "pouw.answer.json").write_text(line + "\n")
    detail = {"node": node, "protocol": CC.ID, "served": work, "certified_work": (answer["public"] or {}).get("work"),
              "evidence": ANSWERS if (period_dir / ANSWERS).exists() else None,
              "answered_by": ["core_check"] + (["pouw-check"] if lean is not None else [])}
    (out / "pouw.detail.json").write_text(json.dumps(detail, indent=1, sort_keys=True) + "\n")
    return answer
