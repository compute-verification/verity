"""PoUW's leg of all-protocols: one period of the torch loop through PoUW's Core.Protocol, `verity/pouw/v1`.

``leg(period_dir, node, out)`` reads the period's records as `verity_torch_loop.serve` writes them (``registration.json``,
``circuit.json``, ``index.json``) and answers the period with PoUW's window check (`core_check.answer`, the Python
reference the difftest holds to the Lean `pouw-check`). With ``VERITY_POUW_CHECK`` naming a built ``pouw-check``, the
Lean answers too and must agree with it byte for byte. It returns the canonical answer {commitment, profile, protocol,
public, reason, verdict}, and writes into the directory ``out``:

* ``pouw.case.json``: the case, ``{"commitment": hex, "evidence": hex, "params": JSON}``, what ``pouw-check serve`` and a
  composition's component case read;
* ``pouw.answer.json``: the answer's canonical line;
* ``pouw.detail.json``: what the loop served (units, K, MACs, max_tokens), labelled not certified; the certified work W
  (the accepted window's ``work.credited``, MACs); ε at each δ of `QUOTE_DELTAS`, what the window's draw establishes
  over W (`window.eps_at`; null when nothing is drawn or no ε <= 1 is established); which verifiers answered; and an H1
  window's operands (`operands.stated`: their root X, each value's root, and what they don't establish yet, among it
  the A rows' tie to the served x rows, a gather, CIR-100), null when it names none.

**The H1 window** (the loop's ``--pouw h1`` mode): when ``<period_dir>/pouw/registration.json`` exists, the case is that
window's. Its commitment is that one-stage registration, and its evidence the proof service's answer for it,
``<period_dir>/pouw/answer.json`` (when absent the evidence is empty, which the check refuses). The parameters are the verifier's own, never
the registration's: the check item `h1_check` (the H1 tile `core_check.TILES` pins, its tile work and query, the scheme
and the entries bound `H1_ENTRIES`; coins live exactly when the loop wrote its salt session's ``salt.json`` beside the
window, which also gives the parameters' ``live``, else seeded), the period's run as the window's context (the integrity registration's ``window.run``), ε = `H1_EPS` at δ = `H1_DELTA`, the
declared range `H1_RANGE` and the fixed length the period's labels give. A registration made at anything else is
refused by the check.

**Without one** the period has no PoUW window, and the leg raises `NoWindow` (the runner's ``--pouw h1`` writes one),
which the runner records as the work row's error, so the composed verdict waits on work instead of rejecting. The
integrity window is not PoUW's to answer, under any law of the integrity plan (``subset:K``, or ``--tie``'s
``work:K``). It is one-stage's: its registration carries none of PoUW's items (`core_check.REGISTERED`) and no
context, and its template ``ScaledMmS8Coordinate_v1{K=128}`` is none of the tiles PoUW's verifier of record pins (FP8
Pc8 tiles and the salted H1 tile at K = 64). So `verity/pouw/v1` refuses it at its parameters, which would reject an
honest period by reject dominance.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import operands as OP
from verity.core.protocols.pouw import window as Wd

LEAN_ENV = "VERITY_POUW_CHECK"
H1_DIR = "pouw"
[H1_TEMPLATE] = [n for n in CC.TILES if n.startswith("PouwH1Tile_v1{")]
H1_ENTRIES = 1024
H1_EPS = "1/2"
H1_DELTA = "1/1000000"
H1_RANGE = [1, 1 << 16]
QUOTE_DELTAS = ("1/100", "1/1000000")
SALT = "salt.json"  # {"answer": issued.answer, "live": issued.live}, written beside the window by the loop (window_salt.issue)
#: core's name for a live window's salt-session inputs in its parameters (#4405); None where core takes no live parameters
LIVE_PARAM = getattr(CC, "LIVE_PARAM", None)


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


class NoWindow(ValueError):
    """The period has no PoUW window (module docstring)."""


def _live(d: Path) -> dict | None:
    """The salt session's inputs the loop wrote beside the window in `d` (``SALT``), or None for a seeded window; always
    None where core takes no live parameters (`LIVE_PARAM`)."""
    f = d / SALT
    return json.loads(f.read_text())["live"] if LIVE_PARAM is not None and f.exists() else None


def h1_check(live: bool = False) -> dict:
    """The check item an H1 window of the loop registers: the pinned H1 tile's, its coins live when its salt is."""
    tile = CC.TILES[H1_TEMPLATE]
    return {**({Wd.COINS: "live"} if live else {}), "entries": H1_ENTRIES, "scheme": CC.SCHEME, "template": H1_TEMPLATE,
            "tile_work": tile.tile_work}


def h1_params(period_dir: Path, work: dict) -> dict:
    """The verifier's parameters for the period's H1 window (module docstring)."""
    rec = json.loads((period_dir / "registration.json").read_text())
    live = _live(period_dir / H1_DIR)
    return {"check": h1_check(live is not None), "context": rec["window"]["run"], "delta": H1_DELTA, "eps": H1_EPS,
            "grid_bits": Wd.GRID_BITS, "length": {"max_tokens": work["max_tokens"], "stop": Wd.STOP_FIXED},
            "query": CC.TILES[H1_TEMPLATE].query, "range": list(H1_RANGE),
            **({LIVE_PARAM: live} if live is not None else {})}


def case(period_dir: Path) -> tuple[dict, dict]:
    """The period's case for `verity/pouw/v1` and what the loop served."""
    work = served(period_dir)
    h1 = period_dir / H1_DIR
    if not (h1 / "registration.json").exists():
        raise NoWindow(f"the period has no PoUW window ({H1_DIR}/registration.json; the runner's --pouw h1 writes it), "
                       "and its integrity window is one-stage's, which PoUW refuses under any law")
    rec, answers, p = json.loads((h1 / "registration.json").read_text()), h1 / "answer.json", h1_params(period_dir, work)
    evidence = CC.line(json.loads(answers.read_text())).encode() if answers.exists() else b""
    return {"commitment": CC.line(rec).encode().hex(), "evidence": evidence.hex(), "params": p}, work


def quotes(c: dict, answer: dict) -> list[dict]:
    """ε at each δ of `QUOTE_DELTAS` that the accepted window's draw establishes over its credited work W, as
    `window.eps_at` computes it (``eps`` null when there is no accepted draw or it establishes no ε <= 1)."""
    public = answer["public"] or {}
    work, drawn, tw = public.get("work"), public.get("drawn"), c["params"]["check"]["tile_work"]
    out = []
    for d in QUOTE_DELTAS:
        eps = None
        if work and drawn:
            try:
                eps = str(Wd.eps_at(len(drawn), Fraction(d), work["credited"], work["n"], tw))
            except ValueError:
                eps = None
        out.append({"delta": d, "eps": eps})
    return out


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
    evidence = period_dir / H1_DIR / "answer.json"
    w = (answer["public"] or {}).get("work")
    detail = {"node": node, "protocol": CC.ID, "window": "h1", "served": work,
              "certified_work": w, "W": w["credited"] if w else 0, "quotes": quotes(c, answer),
              "evidence": str(evidence.relative_to(period_dir)) if evidence.exists() else None,
              "answered_by": ["core_check"] + (["pouw-check"] if lean is not None else []),
              "operands": OP.stated(json.loads((period_dir / H1_DIR / "registration.json").read_text())["window"])}
    (out / "pouw.detail.json").write_text(json.dumps(detail, indent=1, sort_keys=True) + "\n")
    return answer
