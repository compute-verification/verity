"""PoUW's piece of the composed checks (`composite.py --cases`): its case and its answer on it, in one directory.

    pouw_piece.py v1 --out DIR [--exe PATH] (--params P --registration R --answer A | --serve-run SERVE_DIR --window K)
    pouw_piece.py period --out DIR [--exe PATH] (--key F --ledger F --pins F --main F --name NAME --windows F
                                                 | --serve-run SERVE_DIR --window K [--pouw DIR --key-sha512 HEX
                                                   --main F --statements F --live-evidence F])

``v1`` is remote's PoUW (`verity/pouw/v1`, `pouw-check`, `PouwCheck/Protocol.lean`): its params the verifier's
`Params` (``--params``), its commitment the window's registration, one-stage's record (``--registration``, written as
canonical JSON, whose digest the proof service's answer cites), its evidence the service's answer for it (``--answer``).
``period`` is onsite's (`verity/pouw/period/v1`, `pouw-period`, `PouwCheck/Period.lean`): its params ``{key, main,
name}`` (the challenger key it pins, ``--key``, as `ChallengerKey.to_json` writes it or as the deployment's key file
holds it; its copy of main's first-parent history, ``--main``, one commit a line; the period, ``--name``), its
commitment the auditor's ledger (``--ledger``, its lines), its evidence ``{pins, windows}`` (the host's pins log,
``--pins``, its lines; each window's evidence in the close's order, ``--windows``, a JSON list in `credit.PROVED_KEYS`'s
form). Windows carrying their statements' proof files instead (`credit.PERIOD_KEYS`, the drawn calls' ``statements``
and the live ``statements``) are put in that form by `proved`: credit's own period reference runs the verifier of
record on them, and each session's answer is the outcome of each unit it proved, so the Lean judges every one.

The directory, private as `composite.py` reads it (0700, its files 0600: the evidence and params are witness), gets
``params.json``, ``commitment.bin`` and ``evidence.json`` (canonical), ``case.json`` (``{params, commitment, evidence,
blobs}``, the commitment and evidence hex, no blobs: neither protocol reads any) and ``answer.json``: the line ``EXE
check params.json commitment.bin evidence.json`` prints, the Lean's (``--exe``, its package's ``pouw-check`` or
``pouw-period``), or without ``--exe`` the Python reference's (`core_check.case`, `period_check.case`, which the
difftests hold to the Lean). Prints one ``PIECE`` line and exits 0 when the answer accepts, 1 when it rejects, 2 when
it is undecided, as the check does.

``--serve-run SERVE_DIR --window K`` takes the inputs from a served run (`served_verdicts.sh`'s serve dir, or its
``out/serve``), and names its piece MISSING, with why, wherever the run doesn't hold them (`v1_missing`,
`period_missing`): the directory then holds only ``missing.json`` saying why, no case and no answer, so `composite.py`
leaves the protocol out and Lean's verdict on the product names it missing. It prints one ``MISSING`` line and exits
2: nothing was decided. ``period``'s are the closed period ``SERVE_DIR/period.json`` names (`served_verdicts.py period
open|close`) holding window K alone (``--cap 1``); its ledger, pins log and window K's session as PoUW's served piece
fetched them from the gate (``--pouw``, ``pieces/pouw``: ``ledger.jsonl``, ``pins.jsonl``, ``session/<id>/out``), the
key in that session pinned by ``--key-sha512`` (the auditor's); ``--main``; the declared Program
(``h1/window-program.json.gz``); and the drawn calls' statements (``--statements``, a JSON list) and the live evidence
(``--live-evidence``, ``h1_prove/live.py evidence``'s), from the proving runs (`period_inputs`).

The ``PIECE`` line is public: the protocol, its verdict and its refusal's name (`composite.outcome`), with ``REFERENCE``,
credit's own Python verdict on the statements where `proved` ran it; the answer whole is in ``answer.json``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from composite import line, outcome, write_private

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import period_check as PC
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

V1, PERIOD = CC.ID, PC.ID
CASE = {V1: CC.case, PERIOD: PC.case}
EXIT = {"accept": 0, "reject": 1, "undecided": 2}
CASE_FILES = ("params.json", "commitment.bin", "evidence.json", "case.json", "answer.json")


class Missing(Exception):
    """Why a served run holds no piece of the protocol."""


def canonical(x: Any) -> bytes:
    return CC._canon(x).encode()


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text().split("\n") if x]


def challenger_key(doc: dict) -> dict:
    """``{H, W, public}`` (`ChallengerKey.to_json`) from itself or from a deployment's key file (``pubkey``)."""
    return {"H": doc["H"], "W": doc["W"], "public": doc["public"] if "public" in doc else doc["pubkey"]}


def answer(protocol: str, out: Path, exe: str | None) -> dict:
    """The protocol's answer on the case in `out`: the Lean's when `exe` is given, else the Python reference's."""
    if exe is None:
        case = json.loads((out / "case.json").read_text())
        return CASE[protocol]({k: case[k] for k in ("params", "commitment", "evidence")})
    p = subprocess.run([exe, "check", *(str(out / f) for f in ("params.json", "commitment.bin", "evidence.json"))],
                       capture_output=True, text=True, check=False)
    said = p.stdout.strip().splitlines()
    if p.returncode not in EXIT.values() or not said:
        raise RuntimeError(f"{exe} check exited {p.returncode}: {(p.stdout + p.stderr).strip()[-1000:]}")
    return json.loads(said[-1])


def write(out: Path, protocol: str, params: Any, commitment: bytes, evidence: bytes, exe: str | None = None) -> dict:
    """The piece in `out` (module docstring) and its answer."""
    (out / "missing.json").unlink(missing_ok=True)
    case = {"blobs": [], "commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    write_private(out, {"params.json": canonical(params), "commitment.bin": commitment, "evidence.json": evidence,
                        "case.json": (json.dumps(case, sort_keys=True) + "\n").encode()})
    a = answer(protocol, out, exe)
    if a.get("protocol") != protocol:
        raise RuntimeError(f"the answer is {a.get('protocol')}'s, not {protocol}'s")
    write_private(out, {"answer.json": canonical(a) + b"\n"})
    return a


def missing(out: Path, protocol: str, why: str) -> None:
    """A missing piece: no case and no answer, only ``missing.json`` saying why."""
    for f in CASE_FILES:
        (out / f).unlink(missing_ok=True)
    write_private(out, {"missing.json": (json.dumps({"protocol": protocol, "why": why}, sort_keys=True) + "\n").encode()})


def _window(serve_dir: Path, k: int) -> dict:
    path = serve_dir / "windows.jsonl"
    if not path.is_file():
        raise Missing(f"{serve_dir} has no windows.jsonl, so no window {k}")
    line = next((w for w in map(json.loads, lines(path)) if int(w["window"]) == k), None)
    if line is None:
        raise Missing(f"{serve_dir}'s windows.jsonl has no window {k}")
    return line


def v1_missing(serve_dir: Path, k: int) -> str:
    """Why served window `k` has no `verity/pouw/v1` case: v1 checks a tile window the proof service registered and
    answered, and a served vLLM window is H1's, or serves no PoUW."""
    try:
        h1 = _window(serve_dir, k).get("pouw_h1")
    except Missing as e:
        return str(e)
    if not h1:
        return f"window {k} served no PoUW (its line has no pouw_h1: served without --protocols)"
    return (f"window {k}'s PoUW is H1's per-call credit ({h1.get('entry')}), which {V1} doesn't check: {V1} is the tile "
            "window's check, over a one-stage registration of tiles PouwCheck.tiles pins (the Pearl-C tiles at sm120, "
            "PouwH1Tile_v1 at K = 64) and the proof service's answer to it, and an H1 window has neither; an H1 window's "
            f"Lean check is {PERIOD}, onsite's")


def period_missing(serve_dir: Path, k: int) -> str:
    """Why served window `k` has no `verity/pouw/period/v1` case: the first of its period, its record and its drawn
    units' proof outcomes it lacks."""
    try:
        h1 = _window(serve_dir, k).get("pouw_h1")
    except Missing as e:
        return str(e)
    if not h1:
        return f"window {k} served no PoUW (its line has no pouw_h1: served without --protocols)"
    session = h1.get("session") or {}
    if session.get("period") is None:
        return (f"window {k}'s session {session.get('session')} is in no period of the auditor's ledger (reserved with "
                f"no period open: a development window), and {PERIOD} credits only a period's windows, from the "
                "period-open that commits their K, rows and weights")
    if not (serve_dir / f"h1/w{k}/pouw-h1-calls.json").is_file():
        return f"window {k} has no H1 record (h1/w{k}/pouw-h1-calls.json) in {serve_dir}"
    return (f"no proof outcomes of window {k}'s drawn units and live rounds: {PERIOD} reads each one's verdict of the "
            "verifier of record (credit.PROVED_KEYS's proofs and live), from their statements and the period's "
            "ledger and session (--pouw, --statements, --live-evidence)")


def _statements(w: Any) -> list[tuple[str, dict]]:
    """Window `w`'s statements (`credit.PERIOD_KEYS`'s form) by leg: ``proofs`` its drawn calls', ``live`` its rounds'."""
    if not (isinstance(w, dict) and sorted(w) == list(Cr.PERIOD_KEYS) and isinstance(w["statements"], list)
            and isinstance(w["live"], dict) and sorted(w["live"]) == list(Cr.LIVE_STATEMENTS)
            and isinstance(w["live"]["statements"], list)):
        raise ValueError(f"a window is not {{{', '.join(Cr.PERIOD_KEYS)}}} with its live {{"
                         f"{', '.join(Cr.LIVE_STATEMENTS)}}}: {Cr.PERIOD_FORM}")
    return [("proofs", s) for s in w["statements"]] + [("live", s) for s in w["live"]["statements"]]


def _files(circuit: Any, public: Any, sessions: Any) -> tuple[str, str, tuple[str, ...]]:
    return str(circuit), str(public), tuple(map(str, sessions))


def _claimed(public: Path, session: str) -> tuple[int, ...]:
    """The units a session the verifier rejected drew (`proofs.proved_units` on its own record), which its reject
    outcome names; none when its files don't say."""
    try:
        return PF.proved_units(public, Path(session))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return ()


def proved(key: Any, ledger: list[str], pins: list[str], main: list[str], name: str, windows: list[dict],
           run: Any = PF.verify) -> tuple[list[dict], tuple[dict | None, tuple[str, str] | None]]:
    """The period's `windows` in the form ``pouw-period`` reads (`credit.PROVED_KEYS`), from windows carrying their
    statements' proof files (`credit.PERIOD_KEYS`), and credit's own verdict on them: `credit.period_credit` runs the
    verifier of record `run` on the statements, and each session it answers is one outcome a unit it proved
    (`credit.PROOF_KEYS`: accept or reject, the program it names and its registered finding), on its window's leg, the
    drawn calls' ``proofs`` or the live rounds' (a rejected session's: the units its record drew). Outcomes are
    recorded unjudged, so the Lean judges each one, and a statement the reference refused before running the verifier
    has none. ValueError on windows not in that form, or proof files two statements name."""
    where: dict[tuple, tuple[int, str]] = {}
    for i, w in enumerate(windows):
        for leg, s in _statements(w):
            at = _files(s.get("circuit"), s.get("public"), s.get("sessions") or ())
            if at in where:
                raise ValueError(f"two statements name the same proof files ({at[0]}, {at[1]}): each is proved once")
            where[at] = (i, leg)
    found: dict[tuple[int, str], list[dict]] = {(i, leg): [] for i in range(len(windows)) for leg in ("proofs", "live")}

    def recorded(proof: PF.Proof, **kw: Any) -> tuple[Any, Any]:
        got, why = run(proof, **kw)
        at = where.get(_files(proof.circuit, proof.public, proof.sessions))
        if why is None and at is not None:
            found[at] += [{"program": f.program, "registered": f.registered, "unit": u,
                           "verdict": "accept" if f.accepted else "reject"}
                          for f in got for u in (f.units if f.accepted else _claimed(proof.public, f.session))]
        return got, why

    reference = Cr.period_credit(key, ledger, pins, main, name, windows, CC.lms_verify, run=recorded)
    out = [{**{x: w[x] for x in Cr.PROVED_KEYS if x not in ("proofs", "live")}, "proofs": found[i, "proofs"],
            "live": {"descriptor": w["live"]["descriptor"], "proofs": found[i, "live"]}} for i, w in enumerate(windows)]
    return out, reference


def period_inputs(serve_dir: Path, k: int, pouw: Path, key_sha512: str, main: Path, statements: Path | None,
                  live: Path | None) -> tuple[dict, list[str], list[str], list[dict]]:
    """Period ``verity/pouw/period/v1``'s params, ledger, pins log and windows (`credit.PERIOD_KEYS`, statements) from a
    served run's kept records (module docstring), or Missing naming the first record or leg it lacks: the closed
    period ``period.json`` names holding window `k` alone, its record, the session, ledger and pins PoUW's served piece
    fetched (`pouw`), the declared Program, and the drawn calls' `statements` and the `live` evidence."""
    h1 = _window(serve_dir, k).get("pouw_h1")
    if not h1:
        raise Missing(f"window {k} served no PoUW (its line has no pouw_h1: served without --protocols)")
    session = h1.get("session") or {}
    if session.get("period") is None:
        raise Missing(period_missing(serve_dir, k))
    try:
        doc = json.loads((serve_dir / "period.json").read_text())
    except (OSError, ValueError) as e:
        raise Missing(f"{serve_dir} has no period.json ({type(e).__name__}): served_verdicts.py period open|close "
                      "writes it, the period's open and close as the gate answered them") from None
    if "close" not in doc:
        raise Missing(f"period {doc.get('name')} was never closed (period.json has no close), so it earns nothing")
    name = doc["name"]
    if session["period"] != name:
        raise Missing(f"window {k}'s session is in period {session['period']}, not the run's {name}")
    calls = serve_dir / f"h1/w{k}/pouw-h1-calls.json"
    if not calls.is_file():
        raise Missing(f"window {k} has no H1 record (h1/w{k}/pouw-h1-calls.json) in {serve_dir}")
    rec = json.loads(calls.read_text())
    sid = rec["commitment"]["session"]["session"]
    tree = pouw / "session" / sid / "out"
    for p in (pouw / "ledger.jsonl", pouw / "pins.jsonl", tree / "key.json"):
        if not p.exists():
            raise Missing(f"no {p.relative_to(pouw)} in {pouw}: PoUW's served piece (served_verdicts.py pouw) fetches "
                          "the gate's ledger, pins log and the window's session after the period's close")
    ledger = lines(pouw / "ledger.jsonl")
    close = next((json.loads(x)["body"] for x in ledger if json.loads(x).get("kind") == "period-close"
                  and json.loads(x)["body"].get("period") == name), None)
    if close is None or len(close.get("windows") or ()) != 1:
        raise Missing(f"period {name}'s close in the ledger lists {len((close or {}).get('windows') or ())} windows, "
                      "and this piece is built for a period of one window (period-open --cap 1)")
    program = serve_dir / "h1/window-program.json.gz"
    if not program.is_file():
        raise Missing(f"no declared window Program (h1/window-program.json.gz) in {serve_dir}, the circuit the "
                      "period's open committed and credit reads the window's layout from")
    if statements is None:
        raise Missing(f"no statements for window {k}'s drawn units (--statements): the drawn calls are proved as "
                      "templated --registered statements, which the verifier of record doesn't accept yet (PRF-133 "
                      "part 3, PRF-274)")
    if live is None:
        raise Missing(f"no live evidence for window {k}'s rounds (--live-evidence: h1_prove/live.sh on the run's "
                      "--h1-prover dump)")
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    key = PHC.pinned_key(tree / "key.json", key_sha512)
    ev = PHC.window_evidence(rec, tree, {"root": rec["commitment"]["session"]["root"]},
                             json.loads(statements.read_text()), gzip.decompress(program.read_bytes()),
                             json.loads(live.read_text()))
    return {"key": key.to_json(), "main": lines(main), "name": name}, ledger, lines(pouw / "pins.jsonl"), [ev]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("protocol", choices=["v1", "period"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", help="the protocol's Lean executable (pouw-check, pouw-period); else the Python reference")
    ap.add_argument("--serve-run", type=Path)
    ap.add_argument("--window", type=int)
    for f in ("params", "registration", "answer", "key", "ledger", "pins", "main", "windows"):
        ap.add_argument(f"--{f}", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--pouw", type=Path, help="period: PoUW's served piece directory (pieces/pouw)")
    ap.add_argument("--key-sha512", help="period: the SHA-512 of the auditor's challenger key, the verifier's pin")
    ap.add_argument("--statements", type=Path, help="period: the drawn calls' statements, a JSON list")
    ap.add_argument("--live-evidence", type=Path, help="period: the window's live evidence (h1_prove/live.py's)")
    a = ap.parse_args(argv)
    protocol = V1 if a.protocol == "v1" else PERIOD
    given = (["params", "registration", "answer"] if protocol == V1
             else ["key", "ledger", "pins", "main", "name", "windows"])
    if a.serve_run is not None:
        if a.window is None:
            ap.error("--serve-run takes --window")
        if protocol == V1 or a.pouw is None:
            why = (v1_missing if protocol == V1 else period_missing)(a.serve_run, a.window)
        elif not (a.key_sha512 and a.main):
            ap.error("period --serve-run with --pouw takes --key-sha512 and --main (the verifier's own)")
        else:
            try:
                params, ledger, pins, windows = period_inputs(a.serve_run, a.window, a.pouw, a.key_sha512, a.main,
                                                              a.statements, a.live_evidence)
            except Missing as e:
                why = str(e)
            else:
                return period_piece(a.out, params, ledger, pins, windows, a.exe)
        missing(a.out, protocol, why)
        print(f"MISSING {protocol}: {why}", flush=True)
        return EXIT["undecided"]
    if not all(getattr(a, f) is not None for f in given):
        ap.error(f"{a.protocol} takes --serve-run and --window, or {', '.join('--' + f for f in given)}")
    if protocol == PERIOD:
        params = {"key": challenger_key(json.loads(a.key.read_text())), "main": lines(a.main), "name": a.name}
        return period_piece(a.out, params, lines(a.ledger), lines(a.pins), json.loads(a.windows.read_text()), a.exe)
    params = json.loads(a.params.read_text())
    commitment = canonical(json.loads(a.registration.read_text()))
    evidence = canonical(json.loads(a.answer.read_text()))
    return said(protocol, write(a.out, protocol, params, commitment, evidence, a.exe), a.exe)


def said(protocol: str, got: dict, exe: str | None) -> int:
    """The public ``PIECE`` line of the answer `got` (module docstring), and its exit."""
    print(line("PIECE", outcome(protocol, json.dumps(got))) + f" by={'lean' if exe else 'python'}", flush=True)
    return EXIT[got["verdict"]]


def period_piece(out: Path, params: dict, ledger: list[str], pins: list[str], windows: Any, exe: str | None) -> int:
    """The period's piece in `out`: `windows` as given (`credit.PROVED_KEYS`), or carrying their statements
    (`credit.PERIOD_KEYS`) and put in that form by `proved`, whose reference verdict ``REFERENCE`` prints."""
    if isinstance(windows, list) and windows and all(isinstance(w, dict) and "statements" in w for w in windows):
        windows, (_credit, why) = proved(params["key"], ledger, pins, params["main"], params["name"], windows)
        print(line("REFERENCE", outcome(PERIOD, json.dumps({"verdict": "reject" if why else "accept",
                                                            "reason": None if why is None else f"{why[0]}: {why[1]}"}))),
              flush=True)
    evidence = canonical({"pins": pins, "windows": windows})
    return said(PERIOD, write(out, PERIOD, params, canonical(ledger), evidence, exe), exe)


if __name__ == "__main__":
    sys.exit(main())
