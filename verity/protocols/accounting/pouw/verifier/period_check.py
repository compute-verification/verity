"""A PoUW period as a core protocol, ``verity/pouw/period/v1``: the Python the Lean `PouwCheck.Period` (``pouw-period``,
PoUW's verifier for a period and onsite's work component) is held to by ``lean/check/tests/test_period_difftest.py``.

Its parameters are the verifier's own, ``{key, main, name, sessions}``: the challenger key it pins, its copy of main's
first-parent history, the period it credits, and the ordered ids of the sessions the challenger's ledger opened for the
period's windows (each window's and its sample's); without ``sessions``, ``{key, main, name}`` (`ALONE_KEYS`), its
sessions aren't checked against a list. They name no device. Its commitment is the auditor's ledger (the lines the
challenger signed), its digest SHA-512(`COMMITMENT_TAG` ‖ 0 ‖ its bytes). Its evidence is ``{pins, windows}``: the challenger host's pins log and each window's evidence in the close's order
(`credit.PROVED_KEYS`). The check first refuses, by name (``form``), every window in a form the period's proved
guarantee (`GUARANTEE`) doesn't cover: a Pearl-C pass, a circuit that is not an H1 window, a template over per-unit
noise (``PouwH1Tile`` v1, v3 or v4), a J10-R unit over a tile other than ``PouwH1Tile_v5`` (v6), or units at more than
one K. Then it is `credit._period_outcomes`, all or nothing, and last a session of the period's close the parameters
don't list or that repeats one before it is refused (``sessions``). It accepts publishing the period's credit
(``{windows, work, k, delta, eps}``) with the parameters' challenger ``key``, the close's open ``roots`` and the
period's span in ledger entries (``start``, ``length``), naming no device (the accounting layer binds ``key`` to
one), with the profile ``{period, windows}``, or rejects with the first refusal, ``code: why``. Each sampled proof
block's and each round's proof outcome is the verifier's verdict, which the evidence carries."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from verity.protocols.accounting.pouw.verifier import core_check as CC
from verity.protocols.accounting.pouw.verifier import credit as Cr
from verity.protocols.accounting.pouw.verifier import salt as PS
from verity.protocols.accounting.pouw.verifier import verifier as V

ID = "verity/pouw/period/v1"
#: the Lean core protocol whose `serve` gives a credit period's verdict of record
EXE = "pouw-period"
COMMITMENT_TAG = "verity/pouw/period/ledger/v1"
PARAMS_KEYS = ("key", "main", "name", "sessions")
#: the keys of a period's parameters with no session list
ALONE_KEYS = ("key", "main", "name")
EVIDENCE_KEYS = ("pins", "windows")
KEYS_FORM = f"keys are not {', '.join(PARAMS_KEYS)}, or {', '.join(ALONE_KEYS)} alone"
CHALLENGE_PHASES_FORM = ("sessions is not a list of the ids of the sessions the challenger's ledger opened for the "
                         "period (1 to 64 of [A-Za-z0-9._:-] each, the first alphanumeric)")


def params(j: Any) -> dict:
    """`PouwCheck.Period.paramsOf`: the session list where the parameters carry one; without it the period's close is
    checked against no list."""
    if not isinstance(j, dict) or sorted(j) not in (list(PARAMS_KEYS), list(ALONE_KEYS)):
        raise CC.Refused(KEYS_FORM)
    if not isinstance(j["name"], str):
        raise CC.Refused("name is not a string")
    if sorted(j) == list(ALONE_KEYS):
        return j
    ss = j["sessions"]
    if not (type(ss) is list and all(type(s) is str and PS.SESSION.fullmatch(s) for s in ss)):
        raise CC.Refused(CHALLENGE_PHASES_FORM)
    return j


def commitment(b: bytes) -> dict:
    ledger = CC.canonical(b)
    if not isinstance(ledger, list):
        raise CC.Refused("the auditor's ledger is not a list of its lines")
    return {"ledger": ledger, "bytes": b}


def digest(c: dict) -> bytes:
    return hashlib.sha512(COMMITMENT_TAG.encode() + b"\0" + c["bytes"]).digest()


def evidence(b: bytes) -> dict:
    j = CC.canonical(b)
    if not isinstance(j, dict) or sorted(j) != list(EVIDENCE_KEYS):
        raise CC.Refused(f"keys are not {', '.join(EVIDENCE_KEYS)}")
    return j


#: the guarantee a credited period is proved under
GUARANTEE = "Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod"
TILE_ID = "PouwH1Tile_v"
#: the one tile the guarantee covers: a sampled per-call unit over it is proved as its 32 lanes and their combine
COVERED_TILE = "PouwH1Tile_v5{"


def _window_form(nm: str, i: int, w: Any) -> str | None:
    """`PouwCheck.Period.windowForm`: why window `i` of period `nm` is a form `GUARANTEE` doesn't cover, read from its
    evidence alone, or None for a covered window and for evidence too malformed to read (`credit.period_credit`
    refuses it)."""
    if isinstance(w, dict) and sorted(w) == list(Cr.PEARL_C_KEYS):
        return (f"window {i} of period {nm} is a Pearl-C pass ({{context, descriptor, manifest, session}}), which this "
                f"protocol never credits: a Pearl-C pass earns only in a Pearl-C period, and {GUARANTEE} covers H1 "
                "windows at J10-R alone")
    h = w.get("descriptor") if isinstance(w, dict) else None
    if not (type(h) is str and Cr._HEX.fullmatch(h)):
        return None
    raw = bytes.fromhex(h)
    text = raw.decode(errors="replace")
    if text.encode() != raw:  # not UTF-8 (Lean's `String.fromUTF8?` is none)
        return None
    ok, j = Cr._loads(text)
    fn = j.get("program_fn") if ok and isinstance(j, dict) else None
    if type(fn) is not str:
        return None
    defs = j.get("definitions")
    if not fn.startswith(Cr.WINDOW_FN + "{"):
        return (f"window {i} of period {nm} proves {fn.split('{')[0]}, not an H1 window ({Cr.WINDOW_FN}): a "
                "per-tile protocol's tile (PouwH1Tile_v1 at K = 64, 4 × 8, or a Pearl-C leaf) or any other circuit is "
                f"outside {GUARANTEE}, which covers H1 windows at J10-R alone")
    s = next((v for k, v in Cr._record(fn) if k == "TILES"), None)
    ok, ts = Cr._loads(s) if s is not None else (False, None)
    if not ok or type(ts) is not list:
        return None
    ks = []
    for t in ts:
        if not (type(t) is list and len(t) == 2):
            return None
        k, f = t
        tid = f.get("fn") if isinstance(f, dict) else None
        if not (Cr._nat(k) and type(tid) is str):
            return None
        dj = defs.get(tid) if isinstance(defs, dict) else None
        got = Cr._tile(dj, tid, k)
        if got is None:
            return None
        if not got[2]:
            return (f"window {i} of period {nm} is an H1 window whose template at K = {k} takes per-unit noise (its "
                    "parameters end in roots, not J10-R's call and blocks: PouwH1Tile_v1, v3 or v4), which "
                    f"{GUARANTEE} doesn't cover: serve it over J10-R's PouwH1Tile_v5")
        u = Cr._canonical(dj["bindings"]["UNIT"])
        if not (0 < u.count(TILE_ID) == u.count(COVERED_TILE)):
            return (f"window {i} of period {nm} is a J10-R window whose unit at K = {k} is not over PouwH1Tile_v5 "
                    f"alone (v6's one-stream z, or no named tile): {GUARANTEE} covers the per-call unit over "
                    "PouwH1Tile_v5, proved as its 32 lanes and their combine")
        ks.append(k)
    if len(ks) > 1:
        return (f"window {i} of period {nm} has units at {len(ks)} K's ({ks}), so its draw is over that "
                f"many strata, and {GUARANTEE} draws from one tile stratum (TileSampleIsCoin): serve a window's linears "
                "at one K")
    return None


def form(nm: str, windows: Any) -> str | None:
    """`PouwCheck.Period.form`: the first window of `windows` in a form `GUARANTEE` doesn't cover, else None."""
    for i, w in enumerate(windows if type(windows) is list else []):
        if (why := _window_form(nm, i, w)) is not None:
            return why
    return None


def _period_entry(ledger: list, kind: str, name: str) -> dict | None:
    return next((e for e in map(json.loads, ledger) if e["kind"] == kind and e["body"].get("period") == name), None)


def ledger_challenge_phases(ledger: list, name: str) -> list[str]:
    """`PouwCheck.Period.ledgerChallengePhases` on a ledger the challenger's checks have trusted: the sessions period
    `name`'s close lists, in its order (each window's session, then its sample's), the empty list when it closes no
    period `name`."""
    close = _period_entry(ledger, "period-close", name)
    return [] if close is None else [w["session"] for w in close["body"]["windows"]]


def challenge_phases_refusal(name: str, listed: list[str], got: list[str]) -> str | None:
    """`PouwCheck.Period.challengePhasesRefusal`: the first session of `got` the parameters don't list (`listed`) or
    that repeats one before it, else None."""
    seen: set[str] = set()
    for i, s in enumerate(got):
        if s not in listed:
            return (f"session {i} of period {name}'s close ({s}) is not one the verifier's parameters list for the "
                    "period: only a session the challenger's ledger opened for one of its windows counts")
        if s in seen:
            return f"session {i} of period {name}'s close ({s}) repeats a session before it: each session counts once"
        seen.add(s)
    return None


def published(credit: dict, p: dict, ledger: list) -> dict:
    """`PouwCheck.Period.Published.toJson`: the period's credit, its parameters' challenger key, and from the trusted
    ledger its close's open roots and its span (the period-open's seq and the entries from it to its close, inclusive).
    It names no device: the accounting layer binds the challenger key to a registered device."""
    o, c = (_period_entry(ledger, k, p["name"]) for k in ("period-open", "period-close"))
    return {**credit, "key": p["key"], "roots": [w["root"] for w in c["body"]["windows"]], "start": o["seq"],
            "length": c["seq"] + 1 - o["seq"]}


def check_twin(p: dict, c: dict, e: dict) -> tuple[str, str | None, tuple[Any, Any] | None]:
    """The Python twin of `PouwCheck.Period.check`, for the difftest: the first window in a form the period's guarantee doesn't cover (``form``), else the
    period's credit or the first refusal, then a session of its close the parameters don't list or that repeats; on
    acceptance what it publishes (`published`) and its profile."""
    if (why := form(p["name"], e["windows"])) is not None:
        return "reject", f"form: {why}", None
    got, refused = Cr._period_outcomes(p["key"], c["ledger"], e["pins"], p["main"], p["name"], e["windows"],
                                       CC.lms_verify)
    if refused is not None:
        return "reject", f"{refused[0]}: {refused[1]}", None
    if "sessions" in p and (why := challenge_phases_refusal(p["name"], p["sessions"],
                                                     ledger_challenge_phases(c["ledger"], p["name"]))) is not None:
        return "reject", f"sessions: {why}", None
    return "accept", None, (published(got, p, c["ledger"]), {"period": p["name"], "windows": len(got["windows"])})


def answer_twin(params_b: bytes, commitment_b: bytes, evidence_b: bytes) -> dict:
    """The Python twin of `Core.Protocol.answer`, for the difftest: every input gets a verdict, a decoding failure being a rejection naming its input."""
    def out(verdict: str, reason: str | None, d: bytes | None, pub: Any = None, prof: Any = None) -> dict:
        return {"protocol": ID, "verdict": verdict, "reason": reason, "commitment": None if d is None else d.hex(),
                "public": pub, "profile": prof}
    try:
        p = params(CC.canonical(params_b))
    except CC.Refused as e:
        return out("reject", f"params: {e}", None)
    try:
        c = commitment(commitment_b)
    except CC.Refused as e:
        return out("reject", f"commitment: {e}", None)
    d = digest(c)
    try:
        ev = evidence(evidence_b)
    except CC.Refused as e:
        return out("reject", f"evidence: {e}", d)
    verdict, reason, acc = check_twin(p, c, ev)
    return out(verdict, reason, d, *(acc or (None, None)))


def case_twin(j: Any) -> dict:
    """The Python twin of `Core.Protocol.case`, for the difftest, on one `serve` case, ``{"commitment": hex, "evidence": hex, "params": JSON}``: a malformed one
    is rejected with no commitment."""
    try:
        params_j = CC._get(j, "params", "the case")
        c = CC._unhex(CC._str_at(j, "commitment", "the case"))
        e = CC._unhex(CC._str_at(j, "evidence", "the case"))
        params_b = CC._canon(params_j).encode()
    except CC.Refused as err:
        return {"protocol": ID, "verdict": "reject", "reason": f"case: {err}", "commitment": None, "public": None,
                "profile": None}
    return answer_twin(params_b, bytes.fromhex(c), bytes.fromhex(e))


def answer(params_b: bytes, commitment_b: bytes, evidence_b: bytes) -> dict:
    """`Core.Protocol.answer`, the Lean's (`pouw-period serve`, through `verifier.serve_of_record`) on the case of these
    bytes, with `answer_twin` its twin: a disagreement is a rejection."""
    return V.serve_of_record(EXE, CC.serve_case(params_b, commitment_b, evidence_b),
                             answer_twin(params_b, commitment_b, evidence_b))


def case(j: Any) -> dict:
    """`Core.Protocol.case` on one `serve` case, the Lean's (`pouw-period serve`, through `verifier.serve_of_record`),
    with `case_twin` its twin: a disagreement is a rejection."""
    return V.serve_of_record(EXE, j, case_twin(j))


__all__ = ["ALONE_KEYS", "CHALLENGE_PHASES_FORM", "COMMITMENT_TAG", "COVERED_TILE", "EVIDENCE_KEYS", "GUARANTEE",
           "EXE", "ID", "KEYS_FORM", "PARAMS_KEYS", "answer", "answer_twin", "case", "case_twin",
           "challenge_phases_refusal", "check_twin", "commitment", "digest", "evidence", "form", "ledger_challenge_phases",
           "params", "published"]
