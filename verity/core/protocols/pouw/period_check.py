"""A PoUW period as a core protocol, ``verity/pouw/period/v1``: the Python the Lean `PouwCheck.Period` (``pouw-period``,
PoUW's verifier of record for a period and onsite's work component) is held to by ``lean/check/tests/test_period_difftest.py``.

Its parameters are the verifier's own, ``{key, main, name}``: the challenger key it pins, its copy of main's
first-parent history and the period it credits; optionally ``capacity``, the period's monitored capacity in MACs.
It carries no policy parameter: the period surfaces what its draw established, and whoever reads it decides whether it
was idle. Its commitment is the auditor's ledger (the lines the challenger signed), its digest
SHA-512(`COMMITMENT_TAG` ‖ 0 ‖ its bytes). Its evidence is ``{pins, windows}``: the challenger host's pins log and each
window's evidence in the close's order (`credit.PROVED_KEYS`), or ``{fills, pins, windows}`` with the filler windows of
the period's fill period too (`fill`, `fill.FILL_KEYS`). The check first refuses, by name (``form``), every served window
in a form the period's proved guarantee (`GUARANTEE`) doesn't cover: a Pearl-C pass, a circuit that is not an H1 window, a
template over per-unit noise (``PouwH1Tile`` v1, v3 or v4), a J10-R unit over a tile other than ``PouwH1Tile_v5`` (v6),
or units at more than one K. Then it is `credit._period_outcomes`, all or nothing: it accepts with the period's credit
(``{windows, work, k, delta, eps}``) and the profile ``{period, windows}``, or rejects with the first refusal,
``code: why``. With filler windows it is `fill.period_fill` too, also all or nothing, and reported, never decided on:
the credit gains ``fill``, ``{work, k, delta, eps}`` with the profile ``fills``, or ``{refused: "code: why"}``, and the
verdict and the served credit are the same with or without them, since no guarantee covers the fill yet (the verifier
gains no form ahead of its proof); with a ``fill`` the credit carries ``gaps`` (`GAPS`), so no reader takes the fill as
proved. With a capacity C the credit gains ``utilization`` (``{capacity, certified, share}``): C, the certified work,
the served work alone, the work `GUARANTEE` covers, and its share of C. Nothing named certified holds the fill. Each
drawn unit's and each round's proof outcome is the verifier of record's verdict, which the evidence carries."""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F

ID = "verity/pouw/period/v1"
COMMITMENT_TAG = "verity/pouw/period/ledger/v1"
PARAMS_KEYS = ("key", "main", "name")
#: the parameters of a verifier that reads the period's utilization: its monitored capacity
CAPACITY_KEYS = ("capacity", "key", "main", "name")
EVIDENCE_KEYS = ("pins", "windows")
#: the evidence of a period that presents its filler windows too (`fill`)
FILL_EVIDENCE_KEYS = ("fills", "pins", "windows")
#: what a period with a ``fill`` doesn't establish
GAPS = ["fill: no guarantee covers the fill yet, so its work is reported and not certified"]


def params(j: Any) -> dict:
    if not isinstance(j, dict) or sorted(j) not in (list(PARAMS_KEYS), list(CAPACITY_KEYS)):
        raise CC.Refused(f"keys are not {', '.join(PARAMS_KEYS)}, with capacity optional")
    if not isinstance(j["name"], str):
        raise CC.Refused("name is not a string")
    if "capacity" in j and not (type(j["capacity"]) is int and j["capacity"] >= 1):
        raise CC.Refused("capacity is not a positive integer, the period's monitored capacity in MACs")
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
    if not isinstance(j, dict) or sorted(j) not in (list(EVIDENCE_KEYS), list(FILL_EVIDENCE_KEYS)):
        raise CC.Refused(f"keys are not {', '.join(EVIDENCE_KEYS)}")
    return j


#: the guarantee a credited period is proved under
GUARANTEE = "Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod"
TILE_ID = "PouwH1Tile_v"
#: the one tile the guarantee covers: a drawn per-call unit over it is proved as its 32 lanes and their combine
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
                f"many strata, and {GUARANTEE} draws from one tile stratum (TileDrawIsCoin): serve a window's linears "
                "at one K")
    return None


def form(nm: str, windows: Any) -> str | None:
    """`PouwCheck.Period.form`: the first window of `windows` in a form `GUARANTEE` doesn't cover, else None."""
    for i, w in enumerate(windows if type(windows) is list else []):
        if (why := _window_form(nm, i, w)) is not None:
            return why
    return None


def check(p: dict, c: dict, e: dict) -> tuple[str, str | None, tuple[Any, Any] | None]:
    """`PouwCheck.Period.check`: the first served window in a form the period's guarantee doesn't cover (``form``),
    else the period's credit and profile, or the first refusal. The fill is reported and read by no decision."""
    if (why := form(p["name"], e["windows"])) is not None:
        return "reject", f"form: {why}", None
    got, refused = Cr._period_outcomes(p["key"], c["ledger"], e["pins"], p["main"], p["name"], e["windows"],
                                       CC.lms_verify)
    if refused is not None:
        return "reject", f"{refused[0]}: {refused[1]}", None
    profile = {"period": p["name"], "windows": len(got["windows"])}
    if "fills" in e:
        fill, refused = F.period_fill(p["key"], c["ledger"], e["pins"], p["main"], p["name"], e["fills"],
                                      CC.lms_verify)
        if refused is not None:
            got = {**got, "fill": {"refused": f"{refused[0]}: {refused[1]}"}}
        else:
            got, profile = {**got, "fill": fill}, {**profile, "fills": len(fill["work"])}
        got["gaps"] = GAPS
    if "capacity" in p:
        cap, certified = p["capacity"], sum(got["work"])
        got = {**got, "utilization": {"capacity": cap, "certified": certified, "share": str(Fraction(certified, cap))}}
    return "accept", None, (got, profile)


def answer(params_b: bytes, commitment_b: bytes, evidence_b: bytes) -> dict:
    """`Core.Protocol.answer`: every input gets a verdict, a decoding failure being a rejection naming its input."""
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
    verdict, reason, acc = check(p, c, ev)
    return out(verdict, reason, d, *(acc or (None, None)))


def case(j: Any) -> dict:
    """`Core.Protocol.case` on one `serve` case, ``{"commitment": hex, "evidence": hex, "params": JSON}``: a malformed one
    is rejected with no commitment."""
    try:
        params_j = CC._get(j, "params", "the case")
        c = CC._unhex(CC._str_at(j, "commitment", "the case"))
        e = CC._unhex(CC._str_at(j, "evidence", "the case"))
        params_b = CC._canon(params_j).encode()
    except CC.Refused as err:
        return {"protocol": ID, "verdict": "reject", "reason": f"case: {err}", "commitment": None, "public": None,
                "profile": None}
    return answer(params_b, bytes.fromhex(c), bytes.fromhex(e))


__all__ = ["CAPACITY_KEYS", "COMMITMENT_TAG", "COVERED_TILE", "EVIDENCE_KEYS", "GAPS", "GUARANTEE", "ID",
           "PARAMS_KEYS", "answer", "case", "check", "commitment", "digest", "evidence", "form", "params"]
