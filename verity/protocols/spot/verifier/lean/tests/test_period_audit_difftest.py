"""Sampled proofs' period, the audits' half (`Spot.PeriodAudit`), held to the references below through three pieces of
the difftest. The references live here alone: there is no Python verifier of the period.

- `period_asked`: what one-stage v2 is asked at a case (`paramsAt`, `evidenceAt`), against `asked`.
- `period_check_stub`: the period's check with a stand-in for one-stage v2's audit (`stubInner`), against `check`: the
  ledger's half through `test_period_difftest.reference`, each window's session through the challenger's mirror
  (`challenger._audited`, `pins_reason`, `first_parent_reason`, `session_named`, `withheld_on`), and each case answered
  as the stand-in answers it. No proof here is one that one-stage v2 accepts, so an accepted period, an undecided one
  and the records each audit is handed are run through the stand-in alone.
- `period_check`: the period's check on one-stage v2 itself, against one-stage v2's own answer
  (`sampled-proofs-one-stage-zk`) on what `asked` composes.

The cases are a week of one lane under the deployed key, written as the auditor's gate writes it (`Week`): each window
reserved, opened on a commit the host's pins log records and main's first-parent line holds, its challenger's session
signed, logged and closed between its open and its end, which seals it; in its log sometimes one coin after the
window's own root, then the sample's coins after its registration's digest; its case the registration's record, its
public file's digest and one-stage v2's evidence. Then one change each (`NAMES`): a case the stand-in leaves undecided
or rejects, a session that sampled no registration or two, no case or two for the one registration, a case of another
registration, a public file that shares rows or is in no blob, evidence that names its own ledger or verdicts, an open
on a commit the pins log does not record or main's first-parent line does not hold, a session not withheld, a release
log other than the sealed one, two windows' sessions swapped, an end that seals nothing, and the ledger's half's own
refusals; then parameters and evidence of another form, which are refused before the check.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import random
import re
import shutil
from typing import Any

import pytest
from hypothesis import strategies as st

from test_challenger_difftest import DEPLOYED, LEAVES as SIGNABLE, PACKAGE, PIECES, _key, _pairs, lms_verify
from test_difftest import (LEAF, ONE_STAGE_V2, RECEIVED, ROOTS, SCHEME, SETUPS, STAND_IN_CIRCUIT, _hex, _json,
                           _public_of, _sha)
from test_period_difftest import DAY, DRAW, LABEL, LANE, OTHER, SALT, START, STOP, Book, at, reference as ledger_half

import verity_difftest as D
from verity.commitments.crypto import lms
from verity.commitments.identity import canonical_json_text
from verity.protocols.spot.device import challenger as Ch, coins as T
from verity.protocols.spot.verifier.one_stage import draw as DR, partition as P, registration as R

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

KEY = _key(*DEPLOYED, 0)
PUBLIC = lms.public_key(KEY)
KEY_JSON = Ch.ChallengerKey(*DEPLOYED, PUBLIC).to_json()

S = SETUPS[3]
LAW = {"k": 2, "law": "subset"}
DELTA = "1/20"
#: one-stage v2's parameters less what a window sets (`Spot.Period.AUDIT_FORM`)
AUDIT = _json({"anchors": [], "delta": DELTA, "law": LAW, "leaf_layer": LEAF, "log": [],
               "partition": P.partition(S.program, S.query), "ports": S.ports,
               "proofs": {"circuit": _sha(STAND_IN_CIRCUIT), "registered": None}, "program": S.desc, "query": S.query,
               "registered": None, "scheme": SCHEME, "work_table": None})

PARAM_KEYS = ["audit", "key", "label", "lane", "main", "start", "stop"]
#: the keys of one-stage v2's parameters that are a window's (`WINDOW_KEYS`, `UNDERIVED_KEYS`)
AT_A_WINDOW = ["challenger", "window_root", "day", "declarations", "seed_salt"]
AUDIT_FORM = ("the audit's parameters are one-stage v2's less what a window sets: an object with a log, proofs of exactly "
              "circuit and registered, and no challenger, window_root, day, declarations or seed_salt")
EVIDENCE_FORM = ("a period's evidence is {pins, windows}: the host's pins log and, for each window in the ledger's order, "
                 "{audits, session}: its challenger session's files and, for each registration the session sampled, "
                 "{evidence, public, registration}")
CASE_COINS = ("a case's evidence names its coins as {segment} alone: the ledger, main's history, the pins log and the "
              "session's files are the period's own")
ONE_MEMBER = ("only a window of one registration is covered, and a session that sampled several (several ranks', or one "
              "registered again) is refused")
SHARED_ROWS = "the public file's header names shared_rows: a registration whose members share row tables is not covered"
VERDICTS = "the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs"
NO_SESSIONS = "proofs: the evidence names 0 sessions: the drawn units are proved in one session"
HEX = re.compile(r"(?:[0-9a-f]{2})*")
DIGEST = re.compile(r"[0-9a-f]{128}")


# ---- the references -------------------------------------------------------------------------------------------------

def params_refusal(p: Any) -> str | None:
    """Why `AuditParams.of` refuses `p`, of the refusals that are the period's own. The ledger's half's parameters and
    one-stage v2's reading of `audit` at a window are not mirrored: every case here passes both."""
    if not isinstance(p, dict) or sorted(p) != PARAM_KEYS:
        return f"a period's parameters' keys are {', '.join(PARAM_KEYS)}"
    audit = p["audit"]
    if not (isinstance(audit, dict) and isinstance(audit.get("log"), list) and isinstance(audit.get("proofs"), dict)
            and sorted(audit["proofs"]) == ["circuit", "registered"] and not any(k in audit for k in AT_A_WINDOW)):
        return AUDIT_FORM
    return None


def evidence_refusal(e: Any) -> str | None:
    """Why `Evidence.of` refuses `e`."""
    if not (isinstance(e, dict) and sorted(e) == ["pins", "windows"] and isinstance(e["windows"], list)):
        return EVIDENCE_FORM
    for w in e["windows"]:
        if not (isinstance(w, dict) and sorted(w) == ["audits", "session"] and isinstance(w["audits"], list)):
            return EVIDENCE_FORM
        for a in w["audits"]:
            if not (isinstance(a, dict) and sorted(a) == ["evidence", "public", "registration"]
                    and type(a["public"]) is str):
                return EVIDENCE_FORM
    return None


def params_at(p: dict, root: str, pub: str, before: list) -> dict:
    """`paramsAt`: one-stage v2's parameters at the window `root`, for the registration whose public file is `pub`."""
    audit = p["audit"]
    return {**{k: v for k, v in audit.items() if k not in ("log", "proofs")},
            "challenger": {"key": p["key"], "label": p["label"]}, "log": audit["log"] + before,
            "proofs": {"circuit": audit["proofs"]["circuit"], "public": pub, "registered": audit["proofs"]["registered"]},
            "window_root": root}


def evidence_at(p: dict, ledger: Any, pins: Any, session: Any, e: Any) -> dict | None:
    """`evidenceAt`: one-stage v2's evidence for a case, its coins' ledger, main's history, pins log and session files
    the period's; None where the case names more of its coins than their segment (`CASE_COINS`)."""
    if not (isinstance(e, dict) and isinstance(e.get("coins"), dict) and sorted(e["coins"]) == ["segment"]):
        return None
    return {**{k: v for k, v in e.items() if k != "coins"},
            "coins": {"ledger": ledger, "main": p["main"], "pins": pins, "segment": e["coins"]["segment"],
                      "session": session}}


ASKED_KEYS = ["before", "case", "ledger", "params", "pins", "root", "session"]


def asked(j: Any) -> Any:
    """`Spot.Period.askedPiece`."""
    if not isinstance(j, dict) or sorted(j) != ASKED_KEYS:
        raise D.Reject(f"a request's keys are {', '.join(ASKED_KEYS)}")
    k = j["case"]
    if not (isinstance(j["before"], list) and type(j["root"]) is str and isinstance(k, dict)
            and type(k.get("public")) is str):
        raise D.Reject("before is a list, root a string and the case's public a string")
    if (why := params_refusal(j["params"])) is not None:
        return {"refused": f"params: {why}"}
    e = evidence_at(j["params"], j["ledger"], j["pins"], j["session"], k.get("evidence"))
    if e is None:
        return {"refused": f"evidence: {CASE_COINS}"}
    return {"evidence": e, "params": params_at(j["params"], j["root"], k["public"], j["before"])}


def stand_in(params: dict, record: dict, evidence: dict, blobs: list[str]) -> tuple[str, Any, Any]:
    """`stubInner`: the audit as the case's statement tells it, an accept publishing how many records the parameters'
    log holds."""
    match evidence["statement"].get("stub"):
        case "accept":
            return ("accept", {"bound": len(params["log"]), "drawn": [], "per_template": None, "proved": []},
                    {"covered": [], "delta": params["delta"], "drawn": {}, "levels": [], "linked": False,
                     "prescribed": [], "programs": []})
        case "undecided":
            return "undecided", "the stand-in leaves it undecided", None
        case _:
            return "reject", "the stand-in rejects", None


def _reject(code: str, why: str) -> dict:
    return {"reason": f"{code}: {why}", "verdict": "reject"}


def check(j: Any, inner=stand_in) -> Any:
    """`Spot.Period.checkPieceWith` at the audit `inner` (`inner(params, record, evidence, blobs)`: its verdict, then
    what it published and its profile, or its reason)."""
    if not isinstance(j, dict) or sorted(j) != ["blobs", "evidence", "ledger", "params"]:
        raise D.Reject("a request's keys are blobs, evidence, ledger, params")
    if not (isinstance(j["blobs"], list) and isinstance(j["ledger"], list)
            and all(type(h) is str and HEX.fullmatch(h) for h in j["blobs"])):
        raise D.Reject("blobs and ledger are lists, a blob a hex string")
    blobs: dict[str, bytes] = {}
    for h in j["blobs"]:
        blobs.setdefault(hashlib.sha512(bytes.fromhex(h)).hexdigest(), bytes.fromhex(h))
    if (why := params_refusal(j["params"])) is not None:
        return {"refused": f"params: {why}"}
    if (why := evidence_refusal(j["evidence"])) is not None:
        return {"refused": f"evidence: {why}"}
    p, e, ledger = j["params"], j["evidence"], j["ledger"]
    label = p["label"]
    half = ledger_half({"key": p["key"], "label": label, "lane": p["lane"], "ledger": ledger, "start": p["start"],
                        "stop": p["stop"]})
    if "code" in half:
        return _reject(half["code"], half["refused"])
    if len(e["windows"]) != len(half["windows"]):
        return _reject("evidence", f"the evidence holds {len(e['windows'])} windows' sessions, and the ledger lists "
                                   f"{len(half['windows'])} windows in the monitoring period")
    key = Ch.ChallengerKey.from_json(p["key"])
    before: list[dict] = []
    passed: list[dict] = []
    profiles: list[dict] = []
    undecided: str | None = None
    for w, ev in zip(half["windows"], e["windows"]):
        sid, root = w["session"], w["root"]
        try:
            a, opened = Ch._audited(ledger, {"root": root}, key, lms_verify)
            why = (Ch.pins_reason(a.session, opened, e["pins"])
                   or Ch.first_parent_reason(a.session, opened.get("pinned"), p["main"]))
            if why is not None:
                raise Ch.CoinError(why)
            _, roots = Ch.session_named(a, key, ev["session"], lms_verify)
        except Ch.CoinError as x:
            return _reject("coin", str(x))
        if not Ch.withheld_on(ledger, root):
            return _reject("coin", Ch.WITHHELD)
        sampled: list[str] = []
        for r in roots.get(label, ()):
            if (h := r.hex()) != root and h not in sampled:
                sampled.append(h)
        if not sampled:
            return _reject("missing", f"session {sid} released no {label} coin after a registration: its root was "
                                      "committed and never sampled")
        if len(sampled) > 1:
            return _reject("members", f"session {sid} sampled {len(sampled)} registrations: {ONE_MEMBER}")
        want = sampled[0]
        if len(ev["audits"]) != 1:
            return _reject("evidence", f"session {sid}: {len(ev['audits'])} cases for the one registration its log "
                                       "shows sampled")
        k = ev["audits"][0]
        record = k["registration"]
        if not isinstance(record, dict):
            return _reject("evidence", f"session {sid}: registration: the registration record is not an object")
        got = R.digest(record)
        if got != want:
            return _reject("evidence", f"session {sid}: the case is of registration {got[:16]}, and the session's log "
                                       f"shows its {label} coins released after {want[:16]}")
        assert DIGEST.fullmatch(k["public"]), "the reference reads only a case whose public file's digest is one"
        if k["public"] not in blobs:
            return _reject("names", f"session {sid}: public file: no blob has the SHA-512 {k['public']}")
        if "shared_rows" in json.loads(blobs[k["public"]].split(b"\n", 1)[0]):
            return _reject("names", f"session {sid}: {SHARED_ROWS}")
        composed = evidence_at(p, ledger, e["pins"], ev["session"], k["evidence"])
        if composed is None:
            return _reject("evidence", f"session {sid}: {CASE_COINS}")
        if "verdicts" in composed:
            return _reject("evidence", f"session {sid}: {VERDICTS}")
        assert T.coins_within(composed["coins"]), "the reference reads only a case whose coins are within the limits"
        verdict, out, profile = inner(params_at(p, root, k["public"], before), record, composed, j["blobs"])
        here = f"session {sid}, registration {want[:16]}"
        if verdict == "reject":
            return _reject("audit", f"{here}: {out}")
        before.append(record)
        if verdict == "undecided":
            undecided = undecided or f"{here}: {out}"
        else:
            passed.append({**w, "audits": [{"published": out, "registration": want}]})
            profiles.append(profile)
    if undecided is not None:
        return {"reason": f"audit: {undecided}", "verdict": "undecided"}
    return {"profile": profiles, "verdict": "accept",
            "published": {"lane": p["lane"], "others": half["others"], "start": p["start"], "stop": p["stop"],
                          "windows": passed}}


# ---- the cases ------------------------------------------------------------------------------------------------------

@functools.cache
def _record(row: int) -> dict:
    """A registration's record of the test's program at window row `row`: one digest a row."""
    return _json(R.record(S.program, S.query, LAW, SCHEME, [{"port": p, "root": ROOTS[p], **S.ports[p]} for p in S.ports],
                          {"kind": "served", "row": row, "run": "r-test"}, leaf_layer=LEAF))


def _shared(public: bytes) -> bytes:
    """`public` with its header naming `shared_rows`."""
    header, rest = public.split(b"\n", 1)
    return json.dumps({**json.loads(header), "shared_rows": {"x": 1}}, sort_keys=True).encode() + b"\n" + rest


class Week:
    """A week's ledger of lane `LANE` under the deployed key (`b`), with the host's pins log, main's first-parent line,
    each window's evidence in the ledger's order and the blobs the cases name."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.b = Book(rng, KEY, SIGNABLE)
        self.pins: list[str] = []
        self.main: list[str] = []
        self.windows: list[dict] = []
        self.blobs: list[bytes] = [STAND_IN_CIRCUIT]
        self.rows = itertools.count(101)

    def window(self, t: int, name: str = "honest", stub: str = "accept", real: bool = False) -> str:
        """A window of the lane reserved at `t`, with its session and its case, changed as `name` says; its session's
        id. The case's evidence tells the stand-in `stub`, or with `real` is a sample one-stage v2 reads from the
        session's coins, proved in no session."""
        b, rng = self.b, self.rng
        sid = b.reserve(LANE, t, DRAW, withheld=name != "not-withheld")
        root = b.open(sid, t + 60)
        opened = b.events[b.sessions[sid]["open"]][2]
        self.pins.append(Ch.pin_line(opened["pinned"], rng.randbytes(20).hex(), at(t)))
        self.main.insert(rng.randint(0, len(self.main)), opened["pinned"])
        if name != "no-seal":
            opened["pins"] = Ch.pins_seal(self.pins)
        if name == "unpinned":
            opened["pinned"] = rng.randbytes(20).hex()
        elif name == "off-main":
            self.main.remove(opened["pinned"])

        request = b.sessions[sid]["request"]
        ch = Ch.LiveChallenger(request["scheme"], request["epoch"], [tuple(f) for f in request["families"]],
                               rng.randbytes)
        q = b.skip(2)
        ch.sign_record(opened["host"], PUBLIC, lambda m: lms.sign_at(KEY, q, m).encode())
        lines = [canonical_json_text({"event": "commitment", "root": ch.commitment.root.hex()})]
        record = Ch.LiveRecord(ch, LABEL, itertools.count(rng.choice([0, 10 ** 12])).__next__)

        def logged(f):
            n = len(record.log)
            out = f()
            lines.extend(_pairs(record.log[n:]))
            return out

        if real or rng.random() < 0.7:
            logged(lambda: record.record(bytes.fromhex(root)))
        recs = [_record(next(self.rows)) for _ in range({"unsampled": 0, "two-members": 2}.get(name, 1))]
        cases = []
        for rec in recs:
            coins = T.ChallengerCoins(record)
            segment = logged(lambda: coins.commit(bytes.fromhex(R.digest(rec))))
            if real:
                d = logged(lambda: T.draw_first(T.OwnRandomness(coins), DR.derive(S.program, S.query, LAW, None),
                                                S.population))
                proved = DR.proved(d)
                e = {"draw": d, "instances": len(proved), "proved_units": proved,
                     "receipt": R.receipt(rec, "v", received=RECEIVED), "statement": {}}
            else:
                logged(lambda: coins(rng.randint(1, 64)))
                e = {"draw": {}, "instances": 0, "proved_units": [], "receipt": {}, "statement": {"stub": stub}}
            e.update(coins={"segment": segment.to_json()}, proofs={"sessions": []}, served=None, session_roots=None,
                     sessions=[])
            public = _shared(_public_of(rec)) if name == "shared-rows" else _public_of(rec)
            if name == "no-public":
                public += b", in no blob"
            else:
                self.blobs.append(public)
            cases.append({"evidence": e, "public": _sha(public), "registration": rec})
        if name == "twice":
            logged(lambda: record.record(bytes.fromhex(R.digest(recs[0]))))

        hellos = ["x"] * rng.choice([0, 0, 1])
        chain = Ch.log_chain(lines)
        m = Ch.close_message(PUBLIC[:16], opened["host"], ch.commitment, sid, len(lines), chain, None)
        close = Ch.ChallengerClose(sid, len(lines), chain, None, lms.sign_at(KEY, q + 1, m).encode()).to_json()
        b.end(sid, t + 120, complete=True, commitment=ch.commitment.root.hex(), record_leaf=q,
              refused_hellos=len(hellos),
              files={Ch.LOG_FILE: Ch.log_digest(lines), Ch.HELLOS_FILE: Ch.log_digest(hellos),
                     Ch.RECORD_FILE: rng.randbytes(64).hex(), Ch.CLOSE_FILE: rng.randbytes(64).hex()})
        session = {"close": close, "hellos": hellos, "log": lines[:-2] if name == "log-edited" else lines,
                   "record": ch.record.to_json()}

        audits = cases[:1]
        match name:
            case "no-case":
                audits = []
            case "two-cases":
                audits = audits * 2
            case "other-registration":
                audits[0]["registration"] = _record(next(self.rows))
            case "registration-form":
                audits[0]["registration"] = rng.choice(["x", 5, None, []])
            case "case-coins":
                e = audits[0]["evidence"]
                e["coins"] = rng.choice([{**e["coins"], "ledger": []}, {}, None])
            case "verdicts":
                audits[0]["evidence"]["verdicts"] = {}
        self.windows.append({"audits": audits, "session": session})
        return sid

    def params(self) -> dict:
        return {"audit": AUDIT, "key": KEY_JSON, "label": LABEL, "lane": LANE, "main": list(self.main), "start": START,
                "stop": STOP}

    def request(self) -> dict:
        return {"blobs": [x.hex() for x in self.blobs], "evidence": {"pins": list(self.pins), "windows": self.windows},
                "ledger": self.b.lines(), "params": self.params()}


#: each named case's answer: the verdict (or `refused`, before the check) and the code its reason begins with
NAMES = {"honest": ("accept", None), "twice": ("accept", None), "undecided": ("undecided", "audit"),
         "rejected": ("reject", "audit"), "undecided-rejected": ("reject", "audit"), "unsampled": ("reject", "missing"),
         "two-members": ("reject", "members"), "no-case": ("reject", "evidence"), "two-cases": ("reject", "evidence"),
         "other-registration": ("reject", "evidence"), "registration-form": ("reject", "evidence"),
         "case-coins": ("reject", "evidence"), "verdicts": ("reject", "evidence"), "fewer-windows": ("reject", "evidence"),
         "shared-rows": ("reject", "names"), "no-public": ("reject", "names"), "unpinned": ("reject", "coin"),
         "off-main": ("reject", "coin"), "no-seal": ("reject", "coin"), "not-withheld": ("reject", "coin"),
         "log-edited": ("reject", "coin"), "swapped": ("reject", "coin"), "unsealed": ("reject", "coin"),
         "early": ("reject", "early"), "outside": ("reject", "outside"), "open-period": ("reject", "period"),
         "params-keys": ("refused", "params"), "audit-form": ("refused", "params"),
         "evidence-form": ("refused", "evidence")}
#: the cases of two windows or more
SEVERAL = ["undecided-rejected", "swapped"]
REQUESTS = ["request"]


def case(rng: random.Random, name: str) -> Any:
    """The request of case `name` (module docstring): a week of lane `LANE` in one period, changed at one window as
    `name` says. Its generator is seeded from `rng`, so Hypothesis records one draw a case and not each sampled byte."""
    rng = random.Random(rng.getrandbits(64))
    w = Week(rng)
    b = w.b
    n = rng.randint(2, 3) if name in SEVERAL else rng.randint(1, 3)
    pick = 0 if name in SEVERAL else rng.randrange(n)
    if rng.random() < 0.3:
        b.period_open("before", LANE, -2 * DAY)
        b.window(LANE, -2 * DAY + 600)
        b.period_close("before", -2 * DAY + 1200)
    t = 600
    b.period_open("p-0", LANE, t)
    for i in range(n):
        t += 3600
        if rng.random() < 0.3:
            b.window(OTHER, t - 1800, rng.choice([DRAW, SALT]))
        if name == "unsealed" and i == pick:
            b.window(LANE, t)
            w.windows.append({"audits": [], "session": None})
            continue
        stub = "accept"
        if i == pick and name in ("undecided", "undecided-rejected"):
            stub = "undecided"
        elif name == "rejected" and i == pick or name == "undecided-rejected" and i == n - 1:
            stub = "reject"
        w.window(t, name if i == pick else "honest", stub)
    t += 3600
    if name != "open-period":
        b.period_close("p-0", t)
    if name == "outside" or name != "open-period" and rng.random() < 0.4:
        b.window(LANE, t + 3600, DRAW if name == "outside" else rng.choice([SALT, [], None]))
    b.reserve(OTHER, 7 * DAY + rng.randint(0, 3600), SALT)
    j = w.request()
    windows = j["evidence"]["windows"]
    match name:
        case "early":
            j["ledger"] = j["ledger"][:-1]
        case "swapped":
            windows[0], windows[1] = windows[1], windows[0]
        case "fewer-windows":
            windows.pop()
        case "params-keys":
            p = j["params"]
            j["params"] = rng.choice([{k: v for k, v in p.items() if k != "main"}, {**p, "window_root": "00" * 64}, None])
        case "audit-form":
            audit = j["params"]["audit"]
            proofs = audit["proofs"]
            j["params"]["audit"] = rng.choice(
                [{**audit, k: v} for k, v in (("window_root", "00" * 64), ("challenger", None), ("day", 3),
                                              ("declarations", []), ("seed_salt", "5a" * 64), ("log", None),
                                              ("proofs", {**proofs, "public": "00" * 64}),
                                              ("proofs", {"circuit": proofs["circuit"]}), ("proofs", None))]
                + [{k: v for k, v in audit.items() if k != "log"}, None, []])
        case "evidence-form":
            e = j["evidence"]
            first = windows[0]
            j["evidence"] = rng.choice(
                [{"windows": windows}, {**e, "x": 1}, {**e, "windows": None}, None,
                 {**e, "windows": [{**first, "x": 1}] + windows[1:]},
                 {**e, "windows": [{"audits": None, "session": first["session"]}] + windows[1:]},
                 {**e, "windows": [{"audits": [{"evidence": {}, "public": None, "registration": {}}],
                                    "session": first["session"]}] + windows[1:]},
                 {**e, "windows": [{"audits": [{"evidence": {}, "registration": {}}], "session": first["session"]}]
                                  + windows[1:]}])
        case "request":
            j = rng.choice([{k: v for k, v in j.items() if k != "blobs"}, {**j, "x": 1}, None, [], {**j, "blobs": None},
                            {**j, "blobs": ["0g"]}, {**j, "blobs": [5]}, {**j, "ledger": None}])
    return j


INPUTS = st.builds(case, st.randoms(use_true_random=False),
                   st.sampled_from(["honest"] * 3 + list(NAMES) + REQUESTS * 6))
CHECK = D.Port(PIECES, "period_check_stub", check, INPUTS, package=PACKAGE, of_record="lean")


def asked_case(rng: random.Random, name: str) -> Any:
    """A request of `period_asked`: a case of a week's window with the records before it, or changed as `name` says."""
    rng = random.Random(rng.getrandbits(64))
    w = Week(rng)
    w.b.period_open("p-0", LANE, 600)
    w.window(4200, name if name in ("case-coins", "verdicts") else "honest")
    ev = w.windows[0]
    j = {"before": [_record(next(w.rows)) for _ in range(rng.randint(0, 2))], "case": ev["audits"][0],
         "ledger": rng.choice([["a line"], [], None]), "params": w.params(), "pins": list(w.pins),
         "root": rng.randbytes(64).hex(), "session": ev["session"]}
    if rng.random() < 0.5:
        j["params"]["audit"] = {**AUDIT, "log": [_record(next(w.rows))], "held": None}
    match name:
        case "params-keys":
            j["params"].pop("main")
        case "audit-form":
            j["params"]["audit"] = rng.choice([{**AUDIT, "window_root": "00" * 64}, None,
                                               {**AUDIT, "proofs": {**AUDIT["proofs"], "public": "00" * 64}}])
        case "request":
            j = rng.choice([{k: v for k, v in j.items() if k != "pins"}, {**j, "x": 1}, None, {**j, "before": None},
                            {**j, "root": 5}, {**j, "case": {**j["case"], "public": None}}])
    return j


ASKED = D.Port(PIECES, "period_asked", asked,
               st.builds(asked_case, st.randoms(use_true_random=False),
                         st.sampled_from(["honest"] * 4 + ["case-coins", "verdicts", "params-keys", "audit-form"]
                                         + REQUESTS * 2)),
               package=PACKAGE, of_record="lean")

pieces = D.server_fixture(CHECK)
one_stage_v2 = D.server_fixture(ONE_STAGE_V2)


def _answer(pieces, port: D.Port, j: Any) -> dict:
    kind, text = D.replay(port, j, pieces)
    assert kind == "output", (kind, text)
    return json.loads(text)


# ---- the tests ------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_what_one_stage_is_asked_is_what_the_reference_composes(pieces):
    """Lean and the reference agree on what one-stage v2 is asked at a case: the audit's parameters with the window's
    root, the challenger's key and family, the public file's digest and the log with the records before; the case's
    evidence with the period's ledger, main's history, pins log and session files as its coins'."""
    stats = D.run(ASKED, examples=40, server=pieces)
    assert stats.outputs > 20 and stats.rejects > 0, stats


@pytest.mark.slow
def test_the_check_agrees_and_each_case_is_answered_as_named(pieces):
    """Lean and the reference agree on sampled cases and on each named case over two seeds, and each named case is
    answered with its verdict under its own code."""
    stats = D.run(CHECK, examples=30, server=pieces)
    assert stats.outputs > 20 and stats.rejects > 0, stats
    for name, (verdict, code) in NAMES.items():
        for seed in range(2):
            got = _answer(pieces, CHECK, case(random.Random(f"{name}/{seed}"), name))
            if verdict == "refused":
                assert got["refused"].startswith(f"{code}: "), (name, seed, got)
            else:
                assert got["verdict"] == verdict, (name, seed, got)
                assert code is None or got["reason"].startswith(f"{code}: "), (name, seed, got)


def test_three_windows_pass_and_each_audit_is_handed_the_records_before_it(pieces):
    """A week of three windows is accepted with the stand-in: each window is published with its one registration, the
    audit of each handed the log with the records of the registrations audited before it (the stand-in publishes the
    log's size as its bound), and the lane's session of another layout is published among the others."""
    w = Week(random.Random("three"))
    b = w.b
    b.period_open("p", LANE, 100)
    sids = [w.window(3600 * (i + 1)) for i in range(3)]
    b.period_close("p", 5 * 3600)
    other = b.window(LANE, 6 * 3600, SALT)
    b.reserve(OTHER, 7 * DAY, SALT)
    j = w.request()
    got = _answer(pieces, CHECK, j)
    assert got["verdict"] == "accept", got
    published = got["published"]
    assert [x["session"] for x in published["windows"]] == sids
    assert [[a["published"]["bound"] for a in x["audits"]] for x in published["windows"]] == [[0], [1], [2]]
    assert [x["audits"][0]["registration"] for x in published["windows"]] == [
        R.digest(ev["audits"][0]["registration"]) for ev in w.windows]
    assert [s["session"] for s in published["others"]] == [other]
    assert (published["lane"], published["start"], published["stop"]) == (LANE, START, STOP)
    assert len(got["profile"]) == 3 and all(x["delta"] == DELTA for x in got["profile"])

    j["params"]["audit"] = {**AUDIT, "log": [_record(1), _record(2)]}
    got = _answer(pieces, CHECK, j)
    assert [x["audits"][0]["published"]["bound"] for x in got["published"]["windows"]] == [2, 3, 4]


def test_one_stage_v2_itself_answers_each_case(pieces, one_stage_v2):
    """The period's check on one-stage v2 itself answers a case as one-stage v2 answers what the reference composes
    for it. A sample read from the window's session's coins and proved in no session is read by one-stage v2 (its
    coins are the challenger's: the period's ledger, main's history, pins log and session files, composed with the
    case's segment, pass its own reading) and rejected for its proofs; the same case with its segment's commitment
    another's is refused by one-stage v2's reading of the coins. Either way the period rejects under `audit` with
    one-stage v2's reason."""
    def inner(params: dict, record: dict, evidence: dict, blobs: list[str]) -> tuple[str, Any, Any]:
        out = one_stage_v2.exchange(_json({"blobs": blobs, "commitment": _hex(record), "evidence": _hex(evidence),
                                           "params": params}))
        assert out["verdict"] in ("reject", "undecided"), out
        return out["verdict"], out["reason"], None

    w = Week(random.Random("real"))
    b = w.b
    b.period_open("p", LANE, 100)
    sid = w.window(3600, real=True)
    b.period_close("p", 2 * 3600)
    b.reserve(OTHER, 7 * DAY, SALT)
    j = w.request()
    digest = R.digest(w.windows[0]["audits"][0]["registration"])
    want = check(j, inner)
    assert want == {"reason": f"audit: session {sid}, registration {digest[:16]}: {NO_SESSIONS}", "verdict": "reject"}
    kind, text = pieces.call("period_check", j)
    assert kind == "output" and json.loads(text) == want, (kind, text)

    segment = j["evidence"]["windows"][0]["audits"][0]["evidence"]["coins"]["segment"]
    segment["after"] = "00" * 64
    want = check(j, inner)
    assert want["verdict"] == "reject" and want["reason"].startswith(f"audit: session {sid}, registration ")
    assert NO_SESSIONS not in want["reason"], want
    kind, text = pieces.call("period_check", j)
    assert kind == "output" and json.loads(text) == want, (kind, text)


def test_parameters_one_stage_v2_refuses_at_every_window_are_refused_before_the_check(pieces):
    """Parameters whose `audit` one-stage v2 would refuse at every window (a Bernoulli sampling design, a δ of 1, no
    partition) are refused once, as the parameters', as are the ledger's half's own (a monitoring period that ends
    before it starts) and a challenger key at other LMS parameters than the deployed ones."""
    w = Week(random.Random("params"))
    w.b.period_open("p", LANE, 100)
    w.window(3600)
    w.b.period_close("p", 2 * 3600)
    w.b.reserve(OTHER, 7 * DAY, SALT)
    j = w.request()

    def refused(**changes: Any) -> str:
        kind, text = pieces.call("period_check_stub", {**j, "params": {**j["params"], **changes}})
        assert kind == "output", (kind, text)
        return json.loads(text)["refused"]

    for audit in ({**AUDIT, "law": {"law": "bernoulli", "p": "1/2"}}, {**AUDIT, "delta": "1/1"},
                  {**AUDIT, "partition": None}):
        assert refused(audit=audit).startswith("params: audit: "), audit
    assert refused(start=STOP, stop=START).startswith("params: a period's parameters name a family and a lane")
    small = Ch.ChallengerKey(1, 6, lms.public_key(_key(1, 6, 0))).to_json()
    assert refused(key=small).startswith("params: audit: the challenger's key is not at the deployed LMS parameters")


@pytest.mark.parametrize("j", [None, [], {}, {"blobs": [], "evidence": {}, "ledger": [], "params": {}, "x": 1},
                               {"blobs": None, "evidence": {}, "ledger": [], "params": {}},
                               {"blobs": ["0g"], "evidence": {}, "ledger": [], "params": {}},
                               {"blobs": [], "evidence": {}, "ledger": None, "params": {}}],
                         ids=lambda j: D.canonical(j)[:40])
def test_a_malformed_request_is_rejected_by_both(pieces, j):
    with pytest.raises(D.Reject):
        check(j)
    for fn in ("period_check", "period_check_stub"):
        kind, _why = pieces.call(fn, j)
        assert kind == "reject", fn
