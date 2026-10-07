"""PoUW's check of one audit window as a core protocol (`verity/experimental/core/protocols/interface`): the Python
reference of the Lean port `pouw_check` (`verity/experimental/core/protocols/pouw/check`), which the difftest holds to it
answer for answer.

The proof service registers the window, draws, proves and verifies; PoUW states what it registers and what the outcome
must show, and publishes `window.PUBLIC` from it.  This is that composition in one function, the parts of
`benchmarks/pouw/served_zk/served_commit.py` (`public_problem`, `register` and `Challenger.outcome`) that are PoUW's.

* **Params**, canonical JSON: the check item (`check`: the tile template, its work per tile w, the commitment scheme and
  the public bound on entries), ε and δ as ratio strings, the grid's bits, the declared range of tile counts, the
  window's context and the check query's SHA-512 (one-stage's `query_digest` of the tile template's instance query).
* **Commitment**: the window's registration, one-stage's record (`registration.record`) as canonical JSON, its window
  ``{"kind": "served", "context", "public"}`` holding exactly `window.REGISTERED`.  Its digest is the record's own,
  `registration.digest`, the one the service's receipt, draw and outcome cite.
* **Evidence**: the service's answer for that registration, in the core interface's answer form (`answer`'s, with
  protocol one-stage's): its verdict, the registration digest it is for, and on acceptance the integrity profile and the
  draw ``{"draw": {"k", "law": "subset", "population", "units"}}``.
* **The check** accepts when the registration is PoUW's (`window.REGISTERED`, each item the verifier's own at the
  registered level, N = point(level + 1) in the declared range, the law subset:K′ with K′ = `window.k_prime`, the check's
  query and scheme), the service accepted this registration, and its profile and draw are of the registered partition's N
  units drawn subset:K′ at the window's δ.  It publishes exactly `window.PUBLIC` and the service's profile.  The service's
  reject is a reject and its undecided (the verifier of record hasn't run) is undecided.

Every refusal has one reason, the first rule it breaks in the order below; the Lean port writes the same strings.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any

from verity.experimental.protocols.pouw import window as Wd

ID = "verity/pouw/v1"
ONE_STAGE = "verity/sampled-proofs/one-stage/v1"
SCHEME = "frame-v3-sha512/hm96-sha512"
FORMATS = ("verity/registration/v1", "verity/registration/v2")
LEVEL = "unit"
PARAMS_KEYS = ("check", "context", "delta", "eps", "grid_bits", "query", "range")
CHECK_KEYS = ("entries", "scheme", "template", "tile_work")
ANSWER_KEYS = ("commitment", "profile", "protocol", "public", "reason", "verdict")
VERDICTS = ("accept", "reject", "undecided")
_DIGEST = re.compile(r"^[0-9a-f]{128}$")


class Refused(ValueError):
    """A decoder's refusal; `answer` prefixes it with the input it read."""


# ---- canonical JSON and exact ratios, as `Core.canonicalJson` and `Core.Ratio` read them ---------------------------------

def _canon(x: Any) -> str:
    """`Flock.canon`: compact, keys sorted, serde's escaping (non-ASCII as itself), integers only."""
    if isinstance(x, float):
        raise Refused("canonical JSON: a non-integer number")
    if isinstance(x, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _canon(v) for k, v in sorted(x.items())) + "}"
    if isinstance(x, list):
        return "[" + ",".join(_canon(v) for v in x) + "]"
    return json.dumps(x, ensure_ascii=False)


def canonical(b: bytes) -> Any:
    try:
        s = b.decode()
    except UnicodeDecodeError:
        raise Refused("not UTF-8") from None
    try:
        x = json.loads(s)
    except ValueError as e:
        raise Refused(f"not JSON ({e})") from None
    if _canon(x) != s:
        raise Refused("not canonical JSON (compact, keys sorted, integers only)")
    return x


def _nat_of(s: str, ctx: str) -> int:
    if s and all(c in "0123456789" for c in s) and (len(s) == 1 or s[0] != "0"):
        return int(s)
    raise Refused(f'{ctx}: "{s}" is not a natural number in canonical form')


def ratio(s: str, ctx: str) -> Fraction:
    """`Core.Ratio.ofString`: ``n`` or ``n/d`` in lowest terms with d >= 2, one spelling per ratio."""
    parts = s.split("/")
    if len(parts) == 1:
        return Fraction(_nat_of(parts[0], ctx))
    if len(parts) == 2:
        n, d = _nat_of(parts[0], ctx), _nat_of(parts[1], ctx)
        if d <= 1:
            raise Refused(f'{ctx}: "{s}" has denominator {d}')
        if math.gcd(n, d) != 1:
            raise Refused(f'{ctx}: "{s}" is not in lowest terms')
        return Fraction(n, d)
    raise Refused(f'{ctx}: "{s}" is not a ratio n/d')


def _nat(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _str(x: Any) -> bool:
    return isinstance(x, str)


# ---- the three inputs ---------------------------------------------------------------------------------------------------

def params(j: Any) -> dict:
    """The verifier's own parameters of the window."""
    if not isinstance(j, dict) or sorted(j) != list(PARAMS_KEYS):
        raise Refused(f"keys are not {', '.join(PARAMS_KEYS)}")
    c = j["check"]
    if not isinstance(c, dict) or sorted(c) != list(CHECK_KEYS):
        raise Refused(f"check's keys are not {', '.join(CHECK_KEYS)}")
    if not (_nat(c["entries"]) and c["entries"] >= 2):
        raise Refused("check.entries is not a natural number of at least 2")
    if c["scheme"] != SCHEME:
        raise Refused(f"check.scheme is not {SCHEME}")
    if not (_str(c["template"]) and c["template"]):
        raise Refused("check.template is not a non-empty string")
    if not (_nat(c["tile_work"]) and c["tile_work"] >= 1):
        raise Refused("check.tile_work is not a positive natural number")
    if not (_str(j["context"]) and j["context"]):
        raise Refused("context is not a non-empty string")
    out: dict[str, Any] = {"check": c, "context": j["context"]}
    for k, ok, what in (("delta", lambda r: 0 < r < 1, "(0, 1)"), ("eps", lambda r: 0 < r <= 1, "(0, 1]")):
        if not _str(j[k]):
            raise Refused(f"{k} is not a string")
        r = ratio(j[k], k)
        if not ok(r):
            raise Refused(f"{k} is not in {what}")
        out[k] = r
    if not _nat(j["grid_bits"]):
        raise Refused("grid_bits is not a natural number")
    if not (_str(j["query"]) and _DIGEST.match(j["query"])):
        raise Refused("query is not a SHA-512 digest (128 hex digits)")
    rg = j["range"]
    if not (isinstance(rg, list) and len(rg) == 2 and all(_nat(x) for x in rg) and 1 <= rg[0] <= rg[1]):
        raise Refused("range is not [lo, hi] with 1 <= lo <= hi")
    return {**out, "grid_bits": j["grid_bits"], "query": j["query"], "range": tuple(rg)}


def _root(r: Any) -> dict | None:
    if isinstance(r, dict) and sorted(r) == ["domain", "leaves", "port", "root"] and _str(r["domain"]) \
            and _nat(r["leaves"]) and _str(r["port"]) and _str(r["root"]):
        return r
    return None


def commitment(b: bytes) -> dict:
    """The registration: one-stage's record, read for what PoUW's rule reads; `bytes` is what its digest hashes."""
    rec = canonical(b)
    if not isinstance(rec, dict) or rec.get("format") not in FORMATS:
        raise Refused(f"format is not {' or '.join(FORMATS)}")
    for k in ("program", "query", "partition", "scheme"):
        if not _str(rec.get(k)):
            raise Refused(f"{k} is missing or not a string")
    if not _nat(rec.get("population")):
        raise Refused("population is missing or not a natural number")
    if "law" not in rec:
        raise Refused("law is missing")
    roots = rec.get("roots")
    if not isinstance(roots, list) or any(_root(r) is None for r in roots):
        raise Refused("roots is not a list of {domain, leaves, port, root}")
    w = rec.get("window")
    if not (isinstance(w, dict) and sorted(w) == ["context", "kind", "public"] and _str(w["context"]) and _str(w["kind"])
            and isinstance(w["public"], dict)):
        raise Refused("window is not {context, kind, public} with strings and an object")
    return {"bytes": b, "format": rec["format"], "record": rec}


def digest(c: dict) -> bytes:
    """`registration.digest`: SHA-512(format || 0 || the canonical record)."""
    return hashlib.sha512(c["format"].encode() + b"\0" + c["bytes"]).digest()


def _strings(x: Any) -> bool:
    return isinstance(x, list) and all(_str(s) for s in x)


def _draw_json(d: Any) -> Any:
    """A level's draw as `Core.Draw.toJson` writes it, or `Refused`."""
    if d is None:
        return None
    if isinstance(d, dict) and sorted(d) == ["subset"] and _nat(d["subset"]):
        return {"subset": d["subset"]}
    if isinstance(d, dict) and sorted(d) == ["bernoulli"] and _str(d["bernoulli"]):
        return {"bernoulli": str(ratio(d["bernoulli"], "bernoulli"))}
    if isinstance(d, dict) and sorted(d) == ["law", "strata"] and d["law"] == "stratified" and isinstance(d["strata"], list) \
            and all(isinstance(s, dict) and sorted(s) == ["k", "name", "size"] and _str(s["name"]) and _nat(s["size"])
                    and _nat(s["k"]) for s in d["strata"]):
        return {"law": "stratified", "strata": [dict(s) for s in d["strata"]]}
    raise Refused("a draw")


def integrity(j: Any) -> dict:
    """`Core.Integrity` read back from its JSON, written again as `Integrity.toJson` writes it."""
    keys = ["covered", "delta", "drawn", "levels", "linked", "prescribed", "programs"]
    try:
        if not (isinstance(j, dict) and sorted(j) == keys and _strings(j["covered"]) and _str(j["delta"])
                and isinstance(j["drawn"], dict) and all(_nat(v) for v in j["drawn"].values())
                and isinstance(j["levels"], list) and isinstance(j["linked"], bool) and _strings(j["prescribed"])
                and _strings(j["programs"])):
            raise Refused("shape")
        levels = []
        for lv in j["levels"]:
            if not (isinstance(lv, dict) and sorted(lv) == ["draw", "name", "query", "size"] and _str(lv["name"])
                    and (lv["query"] is None or _str(lv["query"])) and _nat(lv["size"])):
                raise Refused("level")
            levels.append({"draw": _draw_json(lv["draw"]), "name": lv["name"], "query": lv["query"], "size": lv["size"]})
        delta = ratio(j["delta"], "delta")
    except Refused:
        raise Refused("profile is not an integrity profile") from None
    return {"covered": sorted(j["covered"]), "delta": str(delta), "drawn": dict(j["drawn"]), "levels": levels,
            "linked": j["linked"], "prescribed": sorted(j["prescribed"]), "programs": list(j["programs"])}


def _subset_draw(j: Any) -> dict:
    d = j.get("draw") if isinstance(j, dict) and sorted(j) == ["draw"] else None
    if not (isinstance(d, dict) and sorted(d) == ["k", "law", "population", "units"] and d["law"] == "subset"
            and _nat(d["k"]) and _nat(d["population"]) and isinstance(d["units"], list) and all(_nat(u) for u in d["units"])):
        raise Refused("public is not {draw} with a subset draw {k, law, population, units}")
    return d


def evidence(b: bytes) -> dict:
    """The service's answer for the registration, in the core interface's answer form."""
    j = canonical(b)
    if not isinstance(j, dict) or sorted(j) != list(ANSWER_KEYS):
        raise Refused(f"keys are not {', '.join(ANSWER_KEYS)}")
    if not _str(j["protocol"]):
        raise Refused("protocol is not a string")
    if j["verdict"] not in VERDICTS:
        raise Refused("verdict is not accept, reject or undecided")
    if not (j["commitment"] is None or _str(j["commitment"])):
        raise Refused("commitment is not null or a string")
    accepted = j["verdict"] == "accept"
    shaped = (j["reason"] is None and j["public"] is not None and j["profile"] is not None) if accepted else \
        (_str(j["reason"]) and j["public"] is None and j["profile"] is None)
    if not shaped:
        raise Refused("an accepted answer has public outputs and a profile and no reason; any other has a reason and neither")
    out = {"protocol": j["protocol"], "verdict": j["verdict"], "commitment": j["commitment"], "reason": j["reason"]}
    if accepted:
        out["profile"] = integrity(j["profile"])
        out["draw"] = _subset_draw(j["public"])
    return out


# ---- the check -----------------------------------------------------------------------------------------------------------

def check_item(p: dict) -> dict:
    return dict(p["check"])


def expected_items(p: dict, ell: int, w: Wd.Work, k: int, root_w: Any) -> dict:
    """The registered items as the verifier computes them at level `ell` (`served_commit.items_of`)."""
    tw = p["check"]["tile_work"]
    return {"check": check_item(p), "root_W": root_w, "work": w.to_json(tw),
            "sampling": {"delta": str(p["delta"]), "eps": str(p["eps"]), "grid_bits": p["grid_bits"],
                         "law": {"k": k, "law": "subset"}}}


def check(p: dict, c: dict, e: dict) -> tuple[str, Any, Any]:
    """(verdict, reason, (public, profile) on acceptance)."""
    rec = c["record"]
    win, items = rec["window"], rec["window"]["public"]
    bits, (lo, hi) = p["grid_bits"], p["range"]
    if win["kind"] != "served" or win["context"] != p["context"]:
        return "reject", (f"window: registered kind {win['kind']} in context {win['context']}, the verifier's served in "
                          f"{p['context']}"), None
    if sorted(items) != sorted(Wd.REGISTERED):
        return "reject", (f"public-items: registered {', '.join(sorted(items))}, declared "
                          f"{', '.join(sorted(Wd.REGISTERED))}"), None
    work = items["work"]
    ell = work.get("level") if isinstance(work, dict) else None
    if not _nat(ell):
        return "reject", "level: the registration names no work level", None
    if ell + 1 > Wd.level(Wd.floor_point(hi, bits), bits) or Wd.point(ell, bits) < lo:
        return "reject", f"level: level {ell} is outside the window's range {lo}..{hi}", None
    w = Wd.Work.at(ell, bits)
    tw = p["check"]["tile_work"]
    k = Wd.k_prime(p["eps"], p["delta"], w.credited * tw, w.n, tw)
    root_w = next((r["root"] for r in rec["roots"] if r["port"] == "W"), None)
    want = expected_items(p, ell, w, k, root_w)
    wrong = [i for i in sorted(want) if _canon(items[i]) != _canon(want[i])]
    if wrong:
        return "reject", f"public-value: {', '.join(wrong)}", None
    if rec["population"] != w.n:
        return "reject", f"population: registered {rec['population']}, the level's {w.n}", None
    law = {"k": k, "law": "subset"}
    if _canon(rec["law"]) != _canon(law):
        return "reject", f"law: registered {_canon(rec['law'])}, the window's {_canon(law)}", None
    if rec["query"] != p["query"]:
        return "reject", f"query: registered {rec['query']}, the check's {p['query']}", None
    if rec["scheme"] != p["check"]["scheme"]:
        return "reject", f"scheme: registered {rec['scheme']}, the check's {p['check']['scheme']}", None
    if e["protocol"] != ONE_STAGE:
        return "reject", f"outcome: protocol {e['protocol']}, not one-stage's {ONE_STAGE}", None
    d = digest(c).hex()
    if e["commitment"] != d:
        return "reject", f"outcome: for registration {e['commitment'] or 'null'}, not this one {d}", None
    if e["verdict"] != "accept":
        return e["verdict"], f"service: {e['reason']}", None
    prof, draw = e["profile"], e["draw"]
    if prof["programs"] != [rec["program"]]:
        return "reject", f"profile: programs {', '.join(prof['programs'])}, not the registered {rec['program']}", None
    if prof["levels"] != [{"draw": {"subset": k}, "name": LEVEL, "query": rec["partition"], "size": w.n}]:
        return "reject", f"profile: its levels are not the registered partition's {w.n} units drawn subset:{k}", None
    if prof["delta"] != str(p["delta"]):
        return "reject", f"profile: delta {prof['delta']}, not the window's {p['delta']}", None
    if prof["drawn"] != {LEVEL: k}:
        return "reject", f"profile: drawn {_canon(prof['drawn'])}, not {k} units", None
    if draw["k"] != k or draw["population"] != w.n:
        return "reject", f"draw: subset:{draw['k']} over {draw['population']}, not subset:{k} over {w.n}", None
    units = draw["units"]
    if len(units) != k or any(b <= a for a, b in zip(units, units[1:])) or (units and units[-1] >= w.n):
        return "reject", f"draw: its {len(units)} units are not {k} ascending distinct units below {w.n}", None
    public = {**{i: items[i] for i in Wd.REGISTERED}, "verdict": True, "drawn": list(units)}
    return "accept", None, (public, prof)


def answer(params_b: bytes, commitment_b: bytes, evidence_b: bytes) -> dict:
    """`Core.Protocol.answer`: every input gets a verdict, a decoding failure being a rejection naming its input."""
    def out(verdict: str, reason: str | None, d: bytes | None, pub: Any = None, prof: Any = None) -> dict:
        return {"protocol": ID, "verdict": verdict, "reason": reason, "commitment": None if d is None else d.hex(),
                "public": pub, "profile": prof}
    try:
        p = params(canonical(params_b))
    except Refused as e:
        return out("reject", f"params: {e}", None)
    try:
        c = commitment(commitment_b)
    except Refused as e:
        return out("reject", f"commitment: {e}", None)
    d = digest(c)
    try:
        ev = evidence(evidence_b)
    except Refused as e:
        return out("reject", f"evidence: {e}", d)
    verdict, reason, acc = check(p, c, ev)
    return out(verdict, reason, d, *(acc or (None, None)))


def case(line: str) -> dict:
    """One `serve` case, ``{"commitment": hex, "evidence": hex, "params": JSON}``, well formed."""
    j = json.loads(line)
    return answer(_canon(j["params"]).encode(), bytes.fromhex(j["commitment"]), bytes.fromhex(j["evidence"]))


def line(a: dict) -> str:
    """The answer's canonical JSON line (`Core.Answer.canonical`)."""
    return _canon(a)


__all__ = ["ANSWER_KEYS", "FORMATS", "ID", "LEVEL", "ONE_STAGE", "Refused", "SCHEME", "answer", "canonical", "case",
           "check", "commitment", "digest", "evidence", "integrity", "line", "params", "ratio"]
