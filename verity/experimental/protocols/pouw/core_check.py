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
* **Evidence**: the service's answer for that registration, read as `Core.Answer.ofJson` and `Core.Integrity.ofJson`
  read it (only what `toJson` writes, in the shapes `answer` returns): its verdict, the registration digest it is for,
  and on acceptance the integrity profile and the public outputs ``{"draw": d}``, `d` the verifier's draw as one-stage's
  `draw.py` writes it (any of its laws; PoUW's rule then holds it to subset:K′).
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
_DIGEST = re.compile(r"[0-9a-f]{128}")


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


def _quote(s: str) -> str:
    """Lean's `String.quote`."""
    esc = {"\n": "\\n", "\t": "\\t", "\\": "\\\\", '"': '\\"'}
    return '"' + "".join(esc.get(c) or (f"\\x{ord(c):02x}" if ord(c) <= 31 or ord(c) == 0x7f else c) for c in s) + '"'


def _nat_of(s: str, ctx: str) -> int:
    if s and all(c in "0123456789" for c in s) and (len(s) == 1 or s[0] != "0"):
        return int(s)
    raise Refused(f"{ctx}: {_quote(s)} is not a natural number in canonical form")


def ratio(s: str, ctx: str) -> Fraction:
    """`Core.Ratio.ofString`: ``n`` or ``n/d`` in lowest terms with d >= 2, one spelling per ratio."""
    parts = s.split("/")
    if len(parts) == 1:
        return Fraction(_nat_of(parts[0], ctx))
    if len(parts) == 2:
        n, d = _nat_of(parts[0], ctx), _nat_of(parts[1], ctx)
        if d <= 1:
            raise Refused(f"{ctx}: {_quote(s)} has denominator {d}")
        if math.gcd(n, d) != 1:
            raise Refused(f"{ctx}: {_quote(s)} is not in lowest terms")
        return Fraction(n, d)
    raise Refused(f"{ctx}: {_quote(s)} is not a ratio n/d")


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
    if not (_str(j["query"]) and _DIGEST.fullmatch(j["query"])):
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


# ---- the evidence, as `Core.Answer.ofJson` and `Core.Integrity.ofJson` read it (with `Flock.J`'s messages) ------------

def _get(j: Any, k: str, ctx: str) -> Any:
    if not isinstance(j, dict) or k not in j:
        raise Refused(f"{ctx}: {k} missing")
    return j[k]


def _has(j: Any, k: str) -> bool:
    return isinstance(j, dict) and k in j


def _as_str(v: Any, ctx: str) -> str:
    if not _str(v):
        raise Refused(f"{ctx}: not a string")
    return v


def _as_nat(v: Any, ctx: str) -> int:
    if not _nat(v):
        raise Refused(f"{ctx}: not a natural number")
    return v


def _as_arr(v: Any, ctx: str) -> list:
    if not isinstance(v, list):
        raise Refused(f"{ctx}: not a list")
    return v


def _as_bool(v: Any, ctx: str) -> bool:
    if not isinstance(v, bool):
        raise Refused(f"{ctx}: not a boolean")
    return v


def _str_at(j: Any, k: str, ctx: str) -> str:
    return _as_str(_get(j, k, ctx), f"{ctx}.{k}")


def _nat_at(j: Any, k: str, ctx: str) -> int:
    return _as_nat(_get(j, k, ctx), f"{ctx}.{k}")


def _arr_at(j: Any, k: str, ctx: str) -> list:
    return _as_arr(_get(j, k, ctx), f"{ctx}.{k}")


def _unhex(s: str) -> str:
    """`Flock.unhex`, keeping the hex: `hexOf` writes back exactly what it accepts."""
    if len(s) % 2:
        raise Refused("odd-length hex")
    if any(c not in "0123456789abcdef" for c in s):
        raise Refused("not lowercase hex")
    return s


def _exactly(written: Any, j: Any, ctx: str) -> None:
    if _canon(written) != _canon(j):
        raise Refused(f"{ctx}: not in the form toJson writes")


def _core_draw(j: Any, ctx: str) -> dict:
    """`Core.Draw.ofJson`, as `Draw.toJson` writes it."""
    if _has(j, "bernoulli"):
        d: dict = {"bernoulli": str(ratio(_str_at(j, "bernoulli", ctx), f"{ctx}.bernoulli"))}
    elif _has(j, "subset"):
        d = {"subset": _nat_at(j, "subset", ctx)}
    elif _has(j, "strata"):
        c = f"{ctx}.strata"
        d = {"law": "stratified", "strata": [{"name": _str_at(s, "name", c), "size": _nat_at(s, "size", c),
                                              "k": _nat_at(s, "k", c)} for s in _arr_at(j, "strata", ctx)]}
    else:
        raise Refused(f"{ctx}: not a bernoulli, subset or stratified draw")
    _exactly(d, j, ctx)
    return d


def _level(j: Any, ctx: str) -> dict:
    """`Core.Level.ofJson`."""
    name = _str_at(j, "name", ctx)
    q = _get(j, "query", ctx)
    query = None if q is None else _as_str(q, f"{ctx}.query")
    size = _nat_at(j, "size", ctx)
    d = _get(j, "draw", ctx)
    lv = {"name": name, "query": query, "size": size, "draw": None if d is None else _core_draw(d, f"{ctx}.draw")}
    _exactly(lv, j, ctx)
    return lv


def integrity(j: Any, ctx: str = "profile") -> dict:
    """`Core.Integrity.ofJson`: `drawn` as (level, count) pairs in level order, for the levels it names."""
    levels: list[dict] = []
    for lv in _arr_at(j, "levels", ctx):
        levels.append(_level(lv, f"{ctx}.levels[{len(levels)}]"))
    counts = _get(j, "drawn", ctx)
    drawn = [(lv["name"], _nat_at(counts, lv["name"], f"{ctx}.drawn")) for lv in levels if _has(counts, lv["name"])]

    def strs(k: str) -> list[str]:
        return [_as_str(v, f"{ctx}.{k}") for v in _arr_at(j, k, ctx)]
    p = {"programs": strs("programs"), "prescribed": strs("prescribed")}
    p["linked"] = _as_bool(_get(j, "linked", ctx), f"{ctx}.linked")
    p["covered"] = strs("covered")
    p["levels"] = levels
    p["delta"] = ratio(_str_at(j, "delta", ctx), f"{ctx}.delta")
    p["drawn"] = drawn
    _exactly(integrity_json(p), j, ctx)
    return p


def integrity_json(p: dict) -> dict:
    """`Core.Integrity.toJson`."""
    return {"programs": p["programs"], "prescribed": sorted(p["prescribed"]), "linked": p["linked"],
            "covered": sorted(p["covered"]), "levels": p["levels"], "delta": str(p["delta"]), "drawn": dict(p["drawn"])}


def answer_of(j: Any, ctx: str = "answer") -> dict:
    """`Core.Answer.ofJson`: only what `Answer.toJson` writes, in the shapes `answer` returns."""
    verdict = _str_at(j, "verdict", ctx)
    if verdict not in VERDICTS:
        raise Refused(f"{ctx}.verdict: not accept, reject or undecided")
    protocol = _str_at(j, "protocol", ctx)
    reason = _get(j, "reason", ctx)
    if reason is not None:
        _as_str(reason, f"{ctx}.reason")
    com = _get(j, "commitment", ctx)
    if com is not None:
        com = _unhex(_as_str(com, f"{ctx}.commitment"))
    pub, prof = _get(j, "public", ctx), _get(j, "profile", ctx)
    if verdict == "accept":
        if not (reason is None and pub is not None and prof is not None):
            raise Refused(f"{ctx}: an accept has no reason, and both its public outputs and its profile")
    elif not (reason is not None and pub is None and prof is None):
        raise Refused(f"{ctx}: a {verdict} has a reason, and neither public outputs nor a profile")
    if sorted(j) != list(ANSWER_KEYS):
        raise Refused(f"{ctx}: not in the form Answer.toJson writes")
    return {"protocol": protocol, "verdict": verdict, "reason": reason, "commitment": com, "public": pub, "profile": prof}


_DRAW_KEYS = {"subset": ["k", "law", "population", "units"], "bernoulli": ["law", "p", "population", "units"],
              "stratified": ["k", "law", "population", "strata", "units"],
              "work": ["k", "law", "population", "strata", "units"]}


def _keys_are(j: Any, ks: list[str], ctx: str) -> None:
    if not isinstance(j, dict) or sorted(j) != ks:
        raise Refused(f"{ctx}: keys are not {', '.join(ks)}")


def _nats_at(j: Any, k: str, ctx: str) -> list[int]:
    return [_as_nat(v, f"{ctx}.{k}") for v in _arr_at(j, k, ctx)]


def _range(v: Any, ctx: str) -> tuple[int, int]:
    if not (isinstance(v, list) and len(v) == 2):
        raise Refused(f"{ctx}: not a range [lo, hi]")
    return _as_nat(v[0], ctx), _as_nat(v[1], ctx)


def _strata(work: bool, j: Any, ctx: str) -> list[dict]:
    out: list[dict] = []
    for s in _arr_at(j, "strata", ctx):
        c = f"{ctx}.strata[{len(out)}]"
        _keys_are(s, ["floor", "k", "template", "units", "work"] if work else ["k", "template", "units"], c)
        st = {"template": _str_at(s, "template", c), "units": [_range(r, f"{c}.units") for r in _arr_at(s, "units", c)],
              "k": _nat_at(s, "k", c)}
        if work:
            st["work"], st["floor"] = _nat_at(s, "work", c), _nat_at(s, "floor", c)
        out.append(st)
    return out


def unit_draw(j: Any) -> dict:
    """The public outputs ``{"draw": d}``, `d` one of one-stage's `draw.py` laws with exactly its keys (`check_draw`'s)."""
    _keys_are(j, ["draw"], "public")
    d, ctx = j["draw"], "public.draw"
    law = _str_at(d, "law", ctx)
    if law not in _DRAW_KEYS:
        raise Refused(f"{ctx}.law: not subset, bernoulli, stratified or work")
    _keys_are(d, (["closure"] if law == "work" and "closure" in d else []) + _DRAW_KEYS[law], ctx)
    out: dict[str, Any] = {"law": law, "population": _nat_at(d, "population", ctx)}
    if law == "bernoulli":
        out["p"] = _str_at(d, "p", ctx)
    else:
        out["k"] = _nat_at(d, "k", ctx)
    if law in ("stratified", "work"):
        out["strata"] = _strata(law == "work", d, ctx)
    if "closure" in d:
        out["closure"] = _nats_at(d, "closure", ctx)
    out["units"] = _nats_at(d, "units", ctx)
    return out


def law_name(d: dict) -> str:
    """`parse_law`'s spelling of a draw's law."""
    return f"bernoulli:{d['p']}" if d["law"] == "bernoulli" else f"{d['law']}:{d['k']}"


def evidence(b: bytes) -> dict:
    """The service's answer for the registration (`answer_of`), and on acceptance its profile and draw."""
    a = answer_of(canonical(b))
    if a["verdict"] == "accept":
        a["profile"] = integrity(a["profile"])
        a["draw"] = unit_draw(a.pop("public"))
    return a


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
    if prof["levels"] != [{"name": LEVEL, "query": rec["partition"], "size": w.n, "draw": {"subset": k}}]:
        return "reject", f"profile: its levels are not the registered partition's {w.n} units drawn subset:{k}", None
    if prof["delta"] != p["delta"]:
        return "reject", f"profile: delta {prof['delta']}, not the window's {p['delta']}", None
    if prof["drawn"] != [(LEVEL, k)]:
        return "reject", f"profile: drawn {_canon(dict(prof['drawn']))}, not {k} units", None
    if not (draw["law"] == "subset" and draw["k"] == k) or draw["population"] != w.n:
        return "reject", f"draw: {law_name(draw)} over {draw['population']}, not subset:{k} over {w.n}", None
    units = draw["units"]
    if len(units) != k or any(b <= a for a, b in zip(units, units[1:])) or (units and units[-1] >= w.n):
        return "reject", f"draw: its {len(units)} units are not {k} ascending distinct units below {w.n}", None
    public = {**{i: items[i] for i in Wd.REGISTERED}, "verdict": True, "drawn": list(units)}
    return "accept", None, (public, integrity_json(prof))


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


def case(j: Any) -> dict:
    """`Core.Protocol.case` on one `serve` case, ``{"commitment": hex, "evidence": hex, "params": JSON}``: a malformed one
    is rejected with no commitment."""
    try:
        params_j = _get(j, "params", "the case")
        c = _unhex(_str_at(j, "commitment", "the case"))
        e = _unhex(_str_at(j, "evidence", "the case"))
        params_b = _canon(params_j).encode()
    except Refused as err:
        return {"protocol": ID, "verdict": "reject", "reason": f"case: {err}", "commitment": None, "public": None,
                "profile": None}
    return answer(params_b, bytes.fromhex(c), bytes.fromhex(e))


def line(a: dict) -> str:
    """The answer's canonical JSON line (`Core.Answer.canonical`)."""
    return _canon(a)


__all__ = ["ANSWER_KEYS", "FORMATS", "ID", "LEVEL", "ONE_STAGE", "Refused", "SCHEME", "answer", "answer_of", "canonical",
           "case", "check", "commitment", "digest", "evidence", "integrity", "integrity_json", "law_name", "line", "params",
           "ratio", "unit_draw"]
