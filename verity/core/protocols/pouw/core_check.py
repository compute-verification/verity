"""PoUW's check of one audit window as a core protocol (`verity/core/protocols/interface`): the Python
reference of the Lean port `pouw_check` (`verity/core/protocols/pouw/lean/check`), which the difftest holds to it
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

`opening` is the rule on one unit's opening beside it (`served_commit.check`'s, which calls it): the tile proof's public
file opens the unit's rows under the registered window root, through its entry and the entry's side trees.

`pass_epoch` is the rule on the pass a window is received from (`served_commit.Challenger.pass_epoch`, which calls it):
its manifest re-starts its epoch under the challenger's own coin commitment, and the window's context names it.
`draw` is the rule on the window's draw (`Challenger.select`'s): the epoch's draw coin opens after the registration,
and the drawn units are Lean's ``Draw.subset`` of the key it derives.  Both read the coin versions in `COIN_VERSIONS`,
and a replay of a recorded draw `COIN_EARLY` too.

Every refusal has one reason, the first rule it breaks in the order below; the Lean port writes the same strings.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN, SALT_DOMAIN
from verity.core.protocols.pouw.protocol import U64, tagged
from verity.ml.commitments import Commitment, Opening, VerificationLimits, verify_opening
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import row_leaf
from verity.ml.randomness import derive, frame

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
#: one-stage's served domain (`registration.served_domain`), the window root's
SERVED_DOMAIN = "verity/one-stage/served-domain/v1"
#: a tile proof's first two rows, in its header's key order: its A row and its B row, each side's fp8-p row schema
#: (`circuit.leaves.ROW_SCHEMA["fp8-p", X | W]`)
OPENED = (("a", "hm96-sha512/row-seg/v1"), ("b", "hm96-sha512/row/v2"))
SALT_BYTES = 192
BC_BYTES = 128


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


def _registered(p: dict, c: dict) -> tuple[str | None, Any]:
    rec = c["record"]
    win, items = rec["window"], rec["window"]["public"]
    bits, (lo, hi) = p["grid_bits"], p["range"]
    if win["kind"] != "served" or win["context"] != p["context"]:
        return (f"window: registered kind {win['kind']} in context {win['context']}, the verifier's served in "
                f"{p['context']}"), None
    if sorted(items) != sorted(Wd.REGISTERED):
        return f"public-items: registered {', '.join(sorted(items))}, declared {', '.join(sorted(Wd.REGISTERED))}", None
    work = items["work"]
    ell = work.get("level") if isinstance(work, dict) else None
    if not _nat(ell):
        return "level: the registration names no work level", None
    if ell + 1 > Wd.level(Wd.floor_point(hi, bits), bits) or Wd.point(ell, bits) < lo:
        return f"level: level {ell} is outside the window's range {lo}..{hi}", None
    w = Wd.Work.at(ell, bits)
    tw = p["check"]["tile_work"]
    k = Wd.k_prime(p["eps"], p["delta"], w.credited * tw, w.n, tw)
    root_w = next((r["root"] for r in rec["roots"] if r["port"] == "W"), None)
    want = expected_items(p, ell, w, k, root_w)
    wrong = [i for i in sorted(want) if _canon(items[i]) != _canon(want[i])]
    if wrong:
        return f"public-value: {', '.join(wrong)}", None
    if rec["population"] != w.n:
        return f"population: registered {rec['population']}, the level's {w.n}", None
    law = {"k": k, "law": "subset"}
    if _canon(rec["law"]) != _canon(law):
        return f"law: registered {_canon(rec['law'])}, the window's {_canon(law)}", None
    if rec["query"] != p["query"]:
        return f"query: registered {rec['query']}, the check's {p['query']}", None
    if rec["scheme"] != p["check"]["scheme"]:
        return f"scheme: registered {rec['scheme']}, the check's {p['check']['scheme']}", None
    return None, (w, k, items)


def registered(p: dict, c: dict) -> str | None:
    """PoUW's rule over the registration (`PouwCheck.registered`): why it fails, or None."""
    return _registered(p, c)[0]


def check(p: dict, c: dict, e: dict) -> tuple[str, Any, Any]:
    """(verdict, reason, (public, profile) on acceptance)."""
    rec = c["record"]
    why, fixed = _registered(p, c)
    if why is not None:
        return "reject", why, None
    w, k, items = fixed
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


# ---- a unit's opening ------------------------------------------------------------------------------------------------------

def window_domain(p: dict, rec: dict) -> CommitmentDomain:
    """The window root's domain: one-stage's served domain (`registration.served_domain`) of the registered program and
    partition, over the check's P entries in the window's context."""
    n = p["check"]["entries"]
    binding = identity_digest_sha512(SERVED_DOMAIN, {"program": rec["program"], "partition": rec["partition"],
                                                     "port": Wd.PORT, "schema": Wd.ENTRY_SCHEMA, "leaves": n,
                                                     "run": p["context"]})
    return CommitmentDomain(binding, Wd.OWNER, RangeIndexedDomain(0, n), hash="sha512")


def verify_path(dom: CommitmentDomain, root: bytes, index: int, leaf: bytes, sibs: list[bytes]) -> bool:
    """Leaf `index` with siblings `sibs` (leaf level first) hashes up to `root` in `dom`."""
    if not 0 <= index < dom.count or len(sibs) != merkle_depth(dom.count):
        return False
    digest = leaf
    for level, s in enumerate(sibs):
        digest = dom.node(level, index >> 1, *((digest, s) if index % 2 == 0 else (s, digest)))
        index >>= 1
    return digest == root


def _fromhex(s: Any, n: int) -> bytes | None:
    """`bytes.fromhex(s)` when `s` is a string of `n` bytes' hex, else None."""
    if not _str(s):
        return None
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return None
    return b if len(b) == n else None


def _path(x: Any) -> list[bytes] | None:
    sibs = [_fromhex(s, 64) for s in x] if isinstance(x, list) else None
    return None if sibs is None or None in sibs else sibs


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _no_constant(s: str) -> Any:
    raise ValueError(f"{s} is not JSON")


def opening(p: dict, c: dict, units: list[int] | None, unit: int, pub: bytes, op: Any) -> str | None:
    """None when the tile proof's public file `pub` opens unit `unit`'s rows under the registered window root, else why
    not (`served_commit.check`).  `units` is the draw's (None for a unit opened undrawn, the filler's); `op` the prover's
    opening ``{"entry", "entry_value", "entry_salt", "window_path", "a": {"index", "path"}, "b": …}``: the unit's entry,
    its salt and its path to the window root, and each side's row with its path to the entry's side root.  The proof's
    first two rows, in its header's key order (serde's, as C-Flock reads it), are its A row and its B row, and its body
    starts with their b ‖ c."""
    if units is not None and unit not in units:
        return f"unit {unit} is not drawn"
    rec = c["record"]
    dom = window_domain(p, rec)
    roots = {r["port"]: r for r in rec["roots"]}
    if roots.get(Wd.PORT, {}).get("domain") != dom.domain_id.hex():
        return "the registered window domain is not the window's served domain"
    root = _fromhex(roots[Wd.PORT]["root"], 64)
    if root is None:
        return "the registered window root is not a SHA-512 digest"
    try:
        entry = Wd.Entry.from_json(op["entry_value"])
        entry.row()
    except (KeyError, TypeError, ValueError):
        return "the opening names no entry"
    e = op.get("entry")
    if not _int(e):
        return "the opening names no entry"
    salt, sibs = _fromhex(op.get("entry_salt"), SALT_BYTES), _path(op.get("window_path"))
    sides = {s: op.get(s) for s, _ in OPENED}
    paths = {s: _path(o.get("path")) if isinstance(o, dict) else None for s, o in sides.items()}
    if salt is None or sibs is None or any(p_ is None or not _int(sides[s].get("index")) for s, p_ in paths.items()):
        return (f"the opening's salt is not {SALT_BYTES} bytes, or a path or a side's index is not one (paths of SHA-512 "
                "digests)")
    if not (0 <= e < dom.count and verify_path(dom, root, e, Wd.entry_leaf(dom, e, Wd.entry_bc(entry, salt)), sibs)):
        return f"entry {e} is not the registered window's entry"
    try:
        at = dict(zip(("a", "b"), Wd.tile_in(entry, unit)))
    except ValueError as x:
        return f"entry {e}: {x}"
    head, nl, body = pub.partition(b"\n")
    try:
        h = json.loads(head.decode(), parse_constant=_no_constant) if nl else None
    except ValueError:
        h = None
    if not isinstance(h, dict):
        return "the proof's public file has no JSON header"
    n = h.get("instances")
    if not _int(n):
        return "the proof's header has no instance count"
    if n != 1:
        return f"the proof has {n} instances, not one tile"
    f = h.get("frame_v3")
    s = f.get("schemas") if isinstance(f, dict) else None
    if not isinstance(s, dict) or [s[k] for k in sorted(s)[:2]] != [schema for _, schema in OPENED]:
        return "the proof's first rows are not an A row and a B row of the registered schemas"
    if len(body) < 2 * BC_BYTES:
        return "the proof's body holds no A row and B row"
    for i, (side, schema) in enumerate(OPENED):
        o = sides[side]
        if o["index"] != at[side]:
            return f"side {side}: the opening is of row {o['index']}, not the unit's row {at[side]}"
        sdom = Wd.side_domain(dom.domain_id, side)
        bc = body[i * BC_BYTES:(i + 1) * BC_BYTES]
        if not (at[side] < sdom.count and verify_path(sdom, entry.root_a if side == "a" else entry.root_b, at[side],
                                                      row_leaf(sdom, at[side], Wd._HM.tree_leaf(Wd._HM.default_key, bc),
                                                               schema), paths[side])):
            return f"side {side}: the proof's committed row is not entry {e}'s row {at[side]}"
    return None


# ---- the pass's epoch and the window's draw --------------------------------------------------------------------------------

MANIFEST_SCHEMA = "verity-vllm/pouw-pearl-c-run/v2"
#: a pass recorded before epoch coins, salted from a beacon: only a replay of a recorded pass reads it
MANIFEST_RECORDED = "verity-vllm/pouw-pearl-c-run/v1"
#: the e2e-service passes' coin commitment (r20261006-19… to -22…), whose binding and weights' root are the first 32 bytes
#: of `tagged`: only a replay of a recorded pass reads it
COINS_RECORDED = "verity/pouw/epoch-coins/v2"
#: a coin commitment's scheme: (the bytes of `tagged` its binding and the weights' root keep, the weights' root's tag)
COIN_VERSIONS = {Co.SCHEME: (64, "verity/pouw/weights/v1"), COINS_RECORDED: (32, "verity/pouw/weights/v2")}
#: the e2e-service's first coin commitments (r20261006-08…, -19… to -21…), named `coins.SCHEME` but on SHA-256 as core
#: was then: bound by `tagged_sha256`, and the draw's key and stream `derive_v1`'s.  Only a replay of a recorded draw
#: reads it, after the current binding (none of their passes was retained, so the pass's rule doesn't)
COIN_EARLY = Co.SCHEME
DRAW_DOMAIN = "verity/pouw/served-zk-draw/v1"
_OPENING = "a coin opening is {epoch, label, after, coin, salt, path}: an integer, a string, hex, and SHA-512 digests"
_MATMULS = "the pass's matmuls are not a list of objects with integer index, m, k, n and weight"
_WEIGHTS = "the pass's weights are not an object from each weight, in decimal, to an object with a hex root"


def _hex(s: Any) -> bytes | None:
    if not _str(s):
        return None
    try:
        return bytes.fromhex(s)
    except ValueError:
        return None


def coins_of(j: Any, recorded: bool) -> tuple[str, int, bytes]:
    """A coin commitment ``{"scheme", "epoch", "root"}`` (`CoinCommitment.from_json`) as (scheme, epoch, root), of a scheme
    the verifier reads: `coins.SCHEME`, and `COINS_RECORDED` only in a replay of a recorded pass."""
    if not isinstance(j, dict) or j.get("scheme") not in ((Co.SCHEME, COINS_RECORDED) if recorded else (Co.SCHEME,)):
        raise Refused(f"a coin commitment's scheme is {Co.SCHEME!r}")
    if not (_int(j.get("epoch")) and j["epoch"] >= 0):
        raise Refused("an epoch is a nonnegative integer")
    root = _hex(j.get("root"))
    if root is None or len(root) != 64:
        raise Refused("a coin commitment's root is 64 bytes")
    return j["scheme"], j["epoch"], root


def opening_of(j: Any) -> Co.CoinOpening:
    """A coin's opening (`CoinOpening.from_json`), its path of SHA-512 digests."""
    if not isinstance(j, dict) or not _int(j.get("epoch")) or not _str(j.get("label")):
        raise Refused(_OPENING)
    hexes = [_hex(j.get(k)) for k in ("after", "coin", "salt")]
    path = [_hex(p) for p in j["path"]] if isinstance(j.get("path"), list) else None
    if None in hexes or path is None or any(p is None or len(p) != 64 for p in path):
        raise Refused(_OPENING)
    return Co.CoinOpening(j["epoch"], j["label"], *hexes, tuple(path))


def tagged_sha256(tag: str, *parts: Any) -> bytes:
    """`protocol.tagged` as it was on SHA-256 (`COIN_EARLY`'s binding)."""
    return hashlib.sha256(tag.encode() + b"\0" + b"".join(frame(p) for p in parts)).digest()


def derive_v1(source: bytes, domain: str, context: dict) -> bytes:
    """`randomness.derive` as it was (``verity/randomness/derive/v1``, SHA-256): `COIN_EARLY`'s draw key."""
    ctx = b"".join(frame(k) + frame(context[k]) for k in sorted(context))
    return hashlib.sha256(b"verity/randomness/derive/v1\0" + frame(source) + frame(domain) + frame(ctx)).digest()


def stream_v1(key: bytes, index: str, n: int) -> bytes:
    """`Key.stream` as it was (``verity/randomness/stream/v1``, SHA-256 blocks): `COIN_EARLY`'s draw stream."""
    head = b"verity/randomness/stream/v1\0" + key + frame(frame(index))
    return b"".join(hashlib.sha256(head + c.to_bytes(8, "big")).digest() for c in range((n + 31) // 32))[:n]


def coin_version(c: tuple[str, int, bytes], o: Co.CoinOpening, label: str, after: bytes, early: bool = False
                 ) -> tuple[str | None, bool]:
    """`coins.coin_reason` under commitment `c`'s scheme (`COIN_VERSIONS`), and whether the coin opened under
    `COIN_EARLY`'s binding, which `early` also tries when the current one doesn't open."""
    scheme, epoch, root = c
    if o.epoch != epoch:
        return f"coin is epoch {o.epoch}'s, not epoch {epoch}'s", False
    if o.label != label:
        return f"coin is the epoch's {o.label!r} coin, not its {label!r} coin", False
    if len(o.coin) != Co.COIN_BYTES or len(o.salt) != Co.HM96.salt_bytes:
        return "coin opening is malformed", False

    def opens(binding: bytes) -> bool:
        return verify_opening(CommitmentDomain(binding, -1, RangeIndexedDomain(len(Co.LABELS)), hash="sha512"),
                              Commitment(root, len(Co.LABELS)),
                              Opening(Co.LABELS.index(label), Co.leaf_value(o.coin, o.salt), o.path), scheme,
                              VerificationLimits())
    if opens(tagged(scheme, epoch)[:COIN_VERSIONS[scheme][0]]):
        old = False
    elif early and scheme == COIN_EARLY and opens(tagged_sha256(scheme, epoch)):
        old = True
    else:
        return "coin does not open against the epoch's coin commitment", False
    if o.after != after:
        return "coin was not opened after what the verifier recorded", old
    return None, old


def coin_reason(c: tuple[str, int, bytes], o: Co.CoinOpening, label: str, after: bytes) -> str | None:
    """`coins.coin_reason` under commitment `c`'s scheme (`COIN_VERSIONS`)."""
    return coin_version(c, o, label, after)[0]


def weights_root(scheme: str, roots: dict[int, bytes]) -> bytes:
    """`audit.weights_root` under a coin commitment's scheme (`COIN_VERSIONS`)."""
    n, tag = COIN_VERSIONS[scheme]
    return tagged(tag, *(p for w in sorted(roots) for p in (w, roots[w])))[:n]


def _manifest(raw: bytes) -> dict:
    try:
        m = json.loads(raw.decode(), parse_constant=_no_constant)
    except ValueError:
        m = None
    if not isinstance(m, dict):
        raise Refused("the pass's manifest is not a JSON object")
    return m


def _workload(m: dict) -> tuple[list[dict], dict[int, bytes]]:
    """The manifest's matmuls (each a shape and 64-bit ids, `Matmul`'s) and its weights' roots."""
    us = m.get("matmuls")
    if not isinstance(us, list) or not all(isinstance(u, dict) and all(_int(u.get(k)) for k in ("index", "m", "k", "n", "weight"))
                                           for u in us):
        raise Refused(_MATMULS)
    for u in us:
        if min(u["m"], u["k"], u["n"]) < 1:
            raise Co.DomainError("a shape has m, k, n >= 1")
        if not (0 <= u["index"] < U64 and 0 <= u["weight"] < U64):
            raise Co.DomainError("a matmul's index and weight id are 64-bit")
    ws = m.get("weights")
    if not isinstance(ws, dict):
        raise Refused(_WEIGHTS)
    roots: dict[int, bytes] = {}
    for w, v in ws.items():
        r = _hex(v.get("root")) if isinstance(v, dict) else None
        if r is None or not (w and all(c in "0123456789" for c in w) and (len(w) == 1 or w[0] != "0")):
            raise Refused(_WEIGHTS)
        roots[int(w)] = r
    return us, roots


def _check_workload(us: list[dict], roots: dict[int, bytes]) -> None:
    """`Workload`'s and `Epoch`'s checks of the weights."""
    if len({u["index"] for u in us}) != len(us):
        raise Co.DomainError("matmul indices are unique")
    dims: dict[int, tuple[int, int]] = {}
    for u in us:
        if dims.setdefault(u["weight"], (u["n"], u["k"])) != (u["n"], u["k"]):
            raise Co.DomainError(f"every matmul on weight {u['weight']} has the weight's (n, k)")
    if set(roots) != set(dims):
        raise Co.DomainError("every weight of the workload is committed, and only those")


def _epoch_salt(m: dict, coins: Any, recorded: bool) -> tuple[bytes, str]:
    """`manifest_epoch`: the pass's epoch salt re-started from its manifest, and how it was had.  Raises `DomainError`
    (the coin rule's) or `Refused` (a manifest it can't read)."""
    schema = m.get("schema")
    if schema == MANIFEST_RECORDED and not recorded:
        raise Co.DomainError(f"the pass's manifest is {schema}, recorded before epoch coins: only a replay of a recorded "
                             "pass reads it")
    if schema not in (MANIFEST_SCHEMA, MANIFEST_RECORDED):
        if not _str(schema):
            raise Refused("the pass's manifest names no schema")
        raise Co.DomainError(f"schema {schema!r}, not {MANIFEST_SCHEMA}" + (f" or {MANIFEST_RECORDED}" if recorded else ""))
    us, roots = _workload(m)
    if schema == MANIFEST_SCHEMA:
        if "salt_opening" not in m:
            raise Co.DomainError("the pass has no salt coin")
        if coins is None:
            raise Co.DomainError("the pass's salt coin opens against the epoch's published coin commitment, which the "
                                 "verifier holds")
        try:
            c = coins_of(coins, recorded)
        except Refused as e:
            raise Co.DomainError(f"the verifier's coins are malformed: {e}") from None
        try:
            opening = opening_of(m["salt_opening"])
            claimed = coins_of(m.get("coins"), recorded)
        except Refused as e:
            raise Co.DomainError(f"the pass's coins are malformed: {e}") from None
        if claimed != c:
            raise Co.DomainError("the pass's coin commitment is not the epoch's")
        _check_workload(us, roots)
        root = weights_root(c[0], roots)
        why = coin_reason(c, opening, "salt", root)
        if why is not None:
            raise Co.DomainError(f"the salt's {why}")
        salt, how = derive(opening.coin, EPOCH_SALT_DOMAIN, {"epoch": c[1], "coins": c[2], "weights": root}).stream("salt", 32), "coin"
    elif "beacon" in m:
        beacon = _hex(m["beacon"])
        if not beacon:
            raise Refused("the recorded pass's beacon is not nonempty hex")
        _check_workload(us, roots)
        salt = derive(beacon, SALT_DOMAIN, {"weights": weights_root(Co.SCHEME, roots)}).stream("salt", 32)
        how = "recorded beacon"
    else:
        raise Co.DomainError("the recorded pass has no beacon")
    if salt.hex() != m.get("salt"):
        raise Co.DomainError(f"the pass's salt is not its {how}'s and the weights'")
    return salt, how


def pass_epoch(coins: Any, raw: bytes | None, context: str, recorded: bool = False) -> tuple[str | None, tuple[str, str] | None]:
    """How the pass's salt was had (``coin``, or ``recorded beacon`` in a replay of a recorded pass) and None, or None and
    the refusal (``coin`` or ``manifest``, and why): its manifest `raw` re-starts its epoch under `coins`, the challenger's
    own coin commitment (its JSON), and the window's `context` names it by its SHA-512."""
    if raw is None:
        return None, ("coin", "the window names no pass, whose epoch the challenger re-starts from its manifest")
    try:
        _, how = _epoch_salt(_manifest(raw), coins, recorded)
    except Co.DomainError as e:
        return None, ("coin", str(e))
    except Refused as e:
        return None, ("manifest", str(e))
    if hashlib.sha512(raw).hexdigest() != context.rsplit("/", 1)[-1]:
        return None, ("manifest", "the pass is not the one the window's parameters name")
    return how, None


def width(n: int) -> int:
    """Bytes per word of an exact uniform draw below `n` (Lean's ``Draw.width``, `Key.uniform`'s)."""
    return (n.bit_length() + 64 + 7) // 8


def _uniform(n: int, src: bytes, pos: int) -> tuple[int, int] | None:
    w = width(n)
    space = 1 << (8 * w)
    while pos + w <= len(src):
        x, pos = int.from_bytes(src[pos:pos + w], "big"), pos + w
        if x < space - space % n:
            return x % n, pos
    return None


def stream_subset(k: int, n: int, src: bytes) -> list[int] | None:
    """Lean's ``Draw.subset`` on one stream (``flock-verify draw --stream``): partial Fisher-Yates, each step an exact
    uniform word of `width` bytes; None when the stream runs out."""
    swapped: dict[int, int] = {}
    pos = 0
    for i in range(k):
        r = _uniform(n - i, src, pos)
        if r is None:
            return None
        j, pos = i + r[0], r[1]
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _plain(s: Any) -> bool:
    return _str(s) and all(" " <= c <= "~" and c not in '"\\' for c in s)


def draw_key_context(stream: str, registration: str, receipt: dict) -> dict:
    """The draw key's context: the stream, the registration and its receipt (as canonical JSON's SHA-512)."""
    if not all(_plain(x) and _plain(v) for x, v in receipt.items()):
        raise ValueError("a receipt is an object of printable ASCII strings")
    body = "{" + ",".join(f'"{k}":"{receipt[k]}"' for k in sorted(receipt)) + "}"
    return {"stream": stream, "registration": registration, "receipt": hashlib.sha512(body.encode()).hexdigest()}


def draw(coins: Any, recorded: bool, opening: Any, stream: Any, registration: Any, receipt: Any, k: Any, population: Any,
         units: Any) -> tuple[str, str] | None:
    """None when `units` are the window's draw: the epoch's draw coin `opening` opens against `coins` (the challenger's
    commitment, its JSON) after the registration `registration` (its digest's hex), and `units` are ``Draw.subset`` of
    `k` over `population` on the stream ``"draw"`` of the key the coin derives under the stream's digest `stream`, the
    registration and its `receipt`; else the refusal (``coin``, ``draw`` or ``sampler``, and why)."""
    try:
        c = coins_of(coins, recorded)
    except Refused as e:
        return "coin", f"the verifier's coins are malformed: {e}"
    if not (_str(registration) and _DIGEST.fullmatch(registration)) or not _str(stream):
        return "draw", "the registration is not a SHA-512 digest, or the stream's digest not a string"
    if not (isinstance(receipt, dict) and all(_plain(x) and _plain(v) for x, v in receipt.items())):
        return "draw", "the receipt is not an object of printable ASCII strings"
    if not (_nat(k) and _nat(population) and 1 <= population and k <= population):
        return "draw", "the law is not subset:k over N with k <= N and N >= 1"
    try:
        o = opening_of(opening)
    except Refused as e:
        return "coin", f"the draw's {e}"
    why, old = coin_version(c, o, "draw", bytes.fromhex(registration), early=recorded)
    if why is not None:
        return "coin", f"the draw's {why}"
    ctx, n = draw_key_context(stream, registration, receipt), 2 * k * width(population) + 64
    src = stream_v1(derive_v1(o.coin, DRAW_DOMAIN, ctx), "draw", n) if old else derive(o.coin, DRAW_DOMAIN, ctx).stream("draw", n)
    got = stream_subset(k, population, src)
    if got is None:
        return "sampler", "the draw's stream ran out"
    if not (isinstance(units, list) and all(_nat(u) for u in units) and units == got):
        return "draw", f"the units are not the draw coin's subset:{k} over {population}"
    return None


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
           "case", "check", "commitment", "digest", "draw", "evidence", "integrity", "integrity_json", "law_name", "line",
           "opening", "params", "pass_epoch", "ratio", "registered", "unit_draw", "verify_path", "window_domain"]
