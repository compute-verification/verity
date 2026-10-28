"""PoUW's check of one audit window as a core protocol (`verity/core/protocols/interface`): the Python
reference beside the Lean `pouw_check` (`verity/core/protocols/pouw/lean/check`), the served window's verifier of record
(`benchmarks/pouw/served_zk/of_record.py`), held to it answer for answer by the difftest.

The proof system registers the window, draws, proves and verifies; PoUW states what it registers and what the outcome
must show, and publishes `window.PUBLIC` from it.  This is that composition in one function, the parts of
`benchmarks/pouw/served_zk/served_commit.py` (`public_problem`, `register` and `Challenger.outcome`) that are PoUW's.

* **Params**, canonical JSON: the check item (`check`: the tile template, its work per tile w, the commitment scheme,
  the public bound on entries and optionally its coin mode, ``coins``, live or seeded: `window.mode`), ε and δ as ratio
  strings, the grid's bits, the declared range of tile counts, the window's context, the check query's SHA-512
  (one-stage's `query_digest` of the tile template's instance query) and the rows' fixed length
  ``{"stop": "fixed", "max_tokens": [...]}``, each slot's max_tokens.  The template is one the verifier pins (`TILES`,
  Lean's `PouwCheck.tiles`), with that template's tile work and query.  A live window's parameters (``coins`` live)
  also hold ``live``, the inputs of its salt session as the verifier holds them (`LIVE`), and no other window's do.
* **Commitment**: the window's registration, one-stage's record (`registration.record`) as canonical JSON, its window
  ``{"kind": "served", "context", "public"}`` holding exactly `window.REGISTERED`, and optionally ``activations``, the
  per-window values it committed after serving, and ``registered``, the registered weights' entries
  (`WINDOW_VALUES`, each ``{program, values}``, `values_root`).  Its digest is the record's own, `registration.digest`,
  the one the service's receipt, draw and outcome cite.
* **Evidence**: the service's answer for that registration, read as `Core.Answer.ofJson` and `Core.Integrity.ofJson`
  read it (only what `toJson` writes, in the shapes `answer` returns): its verdict, the registration digest it is for,
  and on acceptance the integrity profile and the public outputs ``{"draw": d}``, `d` the verifier's draw as one-stage's
  `draw.py` writes it (any of its laws; PoUW's rule then holds it to subset:K′), and a live window's ``salt``, its salt
  coin and the salt it gives (`window_salt`).
* **The check** accepts when the registration is PoUW's (`window.REGISTERED`, its `length` item a fixed-length
  window's at the verifier's max_tokens (`window.length_problem`), each item the verifier's own
  at the registered level, N = point(level + 1) in the declared range, the law subset:K′ with K′ = `window.k_prime`, the check's
  query and scheme), the service accepted this registration, and its profile and draw are of the registered partition's N
  units drawn subset:K′ at the window's δ, and a live window's salt is its live coin's (`window_salt`: taken from a
  salt-only session the auditor's ledger opened on `salt.anchored_epoch_root` of the registered weights' root W, the
  root X of the activations the window committed before its salt, and the answer's epoch, after that root, as a live
  Pearl-C pass's is; a seeded window's answer carries no salt).  It publishes
  exactly `window.PUBLIC` and the service's profile.  The service's reject is a reject and its undecided (the verifier
  of record hasn't run) is undecided.

`opening` is the rule on one unit's opening beside it (`served_commit.check`'s, which calls it): the tile proof's public
file opens the unit's rows under the registered window root, through its entry and the entry's side trees.

`pass_epoch` is the rule on the pass a window is received from (`served_commit.Challenger.pass_epoch`, which calls it):
its manifest re-starts its epoch under the challenger's own coin commitment, and the window's context names it.  In a
certified window, whose verifier holds the network certifier's reveal for it (`salt.revealed`), the pass's salt must be
`salt.pouw_salt` of that reveal and the PoUW window the context names, and its weights those the certifier recorded
before it drew that salt (`salt.committed`).  `recorded_reason` is that last rule alone, which needs no reveal: the
challenger runs it before it asks the certifier for one, since the certifier reveals a window once.  A live Pearl-C
pass (`pouw_pearl_c_host.epoch_of`), whose manifest's coins are a live session's, took its salt coin from a
salt-only session of the auditor host's challenger opened on its salt epoch's root (`salt.epoch_open_root` of its
weights' root and its ``pass_index``): its verifier holds the auditor's ledger and that session's files as `rounds`
reads them (`LIVE`), and the pass passes `live_salt` under them.
`draw` is the rule on the window's draw (`Challenger.select`'s): the epoch's draw coin opens after the registration,
and the drawn units are Lean's ``Draw.subset`` of the key it derives.  Both read the coin versions in `COIN_VERSIONS`,
and a replay of a recorded draw `COIN_EARLY` too.  `work_units` is the same rule under the law ``work:k``: each work
stratum's ``Draw.subset`` on its own stream of that key (`credit.drawn` draws a served H1 window with it).
`salt` is the rule on a live window's salt coin, read through the
same session as `rounds`: S is the session the auditor's signed ledger reserves for the window and R the root its open
entry opened S on, which the verifier recomputes from the window's record and S (`salt.window_open_root`); the
session's signed log shows R alone as what its salt coin followed, and under the commitment it signed the salt coin
opened once, after R.  `live_salt` is its Pearl-C form (LIVESALT2, `Properties.Pouw.LiveSalt`): R is the pass's salt
epoch's root, `salt.epoch_open_root` of its weights' root W and its ``pass_index`` k, the session is reserved for the salt
coin alone (`window.SALT_FAMILIES`) and counts only when the ledger ends it complete, and the pass's coins, salt coin,
salt and session are held to what the ledger and the session's files show.
`rounds` is the rule on a live window's round coins (`window.rounds_reason`, over the proof system's challenger
`sampled_proofs.challenger`): it reads the challenger's coins only through the session the auditor's signed ledger
opens and ends for the window, under the window's pinned key, run from a commit the host's pins log records on main's
first-parent line (`challenger.trusted_session_of`, over `challenger.pinned_session`), never a commitment the prover
hands in; under the commitment that session signed, round r's coin opened once, after the r-th round root, and the
round roots are the ones its signed log shows it received.  `salt` reads its session through the same
`pinned_session`, so a window whose session ran any tree but main's is credited neither.

Every refusal has one reason, the first rule it breaks in the order below; the Lean port writes the same strings.
"""
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
import re
from fractions import Fraction
from typing import Any, NamedTuple

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.core.protocols.profile import Work, WorkStratum
from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN, SALT_DOMAIN
from verity.core.protocols.pouw.protocol import U64, fold_roots, tagged
from verity.ml.commitments import Commitment, Opening, VerificationLimits, domains, verify_opening
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest
from verity.ml.commitments.scheme import DIGEST_BYTES
from verity.ml.crypto import lms
from verity.ml.randomness import derive, frame

ID = domains.POUW_PROTOCOL
ONE_STAGE = domains.ONE_STAGE_PROTOCOL
SCHEME = "frame-v3-sha512/hm96-sha512"
FORMATS = (domains.REGISTRATION_V1, domains.REGISTRATION)
LEVEL = "unit"
PARAMS_KEYS = ("check", "context", "delta", "eps", "grid_bits", "length", "query", "range")
LENGTH_PARAM_KEYS = ("max_tokens", "stop")
CHECK_KEYS = ("entries", "scheme", "template", "tile_work")
ANSWER_KEYS = ("commitment", "profile", "protocol", "public", "reason", "verdict")
VERDICTS = ("accept", "reject", "undecided")
_DIGEST = re.compile(r"[0-9a-f]{128}")
#: the window's optional groups of committed values (`values_root`): the activations it committed after serving, before
#: its salt (X), and the registered weights' entries (W, where it registers no ``W`` root)
WINDOW_VALUES = ("activations", "registered")
#: `values_root`'s domain tag
VALUES_TAG = domains.POUW_PERIOD_VALUES.encode()
#: one-stage's window domain (`registration.window_domain`), the window root's
SERVED_DOMAIN = domains.ONE_STAGE_SERVED_DOMAIN
#: a tile proof's first two rows: its A row and its B row, each side's fp8-p row schema
#: (`circuit.leaves.ROW_SCHEMA["fp8-p", X | W]`)
OPENED = (("a", domains.HM96_SHA512_ROW_SEG), ("b", domains.HM96_SHA512_ROW))
#: the ports of those rows, the pinned circuit's first two (`served_zk/stage_tile.py`'s ``in_rows``)
PC8_PORTS = ("a0", "b0")
SALT_BYTES = 192
BC_BYTES = 128
ROW_V2 = domains.HM96_SHA512_ROW
#: a salted unit's public row: the window's certified salt, the unit's position (a u64) and its entry's side roots
PUBLIC_ROW_BITS = 8 * (PS.SALT_BYTES + 8 + 2 * DIGEST_BYTES["sha512"])
#: a J10-R unit's public row (`h1tile.PouwH1Tile5`'s): `PUBLIC_ROW_BITS`, then its call's first position ``call`` and its
#: ``blocks``, its first row block and its first column block in its call's stream, each a u64
PUBLIC_ROW_J10R_BITS = PUBLIC_ROW_BITS + 8 * 3 * 8
PUBLIC_ROW_SALT_DOMAIN = domains.POUW_PUBLIC_ROW_SALT.encode() + b"\0"


class Salted(NamedTuple):
    """A salted tile template's unit: R A rows and C B rows, then its public row ``pub``, the pinned circuit's first
    R + C + 1 ports in that order, each an ``hm96-sha512/row/v2`` row (`PouwH1Tile`'s ``a0..``, ``b0..``, ``pub``)."""
    r: int
    c: int


class Tile(NamedTuple):
    tile_work: int
    query: str
    circuit: str
    salted: Salted | None = None


#: the tile templates the verifier pins (Lean's `PouwCheck.tiles`, of record): each one's work per tile, its check query's
#: SHA-512, and the SHA-512 of the one circuit a unit's tile proof of it is over (its public file's ``circuit_sha512``)
TILES = {
    "Pc8TileSide_v1{K=4096,TM=1,TN=1,DEV='sm120'}": Tile(
        4096,
        "491c9160f3c0d0c95ef14575e0966bfa9958de58886c3cc46e8008c767118926582acc2199a49dae85a32656cd2b11f90bffc1105aa9d660e98f66f75ccac4af",
        "39d540c10ed1148526fcdf9d01561fcf68e33427f28f55d2dc2f453c09f68696978a530b2c14054f4720af1710afee8c74331f45b4c9a7c0511bfc1e02017c12"),
    "Pc8TileSide_v1{K=128,TM=1,TN=1,DEV='sm120'}": Tile(
        128,
        "63cd98cb86598a2d67169d1a5ea949957c941b3e11aad61aeb14e99fc1a2d7c7f5b7b19e7349bcae68d0ee7400cfbe89b8a6dd20f14319666eb7a7d81a0a6343",
        "84315c5940d5c61a4938182d41330c2f4febbf049ca9cdadd58ec9f1f9dd92a13d9e43501b37e9e0f32914e698667907d768ce49c2ba8be026454bdff78baf24"),
    "Pc8TileHidden_v1{K=4096,TM=1,TN=1,DEV='sm120'}": Tile(
        4096,
        "f48d5a1f247cb17063bcea693733db3d26e91be34967645486ea20df1336b4d9ae14aaf242ba4437f0d2a0123aff81f7ab31d18d437f4abe73363a1ac7bf65fb",
        "5ec6002f654337db2051fd6a98b7ec122a9f33e2d06180012177195674233d7c811c2ffd8ef49bc9a586dcf8e61ee1b16914b9f36e1bd378f9e53d0d40f4926d"),
    "Pc8TileLeafH3_v1{K=4096,DEV='sm120'}": Tile(
        4096,
        "0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647",
        "c4fb8b6c5ce91ad0bc750f315700491911f678ad219cd370baf4359dc751954c9955590eb50304a166e1cc103ffd0d431de31f03c29ad5b6d119a33502b004af"),
    "PouwH1Tile_v1{EXACT=GemmAccumulatorS8_v2{K=64,DOT=BlackwellS8ImmaDot32_v2},R=4,C=8,PAIRS=((0,0),(0,1),(0,2),(0,3),"
    "(0,4),(0,5),(0,6),(0,7),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),"
    "(2,7),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7))}": Tile(
        64 * 32,
        "1083a2eb9d13f77afdb936ecac9e780aa63b7f993795a8ff59d9d05ca7259c455c1220a833979cd975f907e97d9045033c5427950b7953dd33ac26cfdef44631",
        "571e56a77368c64505e46a2affd6360d869cf17d272f86926bbb6d8974f2c39177b351e15c20bbf8fd78cbb87a4951f9e3a1062edab8e4a57905eb99e6074b2f",
        Salted(4, 8)),
}


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
    """The verifier's own parameters of the window: a live window's (``check.coins`` live) also hold ``live``, its salt
    session's inputs as `LIVE` names them, and no other window's do."""
    if not isinstance(j, dict) or sorted(set(j) - {LIVE_PARAM}) != list(PARAMS_KEYS):
        raise Refused(f"keys are not {', '.join(PARAMS_KEYS)}, and optionally {LIVE_PARAM}")
    c = j["check"]
    if not isinstance(c, dict) or sorted(set(c) - {Wd.COINS}) != list(CHECK_KEYS):
        raise Refused(f"check's keys are not {', '.join(CHECK_KEYS)}, and optionally {Wd.COINS}")
    if Wd.COINS in c and c[Wd.COINS] not in Wd.MODES:
        raise Refused(f"check.{Wd.COINS} is not {' or '.join(Wd.MODES)}")
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
    t = TILES.get(c["template"])
    if t is None:
        raise Refused("check.template is not a tile template the verifier pins")
    if c["tile_work"] != t.tile_work:
        raise Refused("check.tile_work is not its pinned template's tile work")
    if j["query"] != t.query:
        raise Refused("query is not its pinned template's query digest")
    ln = j["length"]
    m = ln.get("max_tokens") if isinstance(ln, dict) else None
    if not (isinstance(ln, dict) and sorted(ln) == list(LENGTH_PARAM_KEYS) and ln["stop"] == Wd.STOP_FIXED
            and isinstance(m, list) and m and all(_nat(x) and x >= 1 for x in m)):
        raise Refused("length is not {max_tokens, stop} with stop fixed and max_tokens a non-empty list of positive natural "
                      "numbers")
    live = j.get(LIVE_PARAM)
    if (Wd.mode(c) == "live") != (LIVE_PARAM in j) or (LIVE_PARAM in j and not (isinstance(live, dict)
                                                                                and sorted(live) == list(LIVE))):
        raise Refused(LIVE_PARAMS)
    return {**out, "grid_bits": j["grid_bits"], "query": j["query"], "range": tuple(rg), "max_tokens": list(m),
            "live": live}


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
    if not (isinstance(w, dict) and sorted(set(w) - set(WINDOW_VALUES)) == ["context", "kind", "public"]
            and _str(w["context"]) and _str(w["kind"]) and isinstance(w["public"], dict)):
        raise Refused("window is not {context, kind, public}, with strings and an object, and optionally activations "
                      "and registered")
    return {"bytes": b, "format": rec["format"], "record": rec}


def digest(c: dict) -> bytes:
    """`registration.record_digest` over the record's own bytes: SHA-512(format || 0 || the canonical record)."""
    return R.record_digest(c["format"], c["bytes"])


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
    """`Core.SamplingDesign.ofJson`, as `SamplingDesign.toJson` writes it."""
    if _has(j, "bernoulli"):
        d: dict = {"bernoulli": str(ratio(_str_at(j, "bernoulli", ctx), f"{ctx}.bernoulli"))}
    elif _has(j, "subset"):
        d = {"subset": _nat_at(j, "subset", ctx)}
    elif _has(j, "strata") and isinstance(j, dict) and j.get("law") == "work":
        k, c = _nat_at(j, "k", ctx), f"{ctx}.strata"
        strata = [WorkStratum(_str_at(s, "name", c), _nat_at(s, "size", c), _nat_at(s, "k", c), _nat_at(s, "work", c),
                              _nat_at(s, "floor", c)) for s in _arr_at(j, "strata", ctx)]
        try:
            d = Work(k, tuple(strata)).to_json()
        except ValueError as e:
            raise Refused(f"{ctx}: {e}") from None
    elif _has(j, "strata"):
        c = f"{ctx}.strata"
        d = {"law": "stratified", "strata": [{"name": _str_at(s, "name", c), "size": _nat_at(s, "size", c),
                                              "k": _nat_at(s, "k", c)} for s in _arr_at(j, "strata", ctx)]}
    else:
        raise Refused(f"{ctx}: not a bernoulli, subset, stratified or work draw")
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


def _num_den(p: str, ctx: str) -> str:
    """`pOf`: ``num/den``, each a natural number in canonical decimal (`Flock.Draw.Law.bernoulli`'s), kept as written."""
    parts = p.split("/")
    if len(parts) == 2 and all(t and all(c in "0123456789" for c in t) and str(int(t)) == t for t in parts):
        return p
    raise Refused(f"{ctx}.p: not num/den")


def unit_draw(j: Any) -> dict:
    """The public outputs ``{"draw": d}``, `d` one of one-stage's `draw.py` laws with exactly its keys (`check_draw`'s),
    and a live window's ``salt``, an object (`window_salt` reads it)."""
    if not (isinstance(j, dict) and sorted(j) in (["draw"], ["draw", "salt"])):
        raise Refused("public: keys are not draw, and optionally salt")
    d, ctx = j["draw"], "public.draw"
    law = _str_at(d, "law", ctx)
    if law not in _DRAW_KEYS:
        raise Refused(f"{ctx}.law: not subset, bernoulli, stratified or work")
    _keys_are(d, (["closure"] if law == "work" and "closure" in d else []) + _DRAW_KEYS[law], ctx)
    out: dict[str, Any] = {"law": law, "population": _nat_at(d, "population", ctx)}
    if law == "bernoulli":
        out["p"] = _num_den(_str_at(d, "p", ctx), ctx)
    else:
        out["k"] = _nat_at(d, "k", ctx)
    if law in ("stratified", "work"):
        out["strata"] = _strata(law == "work", d, ctx)
    if "closure" in d:
        out["closure"] = _nats_at(d, "closure", ctx)
    out["units"] = _nats_at(d, "units", ctx)
    if "salt" in j and not isinstance(j["salt"], dict):
        raise Refused("public.salt: not an object")
    return out


def law_name(d: dict) -> str:
    """`parse_law`'s spelling of a draw's law."""
    return f"bernoulli:{d['p']}" if d["law"] == "bernoulli" else f"{d['law']}:{d['k']}"


def evidence(b: bytes) -> dict:
    """The service's answer for the registration (`answer_of`), and on acceptance its profile and draw."""
    a = answer_of(canonical(b))
    a["salt"] = None
    if a["verdict"] == "accept":
        a["profile"] = integrity(a["profile"])
        a["draw"] = unit_draw(a["public"])
        a["salt"] = a.pop("public").get("salt")
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
        if sorted(items) == sorted(Wd.LEGACY_REGISTERED):
            return "window-form: no-fixed-length: registered before the fixed-length rule (19 Oct), so not of record", None
        return f"public-items: registered {', '.join(sorted(items))}, declared {', '.join(sorted(Wd.REGISTERED))}", None
    why = Wd.length_problem(items["length"], p["max_tokens"])
    if why is not None:
        return why, None
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
    why = window_salt(p, rec, e["salt"])
    if why is not None:
        return "reject", f"salt: {why}", None
    public = {**{i: items[i] for i in Wd.REGISTERED}, "verdict": True, "drawn": list(units)}
    return "accept", None, (public, integrity_json(prof))


# ---- a unit's opening ------------------------------------------------------------------------------------------------------

def window_domain(p: dict, rec: dict) -> CommitmentDomain:
    """The window root's domain: one-stage's window domain (`registration.window_domain`) of the registered program and
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
    sibs = [_fromhex(s, DIGEST_BYTES["sha512"]) for s in x] if isinstance(x, list) else None
    return None if sibs is None or None in sibs else sibs


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _no_constant(s: str) -> Any:
    raise ValueError(f"{s} is not JSON")


def public_row(salt: bytes, position: int, roots: tuple[bytes, bytes],
               j10r: tuple[int, tuple[int, int]] | None = None) -> bytes:
    """A salted unit's public row, its `PUBLIC_ROW_BITS` bits as `ml.circuits.seeds` lays them out (`h1unit.pub_words`):
    `salt`, `position` as a u64 and the two roots, each 32 bits a u32 low bit first. With `j10r` ``(call, blocks)``, a
    J10-R unit's (`PUBLIC_ROW_J10R_BITS`): then its call's first position and its two blocks (`j10r_blocks`), each a u64
    in the same layout."""
    x = salt + position.to_bytes(8, "big") + roots[0] + roots[1]
    if j10r is not None:
        x += b"".join(v.to_bytes(8, "big") for v in (j10r[0], *j10r[1]))
    if len(x) * 8 != (PUBLIC_ROW_BITS if j10r is None else PUBLIC_ROW_J10R_BITS):
        raise ValueError(f"a public row is {PUBLIC_ROW_BITS} bits, a J10-R unit's {PUBLIC_ROW_J10R_BITS}")
    return b"".join(x[i:i + 4][::-1] for i in range(0, len(x), 4))


def public_words(salt: bytes, position: int, roots: tuple[bytes, bytes],
                 j10r: tuple[int, tuple[int, int]] | None = None) -> list[int]:
    """`public_row` as a ``row/v2`` row of 16-bit leaves: word w its bits 16 w .. 16 w + 15, low byte first."""
    x = public_row(salt, position, roots, j10r)
    return [x[i] | x[i + 1] << 8 for i in range(0, len(x), 2)]


def n_digits(k: int) -> int:
    """H1's digit count m at contraction `k` (`segment.n_digits` of KS = k / 32): the least m with 128 KS <= 128^m and
    127 KS // 128^(m - 1) <= 127."""
    ks, m = k // 32, 1
    while not (128 * ks <= 128 ** m and 127 * ks // 128 ** (m - 1) <= 127):
        m += 1
    return m


def j10r_blocks(m: int, k: int, tile: tuple[int, int], i: int, j: int) -> tuple[int, int]:
    """`h1tile.j10r_blocks`: unit (I, J) = (`i`, `j`)'s first row block and first column block of its call's one stream at
    call scale, row block R I at R I RB and column block C J at M RB + C J CB, for a call of slot `m` (its A rows), an
    R × C `tile` and K = `k`: RB = ⌈(K + 32 (m_K - 1)) / 512⌉ a row's blocks and CB = ⌈(K + 256) / 512⌉ a column's
    (`h1tile.Layout`'s). Every input is the window's public structure, so the verifier derives them."""
    r, c = tile
    rb, cb = -(-(k + 32 * (n_digits(k) - 1)) // 512), -(-(k + 256) // 512)
    return r * i * rb, m * rb + c * j * cb


def public_row_salt(context: str, position: int) -> bytes:
    """The public row's hm96 salt, which anyone derives from the window's context and the unit's position: three SHA-512
    blocks of `PUBLIC_ROW_SALT_DOMAIN` ‖ u64be(len) ‖ context ‖ u64be(position) ‖ u8(i)."""
    ctx = context.encode()
    head = PUBLIC_ROW_SALT_DOMAIN + len(ctx).to_bytes(8, "big") + ctx + position.to_bytes(8, "big")
    return b"".join(hashlib.sha512(head + bytes([i])).digest() for i in range(SALT_BYTES // DIGEST_BYTES["sha512"]))


#: the ok row the verifier accepts: a unit's one output bit, 1, in its one word (h1_prove's `OK`)
OK_ROW = (1,)
#: the ok row's hm96 salt tag (h1_prove's)
OK_SALT_TAG = domains.POUW_H1_OK_ROW_SALT.encode() + b"\0"


def ok_salt(context: str, position: int) -> bytes:
    """The ok row's hm96 salt, which anyone derives from the window's context and the unit's position: its content is
    the one the verifier accepts (`OK_ROW`), so the salt hides nothing. SHAKE-256 of `OK_SALT_TAG` ‖ context ‖ 0 ‖
    u64be(position), `SALT_BYTES` bytes."""
    return hashlib.shake_256(OK_SALT_TAG + context.encode() + b"\0" + int(position).to_bytes(8, "big")).digest(SALT_BYTES)


def call_reads(u0: int, rows: list[int | None]) -> dict[str, Any]:
    """A ``--registered`` port's reads (`registered.READS_V1`) for a statement over one call's units from unit `u0`:
    ``reads_from`` `u0` and ``reads`` the row each of the call's units reads, in unit order, trailing nones dropped, so
    the file is as long as the call, not the window (Flock's `Registered.Port.readAt`: none below ``reads_from``)."""
    rows = list(rows)
    while rows and rows[-1] is None:
        rows.pop()
    return {G.READS_FROM: u0, "reads": rows}


def port_reads(entry: dict[str, Any]) -> list[int | None]:
    """The row every unit from unit 0 reads at a ``--registered`` port, of either format (Flock's
    `Registered.Port.reads`): none below its ``reads_from`` (absent: 0), then its ``reads``."""
    return [None] * entry.get(G.READS_FROM, 0) + list(entry["reads"])


def public_bc(context: str, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> bytes:
    """The b ‖ c of unit `position`'s public row (`public_row`) under `public_row_salt`: the 128 bytes the verifier
    requires at the public row's place in the proof's body."""
    return Wd._HM.commit_string(Wd._HM.default_key, sha512_row_v2_digest(public_row(salt, position, roots), PUBLIC_ROW_BITS),
                                public_row_salt(context, position))


def certified_salt(raw: bytes | None, context: str, reveal: bytes, committed: bytes | None, stream: bytes | None,
                   prover: bytes | None) -> bytes:
    """The certified window's salt that `pass_epoch` checks the pass's against (`_certified_salt`), which `opening` holds
    every drawn salted unit's public row to.  Raises `Co.DomainError` or `Refused` where `pass_epoch` refuses."""
    if raw is None:
        raise Co.DomainError("the window names no pass, whose epoch the challenger re-starts from its manifest")
    return _certified_salt(_manifest(raw), reveal, context, committed, stream, prover)[0]


def _salt_reason(p: dict, t: Tile | None, salt: Any) -> str | None:
    """A certified window (its context names a PoUW window, `salt.window_of`) opens only a salted template's units, under
    the 64-byte certified salt the verifier holds; any other window, no salt and no salted unit."""
    if PS.window_of(p["context"]) is not None:
        if t is None or t.salted is None:
            return ("the window is certified, and its template's unit reads no salt: no drawn proof of it is held to the "
                    "window's certified salt")
        if type(salt) is not bytes or len(salt) != PS.SALT_BYTES:
            return f"the window is certified, and the verifier holds no {PS.SALT_BYTES}-byte certified salt for it"
    elif salt is not None:
        return "the window is not certified, so it has no certified salt for a unit to read"
    elif t is not None and t.salted is not None:
        return "the window's template's unit reads a salt, which only a certified window has"
    return None


def _ports(lay: Salted | None) -> list[str]:
    """The side rows a tile proof opens, by port name in its circuit's port order: A rows, then B rows. C-Flock reads
    port k's b ‖ c at the body's k-th place, in the order of the circuit's ports, not of the header's keys."""
    return list(PC8_PORTS) if lay is None else [f"a{k}" for k in range(lay.r)] + [f"b{k}" for k in range(lay.c)]


def _sides(o: Any, n: int) -> list[tuple[int, list[bytes]]] | None:
    """A salted unit's n rows of one side, ``[{"index", "path"}, …]``, or None."""
    if not (isinstance(o, list) and len(o) == n and all(isinstance(x, dict) for x in o)):
        return None
    out = [(x.get("index"), _path(x.get("path"))) for x in o]
    return None if any(not _int(i) or s is None for i, s in out) else out


def opening(p: dict, c: dict, units: list[int] | None, unit: int, pub: bytes, op: Any,
            salt: bytes | None = None) -> str | None:
    """None when the tile proof's public file `pub` opens unit `unit`'s rows under the registered window root, else why
    not (`served_commit.check`).  `units` is the draw's (None for a unit opened undrawn, the filler's); `op` the prover's
    opening ``{"entry", "entry_value", "entry_salt", "window_path", "a": {"index", "path"}, "b": …}``: the unit's entry,
    its salt and its path to the window root, and each side's row with its path to the entry's side root.  The proof's
    A row and B row are its circuit's first two ports, ``a0`` and ``b0`` (`PC8_PORTS`), whose schemas the header names,
    and its body starts with their b ‖ c.  The header's keys, in its key order (serde's), list the opened side rows
    first and in port order (`_ports`): a header that orders them otherwise is refused, never read in its own order.

    `salt` is a certified window's salt (`certified_salt`), and a certified window opens only a salted template's units
    (`Tile.salted`, `_salt_reason`).  A salted unit's opening lists each side's rows, ``"a": [{"index", "path"}, …]`` (R
    rows) and ``"b"`` (C rows): tile (i, j) of its entry (`window.tile_in`) is A rows R·i + k and B rows C·j + k, and
    the proof's body starts with their b ‖ c and then its public row's, which must be `public_bc` of the window's context,
    `salt`, the unit's position and the entry's side roots."""
    if units is not None and unit not in units:
        return f"unit {unit} is not drawn"
    t = TILES.get(p["check"]["template"])
    why = _salt_reason(p, t, salt)
    if why is not None:
        return why
    lay = t.salted if t is not None else None
    rec = c["record"]
    dom = window_domain(p, rec)
    roots = {r["port"]: r for r in rec["roots"]}
    if roots.get(Wd.PORT, {}).get("domain") != dom.domain_id.hex():
        return "the registered window domain is not the window's served domain"
    root = _fromhex(roots[Wd.PORT]["root"], DIGEST_BYTES["sha512"])
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
    esalt, sibs = _fromhex(op.get("entry_salt"), SALT_BYTES), _path(op.get("window_path"))
    if lay is None:
        sides = {s: op.get(s) for s, _ in OPENED}
        paths = {s: _path(o.get("path")) if isinstance(o, dict) else None for s, o in sides.items()}
        opened = None if any(p_ is None or not _int(sides[s].get("index")) for s, p_ in paths.items()) else \
            {s: [(sides[s]["index"], paths[s])] for s in paths}
    else:
        opened = {s: _sides(op.get(s), n) for s, n in (("a", lay.r), ("b", lay.c))}
        opened = None if None in opened.values() else opened
    if esalt is None or sibs is None or opened is None:
        return (f"the opening's salt is not {SALT_BYTES} bytes, or a path or a side's index is not one (paths of SHA-512 "
                "digests)")
    if not (0 <= e < dom.count and verify_path(dom, root, e, Wd.entry_leaf(dom, e, Wd.entry_bc(entry, esalt)), sibs)):
        return f"entry {e} is not the registered window's entry"
    try:
        i, j = Wd.tile_in(entry, unit)
    except ValueError as x:
        return f"entry {e}: {x}"
    at = {"a": [i], "b": [j]} if lay is None else {"a": [lay.r * i + k for k in range(lay.r)],
                                                  "b": [lay.c * j + k for k in range(lay.c)]}
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
    if t is None or h.get("circuit_sha512") != t.circuit:
        return "the proof's circuit is not the one the verifier pins for the check's template"
    f = h.get("frame_v3")
    s = f.get("schemas") if isinstance(f, dict) else None
    ports = _ports(lay)
    if lay is None:
        if not isinstance(s, dict) or [s.get(x) for x in ports] != [schema for _, schema in OPENED]:
            return "the proof's first rows are not an A row and a B row of the registered schemas"
        if len(body) < 2 * BC_BYTES:
            return "the proof's body holds no A row and B row"
    else:
        names = ports + ["pub"]
        if not isinstance(s, dict) or any(s.get(x) != ROW_V2 for x in names):
            return f"the proof's rows {', '.join(names)} are not {ROW_V2} rows"
        if len(body) < (lay.r + lay.c + 1) * BC_BYTES:
            return "the proof's body holds no A rows, B rows and public row"
    if sorted(s)[:len(ports)] != ports:
        return f"the proof's header does not list its rows {', '.join(ports)} first, in its circuit's port order"
    k = 0
    for side, schema in OPENED:
        sdom = Wd.side_domain(dom.domain_id, side)
        for want, (idx, path) in zip(at[side], opened[side]):
            if idx != want:
                return f"side {side}: the opening is of row {idx}, not the unit's row {want}"
            bc = body[k * BC_BYTES:(k + 1) * BC_BYTES]
            if not (want < sdom.count and verify_path(sdom, entry.root_a if side == "a" else entry.root_b, want,
                                                      row_leaf(sdom, want, Wd._HM.tree_leaf(Wd._HM.default_key, bc),
                                                               schema if lay is None else ROW_V2), path)):
                return f"side {side}: the proof's committed row is not entry {e}'s row {want}"
            k += 1
    if lay is not None and body[k * BC_BYTES:(k + 1) * BC_BYTES] != public_bc(p["context"], salt, unit,
                                                                             (entry.root_a, entry.root_b)):
        return (f"the proof's public row is not the window's certified salt, unit {unit}'s position and entry {e}'s "
                "side roots")
    return None


# ---- the pass's epoch and the window's draw --------------------------------------------------------------------------------

MANIFEST_SCHEMA = domains.VLLM_PEARL_C_RUN
#: a pass recorded before epoch coins, salted from a beacon: only a replay of a recorded pass reads it
MANIFEST_RECORDED = domains.VLLM_PEARL_C_RUN_V1
#: a pass salted by the network certifier, `salt.pouw_salt` of its PoUW window's reveal: only a certified window's
#: verifier, which holds the reveal, reads it, and that verifier reads no other
MANIFEST_CERTIFIED = domains.VLLM_PEARL_C_RUN_CERTIFIED
#: the e2e-service passes' coin commitment, whose binding and weights' root are the first 32 bytes
#: of `tagged`: only a replay of a recorded pass reads it
COINS_RECORDED = domains.POUW_EPOCH_COINS_RECORDED
#: a coin commitment's scheme: (the bytes of `tagged` its binding and the weights' root keep, the weights' root's tag)
COIN_VERSIONS = {Co.SCHEME: (DIGEST_BYTES["sha512"], domains.POUW_WEIGHTS_ROOT),
                 COINS_RECORDED: (32, domains.POUW_WEIGHTS_ROOT_RECORDED)}
#: the e2e-service's first coin commitments, named `coins.SCHEME` but on SHA-256 as core
#: was then: bound by `tagged_sha256`, and the draw's key and stream `derive_v1`'s.  Only a replay of a recorded draw
#: reads it, after the current binding (none of their passes was retained, so the pass's rule doesn't)
COIN_EARLY = Co.SCHEME
DRAW_DOMAIN = domains.POUW_SERVED_ZK_SAMPLING
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
    if root is None or len(root) != DIGEST_BYTES["sha512"]:
        raise Refused("a coin commitment's root is 64 bytes")
    return j["scheme"], j["epoch"], root


def opening_of(j: Any) -> Co.CoinOpening:
    """A coin's opening (`CoinOpening.from_json`), its path of SHA-512 digests."""
    if not isinstance(j, dict) or not _int(j.get("epoch")) or not _str(j.get("label")):
        raise Refused(_OPENING)
    hexes = [_hex(j.get(k)) for k in ("after", "coin", "salt")]
    path = [_hex(p) for p in j["path"]] if isinstance(j.get("path"), list) else None
    if None in hexes or path is None or any(p is None or len(p) != DIGEST_BYTES["sha512"] for p in path):
        raise Refused(_OPENING)
    return Co.CoinOpening(j["epoch"], j["label"], *hexes, tuple(path))


def tagged_sha256(tag: str, *parts: Any) -> bytes:
    """`protocol.tagged` as it was on SHA-256 (`COIN_EARLY`'s binding)."""
    return hashlib.sha256(tag.encode() + b"\0" + b"".join(frame(p) for p in parts)).digest()


def derive_v1(source: bytes, domain: str, context: dict) -> bytes:
    """`randomness.derive` as it was (``verity/randomness/derive/v1``, SHA-256): `COIN_EARLY`'s draw key."""
    ctx = b"".join(frame(k) + frame(context[k]) for k in sorted(context))
    return hashlib.sha256(domains.RANDOMNESS_DERIVE_V1.encode() + b"\0" + frame(source) + frame(domain) + frame(ctx)).digest()


def stream_v1(key: bytes, index: str, n: int) -> bytes:
    """`Key.stream` as it was (``verity/randomness/stream/v1``, SHA-256 blocks): `COIN_EARLY`'s draw stream."""
    head = domains.RANDOMNESS_STREAM_V1.encode() + b"\0" + key + frame(frame(index))
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
    """`audit.weights_root`, an epoch's weights' root, under a coin commitment's scheme (`COIN_VERSIONS`); a live H1
    window's open root binds `weights.weights_root` instead (`salt`)."""
    n, tag = COIN_VERSIONS[scheme]
    return fold_roots(tag, roots)[:n]


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


def _certified_workload(m: dict) -> tuple[list[dict], dict[int, bytes]]:
    schema = m.get("schema")
    if schema != MANIFEST_CERTIFIED:
        if not _str(schema):
            raise Refused("the pass's manifest names no schema")
        raise Co.DomainError(f"schema {schema!r}, not {MANIFEST_CERTIFIED}: the window is certified")
    return _workload(m)


def _recorded(us: list[dict], roots: dict[int, bytes], committed: bytes | None, stream: bytes | None,
              prover: bytes | None) -> None:
    _check_workload(us, roots)
    if not (type(stream) is bytes and len(stream) == PS.STREAM_BYTES and type(prover) is bytes
            and len(prover) == PS.PROVER_BYTES):
        raise Co.DomainError(f"the verifier holds no {PS.STREAM_BYTES}-byte stream digest and {PS.PROVER_BYTES}-byte "
                             "prover id for the window")
    if type(committed) is not bytes or len(committed) != PS.WEIGHTS_ROOT_BYTES:
        raise Co.DomainError("the verifier holds no record the certifier made before it drew the window's salt")
    if PS.pouw_record(weights_root(Co.SCHEME, roots), stream, prover) != committed:
        raise Co.DomainError("the pass's weights, stream and prover are not those the certifier recorded before it drew "
                             "the window's salt")


def recorded_reason(raw: bytes | None, committed: bytes | None, stream: bytes | None, prover: bytes | None
                    ) -> str | None:
    """Why a certified window's pass `raw` is not over the weights, `stream` and `prover` that `committed` binds, what PoUW
    recorded with the certifier before it drew the window's salt (`salt.committed`); None when it is.  `pass_epoch`'s
    rule, with the same words, short of the reveal, which a challenger asks for only after this passes: the certifier
    reveals a window once, so a registration whose stream or prover wasn't recorded gets no reveal."""
    if raw is None:
        return "the window names no pass, whose epoch the challenger re-starts from its manifest"
    try:
        _recorded(*_certified_workload(_manifest(raw)), committed, stream, prover)
    except (Co.DomainError, Refused) as e:
        return str(e)
    return None


def _certified_salt(m: dict, reveal: bytes, context: str, committed: bytes | None, stream: bytes | None,
                    prover: bytes | None) -> tuple[bytes, str]:
    """A certified window's pass under the window's `stream` and `prover`: `committed`, what the certifier recorded
    before it drew the window's salt (`salt.committed`), is `salt.pouw_record` of the pass's weights' root, the stream
    and the prover, and the pass's salt is `salt.pouw_salt` of `reveal`, the certifier's salt for the PoUW window the
    window's `context` names, the window, the stream and the prover."""
    us, roots = _certified_workload(m)
    if type(reveal) is not bytes or len(reveal) != PS.SALT_BYTES:
        raise Co.DomainError(f"the verifier's reveal is not a {PS.SALT_BYTES}-byte salt")
    window = PS.window_of(context)
    if window is None:
        raise Co.DomainError("the window's context names no PoUW window, whose certifier salt the pass's is")
    _recorded(us, roots, committed, stream, prover)
    salt = PS.pouw_salt(reveal, window, stream, prover)
    if salt.hex() != m.get("salt"):
        raise Co.DomainError("the pass's salt is not the certifier's for its window, stream and prover")
    return salt, "certifier"


#: a live Pearl-C pass's inputs as its verifier holds them (`pass_epoch`'s `live`): the window's pinned challenger key,
#: the auditor's ledger's lines, the host's pins log, the verifier's copy of main's first-parent history and the
#: session's files
LIVE = ("key", "ledger", "main", "pins", "session")
LIVE_FORM = ("a live pass's verifier holds {key, ledger, main, pins, session}: the pinned challenger key, the auditor's "
             "ledger, the host's pins log, main's first-parent history and the session's files")
LIVE_UNHELD = ("the pass's coins are a live session's: only a verifier holding the auditor's ledger reads a pass salted by "
               "a live session")
#: a live window's parameters hold the same inputs for its salt session (`params`, `window_salt`)
LIVE_PARAM = "live"
LIVE_PARAMS = ("live is exactly when check.coins is live, and is then {key, ledger, main, pins, session}: the pinned "
               "challenger key, the auditor's ledger, the host's pins log, main's first-parent history and the salt "
               "session's files")


def _live_salt(m: dict, live: Any, verify: Ch.Verify) -> tuple[bytes, str]:
    """A live Pearl-C pass's salt, `live_salt` under the verifier's `LIVE` inputs `live`: its ``manifest`` refusal is
    `Refused`, any other `DomainError`."""
    if live is None:
        raise Co.DomainError(LIVE_UNHELD)
    if not (isinstance(live, dict) and sorted(live) == list(LIVE)):
        raise Co.DomainError(LIVE_FORM)
    salt, refused = live_salt(live["key"], live["ledger"], live["pins"], live["main"], live["session"], m, verify)
    if refused is not None:
        raise (Refused if refused[0] == "manifest" else Co.DomainError)(refused[1])
    return salt, "live coin"


def _coin_salt(m: dict, coins: Any, recorded: bool, us: list[dict], roots: dict[int, bytes]) -> tuple[bytes, str]:
    """A pass salted by epoch coins the verifier holds as a commitment (`coins_of`): its salt coin opens against them
    after the weights' root (`Epoch.start`'s)."""
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
    return derive(opening.coin, EPOCH_SALT_DOMAIN, {"epoch": c[1], "coins": c[2], "weights": root}).stream("salt", 32), "coin"


def _epoch_salt(m: dict, coins: Any, recorded: bool, reveal: bytes | None = None, context: str = "",
                committed: bytes | None = None, stream: bytes | None = None, prover: bytes | None = None,
                live: Any = None, verify: Ch.Verify | None = None) -> tuple[bytes, str]:
    """`manifest_epoch`: the pass's epoch salt re-started from its manifest, and how it was had.  Raises `DomainError`
    (the coin rule's) or `Refused` (a manifest it can't read)."""
    if reveal is not None:
        return _certified_salt(m, reveal, context, committed, stream, prover)
    schema = m.get("schema")
    if schema == MANIFEST_CERTIFIED:
        raise Co.DomainError(f"the pass's manifest is {MANIFEST_CERTIFIED}, salted by the network certifier: only a "
                             "certified window's verifier, which holds the certifier's reveal, reads it")
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
        if isinstance(m.get("coins"), dict) and m["coins"].get("scheme") == Wd.LIVE_SCHEME:
            salt, how = _live_salt(m, live, lms_verify if verify is None else verify)
        elif coins is None:
            raise Co.DomainError("the pass's salt coin opens against the epoch's published coin commitment, which the "
                                 "verifier holds")
        else:
            salt, how = _coin_salt(m, coins, recorded, us, roots)
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


def pass_epoch(coins: Any, raw: bytes | None, context: str, recorded: bool = False, reveal: bytes | None = None,
               committed: bytes | None = None, stream: bytes | None = None, prover: bytes | None = None,
               live: Any = None, verify: Ch.Verify | None = None) -> tuple[str | None, tuple[str, str] | None]:
    """How the pass's salt was had (``coin``, ``live coin``, ``recorded beacon`` in a replay of a recorded pass, or
    ``certifier``) and None, or None and the refusal (``coin`` or ``manifest``, and why): its manifest `raw` re-starts
    its epoch under `coins`, the challenger's own coin commitment (its JSON), and the window's `context` names it by its
    SHA-512.  A live Pearl-C pass, whose manifest's coins are a live session's (`window.LIVE_SCHEME`), is read only
    through the verifier's `live` inputs (`LIVE`, ``{key, ledger, main, pins, session}``, the challenger's signatures
    checked with `verify`, by default `lms_verify`; a verifier that holds none refuses it) and passes `live_salt` under
    them (``live coin``, its ``salt`` refusals here ``coin``'s); `coins` plays no part in it.  A certified window's
    verifier holds `reveal`, the certifier's salt for the PoUW window the context names, as `salt.revealed` read it,
    `committed`, what PoUW recorded with the certifier before it drew that salt, as `salt.committed` read it, and the
    window's `stream` and `prover`; its window
    passes only a pass salted by it (`MANIFEST_CERTIFIED`, the salt `salt.pouw_salt(reveal, p, stream, prover)`) over
    weights whose root `salt.pouw_record` binds with the stream and the prover in `committed`, and a certified pass
    passes only there."""
    if raw is None:
        return None, ("coin", "the window names no pass, whose epoch the challenger re-starts from its manifest")
    try:
        _, how = _epoch_salt(_manifest(raw), coins, recorded, reveal, context, committed, stream, prover, live, verify)
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


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify` under the verifier's challenger key: LMS over SHA-512 at (64, W, H) (`verity.ml.crypto.lms`).
    A signature that doesn't decode raises `ValueError`, which `challenger.record_reason` reads as one that doesn't
    verify."""
    p = lms.Params(DIGEST_BYTES["sha512"], key.W, key.H)
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


#: a live window's coins as the draw's check of record reads them
SIGNED_LIVE = ("a live window's coins are its challenger's signed record and the verifier's key for it, {key, record} "
               "(`challenger.trusted_commitment`), not a bare commitment")


#: the refusal of a draw whose coins aren't the record of the challenger the window's parameters pin
COIN_KEY = "coin-key"
NO_PINNED_KEY = "the window's parameters pin no challenger key for its draw's record"
OTHER_KEY = "the challenger record is signed under a key other than the challenger key the window's parameters pin"
SEEDED_PINNED = ("the window's parameters pin a challenger key, so its draw's coins are that challenger's signed record, "
                 "not seeded coins")


def _live_coins(coins: dict, pinned: Any) -> tuple[Ch.LiveCommitment | None, tuple[str, str] | None]:
    """A live window's commitment from `coins`, ``{"key", "record"}``: `key` must be `pinned`, the challenger key the
    window's parameters pin (``coin-key``), the record must be signed under it (`challenger.record_reason`) and its
    commitment a live epoch's; else None and the refusal."""
    if sorted(coins) != ["key", "record"]:
        return None, ("coin", SIGNED_LIVE)
    try:
        want = Ch.ChallengerKey.from_json(pinned)
    except Ch.CoinError as e:
        return None, (COIN_KEY, f"{NO_PINNED_KEY}: {e}")
    try:
        key = Ch.ChallengerKey.from_json(coins["key"])
    except Ch.CoinError as e:
        return None, ("coin", f"the verifier's challenger key is malformed: {e}")
    if key != want:
        return None, (COIN_KEY, OTHER_KEY)
    why = Ch.record_reason(want, coins["record"], lms_verify)
    if why is not None:
        return None, ("coin", why)
    c = Ch.ChallengerRecord.from_json(coins["record"]).commitment
    return (None, ("coin", why)) if (why := Wd.live_epoch_reason(c)) is not None else (c, None)


def draw(coins: Any, recorded: bool, opening: Any, stream: Any, registration: Any, receipt: Any, k: Any, population: Any,
         units: Any, key: Any = None) -> tuple[str, str] | None:
    """None when `units` are the window's draw: the epoch's draw coin `opening` opens against `coins` (the challenger's
    commitment, its JSON) after the registration `registration` (its digest's hex), and `units` are ``Draw.subset`` of
    `k` over `population` on the stream ``"draw"`` of the key the coin derives under the stream's digest `stream`, the
    registration and its `receipt`; else the refusal (``coin``, ``coin-key``, ``draw`` or ``sampler``, and why). A
    seeded commitment (`coins_of`) has its ``draw`` coin, and is refused when the window's parameters pin a challenger
    key. A live one (`window.LIVE_SCHEME`, a certified window's) is read only from its challenger's signed record with
    the key it is signed under, ``{"key", "record"}``: that key must be `key`, the challenger key the window's
    parameters pin (``coin-key``, so a record the prover signed under a key of its own, over coins it holds, is
    refused), and this check verifies the record under it (`challenger.record_reason`; a bare live commitment is
    refused, `SIGNED_LIVE`); it has its ``pouw/draw`` coin 0 (`challenger.live_reason`)."""
    live = isinstance(coins, dict) and ("record" in coins or coins.get("scheme") == Wd.LIVE_SCHEME)
    if live:
        c, refused = _live_coins(coins, key)
        if c is None:
            return refused
    elif key is not None:
        return COIN_KEY, SEEDED_PINNED
    else:
        try:
            c = coins_of(coins, recorded)
        except (Refused, Ch.CoinError) as e:
            return "coin", f"the verifier's coins are malformed: {e}"
    if not (_str(registration) and _DIGEST.fullmatch(registration)) or not _str(stream):
        return "draw", "the registration is not a SHA-512 digest, or the stream's digest not a string"
    if not (isinstance(receipt, dict) and all(_plain(x) and _plain(v) for x, v in receipt.items())):
        return "draw", "the receipt is not an object of printable ASCII strings"
    if not (_nat(k) and _nat(population) and 1 <= population and k <= population):
        return "draw", "the law is not subset:k over N with k <= N and N >= 1"
    try:
        o = Ch.LiveOpening.from_json(opening) if live else opening_of(opening)
    except (Refused, Ch.CoinError) as e:
        return "coin", f"the draw's {e}"
    if live:
        why, old = Ch.live_reason(c, o, Wd.DRAW, 0, bytes.fromhex(registration)), False
    else:
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


def work_k(k: int, work: int, size: int, total: int, floor: int) -> int:
    """A stratum's draw under the work law (Lean `Flock.Draw.workK`): ⌈k·work·size/total⌉, at least `floor`, at most
    `size`."""
    return min(size, max(floor, (k * work * size + total - 1) // total))


def work_units(coin: bytes, stream: str, registration: str, receipt: dict, strata: list[dict]) -> list[int] | None:
    """`draw`'s rule under the law ``work:k`` (Lean `workUnits`): the units the draw coin `coin` draws over the work strata
    `strata` (each ``{template, units, work, floor, k}``, its units ``[lo, hi)`` ranges in order), ascending. Stratum s is
    ``Draw.subset`` of its k_s over its n_s units on the stream ``("draw", template)`` of the key the coin derives under
    the stream's digest `stream`, the registration `registration` and its `receipt` (`draw_key_context`), its local index
    j its j-th unit (`Flock.Draw.stratified`); None when a stream runs out."""
    key, out = derive(coin, DRAW_DOMAIN, draw_key_context(stream, registration, receipt)), []
    for s in strata:
        n = sum(hi - lo for lo, hi in s["units"])
        got = stream_subset(s["k"], n, key.stream(("draw", s["template"]), 2 * s["k"] * width(n) + 64))
        if got is None:
            return None
        starts = list(itertools.accumulate((hi - lo for lo, hi in s["units"]), initial=0))
        for j in got:
            r = bisect.bisect_right(starts, j) - 1
            out.append(s["units"][r][0] + j - starts[r])
    return sorted(out)


# ---- a live window's salt and round coins -----------------------------------------------------------------------------

NO_RESERVE = ("the auditor's ledger opens the window's session without reserving it: a window's open root binds the "
              "session a signed reserve entry names")
SALT_FORM = "the window's weights' root and its round-0 request table's root are not SHA-512 digests"
OTHER_WINDOW = ("the session's open root is not the window's: salt.window_open_root of its weights' root, its round-0 "
                "request table's root and the session")
NOT_RECEIVED = "the challenger's log does not show the window's open root as the one root its salt coin followed"


#: a live Pearl-C pass's salt rule (`live_salt`)
RESERVE_LAYOUT = ("the auditor's ledger reserves the pass's session for another layout than a Pearl-C salt session's, its "
                  "salt coin alone")
INCOMPLETE = ("the auditor's ledger does not end the pass's session complete, having released its salt coin and nothing "
              "else")
LIVE_NOT_RECEIVED = "the challenger's log does not show the epoch's root as the one root its salt coin followed"
LIVE_NOT_SIGNED = "the pass's coin commitment is not the one its session's challenger signed"
LIVE_SALT = "the pass's salt is not its live coin's and the weights'"
LIVE_SESSION = ("the pass's session is not the gate's answer, without its secret, for the session the auditor's ledger "
                "opened on its epoch's root: its session, seq, root and commitment")
PASS_FORM = ("the pass's manifest names no epoch: `pass_index` is an integer in [0, 2^64), its epoch's number among its "
             "weights' root's epochs")
#: a live window's salt rule (`window_salt`)
WINDOW_SALT_KEYS = ("coins", "epoch", "opening", "salt", "session")
SEEDED_SALT = "the window's coins are seeded, and its service answer carries a live salt"
NO_WINDOW_SALT = ("the window's coins are live, and its service answer carries no salt: its salt coin is a salt-only "
                  "session's, opened on its salt epoch's root")
WINDOW_SALT_FORM = "the answer's salt is not {coins, epoch, opening, salt, session}"
WINDOW_NO_W = ("the window registers no one weights' root W, which its salt epoch's root binds: a SHA-512 digest at "
               "its W port, or else the root of its registered weights' entries (`registered`), not both")
WINDOW_NO_X = ("the window commits no activations before its salt: its registration's window names no activations' "
               "entries (`activations`), whose root X its salt epoch's root binds")
WINDOW_EPOCH = "the answer's salt names no epoch: `epoch` is an integer in [0, 2^64), its number among W's salt epochs"
WINDOW_NOT_SIGNED = "the answer's salt's coin commitment is not the one its session's challenger signed"
WINDOW_SALT = "the answer's salt is not its live coin's round-0 salt after its epoch's root"
WINDOW_SESSION = ("the answer's salt's session is not the gate's answer, without its secret, for the session the "
                  "auditor's ledger opened on its epoch's root: its session, seq, root and commitment")
#: -h3's row seed reads u64le(call index · 2^32 + row), so a call index is below 2^32 (`pearl_c.row_seed`)
CALL_BOUND = 1 << 32
CALL_RANGE = "a call index is 2^32 or more: -h3's row seed takes call indices below 2^32"


def _salt_session(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, verify: Ch.Verify
                  ) -> tuple[tuple[str, str] | None, Any]:
    """The session the salt rules read (`salt`, `live_salt`): the one the auditor's signed v2 ledger `ledger` opens and
    ends closed for `window` under the pinned challenger key `key`, run from a commit the host's pins log `pins` records
    on main's first-parent line `main` (`challenger.pinned_session`), and its files `session`
    (`challenger.session_named`).  (None, (the ledger's session, its signed commitment, the roots its log shows it
    received)), or the refusal and None."""
    try:
        k = Ch.ChallengerKey.from_json(key)
    except Ch.CoinError as e:
        return ("coin", f"the verifier's challenger key is malformed: {e}"), None
    try:
        a = Ch.pinned_session(ledger, window, k, pins, main, verify)
        c, received = Ch.session_named(a, k, session, verify)
    except Ch.CoinError as e:
        return ("coin", f"the challenger's session is not trusted: {e}"), None
    return None, (a, c, received)


def _named(ledger: list[str], sid: str) -> dict[str, dict]:
    """Session `sid`'s entries in the auditor's ledger (which `pinned_session` has read), by kind."""
    return {e["kind"]: e for e in map(json.loads, ledger) if e["session"] == sid}


def _jeq(a: Any, b: Any) -> bool:
    """`a` and `b` are one JSON value (`SampledProofs.jeq`): one canonical text, a boolean never a number."""
    try:
        return _canon(a) == _canon(b)
    except Refused:
        return False


def _carries(s: Any, sid: str, seq: int, root: bytes, commitment: str) -> bool:
    """Whether `s`, the session a pass's manifest carries, is the gate's open answer for session `sid`, opened at ledger
    entry `seq` on `root` under the commitment whose root is `commitment` (hex), without its hello's ``secret``
    (`pouw_h1_challenger.carried`)."""
    return (isinstance(s, dict) and "secret" not in s and s.get("session") == sid and _int(s.get("seq"))
            and s["seq"] == seq and s.get("root") == root.hex() and s.get("commitment") == commitment)


def _salt_epoch(key: Any, ledger: Any, pins: Any, main: Any, session: Any, r: bytes, verify: Ch.Verify
                ) -> tuple[tuple[str, str] | None, Any]:
    """The salt-only session opened on salt epoch root `r` (`_salt_session` for ``{root: r}``), when it counts
    (LIVESALT2): a Pearl-C salt session's commitment, reserved for its salt coin alone and opened binding that reserve,
    ended complete having released that coin and nothing else, its log showing `r` alone as the root the coin followed.
    (None, (the ledger's session, its signed commitment, its open entry)), or the refusal and None."""
    refused, got = _salt_session(key, ledger, pins, main, {"root": r.hex()}, session, verify)
    if refused is not None:
        return refused, None
    a, c, received = got
    why = Wd.salt_epoch_reason(c)
    if why is not None:
        return ("salt", why), None
    named = _named(ledger, a.session)
    if "reserve" not in named:
        return ("salt", NO_RESERVE), None
    reserve, opened, end = named["reserve"], named["open"], named["end"]["body"]
    if not _jeq(reserve["body"].get("families"), [list(f) for f in Wd.SALT_FAMILIES]):
        return ("salt", RESERVE_LAYOUT), None
    if not _jeq(opened["body"].get("reserve"), reserve["seq"]):
        return ("salt", NO_RESERVE), None
    if not (end.get("complete") is True and _jeq(end.get("released"), {Wd.SALT: 1})):
        return ("salt", INCOMPLETE), None
    if received != {Wd.SALT: (r,)}:
        return ("salt", LIVE_NOT_RECEIVED), None
    return None, (a, c, opened)


def values_root(g: Any) -> bytes | None:
    """The root of a group of committed values the window names (`WINDOW_VALUES`), or None when `g` is not one: ``{program,
    values}``, ``program`` a string and ``values`` a non-empty object of entries by value name, each naming its value
    (``value``) and its ``domain`` and ``root``, SHA-512 digests (core's ``registered`` entries: the served
    loop's `loop.activations` and its weights' pin's ``values``).  The root is SHA-512(`VALUES_TAG` ‖ u64be(len program) ‖
    program ‖ for each name in sorted order: u64be(len name) ‖ name ‖ domain ‖ root), strings as UTF-8, so it fixes
    every value's root under its domain.  An entry's other fields (its rows, words and width) are bound by its domain,
    which the records' audit holds to them (`records_audit.committed_activations`, `pinned`)."""
    if not (isinstance(g, dict) and sorted(g) == ["program", "values"] and _str(g["program"])
            and isinstance(g["values"], dict) and g["values"]):
        return None
    out = VALUES_TAG + _sized(g["program"].encode())
    for name in sorted(g["values"]):
        e = g["values"][name]
        if not (isinstance(e, dict) and e.get("value") == name and all(_str(e.get(f)) and _DIGEST.fullmatch(e[f])
                                                                       for f in ("domain", "root"))):
            return None
        out += _sized(name.encode()) + bytes.fromhex(e["domain"]) + bytes.fromhex(e["root"])
    return hashlib.sha512(out).digest()


def _sized(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def window_anchors(rec: dict) -> tuple[bytes, bytes] | str:
    """What an H1 window's salt epoch's root binds besides its epoch, from its registration `rec` (its ``roots`` and
    ``window``): (W, X), or why it names none.  W is its weights' root: its ``W`` root, a SHA-512 digest, or, when it
    registers none, the root of its registered weights' entries (``window.registered``, `values_root`); a window with
    both names no one W (`WINDOW_NO_W`).  X is the root of the activations it committed after serving
    (``window.activations``, `values_root`; `WINDOW_NO_X`): the anchors H1's cut reads at coin 0 (x, sx, the A rows),
    which `salt.anchored_epoch_root` fixes before the salt."""
    port = next((r["root"] for r in rec["roots"] if r["port"] == "W"), None)
    win = rec["window"]
    if "registered" not in win:
        w = bytes.fromhex(port) if _str(port) and _DIGEST.fullmatch(port) else None
    else:
        w = values_root(win["registered"]) if port is None else None
    if w is None:
        return WINDOW_NO_W
    x = values_root(win.get("activations"))
    return WINDOW_NO_X if x is None else (w, x)


def window_salt(p: dict, rec: dict, s: Any, verify: Ch.Verify = lms_verify) -> str | None:
    """Why the salt `s` an accepted window's service answer carries (``public.salt``, None when it carries none) is not
    its live salt, or None, its registration `rec` (``roots`` and ``window``).  A seeded window's answer carries none
    (`SEEDED_SALT`).  A live window's carries ``{coins, epoch, opening, salt, session}`` (`WINDOW_SALT_FORM`): its salt
    epoch's number k among its weights' root W's epochs, an integer in [0, 2^64), and its salt coin's commitment,
    opening, salt and session as `live_salt` reads a live Pearl-C pass's.  W and X are the registration's
    (`window_anchors`), so R = `salt.anchored_epoch_root(W, X, k)` binds the weights and the activations committed
    before the coin: a window whose activations were committed after its salt names an X its session's R doesn't, and
    the ledger opened no session on its R.  R and the session are read only from the registration and the verifier's
    `p["live"]` inputs (`LIVE`), never the answer: the session opened on ``{root: R}`` counts as `live_salt`'s does
    (`_salt_epoch`), ``coins`` is the commitment it signed, ``opening`` its ``pouw/salt`` coin 0 opened after R
    (`challenger.live_reason`), ``salt`` that coin's round-0 unit salt after R (`h1tile.live_salt`: `window.round_key`'s
    stream 0, `salt.SALT_BYTES`), and ``session`` the gate's open answer for the session without its secret
    (`_carries`)."""
    if Wd.mode(p["check"]) != "live":
        return None if s is None else SEEDED_SALT
    if s is None:
        return NO_WINDOW_SALT
    if not (isinstance(s, dict) and sorted(s) == list(WINDOW_SALT_KEYS)):
        return WINDOW_SALT_FORM
    anchors = window_anchors(rec)
    if isinstance(anchors, str):
        return anchors
    k = s["epoch"]
    if not (_int(k) and 0 <= k < U64):
        return WINDOW_EPOCH
    r = PS.anchored_epoch_root(*anchors, k)
    live = p["live"]
    refused, got = _salt_epoch(live["key"], live["ledger"], live["pins"], live["main"], live["session"], r, verify)
    if refused is not None:
        return refused[1]
    a, c, opened = got
    try:
        claimed = Ch.LiveCommitment.from_json(s["coins"])
    except Ch.CoinError as e:
        return f"the salt's coins are malformed: {e}"
    if claimed != c:
        return WINDOW_NOT_SIGNED
    try:
        o = Ch.LiveOpening.from_json(s["opening"])
    except Ch.CoinError as e:
        return f"the salt's {e}"
    why = Ch.live_reason(c, o, Wd.SALT, 0, r)
    if why is not None:
        return f"the salt's {why}"
    if s["salt"] != Wd.round_key(c, o.coin, 0, r).stream(0, PS.SALT_BYTES).hex():
        return WINDOW_SALT
    if not _carries(s["session"], a.session, opened["seq"], r, a.commitment):
        return WINDOW_SESSION
    return None


def live_salt(key: Any, ledger: Any, pins: Any, main: Any, session: Any, manifest: Any, verify: Ch.Verify
              ) -> tuple[bytes | None, tuple[str, str] | None]:
    """A live Pearl-C pass's salt and None, or None and the refusal (``manifest``, ``coin`` or ``salt``, and why): `salt`'s
    rule with the pass's epoch's root R in place of a window's open root, its session and commitment read from the
    auditor's ledger and the session's own files, never the manifest.  The pass's manifest `manifest` (an object) is
    `MANIFEST_SCHEMA`'s, its every call index below 2^32 (`CALL_RANGE`), W `audit.weights_root` of its workload's weights
    (`_workload`, `_check_workload`), and its ``pass_index`` k, an integer in [0, 2^64) (`PASS_FORM`), its salt epoch's
    number among W's: R is `salt.epoch_open_root` of W and k, computed here and never read from the manifest.  The
    session is the one the auditor's signed v2 ledger `ledger` opens on ``{root: R}`` and ends closed under the pinned
    challenger key `key`, run from a commit the host's pins log `pins` records on main's first-parent line `main`, from
    its files `session` (`_salt_session`), and it counts (LIVESALT2): its signed commitment is a Pearl-C salt session's
    (`window.salt_epoch_reason`); the ledger reserved it for that layout (``families`` `window.SALT_FAMILIES`; a session
    opened without a reserve is refused, `NO_RESERVE`), opened it binding that reserve, and ended it complete, having
    released its salt coin and nothing else (``released`` ``{pouw/salt: 1}``); and its signed log shows R, alone, as the
    root its salt coin followed.  Then the manifest's ``coins`` are that commitment, its ``salt_opening`` the
    commitment's ``pouw/salt`` coin 0 opened after R (`challenger.live_reason`), its ``salt`` `audit.salt_of` of that
    coin over W, and its ``session`` the gate's open answer for the session without its secret: the ledger's session,
    its open's seq, R and the commitment the ledger's end names (`_carries`).  The verifier's signatures are checked
    with `verify`.  A pass of epoch k over W reads the session opened on `epoch_open_root(W, k)`, so the passes of one
    epoch read one session and one salt, and one session salts one epoch.  Whether two passes of an epoch share a call
    index is the Pearl-C period's to refuse."""
    m = manifest
    if not isinstance(m, dict):
        return None, ("manifest", "the pass's manifest is not a JSON object")
    schema = m.get("schema")
    if schema != MANIFEST_SCHEMA:
        return None, ("manifest", "the pass's manifest names no schema") if not _str(schema) else \
            ("coin", f"schema {schema!r}, not {MANIFEST_SCHEMA}")
    try:
        us, roots = _workload(m)
        _check_workload(us, roots)
    except Refused as e:
        return None, ("manifest", str(e))
    except Co.DomainError as e:
        return None, ("coin", str(e))
    if any(u["index"] >= CALL_BOUND for u in us):
        return None, ("manifest", CALL_RANGE)
    k = m.get("pass_index")
    if not (_int(k) and 0 <= k < U64):
        return None, ("manifest", PASS_FORM)
    w = weights_root(Co.SCHEME, roots)
    r = PS.epoch_open_root(w, k)
    refused, got = _salt_epoch(key, ledger, pins, main, session, r, verify)
    if refused is not None:
        return None, refused
    a, c, opened = got
    try:
        claimed = Ch.LiveCommitment.from_json(m.get("coins"))
    except Ch.CoinError as e:
        return None, ("coin", f"the pass's coins are malformed: {e}")
    if claimed != c:
        return None, ("coin", LIVE_NOT_SIGNED)
    try:
        o = Ch.LiveOpening.from_json(m.get("salt_opening"))
    except Ch.CoinError as e:
        return None, ("coin", f"the salt's {e}")
    why = Ch.live_reason(c, o, Wd.SALT, 0, r)
    if why is not None:
        return None, ("coin", f"the salt's {why}")
    salt = derive(o.coin, EPOCH_SALT_DOMAIN, {"epoch": c.epoch, "coins": c.root, "weights": w}).stream("salt", 32)
    if m.get("salt") != salt.hex():
        return None, ("coin", LIVE_SALT)
    if not _carries(m.get("session"), a.session, opened["seq"], r, a.commitment):
        return None, ("coin", LIVE_SESSION)
    return salt, None


def live_epoch(raw: bytes) -> tuple[dict, bytes, int]:
    """A live Pearl-C pass's manifest, read from its bytes `raw`, and its salt epoch as `live_salt` reads it: W,
    `audit.weights_root` of its workload's weights, and k, its ``pass_index``, an integer in [0, 2^64). Raises `Refused`
    when the manifest names none (`PASS_FORM` among them), `DomainError` when its workload is malformed."""
    m = _manifest(raw)
    _, roots = _workload(m)
    k = m.get("pass_index")
    if not (_int(k) and 0 <= k < U64):
        raise Refused(PASS_FORM)
    return m, weights_root(Co.SCHEME, roots), k


def salt(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, weights_root: Any, requests_root: Any,
         opening: Any, verify: Ch.Verify) -> tuple[str, str] | None:
    """None when `opening` is a live window's salt coin, opened after the window's own open root, every coin and root of
    it read through the challenger's own session as `rounds` reads them, never one the prover hands in: the one session
    the auditor's signed v2 ledger `ledger` opens and ends closed for `window` (``{root}``) under the pinned challenger
    key `key`, run from a commit the host's pins log `pins` records and the verifier's copy of main's first-parent
    history `main` holds, from that session's files `session` (`challenger.pinned_session` and `session_named`, its
    signatures checked with `verify`).  S is the session the ledger's signed reserve entry names (a session opened without one is
    refused) and R the root its signed open entry opened S on; R is `salt.window_open_root` of the window's weights'
    root `weights_root` and its round-0 request table's root `requests_root` (hex, from the window's record) and S; the
    session's signed log shows R, alone, as the root its salt coin followed; and under the commitment the session
    signed, a live epoch's, `opening` is the epoch's ``pouw/salt`` coin 0 opened after R (`challenger.live_reason`).
    Else the refusal (``coin`` or ``salt``, and why).  A session opened for one window's weights and requests salts no
    other window, and a window salted in one session passes under no other."""
    refused, got = _salt_session(key, ledger, pins, main, window, session, verify)
    if refused is not None:
        return refused
    a, c, received = got
    why = Wd.live_epoch_reason(c)
    if why is not None:
        return "coin", why
    named = _named(ledger, a.session)
    if "reserve" not in named:
        return "salt", NO_RESERVE
    s, r = a.session, bytes.fromhex(Ch.root_of(named["open"]))
    hexes = [bytes.fromhex(x) if _str(x) and _DIGEST.fullmatch(x) else None for x in (weights_root, requests_root)]
    if None in hexes:
        return "salt", SALT_FORM
    w, t = hexes
    if PS.window_open_root(w, t, s) != r:
        return "salt", OTHER_WINDOW
    if received.get(Wd.SALT, ()) != (r,):
        return "salt", NOT_RECEIVED
    try:
        o = Ch.LiveOpening.from_json(opening)
    except Ch.CoinError as e:
        return "coin", f"the salt's {e}"
    why = Ch.live_reason(c, o, Wd.SALT, 0, r)
    return None if why is None else ("coin", f"the salt's {why}")


def rounds(key: Any, ledger: Any, pins: Any, main: Any, window: Any, session: Any, roots: Any, openings: Any,
           verify: Ch.Verify) -> tuple[str, str] | None:
    """None when `openings` are a live window's round coins, read through the challenger's own session and never a
    commitment the prover hands in: the one session the auditor's signed v2 ledger `ledger` (its lines, the head
    `challenger.ledger_head` took) opens and ends closed for `window` (``{root}``) under the window's pinned challenger key
    `key` (its JSON), run from a commit the host's pins log `pins` records and the verifier's copy of main's first-parent
    history `main` holds, from that session's files `session` (``{close, hellos, log, record}``;
    `challenger.trusted_session_of`, its signatures checked with `verify`).  Under the commitment the challenger signed, round r's coin opened once, after
    `roots[r]`, the r-th round root (hex), every recorded round's coin opened (`window.rounds_reason`), and the roots are
    the round roots the challenger's signed log shows it received, in order; else the refusal (``coin`` or ``rounds``,
    and why)."""
    try:
        k = Ch.ChallengerKey.from_json(key)
    except Ch.CoinError as e:
        return "coin", f"the verifier's challenger key is malformed: {e}"
    try:
        c, received = Ch.trusted_session_of(ledger, window, k, pins, main, session, verify)
    except Ch.CoinError as e:
        return "coin", f"the challenger's session is not trusted: {e}"
    if not (isinstance(roots, list) and all(_str(r) and _DIGEST.fullmatch(r) for r in roots)):
        return "rounds", "the round roots are not a list of SHA-512 digests"
    if not isinstance(openings, list):
        return "rounds", "the round coins are not a list"
    os_ = []
    for i, o in enumerate(openings):
        try:
            os_.append(Ch.LiveOpening.from_json(o))
        except Ch.CoinError as e:
            return "coin", f"round coin {i}: {e}"
    rs = tuple(bytes.fromhex(r) for r in roots)
    why = Wd.rounds_reason(c, list(rs), os_)
    if why is not None:
        return "coin", why
    if received.get(Wd.ROUND, ()) != rs:
        return "rounds", "the round roots are not the roots the challenger's log shows it received, in order"
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


__all__ = ["ANSWER_KEYS", "COIN_KEY", "FORMATS", "ID", "LEVEL", "LIVE", "ONE_STAGE", "Refused", "SCHEME", "SIGNED_LIVE",
           "VALUES_TAG", "WINDOW_VALUES", "answer", "answer_of", "canonical", "case", "check", "commitment", "digest",
           "draw", "evidence", "integrity", "integrity_json", "law_name", "line", "live_epoch", "live_salt", "lms_verify",
           "opening", "params", "pass_epoch", "ratio", "recorded_reason", "registered", "rounds", "salt", "unit_draw",
           "values_root", "verify_path", "window_anchors", "window_domain", "window_salt", "work_k", "work_units"]
