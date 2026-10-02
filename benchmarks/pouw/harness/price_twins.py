"""γ from its Lean pins: every Pearl-C and Pearl-C4 price twin whose statement states γ as a literal rational, keyed by what
an attempt ran, so the attempt ledger (`ledger.py`) looks γ up at both FP32 prices instead of anyone typing it.

A twin is one pinned theorem: the Gγ statement of a device record at a cast (FP8 v1 and v2), or the value lemma of γ's
function (v1-hot, v2-hot, Pearl-C4 v1 and v2). Its key is

  statement   the panel line and version it is γ for, hashing format aside: pearl-c-sm120/v2, pearl-c-fp4/v1, ...
  path        the priced path, as `verity_pouw.serving.KernelVariant.prices` and `GateRecord.prices` name it: cast-8 (the
              statement's), cast-8.72, cast-16, cast-32, cast-32.06; lut256 or rcpApprox for Pearl-C4's block scale
  fadd        the FP32 add's W1 price: 8 (the cheapest bit-exact FADD) or 1047/125 (the measured in-loop one)
  reading     forming-credited, chain-only or chain-cap
  kind        unit (Gγ), tile (GγSampled) or value (the γ function's value, which both read)
  shape       8192 or 16384 (the cube it is stated at); ``sizing`` a hot start's rule (publicConst64 is the ruled one)

Rounded twins (a record's own cast under the loop price, an upper bound) and U-only messages have no priced path and are
left out. Each twin keeps its pin's name, type hash and source, and is ``pinned`` when the Lean store's lean-audit.json
holds that pin with that type hash, ``staged`` otherwise.

  python price_twins.py STORE_ROOT [--out price_twins.json]   regenerate the table from the POUS Project store's pins

Stdlib only. The table is checked in beside this file (`price_twins.json`) and regenerated when a pin lands or moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "price_twins.json"
SCHEMA = "pouw-price-twins/v1"
FADD_PRICES = (Fraction(8), Fraction(1047, 125))
#: the POUS Project store's pin files, relative to its root: the Lean store's audit first (what is pinned), then what is
#: staged with a red-team GO (price-twins-lean/README.md)
LEAN_STORE = "lean/submissions/pouw/lean-audit.json"
STAGED = ("internal/pouw/price-twins-lean/proposed-pins.json", "internal/pouw/price-twins-lean/v2-hot/v2-hot-pins.json",
          "internal/pouw/price-twins-lean/v1-hot/v1-hot-pins.json", "internal/pouw/price-twins-lean/fp4-delta/fp4-delta-pins.json")
CASTS = {"8": "cast-8", "8p72": "cast-8.72", "16": "cast-16", "32": "cast-32", "32p06": "cast-32.06"}
CAST_VALUES = {"cast-8": Fraction(8), "cast-8.72": Fraction(218, 25), "cast-16": Fraction(16), "cast-32": Fraction(32),
               "cast-32.06": Fraction(1603, 50)}
READINGS = {None: "forming-credited", "ChainOnly": "chain-only", "Chain": "chain-cap"}
RULED_SIZING = "publicConst64"

_RAT = r"(\d+)(?: / (\d+))?"
_FP8 = re.compile(r"pearlC(?P<kind>Gamma|Sampled)Sm120(?P<ver>v1|v2)(?P<loop>Loop)?(?:Cast(?P<cast>8p72|32p06|16|32|8))?"
                  r"(?P<read>ChainOnly|Chain)?(?:Rev1|Cap1000)_(?P<shape>8192|16384)")
_HOT = re.compile(r"gammaHot(?P<rev>Rev1)?(?P<read>ChainOnly)?_sm120(?P<ver>v1|v2)Hot(?P<loop>Loop)?Cast(?P<cast>8p72|32p06|16|32|8)"
                  r"_(?P<sizing>[A-Za-z0-9]+)_(?P<shape>8192|16384)")
_FP4 = re.compile(r"gammaFp4(?P<hot>Hot)?(?P<read>ChainOnly)?_sm120(?P<price>Issue|Loop)_(?P<path>lut256|rcpApprox)"
                  r"_(?P<shape>8192|16384)")
_CAST_ARG = re.compile(r"instHSub\.hSub \(?" + _RAT + r"\)?")
_GAMMA_LITERAL = re.compile(r"(?:\(|↑\()" + _RAT + r"\)\)?(?:\.cast)? Pouw\.PearlC\.εPearlC\s*$")
_VALUE_LITERAL = re.compile(r"\(" + _RAT + r"\)\)*\s*$")
_FP4_PRICE = re.compile(r"Fp4Prices\.sm120At \(?" + _RAT + r"\)? Pouw\.PearlC\.Fp4ScalePath\.(lut256|rcpApprox)")


class TwinError(ValueError):
    pass


def _q(num: str, den: str | None) -> Fraction:
    return Fraction(int(num), int(den or 1))


def parse(name: str, pin: dict) -> dict | None:
    """One pin as a twin, or None when it isn't one; TwinError when its name and its signature disagree."""
    short = name.removeprefix("Pouw.PearlC.")
    sig = " ".join(" ".join(pin["signature"]).split())
    row: dict
    if m := _FP8.fullmatch(short):
        loop, cast = bool(m["loop"]), m["cast"]
        dev = re.search(r"Pouw\.PearlC\.devSm120(v1|v2) Pouw\.PearlC\.Prices\.(sm120Loop|sm120)\b", sig)
        if dev is None or dev[1] != m["ver"] or (dev[2] == "sm120Loop") != loop:
            raise TwinError(f"{name}: its signature's device record and prices aren't its name's")
        arg = _CAST_ARG.search(sig)
        if cast is None and (loop or arg):
            return None
        path = CASTS[cast or "8"]
        if arg and _q(arg[1], arg[2]) != CAST_VALUES[path]:
            raise TwinError(f"{name}: its signature prices the cast at {_q(arg[1], arg[2])}, not {path}")
        lit = _GAMMA_LITERAL.search(sig)
        if lit is None:
            return None
        row = {"statement": f"pearl-c-sm120/{m['ver']}", "path": path, "fadd": Fraction(1047, 125) if loop else Fraction(8),
               "reading": READINGS[m["read"]], "kind": "unit" if m["kind"] == "Gamma" else "tile",
               "shape": int(m["shape"]), "sizing": None, "gamma": _q(lit[1], lit[2])}
    elif m := _HOT.fullmatch(short):
        loop = bool(m["loop"])
        prices = re.search(r"Eq d\.prices Pouw\.PearlC\.Prices\.(sm120Loop|sm120)\b", sig)
        g = re.search(r"Eq d\.G (\d+)", sig)
        if prices is None or (prices[1] == "sm120Loop") != loop or g is None or (g[1] == "4") != (m["ver"] == "v1"):
            raise TwinError(f"{name}: its signature's prices or promotion period aren't its name's")
        path, arg = CASTS[m["cast"]], _CAST_ARG.search(sig)
        if arg is None or _q(arg[1], arg[2]) != CAST_VALUES[path]:
            raise TwinError(f"{name}: its signature doesn't price the cast at {path}")
        lit = _VALUE_LITERAL.search(sig)
        if lit is None:
            return None
        row = {"statement": f"pearl-c-sm120/{m['ver']}-hot", "path": path, "fadd": Fraction(1047, 125) if loop else Fraction(8),
               "reading": READINGS[m["read"]], "kind": "value", "shape": int(m["shape"]), "sizing": m["sizing"],
               "gamma": _q(lit[1], lit[2])}
    elif m := _FP4.fullmatch(short):
        fadd = Fraction(1047, 125) if m["price"] == "Loop" else Fraction(8)
        p = _FP4_PRICE.search(sig)
        if p is None or _q(p[1], p[2]) != fadd or p[3] != m["path"]:
            raise TwinError(f"{name}: its signature's FP32 price or scale path isn't its name's")
        lit = _VALUE_LITERAL.search(sig)
        if lit is None:
            return None
        row = {"statement": f"pearl-c-fp4/{'v2' if m['hot'] else 'v1'}", "path": m["path"], "fadd": fadd,
               "reading": READINGS[m["read"]], "kind": "value", "shape": int(m["shape"]), "sizing": None,
               "gamma": _q(lit[1], lit[2])}
    else:
        return None
    return row | {"pin": name, "type_hash": pin.get("type_hash")}


def extract(pins: dict[str, dict], source: str, pinned: dict[str, str]) -> list[dict]:
    """Every twin among `pins`, from `source`; ``status`` pinned when `pinned` (the Lean store's name -> type hash) holds it."""
    out = []
    for name in sorted(pins):
        row = parse(name, pins[name])
        if row is not None:
            row["source"] = source
            row["status"] = "pinned" if pinned.get(name) == row["type_hash"] else "staged"
            out.append(row)
    return out


def _jsonable(row: dict) -> dict:
    return {k: (str(v) if isinstance(v, Fraction) else v) for k, v in row.items()}


def build(root: Path) -> dict:
    """The table from the POUS Project store at `root`: every twin, one per pin (the Lean store's copy of a pin first)."""
    audit = json.loads((root / LEAN_STORE).read_text())
    pinned = {k: v.get("type_hash") for k, v in audit["pins"].items()}
    sources, twins, seen = [], [], set()
    for rel in (LEAN_STORE, *STAGED):
        raw = (root / rel).read_bytes()
        rows = extract(json.loads(raw)["pins"], rel, pinned)
        sources.append({"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "twins": len(rows)})
        for r in rows:
            if r["pin"] not in seen:
                seen.add(r["pin"])
                twins.append(_jsonable(r))
    keys = [(t["statement"], t["path"], t["fadd"], t["reading"], t["kind"], t["shape"], t["sizing"]) for t in twins]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise TwinError(f"two pins state the same twin: {dup[:3]}")
    return {"schema": SCHEMA, "sources": sources, "twins": sorted(twins, key=lambda t: t["pin"])}


def load(path: Path = TABLE) -> list[dict]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise TwinError(f"{path} is {doc.get('schema')!r}, not {SCHEMA}")
    return doc["twins"]


def statement(line: str | None, version: str | None) -> str | None:
    """The statement a panel line and version is γ for: the hashing format (-h1, -h2, ...) is outside γ's statement."""
    if not line or not version:
        return None
    return f"{line}/{re.sub(r'-h[0-9]+$', '', version)}"


def lookup(twins: list[dict], statement_: str | None, path: str | None, n: int | None, k: int | None,
           reading: str = "forming-credited") -> dict:
    """γ at both FP32 prices for one attempt: the twins of its statement, priced path and cube, the larger published.
    ``missing`` says what kept it from publishing."""
    out: dict = {"statement": statement_, "path": path, "reading": reading, "by_price": {}, "published": None,
                 "published_at": None, "missing": []}
    if statement_ is None:
        out["missing"].append("the panel line and version (the statement γ is for)")
    if path is None:
        out["missing"].append("the priced path (GateRecord.prices, from the variant's prices)")
    shape = n if n is not None and n == k and n in (8192, 16384) else None
    if shape is None:
        out["missing"].append(f"a pinned cube for (n, k) = ({n}, {k}): twins are stated at 8192³ and 16384³")
    out["shape"] = shape
    if out["missing"]:
        return out
    for fadd in FADD_PRICES:
        hits = [t for t in twins if t["statement"] == statement_ and t["path"] == path and t["shape"] == shape
                and t["reading"] == reading and t["kind"] in ("unit", "value") and Fraction(t["fadd"]) == fadd
                and t["sizing"] in (None, RULED_SIZING)]
        if not hits:
            out["missing"].append(f"a {reading} twin of {statement_} on {path} at {shape}³, FADD {fadd}")
            continue
        t = hits[0]
        out["by_price"][str(fadd)] = {"gamma": t["gamma"], "pin": t["pin"], "type_hash": t["type_hash"], "status": t["status"],
                                      "source": t["source"]}
    if len(out["by_price"]) == len(FADD_PRICES):
        fadd, v = max(out["by_price"].items(), key=lambda kv: Fraction(kv[1]["gamma"]))
        out["published"], out["published_at"] = v["gamma"], fadd
    out["pins"] = sorted({v["status"] for v in out["by_price"].values()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store_root", type=Path, help="the POUS Project store's root (/cursor/stores/<id>)")
    ap.add_argument("--out", type=Path, default=TABLE)
    a = ap.parse_args(argv)
    try:
        doc = build(a.store_root)
    except (TwinError, OSError, KeyError, ValueError) as ex:
        print(f"price_twins: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    a.out.write_text(json.dumps(doc, indent=0, sort_keys=True) + "\n")
    n = {s: sum(t["status"] == s for t in doc["twins"]) for s in ("pinned", "staged")}
    print(f"{len(doc['twins'])} twins ({n['pinned']} pinned, {n['staged']} staged) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
