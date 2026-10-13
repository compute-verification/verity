"""γ from its Lean pins: every Pearl-C and Pearl-C4 price twin whose statement states γ as a literal rational, keyed by what
an attempt ran, so the attempt ledger (`ledger.py`) looks γ up at both FP32 prices instead of anyone typing it.

A twin is one pinned theorem: the Gγ statement of a device record at a cast (FP8 v1 and v2), or the value lemma of γ's
function (v1-hot, v2-hot, Pearl-C4 v1 and v2). Its key is

  statement   the panel line and version it is γ for, hashing format aside: pearl-c-sm120/v2, pearl-c-fp4/v1, ...
  path        the priced path, as `verity.core.protocols.pouw.serving.KernelVariant.prices` and `GateRecord.prices` name it: cast-8 (the
              statement's), cast-8.72, cast-16, cast-32, cast-32.06; lut256 or rcpApprox for Pearl-C4's block scale
  fadd        the FP32 add's W1 price: 8 (the cheapest bit-exact FADD) or 1047/125 (the measured in-loop one)
  reading     forming-credited, chain-only or chain-cap
  kind        unit (Gγ), tile (GγSampled) or value (the γ function's value, which both read)
  shape       8192 or 16384 (the cube it is stated at); ``sizing`` a hot start's rule (publicConst64 is the ruled one)
  n, k        a linear's (n, k) at any batch m, for a twin stated on `pearlCDomainDevNK` (``_n6144k4096``), its shape None
  layout      a served layout's linears, for a twin stated on `pearlCDomainDevShapes` (``_llama31_8b`` on
              `llama31_8bLinears`), γ the max over them; its shape, n and k None
  cap         FP8 v1 and v2's debit cap ρ, read off the protocol the statement is about (``Rev1`` is v1 at 1/400, ``Cap1000``
              and ``Rev1Cap1000`` are 1/1,000); a lookup reads the statement's cap (`CAPS`), and the others are history

Rounded twins (a record's own cast under the loop price, an upper bound) and U-only messages have no priced path and are
left out. Each twin keeps its pin's name, type hash and source, and is ``pinned`` when the Lean store's lean-audit.json
or the repository's own (`REPO_AUDIT`, read after the store's files) holds that pin with that type hash, ``staged``
otherwise.

A twin lifted to a guarantee is pinned as `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G`, whose signature
states no γ. Its statement is the body of the definition `G`, read from the Lean source of the module that the audit's
`reads` records it in, and checked as a signature is. A lifted twin whose `G` the audit doesn't record, or whose body
states no literal γ, fails the table (`TwinError`).

  python price_twins.py STORE_ROOT [--out price_twins.json]   regenerate the table from the POUS Project store's pins
  python price_twins.py --repository-only                     re-read only the repository's pins, keeping the store's rows

Stdlib only. The table is checked in beside this file (`price_twins.json`) and regenerated when a pin lands or moves. A
store file's source records its sha256, the one record of what was read outside the tree. The repository's audit records
none: the suite fails when the table isn't `refresh` of itself, so it changes only when a twin does, not with every
landing that touches the lock.
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
#: the repository's security analysis audit, where PoUW's pins land now; read after the store's files, so a pin they hold keeps its row
REPO_AUDIT = HERE.parents[2] / "Security" / "lean-audit.json"
REPO_SOURCE = "Security/lean-audit.json"
STAGED = ("internal/pouw/price-twins-lean/proposed-pins.json", "internal/pouw/price-twins-lean/v2-hot/v2-hot-pins.json",
          "internal/pouw/price-twins-lean/v1-hot/v1-hot-pins.json", "internal/pouw/price-twins-lean/fp4-delta/fp4-delta-pins.json")
CASTS = {"8": "cast-8", "8p72": "cast-8.72", "16": "cast-16", "32": "cast-32", "32p06": "cast-32.06"}
CAST_VALUES = {"cast-8": Fraction(8), "cast-8.72": Fraction(218, 25), "cast-16": Fraction(16), "cast-32": Fraction(32),
               "cast-32.06": Fraction(1603, 50)}
READINGS = {None: "forming-credited", "ChainOnly": "chain-only", "Chain": "chain-cap"}
RULED_SIZING = "publicConst64"
#: each FP8 statement's debit cap ρ: v1's moved from 1/400 to 1/1,000 (Daniel, 9:43 AM PDT on 2 Oct), v2's was 1/1,000
CAPS = {"pearl-c-sm120/v1": Fraction(1, 1000), "pearl-c-sm120/v2": Fraction(1, 1000)}
NAME_CAPS = {"Rev1": Fraction(1, 400), "Cap1000": Fraction(1, 1000), "Rev1Cap1000": Fraction(1, 1000)}

_RAT = r"(\d+)(?: / (\d+))?"
#: `Pouw.PearlC.`, which a signature prints and a `Guarantees` definition's source may leave to its `open`
_NS = r"(?:Pouw\.PearlC\.)?"
_FP8 = re.compile(r"pearlC(?P<kind>Gamma|Sampled)Sm120(?P<ver>v1|v2)(?P<loop>Loop)?(?:Cast(?P<cast>8p72|32p06|16|32|8))?"
                  r"(?P<read>ChainOnly|Chain)?(?P<cap>Rev1Cap1000|Rev1|Cap1000)"
                  r"_(?:(?P<shape>8192|16384)|n(?P<n>\d+)k(?P<k>\d+)|(?P<layout>[a-z][a-z0-9_]*))")
_LIFTED = re.compile(r"(\S+) : (Pouw\.Guarantees\.\S+)")
_DEV = re.compile(_NS + r"devSm120(v1|v2) " + _NS + r"Prices\.(sm120Loop|sm120)\b")
_CAP_ARG = re.compile(_NS + r"pearlCProtocolDev\w* \(" + _NS + r"devSm120v[12] " + _NS + r"Prices\.\w+\) "
                      r"\w+ \(" + _RAT + r"\)")
_HOT = re.compile(r"gammaHot(?P<rev>Rev1)?(?P<read>ChainOnly)?_sm120(?P<ver>v1|v2)Hot(?P<loop>Loop)?Cast(?P<cast>8p72|32p06|16|32|8)"
                  r"_(?P<sizing>[A-Za-z0-9]+)_(?P<shape>8192|16384)")
_FP4 = re.compile(r"gammaFp4(?P<hot>Hot)?(?P<read>ChainOnly)?_sm120(?P<price>Issue|Loop)_(?P<path>lut256|rcpApprox)"
                  r"_(?P<shape>8192|16384)")
_DEV_ARG = r"\(" + _NS + r"devSm120v[12] " + _NS + r"Prices\.\w+\)"
_DOMAIN_NK = re.compile(_NS + r"pearlCDomainDevNK " + _DEV_ARG + r" (\d+) (\d+)")
_DOMAIN_SHAPES = re.compile(_NS + r"pearlCDomainDevShapes " + _DEV_ARG + r" " + _NS + r"(\w+)Linears\b")
_CAST_ARG = re.compile(r"instHSub\.hSub \(?" + _RAT + r"\)?")
_CAST_SOURCE = re.compile(r"\(" + _RAT + r" - ")
_GAMMA_LITERAL = re.compile(r"(?:\(|↑\()" + _RAT + r"\)\)?(?:\.cast)? Pouw\.PearlC\.εPearlC\s*$")
_GAMMA_SOURCE = re.compile(r"\(\(" + _RAT + r" : ℚ\) : ℝ\) " + _NS + r"εPearlC\s*$")
_VALUE_LITERAL = re.compile(r"\(" + _RAT + r"\)\)*\s*$")
_FP4_PRICE = re.compile(r"Fp4Prices\.sm120At \(?" + _RAT + r"\)? Pouw\.PearlC\.Fp4ScalePath\.(lut256|rcpApprox)")


class TwinError(ValueError):
    pass


def _q(num: str, den: str | None) -> Fraction:
    return Fraction(int(num), int(den or 1))


def lifted_definitions(audit: dict, lean: Path) -> dict[str, str]:
    """The body of each `Pouw.Guarantees` definition that the audit's `reads` record, from its module's source under `lean`,
    with whitespace collapsed: `def G : Prop :=` and the indented lines after it, in `namespace` G's prefix, or
    `def _root_.G : Prop :=` in any namespace (how `tools/verity/lean/check.py --lift` writes it)."""
    out = {}
    for module, group in audit.get("reads", {}).items():
        names = [d for d in group.get("definitions", {}) if d.startswith("Pouw.Guarantees.")]
        path = lean / (module.replace(".", "/") + ".lean")
        if not names or not path.is_file():
            continue
        text = path.read_text()
        for d in names:
            ns, short = d.rsplit(".", 1)
            m = (re.search(rf"^namespace {re.escape(ns)}\n(?:.*\n)*?def {re.escape(short)} : Prop :=\n((?:[ \t]+\S.*\n)+)",
                           text, re.M)
                 or re.search(rf"^def _root_\.{re.escape(d)} : Prop :=\n((?:[ \t]+\S.*\n)+)", text, re.M))
            if m:
                out[d] = " ".join(m[1].split())
    return out


def _statement(name: str, pin: dict, defs: dict[str, str]) -> tuple[str, bool]:
    """What the pin states, and whether it is a lifted guarantee's (`X : G`, stated by `G`'s body in `defs`)."""
    sig = " ".join(" ".join(pin["signature"]).split())
    if (m := _LIFTED.fullmatch(sig)) and m[1] == name:
        if m[2] not in defs:
            raise TwinError(f"{name}: states {m[2]}, which no Guarantees definition the audit's reads record gives")
        return defs[m[2]], True
    return sig, False


def _short(name: str) -> str:
    """A pin's name before its lift: `Pouw.SecurityProofs.PearlC.GammaX` and `Pouw.PearlC.pearlCGammaX` are both
    `pearlCGammaX`."""
    return re.sub(r"^Pouw\.SecurityProofs\.PearlC\.(?=Gamma|Sampled)", "pearlC", name).removeprefix("Pouw.PearlC.")


def parse(name: str, pin: dict, defs: dict[str, str] | None = None) -> dict | None:
    """One pin as a twin, or None when it isn't one; TwinError when its name and its statement disagree, or when it is a
    lifted guarantee whose definition `defs` (`lifted_definitions`) doesn't give or whose statement isn't a twin."""
    short = _short(name)
    if not any(r.fullmatch(short) for r in (_FP8, _HOT, _FP4)):
        return None
    sig, lifted = _statement(name, pin, defs or {})
    row = _row(name, short, sig)
    if row is None and lifted:
        raise TwinError(f"{name}: its guarantee's statement states no literal γ")
    return None if row is None else row | {"pin": name, "type_hash": pin.get("type_hash")}


def _row(name: str, short: str, sig: str) -> dict | None:
    row: dict
    if m := _FP8.fullmatch(short):
        loop, cast = bool(m["loop"]), m["cast"]
        dev = _DEV.search(sig)
        if dev is None or dev[1] != m["ver"] or (dev[2] == "sm120Loop") != loop:
            raise TwinError(f"{name}: its signature's device record and prices aren't its name's")
        arg = _CAST_ARG.search(sig) or _CAST_SOURCE.search(sig)
        if cast is None and (loop or arg):
            return None
        path = CASTS[cast or "8"]
        if arg and _q(arg[1], arg[2]) != CAST_VALUES[path]:
            raise TwinError(f"{name}: its signature prices the cast at {_q(arg[1], arg[2])}, not {path}")
        lit = _GAMMA_LITERAL.search(sig) or _GAMMA_SOURCE.search(sig)
        if lit is None:
            return None
        rho = _CAP_ARG.search(sig)
        if rho is None or _q(rho[1], rho[2]) != NAME_CAPS[m["cap"]]:
            raise TwinError(f"{name}: its signature's protocol isn't at its name's cap {NAME_CAPS[m['cap']]}")
        row = {"statement": f"pearl-c-sm120/{m['ver']}", "path": path, "fadd": Fraction(1047, 125) if loop else Fraction(8),
               "reading": READINGS[m["read"]], "kind": "unit" if m["kind"] == "Gamma" else "tile",
               "shape": int(m["shape"]) if m["shape"] else None, "sizing": None, "cap": NAME_CAPS[m["cap"]],
               "gamma": _q(lit[1], lit[2])}
        if m["n"]:
            dom = _DOMAIN_NK.search(sig)
            if dom is None or (dom[1], dom[2]) != (m["n"], m["k"]):
                raise TwinError(f"{name}: its signature's domain isn't the linear (n, k) = ({m['n']}, {m['k']})")
            row |= {"n": int(m["n"]), "k": int(m["k"])}
        elif m["layout"]:
            dom = _DOMAIN_SHAPES.search(sig)
            if dom is None or dom[1] != m["layout"]:
                raise TwinError(f"{name}: its signature's domain isn't the layout {m['layout']}Linears")
            row |= {"layout": m["layout"]}
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
               "reading": READINGS[m["read"]], "kind": "value", "shape": int(m["shape"]), "sizing": m["sizing"], "cap": None,
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
               "reading": READINGS[m["read"]], "kind": "value", "shape": int(m["shape"]), "sizing": None, "cap": None,
               "gamma": _q(lit[1], lit[2])}
    else:
        return None
    return row


def extract(pins: dict[str, dict], source: str, pinned: dict[str, str], defs: dict[str, str] | None = None) -> list[dict]:
    """Every twin among `pins`, from `source`; ``status`` pinned when `pinned` (the Lean store's name -> type hash) holds it.
    ``defs`` states the lifted guarantees (`lifted_definitions`)."""
    out = []
    for name in sorted(pins):
        row = parse(name, pins[name], defs)
        if row is not None:
            row["source"] = source
            row["status"] = "pinned" if pinned.get(name) == row["type_hash"] else "staged"
            out.append(row)
    return out


def _jsonable(row: dict) -> dict:
    return {k: (str(v) if isinstance(v, Fraction) else v) for k, v in row.items()}


def guarantees(doc: dict) -> dict[str, dict]:
    """A Lean audit record's guarantees; the store's files, written before they were called that, hold them as `pins`."""
    return doc["guarantees"] if "guarantees" in doc else doc["pins"]


def build(root: Path, repo_audit: Path = REPO_AUDIT) -> dict:
    """The table from the POUS Project store at `root` and the repository's audit: every twin, one per pin (the Lean
    store's copy of a pin first, the repository's last, a pin the store holds by its name before its lift included)."""
    files = {rel: root / rel for rel in (LEAN_STORE, *STAGED)} | {REPO_SOURCE: repo_audit}
    pinned: dict[str, str] = {}
    for rel in (REPO_SOURCE, LEAN_STORE):
        pinned |= {k: v.get("type_hash") for k, v in guarantees(json.loads(files[rel].read_text())).items()}
    defs = lifted_definitions(json.loads(repo_audit.read_text()), repo_audit.parent)
    sources, twins, seen = [], [], set()
    for rel, path in files.items():
        raw = path.read_bytes()
        rows = extract(guarantees(json.loads(raw)), rel, pinned, defs)
        sources.append({"path": rel, "twins": len(rows)} | ({} if rel == REPO_SOURCE else
                                                             {"sha256": hashlib.sha256(raw).hexdigest()}))
        for r in rows:
            if _short(r["pin"]) not in seen:
                seen.add(_short(r["pin"]))
                twins.append(_jsonable(r))
    return _table(sources, twins)


def refresh(doc: dict, repo_audit: Path = REPO_AUDIT) -> dict:
    """The table `doc` with the repository's twins re-read from `repo_audit` and every store file's rows kept: what
    `build` gives when only the repository's audit has changed since `doc` was built, without the store. A kept row
    turns pinned when the repository now holds its pin at its type hash."""
    raw = repo_audit.read_bytes()
    pins = guarantees(json.loads(raw))
    kept = [t | ({"status": "pinned"} if pins.get(t["pin"], {}).get("type_hash") == t["type_hash"] else {})
            for t in doc["twins"] if t["source"] != REPO_SOURCE]
    pinned = {k: v.get("type_hash") for k, v in pins.items()} | {t["pin"]: t["type_hash"] for t in kept
                                                                  if t["status"] == "pinned"}
    seen = {_short(t["pin"]) for t in kept}
    rows = extract(pins, REPO_SOURCE, pinned, lifted_definitions(json.loads(raw), repo_audit.parent))
    sources = [s for s in doc["sources"] if s["path"] != REPO_SOURCE]
    sources.append({"path": REPO_SOURCE, "twins": len(rows)})
    return _table(sources, kept + [_jsonable(r) for r in rows if _short(r["pin"]) not in seen])


def _table(sources: list[dict], twins: list[dict]) -> dict:
    keys = [(t["statement"], t["path"], t["fadd"], t["reading"], t["kind"], t["shape"], t.get("n"), t.get("k"), t.get("layout"),
             t["sizing"], t["cap"]) for t in twins]
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
    """The statement a panel line and version is γ for: the hashing format (-h0, -h3) is outside γ's statement."""
    if not line or not version:
        return None
    return f"{line}/{re.sub(r'-h[0-9]+$', '', version)}"


def lookup(twins: list[dict], statement_: str | None, path: str | None, n: int | None, k: int | None,
           reading: str = "forming-credited", cap: Fraction | None = None, layout: str | None = None) -> dict:
    """γ at both FP32 prices for one attempt: the twins of its statement, priced path and shape, at the statement's cap
    (`CAPS`; ``cap`` reads another, as history), the larger published.  The shape is the cube n³ when n = k is 8192 or
    16384, else the linear (n, k) at any m; ``layout`` (``llama31_8b``) reads a served layout's twin instead, whose γ
    is the max over its linears.  ``missing`` says what kept it from publishing."""
    cap = CAPS.get(statement_ or "") if cap is None else Fraction(cap)
    out: dict = {"statement": statement_, "path": path, "reading": reading, "cap": None if cap is None else str(cap),
                 "by_price": {}, "published": None, "published_at": None, "missing": []}
    if statement_ is None:
        out["missing"].append("the panel line and version (the statement γ is for)")
    if path is None:
        out["missing"].append("the priced path (GateRecord.prices, from the variant's prices)")
    shape = n if layout is None and n is not None and n == k and n in (8192, 16384) else None
    if layout is None and (n is None or k is None):
        out["missing"].append(f"the linear's (n, k), here ({n}, {k})")
    out["shape"], out["layout"] = shape, layout
    where = f"layout {layout}" if layout else f"{shape}³" if shape else f"(n, k) = ({n}, {k})"

    def at(t: dict) -> bool:
        if layout is not None:
            return t.get("layout") == layout
        if shape is not None:
            return t["shape"] == shape
        return t.get("n") == n and t.get("k") == k and t.get("layout") is None
    if out["missing"]:
        return out
    for fadd in FADD_PRICES:
        hits = [t for t in twins if t["statement"] == statement_ and t["path"] == path and at(t)
                and t["reading"] == reading and t["kind"] in ("unit", "value") and Fraction(t["fadd"]) == fadd
                and t["sizing"] in (None, RULED_SIZING)
                and (cap is None if t.get("cap") is None else Fraction(t["cap"]) == cap)]
        if not hits:
            out["missing"].append(f"a {reading} twin of {statement_} on {path} at {where}, FADD {fadd}"
                                  + ("" if cap is None else f", cap {cap}"))
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
    ap.add_argument("store_root", type=Path, nargs="?", help="the POUS Project store's root (/cursor/stores/<id>)")
    ap.add_argument("--repository-only", action="store_true",
                    help="re-read only the repository's audit and keep the checked-in table's store rows (`refresh`)")
    ap.add_argument("--out", type=Path, default=TABLE)
    a = ap.parse_args(argv)
    if (a.store_root is None) != a.repository_only:
        ap.error("give the store's root, or --repository-only")
    try:
        doc = refresh(json.loads(TABLE.read_text())) if a.repository_only else build(a.store_root)
    except (TwinError, OSError, KeyError, ValueError) as ex:
        print(f"price_twins: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    a.out.write_text(json.dumps(doc, indent=0, sort_keys=True) + "\n")
    n = {s: sum(t["status"] == s for t in doc["twins"]) for s in ("pinned", "staged")}
    print(f"{len(doc['twins'])} twins ({n['pinned']} pinned, {n['staged']} staged) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
