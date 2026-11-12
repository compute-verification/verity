"""Each listed guarantee's security in bits, where something gives it a number, from the locks and no build.

    verity lean bits [PKG ...] [--json] [--locks-only]

PKG is a Lake package with a `lean-audit.json` (a path from the repository's root); with none, the three security locks
(`LOCKS`). It prints one line for every guarantee a lock lists under `guarantees`, and one for each point and budget a
number is evaluated at, in the columns: the guarantee, its owner, whether its statement bounds a probability, the error
term, the point and the budget, the bits (−log₂ of the error term there), the source of the number and the row.

A number has one of three sources, strongest first (the README's "Bits"): `proved`, a Lean row at one point and budget,
which the locks hold none of yet; `computed`, an evaluator in Lean with a soundness property, none yet either; and
`mirrored`, a Python mirror of one error term (`bits_mirror.py`), printed only while its pins hold. Lines with no number
come first, then `mirrored`, then ascending bits. A mirrored line's row names the Lean instance, when one is, that bounds
the mirrored expression at that point: `tests/test_bits.py` holds the printed figure under it.

What the locks give is exact: the guarantees, their owners, their hashes. Whether a statement bounds a probability, and
its error term, are read from the statement's text, since a lock's `reads` lists what a package's guarantees read and not
what each one does: the signature, or the definition it names, and the definitions it applies, a few deep, looked for a
probability definition (`probabilities`: the locks' `shape.events` and `PROBABILITY`), the error term the right side of
the `≤` after the last one that has one. Those cells are marked `~`. A guarantee a lock lists as another's witness bounds
nothing, from the lock. `--locks-only` reads no source and leaves the `~` cells out.

Exit code 0 always: the command gates nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY = "lean-audit.json"
LOCKS = ("security", "security/Proofs", "security/Properties")
_spec = importlib.util.spec_from_file_location("lean_bits_mirror", HERE / "bits_mirror.py")
M = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("lean_bits_mirror", M)
_spec.loader.exec_module(M)

#: the spec's probability and expectation definitions besides the locks' `shape.events`: what a guarantee's statement
#: applies when it bounds a probability
PROBABILITY = ("Claims.Pr", "Claims.prUniform", "Claims.Ex", "Claims.PrENNReal", "Claims.Finder.Pr",
               "Hash.Assumptions.LmsForger.Pr", "Verity.Game.prob", "Verity.Game.avg", "Verity.Game.prCoin",
               "Verity.Game.expect", "Verity.Game.prPMF", "Verity.Game.prPMFReal", "Pouw.pr", "Pous.pr",
               "Pouw.RelativeSecurity.prob", "Accounting.Slowdown.Task.prob")
#: how many definitions deep the text pass follows a statement
DEPTH = 3
#: a name the text pass resolves to more definitions than this is too common to follow
AMBIGUOUS = 4
MODIFIERS = r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|public|nonrec|partial|unsafe)\s+)*"
DECL = re.compile(rf"^{MODIFIERS}(def|abbrev|theorem|lemma|structure|class|inductive|instance|opaque)\s+([^\s(:{{\[⦃]+)")
IDENT = re.compile(r"[^\W\d][\w'.!?]*")
FOLLOWED = {"def", "abbrev", "structure", "class", "inductive", "opaque"}
OPEN, CLOSE = "([{⟨⦃", ")]}⟩⦄"
STOPS = ("∧", "∨", "→", "↔", ":=", ",")


def load(pkg: Path) -> dict:
    return json.loads((pkg / POLICY).read_text())


def probabilities(locks: dict[str, dict]) -> set[str]:
    """The probability definitions: every lock's `shape.events` and `PROBABILITY`."""
    return set(PROBABILITY).union(*((lock.get("shape") or {}).get("events") or [] for lock in locks.values()))


def requires(pkg: Path) -> list[Path]:
    """``pkg`` and the packages it requires by path, transitively (their lakefile.toml's `[[require]] path`)."""
    out, todo = [], [pkg.resolve()]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.append(p)
        lakefile = p / "lakefile.toml"
        if lakefile.is_file():
            todo += [(p / r["path"]).resolve() for r in tomllib.loads(lakefile.read_text()).get("require", []) if "path" in r]
    return out


class Sources:
    """The declarations of some packages' Lean sources, read as text: {full name: `Decl`}, the text the declaration's
    lines up to the next line at column 0. Namespaces, sections and `open` are followed as far as a line's first words
    show them."""

    def __init__(self, dirs: list[Path]):
        self.decls: dict[str, Decl] = {}
        self.by_last: dict[str, list[str]] = {}
        files = sorted({f for d in dirs for f in d.rglob("*.lean") if ".lake" not in f.parts})
        for f in files:
            self._read(f)
        for name in self.decls:
            self.by_last.setdefault(name.rsplit(".", 1)[-1], []).append(name)

    def _read(self, f: Path) -> None:
        lines = f.read_text(errors="replace").splitlines()
        scopes: list[tuple[str | None, list[str]]] = [(None, [])]
        at, pending = None, []
        rel = str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f)

        def close(end: int) -> None:
            if at:
                name, kind, start, ctx = at
                self.decls.setdefault(name, Decl(kind, rel, start + 1, "\n".join(lines[start:end]), ctx))

        for i, line in enumerate(lines):
            if not line or line[0].isspace():
                continue
            close(i)
            at = None
            words = line.split()
            if words[0] == "namespace" and len(words) > 1:
                scopes.append((words[1], []))
            elif "section" in words[:2] and words[0] in ("section", "noncomputable", "public"):
                scopes.append((None, []))
            elif words[0] == "end":
                if len(scopes) > 1:
                    scopes.pop()
            elif words[0] == "open":
                names = [w for w in words[1:] if w not in ("in", "scoped") and w[0] not in "(" and not w.startswith("--")]
                names = names[:next((k for k, w in enumerate(names) if w in ("hiding", "renaming")), len(names))]
                (pending if words[-1] == "in" else scopes[-1][1]).extend(names)
            elif m := DECL.match(line):
                ns = ".".join(s for s, _ in scopes if s)
                name = m.group(2)
                name = name[len("_root_."):] if name.startswith("_root_.") else f"{ns}.{name}" if ns else name
                parts = ns.split(".") if ns else []
                bases = tuple(".".join(parts[:k]) for k in range(len(parts), -1, -1))
                opens = tuple(dict.fromkeys(o for _, os in scopes for o in os)) + tuple(pending)
                at = (name, m.group(1), i, (bases, opens))
                pending = []
            elif not line.startswith(("/-", "--", "@[", "set_option", "variable", "attribute", "omit", "include")):
                pending = []
        close(len(lines))

    def resolve(self, token: str, ctx: tuple = (("",), ()), field: bool = False) -> list[str]:
        """The declarations ``token`` names in the context ``ctx`` (the namespaces enclosing it, longest first, and the
        namespaces open there): under one of them, as Lean resolves a name; or, for field notation on a variable (`L.KSReg`,
        `(f x).Pr`), the declarations whose name ends in the field, when there are at most `AMBIGUOUS` of them."""
        bases, opens = ctx
        if not field:
            for base in (*bases, *opens, *(f"{b}.{o}" for o in opens for b in bases if b)):
                if (n := f"{base}.{token}" if base else token) in self.decls:
                    return [n]
            if "." not in token:
                return []
            token = token.split(".", 1)[1]
        found = [n for n in self.by_last.get(token.rsplit(".", 1)[-1], ()) if n == token or n.endswith("." + token)]
        return found if len(found) <= AMBIGUOUS else []


@dataclass(frozen=True)
class Decl:
    kind: str
    file: str
    line: int
    text: str
    ctx: tuple


def bound_after(text: str, at: int) -> str | None:
    """The right side of the first `≤` or `<` after ``at`` at the depth of ``at``, up to a connective at that depth or
    the end of the enclosing bracket; None when there is none before either."""
    depth, i, start = 0, at, None
    while i < len(text):
        c = text[i]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth -= 1
            if depth < 0:
                break
        elif depth == 0 and start is None and c in "≤<" and not text.startswith("<;>", i):
            start = i + 1
        elif depth == 0 and any(text.startswith(s, i) for s in STOPS):
            if start is None:
                return None
            break
        i += 1
    if start is None:
        return None
    out = " ".join(text[start:i].split())
    return out or None


class Reader:
    """Whether a statement bounds a probability, and its error term, from its text (`Sources`)."""

    def __init__(self, sources: Sources, prob: set[str]):
        self.src, self.prob = sources, prob
        self._reaches: dict[tuple[str, int], bool] = {}

    def names(self, text: str, ctx: tuple) -> list[tuple[int, list[str]]]:
        """Each identifier of ``text`` that names a declaration: its end and the declarations it may name."""
        out = []
        for m in IDENT.finditer(text):
            if found := self.src.resolve(m.group(0), ctx, field=text[m.start() - 1:m.start()] == "."):
                out.append((m.end(), found))
        return out

    def reaches(self, name: str, depth: int) -> bool:
        """``name`` is a definition whose text applies a probability definition, or one that does within ``depth`` more
        definitions."""
        key = (name, depth)
        if key not in self._reaches:
            self._reaches[key] = False
            d = self.src.decls[name]
            self._reaches[key] = d.kind in FOLLOWED and any(
                n in self.prob or (n != name and depth > 0 and self.reaches(n, depth - 1))
                for _, ns in self.names(d.text, d.ctx) for n in ns)
        return self._reaches[key]

    def statement(self, record: dict) -> tuple[str, str | None, tuple]:
        """The text of the guarantee's statement, the definition it is when its signature names one, and the context its
        names resolve in."""
        sig = " ".join(record.get("signature") or [])
        m = re.fullmatch(r"\S+ : ([^\s()]+)", sig.strip())
        if m and m.group(1) in self.src.decls:
            d = self.src.decls[m.group(1)]
            return d.text, m.group(1), d.ctx
        return sig, None, (("",), ())

    def read(self, record: dict) -> dict:
        """{"probability": bool, "error": str | None, "from": the definition holding the error term}."""
        text, named, ctx = self.statement(record)
        todo, seen = [(text, named, ctx, DEPTH)], set()
        prob, error, where = False, None, None
        while todo and error is None:
            t, name, c, depth = todo.pop(0)
            for end, found in reversed(self.names(t, c)):
                leads = [n for n in found if n not in self.prob and n != name and depth > 0 and self.reaches(n, depth - 1)]
                if not leads and not any(n in self.prob for n in found):
                    continue
                prob = True
                if (b := bound_after(t, end)) is not None:
                    error, where = b, name
                    break
                for n in leads:
                    if n not in seen:
                        seen.add(n)
                        d = self.src.decls[n]
                        todo.append((d.text, n, d.ctx, depth - 1))
        return {"probability": prob, "error": error, "from": where}


def witnesses(lock: dict) -> set[str]:
    """The guarantees ``lock`` lists as another's witness (`witnesses`, `claims.witnesses`): each states that a bound
    holds at an instance or fails at one, and bounds nothing itself."""
    out: set[str] = set()
    for section in ((lock.get("witnesses") or {}).values(), ((lock.get("claims") or {}).get("witnesses") or {}).values()):
        for entry in section:
            for v in (entry or {}).values() if isinstance(entry, dict) else ():
                out |= {v} if isinstance(v, str) else set(v) if isinstance(v, list) else set()
    return out


def pins_hold(mirror, locks: dict[str, dict]) -> list[str]:
    """What moved among ``mirror``'s pins in ``locks``: each pinned guarantee whose record's hash differs, and each pinned
    definition whose `reads` hash differs, in a lock that lists it."""
    moved = []
    for g, (key, h) in mirror.guarantees.items():
        for lock in locks.values():
            if g in (lock.get("guarantees") or {}) and lock["guarantees"][g].get(key) != h:
                moved.append(f"{g}'s {key}")
    for d, h in mirror.definitions.items():
        for lock in locks.values():
            for group in (lock.get("reads") or {}).values():
                if d in (group.get("definitions") or {}) and group["definitions"][d] != h:
                    moved.append(f"{d}'s reads hash")
    return moved


def bits_of(v: Fraction | float) -> float:
    """−log₂ ``v``, exactly as far as a float holds it for a rational."""
    if isinstance(v, Fraction):
        return math.log2(v.denominator) - math.log2(v.numerator) if v > 0 else math.inf
    return -math.log2(v) if v > 0 else math.inf


def lines(pkgs: list[Path], text: bool = True) -> list[dict]:
    """The table's lines for the guarantees ``pkgs``' locks list, ordered: no number, then `mirrored`, then ascending
    bits."""
    locks = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): load(p) for p in pkgs}
    reader = None
    if text:
        dirs = list(dict.fromkeys(d for p in pkgs for d in requires(p)))
        reader = Reader(Sources(dirs), probabilities(locks))
    wit = {where: witnesses(lock) for where, lock in locks.items()}
    mirrored: dict[str, list] = {}
    for mirror in M.MIRRORS:
        for g in mirror.guarantees:
            mirrored.setdefault(g, []).append(mirror)
    out = []
    for where, lock in locks.items():
        for g, record in sorted((lock.get("guarantees") or {}).items()):
            base = {"guarantee": g, "lock": where, "owner": record.get("owner"), "probability": None, "error": None,
                    "error_from": None, "point": None, "budget": None, "args": None, "bits": None, "source": None,
                    "row": None, "note": None, "witness": False}
            if g in wit[where]:
                base |= {"probability": False, "witness": True}
            elif reader:
                r = reader.read(record)
                base |= {"probability": r["probability"], "error": r["error"], "error_from": r["from"]}
            numbered = False
            for mirror in mirrored.get(g, []):
                if moved := pins_hold(mirror, {where: lock}):
                    base["note"] = (f"{mirror.lean}'s mirror is not printed: {', '.join(moved)} moved; re-read the "
                                    f"statement, update the mirror, then its pin")
                    continue
                for budget, bargs in mirror.budgets.items():
                    for point in mirror.points:
                        args = point.args | bargs
                        out.append(base | {"error": mirror.term, "error_from": mirror.lean, "point": point.name,
                                           "point_about": point.about, "budget": budget,
                                           "args": {k: str(v) for k, v in args.items()},
                                           "bits": bits_of(mirror.value(**args)), "source": "mirrored",
                                           "row": point.instance})
                        numbered = True
            if not numbered:
                out.append(base)
    rank = {None: 0, "mirrored": 1, "computed": 2, "proved": 2}
    out.sort(key=lambda r: (rank[r["source"]], r["bits"] if r["bits"] is not None else 0, r["guarantee"],
                            r["point"] or "", r["budget"] or ""))
    return out


def fmt_bits(b: float | None) -> str:
    return "-" if b is None else "vacuous" if b <= 0 else f"{b:.1f}"


def table(rows: list[dict], text: bool) -> str:
    def cell(s: str | None, mark: bool = False) -> str:
        return "-" if s is None else (("~" if mark else "") + s.replace("|", "\\|"))

    out = ["| guarantee | owner | probability | error term | point | budget | bits | source | row |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        prob = "no, a witness" if r["witness"] else None if r["probability"] is None else "yes" if r["probability"] else "no"
        err = r["error"] if r["error"] is None or len(r["error"]) <= 90 else r["error"][:89] + "…"
        out.append(f"| {r['guarantee']} | {cell(r['owner'])} | {cell(prob, text and not r['witness'])} | "
                   f"{cell(err, text and r['source'] is None)} | {cell(r['point'])} | {cell(r['budget'])} | "
                   f"{fmt_bits(r['bits'])} | {cell(r['source'])} | {cell(r['row'])} |")
    return "\n".join(out)


def summary(rows: list[dict]) -> dict:
    guarantees = {(r["lock"], r["guarantee"]) for r in rows}
    prob = {(r["lock"], r["guarantee"]) for r in rows if r["probability"]}
    numbered = {(r["lock"], r["guarantee"]) for r in rows if r["bits"] is not None}
    by = {}
    for r in rows:
        by[r["source"] or "none"] = by.get(r["source"] or "none", 0) + 1
    return {"lines": len(rows), "guarantees": len(guarantees), "probability": len(prob), "with_a_number": len(numbered),
            "lines_by_source": by, "pins_moved": sorted({r["guarantee"] for r in rows if r["note"]})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity lean bits", description=__doc__.split("\n\n")[0])
    ap.add_argument("pkg", nargs="*", help=f"Lake packages with a {POLICY} (default: {', '.join(LOCKS)})")
    ap.add_argument("--json", action="store_true", help="print the lines and their counts as JSON")
    ap.add_argument("--locks-only", action="store_true",
                    help="read no Lean source: leave out what the statements' text gives (the `~` cells)")
    a = ap.parse_args(argv)
    pkgs = [(ROOT / p) if not Path(p).is_absolute() else Path(p) for p in (a.pkg or LOCKS)]
    rows = lines(pkgs, text=not a.locks_only)
    if a.json:
        print(json.dumps({"lines": rows, "summary": summary(rows)}, indent=1, ensure_ascii=False))
        return 0
    print(table(rows, not a.locks_only))
    s = summary(rows)
    print(f"\n{s['lines']} lines, {s['guarantees']} guarantees, {s['probability']} bound a probability"
          f"{'' if not a.locks_only else ' (not read: --locks-only)'}, {s['with_a_number']} have a number; lines by "
          f"source: " + ", ".join(f"{k} {v}" for k, v in sorted(s["lines_by_source"].items())))
    if not a.locks_only:
        print("~ read from the statement's text, not from the lock")
    for r in rows:
        if r["note"]:
            print(f"{r['guarantee']}: {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
