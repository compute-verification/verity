"""Retired terms in the tree, held to ``allowlists/repo_retired_terms.json``: no file gains a use of a term that
``GLOSSARY.md``'s table of older names retires.  The table is the term source, so a row added there is enforced here
with no change to this module, and its fix is the row's term: ``'X' is retired; say Y (GLOSSARY.md)``.

What a row retires:

* each quoted old name of its first cell (``"C-Flock"``, ``"draw law"``), as a word or as one part of an identifier, in
  any spelling of its parts: ``draw law``, ``draw_law``, ``drawLaw``, ``DRAW_LAW``.  A lower-case part may start upper
  case; a part spelled with capitals (``IR``, ``C-Flock``) matches as spelled or all upper case.  Plurals count, and so do
  a verb's forms where the row says ``(noun, verb)``.  A quote is not enforced where the row narrows it to a sense
  the lint can't tell apart: a quote followed by ``for``, ``meaning``, ``naming``, ``in``, ``as`` or ``of`` narrows every
  quote of its clause (the cell's ``;``-separated part), ``alone`` narrows its own, and one after ``outside`` is the
  exception it names.  A one-word quote followed by a list of senses (``"dark" (compute, fleet, leaves)``) retires
  each phrase (``dark compute``).  The words that ``GLOSSARY.md``'s narrower senses say still carry two senses
  (**Row**, **Transcript**, ...) and those ``LIVE`` names are not enforced alone, only in the phrases that retire them;
* each identifier written in backticks in a clause that says ``until renamed``, when it is one only code spells
  (``Flock.Firewall``, ``FC_FIREWALL*``, ``DrawIsCoin``: a dot, an underscore, a hyphen, an inner capital or a ``*``,
  which matches any identifier characters); a plain word there (``window``), a path, and a name that owns the next
  (``machine_model``'s ``cover``) are not.

The lint reads the whole text of what the lane-text lint reads (``test_repo_lane_text.py``: code, Lean, shell and
config, and Markdown), identifiers, strings and prose alike, outside what ``SKIP`` names, with each reason, and outside
what the word maps of ``tools/move/`` keep: a domain tag or format id (``verity/flock-netlist/v1``), and ``note:``,
``art:``, branch and lane names.  Sites are counted per file: an entry ``{"file", "count"}`` allows ``count`` sites in its
file, a file with more fails, and ``--prune`` lowers what the tree no longer holds.  The ratchet gate
(``adapters/vllm/tools/ratchet_gate.py``) refuses a change that adds a site to a file it touches, grows this list, or
leaves a renamed file's entry behind; it loads this module alone from the change's base, so the module imports only the
standard library and reads ``GLOSSARY.md`` from the tree it runs in.

    python3 tests/lint/test_repo_retired_terms.py --move OLD NEW   # a file (or directory) moved: its entry follows it
    python3 tests/lint/test_repo_retired_terms.py --prune          # delete and lower what this tree no longer holds
    python3 tests/lint/test_repo_retired_terms.py --record         # the list is what this tree holds (a new row's sites)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_retired_terms"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
GLOSSARY = "GLOSSARY.md"
#: the archive isn't developed
LEFT_OUT = ("archive/",)
#: what isn't read, by path prefix (or a whole path), each with its reason
SKIP = {
    GLOSSARY: "the term source: its table of older names says where each old name stays",
    "tools/move/": "the word maps name the words they retire",
    "tools/verity/check/tests/test_move_check.py": "its toy moves rename the words the maps retire",
    "fixtures/": "replicated fixtures and their notes are data, named by what they hold",
    "benchmarks/census/": "data only (AGENTS.md: verity never reads it)",
    f"tests/lint/test_{NAME}.py": "its examples are the terms it refuses",
    "tests/test_retired_terms.py": "the prose lint that holds a few of the terms at zero names them to find them",
}
#: test fixtures: a path with one of these directories is data
FIXTURE_DIRS = ("fixtures", "testdata", "golden")
#: a changelog records what was
HISTORY = ("CHANGELOG.md",)
#: quoted old names the table lists alone that keep another live sense, which the lint can't tell apart
LIVE = {
    "session": "the table retires it by sense (a run, a release, a certifier's attachment, a challenge phase)",
    "kind": "the SP1-era kind; every other kind (a Definition's, a record's, a store's) is live",
    "pin": "a digest pin (`TRUSTED.sha256`, circuit-check's pins) is live; only a pinned guarantee is retired",
    "instance": "Lean's `instance`, Python's `isinstance` and a template instance are live",
    "slack": "the table keeps its other senses, Slack among them",
    "tap": "vLLM's capture taps are live; the certifier's are retired as \"uplink tap\" and \"the auditor's tap\"",
    "remote audit": "Lean's `RemoteAudit` is the remote composite's audit; PoUB's old name is \"remote bandwidth audit\"",
    "M0": "a label the code uses for other things", "M1": "a label the code uses for other things",
    "M2": "a label the code uses for other things",
    "layer 1": "a model's layers", "layer 2": "a model's layers", "layer 3": "a model's layers",
    "commit period": "the Commit period is the term for the steps one Commit covers",
    "IR": "an acronym the code uses for other things (an alias, a cut); \"Verity IR\" and \"IR evaluator\" are retired",
    "reference run": "a run whose output another is compared with is live; the training-cost reference is retired",
}
#: other senses of an enforced word, never counted
OTHER_SENSES = re.compile(r"(?i:power[- ]laws?|laws? of large numbers|(?:Amdahl|Moore|Little|Zipf|Kerckhoffs)'s laws?"
                          r"|power draws?)")
#: what the word maps keep: a domain tag or format id, a note, an artifact, a branch, a lane
KEEP = re.compile(r"[\w.-]+(?:/[\w.-]+)*/v\d+(?:\.\d+)*\b|note:[\w./-]+|art:[0-9a-f]+|cursor/[\w./-]+|lanes/[\w.-]+")
#: a character of what ``KEEP`` keeps
ID = re.compile(r"[\w./:-]")
FIX = (f"Say the term the row gives (GLOSSARY.md's older names). A use the table keeps (a recorded key, a wire format) "
       f"stays only where it already stands. If the file moved, move its entry in the same change: `python3 "
       f"tests/lint/test_{NAME}.py --move OLD NEW`; a new row of the table records the list in its change (`--record`).")

PY = (".py", ".pyi")
LEAN = (".lean",)
MARKDOWN = (".md",)
HASH = (".sh", ".bash", ".toml", ".yaml", ".yml", ".service", ".timer", ".conf", ".cfg", ".ini")
SLASH = (".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".ts", ".js", ".go")

QUOTE = re.compile(r'"([^"\n]+)"')
TICK = re.compile(r"`([^`\n]+)`")
NARROWS = re.compile(r"\b(?:for|meaning|naming|in|as|of)\b")
#: an identifier's parts, as the lint splits a term and a file into words: camel humps, runs of capitals, digits, letters
WORDS = re.compile(r"[A-Z]?[a-zδκ]+|[A-Z]+(?![a-z])|\d+|[δκεΔ]")
INFLECT = {"noun": ("", "s", "es"), "verb": ("", "s", "es", "n", "ing", "ed")}


def holds(path: str) -> bool:
    """Whether this lint reads ``path``: code, Lean, config and Markdown outside ``LEFT_OUT``, ``SKIP``, test fixtures and
    changelogs."""
    if path.startswith(LEFT_OUT) or any(path == s or s.endswith("/") and path.startswith(s) for s in SKIP):
        return False
    if any(d in FIXTURE_DIRS for d in path.split("/")[:-1]) or path.rsplit("/", 1)[-1] in HISTORY:
        return False
    return path.endswith(PY + LEAN + MARKDOWN + HASH + SLASH)


class Term:
    """One retired name: its spelling in the table, what the table says instead, and how it is matched."""

    def __init__(self, old: str, new: str, inflect: str = "noun", code: bool = False):
        self.old, self.new = old, new
        if code:
            body = re.escape(old).replace(r"\.\*", r"\.\w+").replace(r"\*", r"\w*")
            scan = body
        else:
            parts = [p for p in re.split(r"[\s_-]+", old) if p]
            ends = sorted({e for s in INFLECT[inflect] for e in (s, s.upper())}, key=len, reverse=True)
            body = r"[\s_-]?".join(map(self._part, parts)) + "(?:" + "|".join(ends) + ")"
            scan = r"[\s_-]?".join(re.escape(p.lower()) for p in parts) + "(?:" + "|".join(e for e in ends
                                                                                        if e.islower() or not e) + ")"
        self.pattern, self.scan = re.compile(body), re.compile(scan)
        self.anchors = frozenset(w.lower() + s for w in [max(WORDS.findall(old) or [old], key=len)]
                                 for s in INFLECT["verb"])
        self.code = code

    @staticmethod
    def _part(p: str) -> str:
        if p.islower():
            return f"(?:[{p[0]}{p[0].upper()}]{re.escape(p[1:])}|{re.escape(p.upper())})"
        return f"(?:{re.escape(p)}|{re.escape(p.upper())})" if p.upper() != p else re.escape(p)

    def matches(self, text: str, low: str):
        """The spans of this term in ``text`` (``low``, its lower case of the same length) where it stands as a word or as
        one part of an identifier: after no letter or digit but at a camel hump, and before no lower-case letter, nor a
        capital after a capital; a code term before no identifier character.  A word is found in ``low`` by its literal,
        which is fast, and its case then checked in ``text``."""
        found = (self.pattern.finditer(text) if self.code else
                 (m for m in self.scan.finditer(low) if self.pattern.fullmatch(text, *m.span())))
        for m in found:
            s, e = m.span()
            before = text[s - 1] if s else ""
            if before.isalnum() and not (text[s].isupper() and (before.islower() or before.isdigit())):
                continue
            after = text[e] if e < len(text) else ""
            if self.code:
                if after.isalnum() or after == "_":
                    continue
            elif after.isalpha() and (after.islower() or text[e - 1].isupper()):
                continue
            yield s, e


def terms_of(glossary: str) -> list[Term]:
    """The retired terms of ``glossary`` (a ``GLOSSARY.md``'s text), longest first."""
    table = glossary.split("## Older names", 1)[-1].split("\n## ", 1)[0]
    two = glossary.split("still carry two senses", 1)[-1].split("\n## ", 1)[0] if "still carry two senses" in glossary else ""
    narrow = {w.lower() for w in re.findall(r"^\s*- \*\*([^*]+)\*\*", two, re.M)}
    live = {w.lower() for w in LIVE}
    out: dict[str, Term] = {}
    for row in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", row)[1:-1]]
        if len(cells) < 2 or set(cells[0]) <= {"-"} or cells[0].startswith("Old name"):
            continue
        new = re.sub(r"\s+", " ", cells[-1])
        for clause in cells[0].split(";"):
            quotes, narrowed, words = list(QUOTE.finditer(clause)), False, []
            for i, m in enumerate(quotes):
                if clause[:m.start()].rstrip().endswith("outside"):
                    continue
                after = clause[m.end():quotes[i + 1].start() if i + 1 < len(quotes) else len(clause)].split("`", 1)[0]
                paren = re.match(r"\s*\(([^)]*)\)", after)
                rest = after[paren.end():] if paren else after
                if re.match(r"\s*alone\b", rest):
                    continue
                narrowed |= bool(NARROWS.search(rest))
                senses = paren.group(1) if paren else ""
                if "," in senses and senses != "noun, verb" and " " not in m.group(1):
                    words += [(f"{m.group(1)} {s.strip()}", "noun") for s in senses.split(",")]
                else:
                    words.append((m.group(1), "verb" if senses == "noun, verb" else "noun"))
            if not narrowed:
                for w, inflect in words:
                    if w.lower() not in narrow | live:
                        out.setdefault(w, Term(w, new, inflect))
            if "until renamed" in clause:
                for m in TICK.finditer(clause):
                    w = m.group(1)
                    owner = clause.startswith("'s", m.end())
                    if not owner and " " not in w and "/" not in w and re.search(r"[._*-]|[a-z][A-Z]", w):
                        out.setdefault(w, Term(w, new, code=True))
    return sorted(out.values(), key=lambda t: -len(t.old))


def glossary_text() -> str:
    """``GLOSSARY.md`` of the tree this module stands in, or (loaded alone, as the ratchet gate loads it) of the tree the
    command runs in."""
    if (ROOT / GLOSSARY).is_file():
        return (ROOT / GLOSSARY).read_text(encoding="utf-8")
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    return (Path(top) / GLOSSARY).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def terms() -> tuple[tuple[Term, ...], dict[str, list[Term]]]:
    """The retired terms, and each by the words it is found by."""
    found = tuple(terms_of(glossary_text()))
    by: dict[str, list[Term]] = {}
    for t in found:
        for a in t.anchors:
            by.setdefault(a, []).append(t)
    return found, by


def sites_of(text: str, by: dict[str, list[Term]]) -> list[tuple[int, str]]:
    """(line, ``'X' is retired; say Y``) of each site of a retired term in ``text``, ``by`` its terms by word."""
    words = {w.lower() for w in set(WORDS.findall(text))}
    candidates = {t for w in words & by.keys() for t in by[w]}
    if not candidates:
        return []
    low = text.lower()
    if len(low) != len(text):                           # a character whose lower case is longer: lower it alone
        low = "".join(c if len(c.lower()) != 1 else c.lower() for c in text)
    hits = sorted(((s, e, t) for t in candidates for s, e in t.matches(text, low)), key=lambda h: (h[0], -h[1]))
    out, end, line, at = [], -1, 1, 0
    for s, e, t in hits:
        if s < end or kept(text, s, e):
            continue
        end = e
        line, at = line + text.count("\n", at, s), s
        out.append((line, f"{text[s:e]!r} is retired; say {t.new[:120]} (GLOSSARY.md)"))
    return out


def kept(text: str, s: int, e: int) -> bool:
    """Whether the site ``text[s:e]`` stands in what ``KEEP`` keeps or in ``OTHER_SENSES``, read from the run of id
    characters around it and the few words about it."""
    a, b = s, e
    while a and ID.match(text, a - 1):
        a -= 1
    while b < len(text) and ID.match(text, b):
        b += 1
    if any(m.start() <= s - a < m.end() for m in KEEP.finditer(text[a:b])):
        return True
    a = max(0, s - 40)
    return any(m.start() <= s - a < m.end() for m in OTHER_SENSES.finditer(text[a:e + 40]))


def sites(text: str, path: str) -> list[tuple[int, str]]:
    """(line, what) of each site of a retired term in ``path``, ``text`` its content."""
    if not holds(path):
        return []
    return sites_of(text, terms()[1])


def parse(doc: object) -> Counter:
    """Each file's allowed sites, from a list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    counts: Counter = Counter()
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == {"file", "count"} and isinstance(e["file"], str), \
            f"{LIST.name}: an entry is {{file, count}}: {e}"
        assert type(e["count"]) is int and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        assert e["file"] not in counts, f"{LIST.name}: duplicate entry {e}"
        counts[e["file"]] = e["count"]
    return counts


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and holds(p))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[str, int, str], ...]:
    """Each site's file, line and text."""
    return tuple((f, line, what) for f in files() if (ROOT / f).is_file()
                 for line, what in sites((ROOT / f).read_text(encoding="utf-8", errors="replace"), f))


def allowed() -> Counter:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def found() -> Counter:
    return Counter(f for f, *_ in scan())


def test_no_new_retired_terms() -> None:
    have = allowed()
    where: dict[str, list[tuple[int, str]]] = {}
    for f, line, what in scan():
        where.setdefault(f, []).append((line, what))
    new = [f"  {f}: {n} found, {have[f]} allowed\n" + "\n".join(f"    {f}:{line} {what}" for line, what in where[f])
           for f, n in sorted(found().items()) if n > have[f]]
    assert not new, (f"retired terms: {len(new)} file(s) hold more than tests/lint/allowlists/{NAME}.json allows. {FIX}\n"
                     + "\n".join(new))


GLOSSARY_EXAMPLE = """# Glossary
## Older names

| Old name, where it stays | Term |
|---|---|
| "C-Flock"; the stored label `C-Flock` | Flock |
| "law", "draw law"; `law` record keys, the Lean's `Law` | sampling design |
| "draw" (noun, verb), "drawn unit"; `DrawIsCoin`, `Remote.Draw.Session`, `--draw`, until renamed | the sample |
| "firewall"; `flock-firewall`, `Flock.Firewall`, `FC_FIREWALL*`, until renamed | the ZK server |
| "proof unit", "unit class", and "unit" alone meaning one | proof block |
| "components", "product" for a composite's parts | subprotocols; composition |
| "stage", outside "one-stage" | phase |
| "dark" (compute, fleet) | unregistered hardware |
| "session"; "window" for a span of time; `Verity.Window`, `window`, until renamed | a run |
| "IR", "Verity IR" | circuit |

## Narrower senses

- Two words still carry two senses in the code:
  - **Stage**: to stage a statement.
"""


def test_what_is_a_retired_term() -> None:
    got = {t.old: t for t in terms_of(GLOSSARY_EXAMPLE)}
    assert set(got) == {"C-Flock", "law", "draw law", "draw", "drawn unit", "DrawIsCoin", "Remote.Draw.Session", "--draw",
                        "firewall", "flock-firewall", "Flock.Firewall", "FC_FIREWALL*", "proof unit", "unit class",
                        "dark compute", "dark fleet", "Verity.Window", "Verity IR"}, sorted(got)
    by: dict[str, list[Term]] = {}
    for t in got.values():
        for a in t.anchors:
            by.setdefault(a, []).append(t)

    def olds(text: str) -> list[str]:
        return [w.split(" is retired")[0].strip("'\"") for _, w in sites_of(text, by)]
    assert olds("C-Flock's draw law; its drawLaw, draw_law and DRAW_LAW; drawn units, draws, drawing") == [
        "C-Flock", "draw law", "drawLaw", "draw_law", "DRAW_LAW", "drawn units", "draws", "drawing"]
    assert olds("withdraw, drawback, flaw, lawful, Lawrence, unit, a unit of work, one-stage, stage, units_per_vu") == []
    assert olds("a power law, Amdahl's law, the power draw; parse_law, LAW, Law.stratified, lawOf") == [
        "law", "LAW", "Law", "law"]
    assert olds("Flock.Firewall.Frame, FC_FIREWALL_LEAN, flock-firewall, the firewall's, firewall_on") == [
        "Flock.Firewall", "FC_FIREWALL_LEAN", "flock-firewall", "firewall", "firewall"]
    assert olds("proofUnit, ProofUnits, PROOF_UNIT, proof-unit, unit classes, dark compute, the IR, ir_lower, Verity IR") == [
        "proofUnit", "ProofUnits", "PROOF_UNIT", "proof-unit", "unit classes", "dark compute", "Verity IR"]
    assert olds("DrawIsCoin, Remote.Draw.Session, --draw K, Verity.Window, Verity.Windows") == [
        "DrawIsCoin", "Remote.Draw.Session", "--draw", "Verity.Window"]
    assert olds("the domain verity/flock-firewall/v1 and note:x-draw-law, a session, the window") == []
    assert sites_of("a\nb C-Flock\n", by) == [(2, "'C-Flock' is retired; say Flock (GLOSSARY.md)")]


def test_it_reads_what_lane_text_reads_and_skips_what_it_says() -> None:
    assert holds("verity/a.py") and holds("security/A.lean") and holds("README.md") and holds("a/b.toml")
    assert not holds("x.json") and not holds("archive/a.md") and not holds(GLOSSARY) and not holds("tools/move/x_map.toml")
    assert not holds("a/tests/fixtures/b.py") and not holds("a/CHANGELOG.md") and not holds(f"tests/lint/test_{NAME}.py")
    assert sites("C-Flock", "x.json") == []


def test_the_list_is_well_formed_and_an_entry_moves_with_its_file() -> None:
    assert parse({"entries": [{"file": "a.py", "count": 2}]}) == Counter({"a.py": 2})
    for bad in ({"file": "a.py", "count": 0}, {"file": "a.py", "count": True}, {"file": "a.py", "count": 1, "rule": "x"}):
        try:
            parse({"entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    assert moved(Counter({"a/x.py": 2, "a/y.md": 1, "b.py": 1}), "a", "c") == Counter({"c/x.py": 2, "c/y.md": 1, "b.py": 1})


def moved(counts: Counter, old: str, new: str) -> Counter:
    """``counts`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a count already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: Counter = Counter()
    for f, n in counts.items():
        out[new + f[len(old):] if f == old or f.startswith(f"{old}/") else f] += n
    return out


def dump(counts: Counter) -> None:
    lines = [f"    {json.dumps({'file': f, 'count': n}, ensure_ascii=False)}" for f, n in sorted(counts.items()) if n > 0]
    LIST.write_text(f'{{\n  "rule": "retired terms (tests/lint/test_{NAME}.py)",\n  "entries": [\n'
                    + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have = allowed()
        after = moved(have, *sys.argv[2:])
        dump(after)
        print(f"{sum(1 for f in have if f not in after)} entries moved from {sys.argv[2]} to {sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        dump(allowed() & found())
    elif sys.argv[1:] == ["--record"]:
        dump(found())
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
