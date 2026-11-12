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
``art:``, branch and lane names.  A word's other senses aren't counted: by the words about it (``OTHER_SENSES``: a power
law, nvidia-smi's ``power.draw``, ``AUDIT_CAP``), by the code around it (hypothesis's ``draw``) and by the file's place
(``OWN_CAPS``: PoUS's, PoUB's and Flock's own caps).  Sites are counted per file: an entry ``{"file", "count"}`` allows
``count`` sites in its file, a file with more fails, and ``--prune`` lowers what the tree no longer holds.  The ratchet gate
(``adapters/vllm/tools/ratchet_gate.py``) refuses a change that adds a site to a file it touches, grows this list, or
leaves a renamed file's entry behind; it loads this module alone from the change's base, so the module imports only the
standard library and reads ``GLOSSARY.md`` from the tree it runs in.

    python3 tests/lint/test_repo_retired_terms.py --move OLD NEW   # a file (or directory) moved: its entry follows it
    python3 tests/lint/test_repo_retired_terms.py --prune          # delete and lower what this tree no longer holds
    python3 tests/lint/test_repo_retired_terms.py --record         # the list is what this tree holds (a new row's sites)
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_retired_terms"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
GLOSSARY = "GLOSSARY.md"
#: path prefixes this lint leaves to others (the ratchet gate reads it): none
LEFT_OUT = ()
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
#: other senses of an enforced word, never counted: nvidia-smi's fields (``power.draw``, ``compute_cap``) among them, the
#: names of PoUS's cap on an audit's duration and PoUB's lifetime cap, and the stacking property's caps as its Lean names
#: them (``Gates.Caps``, ``FoolingCap``), none of them Pearl's
OTHER_SENSES = re.compile(r"(?i:power[- ]laws?|laws? of large numbers|(?:Amdahl|Moore|Little|Zipf|Kerckhoffs)'s laws?"
                          r"|power[ ._-]draws?)|(?<![^\W_])(?:audit_cap|AUDIT_CAP|auditCap|cap_ns|CAP_NS"
                          r"|schedule_cap|SCHEDULE_CAP)(?![^\W_])"
                          r"|(?<!\w)(?:R_cap(?:_phase|_node(?:_by_class_peak)?)?|caps\.total|FoolingCap|compute_cap)(?!\w)")
#: the files whose "cap" is their protocol's own, by path prefix, with each reason; Pearl's spellings still count there.
#: Flock's are two, in the Lean that states and proves its knowledge soundness and sampling over it: the bound on the reruns
#: one wait of the extractor makes, and a commitment's Merkle cap
OWN_CAPS = {
    "verity/protocols/pous/": "PoUS's caps (an audit's duration, its schemes' bounds); Pearl's is PoUW's",
    "verity/protocols/poub/": "PoUB's lifetime cap on a schedule's bits; Pearl's is PoUW's",
    "security/Definitions/Claims/": "the bound on an extractor's reruns that Flock's claims state (`Deploy.cap`, `Arg.cap`)",
    "security/Proofs/Flock/Claims/KS/": "that bound and the Merkle cap (`Model.Cap`), in Flock's knowledge soundness",
    "security/Proofs/Flock/Claims/Numbers": "that bound at the deployed numbers (`Numbers.cap`): each `Numbers*.lean`",
    "security/Proofs/Flock/SamplingGeneric/": "that bound, of the proof system sampling is proved over (`Arg.cap`)",
}
#: Pearl's `CAP`, which the table spells as a word, so no longer term holds it ("over-cap", "cap check", `capOK` and
#: `Device.cap` are terms of their own)
PEARL_CAP = re.compile(r"(?<!\w)CAPS?(?!\w)")
#: hypothesis's own ``draw`` in a module importing it: a composite strategy's first parameter, through its body, and the
#: data object's ``data.draw(``, as code where the file is Python; a comment or a string there, a ``"draw"`` key anywhere, a
#: longer name, an attribute (``challenger.draw``) and the other forms (``draws``, ``drawn``) aren't
HYPOTHESIS = re.compile(r"^[ \t]*(?:from|import)[ \t]+hypothesis\b", re.M)
COMPOSITE = re.compile(r"^([ \t]*)@(?:\w+\.)*composite\b.*\n(?:\1@.*\n)*\1(?:async[ \t]+)?def[ \t]+\w+[ \t]*\(\s*draw\b",
                       re.M)
BARE_DRAW = re.compile(r"(?<![\w.'\"-])draw(?![\w'\"])")
DATA_DRAW = re.compile(r"(?<![\w.])data\.draw\(")
#: a line that can end a block: its indent, then neither a comment nor a closing bracket
LINE = re.compile(r"^([ \t]*)[^\s#)\]}]", re.M)
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


def sites_of(text: str, by: dict[str, list[Term]], path: str = "") -> list[tuple[int, str]]:
    """(line, ``'X' is retired; say Y``) of each site of a retired term in ``text``, ``by`` its terms by word, ``path`` the
    file's.  Where two terms match one span, the one the table spells longer names it, then the first by spelling."""
    words = {w.lower() for w in set(WORDS.findall(text))}
    candidates = sorted({t for w in words & by.keys() for t in by[w]}, key=lambda t: (-len(t.old), t.old))
    if not candidates:
        return []
    low = text.lower()
    if len(low) != len(text):                           # a character whose lower case is longer: lower it alone
        low = "".join(c if len(c.lower()) != 1 else c.lower() for c in text)
    hits = sorted(((s, e, t) for t in candidates for s, e in t.matches(text, low)), key=lambda h: (h[0], -h[1]))
    draws, caps = hypothesis_draws(text, path), path.startswith(tuple(OWN_CAPS))
    out, end, line, at = [], -1, 1, 0
    for s, e, t in hits:
        if s < end or text[s:e] == "draw" and s in draws or kept(text, s, e, caps):
            continue
        end = e
        line, at = line + text.count("\n", at, s), s
        out.append((line, f"{text[s:e]!r} is retired; say {t.new[:120]} (GLOSSARY.md)"))
    return out


def kept(text: str, s: int, e: int, own_caps: bool = False) -> bool:
    """Whether the site ``text[s:e]`` stands in what ``KEEP`` keeps or in ``OTHER_SENSES``, read from the run of id
    characters around it and the few words about it, or is a "cap" other than ``PEARL_CAP`` in a file ``OWN_CAPS`` names
    (``own_caps``)."""
    a, b = s, e
    while a and ID.match(text, a - 1):
        a -= 1
    while b < len(text) and ID.match(text, b):
        b += 1
    if any(m.start() <= s - a < m.end() for m in KEEP.finditer(text[a:b])):
        return True
    if own_caps and text[s:e].lower() in ("cap", "caps") and not PEARL_CAP.match(text, s):
        return True
    a = max(0, s - 40)
    return any(m.start() <= s - a < m.end() for m in OTHER_SENSES.finditer(text[a:e + 40]))


def hypothesis_draws(text: str, path: str = "") -> frozenset[int]:
    """Where ``text``, at ``path``, says hypothesis's own ``draw`` (``HYPOTHESIS``): the start of each.  A composite's body
    runs to the first line after its header, not blank, a comment or a closing bracket, indented no deeper than its
    decorator.  In a Python file only code counts; any other file's text has no code to tell from prose, so every bare
    ``draw`` in the body counts."""
    if "draw" not in text or not HYPOTHESIS.search(text):
        return frozenset()
    out = {m.end() - len("draw(") for m in DATA_DRAW.finditer(text)}
    for m in COMPOSITE.finditer(text):
        after = text.find("\n", m.end()) + 1 or len(text)
        end = next((n.start() for n in LINE.finditer(text, after) if len(n.group(1)) <= len(m.group(1))), len(text))
        out |= {d.start() for d in BARE_DRAW.finditer(text, m.end() - len("draw"), end)}
    if not out or not path.endswith(PY):
        return frozenset(out)
    return frozenset(out) & code_names(text, "draw", path)


def code_names(text: str, name: str, path: str) -> frozenset[int]:
    """Where ``text``, the Python file ``path``, says ``name`` as code, outside comments and strings: the start of each.
    Raises ``ValueError`` naming ``path`` where the text doesn't tokenize."""
    starts = [0]
    for line in text.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        raise ValueError(f"{path}: doesn't tokenize as Python ({e}), so the retired-terms lint can't tell hypothesis's "
                         f"`draw` from prose in it. Fix the file's syntax.") from e
    return frozenset(starts[t.start[0] - 1] + t.start[1] for t in tokens if t.type == tokenize.NAME and t.string == name)


def sites(text: str, path: str) -> list[tuple[int, str]]:
    """(line, what) of each site of a retired term in ``path``, ``text`` its content."""
    if not holds(path):
        return []
    return sites_of(text, terms()[1], path)


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


SENSES_EXAMPLE = """# Glossary
## Older names

| Old name, where it stays | Term |
|---|---|
| "draw" (noun, verb), "drawn unit"; `DrawIsCoin`, `--draw`, until renamed | the sample |
| Pearl's "debit" and "cap": "cap", "cap check", "over-cap" tile; `capOK`, `Device.cap`, `CAP`, until renamed | the bound |
"""


def said(text: str, path: str = "") -> list[tuple[int, str]]:
    """(line, the old name) of each site of ``SENSES_EXAMPLE``'s terms in ``text``, at ``path``; the default path isn't a
    Python file's, so hypothesis's ``draw`` is found by the rule for text that isn't Python."""
    by: dict[str, list[Term]] = {}
    for t in terms_of(SENSES_EXAMPLE):
        for a in t.anchors:
            by.setdefault(a, []).append(t)
    return [(line, w.split(" is retired")[0].strip("'\"")) for line, w in sites_of(text, by, path)]


def test_nvidia_smi_s_power_draw_is_another_sense_and_the_sample_beside_it_is_not() -> None:
    assert said("nvidia-smi --query-gpu=clocks.sm,power.draw,power.limit; POWER_DRAW, power_draw, its power draws") == []
    assert said("q = 'power.draw,clocks.sm'  # then challenger.draw(k), --draw 3 and the draws") == [
        (1, "draw"), (1, "--draw"), (1, "draws")]


HYPOTHESIS_EXAMPLE = '''from hypothesis import given, strategies as st


@st.composite
def cases(draw, n: int = 3) -> dict:
    k = draw(st.integers(0, n))
    rows = helper(draw, k)
    return {"draw": challenger.draw(k), "units": drawn, "n": draws, "flag": ["--draw"], "coin": DrawIsCoin}


def helper(draw, k):
    return draw(st.just(k))


@given(st.data())
def test_cases(data) -> None:
    data.draw(cases())
'''


def test_hypothesis_s_draw_is_another_sense_and_the_sample_beside_it_is_not() -> None:
    """A composite's first parameter and its uses in the body, and ``data.draw(``, are hypothesis's; in that body a string,
    an attribute, another form or a longer name is the sample, and so is a ``draw`` past the body (an undecorated helper's
    ``draw`` may be the sample: a test's parameter named for it)."""
    sample = [(8, "draw"), (8, "draw"), (8, "drawn"), (8, "draws"), (8, "--draw"), (8, "DrawIsCoin"), (11, "draw"),
              (12, "draw")]
    assert said(HYPOTHESIS_EXAMPLE) == sample
    bare = HYPOTHESIS_EXAMPLE.replace("from hypothesis import given, strategies as st",
                                      "from hypothesis.strategies import composite").replace("@st.composite", "@composite")
    assert said(bare) == sample
    assert sorted(said(HYPOTHESIS_EXAMPLE.replace("from hypothesis", "from elsewhere"))) == sorted(
        sample + [(5, "draw"), (6, "draw"), (7, "draw"), (17, "draw")])


def test_a_draw_in_a_comment_or_string_of_a_composite_body_is_still_the_sample() -> None:
    """In a Python file only code is hypothesis's: the word in a comment or a docstring inside a composite's body is
    prose."""
    text = ('from hypothesis import strategies as st\n\n\n@st.composite\ndef units(draw):\n'
            '    """Each unit we draw, one by one."""\n'
            '    k = draw(st.integers(0, 3))  # draw the count, then the units\n'
            '    return [draw(st.just(i)) for i in range(k)]  # draw\n')
    assert said(text, "verity/protocols/spot/tests/test_units.py") == [(6, "draw"), (7, "draw"), (8, "draw")]


def test_a_composite_shown_in_markdown_keeps_every_bare_draw_of_its_body() -> None:
    """A file that isn't Python is never read as Python, so an example in Markdown with prose after it (an apostrophe,
    which doesn't tokenize) doesn't stop the scan. In its body every bare ``draw`` is kept, the comment's too: this is the
    rule before the code-only one, on purpose, since such a text has no code to tell from prose."""
    text = ("# Units\n\n```python\nfrom hypothesis import strategies as st\n\n\n@st.composite\ndef units(draw):\n"
            "    k = draw(st.integers(0, 3))  # draw the count\n    return [draw(st.just(i)) for i in range(k)]\n```\n\n"
            "It's the sample that counts past the example: one draw.\n")
    assert said(text, "verity/protocols/spot/README.md") == [(13, "draw")]


def test_a_python_file_that_doesn_t_tokenize_stops_the_scan_naming_itself() -> None:
    path = "verity/protocols/spot/tests/test_broken.py"
    text = 'from hypothesis import given, strategies as st\n\n\n@given(st.data())\ndef test_it(data):\n    data.draw(st.just("x))\n'
    try:
        said(text, path)
    except ValueError as e:
        assert path in str(e) and "Fix the file's syntax" in str(e), e
    else:
        raise AssertionError(f"{path} doesn't tokenize, and the scan read it anyway")


def test_the_stacking_property_s_caps_and_nvidia_smi_s_compute_cap_are_other_senses_and_a_cap_beside_them_is_not() -> None:
    """The stacking property's caps by the names its Lean gives them, and nvidia-smi's field, anywhere; a "cap" beside
    them, and a longer name built on one, still count, in their directories too."""
    theirs = ("R_cap, R_cap_phase, R_cap_node, R_cap_node_by_class_peak, caps.total, `Gates.FoolingCap`; "
              'nvidia-smi --query-gpu=name,compute_cap; {"compute_cap": "12.0"}')
    assert said(theirs, "benchmarks/stacked_share/share.py") == []
    assert said("R_cap is the cap; R_capRows, caps.totals, FoolingCaps, a compute cap, compute_cap_max and its cap",
                "benchmarks/stacked_share/beta.py") == [(1, "cap"), (1, "cap"), (1, "caps"), (1, "Caps"), (1, "cap"),
                                                         (1, "cap"), (1, "cap")]


def test_a_cap_of_pous_or_poub_is_another_sense_and_pearl_s_beside_it_is_not() -> None:
    """PoUS's and PoUB's caps by their names anywhere, and any "cap" in their protocols' files; Pearl's spellings
    everywhere, in those files too, against the real table as well."""
    theirs = "the cap: `Params.schedule_cap`, SCHEDULE_CAP, `RTT_ALLOWANCE_CAP_NS`, cap_ns, `AUDIT_CAP`, `auditCap`; v.cap, caps"
    pearls = "\nPearl's: capOK, Device.cap, CAP, an over-cap tile, the cap check"
    pearl = [(2, "capOK"), (2, "Device.cap"), (2, "CAP"), (2, "over-cap"), (2, "cap check")]
    for path in ("verity/protocols/pous/verifier/audit.py", "verity/protocols/poub/verifier/audit.py"):
        assert said(theirs + pearls, path) == pearl
    assert said(theirs + pearls, "verity/protocols/pouw/verifier/pearl.py") == [(1, "cap"), (1, "cap"), (1, "caps"), *pearl]
    assert [w.split("'")[1] for _, w in sites(pearls, "verity/protocols/poub/verifier/audit.py")] == [o for _, o in pearl]


def test_a_cap_of_flock_s_claims_is_another_sense_and_pearl_s_beside_it_is_not() -> None:
    """The bound on an extractor's reruns and a commitment's Merkle cap: any "cap" where Lean states Flock's claims and
    proves its knowledge soundness and sampling over it. Pearl's spellings count there too, against the real table as
    well, and so does every "cap" in the rest of Flock's Lean, beside a kept directory and in PoUW's proofs."""
    theirs = "the cap: `D.cap`, `Numbers.cap`, cap₀, `Model.Cap D`, `IsCapTauZ`, recursiveCaps, capOfRoot, cap_pos; its caps"
    pearls = "\nPearl's: capOK, Device.cap, CAP, an over-cap tile, the cap check"
    pearl = [(2, "capOK"), (2, "Device.cap"), (2, "CAP"), (2, "over-cap"), (2, "cap check")]
    for path in ("security/Definitions/Claims/SamplingTable.lean", "security/Proofs/Flock/Claims/KS/KSTable.lean",
                 "security/Proofs/Flock/Claims/Numbers.lean", "security/Proofs/Flock/Claims/NumbersDeploy.lean",
                 "security/Proofs/Flock/SamplingGeneric/KSTable.lean"):
        assert said(theirs + pearls, path) == pearl, path
        assert [w.split("'")[1] for _, w in sites(pearls, path)] == [o for _, o in pearl], path
    counted = [(1, o) for o in ("cap", "cap", "cap", "cap", "Cap", "Cap", "Caps", "cap", "cap", "caps")]
    for path in ("security/Definitions/Flock/Compiled.lean", "security/Definitions/ClaimsOf.lean",
                 "security/Proofs/Flock/Claims/Aggregate.lean", "security/Proofs/Flock/Masked.lean",
                 "security/Proofs/Pouw/Claims/KS/Bound.lean", "verity/protocols/flock/verifier/Flock/Merkle.lean"):
        assert said(theirs + pearls, path) == counted + pearl, path


def test_two_terms_on_one_span_name_the_same_one_in_every_process() -> None:
    """The ratchet gate compares a base's sites with a head's as strings, each found in its own process, and a set of
    terms iterates in an order that differs between processes."""
    alive, said = [], set()
    for i in range(64):
        pair = [Term("Dampener*", "the slowdown", code=True), Term("*RemotePeriod", "monitoring period", code=True)]
        alive.append(pair)
        by: dict[str, list[Term]] = {}
        for t in pair[::-1] if i % 2 else pair:
            for a in t.anchors:
                by.setdefault(a, []).append(t)
        said |= set(sites_of("the DampenerRemotePeriod\n", by))
    assert said == {(1, "'DampenerRemotePeriod' is retired; say monitoring period (GLOSSARY.md)")}, sorted(said)


def test_two_terms_on_one_site_say_the_same_whatever_order_the_set_holds_them_in() -> None:
    class Placed(Term):
        """A term whose place in a small set is chosen: such a set gives its members in the order of their hashes."""

        def __init__(self, place: int, old: str, new: str):
            super().__init__(old, new, code=True)
            self.place = place

        def __hash__(self) -> int:
            return self.place

    def said(text: str, *rows: tuple[int, str, str]) -> list[tuple[int, str]]:
        by: dict[str, list[Term]] = {}
        for t in (Placed(*row) for row in rows):
            for a in t.anchors:
                by.setdefault(a, []).append(t)
        return sites_of(text, by)

    longer = [(1, "'OldName' is retired; say the longer row's (GLOSSARY.md)")]  # second by spelling: the length decides
    assert said("an OldName", (1, "*Name", "the shorter row's"), (2, "OldNa*", "the longer row's")) == longer
    assert said("an OldName", (2, "*Name", "the shorter row's"), (1, "OldNa*", "the longer row's")) == longer
    first = [(1, "'OldName' is retired; say the first by spelling (GLOSSARY.md)")]  # of one length: the spelling decides
    assert said("an OldName", (1, "OldN*", "the second by spelling"), (2, "*Name", "the first by spelling")) == first
    assert said("an OldName", (2, "OldN*", "the second by spelling"), (1, "*Name", "the first by spelling")) == first


def test_it_reads_what_lane_text_reads_and_skips_what_it_says() -> None:
    assert holds("verity/a.py") and holds("security/A.lean") and holds("README.md") and holds("a/b.toml")
    assert not holds("x.json") and not holds(GLOSSARY) and not holds("tools/move/x_map.toml")
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
