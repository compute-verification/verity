"""`public_lock`: a lock (`lean-audit.json`) as the public monorepo's export ships it (`tools/verity/check/public_export.py`,
which loads this file by path). Stdlib only.

A lock names Lean declarations, namespaces and modules, and Lake packages. The export copies only some of the tree, so
the lock it ships states only what the copied files hold:

* an entry stays only if the copied files have what it is named after (`ENTRIES`: a guarantee, a premise, a module's
  record, a rule over the modules under a prefix), and goes whole otherwise, with everything its record holds;
* an entry that lists witnesses (`WITNESSES`: theorems beside the declaration it is named after, which nothing rests on)
  goes with a witness the copied files lack: it would say of the public tree what no file there shows;
* a runner's entry goes with a file of the command that writes its data (`GENERATORS`) the copied files lack: the public
  tree can't make that data, so the run isn't one it can repeat;
* a premise's entry goes when the guarantees that took it are all gone (`PREMISES`, `TAKES`): the Lean check fails a
  premise no listed guarantee takes, and the lock's own records say which those are;
* a kept entry's record may cite other names (`CITES`: the premises a guarantee takes, a deployed guarantee's gap). Those
  are never dropped: a public statement doesn't rest on what the export leaves out, so a cited name the copied files
  lack refuses the lock (`missing`);
* no entry carries its `owner`, `issue` or `due` (`LANE_FIELDS`), and the sections that hold only those go
  (`LANE_SECTIONS`), so the lock is one the Lean check reads as unowned (`check.py`'s `unowned_lock`), and the `policy`
  of each of the check's controls loses them too (`public_controls`);
* everything else stays as it stands: a kept entry's record (its hashes, its reasons), a `reads` group, which is its
  module's whole, and the sections that name nothing of the repository's Lean (`OPAQUE`). A reason is never reworded
  here: what a kept entry's text still holds that the public tree shouldn't is found (`leftovers`), and the export
  refuses the lock by the entry that holds it (`refusal`) until the lock it comes from is reworded.

A `reads` group's records are of what its module declares in the whole tree, so the exported tree's own check writes them
anew from its build. That lock is held to the one it comes from (`restated`): each guarantee and premise it lists has the
source's record. A statement the copied spec holds whose theorem isn't copied is listed by one reason (`unstated`), and an
entry that build shows to be of nothing in the exported tree goes (`STALE`, `unlisted`).

What the copied files have is the caller's to say, one name at a time: ``has(kind, name)``, for the kinds below. A section
no rule covers refuses the lock (`unruled`), so a section added to the lock's schema gets its rule here before a lock
holding it ships (`tests/test_public_boundary.py` holds the rules to `check.ORDER`).
"""
from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

#: a declaration, by its full name
DECLARATION = "declaration"
#: a compiled constant: a declaration, or one Lean names below the declaration it is local to (a `where` function)
CONSTANT = "constant"
#: a namespace declarations are under
NAMESPACE = "namespace"
#: a module of the repository's Lake packages, by its own name: the one file
MODULE = "module"
#: a prefix of modules: a module and those under it, of which the copied files have some
PREFIX = "prefix"
#: a module prefix a layer may import: the repository's, or a dependency's (Mathlib, the toolchain)
IMPORT = "import"
#: a Lake package, by its path from the lock's own
PACKAGE = "package"
#: a command's argument: a path from the package's directory, or from the repository's as `{repo}/PATH`, held unless it
#: names a tracked file or directory the export copies nothing of
ARGUMENT = "argument"

#: where a lock holds entries, and the kind of name each is under: a dict's keys, a list's items, or the one name a
#: section is. `*` stands for every entry of the section before it. A section the Lean check matches a module's name
#: against as a prefix (`check.under`: a library's root, a layer, an exemption) is of prefixes; one it reads as the
#: module itself (a runner, a module that runs code as it compiles, a layer's exception, a `reads` group) is of modules.
ENTRIES = (
    ("roots", PREFIX), ("proved_in", PACKAGE), ("exempt", PREFIX), ("runs", MODULE), ("assumptions_module", PREFIX),
    ("assumptions", DECLARATION), ("domain", DECLARATION), ("open", DECLARATION),
    ("trust.base", DECLARATION), ("trust.extra", DECLARATION), ("deployed", DECLARATION), ("trust_gap", DECLARATION),
    ("meaning", PREFIX), ("escapes", CONSTANT),
    ("claims.modules", PREFIX), ("claims.witnesses", DECLARATION), ("claims.not", DECLARATION),
    ("claims.generic", DECLARATION),
    ("guarantee_lemmas.witnesses", DECLARATION), ("guarantee_lemmas.not", DECLARATION),
    ("witnesses", DECLARATION), ("unwitnessed", DECLARATION), ("witnesses_assumed", DECLARATION),
    ("compile_time", MODULE), ("layers", PREFIX), ("layers.*.imports", IMPORT), ("layers.*.except", MODULE),
    ("reads_exempt", PREFIX),
    ("trees.definitions", PREFIX), ("trees.properties", PREFIX), ("trees.proofs", PREFIX), ("trees.except", PREFIX),
    ("four_names", NAMESPACE), ("four_names_except", DECLARATION), ("guarantees", DECLARATION),
    ("shape.premises", PREFIX), ("shape.statements", PREFIX), ("shape.properties", DECLARATION),
    ("shape.frozen", DECLARATION), ("shape.conjectures", DECLARATION), ("shape.events", DECLARATION),
    ("shape.inline", DECLARATION),
    ("reads", MODULE),
)
#: where an entry lists its witnesses, `*` standing for the entry: it goes with one the copied files lack
WITNESSES = (
    ("witnesses.*.honest", DECLARATION), ("witnesses.*.nontrivial", DECLARATION),
    ("claims.witnesses.*.fails", DECLARATION), ("claims.witnesses.*.meets", DECLARATION),
    ("guarantee_lemmas.witnesses.*.fails", DECLARATION), ("guarantee_lemmas.witnesses.*.meets", DECLARATION),
    ("guarantee_lemmas.witnesses.*.by", DECLARATION),
)
#: where a runner's entry gives the command that writes its data, `*` standing for the entry: it goes with a file of that
#: command the copied files lack
GENERATORS = (("runs.*.generate", ARGUMENT),)
#: where a kept entry's record cites other names, which the copied files must have. What a statement reads
#: (`premises.reads`) includes the constants Lean generates below a declaration (a structure's `mk`, a type's `rec`).
CITES = (
    ("domain.*.cites", DECLARATION), ("trust.extra.*", DECLARATION), ("trust_gap.*", DECLARATION),
    ("claims.generic.*", DECLARATION),
    ("guarantees.*.assumptions", DECLARATION), ("guarantees.*.premises.assumptions", DECLARATION),
    ("guarantees.*.premises.domain", DECLARATION), ("guarantees.*.premises.open", DECLARATION),
    ("guarantees.*.premises.reads", CONSTANT),
    ("shape.properties.*.rests_on", DECLARATION),
    ("verifier.entries", DECLARATION), ("verifier.stop", DECLARATION), ("verifier.hash", NAMESPACE),
)
#: the sections that list premises: an entry of one goes when no kept guarantee takes it and a dropped one did
PREMISES = ("assumptions", "domain", "open", "witnesses_assumed")
#: where a guarantee's record lists the premises it takes, as they stand
TAKES = ("assumptions", "premises.assumptions", "premises.domain", "premises.open")
#: the rule, in what `public` reports it dropped, of a premise only dropped guarantees took: `UNTAKEN.<section>`
UNTAKEN = "untaken"
#: the reason of the entry the export lists a public statement under when its theorem stays behind (`unstated`)
UNSTATED = "unstated: no theorem in this tree proves it"
#: the sections that name nothing of the repository's Lean: the schema's version, the allowed axioms, and what the
#: dependencies hold (their digests, their modules that may name a bypass, the statements watched in them, their
#: constants the verifier may reach)
OPAQUE = ("format", "axioms", "dependencies", "dependency_unchecked", "upstream", "verifier.allow")
#: the lane's bookkeeping on an entry: who answers for it, and the issue and date that discharge an open premise
LANE_FIELDS = ("owner", "issue", "due")
#: the sections whose entries carry `LANE_FIELDS`
OWNED = ("guarantees", "assumptions", "domain", "open")
#: the sections that hold only `LANE_FIELDS`' defaults
LANE_SECTIONS = ("default_owner", "open_default_issue", "open_seed")
#: the files of the Lean check's controls, from the check's own directory (`check.py`'s `CONTROLS`): each holds
#: {control: {..., "policy": the sections of its fixture's lock the control lays over them}}
CONTROLS = ("tests/fixtures/violations.json", "tests/fixtures/module-violations.json")
#: a control's key saying it needs an owned lock: its point is an owner's or a ticket's rule, so the export leaves it out
NEEDS_OWNED = "owned"
#: how many names a refusal shows
SHOWN = 20
#: a dotted name, as a reason spells a declaration or a module
NAME = re.compile(r"[A-Za-z_][\w']*(?:\.[A-Za-z_][\w']*)+")
#: a path of two components or more, as a reason spells a file or a directory
PATH = re.compile(r"[\w.-]+(?:/[\w.-]+)+")
#: every section a rule covers, a section of subsections by each subsection
RULED = frozenset({path.partition(".*")[0] for path, _ in (*ENTRIES, *WITNESSES, *GENERATORS, *CITES)}
                  | {*OPAQUE, *LANE_SECTIONS})
#: the sections of subsections
NESTED = frozenset(s.partition(".")[0] for s in RULED if "." in s)


def _places(node, parts: list[str], trail: str = "") -> Iterator[tuple[dict, str, str]]:
    """(holder, key, where) of each value ``parts`` reach below ``node``, `*` standing for every key there."""
    if not isinstance(node, dict):
        return
    for k in list(node) if parts[0] == "*" else [parts[0]] if parts[0] in node else []:
        where = f"{trail}[{k}]" if parts[0] == "*" else f"{trail}.{k}".lstrip(".")
        if len(parts) > 1:
            yield from _places(node[k], parts[1:], where)
        else:
            yield node, k, where


def _names(value) -> list[str]:
    """The names a place holds: a dict's keys, a list's strings, or the one string."""
    if isinstance(value, str):
        return [value]
    return [n for n in value if isinstance(n, str)] if isinstance(value, (dict, list)) else []


def _stated(lock: dict, rules) -> Iterator[tuple[str, str, str, str]]:
    """(path, where, kind, name) of every name ``lock`` holds under ``rules``' paths."""
    for path, kind in rules:
        for holder, key, where in _places(lock, path.split(".")):
            for name in _names(holder[key]):
                yield path, where, kind, name


def named(lock: dict) -> Iterator[tuple[str, str, str]]:
    """(where, kind, name) of every name ``lock`` states: its entries' (`ENTRIES`), and those they list as witnesses
    (`WITNESSES`), run to write their data (`GENERATORS`) and cite (`CITES`)."""
    for _, where, kind, name in _stated(lock, (*ENTRIES, *WITNESSES, *GENERATORS, *CITES)):
        yield where, kind, name


def counts(lock: dict) -> dict[str, int]:
    """{`ENTRIES` path: how many entries ``lock`` holds there}, for each path it holds one at."""
    out: dict[str, int] = {}
    for path, *_ in _stated(lock, ENTRIES):
        out[path] = out.get(path, 0) + 1
    return out


def unruled(lock: dict) -> list[str]:
    """The sections of ``lock`` no rule covers (`RULED`), a subsection as `section.subsection`."""
    out = []
    for k, v in lock.items():
        if k in NESTED:
            out += [f"{k}.{s}" for s in v if f"{k}.{s}" not in RULED] if isinstance(v, dict) else [k]
        elif k not in RULED:
            out.append(k)
    return out


def lane_fields(node, where: str = "") -> Iterator[str]:
    """Where ``node`` holds a lane field, at any depth, or as the lock a section of them."""
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{where}.{k}".lstrip(".")
            if k in LANE_FIELDS or not where and k in LANE_SECTIONS:
                yield here
            yield from lane_fields(v, here)
    elif isinstance(node, list):
        for v in node:
            yield from lane_fields(v, where)


def entries(lock: dict) -> Iterator[tuple[str, str, object]]:
    """(section, entry, what it holds) for all of ``lock``: each key of a section that is a dict, with its value under that
    key, and a section that is none whole, as the entry `""`. A section of subsections (`NESTED`) is read by subsection
    (`shape.frozen`)."""
    for k, value in lock.items():
        nested = k in NESTED and isinstance(value, dict)
        for section, v in [(f"{k}.{sub}", v) for sub, v in value.items()] if nested else [(k, value)]:
            if isinstance(v, dict):
                for entry, record in v.items():
                    yield section, entry, {entry: record}
            else:
                yield section, "", v


def _strings(node) -> Iterator[str]:
    """Every string at or below ``node``: a dict's keys too."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def names(text: str) -> list[str]:
    """The dotted names ``text`` spells (`NAME`), each once, in the order it first spells them."""
    return list(dict.fromkeys(NAME.findall(text)))


def paths(text: str) -> list[str]:
    """The paths ``text`` spells (`PATH`), without a sentence's full stop or a leading `./`, each once, in order."""
    return list(dict.fromkeys(p for m in PATH.findall(text) if (p := m.strip(".").lstrip("/"))))


def leftovers(lock: dict, find: Callable[[str], Iterable[tuple[str, str]]]) -> list[tuple[str, str, str, str]]:
    """(section, entry, kind, text) for each (kind, text) ``find`` reports in a string of ``lock``, by the entry that
    holds the string (`entries`): what a kept entry's key, reason or record still spells. The caller says what counts."""
    return [(section, entry, kind, text) for section, entry, held in entries(lock)
            for s in _strings(held) for kind, text in find(s)]


def refusal(left: list[tuple[str, str, str, str, str]]) -> str:
    """What the export says of locks whose kept text still holds ``left``, each (the lock's file, section, entry, kind,
    text) as `leftovers` names it: every one of the first `SHOWN`, and the fix."""
    more = f"; and {len(left) - SHOWN} more" if len(left) > SHOWN else ""
    shown = "; ".join(f"{file}: {section}{f'[{entry}]' if entry else ''} holds {kind} {text!r}"
                      for file, section, entry, kind, text in left[:SHOWN])
    return (f"public_lock: {len(left)} place(s) where a lock's kept text holds lane text, or names what the export leaves "
            f"out: {shown}{more}. The export never rewords a lock, so reword each in the lock itself: drop the citation, "
            "the attribution or the name and keep the sentence, saying what is true now. An entry that can't be stated "
            "without what it names is made public with it or private with it (`public_boundary.toml`); one that a "
            "change on its way removes is listed there under `[[residue]]`, with that change as its `until`")


def public_controls(controls: dict) -> tuple[dict, list[str]]:
    """``controls`` (a file of `CONTROLS`) as the export ships it, and the controls it leaves out. The fixture's lock ships
    unowned (`public`), and a control's `policy` is laid over it, so each `policy` loses its `LANE_FIELDS` and
    `LANE_SECTIONS` too: the lock a control builds is unowned as the clean one is. A control marked `NEEDS_OWNED` is one
    of an owner's or a ticket's rule, which an unowned lock isn't held to, and is left out. `ValueError` on a kept
    control whose expected failure still spells a lane field."""
    out, gone = {}, sorted(name for name, c in controls.items() if c.get(NEEDS_OWNED))
    for name, control in controls.items():
        if name in gone:
            continue
        out[name] = control = copy.deepcopy(control)
        if isinstance(policy := control.get("policy"), dict):
            for section in LANE_SECTIONS:
                policy.pop(section, None)
            for section in OWNED:
                for record in policy[section].values() if isinstance(policy.get(section), dict) else ():
                    if isinstance(record, dict):
                        for field in LANE_FIELDS:
                            record.pop(field, None)
    spelled = re.compile(r'"(?:%s)": ' % "|".join(LANE_FIELDS))
    if left := sorted(name for name, c in out.items() for text in (c.get("fails", ""), *c.get("not", []))
                      if spelled.search(text)):
        raise ValueError(f"public_lock: the expected failure of the controls {', '.join(left)} spells an owner, an issue or "
                         f"a due date, which the lock the export ships doesn't hold: end the expected text before it, or, "
                         f"if the control's point is an owner's or a ticket's rule, mark it `\"{NEEDS_OWNED}\": true`, "
                         f"and the export leaves it out")
    return out, gone


def missing(lock: dict, has: Callable[[str, str], bool]) -> list[str]:
    """What ``lock`` holds that a public lock doesn't: `WHERE: NAME` for each name it states (`named`) that the copied
    files lack, and `WHERE: a lane field` for each `owner`, `issue` or `due` at any depth and each of `LANE_SECTIONS`.
    A lock `public` returned has none, and the export refuses one it wrote that does."""
    return [f"{where}: {name}" for where, kind, name in named(lock) if not has(kind, name)] + \
        [f"{where}: a lane field" for where in lane_fields(lock)]


def restated(source: dict, public: dict) -> list[str]:
    """What ``public`` states otherwise than ``source``, the lock it was exported from, once the Lean check has written
    ``public``'s records anew on the exported tree (`check.py --update`, which settles every difference by recording
    it): `NAME: why` for each guarantee and each premise (`PREMISES`' sections that hold records) ``public`` lists
    that ``source`` doesn't, or whose record differs from ``source``'s in anything but `LANE_FIELDS`, naming the fields.
    A guarantee's record is its statement's digest and what it takes, so a public lock with none of these states each
    guarantee as the lock it came from does."""
    def record(value):
        return {k: v for k, v in value.items() if k not in LANE_FIELDS} if isinstance(value, dict) else value

    out = []
    for section in ("guarantees", *PREMISES):
        theirs, ours = source.get(section), public.get(section)
        for name, mine in sorted(ours.items()) if isinstance(ours, dict) else ():
            if not isinstance(theirs, dict) or name not in theirs:
                out.append(f"{section}[{name}]: the lock it was exported from doesn't list it")
            elif (a := record(theirs[name])) != (b := record(mine)):
                both = isinstance(a, dict) and isinstance(b, dict)
                fields = sorted(k for k in {*a, *b} if a.get(k) != b.get(k)) if both else []
                out.append(f"{section}[{name}]: its record differs" + (f" in {', '.join(fields)}" if fields else ""))
    return out


#: the failures of the Lean check that say a kept entry is of nothing in the exported tree, each with where its entry is
#: listed. Only a build of that tree shows them (which premise a witness takes, which prefix a guarantee reads outside
#: the spec, which exception the trees rule still needs, which conjecture a guarantee reads), so no rule of `ENTRIES`
#: drops these entries
STALE = {
    "witnesses.assumed-stale": ("witnesses_assumed",),
    "reads.exempt-stale": ("reads_exempt",),
    "trees.stale-except": ("trees", "except"),
    "shape.conjecture-unnamed": ("shape", "conjectures"),
}


def unlisted(lock: dict, stale: Iterable[tuple[str, str]]) -> dict:
    """``lock`` less each entry of ``stale``: (code, name), a failure of `STALE` the exported tree's check reports of
    the entry ``name``. ValueError on an entry the lock doesn't hold where the code says it is."""
    out = dict(lock)
    for code, name in stale:
        *outer, last = STALE[code]
        at = out
        for key in outer:
            at[key] = dict(at.get(key) or {})
            at = at[key]
        if not isinstance(held := at.get(last), dict) or name not in held:
            raise ValueError(f"public_lock: {'.'.join(STALE[code])} holds no entry {name!r}, which a `{code}` failure names")
        at[last] = {k: v for k, v in held.items() if k != name}
    return out


def unstated(lock: dict, names: Iterable[str]) -> dict:
    """``lock`` with each of ``names``, a statement of the public spec that no guarantee of the public tree states (the
    Lean check's `shape.unstated`: its theorem is in a file the export leaves out), listed as one (`shape.frozen`, the
    category `unstated`, the reason `UNSTATED`). The reason says only what the public tree shows: the public check
    claims what it replays, so it never says the statement is proved elsewhere."""
    if not (names := sorted(set(names))):
        return lock
    shape = dict(lock.get("shape") or {})
    frozen = dict(shape.get("frozen") or {})
    if held := [n for n in names if n in frozen]:
        raise ValueError(f"public_lock: {', '.join(held)}: `shape.frozen` already holds an entry, so it is no unstated "
                         f"statement the export may list")
    return {**lock, "shape": {**shape, "frozen": {**frozen, **{n: UNSTATED for n in names}}}}


def taken(lock: dict) -> set[str]:
    """The premises the guarantees ``lock`` lists take as they stand, by their records (`TAKES`)."""
    records = lock.get("guarantees")
    return {n for record in (records.values() if isinstance(records, dict) else ())
            for path in TAKES for holder, key, _ in _places(record, path.split(".")) for n in _names(holder[key])}


def public(lock: dict, has: Callable[[str, str], bool]) -> tuple[dict, dict[str, list[str]]]:
    """``lock`` as the export ships it, and {the path of the rule: the names of the entries it dropped}. Each entry named
    after what the copied files lack (``has(kind, name)`` false) is gone (`ENTRIES`), and so is each that lists a
    witness they lack (`WITNESSES`) or a generator's file they lack (`GENERATORS`), and each premise that only dropped
    guarantees took (`PREMISES`, reported under `UNTAKEN`); no entry carries `LANE_FIELDS`, and
    `LANE_SECTIONS` are gone. `ValueError` when a
    section has no rule (`unruled`), or when a kept entry still cites a name the copied files lack (`CITES`)."""
    if bad := unruled(lock):
        raise ValueError(f"public_lock: no rule says what {', '.join(bad)} names, so the export can't tell what of it is "
                         f"public: give each its rule in {Path(__file__).name} (`ENTRIES`, `WITNESSES`, `CITES` or `OPAQUE`)")
    out = copy.deepcopy({k: v for k, v in lock.items() if k not in LANE_SECTIONS})
    dropped: dict[str, list[str]] = {}
    for path, kind in ENTRIES:
        for holder, key, _ in _places(out, path.split(".")):
            value = holder[key]
            if not (gone := {n for n in _names(value) if not has(kind, n)}):
                continue
            dropped.setdefault(path, []).extend(n for n in _names(value) if n in gone)
            if isinstance(value, dict):
                holder[key] = {n: v for n, v in value.items() if n not in gone}
            elif isinstance(value, list):
                holder[key] = [n for n in value if not (isinstance(n, str) and n in gone)]
            else:
                del holder[key]
    for path, kind in (*WITNESSES, *GENERATORS):
        section, _, listed = path.partition(".*.")
        for holder, key, _ in _places(out, section.split(".")):
            for entry, record in list(holder[key].items()) if isinstance(holder[key], dict) else ():
                if any(not has(kind, n) for h, k, _ in _places(record, listed.split(".")) for n in _names(h[k])):
                    del holder[key][entry]
                    dropped.setdefault(path, []).append(entry)
    if untaken := taken(lock) - taken(out):
        for section in PREMISES:
            value = out.get(section)
            if gone := [n for n in _names(value) if n in untaken]:
                out[section] = ({n: v for n, v in value.items() if n not in untaken} if isinstance(value, dict)
                                else [n for n in value if n not in untaken])
                dropped[f"{UNTAKEN}.{section}"] = gone
    for section in OWNED:
        for record in out[section].values() if isinstance(out.get(section), dict) else ():
            if isinstance(record, dict):
                for field in LANE_FIELDS:
                    record.pop(field, None)
    if left := missing(out, has):
        more = f"; and {len(left) - SHOWN} more" if len(left) > SHOWN else ""
        raise ValueError("public_lock: entries the export keeps cite what it doesn't copy, so they don't hold in the public "
                         f"tree: {'; '.join(left[:SHOWN])}{more}. Make what each cites public, or the entry that cites it "
                         "private (`public_boundary.toml`), or drop the citation from the lock")
    return out, dropped
