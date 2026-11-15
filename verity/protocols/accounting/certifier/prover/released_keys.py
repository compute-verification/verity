"""The keys a chain record releases: every JSON file a record releases holds only keys declared public by name, each
with its kind. The check is by key and not by bytes, so a value of any length is covered; the leak guard's pieces stay
beside it, for a private value copied under a public key.

    python3 released_keys.py OUT_DIR [--declared FILE]

The declared data (`DECLARED`, released_keys.json beside this file) holds the lists, one for each form of file a writer
writes, wherever a step copies it:

* ``files``: the record paths it rules, relative to the record's ``out/`` (globs: ``*`` and ``?`` inside one path
  segment, ``**`` any number of segments);
* ``form``: ``json``, one document, or ``jsonl``, one a line;
* ``keys``: each key the file may hold with its kind. A key is named by its path: the object keys from the document's
  root joined by ``/``, down to a value that is no object or list. Its kind is one of `KINDS`: protocol structure, a
  protocol output, a challenge or sample of the verifier's, or a hiding commitment. A key that is none of the four is
  not declared: its writer stops writing it;
* ``maps``: the one wildcard, ``*``, stands for a list's index, or, at a path ``maps`` names, for a map's own keys,
  which are public ids of the form ``maps`` gives there (a regular expression each id matches whole). A named key
  beside the wildcard is read before it.

A file is read strictly: a key the list doesn't name, a map's id that isn't of its form, an object where the list has a
list's index, a list where it has a map or a value, a value where it has deeper keys, and a document that doesn't parse
are each undeclared. An empty list or object and ``null`` hold no value and pass wherever they are. No entry excuses a
key: a list names it, or the file fails.

``not_yet`` names the record paths whose lists are not written yet (globs): a JSON file under one of them is counted
as not yet checked, by step, and never passed silently. A JSON file under no list and no ``not_yet`` glob fails. A file
that is neither ``.json`` nor ``.jsonl`` has no keys to read and is counted by its extension.

The guard (leak_guard.py ``--keys``) reads each file under a record's ``out/`` through `Tally` as it releases it and
moves a file that fails to the quarantine with its other finds. What a report that is itself released says of a
failure is its file and how many keys, never a key's name, which a writer may have made from a value; the quarantine's
report and this command name each key path with its fix.

Exit 0 when every file read holds declared keys only, 3 when one doesn't, 2 when the declared data is malformed.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

DECLARED = Path(__file__).resolve().with_name("released_keys.json")
SCHEMA = "network-traces/released-keys/v1"
KINDS = ("structure", "output", "challenge", "commitment")
FORMS = ("json", "jsonl")
SUFFIXES = (".json", ".jsonl")
TOP = "(top level)"
NOT_JSON, NO_ID = "(not json)", " (not an id)"


class Malformed(Exception):
    """The declared data can't be read as lists of keys."""


def glob(pattern: str) -> re.Pattern:
    """A record path glob as a pattern: ``*`` and ``?`` stay inside one segment, a ``**`` segment is any number."""
    parts = pattern.split("/")
    if "" in parts:
        raise Malformed(f"{pattern!r}: a glob is path segments joined by /")
    out = []
    for i, seg in enumerate(parts):
        last = i == len(parts) - 1
        if seg == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
            continue
        if "**" in seg:
            raise Malformed(f"{pattern!r}: ** is a whole segment")
        out.append("".join("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c) for c in seg)
                   + ("" if last else "/"))
    return re.compile("".join(out))


class Node:
    """A place in a list: the kind of the value there, the named keys below it, and what ``*`` stands for below it
    (`star`; a map's ids when `ids` is their form, else a list's index)."""
    __slots__ = ("kind", "named", "star", "ids")

    def __init__(self):
        self.kind: str | None = None
        self.named: dict[str, Node] = {}
        self.star: Node | None = None
        self.ids: re.Pattern | None = None


class Listing:
    """One list of the declared data."""

    def __init__(self, doc: dict, where: str):
        if not isinstance(doc, dict) or set(doc) - {"maps"} != {"name", "writer", "files", "form", "keys"}:
            raise Malformed(f"{where} needs a name, its writer, files, a form and keys, and at most maps beside them")
        self.name, self.writer, self.form = doc["name"], doc["writer"], doc["form"]
        where = f"{where} ({self.name!r})"
        if not (isinstance(self.name, str) and self.name and isinstance(self.writer, str) and self.writer):
            raise Malformed(f"{where}: its name and its writer are text")
        if self.form not in FORMS:
            raise Malformed(f"{where}: its form is one of {', '.join(FORMS)}, not {self.form!r}")
        if not (isinstance(doc["files"], list) and doc["files"] and all(isinstance(g, str) for g in doc["files"])):
            raise Malformed(f"{where}: its files are record path globs")
        if bad := [g for g in doc["files"] if not g.endswith(SUFFIXES)]:
            raise Malformed(f"{where}: {bad[0]!r} is no .json or .jsonl path, and only those have keys")
        self.globs = list(doc["files"])
        self.files = [glob(g) for g in self.globs]
        if not (isinstance(doc["keys"], dict) and doc["keys"]):
            raise Malformed(f"{where}: its keys are key paths, each with its kind")
        self.root = Node()
        for key, kind in doc["keys"].items():
            if kind not in KINDS:
                raise Malformed(f"{where}: {key!r} has the kind {kind!r}, which is none of {', '.join(KINDS)}. A key "
                                "of no kind is not declared: its writer stops writing it")
            node = self.root
            for seg in self._segments(key, where):
                if node.kind is not None:
                    raise Malformed(f"{where}: {key!r} lies under a key declared as a value")
                if seg == "*":
                    node.star = node.star or Node()
                    node = node.star
                else:
                    node = node.named.setdefault(seg, Node())
            if node.named or node.star is not None:
                raise Malformed(f"{where}: {key!r} is declared as a value and holds deeper keys too")
            node.kind = kind
        maps = doc.get("maps", {})
        if not isinstance(maps, dict):
            raise Malformed(f"{where}: its maps are key paths ending in *, each with its ids' form")
        for key, form in maps.items():
            segs = self._segments(key, where)
            node = self.root
            for seg in segs[:-1]:
                node = (node.star if seg == "*" else node.named.get(seg)) if node is not None else None
            if segs[-1] != "*" or node is None or node.star is None:
                raise Malformed(f"{where}: the map {key!r} is no path ending in * that its keys go through")
            try:
                node.ids = re.compile(form)
            except (re.error, TypeError) as e:
                raise Malformed(f"{where}: the map {key!r} needs its ids' form, a regular expression: {e}") from None
        self._no_names_beside_an_index(self.root, "", where)

    @staticmethod
    def _segments(key: str, where: str) -> list[str]:
        segs = key.split("/") if isinstance(key, str) else [""]
        if "" in segs:
            raise Malformed(f"{where}: {key!r} is no key path (keys joined by /)")
        return segs

    def _no_names_beside_an_index(self, node: Node, at: str, where: str) -> None:
        if node.star is not None and node.ids is None and node.named:
            raise Malformed(f"{where}: {at or 'the root'} holds named keys and a list's index: name the path in maps "
                            "if it is a map of ids")
        for seg, child in [*node.named.items(), *([("*", node.star)] if node.star is not None else [])]:
            self._no_names_beside_an_index(child, f"{at}/{seg}".lstrip("/"), where)

    def undeclared(self, data: bytes) -> dict[str, int]:
        """The key paths of `data` this list doesn't declare, each with how often it occurs (an index and a map's id
        written ``*``, a map's key that is no id of its form as the map's path and `NO_ID`); a document that doesn't
        parse is the one path `NOT_JSON`."""
        out: collections.Counter = collections.Counter()
        try:
            text = data.decode("utf-8")
            docs = ([json.loads(x) for x in text.splitlines() if x.strip()] if self.form == "jsonl"
                    else [json.loads(text)])
        except (ValueError, RecursionError):
            return {NOT_JSON: 1}
        for doc in docs:
            self._walk(self.root, doc, (), out)
        return dict(out)

    def _walk(self, node: Node, value, path: tuple[str, ...], out: collections.Counter) -> None:
        if value is None or value == {} or value == []:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                child = node.named.get(k)
                if child is None and node.ids is not None and node.ids.fullmatch(k):
                    child = node.star
                if child is None:
                    out["/".join((*path, "*")) + NO_ID if node.ids is not None else "/".join((*path, k))] += 1
                else:
                    self._walk(child, v, (*path, k if child is not node.star else "*"), out)
        elif isinstance(value, list):
            if node.star is None or node.ids is not None:
                out["/".join((*path, "*"))] += 1
            else:
                for v in value:
                    self._walk(node.star, v, (*path, "*"), out)
        elif node.kind is None:
            out["/".join(path) or "(the document)"] += 1

    def fix(self, key: str) -> str:
        if key == NOT_JSON:
            return (f"it doesn't parse as {self.form}, so its keys can't be read: its writer ({self.writer}) writes "
                    f"{self.form}")
        if key.endswith(NO_ID):
            return (f"a key of the map {key[:-len(NO_ID)]} is not an id of the form the list {self.name!r} of "
                    f"{DECLARED.name} gives its ids: its writer ({self.writer}) writes ids of that form, or the list "
                    "names the key beside the wildcard with its kind")
        return (f"key {key} is not declared public: give it its kind ({', '.join(KINDS)}) in the list {self.name!r} of "
                f"{DECLARED.name}, with its ids' form in maps if it is a map's, or stop its writer ({self.writer}) "
                "writing it")


class Declared:
    """The declared data: the lists, and the record paths whose lists are not written yet."""

    def __init__(self, doc: dict, source: str = DECLARED.name):
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or set(doc) != {"schema", "lists", "not_yet"}:
            raise Malformed(f"{source} is not {SCHEMA}: a schema, lists and not_yet")
        if not isinstance(doc["lists"], list):
            raise Malformed(f"{source}: lists is a list")
        self.lists = [Listing(d, f"{source} list {i}") for i, d in enumerate(doc["lists"])]
        for what, seen in (("name", [x.name for x in self.lists]), ("glob", [g for x in self.lists for g in x.globs])):
            if twice := sorted({s for s in seen if seen.count(s) > 1}):
                raise Malformed(f"{source}: the {what} {twice[0]!r} is in two lists")
        if not (isinstance(doc["not_yet"], list) and all(isinstance(g, str) for g in doc["not_yet"])):
            raise Malformed(f"{source}: not_yet is a list of record path globs")
        self.not_yet = [glob(g) for g in doc["not_yet"]]

    def listing(self, rel: str) -> Listing | None:
        """The list that rules record path `rel`; two that do are malformed data."""
        got = [x for x in self.lists if any(g.fullmatch(rel) for g in x.files)]
        if len(got) > 1:
            raise Malformed(f"{rel} is under two lists, {got[0].name!r} and {got[1].name!r}")
        return got[0] if got else None

    def waits(self, rel: str) -> bool:
        return any(g.fullmatch(rel) for g in self.not_yet)


def load(path: Path = DECLARED) -> Declared:
    try:
        return Declared(json.loads(path.read_text()), path.name)
    except (OSError, ValueError) as e:
        raise Malformed(f"{path}: {e}") from None


def step(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else TOP


class Tally:
    """What a pass over a record's ``out/`` found: the files checked, those not yet checked by step, those with no
    keys by extension, those a guard met outside ``out/``, and each file that failed."""

    def __init__(self, declared: Declared):
        self.declared = declared
        self.checked = 0
        self.not_yet: collections.Counter = collections.Counter()
        self.no_keys: collections.Counter = collections.Counter()
        self.outside = 0
        self.failed: dict[str, list[dict]] = {}
        self._counted: set[str] = set()

    def claim(self, rel: str) -> Listing | None:
        """Record path `rel` counted, once however often it is read: the list that rules it, or None for a file
        that waits or has no keys. A JSON file under no list and no ``not_yet`` glob fails here."""
        listing = self.declared.listing(rel)
        if rel in self._counted:
            return listing
        self._counted.add(rel)
        if listing is not None:
            self.checked += 1
        elif not rel.endswith(SUFFIXES):
            self.no_keys[Path(rel).suffix or "(none)"] += 1
        elif self.declared.waits(rel):
            self.not_yet[step(rel)] += 1
        else:
            self.failed[rel] = [{"key": "(the file)", "count": 1, "fix": (
                f"no list of {DECLARED.name} names this file: write its list, each key with its kind "
                f"({', '.join(KINDS)}), or stop releasing it")}]
        return listing

    def read(self, rel: str, path: Path) -> int:
        """Record path `rel` (relative to ``out/``), the file at `path`: how many of its keys are not declared, 0 for
        a file that passes, waits or has no keys."""
        listing = self.claim(rel)
        if listing is not None and (bad := listing.undeclared(path.read_bytes())):
            self.failed[rel] = [{"key": k, "count": n, "fix": listing.fix(k)} for k, n in sorted(bad.items())]
        return len(self.failed.get(rel, ()))

    def summary(self) -> dict:
        """The counts a released report may hold: no key's name."""
        return {"declared": SCHEMA, "checked": self.checked, "not_yet": dict(sorted(self.not_yet.items())),
                "no_keys": dict(sorted(self.no_keys.items())), "outside": self.outside, "undeclared": len(self.failed)}


def line(summary: dict) -> str:
    """A pass's counts (`Tally.summary`) in a line."""
    waits = ", ".join(f"{s} {n}" for s, n in summary["not_yet"].items())
    bare = ", ".join(f"{s} {n}" for s, n in summary["no_keys"].items())
    return (f"{summary['checked']} files checked by key, {summary['undeclared']} with a key not declared public; "
            f"{sum(summary['not_yet'].values())} not yet checked{f' ({waits})' if waits else ''}; "
            f"{sum(summary['no_keys'].values())} with no keys{f' ({bare})' if bare else ''}"
            + (f"; {summary['outside']} outside out/, not read by key" if summary["outside"] else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="a record's out/")
    ap.add_argument("--declared", type=Path, default=DECLARED)
    a = ap.parse_args(argv)
    try:
        if not a.out.is_dir():
            raise Malformed(f"no directory {a.out}")
        tally = Tally(load(a.declared))
        for p in sorted(a.out.rglob("*")):
            if p.is_file() and not p.is_symlink():
                tally.read(p.relative_to(a.out).as_posix(), p)
    except Malformed as e:
        print(f"released_keys: refused: {e}", file=sys.stderr)
        return 2
    print(f"released keys: {line(tally.summary())}")
    for rel, bad in tally.failed.items():
        for b in bad:
            print(f"  {rel}: {b['fix']}" + (f" ({b['count']} times)" if b["count"] > 1 else ""))
    return 3 if tally.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
