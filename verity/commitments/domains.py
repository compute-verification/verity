"""One table of domain separators: every versioned string a protocol hashes, derives from or commits under, or names its
records, schemes and protocols by. The table is the Lean's, `Identity.Domains`
(`verity/commitments/identity/Identity/Domains.lean` and the parts it imports), which this module reads at import, beside
`identity.tagged_sha512`, the tagged SHA-512 most of them key; `domains.pyi`, generated from the same files, names its
entries for type checkers.

A constant's name is in today's words. Its bytes are what committed formats, pinned vectors and recorded runs carry, so
they keep the words of their version: a PoUW period's tags say "window", its sampling's say "draw", the network
attester's oldest say "warden". Bytes never change in place: a format in today's words is a new version, a new entry
beside the old one, which stays while anything reads it. An entry carries no terminator; a site that hashes a tag
NUL-terminated appends the NUL.

Code reads a separator from here or from the Lean, never as a literal of its own. ``tests/test_domains.py`` holds this
module's reading and its stub to the Lean, entry for entry; refuses a literal of the table in the code of a protocol that
reads it (its ``READERS``, PoUW first); and refuses an entry that nothing reads and whose bytes appear nowhere else. The entries under
"Not yet read from here" are other protocols' separators, listed so that each has one name before its owner's code
switches to reading it. Flock's statement versions keep their tags in `Flock.Tags`, one structure per version.
"""

import re as _re
from pathlib import Path as _Path

#: the table's file; the wheel ships it, and the parts it imports, beside this module
_LEAN = _Path(__file__).parent / "identity/Identity/Domains.lean"
_ENTRY = _re.compile(r'abbrev ([A-Z][A-Z0-9_]*) : String :=(?: "([^"\\]*)")?')
_VALUE = _re.compile(r'"([^"\\]*)"')
_PART = _re.compile(r"public import Identity\.Domains\.([A-Za-z]+)")
_HEADER = ("", "module", "@[expose] public section", "namespace Domains", "end Domains")


def _read(top: _Path = _LEAN) -> dict[str, str]:
    """Every entry of the Lean table at ``top``, in its order: the entries of the parts it imports, in its order, then its
    own. Refuses, by file and line, a line that is none of a module header, a part's import, a comment or an entry."""
    parts: list[str] = []
    own = _entries(top, parts)
    table: dict[str, str] = {}
    for name, value in [e for p in parts for e in _entries(top.with_suffix("") / f"{p}.lean", [])] + own:
        if name in table:
            raise ValueError(f"{top}: the Lean table names {name} twice")
        table[name] = value
    return table


def _entries(path: _Path, parts: list[str]) -> list[tuple[str, str]]:
    entries, comment, name = [], False, None
    for n, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if comment or s.startswith("/-"):
            body = s if comment else s[2:]
            if "-/" in body[:-2]:
                raise ValueError(f"{path}:{n}: a comment of the domain table closes before the end of its line: {s[:120]}")
            comment = not body.endswith("-/")
        elif name is not None:
            if not (m := _VALUE.fullmatch(s)):
                raise ValueError(f"{path}:{n}: the value of {name}, begun on the line above, is not a string alone on its "
                                 f"line: {s[:120]}")
            entries.append((name, m[1]))
            name = None
        elif s in _HEADER or s.startswith("--"):
            pass
        elif m := _PART.fullmatch(s):
            parts.append(m[1])
        elif m := _ENTRY.fullmatch(s):
            if m[2] is None:
                name = m[1]
            else:
                entries.append((m[1], m[2]))
        else:
            raise ValueError(f"{path}:{n}: not a line of the domain table this module reads (a module header, a part's "
                             f"import, a comment, or `abbrev NAME : String := \"...\"`, its string on the same line or "
                             f"the next): {s[:120]}")
    if comment or name is not None:
        raise ValueError(f"{path}: ends inside {'a comment' if comment else f'the entry {name}'}")
    return entries


#: every entry by name, in the Lean's order
TABLE: dict[str, str] = _read()
globals().update(TABLE)
