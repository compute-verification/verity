#!/usr/bin/env python3
"""`envnames.py FILE...`: the names of the variables each env file sets, and never a value.

    python3 tools/agent-guard/envnames.py FILE...
    ssh HOST python3 - FILE... < tools/agent-guard/envnames.py      # on another machine; a glob in FILE expands there

A FILE is an env file when every line is blank, a `#` comment, or `NAME=VALUE` (`export NAME=VALUE` too), and its line is then
`FILE: NAME NAME ...`. Any other line refuses the whole file, and nothing of a refused file is printed: a glob over a secrets
directory also matches its key files, and a filter that cuts each line at its `=` (`cut -d= -f1`, `sed 's/=.*//'`) prints
every line that has none. Two kinds of line read as an assignment and are refused as well, since their "name" would be a
value's text: the line after a value that leaves a quote open or ends in a backslash (a multi-line value's next line), and a
name with no value or only `=` after it (a line of padded base64). Exits 0 when every FILE was listed, 1 when one was refused
or couldn't be read. Standard library only, one file, so it runs from stdin on a machine without a checkout.
"""
import os
import re
import sys

USAGE = "usage: python3 envnames.py FILE...    (on another machine: ssh HOST python3 - FILE... < tools/agent-guard/envnames.py)"
BLANK = re.compile(r"\s*(#.*)?")
ASSIGNMENT = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(?!=*\s*$)(.*)")
#: a value that ends on its line: plain text, escaped characters, and quoted strings that close
WHOLE = re.compile(r"""(?:[^'"\\]|\\.|'[^']*'|"(?:[^"\\]|\\.)*")*""")


def names(path):
    """(the names `path` sets, 0), or (None, the number of its first line that is not NAME=VALUE); OSError when it can't be read."""
    found, continued = [], False
    with open(path, "rb") as f:
        for number, raw in enumerate(f, 1):
            line = raw.decode("utf-8", "replace").rstrip("\n")
            m = None if continued else ASSIGNMENT.fullmatch(line)
            if m:
                found.append(m[1])
                continued = WHOLE.fullmatch(m[2].rstrip("\r")) is None
            elif continued or not BLANK.fullmatch(line):
                return None, number
    return found, 0


def main(argv):
    if not argv:
        print(USAGE, file=sys.stderr)
        return 2
    status = 0
    for path in argv:
        try:
            if not os.path.isfile(path):
                raise OSError(path)
            found, bad = names(path)
        except OSError:
            print(f"{path}: not a readable file")
            status = 1
            continue
        if found is None:
            print(f"{path}: refused: line {bad} is not NAME=VALUE, so this is not an env file; nothing printed")
            status = 1
        else:
            print(f"{path}: {' '.join(found) or '(no variables)'}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
