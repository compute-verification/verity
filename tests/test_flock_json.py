"""C-Flock's Lean reads JSON only through `Flock.JsonParse.parse`, the total parser the kernel can evaluate.

`Lean.Json.parse` is a `partial def`, an opaque constant to the kernel, so no proof can show that a text it reads parses;
`JsonParse.parse` reads the same language, with the same values and refusal strings (`Flock.Test.JsonParse` holds the
two to each other). A scan of the Lean under `verity/core/service/flock/` refuses a call of `Lean.Json.parse`, written
`Json.parse` under `open Lean` or in full, outside the module that mirrors it and the test that compares the two.
"""

from __future__ import annotations

import re

from test_repository import ROOT, _tracked

TREE = "verity/core/service/flock/"
ALLOWED = {
    "verity/core/service/flock/Flock/JsonParse.lean": "the total parser, whose docstring names the source it mirrors",
    "verity/core/service/flock/Flock/Test/JsonParse.lean": "the test holding the total parser to Lean.Json.parse",
}

#: `/- … -/` and `-- …`, blanked so that line numbers stay
COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
CALL = re.compile(r"(?<![\w.])(?:Lean\.)?Json\.parse\b(?![?'\w])")


def test_flock_reads_json_through_the_total_parser() -> None:
    bad = []
    for path in _tracked():
        if not (path.startswith(TREE) and path.endswith(".lean")) or path in ALLOWED:
            continue
        text = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), (ROOT / path).read_text())
        bad += [f"{path}:{text.count(chr(10), 0, m.start()) + 1}" for m in CALL.finditer(text)]
    assert not bad, ("Lean.Json.parse is partial, so no proof reads through it: call Flock.JsonParse.parse (JsonParse.parse "
                     "inside namespace Flock), which reads the same language:\n  " + "\n  ".join(bad))


def test_allowed_entries_exist() -> None:
    assert all((ROOT / p).exists() for p in ALLOWED), "an ALLOWED entry names a file that no longer exists"
