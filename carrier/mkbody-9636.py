"""#9636's body after the carry: the old body with the parts the carry made stale rewritten.

    python3 mkbody-9636.py OLD.md FACTS.json NEW.md

Rewritten: the paragraph "What the branch holds beside its own change" and the two path lines under it; the README bullet
of "What it does not do" (the file is under its cap now); the tries' run in "The proof"; and the three last sections
("The lock", "Runs at the head", "Merged tree"), whose text comes from FACTS.json's keys `lock`, `runs`, `merged`.
Every other line is the old body's, byte for byte; the script fails if an anchor it replaces is not found exactly once.
"""
import json
import sys

old, facts, new = sys.argv[1:4]
body = open(old).read()
F = json.load(open(facts))


def swap(text, start, end, repl):
    """Replace the span from the line starting with `start` up to (not including) the line starting with `end`."""
    i = text.index("\n" + start) + 1
    assert text.count("\n" + start) == 1, start
    j = len(text) if end is None else text.index("\n" + end, i) + 1
    return text[:i] + repl + text[j:]


body = swap(body, "**What the branch holds beside its own change.**", "- Lean code:", F["holds"] + "\n\n")
for a, b in (("under `verity/protocols/spot/verifier/lean/`;", "under `verity/protocols/proofs/spot/verifier/lean/`;"),
             ("under `verity/protocols/spot/verifier/`;", "under `verity/protocols/proofs/spot/verifier/`;")):
    assert body.count(a) == 1, a
    body = body.replace(a, b)
body = swap(body, "- **`verity/protocols/spot/verifier/README.md` is over its Markdown cap**", "## Not established", "\n")
a = "run `r20261113-030550-5811`, at the head."
assert body.count(a) == 1
body = body.replace(a, F["tries_ref"])
body = swap(body, "## The lock", "## Runs at the head", F["lock"] + "\n\n")
body = swap(body, "## Runs at the head", "## Merged tree", F["runs"] + "\n\n")
body = swap(body, "## Merged tree", None, F["merged"] + "\n")
for stale in ("4a7291116e11", "049c5875989c", "r20261113-", "verity/protocols/spot/"):
    assert stale not in body, (stale, body.count(stale))
open(new, "w").write(body)
print(f"{new}: {len(body.encode())} bytes, {body.count(chr(10))} lines")
