"""Each refusal text of the Lean decoder (`Circuits/Descriptor.lean`'s `throw`s) is the pinned text of a case of the decode difference
test (`lean_decode_texts.json`, `write_lean_texts.py`) or is listed in `NO_CASE` with why none is written or none can be. Sites that spell
one text are one group here: the cases tell them apart by what they edit, not by the text. `decode`, `decodeOf` and `decodeWith` carry
the same clauses; the port reaches `decode`'s."""
from __future__ import annotations

import json
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "verity" / "circuits" / "lean" / "Circuits" / "Descriptor.lean"
TEXTS = json.loads((Path(__file__).parent / "lean_decode_texts.json").read_text())

#: a refusal named by a `let`: its text
NAMED = {
    "shape": "grid-shape: grid {g} is not [run, [[space, index, base, inner_stride, [[radix, step], ...]], ...]]",
    "noLeaf": "references {j} name no leaf of a parameter or a node",
    "UNCHECKED": "its references are read leaf by leaf past {W} leaves to compare their widths, past which this verifier refuses them",
}
UNDECIDED = "a {form} part of {n} references, not decided by structure: {why}"
#: throws that pass an inner refusal on, whose own text a case pins
FORWARDED = {"{pre}: {e}", "definition '{fid}' ret: {e}"}

NO_CASE = {
    "references nest deeper than this verifier follows":
        "concats nest at most `Steps.FUEL` = 2^62 deep (the grid's own parse, `parsePartsIn 4`, reads a fixed shape one level deep)",
    "definitions nest deeper than this verifier follows": "definitions nest at most `Steps.FUEL` = 2^62 deep",
    "program: not UTF-8": "the port's descriptor is a JSON string, which is UTF-8; `Program.decode` takes bytes",
}


def _regex(text: str, interpolated: bool):
    parts = re.split(r"\{(?:[^{}]|\{[^{}]*\})*\}", text) if interpolated else [text]
    return re.compile(".*".join(re.escape(p) for p in parts), re.S)


def sites() -> dict[str, list[int]]:
    """The decoder's refusal texts, each with the lines that throw it."""
    out: dict[str, list[int]] = {}
    for n, line in enumerate(SOURCE.read_text().split("\n"), 1):
        literal = list(re.finditer(r'throw\s*\(?\s*(s!)?"((?:[^"\\]|\\.)*)"', line))
        for m in literal:
            out.setdefault(m.group(2), []).append(n)
        if not literal and (m := re.search(r"throw\s*\(?\s*(\w+)", line)):
            if m[1] in NAMED:
                out.setdefault(NAMED[m[1]], []).append(n)
            elif m[1] == "undecided":
                out.setdefault(UNDECIDED, []).append(n)
    return out


def reached(text: str) -> bool:
    if text in FORWARDED:
        return True
    rx = _regex(text, "{" in text)
    return any(rx.match(m) or rx.match(re.sub(r"^definition '.*?'(?: node \d+| ret|: E_REINTERPRET unchecked): ", "", m, flags=re.S)) for m in TEXTS.values())


def test_every_refusal_text_of_the_lean_decoder_is_pinned_or_unreached():
    unlisted = {t: ls for t, ls in sites().items() if not reached(t) and t not in NO_CASE}
    assert not unlisted, f"refusals no pinned case reaches and NO_CASE does not list (line: text): {unlisted}"


def test_the_unreached_refusals_are_in_the_decoder_and_reached_by_no_case():
    found = sites()
    assert not [t for t in NO_CASE if t not in found], "NO_CASE names a text the decoder no longer throws"
    assert not [t for t in NO_CASE if reached(t)], "NO_CASE lists a text a case now reaches: delete it"

