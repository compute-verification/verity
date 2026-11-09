"""The word maps in `tools/move/`: each concept has one map, so two maps that rename one word in one file reach one name
whichever of their moves runs first. A map lands before its move, so a map that would give a word a second name fails here,
before either move is staged."""
from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

MOVE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rename_for_maps", MOVE / "rename.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def tables(data: dict) -> list[tuple[list[str] | None, dict[str, str]]]:
    """``data``'s word tables, each with the files it renames in: [words] everywhere (None), a [[words_in]] its globs."""
    return [(None, data.get("words", {})), *((t.get("files", []), t.get("words", {})) for t in data.get("words_in", []))]


def meet(a: list[str] | None, b: list[str] | None) -> bool:
    """Whether two tables can rename in one file: [words] meets every table, and two globs meet when one matches the
    other or both hold a wildcard (no tree is read, so two patterns are taken to share a file)."""
    wild = re.compile(r"[*?\[]")
    return a is None or b is None or any(x == y or R._match(x, y) or R._match(y, x) or (wild.search(x) and wild.search(y))
                                         for x in a for y in b)


def everywhere(data: dict) -> R.Words:
    """``data``'s words with each [[words_in]] table widened to every file, so a name is renamed wherever the map could."""
    return R.Words({"words": data.get("words", {}),
                    "words_in": [{"files": ["*"], "words": t.get("words", {})} for t in data.get("words_in", [])]})


def two_names(maps: dict[str, dict]) -> list[str]:
    """Each word two of ``maps`` rename in one file so that the order of their moves decides its name (words compared case
    folded, a map's words applied to the other's name as `rename.py` applies them to an identifier)."""
    words = {p: everywhere(d) for p, d in maps.items()}
    out = []
    ps = sorted(maps)
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            for ga, ta in tables(maps[a]):
                for gb, tb in tables(maps[b]):
                    if not meet(ga, gb):
                        continue
                    theirs = {}
                    for old, new in tb.items():
                        theirs.setdefault(old.casefold(), set()).add(new)
                    for old, x in ta.items():
                        for y in sorted(theirs.get(old.casefold(), ())):
                            xy, yx = words[b].sub(x, "name", protect=False), words[a].sub(y, "name", protect=False)
                            if xy.casefold() != yx.casefold():
                                out.append(f"{old!r}: {a} renames it to {x!r}, which {b} makes {xy!r}; {b} renames it to "
                                           f"{y!r}, which {a} makes {yx!r}")
    return list(dict.fromkeys(out))


def test_two_word_maps_that_rename_one_word_reach_one_name_in_either_order():
    found = two_names({p.name: d for p in sorted(MOVE.glob("*.toml")) if "words" in (d := tomllib.loads(p.read_text()))})
    assert not found, "\n".join(found + ["each concept has one name: settle the word in one map and drop its row from the "
                                         "other, or give the other map the first one's names"])


def test_maps_that_agree_or_compose_pass_and_a_second_name_is_found():
    agree = {"a.toml": {"words": {"redraw": "resample"}}, "b.toml": {"scope": "prose", "words": {"Redraw": "Resample"}}}
    assert two_names(agree) == []
    compose = {"draw.toml": {"words": {}, "words_in": [{"files": ["Pop/*.lean"], "words": {
                   "indepDraw_escapes": "indepSample_escapes", "indepDraw_acceptCoins": "indepSample_acceptCoins"}}]},
               "escape.toml": {"words": {"indepDraw_escapes": "indepDraw_acceptCoins",
                                         "indepSample_escapes": "indepSample_acceptCoins"}}}
    assert two_names(compose) == []
    clash = {**agree, "c.toml": {"words": {}, "words_in": [{"files": ["x/**"], "words": {"redraw": "reroll"}}]}}
    assert two_names(clash) == [
        "'redraw': a.toml renames it to 'resample', which c.toml makes 'resample'; c.toml renames it to 'reroll', which "
        "a.toml makes 'reroll'",
        "'Redraw': b.toml renames it to 'Resample', which c.toml makes 'Resample'; c.toml renames it to 'reroll', which "
        "b.toml makes 'reroll'"]
    apart = {"d.toml": {"words": {}, "words_in": [{"files": ["Security/A.lean"], "words": {"Session": "Run"}}]},
             "e.toml": {"scope": "prose", "words": {}, "words_in": [{"files": ["bench/x.py"], "words": {"session": "phase"}}]}}
    assert two_names(apart) == []
    assert two_names({**apart, "f.toml": {"words": {}, "words_in": [{"files": ["Security/*.lean"],
                                                                     "words": {"session": "attachment"}}]}}) == [
        "'Session': d.toml renames it to 'Run', which f.toml makes 'Run'; f.toml renames it to 'attachment', which "
        "d.toml makes 'attachment'"]
