"""The word maps in `tools/move/`: each concept has one map, so two maps that rename one word reach one name whichever of
their moves runs first. A map lands before its move, so a map that would give a word a second name fails here, before
either move is staged."""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

MOVE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rename_for_maps", MOVE / "rename.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def renames(data: dict) -> list[tuple[str, str]]:
    return [(old, new) for table in (data.get("words", {}), *(t.get("words", {}) for t in data.get("words_in", [])))
            for old, new in table.items()]


def everywhere(data: dict) -> R.Words:
    """``data``'s words with each [[words_in]] table widened to every file, so a name is renamed wherever the map could."""
    return R.Words({"words": data.get("words", {}),
                    "words_in": [{"files": ["*"], "words": t.get("words", {})} for t in data.get("words_in", [])]})


def two_names(maps: dict[str, dict]) -> list[str]:
    """Each word two of ``maps`` rename so that the order of their moves decides its name (words compared case folded, a
    map's words applied to the other's name as `rename.py` applies them to an identifier)."""
    words = {p: everywhere(d) for p, d in maps.items()}
    out = []
    ps = sorted(maps)
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            theirs = {}
            for old, new in renames(maps[b]):
                theirs.setdefault(old.casefold(), set()).add(new)
            for old, x in renames(maps[a]):
                for y in sorted(theirs.get(old.casefold(), ())):
                    xy, yx = words[b].sub(x, "name", protect=False), words[a].sub(y, "name", protect=False)
                    if xy.casefold() != yx.casefold():
                        out.append(f"{old!r}: {a} renames it to {x!r}, which {b} makes {xy!r}; {b} renames it to {y!r}, "
                                   f"which {a} makes {yx!r}")
    return out


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
