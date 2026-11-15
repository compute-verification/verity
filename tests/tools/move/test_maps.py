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


def meet(a: list[str] | None, b: list[str] | None) -> list[tuple[str | None, str | None]]:
    """The pairs of globs, one of each table's, in which two tables can rename in one file: [words] (None) meets every
    glob, and two globs meet when one matches the other or both hold a wildcard and neither's text before its first
    wildcard rules out the other's (no tree is read, so two such patterns are taken to share a file), unless each ends in
    a file extension and the two differ."""
    def head(g: str) -> str:
        return g[:m.start()] if (m := re.search(r"[*?\[]", g)) else g

    def ext(g: str) -> str | None:
        return m.group(1) if (m := re.search(r"\.(\w+)$", re.split(r"[*?\[\]/]", g)[-1])) else None

    def both(x: str, y: str) -> bool:
        return head(x) != x and head(y) != y and (head(x).startswith(head(y)) or head(y).startswith(head(x)))

    def apart(x: str, y: str) -> bool:
        return None not in (ext(x), ext(y)) and ext(x) != ext(y)
    return [(x, y) for x in ([None] if a is None else a) for y in ([None] if b is None else b)
            if x is None or y is None or not apart(x, y) and (x == y or R._match(x, y) or R._match(y, x) or both(x, y))]


def stands(words: R.Words, word: str, where: tuple[str | None, str | None]) -> bool:
    """Whether a map leaves ``word`` standing in every file both globs of ``where`` match, as `rename.py` decides a file's
    words: either glob, read as a path, is one the map's [rewrite] exclude or a [keep_in] glob listing the word holds."""
    return any(g is not None and (words.excluded(g) or word in words.held(g)) for g in where)


def everywhere(data: dict) -> R.Words:
    """``data``'s words with each [[words_in]] table widened to every file, so a name is renamed wherever the map could."""
    return R.Words({"words": data.get("words", {}),
                    "words_in": [{"files": ["*"], "words": t.get("words", {})} for t in data.get("words_in", [])]})


def two_names(maps: dict[str, dict]) -> list[str]:
    """Each word two of ``maps`` rename in one file so that the order of their moves decides its name (words compared case
    folded, a map's words applied to the other's name as `rename.py` applies them to an identifier). Two rows meet only
    where their tables' files meet and neither map leaves its word standing there (`stands`), so two senses of one word,
    each map's in files the other keeps or excludes, are two names. A prose map (`scope = "prose"`) and an identifier
    map are two kinds, so one word may have a row in one map of each."""
    words = {p: everywhere(d) for p, d in maps.items()}
    scopes = {p: R.Words(d) for p, d in maps.items()}
    out = []
    ps = sorted(maps)
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            if (maps[a].get("scope") == "prose") != (maps[b].get("scope") == "prose"):
                continue
            for ga, ta in tables(maps[a]):
                for gb, tb in tables(maps[b]):
                    if not (where := meet(ga, gb)):
                        continue
                    theirs = {}
                    for old, new in tb.items():
                        theirs.setdefault(old.casefold(), set()).add((new, old))
                    for old, x in ta.items():
                        for y, by in sorted(theirs.get(old.casefold(), ())):
                            if all(stands(scopes[a], old, p) or stands(scopes[b], by, p) for p in where):
                                continue
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
        "a.toml makes 'reroll'"]
    kinds = {"ids.toml": {"words": {"firewall": "zk_server", "Firewall": "ZkServer"}},
             "prose.toml": {"scope": "prose", "words": {"Firewall": "ZK server"}}}
    assert two_names(kinds) == []
    assert two_names({**kinds, "more.toml": {"scope": "prose", "words": {"Firewall": "Guard"}}}) == [
        "'Firewall': more.toml renames it to 'Guard', which prose.toml makes 'Guard'; prose.toml renames it to "
        "'ZK server', which more.toml makes 'ZK server'"]
    apart = {"d.toml": {"words": {}, "words_in": [{"files": ["Security/A.lean"], "words": {"Session": "Run"}}]},
             "e.toml": {"scope": "prose", "words": {}, "words_in": [{"files": ["bench/**"], "words": {"session": "phase"}}]}}
    assert two_names(apart) == []
    assert two_names({"g.toml": {"words": {}, "words_in": [{"files": ["**/session.py"], "words": {"SESSION": "PHASE"}}]},
                      "h.toml": {"words": {}, "words_in": [{"files": ["Security/*.lean"],
                                                            "words": {"session": "attachment"}}]}}) == []
    assert two_names({**apart, "f.toml": {"words": {}, "words_in": [{"files": ["Security/*.lean"],
                                                                     "words": {"session": "attachment"}}]}}) == [
        "'Session': d.toml renames it to 'Run', which f.toml makes 'Run'; f.toml renames it to 'attachment', which "
        "d.toml makes 'attachment'"]


def test_two_senses_of_a_word_in_files_one_map_keeps_or_excludes_pass_and_files_both_rename_fail():
    fanin = {"words": {}, "words_in": [{"files": ["bench/slow/model.py", "bench/slow/tests/test_model.py"],
                                        "words": {"legs": "fanin"}}]}
    kept = {"fanin.toml": fanin, "leg.toml": {"words": {"legs": "subprotocols"}, "keep_in": {"bench/slow/**": ["legs"]}}}
    assert two_names(kept) == []
    assert two_names({**kept, "leg.toml": {"words": {"legs": "subprotocols"}, "rewrite": {"exclude": ["bench/**"]}}}) == []
    clash = ["'legs': fanin.toml renames it to 'fanin', which leg.toml makes 'fanin'; leg.toml renames it to "
             "'subprotocols', which fanin.toml makes 'subprotocols'"]
    assert two_names({**kept, "leg.toml": {"words": {"legs": "subprotocols"}, "keep_in": {"bench/fast/**": ["legs"]}}}) \
        == clash
    assert two_names({**kept, "leg.toml": {"words": {"legs": "subprotocols"},
                                           "keep_in": {"bench/slow/model.py": ["legs"]}}}) == clash
    assert two_names({**kept, "leg.toml": {"words": {"Legs": "Subprotocols"}, "keep_in": {"bench/slow/**": ["legs"]}}}) \
        == [clash[0].replace("'subprotocols'", "'Subprotocols'")]
