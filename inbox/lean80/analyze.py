#!/usr/bin/env python3
"""Keep each stage's tree in memory (main's tip through the six maps) and pickle them for the questions."""
import pickle
import posixpath
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "move_main"))
import rename  # noqa: E402
import sim  # noqa: E402


def stages(commit: str):
    texts, names = sim.tree(commit)
    origin = {p: p for p in names}
    snaps = [dict(texts)]
    origins = [dict(origin)]
    for stage in sim.STAGES:
        words = rename.Words(tomllib.loads((HERE / "maps" / f"{stage}.toml").read_text()))
        files = list(origin)
        assert not words.unmatched(files), words.unmatched(files)
        words.hold(files)
        moves = {p: n for p in files if not words.excluded(p) and (n := words.path(p)) != p}
        for p in list(texts):
            if words.excluded(p):
                continue
            text = texts[p]
            lock = posixpath.basename(p) == rename.POLICY
            texts[p] = rename.resorted(text, words.sub(text, p, rename.record_spans(text) if lock else [], Counter(), Counter()), p)
        texts = {moves.get(p, p): t for p, t in texts.items()}
        origin = {moves.get(p, p): o for p, o in origin.items()}
        snaps.append(dict(texts))
        origins.append(dict(origin))
    return snaps, origins


if __name__ == "__main__":
    snaps, origins = stages(sys.argv[1])
    pickle.dump((snaps, origins), open(sys.argv[2], "wb"))
    print("ok", [len(s) for s in snaps])
