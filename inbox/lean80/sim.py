#!/usr/bin/env python3
"""In-memory read of what the six period maps would do to a commit's tree: nothing is written to any checkout.
Usage: sim.py COMMIT OUT.json [--only FILE...]  (reads blobs through `git cat-file --batch`)."""
import json
import posixpath
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "move_main"))
import rename  # noqa: E402

REPO = "/agent/repos/verity"
STAGES = ["period_map", "period_map_2", "period_net_map", "period_proofs_map", "period_proofs_map_2", "period_soundness_map"]


def tree(commit: str, only=None) -> dict[str, str]:
    ls = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "-z", commit], check=True, capture_output=True).stdout.decode()
    entries = []
    for e in ls.split("\0"):
        if not e:
            continue
        meta, path = e.split("\t", 1)
        mode, kind, sha = meta.split()
        if mode == "120000" or kind != "blob":
            continue
        entries.append((path, sha))
    names = [p for p, _ in entries]
    out: dict[str, str] = {}
    want = entries if only is None else [(p, s) for p, s in entries if p in only]
    with subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
        import threading

        def feed():
            for _, sha in want:
                p.stdin.write(sha.encode() + b"\n")
            p.stdin.close()
        threading.Thread(target=feed, daemon=True).start()
        for path, _ in want:
            header = p.stdout.readline().split()
            data = p.stdout.read(int(header[-1]))
            p.stdout.read(1)
            if b"\0" in data[:8000]:
                continue
            try:
                out[path] = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
    return out, names


def run(commit: str, only=None, upto: int = 6):
    texts, names = tree(commit, only)
    report = {"commit": commit, "stages": []}
    origin = {p: p for p in names}          # current path -> path at the commit
    for k, stage in enumerate(STAGES[:upto]):
        words = rename.Words(tomllib.loads((HERE / "maps" / f"{stage}.toml").read_text()))
        files = list(origin)
        missing = words.unmatched(files)
        words.hold(files)
        moves = {p: n for p in files if not words.excluded(p) and (n := words.path(p)) != p}
        counts, kept, changed = Counter(), Counter(), {}
        for p in list(texts):
            if words.excluded(p):
                continue
            text = texts[p]
            lock = posixpath.basename(p) == rename.POLICY
            here = Counter()
            new = rename.resorted(text, words.sub(text, p, rename.record_spans(text) if lock else [], counts, here), p)
            kept.update({(p, w): n for w, n in here.items()})
            if new != text:
                a, b = text.split("\n"), new.split("\n")
                if len(a) == len(b):
                    changed[p] = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
                else:
                    changed[p] = [(0, f"<{len(a)} lines>", f"<{len(b)} lines>")]
                texts[p] = new
        texts = {moves.get(p, p): t for p, t in texts.items()}
        origin = {moves.get(p, p): o for p, o in origin.items()}
        report["stages"].append({
            "stage": stage, "unmatched": missing, "moves": moves,
            "counts": sorted(([w, n, c] for (w, n), c in counts.items()), key=lambda r: -r[2]),
            "kept": sorted([p, w, c] for (p, w), c in kept.items()),
            "changed": {p: [[origin.get(moves.get(p, p), p), *row] for row in rows] for p, rows in changed.items()},
        })
    return report


if __name__ == "__main__":
    commit, out = sys.argv[1], sys.argv[2]
    only = set(sys.argv[4:]) if len(sys.argv) > 3 and sys.argv[3] == "--only" else None
    rep = run(commit, only)
    json.dump(rep, open(out, "w"))
    for s in rep["stages"]:
        print(s["stage"], "files changed", len(s["changed"]), "moved", len(s["moves"]), "words", sum(c for _, _, c in s["counts"]),
              "unmatched", s["unmatched"])
