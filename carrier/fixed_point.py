"""The words check of a carried head by the rename tool's own reading.

    python3 fixed_point.py MAIN OLD_HEAD CARRIED [DIFF_BASE]

For each word move MAIN has and OLD_HEAD lacks (a commit with a line `Move: tools/move/MAP.toml`, crossed by
`tools/move/rename.py --map MAP`), it takes `rename.py` as the move's parent has it and the map as the move has it (what
`restack.py` runs on a branch), and asks `Words.sub`, the function that rewrites one file's text, about each file CARRIED
changes over MAIN: the words it would still rename there. A carried head is a fixed point of every move when each count is 0.
It reads a handful of files in memory: it walks no tree, builds nothing and writes nothing but a scratch copy of
`tools/move/` per move under the temporary directory.
"""
import json
import subprocess
import sys
import tempfile

PROBE = r"""
import json, subprocess, sys, tomllib
from collections import Counter
sys.path.insert(0, sys.argv[1])
import rename as R
move, mp, carried = sys.argv[2], sys.argv[3], sys.argv[4]
files = json.loads(sys.argv[5])
main = sys.argv[6]
words = R.Words(tomllib.loads(subprocess.run(["git", "show", f"{move}:{mp}"], capture_output=True, text=True, check=True).stdout))
out = {}
for p in files:
    raw = subprocess.run(["git", "show", f"{carried}:{p}"], capture_output=True).stdout
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if words.excluded(p):
        continue
    counts, kept = Counter(), Counter()
    lock = p.rsplit("/", 1)[-1] == R.POLICY
    new = R.resorted(text, words.sub(text, p, R.record_spans(text) if lock else [], counts, kept), p)
    if new != text or counts:
        # the same file as main holds it: a word main's own lines still hold there is not the branch's
        mine = Counter()
        raw0 = subprocess.run(["git", "show", f"{main}:{p}"], capture_output=True)
        if raw0.returncode == 0:
            try:
                t0 = raw0.stdout.decode("utf-8")
                words.sub(t0, p, R.record_spans(t0) if lock else [], mine, Counter())
            except UnicodeDecodeError:
                pass
        over = counts - mine
        out[p] = {"all": sorted(f"{o} -> {n}: {c}" for (o, n), c in counts.items()),
                  "over_main": sorted(f"{o} -> {n}: {c}" for (o, n), c in over.items())}
print(json.dumps({"scope": words.scope, "would_rename": out}))
"""


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def main():
    main_sha, old, carried = sys.argv[1:4]
    # DIFF_BASE (a control only): the commit CARRIED's files are taken against, where that is not MAIN
    against = sys.argv[4] if len(sys.argv) > 4 else main_sha
    files = [f for f in git("diff", "--name-only", "--diff-filter=AMR", against, carried).splitlines() if f]
    print(f"files the carried head {carried[:12]} changes over main {main_sha[:12]}: {len(files)}")
    log = git("log", "--topo-order", "--reverse", "--no-merges", "-E", "--grep=^Move: tools/move/", "--format=%H %P%x00%B%x1e",
              main_sha, f"^{old}")
    total, n = 0, 0
    for rec in filter(None, (s.strip() for s in log.split("\x1e"))):
        line, _, body = rec.partition("\x00")
        move, *parents = line.split()
        maps = [ln.split()[1] for ln in body.splitlines() if ln.startswith("Move: tools/move/")]
        if not maps:
            continue
        n += 1
        with tempfile.TemporaryDirectory(prefix="fixed-point-") as d:
            tar = subprocess.run(["git", "archive", parents[0], "tools/move"], capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
            p = subprocess.run([sys.executable, "-c", PROBE, f"{d}/tools/move", move, maps[0], carried, json.dumps(files), main_sha],
                               capture_output=True, text=True)
        if p.returncode:
            print(f"MOVE {move[:12]} {maps[0]}: the probe failed: {p.stderr.strip()[-400:]}")
            total += 1
            continue
        got = json.loads(p.stdout.strip().splitlines()[-1])
        hits = sum(len(v["over_main"]) for v in got["would_rename"].values())
        total += hits
        print(f"MOVE {move[:12]} {maps[0]} (scope {got['scope'] or 'all'}): rename.py would still rename words in "
              f"{len(got['would_rename'])} of the {len(files)} files; in {sum(1 for v in got['would_rename'].values() if v['over_main'])} "
              "of them more than in main's own copy")
        for f, v in got["would_rename"].items():
            print(f"   {f}: in all: {'; '.join(v['all']) or 'a re-sort only'} | beyond main's own copy: {'; '.join(v['over_main']) or 'none'}")
    print(f"TOTAL over {n} word moves the old head lacked: {total} word(s) beyond what main's own copies hold "
          "(0: the carried change is a fixed point of each)")


main()
