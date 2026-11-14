"""Every code name the staged rename writes, old to new, with the files and count: tokens of code (prose and literals blanked)
aligned before and after."""
import collections, difflib, re, subprocess, sys
sys.path.insert(0, "tools/move")
import rename as R
moves = dict(l.rstrip("\n").split("\t")[::-1] for l in open(__import__("os").environ.get("MOVES", "/tmp/kit/m1_moves.tsv")) if l.strip())
TOK = re.compile(r"[A-Za-z_]\w*")
def code(p, t):
    try: spans = R.regions(p, t)
    except Exception: return ""
    buf, i = [], 0
    for s, e, k in spans:
        buf += [t[i:s], " " * (e - s)]; i = e
    return "".join(buf + [t[i:]])
pairs = collections.defaultdict(lambda: [0, set()])
files = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=AMR"], capture_output=True, text=True).stdout.split()
for f in files:
    if not f.endswith((".py", ".sh", ".cu", ".rs", ".toml")): continue
    o = subprocess.run(["git", "show", f"HEAD:{moves.get(f, f)}"], capture_output=True, text=True).stdout
    n = open(f, encoding="utf-8").read()
    a, b = TOK.findall(code(moves.get(f, f), o)), TOK.findall(code(f, n))
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for x, y in zip(a[i1:i2], b[j1:j2]):
                pairs[(x, y)][0] += 1; pairs[(x, y)][1].add(f)
for (x, y), (c, fs) in sorted(pairs.items(), key=lambda kv: (-kv[1][0], kv[0])):
    print(f"{c}\t{x}\t{y}\t{len(fs)}\t{sorted(fs)[0]}")
