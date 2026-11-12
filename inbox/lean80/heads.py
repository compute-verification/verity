import subprocess, sys, tomllib, re, json
from collections import Counter, defaultdict
sys.path.insert(0, "move_main")
import rename
REPO = "/agent/repos/verity"
def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True).stdout.decode("utf-8", "replace")
MAIN = "7d296de0f3e9"
main_files = [p for p in git("ls-tree", "-r", "-z", "--name-only", MAIN).split("\0") if p]
main_set = set(main_files)
by_tail = defaultdict(list)
for p in main_files:
    parts = p.lower().split("/")
    for k in (2, 3, 4):
        by_tail["/".join(parts[-k:])].append(p)
def norm(p):
    if p in main_set:
        return p, False
    parts = p.lower().split("/")
    for k in (4, 3, 2):
        c = by_tail.get("/".join(parts[-k:]), [])
        if len(c) == 1:
            return c[0], True
    return p, True
W = rename.Words(tomllib.loads(open("maps/period_proofs_map.toml").read()))
W.hold(main_files)
FAM = {"PeriodPlays": "i", "PeriodGood": "i", "periodG": "i", "periodRun": "i",
       "windowOf": "ii", "windowIdx": "ii", "windowStreams": "ii", "windowTest": "ii", "keyedWindow": "ii",
       "sessionTreeReason": "iii", "sessionsTreeReason": "iii", "sessionTreesPiece": "iii"}
SCOPED = {}
for globs, table in W.scoped:
    for w in table:
        SCOPED[w] = globs
ASK = re.compile(r"(?<!\w)(periodRun|sessionTreeReason|sessionsTreeReason|windowOf|sessionOf|sessionNamed|trustedSessionOf|window_opens|windowSalt)(?!\w)")
refs = [l.split() for l in git("for-each-ref", "--format=%(refname) %(objectname)", "refs/proghold/pr/").splitlines()]
out = {}
for ref, sha in sorted(refs):
    name = ref.rsplit("/", 1)[1]
    mb = git("merge-base", MAIN, sha).strip()
    moved_layout = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", "5c095bb4ab70", mb]).returncode == 0
    diff = git("diff", "-U0", "-M", "--no-color", mb, sha)
    cur, added, removed = None, defaultdict(list), defaultdict(list)
    ln = 0
    for line in diff.split("\n"):
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("--- ") or line.startswith("diff ") or line.startswith("index ") or line.startswith("rename ") or line.startswith("similarity ") or line.startswith("new file") or line.startswith("deleted file") or line.startswith("old mode") or line.startswith("new mode"):
            if line.startswith("--- a/"):
                old = line[6:]
            continue
        elif line.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line); ln = int(m.group(1))
        elif line.startswith("+") and cur:
            added[cur].append((ln, line[1:])); ln += 1
        elif line.startswith("-") and cur is not None:
            removed[cur].append(line[1:])
    res = {"sha": sha[:12], "mb": mb[:12], "new_layout": moved_layout, "files": len(set(added) | set(removed)),
           "uses": Counter(), "use_files": defaultdict(set), "conflicts": Counter(), "conflict_files": defaultdict(set), "outside": [], "oldpaths": 0}
    for f, lines in added.items():
        p, guessed = norm(f)
        if guessed: res["oldpaths"] += 1
        if W.excluded(p): continue
        for n, text in lines:
            c = Counter()
            W.sub(text, p, counts=c)
            for (w, new), k in c.items():
                g = FAM.get(w, "other")
                res["uses"][g] += k; res["use_files"][g].add(f)
            for m in ASK.finditer(text):
                w = m.group(1)
                if not any(rename._match(p, g) for g in SCOPED[w]):
                    res["outside"].append((f, n, w, text.strip()[:150]))
    for f, lines in removed.items():
        p, _ = norm(f)
        if W.excluded(p): continue
        for text in lines:
            c = Counter()
            W.sub(text, p, counts=c)
            if c:
                gs = {FAM.get(w, "other") for (w, _n) in c}
                for g in gs:
                    res["conflicts"][g] += 1; res["conflict_files"][g].add(f)
    out[name] = res
    u = {g: (len(res["use_files"][g]), res["uses"][g]) for g in ("i", "ii", "iii", "other") if res["uses"][g]}
    cf = {g: (len(res["conflict_files"][g]), res["conflicts"][g]) for g in ("i", "ii", "iii", "other") if res["conflicts"][g]}
    print(f"{name} {res['sha']} mb {res['mb']} {'new' if moved_layout else 'OLD'}-layout files {res['files']} guessed-paths {res['oldpaths']} | added uses {u or '-'} | rewritten lines the move also rewrites {cf or '-'} | outside-row {len(res['outside'])}")
    for o in res["outside"]:
        print("      outside-row:", o)
json.dump({k: {**v, "uses": dict(v["uses"]), "use_files": {g: sorted(s) for g, s in v["use_files"].items()}, "conflicts": dict(v["conflicts"]), "conflict_files": {g: sorted(s) for g, s in v["conflict_files"].items()}} for k, v in out.items()}, open("heads.json", "w"), indent=1)
