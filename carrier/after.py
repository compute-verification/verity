"""After a batch of the pass: read a run's results and its bundle, and make the checks git alone can make.

    python3 after.py MAIN RUN_OUT_DIR [NAME=PARENT[+PARENT...] ...] [--landing NAME,NAME,...] [--prefix refs/pass]

For each `restack-NAME.json` under RUN_OUT_DIR: its row; the carried commit, fetched from `carried.bundle` to
`refs/pass/carried/NAME`; whether it holds MAIN and its own head; and the lines it adds over MAIN that still name an old path
or module of a move's map, the maps of the moves it skipped first.

NAME=PARENTS is the test a late carry must pass, and the pass's own check of a stacked pull request: the carried line merged
with a commit that holds its carried parents (each already under `refs/pass/carried/`) shows no conflict, and the merged tree
is the line's own carried tree.

--landing: the carried results merged onto MAIN one after another in that order, with the conflicts of each step.
"""
import json
import os
import re
import subprocess
import sys
import tomllib

sys.path.insert(0, "tools/move")
import restack as rs  # noqa: E402

# CARRIER: no identity is set here: every scratch commit is the checkout's own (infra's rule of 13 Nov)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True)


PREFIX = "refs/pass"


def carried(name):
    p = sh("git", "rev-parse", "--verify", "-q", f"{PREFIX}/carried/{name}^{{commit}}")
    return p.stdout.strip() if p.returncode == 0 else None


def old_names(move):
    """The old paths of ``move``'s map and their dotted forms, as one pattern; None where the map has no `paths` table."""
    line = rs.move_line(move, None)
    if not line or "--map" not in line[1]:
        return None, None
    path = line[1][line[1].index("--map") + 1]
    text = sh("git", "show", f"{move}:{path}").stdout
    try:
        paths = tomllib.loads(text).get("paths") or {}
    except tomllib.TOMLDecodeError:
        paths = {}
    alts = set()
    for old, new in paths.items():
        if not isinstance(new, str) or new.startswith(old):
            continue  # the new name begins with the old one: every new line would match
        alts.add(re.escape(old))
        dotted = old.rstrip("/").removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        if "." in dotted and "/" in old:
            alts.add(re.escape(dotted) + r"(?![\w])")
    if not alts:
        return path, None
    return path, re.compile(r"(?<![\w./-])(?:" + "|".join(sorted(alts, key=len, reverse=True)) + ")")


def added_lines(main, commit):
    """(file, line) for each line ``commit``'s tree adds over ``main``'s."""
    out = subprocess.run(["git", "diff", "-U0", "--no-renames", "--no-color", main, commit], capture_output=True).stdout
    f = None
    for raw in out.split(b"\n"):
        if raw.startswith(b"+++ "):
            f = raw[6:].decode("utf-8", "replace") if raw.startswith(b"+++ b/") else None
        elif raw.startswith(b"+") and f is not None:
            yield f, raw[1:].decode("utf-8", "replace")


def grep_old(main, commit, moves, label):
    pats = []
    for m in moves:
        path, pat = old_names(m)
        if pat is not None:
            pats.append((path.rsplit("/", 1)[-1], pat))
    hits, n = {}, 0
    for f, text in added_lines(main, commit):
        n += 1
        for name, pat in pats:
            if pat.search(text):
                hits.setdefault((name, f), []).append(text.strip()[:160])
    print(f"   GREP {label}: {n} added lines read against {len(pats)} maps: {sum(len(v) for v in hits.values())} lines name an old path or module, in {len(hits)} (map, file) pairs")
    for (name, f), lines in sorted(hits.items())[:40]:
        print(f"      {name}: {f}: {len(lines)}: {lines[0]}")
    return hits


def main():
    global PREFIX
    args = sys.argv[1:]
    landing = []
    if "--prefix" in args:
        i = args.index("--prefix")
        PREFIX = args[i + 1]
        del args[i:i + 2]
    if "--landing" in args:
        i = args.index("--landing")
        landing = args[i + 1].split(",")
        del args[i:i + 2]
    main_sha, out = rs.rev(args[0], None), args[1]
    pairs = dict(a.split("=", 1) for a in args[2:])
    names = sorted(f[len("restack-"):-len(".json")] for f in os.listdir(out) if f.startswith("restack-") and f.endswith(".json"))
    bundle = os.path.join(out, "carried.bundle")
    if os.path.exists(bundle):
        p = sh("git", "fetch", "-q", bundle, f"+refs/heads/carried/*:{PREFIX}/carried/*", f"+refs/heads/confirmed/*:{PREFIX}/confirmed/*")
        print(f"bundle fetched: exit {p.returncode} {p.stderr.strip()[-200:]}")
    else:
        print("no bundle yet")
    for n in names:
        try:
            r = json.load(open(os.path.join(out, f"restack-{n}.json")))[0]
        except Exception as e:
            print(f"RESULT {n}: no result yet ({type(e).__name__})")
            continue
        skipped = [s["args"][1].rsplit("/", 1)[-1] for s in r.get("skipped", [])]
        print(f"RESULT {n} ({r['head'][:12]}): status {r['status']}, result {str(r.get('result'))[:12]}, crossed {len(r.get('crossed', []))}, "
              f"skipped {len(skipped)} {skipped}, onto {str(r.get('onto'))[:12]}, plain {r['plain']['status']} ({len(r['plain']['files'])})")
        if r["status"] not in ("clean", "clean-skipped", "up-to-date"):
            print(f"   stopped at step {r.get('step')}, at the move {str(r.get('move'))[:12]}; files {r.get('files')}")
        if r.get("error"):
            print("   the tool's words:", r["error"][:1500].replace("\n", "\n   "))
        for s in r.get("steps", []):
            if s.get("step") == "pins":
                print("   pins step:", " | ".join(str(x) for x in s.get("output", [])[-4:])[:700])
        c = carried(n)
        if not c or c != r.get("result"):
            print(f"   carried commit here: {c and c[:12]} (the result names {str(r.get('result'))[:12]})")
            continue
        head = r["head"]
        print(f"   holds main {main_sha[:12]}: {rs.contains(main_sha, c, None)}; holds its own head: {rs.contains(head, c, None)}; "
              f"merges with main with no conflict: {not rs.merge_trees(main_sha, c, None)[1]}; tree {rs.tree_of(c, None)[:12]}")
        if skipped:
            grep_old(main_sha, c, [s["move"] for s in r["skipped"]], f"{n}, the {len(skipped)} skipped moves' maps")
        grep_old(main_sha, c, r.get("crossed", []), f"{n}, the {len(r.get('crossed', []))} crossed moves' maps")
    for child, parents in pairs.items():
        c = carried(child)
        ps = parents.split("+")
        if not c or not all(carried(p) for p in ps):
            print(f"TEST {child} over {parents}: not all carried here yet")
            continue
        m = main_sha
        for p in ps:
            t, conf = rs.merge_trees(m, carried(p), None)
            if conf:
                print(f"TEST {child} over {parents}: main with carried {p} conflicts: {conf}")
            m = rs.commit_tree(t, [m, carried(p)], f"scratch, local only: main with carried {p}\n", None)
        t, conf = rs.merge_trees(c, m, None)
        same = t == rs.tree_of(c, None)
        bases = sh("git", "merge-base", "--all", c, m).stdout.split()
        print(f"TEST {child} merged with a main that holds carried {parents}: conflicts {conf or 'none'}; merge bases {len(bases)}; "
              f"the merged tree is the line's own carried tree: {same}")
        if not same and not conf:
            d = sh("git", "diff", "--name-only", rs.tree_of(c, None), t).stdout.splitlines()
            print(f"   differs in {len(d)} files:", d[:30])
    if landing:
        m = main_sha
        for n in landing:
            c = carried(n)
            if not c:
                print(f"LANDING {n}: not carried here; left out")
                continue
            t, conf = rs.merge_trees(m, c, None)
            print(f"LANDING {n} onto main with those before it: {'no conflict' if not conf else conf}")
            if conf:
                break
            m = rs.commit_tree(t, [m, c], f"scratch, local only: main with carried {n}\n", None)


main()
