"""For each head: `git merge-tree` of the head with the staged head (a plain merge, at the staged tree's root with the
allowlist driver at its moved path), beside the tree restack.py made. Prints one line a head and writes plain-merge.json."""
import json, subprocess, sys

STAGED = "178f49c90c89734ea185f01496a7b6fbbd90b88c"
DRIVER = "merge.vllm-allowlist.driver=python3 adapters/vllm/tests/lint/merge_allowlist.py %O %A %B %P"
WT = "/tmp/cir223b/staged"


def git(*a, ok=(0,)):
    p = subprocess.run(["git", "-c", DRIVER, *a], cwd=WT, capture_output=True)
    if p.returncode not in ok:
        raise SystemExit(f"git {a}: {p.stderr.decode()[:600]}")
    return p


def one(name, head, result):
    p = git("merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", head, STAGED, ok=(0, 1))
    parts = [s.decode() for s in p.stdout.split(b"\0") if s]
    tree, conflicts = parts[0], sorted(set(parts[1:]))
    rec = {"name": name, "head": head, "result": result, "merge_tree": tree, "merge_exit": p.returncode, "conflicts": conflicts}
    if result:
        rt = git("rev-parse", result + "^{tree}").stdout.decode().strip()
        rec["equal"] = rt == tree
        if not rec["equal"]:
            names = [n for n in git("diff", "--name-only", "--no-renames", tree, rt).stdout.decode().splitlines() if n]
            rec["differ_files"] = names
            rec["shortstat"] = " ".join(git("diff", "--shortstat", "--no-renames", tree, rt).stdout.decode().split())
    return rec


rows = [one(*spec.split(",")) for spec in sys.argv[1:]]
for r in rows:
    print(f"#{r['name']} head {r['head'][:12]} result {(r['result'] or '-')[:12]}: plain merge exit {r['merge_exit']}, "
          f"{len(r['conflicts'])} conflicted; equal to the restacked tree: {r.get('equal')}; "
          f"{r.get('shortstat', '')}; files that differ: {len(r.get('differ_files', []))} {r.get('differ_files', [])[:6]}; "
          f"conflicted: {r['conflicts'][:6]}")
json.dump(rows, open("/tmp/cir223b/work/plain-merge.json", "w"), indent=1)
