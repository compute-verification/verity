"""Write the job that carries one branch across the moves of a main commit, stopping where a move's base conflicts.

    python3 mkstage.py OUT.sh MAIN_SHA NAME=SHA [BASE_SHA=TAKE_SHA ...]

The job is mkpass.py's for one branch, with one more thing: where tools/move/restack.py stops in step 1 of a move (the
move's base conflicts with the branch), it crosses the moves before it by name (the tool's own command, which gives the
branch as it stands there), and then, for a base named BASE_SHA here, merges TAKE_SHA instead: a commit of another carried
line that already holds that base, with its conflicts settled there. It makes that merge only when git's own merge of
the two is clean, or when the merge against TAKE_SHA's first parent (the other line as it stood before it merged the base)
is; it says which, and whether the two trees are equal. Then it goes on across the moves left. Anywhere else it stops, and
the bundle holds the branch as far as it came. It pushes nothing: a node's clone has no remote it can reach.
"""
import sys

out, main, pair, *takes = sys.argv[1:]
name, head = pair.split("=", 1)
for n, s in [("main", main), (name, head), *[x for t in takes for x in zip(("base", "take"), t.split("=", 1))]]:
    if len(s) != 40 or any(c not in "0123456789abcdef" for c in s):
        raise SystemExit(f"{n}: a full commit id, not {s!r}")

JOB = r'''set -u
# One branch carried across the moves of a main commit, in stages. It pushes nothing.
MAIN_SHA=@MAIN@
NAME=@NAME@
HEAD_SHA=@HEAD@
TAKES="@TAKES@"
echo "node $(hostname) cores $(nproc) $(python3 -V) git $(git --version | cut -d' ' -f3) uv $(uv --version 2>/dev/null | cut -d' ' -f2)"
: "${GIT_AUTHOR_NAME:?the launch names the git identity of the launching checkout, env GIT_AUTHOR_NAME=... ahead of bash}" "${GIT_AUTHOR_EMAIL:?}" "${GIT_COMMITTER_NAME:?}" "${GIT_COMMITTER_EMAIL:?}"
echo "identity of every commit this job makes: $GIT_AUTHOR_NAME <$GIT_AUTHOR_EMAIL> | $GIT_COMMITTER_NAME <$GIT_COMMITTER_EMAIL>"
for s in $MAIN_SHA $HEAD_SHA $(echo $TAKES | tr '=' ' '); do git cat-file -e "$s^{commit}" 2>/dev/null || { echo "REFUSED: the clone has no commit $s"; exit 2; }; done
TMPDIR="$(mktemp -d -p /workspace/research/scratch restack-pass-XXXXXXXX)" || { echo "REFUSED: no scratch under /workspace/research/scratch"; exit 2; }
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
trap 'exit 143' TERM INT HUP   # a cancel or a timeout still runs the line above
free() { echo "free: $(df -h --output=avail "$TMPDIR" | tail -1 | tr -d ' ') under the scratch, $(df -h --output=avail / | tail -1 | tr -d ' ') on the root disk ($1)"; }
echo "scratch $TMPDIR"; free "at the start"
git branch -f pass/main "$MAIN_SHA" >/dev/null || exit 2
git branch -f "pass/$NAME" "$HEAD_SHA" >/dev/null || exit 2

# 1. Each move's check first, three at a time: the move's script on the move's base in one scratch worktree, no pins step.
cat > "$TMPDIR/moves.py" <<'PY'
import json, sys
sys.path.insert(0, "tools/move")
import restack as rs
why = rs.drivers(sys.executable)
if why:
    raise SystemExit(f"restack: {why}")
moves = [[m, script, list(a)] for m, script, a in rs.moves_on("pass/main", sys.argv[2], None)]
json.dump(moves, open(sys.argv[1], "w"))
print(f"moves to cross: {len(moves)}")
PY
cat > "$TMPDIR/check.py" <<'PY'
import json, sys, time
sys.path.insert(0, "tools/move")
import restack as rs
moves = json.load(open(sys.argv[1]))
i = int(sys.argv[2])
move, script, a = moves[i]
rs.KEEP = True
rs.SCRIPT, rs.SCRIPT_ARGS = script, tuple(a)
t = time.time()
rs.check_generated(move, rs.move_base(move, None), sys.executable, None)
print(f"CHECK {i + 1} of {len(moves)} {move[:12]} {a[-1] if a else script}: is what its script writes on its base, {time.time() - t:.0f}s", flush=True)
PY
python3 "$TMPDIR/moves.py" "$TMPDIR/moves.json" "pass/$NAME" || exit 2
N=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$TMPDIR/moves.json")
if [ "$N" -gt 0 ]; then
  seq 0 $((N - 1)) | xargs -P 3 -I{} sh -c 'python3 "$0" "$1" {} 2>&1 | tail -n 1' "$TMPDIR/check.py" "$TMPDIR/moves.json"
fi
echo "CHECKS DONE at ${SECONDS}s"; free "after the checks"

# 2. The carry, in stages: the tool across the moves it finds; where it stops at a move's base, the moves before it by
#    name, the merge this job was told to take for that base, and the tool again from there.
cat > "$TMPDIR/stage.py" <<'PY'
import json, os, subprocess, sys, time
sys.path.insert(0, "tools/move")
import restack as rs
out, name = os.environ["OUT"], sys.argv[1]
takes = dict(t.split("=", 1) for t in sys.argv[2:])
keep, tip, stage = [], f"pass/{name}", 0

def tool(branch, *args):
    t = time.time()
    p = subprocess.run(["python3", "tools/move/restack.py", branch, *args, "--dry-run", "--kept", "--json"], capture_output=True, text=True)
    tag = f"{name}-stage{stage}-{'named' if '--move-commit' in args else 'found'}"
    open(os.path.join(out, f"restack-{tag}.json"), "w").write(p.stdout)
    open(os.path.join(out, f"restack-{tag}.err"), "w").write(p.stderr)
    try:
        r = json.loads(p.stdout)[0]
    except Exception as e:
        print(f"STAGE {stage}: no result ({e}): {p.stderr[-800:]}")
        return None
    print(f"STAGE {stage} {'named' if '--move-commit' in args else 'found'} on {rs.rev(branch, None)[:12]}: exit {p.returncode}, status {r['status']}, "
          f"result {str(r.get('result'))[:12]}, crossed {len(r.get('crossed', []))}, skipped {len(r.get('skipped', []))}, step {r.get('step')}, "
          f"at the move {str(r.get('move'))[:12]}, base {str(r.get('base'))[:12]}, files {r.get('files')}, {time.time() - t:.0f}s", flush=True)
    if r.get("error"):
        print("   error:", r["error"][:3000].replace("\n", "\n   "))
    for w in r.get("words", [])[:60]:
        print("   stale word:", json.dumps(w)[:300])
    for s in r.get("skipped", []):
        print(f"   skipped the move {s['move'][:12]} {s['args']}")
    return r

def branch(ref, commit):
    subprocess.run(["git", "branch", "-f", ref, commit], check=True)
    keep.append(ref)

while stage < 6:
    stage += 1
    r = tool(tip, "--main", "pass/main")
    if r is None:
        break
    if r.get("result") and r["status"] in ("clean", "clean-skipped", "locks-stale", "pins-stale", "stale-words", "up-to-date"):
        branch(f"carried/{name}", r["result"])
        print(f"CARRIED {name}: {r['result']} status {r['status']}")
        break
    if not (r["status"] == "conflicted" and r.get("step") == "base"):
        print(f"STOPPED {name}: status {r['status']} at step {r.get('step')}")
        break
    base, crossed = r["base"], r.get("crossed", [])
    if crossed:
        c = tool(tip, *[x for m in crossed for x in ("--move-commit", m)], "--main", "pass/main")
        if c is None or not c.get("result") or c["status"] not in ("clean", "stale-words"):
            print(f"STOPPED {name}: the moves before the stop, by name, are not clean")
            break
        tip = c["result"]
    else:
        tip = rs.rev(tip, None)
    branch(f"tip/{name}-before-{base[:12]}", tip)
    print(f"AT THE STOP: the branch as it stands before the base {base[:12]} is {tip}; its merge with the base conflicts in {r.get('files')}")
    take = takes.get(base)
    if not take:
        print(f"STOPPED {name}: no merge was named for the base {base[:12]}")
        break
    if not rs.contains(base, take, None):
        print(f"STOPPED {name}: {take[:12]} does not hold the base {base[:12]}")
        break
    before = rs.rev(f"{take}^1", None)
    t_plain, c_plain = rs.merge_trees(tip, take, None)
    t_forced, c_forced = rs.merge_trees(tip, take, None, base=before)
    print(f"TAKE {take[:12]} for the base {base[:12]}: git's own merge conflicts in {c_plain}; the merge against its first parent "
          f"{before[:12]} conflicts in {c_forced}; the two trees are {'equal' if t_plain == t_forced else 'NOT equal'}")
    if not c_plain:
        tree, how = t_plain, "git's own merge of the two, which is clean"
    elif not c_forced:
        tree, how = t_forced, f"the merge of the two with {before[:12]} as its base (the other line as it stood before it merged that base), which is clean; git's own merge conflicts in {', '.join(c_plain)}"
    else:
        print(f"STOPPED {name}: both merges of {take[:12]} conflict")
        break
    if not c_plain and t_plain != t_forced:
        d = subprocess.run(["git", "diff", "--stat", t_plain, t_forced], capture_output=True, text=True).stdout
        print("   the two trees differ in:\n   " + d[-1500:].replace("\n", "\n   "))
    msg = (f"Merge {take[:12]}, the carried line of #9458 where it merged {base[:12]}, a move's base, into the carry of #9636\n\n"
           f"tools/move/restack.py stops the carry of #9636 at this base: it conflicts with the branch in {', '.join(r.get('files') or [])},\n"
           f"in lines #9458 changed, which #9636 holds. The carried line of #9458 met the same conflict and settled it in {take[:12]}.\n"
           f"This merge takes that commit, so the settlement is #9458's own and is made once. Its tree is {how}.\n")
    tip = rs.commit_tree(tree, [tip, take], msg, None)
    branch(f"took/{name}-{take[:12]}", tip)
    print(f"TOOK {take[:12]}: the branch is now {tip}")
if keep:
    p = subprocess.run(["git", "bundle", "create", os.path.join(out, "carried.bundle"), *keep, "--not", "pass/main", f"pass/{name}", *takes.values()],
                       capture_output=True, text=True)
    print("BUNDLE", keep, "exit", p.returncode, p.stderr.strip().splitlines()[-1:] if p.returncode else "")
    if not p.returncode:
        print("bundle bytes", os.path.getsize(os.path.join(out, "carried.bundle")))
else:
    print("BUNDLE: nothing kept, no bundle")
PY
python3 "$TMPDIR/stage.py" "$NAME" $TAKES
free "at the end"
echo "left in the scratch: $(ls -d "$TMPDIR"/restack-* 2>/dev/null | wc -l) restack directories"
echo "DONE at ${SECONDS}s"
'''
text = JOB.replace("@MAIN@", main).replace("@NAME@", name).replace("@HEAD@", head).replace("@TAKES@", " ".join(takes))
open(out, "w").write(text)
print(f"written {out}: main {main[:12]}, {name} at {head[:12]}, takes {takes}")
