set -u
# One batch of the carry: at most three branches abreast on this node. It pushes nothing.
MAIN_SHA=56f6c267563373883c093381ec5bf02a0ec80880
NAMES=(pr9496 pr9209 pr9595)
declare -A HEAD=([pr9496]=92579572ddcb8a320540f9e6d980501112ff941e [pr9209]=4c8742d0d449fe2d933f005ec9dc3c8ab6899686 [pr9595]=41a71836c34823cc2f7dcc54396ed776c7c8c5a9)
CONFIRM=0
[ "${#NAMES[@]}" -le 3 ] || { echo "REFUSED: more than three branches abreast on one node"; exit 2; }
echo "node $(hostname) cores $(nproc) $(python3 -V) git $(git --version | cut -d' ' -f3) uv $(uv --version 2>/dev/null | cut -d' ' -f2)"
: "${GIT_AUTHOR_NAME:?the launch names the git identity of the launching checkout, env GIT_AUTHOR_NAME=... ahead of bash}" "${GIT_AUTHOR_EMAIL:?}" "${GIT_COMMITTER_NAME:?}" "${GIT_COMMITTER_EMAIL:?}"
echo "identity of every commit this job makes: $GIT_AUTHOR_NAME <$GIT_AUTHOR_EMAIL> | $GIT_COMMITTER_NAME <$GIT_COMMITTER_EMAIL>"
TMPDIR="$(mktemp -d -p /workspace/research/scratch restack-pass-XXXXXXXX)" || { echo "REFUSED: no scratch under /workspace/research/scratch"; exit 2; }
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
trap 'exit 143' TERM INT HUP   # a cancel or a timeout still runs the line above
free() { echo "free: $(df -h --output=avail "$TMPDIR" | tail -1 | tr -d ' ') under the scratch, $(df -h --output=avail / | tail -1 | tr -d ' ') on the root disk ($1)"; }
echo "scratch $TMPDIR"; free "at the start"
echo "scratch of an earlier job of this kind still on this node: $(ls -d /workspace/research/scratch/restack-pass-* 2>/dev/null | grep -v -x "$TMPDIR" | wc -l)"
git branch -f pass/main "$MAIN_SHA" >/dev/null || exit 2
for n in "${NAMES[@]}"; do git branch -f "pass/$n" "${HEAD[$n]}" >/dev/null || exit 2; done

# 1. Each move's check first, three at a time: the move's script on the move's base in one scratch worktree, no pins step.
cat > "$TMPDIR/moves.py" <<'PY'
import json, sys
sys.path.insert(0, "tools/move")
import restack as rs
why = rs.drivers(sys.executable)
if why:
    raise SystemExit(f"restack: {why}")
seen, moves = set(), []
for b in sys.argv[2:]:
    for m, script, a in rs.moves_on("pass/main", b, None):
        if m not in seen:
            seen.add(m)
            moves.append([m, script, list(a)])
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
python3 "$TMPDIR/moves.py" "$TMPDIR/moves.json" "${NAMES[@]/#/pass/}" || exit 2
N=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$TMPDIR/moves.json")
if [ "$N" -gt 0 ]; then
  seq 0 $((N - 1)) | xargs -P 3 -I{} sh -c 'python3 "$0" "$1" {} 2>&1 | tail -n 1' "$TMPDIR/check.py" "$TMPDIR/moves.json"
fi
echo "CHECKS DONE at ${SECONDS}s"; free "after the checks"

# 2. The carries, abreast. --dry-run: report only; the commits stay in this clone until the bundle.
for n in "${NAMES[@]}"; do
  ( s=$SECONDS
    python3 tools/move/restack.py "pass/$n" --main pass/main --dry-run --kept --json > "$OUT/restack-$n.json" 2> "$OUT/restack-$n.err"
    echo "RESTACK $n EXIT $? in $((SECONDS - s))s (at ${SECONDS}s)" ) &
done
wait
echo "ALL RESTACKS DONE at ${SECONDS}s"; free "after the carries"
echo "left in the scratch: $(ls -d "$TMPDIR"/restack-* 2>/dev/null | wc -l) restack directories"

# 3. The results, the confirming command where asked, and the bundle.
cat > "$TMPDIR/results.py" <<'PY'
import json, os, subprocess, sys
out, confirm, names = os.environ["OUT"], sys.argv[1] == "1", sys.argv[2:]
keep = []
for n in names:
    try:
        r = json.load(open(os.path.join(out, f"restack-{n}.json")))[0]
    except Exception as e:
        print(f"RESULT {n}: no result ({e})")
        continue
    print(f"RESULT {n} ({r['head'][:12]}): status {r['status']}, result {str(r.get('result'))[:12]}, crossed {len(r.get('crossed', []))}, "
          f"skipped {len(r.get('skipped', []))}, step {r.get('step')}, at the move {str(r.get('move'))[:12]}, files {r.get('files')}")
    if r.get("error"):
        print("   error:", r["error"][:1200].replace("\n", "\n   "))
    for s in r.get("steps", []):
        if s.get("step") == "pins":
            print("   pins step:", " | ".join(str(x) for x in s.get("output", [])[-4:])[:600])
    kept = sum("as checked before" in x for x in open(os.path.join(out, f"restack-{n}.err")))
    print(f"   checks found kept: {kept}")
    if r.get("result") and r["status"] in ("clean", "clean-skipped", "locks-stale", "pins-stale"):
        subprocess.run(["git", "branch", "-f", f"carried/{n}", r["result"]], check=True)
        keep.append(f"carried/{n}")
    if r["status"] == "conflicted" and r.get("step") == "moved":
        # CARRIER: the branch as it stands after the last move's script, so the last merge can be made (with the move as
        # its base, as the tool makes it) and settled without a second carry
        tip = (r.get("steps") or [{}])[-1].get("commit")
        print(f"   stopped at the last merge: after the last move's script the branch is {tip}; the move {r.get('move')}; onto {r.get('onto')}")
        if tip:
            subprocess.run(["git", "branch", "-f", f"tip/{n}", tip], check=True)
            keep.append(f"tip/{n}")
    if r["status"] == "locks-stale":
        print(f"   locks-stale: {r.get('files')}")
    for s in r.get("skipped", []):
        print(f"   skipped the move {s['move'][:12]} {s['args']}")
    if confirm and r["status"] == "clean-skipped" and r.get("confirm"):
        cmd = [*r["confirm"], "--json"] if "--json" not in r["confirm"] else r["confirm"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        open(os.path.join(out, f"confirm-{n}.json"), "w").write(p.stdout)
        open(os.path.join(out, f"confirm-{n}.err"), "w").write(p.stderr)
        try:
            c = json.loads(p.stdout)[0]
            same = subprocess.run(["git", "rev-parse", f"{c['result']}^{{tree}}"], capture_output=True, text=True).stdout == \
                subprocess.run(["git", "rev-parse", f"{r['result']}^{{tree}}"], capture_output=True, text=True).stdout
            print(f"CONFIRM {n}: exit {p.returncode}, status {c['status']}, result {str(c.get('result'))[:12]}, skipped {len(c.get('skipped', []))}, "
                  f"the same tree as the carry's: {same}")
            if c.get("result") and c["status"] == "clean":
                subprocess.run(["git", "branch", "-f", f"confirmed/{n}", c["result"]], check=True)
                keep.append(f"confirmed/{n}")
        except Exception as e:
            print(f"CONFIRM {n}: exit {p.returncode}, no result ({e}): {p.stderr[-600:]}")
if keep:
    p = subprocess.run(["git", "bundle", "create", os.path.join(out, "carried.bundle"), *keep, "--not", "pass/main", *[f"pass/{n}" for n in names]],
                       capture_output=True, text=True)
    print("BUNDLE", keep, "exit", p.returncode, p.stderr.strip().splitlines()[-1:] if p.returncode else "")
    if not p.returncode:
        print("bundle bytes", os.path.getsize(os.path.join(out, "carried.bundle")))
else:
    print("BUNDLE: nothing carried, no bundle")
PY
python3 "$TMPDIR/results.py" "$CONFIRM" "${NAMES[@]}"
free "at the end"
echo "DONE at ${SECONDS}s"
