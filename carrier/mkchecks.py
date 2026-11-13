"""Write the job that checks one carried head on a node (the brief's step 5), launched from a worktree at that head.

    python3 mkchecks.py OUT.sh MAIN_SHA [--records PKG [--update]] [--build FILE...] [--no-suites]

The job runs, in the run's own clone of the carried head (`--cwd clone`):
  the suites whose inputs the change against MAIN_SHA touches, without the slow tests (`suites.py --changed MAIN --quick`),
  and beside them, where asked, the lane's fast path to PKG's records and its requirers' (`lean_changed.py --records PKG`,
  with `--update` only where asked) or the build of the named Lean files (`lean_changed.py --build FILE...`).
It keeps under $OUT: suites.log (last line `SUITES_EXIT n`), lean.log (last line `LEAN_EXIT n`), the fast path's
`result.json` and each lock it regenerated (`record/`), and `tree.txt` (what the job left changed in the clone).
"""
import shlex
import sys

args = sys.argv[1:]
out, main = args[0], args[1]
rest = args[2:]
suites = "--no-suites" not in rest
update = "--update" in rest
pkg = rest[rest.index("--records") + 1] if "--records" in rest else ""
build = []
if "--build" in rest:
    build = [a for a in rest[rest.index("--build") + 1:] if not a.startswith("--")]
assert len(main) == 40, "MAIN_SHA in full"
assert not (pkg and build), "one of --records and --build"

lean = ""
if pkg:
    lean = f"python3 tools/verity/lean/lean_changed.py --records {shlex.quote(pkg)}{' --update' if update else ''}"
elif build:
    lean = "python3 tools/verity/lean/lean_changed.py --build " + " ".join(shlex.quote(f) for f in build)

JOB = f"""set -u
MAIN={main}
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$MAIN" HEAD && echo "holds main $MAIN" || {{ echo "DOES NOT HOLD main $MAIN"; exit 8; }}
git status --short | head -5
uv sync -q --all-packages --extra torch-cpu || {{ echo "SYNC_FAILED"; exit 9; }}
"""
if suites:
    JOB += """( uv run tools/verity/check/suites.py --changed "$MAIN" --quick > "$OUT/suites.log" 2>&1; echo "SUITES_EXIT $?" >> "$OUT/suites.log" ) &
SUITES=$!
"""
if lean:
    JOB += f"""{lean} > "$OUT/lean.log" 2>&1; LEAN=$?
echo "LEAN_EXIT $LEAN" >> "$OUT/lean.log"
tail -25 "$OUT/lean.log"
RESEARCH_RUN_DIR="${{RESEARCH_RUN_DIR:-}}"
if [ -n "$RESEARCH_RUN_DIR" ] && [ -d "$RESEARCH_RUN_DIR/lean-audit" ]; then
  mkdir -p "$OUT/lean-audit"
  for f in result.json record allowed-record; do
    [ -e "$RESEARCH_RUN_DIR/lean-audit/$f" ] && cp -a "$RESEARCH_RUN_DIR/lean-audit/$f" "$OUT/lean-audit/"
  done
  ( cd "$RESEARCH_RUN_DIR/lean-audit" && find . -maxdepth 5 \\( -name lean-audit.json -o -name report.json -o -name result.json \\) | sort | head -120 ) > "$OUT/lean-audit/files.txt" 2>&1
fi
"""
else:
    JOB += "LEAN=0\n"
if suites:
    JOB += """wait $SUITES
tail -40 "$OUT/suites.log"
"""
JOB += """git status --short > "$OUT/tree.txt" 2>&1
git diff > "$OUT/tree.patch" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
"""
if suites:
    JOB += """S=$(tail -1 "$OUT/suites.log" | sed 's/SUITES_EXIT //')
echo "SUITES_EXIT $S LEAN_EXIT $LEAN"
[ "$S" = 0 ] && [ "$LEAN" = 0 ]
"""
else:
    JOB += """echo "LEAN_EXIT $LEAN"
[ "$LEAN" = 0 ]
"""
open(out, "w").write(JOB)
print(f"written {out}: main {main[:12]}; suites {'on' if suites else 'off'}; lean: {lean or 'none'}")
