#!/bin/bash
# One integrator pass. Exit 2: conflict or compile failure; exit 4: a map leaves sites; exit 3: moved again.
cd /home/ubuntu/wt-collect || exit 1
A=origin/architecture; C=origin/cursor/na-architecture-b51f
git fetch -q origin architecture cursor/na-architecture-b51f || exit 1
S=$(git rev-parse HEAD)
echo "time $(TZ=America/Los_Angeles date +%H:%M) head $(git rev-parse --short=12 HEAD) arch $(git rev-parse --short=12 $A) behind $(git rev-list --count HEAD..$A) collect $(git rev-parse --short=12 $C) behind $(git rev-list --count HEAD..$C)"
mapcommits=$(git log --no-merges --format='%h' $A ^HEAD | while read c; do
  s=$(git show -s --format=%s $c); case "$s" in *restack*) continue;; esac
  git show -s --format=%B $c | grep -qE 'tools/move/(rename|layout)\.py --map|^Move by |^Rename by |^(Move|Layout-move): ' && echo $c; done)
maps=""
for c in $mapcommits; do
  for m in $(git show -s --format=%B $c | grep -oE 'tools/move/[a-z_0-9]+\.toml' | sort -u); do
    git cat-file -e $A:$m 2>/dev/null || continue
    ex=$(git show $A:$m | grep -c 'accounting/certifier/\*\*\|security/\*\*/Certifier/\*\*')
    lay=$(git show $A:$m | grep -c '^\[paths\]')
    cf=$(git show --name-only --format= $c | grep -cE "accounting/certifier/|/Certifier/"); echo "MAP $c $m exclude=$ex layout=$lay certfiles=$cf :: $(git show -s --format=%s $c | cut -c1-80)"
    maps="$maps $m"
  done
done
for R in $A $C; do
  git merge-base --is-ancestor $R HEAD && continue
  if git merge-base --is-ancestor HEAD $R; then git merge -q --ff-only $R || exit 1; continue; fi
  pre=$(git rev-parse HEAD)
  if ! git merge -q --no-ff --no-edit -m "Merge ${R#origin/} ($(git rev-parse --short=12 $R)) into the branch carried onto architecture" $R >/tmp/pass_merge.log 2>&1; then
    AL=tests/lint/allowlists/repo_retired_terms.json
    if git diff --name-only --diff-filter=U | grep -qx $AL; then git show $A:$AL > $AL; git add $AL; echo "ALLOWLIST took architecture's side"; fi
    u=$(git diff --name-only --diff-filter=U)
    if [ -z "$u" ]; then git commit -q --no-edit -m "Merge ${R#origin/} ($(git rev-parse --short=12 $R)) into the branch carried onto architecture" -m "tests/lint/allowlists/repo_retired_terms.json takes architecture's side." || exit 2; else
    if [ -n "$u" ] && ! echo "$u" | grep -qv 'lean-audit\.json$'; then
      for f in $u; do d=$(mktemp -d); git show :1:$f > $d/b; git show :2:$f > $d/o; git show :3:$f > $d/t
        python3 tools/verity/lean/merge.py $d/b $d/o $d/t $f || { echo "LOCK MERGE FAILED $f"; exit 2; }
        cp $d/o $f; git add $f; echo "LOCK merged by the tree's merge.py: $f"; done
      git commit -q --no-edit -m "Merge ${R#origin/} ($(git rev-parse --short=12 $R)) into the branch carried onto architecture" -m "Lock files merged by the tree's tools/verity/lean/merge.py (format 2), which main's driver refused: $(echo $u)" || exit 2
    else
      echo "CONFLICT merging $R"; git status --porcelain | grep -E '^(DU|UD|AU|UA|DD|AA|UU) '; exit 2
    fi; fi
  fi
  bad=0; for f in $(git diff --name-only --diff-filter=AM $pre HEAD -- '*.py'); do python3 -m py_compile "$f" 2>/dev/null || { echo "COMPILE $f"; bad=1; }; done
  [ $bad = 0 ] || { echo "COMPILE FAIL after merging $R"; exit 2; }
done
AL=tests/lint/allowlists/repo_retired_terms.json
if ! git diff --quiet $A HEAD -- $AL; then git show $A:$AL > $AL; git add $AL; git commit -q -m "repo_retired_terms.json takes architecture's side" && echo "ALLOWLIST set to architecture's"; fi
python3 /tmp/allowcheck.py | sed 's/^/AL: /'
if [ "$(git rev-parse HEAD)" != "$S" ]; then kc=$(KEEPREV=HEAD python3 /tmp/keptcheck.py $S 2>&1); echo "$kc" | sed 's/^/KEPT: /' | grep -E 'LOST|tag files|kept'; echo "$kc" | grep -q ': OK$' || [ "${FORCE:-0}" = 1 ] || { echo "KEPT FAIL: inspect before pushing"; exit 5; }; fi
sites=0
for m in $(echo $maps | tr ' ' '\n' | sort -u); do
  if git show HEAD:$m | grep -q '^\[paths\]'; then
    lc=$(python3 /tmp/laycheck.py $m); echo "LAYOUT $lc" | head -8
    echo "$lc" | grep -q 'naming one: 0' || sites=1
    continue
  fi
  out=$(python3 tools/move/rename.py --map $m --dry-run 2>&1)
  n=$(echo "$out" | grep -oE 'files rewritten: [0-9]+' | grep -oE '[0-9]+$')
  if [ -z "$n" ]; then echo "DRY $m: $(echo "$out" | head -1 | cut -c1-90)"; elif [ "$n" != 0 ]; then echo "DRY $m: $n files"; echo "$out" | head -12; sites=1; else echo "DRY $m: 0"; fi
done
[ $sites = 0 ] || [ "${FORCE:-0}" = 1 ] || { echo "SITES: inspect before pushing"; exit 4; }
git fetch -q origin architecture cursor/na-architecture-b51f
if git merge-base --is-ancestor $A HEAD && git merge-base --is-ancestor $C HEAD; then
  git push --atomic origin HEAD:refs/heads/architecture HEAD:refs/heads/cursor/na-architecture-b51f 2>&1 | grep -E '\->|rejected|error'
  git fetch -q origin architecture; git merge-base --is-ancestor HEAD $A && echo "pushed $(git rev-parse --short=12 HEAD)" || { echo PUSH-REJECTED; exit 3; }
else echo "MOVED again; rerun"; exit 3; fi
for f in $(git grep -l 'accounting/certifier/\*\*' HEAD -- 'tools/move/*_map.toml' | sed 's/^HEAD://'); do git show HEAD:$f | grep -q 'security/\*\*/Certifier/\*\*' && basename $f _map.toml; done | sort > /tmp/globmaps.now
comm -13 <(sort /tmp/globmaps.txt) /tmp/globmaps.now | while read m; do echo "NEWGLOB $m $(git log -1 --format=%h -E --grep="${m}_map.toml" HEAD)"; done
w=$(git diff $S HEAD -U0 -- . ':!tools/move' ':!tests/lint/allowlists' ':!**/lean-audit.json' | grep -E '^\+[^+]' | grep -ciE 'warden'); [ "$w" = 0 ] || echo "WARDEN $w new lines name warden since $(git rev-parse --short=12 $S)"
