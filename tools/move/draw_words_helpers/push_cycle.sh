#!/bin/bash
set -e
cd /agent/repos/seat
git fetch -q origin architecture
git reset -q --hard origin/architecture
git clean -fdq
echo "tip $(git rev-parse --short=11 HEAD)"
cp /tmp/kit/p1_final.toml tools/move/draw_words_map.toml
python3 /tmp/kit/exclcheck.py /agent/repos/seat tools/move/draw_words_map.toml | head -3
git add tools/move/draw_words_map.toml
git commit -q -m "tools/move/draw_words_map.toml: the retired word draw renamed to the sample, first piece

The paths no other lane's draw map runs over; the offered paths (the vLLM adapter, PoUW, benchmarks/pouw, the
certifier and its benchmarks, benchmarks/slowdown) and memory accounting's files are excluded."
python3 tools/move/rename.py --map tools/move/draw_words_map.toml > /tmp/kit/push1.out 2>&1
sed -n 2p /tmp/kit/push1.out
git diff --cached --name-only --diff-filter=AMR -- '*.py' | xargs -r python3 -m py_compile && echo COMPILE-OK
git diff --cached --name-only > /tmp/kit/push_files.txt; wc -l < /tmp/kit/push_files.txt
sort /tmp/kit/push_files.txt | diff /tmp/kit/push_expected.txt - > /tmp/kit/push_filediff.txt || { echo "FILES DIFFER"; cat /tmp/kit/push_filediff.txt; exit 3; }
python3 -m pytest tests/lint/test_repo_retired_terms.py -q -p no:cacheprovider 2>&1 | tail -1 | tee /tmp/kit/push_lint.txt
grep -q ' passed' /tmp/kit/push_lint.txt && ! grep -q failed /tmp/kit/push_lint.txt || { echo LINT-RED; exit 4; }
[ "$(compgen -e | grep -c -E '^GIT_(AUTHOR|COMMITTER)')" = 0 ]
git commit -q -m "Rename by tools/move/draw_words_map.toml

Move: tools/move/draw_words_map.toml"
git push origin HEAD:architecture 2>&1 | tail -1
git log --oneline -2
