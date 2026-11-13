uv sync -q --all-packages --extra torch-cpu </dev/null && git rev-parse HEAD
I=$RESEARCH_RUN_DIR/inputs
.venv/bin/python $I/p3_names.py $OUT 4; echo names rc=$?
for a in "43 2" "43 3" "86 2"; do
  set -- $a
  /usr/bin/time -v .venv/bin/python $I/p3_keycost.py $OUT 4 $1 $2 2> $OUT/time_keycost_$1_$2.txt; echo keycost $1 $2 rc=$?
  rg "Maximum resident|Elapsed" $OUT/time_keycost_$1_$2.txt
done
echo "store search (a capture's params.json)"
uv run -q --project tools/research research data select --ids 2>&1 | tail -3
uv run -q --project tools/research research data sql "select id, kind, substr(manifest_json,1,240) from artifacts where manifest_json like '%params.json%' or manifest_json like '%served_param%' limit 40" 2>&1 | tail -45
