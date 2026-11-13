uv sync -q --all-packages --extra torch-cpu </dev/null && git rev-parse HEAD
I=$RESEARCH_RUN_DIR/inputs
/usr/bin/time -v .venv/bin/python $I/p3_census.py $OUT 78 0 frame,dense,sparse 2> $OUT/time_census78.txt; echo census rc=$?
rg "Maximum resident|Elapsed" $OUT/time_census78.txt
/usr/bin/time -v .venv/bin/python $I/p3_keycost.py $OUT 78 2> $OUT/time_keycost78.txt; echo keycost rc=$?
rg "Maximum resident|Elapsed" $OUT/time_keycost78.txt
