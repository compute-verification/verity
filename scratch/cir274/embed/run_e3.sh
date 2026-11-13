set -u
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD | tee $OUT/head.txt; git status --short | tee $OUT/status.txt
uv sync -q --all-packages --extra torch-cpu
bash $I/watch.sh census4 .venv/bin/python $I/embed_census.py $OUT 4 4 > $OUT/stdout_census4.txt 2>&1; tail -2 $OUT/stdout_census4.txt | cut -c1-300
bash $I/watch.sh census5 .venv/bin/python $I/embed_census.py $OUT 4 5 > $OUT/stdout_census5.txt 2>&1; tail -2 $OUT/stdout_census5.txt | cut -c1-300
