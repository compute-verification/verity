#!/usr/bin/env bash
# flock-circuit stacked, candidate 3's opening-only prover (lane rec-v0): tests/test_rec_c3_open.py against this tree's prover
# build (rec-env.sh's $B: 60-circuit.sh MODE=build GPU=0 on Flock's patched checkout). Its openings at m_R 22 and 23 are
# accepted by the opening-only model, and a forged cell and a forged claim value are refused. Run from the verity tree
# (research run --cwd source), pinned by its caller (slot.py). Outputs under out/: host.txt, build.log, binary.sha256,
# pytest.txt, junit.xml.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
export RI_PINNED=${RI_PINNED:-$(taskset -pc $$ | cut -d: -f2 | tr -d ' ')} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$(nproc)}
. verity/ml/flock/pod/rec-env.sh
python3 -c 'import pytest' 2>/dev/null || $UV pip install -q --python $W/py/bin/python pytest || exit 1
FLOCK_CIRCUIT=$B PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider -v --junitxml=$O/junit.xml \
  verity/ml/flock/tests/test_rec_c3_open.py > $O/pytest.txt 2>&1
rc=$?
tail -8 $O/pytest.txt
wait $LP
exit $rc
