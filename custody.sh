#!/usr/bin/env bash
# custody.sh RUN: whether the run's custody publish finished and preserved it.
R=/workspace/research/runs/$1
python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[2],'preserved',d.get('preserved'),'error',d.get('error'),d.get('verified_utc'))" $R/.custody $1 2>&1 | tail -1
tail -2 $R/launcher.log | cut -c1-200
