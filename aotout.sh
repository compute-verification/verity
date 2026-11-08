#!/usr/bin/env bash
# aotout.sh RUN: an aot run's summary tail, its comparison.json and its published outputs, between markers.
R=/workspace/research/runs/$1
echo "@@SUMMARY"; tail -12 $R/out/summary.txt
echo "@@COMPARISON"; cat $R/out/comparison.json; echo
echo "@@OUTPUTS"; python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('outputs')))" /workspace/research/store/attempts/$1.json
echo "@@END"
