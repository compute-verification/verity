#!/usr/bin/env bash
# artcust.sh RUN: the run's custody verdict and its published outputs, once custody has finished.
R=/workspace/research/runs/$1
python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('preserved',d.get('preserved'),'error',d.get('error'),d.get('verified_utc'))" $R/.custody 2>&1 | tail -1
f=/workspace/research/store/attempts/$1.json; [ -f "$f" ] || f=$(find /workspace/research -maxdepth 4 -path '*attempts*' -name "$1.json" 2>/dev/null | head -1)
echo "attempt $f"; [ -n "$f" ] && python3 -c "import json,sys;print('OUTPUTS',json.dumps(json.load(open(sys.argv[1])).get('outputs')))" "$f"
tail -2 $R/launcher.log | cut -c1-200
