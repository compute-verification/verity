#!/usr/bin/env bash
# wipe_gate.sh, then on every exit the record's leak guard (leak_guard.py) on the run dir, before custody publishes it:
# the gate run's kept fetch (KEEP/RUN/fetch.tar.gz, with witness/) unpacked to KEEP/RUN/gate (directories 0700, files
# 0600, whatever modes its members carry), and each file of the
# run dir that holds a piece of its witness moved to KEEP/RUN/quarantine (the guard's exit 3; $OUT/leak-guard.json has
# counts and paths only). Same environment as wipe_gate.sh, inside the same `research run` (--cwd source). Exits with
# wipe_gate.sh's rc when it failed, else the guard's.
set -uo pipefail
: "${OUT:?}" "${RESEARCH_RUN_DIR:?}"
KEEP=$(realpath -m "${KEEP:-/workspace/research/lanes/network-accounting/gate}")
export KEEP

guard() {
  local rc=$? run g grc=0
  trap - EXIT
  run=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' "$OUT/start.json" 2>/dev/null) || run=
  if [ -n "$run" ] && [ -s "$KEEP/$run/fetch.tar.gz" ]; then
    g=$KEEP/$run/gate
    (umask 077 && python3 - "$KEEP/$run/fetch.tar.gz" "$g" <<'EOF'
import shutil, sys, tarfile
from pathlib import Path
dst = Path(sys.argv[2]).resolve()
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(sys.argv[1], "r:gz") as tar:
    for m in tar:
        p = (dst / m.name).resolve()
        if not p.is_relative_to(dst):
            sys.exit(f"the fetch's member {m.name!r} lies outside {dst}")
        if m.isdir():
            p.mkdir(parents=True, exist_ok=True)
        elif m.isfile():
            p.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(p, "wb") as out:
                shutil.copyfileobj(src, out)
EOF
    ) || { echo "leak guard: can't unpack $run's fetch"; exit 3; }
    python3 benchmarks/network_traces/leak_guard.py "$RESEARCH_RUN_DIR" "$KEEP/$run/quarantine" --gates "$g" \
      --report "$OUT/leak-guard.json" || grc=$?
  else
    echo "leak guard: no fetch kept (gate run ${run:-none}), so no witness of it on this machine to hold the record against"
  fi
  [ "$rc" -ne 0 ] && exit "$rc"
  exit "$grc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap guard EXIT
bash benchmarks/network_traces/wipe_gate.sh
