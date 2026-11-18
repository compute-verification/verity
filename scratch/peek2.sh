for r in "$@"; do D=/workspace/research/runs/$r; echo "######## $r"
  grep -E 'LEAN_BUILT|error|record-check|wired-time' $D/stdout.log | cut -c1-900
  [ -f $D/out/probe.txt ] && { echo '--- probe.txt'; cut -c1-700 $D/out/probe.txt; }
  [ -f $D/out/rc.txt ] && { echo '--- rc.txt'; cut -c1-900 $D/out/rc.txt; }
  grep -m4 -E 'WIRES|EDGES|PY_TWIN|Traceback|rror' $D/stdout.log | cut -c1-2500
  grep '"state"' $D/status.json 2>/dev/null | tail -1
done
