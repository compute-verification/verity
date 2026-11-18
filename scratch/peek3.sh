for r in "$@"; do D=/workspace/research/runs/$r; echo "######## $r"
  grep -E 'RECORD|CASE |LEAN_BUILT|build failed|^error|record-check|wired-|PY_TWIN|OWN_DIAG|Traceback' $D/stdout.log | cut -c1-900
  for f in share.txt probe.txt rc.txt; do [ -f $D/out/$f ] && { echo "--- $f"; cut -c1-1500 $D/out/$f; }; done
  grep '"state"' $D/status.json 2>/dev/null | tail -1
done
