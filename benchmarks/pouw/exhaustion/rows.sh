# Sourced by window.sh and audit.sh: the life of the bench's Pearl-C rows (A's and B's operand rows, 512 MB a card at 8,192²),
# which the arm's transcripts name by absolute path under window.sh's scratch (SCRATCH_ROOT/exhaustion.XXXXXX/rows).  With
# AUDIT=0 the window keeps them, marked, for audit.sh's harness verify on the same machine; audit.sh releases them after it, and
# a kept dir whose mark is older than ROWS_KEEP_HOURS (default 6) is swept by the next window.sh or audit.sh that starts there.
# Only marked dirs are swept: a window still waiting for its lease has none, however old.
ROWS_MARK=.kept-rows

rows_exit() {   # rows_exit SCR OUT: window.sh's exit; with AUDIT=0 and rows written, keep SCR's rows only, else remove SCR
  local scr=$1 out=$2
  if [ "${AUDIT:-1}" = 0 ] && [ -n "$(ls -A "$scr/rows" 2>/dev/null)" ]; then
    find "$scr" -mindepth 1 -maxdepth 1 ! -name rows -exec rm -rf {} +
    date -u +%FT%TZ > "$scr/$ROWS_MARK"
    echo "$scr" > "$out/kept-rows"
    echo "rows: kept $scr ($(du -sh "$scr" | cut -f1)) for audit.sh"
  else
    rm -rf "$scr"
  fi
}

rows_sweep() {   # rows_sweep ROOT: remove this user's kept dirs under ROOT whose mark is older than ROWS_KEEP_HOURS
  local m d
  find "$1" -mindepth 2 -maxdepth 2 -name "$ROWS_MARK" -type f -user "$(id -u)" -mmin +$(( ${ROWS_KEEP_HOURS:-6} * 60 )) \
    2>/dev/null | while read -r m; do
      d=${m%/*}
      case ${d##*/} in exhaustion.*) rm -rf "$d" && echo "rows: swept $d (kept over ${ROWS_KEEP_HOURS:-6} h)" ;; esac
    done
}

rows_kept() {   # rows_kept OUT: the kept dir OUT/kept-rows names, if it is one on this machine
  local d
  d=$(cat "$1/kept-rows" 2>/dev/null) || return 1
  case ${d##*/} in exhaustion.*) ;; *) return 1 ;; esac
  [ -d "$d" ] && [ ! -L "$d" ] && [ -f "$d/$ROWS_MARK" ] && echo "$d"
}

rows_claim() {   # rows_claim OUT: renew the kept dir's mark, so that no sweep takes it while this audit runs
  local d
  d=$(rows_kept "$1") && touch "$d/$ROWS_MARK"
  return 0
}

rows_check() {   # rows_check OUT: every file a bench-before transcript names outside OUT is here, at its recorded size
  "${PY:-python3}" - "$1" <<'EOF'
import json, sys
from pathlib import Path
out, named = Path(sys.argv[1]), {}
for m in sorted(out.glob("bench-before/card*/*/*/*/manifest.json")):
    for spec in json.loads(m.read_text()).get("files", {}).values():
        if Path(spec["path"]).is_absolute():
            named.setdefault(spec["path"], spec["bytes"])
missing = [p for p, n in named.items() if not Path(p).is_file() or Path(p).stat().st_size != n]
for p in missing[:4]:
    print(f"rows: missing {p}")
print(f"rows: {len(named) - len(missing)} of {len(named)} files the bench transcripts name outside OUT are here")
sys.exit(1 if missing else 0)
EOF
}

rows_release() {   # rows_release OUT: remove the kept dir OUT/kept-rows names, once its verify is done
  local d
  d=$(rows_kept "$1") || return 0
  rm -rf "$d" && echo "rows: released $d"
}
