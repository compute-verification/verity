#!/usr/bin/env bash
# CIR-263: which catalog build imports verity.channel. circuit-check's records-only build of the whole catalog, as the suite's
# session runs it, into an empty cache, with an audit hook (a sitecustomize on PYTHONPATH) that writes the stack of every import
# of verity.channel and every open of verity/channel.py to $OUT/channel.txt.  Runs in a private clone (--cwd clone).
set -uo pipefail
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
C=$(mktemp -d); H=$(mktemp -d)
cat > "$H/sitecustomize.py" <<'EOF'
import os, sys, traceback

_LOG = os.environ.get("CHANNEL_LOG")


def _hook(event, args):
    if not _LOG:
        return
    hit = (event == "import" and args and args[0] == "verity.channel") or (
        event == "open" and args and isinstance(args[0], (str, bytes)) and os.fsdecode(args[0]).endswith("verity/channel.py"))
    if hit:
        with open(_LOG, "a") as f:
            f.write(f"--- pid {os.getpid()} {event} argv {sys.argv[:3]}\n{''.join(traceback.format_stack()[:-1])}\n")


sys.addaudithook(_hook)
EOF
date -u +"build %FT%TZ"
CHANNEL_LOG="$OUT/channel.txt" PYTHONPATH="$H" uv run python -m circuit_check --all --records-only --cache "$C" --jobs 24 \
  --mem-gb 28 --out "$OUT/records.json" > "$OUT/records.log" 2>&1
rc=$?
echo "build rc=$rc"; tail -5 "$OUT/records.log"
if [ -f "$OUT/channel.txt" ]; then
  echo "verity.channel touched $(grep -c '^--- ' "$OUT/channel.txt") time(s); the first stack:"
  awk 'NR==1,/^$/' "$OUT/channel.txt" | tail -40
else
  echo "verity.channel never touched"
fi
date -u +"end %FT%TZ"
exit $rc
