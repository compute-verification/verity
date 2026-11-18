# Sourced with CARD, log() and a directory: a gpu-lease there, first on PATH, that leases card CARD alone; `real` is the node's.
real=$(command -v gpu-lease) || { log "CARD=$CARD is held under gpu-lease, which this node lacks: nothing run"; return 2; }
mkdir -p "$1" || return 2
printf '#!/usr/bin/env bash\n# leases card %s alone: --on %s after the count\n[ "$1" = status ] && exec %q status\nn=$1; shift\nexec %q "$n" --on %s "$@"\n' \
  "$CARD" "$CARD" "$real" "$real" "$CARD" > "$1/gpu-lease" && chmod +x "$1/gpu-lease" || return 2
export PATH=$1:$PATH
