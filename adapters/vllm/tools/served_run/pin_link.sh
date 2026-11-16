# Sourced by window.sh and setup.sh, from the tree's root: the engine reads an INT8 pin at its checkpoint manifest's
# local_path (under /workspace/hf) whatever HF_HOME is, so that model directory becomes one link to the pin under
# HF_HOME, made where nothing is at that name and removed at exit. A model the INT8 manifest doesn't name (the BF16 base)
# is read where the node keeps it, with no link. Needs REPO, REV, HF_HOME and log(); returns 2 where it can't link.
local_path=$(python3 -c 'import json, sys; print(next((c["local_path"] for c in json.load(open(sys.argv[1]))["checkpoints"] if c["repo"] == sys.argv[2] and c["revision"] == sys.argv[3]), ""))' \
  adapters/vllm/manifests/checkpoints-int8.json "$REPO" "$REV") || return 2
if [ -n "$local_path" ]; then
  LINK=${local_path%/snapshots/*} SRC=$HF_HOME/hub/models--${REPO//\//--}
  if ! { [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$SRC" ]; }; then
    { [ ! -e "$LINK" ] && [ ! -L "$LINK" ]; } || { log "$LINK exists and is not a link to $SRC: nothing touched, nothing run"; return 2; }
    ln -s "$SRC" "$LINK" || { log "can't make the link $LINK: nothing run"; return 2; }
  fi
  trap '[ -L "$LINK" ] && rm -f "$LINK"' EXIT
  [ -s "$local_path/model.safetensors" ] || { log "no weights through $LINK: nothing run"; return 2; }
fi
