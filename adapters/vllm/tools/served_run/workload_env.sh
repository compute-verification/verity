# Sourced from the tree's root with REPO, REV, HF_HOME, VENV, CUDA_HOME and log(): an INT8 pin linked where the engine
# reads it until exit, and the venv's CUDA headers on CPATH where the node's CUDA lacks them.
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
if [ ! -f "$CUDA_HOME/include/cusparse.h" ]; then
  for lib in cusparse cusolver cufft curand; do
    for d in "$VENV"/lib/python*/site-packages/nvidia/"$lib"/include; do [ -d "$d" ] && CPATH=${CPATH:+$CPATH:}$d; done
  done
  export CPATH
fi
