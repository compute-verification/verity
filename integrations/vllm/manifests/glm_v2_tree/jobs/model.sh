# Sourced by the row jobs after ROW is set: the row's model from its name.  ROLE, REPO, REV (the pin, checkpoints.json), CONFIG
# (relative to integrations/vllm), SNAP (the metadata the CPU jobs read, under /workspace/cp/circuits/hf), FS_CK (the pin's
# local_path, which the GPU legs serve), LOCAL_CK (a node's verified local copy), WHAT (for keep reasons), V2 (the name of the
# model's v2 dirs under /workspace/cp/circuits), POST_LOAD (the Parameters its loader's post-load hook makes, which every rank's
# Build records and the capture registers: DeepSeek-V4's mHC broadcast), DERIVE_GB (a request derive's measured peak, where
# `row_records.derive_bytes` plans too little), CENSUS (VERITY_BUILD_CENSUS) and the defaults of REF and SW (its v2 tree and
# row dirs).  `ckpt_bind` serves
# LOCAL_CK at FS_CK when it is verified; `stage_meta PY` puts the pin's non-shard files at SNAP from the hub, each checked against
# checkpoints.json's sha256.
case ${ROW:?ROW} in
  glm53__*) ROLE=GLM53 REPO=zai-org/GLM-5.3 REV=aca966e4e02791568aa6a4ced368624b3d897f42 WHAT=GLM-5.3
            FS_CK=/workspace/fs/glm53/$REV LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
            V2=glm-v2 REF=${REF:-cursor/glm-v2-21f3} POST_LOAD= DERIVE_GB= CENSUS= ;;
  dsv4-pro__*) ROLE=DSV4_PRO REPO=deepseek-ai/DeepSeek-V4-Pro REV=b5968e9190ef611bbf34a7229255be88a0e937c1 WHAT=DeepSeek-V4-Pro
            FS_CK=/workspace/fs/dsv4/$REV LOCAL_CK=/workspace/cp/dsv4-ckpt
            V2=dsv4-v2 REF=${REF:-cursor/dsv4-v2-21f3} POST_LOAD=model.layers.0.hc_attn_fn_broadcast DERIVE_GB=32 CENSUS=skip ;;
  *) echo "row $ROW: no model for it in model.sh"; exit 2 ;;
esac
CONFIG=verity_vllm/engine/profiles/hf_configs/$ROLE.config.json SW=${SW:-/workspace/cp/circuits/$V2/rows}
SNAP=/workspace/cp/circuits/hf/hub/models--${REPO/\//--}/snapshots/$REV
# DeepSeek-V4-Pro's B1 request derive peaks at 21.8 GB, but the census the Build writes beside it (`memory_map.census_entry`)
# ran past 50 min at 161 GB, against `derive_bytes`' 13.5 GB: `--build-jobs auto` ran 8 at once and the cgroup's OOM killer
# took five (r20261020-071632-1bcf).  The census decides no Build, so CENSUS=skip leaves it out; without it the trace's
# largest process was 27.6 GB (r20261020-085740-90ff).  With DERIVE_GB, BUILD_JOBS (the env `--build-jobs` reads) is the
# run's --mem-gb over it.
[ -z "$CENSUS" ] || export VERITY_BUILD_CENSUS=$CENSUS
if [ -n "$DERIVE_GB" ] && [ -z "${BUILD_JOBS:-}" ]; then
  _mem=${RESEARCH_RUN_MEM_GB:-0}
  BUILD_JOBS=$(( ${_mem%.*} / DERIVE_GB )) && [ "$BUILD_JOBS" -ge 1 ] || BUILD_JOBS=1
  export BUILD_JOBS
fi

ckpt_bind() {
  if [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK" && [ ! -L "$FS_CK" ]; then
    if [ -d "$FS_CK" ]; then sudo -n mount --bind "$LOCAL_CK" "$FS_CK"; else mkdir -p "$(dirname "$FS_CK")" && ln -s "$LOCAL_CK" "$FS_CK"; fi
  fi
  if mountpoint -q "$FS_CK"; then echo "checkpoint: local verified copy bound over $FS_CK"
  elif [ -L "$FS_CK" ]; then echo "checkpoint: $FS_CK links the local verified copy $(readlink "$FS_CK")"
  else echo "checkpoint: $FS_CK read over the shared fs"; fi
}

stage_meta() {  # stage_meta PY: the pin's non-shard files at SNAP (none of the shards), refused unless each is checkpoints.json's
  HF_HUB_OFFLINE= "$1" - "$REPO" "$REV" "$SNAP" "$TREE/integrations/vllm/manifests/checkpoints.json" <<'EOF'
import hashlib, json, os, sys, tempfile
repo, rev, snap, pins = sys.argv[1:5]
pin = [c for c in json.load(open(pins))["checkpoints"] if c["repo"] == repo and c["revision"] == rev][0]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
want = {n: f for n, f in pin["files"].items() if not n.endswith(".safetensors")}
todo = [n for n, f in want.items() if not (os.path.isfile(os.path.join(snap, n)) and sha(os.path.join(snap, n)) == f["sha256"])]
if not todo:
    sys.exit(print(f"metadata: {len(want)} files at {snap}, each checkpoints.json's"))
from huggingface_hub import hf_hub_download
os.makedirs(snap, exist_ok=True)
with tempfile.TemporaryDirectory(dir=os.path.dirname(snap)) as t:
    for name in todo:
        h = sha(hf_hub_download(repo, name, revision=rev, local_dir=t))
        if h != want[name]["sha256"]:
            sys.exit(f"{repo}@{rev} {name}: sha256 {h}, checkpoints.json {want[name]['sha256']}")
        os.replace(os.path.join(t, name), os.path.join(snap, name))
        print(f"staged {name} ({want[name]['size']} B, sha256 {h[:16]} = checkpoints.json)")
EOF
}
