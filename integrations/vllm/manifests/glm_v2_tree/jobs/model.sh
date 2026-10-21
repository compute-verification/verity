# Sourced by the row jobs after ROW is set: the row's model from its name.  ROLE, REPO, REV (the pin, checkpoints.json), CONFIG
# (relative to integrations/vllm), SNAP (the metadata the CPU jobs read, under /workspace/cp/circuits/hf), FS_CK (the pin's
# local_path, which the GPU legs serve), LOCAL_CK (a node's verified local copy), WHAT (for keep reasons), V2 (the name of the
# model's v2 dirs under /workspace/cp/circuits), POST_LOAD (the Parameters its loader's post-load hook makes, which every rank's
# Build records and the capture registers: DeepSeek-V4's mHC broadcast), DERIVE_GB (a request derive's measured peak, where
# `row_records.derive_bytes` plans too little), CENSUS (VERITY_BUILD_CENSUS) and the defaults of REF and SW (its v2 tree and
# row dirs).  `ckpt_bind` serves
# LOCAL_CK at FS_CK when it is verified; `row_on ROOT` keeps the row dir on another disk at its path; `stage_meta PY` puts the pin's non-shard files at SNAP from the hub, each checked against
# checkpoints.json's sha256.
case ${ROW:?ROW} in
  glm53__*) ROLE=GLM53 REPO=zai-org/GLM-5.3 REV=aca966e4e02791568aa6a4ced368624b3d897f42 WHAT=GLM-5.3
            FS_CK=/workspace/fs/glm53/$REV LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
            V2=glm-v2 REF=${REF:-cursor/glm-v2-21f3} POST_LOAD= DERIVE_GB= CENSUS=
            # its checkpoints.json pin names config.json and tokenizer.json only; these are the other non-shard files, as
            # vy-glm53-cpu-2's snapshot holds them for the B1 and B8 Builds of record
            META_EXTRA="model.safetensors.index.json=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf
                        tokenizer_config.json=98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc
                        generation_config.json=ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55
                        chat_template.jinja=3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c" ;;
  dsv4-pro__*) ROLE=DSV4_PRO REPO=deepseek-ai/DeepSeek-V4-Pro REV=b5968e9190ef611bbf34a7229255be88a0e937c1 WHAT=DeepSeek-V4-Pro
            FS_CK=/workspace/fs/dsv4/$REV LOCAL_CK=/workspace/cp/dsv4-ckpt
            V2=dsv4-v2 REF=${REF:-cursor/dsv4-v2-21f3} POST_LOAD=model.layers.0.hc_attn_fn_broadcast DERIVE_GB=60 CENSUS=skip META_EXTRA= ;;
  *) echo "row $ROW: no model for it in model.sh"; exit 2 ;;
esac
CONFIG=verity_vllm/engine/profiles/hf_configs/$ROLE.config.json SW=${SW:-/workspace/cp/circuits/$V2/rows}
SNAP=/workspace/cp/circuits/hf/hub/models--${REPO/\//--}/snapshots/$REV
# DeepSeek-V4-Pro's B1 request derive peaks at 47 GB, then the census the Build writes beside it (`memory_map.census_entry`)
# runs past 50 min at 161 GB, against `derive_bytes`' 13.5 GB: `--build-jobs auto` ran 8 at once and the cgroup's OOM killer
# took five (r20261020-071632-1bcf).  The census decides no Build, so CENSUS=skip leaves it out.  With DERIVE_GB, BUILD_JOBS
# (the env `--build-jobs` reads) is the run's --mem-gb over it.
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

row_on() {  # row_on ROOT: the row dir kept on another disk, ROOT/circuits/$V2/rows/$ROW, bound at $SW/$ROW, the path every job
  local src=$1/circuits/$V2/rows/$ROW d=$SW/$ROW  # and the Commit's replay read; binds an absent or empty row dir only
  if mountpoint -q "$d"; then echo "row dir: $(findmnt -no SOURCE "$d") bound at $d"; return 0; fi
  [ -z "$(ls -A "$d" 2>/dev/null)" ] || { echo "row dir: $d already holds files, so ROW_ON=$1 cannot bind it"; return 2; }
  mkdir -p "$src" "$d" && sudo -n mount --bind "$src" "$d" && echo "row dir: $src bound at $d"
}

stage_meta() {  # stage_meta PY: the pin's non-shard files at SNAP (none of the shards) and META_EXTRA's, refused unless each is its sha256
  HF_HUB_OFFLINE= "$1" - "$REPO" "$REV" "$SNAP" "$TREE/integrations/vllm/manifests/checkpoints.json" "$META_EXTRA" <<'EOF'
import hashlib, json, os, sys, tempfile
repo, rev, snap, pins, extra = sys.argv[1:6]
pin = [c for c in json.load(open(pins))["checkpoints"] if c["repo"] == repo and c["revision"] == rev][0]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
want = {n: f for n, f in pin["files"].items() if not n.endswith(".safetensors")}
want |= {n: {"sha256": h} for n, _, h in (e.partition("=") for e in extra.split())}
todo = [n for n, f in want.items() if not (os.path.isfile(os.path.join(snap, n)) and sha(os.path.join(snap, n)) == f["sha256"])]
if not todo:
    sys.exit(print(f"metadata: {len(want)} files at {snap}, each checkpoints.json's or META_EXTRA's"))
from huggingface_hub import hf_hub_download
os.makedirs(snap, exist_ok=True)
with tempfile.TemporaryDirectory(dir=os.path.dirname(snap)) as t:
    for name in todo:
        h = sha(hf_hub_download(repo, name, revision=rev, local_dir=t))
        if h != want[name]["sha256"]:
            sys.exit(f"{repo}@{rev} {name}: sha256 {h}, pinned {want[name]['sha256']}")
        os.replace(os.path.join(t, name), os.path.join(snap, name))
        print(f"staged {name} ({os.path.getsize(os.path.join(snap, name))} B, sha256 {h[:16]} = its pin)")
EOF
}
