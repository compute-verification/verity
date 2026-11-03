#!/usr/bin/env bash
# candidate.sh START [TRAIN_ARGS...]: one candidate on one leased GPU (fixed), run from the pinned measurement commit
# (pin.py) as two jobs, one after the other.
#
# The training job is the island's program, `islands/$CW_ISLAND/$CW_PROGRAM` (CW_ISLAND required, CW_PROGRAM default
# train.py; the island is launch.py's $RESEARCH_RUN_DIR/inputs/island.tar, else this commit's), inside the harness (`python -m
# te_core.run --script`, which counts its MACs, times its steps and stops it at 1.2x GPU_SECONDS_CAP), inside sandbox.py:
# it sees this commit's benchmarks/covert_witness with that one island in islands/, the starts, the training stream (the
# sealed start and its stream only in final mode), the Python it runs on and its leased GPU, all read-only, and one
# directory it writes, where the harness puts each checkpoint's E and its record. No held-out set, no other run and no
# home directory is on any path it can open, and it has no network. sandbox.py records its whole process tree.
#
# The evaluation job (evaluate.py, outside the sandbox, after every process of the training job is gone) checks the
# witnesses and the process tree, rebuilds each E's model from θ with the fixed decoder and scores it. $OUT gets run/ (the
# scored run.json, which the ledger ingests, and the harness's guard.log), witness.json, tree.json, timing.json,
# island.json and island.tar, the submission's meta.json and steps.jsonl, E.bin when it is at most 1 MB (else its
# sha256), and the logs. The training job's directory is scratch, removed at exit. CW_LEASE=none, with CW_DEVICE=cpu,
# runs it without a GPU (a smoke check on a CPU pool).
set -u
here=$(cd "$(dirname "$0")" && pwd)
P=${CW_PYTHON:-/workspace/jobs/venv312/bin/python}
R=${CW_ROOT:-/workspace/research/scratch/covert-witness}
LF=${CW_LF_CORPUS:-/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1}
STARTS=${CW_STARTS:-/workspace/research/te/starts}
SEALED=${CW_SEALED:-/workspace/research/te/sealed}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-training:covert-witness}
# the seconds cap, the training job's hard limit (1.2x the cap, plus loading, encoding and the family checks), the
# evaluation job's, and the GPU lease's in minutes
read -r cap kill ekill lease <<< "$("$P" -c "import sys; sys.path.insert(0, '$here'); import prepare; c = prepare.GPU_SECONDS_CAP
print(int(c), int(1.2 * c + 600), 1200, int((1.2 * c + 600 + 1200) / 60 + 5))")"

if [ "${1:-}" = --leased ]; then
  shift
  start=$1 S=$2 island=$3 program=$4
  shift 4
  fraction=0.98 prev=
  for x in "$@"; do
    case "$x" in --budget-fraction=*) fraction=${x#*=} ;; esac
    [ "$prev" = --budget-fraction ] && fraction=$x
    prev=$x
  done
  T=$S/train
  mkdir -p "$T" "$S/root"
  meta="{\"start\": \"$start\", \"launcher\": \"covert_witness/candidate.sh\", \"node\": \"${CW_NODE:-}\", \"island\": \"$island\"}"
  # the interpreter's prefix, base prefix and every sys.path entry, and each link on the way to its binary and to its
  # venv's home (a bind follows it), outside /usr and each other, none holding a held-out directory
  pyro=$("$P" -I -c 'import os, sys
held = [os.path.realpath(p) for p in sys.argv[2:]]
paths = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, *sys.path) if p and os.path.exists(p)}
cfg = os.path.join(sys.prefix, "pyvenv.cfg")
todo = [sys.argv[1], *(v.strip() for k, _, v in (l.partition("=") for l in open(cfg)) if k.strip() == "home")] \
    if os.path.isfile(cfg) else [sys.argv[1]]
for _ in range(200):
    if not todo:
        break
    x = todo.pop()
    parts = x.split("/")
    for i in range(2, len(parts) + 1):
        q = "/".join(parts[:i])
        if os.path.islink(q):
            paths.add(q)
            todo.append(os.path.normpath(os.path.join(os.path.dirname(q), os.readlink(q))))
out = []
for p in sorted(paths, key=len):
    if p == "/usr" or p.startswith("/usr/") or any(p == q or p.startswith(q + "/") for q in out):
        continue
    bad = [h for h in held if h == p or h.startswith(p + "/")]
    if bad:
        sys.exit(f"candidate: {p}, on the interpreter path, holds {bad[0]}")
    out.append(p)
print(" ".join(out))' "$P" "$R" "$SEALED" "$LF") || { echo "$pyro"; exit 1; }
  ro=(--ro /usr --ro /etc)
  for p in $pyro; do ro+=(--ro "$p"); done
  ro+=(--ro "$here" --tmpfs "$here/islands" --ro "$S/island:$here/islands/$island" --ro "$STARTS" --ro "$LF/train.bin")
  # the training streams, never data v3's ladder validation stream (train/v3/val.bin, held-out text)
  for p in "$R/train/pythia.bin" "$R/train/v3/parts" "$R/train/v3/tokenizer.json" "$R/train/v3/manifest.json"; do
    [ -e "$p" ] && ro+=(--ro "$p")
  done
  [ "${CW_FINAL:-}" = 1 ] && [ -d "$SEALED/starts" ] && ro+=(--ro "$SEALED/starts")
  env=(--env PATH=/usr/local/bin:/usr/bin:/bin --env HOME=/tmp/home --env LANG=C.UTF-8 --env PYTHONDONTWRITEBYTECODE=1
       --env HF_HUB_OFFLINE=1 --env CUDA_CACHE_PATH=/tmp/cuda --env "CW_ROOT=$R" --env "CW_LF_CORPUS=$LF"
       --env "CW_STARTS=$STARTS" --env "CW_SEALED=$SEALED" --env "RESEARCH_RUN_ID=${RESEARCH_RUN_ID:-}"
       --env "RESEARCH_COMMIT=${RESEARCH_SOURCE_SHA:-}")
  for k in CW_DEVICE CW_FINAL CW_NODE CUDA_DEVICE_ORDER OMP_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF; do
    [ -n "${!k:-}" ] && env+=(--env "$k=${!k}")
  done
  dev=()
  if [ -n "${GPU_LEASE_UUID:-}" ]; then
    # the driver's control nodes and the leased GPUs' own, by the minor number the driver gives each UUID
    for n in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia-caps/*; do
      [ -e "$n" ] && dev+=(--dev "$n")
    done
    for u in ${GPU_LEASE_UUID//,/ }; do
      minor=$(grep -l -i "$u" /proc/driver/nvidia/gpus/*/information 2>/dev/null | head -1 | xargs -r awk '/Device Minor/ {print $3}')
      [ -n "$minor" ] || { echo "candidate: no device minor for the leased GPU $u"; exit 1; }
      dev+=(--dev "/dev/nvidia$minor")
    done
    env+=(--env "CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID" --env "GPU_LEASE_UUID=$GPU_LEASE_UUID")
  else
    env+=(--env CUDA_VISIBLE_DEVICES=)
  fi
  t0=$(date +%s.%N)
  timeout --signal=TERM --kill-after=30 "$kill" sudo -n /usr/bin/python3 -I "$here/sandbox.py" --root "$S/root" \
    --uid "$(id -u)" --gid "$(id -g)" --tree "$S/tree.json" --cwd "$here" "${ro[@]}" --rw "$T" "${dev[@]}" "${env[@]}" \
    -- "$P" -m te_core.run --script "islands/$island/$program" --out "$T/run" --families "$T/families" \
    --budget-fraction "$fraction" --budget-minutes "$("$P" -c "print($cap / 60)")" --meta "$meta" \
    -- --start "$start" --out "$T/sub" "$@" > "$OUT/train.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  leased=${GPU_LEASE_UUID:-}
  gpu=$([ -n "$leased" ] && nvidia-smi --query-gpu=name --format=csv,noheader -i "${leased%%,*}" 2>/dev/null | head -1)
  echo "{\"process_seconds\": $("$P" -c "print(round($t1 - $t0, 3))"), \"rc\": $rc, \"gpu\": \"$gpu\"}" > "$S/timing.json"
  cp "$S/timing.json" "$OUT/timing.json"
  [ -f "$S/tree.json" ] && cp "$S/tree.json" "$OUT/tree.json"
  mkdir -p "$OUT/run"
  [ -f "$T/run/guard.log" ] && cp "$T/run/guard.log" "$OUT/run/guard.log"
  [ -f "$T/sub/meta.json" ] || { echo "candidate: the program wrote no submission (rc $rc)"; tail -3 "$OUT/train.log"; exit 1; }
  [ -f "$T/run/run.json" ] || { echo "candidate: the harness wrote no run.json (rc $rc)"; tail -3 "$OUT/train.log"; exit 1; }
  timeout --signal=TERM --kill-after=30 "$ekill" "$P" "$here/evaluate.py" --submission "$T/sub" \
    --timing "$S/timing.json" --run "$T/run/run.json" --tree "$S/tree.json" --island "$S/island.json" \
    --scored "$OUT/run" --out "$OUT/witness.json" > "$OUT/evaluate.log" 2>&1
  erc=$?
  tail -1 "$OUT/evaluate.log"
  exit $erc
fi

start=${1:?candidate: candidate.sh START [TRAIN_ARGS...]}
if [ ! -d "$STARTS/$start" ] && [ -n "${2:-}" ] && [ -d "$STARTS/$2" ]; then
  echo "candidate: harness v3 takes START with no class (given $1 $2): merge origin/cursor/covert-witness-4207 and launch with launch.py START --island NAME"
  exit 2
fi
shift
island=${CW_ISLAND:?candidate: name the island (CW_ISLAND; launch.py sets it)} program=${CW_PROGRAM:-train.py}
case "$island/$program" in
  /*|*..*|*//*) echo "candidate: island $island and program $program must be names inside islands/"; exit 2 ;;
esac
mkdir -p "$R/candidates" "$OUT"
S=$(mktemp -d -p "$R/candidates" "$island.XXXXXX")
trap 'rm -rf "$S"' EXIT
mkdir "$S/island"
tar=${RESEARCH_RUN_DIR:-/nonexistent}/inputs/island.tar
if [ -f "$tar" ]; then
  tar -xf "$tar" -C "$S/island" --no-same-owner --no-same-permissions || { echo "candidate: island.tar does not extract"; exit 2; }
  cp "$tar" "$OUT/island.tar"
  from=sent commit=${CW_ISLAND_COMMIT:-}
else
  [ -d "$here/islands/$island" ] || { echo "candidate: no island $island in this commit and none sent"; exit 2; }
  cp -a "$here/islands/$island/." "$S/island/"
  from=measurement-commit commit=${RESEARCH_SOURCE_SHA:-}
fi
[ -f "$S/island/$program" ] && [ ! -L "$S/island/$program" ] || { echo "candidate: island $island has no file $program"; exit 2; }
"$P" -I -c 'import hashlib, json, os, sys
d, name, program, frm, commit, tree, tar = sys.argv[1:]
files = {}
for top, dirs, names in os.walk(d):
    for n in names:
        p = os.path.join(top, n)
        rel = os.path.relpath(p, d)
        files[rel] = ("link:" + os.readlink(p)) if os.path.islink(p) else hashlib.sha256(open(p, "rb").read()).hexdigest()
digest = hashlib.sha256("".join(f"{v}  {k}\n" for k, v in sorted(files.items())).encode()).hexdigest()
rec = {"name": name, "program": program, "from": frm, "commit": commit, "tree": tree or None, "digest": digest,
       "tar_sha256": hashlib.sha256(open(tar, "rb").read()).hexdigest() if os.path.isfile(tar) else None,
       "program_sha256": files.get(program), "files": files}
print(json.dumps(rec, indent=1))' "$S/island" "$island" "$program" "$from" "$commit" "${CW_ISLAND_TREE:-}" "$tar" \
  > "$S/island.json" || { echo "candidate: could not record the island"; exit 2; }
cp "$S/island.json" "$OUT/island.json"
cp "$S/island/$program" "$OUT/train.py"
echo "{\"start\": \"$start\", \"island\": \"$island\", \"program\": \"$program\", \"args\": \"$*\", \"measurement_commit\": \"${RESEARCH_SOURCE_SHA:-}\"}" > "$OUT/candidate.json"
if [ "${CW_LEASE:-}" = none ]; then
  [ "${CW_DEVICE:-}" = cpu ] || { echo "candidate: CW_LEASE=none runs only with CW_DEVICE=cpu"; exit 2; }
  bash "$0" --leased "$start" "$S" "$island" "$program" "$@"
else
  gpu-lease 1 --wait --max-min "$lease" -- bash "$0" --leased "$start" "$S" "$island" "$program" "$@"
fi
rc=$?
for f in meta.json steps.jsonl; do [ -f "$S/train/sub/$f" ] && cp "$S/train/sub/$f" "$OUT/"; done
if [ -f "$S/train/sub/E.bin" ]; then
  if [ "$(stat -c %s "$S/train/sub/E.bin")" -le 1000000 ]; then cp "$S/train/sub/E.bin" "$OUT/E.bin"; fi
  sha256sum "$S/train/sub/E.bin" | cut -d' ' -f1 > "$OUT/E.sha256"
fi
exit $rc
