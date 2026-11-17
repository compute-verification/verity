#!/bin/bash
# The job vcf-job-run.sh submits to VCF's Slurm cluster with SEND=a payload from payload.sh: it records what the node has
# (result/site.txt), builds the tree's environment with uv on the node's disk, and runs the payload's command under
# `research run --local-only`, leaving the run directory under result/runs/ for vcf-job-run.sh to copy back and publish. It holds
# no credential; it writes only the job's directory, which vcf-job-run.sh removes, and its scratch on the node's disk.
set -u -o pipefail
J=${SLURM_SUBMIT_DIR:?submit me with sbatch}
# the tree, uv, its Pythons, caches and the venv (GBs with CUDA torch) on the node's own disk: the home is a network
# filesystem, where removing them takes minutes
W=$(mktemp -d "${TMPDIR:-/tmp}/verity-XXXXXX") || exit 3
trap 'rm -rf "$W"' EXIT TERM
mkdir -p "$W/home" "$J/result/runs"
export HOME=$W/home XDG_CACHE_HOME=$W/cache XDG_DATA_HOME=$W/data XDG_CONFIG_HOME=$W/config UV_CACHE_DIR=$W/uv-cache \
  UV_PYTHON_INSTALL_DIR=$W/python UV_TOOL_DIR=$W/tools RESEARCH_STORE=$W/store GIT_CEILING_DIRECTORIES=$J
NODE=${SLURMD_NODENAME:-$(hostname -s)}
{
  echo "node $NODE, job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_PARTITION:-?}, cpus ${SLURM_CPUS_PER_TASK:-?}, gpus ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-none}}, user uid $(id -u)"
  echo "kernel $(uname -r); $(nproc) CPUs visible; $(free -g | awk '/Mem:/{print $2}') GB memory on the node"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -n 8
  for c in python3 git nvcc ncu module enroot apptainer singularity docker podman sudo; do
    printf '%s: %s\n' "$c" "$(type -p "$c" 2>/dev/null || type -t "$c" 2>/dev/null || echo absent)"
  done
  echo "python3 $(python3 -c 'import platform; print(platform.python_version())' 2>/dev/null || echo absent)"
  echo "cuda dirs: $(ls -d /usr/local/cuda* 2>/dev/null | tr '\n' ' ')"
  echo "sudo -n: $(sudo -n true 2>&1 && echo yes || echo no)"
  printf 'pypi: '; curl -s -o /dev/null -m 20 -w '%{http_code}\n' https://pypi.org/simple/ || echo unreachable
  df -h --output=avail,target "$J" "$W" | tail -n 2
} > "$J/result/site.txt" 2>&1
tar -C "$W" -xzf "$J/payload.tar.gz" || { echo "no payload: send one with SEND=" >&2; exit 3; }
# shellcheck source=/dev/null
. "$W/request.sh"
command -v uv > /dev/null || { curl -LsSf -m 60 https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$W/bin" sh > /dev/null; export PATH="$W/bin:$PATH"; }
cd "$W/tree" || exit 3
t0=$(date +%s)
# shellcheck disable=SC2086
uv sync -q $VCF_SYNC || { echo "uv sync $VCF_SYNC failed" >&2; exit 4; }
echo "uv $(uv --version | awk '{print $2}'); uv sync $VCF_SYNC in $(( $(date +%s) - t0 )) s" >> "$J/result/site.txt"
export PATH="$W/tree/.venv/bin:$PATH"
# what the record names as the machine: a node of VCF's cluster, shared with its other users, with no lease or slot of
# ours; Slurm's --mem is its memory limit
REMOTE=$(.venv/bin/python -c 'import json, os, sys; e = os.environ.get
print(json.dumps({"machine": "vcf-" + sys.argv[1], "site": "vcf-slurm", "host": "vcf.lucidcomputing.ai", "slurm_job": e("SLURM_JOB_ID"),
                  "partition": e("SLURM_JOB_PARTITION"), "cpus": e("SLURM_CPUS_PER_TASK"), "gpus": e("SLURM_JOB_GPUS"),
                  "mem_mb": e("SLURM_MEM_PER_NODE"), "lease": None, "slot": None, "node_shared": True}))' "$NODE")
.venv/bin/research run --runs-dir "$J/result/runs" --local-only "VCF's cluster holds no store key: vcf-job-run.sh publishes the run dir with its custody" \
  --source-identity "{\"commit\": \"$VCF_COMMIT\", \"tree\": \"$VCF_TREE\", \"dirty\": false, \"dirty_digest\": null}" \
  --remote-json "$REMOTE" --cost-json '{"provider": "vcf", "rate_usd_per_hour": null, "shared": true, "reason": "VCF cluster, no rate"}' \
  ${VCF_TOOL:+--tool "$VCF_TOOL"} --env "RESEARCH_SOURCE_COMMIT=$VCF_COMMIT" \
  --tree-sha "$VCF_COMMIT" --cwd "$W/tree" --timeout "$(( ${VCF_TIMEOUT_MIN:-20} * 60 ))" -- "${VCF_CMD[@]}"
