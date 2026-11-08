#!/usr/bin/env bash
# Sets vy-glm53-h8 up to replay node A's (vy-spot-glm53's) GLM-5.3 serves, as node A runs them:
#   CUDA 12.9.1's toolkit in $CU (nvcc V12.9.86, which FlashInfer's and DeepGEMM's JIT build with on node A; this node's
#   /usr/local/cuda is 13.0), installed from NVIDIA's runfile without root, and an overlay $OV of node A's venv versions of
#   every distribution this node's venv has at another version (nodeA-packages.txt, --no-deps), put first on PYTHONPATH.
# Idempotent: what is already in place is checked, not redone. Under research run --cwd clone of the branch holding spot-send/.
set -uo pipefail
CU=/workspace/cp/cuda-12.9; OV=/workspace/cp/overlay-nodeA; V=/workspace/cp/venv312
HERE=$(cd "$(dirname "$0")" && pwd)
BY=https://cursor.com/agents/bc-5609d8d8-985e-50b9-9216-15dfd5276546
mkdir -p /workspace/tmp; T=$(mktemp -d -p /workspace/tmp); trap 'rm -rf "$T"' EXIT
echo "setup: $(date -u +%T) $(df -BG --output=avail /workspace | tail -1) free"
if "$CU/bin/nvcc" --version 2>/dev/null | grep -q V12.9.86; then
  echo "setup: toolkit already at $CU"
else
  URL=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
  curl -fsSL --retry 3 -o "$T/cuda.run" "$URL" || { echo "setup: download of $URL failed"; exit 3; }
  echo "setup: $(date -u +%T) downloaded $(stat -c %s "$T/cuda.run") bytes, sha256 $(sha256sum "$T/cuda.run" | cut -c1-16)"
  sh "$T/cuda.run" --silent --toolkit --toolkitpath="$CU" --no-man-page --override --tmpdir="$T" \
    || { echo "setup: the runfile's toolkit install failed"; tail -40 /tmp/cuda-installer.log 2>/dev/null; exit 4; }
  rm -f "$T/cuda.run"
fi
"$CU/bin/nvcc" --version | tail -2; echo "setup: toolkit $(du -sh "$CU" | cut -f1)"
"$CU/bin/nvcc" --version | grep -q V12.9.86 || { echo "setup: $CU/bin/nvcc is not V12.9.86"; exit 4; }
UV=$(command -v uv || true)
if [ -z "$UV" ]; then
  curl -fsSL --retry 3 https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz -C "$T" \
    && UV=$(ls "$T"/uv-*/uv) || { echo "setup: no uv"; exit 5; }
fi
"$V/bin/python" - "$HERE/nodeA-packages.txt" > "$T/differ.txt" <<'EOF'
import importlib.metadata as md, sys
for line in open(sys.argv[1]):
    name, ver = line.strip().split("==")
    if name.startswith("nvidia-cuda-nvdisasm"):
        continue  # a disassembler's namespace package, read by no kernel
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        have = None
    if have != ver:
        print(f"{name}=={ver}")
EOF
echo "setup: $(wc -l < "$T/differ.txt") distributions differ from node A's"
mkdir -p "$OV"
while read -r req; do
  "$UV" pip install -q --python "$V/bin/python" --target "$OV" --no-deps "$req" >"$T/uv.log" 2>&1 \
    || { echo "setup: overlay could not install $req: $(tail -1 "$T/uv.log")"; }
done < "$T/differ.txt"
PYTHONPATH=$OV "$V/bin/python" - "$HERE/nodeA-packages.txt" <<'EOF'
import importlib.metadata as md, sys
bad = []
for line in open(sys.argv[1]):
    name, ver = line.strip().split("==")
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        have = None
    if have != ver:
        bad.append(f"{name} {have} (node A {ver})")
print(f"setup: with the overlay, {200 - len(bad)} of 200 distributions at node A's version; differing: {bad}")
EOF
for p in "$CU" "$OV"; do
  [ -e "$p" ] && research keep "$p" --owner @proofs --expires 2026-11-15 --priority low --regenerate "spot-send/h8_setup.sh on cursor/proofs-overheaddef-h8a1" \
    --why "node A's CUDA toolkit and package versions for OVERHEADDEF's cross-machine GLM-5.3 replays on this node" --by "$BY" >/dev/null || true
done
echo "setup: done $(date -u +%T) $(df -BG --output=avail /workspace | tail -1) free"
