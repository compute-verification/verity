#!/usr/bin/env bash
# What a ship build asks of its compiler before it compiles (ship.sh's ship_cuda runs this once a build, in its frame):
# that ship_nvcc.sh, with this toolkit, still gives one cubin for a source wherever it sits. A probe is compiled at two
# places under TMPDIR: beside its header in one, and in the other at another depth under names of other lengths, its
# header in a directory of its own with the #include respelled as a move respells it. Its kernel has internal linkage,
# which nvcc names by a hash of the source's path, so:
#   - nvcc itself must give two cubins. Were they one, the probe would follow no path and the second comparison would
#     show nothing;
#   - ship_nvcc.sh must give one. Were they two, a ship's cubins would follow where their sources sit.
# Either failing refuses the build, naming which. Needs CUDA (the toolkit) and TMPDIR, as ship_nvcc.sh does.
set -euo pipefail
: "${CUDA:?the toolkit, which ship_cuda sets}" "${TMPDIR:?which the frame of ship.sh sets}"
SHIP_NVCC=$(dirname "${BASH_SOURCE[0]}")/ship_nvcc.sh
p=$(mktemp -d "$TMPDIR/ship_probe.XXXXXXXX"); trap 'rm -rf "$p"' EXIT
places=("$p/as_is" "$p/moved/to_another_depth/under_other_names")
includes=("probe.cuh" "headers/of_its_own/probe.cuh")
for i in 0 1; do
  mkdir -p "$(dirname "${places[i]}/${includes[i]}")"
  cat > "${places[i]}/probe.cu" << PROBE
#include "${includes[i]}"
namespace {
__global__ void probe(const unsigned *in, unsigned *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = mix(in[i]);
}
}  // namespace
void launch(const unsigned *in, unsigned *out, int n) { probe<<<(n + 31) / 32, 32>>>(in, out, n); }
PROBE
  echo "__device__ inline unsigned mix(unsigned x) { return x * 2654435761u + 1u; }" > "${places[i]}/${includes[i]}"
done
# a ship's cubin flags (ship_cubin_flags), less ptxas's report
flags=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -cubin)
for cc in "$CUDA/bin/nvcc" "$SHIP_NVCC"; do
  for i in 0 1; do
    "$cc" "${flags[@]}" -o "$p/$(basename "$cc").$i.cubin" "${places[i]}/probe.cu" || {
      echo "ship_probe.sh: $cc doesn't build the probe (above), so nothing says whether a ship's cubins would follow" \
        "where their sources sit: no ship is built with it" >&2
      exit 1; }
  done
done
if cmp -s "$p/nvcc.0.cubin" "$p/nvcc.1.cubin"; then
  echo "ship_probe.sh: the first comparison failed: $CUDA/bin/nvcc itself gives the probe one cubin at two places, so" \
    "the probe follows no path and shows nothing of ship_nvcc.sh. A new toolkit names a kernel of internal linkage" \
    "otherwise: give the probe (in ship_probe.sh) something this nvcc makes follow its source's path" >&2
  exit 1
fi
if ! cmp -s "$p/ship_nvcc.sh.0.cubin" "$p/ship_nvcc.sh.1.cubin"; then
  echo "ship_probe.sh: the second comparison failed: ship_nvcc.sh gives the probe two cubins at two places, so a ship's" \
    "cubins would follow where their sources sit. Either this toolkit ($CUDA) hands its device compiler something of a" \
    "file's place that ship_nvcc.sh doesn't fix, or ship_nvcc.sh changed: look at \`nvcc -dryrun\`'s cicc step and at" \
    "the line markers of the text it compiles, against the three things ship_nvcc.sh's header lists" >&2
  exit 1
fi
