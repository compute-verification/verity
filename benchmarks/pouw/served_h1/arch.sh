# Sourced by served_h1's GPU scripts: the card their CUDA builds (h1u.cu's units kernel, noise.cu's cubin) target. ARCH
# is nvcc's name for it, 120a (the RTX PRO 6000, compute capability 12.0) or 90a (an H100 or H200, 9.0), else the node's
# first GPU's (nvidia-smi, which needs no lease); a build runs before the lease, so it is for the node's card. Sets
# GENCODE (nvcc's -gencode value) and FLASHINFER_ARCH (FlashInfer's FLASHINFER_CUDA_ARCH_LIST entry for that card).
if [ -z "${ARCH:-}" ]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  case "$cap" in
    12.0) ARCH=120a ;;
    9.0) ARCH=90a ;;
    *) echo "served_h1: no build for compute capability '${cap:-none}': set ARCH (120a or 90a)"; exit 2 ;;
  esac
fi
case "$ARCH" in
  120a) FLASHINFER_ARCH=12.0f ;;
  90a) FLASHINFER_ARCH=9.0a ;;
  *) echo "served_h1: ARCH is 120a or 90a, not $ARCH"; exit 2 ;;
esac
GENCODE="arch=compute_$ARCH,code=sm_$ARCH"
echo "served_h1: building for sm_$ARCH"
