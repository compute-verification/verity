# Sourced by a run.sh that serves vLLM on a held card: exports FLASHINFER_CUDA_ARCH_LIST, FlashInfer's build for the
# compute capability nvidia-smi gives the card (CUDA_VISIBLE_DEVICES's first, else the node's first), unless it is set;
# exits 2 naming the capability where it has no build here.
if [ -z "${FLASHINFER_CUDA_ARCH_LIST:-}" ]; then
  _cc=$(nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "${CUDA_VISIBLE_DEVICES%%,*}"} --query-gpu=compute_cap --format=csv,noheader \
    2>/dev/null | head -1 | tr -d ' ') || true
  case $_cc in
    12.0) FLASHINFER_CUDA_ARCH_LIST=12.0f ;;
    10.0) FLASHINFER_CUDA_ARCH_LIST=10.0a ;;
    9.0) FLASHINFER_CUDA_ARCH_LIST=9.0a ;;
    *) echo "flashinfer_arch.sh: no FlashInfer build for compute capability '${_cc}' (benchmarks/harness/flashinfer_arch.sh):" \
         "set FLASHINFER_CUDA_ARCH_LIST to the card's" >&2
       exit 2 ;;
  esac
  unset _cc
fi
export FLASHINFER_CUDA_ARCH_LIST
