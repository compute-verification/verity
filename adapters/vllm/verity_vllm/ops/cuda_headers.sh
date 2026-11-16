# Sourced: `cuda_headers PYTHON` puts on CPATH the include directory of each CUDA library header torch's
# ATen/cuda headers read (CUDAContextLight.h: cuSPARSE's, cuBLAS's and cuSOLVER's) that CUDA_HOME's toolkit lacks, from
# PYTHON's NVIDIA wheels, which torch's own CUDA version pins. A toolkit of nvcc and the runtime alone then still builds
# torch's JIT extensions: the norm and router taps and the Commit's collector. Only headers: no tap links those libraries
# (torch loads them). It prints one line naming the wheels' directories it added; a header in neither place gets one line
# on stderr naming it and the two places looked, and returns 1.
cuda_headers() {
  local sp h d took=""
  sp=$("$1" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null) && [ -n "$sp" ] || return 0
  for h in cusparse.h cusolverDn.h cublas_v2.h cublasLt.h; do
    [ -e "${CUDA_HOME:?}/include/$h" ] && continue
    for d in "$sp"/nvidia/*/include ""; do
      if [ -z "$d" ]; then
        echo "cuda_headers: $h is in neither $CUDA_HOME/include nor the NVIDIA wheels at $sp/nvidia/*/include" >&2
        return 1
      fi
      if [ -e "$d/$h" ]; then
        case ":${CPATH:-}:" in *":$d:"*) ;; *) CPATH=$d${CPATH:+:$CPATH} took="$took $d" ;; esac
        break
      fi
    done
  done
  [ -n "${CPATH:-}" ] && export CPATH
  [ -z "$took" ] || echo "cuda_headers: $CUDA_HOME/include lacks CUDA library headers torch reads; CPATH adds the NVIDIA wheels':$took"
  return 0
}
