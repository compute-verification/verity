# Sourced: `cuda_headers PYTHON` puts on CPATH the include directory of each CUDA library header torch's
# ATen/cuda headers read (CUDAContextLight.h) that CUDA_HOME's toolkit lacks, from PYTHON's NVIDIA wheels, which
# torch's own CUDA version pins. A toolkit of nvcc and the runtime alone (no cuSPARSE, cuSOLVER or cuFFT headers)
# then still builds torch's JIT extensions: the norm and router taps and the Commit's collector.
cuda_headers() {
  local sp h d
  sp=$("$1" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null) || return 0
  for h in cusparse.h cusolverDn.h cufft.h curand.h cublas_v2.h cublasLt.h; do
    [ -e "${CUDA_HOME:?}/include/$h" ] && continue
    for d in "$sp"/nvidia/*/include; do
      if [ -e "$d/$h" ]; then
        case ":${CPATH:-}:" in *":$d:"*) ;; *) CPATH=$d${CPATH:+:$CPATH} ;; esac
        break
      fi
    done
  done
  [ -n "${CPATH:-}" ] && export CPATH
  return 0
}
