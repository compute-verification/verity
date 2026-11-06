# Sourced: CUDA_TK set to a CUDA 13.3 toolkit (upstream Flock's clmad needs its ptxas) where none is installed and none can be,
# without root: the 13-3 packages unpacked under FLOCK_WORK (apt-get download needs no root); no compat library: an R580 driver
# runs 13.3 by minor-version compatibility. A CUDA_TK already set, or an installed /usr/local/cuda-13.3, is left alone.
for c in ${FLOCK_WORK:-/workspace}/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
if [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ]; then
  CR=${FLOCK_WORK:-/workspace}/cuda-root; export CUDA_TK=$CR/usr/local/cuda-13.3
  if [ ! -x $CUDA_TK/bin/nvcc ]; then
    # the 13-3 packages' dependency closure, from apt-cache show (a pinned-out package has no candidate for apt-cache depends)
    todo="cuda-nvcc-13-3 cuda-cudart-dev-13-3"; seen=
    while [ -n "$todo" ]; do
      set -- $todo; q=$1; shift; todo="$*"
      case " $seen " in *" $q "*) continue ;; esac
      seen="$seen $q"
      todo="$todo $(apt-cache show $q 2>/dev/null | awk '/^Depends:/ {sub(/^Depends: /, ""); print; exit}' | tr ',' '\n' \
        | sed 's/(.*//; s/|.*//; s/ //g' | grep -E -- '-13-3$' | tr '\n' ' ')"
    done
    mkdir -p $CR/debs && (cd $CR/debs && apt-get download $(for q in $seen; do echo "$q=$(apt-cache show $q | awk '/^Version:/ {print $2; exit}')"; done)) \
      && for d in $CR/debs/*.deb; do dpkg-deb -x $d $CR; done
    [ -x $CUDA_TK/bin/nvcc ] || { echo "CUDA 13.3 unpack failed"; exit 1; }
  fi
fi
