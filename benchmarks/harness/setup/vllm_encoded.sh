# Sourced by run_point.sh (a spec's `setup`): the bandwidth arms' GPU libraries (benchmarks/vllm_encoded/bench_arm.py),
# built from this source tree as benchmarks/kv_filler/run.sh builds the encoded stack (STACK=...) at its default field
# (1024, masked, FOLD6, 32 query heads) and the top-up, and the VERITY_VLLM_BENCH_*_LIB names the arms read exported.
#
# wt.cu takes nvcc about 9 minutes, so a build is kept on the node in $VERITY_BENCH_LIB_CACHE/KEY (default
# /workspace/research/cache/verity-bench-libs), KEY the hash of nvcc's version, the flags and every file each build reads
# (nvcc -MM): a point whose sources match an earlier one's reuses its libraries, and one whose sources differ (another
# commit's ctypes declarations) builds its own. One point builds a KEY while the others wait on its lock; each copies the
# libraries into $RESEARCH_RUN_DIR/setup, so a retention sweep of the cache never removes a library a running point
# loads. A cache this job can't write is skipped: the point builds in its run directory. nvcc's version, the flags,
# ptxas's reports, every source's and library's sha256 and whether the point built or reused go to
# $RESEARCH_RUN_DIR/out/setup.
VE_RUN=${RESEARCH_RUN_DIR:-$PWD}
VE_CACHE=${VERITY_BENCH_LIB_CACHE:-/workspace/research/cache/verity-bench-libs}
if ! { mkdir -p "$VE_CACHE" 2> /dev/null && [ -w "$VE_CACHE" ]; }; then
  echo "setup: $VE_CACHE is not writable here; building in the run directory" >&2
  VE_CACHE=$VE_RUN/setup-build
fi
VE_FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
VE_FIELD=1024
VE_ORDER=(wt reader paged topup)
declare -A VE_BUILDS=(
  [wt]="-DWT_FIELD=$VE_FIELD -DWT_MASK=1 -DWT_FOLD6=1 -DWT_RECEIPT=1 benchmarks/codec_tensor/wt.cu"
  [reader]="benchmarks/codec_tensor/reader.cu"
  [paged]="-DKV_FIELD=$VE_FIELD -DKV_MASK=1 -DKV_FOLD6=1 -DKV_QHEADS=32 benchmarks/vllm_encoded/paged.cu"
  [topup]="benchmarks/kv_filler/topup.cu"
)
mkdir -p "$VE_CACHE" "$VE_RUN/setup" "$VE_RUN/out/setup"
VE_RECIPE=$(nvcc --version; echo "flags: ${VE_FLAGS[*]}"; for lib in "${VE_ORDER[@]}"; do echo "$lib: ${VE_BUILDS[$lib]}"; done)
VE_SOURCES=$(for lib in "${VE_ORDER[@]}"; do
  # shellcheck disable=SC2086
  nvcc -MM -gencode arch=compute_120a,code=sm_120a -std=c++17 ${VE_BUILDS[$lib]}
done | sed 's/^[^:]*://; s/\\$//' | tr ' ' '\n' | sed '/^$/d' | sort -u)
VE_KEY=$({ echo "$VE_RECIPE"; echo "$VE_SOURCES" | xargs sha256sum; } | sha256sum | cut -c1-32)
VE_LIBS=$VE_CACHE/$VE_KEY
exec {VE_LOCK}> "$VE_CACHE/$VE_KEY.lock"
flock "$VE_LOCK"
if [ -f "$VE_LIBS/build.txt" ]; then
  VE_HOW=reused
else
  VE_HOW=built
  VE_TMP=$(mktemp -d "$VE_CACHE/.$VE_KEY.XXXXXX")
  declare -A VE_PIDS=()
  for lib in "${VE_ORDER[@]}"; do
    # shellcheck disable=SC2086
    nvcc "${VE_FLAGS[@]}" ${VE_BUILDS[$lib]} -o "$VE_TMP/$lib.so" > "$VE_TMP/$lib.ptxas" 2>&1 &
    VE_PIDS[$lib]=$!
  done
  for lib in "${VE_ORDER[@]}"; do
    wait "${VE_PIDS[$lib]}" || {
      echo "setup: nvcc failed for $lib.so" >&2; tail -30 "$VE_TMP/$lib.ptxas" >&2; wait; rm -rf "$VE_TMP"; exit 4; }
  done
  { echo "$VE_RECIPE"; echo "$VE_SOURCES" | xargs sha256sum; (cd "$VE_TMP" && sha256sum -- *.so); } > "$VE_TMP/build.txt"
  mv -T "$VE_TMP" "$VE_LIBS"
  "$VENV/bin/python" -m research keep "$VE_LIBS" --owner @benchmarks --expires 7d --priority low \
    --regenerate "a dispatched point of a spec whose setup is benchmarks/harness/setup/vllm_encoded.sh" \
    --why "verity-bench: the bandwidth arms' libraries for sources keyed $VE_KEY" > "$VE_RUN/out/setup/keep.log" 2>&1 ||
    echo "setup: no retention record for $VE_LIBS (out/setup/keep.log)" >&2
fi
cp "$VE_LIBS"/*.so "$VE_RUN/setup/"
cp "$VE_LIBS"/build.txt "$VE_LIBS"/*.ptxas "$VE_RUN/out/setup/"
flock -u "$VE_LOCK"
exec {VE_LOCK}>&-
echo "$VE_HOW $VE_LIBS" >> "$VE_RUN/out/setup/build.txt"
export VERITY_VLLM_BENCH_WT_LIB=$VE_RUN/setup/wt.so VERITY_VLLM_BENCH_READER_LIB=$VE_RUN/setup/reader.so \
  VERITY_VLLM_BENCH_PAGED_LIB=$VE_RUN/setup/paged.so VERITY_VLLM_BENCH_TOPUP_LIB=$VE_RUN/setup/topup.so
