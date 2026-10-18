# Sourced by run_point.sh (a spec's `setup`): builds the bandwidth arms' GPU libraries (benchmarks/vllm_encoded/bench_arm.py)
# from this source tree, as benchmarks/kv_filler/run.sh builds the encoded stack (STACK=...) at its default field (1024,
# masked, FOLD6, 32 query heads) and the top-up, and exports the VERITY_VLLM_BENCH_*_LIB names the arms read. The libraries
# go to $RESEARCH_RUN_DIR/setup; nvcc's version, the flags, ptxas's reports and every source's and library's sha256 go to
# $RESEARCH_RUN_DIR/out/setup. Built per point, so a library always matches the ctypes declarations of the commit it runs.
VE_LIBS=${RESEARCH_RUN_DIR:-$PWD}/setup
VE_LOGS=${RESEARCH_RUN_DIR:-$PWD}/out/setup
mkdir -p "$VE_LIBS" "$VE_LOGS"
VE_FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
VE_FIELD=1024
declare -A VE_BUILDS=(
  [wt]="-DWT_FIELD=$VE_FIELD -DWT_MASK=1 -DWT_FOLD6=1 -DWT_RECEIPT=1 benchmarks/codec_tensor/wt.cu"
  [reader]="benchmarks/codec_tensor/reader.cu"
  [paged]="-DKV_FIELD=$VE_FIELD -DKV_MASK=1 -DKV_FOLD6=1 -DKV_QHEADS=32 benchmarks/vllm_encoded/paged.cu"
  [topup]="benchmarks/kv_filler/topup.cu"
)
{ nvcc --version; echo "flags: ${VE_FLAGS[*]}"; for lib in wt reader paged topup; do echo "$lib: ${VE_BUILDS[$lib]}"; done; } \
  > "$VE_LOGS/build.txt"
declare -A VE_PIDS=()
for lib in wt reader paged topup; do
  # shellcheck disable=SC2086
  nvcc "${VE_FLAGS[@]}" ${VE_BUILDS[$lib]} -o "$VE_LIBS/$lib.so" > "$VE_LOGS/$lib.ptxas" 2>&1 &
  VE_PIDS[$lib]=$!
done
for lib in wt reader paged topup; do
  wait "${VE_PIDS[$lib]}" || { echo "setup: nvcc failed for $lib.so" >&2; tail -30 "$VE_LOGS/$lib.ptxas" >&2; exit 4; }
done
sha256sum benchmarks/codec_tensor/wt.cu benchmarks/codec_tensor/reader.cu benchmarks/vllm_encoded/paged.cu \
  benchmarks/kv_filler/topup.cu "$VE_LIBS"/*.so >> "$VE_LOGS/build.txt"
export VERITY_VLLM_BENCH_WT_LIB=$VE_LIBS/wt.so VERITY_VLLM_BENCH_READER_LIB=$VE_LIBS/reader.so \
  VERITY_VLLM_BENCH_PAGED_LIB=$VE_LIBS/paged.so VERITY_VLLM_BENCH_TOPUP_LIB=$VE_LIBS/topup.so
