#!/usr/bin/env bash
# The kernels' part of a smoke run on one H100 (sm_90a): for every hm96 leaf kernel that takes its salts as data, the card's
# leaves and roots are held to the reference's for the same salt bytes, and each build takes its machine-code gate on the
# node's own toolkit. One command, which the smoke's runner calls while it holds the card (cwd anywhere):
#
#   bash benchmarks/kernels/smoke_h100.sh OUT
#
# What it checks, one line `PASS|FAIL|SKIP step: reason` each, in two parts:
#   off the card (the CPU, `--dry` runs these and stops)
#     toolkit            CUDA 13.0 as `machine.py toolkit 13.0` names it, the card's architecture and its -gencode
#     python             the Python the tests run with has torch, numpy and pytest
#     pearl_c.build      Pearl-C for sm_90a, `pearl_c_sm90/build.sh`: refused on a stack frame or spill (ptxas) and by its
#                        SASS gate; the debit cubin with its two gates
#     pearl_c.gate       the harness's SASS gate (.FTZ, FP32 MUFU) on both cubins, enforced here (build.sh only records it)
#     pearl_c.fixture    the check's fixture, regenerated: it carries the salts the kernels read
#     pearl_c.host       tests/benchmarks/pouw's Pearl-C sm_90a, hash and key tests. None of them reads a card: they hold
#                        the host twins, and one builds the cubin with the toolkit and takes its gate
#     commit_gpu.build   `commit_gpu.cu` for the card's gencode; commit_gpu.gate, the harness's gate on the binary
#     commit_gpu.cases   `reference.py prepare`'s quick cases, each with its salts
#     collect            pytest's collection of the device tests below
#     card_sku           vLLM's settings read the card file's name from CARD_SKU and refuse it unset, which collection
#                        does not reach (only the tests that serve a window build vLLM's committer); where this host
#                        has a card, `machine.py sku` names its record and the settings build for sm_90a under it
#   on the card
#     card               one card, an H100 (or H200: the same architecture), whose record states sm_90a; CARD_SKU is
#                        set for the steps below to that record's name (`machine.py sku NAME`), which the reason states
#     pearl_c.check      `run.py`'s check on the fixture (check_chains and the rest): every kernel's bytes against the
#                        fixture's, no timing (SHAPES is empty)
#     rows.device        the row engine's tests, `sha512_rows`: the device's leaves against the host's and the reference's
#     slots.device       the slot commit's device tests, `commit_slots/test_gpu.py`: strided and misaligned salt buffers,
#                        graph replay under `resalt`
#     zy.device          the z/y commit's device tests, `test_pouw_h1_registered_device.py`; its one strict xfail (a window
#                        served on the device, which waits for compute-accounting's switch) is named in the reason
#     zy.h1_zy_gpu       `h1_zy_gpu.py` at one repetition: each call's device records equal to the two kernels apart
#     device_libs.gate   the harness's gate on every device library those tests built
#     commit_gpu.selftest, commit_gpu.check, commit_gpu.check_small
#                        the binary's selftest, then `reference.py check` of it on the cases, with and without
#                        COMMIT_GPU_SMALL=1
# On the card a test that skips for want of a device or nvcc is a FAIL: nothing passes by not running.
#
# What it does not check: anything for sm_120a (the RTX PRO 6000's builds wait for that machine); any timing or rate
# (nothing here is a measurement, and no number a step prints is a result); the 512-bit receipt kernels, which are not on
# this branch; the bandwidth audit's kernels, which did not change; Pearl-C's grouped kernels on a card (their card check
# goes through the served path, which has no salts until compute-accounting's switch; their host twins are in
# pearl_c.host); and the served path itself.
#
# OUT (the argument, else $OUT) gets only public verdicts and logs: smoke.json (each step, its verdict, seconds and the
# sha256 of the cubin or binary it built or ran), logs/STEP.log, the gates' records and the checks' lines. Fixtures, cases,
# salts and builds stay in a scratch directory outside OUT (--scratch DIR, else a fresh one under /dev/shm or /tmp,
# removed at exit). Exits nonzero if any step FAILs.
#
#   --dry           run the part off the card and stop; the card's steps are SKIPs. On a machine with no CUDA 13.0 the
#                   builds are SKIPs too.
#   --scratch DIR   keep the scratch in DIR. A run finds there what a `--dry` of the same commit, tree, toolkit and Python
#                   built, and goes straight to the card: `--dry --scratch DIR` before taking the card, then the same
#                   command without `--dry`, holds the card for the card's steps alone.
#
# Needs, outside the hold: the workspace synced with CUDA torch (`uv sync --all-packages --extra torch`; PY names another
# Python), g++, and in the toolkit nvcc, cuobjdump, nvdisasm, cuBLASLt and NVML's library. CUDA_HOME overrides the toolkit.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

DRY=0 SCRATCH="" O=${OUT:-}
while [ $# -gt 0 ]; do
  case $1 in
    --dry) DRY=1 ;;
    --scratch) SCRATCH=${2:?--scratch takes a directory}; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "smoke_h100.sh: no option $1 (--dry, --scratch DIR)" >&2; exit 2 ;;
    *) O=$1 ;;
  esac
  shift
done
[ -n "$O" ] || { echo "smoke_h100.sh: name the output directory: smoke_h100.sh [--dry] [--scratch DIR] OUT" >&2; exit 2; }
mkdir -p "$O/logs" && O=$(cd "$O" && pwd) || exit 2
LOGS=$O/logs
OWN_SCRATCH=0
if [ -z "$SCRATCH" ]; then
  base=/tmp; [ -d /dev/shm ] && [ -w /dev/shm ] && base=/dev/shm
  SCRATCH=$(mktemp -d "$base/kernels-smoke.XXXXXX") || exit 2
  OWN_SCRATCH=1
fi
mkdir -p "$SCRATCH" && chmod 700 "$SCRATCH" && SCRATCH=$(cd "$SCRATCH" && pwd) || exit 2
case "$SCRATCH/" in "$O"/*) echo "smoke_h100.sh: the scratch $SCRATCH is under the output $O: salts would be published" >&2; exit 2 ;; esac
case "$O/" in "$SCRATCH"/*) echo "smoke_h100.sh: the output $O is under the scratch $SCRATCH" >&2; exit 2 ;; esac
[ "$OWN_SCRATCH" = 1 ] && trap 'rm -rf "$SCRATCH"' EXIT

PY=${PY:-$ROOT/.venv/bin/python}
POUW=$ROOT/verity/protocols/resource_accounting/compute/pouw/prover
PEARL=$POUW/kernels/pearl_c_sm90
CG=$ROOT/verity/commitments/cuda/commit_gpu
TAG=a0r0l
ROWS_T=$ROOT/tests/verity/commitments/cuda/sha512_rows
SLOTS_T=$ROOT/tests/verity/commitments/cuda/commit_slots
ZY_T=$ROOT/tests/adapters/vllm
ZY_FILE=tests/protocols/test_pouw_h1_registered_device.py
ROWS=$SCRATCH/rows.txt
SKIPPED=77
CUDA="" GENCODE="" NVCC_VERSION="" GPU="" CARD_OK=0
# the target the smoke's builds are for: Pearl-C's tree states its own (build.sh's ARCH, which refuses another's GENCODE)
TARGET=$(sed -n 's/^ARCH=\([a-z0-9_]*\).*/\1/p' "$PEARL/build.sh")
[ -n "$TARGET" ] || { echo "smoke_h100.sh: $PEARL/build.sh states no ARCH, the target its kernels are built for" >&2; exit 2; }
: > "$ROWS"

sha() { sha256sum "$1" | cut -d' ' -f1; }

# step NAME FUNCTION: runs it with its output in logs/NAME.log. The function leaves its reason in REASON and a built or
# run binary's sha256 in SHA, and returns 0 (PASS), $SKIPPED (SKIP) or anything else (FAIL).
step() {
  local name=$1 t0 rc verdict
  REASON="" SHA=""
  t0=$(date +%s.%N)
  "$2" > "$LOGS/$name.log" 2>&1
  rc=$?
  case $rc in 0) verdict=PASS ;; "$SKIPPED") verdict=SKIP ;; *) verdict=FAIL; REASON="${REASON:-exit $rc}; see logs/$name.log" ;; esac
  REASON=$(printf '%s' "$REASON" | tr '\037\n' '  ')
  printf '%s\037%s\037%s\037%s\037%s\n' "$name" "$verdict" "$(echo "$(date +%s.%N) $t0" | awk '{ printf "%.1f", $1 - $2 }')" "$SHA" "$REASON" >> "$ROWS"
  printf '%-4s %s: %s\n' "$verdict" "$name" "$REASON"
}

# the reason a step that needs the toolkit gives without one: a SKIP under --dry, a FAIL on the card's machine
no_toolkit() { REASON="no CUDA 13.0 toolkit (step toolkit)"; [ "$DRY" = 1 ] && return "$SKIPPED"; return 1; }

# junit XML [card]: pytest's verdicts as a reason on stdout; exit 1 on a failure, an error, no test at all, or (card) a
# skip for want of a device or nvcc
JUNIT='
import re, sys
import xml.etree.ElementTree as ET
card = len(sys.argv) > 2
cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
bad, skips, xfails, missed = [], [], [], []
whys = set()
for c in cases:
    name = c.get("name")
    if c.find("failure") is not None or c.find("error") is not None:
        bad.append(name)
    elif (s := c.find("skipped")) is not None:
        why = (s.get("message") or "").strip()
        if s.get("type") == "pytest.xfail":
            xfails.append(name)
            whys.add(why[:90] + ("..." if len(why) > 90 else ""))
        elif card and re.search(r"cuda|nvcc|device|gpu", why, re.I):
            missed.append(f"{name} ({why})")
        else:
            skips.append(why)
passed = len(cases) - len(bad) - len(skips) - len(xfails) - len(missed)
out = [f"{passed} passed"]
if bad:
    out.append(f"{len(bad)} FAILED: " + ", ".join(bad[:6]))
if missed:
    out.append(f"{len(missed)} skipped on the card: " + ", ".join(missed[:4]))
if skips:
    out.append(f"{len(skips)} skipped (" + "; ".join(sorted(set(skips))[:4]) + ")")
if xfails:
    out.append(f"{len(xfails)} xfailed, strict (" + "; ".join(sorted(whys)) + "): " + ", ".join(xfails[:4])
               + (" and others" if len(xfails) > 4 else ""))
if not cases:
    out.append("no test ran")
print("; ".join(out))
sys.exit(1 if bad or missed or not cases else 0)
'

# tests DIR XML CARD FILES...: pytest on a suite's files from its own directory, then its verdicts as the reason
tests() {
  local dir=$1 xml=$2 card=$3; shift 3
  (cd "$dir" && "$PY" -m pytest -q -rs -p no:cacheprovider --junitxml "$xml" "$@")
  local rc=$?
  [ -s "$xml" ] || { REASON="pytest wrote no verdicts (exit $rc)"; return 1; }
  REASON=$(python3 -c "$JUNIT" "$xml" $card) || return 1
  [ $rc = 0 ] || { REASON="$REASON; pytest exit $rc"; return 1; }
}

# ---- off the card ----------------------------------------------------------------------------------------------------

p_toolkit() {
  local kit arch said
  kit=$(python3 "$ROOT/benchmarks/kernels/machine.py" toolkit 13.0) && [ -x "$kit/bin/nvcc" ] || {
    REASON="machine.py toolkit 13.0 names no toolkit with nvcc: set CUDA_HOME to the node's CUDA 13.0"
    [ "$DRY" = 1 ] && return "$SKIPPED"; return 1; }
  if arch=$(python3 "$ROOT/benchmarks/kernels/machine.py" arch); then said="the machine's record states $arch"
  else arch=$TARGET; said="the machine's record is unread here, so $TARGET is taken, Pearl-C's build's, and the card step holds the card to it"; fi
  [ "$arch" = "$TARGET" ] || { REASON="the machine's card is $arch: this smoke's builds are $TARGET's"; return 1; }
  GENCODE=$(python3 "$ROOT/benchmarks/kernels/machine.py" gencode "$arch") || { REASON="no -gencode for $arch"; return 1; }
  CUDA=$kit
  NVCC_VERSION=$("$CUDA/bin/nvcc" --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  "$CUDA/bin/nvcc" --version > "$O/nvcc.txt" 2>&1
  for tool in cuobjdump nvdisasm; do
    [ -x "$CUDA/bin/$tool" ] || { REASON="$CUDA has no bin/$tool, which the gates read"; return 1; }
  done
  REASON="$CUDA, nvcc $NVCC_VERSION, -gencode $GENCODE ($said)"
}

p_python() {
  local said
  said=$("$PY" -c 'import numpy, pytest, torch; print("torch", torch.__version__, "CUDA build" if torch.version.cuda else "CPU build")') || {
    REASON="$PY lacks torch, numpy or pytest: uv sync --all-packages --extra torch"; return 1; }
  REASON="$PY: $said"
  case $said in *"CPU build") [ "$DRY" = 1 ] || { REASON="$REASON: the card's tests need uv sync --all-packages --extra torch"; return 1; } ;; esac
}

p_pearl_build() {
  [ -n "$CUDA" ] || { no_toolkit; return; }
  CUDA_HOME=$CUDA GENCODE=$GENCODE bash "$PEARL/build.sh" "$SCRATCH/pearl_c" || { REASON="build.sh refused the build (the target, ptxas, the SASS gate or the debit cubin's gates)"; return 1; }
  mkdir -p "$O/pearl_c"
  cp "$SCRATCH/pearl_c"/{sass.json,ptxas.txt,toolchain.txt} "$SCRATCH/pearl_c/debit"/{debit_ptxas.txt,debit_mma.txt} "$O/pearl_c/"
  cp "$PEARL/run.py" "$SCRATCH/pearl_c/"
  SHA=$(sha "$SCRATCH/pearl_c/pearl_c_sm90.cubin")
  REASON="nvcc $NVCC_VERSION: no stack frame or spill, the SASS gate passes; debit cubin $(sha "$SCRATCH/pearl_c/debit/debit_sm90.cubin" | cut -c1-16)"
}

# gate JSON BINARY...: the harness's SASS gate
gate() {
  local json=$1; shift
  POUW_SASS_GATE_CACHE=none python3 "$POUW/sass_gate.py" --cuda-bin "$CUDA/bin" "$@" --json "$json"
}

p_pearl_gate() {
  [ -n "$CUDA" ] || { no_toolkit; return; }
  local cubin=$SCRATCH/pearl_c/pearl_c_sm90.cubin
  [ -s "$cubin" ] || { REASON="no cubin (step pearl_c.build)"; return 1; }
  SHA=$(sha "$cubin")
  gate "$O/pearl_c/harness_gate.json" "$cubin" "$SCRATCH/pearl_c/debit/debit_sm90.cubin" || { REASON="the harness's gate flags an instruction"; return 1; }
  REASON="no .FTZ or FP32 MUFU outside ptxas's pinned sequences, in both cubins"
}

p_pearl_fixture() {
  mkdir -p "$SCRATCH/pearl_c"
  rm -rf "$SCRATCH/pearl_c/check"
  (cd "$ROOT" && "$PY" benchmarks/pouw/pearl_c_sm90/fixture.py "$SCRATCH/pearl_c/check") || { REASON="fixture.py failed"; return 1; }
  ls "$SCRATCH/pearl_c/check"/salts*.bin > /dev/null 2>&1 || { REASON="the fixture holds no salts"; return 1; }
  REASON="256 x 384 x 1024, $(ls "$SCRATCH/pearl_c/check" | wc -l) files with their salts, in the scratch"
}

p_pearl_host() {
  local built="with the toolkit's build and gate"
  [ -n "$CUDA" ] || built="the toolkit's build skipped"
  (cd "$ROOT/tests/benchmarks/pouw" && ls test_pearl_c_sm90*.py test_pearl_c_hash_wide.py test_pearl_c_keys.py) > "$SCRATCH/pearl_host.files"
  # shellcheck disable=SC2046
  VERITY_POUW_NVCC=$CUDA tests "$ROOT/tests/benchmarks/pouw" "$O/logs/pearl_c.host.xml" "" $(cat "$SCRATCH/pearl_host.files") || return 1
  REASON="$REASON; host twins, $built; none reads a card"
}

p_cg_build() {
  [ -n "$CUDA" ] || { no_toolkit; return; }
  mkdir -p "$SCRATCH/commit_gpu"
  (cd "$ROOT" && PATH=$CUDA/bin:$PATH "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 \
    -o "$SCRATCH/commit_gpu/commit_gpu_$TAG" "$CG/commit_gpu.cu" -lcublasLt -lnvidia-ml) || {
    REASON="nvcc failed (it links cuBLASLt and NVML's library)"; return 1; }
  SHA=$(sha "$SCRATCH/commit_gpu/commit_gpu_$TAG")
  REASON="commit_gpu_$TAG, nvcc $NVCC_VERSION, -gencode $GENCODE"
}

p_cg_gate() {
  [ -n "$CUDA" ] || { no_toolkit; return; }
  local bin=$SCRATCH/commit_gpu/commit_gpu_$TAG
  [ -x "$bin" ] || { REASON="no binary (step commit_gpu.build)"; return 1; }
  SHA=$(sha "$bin")
  mkdir -p "$O/commit_gpu"
  gate "$O/commit_gpu/harness_gate.json" "$bin" || { REASON="the harness's gate flags an instruction"; return 1; }
  REASON="no .FTZ or FP32 MUFU outside ptxas's pinned sequences"
}

p_cg_cases() {
  mkdir -p "$SCRATCH/commit_gpu"
  rm -rf "$SCRATCH/commit_gpu/cases"
  (cd "$ROOT" && PYTHONPATH=. "$PY" "$CG/reference.py" prepare "$SCRATCH/commit_gpu/cases" quick > "$SCRATCH/commit_gpu/prepare.jsonl") || {
    REASON="reference.py prepare failed"; return 1; }
  REASON="$(wc -l < "$SCRATCH/commit_gpu/prepare.jsonl") quick cases with their salts, in the scratch"
}

p_collect() {
  local n total=0 said=""
  for pair in "$ROWS_T:test_engine.py" "$SLOTS_T:test_gpu.py" "$ZY_T:$ZY_FILE"; do
    n=$(cd "${pair%%:*}" && "$PY" -m pytest --collect-only -q -p no:cacheprovider "${pair#*:}" | tee /dev/stderr | awk '/::/ { n++ } /^[^ ]+\.py: [0-9]+$/ { n += $2 } END { print n + 0 }')
    [ "${n:-0}" -gt 0 ] || { REASON="pytest collects nothing from ${pair#*:}"; return 1; }
    said="$said${said:+, }${pair#*:} $n"; total=$((total + n))
  done
  REASON="$total tests the card's steps run: $said"
}

# what a --dry's builds answer for: the commit, the tree's edits, the toolkit and the Python
stamp() {
  (cd "$ROOT" && git rev-parse HEAD 2> /dev/null || echo "no git $RANDOM$RANDOM"
   cd "$ROOT" && git status --porcelain 2> /dev/null | sha256sum; cd "$ROOT" && git diff HEAD 2> /dev/null | sha256sum
   echo "${CUDA_HOME:-}"; echo "$PY") | sha256sum | cut -d' ' -f1
}

# vLLM's reading of CARD_SKU (the first argument; empty: unset), from the suite's directory: the arch it builds for, or
# its refusal's first line
vllm_arch() {
  (cd "$ZY_T" && if [ -n "$1" ]; then export CARD_SKU=$1; else unset CARD_SKU; fi
   "$PY" -c 'from verity_vllm import settings
from verity_vllm.config import cuda_arch
try:
    print(cuda_arch(settings.card_sku()))
except ValueError as e:
    print(e)')
}

p_card_sku() {
  local said sku arch
  said=$(vllm_arch "") || { REASON="verity_vllm.settings does not import"; return 1; }
  case $said in "CARD_SKU is not set"*) ;; *) REASON="verity_vllm.settings no longer refuses an unset CARD_SKU by that name (it said: $said), so what the card step exports is not what it reads"; return 1 ;; esac
  if sku=$(python3 "$ROOT/benchmarks/kernels/machine.py" sku 2> "$SCRATCH/card_sku.err") && [ -n "$sku" ]; then
    arch=$(vllm_arch "$sku")
    [ -n "$GENCODE" ] || { REASON="no -gencode to hold vLLM's arch under CARD_SKU=$sku to (step toolkit)"; [ "$DRY" = 1 ] && return "$SKIPPED"; return 1; }
    # vLLM writes the arch as torch does (9.0a); the toolkit step's -gencode ends with the same target without the dot
    [ "${arch/./}" = "${GENCODE##*_}" ] || { REASON="under CARD_SKU=$sku, this host's card's record, vLLM's settings build for '$arch', not for -gencode $GENCODE"; return 1; }
    REASON="vLLM's settings refuse an unset CARD_SKU, and under CARD_SKU=$sku, this host's card's record, build for $arch; the card step sets it"
  elif [ "$DRY" = 1 ]; then
    REASON="vLLM's settings refuse an unset CARD_SKU by that name; no card here names its record ($(head -c 200 "$SCRATCH/card_sku.err" | tr '\n' ' ')), so the name is first read on the card, where the card step sets it"
  else
    REASON="machine.py sku names no card record: $(head -c 300 "$SCRATCH/card_sku.err" | tr '\n' ' ')"; return 1
  fi
}

PREPARED=$SCRATCH/prepared
if [ "$DRY" = 0 ] && [ -s "$PREPARED/stamp" ] && [ "$(cat "$PREPARED/stamp")" = "$(stamp)" ] \
   && ! cut -d $'\037' -f2 "$PREPARED/rows.txt" | grep -qv PASS; then
  cp -r "$PREPARED/out/." "$O/"
  # shellcheck disable=SC1091
  . "$PREPARED/vars"
  while IFS=$'\037' read -r -u 3 name verdict secs sum reason; do
    # a --dry may run with a CPU build of torch; the card's run may not
    [ "$name" = python ] && { step python p_python; continue; }
    reason="$reason [built by the --dry before this run]"
    printf '%s\037%s\037%s\037%s\037%s\n' "$name" "$verdict" "$secs" "$sum" "$reason" >> "$ROWS"
    printf '%-4s %s: %s\n' "$verdict" "$name" "$reason"
  done 3< "$PREPARED/rows.txt"
else
  rm -rf "$PREPARED"
  step toolkit p_toolkit
  step python p_python
  step pearl_c.build p_pearl_build
  step pearl_c.gate p_pearl_gate
  step pearl_c.fixture p_pearl_fixture
  step pearl_c.host p_pearl_host
  step commit_gpu.build p_cg_build
  step commit_gpu.gate p_cg_gate
  step commit_gpu.cases p_cg_cases
  step collect p_collect
  step card_sku p_card_sku
  if [ "$OWN_SCRATCH" = 0 ]; then
    mkdir -p "$PREPARED/out"
    cp -r "$O/." "$PREPARED/out/"
    cp "$ROWS" "$PREPARED/rows.txt"
    printf 'CUDA=%q GENCODE=%q NVCC_VERSION=%q\n' "$CUDA" "$GENCODE" "$NVCC_VERSION" > "$PREPARED/vars"
    stamp > "$PREPARED/stamp"
  fi
fi

# ---- on the card -----------------------------------------------------------------------------------------------------

c_card() {
  local rows row index uuid name arch sku
  rows=$(nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader) || { REASON="nvidia-smi names no card"; return 1; }
  if [ -n "${GPU_LEASE_UUID:-}" ]; then row=$(printf '%s\n' "$rows" | grep -F "$GPU_LEASE_UUID")
  elif [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then row=$(printf '%s\n' "$rows" | grep -E "^$CUDA_VISIBLE_DEVICES,")
  elif [ "$(printf '%s\n' "$rows" | wc -l)" = 1 ]; then row=$rows
  else REASON="several cards and neither GPU_LEASE_UUID nor a one-index CUDA_VISIBLE_DEVICES says which is held"; return 1; fi
  [ -n "$row" ] || { REASON="the held card is not among nvidia-smi's"; return 1; }
  IFS=, read -r index uuid name _ <<< "$row"
  uuid=${uuid# } name=${name# }
  printf '%s\n' "$row" > "$O/gpu.txt"
  arch=$(python3 "$ROOT/benchmarks/kernels/machine.py" arch "$name") || { REASON="no card record names $name"; return 1; }
  [ "$arch" = "$TARGET" ] || { REASON="$name's record states $arch, and the builds are $TARGET's"; return 1; }
  sku=$(python3 "$ROOT/benchmarks/kernels/machine.py" sku "$name") && [ -n "$sku" ] || { REASON="no card record names $name, so there is no CARD_SKU to set"; return 1; }
  [ -z "${CARD_SKU:-}" ] || [ "$CARD_SKU" = "$sku" ] || { REASON="CARD_SKU is $CARD_SKU here, and $name's record is $sku"; return 1; }
  export CUDA_VISIBLE_DEVICES=$index PEARLC_EXPECT_UUID=$uuid CARD_SKU=$sku
  GPU=$name CARD_OK=1
  REASON="$name, index $index, its record states $arch; CARD_SKU=$sku set for the card's steps"
}

# what every later card step needs: the card and the toolkit
held() {
  [ "$CARD_OK" = 1 ] || { REASON="no card (step card)"; return 1; }
  [ -n "$CUDA" ] || { REASON="no CUDA 13.0 toolkit (step toolkit)"; return 1; }
}

c_pearl_check() {
  held || return 1
  local dir=$SCRATCH/pearl_c
  [ -s "$dir/pearl_c_sm90.cubin" ] && [ -s "$dir/check/check.json" ] || { REASON="no cubin or fixture (steps pearl_c.build, pearl_c.fixture)"; return 1; }
  SHA=$(sha "$dir/pearl_c_sm90.cubin")
  mkdir -p "$O/pearl_c"
  (cd "$dir" && SHAPES="" python3 run.py check/check.json > "$O/pearl_c/check.jsonl")
  local rc=$?
  case $rc in
    0) REASON="every kernel's bytes equal the fixture's for the fixture's salts (pearl_c/check.jsonl)" ;;
    3) REASON="run.py refused the device (exit 3)"; return 1 ;;
    4) REASON="a check differs from the fixture (exit 4, pearl_c/check.jsonl)"; return 1 ;;
    *) REASON="run.py exit $rc: it did not reach its check"; return 1 ;;
  esac
}

# device_tests DIR NAME FILE: a suite's device tests, their libraries built under the scratch (the builds' cache is in
# the home directory)
device_tests() {
  held || return 1
  mkdir -p "$SCRATCH/home"
  HOME=$SCRATCH/home PATH=$CUDA/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-} tests "$1" "$O/logs/$2.xml" card "$3"
}
c_rows() { device_tests "$ROWS_T" rows.device test_engine.py; }
c_slots() { device_tests "$SLOTS_T" slots.device test_gpu.py; }
c_zy() { device_tests "$ZY_T" zy.device "$ZY_FILE"; }

c_zy_gpu() {
  held || return 1
  mkdir -p "$SCRATCH/home" "$SCRATCH/zy"
  (cd "$ZY_T" && HOME=$SCRATCH/home PATH=$CUDA/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-} \
    "$PY" tests/protocols/h1_zy_gpu.py --out "$SCRATCH/zy" --reps 1 > "$SCRATCH/zy/h1_zy_gpu.log" 2>&1)
  local rc=$?
  # the verdicts without the script's timings, which one repetition does not measure
  sed -E 's/  kernels .*//; /^step \(/d' "$SCRATCH/zy/h1_zy_gpu.log"
  [ $rc = 0 ] && grep -q '^h1_zy_gpu: PASS' "$SCRATCH/zy/h1_zy_gpu.log" || { REASON="a call's records differ, or the kernels did not build (exit $rc)"; return 1; }
  REASON="each of the four calls' device records equal the two kernels apart"
}

c_libs_gate() {
  held || return 1
  local libs=() names="" sums=""
  mapfile -t libs < <(find "$SCRATCH/home" -name '*cuda*.so' 2> /dev/null | sort)
  [ ${#libs[@]} -gt 0 ] || { REASON="the device tests built no library under the scratch"; return 1; }
  for lib in "${libs[@]}"; do
    names="$names${names:+, }$(basename "$lib")"; sums="$sums${sums:+,}$(basename "$lib")=$(sha "$lib")"
  done
  SHA=$sums
  gate "$O/device_libs_gate.json" "${libs[@]}" || { REASON="the harness's gate flags an instruction in one of: $names"; return 1; }
  REASON="${#libs[@]} libraries pass the harness's gate: $names"
}

cg_bin() {
  held || return 1
  BIN=$SCRATCH/commit_gpu/commit_gpu_$TAG
  [ -x "$BIN" ] && [ -d "$SCRATCH/commit_gpu/cases" ] || { REASON="no binary or cases (steps commit_gpu.build, commit_gpu.cases)"; return 1; }
  SHA=$(sha "$BIN")
  mkdir -p "$O/commit_gpu"
  export LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-}
}

c_cg_selftest() {
  cg_bin || return 1
  "$BIN" selftest > "$O/commit_gpu/selftest.jsonl" || { REASON="the selftest exits $? (commit_gpu/selftest.jsonl)"; return 1; }
  REASON="the binary's selftest passes"
}

# cg_check NAME [VAR=VALUE]: reference.py check of the binary on the cases
cg_check() {
  cg_bin || return 1
  local name=$1; shift
  (cd "$ROOT" && env "$@" PYTHONPATH=. "$PY" "$CG/reference.py" check "$BIN" "$SCRATCH/commit_gpu/cases" "$TAG" > "$O/commit_gpu/$name.jsonl") || {
    REASON="reference.py check exits $?: a case's root differs from the reference's, or the binary did not run (commit_gpu/$name.jsonl)"; return 1; }
  REASON="$(grep -c . "$O/commit_gpu/$name.jsonl") lines, every case's root equal to the reference's for its salts"
}
c_cg_check() { cg_check check; }
c_cg_check_small() { cg_check check_small COMMIT_GPU_SMALL=1; }

dry() { REASON="needs the card (--dry)"; return "$SKIPPED"; }
card_step() { if [ "$DRY" = 1 ]; then step "$1" dry; else step "$1" "$2"; fi; }

card_step card c_card
card_step pearl_c.check c_pearl_check
card_step rows.device c_rows
card_step slots.device c_slots
card_step zy.device c_zy
card_step zy.h1_zy_gpu c_zy_gpu
card_step device_libs.gate c_libs_gate
card_step commit_gpu.selftest c_cg_selftest
card_step commit_gpu.check c_cg_check
card_step commit_gpu.check_small c_cg_check_small

COMMIT=$(cd "$ROOT" && git rev-parse HEAD 2> /dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}")
DRY=$DRY COMMIT=$COMMIT TOOLKIT=$CUDA NVCC=$NVCC_VERSION GENCODE=$GENCODE GPU=$GPU python3 - "$ROWS" "$O/smoke.json" << 'EOF'
import json, os, sys

steps = []
for line in open(sys.argv[1]):
    name, verdict, seconds, sha, reason = line.rstrip("\n").split("\x1f")
    step = {"step": name, "verdict": verdict, "seconds": float(seconds), "reason": reason}
    if "=" in sha:
        step["sha256"] = dict(pair.split("=") for pair in sha.split(","))
    elif sha:
        step["sha256"] = sha
    steps.append(step)
count = {v: sum(s["verdict"] == v for s in steps) for v in ("PASS", "FAIL", "SKIP")}
e = os.environ
out = {"script": "benchmarks/kernels/smoke_h100.sh", "commit": e["COMMIT"], "dry": e["DRY"] == "1", "toolkit": e["TOOLKIT"] or None,
       "nvcc": e["NVCC"] or None, "gencode": e["GENCODE"] or None, "card": e["GPU"] or None,
       "verdict": "FAIL" if count["FAIL"] else "PASS", "count": count, "steps": steps}
with open(sys.argv[2], "w") as f:
    f.write(json.dumps(out, indent=1) + "\n")
print(f"smoke_h100{' --dry' if out['dry'] else ''}: {out['verdict']} ({count['PASS']} pass, {count['FAIL']} fail, {count['SKIP']} skip) -> {sys.argv[2]}")
sys.exit(1 if count["FAIL"] else 0)
EOF
