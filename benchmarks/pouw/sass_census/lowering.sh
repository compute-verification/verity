#!/usr/bin/env bash
# The lowering probe: what ptxas emits on the H200 (sm_90a) and the RTX PRO 6000 (sm_120a) for each PTX form a coverage
# row (benchmarks/pouw/machine_model/data/cover/rows.toml) cites, or that it refuses it. Compiles lowering.cu's cases
# with nvcc -cubin and reads cuobjdump -sass; CPU only, no GPU. Prints a line a case and architecture (its opcodes, or
# ptxas's refusal) against its expectation below, then the verdict; exits 1 if a case lowers otherwise. With $OUT (under
# research run) it also writes lowering.txt and each case's SASS there:
#
#   research run --on auto --cwd source -- bash benchmarks/pouw/sass_census/lowering.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-$(ls -d /usr/local/cuda-13*/bin/nvcc 2>/dev/null | sort -V | tail -1)}
[ -x "${NVCC:-}" ] || NVCC=$(command -v nvcc)
CUOBJDUMP=${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
[ -n "${OUT:-}" ] && mkdir -p "$OUT/sass"

# case arch expectation: `refused`, or space-separated `+RE` (some opcode matches) and `-RE` (none does)
EXPECT='
f32x2 sm_90a refused
f32x2 sm_120a +^FFMA$ -^FFMA2
multimem sm_90a +^LDGMC\.
multimem sm_120a +^LDGMC\.
vabsdiff4 sm_90a +^VABSDIFF4
vabsdiff4 sm_120a +^VIADD +^VIMNMX -^VABSDIFF4
texture sm_90a +^TEX +^TLD4 +^TXD +^TXQ
texture sm_120a +^TEX +^TLD4 +^TXD +^TXQ
surface sm_90a +^SULD +^SUST +^SURED
surface sm_120a +^SULD +^SUST +^SURED
match sm_90a +^MATCH\.ANY +^MATCH\.ALL
match sm_120a +^MATCH\.ANY +^MATCH\.ALL
f16acc sm_90a +^HMMA\.16816\.F16$
f16acc sm_120a +^HMMA\.16816\.F16$
b1xor sm_90a +^BMMA\.168256\.AND\.POPC -XOR
b1xor sm_120a -^BMMA +^IMMA\.
fp6 sm_90a refused
fp6 sm_120a +^QMMA\.16832\.F32\.E3M2\.E2M3$
qmma_sf sm_90a refused
qmma_sf sm_120a +^QMMA\.SF\.16832\.F32\.E4M3\.E4M3\. +^QMMA\.SF\.16832\.F32\.E3M2\.E2M3\.
omma_sp sm_90a refused
omma_sp sm_120a +^OMMA\.SF\.SP\.168128\.
tcgen05 sm_90a refused
tcgen05 sm_120a refused
wgmma sm_90a +^HGMMA\.
wgmma sm_120a refused
dmma sm_90a +^DMMA\.
dmma sm_120a +^DMMA\.
s4 sm_90a +^IMMA\.16832\.S8\.S8$ +^CALL\.
s4 sm_120a +^IMMA\.16832\.S8\.S8$ +^CALL\.
'

say() { echo "$*"; [ -n "${OUT:-}" ] && echo "$*" >>"$OUT/lowering.txt"; return 0; }
say "lowering probe: $("$NVCC" --version | tail -1)"
n=0 bad=0
while read -r case arch expect; do
  [ -n "$case" ] || continue
  n=$((n + 1))
  cubin="$WORK/${case}_$arch.cubin"
  if ! "$NVCC" -arch="$arch" -DCASE_"$case" -cubin -o "$cubin" "$HERE/lowering.cu" >"$WORK/log" 2>&1; then
    why=$(grep -m1 -E "error" "$WORK/log" | sed -E 's/^ptxas [^,]*, line [0-9]+; //')
    if [ "$expect" = refused ]; then say "ok   $case $arch: refused ($why)"; else say "FAIL $case $arch: refused ($why), expected $expect"; bad=$((bad + 1)); fi
    continue
  fi
  "$CUOBJDUMP" -sass "$cubin" >"$WORK/sass"
  [ -n "${OUT:-}" ] && cp "$WORK/sass" "$OUT/sass/${case}_$arch.sass"
  ops=$(sed -nE 's#^\s+/\*[0-9a-f]{4,}\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*).*#\2#p' "$WORK/sass" | sort -u)
  ok=1
  if [ "$expect" = refused ]; then ok=0; else
    for e in $expect; do
      re=${e:1}
      if [ "${e:0:1}" = + ]; then grep -qE "$re" <<<"$ops" || ok=0; else grep -qE "$re" <<<"$ops" && ok=0; fi
    done
  fi
  shown=$(grep -vE '^(NOP|EXIT|BRA|S2R|LDC|LDC\.64|LDCU|LDCU\.64|ULDC|ULDC\.64|STG\.E.*|LDG\.E.*|MOV|IMAD\.WIDE\.U32|IMAD\.SHL\.U32|SHF\.L\.U32|IMAD\.MOV\.U32|CS2R|RET\.REL\.NODEC)$' <<<"$ops" | tr '\n' ' ')
  if [ $ok = 1 ]; then say "ok   $case $arch: $shown"; else say "FAIL $case $arch: $shown, expected $expect"; bad=$((bad + 1)); fi
done <<<"$EXPECT"
say "verdict: $((n - bad))/$n cases lower as expected"
[ $bad = 0 ]
