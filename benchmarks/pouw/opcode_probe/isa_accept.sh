#!/usr/bin/env bash
# What ptxas does with each instruction form A7's menu has no verdict for: a one-instruction PTX kernel a form,
# assembled for one target, and the SASS opcodes it becomes. CPU only; no GPU. One JSON line a form in
# $OUT/isa_accept.jsonl: accepted or not, ptxas's message, the SASS opcodes.
#
#   research run --on auto --cwd source -- bash benchmarks/pouw/opcode_probe/isa_accept.sh [sm_120a ...]
set -uo pipefail
OUT=${OUT:?run under research run}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
BIN=$(dirname "$NVCC")
PTXAS=$BIN/ptxas
CUOBJDUMP=$BIN/cuobjdump
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$PTXAS" --version >"$OUT/ptxas_version.txt" 2>&1
targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(sm_120a)

# form name, then the kernel body (registers declared, operands loaded from p, the instruction, results stored)
forms=(
  add_imm "
 .reg .b32 a, d; ld.global.u32 a, [rp];
 add.u32 d, a, 17;
 st.global.u32 [rp], d;"
  add_reg "
 .reg .b32 a, b, d; ld.global.u32 a, [rp]; ld.global.u32 b, [rp+4];
 add.u32 d, a, b;
 st.global.u32 [rp], d;"
  add3_reg "
 .reg .b32 a, b, c, t, d; ld.global.u32 a, [rp]; ld.global.u32 b, [rp+4]; ld.global.u32 c, [rp+8];
 add.u32 t, a, b; add.u32 d, t, c;
 st.global.u32 [rp], d;"
  mad_lo "
 .reg .b32 a, b, c, d; ld.global.u32 a, [rp]; ld.global.u32 b, [rp+4]; ld.global.u32 c, [rp+8];
 mad.lo.u32 d, a, b, c;
 st.global.u32 [rp], d;"
  ffma2 "
 .reg .b64 a, b, c, d; ld.global.b64 a, [rp]; ld.global.b64 b, [rp+8]; ld.global.b64 c, [rp+16];
 fma.rn.f32x2 d, a, b, c;
 st.global.b64 [rp], d;"
  dmma_m8n8k4 "
 .reg .f64 a, b, c0, c1, d0, d1;
 ld.global.f64 a, [rp]; ld.global.f64 b, [rp+8]; ld.global.f64 c0, [rp+16]; ld.global.f64 c1, [rp+24];
 mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {d0, d1}, {a}, {b}, {c0, c1};
 st.global.f64 [rp], d0; st.global.f64 [rp+8], d1;"
  imma_s4_m16n8k64 "
 .reg .b32 a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3;
 ld.global.v4.u32 {a0, a1, a2, a3}, [rp]; ld.global.v2.u32 {b0, b1}, [rp+16]; ld.global.v4.u32 {c0, c1, c2, c3}, [rp+32];
 mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};
 st.global.v4.u32 [rp], {d0, d1, d2, d3};"
  bmma_b1_and_m16n8k256 "
 .reg .b32 a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3;
 ld.global.v4.u32 {a0, a1, a2, a3}, [rp]; ld.global.v2.u32 {b0, b1}, [rp+16]; ld.global.v4.u32 {c0, c1, c2, c3}, [rp+32];
 mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};
 st.global.v4.u32 [rp], {d0, d1, d2, d3};"
  bmma_b1_xor_m16n8k256 "
 .reg .b32 a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3;
 ld.global.v4.u32 {a0, a1, a2, a3}, [rp]; ld.global.v2.u32 {b0, b1}, [rp+16]; ld.global.v4.u32 {c0, c1, c2, c3}, [rp+32];
 mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};
 st.global.v4.u32 [rp], {d0, d1, d2, d3};"
)
for t in e4m3 e5m2 e3m2 e2m3 e2m1; do
  forms+=("qmma_f8f6f4_${t}" "
 .reg .b32 a0, a1, a2, a3, b0, b1; .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;
 ld.global.v4.u32 {a0, a1, a2, a3}, [rp]; ld.global.v2.u32 {b0, b1}, [rp+16]; ld.global.v4.f32 {c0, c1, c2, c3}, [rp+32];
 mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.${t}.${t}.f32 {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};
 st.global.v4.f32 [rp], {d0, d1, d2, d3};")
done
forms+=(
  tcgen05_alloc "
 .reg .b32 sa; .shared .b32 slot;
 mov.u32 sa, slot;
 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sa], 32;
 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
  redux_add "
 .reg .b32 a, d; ld.global.u32 a, [rp];
 redux.sync.add.u32 d, a, 0xffffffff;
 st.global.u32 [rp], d;"
)
# Two words in one lane instruction: a 64-bit XOR, add or shift that one SASS instruction did would fold two words for
# one slot (the two-word premise); mad.wide.u32 is the IMAD.WIDE that does write two words.
forms+=(
  xor_b64 "
 .reg .b64 a, b, c, t, d; ld.global.b64 a, [rp]; ld.global.b64 b, [rp+8]; ld.global.b64 c, [rp+16];
 xor.b64 t, a, b; xor.b64 d, t, c;
 st.global.b64 [rp], d;"
  add_u64 "
 .reg .b64 a, b, d; ld.global.b64 a, [rp]; ld.global.b64 b, [rp+8];
 add.u64 d, a, b;
 st.global.b64 [rp], d;"
  shl_b64 "
 .reg .b64 a, d; ld.global.b64 a, [rp];
 shl.b64 d, a, 7;
 st.global.b64 [rp], d;"
  mad_wide_u32 "
 .reg .b32 a, b; .reg .b64 c, d; ld.global.u32 a, [rp]; ld.global.u32 b, [rp+4]; ld.global.b64 c, [rp+8];
 mad.wide.u32 d, a, b, c;
 st.global.b64 [rp], d;"
)

for t in "${targets[@]}"; do
  for ((i = 0; i < ${#forms[@]}; i += 2)); do
    name=${forms[i]}
    body=${forms[i + 1]}
    f=$WORK/${t}_${name}
    cat >"$f.ptx" <<EOF
.version 9.0
.target ${t}
.address_size 64
.visible .entry k(.param .u64 p)
{
 .reg .u64 rp;
 ld.param.u64 rp, [p];
${body}
 ret;
}
EOF
    msg=$("$PTXAS" -arch="$t" -O3 -o "$f.cubin" "$f.ptx" 2>&1)
    rc=$?
    ops=""
    if [ $rc -eq 0 ]; then
      "$CUOBJDUMP" -sass "$f.cubin" >"$f.sass" 2>&1
      cp "$f.sass" "$OUT/${t}_${name}.sass"
      ops=$(grep -oE '^\s+/\*[0-9a-f]{4}\*/\s+(@!?U?P[0-9T] )?[A-Z][A-Z0-9_.]*' "$f.sass" | awk '{print $NF}' |
        grep -vE '^(LDG|STG|LDC|LDCU|EXIT|BRA|NOP|ULDC|S2R|S2UR|IMAD\.MOV\.U32|MOV)' | sort | uniq -c |
        awk '{printf "%s%s=%s", (NR > 1 ? "," : ""), $2, $1}')
    fi
    python3 - "$t" "$name" "$rc" "$msg" "$ops" >>"$OUT/isa_accept.jsonl" <<'PY'
import json, sys
t, name, rc, msg, ops = sys.argv[1:]
print(json.dumps({"target": t, "form": name, "accepted": rc == "0", "ptxas": msg.strip()[:400], "sass_ops": ops}))
PY
  done
done
cat "$OUT/isa_accept.jsonl"
