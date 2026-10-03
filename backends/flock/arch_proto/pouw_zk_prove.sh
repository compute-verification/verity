#!/usr/bin/env bash
# One `--zk` session of a staged PoUW unit (class_statement.stage_unit's directory: circuit.txt, inst.bin, pub.bin), proved on
# the GPU and verified offline by the Lean verifier with the verifier's own public input words (TileAnchors.public_inputs):
# builds or finds flock-circuit (70-class-sweep.sh BUILD_ONLY=1) and flock-verify (a copy of verifier/lean under FLOCK_WORK,
# keyed by its tree), serves the session (`serve --zk`, which reads no --public-inputs), proves it (`prove --zk --gpu`), then
# `flock-verify verify --zk --public-inputs` on the recorded session, and again with one anchor bit flipped (must refuse).
# Writes out/summary.json, out/prove.{out,err}, out/serve.log, out/verify*.txt and out/session/ under $RESEARCH_RUN_DIR.
# env: STAGED=dir PUBLIC_INPUTS=file (required)  FLOCK_WORK=/workspace/jobs/proofs-arch  SM=120  PORT=7720  CPUS=16
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
: "${STAGED:?STAGED= is required}" "${PUBLIC_INPUTS:?PUBLIC_INPUTS= is required}"
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/pouw-zk-prove}; O=$R/out; mkdir -p $O
for f in circuit.txt inst.bin pub.bin; do [ -s $STAGED/$f ] || { echo "$STAGED/$f is missing"; exit 1; }; done
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-arch}; mkdir -p $FLOCK_WORK/home
export HOME=$FLOCK_WORK/home
unset CARGO_TARGET_DIR
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
bins=$(OUT=$O/build bash backends/flock/pod/70-class-sweep.sh BUILD_ONLY=1 SM=${SM:-120} 2> $O/build.err) || { echo "$bins"; tail -30 $O/build.err; exit 1; }
B=$(echo "$bins" | sed -n 's/^BINARY=//p')
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}

LT=$(git -C $REPO rev-parse HEAD:backends/flock/verifier/lean 2>/dev/null || tar -C backends/flock/verifier/lean --exclude=.lake -cf - . | sha256sum | cut -c1-16)
L=$FLOCK_WORK/lean-verifier/$LT
if [ ! -x $L/.lake/build/bin/flock-verify ]; then
  rm -rf $L.tmp && mkdir -p $L.tmp && tar -C backends/flock/verifier/lean --exclude=.lake -cf - . | tar -C $L.tmp -xf - || exit 1
  command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || \
    { curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }
  export PATH="$HOME/.elan/bin:$PATH"
  (cd $L.tmp && lake build) > $O/lean-build.log 2>&1 || { tail -30 $O/lean-build.log; exit 1; }
  rm -rf $L && mv $L.tmp $L
fi
V=$L/.lake/build/bin/flock-verify
sha256sum $B $V | tee $O/binaries.sha256
{ nvidia-smi -L; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-}"; env | grep -E '^(FC_|FLOCK_GEMM_TILE=)' | sort; } > $O/host.txt 2>&1
sha256sum $STAGED/circuit.txt $STAGED/inst.bin $STAGED/pub.bin $PUBLIC_INPUTS > $O/inputs.sha256

$B statement --zk --instances $STAGED/pub.bin --circuit $STAGED/circuit.txt --tables $STAGED/tables > $O/statement.txt 2>&1
S=$(mktemp -d); trap 'kill $SP 2>/dev/null; rm -rf $S' EXIT
PIN=$(sha512sum $STAGED/circuit.txt | cut -d' ' -f1)
T=$STAGED/tables
$B serve --zk --listen 127.0.0.1:${PORT:-7720} --out $S/sessions --instances $STAGED/pub.bin --circuit $STAGED/circuit.txt --tables $T \
  --pin $PIN --operator loopback > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q '^SERVING' $O/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
ADDR=$(sed -n 's/^SERVING \([^ ]*\).*/\1/p' $O/serve.log | head -1)
[ -n "$ADDR" ] || { echo "serve never listened"; tail -20 $O/serve.log; exit 1; }
/usr/bin/time -f %M -o $O/prove.maxrss_kb $B prove --zk --gpu --verifier $ADDR --instances $STAGED/inst.bin --circuit $STAGED/circuit.txt \
  --tables $T --warm 0 --runs 1 > $O/prove.out 2> $O/prove.err
PRC=$?
sleep 2; kill $SP 2>/dev/null; wait $SP 2>/dev/null
SESS=$(find $S/sessions -mindepth 1 -maxdepth 1 -type d | sort | head -1)
[ -n "$SESS" ] && cp -r $SESS $O/session

vfy() { $V verify --statement verity/flock-circuit --circuit $STAGED/circuit.txt --public $STAGED/pub.bin --public-inputs "$1" \
          --zk --session-tables 1 --session $O/session; }
vfy $PUBLIC_INPUTS > $O/verify.txt 2>&1; VRC=$?
python3 -c "import sys; b=bytearray(open(sys.argv[1],'rb').read()); b[-1]^=1; open(sys.argv[2],'wb').write(b)" $PUBLIC_INPUTS $S/flipped.bin
vfy $S/flipped.bin > $O/verify-flipped.txt 2>&1; FRC=$?

python3 - "$O" "$PRC" "$VRC" "$FRC" <<'EOF'
import json, sys
from pathlib import Path
o, prc, vrc, frc = Path(sys.argv[1]), *map(int, sys.argv[2:])
def lines(f, tag):
    return [json.loads(l.split(None, 1)[1]) for l in (o / f).read_text().splitlines() if l.startswith(tag)]
live = lines("prove.out", "LIVE")
v, vf = lines("verify.txt", "VERDICT"), lines("verify-flipped.txt", "VERDICT")
L = live[0] if live else {}
s = {"prove_rc": prc, "live_accepted": L.get("accepted"), "live_why": L.get("why"),
     "lean_accepted": bool(v and v[0].get("accepted")), "lean_why": v[0].get("why") if v else (o / "verify.txt").read_text()[-400:], "lean_rc": vrc,
     "flipped_refused": not (vf and vf[0].get("accepted")), "flipped_why": vf[0].get("why") if vf else (o / "verify-flipped.txt").read_text()[-400:],
     "prover_peak_gb": round(int((o / "prove.maxrss_kb").read_text().split()[-1]) / 2**20, 2) if (o / "prove.maxrss_kb").exists() else None}
s |= {k: L.get(k) for k in ("instances", "vus_per_block", "k_log", "nbl", "dense_m", "prove_s", "prove_total_s", "e2e_s", "witness_s", "proof_bytes", "statement_digest")}
st = [json.loads(l.split("\t", 1)[1]) for l in (o / "statement.txt").read_text().splitlines() if l.startswith("STATEMENT\t")]
s["statement"] = st[0] if st else None
s["verdict"] = "ACCEPT" if (prc == 0 and s["live_accepted"] and s["lean_accepted"] and s["flipped_refused"]) else "FAIL"
(o / "summary.json").write_text(json.dumps(s, indent=1))
print("SUMMARY", json.dumps(s))
EOF
