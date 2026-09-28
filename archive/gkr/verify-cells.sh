#!/usr/bin/env bash
# Non-producer file re-verification of A-route-a cells: flock-link and verity-gkr-verify built from this tree, the full
# statement of every recorded size rebuilt here from the captured set (tools/cell.py statement: leaf digests, commitment,
# circuit files, public words, sigma), checked against the verifier pod's digests and the prover's statement files, then
# tools/cell_gate.py on every recorded session (Flock replay from the record + the prime proof against the record's link
# and prime coins, both on this statement), plus tampered-record negatives.  CPU only.
#   per cell TAG: inputs/set-TAG.tgz  inputs/ver-TAG.tgz (verifier records)  inputs/prf-TAG.tgz (cell-N/sI/proof.bin)
#                 inputs/stmt-TAG.sha256 (the prover's statement-N files)   CELLS="TAG:REL:NEG_SIZE ..."  XFAIL="TAG:N ..."
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); RD=${RESEARCH_RUN_DIR:?}; W=/workspace/verify-agkr; O=$RD/out; mkdir -p $O $W/bin
say() { echo "##### $(date -u +%H:%M:%SZ) $*"; }
Q=-1; PER=100000
if [ -r /sys/fs/cgroup/cpu.max ]; then read -r Q PER < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1; fi
NT=$(( Q > 0 ? Q / PER : $(nproc) )); [ "$NT" -lt 1 ] && NT=1; export RAYON_NUM_THREADS=$NT
export PYTHONPATH="$REPO:$REPO/benchmarks/numerical/python:$REPO/tools/research/src:$REPO:$REPO/archive/gkr:$REPO/archive/gkr/tools"
F=/workspace/flock-link/flock; B=$W/bin/flock-link; VB=$W/bin/verity-gkr-verify
say "build flock-link + verity-gkr-verify ($NT threads)"
bash verity/ml/flock/pod/00-setup-cpu.sh > $O/flock-setup.log 2>&1
source $HOME/.cargo/env; export PATH="$HOME/.cargo/bin:$PATH"
( cd $F && git checkout -q b684b12 && git checkout -q -- . && git clean -qfd crates && git apply $REPO/verity/ml/flock/flock-link-b684b12.patch \
  && git apply $REPO/verity/ml/flock/flock-gpu-link-b684b12.patch && cp -r $REPO/verity/ml/flock/live crates/flock-live \
  && { grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml; } \
  && cargo build --release -p flock-live --bin flock-link -j $NT 2>&1 | grep -E '^error|Finished' | head -5 ) && cp $F/target/release/flock-link $B || { echo BUILD FAILED; exit 1; }
( cd archive/gkr/verifier && cargo build --release -j $NT 2>&1 | grep -E '^error|Finished' | head -3 && cp target/release/verity-gkr-verify $VB \
  && cargo test --release -j $NT 2>&1 | grep -E 'test result|FAILED' ) | tee $O/verifier-build.txt
[ -x $VB ] || { echo "verity-gkr-verify build failed"; exit 1; }
sha256sum $B $VB | tee $O/binaries.sha256
PY=$W/venv/bin/python
if ! $PY -c "import numpy, blake3" 2>/dev/null; then
  command -v uv >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"; uv venv $W/venv --python 3.12 >/dev/null 2>&1; uv pip install --python $PY numpy blake3 2>&1 | tail -1
fi
bad=0
for cell in $CELLS; do
  IFS=: read -r TAG REL NEGN <<<"$cell"; C=$O/$TAG; mkdir -p $C; X=$W/$TAG; rm -rf $X; mkdir -p $X/set $X/ver $X/prf
  tar -xzf $RD/inputs/set-$TAG.tgz -C $X/set; tar -xzf $RD/inputs/ver-$TAG.tgz -C $X/ver; tar -xzf $RD/inputs/prf-$TAG.tgz -C $X/prf
  SET=$(dirname "$(find $X/set -name manifest.json | head -1)")
  K=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]+'/manifest.json'))['relation']['statics']['K'])" "$SET")
  $B selftest --vus 8 --k $K > $C/selftest-8.txt 2>&1
  echo "$TAG flock-link selftest (8 VUs, K=$K): $(grep -c '"pass":true' $C/selftest-8.txt) passed, $(grep -c '"pass":false' $C/selftest-8.txt) failed" | tee -a $C/summary.txt
  : > $C/gates.jsonl
  for sd in $(ls -d $X/ver/sessions-* | sort -t- -k2 -n); do
    n=${sd##*sessions-}; M=$X/stmt-$n
    say "$TAG: statement $n VUs (K=$K)"
    $PY archive/gkr/tools/cell.py statement --cell $M --input-set $SET --vus $n --flock $B > $C/statement-$n.log 2>&1 || { tail -3 $C/statement-$n.log; bad=$((bad+1)); continue; }
    want=$(grep "c$n/leaf_digests" $X/ver/out/statements/digests.sha256 | cut -c1-64); got=$(sha256sum $M/leaf_digests.bin | cut -c1-64)
    echo "$TAG $n digests mine $got verifier's ${want:-none} $([ "$got" = "$want" ] && echo EQUAL || echo DIFFER)" | tee -a $C/summary.txt
    [ "$got" = "$want" ] || bad=$((bad+1))
    ( cd $M/statement && for f in *; do echo "$(sha256sum $f | cut -c1-64)  statement-$n/$f"; done; cd $M && for f in cell.json leaf_digests.bin; do echo "$(sha256sum $f | cut -c1-64)  statement-$n/$f"; done ) | sort -k2 > $C/mine-$n.sha256
    grep " statement-$n/" $RD/inputs/stmt-$TAG.sha256 | sort -k2 > $C/prover-$n.sha256
    d=$(join -1 2 -2 2 $C/mine-$n.sha256 $C/prover-$n.sha256 | awk '$2 != $3 {print $1}' | grep -v 'cell.json' | tr '\n' ' ')
    echo "$TAG $n statement files vs the prover's: $(join -1 2 -2 2 $C/mine-$n.sha256 $C/prover-$n.sha256 | wc -l) compared, differing (cell.json's timing excluded): ${d:-none}" | tee -a $C/summary.txt
    [ -z "$d" ] || bad=$((bad+1))
    PC=$($PY -c "import json;print(json.load(open('$M/cell.json'))['prime_commitment'])")
    i=1
    for ses in $(ls -d $sd/l* | sort); do
      o=$C/g-$n-s$i; mkdir -p $o
      $PY archive/gkr/tools/cell_gate.py --session $ses --sigma $M/statement/sigma.txt --producer agkr-real-k --producer agkr-route-a \
        --producer loopback-selftest --out $o --rust $VB --statement $M/statement --proof $X/prf/out/cell-$n/s$i/proof.bin --vus $n \
        --relation $REL --threads $NT --flock $B --digests $M/leaf_digests.bin --prime-commitment $PC > $o/gate.txt 2>&1
      $PY - $o/gate.json "$n" "$i" "$(basename $ses)" <<'EOF' | tee -a $C/gates.jsonl
import json, sys
d = json.load(open(sys.argv[1])); c = d["checks"]
fails = sorted(k for k, v in c.items() if not v["ok"])
print(json.dumps({"vus": int(sys.argv[2]), "s": int(sys.argv[3]), "session": sys.argv[4], "fails": fails,
                  "verified": fails in ([], ["non_producer"]), "prime": c.get("prime", {}).get("detail", "")[:160],
                  "flock_replay": c.get("flock_replay", {}).get("detail", "")[:160], "non_producer": c["non_producer"]["detail"][:120]}))
EOF
      find $o -name '*.bin' -delete
      i=$((i + 1))
    done
  done
  echo "$TAG gates: $(grep -c '"verified": true' $C/gates.jsonl) of $(wc -l < $C/gates.jsonl) sessions verified (every check but non_producer)" | tee -a $C/summary.txt
  # negatives on the first session of NEGN: each must fail a check other than non_producer
  M=$X/stmt-$NEGN; sd=$X/ver/sessions-$NEGN; ses=$(ls -d $sd/l* | sort | head -1); PC=$($PY -c "import json;print(json.load(open('$M/cell.json'))['prime_commitment'])")
  OTHERN=$(ls -d $X/stmt-* | grep -v "stmt-$NEGN$" | head -1); OREL=$([ "$REL" = bf16-ampere+blake3 ] && echo bf16-ampere-k2048+blake3 || echo bf16-ampere+blake3)
  T=$X/neg; mkdir -p $T; : > $C/negatives.jsonl
  $PY - $ses $T <<'EOF'
import copy, json, shutil, sys
from pathlib import Path
src, t = Path(sys.argv[1]), Path(sys.argv[2])
rec = json.loads((src / "session.json").read_text())
P = 2013265921
def flip(s):
    s = str(s)
    return s[:-1] + format(int(s[-1], 16) ^ 1, "x") if all(ch in "0123456789abcdef" for ch in s.lower()) else str(int(s) + 1)
def case(name, f):
    d = t / name; shutil.copytree(src, d, dirs_exist_ok=True); r = copy.deepcopy(rec); f(r); (d / "session.json").write_text(json.dumps(r))
def prime_word(r):
    rd = r["prime"]["rounds"][len(r["prime"]["rounds"]) // 2]; rd["words"][0] = (int(rd["words"][0]) + 1) % P
def point(r):
    r["link"]["points"][0][0] = flip(r["link"]["points"][0][0])
def flock_round(r):
    r["streams"][0]["rounds"][0]["msg_sha256"] = "00" * 32
def ungated(r):
    r["config"]["require_link"] = False
for name, f in [("prime_coin_word_changed", prime_word), ("link_point_changed", point), ("flock_round_digest_changed", flock_round),
                ("require_link_false", ungated)]:
    case(name, f)
EOF
  neg() { local name=$1 s=$2 st=$3 proof=$4 rel=$5 o=$T/out-$1; mkdir -p $o
    $PY archive/gkr/tools/cell_gate.py --session $s --sigma $st/statement/sigma.txt --producer agkr-real-k --producer agkr-route-a \
      --producer loopback-selftest --out $o --rust $VB --statement $st/statement --proof $proof --vus $(python3 -c "import json;print(json.load(open('$st/cell.json'))['vus'])") \
      --relation $rel --threads $NT --flock $B --digests $st/leaf_digests.bin --prime-commitment $($PY -c "import json;print(json.load(open('$st/cell.json'))['prime_commitment'])") > $o/gate.txt 2>&1
    $PY - $o/gate.json $name <<'EOF' | tee -a $C/negatives.jsonl
import json, sys
c = json.load(open(sys.argv[1]))["checks"]; fails = sorted(k for k, v in c.items() if not v["ok"] and k != "non_producer")
print(json.dumps({"neg": sys.argv[2], "rejected": bool(fails), "fails": fails}))
EOF
    find $o -name '*.bin' -delete; }
  P1=$X/prf/out/cell-$NEGN/s1/proof.bin; P2=$X/prf/out/cell-$NEGN/s2/proof.bin
  for n in prime_coin_word_changed link_point_changed flock_round_digest_changed require_link_false; do neg $n $T/$n $M $P1 $REL; done
  neg proof_from_other_session $ses $M $P2 $REL
  neg claimed_as_other_relation $ses $M $P1 $OREL
  [ -n "$OTHERN" ] && neg other_size_statement $ses $OTHERN $P1 $REL
  nr=$(grep -c '"rejected": true' $C/negatives.jsonl); nt=$(wc -l < $C/negatives.jsonl)
  echo "$TAG negatives: $nr of $nt rejected" | tee -a $C/summary.txt; [ $nr = $nt ] || bad=$((bad+1))
  # XFAIL="TAG:N ...": sizes past the cell whose prover run is known broken (reported, not counted)
  ok=$($PY -c "import json,sys;x={int(v.split(':')[1]) for v in sys.argv[2].split() if v.split(':')[0]==sys.argv[3]};print(all(r['verified'] for r in map(json.loads,open(sys.argv[1])) if r['vus'] not in x))" $C/gates.jsonl "${XFAIL:-}" $TAG)
  [ "$ok" = True ] || bad=$((bad+1))
  rm -rf $X/prf
done
echo "TOTAL bad $bad" | tee $O/summary.txt
[ $bad = 0 ]
