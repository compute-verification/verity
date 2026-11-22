#!/usr/bin/env bash
# The registered-reads fixture (tests/test_lean_registered_reads.py): RoPE d64 under the `RegisteredRoPE` program (its
# `cs_table`, 12 rows: the 8 instances' distinct cs rows and 4 decoys, permuted, read a whole row by each of 8 RoPEHead
# Calls), proved by the pinned upstream 967b8d06 build (UP, its `upstream-967b8d06`) over its public-output statement,
# which the tree no longer composes: the circuit and public files come from that statement's composer, verity
# 31f68948601 (inputs/src.tar.gz, `git archive 31f68948601 packages/verity/src backends/numerical/python
# backends/flock/python protocols/sampled_proofs integrations/vllm`), bound to this tree's program and partition objects
# (stage/program.json, stage/partition.json, their digests in META). Loopback serve/prove and offline replay of three
# statements (registered, fresh, afresh), then the vectors: honest, fresh-salts and afresh (their own pub.bin), and the
# honest session with the verifier's reads moved one row (wrong-position) and with none (no-registered). The fixture goes
# into the evidence store (`research data put --kind fixture/v1 --preserve`), its id in out/fixture.txt (else out/fixture).
# Run from the verity tree (research run --cwd source --send src.tar.gz [--send upstream.tar.gz]); CPU only. It exits
# nonzero unless upstream's server accepted each session.
# env: UP=art:fd8516a0… (flock-upstream-build/v1: inputs/upstream.tar.gz, else fetched from the store)  THREADS=16
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=${RESEARCH_RUN_DIR:-/tmp/registered-reads}/inputs; O=${RESEARCH_RUN_DIR:-/tmp/registered-reads}/out; mkdir -p $O
UP=${UP:-art:fd8516a05f93b00ffd242475754ef5895193b5badbbe4f2fa5fea8e74f870ee5}
UP_SHA=3e385058edbd2d04abe0fe95a85ac4c501a68206add690e833ca2cd7177789fe
T=$(mktemp -d); trap 'rm -rf $T' EXIT
mkdir -p $T/up
if [ -f $I/upstream.tar.gz ]; then cp $I/upstream.tar.gz $T/up/
else research data fetch $UP --to $T/up > $O/fetch.log 2>&1 || { tail -5 $O/fetch.log; exit 1; }
fi
for z in $(find $T/up -name '*.tar.gz'); do tar -xzf $z -C $(dirname $z) || exit 1; done
B=$(find $T/up -name upstream-967b8d06 -type f | head -1); chmod +x $B
sha256sum $B | tee $O/binary.sha256; [ "$(sha256sum $B | cut -d' ' -f1)" = $UP_SHA ] || { echo "upstream-967b8d06 is not the pinned build"; exit 1; }
mkdir -p $T/old && tar -xzf $I/src.tar.gz -C $T/old || exit 1
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
$UV venv -q --python 3.12 $T/py && $UV pip install -q --python $T/py/bin/python numpy blake3 || exit 1
OLD=$T/old/packages/verity/src:$T/old/backends/numerical/python:$T/old/backends/flock/python:$T/old/protocols/sampled_proofs:$T/old/integrations/vllm
NEW=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proof/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch
export RAYON_NUM_THREADS=${THREADS:-16}

cat > $T/old_stage.py <<'EOF'
import json, os, sys
from pathlib import Path
import numpy as np
from verity.commitments import hm96
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import coll_concat
from verity.ir.types import Array
from verity_numerical.bench import templates as TM
from verity_numerical.bench.generate import generate
from verity_numerical.bench.input_sets import InputSet
from verity_sampled_proofs.one_stage import partition as P, registered as G

N, DECOYS, VALUE = 8, 4, "cs_table"
out = Path(sys.argv[1])
s = InputSet.open(generate(TM.rope_head(64), N, 20260928, out / "set")["path"])
template = s.subcircuit.definition()
rows = {p.name: np.asarray(s.port(p.name, 0, N), dtype=np.uint16).reshape(N, -1) for p in s.subcircuit.inputs}
outs = {p.name: np.asarray(s.port(p.name, 0, N), dtype=np.uint16).reshape(N, -1) for p in s.subcircuit.outputs}
rng = np.random.default_rng(20261001)
distinct = sorted({tuple(r) for r in rows["cs"].tolist()})
table = distinct + [tuple(r) for r in rng.integers(0, 1 << 16, (DECOYS, rows["cs"].shape[1])).tolist()]
table = [table[k] for k in rng.permutation(len(table))]
pos = [table.index(tuple(r)) for r in rows["cs"].tolist()]
if sys.argv[2:3] == ["--plan"]:
    print(json.dumps({"reads": pos, "registered_rows": len(table)}))
    sys.exit(0)
# the program and partition objects are this tree's (new_stage.py): only their digests enter the old statement
cur = json.loads(Path(sys.argv[2]).read_text())
prog, part = cur["program_digest"], cur["partition_digest"]
(_, xt), (_, ct) = template.signature()[0]
ret = template.signature()[1]
def sig(S):
    return (("x", Array(N, xt)), (VALUE, Array(len(table), ct))), Array(N, ret)
def body(B, S, x, cs):
    return coll_concat(sig(S)[1], [B.call(template, x[i], cs[p]) for i, p in enumerate(pos)])
program = Program(bind(CompositeDefinition("RegisteredRoPE", 0, (), sig, body, register=False)))
query = P.template_instance_query(template)
values = {VALUE: len(table[0])}
assert G.reads(program, query, values) == {"cs": {"value": VALUE, "rows": pos}}
reg = G.Registered.commit(prog, VALUE, [list(r) for r in table])
own = G.reads_file(program, query, values, {VALUE: reg.root.hex()})
wrong = json.loads(json.dumps(own))
wrong["ports"]["cs"]["reads"] = [(r + 1) % len(table) for r in pos]
from verity_flock import circuit as FC
text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=part, program=prog, statics=FC.set_statics(s))
st = out / "stage"
st.mkdir(parents=True, exist_ok=True)
(st / "circuit.txt").write_text(text)
positions, refs = G.table(pos)
cs_rows, cs_salts, entry = reg.statement(positions)
units = list(range(N))
heads = {}
for name, salts, header in (("registered", cs_salts, entry), ("fresh", [hm96.SHA512.fresh_salt() for _ in positions], entry),
                            ("afresh", [hm96.SHA512.fresh_salt() for _ in positions], None)):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    head = FC.write(text, s, 0, N, d / f"inst-{N}.bin", d / f"pub-{N}.bin", None, units, share_rows=True, outs=outs,
                    tables={"x": rows["x"], "cs": np.asarray(cs_rows)}, refs={"x": units, "cs": refs},
                    salts={"x": [hm96.SHA512.fresh_salt() for _ in range(N)], "cs": salts},
                    registered={"cs": header} if header else None)
    heads[name] = {"shared_rows": head["shared_rows"], "registered": sorted(head.get("registered") or {})}
(out / "registered" / "own.json").write_text(json.dumps(own, indent=1))
(out / "registered" / "own-wrong-position.json").write_text(json.dumps(wrong, indent=1))
print(json.dumps({"reads": pos, "registered_rows": len(table), "root": reg.root.hex(), **cur, "circuit_sha512": FC.digest(text),
                  "statements": heads}))
EOF
cat > $T/new_stage.py <<'EOF'
import json, sys
from pathlib import Path
from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array
from verity_flock import subcircuit_templates as TM
from verity.protocols.proof.spot.verifier.one_stage import partition as P

plan, st = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
pos, n_rows = plan["reads"], plan["registered_rows"]
template = TM.rope_head(64).definition()
(_, xt), (_, ct) = template.signature()[0]
ret = template.signature()[1]
def sig(S):
    return (("x", Array(len(pos), xt)), ("cs_table", Array(n_rows, ct))), Array(len(pos), ret)
def body(B, S, x, cs):
    return coll_concat(sig(S)[1], [B.call(template, x[i], cs[p]) for i, p in enumerate(pos)])
program = Program(bind(CompositeDefinition("RegisteredRoPE", 0, (), sig, body, register=False)))
query = P.template_instance_query(template)
part = P.partition(program, query)
assert P.units(program, query) == len(pos)
st.mkdir(parents=True, exist_ok=True)
(st / "program.json").write_bytes(canonical_json(encode_program(program)))
(st / "partition.json").write_text(json.dumps(part, indent=1))
print(json.dumps({"program_digest": P.program_digest(program), "partition_digest": P.digest(part)}))
EOF
cp $T/old_stage.py $T/new_stage.py $0 $O/
PYTHONPATH=$OLD $T/py/bin/python $T/old_stage.py $T/plan-dir --plan > $O/plan.json 2> $O/plan.err || { tail -20 $O/plan.err; exit 1; }
PYTHONPATH=$NEW $T/py/bin/python $T/new_stage.py $O/plan.json $O/stage > $O/current.json 2> $O/current.err || { tail -20 $O/current.err; exit 1; }
PYTHONPATH=$OLD $T/py/bin/python $T/old_stage.py $O $O/current.json > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
cat $O/plan.json $O/current.json; rm -rf $O/set $T/plan-dir
S=$O/stage; PIN=$(sha512sum $S/circuit.txt | cut -d' ' -f1)
OK=1
run() {  # name
  local name=$1 d=$O/$1
  local port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  $B serve --listen 127.0.0.1:$port --out $O/sessions/$name --instances $d/pub-8.bin --circuit $S/circuit.txt --pin $PIN \
    --sessions 1 --operator loopback > $O/$name.serve.log 2>&1 &
  local sp=$!
  for i in $(seq 600); do grep -q SERVING $O/$name.serve.log && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  $B prove --verifier 127.0.0.1:$port --instances $d/inst-8.bin --circuit $S/circuit.txt --runs 1 > $O/$name.prove.txt 2> $O/$name.prove.err
  wait $sp
  rm -f $d/inst-8.bin
  local got=$(grep -c '"accepted":true' $O/sessions/$name/index.jsonl 2>/dev/null)
  echo "$name: $got of 1 accepted by upstream's server"; [ "$got" = 1 ] || OK=0
  $B replay --instances $d/pub-8.bin --circuit $S/circuit.txt --sessions $O/sessions/$name/l* > $O/$name.replay.txt 2>&1
  grep UPSTREAM $O/$name.replay.txt
}
run registered
run fresh
run afresh
V=$O/vectors; mkdir -p $V
for v in honest:registered fresh-salts:fresh afresh:afresh wrong-position:registered no-registered:registered; do
  IFS=: read -r name from <<<"$v"
  mkdir -p $V/$name && cp $O/sessions/$from/l*/* $V/$name/
done
cp $O/fresh/pub-8.bin $V/fresh-salts/pub.bin
cp $O/afresh/pub-8.bin $V/afresh/pub.bin
cp $O/registered/own-wrong-position.json $V/wrong-position/registered.json
echo null > $V/no-registered/registered.json
find $O -name 'inst-*.bin' -delete
[ $OK = 1 ] || exit 1
X=$T/fixture; mkdir -p $X
cp -r $O/stage $O/registered $O/fresh $O/afresh $O/vectors $O/stage.json $O/binary.sha256 $O/*.py $O/*.sh $X/ || exit 1
RESEARCH=$(command -v research || true)
if [ -n "$RESEARCH" ]; then
  $RESEARCH data put --kind fixture/v1 --tree $X --preserve > $O/fixture.txt 2> $O/fixture.err || tail -5 $O/fixture.err
fi
[ -s $O/fixture.txt ] || { mkdir -p $O/fixture && cp -r $X/. $O/fixture/; echo "not in the store (no research CLI, or fixture.err): the fixture is out/fixture" > $O/fixture.txt; }
cat $O/fixture.txt
