#!/usr/bin/env bash
# Sampled proofs' parties end to end (core's, on the toy's machine: verity_all_protocols.parties), every party on this
# machine: their configuration (tests/fixtures/sampled-proofs.toml) honest, tampered (--tamper 0) and with the certifier
# listed but no certifier party, all in one process (--local); then honest again with each party in a process of its own,
# meeting over TCP (apart: the challenger, firewall and workers listen, the verifier dials them). Each block's VERDICT
# line is in $OUT/<case>.log and its public record in $OUT/<case>/, then the opt-in test that asserts the first three
# (test_parties). The states, with the prover's rows and salts, stay in scratch.
#   research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 96 -- bash benchmarks/all_protocols/parties.sh
set -uo pipefail
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > $O/fetch-main.log 2>&1 \
  || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> $O/fetch-main.log 2>&1 \
  || { tail -5 $O/fetch-main.log; exit 4; }
SECONDS=0; bash benchmarks/all_protocols/build.sh $W/build || exit $?
cp $W/build/binaries.sha256 $W/build/tools.json $O/; echo "build ${SECONDS}s" | tee $O/seconds.txt
PY="env PYTHONPATH=$REPO/benchmarks/all_protocols:$REPO/integrations/torch_loop $REPO/.venv/bin/python"
CONFIG=benchmarks/all_protocols/tests/fixtures/sampled-proofs.toml
sed 's/^protocols = \["sampled-proofs"\]$/protocols = ["sampled-proofs", "certifier"]/' $CONFIG > $W/both.toml
grep -q '"certifier"' $W/both.toml || exit 3
for c in honest:$CONFIG: tampered:$CONFIG:0 absent:$W/both.toml:; do
  IFS=: read -r name config tamper <<< "$c"
  SECONDS=0
  $PY -m verity_all_protocols.parties $config --state $W/$name --local --tools $W/build/tools.json \
    ${tamper:+--tamper $tamper} > $O/$name.log 2> $O/$name.err
  echo "$name rc $? ${SECONDS}s" | tee -a $O/seconds.txt
  mkdir -p $O/$name && cp -r $W/$name/record $W/$name/public $W/$name/firewall $O/$name/ 2>/dev/null
  (cd $W/$name && find runs -name record -path '*/runs/*' | while read -r r; do mkdir -p $O/$name/$(dirname $r); cp $r $O/$name/$r; done)
  grep '^VERDICT ' $O/$name.log
done
# apart: three listening parties started first, each printing nothing until the verifier dials; a free port each
SECONDS=0; A=$W/apart; mkdir -p $A
AT=(); for role in challenger firewall workers; do
  AT+=(--at $role=127.0.0.1:$($REPO/.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'))
done
PIDS=(); for role in challenger firewall workers; do
  $PY -m verity_all_protocols.parties $CONFIG --state $A --party $role "${AT[@]}" --tools $W/build/tools.json \
    > $O/apart-$role.log 2> $O/apart-$role.err & PIDS+=($!)
done
for role in challenger firewall workers; do
  port=$(printf '%s\n' "${AT[@]}" | sed -n "s/^$role=127.0.0.1://p")
  # listening (state 0A in /proc/net/tcp), read without connecting: a listener takes one connection, the verifier's
  until grep -qi ":$(printf %04X $port) 00000000:0000 0A" /proc/net/tcp; do sleep 0.2; done
done
$PY -m verity_all_protocols.parties $CONFIG --state $A --party verifier "${AT[@]}" --tools $W/build/tools.json \
  > $O/apart.log 2> $O/apart.err
rc=$?; for p in "${PIDS[@]}"; do wait $p; echo "apart party $p rc $?" | tee -a $O/seconds.txt; done
echo "apart rc $rc ${SECONDS}s" | tee -a $O/seconds.txt
mkdir -p $O/apart && cp -r $A/record $A/public $A/firewall $O/apart/ 2>/dev/null
(cd $A && find runs -name record -path '*/runs/*' | while read -r r; do mkdir -p $O/apart/$(dirname $r); cp $r $O/apart/$r; done)
grep '^VERDICT ' $O/apart.log
SECONDS=0
(cd benchmarks/all_protocols && VERITY_ALL_PROTOCOLS_TOOLS=$W/build/tools.json $REPO/.venv/bin/python -m pytest -q \
  tests/test_parties.py -p no:cacheprovider --basetemp $W/pytest) > $O/pytest.log 2>&1
rc=$?; echo "pytest rc $rc ${SECONDS}s" | tee -a $O/seconds.txt; tail -5 $O/pytest.log
exit $rc
