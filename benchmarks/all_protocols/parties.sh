#!/usr/bin/env bash
# The sampled-proofs roles end to end (verity_all_protocols.parties), every role on this machine: the toy's example
# configuration honest, tampered (--tamper 0) and with the certifier listed but no certifier role, each block's VERDICT
# line in $OUT/<case>.log and its public record in $OUT/<case>/, then the opt-in test that asserts them (test_parties).
# The states, with the prover's rows and salts, stay in scratch.
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
CONFIG=examples/toy_network/config.toml
sed 's/^protocols = \["sampled-proofs"\]$/protocols = ["sampled-proofs", "certifier"]/' $CONFIG > $W/both.toml
grep -q '"certifier"' $W/both.toml || exit 3
for c in honest:$CONFIG: tampered:$CONFIG:0 absent:$W/both.toml:; do
  IFS=: read -r name config tamper <<< "$c"
  SECONDS=0
  $PY -m verity_all_protocols.parties $config --state $W/$name --local --tools $W/build/tools.json \
    ${tamper:+--tamper $tamper} > $O/$name.log 2> $O/$name.err
  echo "$name rc $? ${SECONDS}s" | tee -a $O/seconds.txt
  mkdir -p $O/$name && cp -r $W/$name/record $W/$name/public $W/$name/firewall $O/$name/ 2>/dev/null
  cp $W/$name/challenger/address.json $O/$name/ 2>/dev/null
  grep '^VERDICT ' $O/$name.log
done
SECONDS=0
(cd benchmarks/all_protocols && VERITY_ALL_PROTOCOLS_TOOLS=$W/build/tools.json $REPO/.venv/bin/python -m pytest -q \
  tests/test_parties.py -p no:cacheprovider --basetemp $W/pytest) > $O/pytest.log 2>&1
rc=$?; echo "pytest rc $rc ${SECONDS}s" | tee -a $O/seconds.txt; tail -5 $O/pytest.log
exit $rc
