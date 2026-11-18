#!/bin/bash
# payload.sh REV [--tool NAME] [--sync 'UV SYNC ARGS'] [--run-id ID] [--project P] [--machine NAME [--properties FILE]]
#   [--timeout-min M] [--env K=V]... -- CMD...: writes to stdout the payload vcf-job-run.sh sends to VCF's cluster
# (SEND=FILE): `git archive` of REV under tree/, and request.sh, which names the commit and the command vcf-bench.sh runs there
# under `research run` (as run ID of project P, on machine NAME, whose [properties] copy FILE goes in as machines.d/NAME.toml, for M minutes,
# with each --env). `research run --on NAME` for a slurm machine builds it with these; run it in a checkout; nothing in the payload
# is a credential.
#   infra/vcf/payload.sh origin/main --tool native_peak -- python calibration/native_peak/measure.py --dtype bf16 --sizes 8192 \
#     | uv run --no-sync research pods ssh vy-control-verity -- 'cat > /data/vcf/payloads/native-peak.tar.gz'
set -euo pipefail
REV=${1:?REV}; shift
TOOL= SYNC="--all-packages --extra torch" RUN_ID= PROJECT= MACHINE= PROPS= TMIN= ENVS=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case "$1" in
    --tool) TOOL=${2:?}; shift 2;;
    --sync) SYNC=${2:?}; shift 2;;
    --run-id) RUN_ID=${2:?}; shift 2;;
    --project) PROJECT=${2:?}; shift 2;;
    --machine) MACHINE=${2:?}; shift 2;;
    --properties) PROPS=${2:?}; shift 2;;
    --timeout-min) TMIN=${2:?}; shift 2;;
    --env) ENVS+=("${2:?}"); shift 2;;
    *) echo "payload.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[ "${1:-}" = -- ] && [ $# -gt 1 ] || { echo "payload.sh: give the command after --" >&2; exit 2; }
[ -z "$PROPS" ] || [ -n "$MACHINE" ] || { echo "payload.sh: --properties is the copy of --machine NAME's table" >&2; exit 2; }
shift
COMMIT=$(git rev-parse --verify "$REV^{commit}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
{
  printf 'VCF_COMMIT=%q\nVCF_TREE=%q\nVCF_TOOL=%q\nVCF_SYNC=%q\n' "$COMMIT" "$(git rev-parse "$COMMIT^{tree}")" "$TOOL" "$SYNC"
  printf 'VCF_RUN_ID=%q\nVCF_PROJECT=%q\nVCF_MACHINE=%q\n' "$RUN_ID" "$PROJECT" "$MACHINE"
  [ -z "$TMIN" ] || printf 'VCF_TIMEOUT_MIN=%q\n' "$TMIN"
  printf 'VCF_ENV=('; [ ${#ENVS[@]} = 0 ] || printf '%q ' "${ENVS[@]}"; printf ')\n'
  printf 'VCF_CMD=('; printf '%q ' "$@"; printf ')\n'
} > "$T/request.sh"
git archive --format=tar --prefix=tree/ "$COMMIT" > "$T/payload.tar"
tar -rf "$T/payload.tar" -C "$T" request.sh
if [ -n "$PROPS" ]; then
  mkdir "$T/machines.d" && cp "$PROPS" "$T/machines.d/$MACHINE.toml" && tar -rf "$T/payload.tar" -C "$T" "machines.d/$MACHINE.toml"
fi
gzip -c "$T/payload.tar"
