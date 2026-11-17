#!/bin/bash
# payload.sh REV [--tool NAME] [--sync 'UV SYNC ARGS'] -- CMD...: writes to stdout the payload vcf-job-run.sh sends to VCF's cluster
# (SEND=FILE): `git archive` of REV under tree/, and request.sh, which names the commit and the command vcf-bench.sh runs there
# under `research run`. Run it in a checkout; nothing in the payload is a credential.
#   infra/vcf/payload.sh origin/main --tool native_peak -- python benchmarks/native_peak/measure.py --dtype bf16 --sizes 8192 \
#     | uv run --no-sync research pods ssh vy-control-verity -- 'cat > /data/vcf/payloads/native-peak.tar.gz'
set -euo pipefail
REV=${1:?REV}; shift
TOOL= SYNC="--all-packages --extra torch"
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case "$1" in
    --tool) TOOL=${2:?}; shift 2;;
    --sync) SYNC=${2:?}; shift 2;;
    *) echo "payload.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[ "${1:-}" = -- ] && [ $# -gt 1 ] || { echo "payload.sh: give the command after --" >&2; exit 2; }
shift
COMMIT=$(git rev-parse --verify "$REV^{commit}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
{
  printf 'VCF_COMMIT=%q\nVCF_TREE=%q\nVCF_TOOL=%q\nVCF_SYNC=%q\n' "$COMMIT" "$(git rev-parse "$COMMIT^{tree}")" "$TOOL" "$SYNC"
  printf 'VCF_CMD=('; printf '%q ' "$@"; printf ')\n'
} > "$T/request.sh"
git archive --format=tar --prefix=tree/ "$COMMIT" > "$T/payload.tar"
tar -rf "$T/payload.tar" -C "$T" request.sh
gzip -c "$T/payload.tar"
