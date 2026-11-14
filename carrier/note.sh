#!/bin/bash
# note.sh SLUG BODY.md: write BODY.md (its first line the note's `# ` title) as a note of the proofs lane under the notes'
# front-matter, named by the clock's reading now, and publish it with `research notes sync --path lanes/proofs`. On a
# failed sync the file stays and the caller tries again (`note.sh --sync`).
set -uo pipefail
if [ "$1" != "--sync" ]; then
  slug=$1; body=$2; id="$(date -u +%Y%m%dT%H%MZ)-report-carrier-$slug"; f=~/.research/notes/lanes/proofs/$id.md
  head -1 "$body" | grep -q '^# ' || { echo "the body's first line is the note's title, '# ...'"; exit 2; }
  { printf -- '---\nid: %s\ncampaign: TOP-394\nlane: proofs\nkind: report\nstatus: active\nrepo: compute-verification/verity\norigin: CARRIER, bc-770979f8-f27e-5abe-b9a5-a86b237c9257\n---\n\n' "$id"; cat "$body"; } > "$f"
  echo "wrote $f"
fi
cd /agent/repos/verity && timeout 170 research notes sync --path lanes/proofs 2>&1 | tail -3 | cut -c1-240; echo "sync exit ${PIPESTATUS[0]}"
