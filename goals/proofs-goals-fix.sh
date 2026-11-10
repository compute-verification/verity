#!/usr/bin/env bash
# proofs' Goals page fix for CON-12: one `research goals set` line for each of proofs' 24 open goals.
# Written by HOUSEKEEP and NOT RUN. Read proofs-goals-table.md first: each ASK there names the line it changes.
#   bash proofs-goals-fix.sh          run every line; a refused line is reported and the rest still run
#   DRY=1 bash proofs-goals-fix.sh    print the commands and send nothing
# Every line passes --description, and --blockers or --no-blockers: the site checks a goal's whole text on any
# write, and the texts on the page today predate its rules. `--wait` replaces a goal's whole list of links.
# It sets no priority and no tier.
set -uo pipefail
cd "$(dirname "$0")"
failed=()
g() { if [ -n "${DRY:-}" ]; then printf "research goals set"; printf " %q" "$@"; echo; return; fi
      research goals set "$@" || failed+=("$1"); }

# ASK 1: see the table.
g proofs/slice-clean --status off-track --due 2026-11-13T18:00-08:00 --tag cleanup --tag proofs --no-wait \
  --blockers @blockers/slice-clean.md --description @desc/slice-clean.md

g proofs/spot-bind-term --status on-track --due 2026-11-13T18:00-08:00 --wait goal:ci/fine-grained-check \
  --no-blockers --description @desc/spot-bind-term.md

g proofs/spot-checks-closed --title "Guarantee checks closed or scoped" --status on-track --no-wait --no-blockers \
  --description @desc/spot-checks-closed.md

# ASK 2: see the table.
g proofs/spot-numbers-lean-checked --status on-track --wait goal:ci/fine-grained-check --no-blockers \
  --description @desc/spot-numbers-lean-checked.md

# ASK 3: see the table.
g proofs/spot-fused-cut-served --status off-track --no-wait --blockers @blockers/spot-fused-cut-served.md \
  --description @desc/spot-fused-cut-served.md

# ASK 4: see the table.
g proofs/spot-128k-attention-proved --status off-track --wait goal:circuits/glm-sparse-attention-zk \
  --wait goal:circuits/glm-long-dsa-row --wait goal:vllm/served-128k-window-committed --no-blockers \
  --description @desc/spot-128k-attention-proved.md

# ASK 5: see the table.
g proofs/spot-advice-certified --status off-track --no-wait --blockers @blockers/spot-advice-certified.md \
  --description @desc/spot-advice-certified.md

# ASK 6: see the table.
g proofs/spot-moe-binding --status off-track --no-wait --blockers @blockers/spot-moe-binding.md \
  --description @desc/spot-moe-binding.md

g proofs/spot-one-stage-premises --status on-track --wait goal:ci/fine-grained-check --no-blockers \
  --description @desc/spot-one-stage-premises.md

g proofs/spot-overhead-measured --title "Proof overhead measured on GLM-5.3's served blocks" --status off-track \
  --tag proofs --wait goal:circuits/glm-served-calls-fit-block --no-blockers \
  --description @desc/spot-overhead-measured.md

# ASK 7: see the table.
g proofs/spot-proof-server-steady-state --status off-track --wait goal:circuits/glm-served-calls-fit-block \
  --no-blockers --description @desc/spot-proof-server-steady-state.md

g proofs/spot-ten-percent --status off-track --no-blockers --description @desc/spot-ten-percent.md

# ASK 8: see the table.
g proofs/spot-two-stage-e2e --status off-track --blockers @blockers/spot-two-stage-e2e.md \
  --description @desc/spot-two-stage-e2e.md

# ASK 9: see the table.
g proofs/spot-verifier-cost --title "Verifier cost tracked and driven down" --status off-track \
  --wait goal:circuits/verifier-templates-not-unrolled --no-blockers --description @desc/spot-verifier-cost.md

# ASK 10: see the table.
g proofs/terminology-clean --title "Retired terms gone, and new uses refused" --status off-track --no-wait \
  --blockers @blockers/terminology-clean.md --description @desc/terminology-clean.md

# ASK 11: see the table.
g proofs/spot-figures --status off-track --tag proofs --wait goal:proofs/spot-ten-percent --no-blockers \
  --description @desc/spot-figures.md

g proofs/spot-claims-checked --title "Every claim in the post traced to evidence" --status on-track --tag proofs \
  --no-blockers --description @desc/spot-claims-checked.md

g proofs/spot-output-accounting --status off-track --wait goal:network/cap-certified --no-blockers \
  --description @desc/spot-output-accounting.md

g proofs/spot-post-ready --status off-track --wait goal:proofs/spot-ten-percent \
  --wait goal:proofs/spot-128k-attention-proved --wait goal:proofs/spot-two-stage-e2e \
  --wait goal:proofs/spot-bind-term --wait goal:proofs/weight-security-bridge \
  --wait goal:proofs/spot-one-stage-premises --wait goal:proofs/spot-moe-binding \
  --wait goal:proofs/spot-output-accounting --wait goal:proofs/spot-advice-certified \
  --wait goal:proofs/spot-checks-closed --wait goal:proofs/spot-numbers-lean-checked --wait goal:proofs/spot-figures \
  --wait goal:proofs/spot-claims-checked --wait goal:circuits/glm-compile-cost-measured \
  --wait goal:proofs/spot-glm-replay --wait goal:proofs/spot-architecture-matches --no-blockers \
  --description @desc/spot-post-ready.md

g proofs/weight-security-bridge --status on-track --wait goal:ci/fine-grained-check --no-blockers \
  --description @desc/weight-security-bridge.md

g proofs/zero-knowledge --status off-track --due 2026-11-30T18:00-08:00 --blockers @blockers/zero-knowledge.md \
  --description @desc/zero-knowledge.md

# ASK 12: see the table.
g proofs/zk-leaks-closed --status off-track --blockers @blockers/zk-leaks-closed.md \
  --description @desc/zk-leaks-closed.md

# ASK 13 (PRF-363, yours for Wed): see the table.
g proofs/sampling-generic --status on-track --wait goal:proofs/spot-one-stage-premises \
  --wait goal:proofs/spot-moe-binding --wait goal:proofs/spot-two-stage-sound --no-blockers \
  --description @desc/sampling-generic.md

# ASK 14: see the table.
g proofs/spot-two-stage-sound --status on-track --no-blockers --description @desc/spot-two-stage-sound.md

# ASK 14, a new goal, commented out. If you want it, uncomment it and swap it into proofs/spot-post-ready's
# links in place of goal:proofs/spot-architecture-matches (16 links is the cap).
# g proofs/spot-two-stage-not-vacuous --title "Two-stage guarantee shown not to be vacuous" --owner proofs \
#   --status on-track --due 2026-11-13T18:00-08:00 --tag proofs --description @desc/new-spot-two-stage-not-vacuous.md

if [ "${#failed[@]}" -gt 0 ]; then echo "REFUSED or failed: ${failed[*]}" >&2; exit 1; fi
[ -n "${DRY:-}" ] || echo "all 24 written; check with: research goals show"
