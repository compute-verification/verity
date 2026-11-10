#!/usr/bin/env bash
# proofs' Goals page fix for CON-12: one `research goals set` line a goal, 24 lines: proofs' 24 open goals and one
# new goal. Written by HOUSEKEEP and NOT RUN. Read proofs-goals-table.md first.
#   ASK_128K=e9c0fbf9-c4f2-4858-97bb-6c1f135efb44 bash proofs-goals-fix.sh    run every line (the id is ASK-264's card);
#                                                  a refused line is reported and the rest still run
#   DRY=1 bash proofs-goals-fix.sh                 print the commands and send nothing
# Every line passes --description, and --blockers or --no-blockers: the site checks a goal's whole text on any
# write, and the texts on the page today predate its rules. `--wait` replaces a goal's whole list of links.
# It sets no priority and no tier.
set -uo pipefail
cd "$(dirname "$0")"

# ASK 4: the ask to Daniel on what the post says of the 128k result (ASK-264). Its card's id goes in ASK_128K, and the 128k goal
# waits on it. Without it the script stops before it writes anything; ASK_128K=none runs without the link.
ASK_128K="${ASK_128K:-}"
if [ -z "$ASK_128K" ]; then
  if [ -n "${DRY:-}" ]; then ASK_128K="THE-ASK-CARD-ID"; else
    echo "proofs-goals-fix: set ASK_128K to the id of the ask card on what the 128k result is called (ASK 4 of the" >&2
    echo "table; ASK-264's is e9c0fbf9-c4f2-4858-97bb-6c1f135efb44), as in ASK_128K=<id> bash proofs-goals-fix.sh; ASK_128K=none runs without that link" >&2; exit 2; fi
fi
ask128=(); [ "$ASK_128K" = none ] || ask128=(--wait "ask:$ASK_128K")

failed=()
g() { if [ -n "${DRY:-}" ]; then printf "research goals set"; printf " %q" "$@"; echo; return; fi
      research goals set "$@" || failed+=("$1"); }

# A new goal (ASK 14), first so that proofs/spot-post-ready's line finds it.
g proofs/spot-two-stage-not-vacuous --title "Two-stage guarantee shown not to be vacuous" --owner proofs \
  --status on-track --due 2026-11-13T12:00-08:00 --tag proofs --description @desc/spot-two-stage-not-vacuous.md

g proofs/slice-clean --status off-track --due 2026-11-13T18:00-08:00 --tag cleanup --tag proofs --no-wait \
  --blockers @blockers/slice-clean.md --description @desc/slice-clean.md

g proofs/spot-bind-term --status on-track --due 2026-11-13T18:00-08:00 --wait goal:ci/fine-grained-check \
  --no-blockers --description @desc/spot-bind-term.md

g proofs/spot-checks-closed --title "Guarantee checks closed or scoped" --status on-track --no-blockers \
  --description @desc/spot-checks-closed.md

g proofs/spot-numbers-lean-checked --status on-track --wait goal:ci/fine-grained-check --no-blockers \
  --description @desc/spot-numbers-lean-checked.md

g proofs/spot-fused-cut-served --status off-track --no-wait --blockers @blockers/spot-fused-cut-served.md \
  --description @desc/spot-fused-cut-served.md

# ASK 4, the card exists; the run command now carries its id: see the table.
g proofs/spot-128k-attention-proved --status off-track --wait goal:vllm/served-128k-window-committed \
  --wait goal:circuits/glm-long-dsa-row --wait goal:circuits/glm-served-calls-fit-block \
  --wait goal:circuits/glm-sparse-attention-zk --wait goal:circuits/glm-top-p-zk ${ask128[@]+"${ask128[@]}"} \
  --no-blockers --description @desc/spot-128k-attention-proved.md

g proofs/spot-advice-certified --status off-track --blockers @blockers/spot-advice-certified.md \
  --description @desc/spot-advice-certified.md

g proofs/spot-moe-binding --status off-track --blockers @blockers/spot-moe-binding.md \
  --description @desc/spot-moe-binding.md

g proofs/spot-one-stage-premises --status on-track --wait goal:ci/fine-grained-check --no-blockers \
  --description @desc/spot-one-stage-premises.md

g proofs/spot-overhead-measured --status off-track --tag proofs --wait goal:circuits/glm-served-calls-fit-block \
  --no-blockers --description @desc/spot-overhead-measured.md

g proofs/spot-ten-percent --status off-track --no-blockers --description @desc/spot-ten-percent.md

g proofs/spot-two-stage-e2e --status off-track --no-blockers --description @desc/spot-two-stage-e2e.md

g proofs/spot-verifier-cost --status on-track --wait goal:circuits/verifier-templates-not-unrolled \
  --wait goal:ci/fine-grained-check --no-blockers --description @desc/spot-verifier-cost.md

g proofs/terminology-clean --status off-track --due 2026-11-20T18:00-08:00 --no-wait \
  --blockers @blockers/terminology-clean.md --description @desc/terminology-clean.md

g proofs/spot-figures --status off-track --tag proofs --no-blockers --description @desc/spot-figures.md

g proofs/spot-claims-checked --status on-track --tag proofs --no-blockers --description @desc/spot-claims-checked.md

g proofs/spot-output-accounting --status off-track --no-blockers --description @desc/spot-output-accounting.md

# ASK 17, a text of mine that ASK-268 made false: see the table.
g proofs/spot-post-ready --status off-track --wait goal:proofs/spot-ten-percent \
  --wait goal:proofs/spot-128k-attention-proved --wait goal:proofs/spot-two-stage-e2e \
  --wait goal:proofs/spot-bind-term --wait goal:proofs/weight-security-bridge \
  --wait goal:proofs/spot-one-stage-premises --wait goal:proofs/spot-moe-binding \
  --wait goal:proofs/spot-output-accounting --wait goal:proofs/spot-advice-certified \
  --wait goal:proofs/spot-checks-closed --wait goal:proofs/spot-numbers-lean-checked --wait goal:proofs/spot-figures \
  --wait goal:proofs/spot-claims-checked --wait goal:circuits/glm-compile-cost-measured \
  --wait goal:proofs/spot-glm-replay --wait goal:proofs/spot-two-stage-not-vacuous --no-blockers \
  --description @desc/spot-post-ready.md

g proofs/weight-security-bridge --status on-track --wait goal:ci/fine-grained-check --no-blockers \
  --description @desc/weight-security-bridge.md

g proofs/zero-knowledge --status off-track --due 2026-11-30T18:00-08:00 --blockers @blockers/zero-knowledge.md \
  --description @desc/zero-knowledge.md

g proofs/zk-leaks-closed --status off-track --due 2026-11-30T18:00-08:00 --blockers @blockers/zk-leaks-closed.md \
  --description @desc/zk-leaks-closed.md

g proofs/sampling-generic --status on-track --no-blockers --description @desc/sampling-generic.md

g proofs/spot-two-stage-sound --status on-track --no-blockers --description @desc/spot-two-stage-sound.md

if [ "${#failed[@]}" -gt 0 ]; then echo "REFUSED or failed: ${failed[*]}" >&2; exit 1; fi
[ -n "${DRY:-}" ] || echo "all 24 written; check with: research goals show"
