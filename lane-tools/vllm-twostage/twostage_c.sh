#!/usr/bin/env bash
# vllm-twostage (VLLM-250): (c), launched from a VM holding the notes and a verity checkout, in two steps.
#   [DRY=1] NOT_BEFORE=ISO bash twostage_c.sh serve
#     twostage_serve.sh (with fi_smoke.py) on node A (vy-spot-glm53) from TREE, a worktree at REV made here if missing,
#     --timeout TIMEOUT (default 45m); refused before NOT_BEFORE. Prints the run id: RUN_C.
#   [DRY=1] bash twostage_c.sh first RUN_C
#     RUN_C's published records fetched here (research fetch --all), the window W's (default 1) class's commit.json
#     among them, then two_stage.py first on vy-challenger-1's gate from vy-nebius-cpu-3 (twostage_first.sh, the
#     interior values tools/interior_classG.json), waited on for at most 20 min. stage1.json lands in the first stage's
#     run record (research fetch RUN_F --all), which (b)'s twostage_b.sh is sent. The window's advice binding
#     (out/advice/w<W>/registration.json of RUN_C, or of ADVICE_RUN) goes with it as --advice when published;
#     CHALLENGER=local runs a local session (sampled-proofs-verifier built first) and doesn't wait.
#   [DRY=1] NOT_BEFORE=ISO bash twostage_c.sh advice RUN_C
#     advice_c.py on node A's CPU (no GPU, 4 CPUs, 8 GB) over RUN_C's kept private dir: each boundary window's advice
#     into PRIV/advice/, only its binding into the run's out/advice/w<k>/registration.json. Prints the run id: ADVICE_RUN.
# REV (default 7af4d24fca, main, the gate's pin since Mon 9 Nov 17:56Z): holds C (#8045), P1 (#8066) and #8055.
set -uo pipefail
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REV=${REV:-7af4d24fca}; TREE=${TREE:-/tmp/wt-c}; VERITY=${VERITY:-/agent/repos/verity}
C4=/workspace/research/runs/r20261107-110834-aaff/inputs/c4-val-00000.json.gz
run() { if [ -n "${DRY:-}" ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
if [ ! -d "$TREE" ]; then
  git -C "$VERITY" fetch -q origin main && git -C "$VERITY" worktree add -q --detach "$TREE" "$REV" || exit 2
fi
[ "$(git -C "$TREE" rev-parse HEAD)" = "$(git -C "$VERITY" rev-parse "$REV^{commit}")" ] || { echo "twostage-c: $TREE is not at $REV"; exit 2; }
[ -z "$(git -C "$TREE" status --porcelain --untracked-files=no)" ] || { echo "twostage-c: $TREE has local changes"; exit 2; }
for f in benchmarks/serve_tax/serve_tax.py:"def await_commits" benchmarks/one_stage/two_stage.py:"def resume"; do
  grep -q "${f#*:}" "$TREE/${f%%:*}" || { echo "twostage-c: $TREE lacks ${f#*:} in ${f%%:*}"; exit 2; }
done
case ${1:-} in
  serve)
    nb=${NOT_BEFORE:?NOT_BEFORE: when the slot starts, a UTC time}
    if [ -z "${DRY:-}" ] && [ "$(date -u +%s)" -lt "$(date -u -d "$nb" +%s)" ]; then echo "twostage-c: before $nb"; exit 2; fi
    envs=(); for v in N P O MAX_MODEL_LEN CANON KEEP AFTER END_UTC NEED_MIN; do [ -n "${!v:-}" ] && envs+=(--env "$v=${!v}"); done
    run research run --on vy-spot-glm53 --project verity --source "$TREE" --cwd source --campaign spot-post \
      --timeout "${TIMEOUT:-45m}" --send "$T/twostage_serve.sh" --send "$T/fi_smoke.py" --send "$T/advice_c.py" \
      --env "C4=$C4" "${envs[@]}" -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/twostage_serve.sh' ;;
  advice)
    rc=${2:?usage: twostage_c.sh advice RUN_C}
    nb=${NOT_BEFORE:?NOT_BEFORE: when node A may be read, a UTC time}
    if [ -z "${DRY:-}" ] && [ "$(date -u +%s)" -lt "$(date -u -d "$nb" +%s)" ]; then echo "twostage-c: before $nb"; exit 2; fi
    run research run --on vy-spot-glm53 --project verity --source "$TREE" --cwd source --campaign spot-post --timeout 20m \
      --gpus 0 --cpus 4 --mem-gb 8 --send "$T/advice_c.py" --env "RUN_C=$rc" -- bash -c 'set -u
P=/workspace/research/vllm-twostage/$RUN_C; A=$P/out/boundary-rows
[ -d "$A/server/boundary" ] || { echo "advice: $RUN_C kept no boundary-rows arm"; exit 2; }
[ -e "$P/advice" ] && { echo "advice: $P/advice exists"; exit 2; }
[ -d /workspace/cp/venv312 ] && v=/workspace/cp/venv312 || v=/workspace/venv312
(umask 077; CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/integrations/vllm:$PWD nice -n 10 "$v/bin/python" \
  "$RESEARCH_RUN_DIR/inputs/advice_c.py" "$A" "$P/advice" "$RESEARCH_RUN_DIR/out" --controls > "$P/advice.log" 2>&1); a=$?
grep "^advice-c:" "$P/advice.log"; echo "advice: exit $a (its log: the private dir'"'"'s advice.log)"; exit $a' ;;
  first)
    rc=${2:?usage: twostage_c.sh first RUN_C}; w=$(mktemp -d); ch=${CHALLENGER:-gate}; extra=()
    if [ -z "${DRY:-}" ]; then
      research fetch "$rc" --on vy-spot-glm53 --all > /dev/null || { echo "twostage-c: no records of $rc"; exit 2; }
      cm=$(ls ~/.research/runs/"$rc"/out/boundary/w"${W:-1}"/p*-o*/commit.json 2>/dev/null | head -1)
      [ -f "$cm" ] || { echo "twostage-c: $rc published no w${W:-1} class (out/boundary/)"; exit 2; }
      echo "twostage-c: $rc published: $(cd ~/.research/runs/"$rc" && find out -type f | sort | tr '\n' ' ')"
      cp "$cm" "$w/commit.json"
      ar=${ADVICE_RUN:-$rc}
      [ "$ar" = "$rc" ] || research fetch "$ar" --on vy-spot-glm53 --all > /dev/null || { echo "twostage-c: no records of $ar"; exit 2; }
      reg=~/.research/runs/$ar/out/advice/w${W:-1}/registration.json
      if [ -f "$reg" ]; then cp "$reg" "$w/advice.json"; extra+=(--send "$w/advice.json")
      elif [ -n "${ADVICE_RUN:-}" ]; then echo "twostage-c: $ar published no w${W:-1} advice"; exit 2; fi
    fi
    [ "$ch" = local ] && extra+=(--env CHALLENGER=local) || extra+=(--wait)
    cp "$T/interior_classG.json" "$w/interior.json"
    git -C "$VERITY" fetch -q origin main && git -C "$VERITY" rev-list --first-parent origin/main > "$w/main.txt"
    run research run --on vy-nebius-cpu-3 --project verity --source "$TREE" --cwd source --campaign spot-post \
      --timeout "${TIMEOUT:-30m}" --send "$T/twostage_first.sh" --send "$T/check_stage1.py" --send "$w/main.txt" \
      --send "$w/commit.json" --send "$w/interior.json" --env PEER=vy-nebius-cpu-3 "${extra[@]}" \
      -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/twostage_first.sh' ;;
  *) echo "usage: twostage_c.sh serve | advice RUN_C | first RUN_C"; exit 2 ;;
esac
