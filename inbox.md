[2026-11-16T22:40Z] 8ec792addce1
By hand, after vllm's tied served run (node A) and its compose, at a tree holding assemble.py (#10538's head dd60b559faf4 until it lands; main after). SERVED_RUN is the tied serve's run id, COMPOSE_RUN its compose's, D the serve dir on node A, K the window of record, NODE_A node A's ssh name:

~~~sh
# the compose's out/ (its record/verdicts.json is the record) as a work copy, with the policy and limits the serve fixed before it served
research fetch "$SERVED_RUN" --all && research fetch "$COMPOSE_RUN" --all
WORK=$(mktemp -d) && cp -r ~/.research/runs/"$COMPOSE_RUN"/out "$WORK/out"
cp ~/.research/runs/"$SERVED_RUN"/out/inference-only-policy.json "$WORK/out/"
[ ! -f ~/.research/runs/"$SERVED_RUN"/out/inference-only-limits.json ] || cp ~/.research/runs/"$SERVED_RUN"/out/inference-only-limits.json "$WORK/out/"
# the six verifier-side files the tied serve left in D (no witness: the hand-off's kind of file)
S=$WORK/serve && mkdir -p "$S"
printf '%s\n' deployment-registration.json weights_registration.json pearlplus/serve-phi.json "pearlplus/w$K/registration.json" \
  "pearlplus/w$K/tie/window/registration.json" "pearlplus/w$K/tie/circuit-record.json" > "$WORK/six.txt"
rsync -a --files-from="$WORK/six.txt" "NODE_A:$D/" "$S/"
# the Program, the log, the limits where the serve wrote none, and the count, into the work copy; exit 2 says why it refused
uv run python benchmarks/inference_only/assemble.py --from "$WORK/out" --first "$S/deployment-registration.json" \
  --window "$S/pearlplus/w$K/registration.json" --registration "$S/pearlplus/w$K/tie/window/registration.json" \
  --circuit "$S/pearlplus/w$K/tie/circuit-record.json" --phi "$S/pearlplus/serve-phi.json" --pins "$S/weights_registration.json"
uv run verity bench inference_only --from "$WORK/out"
~~~

- `pearlplus/wK/tie/circuit-record.json` exists only where V3c held R's circuit to the window-circuit record (`window_tie`). Where it's missing, rsync fails on it and assemble refuses with "the serve wrote no circuit". R's own circuit is `v3c-wK-<vtag>/window/circuit.json` in D, and assemble holds whichever file it is given to R's SHA-512.
- `--day` defaults to the policy's `written_utc` day. A serve from 10 AM to 1 PM PST is one UTC day, 18:00 to 21:00.

