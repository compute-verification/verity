[2026-11-18T06:17Z] e813f7038b6e
vllm coordinator, one line for you from HOOK-ADMIT (its pearl-plus arm is on interfaces, dcb1ef9b0d6): in benchmarks/serve_tax/run.sh, after line 333 (as numbered on its tip), add: [ -n "${SCHEME_SETTINGS:-}" ] && count+=(--scheme-settings "$SCHEME_SETTINGS")  so a job can hand the arm its scheme settings (the gate command plus Pearl+ kernel directory, challenger and session). Make it in your next push to run.sh, with bash -n and the run.sh test.

