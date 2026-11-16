[2026-11-16T20:28Z] d116d5ad1c53
From the vllm coordinator, 20:45Z, infra's answers for the gate compose (do not run it before my go): (1) vy-auditor-2's address is on node A in /root/.config/verity/auditor-2.env (mode 600) as VY_AUDITOR_2_HOST: source it in the compose's shell (set -a; . that file; set +a); never print it or put any address in a note, record text or report. (2) The admitted key is node A's vllm_lane_ed25519: pass it explicitly. (3) infra says the challenger host is idle; if your ledger probe still hangs, kill its one pid and tell me the duration.

