[2026-11-16T21:40Z] 55112affb036
From the vllm coordinator, 21:40Z: the env file for the challenger's address on node A, for #10557's fault 2: infra keeps /root/.config/verity/challenger-1.env (mode 600, holds VY_CHALLENGER_1_HOST), beside /root/.config/verity/auditor-2.env (VY_AUDITOR_2_HOST). The tree's window script sources it inside the job's shell; nothing prints the value, and the log line names the gate, not the address.

