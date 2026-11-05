"""Network accounting: the covert capacity of a bucketed, constant-rate certifier's timing channel.

Modules: `grid` (the certifier), `schedule` (the Program's logical schedule and clock-sync advice), `audit` (the exact
check, record and reject), `capacity` (the counts the Lean package proves), `calibration` (sizing from honest runs),
`online` (the honest prover's sync declared window by window) and `replay_chain` (the replay's state between its step
proof blocks, under hiding commitments). Advice v0, vLLM's serving decisions as one canonical bit string, and its
validity check are the integration's (`verity_vllm.scheduler`).
"""
