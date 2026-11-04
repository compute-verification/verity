"""Network accounting: the covert capacity of a bucketed, constant-rate certifier's timing channel.

Modules: `grid` (the certifier), `schedule` (the Program's logical schedule and clock-sync advice), `audit` (the exact
check, record and reject), `capacity` (the counts the Lean package proves), `calibration` (sizing from honest runs),
and `online` (the honest prover's sync declared window by window). The advice format vLLM's decisions are committed in
and its scheduler as Boolean Definitions are vLLM's (`integrations/vllm/verity_vllm/scheduler/`).
"""
