"""Network certification: the covert capacity of a bucketed, constant-rate certifier's timing channel.

Modules: `grid` (the certifier), `schedule` (the Program's logical schedule and clock-sync advice), `audit` (the exact
check, record and reject), `advice` (advice v0: vLLM's serving decisions as one canonical bit string, the committed
form of the content advice) and `replay_chain` (the replay's state between its step proof blocks, under hiding
commitments). The Boolean Definitions core doesn't read, among them `advice_check`
(advice v0's validity, the compile slice's proof block's first part), are in `verity.protocols.accounting.certifier.prover`.
"""
