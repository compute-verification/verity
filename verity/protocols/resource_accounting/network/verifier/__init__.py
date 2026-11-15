"""Network certification: the covert capacity of a bucketed, constant-rate certifier's timing channel.

Modules: `grid` (the certifier), `schedule` (the Program's logical schedule and clock-sync advice), `audit` (the exact
check, record and reject) and `advice` (advice v0: vLLM's serving decisions as one canonical bit string, the committed
form of the content advice). The replay's state between its step proof blocks, under hiding commitments, is the chain
vLLM's compiler commits (`verity_vllm.compiler.replay_chain`), and advice v0's validity, the compile slice's proof
block's first part, is its `verity_vllm.compiler.advice_check`. The Boolean Definitions core doesn't read are in
`verity.protocols.resource_accounting.network.prover`.
"""
