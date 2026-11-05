"""The auditor-trusted devices (AGENTS.md "Design rules"): the network certifier (`certifier`) and the memory challenger
(`memory_challenger`), each the auditor's hardware on a machine the prover doesn't control. What a device signs is witness,
and trust in it and its host is the named assumption `trusted-host`. Each device's Lean is its `lean/`."""
