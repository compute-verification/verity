"""verity.protocols.resource_accounting.network.prover: the network attester's Boolean Definitions that core doesn't read: the
whole window's charge, the ingress-ids tie, the scan's ingress and egress, the lengths rows' proof block and the
response tie's.  They are stated over the certifier's schedule and records in
`verity.protocols.resource_accounting.network.verifier`. The Definitions core's service and tie service read stay there with
the ones they build on (`advice`, `io_request`, `io_tie`, `charge_chain`, `arith_encode`), and so does the head check
(`head_check`). The compile slice, its scheduler step, the replay's state chain and the cache step are the layouts
vLLM's compiler commits, and advice v0's validity and the served rows' statics and tie are its checks
(`verity_vllm.compiler`: `compile_slice`, `replay_step`, `replay_chain`, `cache_step`, `advice_check`, `served_statics`,
`served_tie`).
"""
