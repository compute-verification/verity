"""verity.protocols.resource_accounting.memory.pous.prover: PoUS's Boolean circuits: P2's modular squaring (`circuit`), the decode of one block
(`decode_circuit`), the audit answer (`answer_circuit`) and the session verdict (`verdict_circuit`); the setup's
and a session's statements as windows' records for the proof system's workers (`window`); and the prover's
parties of the deployment layer: the memory attester's session's (`challenger_prover`) and the streamed wipe's
(`wipe_prover`); the stores the wipe's prover keeps the fill in (`wipe_store`) and its end of one wipe of a node,
with the verifier's answer on it (`wipe_run`); the honest responder (`responder`); a period's setup, session and subprotocol
(`period_prover`); and the store of a served model's weights that a serving system's adapter holds (`weight_store`).
PoUS's verifier reads none of them.
"""
