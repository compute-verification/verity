"""The protocol: the composition of the subprotocols. `run` (a deployment over many epochs: start here), `epoch` (one
epoch, the step `run` repeats), `accounting` (the account and accounting's decision), `compliance` (the policy's check
of the validated account), `configuration` (what a deployment composes). It imports the subprotocols, the primitives
and the circuits; nothing in `verity` imports it."""
