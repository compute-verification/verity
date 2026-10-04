"""The protocol: definitions, the interfaces subprotocols meet, the configuration, and the composition.

`accounting` (the account, its interfaces, its decision), `compliance` (the policy, the compliance-proof interface),
`commitments` and `challenges` (what every subprotocol's commitments and coins must be), `configuration` (what a
deployment enables), `epoch` (one epoch: the composition). Stdlib only; it imports no subprotocol, no library and no
integration. Import from the owning module.
"""
