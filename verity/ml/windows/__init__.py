"""What an integration hands Verity (Daniel, 24 Oct: integrations are frontends, and Verity is agnostic to them): a circuit in
proof units with its public layout, registered by digest; when the circuit depends on run-time choices, a registered
compiler and its advice, an opaque bit string whose length is charged; the circuit's values, committed in a
`verity.ml.commitments` scheme and registered before any coin; and on a draw, the drawn units' openings. A window is one
such hand-over. Any integration hands it over the same way, without touching `verity/`.

It imports `verity.ml.circuits` and `verity.ml.commitments` and never `verity.core`: what core's records need from it
(the registration record, the Lean verifier's registered-reads format) comes from the caller, who passes core's module.
Nothing in core imports it, so it is off the soundness path: the verifier recomputes every domain, partition and root it
checks from the registration and its own Program.
"""
