"""What an integration hands Verity (integrations are frontends, and Verity is agnostic to them): a circuit in
proof blocks with its public layout, registered by digest; when the circuit depends on run-time choices, a registered
compiler and its advice, an opaque bit string whose length is charged; the circuit's values, committed in a
`verity.commitments` scheme and registered before any coin; and on a sample, the sampled proof blocks' openings. A window is one
such hand-over. Any integration hands it over the same way, without touching `verity/`: `interface` is what it calls (its
`Circuit`, the `Window` it records each unit's values into, closes and writes, and the verifier's window from the records,
`of_records`).

It imports `verity.circuits`, `verity.commitments` and the catalog (`verity.definitions.ops`), and never a verifier
(`verity.protocols`): what a verifier's records need from it (the registration record, the Lean verifier's
registered-reads format) comes from the caller, who passes the verifier's module. No verifier imports it, so it is off
the soundness path: the verifier recomputes every domain, partition and root it checks from the registration and its
own Program.
"""
