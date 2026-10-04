"""The subprotocols, one per directory. Each stands alone: it imports `verity.primitives` and `verity.circuits`, never
another subprotocol or the protocol, and states in its own terms what it takes and what it establishes (its profile).
`verity.protocol` composes them. Each construction's Lean sits beside its Python."""
